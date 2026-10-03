
import math
import time


class SecureDriveTrafficModel:

    MAX_DETECTION_DISTANCE = 80.0
    SAFE_FRONT_DISTANCE = 18.0
    SAFE_REAR_DISTANCE = 15.0
    CRITICAL_DISTANCE = 5.5
    LANE_CHANGE_TRIGGER_DISTANCE = 32.0
    MIN_OVERTAKE_SPEED_DIFFERENCE = 2.0
    CONFIRMATION_FRAMES = 2
    MIN_LANE_CHANGE_GAIN = 6.0
    MAX_LANE_CHANGE_TIME = 6.0
    FOLLOW_TIME_HEADWAY = 1.5
    MIN_FOLLOW_DISTANCE = 8.0
    EMERGENCY_TTC = 1.0

    def __init__(self, env=None, attack_engine=None):

        self.env = env
        self.attack_engine = attack_engine

        self.lane_change_active = False
        self.target_lane = None

        self.blocked_frames = 0
        self.lane_change_time = 0.0
        self.target_lane_stable_frames = 0

        self.last_update_time = time.monotonic()

        self.last_front_vehicle = None
        self.last_front_distance = None
        self.last_front_speed = None

        self.last_left_front_distance = None
        self.last_left_rear_distance = None
        self.last_right_front_distance = None
        self.last_right_rear_distance = None

        self.last_perception = {}
        self.last_decision = "CRUISE"

        self.last_detected_vehicle_count = 0
        self.last_detection_names = []

    def set_attack_engine(self, attack_engine):
        self.attack_engine = attack_engine

    def reset(self):

        self.lane_change_active = False
        self.target_lane = None

        self.blocked_frames = 0
        self.lane_change_time = 0.0
        self.target_lane_stable_frames = 0

        self.last_update_time = time.monotonic()

        self.last_front_vehicle = None
        self.last_front_distance = None
        self.last_front_speed = None

        self.last_left_front_distance = None
        self.last_left_rear_distance = None
        self.last_right_front_distance = None
        self.last_right_rear_distance = None

        self.last_perception = {}
        self.last_decision = "CRUISE"

        self.last_detected_vehicle_count = 0
        self.last_detection_names = []

    def compute_action(self, vehicle, base_action):

        if vehicle is None:
            self.last_decision = "NO_VEHICLE"
            return self._normal_action(base_action)

        if self.env is None:
            try:
                self.env = vehicle.engine
            except Exception:
                pass

        now = time.monotonic()
        delta_time = max(0.001, min(now - self.last_update_time, 0.25))
        self.last_update_time = now

        current_lane = self._get_current_lane(vehicle)

        if current_lane is None:
            self.last_decision = "NO_LANE"
            return self._normal_action(base_action)

        objects = self._get_perceived_objects(vehicle)

        self.last_detected_vehicle_count = len(objects)
        self.last_detection_names = [
            obj.__class__.__name__ for obj in objects[:10]
        ]

        traffic = self._analyze_traffic(
            vehicle,
            current_lane,
            objects
        )

        ego_speed = self._get_speed(vehicle)
        front_vehicle = traffic["front_vehicle"]

        if front_vehicle is not None:
            front_speed = self._get_speed(front_vehicle)
        else:
            front_speed = ego_speed

        perception = {
            "speed": ego_speed,
            "front_speed": front_speed,
            "front_distance": self._safe_distance(
                traffic["front_distance"]
            ),
            "left_front_distance": self._safe_distance(
                traffic["left_front_distance"]
            ),
            "left_rear_distance": self._safe_distance(
                traffic["left_rear_distance"]
            ),
            "right_front_distance": self._safe_distance(
                traffic["right_front_distance"]
            ),
            "right_rear_distance": self._safe_distance(
                traffic["right_rear_distance"]
            ),
            "position_x": self._position_component(vehicle, 0),
            "position_y": self._position_component(vehicle, 1)
        }

        perception = self._apply_perception_attack(perception)

        ego_speed = self._safe_number(
            perception.get("speed"),
            ego_speed
        )

        front_speed = self._safe_number(
            perception.get("front_speed"),
            front_speed
        )

        front_distance = self._safe_distance(
            perception.get("front_distance")
        )

        left_front = self._safe_distance(
            perception.get("left_front_distance")
        )

        left_rear = self._safe_distance(
            perception.get("left_rear_distance")
        )

        right_front = self._safe_distance(
            perception.get("right_front_distance")
        )

        right_rear = self._safe_distance(
            perception.get("right_rear_distance")
        )

        traffic["front_distance"] = front_distance
        traffic["left_front_distance"] = left_front
        traffic["left_rear_distance"] = left_rear
        traffic["right_front_distance"] = right_front
        traffic["right_rear_distance"] = right_rear

        self.last_perception = dict(perception)

        self.last_front_vehicle = front_vehicle
        self.last_front_distance = (
            front_distance if front_vehicle is not None else None
        )
        self.last_front_speed = (
            front_speed if front_vehicle is not None else None
        )

        self.last_left_front_distance = left_front
        self.last_left_rear_distance = left_rear
        self.last_right_front_distance = right_front
        self.last_right_rear_distance = right_rear

        speed_difference = ego_speed - front_speed

        if self.lane_change_active:

            self.lane_change_time += delta_time

            target_lane_safe, target_front_distance, target_front_speed = (
                self._target_lane_gap_status(
                    vehicle,
                    current_lane,
                    traffic,
                    ego_speed
                )
            )

            if (
                front_vehicle is not None
                and front_distance <= self._emergency_distance(ego_speed)
            ):
                self.last_decision = "LANE_CHANGE_EMERGENCY_BRAKE"
                return self._emergency_brake(base_action)

            if not target_lane_safe:
                self.last_decision = "LANE_CHANGE_TARGET_UNSAFE"

                if target_front_distance <= self._emergency_distance(ego_speed):
                    return self._emergency_brake(base_action)

                if target_front_distance <= self._desired_follow_distance(ego_speed):
                    return self._follow_action(
                        vehicle,
                        target_front_distance,
                        target_front_speed,
                        base_action,
                        ego_speed
                    )

                action = self._normal_action(base_action)
                return [action[0], min(action[1], 0.0)]

            if self._target_lane_reached(vehicle):

                self.lane_change_active = False
                self.target_lane = None

                self.lane_change_time = 0.0
                self.blocked_frames = 0
                self.target_lane_stable_frames = 0

                self.last_decision = "LANE_CHANGE_COMPLETE"

                return self._normal_action(base_action)

            if self.lane_change_time > self.MAX_LANE_CHANGE_TIME:

                self.lane_change_active = False
                self.target_lane = None

                self.lane_change_time = 0.0
                self.target_lane_stable_frames = 0

                self.last_decision = "LANE_CHANGE_TIMEOUT"

                return self._follow_action(
                    vehicle,
                    front_distance,
                    front_speed,
                    base_action,
                    ego_speed
                )

            self.last_decision = "LANE_CHANGING"

            return self._lane_change_action(base_action)

        if front_vehicle is None:

            self.blocked_frames = 0
            self.last_decision = "CRUISE"

            return self._normal_action(base_action)

        desired_distance = self._desired_follow_distance(ego_speed)
        emergency_distance = self._emergency_distance(ego_speed)

        if front_distance <= emergency_distance:

            target_lane = self._find_best_lane(
                vehicle,
                current_lane,
                traffic
            )

            if target_lane is not None:

                self._start_lane_change(target_lane)

                self.last_decision = "EMERGENCY_LANE_CHANGE"

                return self._lane_change_action(base_action)

            self.last_decision = "EMERGENCY_BRAKE"

            return self._emergency_brake(base_action)

        if (
            front_distance <= self.LANE_CHANGE_TRIGGER_DISTANCE
            and speed_difference >= self.MIN_OVERTAKE_SPEED_DIFFERENCE
        ):
            self.blocked_frames += 1
        else:
            self.blocked_frames = 0

        if (
            self.blocked_frames >= self.CONFIRMATION_FRAMES
            and front_distance > emergency_distance
        ):

            target_lane = self._find_best_lane(
                vehicle,
                current_lane,
                traffic
            )

            if target_lane is not None:

                self._start_lane_change(target_lane)

                self.last_decision = "LANE_CHANGE"

                return self._lane_change_action(base_action)

        if front_distance <= desired_distance:

            self.last_decision = "FOLLOW"

            return self._follow_action(
                vehicle,
                front_distance,
                front_speed,
                base_action,
                ego_speed
            )

        self.last_decision = "CRUISE"

        return self._normal_action(base_action)

    def _get_perceived_objects(self, vehicle):

        objects = []
        seen = set()

        engine = self.env

        if engine is None:
            try:
                engine = vehicle.engine
            except Exception:
                engine = None

        if engine is not None:

            try:
                detected = engine.get_objects()

                if isinstance(detected, dict):
                    detected = detected.values()

                objects.extend(list(detected))

            except Exception:
                pass

            try:
                traffic_manager = engine.traffic_manager
                detected = traffic_manager.get_objects()

                if isinstance(detected, dict):
                    detected = detected.values()

                objects.extend(list(detected))

            except Exception:
                pass

        try:
            lidar = getattr(vehicle, "lidar", None)

            if lidar is not None:

                detected = lidar.get_surrounding_objects(vehicle)

                if isinstance(detected, dict):
                    detected = detected.values()

                if detected is not None:
                    objects.extend(list(detected))

        except Exception:
            pass

        result = []

        for obj in objects:

            if obj is None or obj is vehicle:
                continue

            object_id = id(obj)

            if object_id in seen:
                continue

            seen.add(object_id)

            if not hasattr(obj, "position"):
                continue

            if not self._is_vehicle_object(obj):
                continue

            distance = self._distance(vehicle, obj)

            if distance <= self.MAX_DETECTION_DISTANCE:
                result.append(obj)

        return result

    def _is_vehicle_object(self, obj):

        try:
            from metadrive.component.vehicle.base_vehicle import BaseVehicle

            if isinstance(obj, BaseVehicle):
                return True

        except Exception:
            pass

        class_name = obj.__class__.__name__.lower()

        if "vehicle" in class_name or "car" in class_name or "truck" in class_name:
            return True

        if hasattr(obj, "speed_km_h") and hasattr(obj, "position"):
            return True

        if (
            hasattr(obj, "navigation")
            and hasattr(obj, "lane")
            and hasattr(obj, "position")
        ):
            return True

        return False

    def _analyze_traffic(self, vehicle, current_lane, objects):

        adjacent = self._get_adjacent_lanes(
            vehicle,
            current_lane
        )

        lanes = [
            ("current", current_lane),
            ("left", adjacent["left"]),
            ("right", adjacent["right"])
        ]

        lane_coordinates = {}

        for name, lane in lanes:

            if lane is None:
                continue

            try:
                lane_coordinates[name] = lane.local_coordinates(
                    vehicle.position
                )
            except Exception:
                lane_coordinates[name] = None

        front_vehicle = None
        front_distance = float("inf")

        left_front_distance = float("inf")
        left_rear_distance = float("inf")
        left_front_vehicle = None
        left_rear_vehicle = None

        right_front_distance = float("inf")
        right_rear_distance = float("inf")
        right_front_vehicle = None
        right_rear_vehicle = None

        for obj in objects:

            object_position = getattr(obj, "position", None)

            if object_position is None:
                continue

            best_name = None
            best_lateral = float("inf")
            best_longitudinal = None
            best_lane = None

            for name, lane in lanes:

                if lane is None:
                    continue

                if lane_coordinates.get(name) is None:
                    continue

                try:
                    object_long, object_lateral = (
                        lane.local_coordinates(object_position)
                    )

                    ego_long, _ = lane_coordinates[name]

                    longitudinal = float(object_long - ego_long)
                    lateral = abs(float(object_lateral))

                    if lateral < best_lateral:

                        best_lateral = lateral
                        best_longitudinal = longitudinal
                        best_name = name
                        best_lane = lane

                except Exception:
                    continue

            if best_lane is None:
                continue

            lane_width = self._safe_number(
                getattr(best_lane, "width", 3.5),
                3.5
            )

            object_width = self._safe_number(
                getattr(obj, "WIDTH", 1.8),
                1.8
            )

            lateral_limit = (
                lane_width * 0.5
                + min(object_width * 0.25, 0.45)
            )

            if best_lateral > lateral_limit:
                continue

            distance = best_longitudinal

            if distance is None:
                continue

            if best_name == "current":

                if distance > 0 and distance < front_distance:

                    front_distance = distance
                    front_vehicle = obj

            elif best_name == "left":

                if distance > 0 and distance < left_front_distance:
                    left_front_distance = distance
                    left_front_vehicle = obj

                elif distance < 0 and abs(distance) < left_rear_distance:
                    left_rear_distance = abs(distance)
                    left_rear_vehicle = obj

            elif best_name == "right":

                if distance > 0 and distance < right_front_distance:
                    right_front_distance = distance
                    right_front_vehicle = obj

                elif distance < 0 and abs(distance) < right_rear_distance:
                    right_rear_distance = abs(distance)
                    right_rear_vehicle = obj

        return {
            "front_vehicle": front_vehicle,
            "front_distance": front_distance,

            "left_front_distance": left_front_distance,
            "left_rear_distance": left_rear_distance,
            "left_front_vehicle": left_front_vehicle,
            "left_rear_vehicle": left_rear_vehicle,

            "right_front_distance": right_front_distance,
            "right_rear_distance": right_rear_distance,
            "right_front_vehicle": right_front_vehicle,
            "right_rear_vehicle": right_rear_vehicle,

            "left_lane": adjacent["left"],
            "right_lane": adjacent["right"]
        }

    def _get_adjacent_lanes(self, vehicle, current_lane):

        left_lane = None
        right_lane = None

        try:
            ref_lanes = list(
                vehicle.navigation.current_ref_lanes
            )
        except Exception:
            ref_lanes = []

        current_index = self._lane_index(current_lane)

        if current_index is None:
            return {"left": None, "right": None}

        for lane in ref_lanes:

            lane_index = self._lane_index(lane)

            if lane_index is None:
                continue

            if lane_index == current_index - 1:
                left_lane = lane

            elif lane_index == current_index + 1:
                right_lane = lane

        return {
            "left": left_lane,
            "right": right_lane
        }

    def _find_best_lane(self, vehicle, current_lane, traffic):

        candidates = []

        ego_speed = self._get_speed(vehicle)
        ego_speed_ms = max(0.0, ego_speed / 3.6)

        current_front = self._safe_distance(
            traffic["front_distance"]
        )

        base_front_gap = max(
            self.SAFE_FRONT_DISTANCE,
            self._desired_follow_distance(ego_speed)
        )

        base_rear_gap = max(
            self.SAFE_REAR_DISTANCE,
            8.0 + ego_speed_ms * 0.8
        )

        lane_options = (
            (
                "left",
                traffic["left_lane"],
                traffic["left_front_distance"],
                traffic["left_rear_distance"],
                traffic.get("left_front_vehicle"),
                traffic.get("left_rear_vehicle")
            ),
            (
                "right",
                traffic["right_lane"],
                traffic["right_front_distance"],
                traffic["right_rear_distance"],
                traffic.get("right_front_vehicle"),
                traffic.get("right_rear_vehicle")
            )
        )

        for name, lane, front_raw, rear_raw, front_vehicle, rear_vehicle in lane_options:

            if lane is None:
                continue

            front = self._safe_distance(front_raw)
            rear = self._safe_distance(rear_raw)

            front_speed = (
                self._get_speed(front_vehicle)
                if front_vehicle is not None
                else ego_speed
            )

            rear_speed = (
                self._get_speed(rear_vehicle)
                if rear_vehicle is not None
                else ego_speed
            )

            front_closing_speed = max(
                0.0,
                (ego_speed - front_speed) / 3.6
            )

            rear_closing_speed = max(
                0.0,
                (rear_speed - ego_speed) / 3.6
            )

            required_front = max(
                base_front_gap,
                base_front_gap + front_closing_speed * 1.5
            )

            required_rear = max(
                base_rear_gap,
                base_rear_gap + rear_closing_speed * 2.0
            )

            required_front = max(
                required_front,
                min(
                    self.MAX_DETECTION_DISTANCE,
                    current_front + self.MIN_LANE_CHANGE_GAIN
                )
            )

            if not self._lane_is_safe(
                front,
                rear,
                required_front,
                required_rear
            ):
                continue

            front_ttc = (
                front / front_closing_speed
                if front_closing_speed > 0.1
                else float("inf")
            )

            rear_ttc = (
                rear / rear_closing_speed
                if rear_closing_speed > 0.1
                else float("inf")
            )

            if front_ttc < 2.0 or rear_ttc < 2.0:
                continue

            score = (
                min(front, self.MAX_DETECTION_DISTANCE)
                - max(0.0, required_rear - rear) * 0.5
                - front_closing_speed * 2.0
                - rear_closing_speed * 2.5
            )

            candidates.append((lane, score))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0]

    def _target_lane_gap_status(
        self,
        vehicle,
        current_lane,
        traffic,
        ego_speed
    ):

        if self.target_lane is None:
            return True, self.MAX_DETECTION_DISTANCE + 1.0, ego_speed

        if self.target_lane is traffic.get("left_lane"):

            front_distance = self._safe_distance(
                traffic.get("left_front_distance")
            )

            rear_distance = self._safe_distance(
                traffic.get("left_rear_distance")
            )

            front_vehicle = traffic.get("left_front_vehicle")
            rear_vehicle = traffic.get("left_rear_vehicle")

        elif self.target_lane is traffic.get("right_lane"):

            front_distance = self._safe_distance(
                traffic.get("right_front_distance")
            )

            rear_distance = self._safe_distance(
                traffic.get("right_rear_distance")
            )

            front_vehicle = traffic.get("right_front_vehicle")
            rear_vehicle = traffic.get("right_rear_vehicle")

        else:
            return True, self.MAX_DETECTION_DISTANCE + 1.0, ego_speed

        front_speed = (
            self._get_speed(front_vehicle)
            if front_vehicle is not None
            else ego_speed
        )

        rear_speed = (
            self._get_speed(rear_vehicle)
            if rear_vehicle is not None
            else ego_speed
        )

        ego_speed_ms = max(0.0, ego_speed / 3.6)

        front_closing = max(
            0.0,
            (ego_speed - front_speed) / 3.6
        )

        rear_closing = max(
            0.0,
            (rear_speed - ego_speed) / 3.6
        )

        required_front = max(
            self.SAFE_FRONT_DISTANCE,
            self._desired_follow_distance(ego_speed)
        ) + front_closing * 1.5

        required_rear = max(
            self.SAFE_REAR_DISTANCE,
            8.0 + ego_speed_ms * 0.8
        ) + rear_closing * 2.0

        front_ttc = (
            front_distance / front_closing
            if front_closing > 0.1
            else float("inf")
        )

        rear_ttc = (
            rear_distance / rear_closing
            if rear_closing > 0.1
            else float("inf")
        )

        safe = (
            self._lane_is_safe(
                front_distance,
                rear_distance,
                required_front,
                required_rear
            )
            and front_ttc >= 1.5
            and rear_ttc >= 1.5
        )

        return safe, front_distance, front_speed

    def _lane_is_safe(
        self,
        front_distance,
        rear_distance,
        safe_front=None,
        safe_rear=None
    ):

        front_distance = self._safe_distance(front_distance)
        rear_distance = self._safe_distance(rear_distance)

        if safe_front is None:
            safe_front = self.SAFE_FRONT_DISTANCE

        if safe_rear is None:
            safe_rear = self.SAFE_REAR_DISTANCE

        return (
            front_distance >= safe_front
            and rear_distance >= safe_rear
        )

    def _start_lane_change(self, target_lane):

        self.target_lane = target_lane
        self.lane_change_active = True

        self.lane_change_time = 0.0
        self.target_lane_stable_frames = 0

    def _target_lane_reached(self, vehicle):

        if self.target_lane is None:
            self.target_lane_stable_frames = 0
            return False

        try:

            longitudinal, lateral = (
                self.target_lane.local_coordinates(
                    vehicle.position
                )
            )

            lane_heading = self._lane_heading(
                self.target_lane,
                longitudinal
            )

            heading_error = abs(
                self._wrap_angle(
                    lane_heading - float(vehicle.heading_theta)
                )
            )

            lane_width = float(
                getattr(self.target_lane, "width", 3.5)
            )

            centered = abs(float(lateral)) <= min(
                0.7,
                lane_width * 0.22
            )

            aligned = heading_error <= 0.2

            if centered and aligned:
                self.target_lane_stable_frames += 1
            else:
                self.target_lane_stable_frames = 0

            return self.target_lane_stable_frames >= 5

        except Exception:
            self.target_lane_stable_frames = 0
            return False

    def _lane_heading(self, lane, longitudinal):

        try:
            return float(
                lane.heading_theta_at(longitudinal)
            )
        except Exception:
            try:
                return float(
                    lane.heading_at(longitudinal)
                )
            except Exception:
                return 0.0

    @staticmethod
    def _wrap_angle(angle):

        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def _normal_action(self, base_action):

        if base_action is None:
            return [0.0, 0.0]

        try:
            steering = float(base_action[0])
            throttle_brake = float(base_action[1])
        except Exception:
            return [0.0, 0.0]

        return [
            max(-1.0, min(1.0, steering)),
            max(-1.0, min(1.0, throttle_brake))
        ]

    def _lane_change_action(self, base_action):

        return self._normal_action(base_action)

    def _follow_action(
        self,
        vehicle,
        front_distance,
        front_speed,
        base_action,
        perceived_ego_speed=None
    ):

        action = self._normal_action(base_action)

        steering = action[0]
        current_acceleration = action[1]

        if perceived_ego_speed is None:
            ego_speed = self._get_speed(vehicle)
        else:
            ego_speed = perceived_ego_speed

        front_distance = self._safe_distance(front_distance)

        relative_speed = max(
            0.0,
            ego_speed - front_speed
        )

        relative_speed_ms = relative_speed / 3.6

        desired_distance = self._desired_follow_distance(
            ego_speed
        )

        emergency_distance = self._emergency_distance(
            ego_speed
        )

        closing_ttc = (
            front_distance / relative_speed_ms
            if relative_speed_ms > 0.1
            else float("inf")
        )

        if (
            front_distance <= emergency_distance
            or closing_ttc <= self.EMERGENCY_TTC
        ):
            return [steering, -1.0]

        if front_distance <= desired_distance:

            distance_error = (
                front_distance - desired_distance
            )

            braking = max(
                -0.85,
                min(
                    0.0,
                    distance_error * 0.07
                    - relative_speed_ms * 0.16
                )
            )

            return [
                steering,
                min(current_acceleration, braking)
            ]

        if front_distance <= desired_distance + 12.0:

            if relative_speed > 3.0:

                return [
                    steering,
                    min(current_acceleration, 0.12)
                ]

            return [
                steering,
                min(current_acceleration, 0.30)
            ]

        return [steering, current_acceleration]

    def _emergency_brake(self, base_action):

        action = self._normal_action(base_action)

        return [
            action[0],
            -1.0
        ]

    def _desired_follow_distance(self, speed_kmh):

        speed_ms = max(0.0, speed_kmh / 3.6)

        return max(
            self.MIN_FOLLOW_DISTANCE,
            self.MIN_FOLLOW_DISTANCE
            + speed_ms * self.FOLLOW_TIME_HEADWAY
        )

    def _emergency_distance(self, speed_kmh):

        speed_ms = max(0.0, speed_kmh / 3.6)

        return max(
            self.CRITICAL_DISTANCE,
            3.0 + speed_ms * 1.0
        )

    def _get_current_lane(self, vehicle):

        try:
            lane = vehicle.lane

            if lane is not None:
                return lane

        except Exception:
            pass

        try:
            lanes = vehicle.navigation.current_ref_lanes

            if lanes:
                return lanes[0]

        except Exception:
            pass

        return None

    def _lane_index(self, lane):

        try:
            index = lane.index

            if isinstance(index, (list, tuple)):
                return int(index[-1])

            return int(index)

        except Exception:
            return None

    def _get_speed(self, vehicle):

        try:
            return float(vehicle.speed_km_h)
        except Exception:
            pass

        try:
            return float(vehicle.speed) * 3.6
        except Exception:
            pass

        try:
            velocity = vehicle.velocity

            return math.hypot(
                float(velocity[0]),
                float(velocity[1])
            ) * 3.6

        except Exception:
            return 0.0

    def _distance(self, vehicle_a, vehicle_b):

        try:
            a = vehicle_a.position
            b = vehicle_b.position

            return math.hypot(
                float(b[0] - a[0]),
                float(b[1] - a[1])
            )

        except Exception:
            return float("inf")

    def _position_component(self, vehicle, index):

        try:
            return float(vehicle.position[index])
        except Exception:
            return 0.0

    def _apply_perception_attack(self, perception):

        if self.attack_engine is None:
            return perception

        try:

            if not self.attack_engine.is_perception_attack_active():
                return perception

            modified = self.attack_engine.modify_perception(
                dict(perception)
            )

            if isinstance(modified, dict):
                return modified

        except Exception as error:

            print(
                "[Traffic Model] Perception attack error:",
                error
            )

        return perception

    @staticmethod
    def _safe_number(value, fallback=0.0):

        try:
            result = float(value)

            if math.isfinite(result):
                return result

        except (TypeError, ValueError, OverflowError):
            pass

        return float(fallback)

    def _safe_distance(self, value):

        try:
            result = float(value)

            if not math.isfinite(result):
                return self.MAX_DETECTION_DISTANCE + 1.0

            return max(0.0, result)

        except (TypeError, ValueError, OverflowError):
            return self.MAX_DETECTION_DISTANCE + 1.0

    def get_status(self):

        return {
            "decision": self.last_decision,

            "front_vehicle_detected": (
                self.last_front_vehicle is not None
            ),

            "front_distance": self.last_front_distance,
            "front_speed": self.last_front_speed,

            "detected_vehicle_count": (
                self.last_detected_vehicle_count
            ),

            "detected_vehicle_types": list(
                self.last_detection_names
            ),

            "left_front_distance": (
                self.last_left_front_distance
            ),

            "left_rear_distance": (
                self.last_left_rear_distance
            ),

            "right_front_distance": (
                self.last_right_front_distance
            ),

            "right_rear_distance": (
                self.last_right_rear_distance
            ),

            "lane_change_active": self.lane_change_active,
            "target_lane": self.target_lane,

            "perception": dict(self.last_perception)
        }