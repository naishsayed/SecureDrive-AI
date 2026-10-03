
from math import cos, hypot, isfinite, sin


class SecureDriveTrafficModel:
    MAX_DETECTION_DISTANCE = 45.0
    SAFE_FRONT_DISTANCE = 18.0
    SAFE_REAR_DISTANCE = 15.0
    CRITICAL_DISTANCE = 6.0
    LANE_CHANGE_TRIGGER_DISTANCE = 22.0
    MIN_OVERTAKE_SPEED_DIFFERENCE = 2.0
    CONFIRMATION_FRAMES = 2
    MAX_LANE_CHANGE_TIME = 6.0
    FRAME_TIME = 1.0 / 30.0

    def __init__(self, env=None, attack_engine=None):
        self.env = env
        self.attack_engine = attack_engine

        self.lane_change_active = False
        self.target_lane = None
        self.blocked_frames = 0
        self.lane_change_time = 0.0
        self.target_lane_stable_frames = 0

        self.last_front_vehicle = None
        self.last_front_distance = None
        self.last_front_speed = None

        self.last_left_front_distance = None
        self.last_left_rear_distance = None
        self.last_right_front_distance = None
        self.last_right_rear_distance = None

        self.last_perception = {}
        self.last_decision = "CRUISE"

    def set_attack_engine(self, attack_engine):
        self.attack_engine = attack_engine

    def reset(self):
        self.lane_change_active = False
        self.target_lane = None
        self.blocked_frames = 0
        self.lane_change_time = 0.0
        self.target_lane_stable_frames = 0

        self.last_front_vehicle = None
        self.last_front_distance = None
        self.last_front_speed = None

        self.last_left_front_distance = None
        self.last_left_rear_distance = None
        self.last_right_front_distance = None
        self.last_right_rear_distance = None

        self.last_perception = {}
        self.last_decision = "CRUISE"

    def compute_action(self, vehicle, base_action):
        if vehicle is None:
            self.last_decision = "NO_VEHICLE"
            return self._normal_action(base_action)

        if self.env is None:
            try:
                self.env = vehicle.engine
            except Exception:
                pass

        current_lane = self._get_current_lane(vehicle)

        if current_lane is None:
            self.last_decision = "NO_LANE"
            return self._normal_action(base_action)

        objects = self._get_perceived_objects(vehicle)

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
            self.lane_change_time += self.FRAME_TIME

            if front_vehicle is not None:
                if front_distance <= self.CRITICAL_DISTANCE:
                    self.last_decision = "LANE_CHANGE_EMERGENCY_BRAKE"
                    return self._emergency_brake(base_action)

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
                return self._normal_action(base_action)

            self.last_decision = "LANE_CHANGING"
            return self._lane_change_action(base_action)

        if front_vehicle is None:
            self.blocked_frames = 0
            self.last_decision = "CRUISE"
            return self._normal_action(base_action)

        if front_distance > self.LANE_CHANGE_TRIGGER_DISTANCE:
            self.blocked_frames = 0
            self.last_decision = "FOLLOW"

            return self._follow_action(
                vehicle,
                front_distance,
                front_speed,
                base_action,
                ego_speed
            )

        if speed_difference >= self.MIN_OVERTAKE_SPEED_DIFFERENCE:
            self.blocked_frames += 1
        else:
            self.blocked_frames = 0

        if front_distance <= self.CRITICAL_DISTANCE:
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

        if self.blocked_frames >= self.CONFIRMATION_FRAMES:
            target_lane = self._find_best_lane(
                vehicle,
                current_lane,
                traffic
            )

            if target_lane is not None:
                self._start_lane_change(target_lane)
                self.last_decision = "LANE_CHANGE"
                return self._lane_change_action(base_action)

        self.last_decision = "FOLLOW"

        return self._follow_action(
            vehicle,
            front_distance,
            front_speed,
            base_action,
            ego_speed
        )

    def _apply_perception_attack(self, perception):
        if self.attack_engine is None:
            return perception

        try:
            if not self.attack_engine.is_perception_attack_active():
                return perception
        except Exception:
            return perception

        try:
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

    def _get_perceived_objects(self, vehicle):
        objects = []

        try:
            lidar = getattr(vehicle, "lidar", None)

            if lidar is not None:
                detected = lidar.get_surrounding_objects(vehicle)

                if detected is not None:
                    objects = list(detected)

        except Exception:
            objects = []

        if not objects:
            try:
                engine = self.env

                if engine is None:
                    engine = vehicle.engine

                all_objects = engine.get_objects()
                objects = list(all_objects.values())

            except Exception:
                objects = []

        result = []

        for obj in objects:
            if obj is None or obj is vehicle:
                continue

            if not hasattr(obj, "position"):
                continue

            if not hasattr(obj, "lane"):
                continue

            distance = self._distance(vehicle, obj)

            if distance <= self.MAX_DETECTION_DISTANCE:
                result.append(obj)

        return result

    def _analyze_traffic(self, vehicle, current_lane, objects):
        lanes = self._get_adjacent_lanes(
            vehicle,
            current_lane
        )

        left_lane = lanes["left"]
        right_lane = lanes["right"]

        front_vehicle = None
        front_distance = float("inf")

        left_front_distance = float("inf")
        left_rear_distance = float("inf")

        right_front_distance = float("inf")
        right_rear_distance = float("inf")

        for obj in objects:
            obj_lane = getattr(obj, "lane", None)

            if obj_lane is None:
                continue

            if self._same_lane(obj_lane, current_lane):
                distance = self._longitudinal_distance(
                    vehicle.position,
                    obj.position,
                    current_lane
                )

                if distance > 0 and distance < front_distance:
                    front_distance = distance
                    front_vehicle = obj

                continue

            if left_lane is not None:
                if self._same_lane(obj_lane, left_lane):
                    distance = self._longitudinal_distance(
                        vehicle.position,
                        obj.position,
                        left_lane
                    )

                    if distance > 0:
                        left_front_distance = min(
                            left_front_distance,
                            distance
                        )
                    elif distance < 0:
                        left_rear_distance = min(
                            left_rear_distance,
                            abs(distance)
                        )

                    continue

            if right_lane is not None:
                if self._same_lane(obj_lane, right_lane):
                    distance = self._longitudinal_distance(
                        vehicle.position,
                        obj.position,
                        right_lane
                    )

                    if distance > 0:
                        right_front_distance = min(
                            right_front_distance,
                            distance
                        )
                    elif distance < 0:
                        right_rear_distance = min(
                            right_rear_distance,
                            abs(distance)
                        )

        return {
            "front_vehicle": front_vehicle,
            "front_distance": front_distance,
            "left_front_distance": left_front_distance,
            "left_rear_distance": left_rear_distance,
            "right_front_distance": right_front_distance,
            "right_rear_distance": right_rear_distance,
            "left_lane": left_lane,
            "right_lane": right_lane
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

        if not ref_lanes:
            return {"left": None, "right": None}

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

        left_lane = traffic["left_lane"]
        right_lane = traffic["right_lane"]

        if left_lane is not None:
            front = traffic["left_front_distance"]
            rear = traffic["left_rear_distance"]

            if self._lane_is_safe(front, rear):
                candidates.append((left_lane, front, rear))

        if right_lane is not None:
            front = traffic["right_front_distance"]
            rear = traffic["right_rear_distance"]

            if self._lane_is_safe(front, rear):
                candidates.append((right_lane, front, rear))

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return candidates[0][0]

    def _lane_is_safe(self, front_distance, rear_distance):
        front_distance = self._safe_distance(front_distance)
        rear_distance = self._safe_distance(rear_distance)

        if front_distance < self.SAFE_FRONT_DISTANCE:
            return False

        if rear_distance < self.SAFE_REAR_DISTANCE:
            return False

        return True

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
        while angle > 3.141592653589793:
            angle -= 6.283185307179586

        while angle < -3.141592653589793:
            angle += 6.283185307179586

        return angle

    def _normal_action(self, base_action):
        if base_action is None:
            return [0.0, 0.0]

        try:
            if len(base_action) < 2:
                steering = float(base_action[0])

                return [
                    max(-1.0, min(1.0, steering)),
                    0.0
                ]

            steering = float(base_action[0])
            throttle_brake = float(base_action[1])

        except Exception:
            return [0.0, 0.0]

        return [
            max(-1.0, min(1.0, steering)),
            max(-1.0, min(1.0, throttle_brake))
        ]

    def _lane_change_action(self, base_action):
        action = self._normal_action(base_action)

        steering = action[0]
        throttle_brake = action[1]

        if throttle_brake < 0.1:
            throttle_brake = 0.15

        return [
            max(-1.0, min(1.0, steering)),
            max(-1.0, min(1.0, throttle_brake))
        ]

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

        if front_distance <= 7.0:
            return [steering, -0.75]

        if front_distance <= 12.0:
            if ego_speed > front_speed + 1.0:
                return [steering, -0.35]

            return [steering, 0.0]

        if front_distance <= 22.0:
            if ego_speed > front_speed + 3.0:
                return [steering, -0.15]

            return [
                steering,
                min(current_acceleration, 0.2)
            ]

        return [steering, current_acceleration]

    def _emergency_brake(self, base_action):
        action = self._normal_action(base_action)

        return [
            action[0],
            -1.0
        ]

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

    def _same_lane(self, lane_a, lane_b):
        if lane_a is lane_b:
            return True

        try:
            return lane_a.index == lane_b.index
        except Exception:
            return False

    def _lane_index(self, lane):
        try:
            index = lane.index

            if isinstance(index, (list, tuple)):
                return int(index[-1])

            return int(index)

        except Exception:
            return None

    def _longitudinal_distance(
        self,
        ego_position,
        object_position,
        lane
    ):
        try:
            ego_long, _ = lane.local_coordinates(
                ego_position
            )

            object_long, _ = lane.local_coordinates(
                object_position
            )

            return float(object_long - ego_long)

        except Exception:
            try:
                dx = float(
                    object_position[0] - ego_position[0]
                )

                dy = float(
                    object_position[1] - ego_position[1]
                )

                try:
                    ego_long, _ = lane.local_coordinates(
                        ego_position
                    )
                except Exception:
                    ego_long = 0.0

                heading = self._lane_heading(
                    lane,
                    ego_long
                )

                return (
                    dx * cos(heading)
                    + dy * sin(heading)
                )

            except Exception:
                return 0.0

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

            return hypot(
                float(velocity[0]),
                float(velocity[1])
            ) * 3.6

        except Exception:
            return 0.0

    def _distance(self, vehicle_a, vehicle_b):
        try:
            a = vehicle_a.position
            b = vehicle_b.position

            return hypot(
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

    @staticmethod
    def _safe_number(value, fallback=0.0):
        try:
            result = float(value)

            if isfinite(result):
                return result

        except (TypeError, ValueError, OverflowError):
            pass

        return float(fallback)

    def _safe_distance(self, value):
        try:
            result = float(value)

            if not isfinite(result):
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