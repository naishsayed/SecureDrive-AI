from metadrive.policy.idm_policy import IDMPolicy


class TrafficAvoidance:

    def __init__(self):

        self.idm_policy = None

        self.vehicle = None

        self.front_vehicle_detected = False

        self.front_vehicle_distance = float("inf")

        self.front_vehicle_speed = 0.0

        self.avoidance_active = False

        self.emergency_braking = False

        self.selected_avoidance = "NONE"

        self.last_action = [0.0, 0.0]

        self.initialized = False

    def reset(self):

        self.front_vehicle_detected = False

        self.front_vehicle_distance = float("inf")

        self.front_vehicle_speed = 0.0

        self.avoidance_active = False

        self.emergency_braking = False

        self.selected_avoidance = "NONE"

        self.last_action = [0.0, 0.0]

        if self.idm_policy is not None:

            try:

                self.idm_policy.reset()

            except Exception:

                pass

    def _initialize_idm(
        self,
        vehicle
    ):

        if self.idm_policy is not None:

            if self.vehicle is vehicle:

                return

        self.vehicle = vehicle

        try:

            self.idm_policy = IDMPolicy(
                control_object=vehicle,
                random_seed=10
            )

            self.idm_policy.target_speed = 25

            self.idm_policy.enable_lane_change = True

            self.initialized = True

        except Exception as error:

            print(
                "[TRAFFIC] IDM initialization failed:",
                error
            )

            self.idm_policy = None

            self.initialized = False

    def compute_safe_action(
        self,
        vehicle,
        base_action
    ):

        self._initialize_idm(
            vehicle
        )

        if self.idm_policy is None:

            self.selected_avoidance = (
                "IDM_UNAVAILABLE"
            )

            self.last_action = [
                float(base_action[0]),
                float(base_action[1])
            ]

            return self.last_action

        self._update_front_vehicle_status(
            vehicle
        )

        try:

            action = self.idm_policy.act()

            steering = float(
                action[0]
            )

            throttle_brake = float(
                action[1]
            )

            steering = max(
                -1.0,
                min(
                    1.0,
                    steering
                )
            )

            throttle_brake = max(
                -1.0,
                min(
                    1.0,
                    throttle_brake
                )
            )

            self.last_action = [
                steering,
                throttle_brake
            ]

            self._update_decision(
                vehicle
            )

            return self.last_action

        except Exception as error:

            self.selected_avoidance = (
                "IDM_FALLBACK"
            )

            self.last_action = [
                float(base_action[0]),
                float(base_action[1])
            ]

            return self.last_action

    def _update_front_vehicle_status(
        self,
        vehicle
    ):

        self.front_vehicle_detected = False

        self.front_vehicle_distance = (
            float("inf")
        )

        self.front_vehicle_speed = 0.0

        try:

            objects = (
                vehicle.lidar.get_surrounding_objects(
                    vehicle,
                    40
                )
            )

            front_object = None

            front_distance = float(
                "inf"
            )

            for obj in objects:

                if obj is vehicle:

                    continue

                try:

                    lane = obj.lane

                    if lane is not vehicle.lane:

                        continue

                    ego_long, _ = (
                        vehicle.lane.local_coordinates(
                            vehicle.position
                        )
                    )

                    object_long, _ = (
                        vehicle.lane.local_coordinates(
                            obj.position
                        )
                    )

                    distance = (
                        float(object_long)
                        -
                        float(ego_long)
                    )

                    if distance <= 0:

                        continue

                    if distance < front_distance:

                        front_distance = (
                            distance
                        )

                        front_object = obj

                except Exception:

                    continue

            if front_object is not None:

                self.front_vehicle_detected = True

                self.front_vehicle_distance = (
                    front_distance
                )

                try:

                    self.front_vehicle_speed = float(
                        front_object.speed
                    )

                except Exception:

                    self.front_vehicle_speed = 0.0

        except Exception:

            pass

    def _update_decision(
        self,
        vehicle
    ):

        self.avoidance_active = False

        self.emergency_braking = False

        if self.front_vehicle_detected:

            self.avoidance_active = True

            if self.front_vehicle_distance <= 5.0:

                self.emergency_braking = True

                self.selected_avoidance = (
                    "EMERGENCY_BRAKE"
                )

            elif self._is_changing_lane(
                vehicle
            ):

                self.selected_avoidance = (
                    "LANE_CHANGE"
                )

            elif self.front_vehicle_distance <= 15.0:

                self.selected_avoidance = (
                    "FOLLOWING"
                )

            else:

                self.selected_avoidance = (
                    "TRAFFIC_DETECTED"
                )

        else:

            if self._is_changing_lane(
                vehicle
            ):

                self.avoidance_active = True

                self.selected_avoidance = (
                    "LANE_CHANGE"
                )

            else:

                self.selected_avoidance = (
                    "CLEAR"
                )

    def _is_changing_lane(
        self,
        vehicle
    ):

        if self.idm_policy is None:

            return False

        try:

            target_lane = (
                self.idm_policy.routing_target_lane
            )

            current_lane = (
                vehicle.lane
            )

            if target_lane is None:

                return False

            if current_lane is None:

                return False

            return (
                target_lane is not current_lane
            )

        except Exception:

            return False

    def get_status(self):

        target_lane_active = False

        if self.idm_policy is not None:

            try:

                target_lane_active = (
                    self.idm_policy.routing_target_lane
                    is not None
                )

            except Exception:

                target_lane_active = False

        return {

            "vehicle_detected":
                self.front_vehicle_detected,

            "front_distance":
                self.front_vehicle_distance,

            "front_speed":
                self.front_vehicle_speed,

            "avoidance_active":
                self.avoidance_active,

            "emergency_braking":
                self.emergency_braking,

            "decision":
                self.selected_avoidance,

            "lane_change_active":
                self._is_changing_lane(
                    self.vehicle
                ) if self.vehicle is not None
                else False,

            "target_lane":
                target_lane_active

        }