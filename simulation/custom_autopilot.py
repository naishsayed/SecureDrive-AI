import math


class SecureDriveAutopilot:

    def __init__(self):
        self.target_speed = 15.0
        self.max_speed = 17.0

        self.lookahead_min = 5.0
        self.lookahead_gain = 0.22

        self.steering_gain = 1.15
        self.heading_gain = 0.40
        self.lateral_gain = 0.18

        self.max_steering_change = 0.08
        self.last_steering = 0.0

        self.recovery_stable_steps = 0
        self.recovery_complete = False

    def reset(self):
        self.last_steering = 0.0
        self.recovery_stable_steps = 0
        self.recovery_complete = False

    def reset_recovery(self):
        self.recovery_stable_steps = 0
        self.recovery_complete = False

    def compute_action(
        self,
        vehicle,
        recovery=False,
        target_lane=None
    ):
        if vehicle is None:
            return [0.0, 0.0]

        if recovery:
            return self._compute_recovery_action(vehicle)

        self.reset_recovery()

        lane = target_lane

        if lane is None:
            lane = self._select_lane(vehicle)

        if lane is None:
            return self._safe_fallback(vehicle)

        try:
            longitudinal, lateral = lane.local_coordinates(
                vehicle.position
            )
        except Exception:
            return self._safe_fallback(vehicle)

        lookahead = self._get_lookahead(vehicle.speed)

        steering, heading_error = self._steering_to_lane(
            vehicle,
            lane,
            longitudinal,
            lookahead
        )

        target_speed = self._get_target_speed(
            lane,
            longitudinal,
            lookahead,
            heading_error
        )

        throttle_brake = self._speed_control(
            vehicle.speed,
            target_speed
        )

        steering = self._smooth_steering(steering)

        return [
            steering,
            throttle_brake
        ]

    def compute_lane_change_action(
        self,
        vehicle,
        target_lane,
        base_throttle=0.0
    ):
        if vehicle is None or target_lane is None:
            return [
                self.last_steering,
                0.0
            ]

        try:
            longitudinal, lateral = target_lane.local_coordinates(
                vehicle.position
            )
        except Exception:
            return [
                self.last_steering,
                min(0.0, float(base_throttle))
            ]

        heading = float(vehicle.heading_theta)

        target_heading = self._lane_heading(
            target_lane,
            longitudinal
        )

        heading_error = self._wrap_angle(
            target_heading - heading
        )

        speed = max(float(vehicle.speed), 1.0)

        lookahead = max(
            5.0,
            min(11.0, 5.0 + speed * 0.25)
        )

        target_longitudinal = self._clamp_longitudinal(
            target_lane,
            longitudinal + lookahead
        )

        try:
            target_position = target_lane.position(
                target_longitudinal,
                0.0
            )

            dx = float(target_position[0]) - float(vehicle.position[0])
            dy = float(target_position[1]) - float(vehicle.position[1])

        except Exception:
            return [
                self.last_steering,
                min(0.0, float(base_throttle))
            ]

        forward_x = math.cos(heading)
        forward_y = math.sin(heading)

        left_x = -forward_y
        left_y = forward_x

        local_forward = (
            dx * forward_x
            + dy * forward_y
        )

        local_lateral = (
            dx * left_x
            + dy * left_y
        )

        pure_pursuit_angle = math.atan2(
            local_lateral,
            max(local_forward, 1.0)
        )

        lane_width = self._lane_width(target_lane)

        lateral_ratio = (
            float(lateral) / max(lane_width, 1.0)
        )

        steering = (
            pure_pursuit_angle * 0.75
            + heading_error * 0.35
            + lateral_ratio * 0.18
        )

        steering = max(
            -0.55,
            min(0.55, steering)
        )

        steering = self._smooth_steering(
            steering,
            lane_change=True
        )

        throttle = max(
            -0.35,
            min(0.18, float(base_throttle))
        )

        if abs(float(lateral)) > lane_width * 0.8:
            throttle = min(throttle, 0.05)

        return [
            steering,
            throttle
        ]

    def _compute_recovery_action(self, vehicle):
        lane = self._select_lane(vehicle)

        if lane is None:
            self.recovery_stable_steps = 0
            self.recovery_complete = False

            return [
                self.last_steering,
                -0.15
            ]

        try:
            longitudinal, lateral = lane.local_coordinates(
                vehicle.position
            )
        except Exception:
            self.recovery_stable_steps = 0
            self.recovery_complete = False

            return [
                self.last_steering,
                -0.15
            ]

        lane_heading = self._lane_heading(
            lane,
            longitudinal
        )

        heading_error = self._wrap_angle(
            lane_heading - float(vehicle.heading_theta)
        )

        lateral_abs = abs(float(lateral))

        heading_abs = abs(math.degrees(heading_error))

        speed = float(vehicle.speed)

        lookahead = self._get_recovery_lookahead(
            speed,
            lateral_abs
        )

        steering, _ = self._steering_to_lane(
            vehicle,
            lane,
            longitudinal,
            lookahead
        )

        steering = max(
            -0.45,
            min(0.45, steering)
        )

        steering = self._smooth_steering(
            steering,
            recovery=True
        )

        if lateral_abs > 8.0:
            target_speed = 2.5
        elif lateral_abs > 5.0:
            target_speed = 3.5
        elif lateral_abs > 3.0:
            target_speed = 5.0
        elif heading_abs > 30.0:
            target_speed = 5.0
        else:
            target_speed = 7.0

        if speed < 1.0:
            throttle_brake = 0.25
        else:
            throttle_brake = self._speed_control(
                speed,
                target_speed
            )

        aligned = (
            lateral_abs < 1.0
            and heading_abs < 10.0
            and speed < 10.0
            and bool(getattr(vehicle, "on_lane", True))
        )

        if aligned:
            self.recovery_stable_steps += 1
        else:
            self.recovery_stable_steps = 0

        if self.recovery_stable_steps >= 12:
            self.recovery_complete = True
        else:
            self.recovery_complete = False

        return [
            steering,
            throttle_brake
        ]

    def _select_lane(self, vehicle):
        lanes = []

        try:
            current_ref_lanes = (
                vehicle.navigation.current_ref_lanes
            )

            if current_ref_lanes:
                lanes.extend(current_ref_lanes)

        except Exception:
            pass

        try:
            current_lane = vehicle.lane

            if current_lane is not None and current_lane not in lanes:
                lanes.append(current_lane)

        except Exception:
            pass

        if not lanes:
            return None

        vehicle_heading = float(vehicle.heading_theta)

        best_lane = None
        best_score = float("inf")

        for lane in lanes:
            try:
                longitudinal, lateral = lane.local_coordinates(
                    vehicle.position
                )

                lane_heading = self._lane_heading(
                    lane,
                    longitudinal
                )

                heading_error = abs(
                    self._wrap_angle(
                        lane_heading - vehicle_heading
                    )
                )

                lateral_error = abs(float(lateral))

                score = (
                    lateral_error
                    + 1.5 * min(heading_error, math.pi)
                )

                if score < best_score:
                    best_score = score
                    best_lane = lane

            except Exception:
                continue

        return best_lane

    def _steering_to_lane(
        self,
        vehicle,
        lane,
        longitudinal,
        lookahead
    ):
        target_longitudinal = self._clamp_longitudinal(
            lane,
            longitudinal + lookahead
        )

        target_point = lane.position(
            target_longitudinal,
            0.0
        )

        vehicle_position = vehicle.position

        dx = float(target_point[0] - vehicle_position[0])
        dy = float(target_point[1] - vehicle_position[1])

        heading = float(vehicle.heading_theta)

        forward_x = math.cos(heading)
        forward_y = math.sin(heading)

        left_x = -forward_y
        left_y = forward_x

        local_forward = (
            dx * forward_x
            + dy * forward_y
        )

        local_lateral = (
            dx * left_x
            + dy * left_y
        )

        target_angle = math.atan2(
            local_lateral,
            max(local_forward, 0.5)
        )

        lane_heading = self._lane_heading(
            lane,
            longitudinal
        )

        heading_error = self._wrap_angle(
            lane_heading - heading
        )

        _, lateral_error = lane.local_coordinates(
            vehicle.position
        )

        steering = (
            self.steering_gain * target_angle
            + self.heading_gain * heading_error
            + self.lateral_gain * float(lateral_error)
        )

        steering = max(
            -0.75,
            min(0.75, steering)
        )

        return steering, heading_error

    def _get_target_speed(
        self,
        lane,
        longitudinal,
        lookahead,
        heading_error
    ):
        current_heading = self._lane_heading(
            lane,
            longitudinal
        )

        future_longitudinal = self._clamp_longitudinal(
            lane,
            longitudinal + max(lookahead * 1.5, 3.0)
        )

        future_heading = self._lane_heading(
            lane,
            future_longitudinal
        )

        curvature_signal = abs(
            self._wrap_angle(
                future_heading - current_heading
            )
        )

        curve_reduction = min(
            5.0,
            curvature_signal * 18.0
        )

        heading_reduction = min(
            3.0,
            abs(heading_error) * 4.0
        )

        target = (
            self.target_speed
            - curve_reduction
            - heading_reduction
        )

        return max(
            6.0,
            min(self.max_speed, target)
        )

    def _speed_control(self, speed, target_speed):
        error = float(target_speed) - float(speed)

        if error > 6.0:
            return 0.80

        if error > 3.0:
            return 0.55

        if error > 1.0:
            return 0.35

        if error > 0.2:
            return 0.18

        if error < -6.0:
            return -0.85

        if error < -3.0:
            return -0.55

        if error < -1.0:
            return -0.30

        if error < -0.2:
            return -0.12

        return 0.05

    def _safe_fallback(self, vehicle):
        speed = float(vehicle.speed)

        if speed < 6.0:
            throttle_brake = 0.25
        elif speed > 15.0:
            throttle_brake = -0.25
        else:
            throttle_brake = 0.05

        return [
            self.last_steering,
            throttle_brake
        ]

    def _smooth_steering(
        self,
        steering,
        recovery=False,
        lane_change=False
    ):
        if recovery:
            max_change = 0.06
        elif lane_change:
            max_change = 0.055
        else:
            max_change = self.max_steering_change

        lower = self.last_steering - max_change
        upper = self.last_steering + max_change

        steering = max(
            lower,
            min(upper, steering)
        )

        steering = max(
            -0.75,
            min(0.75, steering)
        )

        self.last_steering = steering

        return steering

    def _get_lookahead(self, speed):
        return (
            self.lookahead_min
            + float(speed) * self.lookahead_gain
        )

    def _get_recovery_lookahead(self, speed, lateral_abs):
        if lateral_abs > 8.0:
            return 2.5

        if lateral_abs > 4.0:
            return 3.0

        return 4.0 + float(speed) * 0.10

    def _lane_heading(self, lane, longitudinal):
        longitudinal = self._clamp_longitudinal(
            lane,
            longitudinal
        )

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

    def _clamp_longitudinal(self, lane, longitudinal):
        try:
            length = float(lane.length)

            return max(
                0.0,
                min(length, float(longitudinal))
            )

        except Exception:
            return max(0.0, float(longitudinal))

    @staticmethod
    def _lane_width(lane):
        try:
            return float(lane.width)
        except Exception:
            return 3.5

    @staticmethod
    def _wrap_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle