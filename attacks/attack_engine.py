
from enum import Enum
from collections import deque
import copy
import math
import random
import time


class AttackType(Enum):
    NONE = "NONE"
    SUDDEN_ACCELERATION = "SUDDEN_ACCELERATION"
    SUDDEN_BRAKING = "SUDDEN_BRAKING"
    STEERING_MANIPULATION = "STEERING_MANIPULATION"
    THROTTLE_MANIPULATION = "THROTTLE_MANIPULATION"
    BRAKE_MANIPULATION = "BRAKE_MANIPULATION"
    SPEED_SENSOR_SPOOFING = "SPEED_SENSOR_SPOOFING"
    GPS_SPOOFING = "GPS_SPOOFING"
    SENSOR_NOISE = "SENSOR_NOISE"
    TELEMETRY_DELAY = "TELEMETRY_DELAY"
    COMBINED_ATTACK = "COMBINED_ATTACK"


class AttackEngine:

    PHYSICAL_ATTACKS = {
        AttackType.SUDDEN_ACCELERATION,
        AttackType.SUDDEN_BRAKING,
        AttackType.STEERING_MANIPULATION,
        AttackType.THROTTLE_MANIPULATION,
        AttackType.BRAKE_MANIPULATION,
        AttackType.COMBINED_ATTACK,
    }

    PERCEPTION_ATTACKS = {
        AttackType.SPEED_SENSOR_SPOOFING,
        AttackType.GPS_SPOOFING,
        AttackType.SENSOR_NOISE,
    }

    TELEMETRY_ATTACKS = {
        AttackType.TELEMETRY_DELAY,
    }

    def __init__(self):

        self.active_attack = AttackType.NONE
        self.recovery_active = False

        self.attack_started_at = None
        self.attack_duration = None
        self.attack_intensity = 1.0

        self.speed_spoof_value = 5.0

        self.gps_offset_x = 100.0
        self.gps_offset_y = 100.0

        self.noise_level = 5.0
        self.distance_noise_level = 4.0

        self.delay_steps = 10

        self.telemetry_history = deque(maxlen=500)
        self.perception_history = deque(maxlen=500)

        self.attack_count = 0
        self.last_attack = AttackType.NONE
        self.last_attack_started_at = None

    @staticmethod
    def _clamp(value, minimum=-1.0, maximum=1.0):

        return max(
            minimum,
            min(maximum, float(value))
        )

    def _elapsed(self):

        if self.attack_started_at is None:
            return 0.0

        return max(
            0.0,
            time.monotonic() - self.attack_started_at
        )

    def _expire_attack_if_needed(self):

        if self.active_attack == AttackType.NONE:
            return

        if self.attack_duration is None:
            return

        if self._elapsed() >= self.attack_duration:
            self.deactivate(recover=False)

    def activate(
        self,
        attack_type,
        duration=None,
        intensity=1.0
    ):

        if isinstance(attack_type, str):

            try:
                attack_type = AttackType[attack_type]

            except KeyError:

                try:
                    attack_type = AttackType(attack_type)

                except ValueError:
                    raise ValueError(
                        f"Unknown attack type: {attack_type}"
                    )

        if not isinstance(attack_type, AttackType):
            raise ValueError("Invalid attack type.")

        if attack_type == AttackType.NONE:
            self.deactivate(recover=False)
            return

        if duration is not None and float(duration) <= 0:
            raise ValueError("Attack duration must be positive.")

        intensity = float(intensity)

        if not 0.0 <= intensity <= 1.0:
            raise ValueError(
                "Attack intensity must be between 0 and 1."
            )

        self.active_attack = attack_type
        self.recovery_active = False

        self.attack_started_at = time.monotonic()

        self.attack_duration = (
            None
            if duration is None
            else float(duration)
        )

        self.attack_intensity = intensity

        self.last_attack = attack_type
        self.last_attack_started_at = self.attack_started_at

        self.attack_count += 1

        self.telemetry_history.clear()
        self.perception_history.clear()

        print()
        print("=" * 70)
        print("[ATTACK ENGINE] ATTACK ACTIVATED")
        print(f"Attack: {attack_type.value}")
        print(f"Intensity: {intensity * 100:.0f}%")

        print(
            "Duration: "
            + (
                "Manual"
                if duration is None
                else f"{float(duration):.1f} seconds"
            )
        )

        print("=" * 70)

    def deactivate(self, recover=True):

        previous_attack = self.active_attack

        self.active_attack = AttackType.NONE
        self.recovery_active = False

        self.attack_started_at = None
        self.attack_duration = None

        self.telemetry_history.clear()
        self.perception_history.clear()

        if previous_attack != AttackType.NONE:

            print()
            print("=" * 70)
            print("[ATTACK ENGINE] ATTACK STOPPED")
            print("[ATTACK ENGINE] NORMAL AUTOPILOT CONTROL RESTORED")
            print("=" * 70)

    def complete_recovery(self):

        self.recovery_active = False

    def _attack_wave(self, frequency=0.8):

        elapsed = self._elapsed()

        return math.sin(
            2.0 * math.pi * frequency * elapsed
        )

    def modify_action(self, action, vehicle=None):

        if action is None or len(action) < 2:
            return [0.0, 0.0]

        steering = float(action[0])
        throttle_brake = float(action[1])

        self._expire_attack_if_needed()

        intensity = self.attack_intensity
        attack = self.active_attack

        if attack == AttackType.SUDDEN_ACCELERATION:

            throttle_brake = (
                1.0 * intensity
                + throttle_brake * (1.0 - intensity)
            )

        elif attack == AttackType.SUDDEN_BRAKING:

            throttle_brake = (
                -1.0 * intensity
                + throttle_brake * (1.0 - intensity)
            )

        elif attack == AttackType.STEERING_MANIPULATION:

            interference = (
                self._attack_wave(frequency=0.65)
                * 0.75
                * intensity
            )

            steering = (
                steering * (1.0 - intensity)
                + interference
            )

        elif attack == AttackType.THROTTLE_MANIPULATION:

            pulse = (
                0.55
                + 0.40 * self._attack_wave(frequency=0.45)
            )

            manipulated_throttle = max(
                0.0,
                pulse * intensity
            )

            throttle_brake = (
                throttle_brake * (1.0 - intensity)
                + manipulated_throttle
            )

        elif attack == AttackType.BRAKE_MANIPULATION:

            if throttle_brake < 0.0:

                brake_suppression = (
                    1.0 - 0.90 * intensity
                )

                throttle_brake *= brake_suppression

        elif attack == AttackType.COMBINED_ATTACK:

            steering_interference = (
                self._attack_wave(frequency=0.55)
                * 0.65
                * intensity
            )

            steering = (
                steering * (1.0 - intensity)
                + steering_interference
            )

            throttle_brake = (
                throttle_brake * (1.0 - intensity)
                + 0.80 * intensity
            )

        elif (
            attack in self.PERCEPTION_ATTACKS
            or attack in self.TELEMETRY_ATTACKS
        ):

            steering_interference = (
                self._attack_wave(frequency=0.40)
                * 0.45
                * intensity
            )

            surge = (
                0.35
                + 0.30 * self._attack_wave(frequency=0.30)
            )

            if attack == AttackType.SENSOR_NOISE:

                steering_interference += (
                    random.uniform(-0.20, 0.20)
                    * intensity
                )

            steering = (
                steering * (1.0 - intensity)
                + steering_interference
            )

            throttle_brake = (
                throttle_brake * (1.0 - intensity)
                + surge * intensity
            )

        return [
            self._clamp(steering),
            self._clamp(throttle_brake),
        ]

    def modify_perception(self, perception):

        if not isinstance(perception, dict):
            return perception

        self._expire_attack_if_needed()

        if self.active_attack not in self.PERCEPTION_ATTACKS:
            return perception

        modified = copy.deepcopy(perception)
        intensity = self.attack_intensity

        if self.active_attack == AttackType.SPEED_SENSOR_SPOOFING:

            actual_speed = modified.get("speed")

            if isinstance(actual_speed, (int, float)):

                modified["speed"] = (
                    float(actual_speed) * (1.0 - intensity)
                    + self.speed_spoof_value * intensity
                )

        elif self.active_attack == AttackType.GPS_SPOOFING:

            if "position_x" in modified:

                modified["position_x"] += (
                    self.gps_offset_x * intensity
                )

            if "position_y" in modified:

                modified["position_y"] += (
                    self.gps_offset_y * intensity
                )

            if "position" in modified:

                position = modified["position"]

                if (
                    isinstance(position, (list, tuple))
                    and len(position) >= 2
                ):

                    modified["position"] = (
                        position[0] + self.gps_offset_x * intensity,
                        position[1] + self.gps_offset_y * intensity,
                    )

        elif self.active_attack == AttackType.SENSOR_NOISE:

            for key in (
                "speed",
                "front_speed",
                "position_x",
                "position_y",
                "front_distance",
                "left_front_distance",
                "left_rear_distance",
                "right_front_distance",
                "right_rear_distance",
            ):

                value = modified.get(key)

                if isinstance(value, (int, float)):

                    noise_limit = (
                        self.noise_level
                        if "distance" not in key
                        else self.distance_noise_level
                    )

                    noise = (
                        random.uniform(
                            -noise_limit,
                            noise_limit
                        )
                        * intensity
                    )

                    modified[key] = value + noise

        return modified

    @staticmethod
    def _get_telemetry_value(
        telemetry,
        key,
        default=None
    ):

        if isinstance(telemetry, dict):
            return telemetry.get(key, default)

        return getattr(telemetry, key, default)

    @staticmethod
    def _set_telemetry_value(
        telemetry,
        key,
        value
    ):

        if isinstance(telemetry, dict):
            telemetry[key] = value

        else:
            setattr(telemetry, key, value)

    def modify_telemetry(self, telemetry):

        if telemetry is None:
            return telemetry

        self._expire_attack_if_needed()

        if self.active_attack not in self.TELEMETRY_ATTACKS:
            return telemetry

        if self.active_attack == AttackType.TELEMETRY_DELAY:

            snapshot = copy.deepcopy(telemetry)

            self.telemetry_history.append(snapshot)

            if len(self.telemetry_history) > self.delay_steps:

                delayed_index = -(
                    self.delay_steps + 1
                )

                return copy.deepcopy(
                    self.telemetry_history[delayed_index]
                )

            return copy.deepcopy(
                self.telemetry_history[0]
            )

        return telemetry

    def get_attack_name(self):

        self._expire_attack_if_needed()

        return self.active_attack.value

    def get_active_attack(self):

        self._expire_attack_if_needed()

        return self.active_attack

    def is_recovering(self):

        return False

    def is_attack_active(self):

        self._expire_attack_if_needed()

        return self.active_attack != AttackType.NONE

    def is_physical_attack_active(self):

        self._expire_attack_if_needed()

        return self.active_attack in self.PHYSICAL_ATTACKS

    def is_perception_attack_active(self):

        self._expire_attack_if_needed()

        return self.active_attack in self.PERCEPTION_ATTACKS

    def is_telemetry_attack_active(self):

        self._expire_attack_if_needed()

        return self.active_attack in self.TELEMETRY_ATTACKS

    def get_status(self):

        self._expire_attack_if_needed()

        elapsed = (
            self._elapsed()
            if self.active_attack != AttackType.NONE
            else 0.0
        )

        remaining = None

        if self.attack_duration is not None:

            remaining = max(
                0.0,
                self.attack_duration - elapsed
            )

        return {
            "active": self.active_attack != AttackType.NONE,
            "attack": self.active_attack.value,
            "category": (
                "PHYSICAL"
                if self.active_attack in self.PHYSICAL_ATTACKS
                else "PERCEPTION"
                if self.active_attack in self.PERCEPTION_ATTACKS
                else "TELEMETRY"
                if self.active_attack in self.TELEMETRY_ATTACKS
                else "NONE"
            ),
            "intensity": self.attack_intensity,
            "elapsed": elapsed,
            "duration": self.attack_duration,
            "remaining": remaining,
            "recovering": False,
            "attack_count": self.attack_count,
        }