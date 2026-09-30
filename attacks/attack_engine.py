from enum import Enum
import random


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

        AttackType.COMBINED_ATTACK

    }

    def __init__(self):

        self.active_attack = AttackType.NONE

        self.recovery_active = False

        self.speed_spoof_value = 5.0

        self.gps_offset_x = 100.0

        self.gps_offset_y = 100.0

        self.noise_level = 5.0

        self.telemetry_history = []

        self.delay_steps = 10

    def activate(self, attack_type):

        self.active_attack = attack_type

        self.recovery_active = False

        self.telemetry_history.clear()

        print()

        print("=" * 70)

        print("[ATTACK ENGINE] ATTACK ACTIVATED")

        print(
            f"Attack: {attack_type.value}"
        )

        print("=" * 70)

    def deactivate(self, recover=True):

        previous_attack = self.active_attack

        self.active_attack = AttackType.NONE

        self.telemetry_history.clear()

        if (
            recover
            and previous_attack in self.PHYSICAL_ATTACKS
        ):

            self.recovery_active = True

            print()

            print("=" * 70)

            print(
                "[ATTACK ENGINE] ATTACK STOPPED"
            )

            print(
                "[ATTACK ENGINE] "
                "CUSTOM AUTOPILOT RECOVERY ACTIVATED"
            )

            print(
                "[ATTACK ENGINE] "
                "ExpertPolicy will NOT control recovery"
            )

            print("=" * 70)

        else:

            self.recovery_active = False

            print()

            print("=" * 70)

            print(
                "[ATTACK ENGINE] ATTACK STOPPED"
            )

            print(
                "[ATTACK ENGINE] "
                "No physical recovery required"
            )

            print("=" * 70)

    def complete_recovery(self):

        if not self.recovery_active:

            return

        self.recovery_active = False

        print()

        print("=" * 70)

        print(
            "[SECUREDRIVE-AI] RECOVERY COMPLETE"
        )

        print(
            "[SECUREDRIVE-AI] "
            "SecureDrive autopilot restored"
        )

        print("=" * 70)

    def modify_action(
        self,
        action,
        vehicle=None
    ):

        steering = float(
            action[0]
        )

        throttle_brake = float(
            action[1]
        )

        # ---------------------------------------------------------
        # RECOVERY
        #
        # Recovery itself is handled by SecureDriveAutopilot.
        # Therefore AttackEngine does not modify the action here.
        # ---------------------------------------------------------

        if self.recovery_active:

            return [

                max(
                    -1.0,
                    min(
                        1.0,
                        steering
                    )
                ),

                max(
                    -1.0,
                    min(
                        1.0,
                        throttle_brake
                    )
                )

            ]

        # ---------------------------------------------------------
        # PHYSICAL ATTACKS
        # ---------------------------------------------------------

        if (
            self.active_attack
            == AttackType.SUDDEN_ACCELERATION
        ):

            throttle_brake = 1.0

        elif (
            self.active_attack
            == AttackType.SUDDEN_BRAKING
        ):

            throttle_brake = -1.0

        elif (
            self.active_attack
            == AttackType.STEERING_MANIPULATION
        ):

            steering = 1.0

        elif (
            self.active_attack
            == AttackType.THROTTLE_MANIPULATION
        ):

            throttle_brake = 0.8

        elif (
            self.active_attack
            == AttackType.BRAKE_MANIPULATION
        ):

            throttle_brake = -1.0

        elif (
            self.active_attack
            == AttackType.COMBINED_ATTACK
        ):

            steering = 1.0

            throttle_brake = 0.8

        # ---------------------------------------------------------
        # LIMIT ACTION VALUES
        # ---------------------------------------------------------

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

        return [

            steering,

            throttle_brake

        ]

    # =============================================================
    # TELEMETRY ATTACKS
    # =============================================================

    def modify_telemetry(
        self,
        telemetry
    ):

        # ---------------------------------------------------------
        # SPEED SENSOR SPOOFING
        # ---------------------------------------------------------

        if (
            self.active_attack
            == AttackType.SPEED_SENSOR_SPOOFING
        ):

            telemetry.speed = (
                self.speed_spoof_value
            )

        # ---------------------------------------------------------
        # GPS SPOOFING
        # ---------------------------------------------------------

        elif (
            self.active_attack
            == AttackType.GPS_SPOOFING
        ):

            telemetry.position_x += (
                self.gps_offset_x
            )

            telemetry.position_y += (
                self.gps_offset_y
            )

        # ---------------------------------------------------------
        # SENSOR NOISE
        # ---------------------------------------------------------

        elif (
            self.active_attack
            == AttackType.SENSOR_NOISE
        ):

            telemetry.speed += random.uniform(
                -self.noise_level,
                self.noise_level
            )

            telemetry.position_x += random.uniform(
                -self.noise_level,
                self.noise_level
            )

            telemetry.position_y += random.uniform(
                -self.noise_level,
                self.noise_level
            )

        # ---------------------------------------------------------
        # TELEMETRY DELAY
        # ---------------------------------------------------------

        elif (
            self.active_attack
            == AttackType.TELEMETRY_DELAY
        ):

            self.telemetry_history.append(
                telemetry
            )

            if (
                len(
                    self.telemetry_history
                )
                > self.delay_steps
            ):

                return (
                    self.telemetry_history[
                        -self.delay_steps
                    ]
                )

        # ---------------------------------------------------------
        # COMBINED ATTACK
        # ---------------------------------------------------------

        elif (
            self.active_attack
            == AttackType.COMBINED_ATTACK
        ):

            telemetry.speed = (
                self.speed_spoof_value
            )

            telemetry.position_x += (
                self.gps_offset_x
            )

            telemetry.position_y += (
                self.gps_offset_y
            )

            telemetry.speed += random.uniform(
                -self.noise_level,
                self.noise_level
            )

        return telemetry

    # =============================================================
    # STATUS
    # =============================================================

    def get_attack_name(self):

        if self.recovery_active:

            return "RECOVERY"

        return self.active_attack.value

    def is_recovering(self):

        return self.recovery_active

    def is_attack_active(self):

        return (
            self.active_attack
            != AttackType.NONE
        )

    def get_active_attack(self):

        return self.active_attack