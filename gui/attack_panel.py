from direct.gui.DirectButton import DirectButton
from direct.gui.DirectFrame import DirectFrame
from direct.gui.DirectLabel import DirectLabel
from direct.task import Task
from panda3d.core import TextNode
import math


class AttackPanel:

    def __init__(self, engine, attack_engine):

        self.engine = engine

        self.attack_engine = attack_engine

        self.panel = None

        self.buttons = []

        self.telemetry_panel = None

        self.telemetry_labels = {}

        self.create_panel()

        self.create_realtime_telemetry()

        self.engine.taskMgr.add(
            self.update_realtime_telemetry,
            "securedrive-realtime-telemetry"
        )

    def create_panel(self):

        self.panel = DirectFrame(
            parent=self.engine.a2dBottomCenter,
            frameColor=(0.015, 0.02, 0.025, 0.96),
            frameSize=(-1.45, 1.45, 0.0, 0.34),
            pos=(0, 0, 0.0)
        )

        self.title = DirectLabel(
            parent=self.panel,
            text="SECUREDRIVE-AI  |  ATTACK CONTROL",
            text_scale=0.030,
            text_align=TextNode.ACenter,
            text_fg=(1, 1, 1, 1),
            frameColor=(0.02, 0.04, 0.06, 1),
            frameSize=(-1.45, 1.45, -0.025, 0.025),
            pos=(0, 0, 0.305)
        )

        attacks = [
            ("ACCELERATION", "SUDDEN_ACCELERATION"),
            ("BRAKING", "SUDDEN_BRAKING"),
            ("STEERING", "STEERING_MANIPULATION"),
            ("THROTTLE", "THROTTLE_MANIPULATION"),
            ("BRAKE CTRL", "BRAKE_MANIPULATION"),
            ("SPEED SPOOF", "SPEED_SENSOR_SPOOFING"),
            ("GPS SPOOF", "GPS_SPOOFING"),
            ("SENSOR NOISE", "SENSOR_NOISE"),
            ("TELEMETRY DELAY", "TELEMETRY_DELAY"),
            ("COMBINED", "COMBINED_ATTACK")
        ]

        x_positions = [
            -1.16,
            -0.58,
            0.0,
            0.58,
            1.16
        ]

        for index, (label, attack_name) in enumerate(attacks):

            row = index // 5

            column = index % 5

            x = x_positions[column]

            z = (
                0.235
                if row == 0
                else 0.165
            )

            button = DirectButton(
                parent=self.panel,
                text=label,
                text_scale=0.024,
                text_align=TextNode.ACenter,
                text_fg=(1, 1, 1, 1),
                frameColor=(0.10, 0.12, 0.15, 1),
                frameSize=(-0.25, 0.25, -0.027, 0.027),
                pos=(x, 0, z),
                command=self.activate_attack,
                extraArgs=[attack_name]
            )

            self.buttons.append(button)

        self.stop_button = DirectButton(
            parent=self.panel,
            text="STOP ATTACK",
            text_scale=0.026,
            text_align=TextNode.ACenter,
            text_fg=(1, 1, 1, 1),
            frameColor=(0.55, 0.08, 0.08, 1),
            frameSize=(-0.25, 0.25, -0.027, 0.027),
            pos=(0, 0, 0.085),
            command=self.stop_attack
        )

        self.status = DirectLabel(
            parent=self.panel,
            text="ATTACK: NONE  |  STATUS: READY",
            text_scale=0.022,
            text_align=TextNode.ACenter,
            text_fg=(0.75, 0.85, 0.90, 1),
            frameColor=(0, 0, 0, 0),
            frameSize=(-0.60, 0.60, -0.022, 0.022),
            pos=(0, 0, 0.025)
        )

    def create_realtime_telemetry(self):

        self.telemetry_panel = DirectFrame(
            parent=self.engine.a2dTopLeft,
            frameColor=(0.015, 0.02, 0.025, 0.92),
            frameSize=(0.0, 0.72, -0.78, 0.0),
            pos=(0.035, 0, -0.055)
        )

        title = DirectLabel(
            parent=self.telemetry_panel,
            text="SECUREDRIVE-AI  |  LIVE TELEMETRY",
            text_scale=0.030,
            text_align=TextNode.ALeft,
            text_fg=(1, 1, 1, 1),
            frameColor=(0.02, 0.04, 0.06, 1),
            frameSize=(0.0, 0.72, -0.045, 0.045),
            pos=(0.025, 0, -0.04)
        )

        self.telemetry_labels["title"] = title

        telemetry_items = [

            ("speed", "SPEED       : 0.0 km/h"),

            ("rpm", "WHEEL RPM   : 0 RPM"),

            ("steering", "STEERING    : 0.00"),

            ("throttle", "THROTTLE    : 0.00"),

            ("brake", "BRAKE       : 0.00"),

            ("acceleration", "ACCELERATION: 0.00 m/s²"),

            ("heading", "HEADING     : 0.0°"),

            ("position", "POSITION    : X 0.0 | Y 0.0"),

            ("lane", "LANE OFFSET : 0.00 m"),

            ("status", "STATUS      : NORMAL")

        ]

        start_z = -0.11

        spacing = 0.060

        for index, (key, text) in enumerate(
            telemetry_items
        ):

            label = DirectLabel(
                parent=self.telemetry_panel,
                text=text,
                text_scale=0.022,
                text_align=TextNode.ALeft,
                text_fg=(0.82, 0.88, 0.92, 1),
                frameColor=(0, 0, 0, 0),
                frameSize=(0.0, 0.68, -0.025, 0.025),
                pos=(0.025, 0, start_z - index * spacing)
            )

            self.telemetry_labels[key] = label

    def find_vehicle(self):

        try:

            objects = self.engine.get_objects()

            if objects:

                for obj in objects.values():

                    if (
                        hasattr(obj, "speed")
                        and hasattr(obj, "steering")
                        and hasattr(obj, "throttle_brake")
                    ):

                        return obj

        except Exception:

            pass

        return None

    def calculate_wheel_rpm(self, vehicle):

        try:

            speed_kmh = float(
                vehicle.speed
            )

            speed_ms = (
                speed_kmh / 3.6
            )

            tire_radius = getattr(
                vehicle,
                "TIRE_RADIUS",
                None
            )

            if tire_radius is None:

                tire_radius = 0.34

            tire_radius = float(
                tire_radius
            )

            if tire_radius <= 0:

                tire_radius = 0.34

            circumference = (
                2.0
                * math.pi
                * tire_radius
            )

            wheel_rpm = (
                speed_ms
                / circumference
                * 60.0
            )

            return max(
                0.0,
                wheel_rpm
            )

        except Exception:

            return 0.0

    def update_realtime_telemetry(
        self,
        task
    ):

        vehicle = self.find_vehicle()

        if vehicle is None:

            return Task.cont

        try:

            speed = float(
                vehicle.speed
            )

        except Exception:

            speed = 0.0

        try:

            steering = float(
                vehicle.steering
            )

        except Exception:

            steering = 0.0

        try:

            throttle_brake = float(
                vehicle.throttle_brake
            )

        except Exception:

            throttle_brake = 0.0

        throttle = max(
            0.0,
            throttle_brake
        )

        brake = max(
            0.0,
            -throttle_brake
        )

        rpm = self.calculate_wheel_rpm(
            vehicle
        )

        try:

            acceleration = float(
                getattr(
                    vehicle,
                    "last_speed",
                    speed
                )
            )

            acceleration = (
                speed
                - acceleration
            ) / 3.6 / 0.1

        except Exception:

            acceleration = 0.0

        try:

            heading = math.degrees(
                float(
                    vehicle.heading_theta
                )
            )

        except Exception:

            heading = 0.0

        try:

            position_x = float(
                vehicle.position[0]
            )

            position_y = float(
                vehicle.position[1]
            )

        except Exception:

            position_x = 0.0

            position_y = 0.0

        lane_offset = 0.0

        try:

            navigation = vehicle.navigation

            if navigation is not None:

                lanes = (
                    navigation.current_ref_lanes
                )

                if lanes:

                    _, lateral = (
                        lanes[0].local_coordinates(
                            vehicle.position
                        )
                    )

                    lane_offset = float(
                        lateral
                    )

        except Exception:

            lane_offset = 0.0

        attack = (
            self.attack_engine.get_attack_name()
        )

        if self.attack_engine.is_recovering():

            status = "RECOVERY"

        elif self.attack_engine.is_attack_active():

            status = "ATTACK ACTIVE"

        else:

            status = "NORMAL"

        self.telemetry_labels["speed"][
            "text"
        ] = (
            f"SPEED       : "
            f"{speed:6.2f} km/h"
        )

        self.telemetry_labels["rpm"][
            "text"
        ] = (
            f"WHEEL RPM   : "
            f"{rpm:6.0f} RPM"
        )

        self.telemetry_labels["steering"][
            "text"
        ] = (
            f"STEERING    : "
            f"{steering:6.2f}"
        )

        self.telemetry_labels["throttle"][
            "text"
        ] = (
            f"THROTTLE    : "
            f"{throttle:6.2f}"
        )

        self.telemetry_labels["brake"][
            "text"
        ] = (
            f"BRAKE       : "
            f"{brake:6.2f}"
        )

        self.telemetry_labels["acceleration"][
            "text"
        ] = (
            f"ACCELERATION: "
            f"{acceleration:6.2f} m/s²"
        )

        self.telemetry_labels["heading"][
            "text"
        ] = (
            f"HEADING     : "
            f"{heading:6.1f}°"
        )

        self.telemetry_labels["position"][
            "text"
        ] = (
            f"POSITION    : "
            f"X {position_x:7.1f} | "
            f"Y {position_y:7.1f}"
        )

        self.telemetry_labels["lane"][
            "text"
        ] = (
            f"LANE OFFSET : "
            f"{lane_offset:6.2f} m"
        )

        self.telemetry_labels["status"][
            "text"
        ] = (
            f"STATUS      : "
            f"{status}"
        )

        return Task.cont

    def activate_attack(
        self,
        attack_name
    ):

        from attacks.attack_engine import AttackType

        attack_type = AttackType(
            attack_name
        )

        self.attack_engine.activate(
            attack_type
        )

        self.update_status()

        print()

        print("=" * 70)

        print(
            "[SECUREDRIVE-AI] ATTACK ACTIVATED"
        )

        print(
            f"Attack : {attack_type.value}"
        )

        print("=" * 70)

    def stop_attack(self):

        self.attack_engine.deactivate(
            recover=True
        )

        self.update_status()

        print()

        print("=" * 70)

        print(
            "[SECUREDRIVE-AI] ATTACK STOPPED"
        )

        print(
            "SecureDrive custom recovery controller activated."
        )

        print("=" * 70)

    def update_status(self):

        attack = (
            self.attack_engine.get_attack_name()
        )

        if self.attack_engine.is_recovering():

            status = "RECOVERING"

        elif attack == "NONE":

            status = "READY"

        else:

            status = "ACTIVE"

        self.status["text"] = (
            f"ATTACK: {attack}  |  "
            f"STATUS: {status}"
        )

    def destroy(self):

        try:

            self.engine.taskMgr.remove(
                "securedrive-realtime-telemetry"
            )

        except Exception:

            pass

        if self.panel:

            self.panel.destroy()

            self.panel = None

        if self.telemetry_panel:

            self.telemetry_panel.destroy()

            self.telemetry_panel = None