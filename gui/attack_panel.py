import math
import os

from direct.gui import DirectGuiGlobals as DGG
from direct.gui.DirectButton import DirectButton
from direct.gui.DirectFrame import DirectFrame
from direct.gui.DirectLabel import DirectLabel
from direct.task import Task
from panda3d.core import (
    TextNode,
    PNMImage,
    Texture,
    TransparencyAttrib,
    Filename
)


class AttackPanel:

    # (button label, attack name)
    ATTACKS = [
        ("CAN Injection", "SUDDEN_ACCELERATION"),
        ("CAN Message Injection", "SUDDEN_BRAKING"),
        ("Fuzzing", "STEERING_MANIPULATION"),
        ("CAN Command", "THROTTLE_MANIPULATION"),
        ("CAN Spoofing", "BRAKE_MANIPULATION"),
        ("Sensor Spoofing", "SPEED_SENSOR_SPOOFING"),
        ("GPS/GNSS", "GPS_SPOOFING"),
        ("FDI Attack", "SENSOR_NOISE"),
        ("DoS Attack", "TELEMETRY_DELAY"),
        ("Multi-Vector", "COMBINED_ATTACK")
    ]

    COLORS = {
        # matches the reference picture
        "button": (0.086, 0.137, 0.184, 1.0),
        "hover": (0.125, 0.200, 0.265, 1.0),
        "active": (0.560, 0.075, 0.095, 1.0),
        "active_hover": (0.700, 0.100, 0.120, 1.0),
        "stop": (0.647, 0.114, 0.153, 1.0),
        "stop_hover": (0.820, 0.150, 0.190, 1.0),
        "text": (1.0, 1.0, 1.0, 1.0),
        "title": (0.80, 0.84, 0.88, 1.0),
        "shadow": (0, 0, 0, 0.85),
        "transparent": (0, 0, 0, 0)
    }

    # Rounded-button texture density (texels per aspect2d unit)
    TEXELS_PER_UNIT = 400.0

    def __init__(self, engine, attack_engine):

        self.engine = engine
        self.attack_engine = attack_engine

        self.panel = None
        self.buttons = []
        self.button_attack_names = []
        self.stop_button = None
        self.hovered = set()

        self.telemetry_panel = None
        self.telemetry_labels = {}

        self.previous_speed = None
        self.previous_time = None

        self.font = None
        self.load_font()

        self.create_panel()
        self.create_realtime_telemetry()

        self.engine.taskMgr.add(
            self.update_realtime_telemetry,
            "securedrive-realtime-telemetry"
        )

        self.engine.taskMgr.add(
            self.refresh_attack_buttons,
            "securedrive-refresh-attack-buttons"
        )

    def get_aspect_ratio(self):

        try:
            return max(1.2, float(self.engine.getAspectRatio()))
        except Exception:
            return 1.777

    def load_font(self):
        """Bold sans-serif font like the reference. Falls back to default."""

        fonts_dir = os.path.join(
            os.environ.get("WINDIR", "C:/Windows"),
            "Fonts"
        )

        for file_name in ("segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"):

            try:
                path = os.path.join(fonts_dir, file_name)

                if not os.path.exists(path):
                    continue

                font = self.engine.loader.loadFont(
                    Filename.fromOsSpecific(path)
                )

                if font is not None:
                    font.setPixelsPerUnit(90)
                    self.font = font
                    return

            except Exception:
                continue

    def font_args(self):

        if self.font is None:
            return {}

        return {"text_font": self.font}

    def create_button_texture(self, width, height, radius):
        """Anti-aliased rounded rectangle (white, alpha-masked)."""

        px = self.TEXELS_PER_UNIT

        w = max(8, int(round(width * px)))
        h = max(8, int(round(height * px)))
        r = min(radius * px, w / 2.0, h / 2.0)

        image = PNMImage(w, h, 4)

        cx = w / 2.0
        cy = h / 2.0

        for y in range(h):
            for x in range(w):

                qx = abs(x + 0.5 - cx) - (cx - r)
                qy = abs(y + 0.5 - cy) - (cy - r)

                outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
                inside = min(max(qx, qy), 0.0)

                distance = outside + inside - r

                alpha = min(1.0, max(0.0, 0.5 - distance))

                image.setXelA(x, y, 1, 1, 1, alpha)

        texture = Texture("securedrive-rounded-button")
        texture.load(image)
        texture.setMinfilter(Texture.FTLinear)
        texture.setMagfilter(Texture.FTLinear)
        texture.setWrapU(Texture.WMClamp)
        texture.setWrapV(Texture.WMClamp)

        return texture

    def create_panel(self):

        aspect = self.get_aspect_ratio()

        half_width = aspect * 0.98

        self.panel = DirectFrame(
            parent=self.engine.a2dBottomCenter,
            frameColor=self.COLORS["transparent"],
            frameSize=(-half_width, half_width, 0, 0.01),
            pos=(0, 0, 0)
        )

        column_width = (2.0 * half_width) / 5.0
        gap = 0.024

        button_half_width = column_width / 2.0 - gap / 2.0
        button_half_height = 0.040
        corner_radius = 0.020

        button_texture = self.create_button_texture(
            button_half_width * 2.0,
            button_half_height * 2.0,
            corner_radius
        )

        stop_half_width = 0.19
        stop_half_height = 0.036

        stop_texture = self.create_button_texture(
            stop_half_width * 2.0,
            stop_half_height * 2.0,
            corner_radius
        )

        # Title (top-left, above the first row)
        separator = "·" if self.font is not None else "|"

        self.title = DirectLabel(
            parent=self.panel,
            text=f"SECUREDRIVE-AI {separator} ATTACK CONTROLS",
            text_scale=0.027,
            text_align=TextNode.ALeft,
            text_fg=self.COLORS["title"],
            text_shadow=self.COLORS["shadow"],
            frameColor=self.COLORS["transparent"],
            frameSize=(0, 0.1, -0.02, 0.02),
            pos=(-half_width + gap / 2.0 + 0.005, 0, 0.300),
            **self.font_args()
        )

        for index, (label, attack_name) in enumerate(
            self.ATTACKS
        ):

            row = index // 5
            column = index % 5

            x = -half_width + column_width * (column + 0.5)

            z = 0.235 if row == 0 else 0.135

            button = DirectButton(
                parent=self.panel,
                text=label,
                text_scale=0.029,
                text_align=TextNode.ACenter,
                text_fg=self.COLORS["text"],
                text_shadow=self.COLORS["shadow"],
                text_pos=(0, -0.010),
                frameColor=self.COLORS["button"],
                frameTexture=button_texture,
                frameSize=(
                    -button_half_width,
                    button_half_width,
                    -button_half_height,
                    button_half_height
                ),
                relief=DGG.FLAT,
                pos=(x, 0, z),
                command=self.activate_attack,
                extraArgs=[attack_name],
                rolloverSound=None,
                clickSound=None,
                **self.font_args()
            )

            button.setTransparency(TransparencyAttrib.MAlpha)

            button.bind(
                DGG.WITHIN,
                self.on_button_hover,
                [button, True]
            )

            button.bind(
                DGG.WITHOUT,
                self.on_button_hover,
                [button, False]
            )

            self.buttons.append(button)
            self.button_attack_names.append(attack_name)

        self.stop_button = DirectButton(
            parent=self.panel,
            text="STOP ATTACK",
            text_scale=0.029,
            text_align=TextNode.ACenter,
            text_fg=(1, 1, 1, 1),
            text_shadow=self.COLORS["shadow"],
            text_pos=(0, -0.010),
            frameColor=self.COLORS["stop"],
            frameTexture=stop_texture,
            frameSize=(
                -stop_half_width,
                stop_half_width,
                -stop_half_height,
                stop_half_height
            ),
            relief=DGG.FLAT,
            pos=(0, 0, 0.048),
            command=self.stop_attack,
            rolloverSound=None,
            clickSound=None,
            **self.font_args()
        )

        self.stop_button.setTransparency(TransparencyAttrib.MAlpha)

        self.stop_button.bind(
            DGG.WITHIN,
            self.on_stop_hover,
            [True]
        )

        self.stop_button.bind(
            DGG.WITHOUT,
            self.on_stop_hover,
            [False]
        )

    def on_button_hover(self, button, hovering, *args):

        index = self.buttons.index(button)

        if hovering:
            self.hovered.add(index)
        else:
            self.hovered.discard(index)

        self.update_status()

    def on_stop_hover(self, hovering, *args):

        self.stop_button["frameColor"] = (
            self.COLORS["stop_hover"]
            if hovering
            else self.COLORS["stop"]
        )

    def refresh_attack_buttons(self, task):

        self.update_status()
        return Task.cont

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

        for index, (key, text) in enumerate(telemetry_items):

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

            speed_ms = float(vehicle.speed) / 3.6

            tire_radius = float(
                getattr(vehicle, "TIRE_RADIUS", 0.34)
            )

            if tire_radius <= 0:
                tire_radius = 0.34

            circumference = 2 * math.pi * tire_radius

            return max(
                0.0,
                speed_ms / circumference * 60.0
            )

        except Exception:
            return 0.0

    def update_realtime_telemetry(self, task):

        vehicle = self.find_vehicle()

        if vehicle is None:
            return Task.cont

        try:
            speed = float(vehicle.speed)
        except Exception:
            speed = 0.0

        try:
            steering = float(vehicle.steering)
        except Exception:
            steering = 0.0

        try:
            throttle_brake = float(vehicle.throttle_brake)
        except Exception:
            throttle_brake = 0.0

        throttle = max(0.0, throttle_brake)
        brake = max(0.0, -throttle_brake)

        rpm = self.calculate_wheel_rpm(vehicle)

        current_time = float(task.time)
        acceleration = 0.0

        if (
            self.previous_speed is not None
            and self.previous_time is not None
        ):

            delta_time = max(
                current_time - self.previous_time,
                0.001
            )

            acceleration = (
                (speed - self.previous_speed) / 3.6
            ) / delta_time

        self.previous_speed = speed
        self.previous_time = current_time

        try:
            heading = math.degrees(
                float(vehicle.heading_theta)
            )
        except Exception:
            heading = 0.0

        try:
            position_x = float(vehicle.position[0])
            position_y = float(vehicle.position[1])
        except Exception:
            position_x = 0.0
            position_y = 0.0

        lane_offset = 0.0

        try:

            navigation = vehicle.navigation

            if navigation is not None:

                lanes = navigation.current_ref_lanes

                if lanes:

                    _, lateral = lanes[0].local_coordinates(
                        vehicle.position
                    )

                    lane_offset = float(lateral)

        except Exception:
            lane_offset = 0.0

        if self.attack_engine.is_recovering():
            status = "RECOVERY"

        elif self.attack_engine.is_attack_active():
            status = "ATTACK ACTIVE"

        else:
            status = "NORMAL"

        self.telemetry_labels["speed"]["text"] = (
            f"SPEED       : {speed:6.2f} km/h"
        )

        self.telemetry_labels["rpm"]["text"] = (
            f"WHEEL RPM   : {rpm:6.0f} RPM"
        )

        self.telemetry_labels["steering"]["text"] = (
            f"STEERING    : {steering:6.2f}"
        )

        self.telemetry_labels["throttle"]["text"] = (
            f"THROTTLE    : {throttle:6.2f}"
        )

        self.telemetry_labels["brake"]["text"] = (
            f"BRAKE       : {brake:6.2f}"
        )

        self.telemetry_labels["acceleration"]["text"] = (
            f"ACCELERATION: {acceleration:6.2f} m/s²"
        )

        self.telemetry_labels["heading"]["text"] = (
            f"HEADING     : {heading:6.1f}°"
        )

        self.telemetry_labels["position"]["text"] = (
            f"POSITION    : X {position_x:7.1f} | "
            f"Y {position_y:7.1f}"
        )

        self.telemetry_labels["lane"]["text"] = (
            f"LANE OFFSET : {lane_offset:6.2f} m"
        )

        self.telemetry_labels["status"]["text"] = (
            f"STATUS      : {status}"
        )

        return Task.cont

    def activate_attack(self, attack_name):

        from attacks.attack_engine import AttackType

        attack_type = AttackType(attack_name)

        self.attack_engine.activate(attack_type)

        self.previous_speed = None
        self.previous_time = None

        self.update_status()

        print("[SECUREDRIVE-AI] ATTACK ACTIVATED:", attack_name)

    def stop_attack(self):

        self.attack_engine.deactivate(recover=True)

        self.update_status()

        print("[SECUREDRIVE-AI] ATTACK STOPPED — RECOVERY REQUESTED")

    def update_status(self):

        active = self.attack_engine.is_attack_active()
        active_name = self.attack_engine.get_attack_name()

        for index, (button, attack_name) in enumerate(
            zip(self.buttons, self.button_attack_names)
        ):

            hovering = index in self.hovered

            if active and attack_name == active_name:

                button["frameColor"] = (
                    self.COLORS["active_hover"]
                    if hovering
                    else self.COLORS["active"]
                )

            else:

                button["frameColor"] = (
                    self.COLORS["hover"]
                    if hovering
                    else self.COLORS["button"]
                )

    def destroy(self):

        self.engine.taskMgr.remove(
            "securedrive-realtime-telemetry"
        )

        self.engine.taskMgr.remove(
            "securedrive-refresh-attack-buttons"
        )

        if self.panel is not None:
            self.panel.destroy()
            self.panel = None

        if self.telemetry_panel is not None:
            self.telemetry_panel.destroy()
            self.telemetry_panel = None

        self.buttons.clear()
        self.button_attack_names.clear()
        self.telemetry_labels.clear()