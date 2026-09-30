import csv
import os


class TelemetryLogger:

    def __init__(
        self,
        filepath="data/raw/vehicle_telemetry.csv"
    ):

        self.filepath = filepath

        directory = os.path.dirname(
            self.filepath
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True
            )

        self.file = open(
            self.filepath,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(
            self.file
        )

        self.writer.writerow([

            "timestamp",

            "step",

            "speed",

            "acceleration",

            "steering",

            "throttle",

            "brake",

            "position_x",

            "position_y",

            "position_z",

            "heading",

            "crashed",

            "attack",

            "recovery",

            "on_lane",

            "lateral_offset",

            "heading_error",

            "target_speed"

        ])

        self.file.flush()

    def log(self, telemetry):

        self.writer.writerow([

            telemetry.timestamp,

            telemetry.step,

            telemetry.speed,

            telemetry.acceleration,

            telemetry.steering,

            telemetry.throttle,

            telemetry.brake,

            telemetry.position_x,

            telemetry.position_y,

            telemetry.position_z,

            telemetry.heading,

            telemetry.crashed,

            telemetry.attack,

            telemetry.recovery,

            telemetry.on_lane,

            telemetry.lateral_offset,

            telemetry.heading_error,

            telemetry.target_speed

        ])

        self.file.flush()

    def close(self):

        if not self.file.closed:

            self.file.close()