from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class VehicleTelemetry:

    timestamp: str

    step: int

    speed: float

    acceleration: float

    steering: float

    throttle: float

    brake: float

    position_x: float

    position_y: float

    position_z: float

    heading: float

    crashed: bool

    attack: str

    recovery: bool

    on_lane: bool

    lateral_offset: float

    heading_error: float

    target_speed: float

    def to_dict(self):

        return asdict(self)


def collect_telemetry(
    vehicle,
    step,
    action,
    attack="NONE",
    previous_speed=0.0,
    delta_time=0.1
):

    position = vehicle.position

    position_x = float(
        position[0]
    )

    position_y = float(
        position[1]
    )

    if len(position) >= 3:

        position_z = float(
            position[2]
        )

    else:

        position_z = 0.0

    current_speed = float(
        vehicle.speed
    )

    steering = float(
        action[0]
    )

    throttle_input = float(
        action[1]
    )

    throttle = max(
        0.0,
        throttle_input
    )

    brake = max(
        0.0,
        -throttle_input
    )

    current_speed_ms = (
        current_speed / 3.6
    )

    previous_speed_ms = (
        previous_speed / 3.6
    )

    if delta_time > 0:

        acceleration = (
            current_speed_ms
            - previous_speed_ms
        ) / delta_time

    else:

        acceleration = 0.0

    crashed = False

    try:

        crashed = bool(
            vehicle.crash_vehicle
        )

    except Exception:

        crashed = False

    on_lane = True

    try:

        on_lane = bool(
            vehicle.on_lane
        )

    except Exception:

        on_lane = True

    lateral_offset = 0.0

    heading_error = 0.0

    target_speed = 0.0

    try:

        lane = vehicle.navigation.get_current_lane(
            vehicle
        )

        if lane is not None:

            longitudinal, lateral = (
                lane.local_coordinates(
                    vehicle.position
                )
            )

            lateral_offset = float(
                lateral
            )

            try:

                lane_heading = float(
                    lane.heading_theta_at(
                        longitudinal
                    )
                )

            except Exception:

                lane_heading = float(
                    lane.heading_at(
                        longitudinal
                    )
                )

            vehicle_heading = float(
                vehicle.heading_theta
            )

            heading_error = (
                lane_heading
                - vehicle_heading
            )

            while heading_error > 3.14159265359:

                heading_error -= (
                    2.0 * 3.14159265359
                )

            while heading_error < -3.14159265359:

                heading_error += (
                    2.0 * 3.14159265359
                )

            heading_error = float(
                heading_error
            )

    except Exception:

        lateral_offset = 0.0

        heading_error = 0.0

    try:

        target_speed = float(
            vehicle.navigation.current_lane.width
        )

    except Exception:

        target_speed = 0.0

    return VehicleTelemetry(

        timestamp=datetime.now().isoformat(),

        step=step,

        speed=current_speed,

        acceleration=acceleration,

        steering=steering,

        throttle=throttle,

        brake=brake,

        position_x=position_x,

        position_y=position_y,

        position_z=position_z,

        heading=float(
            vehicle.heading_theta
        ),

        crashed=crashed,

        attack=attack,

        recovery=(
            attack == "RECOVERY"
        ),

        on_lane=on_lane,

        lateral_offset=lateral_offset,

        heading_error=heading_error,

        target_speed=target_speed
    )