
import time

from metadrive import MetaDriveEnv

from simulation.custom_autopilot import SecureDriveAutopilot
from simulation.telemetry import collect_telemetry
from simulation.telemetry_logger import TelemetryLogger

from attacks.attack_engine import AttackEngine
from gui.attack_panel import AttackPanel


CONFIG = {
    "use_render": True,
    "manual_control": False,
    "traffic_density": 0.1,
    "random_traffic": True,
    "random_agent_model": False,
    "random_lane_width": True,
    "random_lane_num": True,
    "on_continuous_line_done": False,
    "out_of_route_done": False,
    "out_of_road_done": False,
    "crash_vehicle_done": False,
    "vehicle_config": {
        "show_lidar": False,
        "show_side_detector": False,
        "show_lane_line_detector": False,
    },
    "map": 4,
    "start_seed": 10,
}


def main():

    print()
    print("=" * 60)
    print("SECUREDRIVE AI")
    print("SECURE AUTONOMOUS VEHICLE SIMULATOR")
    print("=" * 60)
    print()

    print("[SYSTEM] Initializing MetaDrive...")

    env = MetaDriveEnv(CONFIG)

    print("[SYSTEM] MetaDrive initialized.")

    autopilot = SecureDriveAutopilot()
    attack_engine = AttackEngine()
    telemetry_logger = TelemetryLogger()

    print("[SYSTEM] SecureDrive Autopilot READY")
    print("[SYSTEM] Attack Engine READY")
    print("[SYSTEM] Telemetry Logger READY")

    obs, info = env.reset()

    autopilot.reset()
    attack_engine.deactivate(recover=False)

    last_control_action = [0.0, 0.0]
    step_count = 0

    print()
    print("[SYSTEM] Autonomous Driving ON")
    print("[SYSTEM] Direct Action Control ON")
    print("[SYSTEM] Attack Engine READY")
    print()

    attack_panel = AttackPanel(
        env.engine,
        attack_engine
    )

    print("[SYSTEM] Attack GUI INITIALIZING")
    print("[SYSTEM] Graphics Quality PRESERVED")
    print()

    def securedrive_act():

        nonlocal last_control_action

        vehicle = env.agent

        if vehicle is None:

            last_control_action = [0.0, 0.0]

            return last_control_action

        normal_action = autopilot.compute_action(vehicle)

        if attack_engine.is_attack_active():

            final_action = attack_engine.modify_action(
                normal_action,
                vehicle=vehicle
            )

        else:

            final_action = normal_action

        final_action = [
            max(-1.0, min(1.0, float(final_action[0]))),
            max(-1.0, min(1.0, float(final_action[1])))
        ]

        last_control_action = final_action

        return final_action

    try:

        while True:

            step_count += 1

            action = securedrive_act()

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            vehicle = env.agent

            telemetry = collect_telemetry(
                vehicle=vehicle,
                step=step_count,
                action=last_control_action,
                attack=attack_engine.get_attack_name()
            )

            telemetry_logger.log(telemetry)

            if step_count % 10 == 0:

                attack_name = attack_engine.get_attack_name()

                speed_display = (
                    vehicle.speed * 3.6
                    if vehicle is not None
                    else 0.0
                )

                print(
                    f"[STEP {step_count:05d}] "
                    f"Speed: {speed_display:5.1f} km/h | "
                    f"Steering: {last_control_action[0]:+.2f} | "
                    f"Throttle/Brake: {last_control_action[1]:+.2f} | "
                    f"Attack: {attack_name:<22}"
                )

            if terminated or truncated:

                print()
                print("[SYSTEM] Episode finished.")
                print("[SYSTEM] Resetting vehicle...")

                obs, info = env.reset()

                step_count = 0

                autopilot.reset()

                attack_engine.deactivate(
                    recover=False
                )

                last_control_action = [0.0, 0.0]

                time.sleep(1)

    except KeyboardInterrupt:

        print()
        print("[SYSTEM] Simulation stopped by user.")

    finally:

        telemetry_logger.close()

        env.close()

        print("[SYSTEM] SecureDrive AI shutdown complete.")


if __name__ == "__main__":

    main()