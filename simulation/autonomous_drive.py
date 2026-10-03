
import time

from metadrive import MetaDriveEnv
from metadrive.policy.env_input_policy import EnvInputPolicy

from simulation.custom_autopilot import SecureDriveAutopilot
from simulation.telemetry import collect_telemetry
from simulation.telemetry_logger import TelemetryLogger
from simulation.securedrive_traffic_model import SecureDriveTrafficModel

from attacks.attack_engine import AttackEngine
from gui.attack_panel import AttackPanel


CONFIG = {
    "use_render": True,
    "manual_control": False,
    "agent_policy": EnvInputPolicy,
    "traffic_density": 0.10,
    "random_traffic": True,
    "random_agent_model": False,
    "random_lane_width": False,
    "random_lane_num": False,
    "on_continuous_line_done": False,
    "out_of_route_done": False,
    "out_of_road_done": False,
    "crash_vehicle_done": False,
    "crash_object_done": False,
    "vehicle_config": {
        "show_lidar": False,
        "show_side_detector": False,
        "show_lane_line_detector": False,
    },
    "traffic_vehicle_config": {
        "show_lidar": False,
        "show_side_detector": False,
        "show_lane_line_detector": False,
    },
    "map": 4,
    "start_seed": 10,
    "num_scenarios": 10000,
    "horizon": 10000,
}


def clamp_action(action):
    steering = max(-1.0, min(1.0, float(action[0])))
    throttle_brake = max(-1.0, min(1.0, float(action[1])))
    return [steering, throttle_brake]


def main():
    env = None
    telemetry_logger = None

    print()
    print("=" * 64)
    print("SECUREDRIVE-AI")
    print("AUTONOMOUS VEHICLE SECURITY SIMULATOR")
    print("=" * 64)

    try:
        print("[SYSTEM] Initializing MetaDrive...")
        env = MetaDriveEnv(CONFIG)

        autopilot = SecureDriveAutopilot()
        traffic_model = SecureDriveTrafficModel()
        attack_engine = AttackEngine()
        telemetry_logger = TelemetryLogger()

        obs, info = env.reset()

        autopilot.reset()
        traffic_model.reset()
        attack_engine.deactivate(recover=False)

        attack_panel = AttackPanel(
            env.engine,
            attack_engine
        )

        step_count = 0
        last_control_action = [0.0, 0.0]
        recovery_mode_last_frame = False
        controller_error_logged = False

        print("[SYSTEM] MetaDrive initialized.")
        print("[SYSTEM] SecureDrive Autopilot READY")
        print("[SYSTEM] Traffic Model READY")
        print("[SYSTEM] Attack Engine READY")
        print("[SYSTEM] Telemetry Logger READY")
        print("[SYSTEM] Attack GUI READY")
        print("[SYSTEM] Autonomous Control ACTIVE")
        print()

        def compute_control_action():
            nonlocal recovery_mode_last_frame
            nonlocal controller_error_logged

            vehicle = env.agent

            recovering = attack_engine.is_recovering()

            if recovering:
                if not recovery_mode_last_frame:
                    traffic_model.reset()
                    autopilot.reset_recovery()

                    print(
                        "[RECOVERY] SecureDrive recovery controller engaged."
                    )

                action = autopilot.compute_action(
                    vehicle,
                    recovery=True
                )

                if autopilot.recovery_complete:
                    attack_engine.complete_recovery()
                    autopilot.reset_recovery()
                    traffic_model.reset()

                    recovery_mode_last_frame = False

                    print(
                        "[RECOVERY] Vehicle alignment stabilized."
                    )
                else:
                    recovery_mode_last_frame = True

                return clamp_action(action)

            recovery_mode_last_frame = False

            base_action = autopilot.compute_action(
                vehicle,
                recovery=False
            )

            active_attack = attack_engine.get_active_attack()

            if active_attack in AttackEngine.PHYSICAL_ATTACKS:
                action = attack_engine.modify_action(
                    base_action,
                    vehicle=vehicle
                )

                controller_error_logged = False
                return clamp_action(action)

            try:
                traffic_action = traffic_model.compute_action(
                    vehicle,
                    base_action
                )

                if (
                    traffic_model.lane_change_active
                    and traffic_model.target_lane is not None
                ):
                    action = autopilot.compute_lane_change_action(
                        vehicle,
                        traffic_model.target_lane,
                        base_throttle=traffic_action[1]
                    )
                else:
                    action = traffic_action

                controller_error_logged = False

            except Exception as error:
                if not controller_error_logged:
                    print(
                        f"[CONTROLLER] Traffic model error: {error}"
                    )
                    print(
                        "[CONTROLLER] Falling back to autopilot control."
                    )
                    controller_error_logged = True

                action = base_action

            action = attack_engine.modify_action(
                action,
                vehicle=vehicle
            )

            return clamp_action(action)

        while True:
            step_count += 1

            action = compute_control_action()
            last_control_action = action

            obs, reward, terminated, truncated, info = env.step(action)

            vehicle = env.agent

            telemetry = collect_telemetry(
                vehicle=vehicle,
                step=step_count,
                action=last_control_action,
                attack=attack_engine.get_attack_name()
            )

            telemetry = attack_engine.modify_telemetry(
                telemetry
            )

            telemetry_logger.log(telemetry)

            if step_count % 30 == 0:
                status = traffic_model.get_status()

                front_distance = status.get("front_distance")

                if front_distance is None:
                    traffic_status = "CLEAR"
                elif front_distance == float("inf"):
                    traffic_status = "CLEAR"
                else:
                    traffic_status = f"{front_distance:.1f} m"

                print(
                    f"[STEP {step_count:06d}] "
                    f"Speed: {vehicle.speed * 3.6:6.2f} km/h | "
                    f"Steering: {action[0]:6.2f} | "
                    f"Throttle/Brake: {action[1]:6.2f} | "
                    f"Attack: {attack_engine.get_attack_name():20} | "
                    f"Traffic: {traffic_status:10} | "
                    f"Decision: {status.get('decision', 'UNKNOWN')}"
                )

            if terminated or truncated:
                print()
                print("[SYSTEM] Episode ended.")
                print(
                    f"[SYSTEM] Terminated: {terminated} | "
                    f"Truncated: {truncated}"
                )

                obs, info = env.reset()

                step_count = 0
                last_control_action = [0.0, 0.0]
                recovery_mode_last_frame = False
                controller_error_logged = False

                autopilot.reset()
                traffic_model.reset()
                attack_engine.deactivate(recover=False)

                print("[SYSTEM] New episode initialized.")
                print()

    except KeyboardInterrupt:
        print()
        print("[SYSTEM] Shutdown requested.")

    finally:
        if telemetry_logger is not None:
            telemetry_logger.close()

        if env is not None:
            env.close()

        print("[SYSTEM] SecureDrive-AI shutdown complete.")


if __name__ == "__main__":
    main()