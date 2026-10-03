
import time

from metadrive import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy

from simulation.custom_autopilot import SecureDriveAutopilot
from simulation.telemetry import collect_telemetry
from simulation.telemetry_logger import TelemetryLogger
from simulation.securedrive_traffic_model import SecureDriveTrafficModel

from attacks.attack_engine import AttackEngine
from gui.attack_panel import AttackPanel


CONFIG = {
    "use_render": True,
    "manual_control": False,
    "agent_policy": ExpertPolicy,
    "traffic_density": 0.15,
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
    traffic_model = SecureDriveTrafficModel()
    attack_engine = AttackEngine()
    telemetry_logger = TelemetryLogger()

    traffic_model.set_attack_engine(attack_engine)

    print("[SYSTEM] SecureDrive Autopilot READY")
    print("[SYSTEM] SecureDrive Traffic Model READY")
    print("[SYSTEM] Attack Engine READY")
    print("[SYSTEM] Telemetry Logger READY")

    obs, info = env.reset()

    autopilot.reset()
    traffic_model.reset()
    attack_engine.deactivate(recover=False)

    last_control_action = [0.0, 0.0]
    step_count = 0

    print()
    print("[SYSTEM] Autonomous Driving ON")
    print("[SYSTEM] SecureDrive Traffic Model ON")
    print("[SYSTEM] Attack Engine READY")
    print()

    def securedrive_act():

        nonlocal last_control_action

        vehicle = env.agent

        if vehicle is None:
            last_control_action = [0.0, 0.0]
            return last_control_action

        recovery = attack_engine.is_recovering()

        base_action = autopilot.compute_action(
            vehicle,
            recovery=recovery
        )

        attack_active = attack_engine.is_attack_active()
        active_attack = attack_engine.get_active_attack()

        physical_attack = (
            active_attack in AttackEngine.PHYSICAL_ATTACKS
        )

        traffic_model_enabled = (
            not recovery
            and not (attack_active and physical_attack)
        )

        normal_action = base_action

        if traffic_model_enabled:

            traffic_action = traffic_model.compute_action(
                vehicle,
                base_action
            )

            if (
                traffic_model.lane_change_active
                and traffic_model.target_lane is not None
            ):

                normal_action = autopilot.compute_lane_change_action(
                    vehicle,
                    traffic_model.target_lane,
                    base_throttle=traffic_action[1]
                )

                if traffic_action[1] < 0.0:
                    normal_action[1] = traffic_action[1]

            else:
                normal_action = traffic_action

        final_action = attack_engine.modify_action(
            normal_action,
            vehicle=vehicle
        )

        final_action = [
            max(-1.0, min(1.0, float(final_action[0]))),
            max(-1.0, min(1.0, float(final_action[1])))
        ]

        last_control_action = final_action

        return final_action

    attack_panel = AttackPanel(
        env.engine,
        attack_engine
    )

    print("[SYSTEM] Attack GUI INITIALIZING")
    print("[SYSTEM] Graphics Quality PRESERVED")
    print()

    try:

        while True:

            step_count += 1

            action = securedrive_act()

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            vehicle = env.agent

            if (
                attack_engine.is_recovering()
                and autopilot.recovery_complete
            ):

                print()
                print("=" * 60)
                print("[SYSTEM] VEHICLE RECOVERY COMPLETED")
                print("[SYSTEM] Returning to normal autonomous control")
                print("=" * 60)

                attack_engine.complete_recovery()
                autopilot.reset_recovery()
                traffic_model.reset()

            telemetry = collect_telemetry(
                vehicle=vehicle,
                step=step_count,
                action=last_control_action,
                attack=attack_engine.get_attack_name()
            )

            telemetry_logger.log(telemetry)

            if step_count % 10 == 0:

                status = traffic_model.get_status()

                attack_name = attack_engine.get_attack_name()

                if status["front_vehicle_detected"]:

                    traffic_status = (
                        f"NPC {status['front_distance']:.1f}m"
                    )

                else:
                    traffic_status = "CLEAR"

                decision = status["decision"]
                lane_change = status["lane_change_active"]

                print(
                    f"[STEP {step_count:05d}] "
                    f"Speed: {vehicle.speed * 3.6:5.1f} km/h | "
                    f"Attack: {attack_name:<18} | "
                    f"Traffic: {traffic_status:<12} | "
                    f"Decision: {decision:<28} | "
                    f"Lane Change: {str(lane_change):<5}"
                )

            if terminated or truncated:

                print()
                print("[SYSTEM] Episode finished.")
                print("[SYSTEM] Resetting vehicle...")

                obs, info = env.reset()

                step_count = 0

                autopilot.reset()
                traffic_model.reset()
                attack_engine.deactivate(recover=False)

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