from metadrive import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy

from simulation.custom_autopilot import SecureDriveAutopilot
from simulation.telemetry import collect_telemetry
from simulation.telemetry_logger import TelemetryLogger

from attacks.attack_engine import AttackEngine
from gui.attack_panel import AttackPanel


CONFIG = {

    "use_render": True,

    "manual_control": False,

    "agent_policy": ExpertPolicy,

    "traffic_density": 0.0,

    "num_scenarios": 10000,

    "random_agent_model": False,

    "random_lane_width": True,

    "random_lane_num": True,

    "on_continuous_line_done": False,

    "out_of_route_done": False,

    "out_of_road_done": False,

    "vehicle_config": {

        "show_lidar": False,

        "show_side_detector": False,

        "show_lane_line_detector": False,

    },

    "map": 4,

    "start_seed": 10,
}


def run_autonomous_drive():

    print()
    print("=" * 70)
    print("SECUREDRIVE-AI")
    print("SECURE AUTONOMOUS VEHICLE SIMULATOR")
    print("=" * 70)

    print()
    print("[SYSTEM] Initializing MetaDrive...")

    env = MetaDriveEnv(
        config=CONFIG
    )

    autopilot = SecureDriveAutopilot()

    attack_engine = AttackEngine()

    telemetry_logger = TelemetryLogger(
        "data/raw/vehicle_telemetry.csv"
    )

    attack_panel = None

    previous_speed = 0.0

    step_count = 0

    last_control_action = [
        0.0,
        0.0
    ]

    try:

        print()
        print("[SYSTEM] Resetting simulation...")

        env.reset()

        autopilot.reset()

        attack_engine.deactivate(
            recover=False
        )

        print()
        print("=" * 70)
        print("[SECUREDRIVE-AI] AUTONOMOUS DRIVING ENABLED")
        print("[SECUREDRIVE-AI] SecureDrive Autopilot ACTIVE")
        print("[SECUREDRIVE-AI] Attack Engine READY")
        print("[SECUREDRIVE-AI] Telemetry Logger ACTIVE")
        print("=" * 70)

        print()

        agent_id = env.agent.id

        policy = env.engine.get_policy(
            agent_id
        )

        if policy is None:

            raise RuntimeError(
                "SecureDrive-AI could not obtain "
                "the MetaDrive policy for the vehicle."
            )

        print(
            "[SYSTEM] MetaDrive policy acquired successfully."
        )

        print(
            f"[SYSTEM] Policy type: "
            f"{type(policy).__name__}"
        )

        def securedrive_act(
            agent_id=None
        ):

            nonlocal last_control_action

            recovery = (
                attack_engine.is_recovering()
            )

            base_action = autopilot.compute_action(
                env.agent,
                recovery=recovery
            )

            final_action = attack_engine.modify_action(
                base_action,
                vehicle=env.agent
            )

            final_action = [

                float(final_action[0]),

                float(final_action[1])

            ]

            last_control_action = final_action

            return final_action

        policy.act = securedrive_act

        print(
            "[SYSTEM] SecureDrive Autopilot "
            "hook installed."
        )

        attack_panel = AttackPanel(
            env.engine,
            attack_engine
        )

        while True:

            step_count += 1

            vehicle = env.agent

            current_speed = float(
                vehicle.speed
            )

            recovery_before_step = (
                attack_engine.is_recovering()
            )

            obs, reward, terminated, truncated, info = env.step(
                [0.0, 0.0]
            )

            telemetry = collect_telemetry(
                vehicle=vehicle,
                step=step_count,
                action=last_control_action,
                attack=attack_engine.get_attack_name(),
                previous_speed=previous_speed,
                delta_time=0.1
            )

            telemetry = attack_engine.modify_telemetry(
                telemetry
            )

            telemetry_logger.log(
                telemetry
            )

            previous_speed = current_speed

            if (
                recovery_before_step
                and autopilot.recovery_complete
            ):

                attack_engine.complete_recovery()

                attack_panel.update_status()

            if step_count % 10 == 0:

                attack_name = (
                    attack_engine.get_attack_name()
                )

                status = "NORMAL"

                if attack_engine.is_recovering():

                    status = "RECOVERING"

                elif attack_engine.is_attack_active():

                    status = "ATTACK ACTIVE"

                print(
                    f"[TELEMETRY] "
                    f"Step={step_count:05d} | "
                    f"Speed={telemetry.speed:6.2f} km/h | "
                    f"Steering={telemetry.steering:6.2f} | "
                    f"Throttle={telemetry.throttle:5.2f} | "
                    f"Brake={telemetry.brake:5.2f} | "
                    f"Attack={attack_name} | "
                    f"Status={status}"
                )

            if terminated or truncated:

                print()
                print("=" * 70)
                print(
                    "[SECUREDRIVE-AI] EPISODE ENDED"
                )

                print(
                    f"[SECUREDRIVE-AI] "
                    f"Step: {step_count}"
                )

                print(
                    f"[SECUREDRIVE-AI] "
                    f"Speed: {vehicle.speed:.2f} km/h"
                )

                print("=" * 70)

                print()
                print(
                    "[SECUREDRIVE-AI] Starting new episode..."
                )

                env.reset()

                autopilot.reset()

                attack_engine.deactivate(
                    recover=False
                )

                previous_speed = 0.0

                last_control_action = [
                    0.0,
                    0.0
                ]

                attack_panel.update_status()

                continue

    except KeyboardInterrupt:

        print()
        print("=" * 70)
        print("[SECUREDRIVE-AI] Simulation stopped by user")
        print("=" * 70)

    except Exception as error:

        print()
        print("=" * 70)
        print("[SECUREDRIVE-AI] ERROR")
        print("=" * 70)

        print(
            f"{type(error).__name__}: {error}"
        )

        raise

    finally:

        print()
        print("[SYSTEM] Shutting down...")

        if attack_panel is not None:

            attack_panel.destroy()

        telemetry_logger.close()

        env.close()

        print(
            "[SYSTEM] Telemetry saved to "
            "data/raw/vehicle_telemetry.csv"
        )

        print(
            "[SYSTEM] SecureDrive-AI shutdown complete."
        )


if __name__ == "__main__":

    run_autonomous_drive()