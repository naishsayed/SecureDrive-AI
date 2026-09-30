from metadrive.envs import MetaDriveEnv
import time


def main():
    config = {
        "use_render": True,
        "map": "S",
        "num_scenarios": 1,
        "log_level": 50
    }

    env = MetaDriveEnv(config)

    try:
        obs, info = env.reset()

        for step in range(1000):
            action = [0.0, 0.5]

            obs, reward, terminated, truncated, info = env.step(action)

            vehicle = env.vehicle

            speed = vehicle.speed
            position = vehicle.position
            heading = vehicle.heading_theta
            steering = action[0]
            throttle = action[1]

            print(
                f"\r"
                f"Step: {step:04d} | "
                f"Speed: {speed:6.2f} km/h | "
                f"Throttle: {throttle:5.2f} | "
                f"Steering: {steering:5.2f} | "
                f"X: {position[0]:8.2f} | "
                f"Y: {position[1]:8.2f} | "
                f"Heading: {heading:6.2f}",
                end=""
            )

            time.sleep(0.02)

            if terminated or truncated:
                obs, info = env.reset()

    finally:
        env.close()


if __name__ == "__main__":
    main()