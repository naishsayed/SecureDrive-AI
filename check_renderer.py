from metadrive import MetaDriveEnv


CONFIG = dict(
    use_render=True,
    manual_control=False,
    num_scenarios=1,
)


env = MetaDriveEnv(CONFIG)

try:
    env.reset(seed=0)

    gsg = env.engine.win.get_gsg()

    print("\n" + "=" * 70)
    print("METADRIVE GRAPHICS INFORMATION")
    print("=" * 70)

    print("Graphics Pipe:")
    print(env.engine.win.get_pipe().get_interface_name())

    print("\nGPU Vendor:")
    print(gsg.get_driver_vendor())

    print("\nGPU Renderer:")
    print(gsg.get_driver_renderer())

    print("\nDriver Version:")
    print(gsg.get_driver_version())

    print("\nWindow Size:")
    print(
        env.engine.win.get_x_size(),
        "x",
        env.engine.win.get_y_size()
    )

    print("=" * 70)

    input("\nPress ENTER to close...")

finally:
    env.close()