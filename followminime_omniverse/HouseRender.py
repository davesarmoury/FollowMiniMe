import omni.replicator.core as rep

with rep.new_layer():
    # Define paths for the character, the props, the environment and the surface where the assets will be scattered in.

    TRACKER = 'omniverse://localhost/Projects/FollowMe/TrackerTarget.usdc'
    SURFACE = 'omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/display_riser.usd'
    # Environments.  Large so not swapping
    PUDDLES = 'omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Outdoor/Puddles.usd'
    PARK='omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/abandoned_parking_lot.usd'
    WAREHOUSE='omniverse://localhost/NVIDIA/Assets/XR/Stages/Indoor/Warehouse_XR.usd'
    INDOOR='omniverse://localhost/NVIDIA/Assets/XR/Stages/Indoor/Modern_House_XR.usd'

    ENVS=INDOOR

    def tracker():
        tracker = rep.create.from_usd(TRACKER, semantics=[('class', 'tracker')])

        with tracker:
            rep.modify.pose(
                position=rep.distribution.uniform((-200, 100, -200), (200, 200, 200)),
                rotation=rep.distribution.uniform((-90,-90, 0), (-90, 90, 0)),
            )
        return tracker

    # Register randomization
    rep.randomizer.register(tracker)

    # Rotate the world so it is correct
    env = rep.create.from_usd(ENVS)
    with env:
        rep.modify.pose(
            position=(-450,0,550),
            rotation=(-90,0, 0),
        )
    # Make the surface tiny so you can't see it in renders, and move it around so the shots are more varied
    surface = rep.create.from_usd(SURFACE)
    with surface:
        rep.modify.pose(
            position=rep.distribution.uniform((0, 100,  0), (0, 200, 0)),
            scale=(0.001, 0.001, 0.001),
        )

    # Setup camera and attach it to render product
    camera = rep.create.camera(
        focus_distance=600,
        f_stop=1.8
    )
    render_product = rep.create.render_product(camera, resolution=(1920, 1080))

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir="Kitchen", rgb=True, bounding_box_2d_tight=True)
    writer.attach([render_product])

    with rep.trigger.on_frame(num_frames=100):
        rep.randomizer.tracker()
        with camera:
            rep.modify.pose(position=rep.distribution.uniform((-200, 100, -200), (200, 200, 200)), look_at=surface)