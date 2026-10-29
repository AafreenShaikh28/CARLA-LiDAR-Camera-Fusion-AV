"""
agent.py -- the integration loop from Section 7 of the project plan.

Design: every perception stage (2D detection, lane detection, segmentation,
3D fusion+detection) is a small class with a .predict(...) method. Each one
has TWO implementations behind one flag:

  oracle mode  (CFG['use_oracle_perception'] = True, the default)
      "Cheats" using CARLA's own simulator ground truth -- the exact same
      extraction logic as your data collection script (actor.bounding_box,
      world-to-camera/LiDAR projection, traffic_light.get_state()). This
      lets you build, debug, and demo the ENTIRE loop -- routing, the
      stop/avoid/lane-change FSM, PID control -- today, with zero trained
      models. This is Phase 4 work, decoupled from Phases 2-3.

  real mode  (use_oracle_perception = False, or per-stage overrides below)
      Loads your actual trained weights and runs real inference. Flip the
      flag and set the weight paths once a model is ready -- nothing else
      in this file needs to change, because every stage returns the same
      shape of output either way (see each class's docstring).

IMPORTANT: oracle mode is for integration testing / demos only. Any mAP,
driving-score, or ablation numbers you report MUST come from real model
mode -- oracle perception trivially "solves" perception by definition and
would make your evaluation meaningless.
"""

import os
import sys
import glob
import time
import random
import queue
import numpy as np

import carla

# CARLA's agents package (GlobalRoutePlanner, VehiclePIDController, etc.)
# ships inside the CARLA release under PythonAPI/carla/agents, not as a pip
# package -- point this at your install if the import below fails.
# e.g. sys.path.append('C:/CARLA_0.9.15/PythonAPI/carla')
import sys
sys.path.append(r'C:\Users\24AI001\Downloads\ITS Project\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla')

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.controller import VehiclePIDController
except ImportError:
    print("Could not import carla.agents -- add CARLA's PythonAPI/carla "
          "folder to sys.path (see comment above this import).")
    raise


CFG = dict(
    town='Town03',
    fixed_delta_seconds=0.05,
    img_w=800, img_h=600, fov=90,
    lidar_range=60.0, lidar_channels=64, lidar_pps=500000,

    # --- perception mode switch -------------------------------------------
    use_oracle_perception=True,   # master switch; per-stage flags below
                                   # override it individually once that one
                                   # model is trained, so you can mix real +
                                   # oracle stages during the transition.
    yolo_weights=None,            # path once trained, e.g. 'runs/.../best.pt'
    seg_weights=None,             # segmentation checkpoint
    lane_weights=None,            # UFLD-v2 checkpoint
    det3d_config=None,            # OpenPCDet/mmdet3d config
    det3d_ckpt=None,              # OpenPCDet/mmdet3d checkpoint

    # --- decision logic thresholds -----------------------------------------
    max_speed_kmh=30.0,
    stop_distance_m=15.0,      # red light / obstacle dead ahead within this -> stop
    caution_distance_m=30.0,   # slow down within this
    pedestrian_margin_factor=1.5,  # pedestrians get wider stop/caution margins
    route_sample_resolution=2.0,

    # --- lookahead / curvature control --------------------------------------
    # Fixes the "steering at a point almost underneath the car" problem that
    # caused the turn-induced crash: the controller now targets a point AHEAD
    # of the car by a distance that scales with current speed (time-based
    # preview), instead of the nearest untouched route point (~2m away,
    # i.e. ~0.2s of preview at 30km/h -- far too short to react to a turn).
    lookahead_time_horizon=1.5,     # seconds of preview distance
    lookahead_min_distance=5.0,     # meters, floor even at low/zero speed
    curvature_lookahead_indices=10, # how many route points ahead to scan for turns
    curvature_slow_factor=0.5,      # deg of heading change -> km/h reduction

    # --- traffic / pedestrians ----------------------------------------------
    n_vehicles=40,
    n_walkers=30,
    pedestrian_spawn_retries=3,
    pedestrian_running_ratio=0.2,
    pedestrian_max_speed_walk=1.4,
    pedestrian_max_speed_run=3.0,

    destination=None,  # carla.Location; random reachable point if None
    max_sim_seconds=180,
)


# ---------------------------------------------------------------------------
# Traffic + pedestrians -- ported from collect_dataset.py so decide_speed()'s
# obstacle-avoidance branch actually has something to avoid, and the
# traffic-light logic has other vehicles to interact with at junctions.
# ---------------------------------------------------------------------------
def spawn_traffic(world, bp, tm, n_vehicles, exclude_spawn_point=None):
    """Spawns n_vehicles on autopilot via the Traffic Manager. Simpler than
    collect_dataset.py's version (no forced two-wheeler minimums -- that was
    about training-data class balance, irrelevant here); this just needs
    general traffic on the road for the ego to react to."""
    vehicles = []
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    all_vehicle_bps = list(bp.filter('vehicle.*'))

    for sp in spawn_points:
        if len(vehicles) >= n_vehicles:
            break
        if exclude_spawn_point is not None and \
           sp.location.distance(exclude_spawn_point.location) < 2.0:
            continue  # don't spawn on top of the ego
        v = world.try_spawn_actor(random.choice(all_vehicle_bps), sp)
        if v:
            v.set_autopilot(True, tm.get_port())
            vehicles.append(v)

    if len(vehicles) < n_vehicles:
        print(f'  [warn] only spawned {len(vehicles)}/{n_vehicles} traffic vehicles '
              f'(ran out of free spawn points)')
    return vehicles


def spawn_walkers(world, bp, n_walkers):
    """Identical approach to collect_dataset.py's spawn_walkers: spawn on
    the nav mesh, tick, attach AI controllers, tick again, then start()
    them walking/running to random destinations."""
    walker_bps = list(bp.filter('walker.pedestrian.*'))
    if not walker_bps:
        print('  [warn] no walker.pedestrian.* blueprints found -- no pedestrians spawned')
        return [], []

    walkers = []
    attempts = 0
    max_attempts = n_walkers * max(1, CFG.get('pedestrian_spawn_retries', 3))

    while len(walkers) < n_walkers and attempts < max_attempts:
        attempts += 1
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker = world.try_spawn_actor(walker_bp, carla.Transform(loc))
        if walker is not None:
            walkers.append(walker)

    if not walkers:
        print('  [warn] spawned 0 pedestrians -- nav mesh may be missing for this town')
        return [], []

    world.tick()  # walkers must exist server-side before attaching controllers

    controller_bp = bp.find('controller.ai.walker')
    controllers = []
    for walker in walkers:
        controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if controller is not None:
            controllers.append(controller)

    world.tick()  # controllers must exist before start()/go_to_location()

    running_ratio = CFG.get('pedestrian_running_ratio', 0.2)
    walk_speed = CFG.get('pedestrian_max_speed_walk', 1.4)
    run_speed = CFG.get('pedestrian_max_speed_run', 3.0)

    for controller in controllers:
        try:
            controller.start()
            dest = world.get_random_location_from_navigation()
            if dest is not None:
                controller.go_to_location(dest)
            controller.set_max_speed(run_speed if random.random() < running_ratio else walk_speed)
        except RuntimeError as e:
            print(f'  [warn] walker controller failed to start: {e}')

    if len(walkers) < n_walkers:
        print(f'  [warn] only spawned {len(walkers)}/{n_walkers} pedestrians')

    return walkers, controllers


# ---------------------------------------------------------------------------
# Sensor setup + synchronized grab (same pattern as collect_dataset.py)
# ---------------------------------------------------------------------------
def setup_collision_sensor(world, ego, bp):
    """Separate from the synced rgb/lidar queues -- collisions are rare,
    async events, not per-tick data, so this just prints directly from the
    callback rather than going through grab_synced(). Added specifically to
    find out WHAT the car hit when speed flatlines to ~0 while target_speed
    stays high (seen at t=70s in the last run) -- that pattern means
    "physically stuck," and this tells us against what."""
    col_bp = bp.find('sensor.other.collision')
    col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)

    def on_collision(event):
        other = event.other_actor.type_id if event.other_actor else 'unknown'
        impulse = event.normal_impulse
        mag = (impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2) ** 0.5
        print(f'\n  [COLLISION] frame {event.frame}: hit "{other}" '
              f'(impulse magnitude {mag:.0f})')

    col_sensor.listen(on_collision)
    return col_sensor


def setup_sensors(world, ego, bp):
    sensors = {}
    cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))

    rgb = bp.find('sensor.camera.rgb')
    rgb.set_attribute('image_size_x', str(CFG['img_w']))
    rgb.set_attribute('image_size_y', str(CFG['img_h']))
    rgb.set_attribute('fov', str(CFG['fov']))
    sensors['rgb'] = world.spawn_actor(rgb, cam_tf, attach_to=ego)

    lid = bp.find('sensor.lidar.ray_cast')
    lid.set_attribute('range', str(CFG['lidar_range']))
    lid.set_attribute('channels', str(CFG['lidar_channels']))
    lid.set_attribute('points_per_second', str(CFG['lidar_pps']))
    lid.set_attribute('rotation_frequency', str(1.0 / CFG['fixed_delta_seconds']))
    sensors['lidar'] = world.spawn_actor(
        lid, carla.Transform(carla.Location(z=2.4)), attach_to=ego)

    queues = {}
    for name, s in sensors.items():
        q = queue.Queue()
        s.listen(q.put)
        queues[name] = q
    return sensors, queues


def update_spectator(world, ego, height=4.0, back_distance=8.0, pitch=-15.0):
    """Move the spectator (the free camera in the CARLA window) to a
    chase-cam position behind and above the ego vehicle. The spectator does
    NOT follow a newly spawned actor automatically -- without this, the car
    can be driving around the whole time while the viewport stays pointed
    wherever it was left (often nowhere near any spawn point), which looks
    exactly like 'nothing is happening' even though the sim is running
    fine. Call this once right after spawning ego, and again each tick (or
    every few ticks) in the main loop to keep following."""
    tf = ego.get_transform()
    forward = tf.get_forward_vector()
    cam_loc = tf.location - forward * back_distance + carla.Location(z=height)
    cam_rot = carla.Rotation(pitch=pitch, yaw=tf.rotation.yaw)
    world.get_spectator().set_transform(carla.Transform(cam_loc, cam_rot))


def drain_queues(queues):
    """Empty out any backlog (e.g. from warm-up ticks) so the driving loop
    starts from a clean, synchronized state. Same fix collect_dataset.py
    already has -- sensors start listening the moment they're spawned, so
    every world.tick() before this call (including the warm-up loop below)
    pushes a frame into each queue that nobody has read yet. Without
    draining, the first grab_synced() call pulls that stale backlog instead
    of the current frame and raises exactly the 'frame X != expected Y'
    mismatch we hit."""
    for name, q in queues.items():
        n = 0
        while True:
            try:
                q.get_nowait()
                n += 1
            except queue.Empty:
                break
        if n:
            print(f'  drained {n} stale "{name}" frames before driving')


class FrameDropped(Exception):
    """Raised when a sensor genuinely has no data for frame_id this tick --
    most commonly because RGB camera rendering has a persistent one-tick
    GPU pipeline lag relative to LiDAR/world.tick()'s own frame counter
    (not a one-time startup glitch -- it recurs every tick unless handled).
    The caller should skip this tick's control update (reuse the last
    command) and try again next tick, same as collect_dataset.py does."""
    pass


def grab_synced(queues, frame_id, buffers, timeout=2.0):
    """Buffered version matching collect_dataset.py's approach: a sensor
    running persistently behind/ahead of frame_id doesn't crash the loop --
    we buffer whatever arrived and let the next tick's incremented frame_id
    naturally catch up to it."""
    data = {}
    for name, q in queues.items():
        buffered = buffers.get(name)
        if buffered is not None and buffered.frame == frame_id:
            data[name] = buffered
            del buffers[name]
            continue
        if buffered is not None and buffered.frame > frame_id:
            raise FrameDropped(f'{name} has no data for frame {frame_id} yet '
                                f'(next available is {buffered.frame})')
        if buffered is not None and buffered.frame < frame_id:
            del buffers[name]  # stale leftover, discard and fetch fresh
            buffered = None

        while True:
            try:
                item = q.get(timeout=timeout)
            except queue.Empty:
                raise FrameDropped(f'{name} produced nothing for frame '
                                    f'{frame_id} within {timeout}s (stalled/dead sensor)')
            if item.frame == frame_id:
                data[name] = item
                break
            if item.frame > frame_id:
                buffers[name] = item
                raise FrameDropped(f'{name} overshot frame {frame_id} (got {item.frame})')
            # item.frame < frame_id: stale, discard and keep reading
    return data


# ---------------------------------------------------------------------------
# 2D detection (cars, traffic lights, signs) -- YOLO
# ---------------------------------------------------------------------------
class Detector2D:
    """predict(rgb_carla_image) -> list of dicts:
        {'class': str, 'bbox_px': (x1, y1, x2, y2), 'conf': float}
    Oracle mode additionally tags the nearest traffic light's live state
    (Red/Yellow/Green) onto a synthetic 'traffic_light' detection, since
    that's cheap ground truth your YOLO model won't itself classify unless
    you add a light-state head/crop-classifier later (see project plan
    Section 4: "a tiny CNN or color heuristic reads the light state")."""

    def __init__(self, weights_path, use_oracle, world=None, ego=None):
        self.use_oracle = use_oracle or weights_path is None
        self.world = world
        self.ego = ego
        if not self.use_oracle:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
        else:
            print('[Detector2D] oracle mode -- using CARLA GT, not a trained model')

    def predict(self, rgb_image):
        if not self.use_oracle:
            import numpy as np
            arr = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
            arr = arr.reshape((rgb_image.height, rgb_image.width, 4))[:, :, :3]
            result = self.model.predict(arr, verbose=False)[0]
            return [{'class': result.names[int(b.cls)], 'bbox_px': tuple(b.xyxy[0].tolist()),
                     'conf': float(b.conf)} for b in result.boxes]

        # --- oracle: nearest traffic light's real state, ego's lane only ---
        dets = []
        tl = self.ego.get_traffic_light()
        if tl is not None:
            dets.append({'class': 'traffic_light', 'bbox_px': None, 'conf': 1.0,
                         'state': str(tl.get_state())})  # Red / Yellow / Green
        return dets


# ---------------------------------------------------------------------------
# Lane detection
# ---------------------------------------------------------------------------
class LaneDetector:
    """predict(rgb_carla_image) -> list of lane point-lists in image space,
    or None if unavailable. Oracle mode is intentionally a no-op: lane
    *geometry* for routing already comes from the map/waypoint API via
    GlobalRoutePlanner below, so there's no cheap oracle substitute worth
    building here -- this stage stays genuinely empty until UFLD-v2 (or a
    segmentation-based lane extractor) is trained."""

    def __init__(self, weights_path, use_oracle):
        self.use_oracle = use_oracle or weights_path is None
        if not self.use_oracle:
            # TODO: load your trained UFLD-v2 model here once ready.
            raise NotImplementedError('wire up your lane model checkpoint loading here')
        else:
            print('[LaneDetector] not trained yet -- returning None '
                  '(route waypoints from GlobalRoutePlanner are used instead)')

    def predict(self, rgb_image):
        if self.use_oracle:
            return None
        # TODO: real inference
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fusion (PointPainting) + 3D detection
# ---------------------------------------------------------------------------
class Fusion3DDetector:
    """predict(world, lidar_sensor, lidar_measurement, ego) -> list of dicts:
        {'class': str, 'center_lidar': (x, y, z), 'size': (dx, dy, dz), 'yaw': float}
    (same convention as your data collection script's extract_3d_bboxes_lidar
    output, so decide_speed()/next_waypoint() below work unchanged in both
    modes.)

    Oracle mode reuses that exact extraction function as the "3D detector" --
    it's literally the GT box generator from collect_dataset.py, repurposed
    here as a stand-in so you can test obstacle-avoidance logic before
    PointPillars/PointPainting exist."""

    def __init__(self, det3d_config, det3d_ckpt, use_oracle):
        self.use_oracle = use_oracle or det3d_config is None or det3d_ckpt is None
        if not self.use_oracle:
            # TODO: load your OpenPCDet/mmdet3d model + config here.
            raise NotImplementedError('wire up OpenPCDet/mmdet3d model loading here')
        else:
            print('[Fusion3DDetector] oracle mode -- using CARLA GT boxes, '
                  'not PointPainting + PointPillars')

    def predict(self, world, lidar_sensor, lidar_measurement, ego, max_distance=60.0):
        if self.use_oracle:
            return self._oracle_boxes(world, lidar_sensor, lidar_measurement, ego, max_distance)
        # TODO: run segmentation -> paint LiDAR points -> run 3D detector
        raise NotImplementedError

    @staticmethod
    def _oracle_boxes(world, lidar_sensor, lidar_measurement, ego, max_distance):
        # Same math as extract_3d_bboxes_lidar() in collect_dataset.py,
        # trimmed to return structured dicts instead of KITTI text lines.
        w2l = np.array(lidar_sensor.get_transform().get_inverse_matrix())
        ego_loc = ego.get_transform().location
        lidar_yaw = lidar_sensor.get_transform().rotation.yaw
        results = []

        actors = list(world.get_actors().filter('vehicle.*')) + \
                 list(world.get_actors().filter('walker.pedestrian.*'))
        for actor in actors:
            if actor.id == ego.id:
                continue
            if actor.get_transform().location.distance(ego_loc) > max_distance:
                continue

            tf = actor.get_transform()
            bb = actor.bounding_box
            center_world = tf.transform(bb.location)
            center_h = np.array([center_world.x, center_world.y, center_world.z, 1.0])
            center_lidar = w2l.dot(center_h)[:3]
            x, y, z = center_lidar[0], -center_lidar[1], center_lidar[2]

            yaw = -np.deg2rad(tf.rotation.yaw - lidar_yaw)
            yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))

            cls = 'vehicle' if actor.type_id.startswith('vehicle.') else 'pedestrian'
            results.append({
                'class': cls,
                'center_lidar': (float(x), float(y), float(z)),
                'size': (bb.extent.x * 2.0, bb.extent.y * 2.0, bb.extent.z * 2.0),
                'yaw': yaw,
            })
        return results


# ---------------------------------------------------------------------------
# Decision logic (classical -- rules over whatever the stages above return)
# ---------------------------------------------------------------------------
def estimate_curvature_speed_cap(route, route_index, lookahead_indices, slow_factor):
    """Scans the next `lookahead_indices` route points for total heading
    change (a cheap curvature proxy) and returns a km/h speed cap -- a
    straight road gives no cap (999), a sharp upcoming turn gives a low
    cap. This is what was MISSING before: decide_speed() only reacted to
    dets2d/dets3d, never to the road geometry itself, so the car held full
    speed right up to a turn it then couldn't make.

    Heuristic, not physics: total_heading_change in degrees over the
    window is multiplied by slow_factor and subtracted from a baseline.
    Tune slow_factor / the baseline against your own town's turn radii if
    the car is still too fast/slow into corners."""
    end_idx = min(route_index + lookahead_indices, len(route) - 1)
    if end_idx <= route_index + 1:
        return 999.0  # not enough route left to judge, no cap

    total_heading_change = 0.0
    for i in range(route_index, end_idx):
        yaw1 = route[i][0].transform.rotation.yaw
        yaw2 = route[i + 1][0].transform.rotation.yaw
        diff = (yaw2 - yaw1 + 180) % 360 - 180  # wrap to [-180, 180]
        total_heading_change += abs(diff)

    if total_heading_change < 5.0:
        return 999.0  # essentially straight
    return max(10.0, 60.0 - total_heading_change * slow_factor)


def decide_speed(dets2d, dets3d, curvature_speed_cap, max_speed_kmh,
                  stop_distance_m, caution_distance_m, pedestrian_margin_factor):
    """Returns target speed in km/h. Rules, not learned -- per the plan,
    this stays classical regardless of which perception mode is active.
    Three independent caps are combined by taking the minimum: traffic
    light (hard stop), nearest obstacle (class-aware margins -- pedestrians
    get a wider berth than vehicles), and road curvature (new)."""
    for d in dets2d:
        if d['class'] == 'traffic_light' and d.get('state') == 'Red':
            return 0.0

    # dets3d boxes are already in the ego-relative LiDAR frame (x=forward),
    # so "ahead" is just x > 0 and |y| small (roughly in the ego's lane).
    nearest_ahead, nearest_is_pedestrian = None, False
    for d in dets3d:
        x, y, _ = d['center_lidar']
        lateral_gate = 3.0 if d['class'] == 'pedestrian' else 2.0  # wider gate for people
        if x > 0 and abs(y) < lateral_gate:
            if nearest_ahead is None or x < nearest_ahead:
                nearest_ahead = x
                nearest_is_pedestrian = (d['class'] == 'pedestrian')

    obstacle_speed_cap = max_speed_kmh
    if nearest_ahead is not None:
        margin = pedestrian_margin_factor if nearest_is_pedestrian else 1.0
        if nearest_ahead < stop_distance_m * margin:
            obstacle_speed_cap = 0.0
        elif nearest_ahead < caution_distance_m * margin:
            obstacle_speed_cap = max_speed_kmh * 0.4

    return min(max_speed_kmh, obstacle_speed_cap, curvature_speed_cap)


def compute_lookahead_waypoint(route, route_index, lookahead_distance):
    """Walks forward along the route from route_index, accumulating
    distance, until lookahead_distance is reached -- returns that waypoint
    for the controller to steer toward. This is the actual fix for
    "steering at a point almost underneath the car": the old code targeted
    route[route_index] directly, only ~2m ahead regardless of speed."""
    accumulated = 0.0
    idx = route_index
    prev_loc = route[idx][0].transform.location
    while accumulated < lookahead_distance and idx < len(route) - 1:
        idx += 1
        loc = route[idx][0].transform.location
        accumulated += prev_loc.distance(loc)
        prev_loc = loc
    return route[idx][0]


def next_waypoint(route, route_index, ego, current_speed_kmh,
                   lookahead_time_horizon, lookahead_min_distance):
    """Advances route_index for progress tracking (proximity-based, as
    before), but now returns a LOOKAHEAD waypoint for the controller --
    scaled by current speed (time-based preview: faster = look further
    ahead), not the immediate next route point. Lane-change logic is still
    NOT implemented (plan Section 6/Phase 4 stretch) -- this always follows
    the planned route's own lane. Returns (target_waypoint, new_index)."""
    # TODO: when ready, check whether dets3d blocks the current lane and,
    # if a neighboring lane is clear (via waypoint.get_left_lane()/
    # get_right_lane()), splice in a lane-change maneuver here.
    ego_loc = ego.get_transform().location
    while (route_index < len(route) - 1 and
           route[route_index][0].transform.location.distance(ego_loc) < 2.0):
        route_index += 1

    speed_mps = current_speed_kmh / 3.6
    lookahead_distance = max(lookahead_min_distance, speed_mps * lookahead_time_horizon)
    target_wp = compute_lookahead_waypoint(route, route_index, lookahead_distance)
    return target_wp, route_index


# ---------------------------------------------------------------------------
# Main loop (Section 7)
# ---------------------------------------------------------------------------
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    client.load_world(CFG['town'])
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = CFG['fixed_delta_seconds']
    world.apply_settings(settings)

    bp = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    start_tf = random.choice(spawn_points)
    dest_loc = CFG['destination'] or random.choice(spawn_points).location

    ego = world.spawn_actor(bp.filter('vehicle.tesla.model3')[0], start_tf)
    update_spectator(world, ego)  # point the CARLA window at the car immediately
    sensors, queues = setup_sensors(world, ego, bp)
    collision_sensor = setup_collision_sensor(world, ego, bp)

    # --- perception stages: flip CFG['use_oracle_perception'] (or pass
    # per-stage weight paths) once each model is trained; nothing below
    # this block needs to change. ---
    det2d = Detector2D(CFG['yolo_weights'], CFG['use_oracle_perception'], world, ego)
    lanes_model = LaneDetector(CFG['lane_weights'], CFG['use_oracle_perception'])
    det3d = Fusion3DDetector(CFG['det3d_config'], CFG['det3d_ckpt'], CFG['use_oracle_perception'])

    # --- classical routing + control (do not train these) ---
    grp = GlobalRoutePlanner(world.get_map(), CFG['route_sample_resolution'])
    route = grp.trace_route(start_tf.location, dest_loc)
    route_index = 0

    pid = VehiclePIDController(
        ego,
        args_lateral={'K_P': 1.5, 'K_I': 0.05, 'K_D': 0.1, 'dt': CFG['fixed_delta_seconds']},
        args_longitudinal={'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0, 'dt': CFG['fixed_delta_seconds']},
    )

    try:
        for _ in range(10):
            world.tick()  # let the world settle before driving starts

        drain_queues(queues)  # clear the warm-up backlog -- see drain_queues() docstring

        buffers = {}
        last_control = carla.VehicleControl()  # reused on skipped ticks
        skipped = 0

        n_ticks = int(CFG['max_sim_seconds'] / CFG['fixed_delta_seconds'])
        for tick_idx in range(n_ticks):
            frame_id = world.tick()
            try:
                data = grab_synced(queues, frame_id, buffers)
            except FrameDropped as e:
                skipped += 1
                ego.apply_control(last_control)  # coast on the last known command
                continue

            # ---- perception ----
            dets2d = det2d.predict(data['rgb'])
            lane_result = lanes_model.predict(data['rgb'])   # not yet used by decide_speed/next_waypoint
            dets3d = det3d.predict(world, sensors['lidar'], data['lidar'], ego,
                                    max_distance=CFG['lidar_range'])

            # ---- decision (rules over DL outputs) ----
            target_speed = decide_speed(dets2d, dets3d, CFG['max_speed_kmh'],
                                         CFG['stop_distance_m'], CFG['caution_distance_m'])
            target_wp, route_index = next_waypoint(route, route_index, ego, dets3d)

            # ---- control ----
            control = pid.run_step(target_speed, target_wp)
            ego.apply_control(control)
            last_control = control
            update_spectator(world, ego)  # keep the chase-cam on the car

            if route_index >= len(route) - 1 and \
               ego.get_transform().location.distance(dest_loc) < 3.0:
                print('destination reached')
                break

            # periodic progress printout -- every ~5 sim-seconds -- so a
            # long/stuck run is visible instead of silent until it ends
            if tick_idx % 100 == 0:
                speed_kmh = 3.6 * np.linalg.norm([ego.get_velocity().x,
                                                    ego.get_velocity().y,
                                                    ego.get_velocity().z])
                dist_to_dest = ego.get_transform().location.distance(dest_loc)
                print(f'  t={tick_idx * CFG["fixed_delta_seconds"]:.1f}s  '
                      f'speed={speed_kmh:.1f}km/h  target_speed={target_speed:.1f}km/h  '
                      f'route {route_index}/{len(route)}  dist_to_dest={dist_to_dest:.1f}m  '
                      f'2d_dets={len(dets2d)}  3d_dets={len(dets3d)}  '
                      f'steer={control.steer:.2f}  throttle={control.throttle:.2f}  '
                      f'brake={control.brake:.2f}')

        print(f'\n{skipped} ticks skipped due to sensor lag (expected -- RGB '
              f'has a steady one-tick GPU rendering delay)')

    finally:
        for s in list(sensors.values()) + [collision_sensor]:
            try: s.stop()
            except: pass
        for a in [ego] + list(sensors.values()) + [collision_sensor]:
            try: a.destroy()
            except: pass
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == '__main__':
    main()