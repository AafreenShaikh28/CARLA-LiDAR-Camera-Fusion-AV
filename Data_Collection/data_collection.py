# import carla, os, queue, random, json, time
# from collections import Counter
# import numpy as np

# CFG = dict(
#     towns           = ['Town03', 'Town01', 'Town05'],
#     frames_per_town = 10000,          # raise to 7000 once a short run passes
#     tick_skip       = 4,
#     n_vehicles      = 120,
#     n_walkers       = 200,
#     # Minimum number of each two-wheeler class to force-spawn per town,
#     # since random.choice over all vehicle.* blueprints under-represents
#     # them badly (CARLA's default content has far more car blueprints than
#     # bicycle/motorcycle ones). These come out of the n_vehicles budget.
#     min_bicycles    = 50,
#     min_motorcycles = 50,
#     # Distance (meters) within which two-wheeler spawn points are preferred
#     # over random ones, so more instances land big enough to actually learn
#     # from instead of being unusable 6px specks at the far end of lidar_range.
#     two_wheeler_bias_radius = 25.0,
#     # Drop GT 2D boxes smaller than this many pixels in width or height --
#     # they carry ~no learnable signal and just add label noise to training.
#     min_box_px = 12,
#     img_w = 800, img_h = 600, fov = 90,
#     lidar_range = 60.0, lidar_channels = 64, lidar_pps = 500000,
#     weather = carla.WeatherParameters.ClearNoon,
#     out = 'dataset',
# )


# def setup_sensors(world, vehicle, bp):
#     sensors = {}
#     cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))

#     rgb = bp.find('sensor.camera.rgb')
#     rgb.set_attribute('image_size_x', str(CFG['img_w']))
#     rgb.set_attribute('image_size_y', str(CFG['img_h']))
#     rgb.set_attribute('fov', str(CFG['fov']))
#     sensors['rgb'] = world.spawn_actor(rgb, cam_tf, attach_to=vehicle)

#     sem = bp.find('sensor.camera.semantic_segmentation')
#     sem.set_attribute('image_size_x', str(CFG['img_w']))
#     sem.set_attribute('image_size_y', str(CFG['img_h']))
#     sem.set_attribute('fov', str(CFG['fov']))
#     sensors['sem'] = world.spawn_actor(sem, cam_tf, attach_to=vehicle)

#     lid = bp.find('sensor.lidar.ray_cast')
#     lid.set_attribute('range', str(CFG['lidar_range']))
#     lid.set_attribute('channels', str(CFG['lidar_channels']))
#     lid.set_attribute('points_per_second', str(CFG['lidar_pps']))
#     lid.set_attribute('rotation_frequency', '20')   # = 1 / fixed_delta_seconds
#     sensors['lidar'] = world.spawn_actor(
#         lid, carla.Transform(carla.Location(z=2.4)), attach_to=vehicle)

#     queues = {}
#     for name, s in sensors.items():
#         q = queue.Queue()
#         s.listen(q.put)
#         queues[name] = q
#     return sensors, queues


# def drain_queues(queues):
#     """Empty out any backlog (e.g. from warm-up ticks) so the recording
#     loop starts from a clean, synchronized state."""
#     for name, q in queues.items():
#         n = 0
#         while True:
#             try:
#                 q.get_nowait()
#                 n += 1
#             except queue.Empty:
#                 break
#         if n:
#             print(f'  drained {n} stale "{name}" frames before recording')


# class FrameDropped(Exception):
#     """Raised when a sensor genuinely never produced a frame for frame_id
#     (as opposed to just arriving slightly out of order)."""
#     pass


# def grab_synced(queues, frame_id, buffers, timeout=5.0):
#     data = {}
#     for name, q in queues.items():
#         buffered = buffers.get(name)
#         if buffered is not None and buffered.frame == frame_id:
#             data[name] = buffered
#             del buffers[name]
#             continue
#         if buffered is not None and buffered.frame > frame_id:
#             raise FrameDropped(f'{name} has no data for frame {frame_id} '
#                                 f'(next available is {buffered.frame})')
#         if buffered is not None and buffered.frame < frame_id:
#             del buffers[name]   # stale leftover from an earlier overshoot, discard
#             buffered = None
#         while True:
#             try:
#                 item = q.get(timeout=timeout)
#             except queue.Empty:
#                 raise FrameDropped(f'{name} produced nothing for frame '
#                                     f'{frame_id} within {timeout}s (stalled/dead sensor)')
#             if item.frame == frame_id:
#                 data[name] = item
#                 break
#             if item.frame > frame_id:
#                 buffers[name] = item
#                 raise FrameDropped(f'{name} overshot frame {frame_id} (got {item.frame})')
#     return data


# def extract_labels(world, lidar_sensor):
#     """YOUR JOB — see the CARLA bounding-box tutorial.
#     For each vehicle/walker actor:
#       1. read actor.bounding_box (extent + transform)
#       2. transform world frame -> LiDAR frame
#       3. drop actors beyond lidar_range or with no returns
#       4. return KITTI-style rows: class, dims, location, rotation_y
#     """
#     return []          # stub so the pipeline runs; replace this


# # ---------------------------------------------------------------------------
# # 2D bounding-box (YOLO) ground truth
# # ---------------------------------------------------------------------------
# # Maps CARLA actor type prefixes -> YOLO class ids. Extend/renumber to match
# # whatever your model's data.yaml expects.
# YOLO_CLASSES = {
#     'person': 0,
#     'bicycle': 1,
#     'car': 2,
#     'motorcycle': 3,
#     'bus': 5,
#     'truck': 7,
#     'traffic_light': 9,
#     'stop_sign': 11,
# }

# # ---------------------------------------------------------------------------
# # CARLA actor -> COCO/YOLO class mapping
# # ---------------------------------------------------------------------------
# # CARLA type_ids never look like 'car...' or 'person...' — they look like
# # 'vehicle.tesla.model3' or 'walker.pedestrian.0004' — so we can't prefix-
# # match against YOLO_CLASSES directly. Classify by structure instead.

# # Known non-"car" vehicle blueprints (CARLA 0.9.13/0.9.14 default content).
# # Anything vehicle.* not listed here defaults to 'car'.
# BICYCLE_BLUEPRINTS = {
#     'vehicle.bh.crossbike',
#     'vehicle.diamondback.century',
#     'vehicle.gazelle.omafiets',
# }
# MOTORCYCLE_BLUEPRINTS = {
#     'vehicle.harley-davidson.low_rider',
#     'vehicle.kawasaki.ninja',
#     'vehicle.vespa.zx125',
#     'vehicle.yamaha.yzf',
# }
# BUS_BLUEPRINTS = {
#     'vehicle.mitsubishi.fusorosa',
# }
# TRUCK_BLUEPRINTS = {
#     'vehicle.carlamotors.carlacola',
#     'vehicle.carlamotors.firetruck',
#     'vehicle.carlamotors.european_hgv',
#     'vehicle.tesla.cybertruck',
#     'vehicle.mercedes.sprinter',
#     'vehicle.volkswagen.t2',
#     'vehicle.volkswagen.t2_2021',
#     'vehicle.ford.ambulance',
# }


# def classify_actor(actor):
#     """Map a CARLA actor to a (class_name, class_id) pair from YOLO_CLASSES,
#     or (None, None) if it's not a type we're labeling."""
#     type_id = actor.type_id

#     if type_id.startswith('walker.pedestrian'):
#         return 'person', YOLO_CLASSES['person']

#     if not type_id.startswith('vehicle.'):
#         return None, None

#     # Prefer CARLA's own base_type attribute when present (0.9.14+) --
#     # it already speaks car/truck/bus/motorcycle/bicycle.
#     base_type = actor.attributes.get('base_type')
#     if base_type in YOLO_CLASSES:
#         return base_type, YOLO_CLASSES[base_type]

#     if type_id in BICYCLE_BLUEPRINTS:
#         return 'bicycle', YOLO_CLASSES['bicycle']
#     if type_id in MOTORCYCLE_BLUEPRINTS:
#         return 'motorcycle', YOLO_CLASSES['motorcycle']
#     if type_id in BUS_BLUEPRINTS:
#         return 'bus', YOLO_CLASSES['bus']
#     if type_id in TRUCK_BLUEPRINTS:
#         return 'truck', YOLO_CLASSES['truck']

#     # Fallback for any blueprint not in the tables above (e.g. a future
#     # CARLA content update): use wheel count as a last resort.
#     n_wheels = actor.attributes.get('number_of_wheels')
#     if n_wheels == '2':
#         return 'motorcycle', YOLO_CLASSES['motorcycle']

#     return 'car', YOLO_CLASSES['car']
# # ---------------------------------------------------------------------------


# def build_projection_matrix(w, h, fov):
#     """Pinhole camera intrinsics matrix for CARLA's RGB camera."""
#     focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
#     K = np.identity(3)
#     K[0, 0] = K[1, 1] = focal
#     K[0, 2] = w / 2.0
#     K[1, 2] = h / 2.0
#     return K


# def get_image_point(location, K, world_2_camera):
#     """Project a carla.Location (world space) into 2D image pixel coords.
#     Returns (x, y, depth); depth <= 0 means the point is behind the camera."""
#     point = np.array([location.x, location.y, location.z, 1.0])
#     point_camera = np.dot(world_2_camera, point)

#     # CARLA/UE4 world axes are (x-forward, y-right, z-up); standard camera
#     # convention is (x-right, y-down, z-forward). Swap accordingly.
#     point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])

#     point_img = np.dot(K, point_camera)
#     depth = point_img[2]
#     if depth != 0:
#         point_img[0] /= depth
#         point_img[1] /= depth
#     return point_img[0], point_img[1], depth


# def extract_2d_bboxes_yolo(world, rgb_sensor, ego_vehicle, img_w, img_h, fov,
#                             max_distance=None):
#     """Project every vehicle/walker actor's 3D bounding box into the RGB
#     camera's image plane and return YOLO-format ground-truth lines:

#         class_id x_center y_center width height      (all normalized 0..1)

#     Notes / limitations:
#       - This is a pure geometric projection, not a visibility/occlusion
#         check: an actor fully hidden behind a wall or another vehicle will
#         still get a (incorrect) box. If you need real occlusion filtering,
#         raycast from the camera to the actor (world.cast_ray, or check the
#         semantic/depth sensor) and drop actors whose center hit is blocked.
#       - Boxes are clipped to the image bounds; actors that are entirely
#         outside the frame or entirely behind the camera are dropped.
#     """
#     K = build_projection_matrix(img_w, img_h, fov)
#     world_2_camera = np.array(rgb_sensor.get_transform().get_inverse_matrix())

#     actors = list(world.get_actors().filter('vehicle.*')) + \
#              list(world.get_actors().filter('walker.pedestrian.*'))

#     ego_loc = ego_vehicle.get_transform().location
#     lines = []

#     for actor in actors:
#         if actor.id == ego_vehicle.id:
#             continue

#         if max_distance is not None:
#             if actor.get_transform().location.distance(ego_loc) > max_distance:
#                 continue

#         cls_name, cls_id = classify_actor(actor)
#         if cls_id is None:
#             continue  # unmapped actor type, skip

#         bb = actor.bounding_box
#         verts = bb.get_world_vertices(actor.get_transform())

#         xs, ys, in_front = [], [], 0
#         for v in verts:
#             x, y, depth = get_image_point(v, K, world_2_camera)
#             if depth <= 0:
#                 continue
#             in_front += 1
#             xs.append(x)
#             ys.append(y)

#         if in_front == 0:
#             continue  # entirely behind the camera

#         x_min, x_max = max(min(xs), 0.0), min(max(xs), img_w - 1.0)
#         y_min, y_max = max(min(ys), 0.0), min(max(ys), img_h - 1.0)

#         if x_max <= x_min or y_max <= y_min:
#             continue  # off-screen or degenerate after clipping to the frame

#         if (x_max - x_min) < CFG['min_box_px'] or (y_max - y_min) < CFG['min_box_px']:
#             continue  # too small to carry real training signal, just noise

#         x_center = (x_min + x_max) / 2.0 / img_w
#         y_center = (y_min + y_max) / 2.0 / img_h
#         width    = (x_max - x_min) / img_w
#         height   = (y_max - y_min) / img_h

#         lines.append((cls_name,
#                       f'{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'))

#     return lines
# # ---------------------------------------------------------------------------


# # ---------------------------------------------------------------------------
# # 3D bounding boxes, in the LiDAR sensor's own frame (PointPillars / OpenPCDet
# # style) -- for assessing a point-cloud 3D detector rather than an image one.
# # ---------------------------------------------------------------------------
# def world_to_lidar_matrix(lidar_sensor):
#     """4x4 matrix mapping a world-space point into the LiDAR's local frame --
#     the same frame the raw point cloud (saved to velodyne/*.ply) is already
#     expressed in."""
#     return np.array(lidar_sensor.get_transform().get_inverse_matrix())


# def extract_3d_bboxes_lidar(world, lidar_sensor, lidar_measurement, ego_vehicle,
#                              max_distance=None, min_points=1):
#     """Ground-truth 3D boxes in the LiDAR sensor's coordinate frame.

#     Each line:
#         class x y z dx dy dz yaw num_points

#       x, y, z    -- box center, meters, in the LiDAR frame
#                     (x forward, y left, z up -- right-handed, KITTI/velodyne
#                     convention; CARLA itself is left-handed with y to the
#                     right, so y is negated below to convert).
#       dx, dy, dz -- box size along x, y, z (length, width, height), meters.
#       yaw        -- heading around the vertical (z) axis, radians,
#                     counter-clockwise from the LiDAR's +x axis.
#       num_points -- raw points from THIS frame's point cloud that fall
#                     inside the box. Doubles as a free occlusion/visibility
#                     proxy and as the usual KITTI easy/moderate/hard signal --
#                     boxes below `min_points` are dropped outright.

#     IMPORTANT: coordinate-frame conversions (handedness, axis order, sign of
#     yaw) are the single easiest thing to get subtly wrong here, and it can't
#     be verified without a running simulator. Before trusting a full dataset
#     run, sanity-check a handful of frames: plot a few of these boxes over
#     their velodyne/*.ply point cloud (e.g. with Open3D) and confirm they
#     actually wrap the vehicles, with the heading arrow pointing the way the
#     car is facing.
#     """
#     w2l = world_to_lidar_matrix(lidar_sensor)  # world -> lidar-local, 4x4

#     raw = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32).reshape(-1, 4)
#     cloud_xyz = raw[:, :3]  # already expressed in the LiDAR's local frame

#     actors = list(world.get_actors().filter('vehicle.*')) + \
#              list(world.get_actors().filter('walker.pedestrian.*'))

#     ego_loc = ego_vehicle.get_transform().location
#     lidar_yaw = lidar_sensor.get_transform().rotation.yaw
#     lines = []

#     for actor in actors:
#         if actor.id == ego_vehicle.id:
#             continue

#         if max_distance is not None:
#             if actor.get_transform().location.distance(ego_loc) > max_distance:
#                 continue

#         cls_name, cls_id = classify_actor(actor)
#         if cls_id is None:
#             continue  # unmapped actor type, skip

#         tf = actor.get_transform()
#         bb = actor.bounding_box

#         # bb.location is an offset from the actor's own origin (non-zero for
#         # some pedestrian rigs), so transform it through the actor's pose
#         # first to get the box center in world space.
#         center_world = tf.transform(bb.location)
#         center_h = np.array([center_world.x, center_world.y, center_world.z, 1.0])
#         center_lidar = w2l.dot(center_h)[:3]
#         x, y, z = center_lidar[0], -center_lidar[1], center_lidar[2]

#         dx = bb.extent.x * 2.0   # length, along the actor's forward axis
#         dy = bb.extent.y * 2.0   # width
#         dz = bb.extent.z * 2.0   # height

#         yaw = -np.deg2rad(tf.rotation.yaw - lidar_yaw)
#         yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))  # wrap to [-pi, pi]

#         # raw points inside the (rotated) box -- rotate the cloud into the
#         # box's own axes, then do a simple axis-aligned containment test
#         rel = cloud_xyz - np.array([x, y, z])
#         cos_y, sin_y = np.cos(yaw), np.sin(yaw)
#         local_x = rel[:, 0] * cos_y + rel[:, 1] * sin_y
#         local_y = -rel[:, 0] * sin_y + rel[:, 1] * cos_y
#         local_z = rel[:, 2]
#         inside = (
#             (np.abs(local_x) <= dx / 2.0) &
#             (np.abs(local_y) <= dy / 2.0) &
#             (np.abs(local_z) <= dz / 2.0)
#         )
#         num_points = int(inside.sum())

#         if num_points < min_points:
#             continue

#         lines.append(f'{cls_name} {x:.3f} {y:.3f} {z:.3f} '
#                      f'{dx:.3f} {dy:.3f} {dz:.3f} {yaw:.4f} {num_points}')

#     return lines
# # ---------------------------------------------------------------------------


# def spawn_traffic(world, bp, tm, ego_location=None):
#     """Spawn CFG['n_vehicles'] total vehicles, but force a minimum number of
#     bicycles and motorcycles first (CFG['min_bicycles']/['min_motorcycles']),
#     biased toward spawn points near the ego vehicle.

#     Why bias by distance: even after guaranteeing a minimum count, a
#     two-wheeler spawned at a random point across the whole map is usually
#     far from the ego's camera for most/all of the frames it's ever seen in
#     -- producing GT boxes too small to be a useful training example (see
#     min_box_px filtering above). Sorting candidate spawn points by distance
#     to the ego and preferring the closest ones for two-wheelers means more
#     instances end up at a size the model can actually learn from.
#     """
#     vehicles = []
#     spawn_points = world.get_map().get_spawn_points()
#     random.shuffle(spawn_points)

#     if ego_location is not None:
#         # Points within the bias radius, closest-first; everything else
#         # after, still in its (already shuffled) random order.
#         near = sorted(
#             (sp for sp in spawn_points
#              if sp.location.distance(ego_location) <= CFG['two_wheeler_bias_radius']),
#             key=lambda sp: sp.location.distance(ego_location))
#         far = [sp for sp in spawn_points if sp not in near]
#         two_wheeler_spawn_points = near + far
#     else:
#         two_wheeler_spawn_points = spawn_points

#     all_vehicle_bps = list(bp.filter('vehicle.*'))
#     bike_bps = [b for b in all_vehicle_bps if b.id in BICYCLE_BLUEPRINTS]
#     moto_bps = [b for b in all_vehicle_bps if b.id in MOTORCYCLE_BLUEPRINTS]
    
    

#     def spawn_forced(blueprint_pool, n_target, label, candidate_points):
#         """Try spawn points one at a time, closest-to-ego first, until
#         n_target actors of this pool spawn or we run out of points. Points
#         that get tried (spawned or not) are removed from the shared pool so
#         the general fill-in pass below doesn't reuse them."""
#         spawned = 0
#         while spawned < n_target and candidate_points:
#             sp = candidate_points.pop(0)
#             if sp in spawn_points:
#                 spawn_points.remove(sp)
#             chosen_bp = random.choice(blueprint_pool)
#             v = world.try_spawn_actor(chosen_bp, sp)
#             if v:
#                 v.set_autopilot(True, tm.get_port())
#                 vehicles.append(v)
#                 spawned += 1
#         if spawned < n_target:
#             print(f'  [warn] only spawned {spawned}/{n_target} {label} '
#                   f'(ran out of free spawn points near the ego -- try a '
#                   f'larger two_wheeler_bias_radius or fewer n_vehicles)')
#         return spawned

#     if bike_bps:
#         spawn_forced(bike_bps, CFG['min_bicycles'], 'bicycles',
#                      list(two_wheeler_spawn_points))
#     else:
#         print('  [warn] no bicycle blueprints found on this CARLA install -- '
#               'check BICYCLE_BLUEPRINTS against your CARLA version\'s content')

#     if moto_bps:
#         spawn_forced(moto_bps, CFG['min_motorcycles'], 'motorcycles',
#                      [sp for sp in two_wheeler_spawn_points if sp in spawn_points])
#     else:
#         print('  [warn] no motorcycle blueprints found on this CARLA install -- '
#               'check MOTORCYCLE_BLUEPRINTS against your CARLA version\'s content')

#     # Fill the remaining budget with the general vehicle pool (cars, trucks,
#     # buses, and occasionally more two-wheelers by chance).
#     remaining = max(0, CFG['n_vehicles'] - len(vehicles))
#     for sp in spawn_points[:remaining]:
#         v = world.try_spawn_actor(random.choice(all_vehicle_bps), sp)
#         if v:
#             v.set_autopilot(True, tm.get_port())
#             vehicles.append(v)

#     return vehicles


# def collect_town(client, town, start_idx):
#     print(f'\nloading {town} ...')
#     client.load_world(town)
#     time.sleep(10)
#     world = client.get_world()                      # re-fetch: episode restarted
#     world.set_weather(CFG['weather'])

#     bp = world.get_blueprint_library()
#     tm = client.get_trafficmanager()

#     original = world.get_settings()
#     sensors, queues, actors = {}, {}, []            # defined before try
#     class_counts = Counter()                        # GT instance counts, this town

#     try:
#         settings = world.get_settings()
#         settings.synchronous_mode = True
#         settings.fixed_delta_seconds = 0.05
#         settings.no_rendering_mode = False
#         world.apply_settings(settings)
#         tm.set_synchronous_mode(True)                # enable TM sync AFTER world sync

#         ego = world.spawn_actor(
#             bp.filter('vehicle.tesla.model3')[0],
#             random.choice(world.get_map().get_spawn_points()))
#         ego.set_autopilot(True, tm.get_port())
#         actors.append(ego)

#         actors += spawn_traffic(world, bp, tm, ego_location=ego.get_transform().location)

#         sensors, queues = setup_sensors(world, ego, bp)
#         actors += list(sensors.values())

#         root = f"{CFG['out']}/{town}"
#         for d in ['image_2', 'velodyne', 'label_2', 'semantic', 'labels_yolo', 'labels_3d']:
#             os.makedirs(f'{root}/{d}', exist_ok=True)

#         for _ in range(60):                         # let traffic settle
#             world.tick()

#         drain_queues(queues)                        # clear warm-up backlog

#         buffers = {}
#         dropped_frames = 0
#         saved, tick = start_idx, 0
#         while saved < start_idx + CFG['frames_per_town']:
#             frame_id = world.tick()
#             try:
#                 data = grab_synced(queues, frame_id, buffers)
#             except FrameDropped as e:
#                 dropped_frames += 1
#                 print(f'\n  [warn] {e} -- skipping this frame '
#                       f'({dropped_frames} dropped so far)')
#                 continue

#             tick += 1
#             if tick % CFG['tick_skip']:
#                 continue

#             base = f'{saved:06d}'
#             data['rgb'].save_to_disk(f'{root}/image_2/{base}.png')
#             data['sem'].save_to_disk(f'{root}/semantic/{base}.png',
#                                      carla.ColorConverter.CityScapesPalette)
#             data['lidar'].save_to_disk(f'{root}/velodyne/{base}.ply')

#             with open(f'{root}/label_2/{base}.txt', 'w') as f:
#                 f.write('\n'.join(extract_labels(world, sensors['lidar'])))

#             yolo_pairs = extract_2d_bboxes_yolo(
#                 world, sensors['rgb'], ego,
#                 CFG['img_w'], CFG['img_h'], CFG['fov'],
#                 max_distance=CFG['lidar_range'])
#             for cls_name, _ in yolo_pairs:
#                 class_counts[cls_name] += 1
#             with open(f'{root}/labels_yolo/{base}.txt', 'w') as f:
#                 f.write('\n'.join(line for _, line in yolo_pairs))

#             lidar_3d_lines = extract_3d_bboxes_lidar(
#                 world, sensors['lidar'], data['lidar'], ego,
#                 max_distance=CFG['lidar_range'], min_points=1)
#             with open(f'{root}/labels_3d/{base}.txt', 'w') as f:
#                 f.write('\n'.join(lidar_3d_lines))

#             saved += 1
#             print(f'{town}: {saved - start_idx}/{CFG["frames_per_town"]}', end='\r')

#         print(f'\n{town} done ({dropped_frames} frames dropped/skipped)')
#         print(f'{town} GT instance counts (2D labels): '
#               + ', '.join(f'{name}={count}' for name, count in
#                            sorted(class_counts.items(), key=lambda kv: -kv[1]))
#               or '  (none)')
#         return saved

#     finally:
#         for s in sensors.values():                  # stop before destroy
#             try: s.stop()
#             except: pass
#         for a in actors:                            # destroy before restoring
#             try: a.destroy()
#             except: pass
#         tm.set_synchronous_mode(False)
#         world.apply_settings(original)
#         time.sleep(3)
#         print(f'{town} cleaned up')


# if __name__ == '__main__':
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(200.0)

#     idx = 0
#     for town in CFG['towns']:
#         idx = collect_town(client, town, idx)
#         json.dump({'next_idx': idx}, open('progress.json', 'w'))
#     print(f'\ntotal frames: {idx}')

import carla, os, queue, random, json, time
from collections import Counter
import numpy as np

CFG = dict(
    towns           = ['Town03', 'Town01', 'Town05'],
    frames_per_town = 10000,          # raise to 7000 once a short run passes
    tick_skip       = 4,
    n_vehicles      = 120,
    n_walkers       = 200,
    # Minimum number of each two-wheeler class to force-spawn per town,
    # since random.choice over all vehicle.* blueprints under-represents
    # them badly (CARLA's default content has far more car blueprints than
    # bicycle/motorcycle ones). These come out of the n_vehicles budget.
    min_bicycles    = 50,
    min_motorcycles = 50,
    # Distance (meters) within which two-wheeler spawn points are preferred
    # over random ones, so more instances land big enough to actually learn
    # from instead of being unusable 6px specks at the far end of lidar_range.
    two_wheeler_bias_radius = 25.0,
    # Drop GT 2D boxes smaller than this many pixels in width or height --
    # they carry ~no learnable signal and just add label noise to training.
    min_box_px = 12,
    img_w = 800, img_h = 600, fov = 90,
    lidar_range = 60.0, lidar_channels = 64, lidar_pps = 500000,
    weather = carla.WeatherParameters.ClearNoon,
    out = 'dataset',

    # NEW: pedestrian spawning (CFG['n_walkers'] existed before but was
    # never actually used anywhere -- no walkers were ever spawned).
    pedestrian_running_ratio  = 0.2,   # fraction of walkers set to "running" speed
    pedestrian_max_speed_walk = 1.4,   # m/s
    pedestrian_max_speed_run  = 3.0,   # m/s
    pedestrian_spawn_retries  = 3,     # x n_walkers = max spawn attempts

    # NEW: perception-pipeline range/filter knobs, decoupled per your spec
    # (section 3 vs section 6) even though both default to 60 m today.
    yolo_max_distance = 60.0,   # max distance (m) from ego for a 2D YOLO GT box
    min_points        = 1,      # min LiDAR points inside a 3D box to keep it

    # NEW: occlusion / visibility filtering for 2D YOLO GT (section 5).
    # Practical depth-buffer sampling, NOT a per-pixel raycast -- see
    # is_visible_via_depth() docstring for exactly what this does and does
    # not catch.
    visibility_check = True,
    occlusion_margin  = 1.0,    # meters of slack before calling a pixel "occluded"

    # NEW: optional debug/duplicate outputs, off by default to avoid
    # doubling storage use.
    save_lidar_ply      = False,  # also dump the old .ply alongside .bin
    save_depth_raw_png  = False,  # also dump a visualization depth PNG
    save_semantic_visual = True,  # also dump CityScapes-colored semantic PNG

    # NEW: lane GT (section 7).
    lane_sample_distance = 2.0,   # meters between sampled waypoints along a lane
    lane_max_distance    = 60.0,  # how far forward to sample lanes

    # NEW: run/versioning + split bookkeeping (sections 9, 11).
    run_id = 'run_0001',
    weather_name = 'ClearNoon',   # human-readable label for metadata; keep in
                                   # sync with CFG['weather'] above -- carla's
                                   # WeatherParameters object isn't JSON-safe.
    # Not a random per-frame split -- split by town/run, exactly because
    # consecutive frames are highly correlated. Configurable, not hard-coded.
    split_config = {'Town01': 'train', 'Town03': 'train', 'Town05': 'val'},

    validate_after_run = True,    # run validate_dataset() at the end of each town
)


# ---------------------------------------------------------------------------
# NEW: raw CARLA semantic segmentation tag IDs -> this project's class names.
# ---------------------------------------------------------------------------
# These numeric IDs are stored in the red channel of the RAW semantic PNG
# (semantic/000001.png, saved WITHOUT a ColorConverter below) and match
# CARLA's default "CityScapes" tag set on 0.9.13/0.9.14 content
# (carla.CityObjectLabel). If you're on a different CARLA version, verify
# this table against `carla.CityObjectLabel` before trusting SegFormer GT --
# do not assume it's stable across versions.
#
# Edit THIS table (not the raw PNGs) to change SegFormer's training class
# set later; the raw PNGs always keep CARLA's original numeric IDs so
# nothing here is destructive.
CARLA_SEMANTIC_TO_PROJECT = {
    0:  'unlabeled',
    1:  'building',
    2:  'fence',
    3:  'other',
    4:  'pedestrian',
    5:  'pole',
    6:  'roadline',
    7:  'road',
    8:  'sidewalk',
    9:  'vegetation',
    10: 'vehicle',
    11: 'wall',
    12: 'trafficsign',
    13: 'sky',
    14: 'ground',
    15: 'bridge',
    16: 'railtrack',
    17: 'guardrail',
    18: 'trafficlight',
    19: 'static',
    20: 'dynamic',
    21: 'water',
    22: 'terrain',
}
# ---------------------------------------------------------------------------


def setup_sensors(world, vehicle, bp):
    sensors = {}
    cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))

    rgb = bp.find('sensor.camera.rgb')
    rgb.set_attribute('image_size_x', str(CFG['img_w']))
    rgb.set_attribute('image_size_y', str(CFG['img_h']))
    rgb.set_attribute('fov', str(CFG['fov']))
    sensors['rgb'] = world.spawn_actor(rgb, cam_tf, attach_to=vehicle)

    sem = bp.find('sensor.camera.semantic_segmentation')
    sem.set_attribute('image_size_x', str(CFG['img_w']))
    sem.set_attribute('image_size_y', str(CFG['img_h']))
    sem.set_attribute('fov', str(CFG['fov']))
    sensors['sem'] = world.spawn_actor(sem, cam_tf, attach_to=vehicle)

    # NEW: depth sensor -- identical resolution/FOV/transform to RGB so every
    # pixel lines up 1:1, both for the metric depth.npy export and for the
    # occlusion check in extract_2d_bboxes_yolo(). Synchronized the same way
    # as every other sensor here (own queue, picked up generically by
    # grab_synced() below -- no changes needed there).
    depth = bp.find('sensor.camera.depth')
    depth.set_attribute('image_size_x', str(CFG['img_w']))
    depth.set_attribute('image_size_y', str(CFG['img_h']))
    depth.set_attribute('fov', str(CFG['fov']))
    sensors['depth'] = world.spawn_actor(depth, cam_tf, attach_to=vehicle)

    lid = bp.find('sensor.lidar.ray_cast')
    lid.set_attribute('range', str(CFG['lidar_range']))
    lid.set_attribute('channels', str(CFG['lidar_channels']))
    lid.set_attribute('points_per_second', str(CFG['lidar_pps']))
    lid.set_attribute('rotation_frequency', '20')   # = 1 / fixed_delta_seconds
    sensors['lidar'] = world.spawn_actor(
        lid, carla.Transform(carla.Location(z=2.4)), attach_to=vehicle)

    queues = {}
    for name, s in sensors.items():
        q = queue.Queue()
        s.listen(q.put)
        queues[name] = q
    return sensors, queues


def drain_queues(queues):
    """Empty out any backlog (e.g. from warm-up ticks) so the recording
    loop starts from a clean, synchronized state."""
    for name, q in queues.items():
        n = 0
        while True:
            try:
                q.get_nowait()
                n += 1
            except queue.Empty:
                break
        if n:
            print(f'  drained {n} stale "{name}" frames before recording')


class FrameDropped(Exception):
    """Raised when a sensor genuinely never produced a frame for frame_id
    (as opposed to just arriving slightly out of order)."""
    pass


def grab_synced(queues, frame_id, buffers, timeout=5.0):
    data = {}
    for name, q in queues.items():
        buffered = buffers.get(name)
        if buffered is not None and buffered.frame == frame_id:
            data[name] = buffered
            del buffers[name]
            continue
        if buffered is not None and buffered.frame > frame_id:
            raise FrameDropped(f'{name} has no data for frame {frame_id} '
                                f'(next available is {buffered.frame})')
        if buffered is not None and buffered.frame < frame_id:
            del buffers[name]   # stale leftover from an earlier overshoot, discard
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
    return data


# extract_labels() below was always a no-op stub (returned [] unconditionally)
# -- extract_3d_bboxes_lidar() further down was already doing the real
# KITTI-style 3D GT generation and writing it to labels_3d/. Since the new
# directory layout only has labels_3d/ + label_2/, and the labels/yolo,
# label_2/, the dead call to this stub was removed from collect_town(); the
# function itself is left here unchanged in case you were relying on its
# signature/docstring elsewhere.
def extract_labels(world, lidar_sensor):
    """YOUR JOB — see the CARLA bounding-box tutorial.
    For each vehicle/walker actor:
      1. read actor.bounding_box (extent + transform)
      2. transform world frame -> LiDAR frame
      3. drop actors beyond lidar_range or with no returns
      4. return KITTI-style rows: class, dims, location, rotation_y
    """
    return []          # stub so the pipeline runs; replace this


# ---------------------------------------------------------------------------
# 2D bounding-box (YOLO) ground truth
# ---------------------------------------------------------------------------
# Maps CARLA actor type prefixes -> YOLO class ids. Extend/renumber to match
# whatever your model's data.yaml expects.
YOLO_CLASSES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'bus': 5,
    'truck': 7,
    'traffic_light': 9,
    'stop_sign': 11,
}

# ---------------------------------------------------------------------------
# CARLA actor -> COCO/YOLO class mapping
# ---------------------------------------------------------------------------
# CARLA type_ids never look like 'car...' or 'person...' — they look like
# 'vehicle.tesla.model3' or 'walker.pedestrian.0004' — so we can't prefix-
# match against YOLO_CLASSES directly. Classify by structure instead.

# Known non-"car" vehicle blueprints (CARLA 0.9.13/0.9.14 default content).
# Anything vehicle.* not listed here defaults to 'car'.
BICYCLE_BLUEPRINTS = {
    'vehicle.bh.crossbike',
    'vehicle.diamondback.century',
    'vehicle.gazelle.omafiets',
}
MOTORCYCLE_BLUEPRINTS = {
    'vehicle.harley-davidson.low_rider',
    'vehicle.kawasaki.ninja',
    'vehicle.vespa.zx125',
    'vehicle.yamaha.yzf',
}
BUS_BLUEPRINTS = {
    'vehicle.mitsubishi.fusorosa',
}
TRUCK_BLUEPRINTS = {
    'vehicle.carlamotors.carlacola',
    'vehicle.carlamotors.firetruck',
    'vehicle.carlamotors.european_hgv',
    'vehicle.tesla.cybertruck',
    'vehicle.mercedes.sprinter',
    'vehicle.volkswagen.t2',
    'vehicle.volkswagen.t2_2021',
    'vehicle.ford.ambulance',
}


def classify_actor(actor):
    """Map a CARLA actor to a (class_name, class_id) pair from YOLO_CLASSES,
    or (None, None) if it's not a type we're labeling."""
    type_id = actor.type_id

    if type_id.startswith('walker.pedestrian'):
        return 'person', YOLO_CLASSES['person']

    if not type_id.startswith('vehicle.'):
        return None, None

    # Prefer CARLA's own base_type attribute when present (0.9.14+) --
    # it already speaks car/truck/bus/motorcycle/bicycle.
    base_type = actor.attributes.get('base_type')
    if base_type in YOLO_CLASSES:
        return base_type, YOLO_CLASSES[base_type]

    if type_id in BICYCLE_BLUEPRINTS:
        return 'bicycle', YOLO_CLASSES['bicycle']
    if type_id in MOTORCYCLE_BLUEPRINTS:
        return 'motorcycle', YOLO_CLASSES['motorcycle']
    if type_id in BUS_BLUEPRINTS:
        return 'bus', YOLO_CLASSES['bus']
    if type_id in TRUCK_BLUEPRINTS:
        return 'truck', YOLO_CLASSES['truck']

    # Fallback for any blueprint not in the tables above (e.g. a future
    # CARLA content update): use wheel count as a last resort.
    n_wheels = actor.attributes.get('number_of_wheels')
    if n_wheels == '2':
        return 'motorcycle', YOLO_CLASSES['motorcycle']

    return 'car', YOLO_CLASSES['car']
# ---------------------------------------------------------------------------


def build_projection_matrix(w, h, fov):
    """Pinhole camera intrinsics matrix for CARLA's RGB camera."""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(location, K, world_2_camera):
    """Project a carla.Location (world space) into 2D image pixel coords.
    Returns (x, y, depth); depth <= 0 means the point is behind the camera."""
    point = np.array([location.x, location.y, location.z, 1.0])
    point_camera = np.dot(world_2_camera, point)

    # CARLA/UE4 world axes are (x-forward, y-right, z-up); standard camera
    # convention is (x-right, y-down, z-forward). Swap accordingly.
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])

    point_img = np.dot(K, point_camera)
    depth = point_img[2]
    if depth != 0:
        point_img[0] /= depth
        point_img[1] /= depth
    return point_img[0], point_img[1], depth


# NEW: metric depth decoding.
def decode_depth(depth_image):
    """Decode a carla.Image from sensor.camera.depth into metric depth
    (meters), shape (H, W) float32.

    CARLA encodes depth across all three color channels for precision:
        normalized = (R + G*256 + B*256^2) / (256^3 - 1)
        depth_m    = 1000.0 * normalized      # 1000 m = camera far plane

    We decode the raw BGRA buffer directly rather than going through
    carla.ColorConverter.Depth, because that converter produces an 8-bit
    grayscale *visualization* image (lossy, not a numeric training target --
    see section 2's explicit warning about this).
    """
    array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    array = array.reshape((depth_image.height, depth_image.width, 4)).astype(np.float32)
    B, G, R = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    depth_m = 1000.0 * normalized
    return depth_m.astype(np.float32)


# NEW: practical (non-raycast) occlusion check for 2D YOLO GT.
def is_visible_via_depth(depth_array, img_w, img_h, x_min, y_min, x_max, y_max,
                          actor_depth, margin=1.0, n_samples=5):
    """Section 5 visibility rule.

    Samples a handful of pixels inside the projected 2D box against the
    depth camera's decoded metric buffer. A sampled pixel counts as
    "seeing" the actor if the buffer's distance there is close to (or
    farther than) the actor's own estimated distance -- i.e. nothing closer
    is blocking that pixel. If EVERY sampled pixel shows something
    significantly closer than the actor, we call the actor occluded and
    drop the box; if even one sample is clear, we keep it (partial
    occlusion, e.g. a rider on a motorcycle, or a pedestrian half behind a
    parked car, is still a valid training example).

    LIMITATION (documented per instructions, not silently glossed over):
    this is 5 point samples, not a per-pixel raycast/mask. A thin occluder
    (a pole, a sign edge) can slip between all 5 samples and leave a mostly-
    hidden actor incorrectly marked visible; conversely a stray closer
    object caught by all 5 samples on an otherwise-visible actor could drop
    a valid box. If you need pixel-perfect occlusion later, the semantic
    segmentation buffer (semantic/) can be used for a per-pixel class/ID
    mask check instead -- that was intentionally NOT implemented here to
    keep this cheap, per the instruction to avoid expensive per-actor
    raycasts unless necessary.
    """
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    hw, hh = (x_max - x_min) / 4.0, (y_max - y_min) / 4.0
    sample_points = [
        (cx, cy),
        (cx - hw, cy - hh), (cx + hw, cy - hh),
        (cx - hw, cy + hh), (cx + hw, cy + hh),
    ][:n_samples]

    for sx, sy in sample_points:
        px = int(np.clip(sx, 0, img_w - 1))
        py = int(np.clip(sy, 0, img_h - 1))
        buffer_depth = depth_array[py, px]
        if buffer_depth >= actor_depth - margin:
            return True
    return False


def extract_2d_bboxes_yolo(world, rgb_sensor, ego_vehicle, img_w, img_h, fov,
                            max_distance=None,
                            depth_array=None, visibility_check=False,
                            occlusion_margin=1.0):
    """Project every vehicle/walker actor's 3D bounding box into the RGB
    camera's image plane and return YOLO-format ground-truth lines:

        class_id x_center y_center width height      (all normalized 0..1)

    Notes / limitations:
      - This is a geometric projection PLUS an optional depth-buffer
        visibility check (see is_visible_via_depth() for exactly what that
        does and doesn't catch). It is still not a full per-pixel
        occlusion/segmentation mask.
      - Boxes are clipped to the image bounds; actors that are entirely
        outside the frame or entirely behind the camera are dropped.
      - max_distance only controls whether a LABEL is generated -- actors
        beyond it are still rendered normally in the RGB image and still
        exist in the CARLA world (section 3 requirement).
    """
    K = build_projection_matrix(img_w, img_h, fov)
    world_2_camera = np.array(rgb_sensor.get_transform().get_inverse_matrix())

    actors = list(world.get_actors().filter('vehicle.*')) + \
             list(world.get_actors().filter('walker.pedestrian.*'))

    ego_loc = ego_vehicle.get_transform().location
    lines = []

    for actor in actors:
        if actor.id == ego_vehicle.id:
            continue

        if max_distance is not None:
            if actor.get_transform().location.distance(ego_loc) > max_distance:
                continue

        cls_name, cls_id = classify_actor(actor)
        if cls_id is None:
            continue  # unmapped actor type, skip

        bb = actor.bounding_box
        verts = bb.get_world_vertices(actor.get_transform())

        xs, ys, depths, in_front = [], [], [], 0
        for v in verts:
            x, y, depth = get_image_point(v, K, world_2_camera)
            if depth <= 0:
                continue
            in_front += 1
            xs.append(x)
            ys.append(y)
            depths.append(depth)  # NEW: kept for the occlusion check below

        if in_front == 0:
            continue  # entirely behind the camera

        x_min, x_max = max(min(xs), 0.0), min(max(xs), img_w - 1.0)
        y_min, y_max = max(min(ys), 0.0), min(max(ys), img_h - 1.0)

        if x_max <= x_min or y_max <= y_min:
            continue  # off-screen or degenerate after clipping to the frame

        if (x_max - x_min) < CFG['min_box_px'] or (y_max - y_min) < CFG['min_box_px']:
            continue  # too small to carry real training signal, just noise

        # NEW: visibility / occlusion filtering (section 5). Only runs if a
        # decoded depth buffer was actually handed in -- callers that don't
        # care (or don't have a depth frame available) just get the old
        # geometry-only behavior.
        if visibility_check and depth_array is not None:
            actor_depth = float(np.mean(depths))
            if not is_visible_via_depth(depth_array, img_w, img_h,
                                         x_min, y_min, x_max, y_max,
                                         actor_depth, margin=occlusion_margin):
                continue  # no meaningfully visible pixels -- likely fully occluded

        x_center = (x_min + x_max) / 2.0 / img_w
        y_center = (y_min + y_max) / 2.0 / img_h
        width    = (x_max - x_min) / img_w
        height   = (y_max - y_min) / img_h

        lines.append((cls_name,
                      f'{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'))

    return lines
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3D bounding boxes, in the LiDAR sensor's own frame (PointPillars / OpenPCDet
# style) -- for assessing a point-cloud 3D detector rather than an image one.
# ---------------------------------------------------------------------------
def world_to_lidar_matrix(lidar_sensor):
    """4x4 matrix mapping a world-space point into the LiDAR's local frame --
    the same frame the raw point cloud (saved to velodyne/*.bin) is already
    expressed in, EXCEPT for the y-flip both are given below (see BUGFIX
    note in extract_3d_bboxes_lidar and the y-flip in collect_town's .bin
    writer -- they must both flip y the same way or point counts and box
    positions land in mismatched frames)."""
    return np.array(lidar_sensor.get_transform().get_inverse_matrix())


def extract_3d_bboxes_lidar(world, lidar_sensor, lidar_measurement, ego_vehicle,
                             max_distance=None, min_points=1):
    """Ground-truth 3D boxes in the LiDAR sensor's coordinate frame.

    Each line:
        class x y z dx dy dz yaw num_points

      x, y, z    -- box center, meters, in the LiDAR frame
                    (x forward, y left, z up -- right-handed, KITTI/velodyne
                    convention; CARLA itself is left-handed with y to the
                    right, so y is negated below to convert).
      dx, dy, dz -- box size along x, y, z (length, width, height), meters.
      yaw        -- heading around the vertical (z) axis, radians,
                    counter-clockwise from the LiDAR's +x axis.
      num_points -- raw points from THIS frame's point cloud that fall
                    inside the box. Doubles as a free occlusion/visibility
                    proxy and as the usual KITTI easy/moderate/hard signal --
                    boxes below `min_points` are dropped outright.

    IMPORTANT: coordinate-frame conversions (handedness, axis order, sign of
    yaw) are the single easiest thing to get subtly wrong here, and it can't
    be verified without a running simulator. Before trusting a full dataset
    run, sanity-check a handful of frames: plot a few of these boxes over
    their velodyne/*.bin point cloud (e.g. with Open3D) and confirm they
    actually wrap the vehicles, with the heading arrow pointing the way the
    car is facing.
    """
    w2l = world_to_lidar_matrix(lidar_sensor)  # world -> lidar-local, 4x4

    raw = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    cloud_xyz = raw[:, :3].copy()
    # BUGFIX (section 6 coordinate-conversion check): the box center below is
    # converted to the y-left (KITTI) frame via `y = -center_lidar[1]`, but
    # this containment test was comparing that flipped center against the
    # RAW (y-right, native CARLA) cloud -- i.e. two points expressed in
    # mirrored frames, which silently shifted the "inside box" test to
    # y = -true_y and under/over-counted `num_points` for anything not
    # essentially on the y=0 line. Flipping the cloud once here, up front,
    # puts it in the same y-left frame as the box center used below so the
    # containment math is actually comparing like with like. This does NOT
    # change the previously-working `x, y, z` box output convention at all.
    cloud_xyz[:, 1] = -cloud_xyz[:, 1]

    actors = list(world.get_actors().filter('vehicle.*')) + \
             list(world.get_actors().filter('walker.pedestrian.*'))

    ego_loc = ego_vehicle.get_transform().location
    lidar_yaw = lidar_sensor.get_transform().rotation.yaw
    lines = []

    for actor in actors:
        if actor.id == ego_vehicle.id:
            continue

        if max_distance is not None:
            if actor.get_transform().location.distance(ego_loc) > max_distance:
                continue

        cls_name, cls_id = classify_actor(actor)
        if cls_id is None:
            continue  # unmapped actor type, skip

        tf = actor.get_transform()
        bb = actor.bounding_box

        # bb.location is an offset from the actor's own origin (non-zero for
        # some pedestrian rigs), so transform it through the actor's pose
        # first to get the box center in world space.
        center_world = tf.transform(bb.location)
        center_h = np.array([center_world.x, center_world.y, center_world.z, 1.0])
        center_lidar = w2l.dot(center_h)[:3]
        x, y, z = center_lidar[0], -center_lidar[1], center_lidar[2]

        dx = bb.extent.x * 2.0   # length, along the actor's forward axis
        dy = bb.extent.y * 2.0   # width
        dz = bb.extent.z * 2.0   # height

        yaw = -np.deg2rad(tf.rotation.yaw - lidar_yaw)
        yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))  # wrap to [-pi, pi]

        # raw points inside the (rotated) box -- rotate the cloud into the
        # box's own axes, then do a simple axis-aligned containment test.
        # `cloud_xyz` is now y-flipped to match (x, y, z) above -- see
        # BUGFIX note where cloud_xyz is built.
        rel = cloud_xyz - np.array([x, y, z])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        local_x = rel[:, 0] * cos_y + rel[:, 1] * sin_y
        local_y = -rel[:, 0] * sin_y + rel[:, 1] * cos_y
        local_z = rel[:, 2]
        inside = (
            (np.abs(local_x) <= dx / 2.0) &
            (np.abs(local_y) <= dy / 2.0) &
            (np.abs(local_z) <= dz / 2.0)
        )
        num_points = int(inside.sum())

        if num_points < min_points:
            continue

        lines.append(f'{cls_name} {x:.3f} {y:.3f} {z:.3f} '
                     f'{dx:.3f} {dy:.3f} {dz:.3f} {yaw:.4f} {num_points}')

    return lines
# ---------------------------------------------------------------------------


# NEW: lane ground truth (section 7).
def extract_lane_gt(world, ego_vehicle, rgb_sensor, img_w, img_h, fov,
                     max_distance=60.0, sample_distance=2.0):
    """Collects nearby lane waypoints (ego lane + immediate left/right
    neighbours) from CARLA's map/waypoint API, samples points forward along
    each lane up to max_distance, and projects them into the RGB camera's
    image plane.

    This is deliberately a clean, MODEL-INDEPENDENT representation (world
    points + projected image points + marking metadata) -- it is NOT the
    literal Ultra-Fast-Lane-Detection v2 row-anchor annotation format.
    UFLD-v2 needs lane x-positions resampled onto a fixed row-anchor grid at
    its own training resolution, which is a deterministic, purely-numeric
    post-processing step you can run on this JSON later -- it depends on
    training config choices, not on anything the simulator can decide, so
    it isn't invented here.

    LIMITATION (documented per instructions): only the ego's current lane
    and its immediate left/right neighbors are captured, not the full lane
    graph (adjacent-but-not-touching lanes, opposing traffic lanes further
    over, lanes beyond an intersection, etc. are not included). If your
    lane detector needs more than 3 lanes in view, extend `lane_sources`
    below using further get_left_lane()/get_right_lane() calls.
    """
    carla_map = world.get_map()
    ego_loc = ego_vehicle.get_transform().location
    K = build_projection_matrix(img_w, img_h, fov)
    world_2_camera = np.array(rgb_sensor.get_transform().get_inverse_matrix())

    start_wp = carla_map.get_waypoint(ego_loc, project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
    if start_wp is None:
        return []  # ego off-road / no nearby lane -- document, don't fabricate

    lane_sources = {'ego': start_wp}
    left = start_wp.get_left_lane()
    if left is not None and left.lane_type == carla.LaneType.Driving:
        lane_sources['left'] = left
    right = start_wp.get_right_lane()
    if right is not None and right.lane_type == carla.LaneType.Driving:
        lane_sources['right'] = right

    lanes = []
    for lane_label, wp0 in lane_sources.items():
        world_pts, img_pts, visible_flags = [], [], []
        wp = wp0
        traveled = 0.0
        while wp is not None and traveled <= max_distance:
            loc = wp.transform.location
            world_pts.append([loc.x, loc.y, loc.z])
            x, y, depth = get_image_point(loc, K, world_2_camera)
            vis = bool(depth > 0 and 0 <= x <= img_w and 0 <= y <= img_h)
            img_pts.append([float(x), float(y)])
            visible_flags.append(vis)

            nxt = wp.next(sample_distance)
            if not nxt:
                break
            wp = nxt[0]
            traveled += sample_distance

        marking = wp0.right_lane_marking if lane_label == 'right' else wp0.left_lane_marking
        lanes.append({
            'lane_id': lane_label,
            'marking_type': str(marking.type) if marking is not None else None,
            'marking_color': str(marking.color) if marking is not None else None,
            'world_points': world_pts,
            'image_points': img_pts,
            'visible': visible_flags,
        })

    return lanes
# ---------------------------------------------------------------------------


def spawn_traffic(world, bp, tm, ego_location=None):
    """Spawn CFG['n_vehicles'] total vehicles, but force a minimum number of
    bicycles and motorcycles first (CFG['min_bicycles']/['min_motorcycles']),
    biased toward spawn points near the ego vehicle.

    Why bias by distance: even after guaranteeing a minimum count, a
    two-wheeler spawned at a random point across the whole map is usually
    far from the ego's camera for most/all of the frames it's ever seen in
    -- producing GT boxes too small to be a useful training example (see
    min_box_px filtering above). Sorting candidate spawn points by distance
    to the ego and preferring the closest ones for two-wheelers means more
    instances end up at a size the model can actually learn from.
    """
    vehicles = []
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    if ego_location is not None:
        # Points within the bias radius, closest-first; everything else
        # after, still in its (already shuffled) random order.
        near = sorted(
            (sp for sp in spawn_points
             if sp.location.distance(ego_location) <= CFG['two_wheeler_bias_radius']),
            key=lambda sp: sp.location.distance(ego_location))
        far = [sp for sp in spawn_points if sp not in near]
        two_wheeler_spawn_points = near + far
    else:
        two_wheeler_spawn_points = spawn_points

    all_vehicle_bps = list(bp.filter('vehicle.*'))
    bike_bps = [b for b in all_vehicle_bps if b.id in BICYCLE_BLUEPRINTS]
    moto_bps = [b for b in all_vehicle_bps if b.id in MOTORCYCLE_BLUEPRINTS]

    def spawn_forced(blueprint_pool, n_target, label, candidate_points):
        """Try spawn points one at a time, closest-to-ego first, until
        n_target actors of this pool spawn or we run out of points. Points
        that get tried (spawned or not) are removed from the shared pool so
        the general fill-in pass below doesn't reuse them."""
        spawned = 0
        while spawned < n_target and candidate_points:
            sp = candidate_points.pop(0)
            if sp in spawn_points:
                spawn_points.remove(sp)
            chosen_bp = random.choice(blueprint_pool)
            v = world.try_spawn_actor(chosen_bp, sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                vehicles.append(v)
                spawned += 1
        if spawned < n_target:
            print(f'  [warn] only spawned {spawned}/{n_target} {label} '
                  f'(ran out of free spawn points near the ego -- try a '
                  f'larger two_wheeler_bias_radius or fewer n_vehicles)')
        return spawned

    if bike_bps:
        spawn_forced(bike_bps, CFG['min_bicycles'], 'bicycles',
                     list(two_wheeler_spawn_points))
    else:
        print('  [warn] no bicycle blueprints found on this CARLA install -- '
              'check BICYCLE_BLUEPRINTS against your CARLA version\'s content')

    if moto_bps:
        spawn_forced(moto_bps, CFG['min_motorcycles'], 'motorcycles',
                     [sp for sp in two_wheeler_spawn_points if sp in spawn_points])
    else:
        print('  [warn] no motorcycle blueprints found on this CARLA install -- '
              'check MOTORCYCLE_BLUEPRINTS against your CARLA version\'s content')

    # Fill the remaining budget with the general vehicle pool (cars, trucks,
    # buses, and occasionally more two-wheelers by chance).
    remaining = max(0, CFG['n_vehicles'] - len(vehicles))
    for sp in spawn_points[:remaining]:
        v = world.try_spawn_actor(random.choice(all_vehicle_bps), sp)
        if v:
            v.set_autopilot(True, tm.get_port())
            vehicles.append(v)

    return vehicles


# NEW: pedestrian spawning. CFG['n_walkers'] existed in the original CFG but
# nothing ever spawned a walker -- spawn_traffic() only ever handles
# vehicle.* blueprints. This fixes that gap: every collect_town() run now
# actually gets pedestrians, with retries so the count is close to
# n_walkers rather than best-effort.
def spawn_walkers(world, bp, n_walkers):
    """Spawn n_walkers pedestrians with wandering AI controllers, CARLA's
    standard walker pattern (same approach as CARLA's own
    generate_traffic.py example):

      1. try_spawn_actor() a walker.pedestrian.* blueprint at a random
         point from the navigation mesh.
      2. tick once (AI walker controllers are unreliable if start() is
         called in the same tick the walker was spawned in).
      3. spawn a controller.ai.walker attached to each walker, tick again,
         then start() it and send it to another random nav-mesh point.

    Returns (walkers, controllers) -- both must be appended to the town's
    `actors` list for cleanup, and controllers must have .stop() called on
    them BEFORE destroying actors (handled in collect_town's finally block).
    """
    walker_bps = list(bp.filter('walker.pedestrian.*'))
    if not walker_bps:
        print('  [warn] no walker.pedestrian.* blueprints found on this '
              'CARLA install -- no pedestrians will be spawned')
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
        print('  [warn] spawned 0 pedestrians -- nav mesh may be missing '
              'for this town, or n_walkers/pedestrian_spawn_retries is too low')
        return [], []

    world.tick()  # let the walkers exist server-side before attaching controllers

    controller_bp = bp.find('controller.ai.walker')
    controllers = []
    for walker in walkers:
        controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if controller is not None:
            controllers.append(controller)

    world.tick()  # let controllers exist before calling start()/go_to_location()

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
        print(f'  [warn] only spawned {len(walkers)}/{n_walkers} pedestrians '
              f'(nav mesh exhausted or spawn collisions -- this town may just '
              f'have less walkable area; raise pedestrian_spawn_retries to try harder)')

    return walkers, controllers


# NEW: metadata writer (section 8).
def write_metadata(path, frame_id, town, run_id, ego,
                    n_yolo, n_3d, n_lanes, n_lidar_points, sync_status,
                    sensor_frame_ids):
    tf = ego.get_transform()
    vel = ego.get_velocity()
    speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))
    meta = {
        'frame_id': frame_id,
        'town': town,
        'run_id': run_id,
        'timestamp': time.time(),
        'ego_location': {'x': tf.location.x, 'y': tf.location.y, 'z': tf.location.z},
        'ego_rotation': {'pitch': tf.rotation.pitch, 'yaw': tf.rotation.yaw, 'roll': tf.rotation.roll},
        'ego_velocity': {'x': vel.x, 'y': vel.y, 'z': vel.z},
        'ego_speed': speed,
        'rgb_frame_id': sensor_frame_ids.get('rgb'),
        'semantic_frame_id': sensor_frame_ids.get('sem'),
        'depth_frame_id': sensor_frame_ids.get('depth'),
        'lidar_frame_id': sensor_frame_ids.get('lidar'),
        'weather': CFG.get('weather_name'),
        'camera_resolution': [CFG['img_w'], CFG['img_h']],
        'camera_fov': CFG['fov'],
        'lidar_range': CFG['lidar_range'],
        'num_yolo_objects': n_yolo,
        'num_3d_objects': n_3d,
        'num_lanes': n_lanes,
        'num_lidar_points': n_lidar_points,
        'sync_status': sync_status,
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


# UPDATED: directory structure -- kept FLAT and matching the original
# script's own directory names (dataset/<Town>/<dir>/...), not the nested
# dataset/<Town>/<run_id>/... layout from the first pass, because other
# code on top of this collector already reads image_2/velodyne/semantic/
# labels_yolo/labels_3d directly. run_id is still recorded inside every
# metadata/*.json for provenance -- it just no longer changes the path.
#
#   image_2/        RGB PNGs                    (was rgb/)
#   velodyne/        LiDAR, KITTI-style .bin     [N,4] float32 x,y,z,intensity
#                    (was lidar/ -- .bin now lives under the ORIGINAL
#                    velodyne/ name, matching common KITTI convention where
#                    velodyne/ holds .bin files. The debug .ply, if enabled,
#                    goes in the separate velodyne_ply/ so it never collides
#                    with the .bin the downstream code expects here.)
#   semantic/        raw numeric CARLA class-ID PNGs (the actual GT mask)
#   semantic_visual/ CityScapes-colored PNGs, visualization only, optional
#   depth/           decoded metric depth .npy   (H, W) float32, meters
#   depth_raw/       optional visualization depth PNG
#   label_2/         KITTI-style 3D box lines (see note in collect_town --
#                    this dir existed in the original script as a
#                    permanently-empty stub; it's now populated with the
#                    same real 3D GT as labels_3d/ rather than being empty,
#                    since downstream code may already expect this exact
#                    directory to have content. Flag this to your other
#                    code if it was relying on label_2/ being empty.)
#   labels_yolo/     2D YOLO GT .txt (was labels/yolo/)
#   labels_3d/       KITTI-style 3D box lines, same content as label_2/
#   lanes/           lane GT .json
#   metadata/        per-frame metadata .json
def get_run_root(town):
    return f"{CFG['out']}/{town}"


def ensure_dataset_dirs(root):
    subdirs = [
        'image_2', 'velodyne',
        'semantic', 'semantic_visual',
        'depth', 'depth_raw',
        'label_2', 'labels_yolo', 'labels_3d',
        'lanes', 'metadata',
    ]
    if CFG.get('save_lidar_ply'):
        subdirs.append('velodyne_ply')
    for d in subdirs:
        os.makedirs(f'{root}/{d}', exist_ok=True)


# NEW: validation / safety checks (section 13).
def validate_dataset(root, town, sample_limit=None):
    """Walk the saved frames for one town/run and sanity-check every
    modality. Prints the summary table your spec asked for. Returns the
    summary dict in case a caller wants to act on it programmatically."""
    rgb_dir = f'{root}/image_2'
    if not os.path.isdir(rgb_dir):
        print(f'  [validate] no image_2/ directory found under {root}, skipping validation')
        return None

    frame_ids = sorted(f[:-4] for f in os.listdir(rgb_dir) if f.endswith('.png'))
    if sample_limit:
        frame_ids = frame_ids[:sample_limit]
    n = len(frame_ids)

    counts = Counter()
    yolo_class_counts = Counter()
    box3d_class_counts = Counter()
    lidar_point_totals = []
    lane_totals = []
    bad_frames = []

    for fid in frame_ids:
        ok = True

        rgb_path = f'{root}/image_2/{fid}.png'
        if os.path.isfile(rgb_path):
            counts['rgb'] += 1
        else:
            ok = False

        sem_path = f'{root}/semantic/{fid}.png'
        if os.path.isfile(sem_path):
            counts['semantic'] += 1
        else:
            ok = False

        depth_path = f'{root}/depth/{fid}.npy'
        if os.path.isfile(depth_path):
            try:
                d = np.load(depth_path)
                if d.ndim == 2 and np.isfinite(d).all():
                    counts['depth'] += 1
                else:
                    ok = False
            except Exception:
                ok = False
        else:
            ok = False

        lidar_path = f'{root}/velodyne/{fid}.bin'
        if os.path.isfile(lidar_path):
            pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            if pts.dtype == np.float32 and pts.shape[1] == 4:
                counts['lidar'] += 1
                lidar_point_totals.append(pts.shape[0])
            else:
                ok = False
        else:
            ok = False

        yolo_path = f'{root}/labels_yolo/{fid}.txt'
        if os.path.isfile(yolo_path):
            valid = True
            with open(yolo_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        valid = False
                        break
                    cls_id, xc, yc, w, h = parts
                    vals = [float(xc), float(yc), float(w), float(h)]
                    if not all(0.0 <= v <= 1.0 for v in vals):
                        valid = False
                        break
                    yolo_class_counts[cls_id] += 1
            counts['yolo'] += 1 if valid else 0
            ok = ok and valid
        else:
            ok = False

        kitti_path = f'{root}/labels_3d/{fid}.txt'
        if os.path.isfile(kitti_path):
            valid = True
            with open(kitti_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 9:
                        valid = False
                        break
                    cls_name = parts[0]
                    nums = [float(p) for p in parts[1:]]
                    if any(np.isnan(v) or np.isinf(v) for v in nums):
                        valid = False
                        break
                    box3d_class_counts[cls_name] += 1
            counts['3d_labels'] += 1 if valid else 0
            ok = ok and valid
        else:
            ok = False

        lane_path = f'{root}/lanes/{fid}.json'
        if os.path.isfile(lane_path):
            try:
                with open(lane_path) as f:
                    lanes = json.load(f)
                valid = isinstance(lanes, list)
                if valid:
                    for lane in lanes:
                        for (px, py) in lane.get('image_points', []):
                            if not (-50 <= px <= CFG['img_w'] + 50 and
                                    -50 <= py <= CFG['img_h'] + 50):
                                valid = False
                                break
                counts['lanes'] += 1 if valid else 0
                lane_totals.append(len(lanes) if valid else 0)
                ok = ok and valid
            except Exception:
                ok = False
        else:
            ok = False

        meta_path = f'{root}/metadata/{fid}.json'
        if os.path.isfile(meta_path):
            try:
                json.load(open(meta_path))
                counts['metadata'] += 1
            except Exception:
                ok = False
        else:
            ok = False

        if not ok:
            bad_frames.append(fid)

    print(f'\nFrames saved: {n}\n')
    for label, key in [('RGB', 'rgb'), ('LiDAR', 'lidar'), ('Semantic', 'semantic'),
                        ('Depth', 'depth'), ('YOLO', 'yolo'), ('3D labels', '3d_labels'),
                        ('Lane GT', 'lanes'), ('Metadata', 'metadata')]:
        print(f'{label + ":":<12}{counts.get(key, 0)}/{n}')

    print('\nYOLO instances by class:',
          dict(yolo_class_counts) if yolo_class_counts else '(none)')
    print('3D instances by class:',
          dict(box3d_class_counts) if box3d_class_counts else '(none)')
    print(f'frames with a validation issue: {len(bad_frames)}'
          + (f' (e.g. {bad_frames[:5]})' if bad_frames else ''))
    if lidar_point_totals:
        print(f'average LiDAR points/frame: {np.mean(lidar_point_totals):.0f}')
    if lane_totals:
        print(f'average lanes/frame: {np.mean(lane_totals):.2f}')

    return {
        'n_frames': n, 'counts': dict(counts),
        'yolo_class_counts': dict(yolo_class_counts),
        'box3d_class_counts': dict(box3d_class_counts),
        'bad_frames': bad_frames,
    }


def collect_town(client, town, start_idx):
    print(f'\nloading {town} ...')
    client.load_world(town)
    time.sleep(10)
    world = client.get_world()                      # re-fetch: episode restarted
    world.set_weather(CFG['weather'])

    bp = world.get_blueprint_library()
    tm = client.get_trafficmanager()

    original = world.get_settings()
    sensors, queues, actors, controllers = {}, {}, [], []   # defined before try
    class_counts = Counter()                        # GT instance counts, this town

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)                # enable TM sync AFTER world sync

        ego = world.spawn_actor(
            bp.filter('vehicle.tesla.model3')[0],
            random.choice(world.get_map().get_spawn_points()))
        ego.set_autopilot(True, tm.get_port())
        actors.append(ego)

        actors += spawn_traffic(world, bp, tm, ego_location=ego.get_transform().location)

        # NEW: pedestrians -- previously n_walkers was configured but never
        # actually spawned anywhere.
        walkers, controllers = spawn_walkers(world, bp, CFG['n_walkers'])
        actors += walkers
        actors += controllers

        sensors, queues = setup_sensors(world, ego, bp)
        actors += list(sensors.values())

        # UPDATED: new directory layout (section 9) -- dataset/<town>/<run_id>/...
        root = get_run_root(town)
        ensure_dataset_dirs(root)

        for _ in range(60):                         # let traffic settle
            world.tick()

        drain_queues(queues)                        # clear warm-up backlog

        buffers = {}
        dropped_frames = 0
        saved, tick = start_idx, 0
        while saved < start_idx + CFG['frames_per_town']:
            frame_id = world.tick()
            try:
                data = grab_synced(queues, frame_id, buffers)
            except FrameDropped as e:
                dropped_frames += 1
                print(f'\n  [warn] {e} -- skipping this frame '
                      f'({dropped_frames} dropped so far)')
                continue

            tick += 1
            if tick % CFG['tick_skip']:
                continue

            base = f'{saved:06d}'

            # --- RGB -----------------------------------------------------
            data['rgb'].save_to_disk(f'{root}/image_2/{base}.png')

            # --- Semantic (UPDATED: raw class-ID PNG is now the ground
            # truth, saved into the ORIGINAL semantic/ dir; CityScapes
            # color version is a separate, optional visualization in its
            # own semantic_visual/ dir, never the training target itself) -
            data['sem'].save_to_disk(f'{root}/semantic/{base}.png')
            if CFG.get('save_semantic_visual'):
                data['sem'].save_to_disk(f'{root}/semantic_visual/{base}.png',
                                         carla.ColorConverter.CityScapesPalette)

            # --- NEW: Depth ------------------------------------------------
            depth_array = decode_depth(data['depth'])
            np.save(f'{root}/depth/{base}.npy', depth_array)
            if CFG.get('save_depth_raw_png'):
                data['depth'].save_to_disk(f'{root}/depth_raw/{base}.png',
                                           carla.ColorConverter.Depth)

            # --- NEW: LiDAR .bin (UPDATED representation; .ply optional) --
            # Saved into the ORIGINAL velodyne/ dir (as .bin instead of
            # .ply) -- matches common KITTI convention where velodyne/
            # holds .bin point clouds, and keeps the path any downstream
            # code already points at.
            lidar_points = np.frombuffer(data['lidar'].raw_data, dtype=np.float32).reshape(-1, 4).copy()
            # Flip y to the same x-forward,y-left,z-up (KITTI/right-handed)
            # frame the 3D box labels below use -- see the BUGFIX note in
            # extract_3d_bboxes_lidar for why points and boxes must share a
            # frame. Native CARLA lidar points come out x-forward,y-right,
            # z-up (left-handed), matching the world/vertex convention used
            # everywhere else in this file.
            lidar_points[:, 1] = -lidar_points[:, 1]
            lidar_points.tofile(f'{root}/velodyne/{base}.bin')
            if CFG.get('save_lidar_ply'):
                data['lidar'].save_to_disk(f'{root}/velodyne_ply/{base}.ply')

            # --- 2D YOLO GT (UPDATED: yolo_max_distance + visibility) ----
            yolo_pairs = extract_2d_bboxes_yolo(
                world, sensors['rgb'], ego,
                CFG['img_w'], CFG['img_h'], CFG['fov'],
                max_distance=CFG['yolo_max_distance'],
                depth_array=depth_array,
                visibility_check=CFG['visibility_check'],
                occlusion_margin=CFG['occlusion_margin'])
            for cls_name, _ in yolo_pairs:
                class_counts[cls_name] += 1
            with open(f'{root}/labels_yolo/{base}.txt', 'w') as f:
                f.write('\n'.join(line for _, line in yolo_pairs))

            # --- 3D KITTI-style GT (UPDATED: CFG-driven range/min_points) -
            # Written to BOTH labels_3d/ (unchanged from before) AND
            # label_2/ (the original script's permanently-empty stub dir --
            # now populated with the same real content instead of being
            # empty, so any downstream code already pointed at label_2/
            # gets actual data rather than nothing). If your other code
            # relies on label_2/ specifically being empty, tell me and I'll
            # go back to leaving it blank.
            lidar_3d_lines = extract_3d_bboxes_lidar(
                world, sensors['lidar'], data['lidar'], ego,
                max_distance=CFG['lidar_range'], min_points=CFG['min_points'])
            with open(f'{root}/labels_3d/{base}.txt', 'w') as f:
                f.write('\n'.join(lidar_3d_lines))
            with open(f'{root}/label_2/{base}.txt', 'w') as f:
                f.write('\n'.join(lidar_3d_lines))

            # --- NEW: Lane GT ---------------------------------------------
            lanes = extract_lane_gt(
                world, ego, sensors['rgb'], CFG['img_w'], CFG['img_h'], CFG['fov'],
                max_distance=CFG['lane_max_distance'],
                sample_distance=CFG['lane_sample_distance'])
            with open(f'{root}/lanes/{base}.json', 'w') as f:
                json.dump(lanes, f)

            # --- NEW: Metadata ----------------------------------------------
            sensor_frame_ids = {name: item.frame for name, item in data.items()}
            sync_ok = len(set(sensor_frame_ids.values())) == 1
            write_metadata(
                f'{root}/metadata/{base}.json', frame_id, town, CFG['run_id'], ego,
                n_yolo=len(yolo_pairs), n_3d=len(lidar_3d_lines), n_lanes=len(lanes),
                n_lidar_points=lidar_points.shape[0],
                sync_status='ok' if sync_ok else 'mismatch',
                sensor_frame_ids=sensor_frame_ids)

            saved += 1
            print(f'{town}: {saved - start_idx}/{CFG["frames_per_town"]}', end='\r')

        print(f'\n{town} done ({dropped_frames} frames dropped/skipped)')
        print(f'{town} GT instance counts (2D labels): '
              + ', '.join(f'{name}={count}' for name, count in
                           sorted(class_counts.items(), key=lambda kv: -kv[1]))
              or '  (none)')

        if CFG.get('validate_after_run'):
            validate_dataset(root, town)

        return saved

    finally:
        for s in sensors.values():                  # stop before destroy
            try: s.stop()
            except: pass
        for c in controllers:                        # NEW: stop walker AI before destroy
            try: c.stop()
            except: pass
        for a in actors:                             # destroy before restoring
            try: a.destroy()
            except: pass
        tm.set_synchronous_mode(False)
        world.apply_settings(original)
        time.sleep(3)
        print(f'{town} cleaned up')


if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)

    idx = 0
    for town in CFG['towns']:
        idx = collect_town(client, town, idx)
        json.dump({'next_idx': idx}, open('progress.json', 'w'))
    print(f'\ntotal frames: {idx}')
