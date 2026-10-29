"""
verify_dataset.py

Scans an images/ + labels/ pair of folders (YOLO format) and finds:
  - label files with no matching image file at all
  - image files that exist but are truncated/corrupted (can't be fully
    decoded -- this is what "libpng error: PNG input buffer is incomplete"
    means) -- caused by CARLA's async save_to_disk() not finishing before
    the sensor/actor was destroyed
  - image files with no matching label file (orphaned images -- usually
    harmless, but reported for completeness)

By default it only REPORTS problems. Pass --fix to actually delete the
broken pairs (image + label together, so the dataset stays consistent)
and to remove Ultralytics' stale .cache file so it re-scans clean data
on the next training run.

Usage:
    python verify_dataset.py --images_dir yolo_eval/Town03/images/val \
                              --labels_dir yolo_eval/Town03/labels/val
    python verify_dataset.py --images_dir ... --labels_dir ... --fix
"""

import os
import argparse
from PIL import Image


def is_image_valid(path):
    """True if the image can be fully decoded, False if truncated/corrupt."""
    try:
        with Image.open(path) as img:
            img.verify()          # checks structure without full decode
        with Image.open(path) as img:
            img.load()             # actually forces full pixel decode --
                                    # verify() alone misses some truncations
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--labels_dir', required=True)
    ap.add_argument('--fix', action='store_true',
                     help='Delete broken image/label pairs and stale cache '
                          'files. Without this flag, only reports issues.')
    args = ap.parse_args()

    image_files = {os.path.splitext(f)[0]: f
                   for f in os.listdir(args.images_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    label_files = {os.path.splitext(f)[0]: f
                   for f in os.listdir(args.labels_dir)
                   if f.lower().endswith('.txt')}

    all_ids = sorted(set(image_files) | set(label_files))

    missing_images = []     # label exists, image doesn't
    corrupted_images = []   # image exists, but can't be decoded
    orphaned_images = []    # image exists, label doesn't

    print(f'Checking {len(all_ids)} entries...')
    for i, base_id in enumerate(all_ids):
        if i % 500 == 0:
            print(f'  {i}/{len(all_ids)}', end='\r')

        has_label = base_id in label_files
        has_image = base_id in image_files

        if has_label and not has_image:
            missing_images.append(base_id)
            continue
        if has_image and not has_label:
            orphaned_images.append(base_id)
            continue
        if has_image and has_label:
            img_path = os.path.join(args.images_dir, image_files[base_id])
            if not is_image_valid(img_path):
                corrupted_images.append(base_id)

    print(f'\n\n=== Results ===')
    print(f'Missing images (label exists, image does not): {len(missing_images)}')
    print(f'Corrupted/truncated images:                    {len(corrupted_images)}')
    print(f'Orphaned images (image exists, no label):      {len(orphaned_images)}')

    broken = set(missing_images) | set(corrupted_images)
    if broken:
        print(f'\nSample of broken ids: {sorted(broken)[:15]}')

    if not args.fix:
        print('\nRun again with --fix to remove broken pairs and the stale '
              'Ultralytics cache file.')
        return

    print('\nFixing...')
    removed = 0
    for base_id in broken:
        img_path = os.path.join(args.images_dir, image_files.get(base_id, ''))
        lbl_path = os.path.join(args.labels_dir, label_files.get(base_id, ''))
        for p in (img_path, lbl_path):
            if p and os.path.exists(p):
                os.remove(p)
                removed += 1

    # Ultralytics caches label scans as labels_dir.cache (or val.cache) --
    # stale entries will still point at files we just deleted otherwise.
    for cache_name in ('val.cache', 'train.cache'):
        cache_path = os.path.join(args.labels_dir, cache_name)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f'  removed stale cache: {cache_path}')
        parent_cache = os.path.join(os.path.dirname(args.labels_dir), cache_name)
        if os.path.exists(parent_cache):
            os.remove(parent_cache)
            print(f'  removed stale cache: {parent_cache}')

    print(f'\nRemoved {removed} files across {len(broken)} broken entries.')
    print('Dataset should now be safe to train on.')


if __name__ == '__main__':
    main()