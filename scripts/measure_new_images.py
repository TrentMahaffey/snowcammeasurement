#!/usr/bin/env python3
"""
Automated snow depth measurement script.

Finds new hourly images that haven't been measured yet and processes them.
Designed to run via cron job.

Usage:
    python measure_new_images.py [--resort RESORT] [--dry-run] [--verbose]

Examples:
    # Measure all resorts
    python measure_new_images.py

    # Measure only winter_park
    python measure_new_images.py --resort winter_park

    # See what would be measured without actually doing it
    python measure_new_images.py --dry-run --verbose
"""

import argparse
import glob
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
import numpy as np


def _convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration - paths work inside Docker container
DB_PATH = os.environ.get('SNOW_DB_PATH', '/out/snow_measurements.db')
IMAGES_DIR = os.environ.get('SNOW_IMAGES_DIR', '/out')

# Resort configurations - loaded dynamically from resorts/ folder
def load_resort_configs():
    """Load resort configurations from resorts/ calibration files."""
    import glob
    resorts = {}

    resorts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resorts')

    for cal_file in glob.glob(os.path.join(resorts_dir, '*', 'calibration.json')):
        try:
            with open(cal_file) as f:
                config = json.load(f)

            resort_name = config.get('resort', os.path.basename(os.path.dirname(cal_file)))
            image_prefix = config.get('image_prefix', resort_name)

            resorts[resort_name] = {
                'enabled': config.get('enabled', False),
                'image_pattern': f'{image_prefix}_*.jpg',
                'calibration_path': cal_file,
                'has_custom_measurer': os.path.exists(
                    os.path.join(os.path.dirname(cal_file), 'measurer.py')
                ),
            }
        except Exception as e:
            print(f"Error loading {cal_file}: {e}")

    return resorts

RESORTS = load_resort_configs()


def get_existing_images(cursor, resort):
    """Get set of already-measured image paths for a resort."""
    cursor.execute(
        'SELECT image_path FROM snow_measurements WHERE resort = ?',
        (resort,)
    )
    return set(row[0] for row in cursor.fetchall())


def find_hourly_images(resort_config, images_dir):
    """Find hourly images (one per hour) for a resort."""
    pattern = os.path.join(images_dir, resort_config['image_pattern'])
    all_images = glob.glob(pattern)

    # Filter to hourly images (captured within first 2 minutes of hour)
    # Pattern: resort_YYYYMMDD_HH00SS.jpg or resort_YYYYMMDD_HH01SS.jpg
    hourly_images = []
    for img in all_images:
        basename = os.path.basename(img)
        # Match pattern like winter_park_20251128_070036.jpg
        match = re.search(r'_(\d{8})_(\d{2})(0[01])(\d{2})\.jpg$', basename)
        if match:
            hourly_images.append(img)

    return sorted(hourly_images)


def extract_timestamp(image_path):
    """Extract timestamp from image filename."""
    basename = os.path.basename(image_path)
    match = re.search(r'_(\d{8})_(\d{6})\.jpg$', basename)
    if match:
        date_str, time_str = match.groups()
        return f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}'
    return None


def get_measurer(resort, resort_config):
    """Get the appropriate measurer for a resort."""
    # Load calibration
    with open(resort_config['calibration_path']) as f:
        calibration = json.load(f)

    # Check for custom measurer
    if resort_config.get('has_custom_measurer'):
        measurer_dir = os.path.dirname(resort_config['calibration_path'])
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            f"{resort}_measurer",
            os.path.join(measurer_dir, 'measurer.py')
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'get_measurer'):
            return module.get_measurer(db_path=DB_PATH, calibration=calibration)
        elif hasattr(module, 'WinterParkMeasurer'):
            return module.WinterParkMeasurer(calibration)

    # Check if this resort uses marker interpolation (has marker_positions)
    method = calibration.get('method', 'linear')
    marker_positions = calibration.get('marker_positions', {})

    if method == 'marker_interpolation' or marker_positions:
        # Use WinterParkMeasurer for any resort with marker_positions
        resorts_dir = os.path.dirname(resort_config['calibration_path'])
        winter_park_dir = os.path.join(os.path.dirname(resorts_dir), 'winter_park')
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "winter_park_measurer",
            os.path.join(winter_park_dir, 'measurer.py')
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.WinterParkMeasurer(calibration)

    # Fall back to generic measurer (requires pixels_per_inch)
    from snowcammeasurement.measurement import SnowStakeMeasurer
    return SnowStakeMeasurer(
        pixels_per_inch=calibration.get('pixels_per_inch'),
        reference_y=calibration.get('reference_y'),
        debug=False
    )


def measure_image(measurer, image_path, resort):
    """Measure snow depth from an image."""
    import cv2

    # Check if measurer has measure_from_file (custom measurer)
    if hasattr(measurer, 'measure_from_file'):
        result = measurer.measure_from_file(image_path)
        return {
            'snow_depth_inches': result.snow_depth_inches,
            'confidence_score': result.confidence_score,
            'sample_data': json.dumps(_convert_numpy_types(result.samples)) if hasattr(result, 'samples') and result.samples else None,
            'notes': getattr(result, 'notes', ''),
        }
    else:
        # Generic measurer interface
        image = cv2.imread(image_path)
        if image is None:
            return None
        result = measurer.measure(image)
        if isinstance(result, dict):
            return {
                'snow_depth_inches': result.get('snow_depth_inches'),
                'confidence_score': result.get('confidence_score', 0.5),
                'sample_data': None,
                'notes': result.get('notes', ''),
            }
        else:
            return {
                'snow_depth_inches': getattr(result, 'snow_depth_inches', None),
                'confidence_score': getattr(result, 'confidence_score', 0.5),
                'sample_data': None,
                'notes': getattr(result, 'notes', ''),
            }


def process_resort(resort, resort_config, cursor, dry_run=False, verbose=False):
    """Process new images for a single resort."""
    if not resort_config.get('enabled', False):
        if verbose:
            print(f"  {resort}: disabled, skipping")
        return 0

    # Get existing measurements
    existing = get_existing_images(cursor, resort)
    if verbose:
        print(f"  {resort}: {len(existing)} existing measurements")

    # Find hourly images
    hourly_images = find_hourly_images(resort_config, IMAGES_DIR)
    if verbose:
        print(f"  {resort}: {len(hourly_images)} hourly images found")

    # Filter to new images
    new_images = [img for img in hourly_images if os.path.basename(img) not in existing]
    if verbose:
        print(f"  {resort}: {len(new_images)} new images to measure")

    if not new_images:
        return 0

    if dry_run:
        for img in new_images[:5]:
            print(f"    Would measure: {os.path.basename(img)}")
        if len(new_images) > 5:
            print(f"    ... and {len(new_images) - 5} more")
        return len(new_images)

    # Get measurer
    try:
        measurer = get_measurer(resort, resort_config)
    except Exception as e:
        print(f"  {resort}: ERROR loading measurer: {e}")
        return 0

    # Process new images
    added = 0
    for img_path in new_images:
        basename = os.path.basename(img_path)
        timestamp = extract_timestamp(img_path)
        if not timestamp:
            continue

        try:
            result = measure_image(measurer, img_path, resort)
            if result is None:
                continue

            cursor.execute('''
                INSERT OR IGNORE INTO snow_measurements
                (resort, timestamp, snow_depth_inches, confidence_score, image_path, sample_data, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                resort,
                timestamp,
                result['snow_depth_inches'],
                result['confidence_score'],
                basename,
                result['sample_data'],
                result['notes'],
            ))
            added += 1

            if verbose:
                depth_str = f"{result['snow_depth_inches']:.1f}\"" if result['snow_depth_inches'] else "N/A"
                print(f"    Measured {basename}: {depth_str}")

        except Exception as e:
            print(f"    ERROR measuring {basename}: {e}")

    return added


def main():
    parser = argparse.ArgumentParser(description='Measure new snow cam images')
    parser.add_argument('--resort', help='Only process this resort')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if args.verbose:
        print(f"Snow measurement run at {timestamp}")
        print(f"Database: {DB_PATH}")
        print(f"Images: {IMAGES_DIR}")

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Process resorts
    total_added = 0
    resorts_to_process = [args.resort] if args.resort else RESORTS.keys()

    for resort in resorts_to_process:
        if resort not in RESORTS:
            print(f"Unknown resort: {resort}")
            continue

        added = process_resort(
            resort,
            RESORTS[resort],
            cursor,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        total_added += added

        if not args.dry_run and added > 0:
            conn.commit()

    conn.close()

    if args.verbose or total_added > 0:
        action = "Would add" if args.dry_run else "Added"
        print(f"{action} {total_added} new measurements")

    return 0


if __name__ == '__main__':
    sys.exit(main())
