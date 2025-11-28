#!/usr/bin/env python3
"""
Manual calibration tool for snow stake measurements.
Use when OCR auto-calibration doesn't work reliably.
"""

import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

from .config import CalibrationManager


def get_pixel_coordinates(image_path: str, resort: str):
    """Interactive tool to get pixel coordinates from an image.

    Click on the image to get coordinates:
    - Click on "24 hour total" text (reference point = 0 inches)
    - Click on a visible marking (e.g., "6" marking)
    - Enter the inch value of that marking
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Scale down for display if needed
    display_scale = 1.0
    h, w = image.shape[:2]
    max_display = 1200
    if max(h, w) > max_display:
        display_scale = max_display / max(h, w)

    display_image = cv2.resize(image, None, fx=display_scale, fy=display_scale) if display_scale < 1 else image.copy()

    clicks = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coords back to original image coords
            orig_x = int(x / display_scale)
            orig_y = int(y / display_scale)
            clicks.append((orig_x, orig_y))

            # Draw marker
            cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display_image, f"({orig_x}, {orig_y})",
                       (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow(f"Calibrate {resort}", display_image)
            print(f"Clicked: ({orig_x}, {orig_y})")

    cv2.imshow(f"Calibrate {resort}", display_image)
    cv2.setMouseCallback(f"Calibrate {resort}", mouse_callback)

    print("\n" + "="*60)
    print(f"MANUAL CALIBRATION FOR {resort.upper()}")
    print("="*60)
    print("\nStep 1: Click on the '24 hour total' text (reference point)")
    print("Step 2: Click on a visible inch marking (e.g., the '6' or '10')")
    print("Press ESC when done, or 'q' to quit without saving")
    print("="*60 + "\n")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    if len(clicks) < 2:
        print("Need at least 2 clicks (reference + one marking)")
        return None

    return clicks


def calculate_calibration(reference_y: int, marking_y: int, marking_value: int):
    """Calculate pixels per inch from reference and one marking.

    Args:
        reference_y: Y coordinate of reference point (0 inches)
        marking_y: Y coordinate of a known marking
        marking_value: Inch value of that marking

    Returns:
        pixels_per_inch value
    """
    # Markings are ABOVE reference point (lower Y values = higher on image)
    pixel_distance = reference_y - marking_y
    if pixel_distance <= 0:
        print("Error: Marking should be above reference point (lower Y value)")
        return None

    pixels_per_inch = pixel_distance / marking_value
    return pixels_per_inch


def manual_calibrate_cli(args):
    """Command-line manual calibration."""
    print("\n" + "="*60)
    print(f"MANUAL CALIBRATION FOR {args.resort.upper()}")
    print("="*60)

    if args.reference_y is None or args.marking_y is None or args.marking_value is None:
        # Interactive mode with image
        if not args.image:
            print("\nFor manual calibration, either:")
            print("  1. Provide --image to click on coordinates interactively")
            print("  2. Provide --reference-y, --marking-y, and --marking-value directly")
            return 1

        clicks = get_pixel_coordinates(args.image, args.resort)
        if clicks is None or len(clicks) < 2:
            return 1

        reference_y = clicks[0][1]
        marking_y = clicks[1][1]

        print(f"\nReference point (0\"): y={reference_y}")
        print(f"Marking point: y={marking_y}")

        marking_value = int(input("Enter the inch value of the marking you clicked (e.g., 6, 10, 14): "))
    else:
        reference_y = args.reference_y
        marking_y = args.marking_y
        marking_value = args.marking_value

    # Calculate calibration
    pixels_per_inch = calculate_calibration(reference_y, marking_y, marking_value)

    if pixels_per_inch is None:
        return 1

    print(f"\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"Reference point Y: {reference_y}")
    print(f"Marking Y ({marking_value}\"): {marking_y}")
    print(f"Pixels per inch: {pixels_per_inch:.2f}")
    print(f"Pixel distance: {reference_y - marking_y}")
    print("="*60)

    # Verify with other expected markings
    print("\nExpected marking positions (based on 4\" increments):")
    for inch_val in [2, 6, 10, 14, 18]:
        expected_y = reference_y - (inch_val * pixels_per_inch)
        print(f"  {inch_val}\": y={int(expected_y)}")

    if not args.no_save:
        # Save calibration
        config_manager = CalibrationManager(
            config_path=args.config,
            db_path=args.db
        )

        # Parse stake region if provided
        stake_region = None
        if args.stake_region:
            parts = [int(x) for x in args.stake_region.split(',')]
            if len(parts) == 4:
                stake_region = tuple(parts)
                print(f"Stake region: x={parts[0]}, y={parts[1]}, w={parts[2]}, h={parts[3]}")

        config_manager.add_calibration(
            resort=args.resort,
            pixels_per_inch=pixels_per_inch,
            stake_region=stake_region,
            reference_y=reference_y,
            reference_image_path=args.image or "manual",
            notes=f"Manual calibration. Reference y={reference_y}, {marking_value}\" at y={marking_y}"
        )

        print(f"\n✓ Calibration saved for {args.resort}")
        print(f"  Config: {args.config or 'default'}")
        print(f"  Database: {args.db or 'default'}")
    else:
        print("\n(--no-save specified, calibration not saved)")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Manual calibration for snow stake measurements"
    )
    parser.add_argument(
        'resort',
        help="Resort name (e.g., snowmass)"
    )
    parser.add_argument(
        '--image',
        help="Image file for interactive calibration (optional)"
    )
    parser.add_argument(
        '--reference-y',
        type=int,
        help="Y coordinate of reference point (24 hour total text)"
    )
    parser.add_argument(
        '--marking-y',
        type=int,
        help="Y coordinate of a known marking"
    )
    parser.add_argument(
        '--marking-value',
        type=int,
        help="Inch value of the marking at marking-y"
    )
    parser.add_argument(
        '--stake-region',
        type=str,
        help="Stake region as 'x,y,width,height' (e.g., '400,450,150,600')"
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help="Don't save calibration (just display results)"
    )
    parser.add_argument(
        '--db',
        default=None,
        help="Path to database file"
    )
    parser.add_argument(
        '--config',
        default=None,
        help="Path to calibration config file"
    )

    args = parser.parse_args()

    return manual_calibrate_cli(args)


if __name__ == "__main__":
    sys.exit(main())
