#!/usr/bin/env python3
"""
Tilt-aware snow stake calibration.

This calibration system:
1. Detects camera tilt by analyzing text orientation
2. Finds "24 HOUR TOTAL" as the 0" reference (top of text)
3. Uses level indicators at markings to find stake centerline
4. Adjusts all measurements for camera tilt
"""

import cv2
import numpy as np
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Result of tilt-aware calibration."""
    tilt_angle: float  # Camera tilt in degrees (positive = clockwise)
    reference_point: Tuple[int, int]  # (x, y) of 0" reference (top of "24 HOUR TOTAL")
    stake_centerline_x: int  # X coordinate of stake centerline
    pixels_per_inch: float  # Pixels per inch along the (tilt-corrected) stake
    markings: Dict[int, Tuple[int, int]]  # {inch_value: (x, y)} for detected markings
    confidence: float


class TiltAwareCalibrator:
    """Calibrator that accounts for camera tilt."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.reader = None

    def _get_reader(self):
        """Lazy load EasyOCR reader."""
        if self.reader is None:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False)
        return self.reader

    def detect_tilt_from_text(self, image: np.ndarray) -> Tuple[float, List[dict]]:
        """Detect camera tilt by analyzing text orientation.

        Text should be horizontal. If it's tilted, that's our camera tilt.

        Returns:
            Tuple of (tilt_angle_degrees, list of text detections)
        """
        reader = self._get_reader()
        results = reader.readtext(image)

        angles = []
        detections = []

        for (bbox, text, confidence) in results:
            if confidence < 0.5:
                continue

            # bbox is 4 corners: top-left, top-right, bottom-right, bottom-left
            top_left = bbox[0]
            top_right = bbox[1]

            # Calculate angle of text baseline
            dx = top_right[0] - top_left[0]
            dy = top_right[1] - top_left[1]

            if dx != 0:
                angle = math.degrees(math.atan(dy / dx))
                angles.append(angle)

            detections.append({
                'text': text,
                'bbox': bbox,
                'confidence': confidence,
                'angle': angle if dx != 0 else 0,
                'center': ((bbox[0][0] + bbox[2][0]) / 2, (bbox[0][1] + bbox[2][1]) / 2)
            })

        # Use median angle to avoid outliers
        if angles:
            tilt_angle = np.median(angles)
        else:
            tilt_angle = 0.0

        return tilt_angle, detections

    def find_reference_point(self, detections: List[dict]) -> Optional[Tuple[int, int]]:
        """Find the 0" reference point (top of "24 HOUR TOTAL" text).

        Returns:
            (x, y) of reference point or None
        """
        for det in detections:
            text_lower = det['text'].lower()
            # Look for "24 hour" or "hour total" or similar
            if '24' in text_lower and 'hour' in text_lower:
                # Top of text = top-left and top-right corners
                bbox = det['bbox']
                top_y = min(bbox[0][1], bbox[1][1])
                center_x = (bbox[0][0] + bbox[1][0]) / 2
                return (int(center_x), int(top_y))

            if 'hour' in text_lower and 'total' in text_lower:
                bbox = det['bbox']
                top_y = min(bbox[0][1], bbox[1][1])
                center_x = (bbox[0][0] + bbox[1][0]) / 2
                return (int(center_x), int(top_y))

        return None

    def find_stake_markings(self, detections: List[dict], reference_y: int) -> Dict[int, Tuple[int, int]]:
        """Find inch markings on the stake.

        Args:
            detections: OCR detections
            reference_y: Y coordinate of 0" reference

        Returns:
            Dict mapping inch values to (x, y) positions
        """
        markings = {}

        # Valid marking values (2, 6, 10, 14, 18 with 4" spacing)
        valid_values = {2, 6, 10, 14, 18, 22}

        for det in detections:
            text = det['text'].strip()

            # Try to parse as number
            try:
                value = int(text)
            except ValueError:
                continue

            # Check if it's a valid marking value
            if value not in valid_values:
                continue

            # Marking should be ABOVE reference (lower Y value)
            center_y = det['center'][1]
            if center_y >= reference_y:
                continue

            # Store the marking position
            # Use the RIGHT edge of the text as that's where the level indicator starts
            bbox = det['bbox']
            right_x = max(bbox[1][0], bbox[2][0])
            center_y = (bbox[0][1] + bbox[2][1]) / 2

            markings[value] = (int(right_x), int(center_y))

        return markings

    def find_stake_centerline(
        self,
        image: np.ndarray,
        markings: Dict[int, Tuple[int, int]],
        tilt_angle: float
    ) -> Optional[int]:
        """Find the stake centerline X coordinate.

        Uses the level indicators at each marking that point to stake center.

        Args:
            image: Original image
            markings: Detected marking positions
            tilt_angle: Camera tilt angle

        Returns:
            X coordinate of stake centerline
        """
        if not markings:
            return None

        # For each marking, trace the horizontal line (adjusted for tilt) to find stake
        centerline_candidates = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for inch_val, (mark_x, mark_y) in markings.items():
            # Scan horizontally (with tilt adjustment) from marking toward stake
            # The level indicator should be a horizontal line

            # Calculate Y adjustment per pixel due to tilt
            y_per_x = math.tan(math.radians(tilt_angle))

            # Scan to the right of the marking number
            scan_width = 200  # pixels to scan
            line_brightness = []

            for dx in range(10, scan_width):
                x = mark_x + dx
                y = int(mark_y + dx * y_per_x)

                if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                    # Sample a small vertical window
                    y_start = max(0, y - 2)
                    y_end = min(gray.shape[0], y + 3)
                    brightness = np.mean(gray[y_start:y_end, x])
                    line_brightness.append((x, brightness))

            if line_brightness:
                # Look for where the level indicator ends (brightness change)
                # The stake is usually darker than the indicator line
                brightnesses = [b for _, b in line_brightness]
                mean_brightness = np.mean(brightnesses)

                # Find where brightness drops significantly
                for i, (x, b) in enumerate(line_brightness):
                    if b < mean_brightness * 0.7:  # Significant drop
                        centerline_candidates.append(x)
                        break

        if centerline_candidates:
            # Use median of candidates
            return int(np.median(centerline_candidates))

        return None

    def calculate_pixels_per_inch(
        self,
        markings: Dict[int, Tuple[int, int]],
        tilt_angle: float
    ) -> Optional[float]:
        """Calculate pixels per inch from marking positions.

        Args:
            markings: Detected marking positions
            tilt_angle: Camera tilt angle

        Returns:
            Pixels per inch value
        """
        if len(markings) < 2:
            return None

        # Sort markings by inch value
        sorted_marks = sorted(markings.items())

        measurements = []
        for i in range(len(sorted_marks) - 1):
            val1, (x1, y1) = sorted_marks[i]
            val2, (x2, y2) = sorted_marks[i + 1]

            inch_diff = val2 - val1

            # Calculate pixel distance (accounting for tilt)
            # True vertical distance = measured_y_diff / cos(tilt)
            y_diff = y1 - y2  # y1 > y2 since higher inch = lower y

            # Correct for tilt
            corrected_distance = y_diff / math.cos(math.radians(tilt_angle))

            ppi = corrected_distance / inch_diff
            measurements.append(ppi)

        if measurements:
            return np.median(measurements)

        return None

    def calibrate(self, image_path: str) -> Optional[CalibrationResult]:
        """Perform full tilt-aware calibration.

        Args:
            image_path: Path to calibration image

        Returns:
            CalibrationResult or None if calibration failed
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None

        print(f"Image size: {image.shape[1]}x{image.shape[0]}")

        # Step 1: Detect tilt from text
        print("\n1. Detecting camera tilt...")
        tilt_angle, detections = self.detect_tilt_from_text(image)
        print(f"   Detected tilt: {tilt_angle:.2f} degrees")
        print(f"   Found {len(detections)} text elements")

        # Step 2: Find reference point (top of "24 HOUR TOTAL")
        print("\n2. Finding reference point...")
        reference_point = self.find_reference_point(detections)
        if reference_point is None:
            print("   ERROR: Could not find '24 HOUR TOTAL' reference text")
            return None
        print(f"   Reference (0\") at: {reference_point}")

        # Step 3: Find stake markings
        print("\n3. Finding stake markings...")
        markings = self.find_stake_markings(detections, reference_point[1])
        print(f"   Found {len(markings)} markings: {list(markings.keys())}")
        for val, pos in sorted(markings.items()):
            print(f"      {val}\": {pos}")

        # Step 4: Find stake centerline
        print("\n4. Finding stake centerline...")
        centerline_x = self.find_stake_centerline(image, markings, tilt_angle)
        if centerline_x is None:
            # Fallback: estimate from reference point
            centerline_x = reference_point[0] + 100  # Approximate
            print(f"   Using estimated centerline: x={centerline_x}")
        else:
            print(f"   Stake centerline at: x={centerline_x}")

        # Step 5: Calculate pixels per inch
        print("\n5. Calculating pixels per inch...")
        ppi = self.calculate_pixels_per_inch(markings, tilt_angle)
        if ppi is None:
            print("   ERROR: Could not calculate pixels per inch")
            return None
        print(f"   Pixels per inch: {ppi:.2f}")

        # Calculate confidence based on number of markings found
        confidence = min(1.0, len(markings) / 4)

        result = CalibrationResult(
            tilt_angle=tilt_angle,
            reference_point=reference_point,
            stake_centerline_x=centerline_x,
            pixels_per_inch=ppi,
            markings=markings,
            confidence=confidence
        )

        # Save debug image if enabled
        if self.debug:
            self._save_debug_image(image, result, detections)

        return result

    def _save_debug_image(
        self,
        image: np.ndarray,
        result: CalibrationResult,
        detections: List[dict]
    ):
        """Save annotated debug image."""
        debug_img = image.copy()

        # Draw all text detections
        for det in detections:
            bbox = np.array(det['bbox'], dtype=np.int32)
            cv2.polylines(debug_img, [bbox], True, (100, 100, 100), 1)

        # Draw reference point
        ref_x, ref_y = result.reference_point
        cv2.circle(debug_img, (ref_x, ref_y), 10, (0, 255, 0), -1)
        cv2.putText(debug_img, "0\" REF", (ref_x + 15, ref_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw stake centerline
        cv2.line(debug_img,
                (result.stake_centerline_x, 0),
                (result.stake_centerline_x, debug_img.shape[0]),
                (255, 0, 0), 2)

        # Draw markings
        for inch_val, (x, y) in result.markings.items():
            cv2.circle(debug_img, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(debug_img, f'{inch_val}"', (x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw line to centerline
            cv2.line(debug_img, (x, y), (result.stake_centerline_x, y), (0, 255, 255), 1)

        # Draw tilt indicator
        center_x = debug_img.shape[1] // 2
        center_y = 50
        length = 100
        end_x = int(center_x + length * math.cos(math.radians(result.tilt_angle)))
        end_y = int(center_y + length * math.sin(math.radians(result.tilt_angle)))
        cv2.line(debug_img, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3)
        cv2.putText(debug_img, f"Tilt: {result.tilt_angle:.1f}deg",
                   (center_x - 50, center_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add calibration info
        info_y = debug_img.shape[0] - 100
        cv2.putText(debug_img, f"PPI: {result.pixels_per_inch:.2f}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Confidence: {result.confidence:.0%}",
                   (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save
        output_path = "debug_tilt_calibration.jpg"
        cv2.imwrite(output_path, debug_img)
        print(f"\nDebug image saved to: {output_path}")


def main():
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python3 tilt_calibrate.py <resort> <image_path> [--save]")
        sys.exit(1)

    resort = sys.argv[1]
    image_path = sys.argv[2]
    save = '--save' in sys.argv

    calibrator = TiltAwareCalibrator(debug=True)
    result = calibrator.calibrate(image_path)

    if result is None:
        print("\nCalibration FAILED")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("CALIBRATION RESULT")
    print("=" * 60)
    print(f"Resort:           {resort}")
    print(f"Tilt Angle:       {result.tilt_angle:.2f} degrees")
    print(f"Reference Point:  {result.reference_point}")
    print(f"Stake Centerline: x={result.stake_centerline_x}")
    print(f"Pixels per Inch:  {result.pixels_per_inch:.2f}")
    print(f"Markings Found:   {len(result.markings)}")
    print(f"Confidence:       {result.confidence:.0%}")
    print("=" * 60)

    if save:
        # Save to config
        config = {
            'resorts': [{
                'resort': resort,
                'pixels_per_inch': result.pixels_per_inch,
                'tilt_angle': result.tilt_angle,
                'reference_x': result.reference_point[0],
                'reference_y': result.reference_point[1],
                'stake_centerline_x': result.stake_centerline_x,
                'stake_region_x': result.stake_centerline_x - 50,
                'stake_region_y': min(p[1] for p in result.markings.values()) - 50 if result.markings else result.reference_point[1] - 500,
                'stake_region_width': 100,
                'stake_region_height': result.reference_point[1] - (min(p[1] for p in result.markings.values()) - 50) if result.markings else 500,
                'enabled': True,
                'notes': f"Tilt-aware calibration. Tilt={result.tilt_angle:.1f}deg"
            }]
        }

        config_path = 'resort_calibrations.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nCalibration saved to: {config_path}")


if __name__ == "__main__":
    main()
