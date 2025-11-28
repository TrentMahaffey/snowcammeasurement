"""
OCR-based automatic calibration from snow stake markings.
Reads the numbered markings directly from the stake (2", 4", 6", etc.)
and auto-calculates pixel-to-inch ratio.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
import re
from dataclasses import dataclass


@dataclass
class StakeMarking:
    """Represents a detected marking on the snow stake."""
    value: int  # The number (e.g., 2, 4, 6)
    y_position: int  # Y coordinate in image
    confidence: float  # Detection confidence


class StakeOCRCalibrator:
    """Automatically calibrates from stake markings using OCR."""

    def __init__(self, debug: bool = False):
        """Initialize OCR calibrator.

        Args:
            debug: If True, save debug images
        """
        self.debug = debug
        self.reader = None

    def _get_reader(self):
        """Lazy load EasyOCR reader (it's slow to initialize)."""
        if self.reader is None:
            try:
                import easyocr
                self.reader = easyocr.Reader(['en'], gpu=False)
            except ImportError:
                raise ImportError(
                    "EasyOCR not installed. Run: pip install easyocr"
                )
        return self.reader

    def detect_reference_point(
        self,
        image: np.ndarray,
        stake_region: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[int]:
        """Detect the reference point (e.g., '24 hour total' text) marking zero.

        Args:
            image: OpenCV image array
            stake_region: Optional region (x, y, width, height) containing stake

        Returns:
            Y-coordinate of reference point or None
        """
        # Extract region of interest if specified
        roi = image
        roi_offset_y = 0
        if stake_region:
            x, y, w, h = stake_region
            roi = image[y:y+h, x:x+w]
            roi_offset_y = y

        # Preprocess for better OCR
        roi_preprocessed = self._preprocess_for_ocr(roi)

        # Run OCR
        reader = self._get_reader()
        results = reader.readtext(roi_preprocessed)

        # Look for reference text like "24 hour", "hour total", etc.
        reference_patterns = [
            r'24\s*hour',
            r'hour\s*total',
            r'24hr',
            r'total'
        ]

        for (bbox, text, confidence) in results:
            text_lower = text.lower()
            for pattern in reference_patterns:
                if re.search(pattern, text_lower):
                    # Found reference text - return its Y position
                    y_coords = [point[1] for point in bbox]
                    y_position = int(np.mean(y_coords)) + roi_offset_y
                    print(f"Found reference text '{text}' at y={y_position}")
                    return y_position

        return None

    def detect_stake_markings(
        self,
        image: np.ndarray,
        stake_region: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[List[StakeMarking], Optional[int]]:
        """Detect numbered markings on the snow stake and reference point.

        Args:
            image: OpenCV image array
            stake_region: Optional region (x, y, width, height) containing stake

        Returns:
            Tuple of (List of detected StakeMarking objects, reference_y)
        """
        # First, find the reference point (e.g., "24 hour total")
        reference_y = self.detect_reference_point(image, stake_region)

        # Extract region of interest if specified
        roi = image
        roi_offset_y = 0
        if stake_region:
            x, y, w, h = stake_region
            roi = image[y:y+h, x:x+w]
            roi_offset_y = y

        # Preprocess for better OCR
        roi_preprocessed = self._preprocess_for_ocr(roi)

        # Run OCR
        reader = self._get_reader()
        results = reader.readtext(roi_preprocessed)

        # Parse results to find stake markings
        markings = []
        for (bbox, text, confidence) in results:
            # Skip text that looks like reference text
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in ['hour', 'total', 'temp', 'wind', 'snow']):
                continue

            # Extract numbers from text
            numbers = re.findall(r'\b(\d{1,2})\b', text)  # Only 1-2 digit numbers
            if not numbers:
                continue

            # Parse the number
            try:
                value = int(numbers[0])
            except ValueError:
                continue

            # Only accept small incremental markings (2, 6, 10, 14, 18, 22, etc.)
            # Filter out timestamps (like 18:00), dates, large numbers
            if value < 2 or value > 30:  # Reasonable range for stake markings
                continue

            # Prefer increments of 4 (2, 6, 10, 14, 18, 22) or 2 (2, 4, 6, 8...)
            if value % 2 != 0:  # Must be even
                continue

            # Get Y position (center of bounding box)
            y_coords = [point[1] for point in bbox]
            y_position = int(np.mean(y_coords)) + roi_offset_y

            # If we have a reference point, only accept markings ABOVE it
            if reference_y is not None and y_position >= reference_y:
                continue

            markings.append(StakeMarking(
                value=value,
                y_position=y_position,
                confidence=confidence
            ))

        # Sort by position (top to bottom)
        markings.sort(key=lambda m: m.y_position)

        if self.debug:
            self._save_debug_image(image, markings, stake_region)

        return markings, reference_y

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened

    def calculate_calibration(
        self,
        markings: List[StakeMarking],
        reference_y: Optional[int] = None
    ) -> Optional[Dict]:
        """Calculate pixels per inch from detected markings.

        Args:
            markings: List of detected StakeMarking objects
            reference_y: Y-coordinate of reference point (zero/base)

        Returns:
            Calibration dictionary or None if insufficient data
        """
        if len(markings) < 2:
            return None

        # For this stake, markings are spaced 4 inches apart (2, 6, 10, 14, 18, 22)
        # Find pairs with 4-inch spacing first (most reliable)
        measurements = []

        for i in range(len(markings) - 1):
            current = markings[i]
            next_marking = markings[i + 1]

            # Check for 4-inch spacing (2-6, 6-10, 10-14, 14-18, etc.)
            if next_marking.value == current.value + 4:
                pixel_distance = abs(next_marking.y_position - current.y_position)
                inches_distance = 4

                pixels_per_inch = pixel_distance / inches_distance
                avg_confidence = (current.confidence + next_marking.confidence) / 2

                measurements.append({
                    'pixels_per_inch': pixels_per_inch,
                    'from_marking': current.value,
                    'to_marking': next_marking.value,
                    'confidence': avg_confidence
                })

        if not measurements:
            # Fallback: use any consecutive markings
            for i in range(len(markings) - 1):
                current = markings[i]
                next_marking = markings[i + 1]

                pixel_distance = abs(next_marking.y_position - current.y_position)
                inches_distance = abs(next_marking.value - current.value)

                if inches_distance > 0:
                    pixels_per_inch = pixel_distance / inches_distance
                    avg_confidence = (current.confidence + next_marking.confidence) / 2

                    measurements.append({
                        'pixels_per_inch': pixels_per_inch,
                        'from_marking': current.value,
                        'to_marking': next_marking.value,
                        'confidence': avg_confidence
                    })

        if not measurements:
            return None

        # Use weighted average of all measurements
        total_weight = sum(m['confidence'] for m in measurements)
        weighted_ppi = sum(
            m['pixels_per_inch'] * m['confidence']
            for m in measurements
        ) / total_weight

        # Find stake bounds
        stake_top_y = min(m.y_position for m in markings)
        stake_bottom_y = max(m.y_position for m in markings)

        # Find the lowest (smallest number) marking - this is the first one above reference
        lowest_marking = min(markings, key=lambda m: m.value)

        return {
            'pixels_per_inch': weighted_ppi,
            'stake_top_y': stake_top_y,
            'stake_base_y': stake_bottom_y,
            'reference_y': reference_y,  # Y-coordinate of "24 hour total" (zero point)
            'markings_detected': len(markings),
            'measurements_used': len(measurements),
            'lowest_visible_marking': lowest_marking.value,  # Smallest number (e.g., 2")
            'lowest_marking_y': lowest_marking.y_position,
            'confidence': sum(m['confidence'] for m in measurements) / len(measurements),
            'all_markings': [
                {'value': m.value, 'y': m.y_position, 'conf': m.confidence}
                for m in markings
            ]
        }

    def auto_calibrate(
        self,
        image_path: str,
        stake_region: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Dict]:
        """Automatically calibrate from a single image.

        Args:
            image_path: Path to image with visible stake markings
            stake_region: Optional region containing stake

        Returns:
            Calibration dictionary or None if failed
        """
        import os
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None

        # Detect markings and reference point
        markings, reference_y = self.detect_stake_markings(image, stake_region)

        if not markings:
            print("No stake markings detected")
            return None

        if reference_y:
            print(f"\nReference point (zero) at y={reference_y}")
        else:
            print("\nWarning: No reference point found, using fallback calibration")

        print(f"\nDetected {len(markings)} stake markings:")
        for m in markings:
            print(f"  {m.value}\" at y={m.y_position} (confidence: {m.confidence:.2f})")

        # Calculate calibration
        calibration = self.calculate_calibration(markings, reference_y)

        if calibration:
            print(f"\n✓ Auto-calibration successful!")
            print(f"  Pixels per inch: {calibration['pixels_per_inch']:.2f}")
            print(f"  Markings detected: {calibration['markings_detected']}")
            print(f"  Lowest visible: {calibration['lowest_visible_marking']}\"")
            print(f"  Confidence: {calibration['confidence']:.2f}")
            calibration['reference_image_path'] = image_path
        else:
            print("Failed to calculate calibration from detected markings")

        return calibration

    def _save_debug_image(
        self,
        image: np.ndarray,
        markings: List[StakeMarking],
        stake_region: Optional[Tuple[int, int, int, int]]
    ):
        """Save annotated debug image showing detected markings."""
        from datetime import datetime

        debug_image = image.copy()

        # Draw stake region if specified
        if stake_region:
            x, y, w, h = stake_region
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Draw each detected marking
        for marking in markings:
            y = marking.y_position
            cv2.line(debug_image, (0, y), (image.shape[1], y), (0, 255, 0), 2)
            cv2.putText(
                debug_image,
                f"{marking.value}\" ({marking.confidence:.2f})",
                (10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = f"debug_ocr_calibration_{timestamp}.jpg"
        cv2.imwrite(debug_path, debug_image)
        print(f"Debug image saved: {debug_path}")


def quick_calibrate(
    image_path: str,
    stake_region: Optional[Tuple[int, int, int, int]] = None
) -> Optional[Dict]:
    """Quick function to auto-calibrate from an image.

    Args:
        image_path: Path to image with visible stake
        stake_region: Optional (x, y, width, height) containing stake

    Returns:
        Calibration dictionary

    Example:
        >>> calibration = quick_calibrate('snowmass_20251121_120000.png')
        >>> print(f"Pixels per inch: {calibration['pixels_per_inch']}")
    """
    calibrator = StakeOCRCalibrator(debug=True)
    return calibrator.auto_calibrate(image_path, stake_region)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_calibrate.py <image_path> [x y width height]")
        print("\nExample:")
        print("  python ocr_calibrate.py snowmass.png")
        print("  python ocr_calibrate.py snowmass.png 800 400 200 600")
        sys.exit(1)

    image_path = sys.argv[1]
    stake_region = None

    if len(sys.argv) == 6:
        stake_region = tuple(map(int, sys.argv[2:6]))
        print(f"Using stake region: {stake_region}")

    calibration = quick_calibrate(image_path, stake_region)

    if calibration:
        print("\n" + "="*50)
        print("CALIBRATION DATA:")
        print("="*50)
        import json
        print(json.dumps(calibration, indent=2))
    else:
        print("\nCalibration failed!")
        sys.exit(1)
