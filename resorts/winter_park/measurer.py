"""
Winter Park snow depth measurer using marker interpolation.

This camera has significant lens distortion, so linear pixels_per_inch
doesn't work. Instead, we use known marker positions (the numbers on the
stake: 2, 4, 6, 8, etc.) and interpolate between them.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import json
import os


@dataclass
class WinterParkMeasurement:
    """Result of a Winter Park snow depth measurement."""
    snow_depth_inches: Optional[float]
    confidence_score: float
    stake_visible: bool
    snow_line_y: Optional[int]
    samples: Optional[List[Dict]] = None
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None
    depth_avg: Optional[float] = None
    notes: str = ""


class WinterParkMeasurer:
    """
    Measures snow depth using marker interpolation for lens-distorted images.

    Instead of assuming linear pixels_per_inch, this measurer:
    1. Uses known Y positions of inch markers (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)
    2. Detects the snow line (where snow meets the stake)
    3. Interpolates between the nearest markers to calculate depth
    """

    def __init__(self, calibration: Dict[str, Any], debug: bool = False):
        self.calibration = calibration
        self.debug = debug

        # Parse marker positions (keys are strings in JSON)
        self.marker_positions = {}
        for inch_str, y_pos in calibration.get('marker_positions', {}).items():
            self.marker_positions[int(inch_str)] = y_pos

        if not self.marker_positions:
            raise ValueError("Winter Park measurer requires marker_positions in calibration")

        # Sort markers by inch value
        self.sorted_markers = sorted(self.marker_positions.items(), key=lambda x: x[0])

        # Get stake region
        self.stake_region = (
            calibration.get('stake_region_x', 780),
            calibration.get('stake_region_y', 280),
            calibration.get('stake_region_width', 80),
            calibration.get('stake_region_height', 510)
        )
        self.centerline_x = calibration.get('stake_centerline_x', 820)
        self.min_depth_threshold = calibration.get('min_depth_threshold', 1.0)

    def y_to_inches(self, y: int) -> float:
        """
        Convert a Y pixel position to inches using marker interpolation.

        Args:
            y: Y pixel position in the image

        Returns:
            Depth in inches (interpolated between nearest markers)
        """
        # Find the two markers that bracket this Y position
        # Note: Lower Y = higher on image = more inches of snow

        # If Y is above all markers (more snow than 18"), extrapolate from top two
        if y <= self.sorted_markers[-1][1]:  # Above 18" marker
            inch1, y1 = self.sorted_markers[-2]  # 16"
            inch2, y2 = self.sorted_markers[-1]  # 18"
            # Extrapolate
            pixels_per_inch_local = (y1 - y2) / (inch2 - inch1)
            extra_pixels = y2 - y
            extra_inches = extra_pixels / pixels_per_inch_local
            return inch2 + extra_inches

        # If Y is below 0" marker (no snow or below base)
        if y >= self.sorted_markers[0][1]:  # Below 0" marker
            return 0.0

        # Find bracketing markers
        lower_marker = self.sorted_markers[0]  # Start with 0"
        upper_marker = self.sorted_markers[-1]  # Default to top

        for i, (inch, marker_y) in enumerate(self.sorted_markers):
            if marker_y >= y:  # This marker is at or below our Y
                lower_marker = (inch, marker_y)
                if i + 1 < len(self.sorted_markers):
                    upper_marker = self.sorted_markers[i + 1]
                break

        # Interpolate between the two markers
        inch_low, y_low = lower_marker
        inch_high, y_high = upper_marker

        if y_low == y_high:  # Avoid division by zero
            return float(inch_low)

        # Linear interpolation: how far between the markers is our Y?
        # y_low is lower marker (higher Y value, lower inches)
        # y_high is higher marker (lower Y value, higher inches)
        fraction = (y_low - y) / (y_low - y_high)
        depth = inch_low + fraction * (inch_high - inch_low)

        return depth

    def detect_snow_line(self, image: np.ndarray) -> Tuple[Optional[int], List[Dict]]:
        """
        Detect the snow line in the stake region.

        The stake is red with white numbers. Snow is white/bright.
        We look for the transition from bright (snow) to red (stake).

        Returns:
            Tuple of (median_snow_line_y, list of sample measurements)
        """
        x, y, w, h = self.stake_region
        roi = image[y:y+h, x:x+w]

        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # The stake is RED - detect red color
        # Red in HSV: H is around 0-10 or 170-180, high S, medium-high V
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Sample multiple vertical lines across the stake
        num_samples = 10
        sample_positions = np.linspace(5, w - 5, num_samples, dtype=int)
        samples = []

        base_y = self.marker_positions.get(0, y + h)  # 0" marker position

        for idx, sample_x in enumerate(sample_positions):
            sample_data = {
                'sample_index': idx,
                'x_position': x + sample_x,
                'snow_line_y': None,
                'depth_inches': None,
                'valid': False,
                'contrast': None,
                'skip_reason': None
            }

            # Get vertical column from the red mask
            col_start = max(0, sample_x - 2)
            col_end = min(w, sample_x + 3)

            # Red mask column (255 where red, 0 elsewhere)
            red_column = np.mean(red_mask[:, col_start:col_end], axis=1)

            # Gray column for brightness
            gray_column = np.mean(gray[:, col_start:col_end], axis=1)

            # Smooth the columns
            from scipy.ndimage import gaussian_filter1d
            red_smoothed = gaussian_filter1d(red_column, sigma=3)
            gray_smoothed = gaussian_filter1d(gray_column, sigma=3)

            # Find where red stake starts (scanning from bottom up)
            # Bottom should be snow (low red), stake area has high red
            snow_line_idx = None

            # Get relative position of base (0") in ROI
            base_y_rel = base_y - y
            if base_y_rel < 0 or base_y_rel >= h:
                base_y_rel = h - 10  # Fallback to near bottom

            # Scan from base upward looking for red stake
            consecutive_red = 0
            red_threshold = 100  # Threshold for "this is red stake"

            for i in range(min(base_y_rel, h - 1), -1, -1):
                if red_smoothed[i] > red_threshold:
                    consecutive_red += 1
                    if consecutive_red >= 3:
                        # Found stake - snow line is just below this
                        snow_line_idx = i + consecutive_red
                        break
                else:
                    consecutive_red = 0

            # Alternative: look for brightness transition
            if snow_line_idx is None:
                # Snow is bright, stake may be darker
                bottom_brightness = np.mean(gray_smoothed[max(0, base_y_rel-20):base_y_rel]) if base_y_rel > 20 else np.mean(gray_smoothed[-20:])
                top_brightness = np.mean(gray_smoothed[:int(h * 0.3)])

                contrast = bottom_brightness - top_brightness
                sample_data['contrast'] = float(contrast)

                if contrast < 20:
                    sample_data['skip_reason'] = 'low_contrast'
                    samples.append(sample_data)
                    continue

                threshold = (bottom_brightness + top_brightness) / 2
                consecutive_dark = 0

                for i in range(min(base_y_rel, h - 1), -1, -1):
                    if gray_smoothed[i] < threshold:
                        consecutive_dark += 1
                        if consecutive_dark >= 3:
                            snow_line_idx = i + 2
                            break
                    else:
                        consecutive_dark = 0

            if snow_line_idx is not None and snow_line_idx < h:
                snow_line_y_full = y + snow_line_idx
                sample_data['snow_line_y'] = snow_line_y_full
                sample_data['depth_inches'] = self.y_to_inches(snow_line_y_full)
                sample_data['valid'] = True
            else:
                sample_data['skip_reason'] = 'no_transition_found'

            samples.append(sample_data)

        # Calculate median from valid samples
        valid_snow_lines = [s['snow_line_y'] for s in samples if s.get('valid') and s['snow_line_y'] is not None]

        if len(valid_snow_lines) < 3:
            return None, samples

        # Remove outliers using IQR
        candidates = np.array(valid_snow_lines)
        q1, q3 = np.percentile(candidates, [25, 75])
        iqr = max(q3 - q1, 10)  # Minimum IQR of 10 pixels
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered = candidates[(candidates >= lower_bound) & (candidates <= upper_bound)]

        # Mark outliers
        for s in samples:
            if s['snow_line_y'] is not None:
                if s['snow_line_y'] < lower_bound or s['snow_line_y'] > upper_bound:
                    s['valid'] = False
                    s['skip_reason'] = 'iqr_outlier'

        if len(filtered) >= 2:
            median_y = int(np.median(filtered))
        else:
            median_y = int(np.median(candidates))

        return median_y, samples

    def measure_from_file(self, image_path: str) -> WinterParkMeasurement:
        """Measure snow depth from an image file."""
        if not os.path.exists(image_path):
            return WinterParkMeasurement(
                snow_depth_inches=None,
                confidence_score=0.0,
                stake_visible=False,
                snow_line_y=None,
                notes=f"Image not found: {image_path}"
            )

        image = cv2.imread(image_path)
        if image is None:
            return WinterParkMeasurement(
                snow_depth_inches=None,
                confidence_score=0.0,
                stake_visible=False,
                snow_line_y=None,
                notes="Failed to load image"
            )

        return self.measure(image)

    def measure(self, image: np.ndarray) -> WinterParkMeasurement:
        """Measure snow depth from an image array."""
        snow_line_y, samples = self.detect_snow_line(image)

        if snow_line_y is None:
            return WinterParkMeasurement(
                snow_depth_inches=None,
                confidence_score=0.3,
                stake_visible=True,
                snow_line_y=None,
                samples=samples,
                notes="Could not detect snow line"
            )

        # Calculate depth using marker interpolation
        snow_depth = self.y_to_inches(snow_line_y)

        # Get min/max/avg from valid samples
        valid_depths = [s['depth_inches'] for s in samples if s.get('valid') and s['depth_inches'] is not None]

        depth_min = min(valid_depths) if valid_depths else None
        depth_max = max(valid_depths) if valid_depths else None
        depth_avg = sum(valid_depths) / len(valid_depths) if valid_depths else None

        # Apply minimum threshold
        if snow_depth is not None and snow_depth < self.min_depth_threshold:
            snow_depth = None

        # Calculate confidence
        confidence = 0.5
        if len(valid_depths) >= 5:
            confidence += 0.2
        if len(valid_depths) >= 8:
            confidence += 0.1
        if depth_max and depth_min and (depth_max - depth_min) < 2.0:
            confidence += 0.2  # Consistent readings

        return WinterParkMeasurement(
            snow_depth_inches=snow_depth,
            confidence_score=min(1.0, confidence),
            stake_visible=True,
            snow_line_y=snow_line_y,
            samples=samples,
            depth_min=depth_min,
            depth_max=depth_max,
            depth_avg=depth_avg
        )


def load_calibration() -> Dict[str, Any]:
    """Load Winter Park calibration from the calibration.json file."""
    calibration_path = os.path.join(os.path.dirname(__file__), 'calibration.json')
    with open(calibration_path) as f:
        return json.load(f)


def get_measurer(debug: bool = False) -> WinterParkMeasurer:
    """Get a configured Winter Park measurer instance."""
    calibration = load_calibration()
    return WinterParkMeasurer(calibration, debug=debug)
