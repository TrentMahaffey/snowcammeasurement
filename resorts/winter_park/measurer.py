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

        # Get stake region - prefer stake_corners (new format) over legacy fields
        stake_corners = calibration.get('stake_corners', {})
        sc_tl = stake_corners.get('top_left')
        sc_tr = stake_corners.get('top_right')
        sc_bl = stake_corners.get('bottom_left')
        sc_br = stake_corners.get('bottom_right')

        if all([sc_tl, sc_tr, sc_bl, sc_br]):
            # New format: compute bounding box from stake_corners
            x = min(sc_tl[0], sc_bl[0])
            y = min(sc_tl[1], sc_tr[1])
            w = max(sc_tr[0], sc_br[0]) - x
            h = max(sc_bl[1], sc_br[1]) - y
            self.stake_region = (x, y, w, h)
            self.centerline_x = (sc_bl[0] + sc_br[0]) // 2  # Center of bottom edge
        else:
            # Legacy format
            self.stake_region = (
                calibration.get('stake_region_x', 780),
                calibration.get('stake_region_y', 280),
                calibration.get('stake_region_width', 80),
                calibration.get('stake_region_height', 510)
            )
            self.centerline_x = calibration.get('stake_centerline_x', 820)

        self.min_depth_threshold = calibration.get('min_depth_threshold', 1.0)

        # Get sample_bounds for where to take measurements (wider than stake)
        sample_bounds = calibration.get('sample_bounds', {})
        sb_tl = sample_bounds.get('top_left')
        sb_tr = sample_bounds.get('top_right')
        sb_bl = sample_bounds.get('bottom_left')
        sb_br = sample_bounds.get('bottom_right')

        if all([sb_tl, sb_tr, sb_bl, sb_br]):
            # Use sample_bounds for measurement area
            # Apply 65% width (17.5% margin each side) to avoid blown-off edges
            full_width = sb_br[0] - sb_bl[0]
            margin = int(full_width * 0.175)
            self.sample_left_x = sb_bl[0] + margin
            self.sample_right_x = sb_br[0] - margin
            self.sample_left_y = sb_bl[1]
            self.sample_right_y = sb_br[1]
            self.has_sample_bounds = True
        else:
            self.has_sample_bounds = False

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
        Sample lines start at 0" level and follow stake axis tilt.

        Returns:
            Tuple of (median_snow_line_y, list of sample measurements)
        """
        x, y, w, h = self.stake_region

        # Convert full image to grayscale for analysis
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Sample multiple vertical lines across sample_bounds (80% width)
        num_samples = 10
        samples = []

        base_y = self.marker_positions.get(0, y + h)  # 0" marker position
        img_h, img_w = image.shape[:2]

        # Get stake axis for tilt calculation
        stake_axis = self.calibration.get('stake_axis', {})
        axis_top = stake_axis.get('top')
        axis_bottom = stake_axis.get('bottom')

        # Calculate stake tilt (dx per dy)
        if axis_top and axis_bottom:
            axis_dx = axis_bottom[0] - axis_top[0]  # X change
            axis_dy = axis_bottom[1] - axis_top[1]  # Y change
            # tilt_ratio: how much X changes per unit Y (positive = stake leans right going down)
            tilt_ratio = axis_dx / axis_dy if axis_dy != 0 else 0
            axis_center_x = (axis_top[0] + axis_bottom[0]) / 2
        else:
            tilt_ratio = 0
            axis_center_x = self.centerline_x

        # Determine sample X positions
        if self.has_sample_bounds:
            # Use 80% of sample_bounds (already computed with margins)
            sample_x_positions = np.linspace(
                self.sample_left_x, self.sample_right_x, num_samples, dtype=int
            )
        else:
            # Fall back to stake_region
            sample_x_positions = np.linspace(x + 5, x + w - 5, num_samples, dtype=int)

        # Import here to avoid circular imports
        from scipy.ndimage import gaussian_filter1d

        # Define Y scan range - start at 0" marker, scan upward to top of stake
        scan_top_y = y  # Top of stake region
        scan_bottom_y = base_y  # 0" marker position (where samples START)

        # For at_base detection, use sample_bounds corners if available
        # The sample_bounds is a quadrilateral, so we need to interpolate edges per-sample
        if self.has_sample_bounds:
            sample_bounds = self.calibration.get('sample_bounds', {})
            sb_tl = sample_bounds.get('top_left', [0, scan_top_y])
            sb_tr = sample_bounds.get('top_right', [0, scan_top_y])
            sb_bl = sample_bounds.get('bottom_left', [0, scan_bottom_y])
            sb_br = sample_bounds.get('bottom_right', [0, scan_bottom_y])
        else:
            sb_tl = sb_tr = [0, scan_top_y]
            sb_bl = sb_br = [0, scan_bottom_y]

        for idx, sample_x in enumerate(sample_x_positions):
            # Calculate the 0" Y position for this sample's X (accounting for stake tilt)
            # The 0" marker_position is at the stake centerline
            # For samples away from centerline, adjust Y based on tilt
            x_offset_from_axis = sample_x - axis_center_x
            # Positive tilt_ratio means stake leans right going down
            # So for a sample to the RIGHT of axis, its 0" level is HIGHER (lower Y)
            sample_base_y = int(base_y - (x_offset_from_axis * tilt_ratio))
            sample_base_y = max(scan_top_y + 10, min(img_h - 1, sample_base_y))

            sample_data = {
                'sample_index': int(idx),
                'x_position': int(sample_x),
                'base_y': sample_base_y,  # Store where this sample starts (0" level)
                'snow_line_y': None,
                'depth_inches': None,
                'valid': False,
                'contrast': None,
                'skip_reason': None
            }

            # For tilted stake scanning:
            # Instead of vertical column, we sample along the tilted axis
            # For each Y from base upward, calculate corresponding X
            scan_length = sample_base_y - scan_top_y
            if scan_length < 10:
                sample_data['skip_reason'] = 'column_too_short'
                samples.append(sample_data)
                continue

            # Build the tilted column by sampling along stake axis direction
            gray_column = np.zeros(scan_length, dtype=float)
            for i in range(scan_length):
                # Y position (from top down)
                curr_y = scan_top_y + i
                # X position adjusted for tilt (relative to sample_base_y)
                y_from_base = sample_base_y - curr_y
                x_shift = int(y_from_base * tilt_ratio)
                curr_x = sample_x - x_shift  # Shift X based on how far from base

                # Sample 3-pixel wide average at this tilted position
                col_start = max(0, curr_x - 1)
                col_end = min(img_w, curr_x + 2)
                if col_end > col_start and 0 <= curr_y < img_h:
                    gray_column[i] = np.mean(gray_full[curr_y, col_start:col_end])

            # Smooth the column
            gray_smoothed = gaussian_filter1d(gray_column, sigma=3)
            column_height = len(gray_smoothed)

            # Helper function to check if a region is snow-like
            def is_snow_region(check_y: int) -> Tuple[bool, Dict]:
                """Check if region above check_y is snow (white, uniform texture).
                   X position is adjusted for stake tilt at check_y."""
                info = {'brightness': 0, 'color_diff': 0, 'texture_var': 0}
                check_start = max(0, check_y - 25)
                check_end = max(0, check_y - 5)

                # Calculate tilted X position at check_y
                y_from_base = sample_base_y - check_y
                x_shift = int(y_from_base * tilt_ratio)
                tilted_x = sample_x - x_shift

                c_start = max(0, tilted_x - 5)
                c_end = min(img_w, tilted_x + 6)

                if check_end <= check_start:
                    return True, info  # Can't check, assume snow

                region = image[check_start:check_end, c_start:c_end]
                if region.size == 0:
                    return True, info

                avg_bgr = np.mean(region, axis=(0, 1))
                b, g, r = avg_bgr[0], avg_bgr[1], avg_bgr[2]
                max_diff = max(abs(r-g), abs(g-b), abs(r-b))
                min_channel = min(r, g, b)

                gray_reg = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                texture_var = float(np.std(gray_reg))

                info = {'brightness': float(min_channel), 'color_diff': float(max_diff), 'texture_var': texture_var}

                # Check if snow-like
                if texture_var < 20:
                    return min_channel > 60 and max_diff < 60, info
                elif texture_var < 45:
                    return min_channel > 100 and max_diff < 50, info
                return False, info  # High texture = not snow

            # Use gradient-based detection with snow boundary checking
            gradient = np.gradient(gray_smoothed)
            gradient_threshold = 3.0
            snow_line_idx = None
            snow_check_info = {}

            # Scan from bottom (base) upward looking for brightness transition
            for i in range(column_height - 2, max(10, int(column_height * 0.1)), -1):
                local_gradient = abs(gradient[i])
                if local_gradient > gradient_threshold:
                    window_start = max(0, i - 5)
                    window_end = min(column_height - 1, i + 5)
                    brightness_change = abs(gray_smoothed[window_start] - gray_smoothed[window_end])

                    if brightness_change > 15:
                        # Found a gradient - this is a candidate snow line
                        candidate_y = int(scan_top_y + i)
                        is_snow, info = is_snow_region(candidate_y)

                        # Check if the region BELOW this gradient is snow-like
                        # The snow line is where we transition from snow to non-snow
                        check_below_y = int(scan_top_y + min(i + 20, column_height - 1))
                        is_snow_below, _ = is_snow_region(check_below_y)

                        if is_snow_below:
                            # Region below is snow, this could be our snow line
                            snow_line_idx = i
                            snow_check_info = info
                            sample_data['contrast'] = float(brightness_change)

                            if not is_snow:
                                # Region above is NOT snow - confirmed snow line boundary
                                sample_data['stopped_at_pattern'] = True
                            break  # Take first valid gradient with snow below

            # Fallback: absolute brightness threshold
            if snow_line_idx is None:
                max_brightness = np.max(gray_smoothed[:int(column_height * 0.8)])
                if max_brightness > 50:
                    brightness_threshold = max_brightness * 0.6
                    for i in range(column_height - 1, -1, -1):
                        if gray_smoothed[i] > brightness_threshold:
                            candidate_y = int(scan_top_y + i)
                            is_snow, info = is_snow_region(candidate_y)

                            # For fallback, just check if this position looks like snow boundary
                            check_below_y = int(scan_top_y + min(i + 20, column_height - 1))
                            is_snow_below, _ = is_snow_region(check_below_y)

                            if is_snow_below:
                                snow_line_idx = i
                                snow_check_info = info
                                sample_data['contrast'] = float(gray_smoothed[i])
                                if not is_snow:
                                    sample_data['stopped_at_pattern'] = True
                                break

            if snow_line_idx is None:
                sample_data['skip_reason'] = 'no_transition_found'
                samples.append(sample_data)
                continue

            # Convert to full image Y coordinate
            snow_line_y_full = int(scan_top_y + snow_line_idx)
            sample_data['snow_line_y'] = snow_line_y_full
            sample_data['depth_inches'] = float(self.y_to_inches(snow_line_y_full))

            # Calculate tilted X position at the snow line Y
            y_from_base_at_snow = sample_base_y - snow_line_y_full
            x_shift_at_snow = int(y_from_base_at_snow * tilt_ratio)
            snow_line_x = sample_x - x_shift_at_snow
            sample_data['snow_line_x'] = snow_line_x  # X at the detected snow line
            sample_data['tilt_ratio'] = tilt_ratio  # Store tilt for visualization

            # Store the snow check info
            sample_data['above_brightness'] = snow_check_info.get('brightness', 0)
            sample_data['color_diff'] = snow_check_info.get('color_diff', 0)
            sample_data['texture_var'] = snow_check_info.get('texture_var', 0)

            # Check if sample detected at base or extends full height (platform edge)
            # The sample_bounds is a quadrilateral, so interpolate edges at this sample's X
            t = idx / (num_samples - 1) if num_samples > 1 else 0.5
            sample_top_y_at_x = int(sb_tl[1] + t * (sb_tr[1] - sb_tl[1]))
            sample_bottom_y_at_x = int(sb_bl[1] + t * (sb_br[1] - sb_bl[1]))

            base_threshold = 15  # Within 15 pixels of interpolated bottom edge
            sample_height = sample_bottom_y_at_x - sample_top_y_at_x
            line_height = sample_bottom_y_at_x - snow_line_y_full

            # Check against both sample_bounds bottom AND the 0" marker position
            zero_marker_y = self.marker_positions.get(0, base_y)
            at_sample_base = snow_line_y_full >= (sample_bottom_y_at_x - base_threshold)
            at_zero_marker = snow_line_y_full >= (zero_marker_y - base_threshold)

            if at_sample_base or at_zero_marker:
                # Sample detected at the base - no snow
                sample_data['valid'] = False
                sample_data['skip_reason'] = 'at_base'
                sample_data['depth_inches'] = 0.0
            elif (snow_line_y_full >= sample_top_y_at_x and
                  sample_height > 0 and
                  (line_height / sample_height) > 0.92):
                # Sample detection is INSIDE sample_bounds AND spans >92% height
                # This is likely detecting platform edge, not actual snow
                sample_data['valid'] = False
                sample_data['skip_reason'] = 'at_base'
            else:
                sample_data['valid'] = True

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

        # Mark outliers (but don't overwrite existing skip reasons like at_base)
        for s in samples:
            if s['snow_line_y'] is not None and not s.get('skip_reason'):
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

    def is_grayscale_image(self, image: np.ndarray) -> bool:
        """
        Detect if an image is grayscale/night-vision mode.

        Night images from the webcam are in IR/grayscale mode where
        R, G, B channels are nearly identical.
        """
        if len(image.shape) < 3:
            return True  # Already grayscale

        # Sample the center region of the image
        h, w = image.shape[:2]
        center_y, center_x = h // 2, w // 2
        sample_size = 100

        roi = image[center_y-sample_size:center_y+sample_size,
                    center_x-sample_size:center_x+sample_size]

        # Calculate mean difference between R and B channels
        # Color images have significant R-B difference, grayscale has ~0
        b, g, r = cv2.split(roi)
        rb_diff = np.abs(r.astype(float) - b.astype(float)).mean()

        # If R-B difference is very low, it's grayscale
        return rb_diff < 5.0

    def measure(self, image: np.ndarray) -> WinterParkMeasurement:
        """Measure snow depth from an image array."""
        # Skip measurement for grayscale/night images
        if self.is_grayscale_image(image):
            return WinterParkMeasurement(
                snow_depth_inches=None,
                confidence_score=0.0,
                stake_visible=True,
                snow_line_y=None,
                samples=None,
                notes="Skipped: grayscale/night image"
            )

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

        # Check if detection is at or near the base (0" marker)
        # If snow line is within threshold of 0" marker, likely false detection
        # Use ~2" worth of pixels as threshold (0" at 800, 2" at 745 = 55 pixels)
        base_y = self.marker_positions.get(0, 1000)
        two_inch_y = self.marker_positions.get(2, base_y - 55)
        base_threshold_pixels = int((base_y - two_inch_y) * 0.9)  # ~90% of 0-2" range

        if snow_line_y is not None and snow_line_y >= (base_y - base_threshold_pixels):
            # Snow line is at or below base - no meaningful snow detected
            snow_depth = 0.0
            snow_line_y = base_y  # Set to exact base for visualization
            # Mark samples near base as detecting "base" not snow
            for s in samples:
                if s.get('snow_line_y') and s['snow_line_y'] >= (base_y - base_threshold_pixels):
                    s['at_base'] = True
                    s['depth_inches'] = 0.0
                    s['snow_line_y'] = base_y  # Correct visualization position

        # Apply minimum threshold (but keep 0.0 as valid "no snow" reading)
        if snow_depth is not None and snow_depth > 0 and snow_depth < self.min_depth_threshold:
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


def load_calibration(db_path: str = None) -> Dict[str, Any]:
    """Load Winter Park calibration, checking DB first then file.

    Args:
        db_path: Path to the snow_measurements.db database.
                 If provided, will check for DB calibration first.
    """
    # Try database first if path provided
    if db_path:
        try:
            import sqlite3
            from datetime import datetime
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                SELECT config_json FROM calibration_versions
                WHERE resort = 'winter_park' AND effective_from <= ?
                ORDER BY effective_from DESC, id DESC
                LIMIT 1
            ''', (now_str,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return json.loads(row[0])
        except Exception:
            pass  # Fall through to file

    # Fall back to file
    calibration_path = os.path.join(os.path.dirname(__file__), 'calibration.json')
    with open(calibration_path) as f:
        return json.load(f)


def get_measurer(debug: bool = False, db_path: str = None, calibration: Dict[str, Any] = None) -> WinterParkMeasurer:
    """Get a configured Winter Park measurer instance.

    Args:
        debug: Enable debug mode
        db_path: Path to database for loading calibration
        calibration: Pre-loaded calibration dict (overrides db_path)
    """
    if calibration is None:
        calibration = load_calibration(db_path)
    return WinterParkMeasurer(calibration, debug=debug)
