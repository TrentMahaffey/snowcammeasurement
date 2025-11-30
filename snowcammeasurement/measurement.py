"""
Snow depth measurement module using computer vision.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import os


@dataclass
class SampleMeasurement:
    """Individual vertical sample measurement."""
    sample_index: int
    x_position: int
    snow_line_y: Optional[int]
    depth_inches: Optional[float]


@dataclass
class MeasurementResult:
    """Result of a snow depth measurement."""
    snow_depth_inches: Optional[float]  # Median of valid samples
    confidence_score: float
    stake_visible: bool
    raw_pixel_measurement: Optional[int]
    stake_top_y: Optional[int]
    stake_bottom_y: Optional[int]
    snow_line_y: Optional[int]  # Median snow line Y
    notes: str = ""
    samples: Optional[list] = None  # List of SampleMeasurement dicts
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None
    depth_avg: Optional[float] = None


class SnowStakeMeasurer:
    """Measures snow depth from webcam images using computer vision."""

    def __init__(
        self,
        pixels_per_inch: Optional[float] = None,
        stake_region: Optional[Tuple[int, int, int, int]] = None,
        debug: bool = False
    ):
        """Initialize the measurer.

        Args:
            pixels_per_inch: Calibration ratio (pixels per inch of stake)
            stake_region: Region of interest as (x, y, width, height)
            debug: If True, save debug images showing detection
        """
        self.pixels_per_inch = pixels_per_inch
        self.stake_region = stake_region
        self.sample_bounds = None
        self.debug = debug
        self.calibration = None
        self.marker_positions = None  # For marker interpolation
        self._sorted_markers = None

    def y_to_inches(self, y: int) -> Optional[float]:
        """Convert Y pixel position to depth in inches.

        Uses marker interpolation if marker_positions is available,
        otherwise falls back to linear pixels_per_inch calculation.
        """
        # Try marker interpolation first
        if self.marker_positions and self._sorted_markers:
            # Get reference (0" marker position)
            ref_y = self.marker_positions.get('0') or self.marker_positions.get(0)
            if ref_y is None:
                ref_y = self._sorted_markers[0][1]

            # If Y is at or below 0" marker, no snow
            if y >= ref_y:
                return 0.0

            # If Y is above highest marker, extrapolate
            top_marker = self._sorted_markers[-1]
            if y <= top_marker[1]:
                # Extrapolate using top two markers
                if len(self._sorted_markers) >= 2:
                    inch1, y1 = self._sorted_markers[-2]
                    inch2, y2 = self._sorted_markers[-1]
                    if y1 != y2:
                        ppi_local = (y1 - y2) / (inch2 - inch1)
                        extra_pixels = y2 - y
                        extra_inches = extra_pixels / ppi_local
                        return inch2 + extra_inches
                return float(top_marker[0])

            # Find bracketing markers and interpolate
            lower_marker = self._sorted_markers[0]
            upper_marker = self._sorted_markers[-1]

            for i, (inch, marker_y) in enumerate(self._sorted_markers):
                if marker_y >= y:
                    lower_marker = (inch, marker_y)
                    if i + 1 < len(self._sorted_markers):
                        upper_marker = self._sorted_markers[i + 1]
                    break

            inch_low, y_low = lower_marker
            inch_high, y_high = upper_marker

            if y_low == y_high:
                return float(inch_low)

            fraction = (y_low - y) / (y_low - y_high)
            return inch_low + fraction * (inch_high - inch_low)

        # Fall back to linear calculation
        if self.pixels_per_inch and self.calibration:
            ref_y = self.calibration.get('reference_y')
            if ref_y is None:
                marker_positions = self.calibration.get('marker_positions', {})
                ref_y = marker_positions.get('0') or marker_positions.get(0)

            if ref_y:
                pixel_distance = ref_y - y
                if pixel_distance > 0:
                    return pixel_distance / self.pixels_per_inch
                return 0.0

        return None

    def measure_from_file(
        self,
        image_path: str,
        calibration: Optional[Dict[str, Any]] = None
    ) -> MeasurementResult:
        """Measure snow depth from an image file.

        Args:
            image_path: Path to the image file
            calibration: Optional calibration data dict

        Returns:
            MeasurementResult object
        """
        if not os.path.exists(image_path):
            return MeasurementResult(
                snow_depth_inches=None,
                confidence_score=0.0,
                stake_visible=False,
                raw_pixel_measurement=None,
                stake_top_y=None,
                stake_bottom_y=None,
                snow_line_y=None,
                notes=f"Image file not found: {image_path}"
            )

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return MeasurementResult(
                snow_depth_inches=None,
                confidence_score=0.0,
                stake_visible=False,
                raw_pixel_measurement=None,
                stake_top_y=None,
                stake_bottom_y=None,
                snow_line_y=None,
                notes="Failed to load image"
            )

        # Apply calibration if provided
        if calibration:
            self.calibration = calibration
            self.pixels_per_inch = calibration.get('pixels_per_inch')

            # Parse marker_positions for marker interpolation
            raw_markers = calibration.get('marker_positions', {})
            if raw_markers:
                self.marker_positions = {}
                for inch_str, y_pos in raw_markers.items():
                    self.marker_positions[int(inch_str)] = y_pos
                # Sort by inch value for interpolation
                self._sorted_markers = sorted(self.marker_positions.items(), key=lambda x: x[0])

            # Store sample_bounds for use in sample X positioning (if present)
            sample_bounds = calibration.get('sample_bounds', {})
            sb_tl = sample_bounds.get('top_left')
            sb_tr = sample_bounds.get('top_right')
            sb_bl = sample_bounds.get('bottom_left')
            sb_br = sample_bounds.get('bottom_right')
            if all([sb_tl, sb_tr, sb_bl, sb_br]):
                self.sample_bounds = sample_bounds

            # Determine stake_region (vertical scan area)
            # Priority: stake_corners > legacy stake_region fields
            stake_corners = calibration.get('stake_corners', {})
            sc_tl = stake_corners.get('top_left')
            sc_tr = stake_corners.get('top_right')
            sc_bl = stake_corners.get('bottom_left')
            sc_br = stake_corners.get('bottom_right')

            if all([sc_tl, sc_tr, sc_bl, sc_br]):
                # New format: compute bounding box from stake_corners quadrilateral
                # But extend to include sample_bounds bottom if present (to capture snow line)
                x = min(sc_tl[0], sc_bl[0])
                y = min(sc_tl[1], sc_tr[1])
                x_right = max(sc_tr[0], sc_br[0])
                y_bottom = max(sc_bl[1], sc_br[1])

                # If sample_bounds extends lower than stake_corners, include that area
                if self.sample_bounds:
                    y_bottom = max(y_bottom, sb_bl[1], sb_br[1])
                    x = min(x, sb_tl[0], sb_bl[0])
                    x_right = max(x_right, sb_tr[0], sb_br[0])

                w = x_right - x
                h = y_bottom - y
                self.stake_region = (x, y, w, h)
            elif all(calibration.get(k) is not None for k in ['base_region_x', 'base_region_y', 'base_region_width', 'base_region_height']):
                self.stake_region = (
                    calibration['base_region_x'],
                    calibration['base_region_y'],
                    calibration['base_region_width'],
                    calibration['base_region_height']
                )
            elif all(calibration.get(k) is not None for k in ['stake_region_x', 'stake_region_y', 'stake_region_width', 'stake_region_height']):
                self.stake_region = (
                    calibration['stake_region_x'],
                    calibration['stake_region_y'],
                    calibration['stake_region_width'],
                    calibration['stake_region_height']
                )
            elif calibration.get('stake_base_y') and calibration.get('stake_top_y'):
                # Legacy: Extract region around calibrated stake location
                x_margin = 100  # pixels on each side
                img_height, img_width = image.shape[:2]
                stake_center_x = img_width // 2  # Assume stake is centered
                self.stake_region = (
                    max(0, stake_center_x - x_margin),
                    calibration['stake_top_y'],
                    min(img_width, x_margin * 2),
                    calibration['stake_base_y'] - calibration['stake_top_y']
                )

        return self.measure(image)

    def measure(self, image: np.ndarray) -> MeasurementResult:
        """Measure snow depth from an image array.

        Args:
            image: OpenCV image array (BGR format)

        Returns:
            MeasurementResult object
        """
        # Extract region of interest if specified
        roi = image
        roi_offset_x, roi_offset_y = 0, 0
        if self.stake_region:
            x, y, w, h = self.stake_region
            roi = image[y:y+h, x:x+w]
            roi_offset_x, roi_offset_y = x, y

        # Detect stake
        stake_lines = self._detect_stake(roi)

        if not stake_lines:
            return MeasurementResult(
                snow_depth_inches=None,
                confidence_score=0.0,
                stake_visible=False,
                raw_pixel_measurement=None,
                stake_top_y=None,
                stake_bottom_y=None,
                snow_line_y=None,
                notes="No stake detected"
            )

        # Find the most prominent vertical line (likely the stake)
        stake_line = self._select_best_stake_line(stake_lines)

        # Detect snow line (where stake meets snow) - returns median and all samples
        snow_line_y, samples = self._detect_snow_line(roi, stake_line)

        if snow_line_y is None:
            return MeasurementResult(
                snow_depth_inches=None,
                confidence_score=0.3,
                stake_visible=True,
                raw_pixel_measurement=None,
                stake_top_y=stake_line['top_y'] + roi_offset_y,
                stake_bottom_y=stake_line['bottom_y'] + roi_offset_y,
                snow_line_y=None,
                notes="Stake detected but snow line unclear",
                samples=samples
            )

        # Calculate visible stake length in pixels
        visible_pixels = snow_line_y - stake_line['top_y']

        # Get snow depth from valid samples only (those that passed contrast checks and aren't outliers)
        snow_depth_inches = None
        depth_min = None
        depth_max = None
        depth_avg = None

        if samples:
            # Only use samples marked as valid (passed contrast check, not IQR outlier)
            valid_depths = [s['depth_inches'] for s in samples
                           if s.get('valid') and s.get('depth_inches') is not None]
            if valid_depths:
                depth_min = min(valid_depths)
                depth_max = max(valid_depths)
                depth_avg = sum(valid_depths) / len(valid_depths)
                snow_depth_inches = depth_avg  # Use average of valid samples

        # Fallback to old calculation if samples didn't provide depth
        if snow_depth_inches is None and self.pixels_per_inch:
            if self.calibration and 'reference_y' in self.calibration:
                reference_y = self.calibration['reference_y']
                pixel_distance = reference_y - snow_line_y

                if pixel_distance > 0:
                    snow_depth_inches = pixel_distance / self.pixels_per_inch
                else:
                    snow_depth_inches = 0.0

        # Apply minimum depth threshold - readings below threshold are likely glare/noise
        # Glare and reflections create false readings in the 0.5-0.9" range
        # Real snow accumulation that low is rare and hard to distinguish from artifacts
        # Threshold is configurable per-resort (default 1.0")
        min_depth_threshold = 1.0  # Default
        if self.calibration and 'min_depth_threshold' in self.calibration:
            min_depth_threshold = self.calibration['min_depth_threshold']
        if snow_depth_inches is not None and snow_depth_inches < min_depth_threshold:
            snow_depth_inches = None  # Treat as no valid measurement

        # Calculate confidence based on detection quality
        confidence = self._calculate_confidence(stake_line, snow_line_y)

        # Save debug image if enabled
        if self.debug:
            self._save_debug_image(roi, stake_line, snow_line_y)

        return MeasurementResult(
            snow_depth_inches=snow_depth_inches,
            confidence_score=confidence,
            stake_visible=True,
            raw_pixel_measurement=visible_pixels,
            stake_top_y=stake_line['top_y'] + roi_offset_y,
            stake_bottom_y=stake_line['bottom_y'] + roi_offset_y,
            snow_line_y=snow_line_y,
            notes="",
            samples=samples,
            depth_min=depth_min,
            depth_max=depth_max,
            depth_avg=depth_avg
        )

    def _detect_stake(self, image: np.ndarray) -> list:
        """Detect vertical lines (potential stakes) in the image.

        Args:
            image: OpenCV image array

        Returns:
            List of detected line dictionaries
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=20
        )

        if lines is None:
            return []

        # Filter for near-vertical lines
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle from vertical
            if x2 - x1 == 0:
                angle = 0  # Perfectly vertical
            else:
                angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))

            # Keep lines within 15 degrees of vertical
            if angle > 75 or angle < 15:
                vertical_lines.append({
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'top_y': min(y1, y2),
                    'bottom_y': max(y1, y2),
                    'length': abs(y2 - y1),
                    'angle': angle
                })

        return vertical_lines

    def _select_best_stake_line(self, lines: list) -> Optional[dict]:
        """Select the most likely stake from detected lines.

        Args:
            lines: List of detected line dictionaries

        Returns:
            Best stake line dictionary or None
        """
        if not lines:
            return None

        # Score lines by length (longer is better for stakes)
        scored_lines = sorted(lines, key=lambda x: x['length'], reverse=True)

        # Return the longest vertical line
        return scored_lines[0]

    def _detect_snow_line(
        self,
        image: np.ndarray,
        stake_line: dict
    ) -> Tuple[Optional[int], Optional[list]]:
        """Detect where the stake meets the snow (snow line).

        Takes 10 vertical samples across the measurement region and returns
        both the median snow line and all individual sample measurements.

        Args:
            image: OpenCV image array
            stake_line: Dictionary with stake line info

        Returns:
            Tuple of (median_snow_line_y, list of sample dicts) in ROI coordinates
            Sample dicts have: sample_index, x_position, snow_line_y, depth_inches
        """
        # Determine measurement strip parameters
        # Note: 'image' here is already the ROI (cropped to stake_region in measure())
        # So coordinates are relative to the stake region, not full image

        if self.calibration and self.stake_region:
            # Use stake_region for measurement (may be base_region if configured)
            # The image passed here is already the stake_region ROI
            roi_x_offset = self.stake_region[0]  # For converting back to full image coords
            roi_y_offset = self.stake_region[1]
            full_width = image.shape[1]  # Width of the ROI
            full_height = image.shape[0]  # Height of the ROI

            # Determine X range for sample lines
            # Priority: stake_corners (actual stake width) > legacy methods
            # sample_bounds is for visualization, stake_corners is for measurement
            stake_corners = self.calibration.get('stake_corners', {})
            sc_bl = stake_corners.get('bottom_left')
            sc_br = stake_corners.get('bottom_right')

            if sc_bl and sc_br:
                # Use stake_corners for X range (where the actual stake is)
                x_start = sc_bl[0] - roi_x_offset
                x_end = sc_br[0] - roi_x_offset
                actual_width = x_end - x_start
            elif self.sample_bounds:
                # Fall back to sample_bounds if no stake_corners
                sb_bl = self.sample_bounds.get('bottom_left')
                sb_br = self.sample_bounds.get('bottom_right')
                x_start = sb_bl[0] - roi_x_offset
                x_end = sb_br[0] - roi_x_offset
                actual_width = x_end - x_start
            else:
                # Check if base_region is being used (samples span full base, not centered on stake)
                has_base_region = all(self.calibration.get(k) is not None for k in
                    ['base_region_x', 'base_region_y', 'base_region_width', 'base_region_height'])

                if has_base_region:
                    # base_region: Use full width (75% of it) centered on the ROI, not on stake
                    measure_width = int(self.stake_region[2] * 0.75)  # 75% of base region width
                    x_center = full_width // 2  # Center of the ROI (base_region)
                elif 'stake_centerline_x' in self.calibration:
                    # stake_region with centerline: center on stake
                    centerline_x = self.calibration['stake_centerline_x'] - roi_x_offset
                    measure_width = int(self.stake_region[2] * 0.75)  # 75% of stake width
                    x_center = centerline_x
                else:
                    # Fallback: center on ROI
                    measure_width = int(self.stake_region[2] * 0.75)
                    x_center = full_width // 2

                x_start = max(0, x_center - measure_width // 2)
                x_end = min(full_width, x_center + measure_width // 2)
                actual_width = x_end - x_start

            if actual_width < 10:
                return None, None

            # Extract measurement strip from ROI (use full height of ROI)
            gray_region = cv2.cvtColor(image[:, x_start:x_end], cv2.COLOR_BGR2GRAY)
            region_height = gray_region.shape[0]
        else:
            # Fall back to detected stake line
            roi_x_offset = 0
            roi_y_offset = stake_line['top_y']
            x_center = (stake_line['x1'] + stake_line['x2']) // 2
            measure_width = 30

            x_start = max(0, x_center - measure_width // 2)
            x_end = min(image.shape[1], x_center + measure_width // 2)
            actual_width = x_end - x_start

            if actual_width < 10:
                return None, None

            region = image[stake_line['top_y']:stake_line['bottom_y'], x_start:x_end]
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            region_height = gray_region.shape[0]

        if region_height < 10:
            return None, None

        # Get calibration values for depth calculation
        reference_y = None
        if self.calibration:
            reference_y = self.calibration.get('reference_y')
            # Fall back to marker_positions["0"] if no explicit reference_y
            if reference_y is None:
                marker_positions = self.calibration.get('marker_positions', {})
                if '0' in marker_positions:
                    reference_y = marker_positions['0']
                elif 0 in marker_positions:
                    reference_y = marker_positions[0]
        ppi = self.pixels_per_inch

        # Take 10 vertical samples evenly spaced across the measurement width
        num_samples = 10
        sample_positions = np.linspace(0, actual_width - 1, num_samples, dtype=int)
        samples = []

        try:
            from scipy.ndimage import gaussian_filter1d
            use_scipy = True
        except:
            use_scipy = False

        for idx, sample_x in enumerate(sample_positions):
            # Extract vertical column (with small averaging window for noise reduction)
            col_start = max(0, sample_x - 2)
            col_end = min(actual_width, sample_x + 3)
            column = np.mean(gray_region[:, col_start:col_end], axis=1)

            sample_data = {
                'sample_index': int(idx),
                'x_position': int(roi_x_offset + x_start + sample_x),  # Full image x coordinate
                'snow_line_y': None,
                'depth_inches': None,
                'valid': False,
                'contrast': None,
                'skip_reason': None
            }

            if len(column) < 10:
                sample_data['skip_reason'] = 'column_too_short'
                samples.append(sample_data)
                continue

            # Smooth the column
            if use_scipy:
                smoothed = gaussian_filter1d(column, sigma=5)
            else:
                smoothed = column

            # The stake region goes from top (low Y in image) to bottom (high Y)
            # Snow is bright (high values), exposed stake is darker
            # We need to find where snow ends, scanning from reference_y upward

            # Get reference_y position relative to stake region
            ref_y_rel = None
            if reference_y and self.stake_region:
                ref_y_rel = reference_y - self.stake_region[1]  # Convert to region coordinates

            # If we have a reference point, scan from there upward
            # Otherwise fall back to scanning from bottom
            if ref_y_rel is not None and 0 < ref_y_rel <= region_height:
                # Scan from reference point (base of stake/0") upward
                # Snow should be bright at the bottom, stake should be darker above snow line

                # Sample brightness at the very bottom (should be snow or base)
                bottom_brightness = np.mean(smoothed[max(0, ref_y_rel-20):ref_y_rel]) if ref_y_rel > 20 else np.mean(smoothed[-20:])
                # Sample brightness in the upper stake area (exposed stake)
                top_brightness = np.mean(smoothed[:int(region_height * 0.3)])

                # Threshold between snow (bright) and stake (darker)
                # Snow is typically brighter than exposed wooden stake
                threshold = (bottom_brightness + top_brightness) / 2

                # Check contrast and brightness relationship
                # During day: snow (bottom) is bright, stake (top) is darker
                # At night: this relationship inverts (IR lighting), making detection unreliable
                contrast = bottom_brightness - top_brightness  # Should be positive during day
                # Threshold 35 allows right-side samples without backdrop (contrast ~32-50)
                # while still rejecting night images (contrast ~29-31 or negative)
                min_contrast = 35

                # Store contrast for diagnostics
                sample_data['contrast'] = float(contrast)

                if contrast < min_contrast:
                    # Either low contrast OR inverted brightness (night) - skip this sample
                    sample_data['skip_reason'] = 'low_contrast' if contrast >= 0 else 'inverted_brightness'
                    samples.append(sample_data)
                    continue

                # Scan from reference point upward looking for transition from bright to dark
                snow_line_idx = None
                consecutive_dark = 0

                # Ensure we start within bounds (ref_y_rel might equal region_height)
                start_y = min(ref_y_rel - 1, region_height - 1)
                for i in range(start_y, -1, -1):
                    if smoothed[i] < threshold:
                        consecutive_dark += 1
                        if consecutive_dark >= 3:  # Found transition to darker (stake above snow)
                            snow_line_idx = i + 2  # Adjust to snow surface
                            break
                    else:
                        consecutive_dark = 0
            else:
                # Fallback: original method
                lower_region = smoothed[int(region_height * 0.75):]
                upper_region = smoothed[:int(region_height * 0.3)]

                if len(lower_region) < 5 or len(upper_region) < 5:
                    samples.append(sample_data)
                    continue

                snow_brightness = np.percentile(lower_region, 75)
                stake_brightness = np.percentile(upper_region, 25)
                threshold = (snow_brightness + stake_brightness) / 2

                snow_line_idx = None
                consecutive_dark = 0

                for i in range(region_height - 1, -1, -1):
                    if smoothed[i] < threshold:
                        consecutive_dark += 1
                        if consecutive_dark >= 5:
                            snow_line_idx = i + 3
                            break
                    else:
                        consecutive_dark = 0

            if snow_line_idx is not None:
                # Convert to full image Y coordinate
                snow_line_y_full = int(roi_y_offset + snow_line_idx)
                sample_data['snow_line_y'] = snow_line_y_full
                sample_data['valid'] = True

                # Calculate depth in inches using y_to_inches (supports marker interpolation)
                depth = self.y_to_inches(snow_line_y_full)
                if depth is not None:
                    sample_data['depth_inches'] = depth
            else:
                sample_data['skip_reason'] = 'no_transition_found'

            samples.append(sample_data)

        # Calculate median snow line from valid samples
        valid_snow_lines = [s['snow_line_y'] for s in samples if s['snow_line_y'] is not None]

        # Require minimum 4 valid samples (out of 10) for a reliable measurement
        # This prevents night images with 1-2 lucky samples from producing readings
        min_valid_samples = 4
        if len(valid_snow_lines) < min_valid_samples:
            return None, samples

        median_snow_line_y = None
        if len(valid_snow_lines) >= 3:
            # Remove outliers using IQR method
            candidates = np.array(valid_snow_lines)
            q1 = np.percentile(candidates, 25)
            q3 = np.percentile(candidates, 75)
            iqr = q3 - q1
            # Use minimum IQR of 10 pixels (~0.4") to avoid over-filtering when data is consistent
            # This prevents rejecting samples that differ by only a few pixels
            min_iqr = 10
            effective_iqr = max(iqr, min_iqr)
            lower_bound = q1 - 1.5 * effective_iqr
            upper_bound = q3 + 1.5 * effective_iqr
            filtered = candidates[(candidates >= lower_bound) & (candidates <= upper_bound)]

            # Mark samples that are IQR outliers
            for s in samples:
                if s['snow_line_y'] is not None:
                    y = s['snow_line_y']
                    if y < lower_bound or y > upper_bound:
                        s['iqr_outlier'] = True
                        s['valid'] = False  # Mark as invalid since it's filtered
                        if not s.get('skip_reason'):
                            s['skip_reason'] = 'iqr_outlier'

            if len(filtered) >= 2:
                median_snow_line_y = int(np.median(filtered))
            else:
                # After IQR filtering, if fewer than 2 samples remain, use all candidates
                median_snow_line_y = int(np.median(candidates))

        # For shallow snow (< 2"), use only the center sample
        # The stake is centered on its base, so the center sample is most accurate for shallow depths
        if reference_y and ppi and median_snow_line_y:
            initial_depth = (reference_y - median_snow_line_y) / ppi
            if initial_depth < 2.0:
                # Get center sample (index 4 or 5 out of 0-9)
                center_idx = num_samples // 2
                center_sample = samples[center_idx]
                if center_sample['snow_line_y'] is not None and center_sample['valid']:
                    median_snow_line_y = center_sample['snow_line_y']
                    # Mark non-center samples as not used for shallow measurement
                    for s in samples:
                        if s['sample_index'] != center_idx:
                            s['shallow_excluded'] = True

        return median_snow_line_y, samples

    def _calculate_confidence(
        self,
        stake_line: dict,
        snow_line_y: Optional[int]
    ) -> float:
        """Calculate confidence score for the measurement.

        Args:
            stake_line: Dictionary with stake line info
            snow_line_y: Y-coordinate of detected snow line

        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence

        # Longer stake lines are more reliable
        if stake_line['length'] > 200:
            confidence += 0.2
        elif stake_line['length'] > 100:
            confidence += 0.1

        # More vertical lines are more reliable
        if stake_line['angle'] < 5:
            confidence += 0.2
        elif stake_line['angle'] < 10:
            confidence += 0.1

        # Clear snow line detection
        if snow_line_y is not None:
            confidence += 0.1

        return min(1.0, confidence)

    def _save_debug_image(
        self,
        image: np.ndarray,
        stake_line: dict,
        snow_line_y: Optional[int]
    ):
        """Save an annotated debug image showing detections.

        Args:
            image: OpenCV image array
            stake_line: Dictionary with stake line info
            snow_line_y: Y-coordinate of detected snow line
        """
        debug_image = image.copy()

        # Draw stake line
        cv2.line(
            debug_image,
            (stake_line['x1'], stake_line['y1']),
            (stake_line['x2'], stake_line['y2']),
            (0, 255, 0),  # Green
            2
        )

        # Draw snow line if detected
        if snow_line_y is not None:
            cv2.line(
                debug_image,
                (0, snow_line_y),
                (image.shape[1], snow_line_y),
                (0, 0, 255),  # Red
                2
            )

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = f"debug_measurement_{timestamp}.jpg"
        cv2.imwrite(debug_path, debug_image)
        print(f"Debug image saved: {debug_path}")


def calibrate_from_image(
    image_path: str,
    known_stake_length_inches: float,
    stake_region: Optional[Tuple[int, int, int, int]] = None
) -> Dict[str, Any]:
    """Create calibration data from a reference image.

    Args:
        image_path: Path to reference image with visible stake
        known_stake_length_inches: Known length of visible stake in inches
        stake_region: Optional region of interest as (x, y, width, height)

    Returns:
        Dictionary with calibration data
    """
    measurer = SnowStakeMeasurer(stake_region=stake_region, debug=True)
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    result = measurer.measure(image)

    if not result.stake_visible or result.raw_pixel_measurement is None:
        raise ValueError("Could not detect stake in calibration image")

    pixels_per_inch = result.raw_pixel_measurement / known_stake_length_inches

    return {
        'pixels_per_inch': pixels_per_inch,
        'stake_top_y': result.stake_top_y,
        'stake_base_y': result.stake_bottom_y,
        'reference_image_path': image_path,
        'notes': f"Calibrated with {known_stake_length_inches} inch visible stake"
    }
