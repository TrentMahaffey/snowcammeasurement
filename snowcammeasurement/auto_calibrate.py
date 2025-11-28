#!/usr/bin/env python3
"""
Auto-calibration module for snow stake measurement.

Detects inch markers on the stake to automatically calculate:
- pixels_per_inch: Scale based on marker spacing
- tilt_angle: Camera tilt from marker alignment
- reference_y: Y coordinate of stake base (0")
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Try to import OCR libraries, fall back gracefully
HAS_EASYOCR = False
HAS_PYTESSERACT = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    pass

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    pass

HAS_OCR = HAS_EASYOCR or HAS_PYTESSERACT

# Global easyocr reader (lazy init to save memory)
_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None and HAS_EASYOCR:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


@dataclass
class DetectedMarker:
    """A detected inch marker on the stake."""
    inch_value: int
    x: int
    y: int
    confidence: float
    method: str  # 'ocr', 'template', 'edge'


@dataclass
class AutoCalibrationResult:
    """Result of auto-calibration attempt."""
    success: bool
    pixels_per_inch: Optional[float]
    tilt_angle: Optional[float]
    reference_y: Optional[int]
    stake_centerline_x: Optional[int]
    confidence: float
    markers_detected: List[DetectedMarker]
    method: str
    error: Optional[str] = None


class BaseStakeCalibrator:
    """Base class for stake calibrators with common functionality."""

    # Subclasses should override these
    MARKER_INCHES = []
    MARKER_MAPPING = {}  # OCR misread corrections

    def __init__(self, stake_region: Optional[Tuple[int, int, int, int]] = None):
        """Initialize calibrator.

        Args:
            stake_region: Optional (x, y, width, height) to limit search area
        """
        self.stake_region = stake_region

    def calibrate_from_image(
        self,
        image: np.ndarray,
        fallback_config: Optional[Dict] = None
    ) -> AutoCalibrationResult:
        """Attempt auto-calibration from an image.

        Args:
            image: OpenCV image (BGR)
            fallback_config: Fallback calibration values if auto fails

        Returns:
            AutoCalibrationResult
        """
        # Extract region of interest if specified
        roi = image
        roi_offset_x, roi_offset_y = 0, 0
        if self.stake_region:
            x, y, w, h = self.stake_region
            roi = image[y:y+h, x:x+w]
            roi_offset_x, roi_offset_y = x, y

        markers = []

        # Try OCR-based detection first (most accurate)
        if HAS_OCR:
            ocr_markers = self._detect_markers_ocr(roi)
            # Convert ROI-relative coordinates to absolute image coordinates
            for m in ocr_markers:
                m.x += roi_offset_x
                m.y += roi_offset_y
            markers.extend(ocr_markers)

        # If OCR didn't find enough markers, try edge-based detection
        if len(markers) < 3:
            edge_markers = self._detect_markers_edge(roi)
            for m in edge_markers:
                m.x += roi_offset_x
                m.y += roi_offset_y
            # Add edge markers that don't overlap with OCR markers
            for em in edge_markers:
                if not any(abs(em.inch_value - m.inch_value) == 0 for m in markers):
                    markers.append(em)

        # Need at least 2 markers to calculate pixels_per_inch
        if len(markers) < 2:
            if fallback_config:
                return AutoCalibrationResult(
                    success=True,
                    pixels_per_inch=fallback_config.get('pixels_per_inch'),
                    tilt_angle=fallback_config.get('tilt_angle'),
                    reference_y=fallback_config.get('reference_y'),
                    stake_centerline_x=fallback_config.get('stake_centerline_x'),
                    confidence=0.3,
                    markers_detected=markers,
                    method='fallback',
                    error=f'Only {len(markers)} markers detected, using fallback'
                )
            return AutoCalibrationResult(
                success=False,
                pixels_per_inch=None,
                tilt_angle=None,
                reference_y=None,
                stake_centerline_x=None,
                confidence=0.0,
                markers_detected=markers,
                method='failed',
                error=f'Only {len(markers)} markers detected, need at least 2'
            )

        # Sort markers by inch value
        markers.sort(key=lambda m: m.inch_value)

        # Calculate pixels_per_inch from marker spacing
        ppi_values = []
        for i in range(len(markers) - 1):
            m1, m2 = markers[i], markers[i+1]
            inch_diff = m2.inch_value - m1.inch_value
            pixel_diff = m1.y - m2.y  # Y decreases as we go up
            if inch_diff > 0 and pixel_diff > 0:
                ppi = pixel_diff / inch_diff
                ppi_values.append(ppi)

        if not ppi_values:
            return AutoCalibrationResult(
                success=False,
                pixels_per_inch=None,
                tilt_angle=None,
                reference_y=None,
                stake_centerline_x=None,
                confidence=0.0,
                markers_detected=markers,
                method='failed',
                error='Could not calculate pixels_per_inch from markers'
            )

        # Use median to be robust to outliers
        pixels_per_inch = float(np.median(ppi_values))

        # Calculate tilt from marker x positions
        tilt_angle = self._calculate_tilt(markers)

        # Calculate reference_y (0" position) by extrapolating from lowest marker
        lowest_marker = markers[0]  # Sorted by inch value, so first is lowest
        reference_y = int(lowest_marker.y + (lowest_marker.inch_value * pixels_per_inch))

        # Calculate stake centerline from marker x positions
        stake_centerline_x = int(np.median([m.x for m in markers]))

        # Calculate confidence based on number of markers and consistency
        ppi_std = np.std(ppi_values) if len(ppi_values) > 1 else 0
        max_markers = len(self.MARKER_INCHES) if self.MARKER_INCHES else 5
        confidence = min(1.0, len(markers) / max_markers) * max(0.5, 1.0 - ppi_std / pixels_per_inch)

        return AutoCalibrationResult(
            success=True,
            pixels_per_inch=pixels_per_inch,
            tilt_angle=tilt_angle,
            reference_y=reference_y,
            stake_centerline_x=stake_centerline_x,
            confidence=confidence,
            markers_detected=markers,
            method='auto'
        )

    def _detect_markers_ocr(self, roi: np.ndarray) -> List[DetectedMarker]:
        """Detect markers using OCR to find the numbers."""
        markers = []

        if not HAS_OCR:
            return markers

        # Try easyocr first (better at detecting numbers in varied lighting)
        if HAS_EASYOCR:
            try:
                reader = get_easyocr_reader()
                if reader:
                    results = reader.readtext(roi, allowlist='0123456789')

                    # Collect all detected numbers with positions
                    detected = []
                    for (bbox, text, conf) in results:
                        text = text.strip()
                        if text:
                            x = int((bbox[0][0] + bbox[2][0]) / 2)
                            y = int((bbox[0][1] + bbox[2][1]) / 2)
                            detected.append({'text': text, 'x': x, 'y': y, 'conf': conf})

                    # Sort by Y position (top to bottom = highest inch to lowest)
                    detected.sort(key=lambda d: d['y'])

                    # Use the marker mapping for this calibrator
                    for d in detected:
                        inch_value = self.MARKER_MAPPING.get(d['text'])
                        if inch_value:
                            # Check for duplicates
                            if not any(m.inch_value == inch_value for m in markers):
                                markers.append(DetectedMarker(
                                    inch_value=inch_value,
                                    x=d['x'],
                                    y=d['y'],
                                    confidence=d['conf'],
                                    method='easyocr'
                                ))

                    # Validate Y-ordering: markers should be in descending inch order
                    # (higher Y = lower on image = smaller inch value)
                    if len(markers) >= 2:
                        markers.sort(key=lambda m: m.y)
                        expected_order = sorted([m.inch_value for m in markers], reverse=True)
                        actual_order = [m.inch_value for m in markers]
                        if actual_order != expected_order:
                            # Order doesn't match - may need position-based reassignment
                            pass  # Keep the detected values for now

            except Exception as e:
                pass  # easyocr failed, try pytesseract

        # Fall back to pytesseract if easyocr didn't find enough
        if len(markers) < 2 and HAS_PYTESSERACT:
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)

                # Threshold to isolate dark text on light stake
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

                for i, text in enumerate(data['text']):
                    text = text.strip()
                    # Look for our marker numbers using the mapping
                    inch_value = self.MARKER_MAPPING.get(text)
                    if inch_value:
                        x = data['left'][i] + data['width'][i] // 2
                        y = data['top'][i] + data['height'][i] // 2
                        conf = data['conf'][i] / 100.0 if data['conf'][i] > 0 else 0.5

                        # Don't add duplicates
                        if not any(m.inch_value == inch_value for m in markers):
                            markers.append(DetectedMarker(
                                inch_value=inch_value,
                                x=x,
                                y=y,
                                confidence=conf,
                                method='pytesseract'
                            ))
            except Exception as e:
                pass  # OCR failed

        return markers

    def _detect_markers_edge(self, roi: np.ndarray) -> List[DetectedMarker]:
        """Detect markers using edge detection and horizontal line patterns."""
        markers = []

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Look for horizontal lines (marker tick marks)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                                minLineLength=20, maxLineGap=5)

        if lines is None:
            return markers

        # Find horizontal lines (slope near 0)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # Nearly horizontal
                horizontal_lines.append((y1 + y2) // 2)

        # This method is less reliable, so we assign lower confidence
        # The markers would need to be matched to expected inch values
        # based on their relative spacing

        return markers  # For now, return empty - edge detection needs more work

    def _calculate_tilt(self, markers: List[DetectedMarker]) -> float:
        """Calculate camera tilt angle from marker positions."""
        if len(markers) < 2:
            return 0.0

        # Fit a line to the marker positions
        xs = np.array([m.x for m in markers])
        ys = np.array([m.y for m in markers])

        # Linear regression: x = slope * y + intercept
        # (using y as independent variable since stake is vertical)
        if np.std(ys) < 1:
            return 0.0

        slope = np.cov(xs, ys)[0, 1] / np.var(ys)

        # Convert slope to angle (in degrees)
        tilt_angle = np.degrees(np.arctan(slope))

        return float(tilt_angle)


class SnowmassCalibrator(BaseStakeCalibrator):
    """Auto-calibrator for Snowmass snow stake.

    The Snowmass stake has:
    - "24 HOUR" text at the base
    - Inch markers at 2", 6", 10", 14", 18"
    - Yellow/orange stake with black text
    """

    # Known marker positions (inches from base)
    MARKER_INCHES = [2, 6, 10, 14, 18]

    # EasyOCR common misreadings on stake markers
    MARKER_MAPPING = {
        '18': 18, '8': 18,  # 18 or misread as 8
        '14': 14, '4': 14,  # 14 or misread as 4
        '10': 10, '0': 10, '1': 10,  # 10 or misread as 0 or 1
        '6': 6, '5': 6,  # 6 or misread as 5
        '2': 2,
    }


class WinterParkCalibrator(BaseStakeCalibrator):
    """Auto-calibrator for Winter Park snow stake.

    The Winter Park stake has:
    - "Winter Park" header sign
    - Decorative mountain artwork on left side
    - Ruler strip on right side with inch markers at 2, 4, 6, 8, 10, 12, 14, 16, 18
    - 2-inch increments (more markers = better calibration)
    - IR/night vision camera (black & white images)
    """

    # Known marker positions (inches from base) - 2" increments
    MARKER_INCHES = [2, 4, 6, 8, 10, 12, 14, 16, 18]

    # EasyOCR common misreadings for Winter Park markers
    # More markers means more potential for single-digit misreads
    MARKER_MAPPING = {
        '18': 18, '8': 18,  # 18 or misread as 8
        '16': 16,
        '14': 14, '4': 14,  # 14 or misread as 4
        '12': 12,
        '10': 10, '0': 10, '1': 10,  # 10 or misread as 0 or 1
        # For single digits, we need context - could be actual or misread
        # 8 -> could be actual 8 or misread 18
        # 6 -> actual 6
        # 4 -> could be actual 4 or misread 14
        # 2 -> actual 2
        '6': 6,
        '2': 2,
    }

    def _detect_markers_ocr(self, roi: np.ndarray) -> List[DetectedMarker]:
        """Detect markers using OCR - specialized for Winter Park.

        Winter Park has more markers (9 vs 5) and uses IR imaging.
        We need smarter disambiguation for single-digit OCR results.
        """
        markers = []

        if not HAS_OCR or not HAS_EASYOCR:
            return super()._detect_markers_ocr(roi)

        try:
            reader = get_easyocr_reader()
            if not reader:
                return markers

            results = reader.readtext(roi, allowlist='0123456789')

            # Collect all detected numbers with positions
            detected = []
            for (bbox, text, conf) in results:
                text = text.strip()
                if text:
                    x = int((bbox[0][0] + bbox[2][0]) / 2)
                    y = int((bbox[0][1] + bbox[2][1]) / 2)
                    detected.append({'text': text, 'x': x, 'y': y, 'conf': conf})

            if not detected:
                return markers

            # Sort by Y position (top to bottom = highest inch to lowest)
            detected.sort(key=lambda d: d['y'])

            # For Winter Park, we can use relative Y positions to disambiguate
            # Since we have 9 markers at 2" intervals over 16", we expect
            # roughly equal spacing between consecutive markers

            # First pass: assign obvious values (two-digit numbers)
            for d in detected:
                text = d['text']
                if len(text) == 2 and text.isdigit():
                    inch_value = int(text)
                    if inch_value in self.MARKER_INCHES:
                        if not any(m.inch_value == inch_value for m in markers):
                            markers.append(DetectedMarker(
                                inch_value=inch_value,
                                x=d['x'],
                                y=d['y'],
                                confidence=d['conf'],
                                method='easyocr'
                            ))

            # Second pass: disambiguate single digits using Y position
            if len(markers) >= 2:
                # Calculate expected pixels per inch from known markers
                markers.sort(key=lambda m: m.inch_value)
                ppi_estimates = []
                for i in range(len(markers) - 1):
                    m1, m2 = markers[i], markers[i+1]
                    inch_diff = m2.inch_value - m1.inch_value
                    pixel_diff = m1.y - m2.y
                    if inch_diff > 0 and pixel_diff > 0:
                        ppi_estimates.append(pixel_diff / inch_diff)

                if ppi_estimates:
                    estimated_ppi = np.median(ppi_estimates)

                    # Now assign single-digit detections based on expected Y position
                    for d in detected:
                        text = d['text']
                        if len(text) == 1 and text.isdigit():
                            digit = int(text)

                            # Calculate which inch value this Y position suggests
                            # Use the lowest detected marker as reference
                            ref_marker = markers[0]
                            pixels_above_ref = ref_marker.y - d['y']
                            inches_above_ref = pixels_above_ref / estimated_ppi
                            expected_inch = ref_marker.inch_value + inches_above_ref

                            # Find the closest valid marker value
                            possible_values = []
                            if digit == 8:
                                possible_values = [8, 18]
                            elif digit == 6:
                                possible_values = [6, 16]
                            elif digit == 4:
                                possible_values = [4, 14]
                            elif digit == 2:
                                possible_values = [2, 12]
                            elif digit == 0:
                                possible_values = [10]
                            else:
                                continue

                            # Pick the value closest to expected
                            best_value = min(possible_values,
                                           key=lambda v: abs(v - expected_inch))

                            if best_value in self.MARKER_INCHES:
                                if not any(m.inch_value == best_value for m in markers):
                                    markers.append(DetectedMarker(
                                        inch_value=best_value,
                                        x=d['x'],
                                        y=d['y'],
                                        confidence=d['conf'] * 0.8,  # Slightly lower confidence
                                        method='easyocr_disambig'
                                    ))
            else:
                # Not enough reference markers - use simple mapping
                for d in detected:
                    text = d['text']
                    if len(text) == 1:
                        # For single digits without context, prefer the lower value
                        # (less likely to be covered by snow)
                        simple_map = {'8': 8, '6': 6, '4': 4, '2': 2, '0': 10}
                        inch_value = simple_map.get(text)
                        if inch_value and inch_value in self.MARKER_INCHES:
                            if not any(m.inch_value == inch_value for m in markers):
                                markers.append(DetectedMarker(
                                    inch_value=inch_value,
                                    x=d['x'],
                                    y=d['y'],
                                    confidence=d['conf'] * 0.6,
                                    method='easyocr_simple'
                                ))

        except Exception as e:
            pass

        return markers


class AutoCalibrationManager:
    """Manages auto-calibration for multiple resorts."""

    # Resort-specific calibrators
    CALIBRATORS = {
        'snowmass': SnowmassCalibrator,
        'winter_park': WinterParkCalibrator
    }

    def __init__(self, db_path: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize manager.

        Args:
            db_path: Path to database
            config_path: Path to resort calibrations JSON
        """
        from .db import SnowDatabase

        if db_path is None:
            db_path = 'snow_measurements.db'
        self.db = SnowDatabase(db_path)
        self.config_path = config_path or 'resort_calibrations.json'
        self.base_config = self._load_config()

    def _load_config(self) -> Dict:
        """Load base calibration config."""
        if os.path.exists(self.config_path):
            with open(self.config_path) as f:
                return json.load(f)
        return {'resorts': []}

    def get_resort_config(self, resort: str) -> Optional[Dict]:
        """Get base config for a resort."""
        for r in self.base_config.get('resorts', []):
            if r['resort'] == resort:
                return r
        return None

    def calibrate_for_day(
        self,
        resort: str,
        image_path: str,
        date_str: Optional[str] = None
    ) -> AutoCalibrationResult:
        """Run auto-calibration for a specific day.

        Args:
            resort: Resort name
            image_path: Path to a clear image from that day
            date_str: Date string (YYYY-MM-DD), defaults to today

        Returns:
            AutoCalibrationResult
        """
        if date_str is None:
            date_str = date.today().isoformat()

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return AutoCalibrationResult(
                success=False,
                pixels_per_inch=None,
                tilt_angle=None,
                reference_y=None,
                stake_centerline_x=None,
                confidence=0.0,
                markers_detected=[],
                method='failed',
                error=f'Could not load image: {image_path}'
            )

        # Get base config for fallback values
        base_config = self.get_resort_config(resort)

        # Get the appropriate calibrator
        calibrator_class = self.CALIBRATORS.get(resort)
        if calibrator_class is None:
            # Use generic calibrator (just use fallback for now)
            if base_config:
                return AutoCalibrationResult(
                    success=True,
                    pixels_per_inch=base_config.get('pixels_per_inch'),
                    tilt_angle=base_config.get('tilt_angle'),
                    reference_y=base_config.get('reference_y'),
                    stake_centerline_x=base_config.get('stake_centerline_x'),
                    confidence=0.5,
                    markers_detected=[],
                    method='config_fallback',
                    error=f'No auto-calibrator for {resort}, using config'
                )
            return AutoCalibrationResult(
                success=False,
                pixels_per_inch=None,
                tilt_angle=None,
                reference_y=None,
                stake_centerline_x=None,
                confidence=0.0,
                markers_detected=[],
                method='failed',
                error=f'No calibrator or config for {resort}'
            )

        # Get stake region from config
        stake_region = None
        if base_config:
            if all(base_config.get(k) for k in ['stake_region_x', 'stake_region_y',
                                                  'stake_region_width', 'stake_region_height']):
                stake_region = (
                    base_config['stake_region_x'],
                    base_config['stake_region_y'],
                    base_config['stake_region_width'],
                    base_config['stake_region_height']
                )

        # Run calibration
        calibrator = calibrator_class(stake_region)
        result = calibrator.calibrate_from_image(image, base_config)

        # Save to database
        if result.success:
            markers_json = json.dumps([
                {
                    'inch_value': m.inch_value,
                    'x': m.x,
                    'y': m.y,
                    'confidence': m.confidence,
                    'method': m.method
                }
                for m in result.markers_detected
            ])

            self.db.save_daily_calibration(
                resort=resort,
                date=date_str,
                pixels_per_inch=result.pixels_per_inch,
                tilt_angle=result.tilt_angle,
                reference_y=result.reference_y,
                stake_centerline_x=result.stake_centerline_x,
                detection_confidence=result.confidence,
                markers_detected=markers_json,
                source_image_path=image_path,
                calibration_method=result.method
            )

        return result

    def get_calibration_for_date(
        self,
        resort: str,
        date_str: str
    ) -> Dict[str, Any]:
        """Get calibration values for a specific date.

        Tries daily calibration first, falls back to base config.

        Args:
            resort: Resort name
            date_str: Date string (YYYY-MM-DD)

        Returns:
            Calibration dictionary
        """
        # Try daily calibration first
        daily_cal = self.db.get_daily_calibration(resort, date_str)
        if daily_cal and daily_cal.get('pixels_per_inch'):
            return {
                'pixels_per_inch': daily_cal['pixels_per_inch'],
                'tilt_angle': daily_cal.get('tilt_angle'),
                'reference_y': daily_cal.get('reference_y'),
                'stake_centerline_x': daily_cal.get('stake_centerline_x'),
                'calibration_source': 'daily_auto',
                'calibration_date': date_str,
                'calibration_confidence': daily_cal.get('detection_confidence', 0.0)
            }

        # Fall back to base config
        base_config = self.get_resort_config(resort)
        if base_config:
            return {
                'pixels_per_inch': base_config.get('pixels_per_inch'),
                'tilt_angle': base_config.get('tilt_angle'),
                'reference_y': base_config.get('reference_y'),
                'stake_centerline_x': base_config.get('stake_centerline_x'),
                'calibration_source': 'base_config',
                'calibration_date': None,
                'calibration_confidence': 1.0  # Manual config is trusted
            }

        return {}


def run_daily_calibration(
    resort: str,
    images_dir: str,
    date_str: Optional[str] = None,
    prefer_hour: int = 12
) -> AutoCalibrationResult:
    """Run daily calibration using the best image from the day.

    Args:
        resort: Resort name
        images_dir: Directory containing images
        date_str: Date to calibrate (YYYY-MM-DD), defaults to today
        prefer_hour: Preferred hour for calibration (noon has good lighting)

    Returns:
        AutoCalibrationResult
    """
    from pathlib import Path

    if date_str is None:
        date_str = date.today().isoformat()

    # Find images for this resort and date
    date_pattern = date_str.replace('-', '')
    pattern = f"{resort}_{date_pattern}_*.png"

    images = sorted(Path(images_dir).glob(pattern))

    if not images:
        # Try jpg
        pattern = f"{resort}_{date_pattern}_*.jpg"
        images = sorted(Path(images_dir).glob(pattern))

    if not images:
        return AutoCalibrationResult(
            success=False,
            pixels_per_inch=None,
            tilt_angle=None,
            reference_y=None,
            stake_centerline_x=None,
            confidence=0.0,
            markers_detected=[],
            method='failed',
            error=f'No images found for {resort} on {date_str}'
        )

    # Prefer image from around noon (good lighting)
    best_image = None
    best_hour_diff = 24

    for img_path in images:
        try:
            # Parse hour from filename: resort_YYYYMMDD_HHMMSS.ext
            parts = img_path.stem.split('_')
            if len(parts) >= 3:
                time_str = parts[2]
                hour = int(time_str[:2])
                hour_diff = abs(hour - prefer_hour)
                if hour_diff < best_hour_diff:
                    best_hour_diff = hour_diff
                    best_image = img_path
        except:
            continue

    if best_image is None:
        best_image = images[len(images) // 2]  # Middle of the day

    # Run calibration
    manager = AutoCalibrationManager()
    return manager.calibrate_for_day(resort, str(best_image), date_str)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Auto-calibrate snow stake')
    parser.add_argument('resort', help='Resort name')
    parser.add_argument('--image', help='Specific image to use')
    parser.add_argument('--images-dir', default='.', help='Images directory')
    parser.add_argument('--date', help='Date (YYYY-MM-DD)')

    args = parser.parse_args()

    if args.image:
        manager = AutoCalibrationManager()
        result = manager.calibrate_for_day(args.resort, args.image, args.date)
    else:
        result = run_daily_calibration(args.resort, args.images_dir, args.date)

    print(f"Auto-calibration result for {args.resort}:")
    print(f"  Success: {result.success}")
    print(f"  Method: {result.method}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Pixels/inch: {result.pixels_per_inch}")
    print(f"  Tilt angle: {result.tilt_angle}")
    print(f"  Reference Y: {result.reference_y}")
    print(f"  Markers detected: {len(result.markers_detected)}")
    for m in result.markers_detected:
        print(f"    {m.inch_value}\" at ({m.x}, {m.y}) - {m.method}")
    if result.error:
        print(f"  Error: {result.error}")
