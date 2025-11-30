"""
Configuration management for snow depth measurement calibrations.
"""

import json
import os
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from .db import SnowDatabase


@dataclass
class ResortCalibration:
    """Calibration settings for a resort's snow stake."""
    resort: str
    pixels_per_inch: float
    stake_region_x: Optional[int] = None
    stake_region_y: Optional[int] = None
    stake_region_width: Optional[int] = None
    stake_region_height: Optional[int] = None
    stake_base_y: Optional[int] = None
    stake_top_y: Optional[int] = None
    reference_x: Optional[int] = None  # X coordinate of 0" reference point
    reference_y: Optional[int] = None  # Y coordinate of 0" reference point
    stake_centerline_x: Optional[int] = None  # X coordinate of stake centerline
    tilt_angle: Optional[float] = None  # Camera tilt in degrees
    marker_region_x: Optional[int] = None  # X coordinate for drawing inch markers (separate from sampling)
    marker_region_width: Optional[int] = None  # Width for inch marker lines
    marker_positions: Optional[dict] = None  # Explicit Y positions for each inch marker
    base_region_x: Optional[int] = None  # X coordinate of the full sign base (outer box)
    base_region_y: Optional[int] = None  # Y coordinate of the full sign base
    base_region_width: Optional[int] = None  # Width of the full sign base
    base_region_height: Optional[int] = None  # Height of the full sign base
    reference_height_inches: Optional[float] = None
    reference_image_path: Optional[str] = None
    min_depth_threshold: float = 1.0  # Minimum depth in inches to report (filters glare/reflections)
    enabled: bool = True
    notes: str = ""

    def get_stake_region(self) -> Optional[tuple]:
        """Get stake region as tuple (x, y, width, height)."""
        if all([
            self.stake_region_x is not None,
            self.stake_region_y is not None,
            self.stake_region_width is not None,
            self.stake_region_height is not None
        ]):
            return (
                self.stake_region_x,
                self.stake_region_y,
                self.stake_region_width,
                self.stake_region_height
            )
        return None


class CalibrationManager:
    """Manages calibration configurations for multiple resorts."""

    def __init__(
        self,
        config_path: str = None,
        db_path: str = None
    ):
        """Initialize calibration manager.

        Args:
            config_path: Path to JSON config file
            db_path: Path to SQLite database
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                'resort_calibrations.json'
            )
        self.config_path = config_path
        self.db = SnowDatabase(db_path)
        self.calibrations: Dict[str, ResortCalibration] = {}
        self._load_config()

    def _load_config(self):
        """Load calibrations from JSON file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                for resort_data in data.get('resorts', []):
                    cal = ResortCalibration(**resort_data)
                    self.calibrations[cal.resort] = cal

    def save_config(self):
        """Save calibrations to JSON file."""
        data = {
            'resorts': [
                asdict(cal) for cal in self.calibrations.values()
            ]
        }
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_calibration(
        self,
        resort: str,
        pixels_per_inch: float,
        stake_region: Optional[tuple] = None,
        reference_y: Optional[int] = None,
        reference_height_inches: Optional[float] = None,
        reference_image_path: Optional[str] = None,
        notes: str = ""
    ):
        """Add or update calibration for a resort.

        Args:
            resort: Resort name
            pixels_per_inch: Calibration ratio
            stake_region: Optional (x, y, width, height) tuple
            reference_y: Y-coordinate of reference point (zero/base)
            reference_height_inches: Height of reference marking (e.g., lowest visible marking)
            reference_image_path: Path to reference image
            notes: Optional notes
        """
        cal = ResortCalibration(
            resort=resort,
            pixels_per_inch=pixels_per_inch,
            reference_y=reference_y,
            reference_height_inches=reference_height_inches,
            reference_image_path=reference_image_path,
            notes=notes
        )

        if stake_region:
            cal.stake_region_x = stake_region[0]
            cal.stake_region_y = stake_region[1]
            cal.stake_region_width = stake_region[2]
            cal.stake_region_height = stake_region[3]

        self.calibrations[resort] = cal

        # Also save to database
        self.db.set_calibration(
            resort=resort,
            pixels_per_inch=pixels_per_inch,
            reference_y=reference_y,
            reference_height_inches=reference_height_inches,
            reference_image_path=reference_image_path,
            notes=notes
        )

        self.save_config()

    def get_calibration(self, resort: str, timestamp: Optional[Any] = None) -> Optional[ResortCalibration]:
        """Get calibration for a resort, optionally for a specific timestamp.

        Checks database for time-based calibration versions first,
        then falls back to file-based configs.

        Args:
            resort: Resort name
            timestamp: Optional datetime to get calibration effective at that time

        Returns:
            ResortCalibration object or None
        """
        from datetime import datetime as dt

        # First check database for time-based calibration
        try:
            if timestamp:
                db_config = self.db.get_calibration_for_timestamp(resort, timestamp)
            else:
                db_config = self.db.get_current_calibration_version(resort)

            if db_config:
                # Convert dict to ResortCalibration
                return self._dict_to_calibration(resort, db_config)
        except Exception:
            pass  # Fall through to file-based config

        return self.calibrations.get(resort)

    def _dict_to_calibration(self, resort: str, config: Dict[str, Any]) -> ResortCalibration:
        """Convert a config dictionary to a ResortCalibration object."""
        return ResortCalibration(
            resort=resort,
            pixels_per_inch=config.get('pixels_per_inch', 0),
            stake_region_x=config.get('stake_region_x'),
            stake_region_y=config.get('stake_region_y'),
            stake_region_width=config.get('stake_region_width'),
            stake_region_height=config.get('stake_region_height'),
            stake_base_y=config.get('stake_base_y'),
            stake_top_y=config.get('stake_top_y'),
            reference_x=config.get('reference_x'),
            reference_y=config.get('reference_y'),
            stake_centerline_x=config.get('stake_centerline_x'),
            tilt_angle=config.get('tilt_angle'),
            marker_region_x=config.get('marker_region_x'),
            marker_region_width=config.get('marker_region_width'),
            marker_positions=config.get('marker_positions'),
            base_region_x=config.get('base_region_x'),
            base_region_y=config.get('base_region_y'),
            base_region_width=config.get('base_region_width'),
            base_region_height=config.get('base_region_height'),
            reference_height_inches=config.get('reference_height_inches'),
            reference_image_path=config.get('reference_image_path'),
            min_depth_threshold=config.get('min_depth_threshold', 1.0),
            enabled=config.get('enabled', True),
            notes=config.get('notes', '')
        )

    def get_calibration_dict(self, resort: str, timestamp: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Get calibration as dictionary (for compatibility).

        Args:
            resort: Resort name
            timestamp: Optional datetime to get calibration effective at that time

        Returns:
            Dictionary with calibration data or None
        """
        # First check database directly for time-based calibration
        try:
            if timestamp:
                db_config = self.db.get_calibration_for_timestamp(resort, timestamp)
            else:
                db_config = self.db.get_current_calibration_version(resort)

            if db_config:
                return db_config
        except Exception:
            pass

        cal = self.get_calibration(resort, timestamp)
        if cal:
            result = {
                'pixels_per_inch': cal.pixels_per_inch,
                'reference_image_path': cal.reference_image_path,
                'notes': cal.notes
            }
            # Stake region
            if cal.stake_region_x is not None:
                result['stake_region_x'] = cal.stake_region_x
            if cal.stake_region_y is not None:
                result['stake_region_y'] = cal.stake_region_y
            if cal.stake_region_width is not None:
                result['stake_region_width'] = cal.stake_region_width
            if cal.stake_region_height is not None:
                result['stake_region_height'] = cal.stake_region_height
            # Legacy fields
            if cal.stake_base_y is not None:
                result['stake_base_y'] = cal.stake_base_y
            if cal.stake_top_y is not None:
                result['stake_top_y'] = cal.stake_top_y
            # New tilt-aware fields
            if cal.reference_x is not None:
                result['reference_x'] = cal.reference_x
            if cal.reference_y is not None:
                result['reference_y'] = cal.reference_y
            if cal.stake_centerline_x is not None:
                result['stake_centerline_x'] = cal.stake_centerline_x
            if cal.tilt_angle is not None:
                result['tilt_angle'] = cal.tilt_angle
            if cal.reference_height_inches is not None:
                result['reference_height_inches'] = cal.reference_height_inches
            # Base region fields (for separate sampling region from stake outline)
            if cal.base_region_x is not None:
                result['base_region_x'] = cal.base_region_x
            if cal.base_region_y is not None:
                result['base_region_y'] = cal.base_region_y
            if cal.base_region_width is not None:
                result['base_region_width'] = cal.base_region_width
            if cal.base_region_height is not None:
                result['base_region_height'] = cal.base_region_height
            # Marker position fields
            if cal.marker_region_x is not None:
                result['marker_region_x'] = cal.marker_region_x
            if cal.marker_region_width is not None:
                result['marker_region_width'] = cal.marker_region_width
            if cal.marker_positions is not None:
                result['marker_positions'] = cal.marker_positions
            result['min_depth_threshold'] = cal.min_depth_threshold
            return result
        return None

    def enable_resort(self, resort: str):
        """Enable measurements for a resort."""
        if resort in self.calibrations:
            self.calibrations[resort].enabled = True
            self.save_config()

    def disable_resort(self, resort: str):
        """Disable measurements for a resort."""
        if resort in self.calibrations:
            self.calibrations[resort].enabled = False
            self.save_config()

    def is_enabled(self, resort: str) -> bool:
        """Check if measurements are enabled for a resort."""
        cal = self.calibrations.get(resort)
        return cal.enabled if cal else False

    def list_resorts(self) -> list:
        """Get list of all calibrated resorts."""
        return list(self.calibrations.keys())

    def list_enabled_resorts(self) -> list:
        """Get list of resorts with measurements enabled."""
        return [
            resort for resort, cal in self.calibrations.items()
            if cal.enabled
        ]


def initialize_default_config(config_path: str = None):
    """Create a default configuration file with examples."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            'resort_calibrations.json'
        )

    if os.path.exists(config_path):
        print(f"Config file already exists: {config_path}")
        return

    default_config = {
        "resorts": [
            {
                "resort": "example_resort",
                "pixels_per_inch": 10.0,
                "stake_region_x": None,
                "stake_region_y": None,
                "stake_region_width": None,
                "stake_region_height": None,
                "stake_base_y": None,
                "stake_top_y": None,
                "reference_image_path": None,
                "enabled": False,
                "notes": "Calibration needed - set pixels_per_inch after measuring"
            }
        ]
    }

    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)

    print(f"Created default config file: {config_path}")


if __name__ == "__main__":
    initialize_default_config()
