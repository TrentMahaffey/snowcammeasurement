"""
Snowmass snow depth measurer using linear pixels_per_inch.

This camera has minimal lens distortion, so the standard linear
pixels_per_inch calibration works well.
"""

import json
import os
from typing import Dict, Any

# Import the base measurer from the core library
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from snowcammeasurement.measurement import SnowStakeMeasurer, MeasurementResult


def load_calibration() -> Dict[str, Any]:
    """Load Snowmass calibration from the calibration.json file."""
    calibration_path = os.path.join(os.path.dirname(__file__), 'calibration.json')
    with open(calibration_path) as f:
        return json.load(f)


def get_measurer(debug: bool = False) -> SnowStakeMeasurer:
    """Get a configured Snowmass measurer instance."""
    calibration = load_calibration()

    measurer = SnowStakeMeasurer(
        pixels_per_inch=calibration.get('pixels_per_inch'),
        stake_region=(
            calibration.get('stake_region_x'),
            calibration.get('stake_region_y'),
            calibration.get('stake_region_width'),
            calibration.get('stake_region_height')
        ),
        debug=debug
    )
    measurer.calibration = calibration

    return measurer


def measure_from_file(image_path: str, debug: bool = False) -> MeasurementResult:
    """Convenience function to measure from a file."""
    calibration = load_calibration()
    measurer = get_measurer(debug=debug)
    return measurer.measure_from_file(image_path, calibration=calibration)
