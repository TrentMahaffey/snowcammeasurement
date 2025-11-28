"""
Per-resort measurement configurations and custom measurers.

Each resort has its own folder with:
- calibration.json: Camera calibration parameters
- measurer.py: Custom measurer if needed (or uses base measurer)

Usage:
    from resorts.winter_park.measurer import get_measurer
    measurer = get_measurer()
    result = measurer.measure_from_file('image.jpg')
"""

import os
import json
from typing import Dict, Any, List


def get_resort_list() -> List[str]:
    """Get list of configured resorts."""
    resorts_dir = os.path.dirname(__file__)
    resorts = []
    for name in os.listdir(resorts_dir):
        resort_path = os.path.join(resorts_dir, name)
        if os.path.isdir(resort_path) and not name.startswith('_'):
            calibration_file = os.path.join(resort_path, 'calibration.json')
            if os.path.exists(calibration_file):
                resorts.append(name)
    return sorted(resorts)


def load_resort_calibration(resort: str) -> Dict[str, Any]:
    """Load calibration for a specific resort."""
    calibration_path = os.path.join(os.path.dirname(__file__), resort, 'calibration.json')
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f"No calibration found for resort: {resort}")
    with open(calibration_path) as f:
        return json.load(f)


def get_measurer(resort: str, debug: bool = False):
    """
    Get the appropriate measurer for a resort.

    Returns a resort-specific measurer if one exists, otherwise
    returns the base SnowStakeMeasurer with the resort's calibration.
    """
    calibration = load_resort_calibration(resort)

    # Check if resort has a custom measurer
    measurer_path = os.path.join(os.path.dirname(__file__), resort, 'measurer.py')
    if os.path.exists(measurer_path):
        # Import the resort's custom measurer
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"{resort}_measurer", measurer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'get_measurer'):
            return module.get_measurer(debug=debug)

    # Fallback to base measurer
    from snowcammeasurement.measurement import SnowStakeMeasurer

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
