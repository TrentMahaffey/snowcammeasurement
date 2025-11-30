"""
Helper functions for integrating snow measurements into capture scripts.

Add these imports to your capture script:
    from snowcammeasurement import measure_and_store

Then call after saving your image:
    measure_and_store(resort_name, image_path, timestamp)
"""

import os
from datetime import datetime
from typing import Optional

from .db import SnowDatabase
from .measurement import SnowStakeMeasurer
from .config import CalibrationManager


# Global instances (initialized once)
_db = None
_config_manager = None


def _get_db(db_path: Optional[str] = None) -> SnowDatabase:
    """Get or create database instance."""
    global _db
    if _db is None:
        _db = SnowDatabase(db_path)
    return _db


def _get_config_manager(config_path: Optional[str] = None, db_path: Optional[str] = None) -> CalibrationManager:
    """Get or create config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = CalibrationManager(config_path, db_path)
    return _config_manager


def measure_and_store(
    resort: str,
    image_path: str,
    timestamp: Optional[datetime] = None,
    verbose: bool = False,
    config_path: Optional[str] = None,
    db_path: Optional[str] = None
) -> bool:
    """Measure snow depth from an image and store in database.

    This is a convenience function for easy integration into capture scripts.

    Args:
        resort: Resort name (should match calibration config)
        image_path: Path to the captured image
        timestamp: Image timestamp (defaults to now)
        verbose: Print measurement results
        config_path: Optional path to calibration config file
        db_path: Optional path to database file

    Returns:
        True if measurement was successful, False otherwise

    Example:
        ```python
        # In your capture script, after saving the image:
        from snowcammeasurement import measure_and_store

        # ... your existing capture code ...
        image_path = save_image(...)

        # Add snow measurement
        measure_and_store('abasin', image_path, datetime.now())
        ```
    """
    if timestamp is None:
        timestamp = datetime.now()

    try:
        # Get config and check if enabled
        # Pass timestamp to get calibration effective at that time
        config_manager = _get_config_manager(config_path, db_path)
        calibration = config_manager.get_calibration(resort, timestamp)

        if not calibration:
            if verbose:
                print(f"Snow measurement skipped: No calibration for {resort}")
            return False

        if not calibration.enabled:
            if verbose:
                print(f"Snow measurement skipped: Disabled for {resort}")
            return False

        # Initialize measurer
        measurer = SnowStakeMeasurer(
            pixels_per_inch=calibration.pixels_per_inch,
            stake_region=calibration.get_stake_region(),
            debug=False
        )

        # Get calibration dict (for timestamp-specific config)
        cal_dict = config_manager.get_calibration_dict(resort, timestamp)

        # Measure
        result = measurer.measure_from_file(image_path, cal_dict)

        # Store in database
        db = _get_db(db_path)
        db.insert_measurement(
            resort=resort,
            timestamp=timestamp,
            image_path=image_path,
            snow_depth_inches=result.snow_depth_inches,
            confidence_score=result.confidence_score,
            stake_visible=result.stake_visible,
            raw_pixel_measurement=result.raw_pixel_measurement,
            notes=result.notes
        )

        if verbose:
            if result.snow_depth_inches is not None:
                print(f"Snow measurement: {result.snow_depth_inches:.1f} inches "
                      f"(confidence: {result.confidence_score:.2f})")
            else:
                print(f"Snow measurement: {result.notes}")

        return True

    except Exception as e:
        if verbose:
            print(f"Snow measurement error: {e}")
        return False


def quick_measure(image_path: str, pixels_per_inch: float = 10.0) -> Optional[float]:
    """Quick measurement without database storage (for testing).

    Args:
        image_path: Path to image
        pixels_per_inch: Calibration ratio

    Returns:
        Snow depth in inches or None
    """
    try:
        measurer = SnowStakeMeasurer(
            pixels_per_inch=pixels_per_inch,
            debug=True
        )
        result = measurer.measure_from_file(image_path)

        if result.stake_visible:
            print(f"Stake visible: {result.stake_visible}")
            print(f"Raw pixels: {result.raw_pixel_measurement}")
            print(f"Snow depth: {result.snow_depth_inches} inches")
            print(f"Confidence: {result.confidence_score:.2f}")
        else:
            print("No stake detected")

        return result.snow_depth_inches

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Quick test mode
        image_path = sys.argv[1]
        pixels_per_inch = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

        print(f"Testing measurement on {image_path}")
        print(f"Using {pixels_per_inch} pixels per inch\n")

        quick_measure(image_path, pixels_per_inch)
    else:
        print("""
To integrate snow measurement into your capture script:

1. Add this import at the top:
   from snowcammeasurement import measure_and_store

2. After saving your image, add:
   measure_and_store('resort_name', image_path, timestamp, verbose=True)

That's it! The function will:
- Check if calibration exists and is enabled
- Measure snow depth using computer vision
- Store the measurement in the database
- Handle errors gracefully (won't crash your capture script)
        """)
