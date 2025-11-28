"""
Snow Cam Measurement - Computer vision library for measuring snow depth from webcam images.

This library provides tools for:
- Measuring snow depth from snow stake images using computer vision
- Calibrating cameras using OCR, tilt detection, or manual methods
- Storing and analyzing snow depth measurements over time
- Generating accumulation reports and statistics

Basic Usage:
    from snowcammeasurement import measure_and_store, SnowStakeMeasurer

    # Quick integration into capture scripts
    measure_and_store('resort_name', 'path/to/image.jpg')

    # Or use the measurer directly
    measurer = SnowStakeMeasurer(pixels_per_inch=27.5)
    result = measurer.measure_from_file('image.jpg', calibration_dict)
    print(f"Snow depth: {result.snow_depth_inches} inches")

Calibration:
    from snowcammeasurement import CalibrationManager, quick_calibrate

    # Load existing calibrations
    manager = CalibrationManager()
    calibration = manager.get_calibration_dict('resort_name')

    # Or auto-calibrate from an image with visible stake markers
    calibration = quick_calibrate('clear_image.jpg')

Analytics:
    from snowcammeasurement import SnowAnalytics, generate_report

    analytics = SnowAnalytics()
    summary = analytics.get_accumulation_summary('resort_name', days=7)
    report = generate_report('resort_name')
"""

__version__ = "0.1.0"

# Core measurement
from .measurement import (
    SnowStakeMeasurer,
    MeasurementResult,
    SampleMeasurement,
    calibrate_from_image,
)

# Database
from .db import SnowDatabase

# Configuration
from .config import (
    CalibrationManager,
    ResortCalibration,
    initialize_default_config,
)

# Integration helpers
from .integration import (
    measure_and_store,
    quick_measure,
)

# Analytics
from .analytics import (
    SnowAnalytics,
    generate_report,
)

# Calibration tools
from .auto_calibrate import (
    AutoCalibrationManager,
    AutoCalibrationResult,
    DetectedMarker,
    BaseStakeCalibrator,
    SnowmassCalibrator,
    WinterParkCalibrator,
    run_daily_calibration,
)

from .ocr_calibrate import (
    StakeOCRCalibrator,
    StakeMarking,
    quick_calibrate,
)

from .tilt_calibrate import (
    TiltAwareCalibrator,
    CalibrationResult as TiltCalibrationResult,
)

__all__ = [
    # Version
    "__version__",

    # Core measurement
    "SnowStakeMeasurer",
    "MeasurementResult",
    "SampleMeasurement",
    "calibrate_from_image",

    # Database
    "SnowDatabase",

    # Configuration
    "CalibrationManager",
    "ResortCalibration",
    "initialize_default_config",

    # Integration
    "measure_and_store",
    "quick_measure",

    # Analytics
    "SnowAnalytics",
    "generate_report",

    # Auto calibration
    "AutoCalibrationManager",
    "AutoCalibrationResult",
    "DetectedMarker",
    "BaseStakeCalibrator",
    "SnowmassCalibrator",
    "WinterParkCalibrator",
    "run_daily_calibration",

    # OCR calibration
    "StakeOCRCalibrator",
    "StakeMarking",
    "quick_calibrate",

    # Tilt calibration
    "TiltAwareCalibrator",
    "TiltCalibrationResult",
]
