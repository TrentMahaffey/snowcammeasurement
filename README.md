# Snow Cam Measurement

Computer vision library for measuring snow depth from webcam images of snow stakes.

## Features

- **Snow Depth Measurement**: Automatically detect snow stakes and measure snow depth using computer vision
- **Multiple Calibration Methods**: OCR-based, tilt-aware, and manual calibration options
- **Database Storage**: SQLite-based storage for measurements and calibrations
- **Analytics**: Calculate accumulation, daily/storm totals, and generate reports
- **Easy Integration**: Simple helper functions for integrating into existing capture scripts

## Installation

```bash
pip install snowcammeasurement
```

For OCR-based calibration support:
```bash
pip install snowcammeasurement[ocr]
```

For all features including scipy for improved filtering:
```bash
pip install snowcammeasurement[full]
```

## Quick Start

### Integrating into a Capture Script

```python
from snowcammeasurement import measure_and_store

# After capturing and saving your image:
measure_and_store('resort_name', 'path/to/image.jpg', verbose=True)
```

### Direct Measurement

```python
from snowcammeasurement import SnowStakeMeasurer, CalibrationManager

# Load calibration
manager = CalibrationManager()
calibration = manager.get_calibration_dict('resort_name')

# Measure snow depth
measurer = SnowStakeMeasurer(
    pixels_per_inch=calibration['pixels_per_inch'],
    stake_region=(x, y, width, height)
)
result = measurer.measure_from_file('image.jpg', calibration)

print(f"Snow depth: {result.snow_depth_inches} inches")
print(f"Confidence: {result.confidence_score}")
```

### Auto-Calibration

```python
from snowcammeasurement import quick_calibrate

# Calibrate from an image with visible stake markers
calibration = quick_calibrate('clear_daytime_image.jpg')
print(f"Pixels per inch: {calibration['pixels_per_inch']}")
```

### Analytics

```python
from snowcammeasurement import SnowAnalytics, generate_report

analytics = SnowAnalytics()

# Get 7-day summary
summary = analytics.get_accumulation_summary('resort_name', days=7)
print(f"Total accumulation: {summary['total_accumulation_inches']} inches")

# Generate a text report
report = generate_report('resort_name', days=7)
print(report)
```

## Configuration

Calibrations are stored in `resort_calibrations.json`:

```json
{
  "resorts": [
    {
      "resort": "snowmass",
      "pixels_per_inch": 27.375,
      "reference_y": 988,
      "stake_centerline_x": 675,
      "stake_region_x": 455,
      "stake_region_y": 467,
      "stake_region_width": 440,
      "stake_region_height": 521,
      "enabled": true,
      "min_depth_threshold": 1.0
    }
  ]
}
```

## Calibration Methods

### 1. OCR-Based Calibration

Reads numbered markers on the stake using OCR:

```python
from snowcammeasurement import StakeOCRCalibrator

calibrator = StakeOCRCalibrator(debug=True)
calibration = calibrator.auto_calibrate('image.jpg')
```

### 2. Tilt-Aware Calibration

Accounts for camera tilt by analyzing text orientation:

```python
from snowcammeasurement import TiltAwareCalibrator

calibrator = TiltAwareCalibrator(debug=True)
result = calibrator.calibrate('image.jpg')
print(f"Tilt angle: {result.tilt_angle} degrees")
```

### 3. Manual Calibration

Interactive tool using clicks on known points:

```bash
python -m snowcammeasurement.manual_calibrate resort_name --image image.jpg
```

## Database Schema

Measurements are stored in SQLite with the following tables:

- `snow_measurements`: Individual depth measurements
- `measurement_samples`: Per-sample data from multi-sample measurement
- `calibration_data`: Resort calibration settings
- `daily_calibrations`: Auto-detected daily calibration values

## License

MIT
