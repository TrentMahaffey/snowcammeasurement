# Snow Cam Measurement

Computer vision library for measuring snow depth from webcam images of snow stakes.

## What This Does

- Measures snow depth from snow stake images using OpenCV
- Per-resort calibration configs (each resort can have different measurement methods)
- Supports linear pixels_per_inch OR marker_positions for lens-distorted cameras
- Stores measurements in SQLite database
- Web frontend for viewing measurements with debug overlays

## Architecture

```
snowcammeasurement/
├── pyproject.toml              # Package config (pip installable)
├── snowcammeasurement/
│   ├── __init__.py             # Public API exports
│   ├── measurement.py          # Core CV measurement (SnowStakeMeasurer)
│   ├── db.py                   # SQLite database (SnowDatabase)
│   ├── config.py               # Calibration config (CalibrationManager)
│   ├── analytics.py            # Stats and reporting (SnowAnalytics)
│   ├── integration.py          # Helper functions (measure_and_store)
│   ├── frontend.py             # Flask web UI with debug overlays
│   ├── auto_calibrate.py       # Auto calibration per-resort
│   ├── ocr_calibrate.py        # OCR-based calibration
│   ├── tilt_calibrate.py       # Tilt-aware calibration
│   └── manual_calibrate.py     # CLI manual calibration
├── resorts/                    # Per-resort calibration configs
│   ├── __init__.py             # Resort loader utilities
│   ├── snowmass/
│   │   ├── calibration.json    # Linear pixels_per_inch method
│   │   └── measurer.py         # Uses base SnowStakeMeasurer
│   └── winter_park/
│       ├── calibration.json    # marker_positions method (for lens distortion)
│       └── measurer.py         # Custom WinterParkMeasurer with interpolation
├── data/                       # Shared data location
└── debug_images/               # Debug/test images
```

## Per-Resort Calibration

Each resort has its own folder in `resorts/` with:
- `calibration.json` - Camera calibration parameters
- `measurer.py` - Custom measurer if needed (or uses base measurer)

### Calibration Methods

1. **Linear (Snowmass)**: Uses `pixels_per_inch` - works when camera has minimal distortion
2. **Marker Interpolation (Winter Park)**: Uses `marker_positions` dict mapping inch values to Y pixel positions - required when lens distortion makes linear scaling inaccurate

### calibration.json Example (Winter Park)

```json
{
    "resort": "winter_park",
    "enabled": true,
    "method": "marker_interpolation",
    "tilt_angle": 0.0,
    "stake_centerline_x": 815,
    "stake_region_x": 780,
    "stake_region_y": 280,
    "stake_region_width": 70,
    "stake_region_height": 540,
    "min_depth_threshold": 1.0,
    "marker_positions": {
        "0": 800,
        "2": 745,
        "4": 690,
        "6": 635,
        "8": 580,
        "10": 525,
        "12": 470,
        "14": 415,
        "16": 360,
        "18": 305
    }
}
```

## Running the Frontend

```bash
# Docker (current setup)
docker run -d --name snow_frontend \
  -v /home/trent/snowcammeasurement:/app:ro \
  -v /home/trent/snowcamtimelapse/out:/out \
  -p 5001:5000 \
  --restart unless-stopped \
  python:3.11-slim \
  bash -c "pip install -q flask opencv-python-headless scipy && python3 -u /app/snowcammeasurement/frontend.py"
```

Frontend reads from:
- `/app/resorts/*/calibration.json` - per-resort calibration configs
- `/out/snow_measurements.db` - measurement database
- `/out/*.jpg` or `/out/*.png` - captured images (from snowcamtimelapse)

### Frontend Features

- View measurements by resort and date
- Navigate through hourly images
- **Grid button**: Toggle coordinate grid overlay for calibration debugging
- Shows inch marker lines based on calibration (linear or marker_positions)
- Sample line visualization showing where measurements were taken
- Outlier detection and stake-cleared event marking

## Calibrating a New Resort

1. Create folder: `resorts/<resort_name>/`
2. Get a sample image and use the Grid button on frontend to find coordinates
3. Create `calibration.json` with:
   - `stake_region_x/y/width/height`: Bounding box around the stake
   - `stake_centerline_x`: X coordinate of stake center
   - Either `pixels_per_inch` + `reference_y` (linear method)
   - Or `marker_positions` dict (for distorted cameras)
4. Optionally create custom `measurer.py` if special logic needed

## Quick Usage

```python
from resorts import get_measurer

# Get resort-specific measurer
measurer = get_measurer('winter_park')
result = measurer.measure_from_file('/path/to/image.jpg')

print(f"Snow depth: {result.snow_depth_inches}")
```

## Related

- Images are captured by the `snowcamtimelapse` library
- Images shared via volume mount at `/out`
- Database at `/out/snow_measurements.db`
