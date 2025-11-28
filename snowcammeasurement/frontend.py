#!/usr/bin/env python3
"""
Web frontend for snow depth measurements.
Displays images with calibration overlays and estimated snow levels.
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template_string, send_file, request, jsonify
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Configuration
OUT_DIR = os.environ.get('OUT_DIR', '/out')
DB_PATH = os.environ.get('DB_PATH', '/out/snow_measurements.db')
CONFIG_PATH = os.environ.get('CONFIG_PATH', '/out/resort_calibrations.json')

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Snow Depth Measurements</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #0f3460;
        }
        .header h1 { color: #e94560; margin-bottom: 10px; }
        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        .controls select, .controls input, .controls button {
            padding: 10px 15px;
            border-radius: 5px;
            border: 1px solid #0f3460;
            background: #1a1a2e;
            color: #eee;
            font-size: 14px;
        }
        .controls button {
            background: #e94560;
            cursor: pointer;
            font-weight: bold;
        }
        .controls button:hover { background: #ff6b6b; }
        .main-content {
            display: flex;
            flex-wrap: wrap;
            padding: 20px;
            gap: 20px;
        }
        .image-section {
            flex: 2;
            min-width: 600px;
        }
        .stats-section {
            flex: 1;
            min-width: 300px;
        }
        .image-container {
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            position: relative;
        }
        .image-container img {
            width: 100%;
            border-radius: 5px;
        }
        .measurement-overlay {
            position: absolute;
            top: 25px;
            right: 25px;
            background: rgba(0,0,0,0.8);
            padding: 15px 20px;
            border-radius: 10px;
            border: 2px solid #e94560;
        }
        .measurement-value {
            font-size: 48px;
            font-weight: bold;
            color: #00ff88;
        }
        .measurement-label {
            font-size: 14px;
            color: #aaa;
        }
        .timestamp {
            margin-top: 10px;
            font-size: 12px;
            color: #888;
        }
        .stats-card {
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
        }
        .stats-card h3 {
            color: #e94560;
            margin-bottom: 15px;
            border-bottom: 1px solid #0f3460;
            padding-bottom: 10px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #0f3460;
        }
        .stat-row:last-child { border-bottom: none; }
        .stat-value { color: #00ff88; font-weight: bold; }
        .timeline {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 10px;
        }
        .timeline-item {
            width: 40px;
            height: 40px;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .timeline-item:hover { transform: scale(1.2); }
        .timeline-item.active { border: 2px solid #fff; }
        .no-data { background: #333; color: #666; }
        .low { background: #1a472a; }
        .medium { background: #2d5a27; }
        .high { background: #4a7c23; }
        .very-high { background: #6b9b1f; }
        .outlier { background: #8b0000; border: 1px solid #ff4444; }
        .stake-cleared { background: #4a4a00; border: 1px solid #ffff00; }
        .measurement-id {
            font-size: 11px;
            color: #888;
            margin-bottom: 5px;
            font-family: monospace;
        }
        .outlier-warning {
            margin-top: 8px;
            padding: 5px 10px;
            background: rgba(255, 0, 0, 0.3);
            border-radius: 5px;
            font-size: 12px;
            color: #ff6666;
        }
        .stake-cleared-warning {
            background: rgba(255, 255, 0, 0.2);
            color: #ffff66;
        }
        .nav-buttons {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            justify-content: center;
        }
        .nav-btn {
            padding: 10px 20px;
            background: #0f3460;
            border: none;
            border-radius: 5px;
            color: #eee;
            cursor: pointer;
            font-size: 14px;
        }
        .nav-btn:hover { background: #1a4a7a; }
        .nav-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .nav-btn.active { background: #e94560; }
        .grid-btn { background: #2d4a3e; }
        .grid-btn:hover { background: #3d5a4e; }
        .grid-btn.active { background: #4ade80; color: #000; }
        .loading {
            text-align: center;
            padding: 50px;
            color: #888;
        }
        .sample-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .sample-table th, .sample-table td {
            padding: 6px 8px;
            border-bottom: 1px solid #0f3460;
            text-align: left;
        }
        .sample-table th {
            color: #aaa;
            font-weight: normal;
        }
        .sample-table td.depth { color: #00ff88; font-weight: bold; }
        .sample-table td.contrast { color: #88ccff; }
        .sample-table td.valid { color: #00ff88; }
        .sample-table td.invalid { color: #ff6666; }
        .sample-table td.skip { color: #ff9944; font-size: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>❄️ Snow Depth Measurements</h1>
        <div class="controls">
            <select id="resort" onchange="loadData()">
                {% for r in resorts %}
                <option value="{{ r }}" {% if r == resort %}selected{% endif %}>{{ r.title() }}</option>
                {% endfor %}
            </select>
            <input type="date" id="date" value="{{ date }}" onchange="loadData()">
            <button onclick="loadData()">Refresh</button>
        </div>
    </div>

    <div class="main-content">
        <div class="image-section">
            <div class="image-container">
                <img id="main-image" src="/image/{{ resort }}/{{ current_image }}" alt="Snow measurement">
                <div class="measurement-overlay">
                    <div class="measurement-id" id="measurement-id">ID: {{ current_id }}</div>
                    <div class="measurement-value" id="depth-value">{{ current_depth }}</div>
                    <div class="measurement-label">inches of snow</div>
                    <div class="timestamp" id="timestamp">{{ current_time }}</div>
                    {% if current_is_outlier %}
                    <div class="outlier-warning" id="outlier-warning">Outlier: {{ current_outlier_reason }}</div>
                    {% elif current_outlier_reason == 'stake_cleared' %}
                    <div class="outlier-warning stake-cleared-warning" id="outlier-warning">Stake cleared/reset</div>
                    {% else %}
                    <div class="outlier-warning" id="outlier-warning" style="display: none;"></div>
                    {% endif %}
                </div>
            </div>
            <div class="nav-buttons">
                <button class="nav-btn" onclick="prevImage()">← Previous Hour</button>
                <button class="nav-btn" onclick="nextImage()">Next Hour →</button>
                <button class="nav-btn grid-btn" id="grid-btn" onclick="toggleGrid()">📐 Grid</button>
            </div>
        </div>

        <div class="stats-section">
            <div class="stats-card">
                <h3>📊 Daily Summary (Hourly Avg)</h3>
                <div id="daily-summary">
                    <div class="stat-row">
                        <span>Loading...</span>
                    </div>
                </div>
            </div>

            <div class="stats-card">
                <h3>🕐 Hourly Timeline</h3>
                <div class="timeline" id="timeline">
                    {% for m in measurements %}
                    <div class="timeline-item {{ m.class }}"
                         onclick="selectMeasurement({{ loop.index0 }})"
                         title="{{ m.time }}: {{ m.depth }}"
                         {% if loop.index0 == current_index %}style="border: 2px solid #fff;"{% endif %}>
                        {{ m.hour }}
                    </div>
                    {% endfor %}
                </div>
            </div>

            <div class="stats-card">
                <h3>📏 Sample Lines (1-10)</h3>
                <div id="sample-table">
                    <table class="sample-table">
                        <thead>
                            <tr><th>#</th><th>Depth</th><th>Contrast</th><th>Status</th></tr>
                        </thead>
                        <tbody id="sample-rows">
                            <tr><td colspan="4" style="color:#888">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="stats-card">
                <h3>⚙️ Calibration</h3>
                <div class="stat-row">
                    <span>Pixels/inch</span>
                    <span class="stat-value">{{ calibration.ppi }}</span>
                </div>
                <div class="stat-row">
                    <span>Reference Y</span>
                    <span class="stat-value">{{ calibration.ref_y }}</span>
                </div>
                <div class="stat-row">
                    <span>Tilt Angle</span>
                    <span class="stat-value">{{ calibration.tilt }}°</span>
                </div>
                <div class="stat-row">
                    <span>Stake Region</span>
                    <span class="stat-value">{{ calibration.region }}</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let measurements = {{ measurements_json | safe }};
        let currentIndex = {{ current_index }};
        let resort = '{{ resort }}';
        let minDepthThreshold = {{ calibration.min_depth_threshold }};
        let showGrid = false;

        function toggleGrid() {
            showGrid = !showGrid;
            const btn = document.getElementById('grid-btn');
            if (showGrid) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
            updateDisplay();
        }

        function selectMeasurement(idx) {
            if (idx < 0 || idx >= measurements.length) return;
            currentIndex = idx;
            updateDisplay();
        }

        function prevImage() {
            if (currentIndex > 0) {
                currentIndex--;
                updateDisplay();
            }
        }

        function nextImage() {
            if (currentIndex < measurements.length - 1) {
                currentIndex++;
                updateDisplay();
            }
        }

        function updateDisplay() {
            const m = measurements[currentIndex];
            let imgUrl = '/image/' + resort + '/' + m.image + '?t=' + Date.now();
            if (showGrid) imgUrl += '&grid=true';
            document.getElementById('main-image').src = imgUrl;
            document.getElementById('depth-value').textContent = m.depth;
            document.getElementById('timestamp').textContent = m.time;
            document.getElementById('measurement-id').textContent = 'ID: ' + m.id;

            // Handle outlier warning
            const warningEl = document.getElementById('outlier-warning');
            if (m.is_outlier) {
                warningEl.textContent = 'Outlier: ' + m.outlier_reason;
                warningEl.style.display = 'block';
                warningEl.className = 'outlier-warning';
            } else if (m.outlier_reason === 'stake_cleared') {
                warningEl.textContent = 'Stake cleared/reset';
                warningEl.style.display = 'block';
                warningEl.className = 'outlier-warning stake-cleared-warning';
            } else {
                warningEl.style.display = 'none';
            }

            // Update timeline selection
            document.querySelectorAll('.timeline-item').forEach((el, idx) => {
                el.style.border = idx === currentIndex ? '2px solid #fff' : 'none';
            });

            // Load sample data for this measurement
            loadSampleData(m.id);
        }

        function loadSampleData(measurementId) {
            fetch('/api/samples/' + measurementId)
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('sample-rows');
                    if (!data.samples || data.samples.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="4" style="color:#888">No sample data</td></tr>';
                        return;
                    }
                    let html = '';
                    let validCount = 0;
                    data.samples.forEach((s, i) => {
                        const num = i + 1;
                        const depth = s.depth_inches !== null ? s.depth_inches.toFixed(2) + '"' : '-';
                        const contrast = s.contrast !== null ? s.contrast.toFixed(0) : '-';
                        const valid = s.valid;
                        const skip = s.skip_reason || '';

                        if (valid) validCount++;

                        let statusClass = valid ? 'valid' : 'invalid';
                        let statusText = valid ? '✓' : '✗';
                        if (skip) {
                            statusClass = 'skip';
                            statusText = skip.replace('_', ' ');
                        }

                        html += '<tr>';
                        html += '<td>' + num + '</td>';
                        html += '<td class="depth">' + depth + '</td>';
                        html += '<td class="contrast">' + contrast + '</td>';
                        html += '<td class="' + statusClass + '">' + statusText + '</td>';
                        html += '</tr>';
                    });

                    // Add summary row with average of valid samples
                    html += '<tr style="border-top:2px solid #0f3460;background:#1a1a2e">';
                    html += '<td colspan="2" style="text-align:right;font-weight:bold">Avg (' + validCount + ' valid):</td>';
                    if (data.depth_avg !== null && data.depth_avg !== undefined) {
                        // Check if below minimum threshold (per-resort config)
                        if (data.depth_avg < minDepthThreshold) {
                            html += '<td style="font-weight:bold;color:#f87171">' + data.depth_avg.toFixed(2) + '"</td>';
                            html += '<td style="color:#f87171;font-size:10px">below ' + minDepthThreshold + '" threshold</td>';
                        } else {
                            html += '<td colspan="2" style="font-weight:bold;color:#4ade80">' + data.depth_avg.toFixed(2) + '"</td>';
                        }
                    } else {
                        html += '<td colspan="2" style="color:#888">-</td>';
                    }
                    html += '</tr>';

                    tbody.innerHTML = html;
                })
                .catch(err => {
                    document.getElementById('sample-rows').innerHTML = '<tr><td colspan="4" style="color:#ff6666">Error loading</td></tr>';
                });
        }

        function loadData() {
            const resort = document.getElementById('resort').value;
            const date = document.getElementById('date').value;
            window.location.href = '/?resort=' + resort + '&date=' + date;
        }

        function loadDailySummary() {
            const date = document.getElementById('date').value;
            fetch('/api/daily_summary/' + resort + '/' + date)
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('daily-summary');
                    if (data.error) {
                        container.innerHTML = '<div class="stat-row"><span style="color:#888">' + data.error + '</span></div>';
                        return;
                    }

                    let html = '';

                    // Accumulation (sum of positive deltas) - most important stat
                    if (data.accumulation_inches !== undefined) {
                        html += '<div class="stat-row" style="background:#1a3a1a;border-radius:5px;padding:5px">';
                        html += '<span style="font-weight:bold">Day Accumulation</span>';
                        html += '<span class="stat-value" style="color:#4ade80;font-size:1.2em">+' + data.accumulation_inches + '"</span>';
                        html += '</div>';
                    }

                    // Day max with time
                    if (data.day_max) {
                        html += '<div class="stat-row">';
                        html += '<span>Day Max</span>';
                        html += '<span class="stat-value">' + data.day_max.depth_inches + '" @ ' + data.day_max.latest_time + '</span>';
                        html += '</div>';
                    }

                    // Day min with time
                    if (data.day_min) {
                        html += '<div class="stat-row">';
                        html += '<span>Day Min</span>';
                        html += '<span class="stat-value">' + data.day_min.depth_inches + '" @ ' + data.day_min.latest_time + '</span>';
                        html += '</div>';
                    }

                    // Hours with data
                    html += '<div class="stat-row">';
                    html += '<span>Hours with data</span>';
                    html += '<span class="stat-value">' + data.hours_with_data + '</span>';
                    html += '</div>';

                    // Hourly breakdown (collapsed by default) with deltas
                    if (data.hourly_readings && data.hourly_readings.length > 0) {
                        html += '<details style="margin-top:10px">';
                        html += '<summary style="cursor:pointer;color:#e94560">Hourly Breakdown</summary>';
                        html += '<table class="sample-table" style="margin-top:5px;font-size:12px">';
                        html += '<thead><tr><th>Hour</th><th>Depth</th><th>Change</th></tr></thead>';
                        html += '<tbody>';
                        data.hourly_readings.forEach(h => {
                            let deltaStr = '-';
                            let deltaStyle = '';
                            if (h.delta_inches !== null) {
                                if (h.delta_inches > 0) {
                                    deltaStr = '+' + h.delta_inches + '"';
                                    deltaStyle = 'color:#4ade80';  // green for positive
                                } else if (h.delta_inches < 0) {
                                    deltaStr = h.delta_inches + '"';
                                    deltaStyle = 'color:#f87171';  // red for negative
                                } else {
                                    deltaStr = '0"';
                                }
                            }
                            html += '<tr>';
                            html += '<td>' + h.hour + '</td>';
                            html += '<td>' + h.avg_depth_inches + '"</td>';
                            html += '<td style="' + deltaStyle + '">' + deltaStr + '</td>';
                            html += '</tr>';
                        });
                        html += '</tbody></table>';
                        html += '</details>';
                    }

                    container.innerHTML = html;
                })
                .catch(err => {
                    document.getElementById('daily-summary').innerHTML = '<div class="stat-row"><span style="color:#ff6666">Error loading summary</span></div>';
                });
        }

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft') prevImage();
            if (e.key === 'ArrowRight') nextImage();
        });

        // Load data on page load
        if (measurements.length > 0 && measurements[currentIndex]) {
            loadSampleData(measurements[currentIndex].id);
        }
        loadDailySummary();
    </script>
</body>
</html>
'''


def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_calibration(resort):
    """Load calibration for a resort from per-resort folder structure."""
    # First try new per-resort folder structure
    resorts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resorts')
    resort_calibration_path = os.path.join(resorts_dir, resort, 'calibration.json')

    if os.path.exists(resort_calibration_path):
        try:
            with open(resort_calibration_path) as f:
                return json.load(f)
        except:
            pass

    # Fallback to old combined config file
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        for r in config.get('resorts', []):
            if r['resort'] == resort:
                return r
    except:
        pass
    return {}


def get_resorts():
    """Get list of configured resorts from per-resort folder structure."""
    resorts = []

    # First check new per-resort folder structure
    resorts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resorts')
    if os.path.exists(resorts_dir):
        for name in os.listdir(resorts_dir):
            resort_path = os.path.join(resorts_dir, name)
            if os.path.isdir(resort_path) and not name.startswith('_'):
                calibration_file = os.path.join(resort_path, 'calibration.json')
                if os.path.exists(calibration_file):
                    resorts.append(name)

    # Fallback/merge with old config file
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        for r in config.get('resorts', []):
            if r['resort'] not in resorts:
                resorts.append(r['resort'])
    except:
        pass

    return sorted(resorts) if resorts else ['snowmass']


def get_measurements(resort, date_str):
    """Get measurements for a resort on a specific date."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, timestamp, snow_depth_inches, confidence_score, image_path
        FROM snow_measurements
        WHERE resort = ? AND date(timestamp) = ?
        ORDER BY timestamp
    ''', (resort, date_str))

    rows = cursor.fetchall()
    conn.close()

    measurements = []
    prev_depth = None
    MAX_HOURLY_CHANGE = 4.0  # Max inches change per hour before flagging as outlier

    for row in rows:
        measurement_id = row['id']
        ts = datetime.fromisoformat(row['timestamp'])
        depth = row['snow_depth_inches']

        # Outlier detection
        is_outlier = False
        outlier_reason = None

        if depth is not None and prev_depth is not None:
            change = abs(depth - prev_depth)
            if change > MAX_HOURLY_CHANGE:
                # Check if this could be a stake clearing event
                if depth < 1.0 and prev_depth > 3.0:
                    outlier_reason = "stake_cleared"
                else:
                    is_outlier = True
                    outlier_reason = f"spike (+{change:.1f}\")" if depth > prev_depth else f"drop (-{change:.1f}\")"

        # Determine color class based on depth and outlier status
        if depth is None:
            css_class = 'no-data'
            depth_str = 'N/A'
        elif is_outlier:
            css_class = 'outlier'
            depth_str = f"{depth:.1f}\" ⚠️"
        elif depth < 2:
            css_class = 'low'
            depth_str = f"{depth:.1f}\""
        elif depth < 5:
            css_class = 'medium'
            depth_str = f"{depth:.1f}\""
        elif depth < 10:
            css_class = 'high'
            depth_str = f"{depth:.1f}\""
        else:
            css_class = 'very-high'
            depth_str = f"{depth:.1f}\""

        # Get image filename (strip /out/ prefix if present)
        image_path = row['image_path'] or f"{resort}_{ts.strftime('%Y%m%d_%H0000')}.png"
        if image_path.startswith('/out/'):
            image_path = image_path[5:]  # Remove /out/ prefix

        measurements.append({
            'id': measurement_id,
            'time': ts.strftime('%Y-%m-%d %H:%M'),
            'hour': ts.strftime('%H'),
            'depth': depth_str,
            'depth_num': depth if depth is not None else 0,
            'confidence': row['confidence_score'],
            'image': image_path,
            'class': css_class,
            'is_outlier': is_outlier,
            'outlier_reason': outlier_reason
        })

        # Update prev_depth for next iteration (only if not an outlier)
        if depth is not None and not is_outlier:
            prev_depth = depth

    return measurements


def calculate_stats(measurements):
    """Calculate statistics from measurements."""
    depths = [m['depth_num'] for m in measurements if m['depth_num'] > 0]

    if not depths:
        return {'min': 'N/A', 'max': 'N/A', 'avg': 'N/A', 'count': 0}

    return {
        'min': f"{min(depths):.1f}",
        'max': f"{max(depths):.1f}",
        'avg': f"{sum(depths)/len(depths):.1f}",
        'count': len(depths)
    }


@app.route('/')
def index():
    """Main page."""
    resort = request.args.get('resort', 'snowmass')
    date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))

    resorts = get_resorts()
    measurements = get_measurements(resort, date_str)
    calibration = get_calibration(resort)
    stats = calculate_stats(measurements)

    # Find current index (latest with data, or middle)
    current_index = len(measurements) - 1 if measurements else 0
    for i, m in enumerate(measurements):
        if m['depth_num'] > 0:
            current_index = i

    # Format calibration for display
    cal_display = {
        'ppi': f"{calibration.get('pixels_per_inch', 0):.2f}",
        'ref_y': calibration.get('reference_y', 'N/A'),
        'tilt': f"{calibration.get('tilt_angle', 0):.1f}",
        'region': f"{calibration.get('stake_region_x', '?')},{calibration.get('stake_region_y', '?')}",
        'min_depth_threshold': calibration.get('min_depth_threshold', 1.0)
    }

    current_measurement = measurements[current_index] if measurements else {
        'id': 'N/A', 'image': 'none.png', 'depth': 'N/A', 'time': 'No data',
        'is_outlier': False, 'outlier_reason': None
    }

    return render_template_string(
        HTML_TEMPLATE,
        resort=resort,
        resorts=resorts,
        date=date_str,
        measurements=measurements,
        measurements_json=json.dumps(measurements),
        stats=stats,
        calibration=cal_display,
        current_index=current_index,
        current_image=current_measurement.get('image', 'none.png'),
        current_depth=current_measurement.get('depth', 'N/A'),
        current_time=current_measurement.get('time', ''),
        current_id=current_measurement.get('id', 'N/A'),
        current_is_outlier=current_measurement.get('is_outlier', False),
        current_outlier_reason=current_measurement.get('outlier_reason')
    )


@app.route('/image/<resort>/<path:filename>')
def serve_image(resort, filename):
    """Serve an image with optional calibration overlay."""
    import glob

    # Try to find the image
    image_path = None

    # Handle /out/ prefix from database paths
    if filename.startswith('/out/'):
        filename = filename[5:]  # Remove /out/ prefix

    # Check various locations
    paths_to_try = [
        filename,  # As-is
        os.path.join(OUT_DIR, filename),  # In OUT_DIR
        os.path.join(OUT_DIR, os.path.basename(filename)),  # Just basename in OUT_DIR
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            image_path = path
            break

    # If still not found, try pattern matching
    if not image_path:
        base = os.path.basename(filename).split('.')[0]
        pattern = os.path.join(OUT_DIR, f"{base}*.png")
        matches = glob.glob(pattern)
        if matches:
            image_path = matches[0]

    if not image_path or not os.path.exists(image_path):
        # Return a placeholder
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(img, "Image not found", (700, 540),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        _, buffer = cv2.imencode('.jpg', img)
        return send_file(BytesIO(buffer), mimetype='image/jpeg')

    # Look up snow depth and sample data for this image from database
    snow_depth = None
    sample_data = None
    depth_min = None
    depth_max = None
    depth_avg = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Try to match by image path
        cursor.execute('''
            SELECT snow_depth_inches, sample_data, depth_min, depth_max, depth_avg
            FROM snow_measurements
            WHERE resort = ? AND (image_path LIKE ? OR image_path LIKE ?)
            ORDER BY timestamp DESC LIMIT 1
        ''', (resort, f'%{os.path.basename(filename)}', f'%{filename}'))
        row = cursor.fetchone()
        if row:
            snow_depth = row['snow_depth_inches']
            depth_min = row['depth_min']
            depth_max = row['depth_max']
            depth_avg = row['depth_avg']
            if row['sample_data']:
                sample_data = json.loads(row['sample_data'])
        conn.close()
    except Exception as e:
        pass  # Sample data might not exist yet

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return "Image load failed", 404

    # Get calibration
    calibration = get_calibration(resort)

    # Draw coordinate grid for calibration debugging
    show_grid = request.args.get('grid', 'false').lower() == 'true'
    if show_grid:
        img_h, img_w = image.shape[:2]
        # Vertical lines (X coordinates) every 100 pixels
        for gx in range(0, img_w, 100):
            color = (0, 0, 255) if gx % 200 == 0 else (0, 255, 0)
            thickness = 2 if gx % 200 == 0 else 1
            cv2.line(image, (gx, 0), (gx, img_h), color, thickness)
            cv2.putText(image, str(gx), (gx + 3, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        # Horizontal lines (Y coordinates) every 100 pixels
        for gy in range(100, img_h, 100):
            color = (255, 0, 0) if gy % 200 == 0 else (255, 255, 0)
            thickness = 2 if gy % 200 == 0 else 1
            cv2.line(image, (0, gy), (img_w, gy), color, thickness)
            cv2.putText(image, str(gy), (5, gy - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Draw calibration overlay if available
    if calibration:
        import math

        # Draw stake region
        x = calibration.get('stake_region_x')
        y = calibration.get('stake_region_y')
        w = calibration.get('stake_region_width')
        h = calibration.get('stake_region_height')
        centerline_x = calibration.get('stake_centerline_x')
        tilt_angle = calibration.get('tilt_angle', 0.0)

        # Calculate tilt offset per pixel of vertical distance
        tilt_rad = math.radians(tilt_angle)
        dx_per_dy = math.tan(tilt_rad)  # horizontal shift per vertical pixel

        if all([x, y, w, h]):
            # Draw full stake region (outer box, dimmer) - adjusted for tilt
            # Top corners shift based on tilt
            top_shift = int((h) * dx_per_dy)
            pts_outer = np.array([
                [x + top_shift, y],
                [x + w + top_shift, y],
                [x + w, y + h],
                [x, y + h]
            ], np.int32)
            cv2.polylines(image, [pts_outer], True, (128, 64, 0), 1)

            # Draw measurement region (75% width, centered on centerline)
            if centerline_x:
                measure_width = int(w * 0.75)
                measure_x = centerline_x - measure_width // 2
                pts_measure = np.array([
                    [measure_x + top_shift, y],
                    [measure_x + measure_width + top_shift, y],
                    [measure_x + measure_width, y + h],
                    [measure_x, y + h]
                ], np.int32)
                cv2.polylines(image, [pts_measure], True, (255, 100, 0), 2)

                # Draw 10 vertical sample lines (matching the measurement algorithm)
                # Each line stops at its own detected snow line
                num_samples = 10
                ref_y = calibration.get('reference_y')
                ppi = calibration.get('pixels_per_inch', 27.0)

                # If we have sample data from database, use individual snow lines
                if sample_data and len(sample_data) == num_samples:
                    for i, sample in enumerate(sample_data):
                        sample_x = sample.get('x_position', measure_x + int(measure_width * i / (num_samples - 1)))
                        sample_snow_y = sample.get('snow_line_y')
                        sample_depth = sample.get('depth_inches')
                        sample_valid = sample.get('valid', False)
                        sample_skip = sample.get('skip_reason')

                        # Default to reference_y (base) if no snow line detected
                        if sample_snow_y is None:
                            sample_snow_y = ref_y if ref_y else y + h

                        # Choose color based on validity
                        if sample_valid:
                            line_color = (50, 200, 255)  # Light orange for valid
                        elif sample_skip:
                            line_color = (50, 50, 200)   # Red for skipped
                        else:
                            line_color = (100, 100, 100) # Gray for no data

                        # Calculate tilt-adjusted line
                        if sample_snow_y > y:
                            t = (sample_snow_y - y) / h if h > 0 else 1
                            line_top_shift = int(h * dx_per_dy)
                            line_x_at_snow = int(sample_x + line_top_shift * (1 - t))
                            line_x_at_top = sample_x + line_top_shift

                            # Draw line from reference (base) UP to snow line
                            cv2.line(image,
                                (sample_x, ref_y) if ref_y else (sample_x, y + h),
                                (line_x_at_snow, sample_snow_y),
                                line_color, 2)

                            # Draw sample number label at base
                            num_label = str(i + 1)
                            num_x = sample_x - 5
                            num_y = (ref_y if ref_y else y + h) + 18
                            cv2.putText(image, num_label, (num_x, num_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                            # Draw small depth label at snow line (only for valid samples)
                            if sample_depth is not None and sample_valid:
                                label_text = f"{sample_depth:.1f}"
                                label_x = line_x_at_snow - 15
                                label_y = sample_snow_y - 8
                                # Draw small background
                                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                                cv2.rectangle(image, (label_x - 2, label_y - th - 2),
                                             (label_x + tw + 2, label_y + 4), (0, 0, 0), -1)
                                cv2.putText(image, label_text,
                                    (label_x, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)
                else:
                    # Fallback: use single snow_depth for all lines
                    if snow_depth is not None and ref_y and ppi:
                        snow_line_y = int(ref_y - (snow_depth * ppi))
                    else:
                        snow_line_y = ref_y if ref_y else y + h

                    for i in range(num_samples):
                        sample_x = measure_x + int(measure_width * i / (num_samples - 1))
                        if snow_line_y > y:
                            t = (snow_line_y - y) / h if h > 0 else 1
                            line_top_shift = int(h * dx_per_dy)
                            line_x_at_snow = int(sample_x + line_top_shift * (1 - t))
                            # Draw line from base to snow line
                            cv2.line(image,
                                (sample_x, ref_y) if ref_y else (sample_x, y + h),
                                (line_x_at_snow, snow_line_y),
                                (255, 150, 50), 1)

                # Draw summary stats if available
                if depth_min is not None and depth_max is not None and depth_avg is not None:
                    stats_text = f"Min:{depth_min:.1f}\" Avg:{depth_avg:.1f}\" Max:{depth_max:.1f}\""
                    cv2.putText(image, stats_text, (x, y - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 200, 100), 2)
            else:
                cv2.polylines(image, [pts_outer], True, (255, 100, 0), 2)

        # Draw reference line and inch markers
        marker_positions = calibration.get('marker_positions', {})
        ref_y = calibration.get('reference_y')
        ppi = calibration.get('pixels_per_inch', 29.33)

        # Helper function to draw text with background
        def draw_label(img, text, pos, color, font_scale=1.0):
            font = cv2.FONT_HERSHEY_DUPLEX  # Nicer font
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            tx, ty = pos
            # Draw semi-transparent background
            overlay = img.copy()
            padding = 6
            cv2.rectangle(overlay,
                (tx - padding, ty - th - padding),
                (tx + tw + padding, ty + baseline + padding),
                (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            # Draw text with outline for better visibility
            cv2.putText(img, text, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2)  # Outline
            cv2.putText(img, text, (tx, ty), font, font_scale, color, thickness)  # Main text

        if x and w:
            # For tilted lines: left point and right point have different Y values
            # dy across the width = w * tan(tilt)
            dy_across_width = int(w * dx_per_dy)

            if marker_positions:
                # Use marker_positions for non-linear (distorted) cameras
                # Draw lines at actual marker Y positions from calibration
                for inch_str, mark_y in marker_positions.items():
                    inch = int(inch_str)
                    mark_y_right = mark_y + dy_across_width

                    if inch == 0:
                        # Base/reference line (0") in green
                        cv2.line(image, (x, mark_y), (x + w, mark_y_right), (0, 255, 0), 3)
                        draw_label(image, "0\"", (x + w + 12, mark_y_right + 8), (0, 255, 0), 1.1)
                    elif inch % 4 == 0:
                        # Draw every 4 inches in yellow with label
                        if mark_y > y:
                            cv2.line(image, (x, mark_y), (x + w, mark_y_right), (0, 255, 255), 2)
                            draw_label(image, f'{inch}"', (x + w + 12, mark_y_right + 8), (0, 255, 255), 0.9)
                    else:
                        # Draw every 2 inches with smaller label
                        if mark_y > y:
                            cv2.line(image, (x, mark_y), (x + w, mark_y_right), (0, 200, 200), 1)
                            draw_label(image, f'{inch}"', (x + w + 12, mark_y_right + 6), (0, 200, 200), 0.6)

            elif ref_y:
                # Use linear pixels_per_inch (for cameras without distortion)
                # Reference line (0") - tilted
                cv2.line(image, (x, ref_y), (x + w, ref_y + dy_across_width), (0, 255, 0), 3)
                draw_label(image, "0\"", (x + w + 12, ref_y + dy_across_width + 8), (0, 255, 0), 1.1)

                # Draw inch markers (tilted) - every 2 inches
                for inch in range(2, 20, 2):
                    mark_y_left = int(ref_y - (inch * ppi))
                    mark_y_right = mark_y_left + dy_across_width
                    if mark_y_left > y:
                        if inch % 4 == 0:
                            # Every 4 inches - thicker yellow line with larger label
                            cv2.line(image, (x, mark_y_left), (x + w, mark_y_right), (0, 255, 255), 2)
                            draw_label(image, f'{inch}"', (x + w + 12, mark_y_right + 8), (0, 255, 255), 0.9)
                        else:
                            # Every 2 inches - thinner cyan line with smaller label
                            cv2.line(image, (x, mark_y_left), (x + w, mark_y_right), (0, 200, 200), 1)
                            draw_label(image, f'{inch}"', (x + w + 12, mark_y_right + 6), (0, 200, 200), 0.6)

    # Encode and return
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return send_file(BytesIO(buffer), mimetype='image/jpeg')


@app.route('/api/measurements/<resort>/<date>')
def api_measurements(resort, date):
    """API endpoint for measurements."""
    measurements = get_measurements(resort, date)
    stats = calculate_stats(measurements)
    return jsonify({
        'measurements': measurements,
        'stats': stats
    })


@app.route('/api/samples/<int:measurement_id>')
def api_samples(measurement_id):
    """API endpoint for sample data for a specific measurement."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT sample_data, depth_min, depth_max, depth_avg, snow_depth_inches
            FROM snow_measurements WHERE id = ?
        ''', (measurement_id,))
        row = cursor.fetchone()
        conn.close()

        if row and row['sample_data']:
            samples = json.loads(row['sample_data'])
            return jsonify({
                'samples': samples,
                'depth_min': row['depth_min'],
                'depth_max': row['depth_max'],
                'depth_avg': row['depth_avg'],
                'snow_depth_inches': row['snow_depth_inches']
            })
        else:
            return jsonify({'samples': []})
    except Exception as e:
        return jsonify({'samples': [], 'error': str(e)})


@app.route('/api/daily_summary/<resort>/<date>')
def api_daily_summary(resort, date):
    """API endpoint for daily summary with hourly averages and min/max times."""
    try:
        from snow_analytics import SnowAnalytics
        analytics = SnowAnalytics(DB_PATH)

        # Parse date
        date_obj = datetime.strptime(date, '%Y-%m-%d')

        summary = analytics.get_daily_summary(resort, date_obj)
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    print(f"Starting Snow Depth Frontend on port {port}")
    print(f"Database: {DB_PATH}")
    print(f"Images: {OUT_DIR}")
    print(f"Config: {CONFIG_PATH}")

    app.run(host='0.0.0.0', port=port, debug=debug)
