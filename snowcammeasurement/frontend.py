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
        .calibrate-btn { background: #4a3d5e; }
        .calibrate-btn:hover { background: #5a4d6e; }
        .calibrate-btn.active { background: #a855f7; color: #fff; }
        .calibration-panel {
            position: fixed;
            top: 0;
            right: 0;
            width: 350px;
            height: 100vh;
            background: #16213e;
            border-left: 2px solid #a855f7;
            padding: 20px;
            z-index: 1000;
            overflow-y: auto;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        }
        .calibration-panel.open { transform: translateX(0); }
        .calibration-panel h2 { color: #a855f7; margin-bottom: 15px; }
        .calibration-panel label {
            display: block;
            margin-top: 12px;
            color: #aaa;
            font-size: 13px;
        }
        .calibration-panel select,
        .calibration-panel input {
            width: 100%;
            padding: 8px;
            margin-top: 4px;
            border-radius: 4px;
            border: 1px solid #0f3460;
            background: #1a1a2e;
            color: #eee;
            font-size: 14px;
        }
        .calibration-panel .coord-display {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            text-align: center;
        }
        .calibration-panel .coord-display .coords {
            font-size: 24px;
            font-weight: bold;
            color: #a855f7;
            font-family: monospace;
        }
        .calibration-panel .coord-display .hint {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        .calibration-panel .btn-row {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        .calibration-panel button {
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        .calibration-panel .btn-save {
            background: #a855f7;
            color: #fff;
        }
        .calibration-panel .btn-save:hover { background: #9333ea; }
        .calibration-panel .btn-save:disabled {
            background: #444;
            cursor: not-allowed;
        }
        .calibration-panel .btn-cancel {
            background: #374151;
            color: #eee;
        }
        .calibration-panel .btn-cancel:hover { background: #4b5563; }
        .calibration-panel .history-item {
            background: #1a1a2e;
            padding: 10px;
            margin-top: 8px;
            border-radius: 5px;
            font-size: 12px;
            border-left: 3px solid #a855f7;
        }
        .calibration-panel .history-item .date { color: #a855f7; font-weight: bold; }
        .calibration-panel .history-item .notes { color: #888; margin-top: 4px; }
        .cal-data-table {
            width: 100%;
            margin: 10px 0;
            font-size: 11px;
            border-collapse: collapse;
        }
        .cal-data-table th, .cal-data-table td {
            padding: 4px 6px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        .cal-data-table th { color: #a855f7; font-weight: normal; }
        .cal-data-table td { color: #ccc; }
        .cal-data-table .missing { color: #ef4444; font-style: italic; }
        .cal-data-table .section-header {
            background: #1a1a2e;
            color: #888;
            font-weight: bold;
        }
        .cal-toggle {
            background: none;
            border: none;
            color: #a855f7;
            cursor: pointer;
            font-size: 12px;
            padding: 5px 0;
        }
        .cal-toggle:hover { text-decoration: underline; }
        .cal-section {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
        }
        .cal-section h3 {
            color: #a855f7;
            margin: 0 0 10px 0;
            font-size: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .cal-progress {
            background: #0f0f1a;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 15px;
        }
        .cal-progress-bar {
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin: 8px 0;
        }
        .cal-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #a855f7, #6366f1);
            transition: width 0.3s;
        }
        .cal-progress-text {
            font-size: 12px;
            color: #888;
        }
        .cal-progress-count {
            font-size: 18px;
            font-weight: bold;
            color: #a855f7;
        }
        .cal-checklist {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 4px;
            font-size: 11px;
            margin-top: 10px;
        }
        .cal-check-item {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 3px 6px;
            border-radius: 4px;
            background: #1a1a2e;
        }
        .cal-check-item.set { color: #22c55e; background: rgba(34, 197, 94, 0.1); }
        .cal-check-item.unset { color: #666; }
        .cal-check-icon { font-size: 10px; flex-shrink: 0; }
        .cal-check-label { font-size: 10px; flex-shrink: 0; }
        .cal-check-val { font-size: 9px; color: #888; margin-left: auto; font-family: monospace; }
        .cal-history-table {
            width: 100%;
            font-size: 11px;
            border-collapse: collapse;
        }
        .cal-history-table th, .cal-history-table td {
            padding: 6px 8px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        .cal-history-table th { color: #888; font-weight: normal; }
        .cal-history-table td { color: #ccc; }
        .cal-history-table tr:hover { background: #1a1a2e; }
        .image-container.calibrate-mode { cursor: crosshair; }
        .image-container.calibrate-mode img { pointer-events: none; }
        .click-marker {
            position: absolute;
            width: 20px;
            height: 20px;
            border: 2px solid #a855f7;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
            box-shadow: 0 0 10px #a855f7;
        }
        .click-marker::before,
        .click-marker::after {
            content: '';
            position: absolute;
            background: #a855f7;
        }
        .click-marker::before {
            width: 2px;
            height: 30px;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
        }
        .click-marker::after {
            width: 30px;
            height: 2px;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
        }
        .remeasure-btn { background: #5a3d4e; }
        .remeasure-btn:hover { background: #6a4d5e; }
        .regions-btn { background: #3d5a4e; }
        .regions-btn:hover { background: #4d6a5e; }
        .regions-btn.active { background: #5db86b; }
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0,0,0,0.7);
            z-index: 2000;
            display: none;
            align-items: center;
            justify-content: center;
        }
        .modal-overlay.open { display: flex; }
        .modal {
            background: #16213e;
            border-radius: 10px;
            padding: 25px;
            min-width: 400px;
            max-width: 500px;
            border: 2px solid #e94560;
        }
        .modal h2 { color: #e94560; margin-bottom: 20px; }
        .modal label {
            display: block;
            margin-top: 15px;
            color: #aaa;
            font-size: 13px;
        }
        .modal select, .modal input {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            border-radius: 5px;
            border: 1px solid #0f3460;
            background: #1a1a2e;
            color: #eee;
            font-size: 14px;
        }
        .modal .preview-box {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            text-align: center;
        }
        .modal .preview-count {
            font-size: 32px;
            font-weight: bold;
            color: #e94560;
        }
        .modal .preview-label {
            color: #888;
            font-size: 12px;
            margin-top: 5px;
        }
        .modal .preview-dates {
            color: #aaa;
            font-size: 11px;
            margin-top: 10px;
        }
        .modal .btn-row {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        .modal button {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
        }
        .modal .btn-primary { background: #e94560; color: #fff; }
        .modal .btn-primary:hover { background: #ff6b6b; }
        .modal .btn-primary:disabled { background: #444; cursor: not-allowed; }
        .modal .btn-secondary { background: #374151; color: #eee; }
        .modal .btn-secondary:hover { background: #4b5563; }
        .modal .progress-bar {
            height: 8px;
            background: #0f3460;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 15px;
            display: none;
        }
        .modal .progress-bar.active { display: block; }
        .modal .progress-bar .fill {
            height: 100%;
            background: #e94560;
            width: 0%;
            transition: width 0.3s;
        }
        .modal .status-text {
            font-size: 12px;
            color: #888;
            margin-top: 10px;
            text-align: center;
        }
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
                <div id="click-marker" class="click-marker" style="display: none;"></div>
            </div>
            <div class="nav-buttons">
                <button class="nav-btn" onclick="prevImage()">← Previous Hour</button>
                <button class="nav-btn" onclick="nextImage()">Next Hour →</button>
                <button class="nav-btn grid-btn" id="grid-btn" onclick="toggleGrid()">Grid</button>
                <button class="nav-btn regions-btn" id="regions-btn" onclick="toggleRegions()">Regions</button>
                <button class="nav-btn calibrate-btn" id="calibrate-btn" onclick="toggleCalibrationMode()">Calibrate</button>
                <button class="nav-btn remeasure-btn" onclick="remeasureThis()">Re-measure This</button>
            </div>

            <!-- Calibration Section (below image) -->
            <div class="cal-section" id="cal-section" style="display: none;">
                <h3>
                    <span>Calibration Status</span>
                    <span style="font-size: 11px; color: #888;">ID: <span id="cal-status-id">-</span> | Effective: <span id="cal-status-date">-</span></span>
                </h3>

                <!-- Progress Indicator -->
                <div class="cal-progress">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="cal-progress-count"><span id="cal-set-count">0</span>/<span id="cal-total-count">24</span></span>
                        <span class="cal-progress-text">calibration points set</span>
                    </div>
                    <div class="cal-progress-bar">
                        <div class="cal-progress-fill" id="cal-progress-fill" style="width: 0%;"></div>
                    </div>
                    <div class="cal-checklist" id="cal-checklist"></div>
                </div>

                <!-- History Toggle -->
                <h3 style="margin-top: 15px;">
                    <span>Calibration History</span>
                    <button class="cal-toggle" onclick="toggleCalHistory()">Show/Hide</button>
                </h3>
                <div id="cal-history-container" style="display: none;">
                    <table class="cal-history-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Effective From</th>
                                <th>Created</th>
                                <th>Notes</th>
                            </tr>
                        </thead>
                        <tbody id="cal-history-body"></tbody>
                    </table>
                </div>
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

    <!-- Calibration Panel -->
    <div id="calibration-panel" class="calibration-panel">
        <h2>Calibration</h2>

        <button class="cal-toggle" onclick="toggleCalDataTable()">Show/Hide Current Values</button>
        <div id="cal-data-container" style="display: none;">
            <div style="font-size: 11px; color: #888; margin-bottom: 5px;">
                Calibration ID: <span id="cal-current-id">-</span> |
                Effective: <span id="cal-current-date">-</span>
            </div>
            <table class="cal-data-table" id="cal-data-table">
                <tbody id="cal-data-body"></tbody>
            </table>
            <div id="cal-validation-warning" style="color: #ef4444; font-size: 11px; margin-top: 5px; display: none;"></div>
        </div>

        <label>Property to Set</label>
        <select id="cal-property">
            <option value="">-- Select Property --</option>
            <optgroup label="Stake Corners (click to set X,Y)">
                <option value="stake_corners.top_left">Top Left Corner</option>
                <option value="stake_corners.top_right">Top Right Corner</option>
                <option value="stake_corners.bottom_left">Bottom Left Corner</option>
                <option value="stake_corners.bottom_right">Bottom Right Corner</option>
            </optgroup>
            <optgroup label="Stake Axis (click to set X,Y)">
                <option value="stake_axis.top">Axis Top (18" end)</option>
                <option value="stake_axis.bottom">Axis Bottom (0" end)</option>
            </optgroup>
            <optgroup label="Snow Stake Base (click to set X,Y)">
                <option value="sample_bounds.top_left">Base Top Left</option>
                <option value="sample_bounds.top_right">Base Top Right</option>
                <option value="sample_bounds.bottom_left">Base Bottom Left</option>
                <option value="sample_bounds.bottom_right">Base Bottom Right</option>
            </optgroup>
            <optgroup label="Camera">
                <option value="camera_tilt">Camera Tilt (degrees)</option>
            </optgroup>
            <optgroup label="Marker Positions (Y only)">
                <option value="marker_positions.0">0" Marker Y</option>
                <option value="marker_positions.2">2" Marker Y</option>
                <option value="marker_positions.4">4" Marker Y</option>
                <option value="marker_positions.6">6" Marker Y</option>
                <option value="marker_positions.8">8" Marker Y</option>
                <option value="marker_positions.10">10" Marker Y</option>
                <option value="marker_positions.12">12" Marker Y</option>
                <option value="marker_positions.14">14" Marker Y</option>
                <option value="marker_positions.16">16" Marker Y</option>
                <option value="marker_positions.18">18" Marker Y</option>
            </optgroup>
            <optgroup label="Other">
                <option value="min_depth_threshold">Min Depth Threshold</option>
            </optgroup>
        </select>

        <div class="coord-display">
            <div class="coords" id="cal-coords">Click image</div>
            <div class="hint" id="cal-hint">Select a property, then click on the image</div>
        </div>

        <label>Value</label>
        <input type="number" id="cal-value" placeholder="Click image or enter manually">

        <label>Effective From</label>
        <input type="datetime-local" id="cal-effective-from">

        <label>Notes (optional)</label>
        <input type="text" id="cal-notes" placeholder="e.g., Stake replaced">

        <div class="btn-row">
            <button class="btn-cancel" onclick="cancelCalibration()">Cancel</button>
            <button class="btn-save" id="cal-save-btn" onclick="saveCalibration()" disabled>Save</button>
        </div>
    </div>

    <!-- Re-measure Modal -->
    <div id="remeasure-modal" class="modal-overlay">
        <div class="modal">
            <h2>Re-measure Snow Depths</h2>

            <label>Re-measure Range</label>
            <select id="remeasure-mode" onchange="updateRemeasurePreview()">
                <option value="since_calibration">Since Last Calibration</option>
                <option value="last_n_days">Last N Days</option>
                <option value="date_range">Custom Date Range</option>
                <option value="all">All Measurements</option>
            </select>

            <div id="remeasure-days-row" style="display: none;">
                <label>Number of Days</label>
                <input type="number" id="remeasure-days" value="7" min="1" max="365" onchange="updateRemeasurePreview()">
            </div>

            <div id="remeasure-daterange-row" style="display: none;">
                <label>Start Date</label>
                <input type="date" id="remeasure-start" onchange="updateRemeasurePreview()">
                <label>End Date</label>
                <input type="date" id="remeasure-end" onchange="updateRemeasurePreview()">
            </div>

            <div class="preview-box">
                <div class="preview-count" id="remeasure-count">-</div>
                <div class="preview-label">measurements will be re-processed</div>
                <div class="preview-dates" id="remeasure-dates"></div>
            </div>

            <div class="progress-bar" id="remeasure-progress">
                <div class="fill" id="remeasure-progress-fill"></div>
            </div>
            <div class="status-text" id="remeasure-status"></div>

            <div class="btn-row">
                <button class="btn-secondary" onclick="closeRemeasureModal()">Cancel</button>
                <button class="btn-primary" id="remeasure-submit" onclick="executeRemeasure()">Re-measure</button>
            </div>
        </div>
    </div>

    <script>
        let measurements = {{ measurements_json | safe }};
        let currentIndex = {{ current_index }};
        let resort = '{{ resort }}';
        let minDepthThreshold = {{ calibration.min_depth_threshold }};
        let showGrid = false;
        let showRegions = false;

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

        function toggleRegions() {
            showRegions = !showRegions;
            const btn = document.getElementById('regions-btn');
            if (showRegions) {
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
            if (showRegions) imgUrl += '&regions=true';
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

        // ============ Calibration Mode ============
        let calibrationMode = false;
        let clickedX = null;
        let clickedY = null;
        // Store original image dimensions (will be set when image loads)
        let originalImageWidth = 1920;  // Default, updated on load
        let originalImageHeight = 1080;

        function toggleCalibrationMode() {
            calibrationMode = !calibrationMode;
            const btn = document.getElementById('calibrate-btn');
            const panel = document.getElementById('calibration-panel');
            const container = document.querySelector('.image-container');
            const calSection = document.getElementById('cal-section');

            if (calibrationMode) {
                btn.classList.add('active');
                panel.classList.add('open');
                container.classList.add('calibrate-mode');
                calSection.style.display = 'block';
                // Enable grid for easier calibration
                if (!showGrid) toggleGrid();
                // Set default effective date to current image timestamp
                setDefaultEffectiveDate();
                // Load calibration status and history
                updateCalibrationStatus();
            } else {
                btn.classList.remove('active');
                panel.classList.remove('open');
                container.classList.remove('calibrate-mode');
                calSection.style.display = 'none';
                hideClickMarker();
            }
        }

        function setDefaultEffectiveDate() {
            // Always default to NOW so new calibrations become immediately effective
            // Format as local time for datetime-local input (YYYY-MM-DDTHH:MM)
            const now = new Date();
            const year = now.getFullYear();
            const month = String(now.getMonth() + 1).padStart(2, '0');
            const day = String(now.getDate()).padStart(2, '0');
            const hours = String(now.getHours()).padStart(2, '0');
            const minutes = String(now.getMinutes()).padStart(2, '0');
            document.getElementById('cal-effective-from').value = `${year}-${month}-${day}T${hours}:${minutes}`;
        }

        // Handle clicks on the image container
        document.querySelector('.image-container').addEventListener('click', function(e) {
            if (!calibrationMode) return;

            const img = document.getElementById('main-image');
            const rect = img.getBoundingClientRect();

            // Calculate click position relative to image
            const relX = e.clientX - rect.left;
            const relY = e.clientY - rect.top;

            // Scale to original image dimensions
            const scaleX = originalImageWidth / rect.width;
            const scaleY = originalImageHeight / rect.height;

            clickedX = Math.round(relX * scaleX);
            clickedY = Math.round(relY * scaleY);

            // Update display
            document.getElementById('cal-coords').textContent = `X: ${clickedX}, Y: ${clickedY}`;

            // Show click marker at the clicked position (in display coordinates)
            showClickMarker(relX, relY, rect);

            // Auto-fill value based on selected property type
            const prop = document.getElementById('cal-property').value;
            if (prop) {
                if (prop.startsWith('stake_corners.') || prop.startsWith('stake_axis.') || prop.startsWith('sample_bounds.')) {
                    // Point properties: set [X, Y]
                    document.getElementById('cal-value').value = `${clickedX}, ${clickedY}`;
                } else if (prop.startsWith('marker_positions')) {
                    // Marker positions: Y only
                    document.getElementById('cal-value').value = clickedY;
                } else if (prop.includes('_x')) {
                    document.getElementById('cal-value').value = clickedX;
                } else if (prop.includes('_y')) {
                    document.getElementById('cal-value').value = clickedY;
                }
            }

            updateSaveButtonState();
        });

        function showClickMarker(displayX, displayY, imgRect) {
            const marker = document.getElementById('click-marker');
            const container = document.querySelector('.image-container');
            const containerRect = container.getBoundingClientRect();
            const imgElement = document.getElementById('main-image');
            const imgDisplayRect = imgElement.getBoundingClientRect();

            // Position marker relative to container, accounting for image position within container
            // CSS transform: translate(-50%, -50%) already centers the marker
            const offsetX = imgDisplayRect.left - containerRect.left;
            const offsetY = imgDisplayRect.top - containerRect.top;

            marker.style.left = (offsetX + displayX) + 'px';
            marker.style.top = (offsetY + displayY) + 'px';
            marker.style.display = 'block';
        }

        function hideClickMarker() {
            document.getElementById('click-marker').style.display = 'none';
        }

        function updateSaveButtonState() {
            const prop = document.getElementById('cal-property').value;
            const value = document.getElementById('cal-value').value;
            const effectiveFrom = document.getElementById('cal-effective-from').value;

            const canSave = prop && value && effectiveFrom;
            document.getElementById('cal-save-btn').disabled = !canSave;
        }

        // ============ Calibration Data Table ============
        let currentCalibrationData = null;

        function toggleCalDataTable() {
            const container = document.getElementById('cal-data-container');
            if (container.style.display === 'none') {
                container.style.display = 'block';
                loadCalibrationData();
            } else {
                container.style.display = 'none';
            }
        }

        function loadCalibrationData() {
            fetch('/api/calibration/' + resort + '/current')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        currentCalibrationData = data.calibration;
                        renderCalibrationTable(data.calibration, data.id, data.effective_from);
                        validateCalibration(data.calibration);
                    } else {
                        document.getElementById('cal-data-body').innerHTML =
                            '<tr><td colspan="2" class="missing">No calibration found</td></tr>';
                    }
                })
                .catch(err => {
                    document.getElementById('cal-data-body').innerHTML =
                        '<tr><td colspan="2" class="missing">Error loading calibration</td></tr>';
                });
        }

        function renderCalibrationTable(cal, id, effectiveFrom) {
            document.getElementById('cal-current-id').textContent = id || '-';
            document.getElementById('cal-current-date').textContent = effectiveFrom || '-';

            let html = '';

            // Stake Corners
            html += '<tr class="section-header"><td colspan="2">Stake Corners</td></tr>';
            const corners = cal.stake_corners || {};
            ['top_left', 'top_right', 'bottom_left', 'bottom_right'].forEach(k => {
                const val = corners[k];
                const cls = val ? '' : 'missing';
                html += '<tr><td>' + k.replace('_', ' ') + '</td><td class="' + cls + '">' +
                    (val ? '[' + val.join(', ') + ']' : 'NOT SET') + '</td></tr>';
            });

            // Stake Axis
            html += '<tr class="section-header"><td colspan="2">Stake Axis</td></tr>';
            const axis = cal.stake_axis || {};
            ['top', 'bottom'].forEach(k => {
                const val = axis[k];
                const cls = val ? '' : 'missing';
                html += '<tr><td>' + k + '</td><td class="' + cls + '">' +
                    (val ? '[' + val.join(', ') + ']' : 'NOT SET') + '</td></tr>';
            });

            // Sample Bounds (Base)
            html += '<tr class="section-header"><td colspan="2">Snow Stake Base</td></tr>';
            const bounds = cal.sample_bounds || {};
            ['top_left', 'top_right', 'bottom_left', 'bottom_right'].forEach(k => {
                const val = bounds[k];
                const cls = val ? '' : 'missing';
                html += '<tr><td>' + k.replace('_', ' ') + '</td><td class="' + cls + '">' +
                    (val ? '[' + val.join(', ') + ']' : 'NOT SET') + '</td></tr>';
            });

            // Marker Positions
            html += '<tr class="section-header"><td colspan="2">Marker Positions (Y)</td></tr>';
            const markers = cal.marker_positions || {};
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18].forEach(inch => {
                const val = markers[inch] || markers[String(inch)];
                const cls = val !== undefined ? '' : 'missing';
                html += '<tr><td>' + inch + '"</td><td class="' + cls + '">' +
                    (val !== undefined ? val : 'NOT SET') + '</td></tr>';
            });

            // Other
            html += '<tr class="section-header"><td colspan="2">Other</td></tr>';
            html += '<tr><td>camera_tilt</td><td>' + (cal.camera_tilt || 0) + '°</td></tr>';
            html += '<tr><td>min_depth_threshold</td><td>' + (cal.min_depth_threshold || 1.0) + '"</td></tr>';

            document.getElementById('cal-data-body').innerHTML = html;
        }

        function validateCalibration(cal) {
            const warnings = [];

            // Check stake_corners
            const corners = cal.stake_corners || {};
            const missingCorners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
                .filter(k => !corners[k]);
            if (missingCorners.length > 0 && missingCorners.length < 4) {
                warnings.push('Stake corners incomplete: missing ' + missingCorners.join(', '));
            }

            // Check sample_bounds
            const bounds = cal.sample_bounds || {};
            const missingBounds = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
                .filter(k => !bounds[k]);
            if (missingBounds.length > 0 && missingBounds.length < 4) {
                warnings.push('Base corners incomplete: missing ' + missingBounds.join(', '));
            }

            // Check stake_axis
            const axis = cal.stake_axis || {};
            if ((axis.top && !axis.bottom) || (!axis.top && axis.bottom)) {
                warnings.push('Stake axis incomplete: need both top and bottom');
            }

            const warningEl = document.getElementById('cal-validation-warning');
            if (warnings.length > 0) {
                warningEl.innerHTML = '⚠️ ' + warnings.join('<br>⚠️ ');
                warningEl.style.display = 'block';
            } else {
                warningEl.style.display = 'none';
            }
        }

        // Update save button state when inputs change
        document.getElementById('cal-property').addEventListener('change', function() {
            // Update hint based on property type
            const prop = this.value;
            const hint = document.getElementById('cal-hint');
            const valueInput = document.getElementById('cal-value');
            if (prop.startsWith('stake_corners.') || prop.startsWith('stake_axis.') || prop.startsWith('sample_bounds.')) {
                hint.textContent = 'Click to set X, Y point';
                valueInput.placeholder = 'X, Y (e.g., 815, 300)';
                valueInput.type = 'text';
            } else if (prop.startsWith('marker_positions')) {
                hint.textContent = 'Click to set Y coordinate';
                valueInput.placeholder = 'Y coordinate';
                valueInput.type = 'number';
            } else if (prop === 'camera_tilt') {
                hint.textContent = 'Enter camera tilt in degrees';
                valueInput.placeholder = 'Degrees (e.g., 1.5)';
                valueInput.type = 'number';
            } else {
                hint.textContent = 'Enter value manually';
                valueInput.placeholder = 'Value';
                valueInput.type = 'number';
            }
            updateSaveButtonState();
        });
        document.getElementById('cal-value').addEventListener('input', updateSaveButtonState);
        document.getElementById('cal-effective-from').addEventListener('input', updateSaveButtonState);

        function cancelCalibration() {
            // Reset form
            document.getElementById('cal-property').value = '';
            document.getElementById('cal-value').value = '';
            document.getElementById('cal-notes').value = '';
            document.getElementById('cal-coords').textContent = 'Click image';
            document.getElementById('cal-hint').textContent = 'Select a property, then click on the image';
            hideClickMarker();
            clickedX = null;
            clickedY = null;
            updateSaveButtonState();

            // Close panel
            toggleCalibrationMode();
        }

        function saveCalibration() {
            const prop = document.getElementById('cal-property').value;
            const rawValue = document.getElementById('cal-value').value;
            const effectiveFrom = document.getElementById('cal-effective-from').value;
            const notes = document.getElementById('cal-notes').value;

            if (!prop || !rawValue || !effectiveFrom) {
                alert('Please fill in all required fields');
                return;
            }

            // Parse value based on property type
            let value;
            if (prop.startsWith('stake_corners.') || prop.startsWith('stake_axis.') || prop.startsWith('sample_bounds.')) {
                // Parse "X, Y" format into [X, Y] array
                const parts = rawValue.split(',').map(s => parseInt(s.trim()));
                if (parts.length !== 2 || parts.some(isNaN)) {
                    alert('Please enter X, Y coordinates (e.g., 815, 300)');
                    return;
                }
                value = parts;
            } else {
                value = parseFloat(rawValue);
                if (isNaN(value)) {
                    alert('Please enter a valid number');
                    return;
                }
            }

            // Build the config object
            let config = {};
            if (prop.startsWith('marker_positions.')) {
                const inch = prop.split('.')[1];
                config.marker_positions = {};
                config.marker_positions[inch] = value;
            } else if (prop.startsWith('stake_corners.')) {
                const corner = prop.split('.')[1];
                config.stake_corners = {};
                config.stake_corners[corner] = value;
            } else if (prop.startsWith('stake_axis.')) {
                const point = prop.split('.')[1];
                config.stake_axis = {};
                config.stake_axis[point] = value;
            } else if (prop.startsWith('sample_bounds.')) {
                const point = prop.split('.')[1];
                config.sample_bounds = {};
                config.sample_bounds[point] = value;
            } else {
                config[prop] = value;
            }

            // Send to API
            fetch('/api/calibration/' + resort, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    effective_from: effectiveFrom + ':00',  // Add seconds
                    config: config,
                    notes: notes || `Set ${prop} to ${Array.isArray(value) ? '[' + value.join(', ') + ']' : value}`,
                    merge: true  // Merge with existing calibration
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Calibration saved successfully!');
                    // Reset form but keep panel open
                    document.getElementById('cal-property').value = '';
                    document.getElementById('cal-value').value = '';
                    document.getElementById('cal-notes').value = '';
                    document.getElementById('cal-coords').textContent = 'Click image';
                    hideClickMarker();
                    updateSaveButtonState();
                    // Reload calibration data table if visible
                    if (document.getElementById('cal-data-container').style.display !== 'none') {
                        loadCalibrationData();
                    }
                    // Refresh calibration status (progress indicator + history table)
                    updateCalibrationStatus();
                    // Refresh image to show new calibration
                    updateDisplay();
                } else {
                    alert('Error saving calibration: ' + data.error);
                }
            })
            .catch(err => {
                alert('Error saving calibration: ' + err);
            });
        }

        // ============ Calibration Status Section ============
        const CALIBRATION_POINTS = [
            // Stake corners (4 points)
            { key: 'stake_corners.top_left', label: 'Stake TL', category: 'stake' },
            { key: 'stake_corners.top_right', label: 'Stake TR', category: 'stake' },
            { key: 'stake_corners.bottom_left', label: 'Stake BL', category: 'stake' },
            { key: 'stake_corners.bottom_right', label: 'Stake BR', category: 'stake' },
            // Sample bounds (4 points)
            { key: 'sample_bounds.top_left', label: 'Sample TL', category: 'sample' },
            { key: 'sample_bounds.top_right', label: 'Sample TR', category: 'sample' },
            { key: 'sample_bounds.bottom_left', label: 'Sample BL', category: 'sample' },
            { key: 'sample_bounds.bottom_right', label: 'Sample BR', category: 'sample' },
            // Stake axis (2 points)
            { key: 'stake_axis.top', label: 'Axis Top', category: 'axis' },
            { key: 'stake_axis.bottom', label: 'Axis Bot', category: 'axis' },
            // Marker positions (10 markers)
            { key: 'marker_positions.0', label: '0"', category: 'marker' },
            { key: 'marker_positions.2', label: '2"', category: 'marker' },
            { key: 'marker_positions.4', label: '4"', category: 'marker' },
            { key: 'marker_positions.6', label: '6"', category: 'marker' },
            { key: 'marker_positions.8', label: '8"', category: 'marker' },
            { key: 'marker_positions.10', label: '10"', category: 'marker' },
            { key: 'marker_positions.12', label: '12"', category: 'marker' },
            { key: 'marker_positions.14', label: '14"', category: 'marker' },
            { key: 'marker_positions.16', label: '16"', category: 'marker' },
            { key: 'marker_positions.18', label: '18"', category: 'marker' }
        ];

        function getNestedValue(obj, path) {
            const parts = path.split('.');
            let val = obj;
            for (const p of parts) {
                if (val === undefined || val === null) return undefined;
                val = val[p];
            }
            return val;
        }

        function updateCalibrationStatus() {
            fetch('/api/calibration/' + resort + '/current')
                .then(response => response.json())
                .then(data => {
                    if (!data.success) {
                        document.getElementById('cal-status-id').textContent = '-';
                        document.getElementById('cal-status-date').textContent = '-';
                        document.getElementById('cal-set-count').textContent = '0';
                        document.getElementById('cal-progress-fill').style.width = '0%';
                        return;
                    }

                    const cal = data.calibration || {};
                    document.getElementById('cal-status-id').textContent = data.id || '-';
                    document.getElementById('cal-status-date').textContent = data.effective_from || '-';

                    // Count set points
                    let setCount = 0;
                    let checklistHtml = '';

                    CALIBRATION_POINTS.forEach(pt => {
                        const val = getNestedValue(cal, pt.key);
                        const isSet = val !== undefined && val !== null;
                        if (isSet) setCount++;

                        const statusClass = isSet ? 'set' : 'unset';
                        const statusIcon = isSet ? '✓' : '○';
                        // Format value for display
                        let valStr = '';
                        if (isSet) {
                            if (Array.isArray(val)) {
                                valStr = `[${val.join(', ')}]`;
                            } else {
                                valStr = String(val);
                            }
                        }
                        checklistHtml += `<div class="cal-check-item ${statusClass}" title="${pt.key}">
                            <span class="cal-check-icon">${statusIcon}</span>
                            <span class="cal-check-label">${pt.label}</span>
                            <span class="cal-check-val">${valStr}</span>
                        </div>`;
                    });

                    const total = CALIBRATION_POINTS.length;
                    document.getElementById('cal-set-count').textContent = setCount;
                    document.getElementById('cal-total-count').textContent = total;
                    document.getElementById('cal-progress-fill').style.width = (setCount / total * 100) + '%';
                    document.getElementById('cal-checklist').innerHTML = checklistHtml;

                    // Also load history table
                    loadCalHistoryTable();
                })
                .catch(err => {
                    console.error('Error loading calibration status:', err);
                });
        }

        function toggleCalHistory() {
            const container = document.getElementById('cal-history-container');
            if (container.style.display === 'none') {
                container.style.display = 'block';
            } else {
                container.style.display = 'none';
            }
        }

        function loadCalHistoryTable() {
            fetch('/api/calibration/' + resort + '/history?limit=10')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('cal-history-body');
                    if (!data.success || !data.versions || data.versions.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="4" style="color:#888">No history</td></tr>';
                        return;
                    }

                    let html = '';
                    data.versions.forEach(v => {
                        const effDate = new Date(v.effective_from).toLocaleString();
                        const created = v.created_at ? new Date(v.created_at).toLocaleString() : '-';
                        const notes = v.notes || '-';
                        html += `<tr>
                            <td>${v.id}</td>
                            <td>${effDate}</td>
                            <td>${created}</td>
                            <td>${notes}</td>
                        </tr>`;
                    });
                    tbody.innerHTML = html;
                })
                .catch(err => {
                    document.getElementById('cal-history-body').innerHTML =
                        '<tr><td colspan="4" style="color:#f66">Error loading history</td></tr>';
                });
        }

        // Get actual image dimensions when it loads
        document.getElementById('main-image').addEventListener('load', function() {
            // The natural dimensions are the original image size
            if (this.naturalWidth && this.naturalHeight) {
                originalImageWidth = this.naturalWidth;
                originalImageHeight = this.naturalHeight;
            }
        });

        // ============ Re-measure Current Image ============
        function remeasureThis() {
            const m = measurements[currentIndex];
            if (!m) {
                alert('No measurement selected');
                return;
            }

            if (!confirm('Re-measure this image with current calibration?')) {
                return;
            }

            fetch('/api/remeasure_single/' + resort + '/' + m.id, {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Re-measured! Depth: ' + (data.depth !== null ? data.depth.toFixed(1) + '"' : 'N/A'));
                    // Reload page but try to stay on same hour
                    // Store current hour before reload
                    const currentHour = measurements[currentIndex]?.hour || '12';
                    sessionStorage.setItem('returnToHour', currentHour);
                    window.location.reload();
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(err => {
                alert('Error: ' + err);
            });
        }

        // ============ Re-measure Modal ============
        function openRemeasureModal() {
            document.getElementById('remeasure-modal').classList.add('open');
            // Set default date range
            const today = new Date().toISOString().split('T')[0];
            const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
            document.getElementById('remeasure-start').value = weekAgo;
            document.getElementById('remeasure-end').value = today;
            // Reset state
            document.getElementById('remeasure-progress').classList.remove('active');
            document.getElementById('remeasure-status').textContent = '';
            document.getElementById('remeasure-submit').disabled = false;
            updateRemeasurePreview();
        }

        function closeRemeasureModal() {
            document.getElementById('remeasure-modal').classList.remove('open');
        }

        function updateRemeasurePreview() {
            const mode = document.getElementById('remeasure-mode').value;

            // Show/hide relevant inputs
            document.getElementById('remeasure-days-row').style.display =
                mode === 'last_n_days' ? 'block' : 'none';
            document.getElementById('remeasure-daterange-row').style.display =
                mode === 'date_range' ? 'block' : 'none';

            // Build preview URL
            let url = '/api/remeasure/' + resort + '/preview?mode=' + mode;
            if (mode === 'last_n_days') {
                url += '&days=' + document.getElementById('remeasure-days').value;
            } else if (mode === 'date_range') {
                url += '&start_date=' + document.getElementById('remeasure-start').value;
                url += '&end_date=' + document.getElementById('remeasure-end').value;
            }

            // Fetch preview
            document.getElementById('remeasure-count').textContent = '...';
            document.getElementById('remeasure-dates').textContent = '';

            fetch(url)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('remeasure-count').textContent = data.count;
                        let dateStr = '';
                        if (data.start_date && data.end_date) {
                            const start = new Date(data.start_date).toLocaleDateString();
                            const end = new Date(data.end_date).toLocaleDateString();
                            dateStr = start + ' to ' + end;
                        }
                        if (data.calibration_date) {
                            dateStr += ' (calibration: ' + data.calibration_date + ')';
                        }
                        document.getElementById('remeasure-dates').textContent = dateStr;
                    } else {
                        document.getElementById('remeasure-count').textContent = '0';
                        document.getElementById('remeasure-dates').textContent = data.error || data.message || '';
                    }
                })
                .catch(err => {
                    document.getElementById('remeasure-count').textContent = 'Error';
                    document.getElementById('remeasure-dates').textContent = err.toString();
                });
        }

        function executeRemeasure() {
            const mode = document.getElementById('remeasure-mode').value;
            const submitBtn = document.getElementById('remeasure-submit');
            const progressBar = document.getElementById('remeasure-progress');
            const progressFill = document.getElementById('remeasure-progress-fill');
            const statusText = document.getElementById('remeasure-status');

            // Build request body
            let body = { mode: mode };
            if (mode === 'last_n_days') {
                body.days = parseInt(document.getElementById('remeasure-days').value);
            } else if (mode === 'date_range') {
                body.start_date = document.getElementById('remeasure-start').value;
                body.end_date = document.getElementById('remeasure-end').value;
            }

            // Show progress
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';
            progressBar.classList.add('active');
            progressFill.style.width = '10%';
            statusText.textContent = 'Starting re-measurement...';

            fetch('/api/remeasure/' + resort, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            })
            .then(response => response.json())
            .then(data => {
                progressFill.style.width = '100%';

                if (data.success) {
                    const r = data.results;
                    statusText.textContent =
                        `Done! ${r.success} succeeded, ${r.failed} failed, ${r.skipped} skipped`;
                    statusText.style.color = '#4ade80';

                    // Reload page after short delay to show new measurements
                    setTimeout(() => {
                        window.location.reload();
                    }, 2000);
                } else {
                    statusText.textContent = 'Error: ' + data.error;
                    statusText.style.color = '#f87171';
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Re-measure';
                }
            })
            .catch(err => {
                progressFill.style.width = '100%';
                statusText.textContent = 'Error: ' + err.toString();
                statusText.style.color = '#f87171';
                submitBtn.disabled = false;
                submitBtn.textContent = 'Re-measure';
            });
        }

        // Close modal on escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeRemeasureModal();
            }
        });

        // Close modal on overlay click
        document.getElementById('remeasure-modal').addEventListener('click', function(e) {
            if (e.target === this) {
                closeRemeasureModal();
            }
        });

        // On page load, check if we should return to a specific hour (after remeasure)
        (function() {
            const returnToHour = sessionStorage.getItem('returnToHour');
            if (returnToHour) {
                sessionStorage.removeItem('returnToHour');
                // Find measurement with this hour
                for (let i = 0; i < measurements.length; i++) {
                    if (measurements[i].hour === returnToHour) {
                        selectMeasurement(i);
                        break;
                    }
                }
            }
        })();
    </script>
</body>
</html>
'''


def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_calibration(resort, timestamp=None):
    """Load calibration for a resort, checking DB versions first.

    Args:
        resort: Resort name
        timestamp: Optional datetime to get calibration effective at that time.
                   If None, gets the current calibration.
    """
    # First check database for time-based calibration versions
    try:
        from db import SnowDatabase
        db = SnowDatabase(DB_PATH)
        if timestamp:
            db_config = db.get_calibration_for_timestamp(resort, timestamp)
        else:
            db_config = db.get_current_calibration_version(resort)
        if db_config:
            return db_config
    except Exception as e:
        pass  # Fall through to file-based config

    # Try per-resort folder structure
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
    """Get measurements for a resort on a specific date (in MST)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Convert MST date to UTC range for query
    # MST is UTC-7, so MST midnight = UTC 07:00
    # Query for UTC timestamps that fall within the MST day
    # MST date 00:00 to 23:59 = UTC date 07:00 to next day 06:59
    utc_start = f"{date_str} 07:00:00"  # MST midnight in UTC
    # Calculate next day for end time
    from datetime import datetime, timedelta
    next_day = (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    utc_end = f"{next_day} 06:59:59"  # MST 23:59 in UTC

    cursor.execute('''
        SELECT id, timestamp, snow_depth_inches, confidence_score, image_path
        FROM snow_measurements
        WHERE resort = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
    ''', (resort, utc_start, utc_end))

    rows = cursor.fetchall()
    conn.close()

    measurements = []
    prev_depth = None
    MAX_HOURLY_CHANGE = 4.0  # Max inches change per hour before flagging as outlier

    for row in rows:
        measurement_id = row['id']
        ts_utc = datetime.fromisoformat(row['timestamp'])
        # Convert UTC to MST (UTC-7)
        ts = ts_utc - timedelta(hours=7)
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

    # Draw filled regions with 50% opacity
    show_regions = request.args.get('regions', 'false').lower() == 'true'
    if show_regions and calibration:
        overlay = image.copy()

        # Draw stake_corners region (orange fill)
        stake_corners = calibration.get('stake_corners', {})
        sc_tl = stake_corners.get('top_left')
        sc_tr = stake_corners.get('top_right')
        sc_bl = stake_corners.get('bottom_left')
        sc_br = stake_corners.get('bottom_right')
        if all([sc_tl, sc_tr, sc_bl, sc_br]):
            pts_stake = np.array([sc_tl, sc_tr, sc_br, sc_bl], np.int32)
            cv2.fillPoly(overlay, [pts_stake], (0, 100, 255))  # Orange (BGR)
            # Add label
            center_x = (sc_tl[0] + sc_br[0]) // 2
            center_y = (sc_tl[1] + sc_br[1]) // 2
            cv2.putText(overlay, "STAKE", (center_x - 40, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw sample_bounds region (green fill)
        sample_bounds = calibration.get('sample_bounds', {})
        sb_tl = sample_bounds.get('top_left')
        sb_tr = sample_bounds.get('top_right')
        sb_bl = sample_bounds.get('bottom_left')
        sb_br = sample_bounds.get('bottom_right')
        if all([sb_tl, sb_tr, sb_bl, sb_br]):
            pts_base = np.array([sb_tl, sb_tr, sb_br, sb_bl], np.int32)
            cv2.fillPoly(overlay, [pts_base], (0, 200, 0))  # Green (BGR)
            # Add label
            center_x = (sb_tl[0] + sb_br[0]) // 2
            center_y = (sb_tl[1] + sb_br[1]) // 2
            cv2.putText(overlay, "BASE", (center_x - 30, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Blend overlay with original image (50% opacity)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    # Draw calibration overlay if available
    if calibration:
        import math

        # Check for new format (stake_corners, stake_axis) or fall back to legacy
        stake_corners = calibration.get('stake_corners')
        stake_axis = calibration.get('stake_axis')
        camera_tilt = calibration.get('camera_tilt', 0.0)

        if stake_corners and stake_axis:
            # NEW FORMAT: Use stake_corners and stake_axis
            tl = stake_corners.get('top_left', [0, 0])
            tr = stake_corners.get('top_right', [0, 0])
            bl = stake_corners.get('bottom_left', [0, 0])
            br = stake_corners.get('bottom_right', [0, 0])

            axis_top = stake_axis.get('top', [0, 0])
            axis_bottom = stake_axis.get('bottom', [0, 0])

            # Draw stake region quadrilateral
            pts_corners = np.array([tl, tr, br, bl], np.int32)
            cv2.polylines(image, [pts_corners], True, (255, 100, 0), 2)

            # Draw stake axis line (magenta)
            cv2.line(image, tuple(axis_bottom), tuple(axis_top), (255, 0, 255), 2)

            # Draw axis endpoint markers
            cv2.circle(image, tuple(axis_bottom), 6, (0, 255, 0), -1)  # Green = 0" (bottom)
            cv2.circle(image, tuple(axis_top), 6, (0, 255, 255), -1)   # Yellow = top

            # Draw sample bounds quadrilateral if available
            sample_bounds = calibration.get('sample_bounds')
            if sample_bounds:
                sb_tl = sample_bounds.get('top_left')
                sb_tr = sample_bounds.get('top_right')
                sb_bl = sample_bounds.get('bottom_left')
                sb_br = sample_bounds.get('bottom_right')

                # Draw corners with labels
                for pt, label in [(sb_tl, 'TL'), (sb_tr, 'TR'), (sb_bl, 'BL'), (sb_br, 'BR')]:
                    if pt:
                        cv2.circle(image, tuple(pt), 6, (255, 255, 0), -1)  # Cyan
                        cv2.putText(image, label, (pt[0] + 8, pt[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Draw quadrilateral if all 4 corners defined
                if all([sb_tl, sb_tr, sb_bl, sb_br]):
                    pts_sample = np.array([sb_tl, sb_tr, sb_br, sb_bl], np.int32)
                    cv2.polylines(image, [pts_sample], True, (255, 255, 0), 2)

            # For backward compat with drawing code below, compute legacy values
            x = min(tl[0], bl[0])
            y = min(tl[1], tr[1])
            w = max(tr[0], br[0]) - x
            h = max(bl[1], br[1]) - y
            centerline_x = (axis_top[0] + axis_bottom[0]) // 2
            tilt_angle = camera_tilt
            tilt_rad = math.radians(tilt_angle)
            dx_per_dy = math.tan(tilt_rad)
            top_shift = int(h * dx_per_dy) if h else 0
        else:
            # LEGACY FORMAT: Use stake_region_x/y/width/height
            x = calibration.get('stake_region_x')
            y = calibration.get('stake_region_y')
            w = calibration.get('stake_region_width')
            h = calibration.get('stake_region_height')
            centerline_x = calibration.get('stake_centerline_x')
            tilt_angle = calibration.get('tilt_angle', 0.0)

            tilt_rad = math.radians(tilt_angle)
            dx_per_dy = math.tan(tilt_rad)
            top_shift = int((h or 0) * dx_per_dy)

            if all([x, y, w, h]):
                # Draw full stake region (outer box, dimmer) - adjusted for tilt
                pts_outer = np.array([
                    [x + top_shift, y],
                    [x + w + top_shift, y],
                    [x + w, y + h],
                    [x, y + h]
                ], np.int32)
                cv2.polylines(image, [pts_outer], True, (128, 64, 0), 1)

        # Draw measurement region and sample lines (common code for both formats)
        # Get sample bounds if available (new format)
        sample_bounds = calibration.get('sample_bounds', {})
        sb_tl = sample_bounds.get('top_left')
        sb_tr = sample_bounds.get('top_right')
        sb_bl = sample_bounds.get('bottom_left')
        sb_br = sample_bounds.get('bottom_right')
        has_sample_quad = all([sb_tl, sb_tr, sb_bl, sb_br])

        if all([x, y, w, h]) and centerline_x:
                # Use sample_bounds if available, otherwise compute from centerline
                if has_sample_quad:
                    # Use bottom edge of sample quad for base
                    measure_x = sb_bl[0]
                    measure_width = sb_br[0] - sb_bl[0]
                    base_y = (sb_bl[1] + sb_br[1]) // 2  # Average of bottom corners
                else:
                    measure_width = int(w * 0.75)
                    measure_x = centerline_x - measure_width // 2
                    base_y = y + h  # Default to bottom of region

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
                marker_positions = calibration.get('marker_positions', {})
                ref_y = marker_positions.get('0') or marker_positions.get(0) or calibration.get('reference_y') or base_y
                ppi = calibration.get('pixels_per_inch', 27.0)

                # Calculate sample line base positions from sample_bounds (65% width)
                # Use 17.5% margin on each side to avoid blown-off edges
                # Lines start at base, extend up to detected snow line
                if has_sample_quad:
                    full_width = sb_br[0] - sb_bl[0]
                    margin = int(full_width * 0.175)
                    base_left_x = sb_bl[0] + margin
                    base_right_x = sb_br[0] - margin
                    base_left_y = sb_bl[1]
                    base_right_y = sb_br[1]
                else:
                    base_left_x = measure_x
                    base_right_x = measure_x + measure_width
                    base_left_y = base_y
                    base_right_y = base_y

                # If we have sample data from database, use individual snow lines
                if sample_data and len(sample_data) == num_samples:
                    for i, sample in enumerate(sample_data):
                        # Calculate X position along the base (sample_bounds bottom edge)
                        # Interpolate between left and right edges
                        t_x = i / (num_samples - 1) if num_samples > 1 else 0.5
                        base_x = int(base_left_x + t_x * (base_right_x - base_left_x))
                        base_y_at_x = int(base_left_y + t_x * (base_right_y - base_left_y))

                        sample_snow_y = sample.get('snow_line_y')
                        sample_depth = sample.get('depth_inches')
                        sample_valid = sample.get('valid', False)
                        sample_skip = sample.get('skip_reason')

                        # Default to reference_y (base) if no snow line detected
                        if sample_snow_y is None:
                            sample_snow_y = ref_y if ref_y else base_y_at_x

                        # Choose color based on validity
                        if sample_valid:
                            line_color = (50, 200, 255)  # Light orange for valid
                        elif sample_skip:
                            line_color = (50, 50, 200)   # Red for skipped
                        else:
                            line_color = (100, 100, 100) # Gray for no data

                        # Draw line from base (sample_bounds bottom) UP to snow line
                        cv2.line(image,
                            (base_x, base_y_at_x),
                            (base_x, sample_snow_y),
                            line_color, 2)

                        # Draw sample number label at base
                        num_label = str(i + 1)
                        num_x = base_x - 5
                        num_y = base_y_at_x + 18
                        cv2.putText(image, num_label, (num_x, num_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                        # Draw small depth label at snow line (only for valid samples)
                        if sample_depth is not None and sample_valid:
                            label_text = f"{sample_depth:.1f}"
                            label_x = base_x - 15
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
                        # Calculate X position along the base
                        t_x = i / (num_samples - 1) if num_samples > 1 else 0.5
                        base_x = int(base_left_x + t_x * (base_right_x - base_left_x))
                        base_y_at_x = int(base_left_y + t_x * (base_right_y - base_left_y))

                        # Draw line from base to snow line
                        cv2.line(image,
                            (base_x, base_y_at_x),
                            (base_x, snow_line_y),
                            (255, 150, 50), 1)

                # Draw summary stats if available
                if depth_min is not None and depth_max is not None and depth_avg is not None:
                    stats_text = f"Min:{depth_min:.1f}\" Avg:{depth_avg:.1f}\" Max:{depth_max:.1f}\""
                    cv2.putText(image, stats_text, (x, y - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 200, 100), 2)

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

    # Add MST timestamp overlay on bottom right
    try:
        # Extract timestamp from filename (e.g., winter_park_20251127_230039.jpg)
        import re
        match = re.search(r'(\d{8})_(\d{6})', os.path.basename(image_path))
        if match:
            date_str, time_str = match.groups()
            # Parse as UTC
            from datetime import datetime, timedelta
            utc_time = datetime.strptime(f'{date_str}_{time_str}', '%Y%m%d_%H%M%S')
            # Convert to MST (UTC-7)
            mst_time = utc_time - timedelta(hours=7)
            timestamp_text = mst_time.strftime('%Y-%m-%d %H:%M MST')

            # Draw on bottom right with background
            img_h, img_w = image.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(timestamp_text, font, font_scale, thickness)
            padding = 10
            x = img_w - text_w - padding - 10
            y = img_h - padding - 10

            # Semi-transparent background
            cv2.rectangle(image, (x - padding, y - text_h - padding),
                         (x + text_w + padding, y + baseline + padding), (0, 0, 0), -1)
            cv2.putText(image, timestamp_text, (x, y), font, font_scale, (255, 255, 255), thickness)
    except Exception as e:
        pass  # Don't fail if timestamp extraction fails

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
        from analytics import SnowAnalytics
        analytics = SnowAnalytics(DB_PATH)

        # Parse date
        date_obj = datetime.strptime(date, '%Y-%m-%d')

        summary = analytics.get_daily_summary(resort, date_obj)
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/calibration/<resort>', methods=['GET'])
def api_get_calibration(resort):
    """Get calibration for a resort, optionally for a specific timestamp."""
    from db import SnowDatabase

    try:
        db = SnowDatabase(DB_PATH)

        # Check for timestamp parameter
        timestamp_str = request.args.get('timestamp')
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str)
            config = db.get_calibration_for_timestamp(resort, timestamp)
        else:
            config = db.get_current_calibration_version(resort)

        # Fall back to file-based config if no DB version exists
        if not config:
            config = get_calibration(resort)

        if config:
            return jsonify({'success': True, 'calibration': config})
        else:
            return jsonify({'success': False, 'error': 'No calibration found'}), 404

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/<resort>/current', methods=['GET'])
def api_get_current_calibration(resort):
    """Get current calibration with ID and effective date."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Only get calibrations where effective_from <= now (not future dates)
        from datetime import datetime
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            SELECT id, effective_from, config_json
            FROM calibration_versions
            WHERE resort = ? AND effective_from <= ?
            ORDER BY effective_from DESC, id DESC
            LIMIT 1
        ''', (resort, now_str))
        row = cursor.fetchone()
        conn.close()

        if row:
            return jsonify({
                'success': True,
                'id': row['id'],
                'effective_from': row['effective_from'],
                'calibration': json.loads(row['config_json'])
            })
        else:
            # Fall back to file-based config
            config = get_calibration(resort)
            if config:
                return jsonify({
                    'success': True,
                    'id': None,
                    'effective_from': None,
                    'calibration': config
                })
            return jsonify({'success': False, 'error': 'No calibration found'}), 404

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/<resort>', methods=['POST'])
def api_save_calibration(resort):
    """Save a new calibration version for a resort."""
    from db import SnowDatabase

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400

        effective_from_str = data.get('effective_from')
        if not effective_from_str:
            return jsonify({'success': False, 'error': 'effective_from is required'}), 400

        effective_from = datetime.fromisoformat(effective_from_str)
        config = data.get('config', {})
        notes = data.get('notes')
        created_by = data.get('created_by')

        # Merge with existing calibration if this is a partial update
        if data.get('merge', False):
            db = SnowDatabase(DB_PATH)
            existing = db.get_current_calibration_version(resort)
            if not existing:
                existing = get_calibration(resort) or {}

            # Deep merge for nested objects (marker_positions, stake_corners, stake_axis, sample_bounds)
            for nested_key in ['marker_positions', 'stake_corners', 'stake_axis', 'sample_bounds']:
                if nested_key in config and nested_key in existing:
                    merged_nested = existing.get(nested_key, {}).copy()
                    merged_nested.update(config[nested_key])
                    config[nested_key] = merged_nested
                elif nested_key in existing and nested_key not in config:
                    # Preserve existing nested object if not being updated
                    config[nested_key] = existing[nested_key]

            # Shallow merge for other fields
            merged = existing.copy()
            merged.update(config)
            config = merged

        db = SnowDatabase(DB_PATH)
        cal_id = db.save_calibration_version(
            resort=resort,
            effective_from=effective_from,
            config=config,
            notes=notes,
            created_by=created_by
        )

        return jsonify({
            'success': True,
            'id': cal_id,
            'message': f'Calibration saved, effective from {effective_from}'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/<resort>/history', methods=['GET'])
def api_calibration_history(resort):
    """Get calibration version history for a resort."""
    from db import SnowDatabase

    try:
        limit = request.args.get('limit', 20, type=int)
        db = SnowDatabase(DB_PATH)
        versions = db.get_calibration_versions(resort, limit=limit)

        return jsonify({'success': True, 'versions': versions})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def get_measurer_for_resort(resort, calibration):
    """Get the appropriate measurer for a resort.

    Uses custom measurer if available (e.g., WinterParkMeasurer for winter_park),
    otherwise falls back to base SnowStakeMeasurer.
    """
    from measurement import SnowStakeMeasurer

    # Try to load resort-specific measurer
    if resort == 'winter_park':
        try:
            import sys
            sys.path.insert(0, '/app/resorts/winter_park')
            from measurer import WinterParkMeasurer
            return WinterParkMeasurer(calibration)
        except Exception as e:
            print(f"Could not load WinterParkMeasurer: {e}, using base measurer")

    # Fall back to base measurer
    return SnowStakeMeasurer(
        pixels_per_inch=calibration.get('pixels_per_inch'),
        debug=False
    )


@app.route('/api/remeasure_single/<resort>/<int:measurement_id>', methods=['POST'])
def api_remeasure_single(resort, measurement_id):
    """Re-measure a single image with current calibration."""
    from db import SnowDatabase
    from measurement import SnowStakeMeasurer
    import glob

    try:
        db = SnowDatabase(DB_PATH)

        # Get the measurement
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, timestamp, image_path FROM snow_measurements WHERE id = ?',
            (measurement_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({'success': False, 'error': 'Measurement not found'}), 404

        image_path = row['image_path']
        timestamp = datetime.fromisoformat(row['timestamp'])

        # Find the actual image file
        if image_path.startswith('/out/'):
            image_path = image_path[5:]

        actual_path = None
        for path in [image_path, os.path.join(OUT_DIR, image_path), os.path.join(OUT_DIR, os.path.basename(image_path))]:
            if os.path.exists(path):
                actual_path = path
                break

        if not actual_path:
            return jsonify({'success': False, 'error': f'Image not found: {image_path}'}), 404

        # Get CURRENT calibration (not for timestamp)
        calibration = db.get_current_calibration_version(resort)
        if not calibration:
            calibration = get_calibration(resort)
        if not calibration:
            return jsonify({'success': False, 'error': 'No calibration found'}), 400

        # Create measurer and measure (uses resort-specific measurer if available)
        measurer = get_measurer_for_resort(resort, calibration)

        # WinterParkMeasurer has different API - measure_from_file takes only image path
        if resort == 'winter_park' and hasattr(measurer, 'measure_from_file'):
            result = measurer.measure_from_file(actual_path)
        else:
            result = measurer.measure_from_file(actual_path, calibration)

        # Update the database (handle both MeasurementResult and WinterParkMeasurement)
        db.delete_measurement(measurement_id)
        db.insert_measurement(
            resort=resort,
            timestamp=timestamp,
            image_path=row['image_path'],
            snow_depth_inches=result.snow_depth_inches,
            confidence_score=result.confidence_score,
            stake_visible=result.stake_visible,
            raw_pixel_measurement=getattr(result, 'raw_pixel_measurement', None),
            notes=getattr(result, 'notes', ''),
            sample_data=result.samples,
            replace_hourly=True
        )

        return jsonify({
            'success': True,
            'depth': result.snow_depth_inches,
            'confidence': result.confidence_score
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calibration/<resort>/properties', methods=['GET'])
def api_calibration_properties(resort):
    """Get list of calibration properties that can be set."""
    properties = [
        {'key': 'stake_region_x', 'label': 'Stake Region X', 'type': 'x', 'description': 'Left edge of stake bounding box'},
        {'key': 'stake_region_y', 'label': 'Stake Region Y', 'type': 'y', 'description': 'Top edge of stake bounding box'},
        {'key': 'stake_region_width', 'label': 'Stake Region Width', 'type': 'number', 'description': 'Width of stake bounding box'},
        {'key': 'stake_region_height', 'label': 'Stake Region Height', 'type': 'number', 'description': 'Height of stake bounding box'},
        {'key': 'stake_centerline_x', 'label': 'Stake Centerline X', 'type': 'x', 'description': 'X coordinate of stake center'},
        {'key': 'reference_y', 'label': 'Reference Y (0")', 'type': 'y', 'description': 'Y coordinate of 0" reference line'},
        {'key': 'pixels_per_inch', 'label': 'Pixels Per Inch', 'type': 'number', 'description': 'Linear scaling factor'},
        {'key': 'tilt_angle', 'label': 'Tilt Angle', 'type': 'number', 'description': 'Camera tilt in degrees'},
        {'key': 'min_depth_threshold', 'label': 'Min Depth Threshold', 'type': 'number', 'description': 'Minimum depth to report (inches)'},
        {'key': 'marker_positions.0', 'label': '0" Marker Y', 'type': 'y', 'description': 'Y position of 0 inch marker'},
        {'key': 'marker_positions.2', 'label': '2" Marker Y', 'type': 'y', 'description': 'Y position of 2 inch marker'},
        {'key': 'marker_positions.4', 'label': '4" Marker Y', 'type': 'y', 'description': 'Y position of 4 inch marker'},
        {'key': 'marker_positions.6', 'label': '6" Marker Y', 'type': 'y', 'description': 'Y position of 6 inch marker'},
        {'key': 'marker_positions.8', 'label': '8" Marker Y', 'type': 'y', 'description': 'Y position of 8 inch marker'},
        {'key': 'marker_positions.10', 'label': '10" Marker Y', 'type': 'y', 'description': 'Y position of 10 inch marker'},
        {'key': 'marker_positions.12', 'label': '12" Marker Y', 'type': 'y', 'description': 'Y position of 12 inch marker'},
        {'key': 'marker_positions.14', 'label': '14" Marker Y', 'type': 'y', 'description': 'Y position of 14 inch marker'},
        {'key': 'marker_positions.16', 'label': '16" Marker Y', 'type': 'y', 'description': 'Y position of 16 inch marker'},
        {'key': 'marker_positions.18', 'label': '18" Marker Y', 'type': 'y', 'description': 'Y position of 18 inch marker'},
    ]
    return jsonify({'success': True, 'properties': properties})


@app.route('/api/remeasure/<resort>', methods=['POST'])
def api_remeasure(resort):
    """Re-measure snow depths for a resort using current calibration.

    Request body options:
    - mode: 'since_calibration' | 'last_n_days' | 'date_range' | 'all'
    - days: number of days (for 'last_n_days' mode)
    - start_date: ISO date string (for 'date_range' mode)
    - end_date: ISO date string (for 'date_range' mode)
    """
    from db import SnowDatabase
    from measurement import SnowStakeMeasurer
    import glob

    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'since_calibration')
        dry_run = data.get('dry_run', False)

        db = SnowDatabase(DB_PATH)

        # Determine date range based on mode
        if mode == 'since_calibration':
            # Get the most recent calibration's effective_from date
            versions = db.get_calibration_versions(resort, limit=1)
            if versions:
                start_date = datetime.fromisoformat(versions[0]['effective_from'])
            else:
                return jsonify({
                    'success': False,
                    'error': 'No calibration versions found for this resort'
                }), 400
            end_date = datetime.now()

        elif mode == 'last_n_days':
            days = data.get('days', 7)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

        elif mode == 'date_range':
            start_date = datetime.fromisoformat(data['start_date'])
            end_date = datetime.fromisoformat(data['end_date'])

        elif mode == 'all':
            # Get earliest measurement date
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                'SELECT MIN(timestamp) FROM snow_measurements WHERE resort = ?',
                (resort,)
            )
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                start_date = datetime.fromisoformat(row[0])
            else:
                start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()

        else:
            return jsonify({'success': False, 'error': f'Unknown mode: {mode}'}), 400

        # Get all measurements in the date range
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, timestamp, image_path
            FROM snow_measurements
            WHERE resort = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        ''', (resort, start_date.isoformat(), end_date.isoformat()))
        measurements = cursor.fetchall()
        conn.close()

        if dry_run:
            return jsonify({
                'success': True,
                'dry_run': True,
                'count': len(measurements),
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'message': f'Would re-measure {len(measurements)} images'
            })

        # Re-measure each image
        results = {
            'processed': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }

        for m in measurements:
            measurement_id = m['id']
            timestamp = datetime.fromisoformat(m['timestamp'])
            image_path = m['image_path']

            # Find the actual image file
            actual_path = None
            if image_path.startswith('/out/'):
                image_path = image_path[5:]

            paths_to_try = [
                image_path,
                os.path.join(OUT_DIR, image_path),
                os.path.join(OUT_DIR, os.path.basename(image_path)),
            ]

            for path in paths_to_try:
                if os.path.exists(path):
                    actual_path = path
                    break

            if not actual_path:
                # Try pattern matching
                base = os.path.basename(image_path).split('.')[0]
                pattern = os.path.join(OUT_DIR, f"{base}*.png")
                matches = glob.glob(pattern)
                if matches:
                    actual_path = matches[0]

            if not actual_path:
                results['skipped'] += 1
                results['errors'].append({
                    'id': measurement_id,
                    'error': f'Image not found: {image_path}'
                })
                continue

            results['processed'] += 1

            try:
                # Get calibration for this timestamp
                calibration = get_calibration(resort, timestamp)
                if not calibration:
                    results['failed'] += 1
                    results['errors'].append({
                        'id': measurement_id,
                        'error': 'No calibration for timestamp'
                    })
                    continue

                # Create measurer and measure (uses resort-specific measurer if available)
                measurer = get_measurer_for_resort(resort, calibration)

                # WinterParkMeasurer has different API
                if resort == 'winter_park' and hasattr(measurer, 'measure_from_file'):
                    result = measurer.measure_from_file(actual_path)
                else:
                    result = measurer.measure_from_file(actual_path, calibration)

                # Update the database (handle both MeasurementResult and WinterParkMeasurement)
                db.delete_measurement(measurement_id)
                db.insert_measurement(
                    resort=resort,
                    timestamp=timestamp,
                    image_path=image_path,
                    snow_depth_inches=result.snow_depth_inches,
                    confidence_score=result.confidence_score,
                    stake_visible=result.stake_visible,
                    raw_pixel_measurement=getattr(result, 'raw_pixel_measurement', None),
                    notes=getattr(result, 'notes', ''),
                    sample_data=result.samples,
                    replace_hourly=True
                )

                results['success'] += 1

            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'id': measurement_id,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'results': results,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/remeasure/<resort>/preview', methods=['GET'])
def api_remeasure_preview(resort):
    """Preview how many measurements would be re-processed."""
    from db import SnowDatabase

    try:
        mode = request.args.get('mode', 'since_calibration')
        days = request.args.get('days', 7, type=int)
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')

        db = SnowDatabase(DB_PATH)

        # Determine date range based on mode
        if mode == 'since_calibration':
            versions = db.get_calibration_versions(resort, limit=1)
            if versions:
                start_date = datetime.fromisoformat(versions[0]['effective_from'])
                cal_date = start_date.strftime('%Y-%m-%d %H:%M')
            else:
                return jsonify({
                    'success': True,
                    'count': 0,
                    'message': 'No calibration versions found'
                })
            end_date = datetime.now()

        elif mode == 'last_n_days':
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            cal_date = None

        elif mode == 'date_range':
            start_date = datetime.fromisoformat(start_date_str)
            end_date = datetime.fromisoformat(end_date_str)
            cal_date = None

        elif mode == 'all':
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                'SELECT MIN(timestamp), MAX(timestamp) FROM snow_measurements WHERE resort = ?',
                (resort,)
            )
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                start_date = datetime.fromisoformat(row[0])
                end_date = datetime.fromisoformat(row[1]) if row[1] else datetime.now()
            else:
                return jsonify({'success': True, 'count': 0, 'message': 'No measurements found'})
            cal_date = None

        else:
            return jsonify({'success': False, 'error': f'Unknown mode: {mode}'}), 400

        # Count measurements in range
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM snow_measurements
            WHERE resort = ? AND timestamp >= ? AND timestamp <= ?
        ''', (resort, start_date.isoformat(), end_date.isoformat()))
        count = cursor.fetchone()[0]
        conn.close()

        response = {
            'success': True,
            'count': count,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }

        if mode == 'since_calibration' and cal_date:
            response['calibration_date'] = cal_date

        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    print(f"Starting Snow Depth Frontend on port {port}")
    print(f"Database: {DB_PATH}")
    print(f"Images: {OUT_DIR}")
    print(f"Config: {CONFIG_PATH}")

    app.run(host='0.0.0.0', port=port, debug=debug)
