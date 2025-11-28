"""
SQLite database module for storing snow depth measurements.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class SnowDatabase:
    """Manages SQLite database for snow depth measurements."""

    def __init__(self, db_path: str = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Defaults to './snow_measurements.db'
        """
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__),
                'snow_measurements.db'
            )
        self.db_path = db_path
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Main measurements table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS snow_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resort TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    image_path TEXT NOT NULL,
                    snow_depth_inches REAL,
                    confidence_score REAL,
                    stake_visible BOOLEAN,
                    raw_pixel_measurement INTEGER,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(resort, timestamp)
                )
            """)

            # Index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_resort_timestamp
                ON snow_measurements(resort, timestamp DESC)
            """)

            # Calibration data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calibration_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resort TEXT NOT NULL UNIQUE,
                    pixels_per_inch REAL NOT NULL,
                    stake_base_y INTEGER,
                    stake_top_y INTEGER,
                    reference_y INTEGER,
                    reference_height_inches REAL,
                    reference_image_path TEXT,
                    calibration_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
            """)

            # Sample measurements table - stores individual vertical sample data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS measurement_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    measurement_id INTEGER NOT NULL,
                    sample_index INTEGER NOT NULL,
                    x_position INTEGER,
                    snow_line_y INTEGER,
                    depth_inches REAL,
                    FOREIGN KEY (measurement_id) REFERENCES snow_measurements(id),
                    UNIQUE(measurement_id, sample_index)
                )
            """)

            # Add sample statistics columns to main table if they don't exist
            try:
                cursor.execute("ALTER TABLE snow_measurements ADD COLUMN depth_min REAL")
                cursor.execute("ALTER TABLE snow_measurements ADD COLUMN depth_max REAL")
                cursor.execute("ALTER TABLE snow_measurements ADD COLUMN depth_avg REAL")
                cursor.execute("ALTER TABLE snow_measurements ADD COLUMN sample_data TEXT")  # JSON string of all samples
            except:
                pass  # Columns already exist

            # Daily auto-calibration table - stores detected calibration values per day
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_calibrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resort TEXT NOT NULL,
                    date DATE NOT NULL,
                    pixels_per_inch REAL,
                    tilt_angle REAL,
                    reference_y INTEGER,
                    stake_centerline_x INTEGER,
                    detection_confidence REAL,
                    markers_detected TEXT,
                    source_image_path TEXT,
                    calibration_method TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(resort, date)
                )
            """)

            # Index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_cal_resort_date
                ON daily_calibrations(resort, date DESC)
            """)

    def insert_measurement(
        self,
        resort: str,
        timestamp: datetime,
        image_path: str,
        snow_depth_inches: Optional[float] = None,
        confidence_score: Optional[float] = None,
        stake_visible: bool = False,
        raw_pixel_measurement: Optional[int] = None,
        notes: Optional[str] = None,
        sample_data: Optional[List[Dict]] = None
    ) -> int:
        """Insert a snow depth measurement.

        Args:
            resort: Resort name
            timestamp: Measurement timestamp
            image_path: Path to the image file
            snow_depth_inches: Measured snow depth in inches (median/avg)
            confidence_score: Confidence score (0-1)
            stake_visible: Whether stake was detected
            raw_pixel_measurement: Raw pixel count measurement
            notes: Optional notes
            sample_data: List of dicts with keys: sample_index, x_position, snow_line_y, depth_inches

        Returns:
            ID of inserted row
        """
        import json

        # Calculate min/max/avg from sample data if provided
        depth_min = None
        depth_max = None
        depth_avg = None
        sample_json = None

        if sample_data:
            depths = [s['depth_inches'] for s in sample_data if s.get('depth_inches') is not None]
            if depths:
                depth_min = min(depths)
                depth_max = max(depths)
                depth_avg = sum(depths) / len(depths)
            sample_json = json.dumps(sample_data)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO snow_measurements
                (resort, timestamp, image_path, snow_depth_inches,
                 confidence_score, stake_visible, raw_pixel_measurement, notes,
                 depth_min, depth_max, depth_avg, sample_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                resort,
                timestamp,
                image_path,
                snow_depth_inches,
                confidence_score,
                stake_visible,
                raw_pixel_measurement,
                notes,
                depth_min,
                depth_max,
                depth_avg,
                sample_json
            ))
            measurement_id = cursor.lastrowid

            # Also insert individual samples into the samples table
            if sample_data:
                for sample in sample_data:
                    cursor.execute("""
                        INSERT OR REPLACE INTO measurement_samples
                        (measurement_id, sample_index, x_position, snow_line_y, depth_inches)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        measurement_id,
                        sample.get('sample_index'),
                        sample.get('x_position'),
                        sample.get('snow_line_y'),
                        sample.get('depth_inches')
                    ))

            return measurement_id

    def get_measurements(
        self,
        resort: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query snow depth measurements.

        Args:
            resort: Filter by resort name
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results

        Returns:
            List of measurement dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM snow_measurements WHERE 1=1"
            params = []

            if resort:
                query += " AND resort = ?"
                params.append(resort)

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            query += " ORDER BY timestamp DESC"

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_latest_measurement(self, resort: str) -> Optional[Dict[str, Any]]:
        """Get the most recent measurement for a resort.

        Args:
            resort: Resort name

        Returns:
            Latest measurement dictionary or None
        """
        measurements = self.get_measurements(resort=resort, limit=1)
        return measurements[0] if measurements else None

    def set_calibration(
        self,
        resort: str,
        pixels_per_inch: float,
        stake_base_y: Optional[int] = None,
        stake_top_y: Optional[int] = None,
        reference_y: Optional[int] = None,
        reference_height_inches: Optional[float] = None,
        reference_image_path: Optional[str] = None,
        notes: Optional[str] = None
    ):
        """Store calibration data for a resort.

        Args:
            resort: Resort name
            pixels_per_inch: Calibration ratio
            stake_base_y: Y coordinate of stake base
            stake_top_y: Y coordinate of stake top
            reference_y: Y coordinate of reference point (zero/base marking)
            reference_height_inches: Height of reference marking (e.g., lowest visible marking)
            reference_image_path: Path to calibration reference image
            notes: Optional notes
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO calibration_data
                (resort, pixels_per_inch, stake_base_y, stake_top_y,
                 reference_y, reference_height_inches, reference_image_path, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                resort,
                pixels_per_inch,
                stake_base_y,
                stake_top_y,
                reference_y,
                reference_height_inches,
                reference_image_path,
                notes
            ))

    def get_calibration(self, resort: str) -> Optional[Dict[str, Any]]:
        """Get calibration data for a resort.

        Args:
            resort: Resort name

        Returns:
            Calibration dictionary or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM calibration_data WHERE resort = ?",
                (resort,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def save_daily_calibration(
        self,
        resort: str,
        date: str,
        pixels_per_inch: float,
        tilt_angle: Optional[float] = None,
        reference_y: Optional[int] = None,
        stake_centerline_x: Optional[int] = None,
        detection_confidence: Optional[float] = None,
        markers_detected: Optional[str] = None,
        source_image_path: Optional[str] = None,
        calibration_method: str = "auto"
    ):
        """Save auto-detected calibration for a specific day.

        Args:
            resort: Resort name
            date: Date string (YYYY-MM-DD)
            pixels_per_inch: Detected pixels per inch
            tilt_angle: Detected camera tilt in degrees
            reference_y: Detected Y coordinate of 0" reference
            stake_centerline_x: Detected X coordinate of stake center
            detection_confidence: Confidence score (0-1)
            markers_detected: JSON string of detected markers
            source_image_path: Path to image used for calibration
            calibration_method: Method used (auto, manual, fallback)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO daily_calibrations
                (resort, date, pixels_per_inch, tilt_angle, reference_y,
                 stake_centerline_x, detection_confidence, markers_detected,
                 source_image_path, calibration_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                resort,
                date,
                pixels_per_inch,
                tilt_angle,
                reference_y,
                stake_centerline_x,
                detection_confidence,
                markers_detected,
                source_image_path,
                calibration_method
            ))

    def get_daily_calibration(
        self,
        resort: str,
        date: str
    ) -> Optional[Dict[str, Any]]:
        """Get calibration for a specific day.

        Args:
            resort: Resort name
            date: Date string (YYYY-MM-DD)

        Returns:
            Calibration dictionary or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM daily_calibrations WHERE resort = ? AND date = ?",
                (resort, date)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_latest_daily_calibration(
        self,
        resort: str
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent daily calibration for a resort.

        Args:
            resort: Resort name

        Returns:
            Calibration dictionary or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM daily_calibrations WHERE resort = ? ORDER BY date DESC LIMIT 1",
                (resort,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
