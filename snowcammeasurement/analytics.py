"""
Snow depth analytics module for calculating accumulation statistics.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
try:
    from .db import SnowDatabase
except ImportError:
    from db import SnowDatabase


class SnowAnalytics:
    """Analyzes snow depth measurements and calculates accumulation."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize analytics with database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db = SnowDatabase(db_path)

    def get_measurements_df(
        self,
        resort: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get measurements as a pandas DataFrame.

        Args:
            resort: Filter by resort name
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            DataFrame with measurements
        """
        measurements = self.db.get_measurements(
            resort=resort,
            start_date=start_date,
            end_date=end_date
        )

        if not measurements:
            return pd.DataFrame()

        df = pd.DataFrame(measurements)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        return df

    def calculate_accumulation(
        self,
        resort: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Calculate snow accumulation between two times.

        Args:
            resort: Resort name
            start_time: Start datetime
            end_time: End datetime

        Returns:
            Dictionary with accumulation statistics
        """
        df = self.get_measurements_df(
            resort=resort,
            start_date=start_time,
            end_date=end_time
        )

        if df.empty or 'snow_depth_inches' not in df.columns:
            return {
                'resort': resort,
                'start_time': start_time,
                'end_time': end_time,
                'accumulation_inches': None,
                'measurements_count': 0,
                'error': 'No measurements found'
            }

        # Filter out null measurements
        valid_df = df[df['snow_depth_inches'].notna()].copy()

        if valid_df.empty:
            return {
                'resort': resort,
                'start_time': start_time,
                'end_time': end_time,
                'accumulation_inches': None,
                'measurements_count': len(df),
                'error': 'No valid depth measurements'
            }

        # Get first and last valid measurements
        first_depth = valid_df.iloc[0]['snow_depth_inches']
        last_depth = valid_df.iloc[-1]['snow_depth_inches']

        accumulation = last_depth - first_depth

        return {
            'resort': resort,
            'start_time': start_time,
            'end_time': end_time,
            'start_depth_inches': first_depth,
            'end_depth_inches': last_depth,
            'accumulation_inches': accumulation,
            'measurements_count': len(valid_df),
            'avg_confidence': valid_df['confidence_score'].mean(),
            'time_span_hours': (end_time - start_time).total_seconds() / 3600
        }

    def calculate_daily_accumulation(
        self,
        resort: str,
        date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Calculate accumulation for a single day (midnight to midnight).

        Args:
            resort: Resort name
            date: Date to analyze (defaults to yesterday)

        Returns:
            Dictionary with daily accumulation statistics
        """
        if date is None:
            date = datetime.now().date() - timedelta(days=1)
        elif isinstance(date, datetime):
            date = date.date()

        start_time = datetime.combine(date, datetime.min.time())
        end_time = datetime.combine(date, datetime.max.time())

        return self.calculate_accumulation(resort, start_time, end_time)

    def calculate_storm_total(
        self,
        resort: str,
        storm_start: datetime,
        storm_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Calculate total accumulation during a storm period.

        Args:
            resort: Resort name
            storm_start: Storm start datetime
            storm_end: Storm end datetime (defaults to now)

        Returns:
            Dictionary with storm accumulation statistics
        """
        if storm_end is None:
            storm_end = datetime.now()

        result = self.calculate_accumulation(resort, storm_start, storm_end)
        result['storm_duration_hours'] = result['time_span_hours']

        return result

    def get_hourly_averages(
        self,
        resort: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get hourly averaged snow depth readings.

        For each hour, averages all valid readings within that hour interval.

        Args:
            resort: Resort name
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            DataFrame with hourly averaged depth data
        """
        df = self.get_measurements_df(resort, start_date, end_date)

        if df.empty or 'snow_depth_inches' not in df.columns:
            return pd.DataFrame()

        # Filter to valid measurements only
        valid_df = df[df['snow_depth_inches'].notna()].copy()

        if valid_df.empty:
            return pd.DataFrame()

        # Set timestamp as index and resample to hourly, taking mean of all readings
        valid_df = valid_df.set_index('timestamp')
        hourly = valid_df['snow_depth_inches'].resample('h').mean()

        # Also get count of readings per hour
        hourly_count = valid_df['snow_depth_inches'].resample('h').count()

        # Calculate hourly change
        hourly_change = hourly.diff()

        result_df = pd.DataFrame({
            'hour': hourly.index,
            'avg_depth_inches': hourly.values,
            'readings_count': hourly_count.values,
            'hourly_change_inches': hourly_change.values
        })

        # Drop rows with no readings
        result_df = result_df[result_df['readings_count'] > 0]

        return result_df

    def get_daily_summary(
        self,
        resort: str,
        date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get daily summary with hourly averages and min/max with timestamps.

        Args:
            resort: Resort name
            date: Date to analyze (defaults to today)

        Returns:
            Dictionary with daily summary including:
            - hourly_readings: list of {hour, avg_depth, readings_count}
            - day_max: {depth, latest_time} - max depth and latest time it occurred
            - day_min: {depth, latest_time} - min depth and latest time it occurred
        """
        if date is None:
            date = datetime.now().date()
        elif isinstance(date, datetime):
            date = date.date()

        start_time = datetime.combine(date, datetime.min.time())
        end_time = datetime.combine(date, datetime.max.time())

        hourly_df = self.get_hourly_averages(resort, start_time, end_time)

        if hourly_df.empty:
            return {
                'resort': resort,
                'date': str(date),
                'hourly_readings': [],
                'day_max': None,
                'day_min': None,
                'total_readings': 0,
                'error': 'No valid measurements for this day'
            }

        # Calculate hourly deltas (change from previous hour)
        hourly_df = hourly_df.sort_values('hour').reset_index(drop=True)
        hourly_df['delta'] = hourly_df['avg_depth_inches'].diff()

        # Build hourly readings list with deltas
        hourly_readings = []
        for _, row in hourly_df.iterrows():
            delta = row['delta'] if pd.notna(row['delta']) else None
            hourly_readings.append({
                'hour': row['hour'].strftime('%H:%M'),
                'hour_timestamp': row['hour'].isoformat(),
                'avg_depth_inches': round(row['avg_depth_inches'], 2),
                'readings_count': int(row['readings_count']),
                'delta_inches': round(delta, 2) if delta is not None else None
            })

        # Calculate accumulation: sum of positive deltas only
        # This accounts for stake clearing (negative delta) by ignoring it
        positive_deltas = hourly_df['delta'][hourly_df['delta'] > 0]
        accumulation = positive_deltas.sum() if len(positive_deltas) > 0 else 0.0

        # Find max depth and the LATEST time it occurred
        max_depth = hourly_df['avg_depth_inches'].max()
        max_rows = hourly_df[hourly_df['avg_depth_inches'] == max_depth]
        latest_max_time = max_rows['hour'].max()  # Latest occurrence of max

        # Find min depth and the LATEST time it occurred
        min_depth = hourly_df['avg_depth_inches'].min()
        min_rows = hourly_df[hourly_df['avg_depth_inches'] == min_depth]
        latest_min_time = min_rows['hour'].max()  # Latest occurrence of min

        return {
            'resort': resort,
            'date': str(date),
            'hourly_readings': hourly_readings,
            'day_max': {
                'depth_inches': round(max_depth, 2),
                'latest_time': latest_max_time.strftime('%H:%M'),
                'latest_timestamp': latest_max_time.isoformat()
            },
            'day_min': {
                'depth_inches': round(min_depth, 2),
                'latest_time': latest_min_time.strftime('%H:%M'),
                'latest_timestamp': latest_min_time.isoformat()
            },
            'accumulation_inches': round(accumulation, 2),
            'total_readings': int(hourly_df['readings_count'].sum()),
            'hours_with_data': len(hourly_df)
        }

    def get_hourly_accumulation(
        self,
        resort: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get hourly accumulation rates (legacy method, use get_hourly_averages instead).

        Args:
            resort: Resort name
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            DataFrame with hourly accumulation data
        """
        return self.get_hourly_averages(resort, start_date, end_date)

    def get_accumulation_summary(
        self,
        resort: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get a summary of accumulation over the past N days.

        Args:
            resort: Resort name
            days: Number of days to analyze

        Returns:
            Dictionary with accumulation summary
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = self.get_measurements_df(resort, start_date, end_date)

        if df.empty or 'snow_depth_inches' not in df.columns:
            return {
                'resort': resort,
                'days': days,
                'error': 'No measurements found'
            }

        valid_df = df[df['snow_depth_inches'].notna()]

        if valid_df.empty:
            return {
                'resort': resort,
                'days': days,
                'error': 'No valid measurements'
            }

        # Overall stats
        total_accumulation = (
            valid_df.iloc[-1]['snow_depth_inches'] -
            valid_df.iloc[0]['snow_depth_inches']
        )

        # Daily accumulations
        daily_accumulations = []
        for day in range(days):
            date = (end_date - timedelta(days=day)).date()
            daily = self.calculate_daily_accumulation(resort, date)
            if daily.get('accumulation_inches') is not None:
                daily_accumulations.append({
                    'date': str(date),
                    'accumulation_inches': daily['accumulation_inches']
                })

        return {
            'resort': resort,
            'period_days': days,
            'start_date': str(start_date.date()),
            'end_date': str(end_date.date()),
            'total_accumulation_inches': total_accumulation,
            'current_depth_inches': valid_df.iloc[-1]['snow_depth_inches'],
            'measurements_count': len(valid_df),
            'avg_confidence': valid_df['confidence_score'].mean(),
            'daily_accumulations': daily_accumulations,
            'biggest_day': max(
                daily_accumulations,
                key=lambda x: x['accumulation_inches']
            ) if daily_accumulations else None
        }

    def export_to_csv(
        self,
        resort: str,
        output_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """Export measurements to CSV file.

        Args:
            resort: Resort name
            output_path: Path for output CSV file
            start_date: Optional start date filter
            end_date: Optional end date filter
        """
        df = self.get_measurements_df(resort, start_date, end_date)

        if df.empty:
            raise ValueError(f"No measurements found for {resort}")

        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} measurements to {output_path}")

    def get_all_resorts_summary(self) -> List[Dict[str, Any]]:
        """Get current snow depth for all resorts with measurements.

        Returns:
            List of dictionaries with resort summaries
        """
        # Get all measurements to find unique resorts
        all_measurements = self.db.get_measurements()

        if not all_measurements:
            return []

        # Get unique resorts
        resorts = set(m['resort'] for m in all_measurements)

        summaries = []
        for resort in sorted(resorts):
            latest = self.db.get_latest_measurement(resort)
            if latest:
                summaries.append({
                    'resort': resort,
                    'last_measurement': latest['timestamp'],
                    'snow_depth_inches': latest['snow_depth_inches'],
                    'confidence_score': latest['confidence_score'],
                    'stake_visible': latest['stake_visible']
                })

        return summaries


def generate_report(
    resort: str,
    output_file: Optional[str] = None,
    days: int = 7,
    db_path: Optional[str] = None
) -> str:
    """Generate a text report of snow accumulation.

    Args:
        resort: Resort name
        output_file: Optional path to save report
        days: Number of days to include
        db_path: Optional database path

    Returns:
        Report text
    """
    analytics = SnowAnalytics(db_path)
    summary = analytics.get_accumulation_summary(resort, days)

    if 'error' in summary:
        report = f"Error generating report for {resort}: {summary['error']}"
    else:
        report = f"""
Snow Accumulation Report - {resort}
{'=' * 50}

Period: {summary['start_date']} to {summary['end_date']} ({days} days)
Current Depth: {summary['current_depth_inches']:.1f} inches
Total Accumulation: {summary['total_accumulation_inches']:.1f} inches
Measurements: {summary['measurements_count']}
Average Confidence: {summary['avg_confidence']:.2f}

Daily Accumulation:
{'-' * 50}
"""
        for day in summary['daily_accumulations']:
            report += f"{day['date']}: {day['accumulation_inches']:+.1f} inches\n"

        if summary['biggest_day']:
            report += f"\nBiggest Day: {summary['biggest_day']['date']} "
            report += f"({summary['biggest_day']['accumulation_inches']:+.1f} inches)\n"

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")

    return report
