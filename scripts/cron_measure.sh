#!/bin/bash
# Cron wrapper for snow measurement script
# Runs inside Docker container where dependencies are available

LOG_DIR="/home/trent/snowcammeasurement/logs"
LOG_FILE="$LOG_DIR/measure_$(date +%Y%m).log"

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Run measurement script inside Docker container
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_FILE"

docker exec snow_frontend python3 /app/scripts/measure_new_images.py --verbose >> "$LOG_FILE" 2>&1

echo "" >> "$LOG_FILE"

# Keep only last 7 days of detailed logs (compress older)
find "$LOG_DIR" -name "measure_*.log" -mtime +7 -exec gzip {} \; 2>/dev/null

# Keep only last 90 days of compressed logs
find "$LOG_DIR" -name "measure_*.log.gz" -mtime +90 -delete 2>/dev/null
