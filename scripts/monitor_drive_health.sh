#!/bin/bash
# monitor_drive_health.sh — periodic SMART monitoring with trend logging + alerts.
#
# Logs one CSV row per check to ~/drive_health/history.csv and warns when
# critical attributes worsen. Designed to run in the background or via cron.
#
# Run once:        sudo bash scripts/monitor_drive_health.sh /dev/sdb
# Watch loop:      sudo bash scripts/monitor_drive_health.sh /dev/sdb --watch 3600
# Cron example:    0 */6 * * * sudo bash /path/monitor_drive_health.sh /dev/sdb >> /var/log/drive_health.log 2>&1
#
# NOTE: needs sudo for smartctl. Add to sudoers if you want it unattended:
#   abhijitk20 ALL=(ALL) NOPASSWD: /usr/sbin/smartctl

set +e
DEVICE="${1:-}"
WATCH=0
if [ "${2:-}" = "--watch" ]; then
    WATCH="${3:-3600}"
fi

LOG_DIR="$HOME/drive_health"
HIST="$LOG_DIR/history.csv"
mkdir -p "$LOG_DIR"

if [ -z "$DEVICE" ]; then
    echo "Usage: sudo bash $0 /dev/sdX [--watch seconds]"
    echo "Example: sudo bash $0 /dev/sdb --watch 3600"
    exit 1
fi

BASE=$(basename "$DEVICE")

if ! command -v smartctl > /dev/null 2>&1; then
    echo "smartctl not found. Install with: sudo apt install smartmontools"
    exit 1
fi

[ -f "$HIST" ] || echo "timestamp,device,health,reallocated,pending,uncorrectable,temp_c,power_on_hours" > "$HIST"

check_once() {
    local health realloc pending uncorr temp hours row
    health=$(smartctl -H "$DEVICE" 2>/dev/null | grep -i "result" | awk -F: '{print $2}' | xargs)
    realloc=$(smartctl -A "$DEVICE" 2>/dev/null | awk '/Reallocated_Sector/{print $10; exit}')
    pending=$(smartctl -A "$DEVICE" 2>/dev/null | awk '/Current_Pending/{print $10; exit}')
    uncorr=$(smartctl -A "$DEVICE" 2>/dev/null | awk '/Offline_Uncorrectable/{print $10; exit}')
    temp=$(smartctl -A "$DEVICE" 2>/dev/null | awk '/Temperature_Celsius/{print $10; exit}')
    hours=$(smartctl -A "$DEVICE" 2>/dev/null | awk '/Power_On_Hours/{print $10; exit}')

    row="$(date '+%Y-%m-%d %H:%M:%S'),$BASE,${health:-?},${realloc:-0},${pending:-0},${uncorr:-0},${temp:-?},${hours:-?}"
    echo "$row" >> "$HIST"
    echo "[$(date '+%H:%M:%S')] $row"

    # --- alert logic --------------------------------------------------------
    if [ "${pending:-0}" -gt 0 ] 2>/dev/null; then
        echo "🔴 ALERT: $pending pending (unstable) sectors on $BASE — rescue data NOW."
    fi
    if [ "${uncorr:-0}" -gt 0 ] 2>/dev/null; then
        echo "🔴 ALERT: $uncorr uncorrectable sectors on $BASE — data loss occurring."
    fi
    if [ -n "$temp" ] && [ "$temp" -gt 55 ] 2>/dev/null; then
        echo "🟠 ALERT: drive temperature ${temp}°C is high (>55°C). Improve airflow."
    fi

    # --- trend detection: compare last two rows -----------------------------
    if [ "$(wc -l < "$HIST")" -gt 2 ]; then
        prev=$(tail -2 "$HIST" | head -1)
        cur=$(tail -1 "$HIST")
        p_realloc=$(echo "$prev" | cut -d, -f4)
        c_realloc=$(echo "$cur" | cut -d, -f4)
        if [ "${c_realloc:-0}" -gt "${p_realloc:-0}" ] 2>/dev/null; then
            echo "🟠 TREND: reallocated sectors rising ($p_realloc → $c_realloc)."
        fi
    fi
}

if [ "$WATCH" -gt 0 ]; then
    echo "Watching $DEVICE every ${WATCH}s (Ctrl+C to stop). CSV: $HIST"
    while true; do
        check_once
        sleep "$WATCH"
    done
else
    check_once
fi
