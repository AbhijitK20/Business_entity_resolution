#!/bin/bash
# drive_health_check.sh — comprehensive SMART health check for external drives.
#
# WHAT THIS CAN DO:  give early warning (SMART attributes, bad sectors, temps),
#                    detect I/O errors, and tell you when to stop trusting a disk.
# WHAT THIS CANNOT:  repair physical damage. A failing drive must be replaced.
#                    Scripts extend warning time — they don't restore hardware.
#
# Run:  sudo bash scripts/drive_health_check.sh [device]     (default: auto-detect)
# Save: logs to ~/drive_health/<device>_<date>.txt

set +e
DEVICE="${1:-}"
LOG_DIR="$HOME/drive_health"
mkdir -p "$LOG_DIR"

echo "=============================================================="
echo " DRIVE HEALTH CHECK"
echo "=============================================================="

# --- discover candidate drives (non-system, rotational) ---------------------
if [ -z "$DEVICE" ]; then
    echo
    echo "Detected external / rotational drives:"
    lsblk -d -o NAME,SIZE,TYPE,ROTA,TRAN,MODEL | awk 'NR==1 || ($4==1 && $3=="disk")'
    echo
    echo "Usage: sudo bash $0 /dev/sdX"
    echo "Example: sudo bash $0 /dev/sdb"
    exit 0
fi

BASE=$(basename "$DEVICE")
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/${BASE}_${STAMP}.txt"

if ! command -v smartctl > /dev/null 2>&1; then
    echo "smartctl not found. Install with: sudo apt install smartmontools"
    exit 1
fi

{
echo "=== Drive health report: $DEVICE — $(date) ==="
echo

echo "--- [1] Device info ---"
smartctl -i "$DEVICE" 2>&1 | grep -vE "^smartctl|^Copyright|^$"

echo
echo "--- [2] Overall health ---"
smartctl -H "$DEVICE" 2>&1 | grep -iE "result|health"

echo
echo "--- [3] Critical attributes ---"
smartctl -A "$DEVICE" 2>&1 | grep -iE \
    "Reallocated_Sector|Current_Pending|Offline_Uncorrectable|Raw_Read_Error|Spin_Retry|Reallocated_Event|Reported_Uncorrect|Command_Timeout|Temperature|Power_On_Hours|Start_Stop|Load_Cycle|Seek_Error|CRC|End-to-End"

echo
echo "--- [4] Self-test log (last 5) ---"
smartctl -l selftest "$DEVICE" 2>&1 | tail -12

echo
echo "--- [5] Error log (last 5) ---"
smartctl -l error "$DEVICE" 2>&1 | tail -12

echo
echo "--- [6] Kernel I/O errors for $BASE (dmesg) ---"
dmesg 2>/dev/null | grep -i "$BASE" | grep -iE "error|fail|reset|timeout" | tail -15
echo "(empty = none since boot)"

echo
echo "--- [7] ASSESSMENT ---"
} > "$LOG" 2>&1

# assessment logic
realloc=$(smartctl -A "$DEVICE" 2>/dev/null | awk '/Reallocated_Sector/{print $10}')
pending=$(smartctl -A "$DEVICE" 2>/dev/null | awk '/Current_Pending/{print $10}')
uncorr=$(smartctl -A "$DEVICE" 2>/dev/null | awk '/Offline_Uncorrectable/{print $10}')
health=$(smartctl -H "$DEVICE" 2>/dev/null | grep -i "result" | head -1)

{
echo "SMART health : ${health:-unknown}"
echo "Reallocated  : ${realloc:-n/a}   (>0 = sectors already remapped)"
echo "Pending      : ${pending:-n/a}   (>0 = UNSTABLE sectors waiting to fail)"
echo "Uncorrectable: ${uncorr:-n/a}   (>0 = data LOST on read)"
echo
if [ "${pending:-0}" -gt 0 ] 2>/dev/null || [ "${uncorr:-0}" -gt 0 ] 2>/dev/null; then
    echo "🔴 CRITICAL: drive has unstable/unreadable sectors."
    echo "   → Rescue all data NOW. Do not write. Replace the drive."
elif [ "${realloc:-0}" -gt 0 ] 2>/dev/null; then
    echo "🟠 WARNING: drive has remapped sectors (wear)."
    echo "   → Back up now; plan replacement; monitor weekly."
else
    echo "🟢 No critical SMART warnings on this read."
    echo "   → Keep monitoring monthly; always unmount safely."
fi
} | tee -a "$LOG"

echo
echo "Full report saved → $LOG"
