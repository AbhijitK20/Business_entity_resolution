#!/bin/bash
# rescue_queue.sh — waits for the Phone_Backup rescue to finish, then rescues
# the `plugins` folder from the failing drive. Runs as user (no sudo needed:
# source is mounted read-only, destination is world-writable).
#
# Run:  bash scripts/rescue_queue.sh
# Log:  /tmp/rescue_queue.log

set +e
SRC="/mnt/hdd"
DEST="/run/media/abhijitk20/Windows"
PB_DEST="$DEST/Phone_Backup_rescue"
PLUGINS_DEST="$DEST/plugins_rescue"
LOG="/tmp/rescue_queue.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== RESCUE QUEUE STARTED ==="

# ---------------------------------------------------------------------------
# 1. Wait for Phone_Backup streams (mine + root's) to finish
# ---------------------------------------------------------------------------
log "Waiting for Phone_Backup rescue to finish..."
while pgrep -f "rsync .*Phone_Backup" > /dev/null 2>&1; do
    copied=$(du -sh "$PB_DEST" 2>/dev/null | cut -f1)
    log "  Phone_Backup: $copied copied — still running"
    sleep 300
done
log "Phone_Backup rescue streams finished. Size: $(du -sh "$PB_DEST" 2>/dev/null | cut -f1)"

# ---------------------------------------------------------------------------
# 2. Verify no missing Phone_Backup files (against the original list)
# ---------------------------------------------------------------------------
if [ -f /tmp/rescue_files.txt ]; then
    missing=0
    while IFS= read -r f; do
        [ ! -f "$PB_DEST/$f" ] && missing=$((missing + 1))
    done < /tmp/rescue_files.txt
    log "Phone_Backup missing files: $missing"
fi

# ---------------------------------------------------------------------------
# 3. Check space before starting the plugins rescue
# ---------------------------------------------------------------------------
free_gb=$(df -BG "$DEST" | tail -1 | awk '{gsub("G","",$4); print $4}')
log "Free space on destination: ${free_gb}G"

log "=== STARTING PLUGINS RESCUE ==="
mkdir -p "$PLUGINS_DEST"
nice -n 10 rsync -rt --ignore-existing --timeout=120 \
    "$SRC/plugins/" "$PLUGINS_DEST/" >> "$LOG" 2>&1
rc=$?
log "Plugins rsync finished (exit $rc). Size: $(du -sh "$PLUGINS_DEST" 2>/dev/null | cut -f1)"

# ---------------------------------------------------------------------------
# 4. Also rescue amazon_ml datasets if not already safe elsewhere
#    (we already copied them to SSD; this is belt-and-braces)
# ---------------------------------------------------------------------------
if [ ! -d "$DEST/amazon_ml_rescue" ]; then
    log "=== STARTING AMAZON_ML RESCUE (backup copy) ==="
    nice -n 10 rsync -rt --ignore-existing --timeout=120 \
        "$SRC/amazon_ml/" "$DEST/amazon_ml_rescue/" >> "$LOG" 2>&1
    log "amazon_ml rsync finished. Size: $(du -sh "$DEST/amazon_ml_rescue" 2>/dev/null | cut -f1)"
fi

log "=== RESCUE QUEUE COMPLETE ==="
log "Phone_Backup: $(du -sh "$PB_DEST" 2>/dev/null | cut -f1)"
log "plugins:      $(du -sh "$PLUGINS_DEST" 2>/dev/null | cut -f1)"
log "Free space:   $(df -h "$DEST" | tail -1 | awk '{print $4}')"
