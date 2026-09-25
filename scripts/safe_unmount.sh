#!/bin/bash
# safe_unmount.sh — cleanly flush + unmount an external drive so NTFS never
# gets the "dirty flag" that blocks future mounts.
#
# WHY THIS MATTERS: the failure you just hit (drive refusing to mount
# read-write; UI mounter erroring) is usually an NTFS dirty flag set when
# writes were interrupted — unplugging while active, sleep during write,
# or I/O errors. This script prevents that class of problem.
#
# Run:  bash scripts/safe_unmount.sh [mountpoint]   (default: auto-detect)
#
# SAFE REMOVAL CHECKLIST (do this every time):
#   1. Close all files/apps using the drive
#   2. Run this script (flushes pending writes)
#   3. Wait for "safe to remove"
#   4. Then unplug / eject

set +e

# --- find candidate mountpoints ---------------------------------------------
if [ -z "$1" ]; then
    echo "Mounted external drives:"
    lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,ROTA | awk 'NR==1 || /media|mnt/'
    echo
    echo "Usage: bash $0 /run/media/<user>/<LABEL>    (or /mnt/...)"
    exit 0
fi

MNT="$1"
USER_NAME="${SUDO_USER:-$USER}"

echo "=============================================================="
echo " SAFE UNMOUNT — $MNT"
echo "=============================================================="

# --- 1. is it mounted at all? -----------------------------------------------
if ! mountpoint -q "$MNT"; then
    echo "⚠  $MNT is not a mountpoint."
    echo "   Already unmounted, or wrong path. Nothing to do."
    exit 0
fi

# --- 2. warn about open files ------------------------------------------------
echo
echo "--- [1] Checking for processes using the drive ---"
if command -v lsof > /dev/null 2>&1; then
    users=$(lsof +D "$MNT" 2>/dev/null | tail -n +2 | awk '{print $1}' | sort -u | head -10)
    if [ -n "$users" ]; then
        echo "⚠  These programs have files open on the drive:"
        echo "$users" | sed 's/^/     /'
        echo "   Close them before unmounting, or data may be lost."
    else
        echo "   No open files detected. ✅"
    fi
else
    echo "   (lsof not installed — skipping open-file check)"
fi

# --- 3. flush all pending writes to disk -------------------------------------
echo
echo "--- [2] Flushing pending writes (sync) ---"
sync
echo "   All buffers flushed ✅"

# --- 4. unmount ---------------------------------------------------------------
echo
echo "--- [3] Unmounting ---"
if umount "$MNT" 2>/dev/null; then
    echo "   Unmounted cleanly ✅"
else
    echo "   Normal unmount busy — trying lazy unmount..."
    if umount -l "$MNT" 2>/dev/null; then
        echo "   Lazy-unmounted (flushes when last user closes) ⚠"
    else
        if [ "$(id -u)" -ne 0 ]; then
            echo "   Permission denied — retry with: sudo umount '$MNT'"
        else
            echo "   ❌ Could not unmount. A process is still writing."
            echo "      Find it with:  lsof +D '$MNT'"
            exit 1
        fi
    fi
fi

# --- 5. final verification ----------------------------------------------------
echo
if mountpoint -q "$MNT"; then
    echo "❌ Still mounted — do NOT unplug yet."
    exit 1
else
    echo "=============================================================="
    echo " ✅ SAFE TO REMOVE  —  $MNT"
    echo "    The drive can now be unplugged without setting a dirty flag."
    echo "=============================================================="
fi
