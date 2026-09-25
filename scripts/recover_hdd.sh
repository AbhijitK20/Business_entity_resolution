#!/bin/bash
# HDD RECOVERY SCRIPT — read-only where possible, minimal metadata repair only.
# Run:  sudo bash scripts/recover_hdd.sh
#
# Phase 1: health check + MFTMirr repair + read-only mount + inventory
# It never writes user data to /dev/sda2.

set +e
DEV="/dev/sda2"
MNT="/mnt/hdd"

echo "=============================================================="
echo " HDD RECOVERY — $(date)"
echo "=============================================================="

echo
echo "--- [1] SMART health (drive condition) ---"
if command -v smartctl >/dev/null 2>&1; then
    smartctl -a "$DEV" 2>&1 | grep -iE "model|capacity|health|reallocat|pending|uncorrect|error|temperature|power_on" | head -20
else
    echo "smartctl not installed — run: apt install smartmontools"
fi

echo
echo "--- [2] Current kernel errors ---"
dmesg 2>/dev/null | grep -iE "sda|I/O error|ntfs" | tail -8

echo
echo "--- [3] ntfsfix DRY RUN (no writes) ---"
ntfsfix -n "$DEV" 2>&1 | tail -12

echo
echo "--- [4] ntfsfix REAL (minimal metadata repair: MFTMirr sync + dirty flag) ---"
read -r -p "Proceed with ntfsfix repair? [y/N] " ans
if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
    ntfsfix "$DEV" 2>&1 | tail -12
else
    echo "skipped ntfsfix"
fi

echo
echo "--- [5] Mount read-only (force) ---"
mkdir -p "$MNT"
umount "$MNT" 2>/dev/null
mount -t ntfs-3g -o ro,force,noatime "$DEV" "$MNT" 2>&1 | tail -5
if mountpoint -q "$MNT"; then
    echo "✅ MOUNTED READ-ONLY at $MNT"
    echo
    echo "--- [6] Drive inventory (top-level, with sizes) ---"
    ls -la "$MNT" | head -40
    echo
    echo "--- total usage ---"
    du -sh "$MNT"/* 2>/dev/null | sort -rh | head -25
    echo
    echo "--- free space on copy targets ---"
    df -h / "/run/media/abhijitk20/Windows" 2>/dev/null | tail -3
else
    echo "❌ MOUNT FAILED — next step is ntfsclone --rescue (see notes)"
    echo
    echo "--- [6b] ntfsclone rescue attempt (used clusters only) ---"
    TARGET="/run/media/abhijitk20/Windows/hdd_rescue.img"
    echo "Would write to: $TARGET"
    read -r -p "Run ntfsclone --rescue now? [y/N] " ans2
    if [ "$ans2" = "y" ] || [ "$ans2" = "Y" ]; then
        ntfsclone --rescue --overwrite "$TARGET" "$DEV" 2>&1 | tail -15
    fi
fi

echo
echo "=============================================================="
echo " DONE. Paste this whole output back."
echo "=============================================================="
