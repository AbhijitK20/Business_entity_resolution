#!/bin/bash
# HDD RECOVERY v2 — parallel, RESUMABLE rescue.
# Drive is now /dev/sdb2 (re-detected after replug).
# Safe to re-run any time — rsync skips files already copied.
#
# Run:  sudo bash scripts/recover_hdd_v2.sh

set +e
DEV="/dev/sdb2"
MNT="/mnt/hdd"
TARGET="/run/media/abhijitk20/Windows"
DEST="$TARGET/Phone_Backup_rescue"
STREAMS=6

echo "=============================================================="
echo " HDD RECOVERY v2 — parallel resumable rescue — $(date)"
echo "=============================================================="

echo
echo "--- [1] Clean stale mounts ---"
umount -l /mnt/hdd 2>/dev/null
fusermount3 -u /mnt/hdd 2>/dev/null
sleep 1

echo
echo "--- [2] Mount $DEV read-only ---"
mkdir -p "$MNT"
mount -t ntfs-3g -o ro,force,noatime "$DEV" "$MNT" 2>&1 | tail -3
if ! mountpoint -q "$MNT"; then
    echo "❌ Mount failed. Check: lsblk | grep sd"
    exit 1
fi
echo "✅ mounted"

echo
echo "--- [3] Quick sequential read speed test (200MB) ---"
BIG=$(ls -S "$MNT"/Phone_Backup/Store1_DCIM/Camera/*.mp4 2>/dev/null | sed -n '2p')
if [ -n "$BIG" ]; then
    dd if="$BIG" of=/dev/null bs=1M count=200 2>&1 | tail -1
fi

echo
echo "--- [4] Building file list ---"
cd "$MNT/Phone_Backup" || exit 1
find . -type f 2>/dev/null > /tmp/rescue_files.txt
TOTAL=$(wc -l < /tmp/rescue_files.txt)
echo "files to rescue: $TOTAL"

echo
echo "--- [5] Starting $STREAMS parallel resumable rsyncs ---"
mkdir -p "$DEST"
rm -f /tmp/rescue_chunk_*
split -n l/$STREAMS /tmp/rescue_files.txt /tmp/rescue_chunk_
i=0
for f in /tmp/rescue_chunk_*; do
    i=$((i+1))
    setsid nohup rsync -a --files-from="$f" "$MNT/Phone_Backup/" "$DEST/" \
        > "/tmp/rescue_stream_$i.log" 2>&1 < /dev/null &
    echo "  stream $i started (files: $(wc -l < "$f"))"
done

echo
echo "--- [6] Progress check in 30s ---"
sleep 30
echo "copied so far: $(du -sh "$DEST" 2>/dev/null | cut -f1)"
echo "target free:   $(df -h "$TARGET" | tail -1 | awk '{print $4}')"

echo
echo "=============================================================="
echo " COPY RUNNING IN BACKGROUND."
echo " Re-run this script anytime to resume (rsync skips copied files)."
echo " Monitor with:  du -sh $DEST"
echo "=============================================================="
