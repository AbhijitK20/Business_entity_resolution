#!/bin/bash
# v3 — clean the dirty flag so udisks (UI) can mount, then resume the rescue copy.
# Run:  sudo bash scripts/recover_hdd_v3.sh

set +e
DEV="/dev/sdb2"
USER_NAME="${SUDO_USER:-abhijitk20}"
USER_ID=$(id -u "$USER_NAME")
DEST="/run/media/abhijitk20/Windows/Phone_Backup_rescue"

echo "=============================================================="
echo " HDD RECOVERY v3 — clean flag → UI mount → resume copy"
echo "=============================================================="

echo
echo "--- [1] Ensure nothing is mounted / no stale mounts ---"
pkill -TERM -f "rsync -a --files-from" 2>/dev/null
sleep 2
umount /mnt/hdd 2>/dev/null || umount -l /mnt/hdd 2>/dev/null
sleep 1
echo "mounts on $DEV: $(lsblk -no MOUNTPOINT $DEV | grep -c .)"

echo
echo "--- [2] ntfsfix — clear dirty flag (makes udisks/UI mount possible) ---"
ntfsfix "$DEV" 2>&1 | tail -8

echo
echo "--- [3] Try udisks mount AS YOUR USER (UI-visible) ---"
sudo -u "$USER_NAME" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$USER_ID/bus" \
    udisksctl mount -b "$DEV" 2>&1 | tail -3

NEW_MNT=$(lsblk -no MOUNTPOINT "$DEV" | grep . | head -1)

if [ -z "$NEW_MNT" ]; then
    echo
    echo "--- [3b] udisks failed → fallback: manual read-only mount ---"
    mkdir -p /mnt/hdd
    mount -t ntfs-3g -o ro,force,noatime "$DEV" /mnt/hdd 2>&1 | tail -3
    NEW_MNT="/mnt/hdd"
fi

echo
echo "--- [4] Mounted at: $NEW_MNT ---"
if ! mountpoint -q "$NEW_MNT" && [ ! -d "$NEW_MNT/Phone_Backup" ]; then
    echo "❌ No mount available — aborting"
    exit 1
fi

echo
echo "--- [5] Resume the parallel rescue copy (skips copied files) ---"
mkdir -p "$DEST"
cd "$NEW_MNT/Phone_Backup" || exit 1
find . -type f 2>/dev/null > /tmp/rescue_files.txt
echo "files: $(wc -l < /tmp/rescue_files.txt) | already copied: $(du -sh "$DEST" 2>/dev/null | cut -f1)"
rm -f /tmp/rescue_chunk_*
split -n l/6 /tmp/rescue_files.txt /tmp/rescue_chunk_
i=0
for f in /tmp/rescue_chunk_*; do
    i=$((i+1))
    setsid nohup rsync -a --files-from="$f" "$NEW_MNT/Phone_Backup/" "$DEST/" \
        > "/tmp/rescue_stream_$i.log" 2>&1 < /dev/null &
done
echo "  $i streams restarted"

echo
echo "--- [6] Progress after 30s ---"
sleep 30
echo "  copied: $(du -sh "$DEST" 2>/dev/null | cut -f1) of ~74GB"
echo "  free:   $(df -h /run/media/abhijitk20/Windows | tail -1 | awk '{print $4}')"

echo
echo "=============================================================="
echo " Mount: $NEW_MNT   (check your file manager sidebar)"
echo " Monitor: du -sh $DEST"
echo "=============================================================="
