#!/bin/bash
# Switch the HDD from the manual /mnt/hdd mount to a UI-visible udisks mount,
# so it appears in the file manager sidebar again.
# Run:  sudo bash scripts/switch_hdd_to_udisks.sh
#
# The rescue copy is resumable — we restart it from the new mount afterwards.

set +e
USER_NAME="${SUDO_USER:-abhijitk20}"
USER_ID=$(id -u "$USER_NAME")

echo "=============================================================="
echo " Switching HDD to UI-visible mount"
echo "=============================================================="

echo
echo "--- [1] Stop the running rescue copy (resumable) ---"
pkill -TERM -f "rsync -a --files-from"
sleep 4
echo "remaining rsync: $(pgrep -fc 'rsync -a --files-from' 2>/dev/null || echo 0)"

echo
echo "--- [2] Unmount the manual mount ---"
umount /mnt/hdd 2>/dev/null || umount -l /mnt/hdd 2>/dev/null
sleep 1
mountpoint -q /mnt/hdd && echo "still mounted (will retry lazy)" || echo "✅ unmounted"

echo
echo "--- [3] Mount via udisks AS YOUR USER (so it shows in the UI) ---"
sudo -u "$USER_NAME" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$USER_ID/bus" \
    udisksctl mount -b /dev/sdb2 2>&1 | tail -3

echo
echo "--- [4] Result ---"
lsblk -o NAME,SIZE,MOUNTPOINT | grep -A2 sdb
NEW_MNT=$(lsblk -no MOUNTPOINT /dev/sdb2 | head -1)
echo
echo "New mount point: ${NEW_MNT:-NONE}"
if [ -n "$NEW_MNT" ]; then
    echo "✅ Check your file manager sidebar — the drive should be back."
    echo
    echo "--- [5] Restart the rescue copy from the new mount (resumes) ---"
    DEST="/run/media/abhijitk20/Windows/Phone_Backup_rescue"
    mkdir -p "$DEST"
    cd "$NEW_MNT/Phone_Backup" || exit 1
    find . -type f 2>/dev/null > /tmp/rescue_files.txt
    rm -f /tmp/rescue_chunk_*
    split -n l/6 /tmp/rescue_files.txt /tmp/rescue_chunk_
    i=0
    for f in /tmp/rescue_chunk_*; do
        i=$((i+1))
        setsid nohup rsync -a --files-from="$f" "$NEW_MNT/Phone_Backup/" "$DEST/" \
            > "/tmp/rescue_stream_$i.log" 2>&1 < /dev/null &
    done
    echo "  $i parallel rsync streams restarted (resuming from what's copied)"
    echo "  progress: $(du -sh "$DEST" 2>/dev/null | cut -f1) of ~74GB"
else
    echo "❌ udisks mount failed — run: sudo mount -t ntfs-3g -o ro,force /dev/sdb2 /mnt/hdd"
fi

echo
echo "=============================================================="
echo " DONE"
echo "=============================================================="
