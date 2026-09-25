# 💾 DRIVE HEALTH — Honest Guide + Tools

**Created:** 25 Sep 2026 (after the sdb failure during the hackathon)

---

## ⚠️ First, the truth

**No script can "secure a drive from failing."** Physical failure — worn platters,
bad sectors, failing heads, degraded flash — is a hardware fact. Software cannot
undo it.

What scripts **can** do:

| Can do ✅ | Cannot do ❌ |
|-----------|-------------|
| Warn you *early* (SMART trends) | Repair bad sectors |
| Prevent the **dirty-flag** problem (safe unmount) | Restore lost data |
| Detect I/O errors before they cascade | Make a dying drive reliable |
| Automate evacuation when failure starts | Extend hardware lifespan |
| Keep a health history to spot decay | Bypass physical wear |

**The only real protection is: monitor + back up + replace on warning signs.**

---

## What happened to this drive (25 Sep 2026)

**Model:** 3.6 TB external HDD (`sdb`, NTFS)

**Symptoms:**
- Read speed collapsed to 2–4 MB/s (normal: 100+ MB/s)
- `rsync` processes stuck in `D` state (kernel waiting on failed reads)
- NTFS dirty flag set → read-write mount refused; UI mounter errored
- Fallback read-only mount (`/mnt/hdd`) was the only way in

**Diagnosis:** physical degradation — bad sectors causing read retries.
The NTFS dirty flag was a *downstream symptom*, not the root cause.

**Data status:** ✅ **Not lost.** Everything read successfully; the full
amazon_ml dataset and Phone_Backup were rescued byte-exact.

**Lesson:** we copied the dataset to SSD *just before* the drive went
read-only. An hour later and we might have lost access entirely.

---

## 🛠 The tools we built

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/drive_health_check.sh` | Full SMART report + assessment | `sudo bash scripts/drive_health_check.sh /dev/sdb` |
| `scripts/monitor_drive_health.sh` | Periodic monitoring + trend alerts + CSV history | `sudo bash scripts/monitor_drive_health.sh /dev/sdb --watch 3600` |
| `scripts/safe_unmount.sh` | Flush + clean unmount (prevents dirty flag) | `bash scripts/safe_unmount.sh /run/media/user/LABEL` |
| `scripts/rescue_queue.sh` | Automated staged evacuation | `bash scripts/rescue_queue.sh` |

### What each one protects against

```
drive_health_check.sh   → "Is this drive dying right now?"        (diagnosis)
monitor_drive_health.sh → "Is it getting worse over time?"        (early warning)
safe_unmount.sh         → "Will it mount cleanly next time?"      (prevention)
rescue_queue.sh         → "Get the data out before it dies"       (response)
```

---

## 📋 The prevention checklist (do this every time)

### Before unplugging a drive
1. Close every file/app using it
2. Run `bash scripts/safe_unmount.sh <mountpoint>`
3. Wait for **"SAFE TO REMOVE"**
4. *Then* unplug — never yank

### Daily habits for external drives
- **Never** unplug while a copy is running
- **Disable sleep during big transfers** (`systemd-inhibit` or caffeine tool) —
  a drive sleeping mid-write is a top cause of dirty flags
- Keep drives **cool and ventilated** (heat accelerates wear)
- Use a **quality powered USB hub** — under-voltage causes write errors
- Prefer **`rsync` over `cp`** for big copies — it's resumable and verifies

### Every month
```bash
sudo bash scripts/drive_health_check.sh /dev/sdX
```
Look at: `Reallocated`, `Pending`, `Uncorrectable`, `Temperature`.

### Sign up for early warning (cron)
```bash
# every 6 hours, log health
0 */6 * * * sudo bash /home/abhijitk20/Amazon\ ML/scripts/monitor_drive_health.sh /dev/sdb >> /var/log/drive_health.log 2>&1
```

---

## 🚨 Warning signs → action

| Sign | Meaning | Action |
|------|---------|--------|
| Read/write speed drops sharply | Bad-sector retries | **Back up now** |
| `Pending` sectors > 0 | Unstable sectors exist | **Rescue immediately** |
| `Uncorrectable` > 0 | Data already lost on read | **Rescue immediately** |
| `Reallocated` rising over time | Active degradation | Plan replacement |
| Clicking / beeping / vibrating | Mechanical failure | **Stop using — rescue** |
| Temperature > 55 °C | Heat stress | Improve cooling |
| Drive disappears from `lsblk` mid-use | Controller/head trouble | **Rescue immediately** |
| NTFS dirty flag repeatedly | Interrupted writes | Safe-unmount discipline |

---

## 🧯 If a drive starts failing (the exact playbook)

1. **Stop writing to it immediately.** Every write risks more damage.
2. **Mount read-only** if possible:
   ```bash
   sudo mkdir -p /mnt/hdd
   sudo mount -t ntfs-3g -o ro,force,noatime /dev/sdX2 /mnt/hdd
   ```
3. **Rescue in priority order** (most valuable, smallest files first):
   ```bash
   rsync -rt --ignore-existing --timeout=120 /mnt/hdd/important/ /safe/dest/
   ```
4. **Parallelize** 4–6 streams for big folders (multiplies throughput on most
   failing drives — though bad sectors still cap the total).
5. **Use `--timeout=120`** so a stuck bad-sector read doesn't hang the whole run.
6. **Never use `--delete`** during a rescue — you're copying *out*, not syncing.
7. **Verify afterwards:** file count + sizes, then open a few files.
8. **Replace the drive.** Do not "fix and reuse" it for anything important.

---

## 🔍 Why NTFS drives get "dirty" (and how safe_unmount prevents it)

NTFS records a **dirty flag** in its metadata when a volume is mounted
read-write. It's cleared on clean unmount. If writes are interrupted —
unplug while active, sleep mid-write, I/O error — the flag stays set, and
Linux then:
- refuses read-write mounts
- makes the desktop UI mounter fail with *"wrong fs type, bad option,
  bad superblock"*

**The flag is not corruption** — it's a "this volume needs checking"
marker. `sudo ntfsfix /dev/sdX2` clears it, but **only use ntfsfix when
you know the drive is otherwise healthy**; on a physically failing drive,
ntfsfix can't help and you should rescue read-only instead.

`safe_unmount.sh` prevents the flag by flushing buffers and unmounting
cleanly every time.

---

## 📊 Reading a SMART report

```
ID  ATTRIBUTE                 VALUE  WORST  THRESH  RAW       MEANING
 5  Reallocated_Sector_Ct      100    100    010     0         Remapped bad sectors (want 0)
197 Current_Pending_Sector     100    100    000     0         Unstable sectors (want 0!)
198 Offline_Uncorrectable      100    100    000     0         Unreadable sectors (want 0!)
199 UDMA_CRC_Error_Count      200    200    000     0         Cable/connection errors (want 0)
194 Temperature_Celsius        45     --     --      45        Current temp °C
  9 Power_On_Hours             --     --     --   12345        Total hours powered
```

**Green:** 5/197/198 = 0, temp < 50 °C.
**Yellow:** any 5 > 0, or temp 50–55 °C → back up, monitor.
**Red:** 197/198 > 0, health not "PASSED", clicking → rescue now.

---

## 🏁 Bottom line

```
BACKUP > MONITORING > REPAIR ATTEMPTS
```

A drive that has started failing **will not recover**. The scripts above buy
you **time and warning** — the actual protection is having the data somewhere
else *before* the drive dies.
