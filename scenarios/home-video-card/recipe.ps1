# home-video-card scenario — build the image THE WINDOWS WAY.
#
# Runs on windows-latest (CI) or any admin PowerShell. No dosfstools/mtools:
# the FAT32 filesystem is produced by Windows' own format, so the on-disk
# personality (reserved sector count, FSInfo, backup boot, timestamping,
# cluster allocation patterns) matches what real recovery sees.
#
# Sequence (mirrors the Manon case):
#   1. fixed VHD, NT-style init, ONE full FAT32 format
#   2. copy groundtruth/ + manifest.json (the exact content contract)
#   3. delete the doomed files (surviving names = metadata ghosts)
#   4. QUICK FORMAT (the destructive event under test)
#   5. remount, copy postformat/ (new usage clobbers the old data area)
#   6. detach, trim the VHD footer -> image.vfat (raw, FATality-readable)
#
param(
    [string]$ScenarioDir = "$PSScriptRoot",
    [string]$OutDir = "$ScenarioDir",
    [int]$SizeMB = 256
)

$ErrorActionPreference = 'Stop'
$image   = Join-Path $OutDir 'image.vfat'
$vhd     = Join-Path $OutDir 'image.vhd'
$gt      = Join-Path $ScenarioDir 'groundtruth'
$pf      = Join-Path $ScenarioDir 'postformat'
$log     = Join-Path $OutDir 'recipe-log.txt'

# files deleted BEFORE the format: their names must survive as ghosts
$doomed  = @('DCIM\100GOPRO\G0013657.JPG')

function Log($m) { $m | Tee-Object -FilePath $log -Append }

Log "recipe start: $(Get-Date -Format o)"

# ---- 1. fixed VHD + partition + format ------------------------------------
@"
create vdisk file="$vhd" maximum=$SizeMB type=fixed
select vdisk file="$vhd"
attach vdisk
convert mbr
create partition primary
select partition 1
active
format fs=fat32 label=SCENARIO quick
assign letter=W
"@ | diskpart
if ($LASTEXITCODE -ne 0) { throw "diskpart (create/format) failed" }
Start-Sleep -Seconds 2

# SD-card semantics: Windows' format/delete issues a sanitize UNMAP across
# the partition and the VHD driver turns discard into zero writes — which
# would erase everything quick format is supposed to leave behind (real SD
# cards keep the data; that is the whole premise of recovery). Disable
# delete-notify (TRIM) so format/delete only touch metadata, like on SD.
fsutil behavior set disabledeletenotify 1
if ($LASTEXITCODE -ne 0) { throw "fsutil setDisableDeleteNotify failed" }
fsutil behavior query disabledeletenotify
Log "vhd created and FAT32-formatted (windows format, TRIM disabled)"

if (-not (Test-Path 'W:\')) { throw 'drive W: was not assigned' }

# ---- 2. ground truth + manifest -------------------------------------------
Copy-Item -Recurse -Force (Join-Path $gt '*') 'W:\'
Copy-Item -Force (Join-Path $ScenarioDir 'manifest.json') 'W:\'
Log "groundtruth copied: $((Get-ChildItem -Recurse 'W:\').Count) objects"

# ---- 3. doomed files (delete, names become ghosts) ------------------------
foreach ($f in $doomed) {
    if (Test-Path "W:\$f") { Remove-Item -Force "W:\$f"; Log "deleted pre-format: $f" }
}

# ---- 4. QUICK FORMAT (the destructive event under test) -------------------
@"
select vdisk file="$vhd"
select partition 1
format fs=fat32 label=SCENARIO quick
assign letter=W
"@ | diskpart
if ($LASTEXITCODE -ne 0) { throw "diskpart (quick format) failed" }
Start-Sleep -Seconds 2
Log "QUICK FORMAT done"

# ---- 5. post-format usage (clobbers old data) -----------------------------
Copy-Item -Recurse -Force (Join-Path $pf '*') 'W:\'
Log "postformat copied"

# ---- 6. detach and trim to raw -------------------------------------------
@"
select vdisk file="$vhd"
detach vdisk
"@ | diskpart
if ($LASTEXITCODE -ne 0) { throw "diskpart (detach) failed" }
Start-Sleep -Seconds 2

$vhdSize = (Get-Item $vhd).Length
$rawSize = $vhdSize - 512          # fixed VHD footer is the last 512 bytes
$fsIn  = [IO.File]::OpenRead($vhd)
$fsOut = [IO.File]::Create($image)
$fsIn.Seek(0, 'Begin') | Out-Null
$buf = New-Object byte[] (8MB)
$left = $rawSize
while ($left -gt 0) {
    $n = $fsIn.Read($buf, 0, [Math]::Min($buf.Length, $left))
    if ($n -le 0) { break }
    $fsOut.Write($buf, 0, $n)
    $left -= $n
}
$fsIn.Close(); $fsOut.Close()
Remove-Item -Force $vhd
Log "image.vfat written: $rawSize bytes (vhd $vhdSize minus 512B footer)"
Log "recipe done: $(Get-Date -Format o)"
