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
Clear-Content $log -ErrorAction SilentlyContinue

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
if (-not (Test-Path 'W:\')) { throw 'drive W: was not assigned' }
Log "vhd created and FAT32-formatted (windows format)"

# ---- 2. ground truth + manifest -------------------------------------------
Copy-Item -Recurse -Force (Join-Path $gt '*') 'W:\'
Copy-Item -Force (Join-Path $ScenarioDir 'manifest.json') 'W:\'
Log "groundtruth copied: $((Get-ChildItem -Recurse 'W:\').Count) objects"

# ---- 3. doomed files (delete, names become ghosts) ------------------------
foreach ($f in $doomed) {
    if (Test-Path "W:\$f") { Remove-Item -Force "W:\$f"; Log "deleted pre-format: $f" }
}

# ---- 4. pre-format snapshot + QUICK FORMAT (the destructive event) ---------
# Windows may sanitize (UNMAP -> zero) the whole volume even in quick mode;
# the snapshot is what tools/scenario.py composite layers back over the
# zeroed clusters (data only — FATs/root keep their format semantics).
$prefmt = Join-Path $OutDir 'image-prefmt.vfat'
@"
select vdisk file="$vhd"
detach vdisk
"@ | diskpart
if ($LASTEXITCODE -ne 0) { throw 'diskpart (detach for snapshot) failed' }
Start-Sleep -Seconds 2

$vhdSize = (Get-Item $vhd).Length
$rawSize = $vhdSize - 512
function Copy-RawVhd([string]$src, [string]$dst, [long]$bytes) {
    $in = [IO.File]::OpenRead($src)
    $outf = [IO.File]::Create($dst)
    $buf = New-Object byte[] (8MB)
    $left = $bytes
    while ($left -gt 0) {
        $n = $in.Read($buf, 0, [Math]::Min($buf.Length, $left))
        if ($n -le 0) { break }
        $outf.Write($buf, 0, $n)
        $left -= $n
    }
    $in.Close(); $outf.Close()
}
Copy-RawVhd $vhd $prefmt $rawSize
Log "pre-format snapshot written: $prefmt ($rawSize bytes)"

@"
select vdisk file="$vhd"
attach vdisk
select partition 1
assign letter=W
"@ | diskpart
if ($LASTEXITCODE -ne 0) { throw 'diskpart (re-attach) failed' }
Start-Sleep -Seconds 2
if (-not (Test-Path 'W:\')) { throw 'drive W: missing after re-attach' }

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
Copy-RawVhd $vhd $image $rawSize
Remove-Item -Force $vhd
Log "image.vfat written: $rawSize bytes (vhd minus 512B footer)"
Log "recipe done: $(Get-Date -Format o)"
