param(
    [string]$ManifestPath = "tools\\visual_suite.json",
    [string]$PhaseName = "phase1_phase2",
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$OutputRoot = "artifacts\\visual_suite",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if ([System.IO.Path]::IsPathRooted($ManifestPath)) {
    $resolvedManifestPath = [System.IO.Path]::GetFullPath($ManifestPath)
} else {
    $resolvedManifestPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $ManifestPath))
}

if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
} else {
    $resolvedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputRoot))
}

if (-not (Test-Path $resolvedManifestPath)) {
    throw "Visual suite manifest not found at $resolvedManifestPath"
}

$phaseOutputDir = Join-Path $resolvedOutputRoot $PhaseName
if (Test-Path $phaseOutputDir) {
    Get-ChildItem -Path $phaseOutputDir -Force | Remove-Item -Force -Recurse
} else {
    New-Item -ItemType Directory -Path $phaseOutputDir | Out-Null
}

$manifest = Get-Content $resolvedManifestPath -Raw | ConvertFrom-Json
if (-not $manifest.scenes -or $manifest.scenes.Count -eq 0) {
    throw "Visual suite manifest contains no scenes."
}

$captureScript = Join-Path $PSScriptRoot "capture_repro.ps1"
if (-not (Test-Path $captureScript)) {
    throw "Capture script not found at $captureScript"
}

function Format-CapturePlacements {
    param(
        [Parameter(ValueFromPipeline = $true)]
        $Placements
    )

    if (-not $Placements -or $Placements.Count -eq 0) {
        return $null
    }

    $entries = New-Object System.Collections.Generic.List[string]
    foreach ($placement in $Placements) {
        if (-not $placement.target -or -not $placement.face -or -not $placement.block) {
            throw "Each placement entry must include target, face, and block."
        }

        $target = @(
            [int]$placement.target[0],
            [int]$placement.target[1],
            [int]$placement.target[2]
        ) -join ","
        $face = @(
            [int]$placement.face[0],
            [int]$placement.face[1],
            [int]$placement.face[2]
        ) -join ","
        $entries.Add("$target|$face|$($placement.block)")
    }

    return ($entries -join ";")
}

$captureRows = New-Object System.Collections.Generic.List[psobject]
$builtOnce = $false

foreach ($scene in $manifest.scenes) {
    $sceneName = [string]$scene.name
    if ([string]::IsNullOrWhiteSpace($sceneName)) {
        throw "Each scene must define a non-empty name."
    }

    $outputPath = Join-Path $phaseOutputDir "$sceneName.bmp"
    $placements = $null
    if ($scene.PSObject.Properties.Name -contains "placements") {
        $placements = Format-CapturePlacements $scene.placements
    }

    $captureParams = @{
        X = [double]$scene.position.x
        Y = [double]$scene.position.y
        Z = [double]$scene.position.z
        BuildDir = $BuildDir
        Config = $Config
        OutputPath = $outputPath
        SettleFrames = [int]$scene.settleFrames
        TimeOfDay = [double]$scene.timeOfDay
        NearChunks = [int]$scene.nearChunks
        FarBlocks = [int]$scene.farBlocks
        FogStartBlocks = [int]$scene.fogStartBlocks
        FarTerrainEnabled = [bool]$scene.farTerrainEnabled
        DebugView = [int]$scene.debugView
        DirectSun = [bool]$scene.directSun
        KeepOutputDir = $true
    }

    if ($scene.PSObject.Properties.Name -contains "lookAt") {
        $captureParams["LookX"] = [double]$scene.lookAt.x
        $captureParams["LookY"] = [double]$scene.lookAt.y
        $captureParams["LookZ"] = [double]$scene.lookAt.z
    } else {
        $captureParams["Yaw"] = [double]$scene.yaw
        $captureParams["Pitch"] = [double]$scene.pitch
    }

    if ($placements) {
        $captureParams["CapturePlacements"] = $placements
    }

    if ($SkipBuild -or $builtOnce) {
        $captureParams["SkipBuild"] = $true
    }

    & $captureScript @captureParams

    $builtOnce = $true
    $captureRows.Add([pscustomobject]@{
        name = $sceneName
        description = [string]$scene.description
        x = [double]$scene.position.x
        y = [double]$scene.position.y
        z = [double]$scene.position.z
        yaw = if ($scene.PSObject.Properties.Name -contains "yaw") { [double]$scene.yaw } else { "" }
        pitch = if ($scene.PSObject.Properties.Name -contains "pitch") { [double]$scene.pitch } else { "" }
        look_x = if ($scene.PSObject.Properties.Name -contains "lookAt") { [double]$scene.lookAt.x } else { "" }
        look_y = if ($scene.PSObject.Properties.Name -contains "lookAt") { [double]$scene.lookAt.y } else { "" }
        look_z = if ($scene.PSObject.Properties.Name -contains "lookAt") { [double]$scene.lookAt.z } else { "" }
        time_of_day = [double]$scene.timeOfDay
        near_chunks = [int]$scene.nearChunks
        far_blocks = [int]$scene.farBlocks
        fog_start_blocks = [int]$scene.fogStartBlocks
        far_terrain_enabled = [bool]$scene.farTerrainEnabled
        direct_sun = [bool]$scene.directSun
        debug_view = [int]$scene.debugView
        exact_chunks_only = if ($scene.PSObject.Properties.Name -contains "exactChunksOnly") {
            [bool]$scene.exactChunksOnly
        } else {
            $false
        }
        placements = if ($placements) { $placements } else { "" }
        output = [System.IO.Path]::GetFileName($outputPath)
    })
}

$manifestCopyPath = Join-Path $phaseOutputDir "visual_suite.json"
Copy-Item -Path $resolvedManifestPath -Destination $manifestCopyPath -Force

$capturesCsvPath = Join-Path $phaseOutputDir "captures.csv"
$captureRows | Export-Csv -Path $capturesCsvPath -NoTypeInformation

Write-Host "Visual suite captures written to $phaseOutputDir"
