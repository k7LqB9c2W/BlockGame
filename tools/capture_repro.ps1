param(
    [Parameter(Mandatory = $true)]
    [double]$X,
    [Parameter(Mandatory = $true)]
    [double]$Y,
    [Parameter(Mandatory = $true)]
    [double]$Z,
    [double]$Yaw,
    [double]$Pitch,
    [double]$LookX,
    [double]$LookY,
    [double]$LookZ,
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$OutputPath = "artifacts\\repro_capture\\repro.bmp",
    [int]$SettleFrames = 20,
    [double]$TimeOfDay,
    [int]$NearChunks,
    [int]$FarBlocks,
    [int]$FogStartBlocks,
    [bool]$FarTerrainEnabled,
    [int]$DebugView,
    [bool]$DirectSun = $true,
    [string]$CapturePlacements,
    [switch]$KeepOutputDir,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-CMakePath {
    $cmake = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($cmake) {
        return $cmake.Source
    }

    $vsCMake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if (Test-Path $vsCMake) {
        return $vsCMake
    }

    throw "Unable to find cmake.exe."
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$resolvedBuildDir = Resolve-Path (Join-Path $repoRoot $BuildDir)
if ([System.IO.Path]::IsPathRooted($OutputPath)) {
    $resolvedOutputPath = [System.IO.Path]::GetFullPath($OutputPath)
} else {
    $resolvedOutputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
}
$outputDir = Split-Path -Parent $resolvedOutputPath
function Resolve-ExePath {
    param(
        [string]$ResolvedBuildDir,
        [string]$ConfigName
    )

    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($ConfigName)) {
        $candidates.Add((Join-Path $ResolvedBuildDir "$ConfigName\\blockgame.exe"))
    }
    $candidates.Add((Join-Path $ResolvedBuildDir "blockgame.exe"))
    $candidates.Add((Join-Path $ResolvedBuildDir "Release\\blockgame.exe"))
    $candidates.Add((Join-Path $ResolvedBuildDir "release\\blockgame.exe"))

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $candidates[0]
}

$exePath = Resolve-ExePath -ResolvedBuildDir $resolvedBuildDir -ConfigName $Config
$exeDir = Split-Path -Parent $exePath

if (-not $SkipBuild) {
    $cmakePath = Resolve-CMakePath
    if ([string]::IsNullOrWhiteSpace($Config)) {
        & $cmakePath --build $resolvedBuildDir
    } else {
        & $cmakePath --build $resolvedBuildDir --config $Config
    }
}

if (-not (Test-Path $exePath)) {
    throw "BlockGame executable not found at $exePath"
}

if ($outputDir) {
    if (Test-Path $outputDir) {
        if (-not $KeepOutputDir) {
            Get-ChildItem -Path $outputDir -Force | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
        }
    } else {
        New-Item -ItemType Directory -Path $outputDir | Out-Null
    }
}

if (Test-Path $resolvedOutputPath) {
    Remove-Item $resolvedOutputPath -Force -ErrorAction SilentlyContinue
}

$previousValues = @{
    BLOCKGAME_REPRO_CAPTURE = $env:BLOCKGAME_REPRO_CAPTURE
    BLOCKGAME_REPRO_X = $env:BLOCKGAME_REPRO_X
    BLOCKGAME_REPRO_Y = $env:BLOCKGAME_REPRO_Y
    BLOCKGAME_REPRO_Z = $env:BLOCKGAME_REPRO_Z
    BLOCKGAME_REPRO_YAW = $env:BLOCKGAME_REPRO_YAW
    BLOCKGAME_REPRO_PITCH = $env:BLOCKGAME_REPRO_PITCH
    BLOCKGAME_REPRO_LOOK_X = $env:BLOCKGAME_REPRO_LOOK_X
    BLOCKGAME_REPRO_LOOK_Y = $env:BLOCKGAME_REPRO_LOOK_Y
    BLOCKGAME_REPRO_LOOK_Z = $env:BLOCKGAME_REPRO_LOOK_Z
    BLOCKGAME_REPRO_OUTPUT = $env:BLOCKGAME_REPRO_OUTPUT
    BLOCKGAME_REPRO_SETTLE_FRAMES = $env:BLOCKGAME_REPRO_SETTLE_FRAMES
    BLOCKGAME_CAPTURE_TIME_OF_DAY = $env:BLOCKGAME_CAPTURE_TIME_OF_DAY
    BLOCKGAME_CAPTURE_NEAR_CHUNKS = $env:BLOCKGAME_CAPTURE_NEAR_CHUNKS
    BLOCKGAME_CAPTURE_FAR_BLOCKS = $env:BLOCKGAME_CAPTURE_FAR_BLOCKS
    BLOCKGAME_CAPTURE_FOG_START_BLOCKS = $env:BLOCKGAME_CAPTURE_FOG_START_BLOCKS
    BLOCKGAME_CAPTURE_FAR_TERRAIN = $env:BLOCKGAME_CAPTURE_FAR_TERRAIN
    BLOCKGAME_CAPTURE_DEBUG_VIEW = $env:BLOCKGAME_CAPTURE_DEBUG_VIEW
    BLOCKGAME_CAPTURE_DIRECT_SUN = $env:BLOCKGAME_CAPTURE_DIRECT_SUN
    BLOCKGAME_CAPTURE_PLACEMENTS = $env:BLOCKGAME_CAPTURE_PLACEMENTS
}

try {
    $env:BLOCKGAME_REPRO_CAPTURE = "1"
    $env:BLOCKGAME_REPRO_X = "$X"
    $env:BLOCKGAME_REPRO_Y = "$Y"
    $env:BLOCKGAME_REPRO_Z = "$Z"
    $env:BLOCKGAME_REPRO_OUTPUT = $resolvedOutputPath
    $env:BLOCKGAME_REPRO_SETTLE_FRAMES = "$SettleFrames"

    if ($PSBoundParameters.ContainsKey("TimeOfDay")) {
        $env:BLOCKGAME_CAPTURE_TIME_OF_DAY = "$TimeOfDay"
    } else {
        Remove-Item Env:BLOCKGAME_CAPTURE_TIME_OF_DAY -ErrorAction SilentlyContinue
    }
    if ($PSBoundParameters.ContainsKey("NearChunks")) {
        $env:BLOCKGAME_CAPTURE_NEAR_CHUNKS = "$NearChunks"
    } else {
        Remove-Item Env:BLOCKGAME_CAPTURE_NEAR_CHUNKS -ErrorAction SilentlyContinue
    }
    if ($PSBoundParameters.ContainsKey("FarBlocks")) {
        $env:BLOCKGAME_CAPTURE_FAR_BLOCKS = "$FarBlocks"
    } else {
        Remove-Item Env:BLOCKGAME_CAPTURE_FAR_BLOCKS -ErrorAction SilentlyContinue
    }
    if ($PSBoundParameters.ContainsKey("FogStartBlocks")) {
        $env:BLOCKGAME_CAPTURE_FOG_START_BLOCKS = "$FogStartBlocks"
    } else {
        Remove-Item Env:BLOCKGAME_CAPTURE_FOG_START_BLOCKS -ErrorAction SilentlyContinue
    }
    if ($PSBoundParameters.ContainsKey("FarTerrainEnabled")) {
        $env:BLOCKGAME_CAPTURE_FAR_TERRAIN = $(if ($FarTerrainEnabled) { "1" } else { "0" })
    } else {
        Remove-Item Env:BLOCKGAME_CAPTURE_FAR_TERRAIN -ErrorAction SilentlyContinue
    }
    if ($PSBoundParameters.ContainsKey("DebugView")) {
        $env:BLOCKGAME_CAPTURE_DEBUG_VIEW = "$DebugView"
    } else {
        Remove-Item Env:BLOCKGAME_CAPTURE_DEBUG_VIEW -ErrorAction SilentlyContinue
    }
    if ($PSBoundParameters.ContainsKey("DirectSun")) {
        $env:BLOCKGAME_CAPTURE_DIRECT_SUN = $(if ($DirectSun) { "1" } else { "0" })
    } else {
        Remove-Item Env:BLOCKGAME_CAPTURE_DIRECT_SUN -ErrorAction SilentlyContinue
    }
    if ($PSBoundParameters.ContainsKey("CapturePlacements") -and -not [string]::IsNullOrWhiteSpace($CapturePlacements)) {
        $env:BLOCKGAME_CAPTURE_PLACEMENTS = $CapturePlacements
    } else {
        Remove-Item Env:BLOCKGAME_CAPTURE_PLACEMENTS -ErrorAction SilentlyContinue
    }

    $hasLookTarget =
        $PSBoundParameters.ContainsKey("LookX") -and
        $PSBoundParameters.ContainsKey("LookY") -and
        $PSBoundParameters.ContainsKey("LookZ")

    if ($hasLookTarget) {
        $env:BLOCKGAME_REPRO_LOOK_X = "$LookX"
        $env:BLOCKGAME_REPRO_LOOK_Y = "$LookY"
        $env:BLOCKGAME_REPRO_LOOK_Z = "$LookZ"
        Remove-Item Env:BLOCKGAME_REPRO_YAW -ErrorAction SilentlyContinue
        Remove-Item Env:BLOCKGAME_REPRO_PITCH -ErrorAction SilentlyContinue
    } else {
        if (-not ($PSBoundParameters.ContainsKey("Yaw") -and $PSBoundParameters.ContainsKey("Pitch"))) {
            throw "Provide either -Yaw and -Pitch, or -LookX -LookY -LookZ."
        }
        $env:BLOCKGAME_REPRO_YAW = "$Yaw"
        $env:BLOCKGAME_REPRO_PITCH = "$Pitch"
        Remove-Item Env:BLOCKGAME_REPRO_LOOK_X -ErrorAction SilentlyContinue
        Remove-Item Env:BLOCKGAME_REPRO_LOOK_Y -ErrorAction SilentlyContinue
        Remove-Item Env:BLOCKGAME_REPRO_LOOK_Z -ErrorAction SilentlyContinue
    }

    $process = Start-Process -FilePath $exePath `
        -WorkingDirectory $exeDir `
        -PassThru `
        -Wait

    if ($process.ExitCode -ne 0) {
        throw "BlockGame exited with code $($process.ExitCode)"
    }
}
finally {
    foreach ($key in $previousValues.Keys) {
        if ($null -eq $previousValues[$key]) {
            Remove-Item "Env:$key" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$key" $previousValues[$key]
        }
    }
}

Write-Host "Repro screenshot written to $resolvedOutputPath"
