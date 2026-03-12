param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$OutputDir = "$(Join-Path $PSScriptRoot '..\\artifacts\\lod_horizon_sweep')",
    [switch]$SkipBuild,
    [int]$ExactChunks = 48,
    [int]$TotalChunks = 128,
    [int]$FogStartBlocks = 1400
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

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$resolvedBuildDir = Resolve-Path (Join-Path $repoRoot $BuildDir)
if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputDir)
} else {
    $resolvedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
}

if (-not $SkipBuild) {
    $cmakePath = Resolve-CMakePath
    if ([string]::IsNullOrWhiteSpace($Config)) {
        & $cmakePath --build $resolvedBuildDir
    } else {
        & $cmakePath --build $resolvedBuildDir --config $Config
    }
}

$exePath = Resolve-ExePath -ResolvedBuildDir $resolvedBuildDir -ConfigName $Config
$exeDir = Split-Path -Parent $exePath
if (-not (Test-Path $exePath)) {
    throw "BlockGame executable not found at $exePath"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDirName = "exact${ExactChunks}_total${TotalChunks}_$timestamp"
$runDir = Join-Path $resolvedOutputRoot $runDirName
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$envKeys = @(
    "BLOCKGAME_SCREENSHOT_SWEEP",
    "BLOCKGAME_SCREENSHOT_SWEEP_DIR",
    "BLOCKGAME_CAPTURE_EXACT_CHUNKS",
    "BLOCKGAME_CAPTURE_TOTAL_CHUNKS",
    "BLOCKGAME_CAPTURE_FOG_START_BLOCKS"
)

$previousEnv = @{}
foreach ($key in $envKeys) {
    $previousEnv[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
}

try {
    $env:BLOCKGAME_SCREENSHOT_SWEEP = "1"
    $env:BLOCKGAME_SCREENSHOT_SWEEP_DIR = $runDir
    $env:BLOCKGAME_CAPTURE_EXACT_CHUNKS = [string]$ExactChunks
    $env:BLOCKGAME_CAPTURE_TOTAL_CHUNKS = [string]$TotalChunks
    $env:BLOCKGAME_CAPTURE_FOG_START_BLOCKS = [string]$FogStartBlocks

    $process = Start-Process -FilePath $exePath `
        -WorkingDirectory $exeDir `
        -PassThru `
        -Wait

    if ($process.ExitCode -ne 0) {
        throw "BlockGame exited with code $($process.ExitCode)"
    }
}
finally {
    foreach ($key in $envKeys) {
        $previous = $previousEnv[$key]
        if ([string]::IsNullOrEmpty($previous)) {
            Remove-Item "Env:$key" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($key, $previous, "Process")
        }
    }
}

$analysisScript = Join-Path $PSScriptRoot "analyze_horizon_sweep.py"
if (Test-Path $analysisScript) {
    python $analysisScript $runDir
}

Write-Host "LOD sweep screenshots written to $runDir"
