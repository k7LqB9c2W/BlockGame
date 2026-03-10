param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$OutputDir = "$(Join-Path $PSScriptRoot '..\\artifacts\\horizon_sweep')",
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
if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $resolvedOutputDir = [System.IO.Path]::GetFullPath($OutputDir)
} else {
    $resolvedOutputDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
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

if (Test-Path $resolvedOutputDir) {
    Get-ChildItem -Path $resolvedOutputDir -Force | Remove-Item -Force -Recurse
} else {
    New-Item -ItemType Directory -Path $resolvedOutputDir | Out-Null
}

$previousSweep = $env:BLOCKGAME_SCREENSHOT_SWEEP
$previousDir = $env:BLOCKGAME_SCREENSHOT_SWEEP_DIR

try {
    $env:BLOCKGAME_SCREENSHOT_SWEEP = "1"
    $env:BLOCKGAME_SCREENSHOT_SWEEP_DIR = $resolvedOutputDir

    $process = Start-Process -FilePath $exePath `
        -WorkingDirectory $exeDir `
        -PassThru `
        -Wait

    if ($process.ExitCode -ne 0) {
        throw "BlockGame exited with code $($process.ExitCode)"
    }
}
finally {
    if ($null -eq $previousSweep) {
        Remove-Item Env:BLOCKGAME_SCREENSHOT_SWEEP -ErrorAction SilentlyContinue
    } else {
        $env:BLOCKGAME_SCREENSHOT_SWEEP = $previousSweep
    }

    if ($null -eq $previousDir) {
        Remove-Item Env:BLOCKGAME_SCREENSHOT_SWEEP_DIR -ErrorAction SilentlyContinue
    } else {
        $env:BLOCKGAME_SCREENSHOT_SWEEP_DIR = $previousDir
    }
}

$analysisScript = Join-Path $PSScriptRoot "analyze_horizon_sweep.py"
if (Test-Path $analysisScript) {
    python $analysisScript $resolvedOutputDir
}

Write-Host "Fresh sweep screenshots written to $resolvedOutputDir"
