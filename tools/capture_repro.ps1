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
    [string]$BuildDir = "build-release",
    [string]$Config = "Release",
    [string]$OutputPath = "artifacts\\repro_capture\\repro.bmp",
    [int]$SettleFrames = 20,
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
$exePath = Join-Path $resolvedBuildDir "$Config\blockgame.exe"
$exeDir = Split-Path -Parent $exePath

if (-not $SkipBuild) {
    $cmakePath = Resolve-CMakePath
    & $cmakePath --build $resolvedBuildDir --config $Config
}

if (-not (Test-Path $exePath)) {
    throw "BlockGame executable not found at $exePath"
}

if ($outputDir) {
    if (Test-Path $outputDir) {
        Get-ChildItem -Path $outputDir -Force | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
    } else {
        New-Item -ItemType Directory -Path $outputDir | Out-Null
    }
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
}

try {
    $env:BLOCKGAME_REPRO_CAPTURE = "1"
    $env:BLOCKGAME_REPRO_X = "$X"
    $env:BLOCKGAME_REPRO_Y = "$Y"
    $env:BLOCKGAME_REPRO_Z = "$Z"
    $env:BLOCKGAME_REPRO_OUTPUT = $resolvedOutputPath
    $env:BLOCKGAME_REPRO_SETTLE_FRAMES = "$SettleFrames"

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
