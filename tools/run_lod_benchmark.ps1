param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$OutputDir = "$(Join-Path $PSScriptRoot '..\\artifacts\\lod_benchmark')",
    [switch]$SkipBuild,
    [int]$ExactChunks = 48,
    [int]$TotalChunks = 128,
    [int]$MaxScenarioSeconds = 600,
    [string[]]$Scenarios = @(
        "spawn_preload",
        "full_exact_preload",
        "post_release_exact_fill",
        "straight_line_sprint",
        "turn_heavy_traversal",
        "vertical_travel"
    ),
    [int]$FogStartBlocks = 1400,
    [int]$NotRespondingSeconds = 4,
    [int]$PostWriteGraceSeconds = 5,
    [int]$PollMilliseconds = 500
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$chunkBenchmarkScript = Join-Path $PSScriptRoot "run_chunk_benchmark.ps1"
if (-not (Test-Path $chunkBenchmarkScript)) {
    throw "Missing benchmark script at $chunkBenchmarkScript"
}

$arguments = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $chunkBenchmarkScript,
    "-BuildDir", $BuildDir,
    "-Config", $Config,
    "-OutputDir", $OutputDir,
    "-ExactChunks", [string]$ExactChunks,
    "-TotalChunks", [string]$TotalChunks,
    "-MaxScenarioSeconds", [string]$MaxScenarioSeconds,
    "-FogStartBlocks", [string]$FogStartBlocks,
    "-NotRespondingSeconds", [string]$NotRespondingSeconds,
    "-PostWriteGraceSeconds", [string]$PostWriteGraceSeconds,
    "-PollMilliseconds", [string]$PollMilliseconds
)
if ($SkipBuild) {
    $arguments += "-SkipBuild"
}
if ($Scenarios.Count -gt 0) {
    $arguments += "-Scenarios"
    $arguments += $Scenarios
}

powershell @arguments
