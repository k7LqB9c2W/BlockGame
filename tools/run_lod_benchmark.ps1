param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$OutputDir = "$(Join-Path $PSScriptRoot '..\\artifacts\\lod_benchmark')",
    [switch]$SkipBuild,
    [int]$ExactChunks = 48,
    [int]$TotalChunks = 128,
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
    "-FogStartBlocks", [string]$FogStartBlocks,
    "-NotRespondingSeconds", [string]$NotRespondingSeconds,
    "-PostWriteGraceSeconds", [string]$PostWriteGraceSeconds,
    "-PollMilliseconds", [string]$PollMilliseconds
)
if ($SkipBuild) {
    $arguments += "-SkipBuild"
}

powershell @arguments
