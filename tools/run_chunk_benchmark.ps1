param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$OutputDir = "$(Join-Path $PSScriptRoot '..\\artifacts\\chunk_benchmark')",
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

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir = Join-Path $resolvedOutputRoot $timestamp

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

New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$scenarios = @(
    "spawn_preload",
    "straight_line_sprint",
    "turn_heavy_traversal",
    "vertical_travel"
)

$envKeys = @(
    "BLOCKGAME_BENCHMARK",
    "BLOCKGAME_BENCHMARK_SCENARIO",
    "BLOCKGAME_BENCHMARK_OUTPUT",
    "BLOCKGAME_BENCHMARK_BUILD_CONFIG"
)
$previousEnv = @{}
foreach ($key in $envKeys) {
    $previousEnv[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
}

$scenarioObjects = New-Object System.Collections.Generic.List[object]

try {
    foreach ($scenario in $scenarios) {
        $scenarioPath = Join-Path $runDir "$scenario.json"
        $env:BLOCKGAME_BENCHMARK = "1"
        $env:BLOCKGAME_BENCHMARK_SCENARIO = $scenario
        $env:BLOCKGAME_BENCHMARK_OUTPUT = $scenarioPath
        $env:BLOCKGAME_BENCHMARK_BUILD_CONFIG = $Config

        Write-Host "Running chunk benchmark scenario $scenario ..."
        $process = Start-Process -FilePath $exePath `
            -WorkingDirectory $exeDir `
            -PassThru `
            -Wait

        if ($process.ExitCode -ne 0) {
            throw "BlockGame exited with code $($process.ExitCode) while running scenario $scenario"
        }
        if (-not (Test-Path $scenarioPath)) {
            throw "Scenario output not found at $scenarioPath"
        }

        $scenarioObject = Get-Content -Path $scenarioPath -Raw | ConvertFrom-Json
        $scenarioObjects.Add($scenarioObject)
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

$acceptanceView = foreach ($scenario in $scenarioObjects) {
    [pscustomobject]@{
        scenario = $scenario.scenario
        chunk_ready_latency_median_ms = $scenario.stages.chunk_ready_latency.median_ms
        chunk_ready_latency_p95_ms = $scenario.stages.chunk_ready_latency.p95_ms
        relight_avg_ms = $scenario.stages.relight.avg_ms
        upload_backlog_avg = $scenario.queues.upload_backlog.avg_depth
        upload_backlog_p95 = $scenario.queues.upload_backlog.p95_depth
        climate_hit_rate = $scenario.cache.climate.hit_rate
        surface_hit_rate = $scenario.cache.surface.hit_rate
        generated_chunks_per_sec = $scenario.throughput.generated_chunks_per_sec
        uploaded_chunks_per_sec = $scenario.throughput.uploaded_chunks_per_sec
    }
}

$summaryObject = [pscustomobject]@{
    schema_version = 1
    build_config = $Config
    output_dir = $runDir
    scenarios = $scenarioObjects
    acceptance_view = $acceptanceView
}

$summaryJsonPath = Join-Path $runDir "benchmark_summary.json"
$summaryTxtPath = Join-Path $runDir "benchmark_summary.txt"

$summaryObject | ConvertTo-Json -Depth 12 -Compress | Set-Content -Path $summaryJsonPath -NoNewline

$summaryLines = New-Object System.Collections.Generic.List[string]
$summaryLines.Add("BlockGame chunk benchmark")
$summaryLines.Add("Build: $Config")
$summaryLines.Add("Output: $runDir")
$summaryLines.Add("")
foreach ($scenario in $scenarioObjects) {
    $summaryLines.Add("Scenario: $($scenario.scenario)")
    $summaryLines.Add(("  duration_s={0:F2} generated_cps={1:F2} uploaded_cps={2:F2}" -f `
        $scenario.duration_seconds,
        $scenario.throughput.generated_chunks_per_sec,
        $scenario.throughput.uploaded_chunks_per_sec))
    $summaryLines.Add(("  chunk_ready_ms median={0:F2} p95={1:F2}" -f `
        $scenario.stages.chunk_ready_latency.median_ms,
        $scenario.stages.chunk_ready_latency.p95_ms))
    $summaryLines.Add(("  relight_avg_ms={0:F2} upload_backlog avg={1:F2} p95={2:F2}" -f `
        $scenario.stages.relight.avg_ms,
        $scenario.queues.upload_backlog.avg_depth,
        $scenario.queues.upload_backlog.p95_depth))
    $summaryLines.Add(("  climate_hit_rate={0:P2} surface_hit_rate={1:P2}" -f `
        $scenario.cache.climate.hit_rate,
        $scenario.cache.surface.hit_rate))
    $summaryLines.Add(("  frame_avg_ms={0:F2} frame_p95_ms={1:F2} avg_fps={2:F2}" -f `
        $scenario.frame.avg_ms,
        $scenario.frame.p95_ms,
        $scenario.frame.avg_fps))
    $summaryLines.Add("")
}

$summaryLines | Set-Content -Path $summaryTxtPath

Write-Host "Chunk benchmark summary written to $summaryJsonPath"
Write-Host "Chunk benchmark text summary written to $summaryTxtPath"
