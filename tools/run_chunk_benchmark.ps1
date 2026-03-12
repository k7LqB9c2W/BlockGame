param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$OutputDir = "$(Join-Path $PSScriptRoot '..\\artifacts\\chunk_benchmark')",
    [switch]$SkipBuild,
    [int]$NotRespondingSeconds = 4,
    [int]$PostWriteGraceSeconds = 5,
    [int]$PollMilliseconds = 500
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

function Write-WatchdogReport {
    param(
        [string]$Path,
        [hashtable]$Report
    )

    $Report | ConvertTo-Json -Depth 8 -Compress | Set-Content -Path $Path -NoNewline
}

function Invoke-BenchmarkScenario {
    param(
        [string]$ExePath,
        [string]$ExeDir,
        [string]$Scenario,
        [string]$ScenarioPath,
        [int]$HangThresholdSeconds,
        [int]$PostWriteThresholdSeconds,
        [int]$PollIntervalMilliseconds
    )

    $watchdogPath = [System.IO.Path]::ChangeExtension($ScenarioPath, ".watchdog.json")
    Remove-Item -Path $ScenarioPath -ErrorAction SilentlyContinue
    Remove-Item -Path $watchdogPath -ErrorAction SilentlyContinue

    $startedAt = Get-Date
    $process = Start-Process -FilePath $ExePath `
        -WorkingDirectory $ExeDir `
        -PassThru

    $nonResponsiveSince = $null
    $lastOutputWriteUtc = $null
    $watchdogReason = "completed"
    $killedByWatchdog = $false
    $hungAfterWrite = $false

    while (-not $process.HasExited) {
        Start-Sleep -Milliseconds $PollIntervalMilliseconds
        $process.Refresh()
        $now = Get-Date

        if (Test-Path $ScenarioPath) {
            $lastOutputWriteUtc = (Get-Item $ScenarioPath).LastWriteTimeUtc
        }

        $hasWindow = $process.MainWindowHandle -ne 0
        if ($hasWindow -and -not $process.Responding) {
            if ($null -eq $nonResponsiveSince) {
                $nonResponsiveSince = $now
            }
            if (($now - $nonResponsiveSince).TotalSeconds -ge $HangThresholdSeconds) {
                $watchdogReason = "window_not_responding"
                $killedByWatchdog = $true
            }
        } else {
            $nonResponsiveSince = $null
        }

        if (-not $killedByWatchdog -and $null -ne $lastOutputWriteUtc) {
            $secondsSinceWrite = ($now.ToUniversalTime() - $lastOutputWriteUtc).TotalSeconds
            if ($secondsSinceWrite -ge $PostWriteThresholdSeconds) {
                $watchdogReason = "post_write_shutdown_hang"
                $hungAfterWrite = $true
                $killedByWatchdog = $true
            }
        }

        if ($killedByWatchdog) {
            try {
                Stop-Process -Id $process.Id -Force -ErrorAction Stop
            } catch {
            }
            break
        }
    }

    try {
        $process.WaitForExit(5000) | Out-Null
    } catch {
    }
    $process.Refresh()

    $finishedAt = Get-Date
    $scenarioOutputPresent = Test-Path $ScenarioPath
    $exitCode = $null
    if ($process.HasExited) {
        $exitCode = $process.ExitCode
    }

    if (-not $killedByWatchdog -and $exitCode -ne 0) {
        $watchdogReason = "nonzero_exit"
    }

    $report = [ordered]@{
        schema_version = 1
        scenario = $Scenario
        killed_by_watchdog = $killedByWatchdog
        reason = $watchdogReason
        hung_after_output_write = $hungAfterWrite
        exit_code = $exitCode
        started_at_utc = $startedAt.ToUniversalTime().ToString("o")
        finished_at_utc = $finishedAt.ToUniversalTime().ToString("o")
        runtime_seconds = [Math]::Round(($finishedAt - $startedAt).TotalSeconds, 3)
        poll_milliseconds = $PollIntervalMilliseconds
        not_responding_threshold_seconds = $HangThresholdSeconds
        post_write_grace_seconds = $PostWriteThresholdSeconds
        scenario_output_path = $ScenarioPath
        scenario_output_present = $scenarioOutputPresent
        last_output_write_utc = if ($null -ne $lastOutputWriteUtc) { $lastOutputWriteUtc.ToString("o") } else { $null }
        main_window_handle = $process.MainWindowHandle
        final_process_responding = if ($process.HasExited) { $null } else { $process.Responding }
    }
    Write-WatchdogReport -Path $watchdogPath -Report $report

    [pscustomobject]@{
        exit_code = $exitCode
        watchdog_reason = $watchdogReason
        watchdog_path = $watchdogPath
        scenario_output_present = $scenarioOutputPresent
        killed_by_watchdog = $killedByWatchdog
    }
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
$watchdogReports = New-Object System.Collections.Generic.List[object]

try {
    foreach ($scenario in $scenarios) {
        $scenarioPath = Join-Path $runDir "$scenario.json"
        $env:BLOCKGAME_BENCHMARK = "1"
        $env:BLOCKGAME_BENCHMARK_SCENARIO = $scenario
        $env:BLOCKGAME_BENCHMARK_OUTPUT = $scenarioPath
        $env:BLOCKGAME_BENCHMARK_BUILD_CONFIG = $Config

        Write-Host "Running chunk benchmark scenario $scenario ..."
        $result = Invoke-BenchmarkScenario `
            -ExePath $exePath `
            -ExeDir $exeDir `
            -Scenario $scenario `
            -ScenarioPath $scenarioPath `
            -HangThresholdSeconds $NotRespondingSeconds `
            -PostWriteThresholdSeconds $PostWriteGraceSeconds `
            -PollIntervalMilliseconds $PollMilliseconds

        $watchdogReport = Get-Content -Path $result.watchdog_path -Raw | ConvertFrom-Json
        $watchdogReports.Add($watchdogReport)

        if (-not $result.scenario_output_present) {
            throw "Scenario $scenario did not produce output. Watchdog report: $($result.watchdog_path)"
        }

        try {
            $scenarioObject = Get-Content -Path $scenarioPath -Raw | ConvertFrom-Json
        } catch {
            throw "Scenario output for $scenario is not valid JSON. Watchdog report: $($result.watchdog_path)"
        }

        $scenarioObject | Add-Member -NotePropertyName watchdog_reason -NotePropertyValue $result.watchdog_reason -Force
        $scenarioObject | Add-Member -NotePropertyName watchdog_report_path -NotePropertyValue $result.watchdog_path -Force
        $scenarioObject | Add-Member -NotePropertyName process_exit_code -NotePropertyValue $result.exit_code -Force
        $scenarioObjects.Add($scenarioObject)

        if ($result.watchdog_reason -ne "completed") {
            Write-Warning "Scenario $scenario finished with watchdog reason '$($result.watchdog_reason)'. Using written JSON at $scenarioPath."
        }
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
        watchdog_reason = $scenario.watchdog_reason
        chunk_ready_latency_median_ms = $scenario.stages.chunk_ready_latency.median_ms
        chunk_ready_latency_p95_ms = $scenario.stages.chunk_ready_latency.p95_ms
        relight_avg_ms = $scenario.stages.relight.avg_ms
        upload_backlog_avg = $scenario.queues.upload_backlog.avg_depth
        upload_backlog_p95 = $scenario.queues.upload_backlog.p95_depth
        climate_hit_rate = $scenario.cache.climate.hit_rate
        surface_hit_rate = $scenario.cache.surface.hit_rate
        generated_chunks_per_sec = $scenario.throughput.generated_chunks_per_sec
        uploaded_chunks_per_sec = $scenario.throughput.uploaded_chunks_per_sec
        pooled_chunks = if ($scenario.final_profiling) { $scenario.final_profiling.pooled_chunks } else { $null }
        pooled_chunk_bytes = if ($scenario.final_profiling) { $scenario.final_profiling.pooled_chunk_bytes } else { $null }
        pooled_chunk_budget_bytes = if ($scenario.final_profiling) { $scenario.final_profiling.pooled_chunk_budget_bytes } else { $null }
    }
}

$summaryObject = [pscustomobject]@{
    schema_version = 2
    build_config = $Config
    output_dir = $runDir
    scenarios = $scenarioObjects
    watchdog_reports = $watchdogReports
    acceptance_view = $acceptanceView
}

$summaryJsonPath = Join-Path $runDir "benchmark_summary.json"
$summaryTxtPath = Join-Path $runDir "benchmark_summary.txt"

$summaryObject | ConvertTo-Json -Depth 12 -Compress | Set-Content -Path $summaryJsonPath -NoNewline

$summaryLines = New-Object System.Collections.Generic.List[string]
$summaryLines.Add("BlockGame chunk benchmark")
$summaryLines.Add("Build: $Config")
$summaryLines.Add("Output: $runDir")
$summaryLines.Add(("Watchdog: not_responding={0}s post_write_grace={1}s poll_ms={2}" -f `
    $NotRespondingSeconds,
    $PostWriteGraceSeconds,
    $PollMilliseconds))
$summaryLines.Add("")
foreach ($scenario in $scenarioObjects) {
    $summaryLines.Add("Scenario: $($scenario.scenario)")
    $summaryLines.Add(("  watchdog_reason={0} exit_code={1}" -f `
        $scenario.watchdog_reason,
        $scenario.process_exit_code))
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
    if ($scenario.final_profiling) {
        $summaryLines.Add(("  pool chunks={0} bytes_mib={1:F2} budget_mib={2:F2}" -f `
            $scenario.final_profiling.pooled_chunks,
            ($scenario.final_profiling.pooled_chunk_bytes / 1MB),
            ($scenario.final_profiling.pooled_chunk_budget_bytes / 1MB)))
    }
    $summaryLines.Add(("  frame_avg_ms={0:F2} frame_p95_ms={1:F2} avg_fps={2:F2}" -f `
        $scenario.frame.avg_ms,
        $scenario.frame.p95_ms,
        $scenario.frame.avg_fps))
    $summaryLines.Add("")
}

$summaryLines | Set-Content -Path $summaryTxtPath

Write-Host "Chunk benchmark summary written to $summaryJsonPath"
Write-Host "Chunk benchmark text summary written to $summaryTxtPath"
