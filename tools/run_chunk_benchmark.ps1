param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$OutputDir = "$(Join-Path $PSScriptRoot '..\\artifacts\\chunk_benchmark')",
    [switch]$SkipBuild,
    [int]$ExactChunks = 48,
    [int]$TotalChunks = 0,
    [string[]]$Scenarios = @(),
    [int]$MaxScenarioSeconds = 300,
    [int]$FogStartBlocks = 1400,
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

$defaultScenarios = @(
    "player_idle_exact_fill"
)
$scenarios = if ($Scenarios.Count -gt 0) { $Scenarios } else { $defaultScenarios }

$resolvedTotalChunks = if ($TotalChunks -gt 0) { $TotalChunks } else { $ExactChunks }

$envKeys = @(
    "BLOCKGAME_BENCHMARK",
    "BLOCKGAME_BENCHMARK_SCENARIO",
    "BLOCKGAME_BENCHMARK_OUTPUT",
    "BLOCKGAME_BENCHMARK_BUILD_CONFIG",
    "BLOCKGAME_BENCHMARK_EXACT_CHUNKS",
    "BLOCKGAME_BENCHMARK_TOTAL_CHUNKS",
    "BLOCKGAME_BENCHMARK_FOG_START_BLOCKS",
    "BLOCKGAME_BENCHMARK_MAX_DURATION_SECONDS"
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
        $env:BLOCKGAME_BENCHMARK_EXACT_CHUNKS = [string]$ExactChunks
        $env:BLOCKGAME_BENCHMARK_TOTAL_CHUNKS = [string]$resolvedTotalChunks
        $env:BLOCKGAME_BENCHMARK_FOG_START_BLOCKS = [string]$FogStartBlocks
        $env:BLOCKGAME_BENCHMARK_MAX_DURATION_SECONDS = [string]$MaxScenarioSeconds

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
    $finalProfiling = $scenario.final_profiling
    $hasLodReadyTiles = $null -ne $finalProfiling -and ($finalProfiling.PSObject.Properties.Name -contains "lod_ready_tiles")
    $hasLodActiveTiles = $null -ne $finalProfiling -and ($finalProfiling.PSObject.Properties.Name -contains "lod_active_tiles")
    $hasLodBuiltTiles = $null -ne $finalProfiling -and ($finalProfiling.PSObject.Properties.Name -contains "lod_tiles_built_last_update")
    [pscustomobject]@{
        scenario = $scenario.scenario
        watchdog_reason = $scenario.watchdog_reason
        chunk_ready_latency_median_ms = $scenario.stages.chunk_ready_latency.median_ms
        chunk_ready_latency_p95_ms = $scenario.stages.chunk_ready_latency.p95_ms
        relight_avg_ms = $scenario.stages.relight.avg_ms
        frame_p95_ms = $scenario.frame.p95_ms
        frame_max_ms = $scenario.frame.max_ms
        spike_count_over_50_ms = if ($scenario.frame_spikes) { $scenario.frame_spikes.count_over_50_ms } else { $null }
        spike_count_over_100_ms = if ($scenario.frame_spikes) { $scenario.frame_spikes.count_over_100_ms } else { $null }
        upload_backlog_avg = $scenario.queues.upload_backlog.avg_depth
        upload_backlog_p95 = $scenario.queues.upload_backlog.p95_depth
        climate_hit_rate = $scenario.cache.climate.hit_rate
        surface_hit_rate = $scenario.cache.surface.hit_rate
        generated_chunks_per_sec = $scenario.throughput.generated_chunks_per_sec
        uploaded_chunks_per_sec = $scenario.throughput.uploaded_chunks_per_sec
        exact_chunks = $scenario.render_settings.exact_chunks
        total_chunks = $scenario.render_settings.total_chunks
        lod_mode = $scenario.render_settings.lod_mode
        relight_region_chunks_avg = if ($scenario.relight_detail) { $scenario.relight_detail.region_chunks.avg } else { $null }
        relight_region_chunks_p95 = if ($scenario.relight_detail) { $scenario.relight_detail.region_chunks.p95 } else { $null }
        relight_changed_chunks_avg = if ($scenario.relight_detail) { $scenario.relight_detail.changed_chunks.avg } else { $null }
        relight_changed_chunks_p95 = if ($scenario.relight_detail) { $scenario.relight_detail.changed_chunks.p95 } else { $null }
        relight_external_snapshot_chunks_avg = if ($scenario.relight_detail) { $scenario.relight_detail.external_snapshot_chunks.avg } else { $null }
        relight_external_snapshot_chunks_p95 = if ($scenario.relight_detail) { $scenario.relight_detail.external_snapshot_chunks.p95 } else { $null }
        relight_sky_above_chunk_scans_avg = if ($scenario.relight_detail) { $scenario.relight_detail.sky_above_chunk_scans.avg } else { $null }
        relight_sky_above_chunk_scans_p95 = if ($scenario.relight_detail) { $scenario.relight_detail.sky_above_chunk_scans.p95 } else { $null }
        vertical_radius_delta_avg = if ($scenario.relight_detail) { $scenario.relight_detail.vertical_radius_delta.avg } else { $null }
        vertical_radius_delta_p95 = if ($scenario.relight_detail) { $scenario.relight_detail.vertical_radius_delta.p95 } else { $null }
        lod_ready_tiles = if ($hasLodReadyTiles) { $finalProfiling.lod_ready_tiles } else { $null }
        lod_active_tiles = if ($hasLodActiveTiles) { $finalProfiling.lod_active_tiles } else { $null }
        lod_tiles_built_last_update = if ($hasLodBuiltTiles) { $finalProfiling.lod_tiles_built_last_update } else { $null }
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
$summaryLines.Add("Exact/Total/Fog: $ExactChunks / $resolvedTotalChunks / $FogStartBlocks")
$summaryLines.Add("Scenario timeout: $MaxScenarioSeconds s")
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
    if ($scenario.PSObject.Properties.Name -contains "timed_out") {
        $summaryLines.Add(("  timed_out={0} max_duration_s={1}" -f `
            $scenario.timed_out,
            $scenario.max_duration_seconds))
    }
    $summaryLines.Add(("  duration_s={0:F2} generated_cps={1:F2} uploaded_cps={2:F2}" -f `
        $scenario.duration_seconds,
        $scenario.throughput.generated_chunks_per_sec,
        $scenario.throughput.uploaded_chunks_per_sec))
    if ($scenario.milestones) {
        $summaryLines.Add(("  milestones release_s={0:F2} steady_state_s={1:F2} full_ready_s={2:F2}" -f `
            $scenario.milestones.player_release_seconds,
            $scenario.milestones.steady_state_seconds,
            $scenario.milestones.full_exact_ready_seconds))
        $summaryLines.Add(("  release_exact ready={0} required={1}" -f `
            $scenario.milestones.player_release_exact_ready_chunks,
            $scenario.milestones.player_release_exact_required_chunks))
    }
    if ($scenario.throughput.exact_gpu_builds_committed_per_sec) {
        $summaryLines.Add(("  exact_gpu_cps submitted={0:F2} committed={1:F2}" -f `
            $scenario.throughput.exact_gpu_builds_submitted_per_sec,
            $scenario.throughput.exact_gpu_builds_committed_per_sec))
    }
    $summaryLines.Add(("  chunk_ready_ms median={0:F2} p95={1:F2}" -f `
        $scenario.stages.chunk_ready_latency.median_ms,
        $scenario.stages.chunk_ready_latency.p95_ms))
    if ($scenario.stages.chunk_ready_wait_generate) {
        $summaryLines.Add(("  ready_breakdown_avg_ms wait_generate={0:F2} generate={1:F2} wait_mesh_enqueue={2:F2} wait_mesh_start={3:F2}" -f `
            $scenario.stages.chunk_ready_wait_generate.avg_ms,
            $scenario.stages.chunk_ready_generate.avg_ms,
            $scenario.stages.chunk_ready_wait_mesh_enqueue.avg_ms,
            $scenario.stages.chunk_ready_wait_mesh_start.avg_ms))
        $summaryLines.Add(("  ready_breakdown_avg_ms mesh={0:F2} wait_upload={1:F2} upload_to_ready={2:F2}" -f `
            $scenario.stages.chunk_ready_mesh.avg_ms,
            $scenario.stages.chunk_ready_wait_upload.avg_ms,
            $scenario.stages.chunk_ready_upload_to_ready.avg_ms))
    }
    $summaryLines.Add(("  relight_avg_ms={0:F2} upload_backlog avg={1:F2} p95={2:F2}" -f `
        $scenario.stages.relight.avg_ms,
        $scenario.queues.upload_backlog.avg_depth,
        $scenario.queues.upload_backlog.p95_depth))
    if ($scenario.stages.exact_gpu_total) {
        $summaryLines.Add(("  exact_gpu_avg_ms synth={0:F2} stamp={1:F2} light={2:F2} face_count={3:F2}" -f `
            $scenario.stages.exact_gpu_synth.avg_ms,
            $scenario.stages.exact_gpu_stamp.avg_ms,
            $scenario.stages.exact_gpu_light.avg_ms,
            $scenario.stages.exact_gpu_face_count.avg_ms))
        $summaryLines.Add(("  exact_gpu_avg_ms face_prefix={0:F2} face_emit={1:F2} total={2:F2}" -f `
            $scenario.stages.exact_gpu_face_prefix.avg_ms,
            $scenario.stages.exact_gpu_face_emit.avg_ms,
            $scenario.stages.exact_gpu_total.avg_ms))
        if ($scenario.stages.exact_gpu_prepare_cpu -and $scenario.stages.exact_gpu_submit_cpu -and $scenario.stages.exact_gpu_commit_cpu) {
            $summaryLines.Add(("  exact_gpu_cpu_avg_ms prepare={0:F2} submit={1:F2} commit={2:F2}" -f `
                $scenario.stages.exact_gpu_prepare_cpu.avg_ms,
                $scenario.stages.exact_gpu_submit_cpu.avg_ms,
                $scenario.stages.exact_gpu_commit_cpu.avg_ms))
        }
    }
    if ($scenario.queues.column_prefetch_backlog) {
        $summaryLines.Add(("  column_prefetch_backlog avg={0:F2} p95={1:F2}" -f `
            $scenario.queues.column_prefetch_backlog.avg_depth,
            $scenario.queues.column_prefetch_backlog.p95_depth))
    }
    if ($scenario.stages.visible_scan -and $scenario.stages.ensure_volume -and $scenario.stages.eviction -and $scenario.stages.upload_drain) {
        $summaryLines.Add(("  update_avg_ms={0:F2} residual_avg_ms={1:F2} dense_residency_avg_ms={2:F2} visible_scan_avg_ms={3:F2}" -f `
            $scenario.stages.update.avg_ms,
            $scenario.stages.update_residual.avg_ms,
            $scenario.stages.dense_residency.avg_ms,
            $scenario.stages.visible_scan.avg_ms))
        $summaryLines.Add(("  ensure_volume_avg_ms={0:F2} scheduling_avg_ms={1:F2} eviction_avg_ms={2:F2} upload_drain_avg_ms={3:F2}" -f `
            $scenario.stages.ensure_volume.avg_ms,
            $scenario.stages.scheduling.avg_ms,
            $scenario.stages.eviction.avg_ms,
            $scenario.stages.upload_drain.avg_ms))
        $summaryLines.Add(("  upload_pick_avg_ms={0:F2} upload_queue_age_avg_ms={1:F2} upload_queue_age_p95_ms={2:F2}" -f `
            $scenario.stages.upload_queue_pick.avg_ms,
            $scenario.stages.upload_queue_age.avg_ms,
            $scenario.stages.upload_queue_age.p95_ms))
        if ($scenario.stages.far_terrain_update) {
            $summaryLines.Add(("  far_terrain_avg_ms={0:F2} pool_trim_avg_ms={1:F2} priority_avg_ms={2:F2} upload_prepare_avg_ms={3:F2}" -f `
                $scenario.stages.far_terrain_update.avg_ms,
                $scenario.stages.pool_trim.avg_ms,
                $scenario.stages.priority_update.avg_ms,
                $scenario.stages.upload_prepare.avg_ms))
            $summaryLines.Add(("  upload_begin_avg_ms={0:F2} upload_finalize_avg_ms={1:F2} column_lookup_avg_ms={2:F2} column_sample_avg_ms={3:F2}" -f `
                $scenario.stages.upload_context_begin.avg_ms,
                $scenario.stages.upload_finalize.avg_ms,
                $scenario.stages.column_height_lookup.avg_ms,
                $scenario.stages.column_height_sample.avg_ms))
            $summaryLines.Add(("  commit_collect_avg_ms={0:F2} commit_scan_avg_ms={1:F2} commit_mesh_avg_ms={2:F2} commit_page_avg_ms={3:F2} commit_release_avg_ms={4:F2}" -f `
                $scenario.stages.commit_collect.avg_ms,
                $scenario.stages.commit_chunk_scan.avg_ms,
                $scenario.stages.commit_mesh_state.avg_ms,
                $scenario.stages.commit_page_state.avg_ms,
                $scenario.stages.commit_release.avg_ms))
            $summaryLines.Add(("  commit_mesh_wait_avg_ms={0:F2} commit_mesh_locked_avg_ms={1:F2}" -f `
                $scenario.stages.commit_mesh_lock_wait.avg_ms,
                $scenario.stages.commit_mesh_locked.avg_ms))
            $summaryLines.Add(("  generate_lock_avg_ms={0:F2} upload_mesh_lock_avg_ms={1:F2} neighborhood_lock_avg_ms={2:F2} skylight_cache_lock_avg_ms={3:F2}" -f `
                $scenario.stages.generate_blocks_mesh_lock.avg_ms,
                $scenario.stages.upload_chunk_mesh_lock.avg_ms,
                $scenario.stages.neighborhood_snapshot_lock.avg_ms,
                $scenario.stages.sky_light_cache_lock.avg_ms))
        }
    }
    if ($scenario.upload_detail) {
        $summaryLines.Add(("  upload_detail attempts_avg={0:F2} uploaded_chunks_avg={1:F2} uploaded_bytes_avg_kib={2:F2} scan_entries_avg={3:F2}" -f `
            $scenario.upload_detail.attempts_per_frame.avg,
            $scenario.upload_detail.uploaded_chunks_per_frame.avg,
            ($scenario.upload_detail.uploaded_bytes_per_frame.avg / 1KB),
            $scenario.upload_detail.queue_scan_entries.avg))
        $summaryLines.Add(("  upload_detail totals expired={0:F0} not_ready={1:F0} pending_mesh={2:F0} column_limit={3:F0} budget={4:F0} retry={5:F0}" -f `
            $scenario.upload_detail.expired_entries_per_frame.total,
            $scenario.upload_detail.skipped_not_ready_per_frame.total,
            $scenario.upload_detail.skipped_pending_mesh_per_frame.total,
            $scenario.upload_detail.column_limited_per_frame.total,
            $scenario.upload_detail.budget_deferred_per_frame.total,
            $scenario.upload_detail.retry_failures_per_frame.total))
        $summaryLines.Add(("  upload_detail totals scan_limit_hits={0:F0} begin_failures={1:F0} stale_pending_meshes={2:F0}" -f `
            $scenario.upload_detail.scan_limit_hits_per_frame.total,
            $scenario.upload_detail.begin_failures_per_frame.total,
            $scenario.upload_detail.stale_pending_meshes_per_frame.total))
    }
    if ($scenario.relight_detail) {
        $summaryLines.Add(("  relight region_chunks avg={0:F2} p95={1:F2} changed_chunks avg={2:F2} p95={3:F2}" -f `
            $scenario.relight_detail.region_chunks.avg,
            $scenario.relight_detail.region_chunks.p95,
            $scenario.relight_detail.changed_chunks.avg,
            $scenario.relight_detail.changed_chunks.p95))
        $summaryLines.Add(("  relight external_snapshots avg={0:F2} p95={1:F2} sky_above_scans avg={2:F2} p95={3:F2}" -f `
            $scenario.relight_detail.external_snapshot_chunks.avg,
            $scenario.relight_detail.external_snapshot_chunks.p95,
            $scenario.relight_detail.sky_above_chunk_scans.avg,
            $scenario.relight_detail.sky_above_chunk_scans.p95))
        $summaryLines.Add(("  relight sky_seed avg={0:F2} p95={1:F2} sky_nodes avg={2:F2} p95={3:F2}" -f `
            $scenario.relight_detail.sky_seed_nodes.avg,
            $scenario.relight_detail.sky_seed_nodes.p95,
            $scenario.relight_detail.sky_nodes_processed.avg,
            $scenario.relight_detail.sky_nodes_processed.p95))
        $summaryLines.Add(("  vertical_radius_delta avg={0:F2} p95={1:F2}" -f `
            $scenario.relight_detail.vertical_radius_delta.avg,
            $scenario.relight_detail.vertical_radius_delta.p95))
    }
    $summaryLines.Add(("  climate_hit_rate={0:P2} surface_hit_rate={1:P2}" -f `
        $scenario.cache.climate.hit_rate,
        $scenario.cache.surface.hit_rate))
    if ($scenario.final_profiling) {
        $summaryLines.Add(("  pool chunks={0} bytes_mib={1:F2} budget_mib={2:F2}" -f `
            $scenario.final_profiling.pooled_chunks,
            ($scenario.final_profiling.pooled_chunk_bytes / 1MB),
            ($scenario.final_profiling.pooled_chunk_budget_bytes / 1MB)))
        if ($scenario.final_profiling.PSObject.Properties.Name -contains "exact_gpu_total_bytes") {
            $summaryLines.Add(("  exact_gpu_mem_mib total={0:F2} pages={1:F2} voxels={2:F2} light={3:F2}" -f `
                ($scenario.final_profiling.exact_gpu_total_bytes / 1MB),
                ($scenario.final_profiling.exact_gpu_page_bytes / 1MB),
                ($scenario.final_profiling.exact_gpu_voxel_bytes / 1MB),
                ($scenario.final_profiling.exact_gpu_light_scratch_bytes / 1MB)))
            $summaryLines.Add(("  exact_gpu_mem_mib columns={0:F2} sparse={1:F2} scratch={2:F2} upload={3:F2} readback={4:F2}" -f `
                ($scenario.final_profiling.exact_gpu_column_bytes / 1MB),
                ($scenario.final_profiling.exact_gpu_sparse_voxel_bytes / 1MB),
                ($scenario.final_profiling.exact_gpu_scratch_bytes / 1MB),
                ($scenario.final_profiling.exact_gpu_upload_scratch_bytes / 1MB),
                ($scenario.final_profiling.exact_gpu_readback_bytes / 1MB)))
            $summaryLines.Add(("  vram_local_mib usage={0:F2} budget={1:F2} non_local_usage={2:F2} non_local_budget={3:F2}" -f `
                ($scenario.final_profiling.gpu_local_usage_bytes / 1MB),
                ($scenario.final_profiling.gpu_local_budget_bytes / 1MB),
                ($scenario.final_profiling.gpu_non_local_usage_bytes / 1MB),
                ($scenario.final_profiling.gpu_non_local_budget_bytes / 1MB)))
        }
    }
    $summaryLines.Add(("  frame_avg_ms={0:F2} frame_p95_ms={1:F2} avg_fps={2:F2}" -f `
        $scenario.frame.avg_ms,
        $scenario.frame.p95_ms,
        $scenario.frame.avg_fps))
    if ($scenario.frame_spikes) {
        $summaryLines.Add(("  spikes over50={0} over100={1} streak33={2}" -f `
            $scenario.frame_spikes.count_over_50_ms,
            $scenario.frame_spikes.count_over_100_ms,
            $scenario.frame_spikes.longest_streak_over_33_3_ms))
        if ($scenario.frame_spikes.worst -and $scenario.frame_spikes.worst.Count -gt 0) {
            $worst = $scenario.frame_spikes.worst[0]
            $summaryLines.Add(("  worst_spike_ms={0:F2} source={1} update_ms={2:F2} present_ms={3:F2}" -f `
                $worst.frame_ms,
                $worst.suspected_source,
                $worst.chunk_update_ms,
                $worst.renderer_present_ms))
            if ($null -ne $worst.chunk_ensure_volume_ms) {
                $summaryLines.Add(("  worst_spike detail residual_ms={0:F2} dense_residency_ms={1:F2} scan_ms={2:F2} ensure_ms={3:F2}" -f `
                    $worst.chunk_update_residual_ms,
                    $worst.chunk_dense_residency_ms,
                    $worst.chunk_missing_scan_ms,
                    $worst.chunk_ensure_volume_ms))
                $summaryLines.Add(("  worst_spike detail schedule_ms={0:F2} evict_ms={1:F2} upload_ms={2:F2} queue_age_ms={3:F2}" -f `
                    $worst.chunk_scheduling_ms,
                    $worst.chunk_eviction_ms,
                    $worst.chunk_upload_ms,
                    $worst.chunk_upload_queue_age_ms))
                $summaryLines.Add(("  worst_spike detail upload_pick_ms={0:F2} far_terrain_ms={1:F2} prefetch_backlog={2}" -f `
                    $worst.chunk_upload_queue_pick_ms,
                    $worst.chunk_far_terrain_update_ms,
                    $worst.column_prefetch_backlog))
                $summaryLines.Add(("  worst_spike detail upload_prepare_ms={0:F2} upload_begin_ms={1:F2} upload_finalize_ms={2:F2} column_lookup_ms={3:F2} column_sample_ms={4:F2}" -f `
                    $worst.chunk_upload_prepare_ms,
                    $worst.chunk_upload_context_begin_ms,
                    $worst.chunk_upload_finalize_ms,
                    $worst.chunk_column_height_lookup_ms,
                    $worst.chunk_column_height_sample_ms))
                $summaryLines.Add(("  worst_spike detail commit_collect_ms={0:F2} commit_scan_ms={1:F2} commit_mesh_ms={2:F2} commit_page_ms={3:F2} commit_release_ms={4:F2}" -f `
                    $worst.chunk_commit_collect_ms,
                    $worst.chunk_commit_chunk_scan_ms,
                    $worst.chunk_commit_mesh_state_ms,
                    $worst.chunk_commit_page_state_ms,
                    $worst.chunk_commit_release_ms))
                $summaryLines.Add(("  worst_spike detail commit_mesh_wait_ms={0:F2} commit_mesh_locked_ms={1:F2}" -f `
                    $worst.chunk_commit_mesh_lock_wait_ms,
                    $worst.chunk_commit_mesh_locked_ms))
                $summaryLines.Add(("  worst_spike upload attempts={0} scan_entries={1} expired={2} not_ready={3} pending_mesh={4} column_limit={5} budget={6} retry={7} stale_pending={8}" -f `
                    $worst.upload_attempts_this_frame,
                    $worst.upload_queue_scan_entries_this_frame,
                    $worst.upload_skipped_expired_this_frame,
                    $worst.upload_skipped_not_ready_this_frame,
                    $worst.upload_skipped_pending_mesh_this_frame,
                    $worst.upload_column_limited_this_frame,
                    $worst.upload_budget_deferred_this_frame,
                    $worst.upload_retry_failures_this_frame,
                    $worst.upload_stale_pending_meshes_this_frame))
            }
            $summaryLines.Add(("  worst_spike relight region={0} changed={1} external={2} sky_scans={3} vertical_delta={4}" -f `
                $worst.relight_region_chunks_this_frame,
                $worst.relight_changed_chunks_this_frame,
                $worst.relight_external_snapshot_chunks_this_frame,
                $worst.relight_sky_above_chunk_scans_this_frame,
                $worst.chunk_vertical_radius_delta))
        }
    }
    $summaryLines.Add("")
}

$summaryLines | Set-Content -Path $summaryTxtPath

Write-Host "Chunk benchmark summary written to $summaryJsonPath"
Write-Host "Chunk benchmark text summary written to $summaryTxtPath"
