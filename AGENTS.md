# Repository Guidelines
developer_instructions = '''
Use `rtk` as the default wrapper for almost every shell command it supports, not just when the gain is obvious. Reach for plain commands for shell builtins or cases where wrapping would be awkward or incorrect, such as `cd`, `export`, `alias`, heredocs, raw shell control flow, commands that `rtk` does not support, and all `npm`/`npx` commands. Examples: default to `rtk git status`, `rtk ls`, `rtk find`, `rtk grep`, `rtk pytest`, `rtk vitest`, `rtk diff`, `rtk wc`, `rtk curl`, `rtk docker`, and `rtk kubectl`. Use plain `npm` and plain `npx`. If `rtk` would change semantics, hide information you need, or make the result less reliable for the task, use the normal command instead.
'''
## Project Structure & Module Organization
- `directory.md` is the canonical authored-code map for this repo. Read it before broad exploration when you need to find major systems quickly.
- When a change adds, removes, renames, or moves a major component, update `directory.md` in the same change so the map stays current.
- Keep `AGENTS.md` and `agents.md` synchronized if either one is edited.
- `src/` holds gameplay logic, camera controls, chunk streaming, and the `main.cpp` entry point.
- `include/` bundles third-party headers (GLFW, GLM, stb_image, glad) consumed by the MSVC build.
- `libs/` stores prebuilt GLFW binaries; keep DLLs beside `blockgame.exe` for runtime stability.
- Root assets (`block_atlas.png`, guides, compiled `.obj` intermediates) support rendering validation; avoid checking new binaries without need.
- If they are installed, prefer `rg` for text search, `fd` for file discovery, and `bat` for file viewing.

## Build, Test, and Development Commands
- Configure the project with CMake: `cmake -S . -B build`. Pass `-DCMAKE_BUILD_TYPE=Release` (default) or `-DCMAKE_BUILD_TYPE=Debug` when using single-config generators like Ninja.
- Build with `cmake --build build` (single-config) or `cmake --build build --config Release` when using multi-config generators such as Visual Studio.
- Clean artifacts via `cmake --build build --target clean` if you need a fresh compile.
- Run the produced `blockgame.exe` from the `build` output directory for quick smoke tests.
- The legacy `build_blockgame.bat` script is deprecated; do not use it going forward.
- `dxc` is the default shader compiler backend. `blockgame_shader_precompiler.exe` remains the manifest/incremental wrapper and should call `dxc` unless a developer is explicitly investigating the legacy compiler path.

## Coding Style & Naming Conventions
- Mirror the existing C++20 style: 4-space indentation, braces on their own lines, and standard headers before local includes.
- Classes and structs use PascalCase (`Camera`, `ChunkManager`), member functions camelCase, constants `SCREAMING_SNAKE_CASE`.
- Lean on `<glm/...>` for math, STL containers (`<array>`, `<vector>`) for storage, and `std::` algorithms before bespoke loops.
- Keep headers lightweight; inline-only helpers live in `.inl` files (see `text_overlay.inl`).

## Testing Guidelines
- No automated suite yet; launch the freshly built `blockgame.exe`, review console output, and inspect `debug_output.txt` when adjusting streaming logic.
- For rendering tweaks, compare against `block_atlas_guide.txt` and capture before/after screenshots to attach to reviews.
- When adding tests, stage them under a new `tests/` directory and document the invocation alongside the script or CMake target.

## Commit & Pull Request Guidelines
- Follow the Git history: short, imperative titles (`Clamp chunk streaming to non-negative Y`); squash noisy fixups before pushing.
- Reference issue numbers in the body when relevant and describe gameplay or rendering impact plainly.
- PRs should include a purpose summary, build mode exercised (debug/release), reproduction steps, and visuals for GPU-facing changes.
- Keep binaries out of version control unless preparing a release; update `CMakeLists.txt` whenever dependencies shift.


## Debugging / Repro
- BlockGame supports renderer-driven automated screenshots for visual debugging.
- Sweep captures: run `powershell -ExecutionPolicy Bypass -File tools\run_horizon_sweep.ps1` to launch the game, rotate through a pose grid, delete old sweep screenshots, save fresh captures under `artifacts/horizon_sweep`, and write `captures.csv` plus `analysis.csv`.
- LOD sweep captures: run `powershell -ExecutionPolicy Bypass -File tools\run_lod_horizon_sweep.ps1 -BuildDir build -Config Release -ExactChunks 48 -TotalChunks 128` to capture timestamped distant-terrain sweeps without overwriting earlier runs. Artifacts land under `artifacts\lod_horizon_sweep\exact<exact>_total<total>_<timestamp>\`.
- Single-view repro captures: run `powershell -ExecutionPolicy Bypass -File tools\capture_repro.ps1 -X <x> -Y <y> -Z <z> -Yaw <yaw> -Pitch <pitch>` to teleport to an exact camera pose, capture one screenshot, and delete old repro screenshots in `artifacts/repro_capture`.
- Single-view repro also supports a look target instead of yaw/pitch: `tools\capture_repro.ps1 -X <x> -Y <y> -Z <z> -LookX <x> -LookY <y> -LookZ <z>`.
- The in-game debug overlay now includes `Yaw/Pitch`, `Front`, and `Hit Block` so a screenshot contains enough information to recreate the exact view later.
- Chunk benchmark automation: run `powershell -ExecutionPolicy Bypass -File tools\run_chunk_benchmark.ps1 -BuildDir build -Config Release` to execute the canonical `player_idle_exact_fill` benchmark. This is the benchmark to optimize exact-chunk loading against going forward: it uses the normal `12`-chunk startup preload, releases the player, then leaves the player standing still while the exact `48` ring fills for up to 5 minutes or until completion. Add `-SkipBuild` if `build\Release\blockgame.exe` is already current. Use `-Scenarios <name>` only when you explicitly want a non-default diagnostic scenario such as `full_exact_preload` or movement stress tests.
- LOD benchmark automation: run `powershell -ExecutionPolicy Bypass -File tools\run_lod_benchmark.ps1 -BuildDir build -Config Release -ExactChunks 48 -TotalChunks 128` to execute the same scenarios with distant visual terrain active. Change `-TotalChunks` up to `500` for long-range LOD stress.
- Chunk benchmarks must be killed quickly when the game window becomes `Not Responding`; use the watchdog in `tools\run_chunk_benchmark.ps1` so hung runs are terminated early, a watchdog reason file is written, and CPU/GPU time plus RAM are not wasted during unattended benchmarking.
- Legacy far terrain mode is obsolete. The replacement path is the Exact/Total chunk LOD system; do not revive the old far-terrain toggle or old far-block UI.
- Chunk benchmark artifacts land in `artifacts\chunk_benchmark\<timestamp>\`. Read `benchmark_summary.json` first for Codex or scripts, and `benchmark_summary.txt` for a quick human summary.
- Benchmark interpretation:
  `throughput.generated_chunks_per_sec` / `throughput.uploaded_chunks_per_sec` show chunk streaming throughput.
  `stages.sample|generate|relight|mesh|upload|far_build.avg_ms` are per-work-item averages.
  `stages.chunk_ready_latency.median_ms` and `p95_ms` measure request-to-first-ready latency for chunks.
  `queues.*` are backlog depths sampled once per streaming update.
  `cache.climate.hit_rate` and `cache.surface.hit_rate` are cache efficiency for the terrain fragment caches.
  `milestones.player_release_seconds|steady_state_seconds|full_exact_ready_seconds` separate time-to-interactive from full exact-fill completion.
  Percentiles are low-overhead histogram estimates, so values are best read as approximate bands rather than exact microsecond-precise timings.
- After completing a user-requested task or answer, run `.\alert.ps1 -Task "<short task summary>"` so the user gets the phone notification that work is done.

## Visual Default Policy
- The canonical BlockGame lighting target is the plain `Base Game` look, not the enhanced atmosphere mode.
- The Lighting Lab `Enhanced Atmosphere` toggle is intentionally non-default at startup and should stay that way unless the user explicitly asks for a project-wide default change.
- When tuning terrain lighting, AO, fog, mip behavior, chunk shading, or screenshot regressions, evaluate the base-game look first before using enhanced atmosphere as an optional comparison.
- If you add presets, comments, or UI labels in this area, make it obvious that `Base Game` is the default shipping/editing baseline and `Enhanced Atmosphere` is an optional cinematic layer.
