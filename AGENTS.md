# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds gameplay logic, camera controls, chunk streaming, and the `main.cpp` entry point.
- `include/` bundles third-party headers (GLFW, GLM, stb_image, glad) consumed by the MSVC build.
- `libs/` stores prebuilt GLFW binaries; keep DLLs beside `blockgame.exe` for runtime stability.
- Root assets (`block_atlas.png`, guides, compiled `.obj` intermediates) support rendering validation; avoid checking new binaries without need.

## Build, Test, and Development Commands
- Configure the project with CMake: `cmake -S . -B build`. Pass `-DCMAKE_BUILD_TYPE=Release` (default) or `-DCMAKE_BUILD_TYPE=Debug` when using single-config generators like Ninja.
- Build with `cmake --build build` (single-config) or `cmake --build build --config Release` when using multi-config generators such as Visual Studio.
- Clean artifacts via `cmake --build build --target clean` if you need a fresh compile.
- Run the produced `blockgame.exe` from the `build` output directory for quick smoke tests.
- The legacy `build_blockgame.bat` script is deprecated; do not use it going forward.

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

## WHAT IS CUBYZ?
- REMINDER, OUR GAME THAT YOU WILL 99.999% OF THE TIME WILL BE EDITING IS BLOCK GAME.
- WE ARE USING CUBYZ as a "REFERENCE" FOR SOME FEATURES, LIKE TERRAIN GEN!. THATS NOT THE MAIN GAME THOUGH, JUST REMEMBER WE ALWAYS WILL USE BLOCKGAME.
- BUT I CAN REFERENCE CUBYZ SOMEITMES, like "ADD A FEATURE FROM CUBYZ"

## Debugging / Repro
- BlockGame supports renderer-driven automated screenshots for visual debugging.
- Sweep captures: run `powershell -ExecutionPolicy Bypass -File tools\run_horizon_sweep.ps1` to launch the game, rotate through a pose grid, delete old sweep screenshots, save fresh captures under `artifacts/horizon_sweep`, and write `captures.csv` plus `analysis.csv`.
- Single-view repro captures: run `powershell -ExecutionPolicy Bypass -File tools\capture_repro.ps1 -X <x> -Y <y> -Z <z> -Yaw <yaw> -Pitch <pitch>` to teleport to an exact camera pose, capture one screenshot, and delete old repro screenshots in `artifacts/repro_capture`.
- Single-view repro also supports a look target instead of yaw/pitch: `tools\capture_repro.ps1 -X <x> -Y <y> -Z <z> -LookX <x> -LookY <y> -LookZ <z>`.
- The in-game debug overlay now includes `Yaw/Pitch`, `Front`, and `Hit Block` so a screenshot contains enough information to recreate the exact view later.
- Chunk benchmark automation: run `powershell -ExecutionPolicy Bypass -File tools\run_chunk_benchmark.ps1 -BuildDir build -Config Release` to execute `spawn_preload`, `straight_line_sprint`, `turn_heavy_traversal`, and `vertical_travel` in sequence. Add `-SkipBuild` if `build\Release\blockgame.exe` is already current.
- Chunk benchmarks must be killed quickly when the game window becomes `Not Responding`; use the watchdog in `tools\run_chunk_benchmark.ps1` so hung runs are terminated early, a watchdog reason file is written, and CPU/GPU time plus RAM are not wasted during unattended benchmarking.
- Far terrain mode is obsolete, not needed for current BlockGame work, and is being phased out. Do not re-enable it in gameplay, UI, captures, or benchmarks unless the user explicitly asks for temporary restoration work.
- Chunk benchmark artifacts land in `artifacts\chunk_benchmark\<timestamp>\`. Read `benchmark_summary.json` first for Codex or scripts, and `benchmark_summary.txt` for a quick human summary.
- Benchmark interpretation:
  `throughput.generated_chunks_per_sec` / `throughput.uploaded_chunks_per_sec` show chunk streaming throughput.
  `stages.sample|generate|relight|mesh|upload|far_build.avg_ms` are per-work-item averages.
  `stages.chunk_ready_latency.median_ms` and `p95_ms` measure request-to-first-ready latency for chunks.
  `queues.*` are backlog depths sampled once per streaming update.
  `cache.climate.hit_rate` and `cache.surface.hit_rate` are cache efficiency for the terrain fragment caches.
  Percentiles are low-overhead histogram estimates, so values are best read as approximate bands rather than exact microsecond-precise timings.

## Visual Default Policy
- The canonical BlockGame lighting target is the plain `Base Game` look, not the enhanced atmosphere mode.
- The Lighting Lab `Enhanced Atmosphere` toggle is intentionally non-default at startup and should stay that way unless the user explicitly asks for a project-wide default change.
- When tuning terrain lighting, AO, fog, mip behavior, chunk shading, or screenshot regressions, evaluate the base-game look first before using enhanced atmosphere as an optional comparison.
- If you add presets, comments, or UI labels in this area, make it obvious that `Base Game` is the default shipping/editing baseline and `Enhanced Atmosphere` is an optional cinematic layer.
