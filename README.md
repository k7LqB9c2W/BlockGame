# BlockGame

<img width="1920" height="1005" alt="image" src="https://github.com/user-attachments/assets/939ebb8c-8bf9-4fc3-93b6-949205f24f8d" />

BlockGame is a Windows voxel sandbox prototype written in C++20. It features procedural terrain generation, chunk streaming, block interaction, far-terrain rendering, and an in-game Dear ImGui debug UI.

The visual baseline for the project is the plain "base game" terrain look. The more cinematic atmosphere path is kept as an optional enhancement and is intentionally disabled by default at startup.

The renderer is now **Direct3D 12 only**. There is no OpenGL backend toggle anymore. Any normal build of this repository produces the D3D12 version of the game.

## Current Status

- Rendering backend: Direct3D 12
- UI/debug backend: Dear ImGui with GLFW + D3D12
- Platform target: Windows
- Build system: CMake
- Primary runtime asset: `block_atlas.png`

## Features

- Procedural world generation with biome and climate systems
- Streamed chunk loading and meshing around the player
- Far-terrain rendering beyond the near chunk radius
- Block destruction and placement
- Spawn preload flow to avoid dropping the player into an incomplete world
- Debug overlay and profiling overlay
- In-game ImGui panels for render-distance settings and teleportation

## Controls

- `W`, `A`, `S`, `D`: move
- `Mouse`: look
- `Space`: jump
- `Left click`: destroy block
- `Right click`: place block
- `F1`: toggle debug overlay
- `N`: open render-distance settings
- `F2`: open teleport panel
- `F3`: toggle far terrain
- `Esc`: quit when no ImGui modal/tool window is open

## Requirements

You need the following on Windows:

- A GPU and driver with Direct3D 12 support
- CMake 3.20 or newer
- Visual Studio Build Tools or Visual Studio with the C++ desktop toolchain
- A working `vcpkg` installation with the required packages available

This project expects the D3D12 ImGui path. The current CMake setup auto-detects `vcpkg` from either:

- `VCPKG_ROOT`
- `C:\vcpkg`

Required packages:

- `imgui[dx12-binding,glfw-binding]:x64-windows`
- `glfw3:x64-windows`

In practice, installing the ImGui package above usually brings in GLFW as needed.

## Quick Start

### Option 1: Batch file

The repository includes `build_blockgame.bat` as a convenience wrapper.

Examples:

```bat
build_blockgame.bat release
build_blockgame.bat debug
build_blockgame.bat run
build_blockgame.bat clean
```

What it does:

- Locates the Visual Studio developer environment
- Locates `cmake` and `ninja`
- Configures a CMake build
- Builds the `blockgame` target

Default output folders:

- `build-dx12\Release`
- `build-dx12\Debug`

Important note: the folder name does **not** choose the renderer. The project itself is already D3D12-only. The batch file now uses `build-dx12` just to make that explicit.

### Option 2: CMake + Ninja

```bat
cmake -S . -B build-dx12 -G Ninja
cmake --build build-dx12
```

For a Debug build:

```bat
cmake -S . -B build-dx12-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-dx12-debug
```

For a Release build:

```bat
cmake -S . -B build-dx12-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-dx12-release
```

### Option 3: CMake + Visual Studio generator

Use the generator name that matches your installed Visual Studio version.

Example:

```bat
cmake -S . -B build-dx12-vs -G "Visual Studio 18 2026" -A x64
cmake --build build-dx12-vs --config Release
cmake --build build-dx12-vs --config Debug
```

## Running

After building, run `blockgame.exe` from the output directory so runtime DLLs and assets are beside the executable.

Typical output locations:

- Ninja Release: `build-dx12-release\blockgame.exe`
- Ninja Debug: `build-dx12-debug\blockgame.exe`
- Batch script Release: `build-dx12\Release\blockgame.exe`
- Batch script Debug: `build-dx12\Debug\blockgame.exe`
- Visual Studio multi-config: `build-dx12-vs\Release\blockgame.exe` or `build-dx12-vs\Debug\blockgame.exe`

The build copies these runtime resources automatically when present:

- required DLLs from linked packages
- `block_atlas.png`
- the `assets/` directory

## Build Notes

### Is a special build folder required for D3D12?

No. The renderer is chosen by the source and build configuration, not by the build directory name.

That means:

- `build_blockgame.bat release` builds D3D12 automatically
- `cmake -S . -B build` also builds D3D12 automatically
- `cmake -S . -B build-dx12` also builds D3D12 automatically

The folder name is only a convention. This repository now uses `build-dx12` in the batch script because it better reflects the project state.

### Does the project still support OpenGL?

No. The active game code has been ported to Direct3D 12.

## Project Layout

```text
src/
  main.cpp                Application startup, main loop, gameplay flow, ImGui windows
  renderer.cpp/.h         Direct3D 12 renderer, swapchain, pipelines, texture upload
  chunk_manager.cpp/.h    Chunk streaming, meshing, world interaction, render packet generation
  input_context.cpp/.h    GLFW input handling and ImGui-aware input gating
  terrain/                Terrain generation, biome database, climate and surface logic

include/
  Third-party headers used by the project build

libs/
  Repository-local runtime libraries used by older setups

assets/
  Runtime assets copied beside the executable when present
```

## Rendering Architecture

At a high level, the rendering path now looks like this:

1. GLFW creates a no-API window.
2. The renderer creates the DXGI factory, D3D12 device, command queue, swapchain, descriptor heaps, depth buffer, and graphics pipelines.
3. `ChunkManager` generates and streams chunk mesh data.
4. The renderer uploads texture and mesh data to D3D12 resources.
5. World geometry is rendered through D3D12 command lists.
6. Dear ImGui renders debug and tool UI through the D3D12 backend.
7. The swapchain presents the final frame.

## Gameplay and World Systems

The project currently includes:

- biome-aware terrain generation
- collision and movement
- block raycasting
- block destruction/placement
- chunk visibility and frustum-aware render-data generation
- startup preload sequencing
- profiling and streaming diagnostics

## Passive Mob AI

BlockGame now has a first-pass passive-mob runtime for debug-spawned animals such as pigs and cows.

- Mob geometry is loaded from `assets/mobs/*.json`, with textures resolved from a matching `.png` when present.
- Passive mobs currently share one very small AI loop: `Idle` for a random few seconds, then pick a short wander target, then `Walk` to it, then return to `Idle`.
- Bedrock leg bones can be animated at runtime, and pigs now swing their legs while walking so the motion is not fully rigid.
- Idle passive mobs can also do a small procedural head look-around, with bounded up/down and side-to-side motion that returns to neutral while walking.
- The wander step is intentionally cheap. It does not pathfind, does not run herd logic, and only checks a few chunk/terrain queries plus a small mob AABB sweep to keep movement on nearby walkable ground.
- Passive mobs now use collision-lite locomotion: shared gravity, simple block collision, and an automatic fixed hop over jumpable 1-block ledges instead of a full player-style movement stack.
- Despawn is chunk-radius based, not block-distance based. If a mob's chunk moves outside the player's current `Exact Chunks` radius, it is removed from `MobSystem`.
- There is no far-distance persistence layer yet. If you spawn pigs and then travel far enough away, those pigs are gone until you spawn new ones again.
- The current F1 debug UI exposes `Spawn Pig` and `Spawn Cow` buttons so the passive-mob path can be tested without natural spawning or combat/gameplay systems.

## Troubleshooting

### CMake cannot find `imgui` or `glfw3`

Make sure the required packages are installed in `vcpkg` and that either:

- `VCPKG_ROOT` points to your `vcpkg` install, or
- `C:\vcpkg` exists

### The batch file cannot find `cmake`, `ninja`, or Visual Studio tools

Install:

- Visual Studio Build Tools with the C++ workload
- CMake
- Ninja, or the Visual Studio-bundled Ninja/CMake tools

The batch file already tries to find the Visual Studio copies automatically.

### The game launches but rendering fails immediately

Check the following first:

- your GPU supports Direct3D 12
- your graphics driver is current
- `block_atlas.png` exists beside the executable after build
- runtime DLLs were copied successfully

### Input feels blocked

The debug/config panels intentionally capture input while they are open. Close the ImGui tool window you opened and mouse-look/gameplay input will resume.

## Benchmarking

BlockGame now includes an automated chunk-streaming benchmark pass with fixed scenarios for:

- `spawn_preload`
- `straight_line_sprint`
- `turn_heavy_traversal`
- `vertical_travel`

Run it with:

```bat
powershell -ExecutionPolicy Bypass -File tools\run_chunk_benchmark.ps1 -BuildDir build -Config Release
```

If the current `build\Release\blockgame.exe` is already up to date, skip the build step:

```bat
powershell -ExecutionPolicy Bypass -File tools\run_chunk_benchmark.ps1 -BuildDir build -Config Release -SkipBuild
```

The script creates a timestamped output folder under:

- `artifacts\chunk_benchmark\<timestamp>\benchmark_summary.json`
- `artifacts\chunk_benchmark\<timestamp>\benchmark_summary.txt`
- `artifacts\chunk_benchmark\<timestamp>\<scenario>.json`

`benchmark_summary.json` is the primary machine-readable artifact and is intentionally compact so Codex can inspect it without needing a bulky CSV. `benchmark_summary.txt` is the short human summary.

### How To Read The Benchmark Output

- `throughput.generated_chunks_per_sec`: generated chunk throughput over the scenario duration.
- `throughput.uploaded_chunks_per_sec`: chunks that actually reached the GPU upload stage per second.
- `stages.sample|generate|relight|mesh|upload|far_build.avg_ms`: average cost per completed work item in that stage.
- `stages.chunk_ready_latency.median_ms` and `p95_ms`: request-to-first-ready latency for streamed chunks.
- `queues.job_backlog`, `queues.upload_backlog`, `queues.far_build_backlog`, `queues.far_upload_backlog`: backlog depths sampled once per streaming update.
- `cache.climate.hit_rate` and `cache.surface.hit_rate`: effectiveness of the climate and surface fragment caches.
- `frame.avg_ms`, `frame.p95_ms`, `frame.avg_fps`: frame pacing context for the same run.

Important note: percentile values are collected with low-overhead histograms, so they should be treated as approximate profiling bands rather than exact microsecond-precise measurements.

## LOD Testing

BlockGame now has a visual-only distant terrain path driven by `Exact Chunks` plus `Total Chunks`.
If `Total Chunks` is greater than `Exact Chunks`, exact voxel chunks still handle gameplay while the distant terrain path fills the horizon outside the exact radius.
Within the exact radius, nearby chunks keep full CPU voxel/light data for gameplay and edits, while older non-interactive exact chunks can fall back to mesh-only residency and regenerate CPU data from worldgen plus the in-memory block edit overlay when needed. Exact chunk residency is also now interval-based per column instead of one contiguous vertical slab, so tall mountains no longer force every chunk between the player and the summit to stay exact. The climate generator now uses smaller climate fragments plus bounded fragment/seed caches so long exploration sessions do not let biome-climate memory grow without limit. Together, those changes keep exact rendering intact while significantly reducing RAM growth in tall terrain.

Run an LOD benchmark with:

```bat
powershell -ExecutionPolicy Bypass -File tools\run_lod_benchmark.ps1 -BuildDir build -Config Release -ExactChunks 48 -TotalChunks 128
```

The LOD benchmark writes timestamped runs under `artifacts\lod_benchmark\<timestamp>\`.
The summary JSON includes:

- `render_settings.lod_mode`
- `final_profiling.lod_active_tiles`
- `final_profiling.lod_ready_tiles`
- `final_profiling.lod_build_avg_ms`

Run a timestamped LOD screenshot sweep with:

```bat
powershell -ExecutionPolicy Bypass -File tools\run_lod_horizon_sweep.ps1 -BuildDir build -Config Release -ExactChunks 48 -TotalChunks 128
```

Those captures are preserved under `artifacts\lod_horizon_sweep\exact<exact>_total<total>_<timestamp>\` so you can compare progress across LOD iterations instead of overwriting the previous sweep.

## Development Notes

- The project is currently Windows-only because the renderer is Direct3D 12.
- The build no longer depends on GLAD or the old OpenGL text/texture pipeline.
- Dear ImGui is used for the debug overlay, loading/status windows, render-distance controls, and teleport tools.
- Lighting and terrain readability work should target the base-game look first.
- The Lighting Lab's `Enhanced Atmosphere` toggle is a non-default visual mode, closer to an optional shader-pack style enhancement than the canonical shipping look.
- If you reset renderer defaults or add new lighting presets, preserve `Base Game` as the startup/default presentation unless there is an explicit project-wide decision to change that.

## Verification Snapshot

The current D3D12 port has been validated with:

- successful `Debug` build
- successful `Release` build
- short launch smoke test on the built executable
- source scan confirming no remaining OpenGL calls in `src/`

## License / Assets

This repository does not currently declare a formal license in the root. If you plan to distribute the project, add one explicitly and verify that all bundled assets and third-party dependencies are compatible with your intended use.
