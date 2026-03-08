# BlockGame

<img width="1919" height="992" alt="BlockGame screenshot" src="https://github.com/user-attachments/assets/0a21b184-a01f-4011-9130-0ac2cd4deaaf" />

BlockGame is a Windows voxel sandbox prototype written in C++20. It features procedural terrain generation, chunk streaming, block interaction, far-terrain rendering, and an in-game Dear ImGui debug UI.

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

## Development Notes

- The project is currently Windows-only because the renderer is Direct3D 12.
- The build no longer depends on GLAD or the old OpenGL text/texture pipeline.
- Dear ImGui is used for the debug overlay, loading/status windows, render-distance controls, and teleport tools.

## Verification Snapshot

The current D3D12 port has been validated with:

- successful `Debug` build
- successful `Release` build
- short launch smoke test on the built executable
- source scan confirming no remaining OpenGL calls in `src/`

## License / Assets

This repository does not currently declare a formal license in the root. If you plan to distribute the project, add one explicitly and verify that all bundled assets and third-party dependencies are compatible with your intended use.
