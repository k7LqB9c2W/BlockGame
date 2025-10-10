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
