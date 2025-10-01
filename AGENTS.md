# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds gameplay logic, camera controls, chunk streaming, and the `main.cpp` entry point.
- `include/` bundles third-party headers (GLFW, GLM, stb_image, glad) consumed by the MSVC build.
- `libs/` stores prebuilt GLFW binaries; keep DLLs beside `blockgame.exe` for runtime stability.
- Root assets (`block_atlas.png`, guides, compiled `.obj` intermediates) support rendering validation; avoid checking new binaries without need.

## Build, Test, and Development Commands
- `build_blockgame.bat release` builds an optimized 64-bit executable with `/O2` and copies GLFW runtime when present.
- `build_blockgame.bat debug` compiles with `/Zi` symbols and `/MDd`; prefer this during feature work to retain asserts and richer logs.
- `build_blockgame.bat run` performs a release build then launches `blockgame.exe` for fast smoke testing.
- `build_blockgame.bat clean` clears `.obj/.pdb/.ilk` artifacts so you can verify a fresh compile.
- Currently we are using the .bat file, not the cmakelists to build.

## Coding Style & Naming Conventions
- Mirror the existing C++20 style: 4-space indentation, braces on their own lines, and standard headers before local includes.
- Classes and structs use PascalCase (`Camera`, `ChunkManager`), member functions camelCase, constants `SCREAMING_SNAKE_CASE`.
- Lean on `<glm/...>` for math, STL containers (`<array>`, `<vector>`) for storage, and `std::` algorithms before bespoke loops.
- Keep headers lightweight; inline-only helpers live in `.inl` files (see `text_overlay.inl`).

## Testing Guidelines
- No automated suite yet; run `build_blockgame.bat run`, review console output, and inspect `debug_output.txt` when adjusting streaming logic.
- For rendering tweaks, compare against `block_atlas_guide.txt` and capture before/after screenshots to attach to reviews.
- When adding tests, stage them under a new `tests/` directory and document the invocation alongside the script or CMake target.

## Commit & Pull Request Guidelines
- Follow the Git history: short, imperative titles (`Clamp chunk streaming to non-negative Y`); squash noisy fixups before pushing.
- Reference issue numbers in the body when relevant and describe gameplay or rendering impact plainly.
- PRs should include a purpose summary, build mode exercised (debug/release), reproduction steps, and visuals for GPU-facing changes.
- Keep binaries out of version control unless preparing a release; update `build_blockgame.bat` or `CMakeLists.txt` whenever dependencies shift.
