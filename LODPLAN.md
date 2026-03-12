# BlockGame D3D12 LOD System Plan for 500-Chunk Total Radius

## Summary
- This plan interprets your target as `500 chunks` total radius, which is about `8000 blocks` at `16 blocks/chunk`. It does **not** interpret it as `500 blocks`.
- `48 chunks` remains the maximum exact voxel radius. Any total radius above `48` automatically uses LOD beyond that point.
- The new LOD system replaces the obsolete far-terrain path entirely. It is a new subsystem, not a revival of the current disabled far-terrain code in [chunk_manager.cpp](/Users/Jacob/Documents/code/BlockGame/src/chunk_manager.cpp).
- The GPU path stays **inside D3D12**. Do not add OpenCL. Use D3D12 graphics + compute + indirect drawing so the system is ready for future GPU-side terrain math and generation.
- The design is intentionally a better fit for BlockGame than Distant Horizons’ Java/OpenGL implementation: shared indexed grid patches, D3D12 compute-ready source pages, exact/LOD overlap to hide pop-in, and an `ExecuteIndirect`-ready submission path.

## Current Repo Baseline
- Exact chunk distance is currently capped at `48` in [chunk_manager.h](/Users/Jacob/Documents/code/BlockGame/src/chunk_manager.h).
- The old far-terrain path is already hard-disabled in [chunk_manager.cpp](/Users/Jacob/Documents/code/BlockGame/src/chunk_manager.cpp).
- The UI still exposes `Near Chunks` plus `Far Blocks` in [main.cpp](/Users/Jacob/Documents/code/BlockGame/src/main.cpp).
- The renderer still submits near/far terrain with normal indexed draws in [renderer.cpp](/Users/Jacob/Documents/code/BlockGame/src/renderer.cpp), so this is the right time to redesign distant rendering cleanly rather than pile more work onto the dead far-terrain model.

## Architecture
### 1. Player-facing control model
- Replace the current render-distance controls with:
  - `Exact Radius (Chunks)`: `1..48`
  - `Total Radius (Chunks)`: `1..500`
  - `Fog Start (Blocks)`: keep as a separate tuning control
- Show a derived readout: `Total Radius ≈ N * 16 blocks`.
- Rule:
  - if `totalRadiusChunks <= exactRadiusChunks`, no LOD is active
  - if `totalRadiusChunks > exactRadiusChunks`, LOD starts immediately outside the exact radius
- Remove `Far Blocks`, `farTerrainEnabled`, and the current fake `lodEnabled` toggle.

### 2. Representation split
- Keep the current exact chunk system for `0..48 chunks`.
- Add a new `TerrainLodManager` responsible only for distant visual terrain.
- LOD terrain is **visual-only**:
  - no collision
  - no block edits
  - no raycast targeting
  - exact chunks always take over when the player approaches
- The LOD system does **not** use `Chunk` objects or `FarChunk` shells. It uses its own page/tile cache.

### 3. LOD data model
- Introduce:
  - `LodLevelConfig`
  - `LodSourcePageKey`
  - `LodSourcePageCpuPayload`
  - `LodPageGpuSlot`
  - `LodTileKey`
  - `LodTileInstance`
  - `LodProfilingSnapshot`
- Use a shared source-page model instead of per-tile CPU meshes.
- Each `LodSourcePage` covers `512 x 512 blocks` at a base sample pitch of `4 blocks`.
- Each page stores:
  - height samples: `129 x 129`, `R16_UINT`
  - water-surface samples: `129 x 129`, `R16_UINT`
  - material IDs: `128 x 128`, `R8_UINT`
  - flags/masks: `128 x 128`, `R8_UINT`
  - normals: `128 x 128`, `RG8_SNORM`
  - mip chain down to the coarsest LOD level
- Each visible LOD tile references a subregion of one or more source pages and is drawn by instancing a shared indexed grid patch.

### 4. LOD level configuration
Use these five visual-only levels beyond exact terrain:

- `LOD1`: `49..80 chunks`, sample step `4 blocks`, patch extent `128 blocks`, grid `32 x 32`
- `LOD2`: `81..128 chunks`, sample step `8 blocks`, patch extent `256 blocks`, grid `32 x 32`
- `LOD3`: `129..192 chunks`, sample step `16 blocks`, patch extent `512 blocks`, grid `32 x 32`
- `LOD4`: `193..320 chunks`, sample step `32 blocks`, patch extent `1024 blocks`, grid `32 x 32`
- `LOD5`: `321..500 chunks`, sample step `64 blocks`, patch extent `2048 blocks`, grid `32 x 32`

Rules:
- tile size is always `sampleStepBlocks * 32`
- every level uses a `2-cell` morph band
- every tile gets skirts with depth `max(sampleStepBlocks * 2, 8 blocks)`
- page/tile origins snap to their native grid so rings do not shimmer during motion

### 5. Render path
- Add a reusable static indexed grid mesh for all LOD tiles.
- Near terrain remains the existing voxel mesh path.
- LOD terrain uses:
  - one shared VB/IB
  - per-tile instance data
  - vertex-shader height displacement from source-page textures/buffers
  - material/normal lookup in shader
- This means BlockGame starts using GPU math for terrain shape immediately, even before compute generation lands.
- Near chunks render first and write depth.
- LOD tiles render after near chunks with depth test enabled, so exact chunks naturally override LOD when both are present.

### 6. Exact/LOD transition strategy
- Keep a `4-chunk overlap band` between exact terrain and `LOD1`.
- Inside the overlap:
  - exact chunks always win visually through depth
  - LOD remains present underneath until exact chunks are ready
- Fade rule:
  - use distance-based dither fade between adjacent LOD levels
  - exact-to-LOD uses depth precedence plus a short dither fade at the edge
- This is mandatory to avoid visible holes or pop-in during streaming.

### 7. Terrain content rules
- Far LOD terrain approximates the macro surface only.
- Explicitly out of scope beyond exact terrain:
  - caves
  - overhang fidelity
  - voxel-perfect cliffs
  - editable distant blocks
- Water gets its own far-surface representation from the water-height samples.
- Forests and biome identity should not disappear:
  - encode canopy/biome tint into LOD material/flags in v1
  - do not add far tree geometry in v1
  - optional later: impostor/tree-cluster pass for `LOD1` and `LOD2` only

## D3D12 GPU Math Plan
### 8. Required D3D12 path
- Stay in D3D12. Do not add OpenCL.
- Use:
  - graphics queue for LOD drawing
  - copy queue for page uploads / static resource staging
  - compute queue for page synthesis, mip generation, culling, and indirect-arg generation
- Synchronization:
  - copy fence -> compute fence -> graphics fence
  - use normal D3D12 resource state transitions and UAV barriers
- The renderer must be `ExecuteIndirect`-ready from the first LOD milestone even if the first shipping cut still allows CPU-built visible-tile lists.

### 9. Rollout of GPU math
- `Milestone 1`: GPU vertex math only
  - CPU builds source pages
  - GPU vertex shader displaces the shared grid
- `Milestone 2`: GPU page processing
  - compute builds normals, mip chain, min/max, and optional canopy tint from uploaded base height/material data
- `Milestone 3`: GPU culling and indirect draw
  - compute performs frustum culling
  - optional Hi-Z/occlusion later
  - compute writes visible tile list and indirect args
  - graphics uses `ExecuteIndirect`
- `Milestone 4`: GPU page synthesis for terrain generation
  - port the macro surface sampling path to HLSL compute
  - generate height/water/material pages on GPU directly from worldgen parameters
  - keep CPU fallback for unsupported/failed compute init

### 10. Why this is better for BlockGame than copying DH directly
- Better than DH for this engine:
  - D3D12-native queues/fences instead of a second compute API
  - shared patch instancing instead of heavy per-section mesh uploads
  - exact/LOD depth overlap to hide streaming pop
  - page-atlas design that naturally supports compute, culling, and indirect draw
- Useful DH inspiration:
  - distant terrain must be a separate representation
  - quality should decay gradually with distance
  - caching/persistence can matter later
- Useful C2ME inspiration:
  - batch work aggressively
  - cache reusable worldgen results
  - keep fallback paths and failure handling explicit
- Do **not** copy from C2ME:
  - OpenCL runtime choice
  - Java task graph shapes verbatim
  - device-whitelist/blocklist complexity as the primary shipping path

## Implementation Phases
### Phase A: Remove obsolete far terrain and repurpose settings
- Delete the old far-terrain manager path and `FarChunk`-based shell usage.
- Replace settings and UI with `Exact Radius` and `Total Radius`.
- Rename constants:
  - `kMaxUserRenderDistance` -> `kMaxExactRenderDistanceChunks = 48`
  - add `kMaxTotalRenderDistanceChunks = 500`
- Keep fog start, but derive default far plane from `totalRadiusChunks * 16 + padding`.

### Phase B: CPU-backed LOD source-page cache
- Build `TerrainLodManager` with its own page cache and tile cache.
- Implement `LodSourceBuilder` using existing terrain sampling and climate/surface caches.
- Add batched page sampling API:
  - page-aligned surface sampling at base pitch `4 blocks`
  - return height, water, material, biome-tint metadata in one batch
- Add GPU atlas allocation and page LRU eviction.
- Default budget:
  - `<=4 GB VRAM`: `192` pages
  - `4..8 GB VRAM`: `384` pages
  - `>8 GB VRAM`: `512` pages

### Phase C: Shared-patch LOD renderer
- Add static `32 x 32` grid patch VB/IB.
- Add per-instance tile buffer.
- Add LOD terrain shaders:
  - VS: sample height/water, displace vertices
  - PS: material lookup, biome tint, fog integration
- Render order:
  - exact terrain
  - LOD terrain
  - optional separate far-water pass

### Phase D: Transitions, invalidation, and edit behavior
- Exact/LOD overlap band enabled by default.
- When blocks change near the exact/LOD boundary, invalidate the intersecting source pages.
- Far-only LOD pages are regenerated lazily.
- Teleports:
  - keep old LOD tiles visible until the first visible ring of the new camera origin is ready
  - prioritize center-visible pages before peripheral pages

### Phase E: D3D12 compute page processing
- Add compute shader passes for:
  - normal generation
  - mip generation
  - min/max reduction
  - optional canopy/coverage mask derivation
- Keep CPU page synthesis as fallback.

### Phase F: GPU culling and indirect draw
- Add a dedicated compute pass for visible-tile compaction.
- Emit indirect args for LOD draws.
- Switch LOD terrain submission from CPU loops to `ExecuteIndirect`.
- Optional next step: Hi-Z occlusion culling after frustum culling is stable.

### Phase G: GPU terrain-page synthesis
- Port the macro terrain sampling path used by `surfaceMap/climateMap` into HLSL compute.
- Generate source pages directly on GPU from page coordinates, world seed, and biome/worldgen constants.
- Keep CPU parity/fallback for validation and unsupported cases.

### Phase H: Optional persistence
- Add an on-disk LOD cache only after runtime behavior is correct.
- Cache key:
  - world seed
  - biome/worldgen profile hash
  - page coord
  - schema version
- This is optional for v1, but it is the right follow-up if you want “nearly instant” revisit loads closer to Distant Horizons behavior.

## Important API / Interface Changes
- Replace `RenderDistanceSettings` with:
  - `int exactChunks`
  - `int totalChunks`
  - `int fogStartBlocks`
- Add to `ChunkManager`:
  - `exactRenderDistanceChunks()`
  - `totalRenderDistanceChunks()`
  - `setExactRenderDistanceChunks(int)`
  - `setTotalRenderDistanceChunks(int)`
  - `lodProfilingSnapshot()`
- Remove or deprecate:
  - `farRenderDistanceBlocks()`
  - `setFarRenderDistanceBlocks(int)`
  - `setFarTerrainEnabled(bool)`
  - `setLodEnabled(bool)`
  - `lodEnabled()`
- Add `TerrainLodManager` public surface:
  - `update(camera, settings, exactCoverage)`
  - `collectRenderSubmission(...)`
  - `invalidateWorldRegion(...)`
  - `profilingSnapshot()`
- Extend benchmark JSON with:
  - `lod.page_build`
  - `lod.page_upload`
  - `lod.compute_ms`
  - `lod.cull_ms`
  - `lod.indirect_ms`
  - `lod.visible_tiles`
  - `lod.missing_tiles`
  - `lod.page_cache_hit_rate`
  - `lod.exact_overlap_tiles`

## Tests and Benchmarks
### Functional scenarios
- Exact radius `48`, total radius `48`: LOD must be fully inactive.
- Exact `48`, total `64`: first LOD ring visible, no cracks at boundary.
- Exact `48`, total `500`: continuous horizon coverage, no empty bands.
- Rapid sprint: no visible LOD ring desync or missing outer ring.
- High mountain / valley view: no seam cracks between LOD levels.
- Ocean/coastline: water surface lines up with far terrain.
- Teleport far away: old LOD holds until new center-visible pages appear.
- Break/place blocks near chunk `48` boundary: source-page invalidation works and no hole/persistent stale LOD remains.

### Performance scenarios
- `lod_static_128`
- `lod_static_500`
- `lod_sprint_128`
- `lod_sprint_500`
- `lod_turn_heavy_500`
- `lod_vertical_500`
- `lod_teleport_500`
- `lod_spin_500`

### Acceptance criteria
- Current `48`-chunk exact-terrain frame pacing must not regress by more than `10%`.
- At `500` total chunks, there must be no empty horizon gaps once the first LOD settle pass completes.
- No exact/LOD boundary cracks or temporary “sky holes.”
- No multi-second stalls from LOD updates.
- If compute initialization fails, the CPU-backed page builder must still render LOD correctly.
- Do **not** use your friend’s `2000+ FPS` number as acceptance. That is hardware/content dependent. Use frame pacing, visible coverage, and no-gap/no-stall criteria instead.

## Assumptions and Defaults
- D3D12 remains the only graphics/compute API.
- OpenCL is out of scope.
- LOD beyond `48` chunks is visual-only by default.
- The first version approximates the macro surface and may not preserve caves/overhangs.
- Exact terrain keeps gameplay truth; LOD is presentation.
- Two sliders are the shipping UI model.
- Total radius cap is `500 chunks`, not `500 blocks`.
- Old far terrain is removed as part of this work, not retained behind a toggle.

## Inspiration and Research Basis
- Current exact-distance cap and far-terrain API surface: [chunk_manager.h](/Users/Jacob/Documents/code/BlockGame/src/chunk_manager.h), [chunk_manager.cpp](/Users/Jacob/Documents/code/BlockGame/src/chunk_manager.cpp), [main.cpp](/Users/Jacob/Documents/code/BlockGame/src/main.cpp), [renderer.cpp](/Users/Jacob/Documents/code/BlockGame/src/renderer.cpp)
- Local Distant Horizons baseline: [Readme.md](/Users/Jacob/Documents/code/BlockGame/distant-horizons/Readme.md)
- Local C2ME baseline: [README.md](/Users/Jacob/Documents/code/BlockGame/build/release/c2me-example/README.md), [ModuleEntryPoint.java](/Users/Jacob/Documents/code/BlockGame/build/release/c2me-example/c2me-opts-accel-opencl/src/main/java/com/ishland/c2me/opts/accel/opencl/ModuleEntryPoint.java), [Config.java](/Users/Jacob/Documents/code/BlockGame/build/release/c2me-example/c2me-opts-accel-opencl/src/main/java/com/ishland/c2me/opts/accel/opencl/common/Config.java), [BatchingBiomeNoiseStatus.java](/Users/Jacob/Documents/code/BlockGame/build/release/c2me-example/c2me-opts-accel-opencl/src/main/java/com/ishland/c2me/opts/accel/opencl/common/chunksystem_integration/BatchingBiomeNoiseStatus.java), [Stage1Cache.java](/Users/Jacob/Documents/code/BlockGame/build/release/c2me-example/c2me-opts-accel-opencl/src/main/java/com/ishland/c2me/opts/accel/opencl/common/gen/cache/Stage1Cache.java)
- Official D3D12 references: [Indirect drawing and GPU culling](https://learn.microsoft.com/en-us/windows/win32/direct3d12/indirect-drawing-and-gpu-culling-), [Indirect Drawing](https://learn.microsoft.com/en-us/windows/win32/direct3d12/indirect-drawing), [Multi-engine synchronization](https://learn.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization), [Using Resource Barriers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12)
- OpenCL reference used only to justify not adopting it here: [Khronos OpenCL overview](https://www.khronos.org/opencl)
