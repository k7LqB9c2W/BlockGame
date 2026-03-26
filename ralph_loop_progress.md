# Completed slices
- Step 5 input contract tightened: `ExactChunkColumnDescriptor` now carries authoritative `grassTintIndex`, and exact CPU meshing reuses stored descriptor tint data instead of re-sampling biome tint per column.
- Step 5 GPU-input packing added: chunk generation now packs GPU-ready exact column descriptors plus sparse voxel edits for structures, pending structure edits, and block overlays.
- Residency groundwork tightened: exact chunk residency can now recognize a truly GPU-resident nonlocal chunk independently from the legacy CPU upload state checks.
- Exact compute shader set added and compiling: synth, generic sparse stamp, first-pass light, face count, prefix, and face emit shaders now exist with a shared exact chunk packing contract.

# Current architecture state
- Steps 1-4 groundwork is present: nonlocal exact chunks already store compact column descriptors and CPU-authored sparse structure voxel edits, and CPU-dense block/light arrays are released outside the local interaction bubble.
- Nonlocal exact chunks now also store GPU-ready packed descriptor/edit payloads that match the intended exact compute inputs.
- Nonlocal exact chunks still route through the legacy CPU mesh/upload path after generation.
- `uploadReadyMeshes()` still performs nonlocal exact CPU uploads, and `buildRenderData()` still only sources exact near terrain from CPU-uploaded near pages.
- Far LOD already has reusable GPU compute context, GPU-generated mesh page allocation, GPU cull records, and `ExecuteIndirect` draw wiring that should be extended to exact chunks.
- The exact HLSL side is now defined against the packed CPU descriptor/edit contract, but the renderer-side submission/commit loop is not yet wired to dispatch those shaders.

# Remaining gaps for step 5
- Add a nonlocal exact GPU build lifecycle distinct from CPU mesh/upload state.
- Upload exact column descriptors plus generic sparse structure voxel edits as GPU inputs per nonlocal exact chunk.
- Dispatch exact GPU synth and generic structure stamp before chunk visibility.
- Remove nonlocal exact dependence on CPU dense voxel materialization during content build.

# Remaining gaps for step 6
- Add GPU exact lighting buffers/passes that write the near renderer's expected light channels.
- Replace nonlocal exact CPU relight/base-light fallback with GPU lighting inputs and conservative seam handling.
- Stop rebuilding nonlocal sky/light data from transient CPU chunk materialization.

# Remaining gaps for step 7
- Allocate exact GPU-generated near mesh pages and draw-record slots.
- Dispatch exact GPU face build and commit nonlocal exact chunks without CPU vertex/index uploads.
- Add near exact GPU cull plus `ExecuteIndirect` rendering while preserving shadow rendering coverage.
- Remove nonlocal exact usage of `uploadReadyMeshes()` and related CPU upload telemetry.

# Known regressions
- None newly introduced by the current slice.
- The existing nonlocal exact path still depends on CPU mesh/upload work; this is the main remaining regression against the target architecture.

# Bench results
- No runtime benchmark run yet for the current Ralph loop slices.
- Local verification: `cmake --build build --config Release` passes after descriptor packing and residency-state changes.
- Local verification: exact compute shaders compile successfully through the shader precompiler during the Release build.

# Exact files changed
- `src/terrain/terrain_generator.h`
- `src/terrain/terrain_generator.cpp`
- `src/chunk_manager.cpp`
- `src/shader_manifest.h`
- `assets/shaders/exact_chunk_common.hlsli`
- `assets/shaders/exact_chunk_synth_cs.hlsl`
- `assets/shaders/exact_chunk_structure_stamp_cs.hlsl`
- `assets/shaders/exact_chunk_light_cs.hlsl`
- `assets/shaders/exact_chunk_face_count_cs.hlsl`
- `assets/shaders/exact_chunk_face_prefix_cs.hlsl`
- `assets/shaders/exact_chunk_face_emit_cs.hlsl`
- `ralph_loop_progress.md`

# Next smallest slice
- Add a dedicated exact-chunk GPU context plus nonlocal exact build request/commit queues, then route nonlocal `JobType::Mesh` work away from `uploadReadyMeshes()` and into exact GPU synth/stamp/light/emit submission.
