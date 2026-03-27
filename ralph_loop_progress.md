# Completed slices
- Step 5 input contract tightened: `ExactChunkColumnDescriptor` now carries authoritative `grassTintIndex`, and exact CPU meshing reuses stored descriptor tint data instead of re-sampling biome tint per column.
- Step 5 GPU-input packing added: chunk generation now packs GPU-ready exact column descriptors plus sparse voxel edits for structures, pending structure edits, and block overlays.
- Residency groundwork tightened: exact chunk residency can now recognize a truly GPU-resident nonlocal chunk independently from the legacy CPU upload state checks.
- Exact compute shader set added and compiling: synth, generic sparse stamp, first-pass light, face count, prefix, and face emit shaders now exist with a shared exact chunk packing contract.
- Exact nonlocal build loop is now live in C++: nonlocal exact `JobType::Mesh` work queues exact GPU synth/stamp/light/emit builds, exact commit allocates near mesh pages plus draw-record slots, and renderer-side upload sync now includes the exact compute fence.
- Near exact rendering now reuses the shared GPU cull + `ExecuteIndirect` path: near render batches can source GPU draw records from exact-resident pages and render without CPU vertex/index uploads.
- Startup crash slice shipped: exact descriptor/sparse upload buffers are now kept alive until the copy flush completes, which removed the device-removed startup failure caused by freed upload resources.
- Startup follow-up guard shipped: CPU relight now excludes GPU-resident exact chunks, preventing the old relight path from dereferencing nonlocal GPU-resident chunks during startup.

# Current architecture state
- Steps 1-4 groundwork is present: nonlocal exact chunks already store compact column descriptors and CPU-authored sparse structure voxel edits, and CPU-dense block/light arrays are released outside the local interaction bubble.
- Nonlocal exact chunks now store GPU-ready packed descriptor/edit payloads, submit exact GPU synth/stamp/light/face passes through the shared compute context pattern, and commit results into near-page GPU mesh buffers plus draw-record slots.
- Nonlocal exact `JobType::Mesh` work no longer depends on CPU vertex/index generation outside the CPU-authoritative bubble; `buildRenderData()` skips CPU draw submission for exact GPU-resident chunks and exposes their GPU draw records to the renderer.
- Renderer-side near and far passes now share the same GPU cull + indirect draw path, and graphics queue synchronization includes the exact compute fence before drawing committed exact pages.
- Legacy CPU upload logic still exists for CPU-authoritative/local chunks and fallback/debug responsibilities, but it is no longer the intended runtime path for nonlocal exact meshing.
- Current runtime blocker is no longer startup crash/hang; it is exact preload progress stalling partway through large startup fills while the window remains responsive.

# Remaining gaps for step 5
- Resolve the current exact preload stall so the GPU synth/stamp path continues progressing past the initial resident set during large startup fills.
- Re-verify that first visibility is still gated on synth + stamp completion for all nonlocal exact chunks once the preload stall is fixed.
- Re-run structure-first-visibility checks after the stall fix to confirm sparse structure stamping still lands before visibility in benchmark-sized loads.

# Remaining gaps for step 6
- Validate that the GPU first-pass lighting path is covering all nonlocal exact chunks once the preload stall is removed.
- Audit seam-dirty relight scheduling now that CPU relight skips GPU-resident chunks, and ensure conservative seam defaults plus reruns still fire where needed.
- Add or extend runtime telemetry for exact GPU lighting cost during the larger traversal benchmarks if current counters are insufficient.

# Remaining gaps for step 7
- Resolve the startup/preload plateau that currently leaves many requested exact chunks unready even though near exact GPU cull and indirect draw are wired.
- Validate shadow rendering and draw-record lifetime under churn for exact GPU-resident near chunks.
- Track residual nonlocal exact CPU upload telemetry during benchmarks and retire redundant branches only after runtime parity is proven.

# Known regressions
- Startup crash/not-responding is fixed, but `spawn_preload` currently plateaus with many exact chunks still pending (`1219 / 4629` ready in the latest run) while the window stays responsive.
- Runtime parity for the new exact GPU path is not proven yet; shadow/traversal coverage and long-run preload completion still need validation.

# Bench results
- Local verification: `cmake --build build --config Release` passes after the startup crash fixes and relight guard.
- Focused startup smoke: launched `build\\Release\\blockgame.exe`, held for 20 seconds, window remained responsive, and `build\\Release\\blockgame_crash.log` did not update.
- Benchmark smoke: `tools\\run_chunk_benchmark.ps1 -SkipBuild -ExactChunks 16 -TotalChunks 16 -Scenarios spawn_preload -MaxScenarioSeconds 45 -NotRespondingSeconds 15` no longer hit the watchdog; the run stayed responsive for several minutes but exact preload stalled at `1219 / 4629` ready chunks.

# Exact files changed
- `src/terrain/terrain_generator.h`
- `src/terrain/terrain_generator.cpp`
- `src/chunk_manager.cpp`
- `src/chunk_manager.h`
- `src/chunk_manager_gpu_contexts.inl`
- `src/renderer.h`
- `src/renderer.cpp`
- `src/main.cpp`
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
- Trace why exact GPU residency/build readiness stops advancing after the first startup batches, then fix that single stall point before taking on broader parity validation.
