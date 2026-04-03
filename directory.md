# BlockGame Directory Map

This file is the maintained map of the authored BlockGame codebase.

Use it to find major systems quickly without re-reading the whole repository.
Do not put line numbers in this file. Keep it about component names, ownership, and file locations.

Maintenance rule:
- When a change adds, removes, renames, or moves a major component, update this file in the same change.
- If a file stops owning a component listed here, remove or rewrite the entry instead of leaving stale notes behind.

Scope:
- This map covers authored gameplay, renderer, shader, config, and tooling files.
- It intentionally does not catalog vendored headers under `include/`, prebuilt binaries under `libs/`, or generated output under `build/`.

## Fast Navigation

- Terrain shape, biome blending, or worldgen tuning: start with `src/terrain/`, then read `assets/worldgen.toml` and `assets/biomes/`.
- Exact chunk loading, block edits, lighting propagation, or startup streaming: start with `src/chunk_manager.h`, `src/chunk_manager.cpp`, and `src/chunk_manager_support.h`.
- Far distance terrain or horizon continuity: start with `src/chunk_manager_far_terrain.inl`, `src/chunk_manager_gpu_contexts.inl`, `src/terrain/far_lod_worldgen.*`, and the `assets/shaders/far_lod_*` plus `assets/shaders/lod_gpu_cull.hlsl` files.
- Renderer, fog, sky, atmosphere, or tone mapping: start with `src/renderer.h`, `src/renderer.cpp`, and `assets/shaders/`.
- Player controls, UI toggles, teleport, or debug windows: start with `src/main.cpp`, `src/input_context.*`, and `src/camera.*`.
- Mob import, animation, or AI: start with `src/mob_model.*`, `src/mob_system.*`, and `assets/mobs/`.
- Benchmarks, capture automation, or reproducible screenshots: start with `src/main.cpp` benchmark/capture structs and the `tools/` PowerShell/Python helpers.

## Top-Level Build And Repo Files

- `CMakeLists.txt`: root build graph; defines the `blockgame` executable, the `blockgame_shader_precompiler` helper, post-build asset copying, and release shader precompilation.
- `cmake/precompile_shaders.cmake`: release-only shader precompile step that runs the shader precompiler after build.
- `blockgame.rc`: Windows resource file that binds the application icon.
- `alert.ps1`: post-task notification helper used to ping the user when work is complete.
- `biome_query.py`: root copy of the biome/climate query utility; same role as `tools/biome_query.py`.
- `AGENTS.md`: uppercase repository instructions consumed by future sessions.
- `agents.md`: lowercase duplicate instructions file; keep it synchronized with `AGENTS.md`.

## Runtime Entry, Loop, And UI

- `src/main.cpp`: application bootstrap and the main game loop.
  Owns these major runtime components:
  `BenchmarkScenarioKind`
  `BenchmarkConfig`
  `BenchmarkSpikeRecord`
  `BenchmarkSpikeSummary`
  `BenchmarkRuntimeState`
  `BenchmarkFrameSummary`
  `ScreenshotSweepConfig`
  `ScreenshotSweepState`
  `ScreenshotReproConfig`
  `ScreenshotReproState`
  `CapturePlacementAction`
  `CaptureOverridesConfig`
  `AABB`
  `AxisMoveResult`
  It also owns:
  crash logging and hang watchdog setup
  launch-time debug environment flags
  benchmark env parsing and benchmark JSON/progress writers
  screenshot sweep and single-view repro automation
  capture-time block placement overrides
  player physics and movement stepping
  world startup/preload gating
  ImGui windows named `Debug Overlay`, `Controls`, `Lighting Lab`, `LOD Diagnostics`, `Loading Overlay`, `Render Distance`, `Teleport`, `Block Picker`, and `Profiling Overlay`

## Camera And Input

- `src/camera.h`: declares the `Camera` component and its movement/orientation state.
- `src/camera.cpp`: implements `Camera::processMouse` and `Camera::updateVectors`.
- `src/input_context.h`: declares `InputContext`, `PlayerInputState`, gameplay/UI capture helpers, and GLFW callback entry points.
- `src/input_context.cpp`: implements the GLFW callbacks, gameplay mouse capture rules, key handling, UI toggles, block placement mode switching, and `computePlayerInputState`.

## Chunk Streaming, World State, And Block Interaction

- `src/chunk_manager.h`: public chunk/world API and shared render data types.
  Main public component names:
  `StreamingPhase`
  `VerticalStreamingConfig`
  `BlockId`
  `RaycastHit`
  `WorldVertex`
  `MobVertex`
  `BlockTextureAtlasConfig`
  `LightSample`
  `ChunkRenderBatch`
  `ExactChunkRenderBatch`
  `MobRenderBatch`
  `WorldRenderData`
  `RenderDistanceSettings`
  `ChunkProfilingSnapshot`
  `BenchmarkStageStats`
  `BenchmarkQueueDepthStats`
  `BenchmarkCacheStats`
  `ChunkBenchmarkReport`
  `StreamingStatusSnapshot`
  `LodDiagnosticsTileSnapshot`
  `LodDiagnosticsSnapshot`
  `RecentEditHoleChunkInfo`
  `RecentEditHoleDebugSnapshot`
  `Frustum`
  `ChunkManager`

- `src/chunk_manager.cpp`: core world subsystem implementation behind `ChunkManager::Impl`.
  This is the single biggest ownership file in the repo. Major internal component names here include:
  `AtomicLatencyHistogram`
  `AtomicCountHistogram`
  `AtomicDepthHistogram`
  `ChunkBenchmarkMetrics`
  `BlockLightingProperties`
  `GrassTintIndex`
  `DefaultTreeCandidate`
  `DefaultTreeBlockPalette`
  `DarkOakTreeCandidate`
  `AcaciaTreeCandidate`
  `StructureVoxelEdit`
  `GpuExactColumnDescriptor`
  `GpuExactSparseVoxel`
  `GpuWorldgenPageColumn`
  `GpuExactDescriptorBuildParams`
  `GpuExactPrepassRecord`
  `MeshData`
  `ChunkState`
  `ExactChunkResidencyMode`
  `ChunkYInterval`
  `ColumnChunkIntervals`
  `Chunk`
  `ProfilingCounters`
  `PendingStructureEdit`
  `BlockEditOverlayEntry`
  `UploadQueueBucket`
  `UploadQueueEntry`
  `PendingCommitQueueEntry`
  `UploadQueueEntryComparer`
  `ChunkBuildScratch`
  `ChunkManager::Impl`
  Major responsibilities inside this file:
  exact chunk lifecycle and startup preload
  CPU terrain generation and chunk meshing
  exact-GPU chunk descriptor/synthesis/light/emit orchestration
  chunk upload, commit, and residency page management
  relighting, skylight propagation, and block-light debug
  block breaking, block placement, and overlay edits
  tree/structure spawning and structure edit dispatch
  worldgen page dependency tracking
  structure region dependency tracking
  vertical streaming radius control
  far-LOD shell coordination
  profiling, benchmark metrics, LOD diagnostics, and recent-edit-hole diagnostics

- `src/chunk_manager_support.h`: small shared chunk-manager internals.
  Main component names:
  `ChunkHasher`
  `ColumnHasher`
  `JobType`
  `JobServiceClass`
  `Job`
  `JobQueueSnapshot`
  `ChunkPriorityKey`
  `JobQueue`
  `ChunkBlockView`
  `ColumnManager`

- `src/chunk_manager_job_queue.cpp`: implementation of `JobQueue`, including priority updates, stage balancing, and worker accounting.

- `src/chunk_manager_column_manager.cpp`: implementation of `ColumnManager`, which keeps highest-solid-block information per world column.

- `src/chunk_manager_structure_registry.inl`: internal structure cache and query system used by exact chunks and far LOD.
  Main component names:
  `StructureAabb`
  `StructureType`
  `StructureInstance`
  `StructureRegionKey`
  `StructureRegionKeyHasher`
  `StructureChunkColumnHasher`
  `StructureChunkColumnSpan`
  `StructureChunkVoxelSpan`
  `StructureBvhNode`
  `StructureBvh`
  `StructureRegion`
  `StructureRegistryProfilingSnapshot`
  `StructureRegistry`
  This file owns structure voxelization, BVH queries, per-region instance caches, per-chunk voxel edit spans, and profiling counters.

- `src/chunk_manager_gpu_contexts.inl`: internal GPU helper classes shared by chunk streaming.
  Main component names:
  `UploadContext`
  `FarLodGpuContext`
  `BlockFace`
  This file owns copy-queue uploads plus the compute/cull pipeline plumbing used by far LOD and exact GPU chunk work.

- `src/chunk_manager_far_terrain.inl`: internal far-terrain shell manager.
  Main component names:
  `FarTerrainManager`
  `FarLodChunkKey`
  `FarLodLevelConfig`
  `FarLodVoxel`
  `GpuBlockFaceUv`
  `GpuStructureInstance`
  `GpuStructureRegionState`
  `FarLodChunkCpu`
  `FarLodChunkGpuState`
  This file owns far-LOD tile levels, CPU/GPU shell state, distant structure injection, and far draw record residency.

- `src/lod_page_compute_context.inl`: standalone compute-context helper for the older page-based LOD synthesis path.
  Main component names:
  `PageComputeContext`
  `PageComputeContext::Summary`
  `PageComputeContext::PassTimings`

## Terrain Generation And Biomes

- `src/terrain/worldgen_profile.h`: declares `FbmSettings`, `NoiseProfile`, and `WorldgenProfile`.
- `src/terrain/worldgen_profile.cpp`: loads `WorldgenProfile` from `assets/worldgen.toml`.

- `src/terrain/biome_database.h`: biome schema and lookup API.
  Main component names:
  `BiomeDefinition`
  `BiomeDefinition::GenerationProperties`
  `BiomeDefinition::TransitionBiomeDefinition`
  `BiomeDefinition::SubBiomeDefinition`
  `BiomeDefinition::InterpolationCurve`
  `BiomeDefinition::SoilCreepSettings`
  `BiomeDefinition::StripeSettings`
  `BiomeDefinition::WaterFillSettings`
  `BiomeDefinition::TerrainSettings`
  `BiomeDatabase`

- `src/terrain/biome_database.cpp`: parses biome TOML files, resolves flags/properties, computes derived biome data, and builds the `BiomeDatabase`.

- `src/terrain/climate_map.h`: climate-layer interfaces and cached fragment model.
  Main component names:
  `BiomeBlend`
  `ClimateSample`
  `ClimateFragment`
  `ClimateGenerator`
  `NoiseVoronoiClimateGenerator`
  `ClimateMap`

- `src/terrain/climate_map.cpp`: implements biome site placement, transition/sub-biome expansion, candidate seed caching, climate fragment generation, and the `ClimateMap` fragment cache.

- `src/terrain/surface_map.h`: surface-shape interfaces and cache.
  Main component names:
  `SurfaceColumn`
  `SurfaceFragment`
  `SurfaceGenerator`
  `MapGenV1`
  `SurfaceMap`

- `src/terrain/surface_map.cpp`: implements `MapGenV1`, Perlin/fbm/ridge noise helpers, surface fragment filling, and the `SurfaceMap` cache.

- `src/terrain/terrain_generator.h`: chunk materialization API.
  Main component names:
  `ColumnSample`
  `ColumnSample::BlendDebugInfo`
  `ColumnBuildResult`
  `ChunkGenerationSummary`
  `ExactChunkColumnDescriptor`
  `TerrainColumnBlocks`
  `TerrainGenerator`

- `src/terrain/terrain_generator.cpp`: derives final exact-chunk column descriptors, chooses terrain blocks, and materializes chunk block data from climate/surface samples.

- `src/terrain/far_lod_worldgen.h`: packed worldgen tables sent to far-LOD compute passes.
  Main component names:
  `FarLodCoastProfile`
  `FarLodBiomeFlags`
  `FarLodGpuFloat2`
  `FarLodGpuWorldgenHeader`
  `FarLodGpuBiome`
  `FarLodGpuBiomeSelection`
  `FarLodGpuTransitionBiome`
  `FarLodGpuSubBiome`
  `FarLodWorldgenTables`
  `FarLodColumnSample`

- `src/terrain/far_lod_worldgen.cpp`: builds the packed far-LOD worldgen tables from `BiomeDatabase` and `WorldgenProfile`.

## Mobs

- `src/mob_model.h`: declares the imported model representation.
  Main component names:
  `MobPartAnimationRole`
  `MobModelPart`
  `MobModel`
  `MobModelLibrary`

- `src/mob_model.cpp`: lightweight Bedrock-style geometry parser and loader.
  Important internal names:
  `JsonValue`
  `JsonArray`
  `JsonObjectEntry`
  `JsonObject`
  `JsonParser`
  `BedrockCube`
  `BedrockBone`
  `ParsedModelGeometry`
  `FaceUvRect`

- `src/mob_system.h`: runtime passive-mob owner.
  Main component names:
  `MobTextureBinding`
  `MobSystem`
  `MobSystem::PassiveState`
  `MobSystem::MobInstance`

- `src/mob_system.cpp`: passive mob definitions, movement/collision updates, head-look and walk-cycle behavior, spawning, and render-batch emission.

## Renderer

- `src/renderer.h`: public renderer surface and all major render-setting structs.
  Main component names:
  `LoadedTexture`
  `TerrainDebugView`
  `AtmosphereSettings`
  `TonemapSettings`
  `RenderDebugSettings`
  `EnvironmentState`
  `RendererProfilingSnapshot`
  `Renderer`

- `src/renderer.cpp`: D3D12 implementation.
  Major internal names:
  `RuntimeAtlasInfo`
  `FrustumPlane`
  `Renderer::AtmosphereRenderer`
  This file owns:
  device/swapchain setup
  command queues and frame resources
  descriptor heaps and SRV allocation
  depth buffer, depth pyramid, and shadow map resources
  pipeline creation for terrain, exact terrain, mobs, sky, clouds, tonemap, cull, and indirect draws
  GPU culling for far batches and exact batches
  screenshot readback
  atmosphere LUT generation and sky rendering

- `src/shader_manifest.h`: shader build manifest and compiled shader naming helpers.
  Main component names:
  `ShaderCompileSpec`
  `kBlockGameShaderCompileSpecs`
  `compiledShaderFileName`
  `compiledShaderPath`
  `compiledShaderPathForSource`

## Shader Files

### Shared Includes

- `assets/shaders/world_lighting_common.hlsli`: shared terrain lighting and fog decode helpers.
- `assets/shaders/exact_chunk_common.hlsli`: shared exact-GPU structs such as `GpuExactColumnDescriptor`, `GpuExactFaceDescriptor`, `GpuExactPageMetadata`, `GpuExactChunkAllocationRecord`, and `GpuBlockFaceUv`.
- `assets/shaders/atmosphere_common.hlsli`: shared atmosphere constants and sample structs.
- `assets/shaders/base_game_sky_common.hlsli`: shared base-sky color helpers.

### Terrain, Exact Terrain, Shadow, And Highlight

- `assets/shaders/world_vs.hlsl`: standard terrain vertex shader.
- `assets/shaders/world_near_ps.hlsl`: near terrain pixel shader.
- `assets/shaders/world_far_ps.hlsl`: far terrain pixel shader.
- `assets/shaders/exact_world_vs.hlsl`: exact-GPU terrain vertex shader.
- `assets/shaders/exact_shadow_vs.hlsl`: exact-GPU shadow vertex shader.
- `assets/shaders/shadow_vs.hlsl`: CPU-mesh shadow vertex shader.
- `assets/shaders/shadow_ps.hlsl`: shadow pixel shader.
- `assets/shaders/block_outline_vs.hlsl`: selected-block outline vertex shader.
- `assets/shaders/block_outline_ps.hlsl`: selected-block outline pixel shader.

### Exact Chunk GPU Build Pipeline

- `assets/shaders/exact_chunk_descriptor_gen_cs.hlsl`: builds exact column descriptors from worldgen pages and skylight inputs.
- `assets/shaders/exact_chunk_synth_cs.hlsl`: synthesizes exact chunk voxel state.
- `assets/shaders/exact_chunk_structure_stamp_cs.hlsl`: stamps structure edits into exact chunk scratch data.
- `assets/shaders/exact_chunk_halo_cache_cs.hlsl`: prepares halo/neighborhood cache state for exact chunk work.
- `assets/shaders/exact_chunk_light_cs.hlsl`: exact chunk lighting compute pass.
- `assets/shaders/exact_chunk_seam_export_cs.hlsl`: exports seam data used across chunk/page boundaries.
- `assets/shaders/exact_chunk_face_count_cs.hlsl`: counts visible faces for exact chunks.
- `assets/shaders/exact_chunk_face_prefix_cs.hlsl`: prefix scan for exact chunk face emission.
- `assets/shaders/exact_chunk_face_emit_cs.hlsl`: emits exact chunk draw data and completion records.
- `assets/shaders/exact_gpu_cull.hlsl`: exact draw visibility cull and indirect command build.

### Far LOD GPU Build Pipeline

- `assets/shaders/far_lod_column_atlas_update_canonical_cs.hlsl`: updates far-LOD atlas samples plus chunk-seed/sample caches used by distant terrain generation.
- `assets/shaders/far_lod_chunk_synth_cs.hlsl`: synthesizes far-LOD chunk columns.
- `assets/shaders/far_lod_chunk_structure_stamp_cs.hlsl`: stamps distant structures into far-LOD voxel data.
- `assets/shaders/far_lod_chunk_face_count_cs.hlsl`: counts visible faces for far-LOD tiles.
- `assets/shaders/far_lod_chunk_face_prefix_cs.hlsl`: grouped prefix scan passes for far-LOD face output.
- `assets/shaders/far_lod_chunk_face_emit_cs.hlsl`: emits far-LOD vertices, indices, and draw records.
- `assets/shaders/lod_gpu_cull.hlsl`: far-LOD visibility cull and indirect draw build.
- `assets/shaders/lod_page_compute.hlsl`: page-based compute path for column synthesis, structure stamping, and face masks.

### Sky, Atmosphere, Clouds, Post

- `assets/shaders/fullscreen_vs.hlsl`: fullscreen triangle vertex shader used by post and LUT passes.
- `assets/shaders/base_sky_ps.hlsl`: base-game sky dome pixel shader.
- `assets/shaders/clouds_vs.hlsl`: cloud layer vertex shader.
- `assets/shaders/clouds_ps.hlsl`: cloud layer pixel shader.
- `assets/shaders/tone_map_ps.hlsl`: final scene tonemap shader.
- `assets/shaders/atmosphere_transmittance_ps.hlsl`: atmosphere transmittance LUT pass.
- `assets/shaders/atmosphere_multiscattering_ps.hlsl`: atmosphere multi-scattering LUT pass.
- `assets/shaders/atmosphere_skyview_ps.hlsl`: sky-view LUT pass.
- `assets/shaders/atmosphere_sky_ps.hlsl`: final enhanced-atmosphere sky shader.
- `assets/shaders/atmosphere_aerial_perspective_ps.hlsl`: aerial-perspective LUT/volume pass.

### Utilities And Other Render Paths

- `assets/shaders/depth_pyramid.hlsl`: depth-pyramid compute builder used by GPU cull passes.
- `assets/shaders/mob_vs.hlsl`: mob vertex shader.
- `assets/shaders/mob_ps.hlsl`: mob pixel shader with optional aerial perspective and shadow inputs.

## Authored Gameplay Data And Assets

- `assets/worldgen.toml`: global worldgen seed, climate generator selection, sea level, and the `noise.main`, `noise.medium`, `noise.detail`, and `noise.mountain` FBM profiles.

- `assets/biomes/beach.toml`: biome definition for `Beach`.
- `assets/biomes/birch_forest.toml`: biome definition for `Birch Forest`.
- `assets/biomes/coast_cliffs.toml`: biome definition for `Coastal Cliffs`.
- `assets/biomes/dark_forest.toml`: biome definition for `Dark Forest`.
- `assets/biomes/desert.toml`: biome definition for `Desert`.
- `assets/biomes/forest.toml`: biome definition for `Forest`.
- `assets/biomes/grasslands.toml`: biome definition for `Grasslands`.
- `assets/biomes/little_mountains.toml`: biome definition for `Little Mountains`.
- `assets/biomes/ocean.toml`: biome definition for `Ocean`.
- `assets/biomes/ocean_shelf.toml`: biome definition for `Ocean Shelf`.
- `assets/biomes/savanna.toml`: biome definition for `Savanna`.
- `assets/biomes/taiga.toml`: biome definition for `Taiga`.
  Every biome TOML can own some mix of:
  block palette
  tree generation flags
  radius and spawn weighting
  height/roughness/hills/mountains tuning
  interpolation and property masks
  transition biome lists
  sub-biome lists
  soil creep
  stripes
  water fill
  coast profile selection

- `assets/mobs/README.md`: rules for mob geometry import, texture pairing, and expected Bedrock bone names.
- `assets/mobs/pig.geo.json`: pig geometry definition with body, head, and leg bones.
- `assets/mobs/cow.geo.json`: cow geometry definition with body, head, and leg bones.
- `assets/mobs/pig.png`: pig texture used by the imported pig model.
- `assets/mobs/cow.png`: cow texture used by the imported cow model.
- `block_atlas.png`: terrain texture atlas loaded by the renderer and chunk manager.
- `block_atlas_guide.txt`: atlas layout reference used when validating tile placement.
- `assets/fonts/arial.ttf`: UI font asset loaded by the renderer/ImGui path.
- `assets/icon/icon.ico`: application icon used by `blockgame.rc`.

## Tooling, Benchmarks, Capture, And Content Helpers

- `tools/shader_precompiler.cpp`: incremental shader precompiler; walks `#include` dependencies, uses `shader_manifest.h`, prefers DXC, and falls back to D3DCompiler for legacy targets.

- `tools/run_chunk_benchmark.ps1`: canonical chunk benchmark runner.
  Owns:
  `Resolve-CMakePath`
  `Resolve-ExePath`
  `Write-WatchdogReport`
  `Invoke-BenchmarkScenario`
  It sets benchmark env vars, launches the game, kills hung runs, and writes summary/watchdog artifacts.

- `tools/run_lod_benchmark.ps1`: wrapper that reuses `run_chunk_benchmark.ps1` for LOD-enabled scenario sets.

- `tools/run_horizon_sweep.ps1`: launches the game in screenshot sweep mode, clears the output directory, and runs `tools/analyze_horizon_sweep.py`.

- `tools/run_lod_horizon_sweep.ps1`: timestamped LOD sweep launcher that sets exact/total chunk overrides before running the sweep and post-analysis.

- `tools/capture_repro.ps1`: single-camera repro capture helper.
  Supports:
  explicit `-Yaw` and `-Pitch`
  or `-LookX`, `-LookY`, `-LookZ`
  plus env overrides for time of day, exact/total chunks, fog, direct sun, debug view, block placements, and LOD-ready waiting.

- `tools/run_visual_suite.ps1`: manifest-driven capture runner for a fixed gallery of visual regression scenes.
  Main helper name:
  `Format-CapturePlacements`

- `tools/visual_suite.json`: named capture-scene manifest.
  Current scene names:
  `noon_hilltop`
  `shoreline_grazing`
  `forest_edge`
  `under_canopy_shade`
  `cave_mouth_overhang`
  `interior_lamp_test`
  `chunk_seam`
  `horizon_vista_exact`
  `horizon_vista_far`

- `tools/analyze_horizon_sweep.py`: BMP analyzer that scores captures for dark horizon bands and writes `analysis.csv`.
  Main component name:
  `CaptureScore`

- `tools/biome_query.py`: interactive biome/climate query model for inspecting biome influence and nearest matches outside the game.
  Main component names:
  `_Random`
  `TransitionDefinition`
  `SubBiomeDefinition`
  `BiomeDefinition`
  `BiomeDatabase`
  `BiomeSeed`
  `ChunkSeeds`
  `BiomeContribution`
  `ClimateSample`
  `ClimateModel`
  `WorldgenProfile`

- `tools/find_little_mountains.py`: deterministic search helper for locating nearby `Little Mountains` sites.
  Main component names:
  `Biome`
  `BiomeSite`

- `tools/sample_little_mountains.cpp`: standalone terrain sampling experiment for the Little Mountains biome and related noise behavior.
  Main component names:
  `OpenSimplexNoise`
  `BiomeDefinition`
  `LittleMountainSample`
  `PerlinNoise`
  `TerrainBasisSample`
  `TerrainBasisSampler`
  `LittleMountainSampler`
  `Biome`
  `BiomeSite`
  `LittleMountainColumnResult`

- `tools/update_block_atlas.ps1`: appends one or more 16x16 tiles to the bottom of `block_atlas.png`.

## Third-Party And Generated Areas To Ignore During Normal Navigation

- `include/`: vendored headers such as GLFW, GLM, stb, glad, and toml++.
- `libs/`: prebuilt GLFW binaries.
- `build/`: generated build outputs and copied runtime assets.
- `artifacts/`: generated benchmark and capture outputs.
