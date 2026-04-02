#pragma once
// chunk_manager.h
// Declares the chunk streaming, terrain meshing, and GPU upload subsystem used by BlockGame.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <d3d12.h>
#include <glm/glm.hpp>

namespace terrain
{
struct ColumnSample;
}

inline constexpr float kNearPlane = 0.1f;
inline constexpr float kDefaultFarPlane = 256.0f;
inline constexpr float kFarPlanePadding = 96.0f;
inline constexpr float kCameraEyeHeight = 1.7f;
inline constexpr float kEpsilon = 1e-6f;
inline constexpr float kMaxRayDistance = 8.0f;
inline constexpr float kPlayerWidth = 0.6f;
inline constexpr float kPlayerHeight = 1.8f;
inline constexpr float kGravity = -20.0f;
inline constexpr float kJumpVelocity = 8.0f;
inline constexpr float kTerminalVelocity = -50.0f;
inline constexpr float kHorizontalDamping = 0.80f;
inline constexpr float kGroundSnapTolerance = 1e-3f;
inline constexpr float kAxisCollisionEpsilon = 1e-4f;

inline constexpr int kChunkEdgeLength = 16;
inline constexpr int kChunkSizeX = kChunkEdgeLength;
inline constexpr int kChunkSizeY = kChunkEdgeLength;
inline constexpr int kChunkSizeZ = kChunkEdgeLength;
inline constexpr int kChunkBlockCount = kChunkEdgeLength * kChunkEdgeLength * kChunkEdgeLength;
inline constexpr int kAtlasTileSizePixels = 16;
inline constexpr int kDefaultNearRenderDistance = 48;
inline constexpr int kDefaultViewDistance = kDefaultNearRenderDistance;
inline constexpr int kMaxExactRenderDistanceChunks = 48;
inline constexpr int kMaxUserRenderDistance = kMaxExactRenderDistanceChunks;
inline constexpr int kDefaultTotalRenderDistanceChunks = kDefaultNearRenderDistance;
inline constexpr int kMaxTotalRenderDistanceChunks = 500;
inline constexpr int kHiddenExactPreloadBufferChunks = 3;
inline constexpr int kDefaultStartupExactPreloadChunks = 12;
inline constexpr int kExtendedViewDistance = 320;
inline constexpr int kDefaultFarFogStartBlocks = 1400;

enum class StreamingPhase : std::uint8_t
{
    SpawnResolve = 0,
    ExactPreload,
    InteractiveNearOnly,
    FarRamp,
    SteadyState
};

struct VerticalStreamingConfig
{
    int minRadiusChunks{2};
    int maxRadiusChunks{320};
    int columnSlackChunks{1};
    int sampleRadiusChunks{3};
    int horizontalEvictionSlack{1};
    int verticalEvictionDeadbandChunks{1};
    int verticalEvictionExtraSlackChunks{0};
    int baseEvictionChunksPerFrame{64};
    int maxEvictionChunksPerFrame{128};
    int evictionBudgetBoostDivisor{96};
    int uploadBasePerColumn{8};
    int uploadRampDivisor{1};
    int uploadMaxPerColumn{56};
    int maxGenerationJobsPerColumn{24};
    int backlogColumnCapReleaseThreshold{2048};
    int verticalRadiusFalloffStep{8};

    struct GenerationBudgetSettings
    {
        int baseJobsPerFrame{96};
        float jobsPerHorizontalRing{5.0f};
        float jobsPerVerticalLayer{2.0f};
        int backlogStartThreshold{128};
        int backlogStepSize{128};
        int backlogBoostPerStep{40};
        int maxJobsPerFrame{1536};
        int minRingExpansionsPerFrame{1};
        int maxRingExpansionsPerFrame{12};
        int backlogRingStepSize{128};
        int columnCapBoostPerStep{6};
    } generationBudget{};

    int maxWorkerThreads{0};
};

inline constexpr VerticalStreamingConfig kVerticalStreamingConfig{};
inline constexpr std::size_t kUploadBudgetBytesPerFrame = 64ull * 1024ull * 1024ull;

inline constexpr std::size_t kMinBufferSizeBytes = 4ull * 1024ull;
inline constexpr std::size_t kUploadQueueScanLimit = 64ull;
inline constexpr int kBiomeSizeInChunks = 30; // Controls the width/height of each biome in chunks.

float computeFarPlaneForViewDistance(int viewDistance) noexcept;
float computeFarPlaneForDistanceBlocks(int farDistanceBlocks) noexcept;
extern float kFarPlane;

enum class BlockId : std::uint8_t
{
    Air = 0,
    Grass = 1,
    Wood = 2,
    Leaves = 3,
    Sand = 4,
    Water = 5,
    Stone = 6,
    SpruceLog = 7,
    SpruceLeaves = 8,
    Podzol = 9,
    DebugLamp = 10,
    DarkOakLog = 11,
    DarkOakLeaves = 12,
    BirchLog = 13,
    BirchLeaves = 14,
    AcaciaLog = 15,
    AcaciaLeaves = 16,
    Count
};

constexpr std::size_t toIndex(BlockId block) noexcept
{
    return static_cast<std::size_t>(block);
}

inline bool isSolid(BlockId block) noexcept
{
    return block != BlockId::Air;
}

struct RaycastHit
{
    bool hit{false};
    glm::ivec3 blockPos{0};
    glm::ivec3 faceNormal{0};
    float distance{0.0f};
};

struct WorldVertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 tileCoord;
    glm::vec2 atlasBase;
    glm::vec2 atlasSize;
    std::uint32_t lightingData{0};
};

struct MobVertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec4 color{1.0f};
};

struct BlockTextureAtlasConfig
{
    glm::ivec2 textureSizePixels{0};
    int tileSizePixels{0};
    int tileStridePixels{0};
    int tilePaddingPixels{0};
};

struct LightSample
{
    std::uint8_t sky{15};
    std::uint8_t block{0};
};

struct ChunkRenderBatch
{
    D3D12_VERTEX_BUFFER_VIEW vertexBufferView{};
    D3D12_INDEX_BUFFER_VIEW indexBufferView{};
    std::vector<std::uint32_t> indexCounts;
    std::vector<std::uint32_t> firstIndexLocations;
    std::vector<std::int32_t> baseVertices;
    struct GpuCullRecord
    {
        static constexpr std::uint32_t kReservedActiveBit = 1u << 30u;
        static constexpr std::uint32_t kReservedOverflowBit = 1u << 31u;
        static constexpr std::uint32_t kReservedFaceCountMask = kReservedActiveBit - 1u;

        glm::vec4 boundsMin{0.0f};
        glm::vec4 boundsMax{0.0f};
        std::uint32_t indexCount{0};
        std::uint32_t firstIndexLocation{0};
        std::int32_t baseVertex{0};
        std::uint32_t reserved{0};
    };
    std::vector<GpuCullRecord> gpuCullRecords;
    ID3D12Resource* gpuCullRecordBuffer{nullptr};
    std::uint32_t gpuCullRecordCount{0};
    bool supportsGpuCull{false};
    std::uint32_t debugPageIndex{0};
};

struct ExactChunkRenderBatch
{
    struct GpuCullRecord
    {
        static constexpr std::uint32_t kReservedActiveBit = 1u << 30u;
        static constexpr std::uint32_t kReservedOverflowBit = 1u << 31u;
        static constexpr std::uint32_t kReservedFaceCountMask = kReservedActiveBit - 1u;

        glm::vec4 boundsMin{0.0f};
        glm::vec4 boundsMax{0.0f};
        std::uint32_t faceCount{0};
        std::uint32_t faceOffset{0};
        std::uint32_t reserved0{0};
        std::uint32_t reserved{0};
    };

    ID3D12Resource* faceDescriptorBuffer{nullptr};
    ID3D12Resource* drawRecordBuffer{nullptr};
    ID3D12Resource* drawRecordMetadataBuffer{nullptr};
    std::vector<std::uint32_t> faceOffsets;
    std::vector<std::uint32_t> faceCounts;
    std::vector<std::uint32_t> recordIndices;
    std::uint32_t gpuCullRecordCount{0};
    bool supportsGpuCull{false};
    std::uint32_t debugPageIndex{0};
};

struct MobRenderBatch
{
    std::vector<MobVertex> vertices;
    std::vector<std::uint32_t> indices;
    D3D12_GPU_DESCRIPTOR_HANDLE textureSrv{};
    bool hasTexture{false};
};

struct WorldRenderData
{
    glm::ivec3 highlightedBlock{0};
    bool hasHighlight{false};
    std::vector<ChunkRenderBatch> nearBatches;
    std::vector<ExactChunkRenderBatch> exactNearBatches;
    std::vector<ChunkRenderBatch> farBatches;
    std::vector<MobRenderBatch> mobBatches;
    ID3D12Resource* exactBlockUvBuffer{nullptr};
    std::uint32_t exactBlockUvCount{0};
};

struct RenderDistanceSettings
{
    int exactChunks{kDefaultNearRenderDistance};
    int totalChunks{kDefaultTotalRenderDistanceChunks};
    int fogStartBlocks{kDefaultFarFogStartBlocks};
};

struct ChunkProfilingSnapshot
{
    StreamingPhase phase{StreamingPhase::SteadyState};
    double averageGenerationMs{0.0};
    double averageRelightMs{0.0};
    double averageMeshingMs{0.0};
    double updateMsLastFrame{0.0};
    double updateResidualMsLastFrame{0.0};
    double denseResidencyMsLastFrame{0.0};
    double verticalRadiusMsLastFrame{0.0};
    double priorityUpdateMsLastFrame{0.0};
    double uploadBudgetMsLastFrame{0.0};
    double missingScanMsLastFrame{0.0};
    double ensureVolumeMsLastFrame{0.0};
    double ensureVolumeColumnPrepMsLastFrame{0.0};
    double ensureVolumeSortMsLastFrame{0.0};
    double ensureVolumeDispatchMsLastFrame{0.0};
    double ensureVolumeChunkLookupMsLastFrame{0.0};
    double ensureVolumeEnqueueMsLastFrame{0.0};
    double schedulingMsLastFrame{0.0};
    double evictionMsLastFrame{0.0};
    double relightMsLastFrame{0.0};
    double uploadMsLastFrame{0.0};
    double uploadQueueAgeMsLastFrame{0.0};
    double uploadQueuePickMsLastFrame{0.0};
    double poolTrimMsLastFrame{0.0};
    double farTerrainUpdateMsLastFrame{0.0};
    double columnHeightLookupMsLastFrame{0.0};
    double columnHeightSampleMsLastFrame{0.0};
    double uploadPrepareMsLastFrame{0.0};
    double uploadContextBeginMsLastFrame{0.0};
    double uploadFinalizeMsLastFrame{0.0};
    double commitCollectMsLastFrame{0.0};
    double commitChunkScanMsLastFrame{0.0};
    double commitMeshLockWaitMsLastFrame{0.0};
    double commitMeshLockedMsLastFrame{0.0};
    double commitMeshStateMsLastFrame{0.0};
    double commitPageStateMsLastFrame{0.0};
    double commitReleaseMsLastFrame{0.0};
    double startupStateMsLastFrame{0.0};
    double benchmarkBookkeepingMsLastFrame{0.0};
    double farBuildMsAverage{0.0};
    double farCollectMsLastFrame{0.0};
    double farUploadMsLastFrame{0.0};
    double lodGpuSynthesisMs{0.0};
    double lodGpuStampMs{0.0};
    double lodGpuFaceBuildMs{0.0};
    double lodGpuCullMs{0.0};
    double lodIndirectBuildMs{0.0};
    double exactGpuSynthMs{0.0};
    double exactGpuStampMs{0.0};
    double exactGpuLightMs{0.0};
    double exactGpuFaceCountMs{0.0};
    double exactGpuFacePrefixMs{0.0};
    double exactGpuAllocateMs{0.0};
    double exactGpuFaceEmitMs{0.0};
    double exactGpuTotalMs{0.0};
    double structureQueryMs{0.0};
    double structureCacheHitRate{0.0};
    std::uint64_t structureRegionsBuilt{0};
    std::size_t uploadedBytes{0};
    std::uint64_t relightRegionChunks{0};
    std::uint64_t relightChangedChunks{0};
    std::uint64_t relightExternalSnapshotChunks{0};
    std::uint64_t relightSkyAboveChunkScans{0};
    std::uint64_t relightSkySeedNodes{0};
    std::uint64_t relightBlockSeedNodes{0};
    std::uint64_t relightSkyNodesProcessed{0};
    std::uint64_t relightBlockNodesProcessed{0};
    int generatedChunks{0};
    int relitChunks{0};
    int relightBatches{0};
    int meshedChunks{0};
    int uploadedChunks{0};
    int throttledUploads{0};
    int deferredUploads{0};
    int uploadAttemptsLastFrame{0};
    int uploadQueueScanEntriesLastFrame{0};
    int uploadSkippedExpiredLastFrame{0};
    int uploadSkippedNotReadyLastFrame{0};
    int uploadSkippedPendingMeshLastFrame{0};
    int uploadColumnLimitedLastFrame{0};
    int uploadBudgetDeferredLastFrame{0};
    int uploadRetryFailuresLastFrame{0};
    int uploadScanLimitHitsLastFrame{0};
    int uploadBeginFailuresLastFrame{0};
    int uploadStalePendingMeshesLastFrame{0};
    int evictedChunks{0};
    int verticalRadius{0};
    int verticalRadiusDelta{0};
    int generationBudget{0};
    int generationJobsIssued{0};
    int ringExpansionBudget{0};
    int ringExpansionsUsed{0};
    int missingChunks{0};
    int generationColumnCap{0};
    int generationBacklogSteps{0};
    int workerThreads{0};
    std::size_t uploadBudgetBytes{0};
    std::size_t uploadedBytesLastFrame{0};
    int uploadColumnLimit{0};
    int pendingUploadChunks{0};
    int jobQueueDepth{0};
    int uploadQueueDepth{0};
    int columnPrefetchQueueDepth{0};
    int ensureVolumeColumnsVisitedLastFrame{0};
    int ensureVolumeCandidatesBuiltLastFrame{0};
    int ensureVolumeExistingChunkSkipsLastFrame{0};
    int ensureVolumeColumnCapSkipsLastFrame{0};
    int exactChunksReady{0};
    int exactChunksPending{0};
    int exactCpuAuthoritativeChunks{0};
    int exactGpuResidentNonlocalChunks{0};
    int exactCpuMaterializingChunks{0};
    int exactGpuPendingRetireChunks{0};
    int exactGpuPageCount{0};
    std::size_t exactGpuPageBytes{0};
    std::size_t exactGpuColumnBytes{0};
    std::size_t exactGpuSparseVoxelBytes{0};
    std::size_t exactGpuVoxelBytes{0};
    std::size_t exactGpuLightScratchBytes{0};
    std::size_t exactGpuScratchBytes{0};
    std::size_t exactGpuUploadScratchBytes{0};
    std::size_t exactGpuReadbackBytes{0};
    std::size_t exactGpuTotalBytes{0};
    std::size_t gpuLocalUsageBytes{0};
    std::size_t gpuLocalBudgetBytes{0};
    std::size_t gpuLocalAvailableForReservationBytes{0};
    std::size_t gpuNonLocalUsageBytes{0};
    std::size_t gpuNonLocalBudgetBytes{0};
    int exactGpuBuildOverflows{0};
    int exactGpuBuildReadbackFailures{0};
    int exactGpuBuildResourceFailures{0};
    int exactGpuBuildStaleCancels{0};
    int exactGpuQueuedBuilds{0};
    int exactGpuPendingBuilds{0};
    std::uint64_t exactGpuBuildsSubmitted{0};
    std::uint64_t exactGpuBuildsCommitted{0};
    std::uint64_t exactGpuMeshReplacements{0};
    std::size_t pooledChunkCount{0};
    std::size_t pooledChunkBytes{0};
    std::size_t pooledChunkBudgetBytes{0};
    int farActiveTiles{0};
    int farDirtyTiles{0};
    int farShellTilesReady{0};
    int farTilesBuilt{0};
    int farTilesQueued{0};
    int farTilesPendingUpload{0};
};

struct BenchmarkStageStats
{
    std::uint64_t count{0};
    double totalMs{0.0};
    double averageMs{0.0};
    double medianMs{0.0};
    double p95Ms{0.0};
    double p99Ms{0.0};
    double maxMs{0.0};
};

struct BenchmarkQueueDepthStats
{
    std::uint64_t sampleCount{0};
    double averageDepth{0.0};
    double medianDepth{0.0};
    double p95Depth{0.0};
    double maxDepth{0.0};
};

struct BenchmarkCacheStats
{
    std::uint64_t hits{0};
    std::uint64_t misses{0};
    std::uint64_t fills{0};
    double hitRate{0.0};
};

struct ChunkBenchmarkReport
{
    BenchmarkStageStats sampleStage{};
    BenchmarkStageStats generateStage{};
    BenchmarkStageStats relightStage{};
    BenchmarkStageStats meshStage{};
    BenchmarkStageStats uploadStage{};
    BenchmarkStageStats updateStage{};
    BenchmarkStageStats updateResidualStage{};
    BenchmarkStageStats denseResidencyStage{};
    BenchmarkStageStats verticalRadiusStage{};
    BenchmarkStageStats priorityUpdateStage{};
    BenchmarkStageStats uploadBudgetPrepStage{};
    BenchmarkStageStats visibleScanStage{};
    BenchmarkStageStats ensureVolumeStage{};
    BenchmarkStageStats ensureVolumeColumnPrepStage{};
    BenchmarkStageStats ensureVolumeSortStage{};
    BenchmarkStageStats ensureVolumeDispatchStage{};
    BenchmarkStageStats ensureVolumeChunkLookupStage{};
    BenchmarkStageStats ensureVolumeEnqueueStage{};
    BenchmarkStageStats schedulingStage{};
    BenchmarkStageStats evictionStage{};
    BenchmarkStageStats mainThreadRelightStage{};
    BenchmarkStageStats uploadDrainStage{};
    BenchmarkStageStats uploadQueuePickStage{};
    BenchmarkStageStats poolTrimStage{};
    BenchmarkStageStats farTerrainUpdateStage{};
    BenchmarkStageStats columnHeightLookupStage{};
    BenchmarkStageStats columnHeightSampleStage{};
    BenchmarkStageStats uploadPrepareStage{};
    BenchmarkStageStats uploadContextBeginStage{};
    BenchmarkStageStats uploadFinalizeStage{};
    BenchmarkStageStats commitCollectStage{};
    BenchmarkStageStats commitChunkScanStage{};
    BenchmarkStageStats commitMeshLockWaitStage{};
    BenchmarkStageStats commitMeshLockedStage{};
    BenchmarkStageStats commitMeshStateStage{};
    BenchmarkStageStats commitPageStateStage{};
    BenchmarkStageStats commitReleaseStage{};
    BenchmarkStageStats generateBlocksMeshLockStage{};
    BenchmarkStageStats generateBaseTerrainStateStage{};
    BenchmarkStageStats generateStructureResolveStage{};
    BenchmarkStageStats generateGpuInputPrepStage{};
    BenchmarkStageStats generateCpuMaterializeStage{};
    BenchmarkStageStats generatePublishStage{};
    BenchmarkStageStats generateWorldgenPageMutexWaitStage{};
    BenchmarkStageStats generateWorldgenPagePendingWaitStage{};
    BenchmarkStageStats generateWorldgenPageBuildStage{};
    BenchmarkStageStats generateWorldgenPageBuildSurfaceStage{};
    BenchmarkStageStats generateWorldgenPageBuildPopulateStage{};
    BenchmarkStageStats generateWorldgenPageAccessCalls{};
    BenchmarkStageStats generateWorldgenPageUniqueKeys{};
    BenchmarkStageStats generateWorldgenPageColdBuildCount{};
    BenchmarkStageStats generateWorldgenPagePendingWaitCount{};
    BenchmarkStageStats buildChunkCpuWarmPagesStage{};
    BenchmarkStageStats buildChunkCpuDescribeColumnsStage{};
    BenchmarkStageStats buildChunkCpuMaterializeColumnsStage{};
    BenchmarkStageStats buildChunkCpuApplyStructureEditsStage{};
    BenchmarkStageStats buildChunkCpuApplyPendingEditsStage{};
    BenchmarkStageStats buildChunkCpuApplyOverlayStage{};
    BenchmarkStageStats uploadChunkMeshLockStage{};
    BenchmarkStageStats neighborhoodSnapshotLockStage{};
    BenchmarkStageStats skyLightCacheLockStage{};
    BenchmarkStageStats startupStateStage{};
    BenchmarkStageStats benchmarkBookkeepingStage{};
    BenchmarkStageStats farBuildStage{};
    BenchmarkStageStats lodGpuSynthesisStage{};
    BenchmarkStageStats lodGpuStampStage{};
    BenchmarkStageStats lodGpuFaceBuildStage{};
    BenchmarkStageStats lodGpuCullStage{};
    BenchmarkStageStats lodIndirectBuildStage{};
    BenchmarkStageStats exactGpuSynthStage{};
    BenchmarkStageStats exactGpuStampStage{};
    BenchmarkStageStats exactGpuLightStage{};
    BenchmarkStageStats exactGpuFaceCountStage{};
    BenchmarkStageStats exactGpuFacePrefixStage{};
    BenchmarkStageStats exactGpuAllocateStage{};
    BenchmarkStageStats exactGpuFaceEmitStage{};
    BenchmarkStageStats exactGpuTotalStage{};
    BenchmarkStageStats exactGpuBuildQueueWaitStage{};
    BenchmarkStageStats exactGpuBuildQueueDepthEnqueue{};
    BenchmarkStageStats exactGpuBuildQueueDepthStart{};
    BenchmarkStageStats chunkReadyLatency{};
    BenchmarkStageStats chunkReadyWaitGenerateStage{};
    BenchmarkStageStats chunkReadyRequestQueuedGenerateStage{};
    BenchmarkStageStats chunkReadyRequestQueuedMeshStage{};
    BenchmarkStageStats chunkReadyRequestQueuedPrefetchStage{};
    BenchmarkStageStats chunkReadyRequestQueuedBulkStage{};
    BenchmarkStageStats chunkReadyRequestLatencySensitiveOutstandingStage{};
    BenchmarkStageStats chunkReadyStartQueuedGenerateStage{};
    BenchmarkStageStats chunkReadyStartQueuedMeshStage{};
    BenchmarkStageStats chunkReadyStartQueuedPrefetchStage{};
    BenchmarkStageStats chunkReadyStartQueuedBulkStage{};
    BenchmarkStageStats chunkReadyStartActiveGenerateStage{};
    BenchmarkStageStats chunkReadyStartActiveMeshStage{};
    BenchmarkStageStats chunkReadyStartActivePrefetchStage{};
    BenchmarkStageStats chunkReadyStartActiveBulkStage{};
    BenchmarkStageStats chunkReadyStartLatencySensitiveOutstandingStage{};
    BenchmarkStageStats chunkReadyGenerateStage{};
    BenchmarkStageStats chunkReadyWaitMeshEnqueueStage{};
    BenchmarkStageStats chunkReadyWaitMeshStartStage{};
    BenchmarkStageStats chunkReadyMeshStage{};
    BenchmarkStageStats chunkReadyWaitUploadStage{};
    BenchmarkStageStats chunkReadyUploadToReadyStage{};
    BenchmarkStageStats chunkReadyGenerateAttempts{};
    BenchmarkStageStats chunkDependencyWorldgenDeferrals{};
    BenchmarkStageStats chunkDependencyStructureDeferrals{};
    BenchmarkStageStats worldgenDependencyReadyToGenerate{};
    BenchmarkStageStats structureDependencyReadyToGenerate{};
    BenchmarkStageStats chunkReadyStructureDeferralCount{};
    BenchmarkStageStats chunkReadyFirstStructureDeferralToReadyStage{};
    BenchmarkStageStats uploadQueueAgeStage{};
    BenchmarkStageStats structureQueryStage{};
    BenchmarkStageStats generateDeferredStructureMissingRegions{};
    BenchmarkStageStats ensureVolumeColumnsVisited{};
    BenchmarkStageStats ensureVolumeCandidatesBuilt{};
    BenchmarkStageStats ensureVolumeExistingChunkSkips{};
    BenchmarkStageStats ensureVolumeColumnCapSkips{};
    BenchmarkStageStats verticalRadiusDelta{};
    BenchmarkStageStats uploadQueueScanEntries{};
    BenchmarkStageStats uploadAttemptsPerFrame{};
    BenchmarkStageStats uploadChunksPerFrame{};
    BenchmarkStageStats uploadBytesPerFrame{};
    BenchmarkStageStats uploadExpiredEntriesPerFrame{};
    BenchmarkStageStats uploadSkippedNotReadyPerFrame{};
    BenchmarkStageStats uploadSkippedPendingMeshPerFrame{};
    BenchmarkStageStats uploadColumnLimitedPerFrame{};
    BenchmarkStageStats uploadBudgetDeferredPerFrame{};
    BenchmarkStageStats uploadRetryFailuresPerFrame{};
    BenchmarkStageStats uploadScanLimitHitsPerFrame{};
    BenchmarkStageStats uploadBeginFailuresPerFrame{};
    BenchmarkStageStats uploadStalePendingMeshesPerFrame{};
    BenchmarkStageStats relightRegionChunks{};
    BenchmarkStageStats relightChangedChunks{};
    BenchmarkStageStats relightExternalSnapshotChunks{};
    BenchmarkStageStats relightSkyAboveChunkScans{};
    BenchmarkStageStats relightSkySeedNodes{};
    BenchmarkStageStats relightBlockSeedNodes{};
    BenchmarkStageStats relightSkyNodesProcessed{};
    BenchmarkStageStats relightBlockNodesProcessed{};
    BenchmarkQueueDepthStats jobQueueDepth{};
    BenchmarkQueueDepthStats uploadQueueDepth{};
    BenchmarkQueueDepthStats columnPrefetchQueueDepth{};
    BenchmarkQueueDepthStats worldgenDependencyQueueDepth{};
    BenchmarkQueueDepthStats structureDependencyQueueDepth{};
    BenchmarkQueueDepthStats farBuildQueueDepth{};
    BenchmarkQueueDepthStats farUploadQueueDepth{};
    BenchmarkCacheStats climateCache{};
    BenchmarkCacheStats surfaceCache{};
    BenchmarkCacheStats structureCache{};
    std::uint64_t generatedChunks{0};
    std::uint64_t meshedChunks{0};
    std::uint64_t uploadedChunks{0};
    std::uint64_t farBuiltTiles{0};
    std::uint64_t structureRegionsBuilt{0};
    std::uint64_t uploadedBytes{0};
};

struct StreamingStatusSnapshot
{
    StreamingPhase phase{StreamingPhase::SteadyState};
    int exactReadyChunks{0};
    int exactRequiredChunks{0};
    bool exactRequiredChunksApproximate{false};
    int exactRequiredChunksAuthoritative{0};
    int exactConfiguredRadiusChunks{0};
    int exactSchedulingRadiusChunks{0};
    int exactTrackedRadiusChunks{0};
    int exactPlanRadius{0};
    int exactPlanVisibleRadius{0};
    int exactPlanPreloadRadius{0};
    int exactProtectedReadyChunks{0};
    int exactProtectedRequiredChunks{0};
    int exactMissingStateChunks{0};
    int exactWaitingDependenciesChunks{0};
    int exactQueuedGenerateChunks{0};
    int exactGeneratingChunks{0};
    int exactMeshingChunks{0};
    bool exactCoverageReconciling{false};
    int exactPendingUploads{0};
    int farActiveTiles{0};
    int farDirtyTiles{0};
    int farReadyTiles{0};
    int farQueuedTiles{0};
    int farPendingUploadTiles{0};
    bool playerReleaseReady{true};
    const char* blockingReason{"ready"};
};

struct LodDiagnosticsTileSnapshot
{
    int level{0};
    glm::ivec2 tileCoord{0};
    float distanceSq{0.0f};
    bool active{false};
    bool dirty{false};
    bool inFlight{false};
    std::uint32_t indexCount{0};
    int blockScaleBlocks{0};
    int chunkSpanBlocks{0};
    glm::ivec3 worldMin{0};
    glm::ivec3 worldMax{0};
};

struct LodDiagnosticsSnapshot
{
    int activeTiles{0};
    int readyTiles{0};
    int dirtyTiles{0};
    int inFlightTiles{0};
    double averageBuildMs{0.0};
    double averageGpuSynthesisMs{0.0};
    double averageGpuStampMs{0.0};
    double averageGpuFaceBuildMs{0.0};
    std::vector<LodDiagnosticsTileSnapshot> tiles;
};

struct RecentEditHoleChunkInfo
{
    glm::ivec3 coord{0};
    bool present{false};
    std::string stateLabel{"Missing"};
    bool hasBlocks{false};
    bool meshReady{false};
    bool queuedForUpload{false};
    int inFlight{0};
    std::uint32_t indexCount{0};
    std::uint32_t bufferPageIndex{(std::numeric_limits<std::uint32_t>::max)()};
    int columnMinChunkY{0};
    int columnMaxChunkY{0};
    int columnHeight{(std::numeric_limits<int>::min)()};
    std::string heightSource{"none"};
    bool wouldEvict{false};
};

struct RecentEditHoleDebugSnapshot
{
    bool hasRecentEdit{false};
    std::string editKind{};
    glm::ivec3 editWorldPos{0};
    glm::ivec3 editChunkCoord{0};
    double ageSeconds{0.0};
    int cameraChunkY{0};
    int verticalRadius{0};
    std::vector<std::string> recentEvents;
    std::vector<RecentEditHoleChunkInfo> chunks;
};

struct Frustum
{
    std::array<glm::vec4, 6> planes{};

    static Frustum fromMatrix(const glm::mat4& matrix);
    [[nodiscard]] bool intersectsAABB(const glm::vec3& minCorner, const glm::vec3& maxCorner) const noexcept;
};

class ChunkManager
{
public:
    explicit ChunkManager(unsigned seed);
    ~ChunkManager();

    ChunkManager(const ChunkManager&) = delete;
    ChunkManager& operator=(const ChunkManager&) = delete;
    ChunkManager(ChunkManager&&) = delete;
    ChunkManager& operator=(ChunkManager&&) = delete;

    void initializeRendering(ID3D12Device* device);
    void setRenderSynchronization(ID3D12Fence* graphicsFence, std::uint64_t graphicsFenceValue);
    [[nodiscard]] ID3D12Fence* uploadFence() const noexcept;
    [[nodiscard]] std::uint64_t lastSubmittedUploadFenceValue() const noexcept;
    [[nodiscard]] ID3D12Fence* farUploadFence() const noexcept;
    [[nodiscard]] std::uint64_t lastSubmittedFarUploadFenceValue() const noexcept;
    [[nodiscard]] ID3D12Fence* exactGpuFence() const noexcept;
    [[nodiscard]] std::uint64_t lastSubmittedExactGpuFenceValue() const noexcept;
    void setBlockTextureAtlasConfig(const BlockTextureAtlasConfig& config);
    void update(const glm::vec3& cameraPos);
    void update(const glm::vec3& cameraPos, const glm::vec3& cameraForward);
    WorldRenderData buildRenderData(const Frustum& frustum) const;

    float surfaceHeight(float worldX, float worldZ) const noexcept;
    terrain::ColumnSample sampleColumnAt(const glm::vec3& worldPos,
                                         int slabMinWorldY = (std::numeric_limits<int>::min)(),
                                         int slabMaxWorldY = (std::numeric_limits<int>::max)()) const;
    void clear();

    bool destroyBlock(const glm::ivec3& worldPos);
    bool placeBlock(const glm::ivec3& targetBlockPos,
                    const glm::ivec3& faceNormal,
                    BlockId block = BlockId::Grass);

    RaycastHit raycast(const glm::vec3& origin, const glm::vec3& direction) const;
    void updateHighlight(const glm::vec3& cameraPos, const glm::vec3& cameraDirection);

    void toggleViewDistance();
    int viewDistance() const noexcept;
    int exactRenderDistanceChunks() const noexcept;
    int totalRenderDistanceChunks() const noexcept;
    int nearRenderDistance() const noexcept;
    int farRenderDistanceBlocks() const noexcept;
    RenderDistanceSettings renderDistanceSettings() const noexcept;
    void setRenderDistance(int distance) noexcept;
    void setExactRenderDistanceChunks(int chunks) noexcept;
    void setTotalRenderDistanceChunks(int chunks) noexcept;
    void setNearRenderDistance(int chunks) noexcept;
    void setFarRenderDistanceBlocks(int blocks) noexcept;
    void setFogStartBlocks(int blocks) noexcept;
    void setLodEnabled(bool enabled);
    bool lodEnabled() const noexcept;
    void setFarTerrainEnabled(bool enabled);
    bool farTerrainEnabled() const noexcept;

    BlockId blockAt(const glm::ivec3& worldPos) const noexcept;
    LightSample lightAt(const glm::ivec3& worldPos) const noexcept;
    glm::vec3 findSafeSpawnPosition(float worldX, float worldZ) const;
    void beginSpawnPreload(const glm::vec3& spawnPos);
    bool isSpawnPreloadReady() const noexcept;
    bool playerReleaseReady() const noexcept;
    StreamingPhase streamingPhase() const noexcept;
    void setStartupExactPreloadChunks(int chunks) noexcept;
    void setStartupEnabled(bool enabled) noexcept;
    bool startupEnabled() const noexcept;
    StreamingStatusSnapshot streamingStatusSnapshot() const noexcept;
    LodDiagnosticsSnapshot lodDiagnosticsSnapshot(const glm::vec3& cameraPos) const;
    RecentEditHoleDebugSnapshot recentEditHoleDebugSnapshot(const glm::vec3& cameraPos) const;
    std::string exactLightingDebugSnapshot(const glm::ivec3& worldPos) const;
    void writeLodDebugSnapshot(const std::filesystem::path& outputPath, const glm::vec3& cameraPos) const;

    ChunkProfilingSnapshot sampleProfilingSnapshot();
    void setBenchmarkMetricsEnabled(bool enabled) noexcept;
    bool benchmarkMetricsEnabled() const noexcept;
    void resetBenchmarkMetrics();
    ChunkBenchmarkReport benchmarkReport() const;
    std::string biomeNameAt(const glm::vec3& worldPos) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

