// chunk_manager.cpp
// Implements the ChunkManager orchestration layer that coordinates streaming, world updates, and render uploads.

#include "chunk_manager.h"
#include "chunk_manager_support.h"

#include "terrain/biome_database.h"
#include "terrain/climate_map.h"
#include "terrain/far_lod_worldgen.h"
#include "terrain/surface_map.h"
#include "terrain/terrain_generator.h"
#include "terrain/worldgen_profile.h"
#include "shader_manifest.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/noise.hpp>
#include <d3dcompiler.h>
#include <wrl/client.h>

namespace
{
std::atomic<int> gActiveVerticalRadius{kVerticalStreamingConfig.minRadiusChunks};
constexpr glm::ivec3 kRelightNeighborPadding(1, 1, 1);
constexpr glm::ivec3 kRelightReservationPadding(1, 1, 1);
constexpr std::size_t kRelightMaxPendingDirtyCoordsPerRegion = 12;
constexpr std::uint64_t kRelightBaseBudgetUnits = 2'500'000ull;
constexpr std::uint64_t kRelightPerWorkerBudgetUnits = 500'000ull;
constexpr std::uint64_t kRelightBacklogBudgetUnitsPerRegion = 49'152ull;
constexpr std::uint64_t kRelightMaxBudgetUnits = 8'000'000ull;
constexpr std::uint64_t kRelightMinBudgetUnits = 1'500'000ull;
constexpr int kRelightMinBatchBudget = 3;
constexpr int kRelightMaxBatchBudget = 16;
using SteadyClock = std::chrono::steady_clock;

[[nodiscard]] std::uint64_t steadyMicrosNow() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            SteadyClock::now().time_since_epoch()).count());
}

[[nodiscard]] std::uint64_t percentileRankCount(std::uint64_t count, double percentile) noexcept
{
    if (count == 0)
    {
        return 0;
    }

    const double scaled = std::ceil(static_cast<double>(count) * percentile);
    return static_cast<std::uint64_t>(std::clamp(scaled, 1.0, static_cast<double>(count)));
}

[[nodiscard]] std::string formatDxHr(HRESULT hr)
{
    std::ostringstream stream;
    stream << "0x" << std::hex << std::uppercase << static_cast<std::uint32_t>(hr);
    return stream.str();
}

[[nodiscard]] const char* dredBreadcrumbOpName(D3D12_AUTO_BREADCRUMB_OP op) noexcept
{
    switch (op)
    {
    case D3D12_AUTO_BREADCRUMB_OP_SETMARKER: return "SetMarker";
    case D3D12_AUTO_BREADCRUMB_OP_BEGINEVENT: return "BeginEvent";
    case D3D12_AUTO_BREADCRUMB_OP_ENDEVENT: return "EndEvent";
    case D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED: return "DrawInstanced";
    case D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED: return "DrawIndexedInstanced";
    case D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT: return "ExecuteIndirect";
    case D3D12_AUTO_BREADCRUMB_OP_DISPATCH: return "Dispatch";
    case D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION: return "CopyBufferRegion";
    case D3D12_AUTO_BREADCRUMB_OP_COPYTEXTUREREGION: return "CopyTextureRegion";
    case D3D12_AUTO_BREADCRUMB_OP_COPYRESOURCE: return "CopyResource";
    case D3D12_AUTO_BREADCRUMB_OP_COPYTILES: return "CopyTiles";
    case D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCE: return "ResolveSubresource";
    case D3D12_AUTO_BREADCRUMB_OP_CLEARRENDERTARGETVIEW: return "ClearRenderTargetView";
    case D3D12_AUTO_BREADCRUMB_OP_CLEARUNORDEREDACCESSVIEW: return "ClearUnorderedAccessView";
    case D3D12_AUTO_BREADCRUMB_OP_CLEARDEPTHSTENCILVIEW: return "ClearDepthStencilView";
    case D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER: return "ResourceBarrier";
    case D3D12_AUTO_BREADCRUMB_OP_EXECUTEBUNDLE: return "ExecuteBundle";
    case D3D12_AUTO_BREADCRUMB_OP_PRESENT: return "Present";
    case D3D12_AUTO_BREADCRUMB_OP_RESOLVEQUERYDATA: return "ResolveQueryData";
    case D3D12_AUTO_BREADCRUMB_OP_BEGINSUBMISSION: return "BeginSubmission";
    case D3D12_AUTO_BREADCRUMB_OP_ENDSUBMISSION: return "EndSubmission";
    case D3D12_AUTO_BREADCRUMB_OP_DISPATCHRAYS: return "DispatchRays";
    case D3D12_AUTO_BREADCRUMB_OP_DISPATCHMESH: return "DispatchMesh";
    default: return "Unknown";
    }
}

[[nodiscard]] std::string collectDeviceDredMessages(ID3D12Device* device)
{
    if (device == nullptr)
    {
        return {};
    }

    const HRESULT removedReason = device->GetDeviceRemovedReason();
    if (!FAILED(removedReason))
    {
        return {};
    }

    Microsoft::WRL::ComPtr<ID3D12DeviceRemovedExtendedData1> dred;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dred))))
    {
        return {};
    }

    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 breadcrumbs{};
    D3D12_DRED_PAGE_FAULT_OUTPUT1 pageFault{};
    dred->GetAutoBreadcrumbsOutput1(&breadcrumbs);
    dred->GetPageFaultAllocationOutput1(&pageFault);

    std::ostringstream stream;
    stream << " removedReason=" << formatDxHr(removedReason);
    if (breadcrumbs.pHeadAutoBreadcrumbNode != nullptr)
    {
        stream << " autoBreadcrumbs:";
        UINT nodeCount = 0;
        for (const D3D12_AUTO_BREADCRUMB_NODE1* node = breadcrumbs.pHeadAutoBreadcrumbNode;
             node != nullptr && nodeCount < 8;
             node = node->pNext, ++nodeCount)
        {
            const UINT lastCompleted =
                node->pLastBreadcrumbValue != nullptr ? static_cast<UINT>(*node->pLastBreadcrumbValue) : 0u;
            stream << " [node " << nodeCount
                   << " cl=" << (node->pCommandListDebugNameA != nullptr ? node->pCommandListDebugNameA : "<unnamed>")
                   << " queue=" << (node->pCommandQueueDebugNameA != nullptr ? node->pCommandQueueDebugNameA : "<unnamed>")
                   << " completed=" << lastCompleted << "/" << node->BreadcrumbCount;
            if (node->pCommandHistory != nullptr && node->BreadcrumbCount > 0)
            {
                const UINT historyIndex = lastCompleted < node->BreadcrumbCount ? lastCompleted : (node->BreadcrumbCount - 1u);
                stream << " lastOp=" << dredBreadcrumbOpName(node->pCommandHistory[historyIndex]);
            }
            stream << "]";
        }
    }

    if (pageFault.PageFaultVA != 0)
    {
        stream << " pageFaultVA=0x" << std::hex << std::uppercase << pageFault.PageFaultVA << std::dec;
        if (pageFault.pHeadExistingAllocationNode != nullptr)
        {
            const D3D12_DRED_ALLOCATION_NODE1* node = pageFault.pHeadExistingAllocationNode;
            stream << " existingAllocation="
                   << (node->ObjectNameA != nullptr ? node->ObjectNameA : "<unnamed>")
                   << " type=" << static_cast<int>(node->AllocationType);
        }
        if (pageFault.pHeadRecentFreedAllocationNode != nullptr)
        {
            const D3D12_DRED_ALLOCATION_NODE1* node = pageFault.pHeadRecentFreedAllocationNode;
            stream << " recentFreed="
                   << (node->ObjectNameA != nullptr ? node->ObjectNameA : "<unnamed>")
                   << " type=" << static_cast<int>(node->AllocationType);
        }
    }

    return stream.str();
}

inline void updateAtomicMax(std::atomic<std::uint64_t>& current, std::uint64_t value) noexcept
{
    std::uint64_t observed = current.load(std::memory_order_relaxed);
    while (observed < value &&
           !current.compare_exchange_weak(observed,
                                          value,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed))
    {
    }
}

// Benchmark helpers stay near the top of the file so profiling code is easy to find and keep separate from streaming logic.
class AtomicLatencyHistogram
{
public:
    static constexpr std::size_t kBucketCount = 64;

    void reset() noexcept
    {
        count_.store(0, std::memory_order_relaxed);
        totalMicros_.store(0, std::memory_order_relaxed);
        maxMicros_.store(0, std::memory_order_relaxed);
        for (auto& bucket : buckets_)
        {
            bucket.store(0, std::memory_order_relaxed);
        }
    }

    void recordMicros(std::uint64_t micros) noexcept
    {
        count_.fetch_add(1, std::memory_order_relaxed);
        totalMicros_.fetch_add(micros, std::memory_order_relaxed);
        updateAtomicMax(maxMicros_, micros);
        buckets_[bucketIndexForMicros(micros)].fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] BenchmarkStageStats snapshot() const noexcept
    {
        BenchmarkStageStats stats{};
        stats.count = count_.load(std::memory_order_relaxed);
        const std::uint64_t totalMicros = totalMicros_.load(std::memory_order_relaxed);
        const std::uint64_t maxMicros = maxMicros_.load(std::memory_order_relaxed);
        stats.totalMs = static_cast<double>(totalMicros) / 1000.0;
        if (stats.count > 0)
        {
            stats.averageMs = stats.totalMs / static_cast<double>(stats.count);
            stats.medianMs = microsToMs(percentileMicros(0.50, maxMicros));
            stats.p95Ms = microsToMs(percentileMicros(0.95, maxMicros));
            stats.p99Ms = microsToMs(percentileMicros(0.99, maxMicros));
            stats.maxMs = microsToMs(maxMicros);
        }
        return stats;
    }

private:
    [[nodiscard]] static std::size_t bucketIndexForMicros(std::uint64_t micros) noexcept
    {
        if (micros == 0)
        {
            return 0;
        }

        const unsigned width = std::bit_width(micros);
        return static_cast<std::size_t>(std::min<unsigned>(width, static_cast<unsigned>(kBucketCount - 1)));
    }

    [[nodiscard]] static std::uint64_t bucketUpperBoundMicros(std::size_t bucketIndex,
                                                              std::uint64_t maxMicros) noexcept
    {
        if (bucketIndex == 0)
        {
            return 0;
        }
        if (bucketIndex + 1 >= kBucketCount)
        {
            return maxMicros;
        }

        return (std::uint64_t{1} << bucketIndex) - 1u;
    }

    [[nodiscard]] std::uint64_t percentileMicros(double percentile, std::uint64_t maxMicros) const noexcept
    {
        const std::uint64_t count = count_.load(std::memory_order_relaxed);
        const std::uint64_t target = percentileRankCount(count, percentile);
        if (target == 0)
        {
            return 0;
        }

        std::uint64_t running = 0;
        for (std::size_t bucketIndex = 0; bucketIndex < buckets_.size(); ++bucketIndex)
        {
            running += buckets_[bucketIndex].load(std::memory_order_relaxed);
            if (running >= target)
            {
                return bucketUpperBoundMicros(bucketIndex, maxMicros);
            }
        }

        return maxMicros;
    }

    [[nodiscard]] static double microsToMs(std::uint64_t micros) noexcept
    {
        return static_cast<double>(micros) / 1000.0;
    }

    std::atomic<std::uint64_t> count_{0};
    std::atomic<std::uint64_t> totalMicros_{0};
    std::atomic<std::uint64_t> maxMicros_{0};
    std::array<std::atomic<std::uint64_t>, kBucketCount> buckets_{};
};

class AtomicCountHistogram
{
public:
    static constexpr std::size_t kBucketCount = 64;

    void reset() noexcept
    {
        count_.store(0, std::memory_order_relaxed);
        total_.store(0, std::memory_order_relaxed);
        max_.store(0, std::memory_order_relaxed);
        for (auto& bucket : buckets_)
        {
            bucket.store(0, std::memory_order_relaxed);
        }
    }

    void record(std::uint64_t value) noexcept
    {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_.fetch_add(value, std::memory_order_relaxed);
        updateAtomicMax(max_, value);
        buckets_[bucketIndexForValue(value)].fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] BenchmarkStageStats snapshot() const noexcept
    {
        BenchmarkStageStats stats{};
        stats.count = count_.load(std::memory_order_relaxed);
        const std::uint64_t total = total_.load(std::memory_order_relaxed);
        const std::uint64_t maxValue = max_.load(std::memory_order_relaxed);
        stats.totalMs = static_cast<double>(total);
        if (stats.count > 0)
        {
            stats.averageMs = stats.totalMs / static_cast<double>(stats.count);
            stats.medianMs = static_cast<double>(percentileValue(0.50, maxValue));
            stats.p95Ms = static_cast<double>(percentileValue(0.95, maxValue));
            stats.p99Ms = static_cast<double>(percentileValue(0.99, maxValue));
            stats.maxMs = static_cast<double>(maxValue);
        }
        return stats;
    }

private:
    [[nodiscard]] static std::size_t bucketIndexForValue(std::uint64_t value) noexcept
    {
        if (value == 0)
        {
            return 0;
        }

        const unsigned width = std::bit_width(value);
        return static_cast<std::size_t>(std::min<unsigned>(width, static_cast<unsigned>(kBucketCount - 1)));
    }

    [[nodiscard]] static std::uint64_t bucketUpperBoundValue(std::size_t bucketIndex,
                                                             std::uint64_t maxValue) noexcept
    {
        if (bucketIndex == 0)
        {
            return 0;
        }
        if (bucketIndex + 1 >= kBucketCount)
        {
            return maxValue;
        }

        return (std::uint64_t{1} << bucketIndex) - 1u;
    }

    [[nodiscard]] std::uint64_t percentileValue(double percentile, std::uint64_t maxValue) const noexcept
    {
        const std::uint64_t count = count_.load(std::memory_order_relaxed);
        const std::uint64_t target = percentileRankCount(count, percentile);
        if (target == 0)
        {
            return 0;
        }

        std::uint64_t running = 0;
        for (std::size_t bucketIndex = 0; bucketIndex < buckets_.size(); ++bucketIndex)
        {
            running += buckets_[bucketIndex].load(std::memory_order_relaxed);
            if (running >= target)
            {
                return bucketUpperBoundValue(bucketIndex, maxValue);
            }
        }

        return maxValue;
    }

    std::atomic<std::uint64_t> count_{0};
    std::atomic<std::uint64_t> total_{0};
    std::atomic<std::uint64_t> max_{0};
    std::array<std::atomic<std::uint64_t>, kBucketCount> buckets_{};
};

template <std::size_t MaxBucketInclusive>
class AtomicDepthHistogram
{
public:
    static constexpr std::size_t kBucketCount = MaxBucketInclusive + 1;

    void reset() noexcept
    {
        count_.store(0, std::memory_order_relaxed);
        total_.store(0, std::memory_order_relaxed);
        max_.store(0, std::memory_order_relaxed);
        for (auto& bucket : buckets_)
        {
            bucket.store(0, std::memory_order_relaxed);
        }
    }

    void record(std::uint64_t value) noexcept
    {
        count_.fetch_add(1, std::memory_order_relaxed);
        total_.fetch_add(value, std::memory_order_relaxed);
        updateAtomicMax(max_, value);
        buckets_[bucketIndex(value)].fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] BenchmarkQueueDepthStats snapshot() const noexcept
    {
        BenchmarkQueueDepthStats stats{};
        stats.sampleCount = count_.load(std::memory_order_relaxed);
        const std::uint64_t total = total_.load(std::memory_order_relaxed);
        const std::uint64_t maxValue = max_.load(std::memory_order_relaxed);
        if (stats.sampleCount > 0)
        {
            stats.averageDepth = static_cast<double>(total) / static_cast<double>(stats.sampleCount);
            stats.medianDepth = static_cast<double>(percentileValue(0.50, maxValue));
            stats.p95Depth = static_cast<double>(percentileValue(0.95, maxValue));
            stats.maxDepth = static_cast<double>(maxValue);
        }
        return stats;
    }

private:
    [[nodiscard]] static std::size_t bucketIndex(std::uint64_t value) noexcept
    {
        return static_cast<std::size_t>(std::min<std::uint64_t>(value, MaxBucketInclusive));
    }

    [[nodiscard]] std::uint64_t percentileValue(double percentile, std::uint64_t maxValue) const noexcept
    {
        const std::uint64_t count = count_.load(std::memory_order_relaxed);
        const std::uint64_t target = percentileRankCount(count, percentile);
        if (target == 0)
        {
            return 0;
        }

        std::uint64_t running = 0;
        for (std::size_t bucketIndex = 0; bucketIndex < buckets_.size(); ++bucketIndex)
        {
            running += buckets_[bucketIndex].load(std::memory_order_relaxed);
            if (running >= target)
            {
                if (bucketIndex + 1 >= buckets_.size())
                {
                    return maxValue;
                }
                return static_cast<std::uint64_t>(bucketIndex);
            }
        }

        return maxValue;
    }

    std::atomic<std::uint64_t> count_{0};
    std::atomic<std::uint64_t> total_{0};
    std::atomic<std::uint64_t> max_{0};
    std::array<std::atomic<std::uint64_t>, kBucketCount> buckets_{};
};

struct ChunkBenchmarkMetrics
{
    void setEnabled(bool enabled) noexcept
    {
        enabled_.store(enabled, std::memory_order_release);
    }

    [[nodiscard]] bool isEnabled() const noexcept
    {
        return enabled_.load(std::memory_order_acquire);
    }

    void reset() noexcept
    {
        sampleStage.reset();
        generateStage.reset();
        relightStage.reset();
        meshStage.reset();
        uploadStage.reset();
        updateStage.reset();
        updateResidualStage.reset();
        denseResidencyStage.reset();
        verticalRadiusStage.reset();
        priorityUpdateStage.reset();
        uploadBudgetPrepStage.reset();
        visibleScanStage.reset();
        ensureVolumeStage.reset();
        schedulingStage.reset();
        evictionStage.reset();
        mainThreadRelightStage.reset();
        uploadDrainStage.reset();
        uploadQueuePickStage.reset();
        poolTrimStage.reset();
        farTerrainUpdateStage.reset();
        columnHeightLookupStage.reset();
        columnHeightSampleStage.reset();
        uploadPrepareStage.reset();
        uploadContextBeginStage.reset();
        uploadFinalizeStage.reset();
        commitCollectStage.reset();
        commitChunkScanStage.reset();
        commitMeshLockWaitStage.reset();
        commitMeshLockedStage.reset();
        commitMeshStateStage.reset();
        commitPageStateStage.reset();
        commitReleaseStage.reset();
        generateBlocksMeshLockStage.reset();
        uploadChunkMeshLockStage.reset();
        neighborhoodSnapshotLockStage.reset();
        skyLightCacheLockStage.reset();
        startupStateStage.reset();
        benchmarkBookkeepingStage.reset();
        farBuildStage.reset();
        lodGpuSynthesisStage.reset();
        lodGpuStampStage.reset();
        lodGpuFaceBuildStage.reset();
        lodGpuCullStage.reset();
        lodIndirectBuildStage.reset();
        chunkReadyLatency.reset();
        chunkReadyWaitGenerateStage.reset();
        chunkReadyGenerateStage.reset();
        chunkReadyWaitMeshEnqueueStage.reset();
        chunkReadyWaitMeshStartStage.reset();
        chunkReadyMeshStage.reset();
        chunkReadyWaitUploadStage.reset();
        chunkReadyUploadToReadyStage.reset();
        uploadQueueAgeStage.reset();
        structureQueryStage.reset();
        verticalRadiusDelta.reset();
        uploadQueueScanEntries.reset();
        uploadAttemptsPerFrame.reset();
        uploadChunksPerFrame.reset();
        uploadBytesPerFrame.reset();
        uploadExpiredEntriesPerFrame.reset();
        uploadSkippedNotReadyPerFrame.reset();
        uploadSkippedPendingMeshPerFrame.reset();
        uploadColumnLimitedPerFrame.reset();
        uploadBudgetDeferredPerFrame.reset();
        uploadRetryFailuresPerFrame.reset();
        uploadScanLimitHitsPerFrame.reset();
        uploadBeginFailuresPerFrame.reset();
        uploadStalePendingMeshesPerFrame.reset();
        relightRegionChunks.reset();
        relightChangedChunks.reset();
        relightExternalSnapshotChunks.reset();
        relightSkyAboveChunkScans.reset();
        relightSkySeedNodes.reset();
        relightBlockSeedNodes.reset();
        relightSkyNodesProcessed.reset();
        relightBlockNodesProcessed.reset();
        jobQueueDepth.reset();
        uploadQueueDepth.reset();
        columnPrefetchQueueDepth.reset();
        farBuildQueueDepth.reset();
        farUploadQueueDepth.reset();
        generatedChunks.store(0, std::memory_order_relaxed);
        meshedChunks.store(0, std::memory_order_relaxed);
        uploadedChunks.store(0, std::memory_order_relaxed);
        farBuiltTiles.store(0, std::memory_order_relaxed);
        uploadedBytes.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] ChunkBenchmarkReport snapshot() const noexcept
    {
        ChunkBenchmarkReport report{};
        report.sampleStage = sampleStage.snapshot();
        report.generateStage = generateStage.snapshot();
        report.relightStage = relightStage.snapshot();
        report.meshStage = meshStage.snapshot();
        report.uploadStage = uploadStage.snapshot();
        report.updateStage = updateStage.snapshot();
        report.updateResidualStage = updateResidualStage.snapshot();
        report.denseResidencyStage = denseResidencyStage.snapshot();
        report.verticalRadiusStage = verticalRadiusStage.snapshot();
        report.priorityUpdateStage = priorityUpdateStage.snapshot();
        report.uploadBudgetPrepStage = uploadBudgetPrepStage.snapshot();
        report.visibleScanStage = visibleScanStage.snapshot();
        report.ensureVolumeStage = ensureVolumeStage.snapshot();
        report.schedulingStage = schedulingStage.snapshot();
        report.evictionStage = evictionStage.snapshot();
        report.mainThreadRelightStage = mainThreadRelightStage.snapshot();
        report.uploadDrainStage = uploadDrainStage.snapshot();
        report.uploadQueuePickStage = uploadQueuePickStage.snapshot();
        report.poolTrimStage = poolTrimStage.snapshot();
        report.farTerrainUpdateStage = farTerrainUpdateStage.snapshot();
        report.columnHeightLookupStage = columnHeightLookupStage.snapshot();
        report.columnHeightSampleStage = columnHeightSampleStage.snapshot();
        report.uploadPrepareStage = uploadPrepareStage.snapshot();
        report.uploadContextBeginStage = uploadContextBeginStage.snapshot();
        report.uploadFinalizeStage = uploadFinalizeStage.snapshot();
        report.commitCollectStage = commitCollectStage.snapshot();
        report.commitChunkScanStage = commitChunkScanStage.snapshot();
        report.commitMeshLockWaitStage = commitMeshLockWaitStage.snapshot();
        report.commitMeshLockedStage = commitMeshLockedStage.snapshot();
        report.commitMeshStateStage = commitMeshStateStage.snapshot();
        report.commitPageStateStage = commitPageStateStage.snapshot();
        report.commitReleaseStage = commitReleaseStage.snapshot();
        report.generateBlocksMeshLockStage = generateBlocksMeshLockStage.snapshot();
        report.uploadChunkMeshLockStage = uploadChunkMeshLockStage.snapshot();
        report.neighborhoodSnapshotLockStage = neighborhoodSnapshotLockStage.snapshot();
        report.skyLightCacheLockStage = skyLightCacheLockStage.snapshot();
        report.startupStateStage = startupStateStage.snapshot();
        report.benchmarkBookkeepingStage = benchmarkBookkeepingStage.snapshot();
        report.farBuildStage = farBuildStage.snapshot();
        report.lodGpuSynthesisStage = lodGpuSynthesisStage.snapshot();
        report.lodGpuStampStage = lodGpuStampStage.snapshot();
        report.lodGpuFaceBuildStage = lodGpuFaceBuildStage.snapshot();
        report.lodGpuCullStage = lodGpuCullStage.snapshot();
        report.lodIndirectBuildStage = lodIndirectBuildStage.snapshot();
        report.chunkReadyLatency = chunkReadyLatency.snapshot();
        report.chunkReadyWaitGenerateStage = chunkReadyWaitGenerateStage.snapshot();
        report.chunkReadyGenerateStage = chunkReadyGenerateStage.snapshot();
        report.chunkReadyWaitMeshEnqueueStage = chunkReadyWaitMeshEnqueueStage.snapshot();
        report.chunkReadyWaitMeshStartStage = chunkReadyWaitMeshStartStage.snapshot();
        report.chunkReadyMeshStage = chunkReadyMeshStage.snapshot();
        report.chunkReadyWaitUploadStage = chunkReadyWaitUploadStage.snapshot();
        report.chunkReadyUploadToReadyStage = chunkReadyUploadToReadyStage.snapshot();
        report.uploadQueueAgeStage = uploadQueueAgeStage.snapshot();
        report.structureQueryStage = structureQueryStage.snapshot();
        report.verticalRadiusDelta = verticalRadiusDelta.snapshot();
        report.uploadQueueScanEntries = uploadQueueScanEntries.snapshot();
        report.uploadAttemptsPerFrame = uploadAttemptsPerFrame.snapshot();
        report.uploadChunksPerFrame = uploadChunksPerFrame.snapshot();
        report.uploadBytesPerFrame = uploadBytesPerFrame.snapshot();
        report.uploadExpiredEntriesPerFrame = uploadExpiredEntriesPerFrame.snapshot();
        report.uploadSkippedNotReadyPerFrame = uploadSkippedNotReadyPerFrame.snapshot();
        report.uploadSkippedPendingMeshPerFrame = uploadSkippedPendingMeshPerFrame.snapshot();
        report.uploadColumnLimitedPerFrame = uploadColumnLimitedPerFrame.snapshot();
        report.uploadBudgetDeferredPerFrame = uploadBudgetDeferredPerFrame.snapshot();
        report.uploadRetryFailuresPerFrame = uploadRetryFailuresPerFrame.snapshot();
        report.uploadScanLimitHitsPerFrame = uploadScanLimitHitsPerFrame.snapshot();
        report.uploadBeginFailuresPerFrame = uploadBeginFailuresPerFrame.snapshot();
        report.uploadStalePendingMeshesPerFrame = uploadStalePendingMeshesPerFrame.snapshot();
        report.relightRegionChunks = relightRegionChunks.snapshot();
        report.relightChangedChunks = relightChangedChunks.snapshot();
        report.relightExternalSnapshotChunks = relightExternalSnapshotChunks.snapshot();
        report.relightSkyAboveChunkScans = relightSkyAboveChunkScans.snapshot();
        report.relightSkySeedNodes = relightSkySeedNodes.snapshot();
        report.relightBlockSeedNodes = relightBlockSeedNodes.snapshot();
        report.relightSkyNodesProcessed = relightSkyNodesProcessed.snapshot();
        report.relightBlockNodesProcessed = relightBlockNodesProcessed.snapshot();
        report.jobQueueDepth = jobQueueDepth.snapshot();
        report.uploadQueueDepth = uploadQueueDepth.snapshot();
        report.columnPrefetchQueueDepth = columnPrefetchQueueDepth.snapshot();
        report.farBuildQueueDepth = farBuildQueueDepth.snapshot();
        report.farUploadQueueDepth = farUploadQueueDepth.snapshot();
        report.generatedChunks = generatedChunks.load(std::memory_order_relaxed);
        report.meshedChunks = meshedChunks.load(std::memory_order_relaxed);
        report.uploadedChunks = uploadedChunks.load(std::memory_order_relaxed);
        report.farBuiltTiles = farBuiltTiles.load(std::memory_order_relaxed);
        report.uploadedBytes = uploadedBytes.load(std::memory_order_relaxed);
        return report;
    }

    AtomicLatencyHistogram sampleStage{};
    AtomicLatencyHistogram generateStage{};
    AtomicLatencyHistogram relightStage{};
    AtomicLatencyHistogram meshStage{};
    AtomicLatencyHistogram uploadStage{};
    AtomicLatencyHistogram updateStage{};
    AtomicLatencyHistogram updateResidualStage{};
    AtomicLatencyHistogram denseResidencyStage{};
    AtomicLatencyHistogram verticalRadiusStage{};
    AtomicLatencyHistogram priorityUpdateStage{};
    AtomicLatencyHistogram uploadBudgetPrepStage{};
    AtomicLatencyHistogram visibleScanStage{};
    AtomicLatencyHistogram ensureVolumeStage{};
    AtomicLatencyHistogram schedulingStage{};
    AtomicLatencyHistogram evictionStage{};
    AtomicLatencyHistogram mainThreadRelightStage{};
    AtomicLatencyHistogram uploadDrainStage{};
    AtomicLatencyHistogram uploadQueuePickStage{};
    AtomicLatencyHistogram poolTrimStage{};
    AtomicLatencyHistogram farTerrainUpdateStage{};
    AtomicLatencyHistogram columnHeightLookupStage{};
    AtomicLatencyHistogram columnHeightSampleStage{};
    AtomicLatencyHistogram uploadPrepareStage{};
    AtomicLatencyHistogram uploadContextBeginStage{};
    AtomicLatencyHistogram uploadFinalizeStage{};
    AtomicLatencyHistogram commitCollectStage{};
    AtomicLatencyHistogram commitChunkScanStage{};
    AtomicLatencyHistogram commitMeshLockWaitStage{};
    AtomicLatencyHistogram commitMeshLockedStage{};
    AtomicLatencyHistogram commitMeshStateStage{};
    AtomicLatencyHistogram commitPageStateStage{};
    AtomicLatencyHistogram commitReleaseStage{};
    AtomicLatencyHistogram generateBlocksMeshLockStage{};
    AtomicLatencyHistogram uploadChunkMeshLockStage{};
    AtomicLatencyHistogram neighborhoodSnapshotLockStage{};
    AtomicLatencyHistogram skyLightCacheLockStage{};
    AtomicLatencyHistogram startupStateStage{};
    AtomicLatencyHistogram benchmarkBookkeepingStage{};
    AtomicLatencyHistogram farBuildStage{};
    AtomicLatencyHistogram lodGpuSynthesisStage{};
    AtomicLatencyHistogram lodGpuStampStage{};
    AtomicLatencyHistogram lodGpuFaceBuildStage{};
    AtomicLatencyHistogram lodGpuCullStage{};
    AtomicLatencyHistogram lodIndirectBuildStage{};
    AtomicLatencyHistogram chunkReadyLatency{};
    AtomicLatencyHistogram chunkReadyWaitGenerateStage{};
    AtomicLatencyHistogram chunkReadyGenerateStage{};
    AtomicLatencyHistogram chunkReadyWaitMeshEnqueueStage{};
    AtomicLatencyHistogram chunkReadyWaitMeshStartStage{};
    AtomicLatencyHistogram chunkReadyMeshStage{};
    AtomicLatencyHistogram chunkReadyWaitUploadStage{};
    AtomicLatencyHistogram chunkReadyUploadToReadyStage{};
    AtomicLatencyHistogram uploadQueueAgeStage{};
    AtomicLatencyHistogram structureQueryStage{};
    AtomicCountHistogram verticalRadiusDelta{};
    AtomicCountHistogram uploadQueueScanEntries{};
    AtomicCountHistogram uploadAttemptsPerFrame{};
    AtomicCountHistogram uploadChunksPerFrame{};
    AtomicCountHistogram uploadBytesPerFrame{};
    AtomicCountHistogram uploadExpiredEntriesPerFrame{};
    AtomicCountHistogram uploadSkippedNotReadyPerFrame{};
    AtomicCountHistogram uploadSkippedPendingMeshPerFrame{};
    AtomicCountHistogram uploadColumnLimitedPerFrame{};
    AtomicCountHistogram uploadBudgetDeferredPerFrame{};
    AtomicCountHistogram uploadRetryFailuresPerFrame{};
    AtomicCountHistogram uploadScanLimitHitsPerFrame{};
    AtomicCountHistogram uploadBeginFailuresPerFrame{};
    AtomicCountHistogram uploadStalePendingMeshesPerFrame{};
    AtomicCountHistogram relightRegionChunks{};
    AtomicCountHistogram relightChangedChunks{};
    AtomicCountHistogram relightExternalSnapshotChunks{};
    AtomicCountHistogram relightSkyAboveChunkScans{};
    AtomicCountHistogram relightSkySeedNodes{};
    AtomicCountHistogram relightBlockSeedNodes{};
    AtomicCountHistogram relightSkyNodesProcessed{};
    AtomicCountHistogram relightBlockNodesProcessed{};
    AtomicDepthHistogram<4096> jobQueueDepth{};
    AtomicDepthHistogram<4096> uploadQueueDepth{};
    AtomicDepthHistogram<4096> columnPrefetchQueueDepth{};
    AtomicDepthHistogram<1024> farBuildQueueDepth{};
    AtomicDepthHistogram<1024> farUploadQueueDepth{};
    std::atomic<std::uint64_t> generatedChunks{0};
    std::atomic<std::uint64_t> meshedChunks{0};
    std::atomic<std::uint64_t> uploadedChunks{0};
    std::atomic<std::uint64_t> farBuiltTiles{0};
    std::atomic<std::uint64_t> uploadedBytes{0};

private:
    std::atomic<bool> enabled_{false};
};

constexpr std::size_t kMaxFarLodAtlasUpdateCellsPerSubmission = 1024u;
constexpr std::uint32_t kFarLodChunkSeedCountPerCacheEntry = 64u;
}

float computeFarPlaneForViewDistance(int viewDistance) noexcept
{
    const int verticalRadius = std::max(gActiveVerticalRadius.load(std::memory_order_relaxed),
                                        kVerticalStreamingConfig.minRadiusChunks);
    const double horizontalSpan = static_cast<double>(viewDistance + 1)
                                  * static_cast<double>(std::max(kChunkSizeX, kChunkSizeZ));
    const double verticalSpan = static_cast<double>(verticalRadius + 1) * static_cast<double>(kChunkSizeY);
    const double diagonal = std::hypot(horizontalSpan, verticalSpan);
    const double farPlane = std::max(diagonal + static_cast<double>(kFarPlanePadding),
                                     static_cast<double>(kDefaultFarPlane));
    return static_cast<float>(farPlane);
}

float computeFarPlaneForDistanceBlocks(int farDistanceBlocks) noexcept
{
    const int clampedBlocks = std::max(farDistanceBlocks, 1);
    const int verticalRadius = std::max(gActiveVerticalRadius.load(std::memory_order_relaxed),
                                        kVerticalStreamingConfig.minRadiusChunks);
    const double horizontalSpan = static_cast<double>(clampedBlocks);
    const double verticalSpan = static_cast<double>(verticalRadius + 1) * static_cast<double>(kChunkSizeY);
    const double diagonal = std::hypot(horizontalSpan, verticalSpan);
    const double farPlane = std::max(diagonal + static_cast<double>(kFarPlanePadding),
                                     static_cast<double>(kDefaultFarPlane));
    return static_cast<float>(farPlane);
}

float kFarPlane = computeFarPlaneForViewDistance(kDefaultNearRenderDistance);

Frustum Frustum::fromMatrix(const glm::mat4& matrix)
{
    Frustum frustum;
    const glm::vec4 row0(matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]);
    const glm::vec4 row1(matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]);
    const glm::vec4 row2(matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]);
    const glm::vec4 row3(matrix[0][3], matrix[1][3], matrix[2][3], matrix[3][3]);

    frustum.planes[0] = row3 + row0; // Left
    frustum.planes[1] = row3 - row0; // Right
    frustum.planes[2] = row3 + row1; // Bottom
    frustum.planes[3] = row3 - row1; // Top
    frustum.planes[4] = row3 + row2; // Near
    frustum.planes[5] = row3 - row2; // Far

    for (auto& plane : frustum.planes)
    {
        const float length = std::sqrt(plane.x * plane.x + plane.y * plane.y + plane.z * plane.z);
        if (length > 0.0f)
        {
            plane /= length;
        }
    }

    return frustum;
}

bool Frustum::intersectsAABB(const glm::vec3& minCorner, const glm::vec3& maxCorner) const noexcept
{
    for (const auto& plane : planes)
    {
        glm::vec3 positiveVertex = minCorner;
        if (plane.x >= 0.0f) positiveVertex.x = maxCorner.x;
        if (plane.y >= 0.0f) positiveVertex.y = maxCorner.y;
        if (plane.z >= 0.0f) positiveVertex.z = maxCorner.z;

        if (glm::dot(glm::vec3(plane), positiveVertex) + plane.w < 0.0f)
        {
            return false;
        }
    }
    return true;
}

namespace
{
constexpr int chunksToBlocks(int chunks) noexcept
{
    return std::max(chunks, 1) * kChunkSizeX;
}

constexpr int blocksToChunkRadiusCeil(int blocks) noexcept
{
    return std::max(1, (std::max(blocks, 1) + kChunkSizeX - 1) / kChunkSizeX);
}

using Vertex = WorldVertex;
inline constexpr std::size_t kChunkPoolMinBudgetBytes = 16ull * 1024ull * 1024ull;
inline constexpr std::size_t kChunkPoolBaseBudgetBytes = 96ull * 1024ull * 1024ull;
inline constexpr double kChunkPoolMaxUploadPressure = 1.5;
inline constexpr double kChunkPoolUploadPressureDivisor = 32.0;
inline constexpr int kDenseCpuHorizontalRadiusMin = 4;
inline constexpr int kDenseCpuHorizontalRadiusMax = 8;
inline constexpr int kDenseCpuVerticalRadiusMin = 4;
inline constexpr int kDenseCpuVerticalRadiusMax = 8;
inline constexpr int kDenseCpuHydrationBudgetMin = 2;
inline constexpr int kDenseCpuHydrationBudgetMax = 4;
inline constexpr int kDenseCpuDemotionBudgetMin = 16;
inline constexpr int kDenseCpuDemotionBudgetMax = 64;
inline constexpr std::uint64_t kDenseCpuDemotionGraceFrames = 120u;
inline constexpr int kExactPlayerBandRadiusMax = 6;
inline constexpr int kExactPlayerBandHorizontalRadius = 8;
inline constexpr int kExactSurfaceShellBelowSlackChunks = 1;
inline constexpr int kExactSurfaceShellAirAboveChunks = 1;
inline constexpr int kMovementEnvelopeCoreRadiusMin = 4;
inline constexpr int kMovementEnvelopeCoreRadiusMax = 8;
inline constexpr int kMovementEnvelopeTurnReserveRadiusMin = 8;
inline constexpr int kMovementEnvelopeTurnReserveRadiusMax = 18;
inline constexpr int kMovementEnvelopeTurnReserveHalfWidthMin = 5;
inline constexpr int kMovementEnvelopeTurnReserveHalfWidthMax = 14;
inline constexpr int kMovementEnvelopeCorridorHalfWidthMin = 2;
inline constexpr int kMovementEnvelopeCorridorHalfWidthMax = 8;
inline constexpr int kMovementEnvelopeCorridorWidthStep = 6;
inline constexpr int kMovementEnvelopeRearSlackChunks = 2;

// Debug logging and D3D helper utilities stay near the top because multiple chunk subsystems depend on them.
[[nodiscard]] bool chunkManagerDebugLoggingEnabled() noexcept;
void chunkManagerDebugLog(const std::string& message);
[[nodiscard]] bool exactUploadDebugLoggingEnabled() noexcept;
void exactUploadDebugLog(const std::string& message);
[[nodiscard]] bool lodVisibilityDebugLoggingEnabled() noexcept;
void lodVisibilityDebugLog(const std::string& message);
[[nodiscard]] std::string hexU32(std::uint32_t value);
[[nodiscard]] std::string hexHr(HRESULT hr);
[[nodiscard]] const char* resourceStateName(D3D12_RESOURCE_STATES state) noexcept;
void setDebugObjectName(ID3D12Object* object, const std::wstring& name);

void throwIfFailedDx(HRESULT hr, const char* message)
{
    if (FAILED(hr))
    {
        if (chunkManagerDebugLoggingEnabled())
        {
            chunkManagerDebugLog(std::string(message) + " (hr=" + hexHr(hr) + ")");
        }
        throw std::runtime_error(message);
    }
}

Microsoft::WRL::ComPtr<ID3D12Resource> createUploadBuffer(ID3D12Device* device,
                                                          std::uint64_t sizeInBytes,
                                                          std::byte*& mappedData)
{
    mappedData = nullptr;
    if (device == nullptr || sizeInBytes == 0)
    {
        return {};
    }

    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = sizeInBytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    throwIfFailedDx(device->CreateCommittedResource(&heapProps,
                                                    D3D12_HEAP_FLAG_NONE,
                                                    &desc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ,
                                                    nullptr,
                                                    IID_PPV_ARGS(&resource)),
                    "failed to create upload buffer");

    void* mapped = nullptr;
    throwIfFailedDx(resource->Map(0, nullptr, &mapped), "failed to map upload buffer");
    mappedData = static_cast<std::byte*>(mapped);
    return resource;
}

Microsoft::WRL::ComPtr<ID3D12Resource> createDefaultBuffer(ID3D12Device* device,
                                                           std::uint64_t sizeInBytes,
                                                           D3D12_RESOURCE_STATES initialState,
                                                           D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
    if (device == nullptr || sizeInBytes == 0)
    {
        return {};
    }

    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = sizeInBytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    const HRESULT hr = device->CreateCommittedResource(&heapProps,
                                                       D3D12_HEAP_FLAG_NONE,
                                                       &desc,
                                                       initialState,
                                                       nullptr,
                                                       IID_PPV_ARGS(&resource));
    if (FAILED(hr) && chunkManagerDebugLoggingEnabled())
    {
        std::ostringstream stream;
        stream << "createDefaultBuffer failed size=" << sizeInBytes
               << " initialState=" << resourceStateName(initialState)
               << " flags=" << hexU32(static_cast<std::uint32_t>(flags))
               << " hr=" << hexHr(hr);
        const std::string dredMessages = collectDeviceDredMessages(device);
        if (!dredMessages.empty())
        {
            stream << dredMessages;
        }
        chunkManagerDebugLog(stream.str());
    }
    throwIfFailedDx(hr, "failed to create default buffer");
    return resource;
}

[[nodiscard]] bool chunkManagerDebugLoggingEnabled() noexcept
{
    static const bool enabled = []()
    {
        const char* value = std::getenv("BLOCKGAME_RENDER_DEBUG_LOG");
        return value != nullptr && std::string_view(value) != "0" && std::string_view(value) != "false";
    }();
    return enabled;
}

void appendDebugLogLine(const char* envVarName,
                        const std::filesystem::path& defaultPath,
                        const std::string& message)
{
    const char* fileValue = std::getenv(envVarName);
    const std::filesystem::path logPath =
        (fileValue != nullptr && *fileValue != '\0') ? std::filesystem::path(fileValue)
                                                     : defaultPath;
    std::ofstream out(logPath, std::ios::app);
    if (out)
    {
        out << message << '\n';
    }
}

void chunkManagerDebugLog(const std::string& message)
{
    if (!chunkManagerDebugLoggingEnabled())
    {
        return;
    }
    std::cerr << message << std::endl;
    appendDebugLogLine("BLOCKGAME_RENDER_DEBUG_LOG_FILE", "gpudebug.log", message);
}

[[nodiscard]] int hiddenExactPreloadBufferChunks(const RenderDistanceSettings& renderSettings) noexcept
{
    return (renderSettings.totalChunks <= renderSettings.exactChunks) ? kHiddenExactPreloadBufferChunks : 0;
}

[[nodiscard]] bool exactUploadDebugLoggingEnabled() noexcept
{
    static const bool enabled = []()
    {
        const char* value = std::getenv("BLOCKGAME_EXACT_UPLOAD_DEBUG");
        return value != nullptr && std::string_view(value) != "0" && std::string_view(value) != "false";
    }();
    return enabled;
}

void exactUploadDebugLog(const std::string& message)
{
    if (!exactUploadDebugLoggingEnabled())
    {
        return;
    }

    std::cerr << message << std::endl;
    appendDebugLogLine("BLOCKGAME_EXACT_UPLOAD_DEBUG_FILE", "exactuploaddebug.log", message);
}

[[nodiscard]] bool lodVisibilityDebugLoggingEnabled() noexcept
{
    static const bool enabled = []()
    {
        const char* value = std::getenv("BLOCKGAME_LOD_VIS_DEBUG");
        return value != nullptr && std::string_view(value) != "0" && std::string_view(value) != "false";
    }();
    return enabled;
}

void lodVisibilityDebugLog(const std::string& message)
{
    if (!lodVisibilityDebugLoggingEnabled())
    {
        return;
    }

    std::cerr << message << std::endl;
    appendDebugLogLine("BLOCKGAME_LOD_VIS_DEBUG_FILE", "loddebug.log", message);
}

[[nodiscard]] std::string hexU32(std::uint32_t value)
{
    std::ostringstream stream;
    stream << "0x" << std::hex << std::uppercase << value;
    return stream.str();
}

[[nodiscard]] std::string hexPtr(const void* value)
{
    std::ostringstream stream;
    stream << "0x" << std::hex << std::uppercase
           << static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(value));
    return stream.str();
}

[[nodiscard]] std::string hexHr(HRESULT hr)
{
    return hexU32(static_cast<std::uint32_t>(hr));
}

[[nodiscard]] const char* resourceStateName(D3D12_RESOURCE_STATES state) noexcept
{
    switch (state)
    {
    case D3D12_RESOURCE_STATE_COMMON:
        return "COMMON";
    case D3D12_RESOURCE_STATE_UNORDERED_ACCESS:
        return "UNORDERED_ACCESS";
    case D3D12_RESOURCE_STATE_COPY_SOURCE:
        return "COPY_SOURCE";
    case D3D12_RESOURCE_STATE_COPY_DEST:
        return "COPY_DEST";
    case D3D12_RESOURCE_STATE_GENERIC_READ:
        return "GENERIC_READ";
    default:
        return "OTHER";
    }
}

void setDebugObjectName(ID3D12Object* object, const std::wstring& name)
{
    if (object == nullptr || !chunkManagerDebugLoggingEnabled())
    {
        return;
    }
    object->SetName(name.c_str());
}

Microsoft::WRL::ComPtr<ID3D12Resource> createReadbackBuffer(ID3D12Device* device,
                                                            std::uint64_t sizeInBytes,
                                                            std::byte*& mappedData)
{
    mappedData = nullptr;
    if (device == nullptr || sizeInBytes == 0)
    {
        return {};
    }

    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = sizeInBytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    throwIfFailedDx(device->CreateCommittedResource(&heapProps,
                                                    D3D12_HEAP_FLAG_NONE,
                                                    &desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST,
                                                    nullptr,
                                                    IID_PPV_ARGS(&resource)),
                    "failed to create readback buffer");

    void* mapped = nullptr;
    throwIfFailedDx(resource->Map(0, nullptr, &mapped), "failed to map readback buffer");
    mappedData = static_cast<std::byte*>(mapped);
    return resource;
}

Microsoft::WRL::ComPtr<ID3DBlob> loadShaderBytecodeLocal(const std::string& path,
                                                         const char* entryPoint,
                                                         const char* target)
{
#if defined(BLOCKGAME_USE_PRECOMPILED_SHADERS)
    Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
    const std::filesystem::path compiledPath =
        compiledShaderPathForSource(std::filesystem::path(path), entryPoint, target);
    const std::wstring wideCompiledPath = compiledPath.wstring();
    const HRESULT hr = D3DReadFileToBlob(wideCompiledPath.c_str(), &bytecode);
    if (FAILED(hr))
    {
        throw std::runtime_error("failed to load precompiled shader blob: " + compiledPath.string());
    }
    return bytecode;
#else
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifndef NDEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
    Microsoft::WRL::ComPtr<ID3DBlob> errors;
    const std::wstring widePath = std::filesystem::path(path).wstring();
    const HRESULT hr = D3DCompileFromFile(widePath.c_str(),
                                          nullptr,
                                          D3D_COMPILE_STANDARD_FILE_INCLUDE,
                                          entryPoint,
                                          target,
                                          flags,
                                          0,
                                          &bytecode,
                                          &errors);
    if (FAILED(hr))
    {
        std::string message = "shader compilation failed for " + path;
        if (errors)
        {
            message += ": ";
            message.append(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
        }
        throw std::runtime_error(message);
    }

    return bytecode;
#endif
}

D3D12_RESOURCE_BARRIER transitionBarrier(ID3D12Resource* resource,
                                         D3D12_RESOURCE_STATES before,
                                         D3D12_RESOURCE_STATES after) noexcept
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return barrier;
}

#include "chunk_manager_gpu_contexts.inl"

struct BlockLightingProperties
{
    bool opaque{true};
    std::uint8_t skyAttenuation{kMaxLightLevel};
    std::uint8_t blockEmission{0};
    bool aoSolid{true};
};

constexpr std::array<BlockLightingProperties, toIndex(BlockId::Count)> kBlockLightingTable{{
    {false, 0, 0, false},              // Air
    {true, kMaxLightLevel, 0, true},   // Grass
    {true, kMaxLightLevel, 0, true},   // Wood
    {false, 1, 0, true},               // Leaves
    {true, kMaxLightLevel, 0, true},   // Sand
    {false, 2, 0, false},              // Water
    {true, kMaxLightLevel, 0, true},   // Stone
    {true, kMaxLightLevel, 0, true},   // SpruceLog
    {false, 1, 0, true},               // SpruceLeaves
    {true, kMaxLightLevel, 0, true},   // Podzol
    {true, kMaxLightLevel, 14, true},  // DebugLamp
    {true, kMaxLightLevel, 0, true},   // DarkOakLog
    {false, 1, 0, true},               // DarkOakLeaves
    {true, kMaxLightLevel, 0, true},   // BirchLog
    {false, 1, 0, true},               // BirchLeaves
    {true, kMaxLightLevel, 0, true},   // AcaciaLog
    {false, 1, 0, true},               // AcaciaLeaves
}};

inline const BlockLightingProperties& blockLightingProperties(BlockId block) noexcept
{
    return kBlockLightingTable[toIndex(block)];
}

inline bool isOpaqueForLighting(BlockId block) noexcept
{
    return blockLightingProperties(block).opaque;
}

inline std::uint8_t packLightLevels(std::uint8_t sky, std::uint8_t block) noexcept
{
    return static_cast<std::uint8_t>(((std::min<std::uint8_t>)(sky, kMaxLightLevel) << 4) |
                                     (std::min<std::uint8_t>)(block, kMaxLightLevel));
}

inline std::uint8_t skyLightFromPacked(std::uint8_t packed) noexcept
{
    return static_cast<std::uint8_t>((packed >> 4) & 0x0F);
}

inline std::uint8_t blockLightFromPacked(std::uint8_t packed) noexcept
{
    return static_cast<std::uint8_t>(packed & 0x0F);
}

inline void setSkyLight(std::uint8_t& packed, std::uint8_t sky) noexcept
{
    packed = packLightLevels(sky, blockLightFromPacked(packed));
}

inline void setBlockLight(std::uint8_t& packed, std::uint8_t block) noexcept
{
    packed = packLightLevels(skyLightFromPacked(packed), block);
}

inline std::uint8_t propagationLossFor(BlockId block) noexcept
{
    return static_cast<std::uint8_t>(1 + blockLightingProperties(block).skyAttenuation);
}

inline bool isAoSolid(BlockId block) noexcept
{
    return blockLightingProperties(block).aoSolid;
}

inline std::uint32_t packVertexLighting(std::uint8_t packedLight,
                                        std::uint8_t aoLevel = 0,
                                        std::uint8_t flags = 0) noexcept
{
    return static_cast<std::uint32_t>(packedLight) |
           (static_cast<std::uint32_t>(aoLevel & 0x03u) << 8) |
           (static_cast<std::uint32_t>(flags) << 10);
}

inline std::uint32_t applyVertexFlags(std::uint32_t packedLighting, std::uint8_t flags) noexcept
{
    return (packedLighting & ~(static_cast<std::uint32_t>(0x3Fu) << 10)) |
           (static_cast<std::uint32_t>(flags & 0x3Fu) << 10);
}

inline std::uint8_t aoLevelFromPackedVertexLighting(std::uint32_t packed) noexcept
{
    return static_cast<std::uint8_t>((packed >> 8) & 0x03u);
}

inline std::uint8_t vertexFlagsFromPackedLighting(std::uint32_t packed) noexcept
{
    return static_cast<std::uint8_t>((packed >> 10) & 0x3Fu);
}

constexpr std::uint8_t kMaterialFlagGrassTintShiftCpu = 2u;
constexpr std::uint8_t kMaterialFlagGrassTintMaskCpu = 0x1Cu;
constexpr std::uint8_t kMaterialFlagGrassSideTintCpu = 0x20u;

enum class GrassTintIndex : std::uint8_t
{
    None = 0,
    Default = 1,
    DarkForest = 2,
    Taiga = 3,
    Warm = 4,
};

inline std::uint8_t packGrassTintFlags(GrassTintIndex tintIndex, bool sideTintOnly) noexcept
{
    const std::uint8_t tintBits =
        (static_cast<std::uint8_t>(tintIndex) << kMaterialFlagGrassTintShiftCpu) & kMaterialFlagGrassTintMaskCpu;
    return static_cast<std::uint8_t>(tintBits | (sideTintOnly ? kMaterialFlagGrassSideTintCpu : 0u));
}

inline GrassTintIndex grassTintIndexForBiome(const terrain::BiomeDefinition* biome) noexcept
{
    if (!biome)
    {
        return GrassTintIndex::Default;
    }
    if (biome->id == "dark_forest")
    {
        return GrassTintIndex::DarkForest;
    }
    if (terrain::isTaigaBiome(*biome))
    {
        return GrassTintIndex::Taiga;
    }
    if (biome->id == "savanna" || biome->id == "desert")
    {
        return GrassTintIndex::Warm;
    }
    return GrassTintIndex::Default;
}

inline int lightingMetricFromPackedVertex(std::uint32_t packed) noexcept
{
    const std::uint8_t packedLight = static_cast<std::uint8_t>(packed & 0xFFu);
    const int sky = static_cast<int>(skyLightFromPacked(packedLight));
    const int block = static_cast<int>(blockLightFromPacked(packedLight));
    const int ao = static_cast<int>(aoLevelFromPackedVertexLighting(packed));
    return sky * 24 + block * 18 + (3 - ao) * 20;
}

inline bool hasUniformCornerLighting(const std::array<std::uint32_t, 4>& lightingData) noexcept
{
    return lightingData[0] == lightingData[1] &&
           lightingData[0] == lightingData[2] &&
           lightingData[0] == lightingData[3];
}

inline std::size_t cornerIndexForSigns(int uSign, int vSign) noexcept
{
    if (uSign > 0)
    {
        return vSign > 0 ? 2u : 1u;
    }
    return vSign > 0 ? 3u : 0u;
}

inline bool isAlphaCutoutBlock(BlockId block) noexcept
{
    return block == BlockId::Leaves ||
           block == BlockId::SpruceLeaves ||
           block == BlockId::DarkOakLeaves ||
           block == BlockId::BirchLeaves ||
           block == BlockId::AcaciaLeaves;
}

inline bool isNonOpaqueBlock(BlockId block) noexcept
{
    return block == BlockId::Air || block == BlockId::Water || isAlphaCutoutBlock(block);
}

inline bool shouldRenderBlockFace(BlockId owningBlock, BlockId neighborBlock) noexcept
{
    if (owningBlock == BlockId::Air)
    {
        return false;
    }

    if (neighborBlock == BlockId::Air)
    {
        return true;
    }

    if (isAlphaCutoutBlock(owningBlock))
    {
        if (isAlphaCutoutBlock(neighborBlock))
        {
            return owningBlock != neighborBlock;
        }

        return neighborBlock == BlockId::Water;
    }

    if (owningBlock == BlockId::Water)
    {
        return neighborBlock == BlockId::Air;
    }

    return isNonOpaqueBlock(neighborBlock);
}

constexpr int kTaigaSpruceCellSize = 14;
constexpr int kTaigaSpruceMinTrunkHeight = 25;
constexpr std::uint8_t kVertexFlagWater = 0x01u;
constexpr std::uint8_t kVertexFlagFarLod = 0x02u;
constexpr int kTaigaSpruceMaxTrunkHeight = 31;
constexpr int kTaigaSpruceMinBareTrunkHeight = 5;
constexpr int kTaigaSpruceMaxBareTrunkHeight = 9;
constexpr int kTaigaSpruceMaxLeafRadius = 4;
constexpr float kTreeBiomeWeightThreshold = 0.55f;
constexpr int kDefaultTreeMinHeight = 6;
constexpr int kDefaultTreeMaxHeight = 8;
constexpr int kDefaultTreeMaxRadius = 2;
constexpr int kDefaultTreeConflictSearchRadius = (kDefaultTreeMaxRadius * 2) + 1;
constexpr int kDarkOakCellSize = 6;
constexpr int kDarkOakMinTrunkHeight = 6;
constexpr int kDarkOakMaxTrunkHeight = 10;
constexpr int kDarkOakBranchMaxLength = 3;
constexpr int kDarkOakCanopyLayers = 5;
constexpr int kDarkOakCanopyBaseOffset = 2;
constexpr int kDarkOakCanopyTopOffset = kDarkOakCanopyLayers - kDarkOakCanopyBaseOffset - 1;
constexpr int kDarkOakMaxHorizontalReach = kDarkOakBranchMaxLength + 1;
constexpr std::array<int, kDarkOakCanopyLayers> kDarkOakCanopyRadii{{2, 2, 2, 1, 1}};
constexpr int kAcaciaCellSize = 11;
constexpr int kAcaciaMinTrunkHeight = 6;
constexpr int kAcaciaMaxTrunkHeight = 9;
constexpr int kAcaciaMaxLeanLength = 3;
constexpr int kAcaciaMainCanopyLayers = 4;
constexpr int kAcaciaSecondaryCanopyLayers = 3;
constexpr int kAcaciaMainCanopyTopOffset = 2;
constexpr int kAcaciaMaxHorizontalReach = 7;
constexpr std::array<int, kAcaciaMainCanopyLayers> kAcaciaMainCanopyRadii{{3, 3, 2, 1}};
constexpr std::array<int, kAcaciaSecondaryCanopyLayers> kAcaciaSecondaryCanopyRadii{{2, 2, 1}};

struct DefaultTreeCandidate
{
    int originX{0};
    int originZ{0};
    int groundWorldY{0};
    int trunkHeight{0};
    float priority{0.0f};
    BlockId trunkBlock{BlockId::Wood};
    BlockId leavesBlock{BlockId::Leaves};
};

struct DefaultTreeBlockPalette
{
    BlockId trunkBlock{BlockId::Wood};
    BlockId leavesBlock{BlockId::Leaves};
};

struct DarkOakTreeCandidate
{
    int originX{0};
    int originZ{0};
    int groundWorldY{0};
    int trunkHeight{0};
    float priority{0.0f};
};

struct AcaciaTreeCandidate
{
    int originX{0};
    int originZ{0};
    int groundWorldY{0};
    int trunkHeight{0};
    float priority{0.0f};
};

inline float hashToUnitFloat32(int x, int y, int z) noexcept
{
    constexpr std::uint32_t kMulX = 374761393u;
    constexpr std::uint32_t kMulY = 668265263u;
    constexpr std::uint32_t kMulZ = 2147483647u;
    constexpr std::uint32_t kMixMul = 1274126177u;
    constexpr std::uint32_t kMask24 = 0x00FFFFFFu;

    const auto widen = [](int value) noexcept -> std::uint32_t {
        return static_cast<std::uint32_t>(value);
    };

    std::uint32_t h = widen(x) * kMulX;
    h ^= widen(y) * kMulY;
    h ^= widen(z) * kMulZ;
    h = (h ^ (h >> 13)) * kMixMul;
    h ^= (h >> 16);
    return static_cast<float>(h & kMask24) / static_cast<float>(kMask24);
}

inline glm::ivec2 taigaSpruceOriginForCell(int cellX, int cellZ) noexcept
{
    const int offsetX = 3 + static_cast<int>(hashToUnitFloat(cellX, 911, cellZ) * 4.0f);
    const int offsetZ = 3 + static_cast<int>(hashToUnitFloat(cellX, 977, cellZ) * 4.0f);
    return glm::ivec2(cellX * kTaigaSpruceCellSize + offsetX,
                      cellZ * kTaigaSpruceCellSize + offsetZ);
}

inline bool isTaigaSpruceOrigin(int worldX, int worldZ) noexcept
{
    const int cellX = floorDiv(worldX, kTaigaSpruceCellSize);
    const int cellZ = floorDiv(worldZ, kTaigaSpruceCellSize);
    return taigaSpruceOriginForCell(cellX, cellZ) == glm::ivec2(worldX, worldZ);
}

inline float taigaSpruceOccupancyChance(const BiomeDefinition& biome) noexcept
{
    return std::clamp(0.40f + std::max(biome.treeDensityMultiplier, 0.0f) * 0.20f, 0.45f, 0.90f);
}

inline bool shouldSpawnTaigaSpruce(const BiomeDefinition& biome, int worldX, int groundWorldY, int worldZ) noexcept
{
    if (!terrain::isTaigaBiome(biome) || !isTaigaSpruceOrigin(worldX, worldZ))
    {
        return false;
    }

    const int cellX = floorDiv(worldX, kTaigaSpruceCellSize);
    const int cellZ = floorDiv(worldZ, kTaigaSpruceCellSize);
    const float occupancyRoll = hashToUnitFloat(cellX, groundWorldY + 151, cellZ);
    return occupancyRoll <= taigaSpruceOccupancyChance(biome);
}

inline int taigaSpruceTrunkHeight(int worldX, int groundWorldY, int worldZ) noexcept
{
    int height = kTaigaSpruceMinTrunkHeight +
                 static_cast<int>(hashToUnitFloat(worldX, groundWorldY + 37, worldZ) *
                                  static_cast<float>(kTaigaSpruceMaxTrunkHeight - kTaigaSpruceMinTrunkHeight + 1));
    return std::clamp(height, kTaigaSpruceMinTrunkHeight, kTaigaSpruceMaxTrunkHeight);
}

inline int taigaSpruceBareTrunkHeight(int worldX, int groundWorldY, int worldZ) noexcept
{
    int height = kTaigaSpruceMinBareTrunkHeight +
                 static_cast<int>(hashToUnitFloat(worldX, groundWorldY + 83, worldZ) *
                                  static_cast<float>(kTaigaSpruceMaxBareTrunkHeight - kTaigaSpruceMinBareTrunkHeight + 1));
    return std::clamp(height, kTaigaSpruceMinBareTrunkHeight, kTaigaSpruceMaxBareTrunkHeight);
}

inline int taigaSpruceLeafRadiusForLayer(int layerFromBottom, int totalLayers) noexcept
{
    if (totalLayers <= 1)
    {
        return 0;
    }

    const float t = static_cast<float>(layerFromBottom) / static_cast<float>(std::max(totalLayers - 1, 1));
    int radius = 1 + static_cast<int>(std::round((1.0f - t) * 3.0f));

    if (layerFromBottom % 3 == 0 && layerFromBottom < (totalLayers * 3) / 4)
    {
        radius = std::min(radius + 1, kTaigaSpruceMaxLeafRadius);
    }

    if (t > 0.88f)
    {
        radius = 1;
    }
    if (t > 0.97f)
    {
        radius = 0;
    }

    return std::clamp(radius, 0, kTaigaSpruceMaxLeafRadius);
}

inline int distanceToInclusiveRange(int value, int minValue, int maxValue) noexcept
{
    if (value < minValue)
    {
        return minValue - value;
    }
    if (value > maxValue)
    {
        return value - maxValue;
    }
    return 0;
}

inline bool taigaSpruceLeafOccupiesCell(int originX,
                                        int originZ,
                                        int worldX,
                                        int worldZ,
                                        int radius,
                                        int layerFromBottom,
                                        int totalLayers) noexcept
{
    if (radius <= 0)
    {
        return false;
    }

    if (worldX >= originX && worldX <= originX + 1 &&
        worldZ >= originZ && worldZ <= originZ + 1)
    {
        return false;
    }

    const int dx = distanceToInclusiveRange(worldX, originX, originX + 1);
    const int dz = distanceToInclusiveRange(worldZ, originZ, originZ + 1);
    const int chebyshev = std::max(dx, dz);
    if (chebyshev > radius)
    {
        return false;
    }

    int manhattanAllowance = radius + 1;
    if (radius >= 4 && layerFromBottom < totalLayers / 3)
    {
        ++manhattanAllowance;
    }

    return (dx + dz) <= manhattanAllowance;
}

inline glm::ivec2 darkOakOriginForCell(int cellX, int cellZ) noexcept
{
    const int offsetX = 1 + static_cast<int>(hashToUnitFloat32(cellX, 1301, cellZ) * 3.0f);
    const int offsetZ = 1 + static_cast<int>(hashToUnitFloat32(cellX, 1427, cellZ) * 3.0f);
    return glm::ivec2(cellX * kDarkOakCellSize + offsetX,
                      cellZ * kDarkOakCellSize + offsetZ);
}

inline bool isDarkOakOrigin(int worldX, int worldZ) noexcept
{
    const int cellX = floorDiv(worldX, kDarkOakCellSize);
    const int cellZ = floorDiv(worldZ, kDarkOakCellSize);
    return darkOakOriginForCell(cellX, cellZ) == glm::ivec2(worldX, worldZ);
}

inline float darkOakPriority(int originX, int groundWorldY, int originZ) noexcept
{
    return hashToUnitFloat32(originX, groundWorldY + 887, originZ);
}

inline int darkOakTrunkHeight(int originX, int groundWorldY, int originZ) noexcept
{
    const int height = kDarkOakMinTrunkHeight +
                       static_cast<int>(hashToUnitFloat32(originX, groundWorldY + 461, originZ) *
                                        static_cast<float>(kDarkOakMaxTrunkHeight - kDarkOakMinTrunkHeight + 1));
    return std::clamp(height, kDarkOakMinTrunkHeight, kDarkOakMaxTrunkHeight);
}

inline float darkOakSpawnChance(const BiomeDefinition& biome, float normalizedDensity) noexcept
{
    const float baseChance = 0.62f + normalizedDensity * 0.28f;
    const float densityScale = 0.88f + std::max(biome.treeDensityMultiplier, 0.0f) * 0.09f;
    return std::clamp(baseChance * densityScale, 0.68f, 0.98f);
}

inline int darkOakBranchLength(int originX, int groundWorldY, int originZ, int dir) noexcept
{
    const int length = 1 + static_cast<int>(hashToUnitFloat32(originX + dir * 37,
                                                              groundWorldY + 557,
                                                              originZ + dir * 53) * 3.0f);
    return std::clamp(length, 1, kDarkOakBranchMaxLength);
}

inline int darkOakBranchCount(int originX, int groundWorldY, int originZ) noexcept
{
    const int count = 1 + static_cast<int>(hashToUnitFloat32(originX, groundWorldY + 719, originZ) * 4.0f);
    return std::clamp(count, 1, 4);
}

inline float darkOakBranchScore(int originX, int groundWorldY, int originZ, int dir) noexcept
{
    return hashToUnitFloat32(originX + dir * 97, groundWorldY + 683, originZ + dir * 109);
}

inline bool darkOakBranchActive(int originX, int groundWorldY, int originZ, int dir) noexcept
{
    const float score = darkOakBranchScore(originX, groundWorldY, originZ, dir);
    int rank = 0;
    for (int otherDir = 0; otherDir < 4; ++otherDir)
    {
        if (otherDir == dir)
        {
            continue;
        }

        const float otherScore = darkOakBranchScore(originX, groundWorldY, originZ, otherDir);
        if (otherScore > score || (otherScore == score && otherDir < dir))
        {
            ++rank;
        }
    }

    return rank < darkOakBranchCount(originX, groundWorldY, originZ);
}

inline int darkOakBranchWorldY(int originX, int groundWorldY, int originZ, int trunkHeight, int dir) noexcept
{
    const int verticalOffset =
        static_cast<int>(hashToUnitFloat32(originX + dir * 67, groundWorldY + 601, originZ + dir * 79) * 3.0f);
    return std::max(groundWorldY + trunkHeight / 2,
                    groundWorldY + trunkHeight - 2 - std::clamp(verticalOffset, 0, 2));
}

inline int darkOakBranchLane(int originX, int groundWorldY, int originZ, int dir) noexcept
{
    return hashToUnitFloat32(originX + dir * 19, groundWorldY + 643, originZ + dir * 29) < 0.5f ? 0 : 1;
}

inline bool darkOakCanopyOccupiesCell(int originX,
                                      int originZ,
                                      int worldX,
                                      int worldZ,
                                      int layer) noexcept
{
    if (layer < 0 || layer >= kDarkOakCanopyLayers)
    {
        return false;
    }

    const int radius = kDarkOakCanopyRadii[static_cast<std::size_t>(layer)];
    const int dx = distanceToInclusiveRange(worldX, originX, originX + 1);
    const int dz = distanceToInclusiveRange(worldZ, originZ, originZ + 1);
    const int chebyshev = std::max(dx, dz);
    if (radius == 0)
    {
        return chebyshev == 0;
    }
    if (chebyshev > radius)
    {
        return false;
    }

    return true;
}

inline glm::ivec2 acaciaOriginForCell(int cellX, int cellZ) noexcept
{
    const int offsetX = 2 + static_cast<int>(hashToUnitFloat32(cellX, 1703, cellZ) * 7.0f);
    const int offsetZ = 2 + static_cast<int>(hashToUnitFloat32(cellX, 1811, cellZ) * 7.0f);
    return glm::ivec2(cellX * kAcaciaCellSize + offsetX,
                      cellZ * kAcaciaCellSize + offsetZ);
}

inline bool isAcaciaOrigin(int worldX, int worldZ) noexcept
{
    const int cellX = floorDiv(worldX, kAcaciaCellSize);
    const int cellZ = floorDiv(worldZ, kAcaciaCellSize);
    return acaciaOriginForCell(cellX, cellZ) == glm::ivec2(worldX, worldZ);
}

inline float acaciaPriority(int originX, int groundWorldY, int originZ) noexcept
{
    return hashToUnitFloat32(originX, groundWorldY + 919, originZ);
}

inline int acaciaTrunkHeight(int originX, int groundWorldY, int originZ) noexcept
{
    const int height = kAcaciaMinTrunkHeight +
                       static_cast<int>(hashToUnitFloat32(originX, groundWorldY + 947, originZ) *
                                        static_cast<float>(kAcaciaMaxTrunkHeight - kAcaciaMinTrunkHeight + 1));
    return std::clamp(height, kAcaciaMinTrunkHeight, kAcaciaMaxTrunkHeight);
}

inline float acaciaSpawnChance(const BiomeDefinition& biome, float normalizedDensity) noexcept
{
    const float baseChance = 0.28f + normalizedDensity * 0.18f;
    const float densityScale = 0.65f + std::max(biome.treeDensityMultiplier, 0.0f) * 0.12f;
    return std::clamp(baseChance * densityScale, 0.20f, 0.78f);
}

inline int acaciaLeanDir(int originX, int groundWorldY, int originZ) noexcept
{
    return static_cast<int>(hashToUnitFloat32(originX, groundWorldY + 971, originZ) * 4.0f) & 3;
}

inline int acaciaLeanLength(int originX, int groundWorldY, int originZ) noexcept
{
    const int length = 1 + static_cast<int>(hashToUnitFloat32(originX, groundWorldY + 997, originZ) * 3.0f);
    return std::clamp(length, 1, kAcaciaMaxLeanLength);
}

inline int acaciaBendStart(int originX, int groundWorldY, int originZ, int trunkHeight) noexcept
{
    const int minStart = std::max(2, trunkHeight / 3);
    const int maxStart = std::max(minStart, trunkHeight - 3);
    const int span = maxStart - minStart + 1;
    const int offset = static_cast<int>(hashToUnitFloat32(originX, groundWorldY + 1031, originZ) *
                                        static_cast<float>(span));
    return std::clamp(minStart + offset, minStart, maxStart);
}

inline bool acaciaHasSecondaryBranch(int originX, int groundWorldY, int originZ) noexcept
{
    return hashToUnitFloat32(originX, groundWorldY + 1063, originZ) < 0.55f;
}

inline int acaciaSecondaryDir(int originX, int groundWorldY, int originZ, int primaryDir) noexcept
{
    const int delta = 1 + (static_cast<int>(hashToUnitFloat32(originX, groundWorldY + 1097, originZ) * 3.0f) % 3);
    return (primaryDir + delta) & 3;
}

inline int acaciaSecondaryLength(int originX, int groundWorldY, int originZ) noexcept
{
    const int length = 1 + static_cast<int>(hashToUnitFloat32(originX, groundWorldY + 1129, originZ) * 2.0f);
    return std::clamp(length, 1, 2);
}

inline bool acaciaCanopyOccupiesCell(int centerX,
                                     int centerZ,
                                     int worldX,
                                     int worldZ,
                                     int radius,
                                     int layer,
                                     bool secondary) noexcept
{
    if (radius <= 0)
    {
        return worldX == centerX && worldZ == centerZ;
    }

    const int dx = std::abs(worldX - centerX);
    const int dz = std::abs(worldZ - centerZ);
    const int chebyshev = std::max(dx, dz);
    if (chebyshev > radius)
    {
        return false;
    }

    if (layer == 0)
    {
        return (dx + dz) <= (secondary ? radius + 1 : radius + 2);
    }

    if (chebyshev == radius && dx + dz > radius + 1)
    {
        return false;
    }

    return true;
}

inline int defaultTreeTrunkHeight(int worldX, int groundWorldY, int worldZ) noexcept
{
    int height = kDefaultTreeMinHeight +
                 static_cast<int>(hashToUnitFloat(worldX, groundWorldY + 1, worldZ) *
                                  static_cast<float>(kDefaultTreeMaxHeight - kDefaultTreeMinHeight + 1));
    return std::clamp(height, kDefaultTreeMinHeight, kDefaultTreeMaxHeight);
}

inline DefaultTreeBlockPalette defaultTreeBlockPaletteForBiome(const BiomeDefinition& biome,
                                                               int worldX,
                                                               int groundWorldY,
                                                               int worldZ) noexcept
{
    if (biome.id == "birch_forest")
    {
        return DefaultTreeBlockPalette{BlockId::BirchLog, BlockId::BirchLeaves};
    }

    if (biome.id == "forest")
    {
        const float birchRoll = hashToUnitFloat(worldX, groundWorldY + 313, worldZ);
        if (birchRoll < 0.30f)
        {
            return DefaultTreeBlockPalette{BlockId::BirchLog, BlockId::BirchLeaves};
        }
    }

    return DefaultTreeBlockPalette{};
}

inline float defaultTreeSpawnThreshold(const BiomeDefinition& biome, float normalizedDensity) noexcept
{
    const float spawnThresholdBase = 0.015f + normalizedDensity * 0.02f;
    return std::clamp(spawnThresholdBase * std::max(biome.treeDensityMultiplier, 0.0f), 0.0f, 1.0f);
}

inline float defaultTreePriority(int worldX, int groundWorldY, int worldZ) noexcept
{
    return hashToUnitFloat(worldX, groundWorldY + 211, worldZ);
}

inline bool shouldDefaultTreeWinTie(const DefaultTreeCandidate& candidate,
                                    const DefaultTreeCandidate& other) noexcept
{
    if (candidate.priority != other.priority)
    {
        return candidate.priority > other.priority;
    }

    if (candidate.originX != other.originX)
    {
        return candidate.originX < other.originX;
    }

    return candidate.originZ < other.originZ;
}

template <typename Callback>
inline bool forEachDefaultTreeBlock(int originX,
                                    int originZ,
                                    int groundWorldY,
                                    int trunkHeight,
                                    BlockId trunkBlock,
                                    BlockId leavesBlock,
                                    Callback&& callback)
{
    for (int dy = 0; dy < trunkHeight; ++dy)
    {
        if (callback(originX, groundWorldY + dy, originZ, trunkBlock))
        {
            return true;
        }
    }

    const int canopyBaseWorld = groundWorldY + trunkHeight - 3;
    const int canopyTopWorld = groundWorldY + trunkHeight;
    for (int worldY = canopyBaseWorld; worldY <= canopyTopWorld; ++worldY)
    {
        const int layer = worldY - canopyBaseWorld;
        int radius = kDefaultTreeMaxRadius;
        if (worldY >= canopyTopWorld - 1)
        {
            radius = 1;
        }

        for (int dx = -radius; dx <= radius; ++dx)
        {
            for (int dz = -radius; dz <= radius; ++dz)
            {
                if (std::abs(dx) == radius && std::abs(dz) == radius && radius > 1)
                {
                    continue;
                }

                if (dx == 0 && dz == 0 && worldY <= groundWorldY + trunkHeight - 1)
                {
                    continue;
                }

                if (layer == 0 && std::abs(dx) + std::abs(dz) > 3)
                {
                    continue;
                }

                if (callback(originX + dx, worldY, originZ + dz, leavesBlock))
                {
                    return true;
                }
            }
        }
    }

    return false;
}

inline bool rangesTouchWithinMargin(int minA, int maxA, int minB, int maxB, int margin) noexcept
{
    return maxA + margin >= minB && maxB + margin >= minA;
}

inline bool defaultTreesTouchOrOverlap(const DefaultTreeCandidate& a,
                                       const DefaultTreeCandidate& b) noexcept
{
    const int minAX = a.originX - kDefaultTreeMaxRadius;
    const int maxAX = a.originX + kDefaultTreeMaxRadius;
    const int minAZ = a.originZ - kDefaultTreeMaxRadius;
    const int maxAZ = a.originZ + kDefaultTreeMaxRadius;
    const int minAY = a.groundWorldY;
    const int maxAY = a.groundWorldY + a.trunkHeight;

    const int minBX = b.originX - kDefaultTreeMaxRadius;
    const int maxBX = b.originX + kDefaultTreeMaxRadius;
    const int minBZ = b.originZ - kDefaultTreeMaxRadius;
    const int maxBZ = b.originZ + kDefaultTreeMaxRadius;
    const int minBY = b.groundWorldY;
    const int maxBY = b.groundWorldY + b.trunkHeight;

    if (!rangesTouchWithinMargin(minAX, maxAX, minBX, maxBX, 1) ||
        !rangesTouchWithinMargin(minAY, maxAY, minBY, maxBY, 1) ||
        !rangesTouchWithinMargin(minAZ, maxAZ, minBZ, maxBZ, 1))
    {
        return false;
    }

    return forEachDefaultTreeBlock(a.originX,
                                   a.originZ,
                                   a.groundWorldY,
                                   a.trunkHeight,
                                   a.trunkBlock,
                                   a.leavesBlock,
                                   [&](int ax, int ay, int az, BlockId) {
                                       return forEachDefaultTreeBlock(b.originX,
                                                                      b.originZ,
                                                                      b.groundWorldY,
                                                                      b.trunkHeight,
                                                                      b.trunkBlock,
                                                                      b.leavesBlock,
                                                                      [&](int bx, int by, int bz, BlockId) {
                                                                          const int dx = std::abs(ax - bx);
                                                                          const int dy = std::abs(ay - by);
                                                                          const int dz = std::abs(az - bz);
                                                                          return (dx + dy + dz) <= 1;
                                                                      });
                                   });
}

template <typename SampleColumnFn, typename DensityFn>
inline bool tryBuildDefaultTreeCandidate(int originX,
                                         int originZ,
                                         const ColumnSample& columnSample,
                                         SampleColumnFn&& sampleColumn,
                                         DensityFn&& densityAt,
                                         DefaultTreeCandidate& outCandidate)
{
    if (!columnSample.dominantBiome || !columnSample.dominantBiome->generatesTrees)
    {
        return false;
    }

    if (columnSample.dominantWeight < kTreeBiomeWeightThreshold)
    {
        return false;
    }

    const BiomeDefinition& biome = *columnSample.dominantBiome;
    if (terrain::isTaigaBiome(biome) || biome.id == "dark_forest" || biome.id == "savanna")
    {
        return false;
    }

    const int groundWorldY = columnSample.surfaceY;
    if (groundWorldY <= 2)
    {
        return false;
    }

    const float density = densityAt(originX, originZ);
    const float normalizedDensity = std::clamp((density + 1.0f) * 0.5f, 0.0f, 1.0f);
    const float randomValue = hashToUnitFloat(originX, groundWorldY, originZ);
    if (randomValue > defaultTreeSpawnThreshold(biome, normalizedDensity))
    {
        return false;
    }

    for (int dx = -1; dx <= 1; ++dx)
    {
        for (int dz = -1; dz <= 1; ++dz)
        {
            if (dx == 0 && dz == 0)
            {
                continue;
            }

            const ColumnSample neighborSample = sampleColumn(originX + dx, originZ + dz);
            if (std::abs(neighborSample.surfaceY - groundWorldY) > 1)
            {
                return false;
            }
        }
    }

    outCandidate.originX = originX;
    outCandidate.originZ = originZ;
    outCandidate.groundWorldY = groundWorldY;
    outCandidate.trunkHeight = defaultTreeTrunkHeight(originX, groundWorldY, originZ);
    outCandidate.priority = defaultTreePriority(originX, groundWorldY, originZ);
    const DefaultTreeBlockPalette palette = defaultTreeBlockPaletteForBiome(biome, originX, groundWorldY, originZ);
    outCandidate.trunkBlock = palette.trunkBlock;
    outCandidate.leavesBlock = palette.leavesBlock;
    return true;
}

template <typename SampleColumnFn, typename DensityFn>
inline bool defaultTreeHasSpacingConflict(const DefaultTreeCandidate& candidate,
                                          SampleColumnFn&& sampleColumn,
                                          DensityFn&& densityAt)
{
    for (int dx = -kDefaultTreeConflictSearchRadius; dx <= kDefaultTreeConflictSearchRadius; ++dx)
    {
        for (int dz = -kDefaultTreeConflictSearchRadius; dz <= kDefaultTreeConflictSearchRadius; ++dz)
        {
            if (dx == 0 && dz == 0)
            {
                continue;
            }

            const int neighborX = candidate.originX + dx;
            const int neighborZ = candidate.originZ + dz;
            const ColumnSample neighborSample = sampleColumn(neighborX, neighborZ);

            DefaultTreeCandidate neighborCandidate{};
            if (!tryBuildDefaultTreeCandidate(neighborX,
                                              neighborZ,
                                              neighborSample,
                                              sampleColumn,
                                              densityAt,
                                              neighborCandidate))
            {
                continue;
            }

            if (!defaultTreesTouchOrOverlap(candidate, neighborCandidate))
            {
                continue;
            }

            if (!shouldDefaultTreeWinTie(candidate, neighborCandidate))
            {
                return true;
            }
        }
    }

    return false;
}

inline bool shouldDarkOakWinTie(const DarkOakTreeCandidate& candidate,
                                const DarkOakTreeCandidate& other) noexcept
{
    if (candidate.priority != other.priority)
    {
        return candidate.priority > other.priority;
    }

    if (candidate.originX != other.originX)
    {
        return candidate.originX < other.originX;
    }

    return candidate.originZ < other.originZ;
}

inline bool shouldAcaciaWinTie(const AcaciaTreeCandidate& candidate,
                               const AcaciaTreeCandidate& other) noexcept
{
    if (candidate.priority != other.priority)
    {
        return candidate.priority > other.priority;
    }

    if (candidate.originX != other.originX)
    {
        return candidate.originX < other.originX;
    }

    return candidate.originZ < other.originZ;
}

template <typename Callback>
inline bool forEachAcaciaTreeBlock(int originX,
                                   int originZ,
                                   int groundWorldY,
                                   int trunkHeight,
                                   BlockId trunkBlock,
                                   BlockId leavesBlock,
                                   Callback&& callback)
{
    static constexpr std::array<int, 4> kDirX{{1, -1, 0, 0}};
    static constexpr std::array<int, 4> kDirZ{{0, 0, 1, -1}};

    const int leanDir = acaciaLeanDir(originX, groundWorldY, originZ);
    const int leanLength = acaciaLeanLength(originX, groundWorldY, originZ);
    const int bendStart = acaciaBendStart(originX, groundWorldY, originZ, trunkHeight);

    int tipX = originX;
    int tipZ = originZ;
    int tipY = groundWorldY;

    for (int dy = 0; dy < trunkHeight; ++dy)
    {
        if (dy >= bendStart)
        {
            const int leanStep = std::min(dy - bendStart + 1, leanLength);
            tipX = originX + kDirX[static_cast<std::size_t>(leanDir)] * leanStep;
            tipZ = originZ + kDirZ[static_cast<std::size_t>(leanDir)] * leanStep;
        }
        else
        {
            tipX = originX;
            tipZ = originZ;
        }

        tipY = groundWorldY + dy;
        if (callback(tipX, tipY, tipZ, trunkBlock))
        {
            return true;
        }
    }

    const int mainCenterX = tipX;
    const int mainCenterZ = tipZ;
    const int mainBaseWorldY = tipY - kAcaciaMainCanopyTopOffset;
    for (int layer = 0; layer < kAcaciaMainCanopyLayers; ++layer)
    {
        const int radius = kAcaciaMainCanopyRadii[static_cast<std::size_t>(layer)];
        const int worldY = mainBaseWorldY + layer;
        for (int worldX = mainCenterX - radius; worldX <= mainCenterX + radius; ++worldX)
        {
            for (int worldZ = mainCenterZ - radius; worldZ <= mainCenterZ + radius; ++worldZ)
            {
                if (!acaciaCanopyOccupiesCell(mainCenterX, mainCenterZ, worldX, worldZ, radius, layer, false))
                {
                    continue;
                }

                if (worldX == tipX && worldZ == tipZ && worldY <= tipY)
                {
                    continue;
                }

                if (callback(worldX, worldY, worldZ, leavesBlock))
                {
                    return true;
                }
            }
        }
    }

    const int hangingLeavesWorldY = mainBaseWorldY - 1;
    for (int worldX = mainCenterX - 2; worldX <= mainCenterX + 2; ++worldX)
    {
        for (int worldZ = mainCenterZ - 2; worldZ <= mainCenterZ + 2; ++worldZ)
        {
            const int dx = std::abs(worldX - mainCenterX);
            const int dz = std::abs(worldZ - mainCenterZ);
            if (std::max(dx, dz) != 2 || dx + dz > 3)
            {
                continue;
            }

            if (callback(worldX, hangingLeavesWorldY, worldZ, leavesBlock))
            {
                return true;
            }
        }
    }

    if (!acaciaHasSecondaryBranch(originX, groundWorldY, originZ))
    {
        return false;
    }

    const int secondaryDir = acaciaSecondaryDir(originX, groundWorldY, originZ, leanDir);
    const int secondaryLength = acaciaSecondaryLength(originX, groundWorldY, originZ);
    const int branchStartY = std::max(groundWorldY + bendStart, tipY - 2);
    int branchX = originX;
    int branchZ = originZ;
    for (int step = 1; step <= secondaryLength; ++step)
    {
        branchX = tipX + kDirX[static_cast<std::size_t>(secondaryDir)] * step;
        branchZ = tipZ + kDirZ[static_cast<std::size_t>(secondaryDir)] * step;
        const int branchY = branchStartY + std::min(step - 1, 1);
        if (callback(branchX, branchY, branchZ, trunkBlock))
        {
            return true;
        }
    }

    const int secondaryCenterX = branchX;
    const int secondaryCenterZ = branchZ;
    const int secondaryBaseWorldY = branchStartY - 1;
    for (int layer = 0; layer < kAcaciaSecondaryCanopyLayers; ++layer)
    {
        const int radius = kAcaciaSecondaryCanopyRadii[static_cast<std::size_t>(layer)];
        const int worldY = secondaryBaseWorldY + layer;
        for (int worldX = secondaryCenterX - radius; worldX <= secondaryCenterX + radius; ++worldX)
        {
            for (int worldZ = secondaryCenterZ - radius; worldZ <= secondaryCenterZ + radius; ++worldZ)
            {
                if (!acaciaCanopyOccupiesCell(secondaryCenterX, secondaryCenterZ, worldX, worldZ, radius, layer, true))
                {
                    continue;
                }

                if ((worldX == branchX && worldZ == branchZ && worldY <= branchStartY + 1) ||
                    (worldX == tipX && worldZ == tipZ && worldY <= tipY))
                {
                    continue;
                }

                if (callback(worldX, worldY, worldZ, leavesBlock))
                {
                    return true;
                }
            }
        }
    }

    return false;
}

template <typename Callback>
inline bool forEachDarkOakTreeBlock(int originX,
                                    int originZ,
                                    int groundWorldY,
                                    int trunkHeight,
                                    BlockId trunkBlock,
                                    BlockId leavesBlock,
                                    Callback&& callback)
{
    const int trunkTopWorld = groundWorldY + trunkHeight - 1;
    const int canopyBaseWorld = groundWorldY + trunkHeight - kDarkOakCanopyBaseOffset;

    for (int trunkX = 0; trunkX < 2; ++trunkX)
    {
        for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
        {
            for (int dy = 0; dy < trunkHeight; ++dy)
            {
                if (callback(originX + trunkX, groundWorldY + dy, originZ + trunkZ, trunkBlock))
                {
                    return true;
                }
            }
        }
    }

    static constexpr std::array<int, 4> kDirX{{1, -1, 0, 0}};
    static constexpr std::array<int, 4> kDirZ{{0, 0, 1, -1}};
    static constexpr std::array<int, 4> kSideX{{0, 0, 1, 1}};
    static constexpr std::array<int, 4> kSideZ{{1, 1, 0, 0}};

    for (int dir = 0; dir < 4; ++dir)
    {
        if (!darkOakBranchActive(originX, groundWorldY, originZ, dir))
        {
            continue;
        }

        const int length = darkOakBranchLength(originX, groundWorldY, originZ, dir);
        const int branchWorldY = darkOakBranchWorldY(originX, groundWorldY, originZ, trunkHeight, dir);
        const int lane = darkOakBranchLane(originX, groundWorldY, originZ, dir);

        int tipX = originX;
        int tipZ = originZ;
        const int shoulderBaseWorldY = std::max(groundWorldY + trunkHeight / 2 - 1, branchWorldY - 2);
        for (int shoulderY = shoulderBaseWorldY; shoulderY <= trunkTopWorld; ++shoulderY)
        {
            for (int side = 0; side < 2; ++side)
            {
                int shoulderX = originX;
                int shoulderZ = originZ;
                if (dir == 0)
                {
                    shoulderX = originX + 2;
                    shoulderZ = originZ + side;
                }
                else if (dir == 1)
                {
                    shoulderX = originX - 1;
                    shoulderZ = originZ + side;
                }
                else if (dir == 2)
                {
                    shoulderX = originX + side;
                    shoulderZ = originZ + 2;
                }
                else
                {
                    shoulderX = originX + side;
                    shoulderZ = originZ - 1;
                }

                if (callback(shoulderX, shoulderY, shoulderZ, trunkBlock))
                {
                    return true;
                }
            }
        }

        for (int leafY = canopyBaseWorld; leafY <= canopyBaseWorld + 1; ++leafY)
        {
            for (int outward = 0; outward <= 2; ++outward)
            {
                for (int lateral = -1; lateral <= 2; ++lateral)
                {
                    int leafX = originX;
                    int leafZ = originZ;
                    if (dir == 0)
                    {
                        leafX = originX + 2 + outward;
                        leafZ = originZ + lateral;
                    }
                    else if (dir == 1)
                    {
                        leafX = originX - 1 - outward;
                        leafZ = originZ + lateral;
                    }
                    else if (dir == 2)
                    {
                        leafX = originX + lateral;
                        leafZ = originZ + 2 + outward;
                    }
                    else
                    {
                        leafX = originX + lateral;
                        leafZ = originZ - 1 - outward;
                    }

                    if (leafX >= originX && leafX <= originX + 1 &&
                        leafZ >= originZ && leafZ <= originZ + 1 &&
                        leafY <= trunkTopWorld)
                    {
                        continue;
                    }

                    if (callback(leafX, leafY, leafZ, leavesBlock))
                    {
                        return true;
                    }
                }
            }
        }

        for (int step = 1; step <= length; ++step)
        {
            int branchX = originX + lane;
            int branchZ = originZ + lane;
            if (dir == 0)
            {
                branchX = originX + 1 + step;
                branchZ = originZ + lane;
            }
            else if (dir == 1)
            {
                branchX = originX - step;
                branchZ = originZ + lane;
            }
            else if (dir == 2)
            {
                branchX = originX + lane;
                branchZ = originZ + 1 + step;
            }
            else
            {
                branchX = originX + lane;
                branchZ = originZ - step;
            }

            tipX = branchX;
            tipZ = branchZ;
            if (callback(branchX, branchWorldY, branchZ, trunkBlock))
            {
                return true;
            }
        }

        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int back = 0; back <= 1; ++back)
            {
                for (int lateral = -1; lateral <= 1; ++lateral)
                {
                    if (std::abs(lateral) + back + std::abs(dy) > 2)
                    {
                        continue;
                    }

                    const int leafX = tipX - kDirX[static_cast<std::size_t>(dir)] * back +
                                      kSideX[static_cast<std::size_t>(dir)] * lateral;
                    const int leafZ = tipZ - kDirZ[static_cast<std::size_t>(dir)] * back +
                                      kSideZ[static_cast<std::size_t>(dir)] * lateral;
                    const int leafY = branchWorldY + dy;
                    if (leafX >= originX && leafX <= originX + 1 &&
                        leafZ >= originZ && leafZ <= originZ + 1 &&
                        leafY <= groundWorldY + trunkHeight - 1)
                    {
                        continue;
                    }

                    if (callback(leafX, leafY, leafZ, leavesBlock))
                    {
                        return true;
                    }
                }
            }
        }
    }

    for (int layer = 0; layer < kDarkOakCanopyLayers; ++layer)
    {
        const int radius = kDarkOakCanopyRadii[static_cast<std::size_t>(layer)];
        const int worldY = canopyBaseWorld + layer;
        for (int worldX = originX - radius; worldX <= originX + 1 + radius; ++worldX)
        {
            for (int worldZ = originZ - radius; worldZ <= originZ + 1 + radius; ++worldZ)
            {
                if (!darkOakCanopyOccupiesCell(originX, originZ, worldX, worldZ, layer))
                {
                    continue;
                }

                if (worldX >= originX && worldX <= originX + 1 &&
                    worldZ >= originZ && worldZ <= originZ + 1 &&
                    worldY <= groundWorldY + trunkHeight - 1)
                {
                    continue;
                }

                if (callback(worldX, worldY, worldZ, leavesBlock))
                {
                    return true;
                }
            }
        }
    }

    const int hangingLeavesWorldY = canopyBaseWorld - 1;
    for (int worldX = originX - 2; worldX <= originX + 3; ++worldX)
    {
        for (int worldZ = originZ - 2; worldZ <= originZ + 3; ++worldZ)
        {
            const int dx = distanceToInclusiveRange(worldX, originX, originX + 1);
            const int dz = distanceToInclusiveRange(worldZ, originZ, originZ + 1);
            const int chebyshev = std::max(dx, dz);
            const int manhattan = dx + dz;
            if (chebyshev == 0 || chebyshev > 2)
            {
                continue;
            }
            if (manhattan > 2)
            {
                continue;
            }

            if (callback(worldX, hangingLeavesWorldY, worldZ, leavesBlock))
            {
                return true;
            }
        }
    }

    return false;
}

inline bool darkOakTreesTouchOrOverlap(const DarkOakTreeCandidate& a,
                                       const DarkOakTreeCandidate& b) noexcept
{
    const int minAX = a.originX - kDarkOakBranchMaxLength;
    const int maxAX = a.originX + 1 + kDarkOakBranchMaxLength;
    const int minAZ = a.originZ - kDarkOakBranchMaxLength;
    const int maxAZ = a.originZ + 1 + kDarkOakBranchMaxLength;
    const int minAY = a.groundWorldY;
    const int maxAY = a.groundWorldY + a.trunkHeight + kDarkOakCanopyTopOffset;

    const int minBX = b.originX - kDarkOakBranchMaxLength;
    const int maxBX = b.originX + 1 + kDarkOakBranchMaxLength;
    const int minBZ = b.originZ - kDarkOakBranchMaxLength;
    const int maxBZ = b.originZ + 1 + kDarkOakBranchMaxLength;
    const int minBY = b.groundWorldY;
    const int maxBY = b.groundWorldY + b.trunkHeight + kDarkOakCanopyTopOffset;

    return rangesTouchWithinMargin(minAX, maxAX, minBX, maxBX, 0) &&
           rangesTouchWithinMargin(minAY, maxAY, minBY, maxBY, 0) &&
           rangesTouchWithinMargin(minAZ, maxAZ, minBZ, maxBZ, 0);
}

inline bool acaciaTreesTouchOrOverlap(const AcaciaTreeCandidate& a,
                                      const AcaciaTreeCandidate& b) noexcept
{
    const int minAX = a.originX - kAcaciaMaxHorizontalReach;
    const int maxAX = a.originX + kAcaciaMaxHorizontalReach;
    const int minAZ = a.originZ - kAcaciaMaxHorizontalReach;
    const int maxAZ = a.originZ + kAcaciaMaxHorizontalReach;
    const int minAY = a.groundWorldY;
    const int maxAY = a.groundWorldY + a.trunkHeight + 1;

    const int minBX = b.originX - kAcaciaMaxHorizontalReach;
    const int maxBX = b.originX + kAcaciaMaxHorizontalReach;
    const int minBZ = b.originZ - kAcaciaMaxHorizontalReach;
    const int maxBZ = b.originZ + kAcaciaMaxHorizontalReach;
    const int minBY = b.groundWorldY;
    const int maxBY = b.groundWorldY + b.trunkHeight + 1;

    return rangesTouchWithinMargin(minAX, maxAX, minBX, maxBX, 0) &&
           rangesTouchWithinMargin(minAY, maxAY, minBY, maxBY, 0) &&
           rangesTouchWithinMargin(minAZ, maxAZ, minBZ, maxBZ, 0);
}

template <typename SampleColumnFn, typename SurfaceBlockFn, typename DensityFn>
inline bool tryBuildDarkOakCandidate(int originX,
                                     int originZ,
                                     const ColumnSample& columnSample,
                                     SampleColumnFn&& sampleColumn,
                                     SurfaceBlockFn&& surfaceBlockAt,
                                     DensityFn&& densityAt,
                                     DarkOakTreeCandidate& outCandidate)
{
    if (!columnSample.dominantBiome || !columnSample.dominantBiome->generatesTrees)
    {
        return false;
    }

    const BiomeDefinition& biome = *columnSample.dominantBiome;
    if (biome.id != "dark_forest" ||
        columnSample.dominantWeight < kTreeBiomeWeightThreshold ||
        !isDarkOakOrigin(originX, originZ))
    {
        return false;
    }

    int groundWorldY = std::numeric_limits<int>::min();
    for (int trunkX = 0; trunkX < 2; ++trunkX)
    {
        for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
        {
            const ColumnSample trunkSample = sampleColumn(originX + trunkX, originZ + trunkZ);
            if (!trunkSample.dominantBiome || trunkSample.dominantBiome->id != biome.id)
            {
                return false;
            }
            if (trunkSample.dominantWeight < kTreeBiomeWeightThreshold)
            {
                return false;
            }

            const BlockId surfaceBlock = surfaceBlockAt(originX + trunkX, originZ + trunkZ, trunkSample);
            if (surfaceBlock != BlockId::Grass && surfaceBlock != BlockId::Podzol)
            {
                return false;
            }

            if (groundWorldY == std::numeric_limits<int>::min())
            {
                groundWorldY = trunkSample.surfaceY;
            }
            else if (trunkSample.surfaceY != groundWorldY)
            {
                return false;
            }
        }
    }

    if (groundWorldY <= 2)
    {
        return false;
    }

    for (int dx = -2; dx <= 3; ++dx)
    {
        for (int dz = -2; dz <= 3; ++dz)
        {
            const ColumnSample neighborSample = sampleColumn(originX + dx, originZ + dz);
            if (!neighborSample.dominantBiome)
            {
                return false;
            }
            if (std::abs(neighborSample.surfaceY - groundWorldY) > 2)
            {
                return false;
            }
        }
    }

    const float density = densityAt(originX + 1, originZ + 1);
    const float normalizedDensity = std::clamp((density + 1.0f) * 0.5f, 0.0f, 1.0f);
    const int cellX = floorDiv(originX, kDarkOakCellSize);
    const int cellZ = floorDiv(originZ, kDarkOakCellSize);
    const float occupancyRoll = hashToUnitFloat32(cellX, groundWorldY + 509, cellZ);
    if (occupancyRoll > darkOakSpawnChance(biome, normalizedDensity))
    {
        return false;
    }

    outCandidate.originX = originX;
    outCandidate.originZ = originZ;
    outCandidate.groundWorldY = groundWorldY;
    outCandidate.trunkHeight = darkOakTrunkHeight(originX, groundWorldY, originZ);
    outCandidate.priority = darkOakPriority(originX, groundWorldY, originZ);
    return true;
}

template <typename SampleColumnFn, typename SurfaceBlockFn, typename DensityFn>
inline bool tryBuildAcaciaCandidate(int originX,
                                    int originZ,
                                    const ColumnSample& columnSample,
                                    SampleColumnFn&& sampleColumn,
                                    SurfaceBlockFn&& surfaceBlockAt,
                                    DensityFn&& densityAt,
                                    AcaciaTreeCandidate& outCandidate)
{
    if (!columnSample.dominantBiome || !columnSample.dominantBiome->generatesTrees)
    {
        return false;
    }

    const BiomeDefinition& biome = *columnSample.dominantBiome;
    if (biome.id != "savanna" ||
        columnSample.dominantWeight < kTreeBiomeWeightThreshold ||
        !isAcaciaOrigin(originX, originZ))
    {
        return false;
    }

    const BlockId surfaceBlock = surfaceBlockAt(originX, originZ, columnSample);
    if (surfaceBlock != BlockId::Grass)
    {
        return false;
    }

    const int groundWorldY = columnSample.surfaceY;
    if (groundWorldY <= 2)
    {
        return false;
    }

    for (int dx = -2; dx <= 2; ++dx)
    {
        for (int dz = -2; dz <= 2; ++dz)
        {
            const ColumnSample neighborSample = sampleColumn(originX + dx, originZ + dz);
            if (!neighborSample.dominantBiome || neighborSample.dominantBiome->id != biome.id)
            {
                return false;
            }
            if (std::abs(neighborSample.surfaceY - groundWorldY) > 2)
            {
                return false;
            }
        }
    }

    const float density = densityAt(originX, originZ);
    const float normalizedDensity = std::clamp((density + 1.0f) * 0.5f, 0.0f, 1.0f);
    const int cellX = floorDiv(originX, kAcaciaCellSize);
    const int cellZ = floorDiv(originZ, kAcaciaCellSize);
    const float occupancyRoll = hashToUnitFloat32(cellX, groundWorldY + 1153, cellZ);
    if (occupancyRoll > acaciaSpawnChance(biome, normalizedDensity))
    {
        return false;
    }

    outCandidate.originX = originX;
    outCandidate.originZ = originZ;
    outCandidate.groundWorldY = groundWorldY;
    outCandidate.trunkHeight = acaciaTrunkHeight(originX, groundWorldY, originZ);
    outCandidate.priority = acaciaPriority(originX, groundWorldY, originZ);
    return true;
}

template <typename SampleColumnFn, typename SurfaceBlockFn, typename DensityFn>
inline bool darkOakHasSpacingConflict(const DarkOakTreeCandidate& candidate,
                                      SampleColumnFn&& sampleColumn,
                                      SurfaceBlockFn&& surfaceBlockAt,
                                      DensityFn&& densityAt)
{
    const int cellX = floorDiv(candidate.originX, kDarkOakCellSize);
    const int cellZ = floorDiv(candidate.originZ, kDarkOakCellSize);
    for (int neighborCellZ = cellZ - 1; neighborCellZ <= cellZ + 1; ++neighborCellZ)
    {
        for (int neighborCellX = cellX - 1; neighborCellX <= cellX + 1; ++neighborCellX)
        {
            const glm::ivec2 neighborOrigin = darkOakOriginForCell(neighborCellX, neighborCellZ);
            if (neighborOrigin.x == candidate.originX && neighborOrigin.y == candidate.originZ)
            {
                continue;
            }

            const ColumnSample neighborSample = sampleColumn(neighborOrigin.x, neighborOrigin.y);
            DarkOakTreeCandidate neighborCandidate{};
            if (!tryBuildDarkOakCandidate(neighborOrigin.x,
                                          neighborOrigin.y,
                                          neighborSample,
                                          sampleColumn,
                                          surfaceBlockAt,
                                          densityAt,
                                          neighborCandidate))
            {
                continue;
            }

            if (!darkOakTreesTouchOrOverlap(candidate, neighborCandidate))
            {
                continue;
            }

            if (!shouldDarkOakWinTie(candidate, neighborCandidate))
            {
                return true;
            }
        }
    }

    return false;
}

template <typename SampleColumnFn, typename SurfaceBlockFn, typename DensityFn>
inline bool acaciaHasSpacingConflict(const AcaciaTreeCandidate& candidate,
                                     SampleColumnFn&& sampleColumn,
                                     SurfaceBlockFn&& surfaceBlockAt,
                                     DensityFn&& densityAt)
{
    for (int dx = -kAcaciaCellSize; dx <= kAcaciaCellSize; ++dx)
    {
        for (int dz = -kAcaciaCellSize; dz <= kAcaciaCellSize; ++dz)
        {
            if (dx == 0 && dz == 0)
            {
                continue;
            }

            const int neighborX = candidate.originX + dx;
            const int neighborZ = candidate.originZ + dz;
            const ColumnSample neighborSample = sampleColumn(neighborX, neighborZ);

            AcaciaTreeCandidate neighborCandidate{};
            if (!tryBuildAcaciaCandidate(neighborX,
                                         neighborZ,
                                         neighborSample,
                                         sampleColumn,
                                         surfaceBlockAt,
                                         densityAt,
                                         neighborCandidate))
            {
                continue;
            }

            if (!acaciaTreesTouchOrOverlap(candidate, neighborCandidate))
            {
                continue;
            }

            if (!shouldAcaciaWinTie(candidate, neighborCandidate))
            {
                return true;
            }
        }
    }

    return false;
}

#include "chunk_manager_structure_registry.inl"

struct MeshData
{
    static constexpr std::size_t kReusableVertexCapacity = 4096;
    static constexpr std::size_t kReusableIndexCapacity = 6144;

    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;

    MeshData()
    {
        vertices.reserve(4096);
        indices.reserve(6144);
    }

    void clear()
    {
        vertices.clear();
        indices.clear();
    }

    void trimForReuse()
    {
        trimVector(vertices, kReusableVertexCapacity);
        trimVector(indices, kReusableIndexCapacity);
    }

    bool empty() const
    {
        return vertices.empty() || indices.empty();
    }

    [[nodiscard]] std::size_t retainedBytes() const noexcept
    {
        return vertices.capacity() * sizeof(Vertex) +
               indices.capacity() * sizeof(std::uint32_t);
    }

private:
    template <typename T>
    static void trimVector(std::vector<T>& values, std::size_t targetCapacity)
    {
        if (values.capacity() <= targetCapacity * 2)
        {
            values.clear();
            return;
        }

        std::vector<T> trimmed;
        trimmed.reserve(targetCapacity);
        values.swap(trimmed);
    }
};

enum class ChunkState : std::uint8_t
{
    Empty = 0,
    Generating,
    Meshing,
    Ready,
    Uploaded,
    Remeshing
};

const char* chunkStateLabel(ChunkState state) noexcept
{
    switch (state)
    {
    case ChunkState::Empty:
        return "Empty";
    case ChunkState::Generating:
        return "Generating";
    case ChunkState::Meshing:
        return "Meshing";
    case ChunkState::Ready:
        return "Ready";
    case ChunkState::Uploaded:
        return "Uploaded";
    case ChunkState::Remeshing:
        return "Remeshing";
    }
    return "Unknown";
}

} // namespace

constexpr std::uint32_t kInvalidChunkBufferPage = std::numeric_limits<std::uint32_t>::max();

namespace chunk_manager_detail
{
struct ChunkYInterval
{
    int minChunkY{0};
    int maxChunkY{0};
};

struct ColumnChunkIntervals
{
    static constexpr std::size_t kMaxIntervals = 8;

    std::array<ChunkYInterval, kMaxIntervals> intervals{};
    std::uint8_t count{0};

    [[nodiscard]] bool empty() const noexcept
    {
        return count == 0;
    }

    [[nodiscard]] int minChunkY() const noexcept
    {
        return empty() ? 0 : intervals[0].minChunkY;
    }

    [[nodiscard]] int maxChunkY() const noexcept
    {
        return empty() ? 0 : intervals[count - 1].maxChunkY;
    }
};

struct Chunk
{
    static constexpr std::size_t kColumnCount =
        static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ);

    struct PendingRenderMesh
    {
        std::uint32_t pageIndex{kInvalidChunkBufferPage};
        std::size_t vertexOffset{0};
        std::size_t indexOffset{0};
        std::size_t vertexCount{0};
        std::size_t indexCount{0};
        std::uint32_t meshVersion{0};
        UINT64 uploadFenceValue{0};

        [[nodiscard]] bool valid() const noexcept
        {
            return pageIndex != kInvalidChunkBufferPage;
        }
    };

    explicit Chunk(const glm::ivec3& c)
        : coord(c),
          minWorldY(c.y * kChunkSizeY),
          maxWorldY(minWorldY + kChunkSizeY - 1),
          blocks(kChunkBlockCount, BlockId::Air),
          lightLevels(kChunkBlockCount, packLightLevels(kMaxLightLevel, 0)),
          state(ChunkState::Empty)
    {
    }

    void reset(const glm::ivec3& c)
    {
        coord = c;
        minWorldY = c.y * kChunkSizeY;
        maxWorldY = minWorldY + kChunkSizeY - 1;
        if (blocks.size() != static_cast<std::size_t>(kChunkBlockCount))
        {
            blocks.assign(kChunkBlockCount, BlockId::Air);
        }
        else
        {
            std::fill(blocks.begin(), blocks.end(), BlockId::Air);
        }
        if (lightLevels.size() != static_cast<std::size_t>(kChunkBlockCount))
        {
            lightLevels.assign(kChunkBlockCount, packLightLevels(kMaxLightLevel, 0));
        }
        else
        {
            std::fill(lightLevels.begin(), lightLevels.end(), packLightLevels(kMaxLightLevel, 0));
        }
        state.store(ChunkState::Empty, std::memory_order_relaxed);
        meshData.trimForReuse();
        meshReady.store(false, std::memory_order_relaxed);
        hasBlocks.store(false, std::memory_order_relaxed);
        queuedForUpload.store(false, std::memory_order_relaxed);
        queuedUploadBucket.store(std::numeric_limits<std::uint8_t>::max(), std::memory_order_relaxed);
        uploadQueueTicket.store(0, std::memory_order_relaxed);
        queuedForCommit.store(false, std::memory_order_relaxed);
        commitQueueTicket.store(0, std::memory_order_relaxed);
        indexCount.store(0, std::memory_order_relaxed);
        vertexCount.store(0, std::memory_order_relaxed);
        bufferPageIndex.store(kInvalidChunkBufferPage, std::memory_order_relaxed);
        vertexOffset.store(0, std::memory_order_relaxed);
        indexOffset.store(0, std::memory_order_relaxed);
        inFlight.store(0, std::memory_order_relaxed);
        requestTimestampMicros.store(0, std::memory_order_relaxed);
        initialReadyRecorded.store(false, std::memory_order_relaxed);
        generateStartTimestampMicros.store(0, std::memory_order_relaxed);
        generateDoneTimestampMicros.store(0, std::memory_order_relaxed);
        meshQueuedTimestampMicros.store(0, std::memory_order_relaxed);
        meshStartTimestampMicros.store(0, std::memory_order_relaxed);
        meshDoneTimestampMicros.store(0, std::memory_order_relaxed);
        uploadQueuedTimestampMicros.store(0, std::memory_order_relaxed);
        uploadStartTimestampMicros.store(0, std::memory_order_relaxed);
        lightBoundaryDirtyMask = 0;
        pendingMeshRefresh.store(false, std::memory_order_relaxed);
        meshVersion.store(0, std::memory_order_relaxed);
        generationEpoch.store(0, std::memory_order_relaxed);
        lightingRevision.store(0, std::memory_order_relaxed);
        appliedLightingRevision.store(0, std::memory_order_relaxed);
        skyLightCacheGeneration.store(0, std::memory_order_relaxed);
        skyLightFromAboveCache.fill(kMaxLightLevel);
        cpuDataResident = true;
        lastDenseFrameTouched = 0;
        pendingMesh = {};
    }

    void ensureCpuDataAllocated()
    {
        if (blocks.size() != static_cast<std::size_t>(kChunkBlockCount))
        {
            blocks.assign(kChunkBlockCount, BlockId::Air);
        }
        else
        {
            std::fill(blocks.begin(), blocks.end(), BlockId::Air);
        }

        if (lightLevels.size() != static_cast<std::size_t>(kChunkBlockCount))
        {
            lightLevels.assign(kChunkBlockCount, packLightLevels(kMaxLightLevel, 0));
        }
        else
        {
            std::fill(lightLevels.begin(), lightLevels.end(), packLightLevels(kMaxLightLevel, 0));
        }

        cpuDataResident = true;
    }

    void releaseCpuData()
    {
        std::vector<BlockId>().swap(blocks);
        std::vector<std::uint8_t>().swap(lightLevels);
        cpuDataResident = false;
    }

    glm::ivec3 coord;
    int minWorldY{0};
    int maxWorldY{0};
    std::vector<BlockId> blocks;
    std::vector<std::uint8_t> lightLevels;
    std::atomic<ChunkState> state;

    std::atomic<std::uint32_t> indexCount{0};
    std::atomic<std::size_t> vertexCount{0};
    std::atomic<std::uint32_t> bufferPageIndex{kInvalidChunkBufferPage};
    std::atomic<std::size_t> vertexOffset{0};
    std::atomic<std::size_t> indexOffset{0};
    std::atomic<bool> queuedForUpload{false};
    std::atomic<std::uint8_t> queuedUploadBucket{std::numeric_limits<std::uint8_t>::max()};
    std::atomic<std::uint64_t> uploadQueueTicket{0};
    std::atomic<bool> queuedForCommit{false};
    std::atomic<std::uint64_t> commitQueueTicket{0};

    mutable std::mutex meshMutex;
    MeshData meshData;
    std::atomic<bool> meshReady{false};
    std::atomic<bool> hasBlocks{false};
    std::atomic<int> inFlight{0};
    std::atomic<long long> requestTimestampMicros{0};
    std::atomic<bool> initialReadyRecorded{false};
    std::atomic<long long> generateStartTimestampMicros{0};
    std::atomic<long long> generateDoneTimestampMicros{0};
    std::atomic<long long> meshQueuedTimestampMicros{0};
    std::atomic<long long> meshStartTimestampMicros{0};
    std::atomic<long long> meshDoneTimestampMicros{0};
    std::atomic<long long> uploadQueuedTimestampMicros{0};
    std::atomic<long long> uploadStartTimestampMicros{0};
    std::uint8_t lightBoundaryDirtyMask{0};
    std::atomic<bool> pendingMeshRefresh{false};
    std::atomic<std::uint32_t> meshVersion{0};
    std::atomic<std::uint32_t> generationEpoch{0};
    std::atomic<std::uint64_t> lightingRevision{0};
    std::atomic<std::uint64_t> appliedLightingRevision{0};
    std::atomic<std::uint64_t> skyLightCacheGeneration{0};
    std::array<std::uint8_t, kChunkSizeX * kChunkSizeZ> skyLightFromAboveCache{};
    bool cpuDataResident{true};
    std::uint64_t lastDenseFrameTouched{0};
    PendingRenderMesh pendingMesh{};
};

} // namespace chunk_manager_detail

using chunk_manager_detail::Chunk;
using chunk_manager_detail::ChunkYInterval;
using chunk_manager_detail::ColumnChunkIntervals;

namespace
{
[[nodiscard]] ChunkBlockView makeChunkBlockView(const Chunk& chunk) noexcept
{
    return ChunkBlockView{chunk.coord, chunk.minWorldY, std::span<const BlockId>(chunk.blocks)};
}

struct ProfilingCounters
{
    std::atomic<long long> generationMicros{0};
    std::atomic<long long> relightMicros{0};
    std::atomic<long long> meshingMicros{0};
    std::atomic<std::size_t> uploadedBytes{0};
    std::atomic<std::uint64_t> relightRegionChunks{0};
    std::atomic<std::uint64_t> relightChangedChunks{0};
    std::atomic<std::uint64_t> relightExternalSnapshotChunks{0};
    std::atomic<std::uint64_t> relightSkyAboveChunkScans{0};
    std::atomic<std::uint64_t> relightSkySeedNodes{0};
    std::atomic<std::uint64_t> relightBlockSeedNodes{0};
    std::atomic<std::uint64_t> relightSkyNodesProcessed{0};
    std::atomic<std::uint64_t> relightBlockNodesProcessed{0};
    std::atomic<int> generatedChunks{0};
    std::atomic<int> relitChunks{0};
    std::atomic<int> relightBatches{0};
    std::atomic<int> meshedChunks{0};
    std::atomic<int> uploadedChunks{0};
    std::atomic<int> throttledUploads{0};
    std::atomic<int> deferredUploads{0};
    std::atomic<int> evictedChunks{0};
};

struct PendingStructureEdit
{
    glm::ivec3 chunkCoord{0};
    glm::ivec3 worldPos{0};
    BlockId block{BlockId::Air};
    bool replaceSolid{false};
};

struct BlockEditOverlayEntry
{
    std::uint16_t localIndex{0};
    BlockId block{BlockId::Air};
};

[[nodiscard]] bool chunkAwaitingInitialReady(const Chunk& chunk) noexcept
{
    return chunk.requestTimestampMicros.load(std::memory_order_acquire) > 0 &&
           !chunk.initialReadyRecorded.load(std::memory_order_acquire);
}

[[nodiscard]] bool chunkAwaitingInitialVisibleReady(const Chunk& chunk) noexcept
{
    return chunkAwaitingInitialReady(chunk) &&
           chunk.hasBlocks.load(std::memory_order_acquire);
}

enum class UploadQueueBucket : std::uint8_t
{
    InitialVisible = 0,
    NearFrontVisible = 1,
    Background = 2,
    Retry = 3
};

inline constexpr std::size_t kUploadQueueBucketCount = 4;

[[nodiscard]] constexpr std::size_t uploadQueueBucketIndex(UploadQueueBucket bucket) noexcept
{
    return static_cast<std::size_t>(bucket);
}

struct UploadQueueEntry
{
    std::weak_ptr<Chunk> chunk;
    ChunkPriorityKey priority{};
    std::uint64_t ticket{0};
    std::uint64_t sequence{0};
};

struct PendingCommitQueueEntry
{
    std::weak_ptr<Chunk> chunk;
    std::uint64_t ticket{0};
    UINT64 uploadFenceValue{0};
};

[[nodiscard]] int compareChunkPriorityKeysLocal(const ChunkPriorityKey& lhs,
                                                const ChunkPriorityKey& rhs) noexcept
{
    if (lhs.supportBucket != rhs.supportBucket)
    {
        return lhs.supportBucket < rhs.supportBucket ? -1 : 1;
    }
    if (lhs.horizontalDistance != rhs.horizontalDistance)
    {
        return lhs.horizontalDistance < rhs.horizontalDistance ? -1 : 1;
    }
    if (lhs.forwardBucket != rhs.forwardBucket)
    {
        return lhs.forwardBucket < rhs.forwardBucket ? -1 : 1;
    }
    if (lhs.verticalDistance != rhs.verticalDistance)
    {
        return lhs.verticalDistance < rhs.verticalDistance ? -1 : 1;
    }
    if (lhs.axisDistance != rhs.axisDistance)
    {
        return lhs.axisDistance < rhs.axisDistance ? -1 : 1;
    }
    return 0;
}

struct UploadQueueEntryComparer
{
    bool operator()(const UploadQueueEntry& lhs, const UploadQueueEntry& rhs) const noexcept
    {
        const int priorityComparison = compareChunkPriorityKeysLocal(lhs.priority, rhs.priority);
        if (priorityComparison != 0)
        {
            return priorityComparison > 0;
        }

        return lhs.sequence > rhs.sequence;
    }
};

#include "chunk_manager_far_terrain.inl"

struct ChunkBuildScratch;

} // namespace

struct ChunkManager::Impl
{
    explicit Impl(unsigned seed);
    ~Impl();

    void initializeRendering(ID3D12Device* device);
    void setRenderSynchronization(ID3D12Fence* graphicsFence, std::uint64_t graphicsFenceValue);
    [[nodiscard]] ID3D12Fence* uploadFence() const noexcept;
    [[nodiscard]] std::uint64_t lastSubmittedUploadFenceValue() const noexcept;
    [[nodiscard]] ID3D12Fence* farUploadFence() const noexcept;
    [[nodiscard]] std::uint64_t lastSubmittedFarUploadFenceValue() const noexcept;
    void setBlockTextureAtlasConfig(const BlockTextureAtlasConfig& config);
    void update(const glm::vec3& cameraPos);
    void update(const glm::vec3& cameraPos, const glm::vec3& cameraForward);
    WorldRenderData buildRenderData(const Frustum& frustum) const;

    float surfaceHeight(float worldX, float worldZ) const noexcept;
    ColumnSample sampleColumnAt(const glm::vec3& worldPos,
                                int slabMinWorldY = std::numeric_limits<int>::min(),
                                int slabMaxWorldY = std::numeric_limits<int>::max()) const;
    void clear();

    bool destroyBlock(const glm::ivec3& worldPos);
    bool placeBlock(const glm::ivec3& targetBlockPos, const glm::ivec3& faceNormal, BlockId block);

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
    void setStartupEnabled(bool enabled) noexcept;
    bool startupEnabled() const noexcept;
    StreamingStatusSnapshot streamingStatusSnapshot() const noexcept;
    LodDiagnosticsSnapshot lodDiagnosticsSnapshot(const glm::vec3& cameraPos) const;
    RecentEditHoleDebugSnapshot recentEditHoleDebugSnapshot(const glm::vec3& cameraPos) const;
    void writeLodDebugSnapshot(const std::filesystem::path& outputPath, const glm::vec3& cameraPos) const;
    ChunkProfilingSnapshot sampleProfilingSnapshot();
    void setBenchmarkMetricsEnabled(bool enabled) noexcept;
    bool benchmarkMetricsEnabled() const noexcept;
    void resetBenchmarkMetrics();
    ChunkBenchmarkReport benchmarkReport() const;
    std::string biomeNameAt(const glm::vec3& worldPos) const;

private:
    enum class ColumnHeightPrefetchPriority : std::uint8_t;

    struct TreeDensityNoise
    {
        TreeDensityNoise() = default;

        explicit TreeDensityNoise(unsigned seed)
        {
            reseed(seed);
        }

        void reseed(unsigned seed)
        {
            seed_ = seed;

            std::mt19937 rng(seed_);
            std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
            for (auto& offset : octaveOffsets_)
            {
                offset = {dist(rng), dist(rng)};
            }
        }

        [[nodiscard]] float fbm(float x,
                                float y,
                                int octaves,
                                float persistence,
                                float lacunarity) const noexcept
        {
            float amplitude = 1.0f;
            float frequency = 1.0f;
            float value = 0.0f;
            float normalization = 0.0f;

            const int octaveCount = std::min<int>(octaves, static_cast<int>(octaveOffsets_.size()));
            for (int i = 0; i < octaveCount; ++i)
            {
                const glm::vec2 sample{x * frequency + octaveOffsets_[i].x,
                                       y * frequency + octaveOffsets_[i].y};
                value += glm::perlin(sample) * amplitude;
                normalization += amplitude;

                amplitude *= persistence;
                frequency *= lacunarity;
            }

            if (normalization > 0.0f)
            {
                value /= normalization;
            }

            return value;
        }

        [[nodiscard]] const std::array<glm::vec2, 16>& octaveOffsets() const noexcept
        {
            return octaveOffsets_;
        }

    private:
        unsigned seed_{0};
        std::array<glm::vec2, 16> octaveOffsets_{};
    };

    terrain::WorldgenProfile worldgenProfile_{};
    terrain::BiomeDatabase biomeDatabase_;
    std::unique_ptr<terrain::ClimateMap> climateMap_;
    std::unique_ptr<terrain::SurfaceMap> surfaceMap_;
    std::unique_ptr<terrain::TerrainGenerator> terrainGenerator_;
    int globalSeaLevel_{20};
    TreeDensityNoise noise_{};
    mutable StructureRegistry structureRegistry_{};

    void startWorkerThreads();
    void stopWorkerThreads();
    void workerThreadFunction();
    void columnHeightPrefetchThreadFunction();
    bool acquireNextColumnHeightPrefetch(glm::ivec2& column,
                                         std::uint64_t& token,
                                         ColumnHeightPrefetchPriority& priority);
    void finishColumnHeightPrefetch(const glm::ivec2& column, std::uint64_t token) const;
    void enqueueJob(const std::shared_ptr<Chunk>& chunk,
                    JobType type,
                    const glm::ivec3& coord,
                    std::uint32_t generationEpoch = 0,
                    bool initialReadyPriority = false);
    void processJob(const Job& job);
    std::shared_ptr<Chunk> popNextChunkForUpload();
    void queueChunkForUpload(const std::shared_ptr<Chunk>& chunk, bool retryBucket = false);
    void requeueChunkForUpload(const std::shared_ptr<Chunk>& chunk, bool retryBucket);
    [[nodiscard]] ChunkPriorityKey buildUploadPriorityKey(const glm::ivec3& coord,
                                                          const glm::ivec3& origin,
                                                          const glm::vec3& forward) const noexcept;
    [[nodiscard]] UploadQueueBucket classifyUploadQueueBucket(const Chunk& chunk,
                                                              const glm::ivec3& origin,
                                                              const glm::vec3& forward,
                                                              bool retryBucket) const noexcept;
    void queueChunkForCommit(const std::shared_ptr<Chunk>& chunk, UINT64 uploadFenceValue);

    enum class ChunkBufferPageState : std::uint8_t
    {
        Available = 0,
        PendingOpen,
        PendingUploaded,
        Resident,
        Retiring
    };

    struct ChunkBufferPage
    {
        struct Range
        {
            std::size_t offset{0};
            std::size_t size{0};
        };

        Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> indexBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> vertexUploadBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> indexUploadBuffer;
        D3D12_VERTEX_BUFFER_VIEW vertexView{};
        D3D12_INDEX_BUFFER_VIEW indexView{};
        std::byte* mappedVertexData{nullptr};
        std::byte* mappedIndexData{nullptr};
        D3D12_RESOURCE_STATES vertexState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES indexState{D3D12_RESOURCE_STATE_COMMON};
        std::size_t vertexCapacity{0};
        std::size_t indexCapacity{0};
        std::size_t vertexCursor{0};
        std::size_t indexCursor{0};
        std::size_t residentChunks{0};
        std::size_t pendingChunks{0};
        UINT64 pendingBatchId{0};
        UINT64 uploadFenceValue{0};
        UINT64 retireFenceValue{0};
        ChunkBufferPageState state{ChunkBufferPageState::Available};
    };

    struct ChunkAllocation
    {
        std::uint32_t pageIndex{kInvalidChunkBufferPage};
        std::size_t vertexOffset{0};
        std::size_t indexOffset{0};
    };

    struct DeferredPendingChunkRelease
    {
        std::uint32_t pageIndex{kInvalidChunkBufferPage};
        std::size_t vertexOffset{0};
        std::size_t indexOffset{0};
        std::size_t vertexCount{0};
        std::size_t indexCount{0};
        UINT64 uploadFenceValue{0};
    };

    static std::size_t nextPowerOfTwo(std::size_t value) noexcept;
    ChunkBufferPage createBufferPage(std::size_t vertexCount, std::size_t indexCount);
    [[nodiscard]] static const char* chunkBufferPageStateLabel(ChunkBufferPageState state) noexcept;
    [[nodiscard]] std::string summarizeChunkBufferPagesLocked() const;
    void ensureChunkBufferPageUploadBuffers(ChunkBufferPage& page);
    static void releaseChunkBufferPageUploadBuffers(ChunkBufferPage& page) noexcept;
    ChunkAllocation acquireChunkAllocation(std::size_t vertexCount, std::size_t indexCount, UINT64 uploadBatchId);
    static void resetChunkBufferPage(ChunkBufferPage& page) noexcept;
    void sealPendingChunkUploadPages(UINT64 uploadBatchId, UINT64 uploadFenceValue);
    void collectReusableChunkBufferPages();
    void releaseChunkAllocationRange(std::uint32_t pageIndex,
                                     std::size_t vertexOffset,
                                     std::size_t vertexCount,
                                     std::size_t indexOffset,
                                     std::size_t indexCount,
                                     bool residentAllocation);
    void collectDeferredPendingChunkReleases();
    void deferPendingChunkRelease(const Chunk::PendingRenderMesh& pendingMesh);
    void commitPendingChunkUploads();
    void releaseChunkAllocation(Chunk& chunk);
    void recycleChunkGPU(Chunk& chunk);
    void destroyBufferPages();
    int computeVerticalRadius(const glm::ivec3& center, int horizontalRadius, int cameraWorldY);
    int updateEvictionCenterChunkY(int targetChunkY) noexcept;
    int computeEvictionBudget(std::size_t pendingEvictions) const noexcept;
    int columnRadiusFor(const glm::ivec2& column,
                        const glm::ivec2& cameraColumn,
                        int cameraChunkY,
                        int verticalRadius) const;
    int columnRadiusForHeight(const glm::ivec2& column,
                              const glm::ivec2& cameraColumn,
                              int cameraChunkY,
                              int verticalRadius,
                              int columnHeight) const;
    int cameraBandRadiusForColumn(const glm::ivec2& column,
                                  const glm::ivec2& cameraColumn,
                                  int verticalRadius) const;
    int surfaceShellFloorChunkForHeight(const glm::ivec2& column, int columnHeight) const;
    [[nodiscard]] bool columnUsesPlayerBand(const glm::ivec2& column,
                                            const glm::ivec2& cameraColumn) const noexcept;
    ColumnChunkIntervals playerBandIntervalsForColumn(const glm::ivec2& column,
                                                      const glm::ivec2& cameraColumn,
                                                      int cameraChunkY,
                                                      int verticalRadius) const;
    std::pair<int, int> columnSpanFor(const glm::ivec2& column,
                                      const glm::ivec2& cameraColumn,
                                      int cameraChunkY,
                                      int verticalRadius) const;
    std::pair<int, int> columnSpanForHeight(const glm::ivec2& column,
                                            const glm::ivec2& cameraColumn,
                                            int cameraChunkY,
                                            int verticalRadius,
                                            int columnHeight) const;
    ColumnChunkIntervals columnIntervalsFor(const glm::ivec2& column,
                                            const glm::ivec2& cameraColumn,
                                            int cameraChunkY,
                                            int verticalRadius) const;
    ColumnChunkIntervals columnIntervalsForHeight(const glm::ivec2& column,
                                                  const glm::ivec2& cameraColumn,
                                                  int cameraChunkY,
                                                  int verticalRadius,
                                                  int columnHeight) const;
    static void addColumnChunkInterval(ColumnChunkIntervals& intervals, int minChunkY, int maxChunkY) noexcept;
    static void mergeColumnChunkIntervals(ColumnChunkIntervals& dst,
                                          const ColumnChunkIntervals& src) noexcept;
    static bool chunkYWithinIntervals(int chunkY, const ColumnChunkIntervals& intervals, int slackChunks = 0) noexcept;
    static int chunkYDistanceToIntervals(int chunkY, const ColumnChunkIntervals& intervals, int slackChunks = 0) noexcept;
    enum class ColumnSlabOccupancyState : std::uint8_t
    {
        DefinitelyEmpty = 0,
        DefinitelyOccupied,
        MaybeOccupied
    };
    struct ColumnSlabOccupancy
    {
        ColumnChunkIntervals terrainIntervals{};
        ColumnChunkIntervals surfaceShellIntervals{};
        ColumnChunkIntervals waterIntervals{};
        ColumnChunkIntervals structureIntervals{};
        ColumnChunkIntervals editIntervals{};
        ColumnChunkIntervals occupiedIntervals{};
        ColumnChunkIntervals maybeIntervals{};
        int highestOccupiedChunkY{-1};
    };
    [[nodiscard]] int adjustedSurfaceYForColumn(const terrain::SurfaceColumn& surfaceColumn,
                                                float neighborAverage) const noexcept;
    ColumnSlabOccupancy buildColumnSlabOccupancy(const glm::ivec2& column) const;
    ColumnSlabOccupancy cachedColumnSlabOccupancy(const glm::ivec2& column) const;
    bool tryGetCachedColumnSlabOccupancy(const glm::ivec2& column, ColumnSlabOccupancy& out) const;
    [[nodiscard]] static ColumnSlabOccupancyState classifyColumnSlab(const ColumnSlabOccupancy& occupancy,
                                                                     int chunkY) noexcept;
    void invalidateColumnSlabOccupancy(const glm::ivec2& column) const;
    void invalidateAllColumnSlabOccupancy() const;
    void updateMovementEnvelopeState(const glm::ivec3& center, const glm::ivec3& previousCenter) noexcept;
    enum class MovementEnvelopeBucket : std::uint8_t
    {
        Core = 0,
        Corridor,
        TurnReserve,
        Background
    };
    enum class SchedulingBand : std::uint8_t
    {
        Protected = 0,
        Background
    };
    [[nodiscard]] MovementEnvelopeBucket movementEnvelopeBucketForColumn(const glm::ivec2& column,
                                                                         const glm::ivec3& center,
                                                                         int horizontalRadius,
                                                                         int lookaheadChunks = 0) const noexcept;
    void prefetchVisibleColumnHeights(const glm::ivec3& center,
                                      const glm::ivec3& previousCenter,
                                      int horizontalRadius);
    void resetColumnBudgets();
    int baseUploadsPerColumnLimit(int verticalRadius) const noexcept;
    std::size_t estimateUploadQueueSize();
    std::size_t estimateInitialReadyUploadQueueSize();
    struct UploadBudgets
    {
        std::size_t byteBudget{kUploadBudgetBytesPerFrame};
        int columnLimit{kVerticalStreamingConfig.uploadBasePerColumn};
        int chunkLimit{1};
        std::size_t queueSize{0};
        double timeBudgetMs{4.0};
    };
    UploadBudgets computeUploadBudgets(int verticalRadius);
    static int computeBacklogSteps(int backlog, int threshold, int stepSize) noexcept;
    int computeGenerationBudget(int horizontalRadius, int verticalRadius, int backlogSteps) const;
    int computeRingExpansionBudget(int backlogChunks) const;
    int computeColumnJobCap(int backlogSteps, int backlogChunks) const;
    struct VisibleChunkCoverage
    {
        int missing{0};
        int ready{0};
        int required{0};
        int protectedMissing{0};
        int protectedReady{0};
        int protectedRequired{0};
    };
    VisibleChunkCoverage scanVisibleChunkCoverage(const glm::ivec3& center,
                                                  int horizontalRadius,
                                                  int verticalRadius) const;
    int estimateMissingChunks(const glm::ivec3& center, int horizontalRadius, int verticalRadius) const;
    StreamingStatusSnapshot computeStreamingStatusSnapshot() const noexcept;

    struct RingProgress
    {
        bool fullyLoaded{false};
        bool budgetExhausted{false};
    };

    RingProgress ensureVolume(const glm::ivec3& center,
                              int horizontalRadius,
                              int verticalRadius,
                              int& jobBudget,
                              SchedulingBand band);
    void removeDistantChunks(const glm::ivec3& center, int horizontalThreshold, int verticalThreshold);
    bool ensureChunkAsync(const glm::ivec3& coord, bool forceResident = false);
    void uploadReadyMeshes();
    bool uploadChunkMesh(Chunk& chunk, UINT64 uploadBatchId);
    void buildChunkMeshAsync(Chunk& chunk);
    void updateDenseChunkResidency(const glm::ivec3& centerChunk);
    [[nodiscard]] int denseCpuHorizontalRadius() const noexcept;
    [[nodiscard]] int denseCpuVerticalRadius() const noexcept;
    [[nodiscard]] bool shouldKeepChunkCpuDense(const Chunk& chunk,
                                               const glm::ivec3& centerChunk,
                                               int horizontalRadius,
                                               int verticalRadius) const;
    [[nodiscard]] bool ensureChunkCpuDataResident(Chunk& chunk);
    [[nodiscard]] bool chunkHasPendingStructureEdits(const glm::ivec3& coord) const;
    void releaseChunkCpuData(Chunk& chunk);
    static glm::ivec3 worldToChunkCoords(int worldX, int worldY, int worldZ) noexcept;
    static std::size_t estimateChunkRetainedBytes(const Chunk& chunk) noexcept;
    std::size_t chunkPoolBudgetBytes() const noexcept;
    void trimChunkPoolToBudget();
    void trimChunkPoolToBudgetLocked(std::size_t budgetBytes);
    std::shared_ptr<Chunk> acquireChunk(const glm::ivec3& coord);

    std::shared_ptr<Chunk> getChunkShared(const glm::ivec3& coord) noexcept;
    std::shared_ptr<const Chunk> getChunkShared(const glm::ivec3& coord) const noexcept;
    Chunk* getChunk(const glm::ivec3& coord) noexcept;
    const Chunk* getChunk(const glm::ivec3& coord) const noexcept;
    void requestChunkRemesh(const std::shared_ptr<Chunk>& chunk);
    void requestChunkRemeshFromRelight(const std::shared_ptr<Chunk>& chunk);
    void markNeighborsForRemeshingIfNeeded(const glm::ivec3& coord, int localX, int localY, int localZ);
    void relightAroundChunk(const glm::ivec3& centerCoord);
    void queueChunkForLightingRemesh(const std::shared_ptr<Chunk>& chunk);
    std::uint8_t packedLightAtWorld(const glm::ivec3& worldPos) const noexcept;
    void noteChunkReadyLatency(Chunk& chunk);
    [[nodiscard]] bool generateChunkBlocks(Chunk& chunk, std::uint32_t generationEpoch);
    void buildChunkCpuBlocks(const Chunk& chunk,
                             ChunkBuildScratch& scratch,
                             bool includePendingStructureEdits,
                             std::array<ColumnBuildResult, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)>&
                                 columnResults,
                             std::vector<PendingStructureEdit>* consumedPendingEdits = nullptr);
    void rebuildChunkBaseLighting(Chunk& chunk) const;
    ColumnSample sampleColumn(int worldX,
                              int worldZ,
                              int slabMinWorldY = std::numeric_limits<int>::min(),
                              int slabMaxWorldY = std::numeric_limits<int>::max(),
                              bool includeBlendDebug = false) const;
    int ensureColumnHeightCached(const glm::ivec2& column, int worldX, int worldZ) const;
    bool tryGetCachedColumnHeight(const glm::ivec2& column, int worldX, int worldZ, int& outHeight) const;
    bool tryGetPredictedColumnHeight(const glm::ivec2& column, int& outHeight) const;
    int cacheSampledColumnHeight(const glm::ivec2& column, int worldX, int worldZ) const;
    enum class ColumnHeightPrefetchPriority : std::uint8_t
    {
        Background = 0,
        Normal,
        Visible,
        Critical
    };
    struct ColumnHeightPrefetchRequest
    {
        glm::ivec2 column{0};
        std::uint64_t token{0};
        std::uint64_t sequence{0};
        std::uint32_t distance{0};
        ColumnHeightPrefetchPriority priority{ColumnHeightPrefetchPriority::Normal};
    };
    struct ColumnHeightPrefetchRequestState
    {
        std::uint64_t token{0};
        ColumnHeightPrefetchPriority priority{ColumnHeightPrefetchPriority::Normal};
        bool inFlight{false};
    };
    struct ColumnHeightPrefetchRequestCompare
    {
        bool operator()(const ColumnHeightPrefetchRequest& lhs,
                        const ColumnHeightPrefetchRequest& rhs) const noexcept
        {
            if (lhs.priority != rhs.priority)
            {
                return lhs.priority < rhs.priority;
            }
            if (lhs.distance != rhs.distance)
            {
                return lhs.distance > rhs.distance;
            }
            return lhs.sequence > rhs.sequence;
        }
    };
    void requestColumnHeightPrefetch(const glm::ivec2& column,
                                     ColumnHeightPrefetchPriority priority = ColumnHeightPrefetchPriority::Normal) const;
    void mergePredictedColumnHeight(const glm::ivec2& column, int height) const;
    void refreshPredictedColumnHeightFromLoadedData(const glm::ivec2& column) const;
    void invalidatePredictedColumn(const glm::ivec2& column) const;
    std::vector<PendingStructureEdit> takePendingStructureEdits(const glm::ivec3& coord);
    std::vector<PendingStructureEdit> copyPendingStructureEdits(const glm::ivec3& coord) const;
    bool applyPendingStructureEditsLocked(Chunk& chunk);
    static std::uint16_t blockOverlayLocalIndex(int localX, int localY, int localZ) noexcept;
    void applyBlockEditOverlay(ChunkBuildScratch& scratch, const glm::ivec3& chunkCoord) const;
    void recordBlockEditOverlay(const glm::ivec3& worldPos, BlockId block);
    bool tryGetBlockEditOverlay(const glm::ivec3& worldPos, BlockId& outBlock) const;
    void dispatchStructureEdits(const std::vector<PendingStructureEdit>& edits);
    std::vector<StructureInstance> queryStructureInstances(const glm::ivec3& minWorld,
                                                           const glm::ivec3& maxWorld,
                                                           int lodLevel = 0) const;
    static bool chunkHasSolidBlocks(const Chunk& chunk) noexcept;
    void recycleChunkObject(std::shared_ptr<Chunk> chunk);
    std::pair<glm::vec2, glm::vec2> atlasUvFor(BlockId block, BlockFace face) const;
    void noteRecentEdit(const char* kind, const glm::ivec3& worldPos, const glm::ivec3& chunkCoord);
    void appendRecentEditDebugEvent(const std::string& event);
    bool shouldTrackRecentEditChunk(const glm::ivec3& coord) const;

    using RelightCoordGenerationMap = std::unordered_map<glm::ivec3, std::uint64_t, ChunkHasher>;
    using RelightCoordSet = std::unordered_set<glm::ivec3, ChunkHasher>;

    struct PendingRelightBatch
    {
        bool valid{false};
        glm::ivec3 minCoord{0};
        glm::ivec3 maxCoord{0};
        glm::ivec3 reservedMinCoord{0};
        glm::ivec3 reservedMaxCoord{0};
        RelightCoordGenerationMap dirtyCoordGenerations{};
        RelightCoordSet forceRemeshCoords{};
        bool containsInitialReadyCoord{false};
        std::uint64_t maxGeneration{0};
        std::uint64_t estimatedCostUnits{0};
        std::uint64_t sequence{0};
    };

    struct ActiveRelightRegion
    {
        glm::ivec3 minCoord{0};
        glm::ivec3 maxCoord{0};
        RelightCoordGenerationMap dirtyCoordGenerations{};
        std::uint64_t maxGeneration{0};
        std::uint64_t sequence{0};
    };

    struct ChunkNeighborhoodSnapshot
    {
        static constexpr int kExtentX = kChunkSizeX + 2;
        static constexpr int kExtentY = kChunkSizeY + 2;
        static constexpr int kExtentZ = kChunkSizeZ + 2;
        static constexpr std::size_t kVoxelCount =
            static_cast<std::size_t>(kExtentX) *
            static_cast<std::size_t>(kExtentY) *
            static_cast<std::size_t>(kExtentZ);

        explicit ChunkNeighborhoodSnapshot(int chunkMinWorldY)
            : chunkMinWorldY(chunkMinWorldY)
        {
            blocks.fill(BlockId::Air);
            for (int sampleY = -1; sampleY <= kChunkSizeY; ++sampleY)
            {
                const std::uint8_t defaultLight =
                    (chunkMinWorldY + sampleY < 0) ? packLightLevels(0, 0)
                                                   : packLightLevels(kMaxLightLevel, 0);
                for (int sampleZ = -1; sampleZ <= kChunkSizeZ; ++sampleZ)
                {
                    for (int sampleX = -1; sampleX <= kChunkSizeX; ++sampleX)
                    {
                        lightLevels[index(sampleX, sampleY, sampleZ)] = defaultLight;
                    }
                }
            }
        }

        [[nodiscard]] static constexpr bool contains(int localX, int localY, int localZ) noexcept
        {
            return localX >= -1 && localX <= kChunkSizeX &&
                   localY >= -1 && localY <= kChunkSizeY &&
                   localZ >= -1 && localZ <= kChunkSizeZ;
        }

        [[nodiscard]] static constexpr std::size_t index(int localX, int localY, int localZ) noexcept
        {
            return static_cast<std::size_t>(localY + 1) * static_cast<std::size_t>(kExtentX * kExtentZ) +
                   static_cast<std::size_t>(localZ + 1) * static_cast<std::size_t>(kExtentX) +
                   static_cast<std::size_t>(localX + 1);
        }

        void set(int localX, int localY, int localZ, BlockId block, std::uint8_t packedLight) noexcept
        {
            if (!contains(localX, localY, localZ))
            {
                return;
            }

            const std::size_t voxelIndex = index(localX, localY, localZ);
            blocks[voxelIndex] = block;
            lightLevels[voxelIndex] = packedLight;
        }

        [[nodiscard]] BlockId blockAt(int localX, int localY, int localZ) const noexcept
        {
            if (!contains(localX, localY, localZ))
            {
                return BlockId::Air;
            }

            return blocks[index(localX, localY, localZ)];
        }

        [[nodiscard]] std::uint8_t lightAt(int localX, int localY, int localZ) const noexcept
        {
            if (!contains(localX, localY, localZ))
            {
                return (chunkMinWorldY + localY < 0) ? packLightLevels(0, 0)
                                                     : packLightLevels(kMaxLightLevel, 0);
            }

            return lightLevels[index(localX, localY, localZ)];
        }

        int chunkMinWorldY{0};
        std::array<BlockId, kVoxelCount> blocks{};
        std::array<std::uint8_t, kVoxelCount> lightLevels{};
    };

    struct PooledChunkEntry
    {
        std::shared_ptr<Chunk> chunk{};
        std::size_t retainedBytes{0};
    };

    static bool relightRegionsOverlap(const glm::ivec3& minA,
                                      const glm::ivec3& maxA,
                                      const glm::ivec3& minB,
                                      const glm::ivec3& maxB) noexcept;
    static glm::ivec3 relightRegionAnchor(const PendingRelightBatch& batch) noexcept;
    void recomputePendingRelightBatchBounds(PendingRelightBatch& batch) const noexcept;
    void mergePendingRelightBatch(PendingRelightBatch& dst, PendingRelightBatch&& src) const;
    std::unordered_set<glm::ivec3, ChunkHasher> expandRelightCoords(const RelightCoordGenerationMap& dirtyCoords) const;
    std::uint64_t estimatePendingRelightBatchCost(const PendingRelightBatch& batch) const;
    void markSkyLightColumnDirty(const glm::ivec2& column);
    std::uint64_t currentSkyLightColumnGeneration(const glm::ivec2& column);
    void ensureSkyLightColumnCacheForChunks(const std::vector<std::shared_ptr<Chunk>>& chunks);
    std::uint64_t computeRelightBudgetUnits();
    int computeRelightBatchBudget();
    void resetRelightBudgetForFrame();
    void queueRelightRequest(const glm::ivec3& centerCoord, bool forceRemesh);
    bool takePendingRelightBatch(PendingRelightBatch& batch);
    void releasePendingRelightBatch(const PendingRelightBatch& batch);
    void processPendingRelightRequests(int maxBatches);
    void relightChunkRegion(const PendingRelightBatch& batch);
    ChunkNeighborhoodSnapshot captureChunkNeighborhoodSnapshot(
        const Chunk& chunk,
        const std::vector<BlockId>& centerBlocks,
        const std::vector<std::uint8_t>& centerLightLevels);

    glm::ivec2 atlasTextureSizePixels_{1, 1};
    int atlasTileSizePixels_{kAtlasTileSizePixels};
    int atlasTileStridePixels_{kAtlasTileSizePixels};
    int atlasTilePaddingPixels_{0};
    struct FaceUV
    {
        glm::vec2 base{0.0f};
        glm::vec2 size{1.0f};
    };

    struct BlockUVSet
    {
        std::array<FaceUV, kBlockFaceCount> faces{};
    };

    std::array<BlockUVSet, toIndex(BlockId::Count)> blockUVTable_{};
    bool blockAtlasConfigured_{false};
    RenderDistanceSettings renderSettings_{};
    FarTerrainManager farTerrainManager_{};

    struct StartupStreamingState
    {
        StreamingPhase phase{StreamingPhase::SpawnResolve};
        double phaseTimeSeconds{0.0};
        double totalTimeSeconds{0.0};
        double healthyTimeSeconds{0.0};
        int exactNearCurrentChunks{0};
        int farCurrentBlocks{0};
        bool preloadStarted{false};
        bool playerReleaseReady{false};
        glm::ivec3 spawnChunk{0};
    };

    std::array<std::priority_queue<UploadQueueEntry,
                                   std::vector<UploadQueueEntry>,
                                   UploadQueueEntryComparer>,
               kUploadQueueBucketCount>
        uploadQueues_{};
    std::mutex uploadQueueMutex_;
    std::size_t queuedUploadCount_{0};
    std::size_t initialVisibleUploadCount_{0};
    std::deque<PendingCommitQueueEntry> pendingCommitQueue_{};
    std::mutex pendingCommitQueueMutex_;
    std::vector<ChunkBufferPage> bufferPages_;
    mutable std::mutex bufferPageMutex_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    UploadContext uploadContext_{};

    std::unordered_map<glm::ivec3, std::shared_ptr<Chunk>, ChunkHasher> chunks_;
    mutable std::mutex chunksMutex;
    const glm::vec3 lightDirection_{glm::normalize(glm::vec3(0.5f, -1.0f, 0.2f))};
    JobQueue jobQueue_;
    ColumnManager columnManager_;
    mutable std::mutex predictedColumnMutex_;
    mutable std::unordered_map<glm::ivec2, int, ColumnHasher> predictedColumnHeights_;
    mutable std::mutex columnSlabOccupancyMutex_;
    mutable std::unordered_map<glm::ivec2, ColumnSlabOccupancy, ColumnHasher> columnSlabOccupancyCache_{};
    mutable std::mutex columnHeightPrefetchMutex_;
    mutable std::condition_variable columnHeightPrefetchCondition_;
    mutable std::priority_queue<ColumnHeightPrefetchRequest,
                                std::vector<ColumnHeightPrefetchRequest>,
                                ColumnHeightPrefetchRequestCompare> pendingColumnHeightPrefetchQueue_{};
    mutable std::unordered_map<glm::ivec2, ColumnHeightPrefetchRequestState, ColumnHasher>
        pendingColumnHeightPrefetchRequests_{};
    mutable std::uint64_t nextColumnHeightPrefetchToken_{1};
    mutable std::uint64_t nextColumnHeightPrefetchSequence_{1};
    std::unordered_map<glm::ivec3, std::vector<PendingStructureEdit>, ChunkHasher> pendingStructureEdits_;
    mutable std::mutex pendingStructureMutex_;
    std::unordered_map<glm::ivec3, std::vector<BlockEditOverlayEntry>, ChunkHasher> blockEditOverlays_;
    mutable std::mutex blockEditOverlayMutex_;

    std::vector<std::thread> workerThreads_;
    std::vector<std::thread> columnHeightPrefetchThreads_;
    std::size_t workerThreadCount_{0};
    std::size_t columnHeightPrefetchWorkerCount_{0};
    std::atomic<bool> shouldStop_;

    glm::ivec3 highlightedBlock_{0};
    bool hasHighlight_{false};

    int viewDistance_;
    int targetViewDistance_;
    std::deque<PooledChunkEntry> chunkPool_;
    std::size_t chunkPoolBytes_{0};
    std::size_t chunkPoolBudgetBytes_{kChunkPoolBaseBudgetBytes};
    std::mutex chunkPoolMutex_;
    ProfilingCounters profilingCounters_{};
    mutable ChunkBenchmarkMetrics benchmarkMetrics_{};
    std::vector<PendingRelightBatch> pendingRelightRegions_{};
    RelightCoordGenerationMap pendingRelightCoordGenerations_{};
    RelightCoordGenerationMap activeRelightCoordGenerations_{};
    std::mutex relightStateMutex_;
    std::atomic<int> activeRelightProcessors_{0};
    std::vector<ActiveRelightRegion> activeRelightRegions_{};
    std::uint64_t nextRelightGeneration_{1};
    std::uint64_t nextPendingRelightSequence_{1};
    std::uint64_t relightBudgetUnitsThisFrame_{0};
    std::uint64_t relightBudgetUnitsRemaining_{0};
    int relightBatchBudgetThisFrame_{0};
    int relightBatchBudgetRemaining_{0};
    mutable std::mutex skyLightCacheMutex_;
    std::unordered_map<glm::ivec2, std::uint64_t, ColumnHasher> skyLightColumnGenerations_{};
    std::unordered_map<glm::ivec2, int, ColumnHasher> jobsScheduledThisFrame_{};
    int lastVerticalRadius_{kVerticalStreamingConfig.minRadiusChunks};
    int lastVerticalRadiusDelta_{0};
    int evictionCenterChunkY_{0};
    bool evictionCenterInitialized_{false};
    int uploadColumnLimitThisFrame_{kVerticalStreamingConfig.uploadBasePerColumn};
    int uploadChunkLimitThisFrame_{1};
    std::size_t uploadBudgetBytesThisFrame_{kUploadBudgetBytesPerFrame};
    double uploadBudgetMsThisFrame_{4.0};
    UINT64 nextUploadBatchId_{1};
    ID3D12Fence* renderFence_{nullptr};
    UINT64 renderFenceValue_{0};
    std::deque<DeferredPendingChunkRelease> deferredPendingChunkReleases_{};
    double updateMsLastFrame_{0.0};
    double updateResidualMsLastFrame_{0.0};
    double denseResidencyMsLastFrame_{0.0};
    double verticalRadiusMsLastFrame_{0.0};
    double priorityUpdateMsLastFrame_{0.0};
    double uploadBudgetPrepMsLastFrame_{0.0};
    double missingScanMsLastFrame_{0.0};
    double ensureVolumeMsLastFrame_{0.0};
    double schedulingMsLastFrame_{0.0};
    double evictionMsLastFrame_{0.0};
    double relightMsLastFrame_{0.0};
    std::size_t lastUploadBytesUsed_{0};
    std::size_t pendingUploadsLastFrame_{0};
    double uploadQueueAgeMsLastFrame_{0.0};
    int uploadAttemptsLastFrame_{0};
    int uploadQueueScanEntriesLastFrame_{0};
    int uploadSkippedExpiredLastFrame_{0};
    int uploadSkippedNotReadyLastFrame_{0};
    int uploadSkippedPendingMeshLastFrame_{0};
    int uploadColumnLimitedLastFrame_{0};
    int uploadBudgetDeferredLastFrame_{0};
    int uploadRetryFailuresLastFrame_{0};
    int uploadScanLimitHitsLastFrame_{0};
    int uploadBeginFailuresLastFrame_{0};
    int uploadStalePendingMeshesLastFrame_{0};
    double uploadQueuePickMsLastFrame_{0.0};
    double poolTrimMsLastFrame_{0.0};
    double farTerrainUpdateMsLastFrame_{0.0};
    double columnHeightLookupMsLastFrame_{0.0};
    double columnHeightSampleMsLastFrame_{0.0};
    double uploadPrepareMsLastFrame_{0.0};
    double uploadContextBeginMsLastFrame_{0.0};
    double uploadFinalizeMsLastFrame_{0.0};
    double commitCollectMsLastFrame_{0.0};
    double commitChunkScanMsLastFrame_{0.0};
    double commitMeshLockWaitMsLastFrame_{0.0};
    double commitMeshLockedMsLastFrame_{0.0};
    double commitMeshStateMsLastFrame_{0.0};
    double commitPageStateMsLastFrame_{0.0};
    double commitReleaseMsLastFrame_{0.0};
    double startupStateMsLastFrame_{0.0};
    double benchmarkBookkeepingMsLastFrame_{0.0};
    std::uint64_t nextUploadQueueTicket_{1};
    std::uint64_t nextUploadQueueSequence_{1};
    std::uint64_t nextCommitQueueTicket_{1};
    int generationColumnCapThisFrame_{kVerticalStreamingConfig.maxGenerationJobsPerColumn};
    int lastGenerationBudget_{kVerticalStreamingConfig.generationBudget.baseJobsPerFrame};
    int lastGenerationJobsIssued_{0};
    int lastRingBudget_{kVerticalStreamingConfig.generationBudget.minRingExpansionsPerFrame};
    int lastRingExpansionsUsed_{0};
    int lastMissingChunks_{0};
    int cachedExactReadyChunks_{0};
    int cachedExactRequiredChunks_{0};
    int lastProtectedMissingChunks_{0};
    int lastProtectedReadyChunks_{0};
    int lastProtectedRequiredChunks_{0};
    int lastColumnCap_{kVerticalStreamingConfig.maxGenerationJobsPerColumn};
    int lastBacklogSteps_{0};
    bool startupEnabled_{true};
    StartupStreamingState startupState_{};
    mutable std::mutex schedulingPriorityMutex_;
    glm::ivec3 schedulingPriorityOrigin_{0};
    glm::vec3 schedulingPriorityForward_{0.0f, 0.0f, -1.0f};
    glm::ivec3 lastJobQueuePriorityOrigin_{0};
    glm::vec2 lastJobQueuePriorityForwardXZ_{0.0f, -1.0f};
    std::chrono::steady_clock::time_point lastJobQueuePriorityRefreshTime_{};
    glm::vec3 lastCameraForward_{0.0f, 0.0f, -1.0f};
    glm::ivec3 lastCenterChunk_{0};
    glm::vec2 movementEnvelopeForwardXZ_{0.0f, -1.0f};
    int lastHorizontalMovementShift_{0};
    int lastVerticalMovementShift_{0};
    bool protectedPressureActive_{false};
    bool severeProtectedPressureActive_{false};
    std::chrono::steady_clock::time_point lastUpdateTime_{};
    std::uint64_t updateFrameIndex_{0};
    double smoothedFrameMs_{16.0};
    double lastUploadMsUsed_{0.0};
    int farWorkerCount_{1};
    glm::vec3 lastCameraPosition_{0.0f};
    int lastLoggedGenerationBudget_{-1};
    int lastLoggedRingBudget_{-1};
    int lastLoggedColumnCap_{-1};

    struct RecentEditDebugState
    {
        bool valid{false};
        std::string kind{};
        glm::ivec3 worldPos{0};
        glm::ivec3 chunkCoord{0};
        std::chrono::steady_clock::time_point timestamp{};
    };

    mutable std::mutex recentEditDebugMutex_;
    RecentEditDebugState recentEditDebug_{};
    std::deque<std::string> recentEditDebugEvents_{};
};

inline bool storeFirstBenchmarkTimestamp(std::atomic<long long>& current, std::uint64_t micros) noexcept
{
    long long expected = 0;
    const long long value = static_cast<long long>(micros);
    return current.compare_exchange_strong(expected,
                                           value,
                                           std::memory_order_relaxed,
                                           std::memory_order_relaxed);
}

[[nodiscard]] inline std::uint64_t loadBenchmarkTimestamp(const std::atomic<long long>& current) noexcept
{
    const long long value = current.load(std::memory_order_relaxed);
    return value > 0 ? static_cast<std::uint64_t>(value) : 0u;
}




























// ChunkManager::Impl methods (to be filled)

ChunkManager::Impl::Impl(unsigned seed)
    : worldgenProfile_(terrain::WorldgenProfile::load("assets/worldgen.toml")),
      biomeDatabase_("assets/biomes"),
      globalSeaLevel_(worldgenProfile_.seaLevel),
      noise_(worldgenProfile_.effectiveSeed(seed)),
      shouldStop_(false),
      viewDistance_(renderSettings_.exactChunks),
      targetViewDistance_(renderSettings_.exactChunks)
{
    const unsigned effectiveSeed = worldgenProfile_.effectiveSeed(seed);

    noise_.reseed(effectiveSeed);

    if (biomeDatabase_.biomeCount() == 0)
    {
        throw std::runtime_error("Biome database is empty");
    }

    const auto& climateGeneratorName = worldgenProfile_.climateGenerator;
    std::unique_ptr<terrain::ClimateGenerator> climateGenerator;
    if (climateGeneratorName == "legacy" || climateGeneratorName == "voronoi"
        || climateGeneratorName == "noise_voronoi")
    {
        climateGenerator = std::make_unique<terrain::NoiseVoronoiClimateGenerator>(
            biomeDatabase_, worldgenProfile_, effectiveSeed, kChunkSizeX, kBiomeSizeInChunks);
    }
    else
    {
        throw std::runtime_error("Unsupported climate_generator '" + climateGeneratorName
                                 + "' in assets/worldgen.toml");
    }

    constexpr std::size_t kClimateCacheFragments = 128;
    constexpr std::size_t kSurfaceCacheFragments = 192;

    climateMap_ = std::make_unique<terrain::ClimateMap>(std::move(climateGenerator), kClimateCacheFragments);

    surfaceMap_ = std::make_unique<terrain::SurfaceMap>(
        std::make_unique<terrain::MapGenV1>(biomeDatabase_, *climateMap_, worldgenProfile_, effectiveSeed),
        kSurfaceCacheFragments);

    terrainGenerator_ = std::make_unique<terrain::TerrainGenerator>(
        *climateMap_,
        *surfaceMap_,
        biomeDatabase_,
        globalSeaLevel_,
        [this](int worldX, int worldZ, int slabMin, int slabMax) {
            return this->sampleColumn(worldX, worldZ, slabMin, slabMax);
        });

    structureRegistry_.configure(
        [this](int worldX, int worldZ) {
            return this->sampleColumn(worldX, worldZ);
        },
        [this](int worldX, int worldZ, const ColumnSample& sample) {
            if (!sample.dominantBiome)
            {
                return BlockId::Air;
            }
            const terrain::TerrainColumnBlocks blocks =
                terrain::resolveTerrainColumnBlocks(*sample.dominantBiome, sample, worldX, worldZ, globalSeaLevel_);
            return blocks.surfaceBlock;
        },
        [this](int worldX, int worldZ) noexcept {
            return noise_.fbm(static_cast<float>(worldX) * 0.05f,
                              static_cast<float>(worldZ) * 0.05f,
                              4,
                              0.55f,
                              2.0f);
        });

    gActiveVerticalRadius.store(kVerticalStreamingConfig.minRadiusChunks, std::memory_order_relaxed);
    farTerrainManager_.setEnabled(false);
    farTerrainManager_.setDistanceBlocks(0);
    farTerrainManager_.setFogStartBlocks(renderSettings_.fogStartBlocks);
    farTerrainManager_.setSeaLevel(globalSeaLevel_);
    terrain::FarLodWorldgenTables farLodTables =
        terrain::buildFarLodWorldgenTables(biomeDatabase_, worldgenProfile_, effectiveSeed);
    const auto& treeDensityOffsets = noise_.octaveOffsets();
    for (std::size_t i = 0; i < farLodTables.header.treeDensityOctaveOffsets.size(); ++i)
    {
        farLodTables.header.treeDensityOctaveOffsets[i].x = treeDensityOffsets[i].x;
        farLodTables.header.treeDensityOctaveOffsets[i].y = treeDensityOffsets[i].y;
    }
    farTerrainManager_.setWorldgenTables(farLodTables);
    farTerrainManager_.setStructureFieldSources(
        [this](int worldX, int worldZ)
        {
            return this->sampleColumn(worldX, worldZ);
        },
        [this](int worldX, int worldZ, const ColumnSample& sample)
        {
            if (!sample.dominantBiome)
            {
                return BlockId::Air;
            }
            const terrain::TerrainColumnBlocks blocks =
                terrain::resolveTerrainColumnBlocks(*sample.dominantBiome, sample, worldX, worldZ, globalSeaLevel_);
            return blocks.surfaceBlock;
        },
        [this](int worldX, int worldZ) noexcept
        {
            return noise_.fbm(static_cast<float>(worldX) * 0.05f,
                              static_cast<float>(worldZ) * 0.05f,
                              4,
                              0.55f,
                              2.0f);
        });
    farTerrainManager_.setBenchmarkMetrics(&benchmarkMetrics_);
    const unsigned concurrency = std::max(2u, std::thread::hardware_concurrency());
    farWorkerCount_ = static_cast<int>(std::clamp(concurrency / 3u, 1u, 4u));
    farTerrainManager_.setWorkerCount(static_cast<std::size_t>(farWorkerCount_));
    kFarPlane = computeFarPlaneForDistanceBlocks(chunksToBlocks(renderSettings_.exactChunks));
    startWorkerThreads();
}

ChunkManager::Impl::~Impl()
{
    setRenderSynchronization(nullptr, 0);
    stopWorkerThreads();
    farTerrainManager_.shutdown();
    clear();
    destroyBufferPages();
    uploadContext_.shutdown();
}

void ChunkManager::Impl::initializeRendering(ID3D12Device* device)
{
    device_ = device;
    uploadContext_.initialize(device_.Get());
    farTerrainManager_.setDevice(device_.Get());
    destroyBufferPages();
}

void ChunkManager::Impl::setRenderSynchronization(ID3D12Fence* graphicsFence, std::uint64_t graphicsFenceValue)
{
    renderFence_ = graphicsFence;
    renderFenceValue_ = static_cast<UINT64>(graphicsFenceValue);
    uploadContext_.setGraphicsFenceDependency(nullptr, 0);
    farTerrainManager_.setRenderSynchronization(graphicsFence, static_cast<UINT64>(graphicsFenceValue));
}

ID3D12Fence* ChunkManager::Impl::uploadFence() const noexcept
{
    return nullptr;
}

std::uint64_t ChunkManager::Impl::lastSubmittedUploadFenceValue() const noexcept
{
    return 0;
}

ID3D12Fence* ChunkManager::Impl::farUploadFence() const noexcept
{
    return farTerrainManager_.uploadFence();
}

std::uint64_t ChunkManager::Impl::lastSubmittedFarUploadFenceValue() const noexcept
{
    return static_cast<std::uint64_t>(farTerrainManager_.lastSubmittedUploadFenceValue());
}

void ChunkManager::Impl::setBlockTextureAtlasConfig(const BlockTextureAtlasConfig& config)
{
    if (config.tileSizePixels <= 0 || config.textureSizePixels.x <= 0 || config.textureSizePixels.y <= 0)
    {
        std::cerr << "Invalid block atlas dimensions provided" << std::endl;
        blockAtlasConfigured_ = false;
        return;
    }

    atlasTextureSizePixels_ = config.textureSizePixels;
    atlasTileSizePixels_ = config.tileSizePixels;
    atlasTileStridePixels_ = (config.tileStridePixels > 0) ? config.tileStridePixels : config.tileSizePixels;
    atlasTilePaddingPixels_ = std::max(config.tilePaddingPixels, 0);
    const glm::vec2 atlasTexelScale(
        1.0f / static_cast<float>(atlasTextureSizePixels_.x),
        1.0f / static_cast<float>(atlasTextureSizePixels_.y));

    for (auto& blockEntry : blockUVTable_)
    {
        for (auto& face : blockEntry.faces)
        {
            face.base = glm::vec2(0.0f);
            face.size = glm::vec2(static_cast<float>(atlasTileSizePixels_) * atlasTexelScale.x,
                                  static_cast<float>(atlasTileSizePixels_) * atlasTexelScale.y);
        }
    }

    auto assignFace = [&](BlockId block, BlockFace face, const glm::ivec2& tile)
    {
        const glm::ivec2 tilePixelOrigin(tile.x * atlasTileStridePixels_ + atlasTilePaddingPixels_,
                                         tile.y * atlasTileStridePixels_ + atlasTilePaddingPixels_);
        const glm::vec2 base = glm::vec2(static_cast<float>(tilePixelOrigin.x) * atlasTexelScale.x,
                                         static_cast<float>(tilePixelOrigin.y) * atlasTexelScale.y);
        auto& faceUV = blockUVTable_[toIndex(block)].faces[toIndex(face)];
        faceUV.base = base;
        faceUV.size = glm::vec2(static_cast<float>(atlasTileSizePixels_) * atlasTexelScale.x,
                                static_cast<float>(atlasTileSizePixels_) * atlasTexelScale.y);
    };

    assignFace(BlockId::Grass, BlockFace::Top, {0, 0});
    assignFace(BlockId::Grass, BlockFace::Bottom, {0, 2});
    for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Grass, face, {0, 1});
    }

    assignFace(BlockId::Wood, BlockFace::Top, {0, 4});
    assignFace(BlockId::Wood, BlockFace::Bottom, {0, 4});
    for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Wood, face, {0, 3});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Leaves, face, {0, 5});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Sand, face, {0, 6});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Water, face, {0, 7});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Stone, face, {0, 8});
    }

    assignFace(BlockId::SpruceLog, BlockFace::Top, {0, 9});
    assignFace(BlockId::SpruceLog, BlockFace::Bottom, {0, 9});
    for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::SpruceLog, face, {0, 10});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::SpruceLeaves, face, {0, 11});
    }

    assignFace(BlockId::Podzol, BlockFace::Top, {0, 13});
    assignFace(BlockId::Podzol, BlockFace::Bottom, {0, 2});
    for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Podzol, face, {0, 12});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::DebugLamp, face, {0, 8});
    }

    assignFace(BlockId::DarkOakLog, BlockFace::Top, {0, 14});
    assignFace(BlockId::DarkOakLog, BlockFace::Bottom, {0, 14});
    for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::DarkOakLog, face, {0, 15});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::DarkOakLeaves, face, {0, 16});
    }

    assignFace(BlockId::BirchLog, BlockFace::Top, {0, 17});
    assignFace(BlockId::BirchLog, BlockFace::Bottom, {0, 17});
    for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::BirchLog, face, {0, 18});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::BirchLeaves, face, {0, 19});
    }

    assignFace(BlockId::AcaciaLog, BlockFace::Top, {0, 20});
    assignFace(BlockId::AcaciaLog, BlockFace::Bottom, {0, 20});
    for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::AcaciaLog, face, {0, 21});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::AcaciaLeaves, face, {0, 22});
    }

    blockAtlasConfigured_ = true;
    {
        std::vector<FarTerrainManager::GpuBlockFaceUv> uvTable;
        uvTable.resize(toIndex(BlockId::Count) * kBlockFaceCount);
        for (std::size_t blockIndex = 0; blockIndex < toIndex(BlockId::Count); ++blockIndex)
        {
            const BlockUVSet& uvSet = blockUVTable_[blockIndex];
            for (std::size_t faceIndex = 0; faceIndex < kBlockFaceCount; ++faceIndex)
            {
                const FaceUV& faceUv = uvSet.faces[faceIndex];
                uvTable[blockIndex * kBlockFaceCount + faceIndex] =
                    FarTerrainManager::GpuBlockFaceUv{faceUv.base, faceUv.size};
            }
        }
        farTerrainManager_.setBlockUvTable(uvTable);
    }
}

std::pair<glm::vec2, glm::vec2> ChunkManager::Impl::atlasUvFor(BlockId block, BlockFace face) const
{
    const FaceUV& uv = blockUVTable_[toIndex(block)].faces[toIndex(face)];
    return {uv.base, uv.size};
}

void ChunkManager::Impl::update(const glm::vec3& cameraPos)
{
    update(cameraPos, lastCameraForward_);
}

void ChunkManager::Impl::update(const glm::vec3& cameraPos, const glm::vec3& cameraForward)
{
    const auto updateStart = std::chrono::steady_clock::now();
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();
    lastCameraPosition_ = cameraPos;
    if (glm::dot(cameraForward, cameraForward) > kEpsilon)
    {
        lastCameraForward_ = glm::normalize(cameraForward);
    }

    const glm::ivec3 previousCenterChunk = lastCenterChunk_;

    const auto now = std::chrono::steady_clock::now();
    double frameSeconds = 1.0 / 60.0;
    if (lastUpdateTime_.time_since_epoch().count() != 0)
    {
        frameSeconds = std::chrono::duration<double>(now - lastUpdateTime_).count();
    }
    lastUpdateTime_ = now;
    ++updateFrameIndex_;
    frameSeconds = std::clamp(frameSeconds, 1.0 / 240.0, 0.25);
    smoothedFrameMs_ = smoothedFrameMs_ * 0.90 + frameSeconds * 1000.0 * 0.10;

    const int worldX = static_cast<int>(std::floor(cameraPos.x));
    const int worldY = static_cast<int>(std::floor(cameraPos.y));
    const int worldZ = static_cast<int>(std::floor(cameraPos.z));
    const int clampedWorldY = std::max(worldY, 0);
    const glm::ivec3 centerChunk = worldToChunkCoords(worldX, clampedWorldY, worldZ);
    lastCenterChunk_ = centerChunk;
    updateMovementEnvelopeState(centerChunk, previousCenterChunk);
    glm::vec3 priorityForward(lastCameraForward_.x, lastCameraForward_.y, lastCameraForward_.z);
    priorityForward.x = movementEnvelopeForwardXZ_.x;
    priorityForward.z = movementEnvelopeForwardXZ_.y;
    if (glm::dot(priorityForward, priorityForward) > kEpsilon)
    {
        priorityForward = glm::normalize(priorityForward);
    }
    else
    {
        priorityForward = lastCameraForward_;
    }
    {
        std::lock_guard<std::mutex> lock(schedulingPriorityMutex_);
        schedulingPriorityOrigin_ = centerChunk;
        schedulingPriorityForward_ = priorityForward;
    }

    if (!startupEnabled_ || !startupState_.preloadStarted)
    {
        startupState_.phase = StreamingPhase::SteadyState;
        startupState_.exactNearCurrentChunks = renderSettings_.exactChunks;
        startupState_.farCurrentBlocks = 0;
        startupState_.playerReleaseReady = true;
    }
    else
    {
        if (startupState_.phase == StreamingPhase::SpawnResolve)
        {
            startupState_.phase = StreamingPhase::ExactPreload;
        }
        startupState_.phaseTimeSeconds += frameSeconds;
        startupState_.totalTimeSeconds += frameSeconds;
    }

    targetViewDistance_ = std::clamp(startupState_.exactNearCurrentChunks, 1, renderSettings_.exactChunks);
    const int preloadTargetViewDistance = std::clamp(targetViewDistance_ + hiddenExactPreloadBufferChunks(renderSettings_),
                                                     1,
                                                     kMaxExactRenderDistanceChunks);

    resetColumnBudgets();
    const auto verticalRadiusStart = std::chrono::steady_clock::now();
    const int previousVerticalRadius = lastVerticalRadius_;
    const int verticalRadius = computeVerticalRadius(centerChunk, targetViewDistance_, clampedWorldY);
    verticalRadiusMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - verticalRadiusStart).count();
    lastVerticalRadiusDelta_ = std::abs(verticalRadius - previousVerticalRadius);
    lastVerticalRadius_ = verticalRadius;
    if (benchmarkEnabled)
    {
        benchmarkMetrics_.verticalRadiusDelta.record(static_cast<std::uint64_t>(lastVerticalRadiusDelta_));
    }
    gActiveVerticalRadius.store(verticalRadius, std::memory_order_relaxed);
    const bool lodActive = renderSettings_.totalChunks > renderSettings_.exactChunks;
    const int visibleDistanceBlocks = chunksToBlocks(lodActive ? renderSettings_.totalChunks
                                                               : renderSettings_.exactChunks);
    kFarPlane = computeFarPlaneForDistanceBlocks(visibleDistanceBlocks);

    const glm::vec2 desiredPriorityForwardXZ = normalizePriorityForwardXZ(priorityForward);
    const glm::ivec3 priorityDelta = centerChunk - lastJobQueuePriorityOrigin_;
    const int priorityHorizontalShift = std::max(std::abs(priorityDelta.x), std::abs(priorityDelta.z));
    const int priorityVerticalShift = std::abs(priorityDelta.y);
    const float priorityFacingDot = glm::dot(lastJobQueuePriorityForwardXZ_, desiredPriorityForwardXZ);
    const double priorityRefreshAgeMs = (lastJobQueuePriorityRefreshTime_ == SteadyClock::time_point{})
        ? std::numeric_limits<double>::infinity()
        : std::chrono::duration<double, std::milli>(now - lastJobQueuePriorityRefreshTime_).count();

    constexpr int kJobQueuePriorityShiftThreshold = 2;
    constexpr float kJobQueuePriorityFacingThreshold = 0.85f;
    constexpr double kJobQueuePriorityRefreshIntervalMs = 250.0;

    const bool shouldRefreshJobQueuePriority =
        priorityRefreshAgeMs >= kJobQueuePriorityRefreshIntervalMs ||
        priorityHorizontalShift >= kJobQueuePriorityShiftThreshold ||
        priorityVerticalShift >= kJobQueuePriorityShiftThreshold ||
        priorityFacingDot <= kJobQueuePriorityFacingThreshold;
    priorityUpdateMsLastFrame_ = 0.0;
    if (shouldRefreshJobQueuePriority)
    {
        const auto priorityUpdateStart = std::chrono::steady_clock::now();
        const bool updatedPriority = jobQueue_.tryUpdatePriorityState(centerChunk, priorityForward);
        priorityUpdateMsLastFrame_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - priorityUpdateStart).count();
        if (updatedPriority)
        {
            lastJobQueuePriorityOrigin_ = centerChunk;
            lastJobQueuePriorityForwardXZ_ = desiredPriorityForwardXZ;
            lastJobQueuePriorityRefreshTime_ = now;
        }
    }

    if (viewDistance_ > preloadTargetViewDistance)
    {
        viewDistance_ = preloadTargetViewDistance;
    }

    prefetchVisibleColumnHeights(centerChunk, previousCenterChunk, targetViewDistance_);

    const auto missingScanStart = std::chrono::steady_clock::now();
    const VisibleChunkCoverage visibleCoverage =
        scanVisibleChunkCoverage(centerChunk, targetViewDistance_, verticalRadius);
    const int missingChunks = visibleCoverage.missing;
    missingScanMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - missingScanStart).count();
    lastProtectedMissingChunks_ = visibleCoverage.protectedMissing;
    lastProtectedReadyChunks_ = visibleCoverage.protectedReady;
    lastProtectedRequiredChunks_ = visibleCoverage.protectedRequired;
    protectedPressureActive_ = visibleCoverage.protectedMissing > 0;
    severeProtectedPressureActive_ =
        protectedPressureActive_ &&
        (visibleCoverage.protectedMissing >= 24 ||
         missingChunks >= 96 ||
         (lastHorizontalMovementShift_ > 0 && visibleCoverage.protectedMissing >= 8));

    const auto uploadBudgetStart = std::chrono::steady_clock::now();
    UploadBudgets uploadBudgets = computeUploadBudgets(verticalRadius);
    uploadBudgetPrepMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - uploadBudgetStart).count();
    uploadBudgetBytesThisFrame_ = uploadBudgets.byteBudget;
    uploadColumnLimitThisFrame_ = uploadBudgets.columnLimit;
    uploadChunkLimitThisFrame_ = uploadBudgets.chunkLimit;
    resetRelightBudgetForFrame();
    uploadBudgetMsThisFrame_ = uploadBudgets.timeBudgetMs;
    pendingUploadsLastFrame_ = uploadBudgets.queueSize;

    const int backlogSteps = computeBacklogSteps(missingChunks,
                                                 kVerticalStreamingConfig.generationBudget.backlogStartThreshold,
                                                 kVerticalStreamingConfig.generationBudget.backlogStepSize);
    int columnCap = computeColumnJobCap(backlogSteps, missingChunks);
    if (columnCap <= 0)
    {
        columnCap = std::numeric_limits<int>::max();
    }

    generationColumnCapThisFrame_ = columnCap;

    int generationBudgetTarget =
        computeGenerationBudget(targetViewDistance_, verticalRadius, backlogSteps);
    int ringBudget = computeRingExpansionBudget(missingChunks);
    const bool exactOnly = renderSettings_.totalChunks <= renderSettings_.exactChunks;
    if (exactOnly)
    {
        generationBudgetTarget += 6 + std::max(verticalRadius - kVerticalStreamingConfig.minRadiusChunks, 0);
        ringBudget += 1;
    }
    if (protectedPressureActive_)
    {
        generationBudgetTarget += std::max(8, visibleCoverage.protectedMissing / 8);
        ringBudget += 1;
        if (columnCap < std::numeric_limits<int>::max())
        {
            columnCap += 2;
        }
    }
    if (severeProtectedPressureActive_)
    {
        generationBudgetTarget += std::max(12, visibleCoverage.protectedMissing / 4);
        ringBudget += 2;
        columnCap = std::numeric_limits<int>::max();
    }
    generationColumnCapThisFrame_ = columnCap;

    const std::size_t workerSlots = std::max<std::size_t>(workerThreadCount_, 1);
    const std::size_t outstandingGenerateCap = std::clamp<std::size_t>(
        96u + workerSlots * 32u +
            static_cast<std::size_t>(std::max(verticalRadius, 0)) * 6u +
            (protectedPressureActive_ ? 64u : 0u) +
            (severeProtectedPressureActive_ ? 64u : 0u),
        96u,
        512u);
    const std::size_t outstandingGenerateJobs = jobQueue_.outstanding(JobType::Generate);
    const int generationHeadroom = (outstandingGenerateJobs >= outstandingGenerateCap)
        ? 0
        : static_cast<int>(std::min<std::size_t>(outstandingGenerateCap - outstandingGenerateJobs,
                                                 static_cast<std::size_t>(std::numeric_limits<int>::max())));
    const int effectiveGenerationBudgetTarget = std::min(generationBudgetTarget, generationHeadroom);

    lastGenerationBudget_ = effectiveGenerationBudgetTarget;
    lastRingBudget_ = ringBudget;
    lastColumnCap_ = generationColumnCapThisFrame_;
    lastBacklogSteps_ = backlogSteps;

    int jobBudget = effectiveGenerationBudgetTarget;
    ensureVolumeMsLastFrame_ = 0.0;
    const auto timedEnsureVolume = [&](int horizontalRadius, SchedulingBand band)
    {
        if (!benchmarkEnabled)
        {
            return ensureVolume(centerChunk, horizontalRadius, verticalRadius, jobBudget, band);
        }

        const auto ensureVolumeStart = std::chrono::steady_clock::now();
        RingProgress progress = ensureVolume(centerChunk, horizontalRadius, verticalRadius, jobBudget, band);
        ensureVolumeMsLastFrame_ +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ensureVolumeStart).count();
        return progress;
    };

    const auto schedulingStart = std::chrono::steady_clock::now();
    for (int ring = 0; ring <= viewDistance_ && jobBudget > 0; ++ring)
    {
        RingProgress progress = timedEnsureVolume(ring, SchedulingBand::Protected);
        if (progress.budgetExhausted)
        {
            break;
        }
    }

    int ringsExpanded = 0;
    while (jobBudget > 0 && viewDistance_ < preloadTargetViewDistance && ringsExpanded < ringBudget)
    {
        const int nextRing = viewDistance_ + 1;
        RingProgress progress = timedEnsureVolume(nextRing, SchedulingBand::Protected);

        if (progress.budgetExhausted)
        {
            break;
        }

        if (progress.fullyLoaded)
        {
            ++viewDistance_;
            ++ringsExpanded;
            continue;
        }

        break;
    }

    if (exactOnly && jobBudget > 0)
    {
        const std::size_t lookaheadTrigger =
            std::max<std::size_t>(64u, workerSlots * 16u);
        const std::size_t outstandingGenerateJobsAfterExpansion = jobQueue_.outstanding(JobType::Generate);
        if (outstandingGenerateJobsAfterExpansion <= lookaheadTrigger)
        {
            const int protectedLookaheadRings = severeProtectedPressureActive_ ? 8 :
                                                (protectedPressureActive_ ? 6 : 4);
            const int lookaheadEnd = std::min(preloadTargetViewDistance, viewDistance_ + protectedLookaheadRings);
            for (int ring = viewDistance_ + 1; ring <= lookaheadEnd && jobBudget > 0; ++ring)
            {
                RingProgress progress = timedEnsureVolume(ring, SchedulingBand::Protected);
                if (progress.budgetExhausted)
                {
                    break;
                }
            }
        }
    }

    if (!protectedPressureActive_ && jobBudget > 0)
    {
        for (int ring = 0; ring <= viewDistance_ && jobBudget > 0; ++ring)
        {
            RingProgress progress = timedEnsureVolume(ring, SchedulingBand::Background);
            if (progress.budgetExhausted)
            {
                break;
            }
        }
    }
    schedulingMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - schedulingStart).count();

    lastGenerationJobsIssued_ =
        std::clamp(effectiveGenerationBudgetTarget - jobBudget, 0, effectiveGenerationBudgetTarget);
    lastRingExpansionsUsed_ = ringsExpanded;
    lastMissingChunks_ = std::max(0, missingChunks - lastGenerationJobsIssued_);
    cachedExactReadyChunks_ = visibleCoverage.ready;
    cachedExactRequiredChunks_ = visibleCoverage.required;

    const auto evictionStart = std::chrono::steady_clock::now();
    removeDistantChunks(centerChunk,
                        preloadTargetViewDistance + kVerticalStreamingConfig.horizontalEvictionSlack,
                        verticalRadius);
    evictionMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - evictionStart).count();

    const bool allowMainThreadRelightWindow =
        startupEnabled_ &&
        startupState_.preloadStarted &&
        startupState_.phase == StreamingPhase::ExactPreload;
    const bool allowMainThreadRelightAssist =
        exactOnly &&
        missingChunks > 0 &&
        smoothedFrameMs_ <= 15.5;
    relightMsLastFrame_ = 0.0;
    if (allowMainThreadRelightWindow || allowMainThreadRelightAssist)
    {
        const auto relightStart = std::chrono::steady_clock::now();
        const int activeRelightProcessors = activeRelightProcessors_.load(std::memory_order_acquire);
        const bool allowMainThreadRelightNow =
            workerThreadCount_ == 0 ||
            activeRelightProcessors == 0 ||
            (allowMainThreadRelightAssist && activeRelightProcessors < 2);
        if (allowMainThreadRelightNow)
        {
            processPendingRelightRequests(allowMainThreadRelightAssist && severeProtectedPressureActive_ ? 2 : 1);
        }
        relightMsLastFrame_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - relightStart).count();
    }
    uploadReadyMeshes();
    const auto denseResidencyStart = std::chrono::steady_clock::now();
    updateDenseChunkResidency(centerChunk);
    denseResidencyMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - denseResidencyStart).count();
    const auto poolTrimStart = std::chrono::steady_clock::now();
    trimChunkPoolToBudget();
    poolTrimMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - poolTrimStart).count();

    const auto farTerrainUpdateStart = std::chrono::steady_clock::now();
    if (renderSettings_.totalChunks > renderSettings_.exactChunks)
    {
        const std::size_t exactPendingUploads = estimateUploadQueueSize();
        const int farQueuedTiles = farTerrainManager_.queuedTileCount();
        const int farPendingUploadTiles = farTerrainManager_.pendingUploadTileCount();
        const bool exactUnderPressure = missingChunks > 8 || exactPendingUploads > 8;
        double lodUploadBudgetMs = std::clamp(uploadBudgets.timeBudgetMs * 0.50, 0.50, 2.00);
        if (!exactUnderPressure && (farQueuedTiles > 32 || farPendingUploadTiles > 12))
        {
            // When the far shell is already backlogged, spend more of the frame draining queued work
            // instead of letting completed tiles wait multiple frames before they become resident.
            lodUploadBudgetMs = std::clamp(uploadBudgets.timeBudgetMs * 0.90 + 0.50, 1.25, 4.00);
        }
        else if (farQueuedTiles > 12 || farPendingUploadTiles > 4)
        {
            lodUploadBudgetMs = std::clamp(uploadBudgets.timeBudgetMs * 0.70 + 0.25, 0.75, 2.75);
        }
        farTerrainManager_.setEnabled(true);
        farTerrainManager_.setDistanceBlocks(chunksToBlocks(renderSettings_.totalChunks));
        farTerrainManager_.setSeaLevel(globalSeaLevel_);
        // Keep the far-LOD worker pool stable. Changing the count tears down the workers and
        // clears the queued LOD work, which can prevent the shell from ever filling under
        // fluctuating exact-streaming pressure.
        farTerrainManager_.setWorkerCount(static_cast<std::size_t>(std::max(farWorkerCount_, 1)));
        farTerrainManager_.setBacklogPressure(missingChunks, exactPendingUploads);
        farTerrainManager_.update(centerChunk,
                                  lastCameraForward_,
                                  targetViewDistance_,
                                  chunksToBlocks(renderSettings_.totalChunks),
                                  lodUploadBudgetMs,
                                  [this](int worldX, int worldZ, int slabMinWorldY, int slabMaxWorldY)
                                  {
                                      return sampleColumn(worldX, worldZ, slabMinWorldY, slabMaxWorldY);
                                  });
    }
    else
    {
        farTerrainManager_.setDistanceBlocks(0);
        farTerrainManager_.clear();
    }
    farTerrainUpdateMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - farTerrainUpdateStart).count();

    const auto startupStateStart = std::chrono::steady_clock::now();
    if (startupEnabled_ && startupState_.preloadStarted)
    {
        const bool exactOnlyMode = renderSettings_.totalChunks <= renderSettings_.exactChunks;
        const bool nearReady = missingChunks == 0;
        const bool uploadReady = pendingUploadsLastFrame_ <= 8;
        const bool exactReady = nearReady && uploadReady;
        switch (startupState_.phase)
        {
        case StreamingPhase::ExactPreload:
            startupState_.farCurrentBlocks = 0;
            startupState_.playerReleaseReady = exactReady;
            if (exactReady)
            {
                startupState_.phaseTimeSeconds = 0.0;
                startupState_.healthyTimeSeconds = 0.0;
                startupState_.playerReleaseReady = true;
                if (exactOnlyMode)
                {
                    startupState_.phase = StreamingPhase::SteadyState;
                    startupState_.exactNearCurrentChunks = renderSettings_.exactChunks;
                }
                else
                {
                    startupState_.phase = StreamingPhase::InteractiveNearOnly;
                    startupState_.exactNearCurrentChunks = std::min(renderSettings_.exactChunks, 6);
                }
            }
            else if (startupState_.phaseTimeSeconds >= 2.0 && startupState_.exactNearCurrentChunks > 4)
            {
                --startupState_.exactNearCurrentChunks;
                startupState_.phaseTimeSeconds = 0.0;
            }
            break;
        case StreamingPhase::InteractiveNearOnly:
            startupState_.playerReleaseReady = true;
            startupState_.farCurrentBlocks = 0;
            if (exactReady)
            {
                startupState_.healthyTimeSeconds += frameSeconds;
            }
            else
            {
                startupState_.healthyTimeSeconds = 0.0;
            }

            if (startupState_.healthyTimeSeconds >= 0.75)
            {
                startupState_.healthyTimeSeconds = 0.0;
                if (startupState_.exactNearCurrentChunks < std::min(renderSettings_.exactChunks, 8))
                {
                    startupState_.exactNearCurrentChunks = std::min(renderSettings_.exactChunks, 8);
                }
                else
                {
                    startupState_.phase = StreamingPhase::SteadyState;
                    startupState_.phaseTimeSeconds = 0.0;
                    startupState_.farCurrentBlocks = 0;
                }
            }
            break;
        case StreamingPhase::FarRamp:
            startupState_.playerReleaseReady = true;
            if (startupState_.exactNearCurrentChunks < renderSettings_.exactChunks && exactReady)
            {
                startupState_.healthyTimeSeconds += frameSeconds;
                if (startupState_.healthyTimeSeconds >= 0.75)
                {
                    startupState_.exactNearCurrentChunks = renderSettings_.exactChunks;
                    startupState_.healthyTimeSeconds = 0.0;
                }
            }
            else
            {
                startupState_.phase = StreamingPhase::SteadyState;
                startupState_.phaseTimeSeconds = 0.0;
            }
            break;
        case StreamingPhase::SteadyState:
            startupState_.playerReleaseReady = true;
            startupState_.exactNearCurrentChunks = renderSettings_.exactChunks;
            startupState_.farCurrentBlocks = 0;
            break;
        case StreamingPhase::SpawnResolve:
            startupState_.phase = StreamingPhase::ExactPreload;
            startupState_.phaseTimeSeconds = 0.0;
            startupState_.playerReleaseReady = false;
            startupState_.farCurrentBlocks = 0;
            break;
        }
    }
    startupStateMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startupStateStart).count();

    benchmarkBookkeepingMsLastFrame_ = 0.0;
    if (benchmarkEnabled)
    {
        const auto benchmarkMetricsStart = std::chrono::steady_clock::now();
        benchmarkMetrics_.jobQueueDepth.record(jobQueue_.size());
        benchmarkMetrics_.uploadQueueDepth.record(estimateUploadQueueSize());
        {
            std::lock_guard<std::mutex> prefetchLock(columnHeightPrefetchMutex_);
            benchmarkMetrics_.columnPrefetchQueueDepth.record(
                static_cast<std::uint64_t>(pendingColumnHeightPrefetchRequests_.size()));
        }
        benchmarkMetrics_.farBuildQueueDepth.record(
            static_cast<std::uint64_t>(std::max(farTerrainManager_.buildQueueDepth(), 0)));
        benchmarkMetrics_.farUploadQueueDepth.record(
            static_cast<std::uint64_t>(std::max(farTerrainManager_.pendingUploadTileCount(), 0)));
        benchmarkBookkeepingMsLastFrame_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - benchmarkMetricsStart)
                .count();
    }

    updateMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - updateStart).count();
    const double accountedUpdateMs =
        verticalRadiusMsLastFrame_ +
        priorityUpdateMsLastFrame_ +
        uploadBudgetPrepMsLastFrame_ +
        missingScanMsLastFrame_ +
        schedulingMsLastFrame_ +
        evictionMsLastFrame_ +
        relightMsLastFrame_ +
        lastUploadMsUsed_ +
        uploadPrepareMsLastFrame_ +
        denseResidencyMsLastFrame_ +
        poolTrimMsLastFrame_ +
        farTerrainUpdateMsLastFrame_ +
        startupStateMsLastFrame_ +
        benchmarkBookkeepingMsLastFrame_;
    updateResidualMsLastFrame_ = std::max(0.0, updateMsLastFrame_ - accountedUpdateMs);
    if (benchmarkEnabled)
    {
        const auto toMicros = [](double milliseconds) -> std::uint64_t
        {
            return static_cast<std::uint64_t>(std::max(0.0, milliseconds) * 1000.0);
        };

        benchmarkMetrics_.updateStage.recordMicros(toMicros(updateMsLastFrame_));
        benchmarkMetrics_.updateResidualStage.recordMicros(toMicros(updateResidualMsLastFrame_));
        benchmarkMetrics_.denseResidencyStage.recordMicros(toMicros(denseResidencyMsLastFrame_));
        benchmarkMetrics_.verticalRadiusStage.recordMicros(toMicros(verticalRadiusMsLastFrame_));
        benchmarkMetrics_.priorityUpdateStage.recordMicros(toMicros(priorityUpdateMsLastFrame_));
        benchmarkMetrics_.uploadBudgetPrepStage.recordMicros(toMicros(uploadBudgetPrepMsLastFrame_));
        benchmarkMetrics_.uploadPrepareStage.recordMicros(toMicros(uploadPrepareMsLastFrame_));
        benchmarkMetrics_.uploadContextBeginStage.recordMicros(toMicros(uploadContextBeginMsLastFrame_));
        benchmarkMetrics_.visibleScanStage.recordMicros(toMicros(missingScanMsLastFrame_));
        benchmarkMetrics_.ensureVolumeStage.recordMicros(toMicros(ensureVolumeMsLastFrame_));
        benchmarkMetrics_.schedulingStage.recordMicros(toMicros(schedulingMsLastFrame_));
        benchmarkMetrics_.evictionStage.recordMicros(toMicros(evictionMsLastFrame_));
        benchmarkMetrics_.mainThreadRelightStage.recordMicros(toMicros(relightMsLastFrame_));
        benchmarkMetrics_.uploadDrainStage.recordMicros(toMicros(lastUploadMsUsed_));
        benchmarkMetrics_.uploadQueuePickStage.recordMicros(toMicros(uploadQueuePickMsLastFrame_));
        benchmarkMetrics_.poolTrimStage.recordMicros(toMicros(poolTrimMsLastFrame_));
        benchmarkMetrics_.farTerrainUpdateStage.recordMicros(toMicros(farTerrainUpdateMsLastFrame_));
        benchmarkMetrics_.uploadFinalizeStage.recordMicros(toMicros(uploadFinalizeMsLastFrame_));
        benchmarkMetrics_.commitCollectStage.recordMicros(toMicros(commitCollectMsLastFrame_));
        benchmarkMetrics_.commitChunkScanStage.recordMicros(toMicros(commitChunkScanMsLastFrame_));
        benchmarkMetrics_.commitMeshLockWaitStage.recordMicros(toMicros(commitMeshLockWaitMsLastFrame_));
        benchmarkMetrics_.commitMeshLockedStage.recordMicros(toMicros(commitMeshLockedMsLastFrame_));
        benchmarkMetrics_.commitMeshStateStage.recordMicros(toMicros(commitMeshStateMsLastFrame_));
        benchmarkMetrics_.commitPageStateStage.recordMicros(toMicros(commitPageStateMsLastFrame_));
        benchmarkMetrics_.commitReleaseStage.recordMicros(toMicros(commitReleaseMsLastFrame_));
        benchmarkMetrics_.startupStateStage.recordMicros(toMicros(startupStateMsLastFrame_));
        benchmarkMetrics_.benchmarkBookkeepingStage.recordMicros(toMicros(benchmarkBookkeepingMsLastFrame_));
    }
}

WorldRenderData ChunkManager::Impl::buildRenderData(const Frustum& frustum) const
{
    WorldRenderData renderData;
    renderData.highlightedBlock = highlightedBlock_;
    renderData.hasHighlight = hasHighlight_;
    farTerrainManager_.setVisibility(frustum, lastCameraPosition_);
    const int exactDrawRadiusChunks =
        std::max(targetViewDistance_ + hiddenExactPreloadBufferChunks(renderSettings_), targetViewDistance_);
    const glm::ivec3 cameraChunk = worldToChunkCoords(static_cast<int>(std::floor(lastCameraPosition_.x)),
                                                      0,
                                                      static_cast<int>(std::floor(lastCameraPosition_.z)));

    std::vector<std::pair<glm::ivec3, std::shared_ptr<Chunk>>> snapshot;
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        snapshot.reserve(chunks_.size());
        for (const auto& entry : chunks_)
        {
            snapshot.push_back(entry);
        }
    }

    {
        std::lock_guard<std::mutex> pageLock(bufferPageMutex_);
        const std::size_t pageCount = bufferPages_.size();
        renderData.nearBatches.resize(pageCount);
        for (std::size_t i = 0; i < pageCount; ++i)
        {
            renderData.nearBatches[i].vertexBufferView = bufferPages_[i].vertexView;
            renderData.nearBatches[i].indexBufferView = bufferPages_[i].indexView;
        }
    }

    for (const auto& [coord, chunkPtr] : snapshot)
    {
        if (!chunkPtr)
        {
            continue;
        }

        ChunkState state = chunkPtr->state.load();
        const std::uint32_t indexCount = chunkPtr->indexCount.load(std::memory_order_acquire);
        // A remeshed chunk keeps its previous uploaded allocation until the new mesh is copied to GPU.
        // Keep drawing that old allocation while the chunk sits in Ready awaiting upload, otherwise
        // edits can briefly punch holes in the world for a frame or two.
        if ((state != ChunkState::Uploaded &&
             state != ChunkState::Remeshing &&
             state != ChunkState::Ready) || indexCount == 0)
        {
            continue;
        }

        const glm::vec3 minCorner(static_cast<float>(coord.x * kChunkSizeX),
                                  static_cast<float>(chunkPtr->minWorldY),
                                  static_cast<float>(coord.z * kChunkSizeZ));
        const glm::vec3 maxCorner(static_cast<float>((coord.x + 1) * kChunkSizeX),
                                  static_cast<float>(chunkPtr->maxWorldY + 1),
                                  static_cast<float>((coord.z + 1) * kChunkSizeZ));

        if (!frustum.intersectsAABB(minCorner, maxCorner))
        {
            continue;
        }

        // Keep the newer fog behavior by drawing exact terrain into the hidden outer preload
        // buffer, but make that coverage square instead of radial.
        const int horizontalChunkDistance =
            std::max(std::abs(coord.x - cameraChunk.x), std::abs(coord.z - cameraChunk.z));
        if (horizontalChunkDistance > exactDrawRadiusChunks)
        {
            continue;
        }

        const std::uint32_t pageIndex = chunkPtr->bufferPageIndex.load(std::memory_order_acquire);
        if (pageIndex == kInvalidChunkBufferPage || pageIndex >= renderData.nearBatches.size())
        {
            continue;
        }

        const std::size_t vertexOffset = chunkPtr->vertexOffset.load(std::memory_order_acquire);
        const std::size_t indexOffset = chunkPtr->indexOffset.load(std::memory_order_acquire);
        if (vertexOffset > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
            indexOffset > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            continue;
        }

        ChunkRenderBatch& batch = renderData.nearBatches[pageIndex];
        batch.indexCounts.push_back(indexCount);
        batch.firstIndexLocations.push_back(static_cast<std::uint32_t>(indexOffset));
        batch.baseVertices.push_back(static_cast<std::int32_t>(vertexOffset));
    }

    auto emptyIt = std::remove_if(renderData.nearBatches.begin(),
                                  renderData.nearBatches.end(),
                                  [](const ChunkRenderBatch& batch)
                                  {
                                      return batch.indexCounts.empty();
                                  });
    renderData.nearBatches.erase(emptyIt, renderData.nearBatches.end());
    renderData.farBatches = farTerrainManager_.buildRenderBatches(frustum);

    return renderData;
}

float ChunkManager::Impl::surfaceHeight(float worldX, float worldZ) const noexcept
{
    const int wx = static_cast<int>(std::floor(worldX));
    const int wz = static_cast<int>(std::floor(worldZ));
    const int cachedHeight = columnManager_.highestSolidBlock(wx, wz);
    if (cachedHeight != ColumnManager::kNoHeight)

    {
        return static_cast<float>(cachedHeight + 1);
    }

    const ColumnSample sample = sampleColumn(wx, wz);
    return static_cast<float>(sample.surfaceY + 1);

}

ColumnSample ChunkManager::Impl::sampleColumnAt(const glm::vec3& worldPos,
                                                int slabMinWorldY,
                                                int slabMaxWorldY) const
{
    const int worldX = static_cast<int>(std::floor(worldPos.x));
    const int worldZ = static_cast<int>(std::floor(worldPos.z));
    return sampleColumn(worldX, worldZ, slabMinWorldY, slabMaxWorldY, true);
}

void ChunkManager::Impl::clear()
{
    while (true)
    {
        std::vector<glm::ivec3> coords;
        {
            std::lock_guard<std::mutex> lock(chunksMutex);
            coords.reserve(chunks_.size());
            for (const auto& [coord, chunkPtr] : chunks_)
            {
                coords.push_back(coord);
            }
        }

        if (coords.empty())
        {
            break;
        }

        bool removedAny = false;
        for (const glm::ivec3& coord : coords)
        {
            std::shared_ptr<Chunk> chunk;
            {
                std::lock_guard<std::mutex> lock(chunksMutex);
                auto it = chunks_.find(coord);
                if (it == chunks_.end())
                {
                    continue;
                }

                if (it->second->inFlight.load(std::memory_order_acquire) != 0)
                {
                    continue;
                }

                chunk = it->second;
                chunks_.erase(it);
                removedAny = true;
            }

        if (chunk)
        {
            columnManager_.removeChunk(chunk->coord);
            invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});
            markSkyLightColumnDirty({chunk->coord.x, chunk->coord.z});
            recycleChunkGPU(*chunk);
            recycleChunkObject(std::move(chunk));

        }
    }

        if (!removedAny)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    {
        std::lock_guard<std::mutex> lock(uploadQueueMutex_);
        for (auto& queue : uploadQueues_)
        {
            queue = {};
        }
        queuedUploadCount_ = 0;
        initialVisibleUploadCount_ = 0;
    }
    {
        std::lock_guard<std::mutex> lock(pendingCommitQueueMutex_);
        pendingCommitQueue_.clear();
    }
    uploadContext_.waitForIdle();
    deferredPendingChunkReleases_.clear();
    renderFence_ = nullptr;
    renderFenceValue_ = 0;
    {
        std::lock_guard<std::mutex> lock(bufferPageMutex_);
        for (ChunkBufferPage& page : bufferPages_)
        {
            resetChunkBufferPage(page);
        }
    }
    {
        std::lock_guard<std::mutex> lock(relightStateMutex_);
        pendingRelightRegions_.clear();
        pendingRelightCoordGenerations_.clear();
        activeRelightCoordGenerations_.clear();
        activeRelightRegions_.clear();
        relightBudgetUnitsThisFrame_ = 0;
        relightBudgetUnitsRemaining_ = 0;
        relightBatchBudgetThisFrame_ = 0;
        relightBatchBudgetRemaining_ = 0;
        nextRelightGeneration_ = 1;
        nextPendingRelightSequence_ = 1;
    }
    activeRelightProcessors_.store(0, std::memory_order_release);
    {
        std::lock_guard<std::mutex> cacheLock(skyLightCacheMutex_);
        skyLightColumnGenerations_.clear();
    }
    evictionCenterChunkY_ = 0;
    evictionCenterInitialized_ = false;
    farTerrainManager_.clear();
    columnManager_.clear();
    structureRegistry_.clear();
    {
        std::lock_guard<std::mutex> lock(predictedColumnMutex_);
        predictedColumnHeights_.clear();
    }
    invalidateAllColumnSlabOccupancy();
    {
        std::lock_guard<std::mutex> lock(columnHeightPrefetchMutex_);
        pendingColumnHeightPrefetchQueue_ = {};
        pendingColumnHeightPrefetchRequests_.clear();
        nextColumnHeightPrefetchToken_ = 1;
        nextColumnHeightPrefetchSequence_ = 1;
    }
    {
        std::lock_guard<std::mutex> lock(pendingStructureMutex_);
        pendingStructureEdits_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(blockEditOverlayMutex_);
        blockEditOverlays_.clear();
    }

    uploadBudgetBytesThisFrame_ = kUploadBudgetBytesPerFrame;
    uploadColumnLimitThisFrame_ = kVerticalStreamingConfig.uploadBasePerColumn;
    lastUploadBytesUsed_ = 0;
    pendingUploadsLastFrame_ = 0;
    lastMissingChunks_ = 0;
    cachedExactReadyChunks_ = 0;
    cachedExactRequiredChunks_ = 0;
    lastProtectedMissingChunks_ = 0;
    lastProtectedReadyChunks_ = 0;
    lastProtectedRequiredChunks_ = 0;
    protectedPressureActive_ = false;
    severeProtectedPressureActive_ = false;
    movementEnvelopeForwardXZ_ = glm::vec2{0.0f, -1.0f};
    lastHorizontalMovementShift_ = 0;
    lastVerticalMovementShift_ = 0;
    lastJobQueuePriorityOrigin_ = glm::ivec3{0};
    lastJobQueuePriorityForwardXZ_ = glm::vec2{0.0f, -1.0f};
    lastJobQueuePriorityRefreshTime_ = SteadyClock::time_point{};
    updateFrameIndex_ = 0;

    if (climateMap_)
    {
        climateMap_->clear();
    }

    if (surfaceMap_)
    {
        surfaceMap_->clear();
    }

    {
        std::lock_guard<std::mutex> lock(chunkPoolMutex_);
        trimChunkPoolToBudgetLocked(kChunkPoolMinBudgetBytes);
    }

}

bool ChunkManager::Impl::destroyBlock(const glm::ivec3& worldPos)
{
    const glm::ivec3 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
    auto chunk = getChunkShared(chunkCoord);
    if (!chunk)
    {
        return false;
    }

    ChunkState currentState = chunk->state.load();
    if (currentState != ChunkState::Uploaded && currentState != ChunkState::Remeshing)
    {
        return false;
    }

    if (worldPos.y < chunk->minWorldY || worldPos.y > chunk->maxWorldY)
    {
        return false;
    }
    const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
    const int localY = worldPos.y - chunk->minWorldY;
    const std::size_t blockIdx = blockIndex(local.x, localY, local.z);


    if (!chunk->cpuDataResident && !ensureChunkCpuDataResident(*chunk))
    {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(chunk->meshMutex);
        if (!isSolid(chunk->blocks[blockIdx]))
        {
            return false;
        }

        chunk->blocks[blockIdx] = BlockId::Air;
        chunk->lastDenseFrameTouched = updateFrameIndex_;
        if (chunk->hasBlocks.load(std::memory_order_relaxed))
        {
            chunk->hasBlocks.store(chunkHasSolidBlocks(*chunk), std::memory_order_relaxed);
        }

        columnManager_.updateColumn(makeChunkBlockView(*chunk), local.x, local.z);
    }

    recordBlockEditOverlay(worldPos, BlockId::Air);
    refreshPredictedColumnHeightFromLoadedData({chunk->coord.x, chunk->coord.z});
    markSkyLightColumnDirty({chunk->coord.x, chunk->coord.z});
    // Player edits must always enqueue a geometry refresh immediately; relying on the
    // deferred relight pass alone can leave the old uploaded mesh visible as a ghost block.
    requestChunkRemesh(chunk);
    queueRelightRequest(chunkCoord, true);
    markNeighborsForRemeshingIfNeeded(chunkCoord, local.x, localY, local.z);
    farTerrainManager_.invalidateWorldBlock(worldPos);
    noteRecentEdit("destroy", worldPos, chunkCoord);

    return true;
}

bool ChunkManager::Impl::placeBlock(const glm::ivec3& targetBlockPos, const glm::ivec3& faceNormal, BlockId block)
{
    const glm::ivec3 placePos = targetBlockPos + faceNormal;

    const glm::ivec3 chunkCoord = worldToChunkCoords(placePos.x, placePos.y, placePos.z);
    auto chunk = getChunkShared(chunkCoord);
    if (!chunk)
    {
        return false;
    }

    ChunkState currentState = chunk->state.load();
    if (currentState != ChunkState::Uploaded && currentState != ChunkState::Remeshing)
    {
        return false;
    }

    if (placePos.y < chunk->minWorldY || placePos.y > chunk->maxWorldY)
    {
        return false;
    }
    const glm::ivec3 local = localBlockCoords(placePos, chunkCoord);
    const int localY = placePos.y - chunk->minWorldY;
    const std::size_t blockIdx = blockIndex(local.x, localY, local.z);

    if (!chunk->cpuDataResident && !ensureChunkCpuDataResident(*chunk))
    {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(chunk->meshMutex);
        if (isSolid(chunk->blocks[blockIdx]))
        {
            return false;
        }

        chunk->blocks[blockIdx] = block;
        chunk->lastDenseFrameTouched = updateFrameIndex_;
        chunk->hasBlocks.store(true, std::memory_order_relaxed);

        columnManager_.updateColumn(makeChunkBlockView(*chunk), local.x, local.z);
    }

    recordBlockEditOverlay(placePos, block);
    refreshPredictedColumnHeightFromLoadedData({chunk->coord.x, chunk->coord.z});
    markSkyLightColumnDirty({chunk->coord.x, chunk->coord.z});
    // Keep edit feedback immediate even if lighting catches up on a later relight batch.
    requestChunkRemesh(chunk);
    queueRelightRequest(chunkCoord, true);
    markNeighborsForRemeshingIfNeeded(chunkCoord, local.x, localY, local.z);
    farTerrainManager_.invalidateWorldBlock(placePos);
    noteRecentEdit("place", placePos, chunkCoord);

    return true;
}

RaycastHit ChunkManager::Impl::raycast(const glm::vec3& origin, const glm::vec3& direction) const
{
    RaycastHit result;

    const float dirLengthSq = glm::dot(direction, direction);
    if (dirLengthSq < kEpsilon * kEpsilon)
    {
        return result;
    }

    const glm::vec3 dir = glm::normalize(direction);
    glm::ivec3 currentBlock{
        static_cast<int>(std::floor(origin.x)),
        static_cast<int>(std::floor(origin.y)),
        static_cast<int>(std::floor(origin.z))
    };

    glm::ivec3 stepVec;
    glm::vec3 tMax;
    glm::vec3 tDelta;

    auto initializeAxis = [](float dirComponent, float originComponent, int blockComponent, int& stepOut, float& tMaxOut, float& tDeltaOut)
    {
        if (dirComponent > 0.0f)
        {
            stepOut = 1;
            const float nextBoundary = static_cast<float>(blockComponent + 1);
            tMaxOut = (nextBoundary - originComponent) / dirComponent;
            tDeltaOut = 1.0f / dirComponent;
        }
        else if (dirComponent < 0.0f)
        {
            stepOut = -1;
            const float nextBoundary = static_cast<float>(blockComponent);
            tMaxOut = (nextBoundary - originComponent) / dirComponent;
            tDeltaOut = -1.0f / dirComponent;
        }
        else
        {
            stepOut = 0;
            tMaxOut = std::numeric_limits<float>::infinity();
            tDeltaOut = std::numeric_limits<float>::infinity();
        }
    };

    initializeAxis(dir.x, origin.x, currentBlock.x, stepVec.x, tMax.x, tDelta.x);
    initializeAxis(dir.y, origin.y, currentBlock.y, stepVec.y, tMax.y, tDelta.y);
    initializeAxis(dir.z, origin.z, currentBlock.z, stepVec.z, tMax.z, tDelta.z);

    glm::ivec3 previousBlock = currentBlock;

    while (true)
    {
        int axis = 0;
        if (tMax.y < tMax.x)
        {
            axis = 1;
        }
        if (tMax.z < tMax[axis])
        {
            axis = 2;
        }

        const float nextT = tMax[axis];
        if (nextT > kMaxRayDistance)
        {
            break;
        }

        previousBlock = currentBlock;
        currentBlock[axis] += stepVec[axis];
        tMax[axis] += tDelta[axis];

        if (isSolid(blockAt(currentBlock)))
        {
            result.hit = true;
            result.blockPos = currentBlock;
            result.distance = nextT;
            result.faceNormal = previousBlock - currentBlock;
            break;
        }
    }

    return result;
}

void ChunkManager::Impl::updateHighlight(const glm::vec3& cameraPos, const glm::vec3& cameraDirection)
{
    RaycastHit hit = raycast(cameraPos, cameraDirection);
    if (hit.hit)
    {
        highlightedBlock_ = hit.blockPos;
        hasHighlight_ = true;
    }
    else
    {
        hasHighlight_ = false;
    }
}

void ChunkManager::Impl::toggleViewDistance()
{
    try
    {
        if (targetViewDistance_ == kDefaultNearRenderDistance)
        {
            std::cout << "Switching to extended near render distance..." << std::endl;
            setNearRenderDistance(kMaxUserRenderDistance);
            const long long width = static_cast<long long>(targetViewDistance_) * 2ll + 1ll;
            const long long totalColumns = width * width;
            std::cout << "Extended near render distance target: " << targetViewDistance_ << " chunks (total: "
                      << totalColumns << " chunks)" << std::endl;
        }
        else
        {
            std::cout << "Switching to default near render distance..." << std::endl;
            setNearRenderDistance(kDefaultNearRenderDistance);
            const long long width = static_cast<long long>(targetViewDistance_) * 2ll + 1ll;
            const long long totalColumns = width * width;
            std::cout << "Default near render distance target: " << targetViewDistance_
                      << " chunks (total: " << totalColumns << " chunks)" << std::endl;
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error toggling view distance: " << ex.what() << std::endl;
        targetViewDistance_ = kDefaultNearRenderDistance;
        viewDistance_ = std::min(viewDistance_, targetViewDistance_);
        renderSettings_.exactChunks = targetViewDistance_;
        if (renderSettings_.totalChunks < renderSettings_.exactChunks)
        {
            renderSettings_.totalChunks = renderSettings_.exactChunks;
        }
        kFarPlane = computeFarPlaneForDistanceBlocks(
            chunksToBlocks(std::max(renderSettings_.exactChunks, renderSettings_.totalChunks)));
    }
}

int ChunkManager::Impl::viewDistance() const noexcept
{
    return targetViewDistance_;
}

int ChunkManager::Impl::exactRenderDistanceChunks() const noexcept
{
    return renderSettings_.exactChunks;
}

int ChunkManager::Impl::totalRenderDistanceChunks() const noexcept
{
    return renderSettings_.totalChunks;
}

int ChunkManager::Impl::nearRenderDistance() const noexcept
{
    return exactRenderDistanceChunks();
}

int ChunkManager::Impl::farRenderDistanceBlocks() const noexcept
{
    return chunksToBlocks(renderSettings_.totalChunks);
}

RenderDistanceSettings ChunkManager::Impl::renderDistanceSettings() const noexcept
{
    return renderSettings_;
}

void ChunkManager::Impl::setRenderDistance(int distance) noexcept
{
    setExactRenderDistanceChunks(distance);
}

void ChunkManager::Impl::setExactRenderDistanceChunks(int chunks) noexcept
{
    try
    {
        const int clampedDistance = std::clamp(chunks, 1, kMaxExactRenderDistanceChunks);
        renderSettings_.exactChunks = clampedDistance;
        if (renderSettings_.totalChunks < renderSettings_.exactChunks)
        {
            renderSettings_.totalChunks = renderSettings_.exactChunks;
        }
        if (!startupEnabled_ || !startupState_.preloadStarted || startupState_.phase == StreamingPhase::SteadyState)
        {
            targetViewDistance_ = clampedDistance;
            startupState_.exactNearCurrentChunks = clampedDistance;
        }
        else
        {
            startupState_.exactNearCurrentChunks = std::min(startupState_.exactNearCurrentChunks, clampedDistance);
            targetViewDistance_ = std::min(startupState_.exactNearCurrentChunks, clampedDistance);
        }
        farTerrainManager_.setEnabled(renderSettings_.totalChunks > renderSettings_.exactChunks);
        farTerrainManager_.setDistanceBlocks(chunksToBlocks(renderSettings_.totalChunks));
        kFarPlane = computeFarPlaneForDistanceBlocks(
            chunksToBlocks(std::max(renderSettings_.exactChunks, renderSettings_.totalChunks)));
        if (chunks != clampedDistance)
        {
            std::cout << "Exact render distance request " << chunks << " clamped to " << clampedDistance << " chunks"
                      << std::endl;
        }

        if (viewDistance_ > targetViewDistance_)
        {
            viewDistance_ = targetViewDistance_;
        }

        const long long width = static_cast<long long>(targetViewDistance_) * 2ll + 1ll;
        const long long totalColumns = width * width;
        std::cout << "Exact render distance set to: " << targetViewDistance_ << " chunks (total exact columns: "
                  << totalColumns << ")" << std::endl;
        trimChunkPoolToBudget();
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error setting exact render distance: " << ex.what() << std::endl;
    }
}

void ChunkManager::Impl::setTotalRenderDistanceChunks(int chunks) noexcept
{
    try
    {
        renderSettings_.totalChunks = std::clamp(chunks, 1, kMaxTotalRenderDistanceChunks);
        startupState_.farCurrentBlocks = 0;
        farTerrainManager_.setEnabled(renderSettings_.totalChunks > renderSettings_.exactChunks);
        farTerrainManager_.setDistanceBlocks(chunksToBlocks(renderSettings_.totalChunks));
        if (renderSettings_.totalChunks > renderSettings_.exactChunks)
        {
            std::cout << "LOD render distance active: exact " << renderSettings_.exactChunks
                      << " chunks, total " << renderSettings_.totalChunks << " chunks." << std::endl;
        }
        kFarPlane = computeFarPlaneForDistanceBlocks(
            chunksToBlocks(std::max(renderSettings_.exactChunks, renderSettings_.totalChunks)));
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error setting total render distance: " << ex.what() << std::endl;
    }
}

void ChunkManager::Impl::setNearRenderDistance(int chunks) noexcept
{
    setExactRenderDistanceChunks(chunks);
}

void ChunkManager::Impl::setFarRenderDistanceBlocks(int blocks) noexcept
{
    setTotalRenderDistanceChunks(blocksToChunkRadiusCeil(blocks));
}

void ChunkManager::Impl::setFogStartBlocks(int blocks) noexcept
{
    renderSettings_.fogStartBlocks = std::max(blocks, 0);
    farTerrainManager_.setFogStartBlocks(renderSettings_.fogStartBlocks);
}

void ChunkManager::Impl::setFarTerrainEnabled(bool enabled)
{
    (void)enabled;
    // Legacy far-terrain toggles are obsolete. LOD now follows Exact/Total distance settings.
    startupState_.farCurrentBlocks = 0;
    farTerrainManager_.setEnabled(renderSettings_.totalChunks > renderSettings_.exactChunks);
    farTerrainManager_.setDistanceBlocks(chunksToBlocks(renderSettings_.totalChunks));
    trimChunkPoolToBudget();
    std::cout << "[ChunkManager] Legacy far-terrain toggle is obsolete. LOD follows Exact/Total render distance."
              << std::endl;
}

bool ChunkManager::Impl::farTerrainEnabled() const noexcept
{
    return renderSettings_.totalChunks > renderSettings_.exactChunks;
}

void ChunkManager::Impl::setLodEnabled(bool enabled)
{
    setFarTerrainEnabled(enabled);
}

bool ChunkManager::Impl::lodEnabled() const noexcept
{
    return renderSettings_.totalChunks > renderSettings_.exactChunks;
}

BlockId ChunkManager::Impl::blockAt(const glm::ivec3& worldPos) const noexcept
{
    BlockId overlayBlock = BlockId::Air;
    if (tryGetBlockEditOverlay(worldPos, overlayBlock))
    {
        return overlayBlock;
    }

    const glm::ivec3 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
    auto chunk = getChunkShared(chunkCoord);
    if (!chunk)
    {
        return BlockId::Air;
    }

    if (worldPos.y < chunk->minWorldY || worldPos.y > chunk->maxWorldY)
    {
        return BlockId::Air;
    }
    const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
    if (!chunk->cpuDataResident || chunk->blocks.size() != static_cast<std::size_t>(kChunkBlockCount))
    {
        return BlockId::Air;
    }
    const int localY = worldPos.y - chunk->minWorldY;
    return chunk->blocks[blockIndex(local.x, localY, local.z)];

}

LightSample ChunkManager::Impl::lightAt(const glm::ivec3& worldPos) const noexcept
{
    const std::uint8_t packed = packedLightAtWorld(worldPos);
    return LightSample{skyLightFromPacked(packed), blockLightFromPacked(packed)};
}

glm::vec3 ChunkManager::Impl::findSafeSpawnPosition(float worldX, float worldZ) const
{
    const int baseX = static_cast<int>(std::floor(worldX));
    const int baseZ = static_cast<int>(std::floor(worldZ));
    int highestSolid = columnManager_.highestSolidBlock(baseX, baseZ);

    auto mergeHeight = [](int current, int candidate)
    {
        if (candidate == ColumnManager::kNoHeight)
        {
            return current;
        }
        if (current == ColumnManager::kNoHeight)
        {
            return candidate;
        }
        return std::max(current, candidate);
    };

    const ColumnSample baseSample = sampleColumn(baseX, baseZ);
    auto sampleColumnAt = [&](int worldX, int worldZ) -> ColumnSample
    {
        return sampleColumn(worldX, worldZ);
    };
    auto computeDefaultTreeDensity = [&](int worldX, int worldZ) noexcept
    {
        return noise_.fbm(static_cast<float>(worldX) * 0.05f,
                          static_cast<float>(worldZ) * 0.05f,
                          4,
                          0.55f,
                          2.0f);
    };
    auto resolvedSurfaceBlockAt = [&](int worldX, int worldZ, const ColumnSample& sample) -> BlockId
    {
        if (!sample.dominantBiome)
        {
            return BlockId::Air;
        }
        const terrain::TerrainColumnBlocks blocks =
            terrain::resolveTerrainColumnBlocks(*sample.dominantBiome, sample, worldX, worldZ, globalSeaLevel_);
        return blocks.surfaceBlock;
    };

    auto predictTreeCanopyTop = [&](int originX, int originZ, const ColumnSample& columnSample, int targetX, int targetZ) -> int
    {
        if (!columnSample.dominantBiome || !columnSample.dominantBiome->generatesTrees)
        {
            return ColumnManager::kNoHeight;
        }

        if (columnSample.dominantWeight < kTreeBiomeWeightThreshold)
        {
            return ColumnManager::kNoHeight;
        }

        const int groundWorldY = columnSample.surfaceY;
        if (groundWorldY <= 2)
        {
            return ColumnManager::kNoHeight;
        }

        const BiomeDefinition& biome = *columnSample.dominantBiome;

        if (terrain::isTaigaBiome(biome))
        {
            if (!shouldSpawnTaigaSpruce(biome, originX, groundWorldY, originZ))
            {
                return ColumnManager::kNoHeight;
            }

            int anchorGroundY = std::numeric_limits<int>::min();
            for (int trunkX = 0; trunkX < 2; ++trunkX)
            {
                for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
                {
                    const ColumnSample trunkSample = sampleColumn(originX + trunkX, originZ + trunkZ);
                    if (!trunkSample.dominantBiome || !terrain::isTaigaBiome(*trunkSample.dominantBiome))
                    {
                        return ColumnManager::kNoHeight;
                    }
                    if (trunkSample.dominantWeight < kTreeBiomeWeightThreshold)
                    {
                        return ColumnManager::kNoHeight;
                    }

                    const terrain::TerrainColumnBlocks blocks =
                        terrain::resolveTerrainColumnBlocks(*trunkSample.dominantBiome,
                                                            trunkSample,
                                                            originX + trunkX,
                                                            originZ + trunkZ,
                                                            globalSeaLevel_);
                    if (blocks.surfaceBlock != BlockId::Grass && blocks.surfaceBlock != BlockId::Podzol)
                    {
                        return ColumnManager::kNoHeight;
                    }

                    if (anchorGroundY == std::numeric_limits<int>::min())
                    {
                        anchorGroundY = trunkSample.surfaceY;
                    }
                    else if (trunkSample.surfaceY != anchorGroundY)
                    {
                        return ColumnManager::kNoHeight;
                    }
                }
            }

            for (int dx = -2; dx <= 3; ++dx)
            {
                for (int dz = -2; dz <= 3; ++dz)
                {
                    const ColumnSample neighborSample = sampleColumn(originX + dx, originZ + dz);
                    if (!neighborSample.dominantBiome)
                    {
                        return ColumnManager::kNoHeight;
                    }
                    if (std::abs(neighborSample.surfaceY - anchorGroundY) > 1)
                    {
                        return ColumnManager::kNoHeight;
                    }
                }
            }

            const int trunkHeight = taigaSpruceTrunkHeight(originX, anchorGroundY, originZ);
            const int bareTrunkHeight = taigaSpruceBareTrunkHeight(originX, anchorGroundY, originZ);
            const int canopyBaseWorld = anchorGroundY + bareTrunkHeight + 1;
            const int canopyTopWorld = anchorGroundY + trunkHeight;
            const int totalLayers = std::max(1, canopyTopWorld - canopyBaseWorld + 1);

            int highestCover = ColumnManager::kNoHeight;
            if (targetX >= originX && targetX <= originX + 1 &&
                targetZ >= originZ && targetZ <= originZ + 1)
            {
                highestCover = canopyTopWorld + 1;
            }

            for (int worldY = canopyBaseWorld; worldY <= canopyTopWorld; ++worldY)
            {
                const int layerFromBottom = worldY - canopyBaseWorld;
                const int radius = taigaSpruceLeafRadiusForLayer(layerFromBottom, totalLayers);
                if (taigaSpruceLeafOccupiesCell(originX,
                                               originZ,
                                               targetX,
                                               targetZ,
                                               radius,
                                               layerFromBottom,
                                               totalLayers))
                {
                    highestCover = std::max(highestCover, worldY);
                }
            }

            return highestCover;
        }

        if (biome.id == "dark_forest")
        {
            DarkOakTreeCandidate darkOakCandidate{};
            if (!tryBuildDarkOakCandidate(originX,
                                          originZ,
                                          columnSample,
                                          sampleColumnAt,
                                          resolvedSurfaceBlockAt,
                                          computeDefaultTreeDensity,
                                          darkOakCandidate))
            {
                return ColumnManager::kNoHeight;
            }

            if (darkOakHasSpacingConflict(darkOakCandidate,
                                          sampleColumnAt,
                                          resolvedSurfaceBlockAt,
                                          computeDefaultTreeDensity))
            {
                return ColumnManager::kNoHeight;
            }

            int highestCover = ColumnManager::kNoHeight;
            forEachDarkOakTreeBlock(darkOakCandidate.originX,
                                    darkOakCandidate.originZ,
                                    darkOakCandidate.groundWorldY,
                                    darkOakCandidate.trunkHeight,
                                    BlockId::DarkOakLog,
                                    BlockId::DarkOakLeaves,
                                    [&](int blockX, int blockY, int blockZ, BlockId) {
                                        if (blockX == targetX && blockZ == targetZ)
                                        {
                                            highestCover = std::max(highestCover, blockY);
                                        }
                                        return false;
                                    });
            return highestCover;
        }

        if (biome.id == "savanna")
        {
            AcaciaTreeCandidate acaciaCandidate{};
            if (!tryBuildAcaciaCandidate(originX,
                                         originZ,
                                         columnSample,
                                         sampleColumnAt,
                                         resolvedSurfaceBlockAt,
                                         computeDefaultTreeDensity,
                                         acaciaCandidate))
            {
                return ColumnManager::kNoHeight;
            }

            if (acaciaHasSpacingConflict(acaciaCandidate,
                                         sampleColumnAt,
                                         resolvedSurfaceBlockAt,
                                         computeDefaultTreeDensity))
            {
                return ColumnManager::kNoHeight;
            }

            int highestCover = ColumnManager::kNoHeight;
            forEachAcaciaTreeBlock(acaciaCandidate.originX,
                                   acaciaCandidate.originZ,
                                   acaciaCandidate.groundWorldY,
                                   acaciaCandidate.trunkHeight,
                                   BlockId::AcaciaLog,
                                   BlockId::AcaciaLeaves,
                                   [&](int blockX, int blockY, int blockZ, BlockId) {
                                       if (blockX == targetX && blockZ == targetZ)
                                       {
                                           highestCover = std::max(highestCover, blockY);
                                       }
                                       return false;
                                   });
            return highestCover;
        }

        DefaultTreeCandidate candidate{};
        if (!tryBuildDefaultTreeCandidate(originX,
                                          originZ,
                                          columnSample,
                                          sampleColumnAt,
                                          computeDefaultTreeDensity,
                                          candidate))
        {
            return ColumnManager::kNoHeight;
        }

        if (defaultTreeHasSpacingConflict(candidate, sampleColumnAt, computeDefaultTreeDensity))
        {
            return ColumnManager::kNoHeight;
        }

        int highestCover = ColumnManager::kNoHeight;
        if (targetX == originX && targetZ == originZ)
        {
            highestCover = candidate.groundWorldY + candidate.trunkHeight;
        }

        const int canopyBaseWorld = candidate.groundWorldY + candidate.trunkHeight - 3;
        const int canopyTopWorld = candidate.groundWorldY + candidate.trunkHeight;
        for (int worldY = canopyBaseWorld; worldY <= canopyTopWorld; ++worldY)
        {
            const int layer = worldY - canopyBaseWorld;
            int radius = kDefaultTreeMaxRadius;
            if (worldY >= canopyTopWorld - 1)
            {
                radius = 1;
            }

            for (int dx = -radius; dx <= radius; ++dx)
            {
                for (int dz = -radius; dz <= radius; ++dz)
                {
                    if (std::abs(dx) == radius && std::abs(dz) == radius && radius > 1)
                    {
                        continue;
                    }

                    if (dx == 0 && dz == 0 && worldY <= candidate.groundWorldY + candidate.trunkHeight - 1)
                    {
                        continue;
                    }

                    if (layer == 0 && std::abs(dx) + std::abs(dz) > 3)
                    {
                        continue;
                    }

                    if (originX + dx == targetX && originZ + dz == targetZ)
                    {
                        highestCover = std::max(highestCover, worldY);
                    }
                }
            }
        }

        return highestCover;
    };

    int predictedHighest = ColumnManager::kNoHeight;
    if (baseSample.dominantBiome)
    {
        predictedHighest = mergeHeight(predictedHighest, baseSample.surfaceY);
    }

    for (int originX = baseX - kTaigaSpruceMaxLeafRadius; originX <= baseX + kTaigaSpruceMaxLeafRadius; ++originX)
    {
        for (int originZ = baseZ - kTaigaSpruceMaxLeafRadius; originZ <= baseZ + kTaigaSpruceMaxLeafRadius; ++originZ)
        {
            const ColumnSample originSample =
                (originX == baseX && originZ == baseZ) ? baseSample : sampleColumn(originX, originZ);
            predictedHighest = mergeHeight(predictedHighest,
                                           predictTreeCanopyTop(originX, originZ, originSample, baseX, baseZ));
        }
    }

    highestSolid = mergeHeight(highestSolid, predictedHighest);
    if (highestSolid == ColumnManager::kNoHeight)
    {
        highestSolid = 0;
    }

    const int clearanceHeight = static_cast<int>(std::ceil(kPlayerHeight)) + 2;
    const int spawnFeetY = std::max(highestSolid + 1, baseSample.surfaceY + 2) + clearanceHeight;
    std::cout << "Predicted spawn at height: " << (spawnFeetY + kCameraEyeHeight)
              << " (feet at: " << spawnFeetY << ")" << std::endl;
    const float fallbackY = static_cast<float>(spawnFeetY) + kCameraEyeHeight;
    return glm::vec3(worldX, fallbackY, worldZ);
}

void ChunkManager::Impl::beginSpawnPreload(const glm::vec3& spawnPos)
{
    startupState_ = StartupStreamingState{};
    startupState_.phase = StreamingPhase::ExactPreload;
    startupState_.preloadStarted = true;
    startupState_.spawnChunk = worldToChunkCoords(static_cast<int>(std::floor(spawnPos.x)),
                                                  std::max(static_cast<int>(std::floor(spawnPos.y)), 0),
                                                  static_cast<int>(std::floor(spawnPos.z)));
    startupState_.exactNearCurrentChunks = std::min(renderSettings_.exactChunks, 6);
    startupState_.farCurrentBlocks = 0;
    lastMissingChunks_ = 0;
    cachedExactReadyChunks_ = 0;
    cachedExactRequiredChunks_ = 0;
    lastProtectedMissingChunks_ = 0;
    lastProtectedReadyChunks_ = 0;
    lastProtectedRequiredChunks_ = 0;
    protectedPressureActive_ = false;
    severeProtectedPressureActive_ = false;
    movementEnvelopeForwardXZ_ = normalizePriorityForwardXZ(lastCameraForward_);
    lastHorizontalMovementShift_ = 0;
    lastVerticalMovementShift_ = 0;
    lastJobQueuePriorityOrigin_ = startupState_.spawnChunk;
    lastJobQueuePriorityForwardXZ_ = normalizePriorityForwardXZ(lastCameraForward_);
    lastJobQueuePriorityRefreshTime_ = SteadyClock::time_point{};
    targetViewDistance_ = startupState_.exactNearCurrentChunks;
    if (viewDistance_ > targetViewDistance_)
    {
        viewDistance_ = targetViewDistance_;
    }
    farTerrainManager_.setDistanceBlocks(startupState_.farCurrentBlocks);
    farTerrainManager_.clear();
}

bool ChunkManager::Impl::isSpawnPreloadReady() const noexcept
{
    return !startupEnabled_ || !startupState_.preloadStarted || startupState_.phase != StreamingPhase::ExactPreload;
}

bool ChunkManager::Impl::playerReleaseReady() const noexcept
{
    return !startupEnabled_ || !startupState_.preloadStarted || startupState_.playerReleaseReady;
}

StreamingPhase ChunkManager::Impl::streamingPhase() const noexcept
{
    if (!startupEnabled_ || !startupState_.preloadStarted)
    {
        return StreamingPhase::SteadyState;
    }

    return startupState_.phase;
}

void ChunkManager::Impl::setStartupEnabled(bool enabled) noexcept
{
    startupEnabled_ = enabled;
    if (!startupEnabled_)
    {
        startupState_.phase = StreamingPhase::SteadyState;
        startupState_.playerReleaseReady = true;
        startupState_.exactNearCurrentChunks = renderSettings_.exactChunks;
        startupState_.farCurrentBlocks = 0;
        targetViewDistance_ = renderSettings_.exactChunks;
        farTerrainManager_.setDistanceBlocks(0);
    }
}

bool ChunkManager::Impl::startupEnabled() const noexcept
{
    return startupEnabled_;
}

StreamingStatusSnapshot ChunkManager::Impl::computeStreamingStatusSnapshot() const noexcept
{
    StreamingStatusSnapshot snapshot{};
    snapshot.phase = streamingPhase();
    snapshot.playerReleaseReady = playerReleaseReady();
    snapshot.exactPendingUploads = static_cast<int>(
        std::min<std::size_t>(pendingUploadsLastFrame_, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    snapshot.farActiveTiles = farTerrainManager_.activeTileCount();
    snapshot.farDirtyTiles = farTerrainManager_.dirtyTileCount();
    snapshot.farReadyTiles = farTerrainManager_.readyTileCount();
    snapshot.farQueuedTiles = farTerrainManager_.queuedTileCount();
    snapshot.farPendingUploadTiles = farTerrainManager_.pendingUploadTileCount();

    snapshot.exactReadyChunks = cachedExactReadyChunks_;
    snapshot.exactRequiredChunks = cachedExactRequiredChunks_;

    if (snapshot.playerReleaseReady)
    {
        snapshot.blockingReason = "ready";
    }
    else if (snapshot.exactReadyChunks < snapshot.exactRequiredChunks)
    {
        snapshot.blockingReason = "waiting for exact chunks";
    }
    else if (snapshot.exactPendingUploads > 8)
    {
        snapshot.blockingReason = "waiting for mesh uploads";
    }
    else
    {
        snapshot.blockingReason = "stabilizing preload";
    }

    return snapshot;
}

StreamingStatusSnapshot ChunkManager::Impl::streamingStatusSnapshot() const noexcept
{
    return computeStreamingStatusSnapshot();
}

LodDiagnosticsSnapshot ChunkManager::Impl::lodDiagnosticsSnapshot(const glm::vec3& cameraPos) const
{
    return farTerrainManager_.diagnosticsSnapshot(cameraPos);
}

void ChunkManager::Impl::writeLodDebugSnapshot(const std::filesystem::path& outputPath,
                                               const glm::vec3& cameraPos) const
{
    farTerrainManager_.writeDebugSnapshot(outputPath, cameraPos);
}

ChunkProfilingSnapshot ChunkManager::Impl::sampleProfilingSnapshot()
{
    ChunkProfilingSnapshot snapshot{};
    const StreamingStatusSnapshot status = computeStreamingStatusSnapshot();
    snapshot.phase = status.phase;

    const int generated = profilingCounters_.generatedChunks.exchange(0, std::memory_order_relaxed);
    const int relit = profilingCounters_.relitChunks.exchange(0, std::memory_order_relaxed);
    const int relightBatches = profilingCounters_.relightBatches.exchange(0, std::memory_order_relaxed);
    const int meshed = profilingCounters_.meshedChunks.exchange(0, std::memory_order_relaxed);
    const int uploaded = profilingCounters_.uploadedChunks.exchange(0, std::memory_order_relaxed);
    const std::uint64_t relightRegionChunks =
        profilingCounters_.relightRegionChunks.exchange(0, std::memory_order_relaxed);
    const std::uint64_t relightChangedChunks =
        profilingCounters_.relightChangedChunks.exchange(0, std::memory_order_relaxed);
    const std::uint64_t relightExternalSnapshotChunks =
        profilingCounters_.relightExternalSnapshotChunks.exchange(0, std::memory_order_relaxed);
    const std::uint64_t relightSkyAboveChunkScans =
        profilingCounters_.relightSkyAboveChunkScans.exchange(0, std::memory_order_relaxed);
    const std::uint64_t relightSkySeedNodes =
        profilingCounters_.relightSkySeedNodes.exchange(0, std::memory_order_relaxed);
    const std::uint64_t relightBlockSeedNodes =
        profilingCounters_.relightBlockSeedNodes.exchange(0, std::memory_order_relaxed);
    const std::uint64_t relightSkyNodesProcessed =
        profilingCounters_.relightSkyNodesProcessed.exchange(0, std::memory_order_relaxed);
    const std::uint64_t relightBlockNodesProcessed =
        profilingCounters_.relightBlockNodesProcessed.exchange(0, std::memory_order_relaxed);

    snapshot.generatedChunks = generated;
    snapshot.relitChunks = relit;
    snapshot.relightBatches = relightBatches;
    snapshot.meshedChunks = meshed;
    snapshot.uploadedChunks = uploaded;
    snapshot.uploadedBytes = profilingCounters_.uploadedBytes.exchange(0, std::memory_order_relaxed);
    snapshot.relightRegionChunks = relightRegionChunks;
    snapshot.relightChangedChunks = relightChangedChunks;
    snapshot.relightExternalSnapshotChunks = relightExternalSnapshotChunks;
    snapshot.relightSkyAboveChunkScans = relightSkyAboveChunkScans;
    snapshot.relightSkySeedNodes = relightSkySeedNodes;
    snapshot.relightBlockSeedNodes = relightBlockSeedNodes;
    snapshot.relightSkyNodesProcessed = relightSkyNodesProcessed;
    snapshot.relightBlockNodesProcessed = relightBlockNodesProcessed;
    snapshot.throttledUploads = profilingCounters_.throttledUploads.exchange(0, std::memory_order_relaxed);
    snapshot.deferredUploads = profilingCounters_.deferredUploads.exchange(0, std::memory_order_relaxed);
    snapshot.uploadAttemptsLastFrame = uploadAttemptsLastFrame_;
    snapshot.uploadQueueScanEntriesLastFrame = uploadQueueScanEntriesLastFrame_;
    snapshot.uploadSkippedExpiredLastFrame = uploadSkippedExpiredLastFrame_;
    snapshot.uploadSkippedNotReadyLastFrame = uploadSkippedNotReadyLastFrame_;
    snapshot.uploadSkippedPendingMeshLastFrame = uploadSkippedPendingMeshLastFrame_;
    snapshot.uploadColumnLimitedLastFrame = uploadColumnLimitedLastFrame_;
    snapshot.uploadBudgetDeferredLastFrame = uploadBudgetDeferredLastFrame_;
    snapshot.uploadRetryFailuresLastFrame = uploadRetryFailuresLastFrame_;
    snapshot.uploadScanLimitHitsLastFrame = uploadScanLimitHitsLastFrame_;
    snapshot.uploadBeginFailuresLastFrame = uploadBeginFailuresLastFrame_;
    snapshot.uploadStalePendingMeshesLastFrame = uploadStalePendingMeshesLastFrame_;
    snapshot.evictedChunks = profilingCounters_.evictedChunks.exchange(0, std::memory_order_relaxed);
    snapshot.verticalRadius = lastVerticalRadius_;
    snapshot.verticalRadiusDelta = lastVerticalRadiusDelta_;
    snapshot.generationBudget = lastGenerationBudget_;
    snapshot.generationJobsIssued = lastGenerationJobsIssued_;
    snapshot.ringExpansionBudget = lastRingBudget_;
    snapshot.ringExpansionsUsed = lastRingExpansionsUsed_;
    snapshot.missingChunks = lastMissingChunks_;
    snapshot.generationBacklogSteps = lastBacklogSteps_;
    snapshot.generationColumnCap =
        (lastColumnCap_ >= std::numeric_limits<int>::max()) ? -1 : std::max(lastColumnCap_, 0);
    snapshot.workerThreads = static_cast<int>(workerThreadCount_);

    const long long genMicros = profilingCounters_.generationMicros.exchange(0, std::memory_order_relaxed);
    const long long relightMicros = profilingCounters_.relightMicros.exchange(0, std::memory_order_relaxed);
    const long long meshMicros = profilingCounters_.meshingMicros.exchange(0, std::memory_order_relaxed);

    if (generated > 0)
    {
        snapshot.averageGenerationMs = static_cast<double>(genMicros) /
                                       (1000.0 * static_cast<double>(generated));
    }
    if (meshed > 0)
    {
        snapshot.averageMeshingMs = static_cast<double>(meshMicros) /
                                    (1000.0 * static_cast<double>(meshed));
    }
    if (relit > 0)
    {
        snapshot.averageRelightMs = static_cast<double>(relightMicros) /
                                    (1000.0 * static_cast<double>(relit));
    }

    snapshot.uploadBudgetBytes = uploadBudgetBytesThisFrame_;
    snapshot.uploadedBytesLastFrame = lastUploadBytesUsed_;
    snapshot.uploadColumnLimit = uploadColumnLimitThisFrame_;
    snapshot.updateMsLastFrame = updateMsLastFrame_;
    snapshot.updateResidualMsLastFrame = updateResidualMsLastFrame_;
    snapshot.denseResidencyMsLastFrame = denseResidencyMsLastFrame_;
    snapshot.verticalRadiusMsLastFrame = verticalRadiusMsLastFrame_;
    snapshot.priorityUpdateMsLastFrame = priorityUpdateMsLastFrame_;
    snapshot.uploadBudgetMsLastFrame = uploadBudgetPrepMsLastFrame_;
    snapshot.missingScanMsLastFrame = missingScanMsLastFrame_;
    snapshot.ensureVolumeMsLastFrame = ensureVolumeMsLastFrame_;
    snapshot.schedulingMsLastFrame = schedulingMsLastFrame_;
    snapshot.evictionMsLastFrame = evictionMsLastFrame_;
    snapshot.relightMsLastFrame = relightMsLastFrame_;
    snapshot.uploadMsLastFrame = lastUploadMsUsed_;
    snapshot.uploadQueueAgeMsLastFrame = uploadQueueAgeMsLastFrame_;
    snapshot.uploadQueuePickMsLastFrame = uploadQueuePickMsLastFrame_;
    snapshot.poolTrimMsLastFrame = poolTrimMsLastFrame_;
    snapshot.farTerrainUpdateMsLastFrame = farTerrainUpdateMsLastFrame_;
    snapshot.columnHeightLookupMsLastFrame = columnHeightLookupMsLastFrame_;
    snapshot.columnHeightSampleMsLastFrame = columnHeightSampleMsLastFrame_;
    snapshot.uploadPrepareMsLastFrame = uploadPrepareMsLastFrame_;
    snapshot.uploadContextBeginMsLastFrame = uploadContextBeginMsLastFrame_;
    snapshot.uploadFinalizeMsLastFrame = uploadFinalizeMsLastFrame_;
    snapshot.commitCollectMsLastFrame = commitCollectMsLastFrame_;
    snapshot.commitChunkScanMsLastFrame = commitChunkScanMsLastFrame_;
    snapshot.commitMeshLockWaitMsLastFrame = commitMeshLockWaitMsLastFrame_;
    snapshot.commitMeshLockedMsLastFrame = commitMeshLockedMsLastFrame_;
    snapshot.commitMeshStateMsLastFrame = commitMeshStateMsLastFrame_;
    snapshot.commitPageStateMsLastFrame = commitPageStateMsLastFrame_;
    snapshot.commitReleaseMsLastFrame = commitReleaseMsLastFrame_;
    snapshot.startupStateMsLastFrame = startupStateMsLastFrame_;
    snapshot.benchmarkBookkeepingMsLastFrame = benchmarkBookkeepingMsLastFrame_;
    const std::size_t pendingUploads = pendingUploadsLastFrame_;
    snapshot.pendingUploadChunks = static_cast<int>(
        std::min<std::size_t>(pendingUploads, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    snapshot.jobQueueDepth = static_cast<int>(std::min<std::size_t>(
        jobQueue_.size(),
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    snapshot.uploadQueueDepth = static_cast<int>(std::min<std::size_t>(
        estimateUploadQueueSize(),
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    {
        std::lock_guard<std::mutex> lock(columnHeightPrefetchMutex_);
        snapshot.columnPrefetchQueueDepth = static_cast<int>(std::min<std::size_t>(
            pendingColumnHeightPrefetchRequests_.size(),
            static_cast<std::size_t>(std::numeric_limits<int>::max())));
    }
    {
        std::lock_guard<std::mutex> lock(chunkPoolMutex_);
        snapshot.pooledChunkCount = chunkPool_.size();
        snapshot.pooledChunkBytes = chunkPoolBytes_;
        snapshot.pooledChunkBudgetBytes = chunkPoolBudgetBytes_;
    }
    snapshot.farBuildMsAverage = farTerrainManager_.averageBuildMs();
    snapshot.lodGpuSynthesisMs = farTerrainManager_.averageGpuSynthesisMs();
    snapshot.lodGpuStampMs = farTerrainManager_.averageGpuStampMs();
    snapshot.lodGpuFaceBuildMs = farTerrainManager_.averageGpuFaceBuildMs();
    snapshot.farCollectMsLastFrame = farTerrainManager_.lastCollectMs();
    snapshot.farUploadMsLastFrame = farTerrainManager_.lastUploadMs();
    snapshot.farActiveTiles = farTerrainManager_.activeTileCount();
    snapshot.farDirtyTiles = farTerrainManager_.dirtyTileCount();
    snapshot.farShellTilesReady = farTerrainManager_.readyTileCount();
    snapshot.farTilesBuilt = farTerrainManager_.builtTilesLastUpdate();
    snapshot.farTilesQueued = farTerrainManager_.queuedTileCount();
    snapshot.farTilesPendingUpload = farTerrainManager_.pendingUploadTileCount();
    const StructureRegistryProfilingSnapshot structureSnapshot = structureRegistry_.profilingSnapshot();
    snapshot.structureQueryMs = structureSnapshot.averageQueryMs;
    snapshot.structureCacheHitRate = structureSnapshot.cacheHitRate;
    snapshot.structureRegionsBuilt = structureSnapshot.regionsBuilt;
    snapshot.exactChunksReady = status.exactReadyChunks;
    snapshot.exactChunksPending = std::max(status.exactRequiredChunks - status.exactReadyChunks, 0);

    return snapshot;
}

void ChunkManager::Impl::setBenchmarkMetricsEnabled(bool enabled) noexcept
{
    benchmarkMetrics_.setEnabled(enabled);
    if (climateMap_)
    {
        climateMap_->setProfilingEnabled(enabled);
    }
    if (surfaceMap_)
    {
        surfaceMap_->setProfilingEnabled(enabled);
    }
    structureRegistry_.setProfilingEnabled(enabled);
}

bool ChunkManager::Impl::benchmarkMetricsEnabled() const noexcept
{
    return benchmarkMetrics_.isEnabled();
}

void ChunkManager::Impl::resetBenchmarkMetrics()
{
    benchmarkMetrics_.reset();
    if (climateMap_)
    {
        climateMap_->resetProfiling();
    }
    if (surfaceMap_)
    {
        surfaceMap_->resetProfiling();
    }
    structureRegistry_.resetProfiling();
}

ChunkBenchmarkReport ChunkManager::Impl::benchmarkReport() const
{
    ChunkBenchmarkReport report = benchmarkMetrics_.snapshot();

    if (climateMap_)
    {
        const terrain::ClimateMap::CacheProfilingSnapshot climate = climateMap_->profilingSnapshot();
        report.climateCache.hits = climate.hits;
        report.climateCache.misses = climate.misses;
        report.climateCache.fills = climate.fills;
        const std::uint64_t total = climate.hits + climate.misses;
        if (total > 0)
        {
            report.climateCache.hitRate = static_cast<double>(climate.hits) / static_cast<double>(total);
        }
    }

    if (surfaceMap_)
    {
        const terrain::SurfaceMap::CacheProfilingSnapshot surface = surfaceMap_->profilingSnapshot();
        report.surfaceCache.hits = surface.hits;
        report.surfaceCache.misses = surface.misses;
        report.surfaceCache.fills = surface.fills;
        const std::uint64_t total = surface.hits + surface.misses;
        if (total > 0)
        {
            report.surfaceCache.hitRate = static_cast<double>(surface.hits) / static_cast<double>(total);
        }
    }

    const StructureRegistryProfilingSnapshot structureSnapshot = structureRegistry_.profilingSnapshot();
    report.structureCache.hits = structureSnapshot.cacheHits;
    report.structureCache.misses = structureSnapshot.cacheMisses;
    report.structureCache.fills = structureSnapshot.regionsBuilt;
    report.structureCache.hitRate = structureSnapshot.cacheHitRate;
    report.structureRegionsBuilt = structureSnapshot.regionsBuilt;

    return report;
}

std::string ChunkManager::Impl::biomeNameAt(const glm::vec3& worldPos) const
{
    const int worldX = static_cast<int>(std::floor(worldPos.x));
    const int worldZ = static_cast<int>(std::floor(worldPos.z));
    const ColumnSample sample = sampleColumn(worldX, worldZ);
    if (sample.dominantBiome)
    {
        return sample.dominantBiome->name;
    }

    return "Unknown";
}

void ChunkManager::Impl::startWorkerThreads()
{
    shouldStop_.store(false, std::memory_order_release);

    unsigned concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0)
    {
        concurrency = 2;
    }

    unsigned desired = 1u;
    if (concurrency >= 16)
    {
        desired = 8u;
    }
    else if (concurrency >= 12)
    {
        desired = 6u;
    }
    else if (concurrency >= 8)
    {
        desired = 5u;
    }
    else if (concurrency >= 6)
    {
        desired = 4u;
    }
    else
    {
        desired = std::max(1u, concurrency > 1 ? concurrency - 1 : 1u);
    }

    if (kVerticalStreamingConfig.maxWorkerThreads > 0)
    {
        desired = std::min(desired, static_cast<unsigned>(kVerticalStreamingConfig.maxWorkerThreads));
    }

    workerThreadCount_ = static_cast<std::size_t>(desired);
    jobQueue_.setWorkerConcurrency(workerThreadCount_);
    workerThreads_.reserve(workerThreadCount_);

    for (std::size_t i = 0; i < workerThreadCount_; ++i)
    {
        workerThreads_.emplace_back(&ChunkManager::Impl::workerThreadFunction, this);
    }

    unsigned prefetchDesired = 1u;
    const bool exactOnly = renderSettings_.totalChunks <= renderSettings_.exactChunks;
    if (exactOnly)
    {
        prefetchDesired = std::clamp(std::max(2u, concurrency / 2), 2u, 8u);
    }
    else
    {
        if (concurrency >= 8 && desired >= 4u)
        {
            prefetchDesired = 2u;
        }
        if (concurrency >= 12 && desired >= 5u)
        {
            prefetchDesired = 3u;
        }
        if (concurrency >= 16 && desired >= 6u)
        {
            prefetchDesired = 4u;
        }
        if (concurrency >= 20 && desired >= 7u)
        {
            prefetchDesired = 5u;
        }
    }

    columnHeightPrefetchWorkerCount_ = static_cast<std::size_t>(prefetchDesired);
    columnHeightPrefetchThreads_.reserve(columnHeightPrefetchWorkerCount_);
    for (std::size_t i = 0; i < columnHeightPrefetchWorkerCount_; ++i)
    {
        columnHeightPrefetchThreads_.emplace_back(&ChunkManager::Impl::columnHeightPrefetchThreadFunction, this);
    }
}

void ChunkManager::Impl::stopWorkerThreads()
{
    shouldStop_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(columnHeightPrefetchMutex_);
        pendingColumnHeightPrefetchQueue_ = {};
        pendingColumnHeightPrefetchRequests_.clear();
        nextColumnHeightPrefetchToken_ = 1;
        nextColumnHeightPrefetchSequence_ = 1;
    }
    columnHeightPrefetchCondition_.notify_all();
    std::vector<Job> cancelledJobs = jobQueue_.stop();
    for (const Job& job : cancelledJobs)
    {
        if (job.chunk)
        {
            job.chunk->inFlight.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    for (auto& thread : workerThreads_)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
    workerThreads_.clear();
    for (auto& thread : columnHeightPrefetchThreads_)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
    columnHeightPrefetchThreads_.clear();
    workerThreadCount_ = 0;
    columnHeightPrefetchWorkerCount_ = 0;
}

bool ChunkManager::Impl::acquireNextColumnHeightPrefetch(glm::ivec2& column,
                                                         std::uint64_t& token,
                                                         ColumnHeightPrefetchPriority& priority)
{
    std::unique_lock<std::mutex> lock(columnHeightPrefetchMutex_);
    while (true)
    {
        columnHeightPrefetchCondition_.wait(lock,
                                            [this]
                                            {
                                                return shouldStop_.load(std::memory_order_acquire) ||
                                                       !pendingColumnHeightPrefetchQueue_.empty();
                                            });

        if (shouldStop_.load(std::memory_order_acquire) && pendingColumnHeightPrefetchQueue_.empty())
        {
            return false;
        }

        while (!pendingColumnHeightPrefetchQueue_.empty())
        {
            const ColumnHeightPrefetchRequest request = pendingColumnHeightPrefetchQueue_.top();
            pendingColumnHeightPrefetchQueue_.pop();

            auto requestIt = pendingColumnHeightPrefetchRequests_.find(request.column);
            if (requestIt == pendingColumnHeightPrefetchRequests_.end())
            {
                continue;
            }

            ColumnHeightPrefetchRequestState& state = requestIt->second;
            if (state.inFlight ||
                state.token != request.token ||
                state.priority != request.priority)
            {
                continue;
            }

            state.inFlight = true;
            column = request.column;
            token = request.token;
            priority = request.priority;
            return true;
        }

        if (shouldStop_.load(std::memory_order_acquire))
        {
            return false;
        }
    }
}

void ChunkManager::Impl::finishColumnHeightPrefetch(const glm::ivec2& column, std::uint64_t token) const
{
    std::lock_guard<std::mutex> lock(columnHeightPrefetchMutex_);
    auto requestIt = pendingColumnHeightPrefetchRequests_.find(column);
    if (requestIt != pendingColumnHeightPrefetchRequests_.end() && requestIt->second.token == token)
    {
        pendingColumnHeightPrefetchRequests_.erase(requestIt);
    }
}

void ChunkManager::Impl::workerThreadFunction()
{
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
#endif
    while (true)
    {
        try
        {
            Job job = jobQueue_.waitAndPop();
            struct QueueCompletionGuard
            {
                JobQueue& queue;
                JobType type;

                ~QueueCompletionGuard()
                {
                    queue.jobCompleted(type);
                }
            } queueCompletionGuard{jobQueue_, job.type};
            processJob(job);
            processPendingRelightRequests(1);
        }
        catch (const std::runtime_error&)
        {
            break;
        }
        catch (const std::exception& ex)
        {
            std::cerr << "Worker thread error: " << ex.what() << std::endl;
        }
    }
}

void ChunkManager::Impl::enqueueJob(const std::shared_ptr<Chunk>& chunk,
                                    JobType type,
                                    const glm::ivec3& coord,
                                    std::uint32_t generationEpoch,
                                    bool initialReadyPriority)
{
    if (!chunk)
    {
        return;
    }

    if (shouldStop_.load(std::memory_order_acquire))
    {
        return;
    }

    chunk->inFlight.fetch_add(1, std::memory_order_relaxed);
    if (!jobQueue_.push(Job(type, coord, chunk, generationEpoch, initialReadyPriority)))
    {
        chunk->inFlight.fetch_sub(1, std::memory_order_relaxed);
    }
}

void ChunkManager::Impl::processJob(const Job& job)
{
    std::shared_ptr<Chunk> chunk = job.chunk;
    if (!chunk)
    {
        return;
    }

    struct FlightGuard
    {
        Chunk* chunkPtr;
        explicit FlightGuard(Chunk* ptr) : chunkPtr(ptr) {}
        ~FlightGuard()
        {
            if (chunkPtr)
            {
                chunkPtr->inFlight.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    } guard(chunk.get());

    if (job.type == JobType::Generate)
    {
        if (benchmarkMetrics_.isEnabled())
        {
            storeFirstBenchmarkTimestamp(chunk->generateStartTimestampMicros, steadyMicrosNow());
        }
        const auto generateStart = SteadyClock::now();
        const bool published = generateChunkBlocks(*chunk, job.generationEpoch);
        const auto generateEnd = SteadyClock::now();
        const auto generateMicros =
            std::chrono::duration_cast<std::chrono::microseconds>(generateEnd - generateStart).count();
        profilingCounters_.generationMicros.fetch_add(generateMicros, std::memory_order_relaxed);
        profilingCounters_.generatedChunks.fetch_add(1, std::memory_order_relaxed);
        if (benchmarkMetrics_.isEnabled())
        {
            benchmarkMetrics_.generateStage.recordMicros(static_cast<std::uint64_t>(generateMicros));
            benchmarkMetrics_.generatedChunks.fetch_add(1, std::memory_order_relaxed);
        }

        if (!published)
        {
            return;
        }

        if (benchmarkMetrics_.isEnabled())
        {
            storeFirstBenchmarkTimestamp(chunk->generateDoneTimestampMicros, steadyMicrosNow());
        }

        queueRelightRequest(job.chunkCoord, chunk->hasBlocks.load(std::memory_order_acquire));

        if (chunk->hasBlocks.load(std::memory_order_acquire))
        {
            chunk->pendingMeshRefresh.store(false, std::memory_order_release);
            chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
            if (shouldTrackRecentEditChunk(chunk->coord))
            {
                std::ostringstream stream;
                stream << "generate -> relight queue chunk=(" << chunk->coord.x << ", " << chunk->coord.y << ", " << chunk->coord.z << ")";
                appendRecentEditDebugEvent(stream.str());
            }
        }
        else
        {
            chunk->state.store(ChunkState::Uploaded, std::memory_order_release);
            chunk->meshReady.store(false, std::memory_order_release);
            chunk->indexCount.store(0, std::memory_order_release);
            noteChunkReadyLatency(*chunk);
            if (shouldTrackRecentEditChunk(chunk->coord))
            {
                std::ostringstream stream;
                stream << "generate empty chunk=(" << chunk->coord.x << ", " << chunk->coord.y << ", " << chunk->coord.z << ")";
                appendRecentEditDebugEvent(stream.str());
            }
        }
    }
    else if (job.type == JobType::Mesh)
    {
        if (benchmarkMetrics_.isEnabled())
        {
            storeFirstBenchmarkTimestamp(chunk->meshStartTimestampMicros, steadyMicrosNow());
        }
        const auto meshStart = SteadyClock::now();
        chunk->pendingMeshRefresh.store(false, std::memory_order_release);
        buildChunkMeshAsync(*chunk);
        const auto meshEnd = SteadyClock::now();
        const auto meshMicros =
            std::chrono::duration_cast<std::chrono::microseconds>(meshEnd - meshStart).count();
        chunk->meshVersion.fetch_add(1, std::memory_order_acq_rel);
        profilingCounters_.meshingMicros.fetch_add(meshMicros, std::memory_order_relaxed);
        profilingCounters_.meshedChunks.fetch_add(1, std::memory_order_relaxed);
        if (benchmarkMetrics_.isEnabled())
        {
            benchmarkMetrics_.meshStage.recordMicros(static_cast<std::uint64_t>(meshMicros));
            benchmarkMetrics_.meshedChunks.fetch_add(1, std::memory_order_relaxed);
        }
        if (benchmarkMetrics_.isEnabled())
        {
            storeFirstBenchmarkTimestamp(chunk->meshDoneTimestampMicros, steadyMicrosNow());
        }

        const bool meshEmpty = chunk->meshData.empty();
        if (meshEmpty)
        {
            chunk->state.store(ChunkState::Uploaded, std::memory_order_release);
            noteChunkReadyLatency(*chunk);
        }
        else
        {
            chunk->state.store(ChunkState::Ready, std::memory_order_release);
        }

        if (shouldTrackRecentEditChunk(chunk->coord))
        {
            std::ostringstream stream;
            stream << "mesh done chunk=(" << chunk->coord.x << ", " << chunk->coord.y << ", " << chunk->coord.z
                   << ") empty=" << (meshEmpty ? "yes" : "no")
                   << " pendingRefresh=" << (chunk->pendingMeshRefresh.load(std::memory_order_acquire) ? "yes" : "no");
            appendRecentEditDebugEvent(stream.str());
        }

        if (chunk->pendingMeshRefresh.exchange(false, std::memory_order_acq_rel))
        {
            chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
            enqueueJob(chunk,
                       JobType::Mesh,
                       job.chunkCoord,
                       0,
                       chunkAwaitingInitialVisibleReady(*chunk));
            if (shouldTrackRecentEditChunk(chunk->coord))
            {
                std::ostringstream stream;
                stream << "mesh result superseded chunk=(" << chunk->coord.x << ", " << chunk->coord.y << ", " << chunk->coord.z << ")";
                appendRecentEditDebugEvent(stream.str());
            }
            return;
        }

        if (chunk->meshData.empty())
        {
            recycleChunkGPU(*chunk);
            chunk->meshReady.store(false, std::memory_order_release);
            chunk->indexCount.store(0, std::memory_order_release);
            if (shouldTrackRecentEditChunk(chunk->coord))
            {
                std::ostringstream stream;
                stream << "mesh empty -> recycle GPU chunk=(" << chunk->coord.x << ", " << chunk->coord.y << ", " << chunk->coord.z << ")";
                appendRecentEditDebugEvent(stream.str());
            }
            return;
        }

        if (chunk->state.load(std::memory_order_acquire) == ChunkState::Ready)
        {
            queueChunkForUpload(chunk);
        }
    }
}

ChunkPriorityKey ChunkManager::Impl::buildUploadPriorityKey(const glm::ivec3& coord,
                                                            const glm::ivec3& origin,
                                                            const glm::vec3& forward) const noexcept
{
    const glm::vec2 forwardXZ = normalizePriorityForwardXZ(forward);
    const int dx = coord.x - origin.x;
    const int dy = coord.y - origin.y;
    const int dz = coord.z - origin.z;
    const int horizontalDistance = std::max(std::abs(dx), std::abs(dz));
    const int verticalDistance = std::abs(dy);

    int supportBucket = 3;
    if (horizontalDistance == 0 && verticalDistance <= 8)
    {
        supportBucket = 0;
    }
    else if (horizontalDistance <= 1 && coord.y >= origin.y - 2 && coord.y <= origin.y + 2)
    {
        supportBucket = 1;
    }
    else if (horizontalDistance <= 2 && verticalDistance <= 2)
    {
        supportBucket = 2;
    }

    float facingDot = 1.0f;
    const glm::vec2 delta(static_cast<float>(dx), static_cast<float>(dz));
    if (glm::dot(delta, delta) > kEpsilon)
    {
        facingDot = glm::dot(glm::normalize(delta), forwardXZ);
    }

    int forwardBucket = 2;
    if (facingDot >= 0.5f)
    {
        forwardBucket = 0;
    }
    else if (facingDot >= -0.2f)
    {
        forwardBucket = 1;
    }

    return ChunkPriorityKey{
        supportBucket,
        horizontalDistance,
        forwardBucket,
        verticalDistance,
        std::abs(dx) + verticalDistance + std::abs(dz)};
}

UploadQueueBucket ChunkManager::Impl::classifyUploadQueueBucket(const Chunk& chunk,
                                                                const glm::ivec3& origin,
                                                                const glm::vec3& forward,
                                                                bool retryBucket) const noexcept
{
    if (retryBucket)
    {
        return UploadQueueBucket::Retry;
    }

    if (chunkAwaitingInitialVisibleReady(chunk))
    {
        return UploadQueueBucket::InitialVisible;
    }

    const ChunkPriorityKey priority = buildUploadPriorityKey(chunk.coord, origin, forward);
    const int frontSpan = std::max(6, targetViewDistance_ / 4);
    const bool nearSupport = priority.supportBucket <= 2;
    const bool frontVisible = priority.forwardBucket == 0 &&
                              priority.horizontalDistance <= frontSpan &&
                              priority.verticalDistance <= std::max(4, lastVerticalRadius_ / 2);
    return (nearSupport || frontVisible) ? UploadQueueBucket::NearFrontVisible : UploadQueueBucket::Background;
}

std::shared_ptr<Chunk> ChunkManager::Impl::popNextChunkForUpload()
{
    glm::ivec3 priorityOrigin{0};
    glm::vec3 priorityForward{0.0f, 0.0f, -1.0f};
    {
        std::lock_guard<std::mutex> priorityLock(schedulingPriorityMutex_);
        priorityOrigin = schedulingPriorityOrigin_;
        priorityForward = schedulingPriorityForward_;
    }

    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    for (std::size_t bucketIndex = 0; bucketIndex < uploadQueues_.size(); ++bucketIndex)
    {
        auto& queue = uploadQueues_[bucketIndex];
        while (!queue.empty())
        {
            UploadQueueEntry entry = queue.top();
            queue.pop();
            ++uploadQueueScanEntriesLastFrame_;

            const std::shared_ptr<Chunk> chunk = entry.chunk.lock();
            if (!chunk)
            {
                ++uploadSkippedExpiredLastFrame_;
                continue;
            }

            if (!chunk->queuedForUpload.load(std::memory_order_acquire) ||
                chunk->uploadQueueTicket.load(std::memory_order_acquire) != entry.ticket)
            {
                continue;
            }

            const UploadQueueBucket desiredBucket = classifyUploadQueueBucket(*chunk, priorityOrigin, priorityForward, false);
            if (desiredBucket != static_cast<UploadQueueBucket>(bucketIndex) &&
                desiredBucket != UploadQueueBucket::Retry)
            {
                const std::uint64_t ticket = nextUploadQueueTicket_++;
                chunk->uploadQueueTicket.store(ticket, std::memory_order_release);
                chunk->queuedUploadBucket.store(static_cast<std::uint8_t>(desiredBucket), std::memory_order_release);
                uploadQueues_[uploadQueueBucketIndex(desiredBucket)].push(UploadQueueEntry{
                    chunk,
                    buildUploadPriorityKey(chunk->coord, priorityOrigin, priorityForward),
                    ticket,
                    nextUploadQueueSequence_++});
                continue;
            }

            const std::uint8_t queuedBucket = chunk->queuedUploadBucket.load(std::memory_order_acquire);
            if (queuedUploadCount_ > 0)
            {
                --queuedUploadCount_;
            }
            if (queuedBucket == static_cast<std::uint8_t>(UploadQueueBucket::InitialVisible) &&
                initialVisibleUploadCount_ > 0)
            {
                --initialVisibleUploadCount_;
            }
            chunk->queuedForUpload.store(false, std::memory_order_release);
            chunk->queuedUploadBucket.store(std::numeric_limits<std::uint8_t>::max(), std::memory_order_release);
            chunk->uploadQueueTicket.store(0, std::memory_order_release);
            return chunk;
        }
    }

    return nullptr;
}

void ChunkManager::Impl::queueChunkForUpload(const std::shared_ptr<Chunk>& chunk, bool retryBucket)
{
    if (!chunk)
    {
        return;
    }

    glm::ivec3 priorityOrigin{0};
    glm::vec3 priorityForward{0.0f, 0.0f, -1.0f};
    {
        std::lock_guard<std::mutex> priorityLock(schedulingPriorityMutex_);
        priorityOrigin = schedulingPriorityOrigin_;
        priorityForward = schedulingPriorityForward_;
    }

    const UploadQueueBucket bucket = classifyUploadQueueBucket(*chunk, priorityOrigin, priorityForward, retryBucket);
    const std::uint8_t desiredBucket = static_cast<std::uint8_t>(bucket);

    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    if (chunk->queuedForUpload.load(std::memory_order_acquire))
    {
        const std::uint8_t existingBucket = chunk->queuedUploadBucket.load(std::memory_order_acquire);
        if (existingBucket <= desiredBucket)
        {
            return;
        }

        if (desiredBucket == static_cast<std::uint8_t>(UploadQueueBucket::InitialVisible) &&
            existingBucket != static_cast<std::uint8_t>(UploadQueueBucket::InitialVisible))
        {
            ++initialVisibleUploadCount_;
        }
    }
    else
    {
        ++queuedUploadCount_;
        if (desiredBucket == static_cast<std::uint8_t>(UploadQueueBucket::InitialVisible))
        {
            ++initialVisibleUploadCount_;
        }
    }

    const std::uint64_t ticket = nextUploadQueueTicket_++;
    chunk->queuedForUpload.store(true, std::memory_order_release);
    chunk->queuedUploadBucket.store(desiredBucket, std::memory_order_release);
    chunk->uploadQueueTicket.store(ticket, std::memory_order_release);
    uploadQueues_[uploadQueueBucketIndex(bucket)].push(UploadQueueEntry{
        chunk,
        buildUploadPriorityKey(chunk->coord, priorityOrigin, priorityForward),
        ticket,
        nextUploadQueueSequence_++});
    storeFirstBenchmarkTimestamp(chunk->uploadQueuedTimestampMicros, steadyMicrosNow());

    if (shouldTrackRecentEditChunk(chunk->coord))
    {
        std::ostringstream stream;
        stream << "queue upload chunk=(" << chunk->coord.x << ", " << chunk->coord.y << ", " << chunk->coord.z
               << ") idx=" << chunk->indexCount.load(std::memory_order_acquire)
               << " bucket=" << static_cast<int>(desiredBucket);
        appendRecentEditDebugEvent(stream.str());
    }
}

void ChunkManager::Impl::requeueChunkForUpload(const std::shared_ptr<Chunk>& chunk, bool retryBucket)
{
    queueChunkForUpload(chunk, retryBucket);
}

void ChunkManager::Impl::queueChunkForCommit(const std::shared_ptr<Chunk>& chunk, UINT64 uploadFenceValue)
{
    if (!chunk || uploadFenceValue == 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(pendingCommitQueueMutex_);
    if (chunk->queuedForCommit.load(std::memory_order_acquire))
    {
        return;
    }

    const std::uint64_t ticket = nextCommitQueueTicket_++;
    chunk->queuedForCommit.store(true, std::memory_order_release);
    chunk->commitQueueTicket.store(ticket, std::memory_order_release);
    pendingCommitQueue_.push_back(PendingCommitQueueEntry{chunk, ticket, uploadFenceValue});
}

std::size_t ChunkManager::Impl::nextPowerOfTwo(std::size_t value) noexcept
{
    if (value <= 1)
    {
        return 1;
    }

    value -= 1;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
#if SIZE_MAX > 0xffffffffu
    value |= value >> 32;
#endif
    return value + 1;
}

const char* ChunkManager::Impl::chunkBufferPageStateLabel(ChunkBufferPageState state) noexcept
{
    switch (state)
    {
    case ChunkBufferPageState::Available: return "available";
    case ChunkBufferPageState::PendingOpen: return "pending_open";
    case ChunkBufferPageState::PendingUploaded: return "pending_uploaded";
    case ChunkBufferPageState::Resident: return "resident";
    case ChunkBufferPageState::Retiring: return "retiring";
    default: return "unknown";
    }
}

std::string ChunkManager::Impl::summarizeChunkBufferPagesLocked() const
{
    struct Totals
    {
        std::size_t count{0};
        std::size_t residentChunks{0};
        std::size_t pendingChunks{0};
        std::uint64_t bytes{0};
    };

    Totals available{};
    Totals pendingOpen{};
    Totals pendingUploaded{};
    Totals resident{};
    Totals retiring{};

    auto accumulate = [](Totals& totals, const ChunkBufferPage& page)
    {
        ++totals.count;
        totals.residentChunks += page.residentChunks;
        totals.pendingChunks += page.pendingChunks;
        if (page.vertexBuffer != nullptr)
        {
            totals.bytes += static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex));
        }
        if (page.indexBuffer != nullptr)
        {
            totals.bytes += static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t));
        }
        if (page.vertexUploadBuffer != nullptr)
        {
            totals.bytes += static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex));
        }
        if (page.indexUploadBuffer != nullptr)
        {
            totals.bytes += static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t));
        }
    };

    for (const ChunkBufferPage& page : bufferPages_)
    {
        switch (page.state)
        {
        case ChunkBufferPageState::Available: accumulate(available, page); break;
        case ChunkBufferPageState::PendingOpen: accumulate(pendingOpen, page); break;
        case ChunkBufferPageState::PendingUploaded: accumulate(pendingUploaded, page); break;
        case ChunkBufferPageState::Resident: accumulate(resident, page); break;
        case ChunkBufferPageState::Retiring: accumulate(retiring, page); break;
        default: break;
        }
    }

    auto appendTotals = [](std::ostringstream& stream, const char* label, const Totals& totals)
    {
        stream << ' ' << label
               << "{pages=" << totals.count
               << ",residentChunks=" << totals.residentChunks
               << ",pendingChunks=" << totals.pendingChunks
               << ",mib=" << std::fixed << std::setprecision(2)
               << (static_cast<double>(totals.bytes) / (1024.0 * 1024.0))
               << "}";
    };

    std::ostringstream stream;
    stream << "exact upload pages total=" << bufferPages_.size();
    appendTotals(stream, "available", available);
    appendTotals(stream, "pending_open", pendingOpen);
    appendTotals(stream, "pending_uploaded", pendingUploaded);
    appendTotals(stream, "resident", resident);
    appendTotals(stream, "retiring", retiring);
    return stream.str();
}

void ChunkManager::Impl::ensureChunkBufferPageUploadBuffers(ChunkBufferPage& page)
{
    if (page.vertexUploadBuffer == nullptr)
    {
        page.vertexUploadBuffer = createUploadBuffer(device_.Get(),
                                                     static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex)),
                                                     page.mappedVertexData);
    }

    if (page.indexUploadBuffer == nullptr)
    {
        page.indexUploadBuffer = createUploadBuffer(device_.Get(),
                                                    static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t)),
                                                    page.mappedIndexData);
    }
}

void ChunkManager::Impl::releaseChunkBufferPageUploadBuffers(ChunkBufferPage& page) noexcept
{
    page.vertexUploadBuffer.Reset();
    page.indexUploadBuffer.Reset();
    page.mappedVertexData = nullptr;
    page.mappedIndexData = nullptr;
}

ChunkManager::Impl::ChunkBufferPage ChunkManager::Impl::createBufferPage(std::size_t vertexCount, std::size_t indexCount)
{
    static constexpr std::size_t kDefaultVertexCapacity = 65536;
    static constexpr std::size_t kDefaultIndexCapacity = 98304;

    ChunkBufferPage page;
    page.vertexCapacity = std::max(nextPowerOfTwo(vertexCount), kDefaultVertexCapacity);
    page.indexCapacity = std::max(nextPowerOfTwo(indexCount), kDefaultIndexCapacity);
    page.vertexBuffer = createDefaultBuffer(device_.Get(),
                                            static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex)),
                                            D3D12_RESOURCE_STATE_COMMON);
    page.indexBuffer = createDefaultBuffer(device_.Get(),
                                           static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t)),
                                           D3D12_RESOURCE_STATE_COMMON);
    ensureChunkBufferPageUploadBuffers(page);
    page.vertexView.BufferLocation = page.vertexBuffer ? page.vertexBuffer->GetGPUVirtualAddress() : 0;
    page.vertexView.SizeInBytes = static_cast<UINT>(page.vertexCapacity * sizeof(Vertex));
    page.vertexView.StrideInBytes = sizeof(Vertex);
    page.indexView.BufferLocation = page.indexBuffer ? page.indexBuffer->GetGPUVirtualAddress() : 0;
    page.indexView.SizeInBytes = static_cast<UINT>(page.indexCapacity * sizeof(std::uint32_t));
    page.indexView.Format = DXGI_FORMAT_R32_UINT;
    resetChunkBufferPage(page);

    if (exactUploadDebugLoggingEnabled())
    {
        std::ostringstream stream;
        const std::uint64_t vertexBytes = static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex));
        const std::uint64_t indexBytes = static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t));
        stream << "exact upload page created"
               << " vertexCapacity=" << page.vertexCapacity
               << " indexCapacity=" << page.indexCapacity
               << " totalPageMiB="
               << std::fixed << std::setprecision(2)
               << (static_cast<double>((vertexBytes + indexBytes) * 2ull) / (1024.0 * 1024.0));
        exactUploadDebugLog(stream.str());
    }

    return page;
}

void ChunkManager::Impl::resetChunkBufferPage(ChunkBufferPage& page) noexcept
{
    page.vertexCursor = 0;
    page.indexCursor = 0;
    page.residentChunks = 0;
    page.pendingChunks = 0;
    page.pendingBatchId = 0;
    page.uploadFenceValue = 0;
    page.retireFenceValue = 0;
    page.state = ChunkBufferPageState::Available;
}

ChunkManager::Impl::ChunkAllocation ChunkManager::Impl::acquireChunkAllocation(std::size_t vertexCount,
                                                                               std::size_t indexCount,
                                                                               UINT64 uploadBatchId)
{
    ChunkAllocation allocation{};
    if (vertexCount == 0 || indexCount == 0 || uploadBatchId == 0)
    {
        return allocation;
    }

    auto tryAllocateInPage = [&](ChunkBufferPage& page, std::uint32_t pageIndex) -> bool
    {
        if (page.state == ChunkBufferPageState::Retiring ||
            page.state == ChunkBufferPageState::Resident ||
            page.state == ChunkBufferPageState::PendingUploaded)
        {
            return false;
        }

        if (page.state == ChunkBufferPageState::Available)
        {
            resetChunkBufferPage(page);
            page.state = ChunkBufferPageState::PendingOpen;
            page.pendingBatchId = uploadBatchId;
        }

        if (page.state != ChunkBufferPageState::PendingOpen || page.pendingBatchId != uploadBatchId)
        {
            return false;
        }

        ensureChunkBufferPageUploadBuffers(page);

        if (page.vertexCursor + vertexCount > page.vertexCapacity ||
            page.indexCursor + indexCount > page.indexCapacity)
        {
            return false;
        }

        allocation.pageIndex = pageIndex;
        allocation.vertexOffset = page.vertexCursor;
        allocation.indexOffset = page.indexCursor;
        page.vertexCursor += vertexCount;
        page.indexCursor += indexCount;
        ++page.pendingChunks;
        return true;
    };

    std::lock_guard<std::mutex> lock(bufferPageMutex_);
    for (std::uint32_t pageIndex = 0; pageIndex < bufferPages_.size(); ++pageIndex)
    {
        ChunkBufferPage& page = bufferPages_[pageIndex];
        if (tryAllocateInPage(page, pageIndex))
        {
            return allocation;
        }
    }

    ChunkBufferPage newPage{};
    try
    {
        newPage = createBufferPage(vertexCount, indexCount);
    }
    catch (const std::exception& ex)
    {
        if (exactUploadDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "exact upload page creation failed"
                   << " requestVerts=" << vertexCount
                   << " requestIdx=" << indexCount
                   << " uploadBatchId=" << uploadBatchId
                   << " error=" << ex.what()
                   << " | " << summarizeChunkBufferPagesLocked();
            exactUploadDebugLog(stream.str());
        }
        throw;
    }
    newPage.state = ChunkBufferPageState::PendingOpen;
    newPage.pendingBatchId = uploadBatchId;
    bufferPages_.push_back(std::move(newPage));
    const std::uint32_t newIndex = static_cast<std::uint32_t>(bufferPages_.size() - 1);
    ChunkBufferPage& page = bufferPages_.back();
    const bool allocated = tryAllocateInPage(page, newIndex);
    (void)allocated;
    return allocation;
}

void ChunkManager::Impl::sealPendingChunkUploadPages(UINT64 uploadBatchId, UINT64 uploadFenceValue)
{
    if (uploadBatchId == 0 || uploadFenceValue == 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(bufferPageMutex_);
    for (ChunkBufferPage& page : bufferPages_)
    {
        if (page.state != ChunkBufferPageState::PendingOpen || page.pendingBatchId != uploadBatchId)
        {
            continue;
        }

        page.state = ChunkBufferPageState::PendingUploaded;
        page.pendingBatchId = 0;
        page.uploadFenceValue = uploadFenceValue;
        if (exactUploadDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "exact upload page sealed"
                   << " page=" << (&page - bufferPages_.data())
                   << " uploadFence=" << uploadFenceValue
                   << " residentChunks=" << page.residentChunks
                   << " pendingChunks=" << page.pendingChunks
                   << " state=" << chunkBufferPageStateLabel(page.state);
            exactUploadDebugLog(stream.str());
        }
    }
}

void ChunkManager::Impl::columnHeightPrefetchThreadFunction()
{
#ifdef _WIN32
    // Keep speculative background work below normal, but let visible discovery run at
    // normal priority so exact-only streaming can feed generation continuously.
    int currentThreadPriority = THREAD_PRIORITY_BELOW_NORMAL;
    SetThreadPriority(GetCurrentThread(), currentThreadPriority);
#endif

    while (true)
    {
        glm::ivec2 column{0};
        std::uint64_t token = 0;
        ColumnHeightPrefetchPriority priority = ColumnHeightPrefetchPriority::Normal;
        if (!acquireNextColumnHeightPrefetch(column, token, priority))
        {
            return;
        }

#ifdef _WIN32
        const int desiredThreadPriority = priority >= ColumnHeightPrefetchPriority::Visible
            ? THREAD_PRIORITY_NORMAL
            : THREAD_PRIORITY_BELOW_NORMAL;
        if (desiredThreadPriority != currentThreadPriority)
        {
            SetThreadPriority(GetCurrentThread(), desiredThreadPriority);
            currentThreadPriority = desiredThreadPriority;
        }
#endif

        const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
        const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
        int cachedHeight = ColumnManager::kNoHeight;
        const bool haveHeight = tryGetCachedColumnHeight(column, worldX, worldZ, cachedHeight);
        ColumnSlabOccupancy occupancy{};
        const bool haveOccupancy = tryGetCachedColumnSlabOccupancy(column, occupancy);
        if (haveHeight && haveOccupancy)
        {
            finishColumnHeightPrefetch(column, token);
            continue;
        }

        const bool shouldBuildOccupancy =
            priority >= ColumnHeightPrefetchPriority::Visible;
        ColumnSlabOccupancy resolvedOccupancy = occupancy;
        bool resolvedHaveOccupancy = haveOccupancy;
        if (shouldBuildOccupancy && !haveOccupancy)
        {
            resolvedOccupancy = cachedColumnSlabOccupancy(column);
            resolvedHaveOccupancy = true;
        }
        if (!haveHeight)
        {
            if (resolvedHaveOccupancy && resolvedOccupancy.highestOccupiedChunkY >= 0)
            {
                mergePredictedColumnHeight(column,
                                           resolvedOccupancy.highestOccupiedChunkY * kChunkSizeY + (kChunkSizeY - 1));
            }
            else
            {
                (void)cacheSampledColumnHeight(column, worldX, worldZ);
            }
        }
        finishColumnHeightPrefetch(column, token);
    }
}

void ChunkManager::Impl::collectReusableChunkBufferPages()
{
    const UINT64 completedRenderFenceValue = (renderFence_ != nullptr)
        ? renderFence_->GetCompletedValue()
        : std::numeric_limits<UINT64>::max();

    std::lock_guard<std::mutex> lock(bufferPageMutex_);
    for (ChunkBufferPage& page : bufferPages_)
    {
        if (page.state == ChunkBufferPageState::Retiring && completedRenderFenceValue >= page.retireFenceValue)
        {
            const std::uint32_t pageIndex = static_cast<std::uint32_t>(&page - bufferPages_.data());
            releaseChunkBufferPageUploadBuffers(page);
            resetChunkBufferPage(page);
            if (exactUploadDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "exact upload page recycled"
                       << " page=" << pageIndex
                       << " completedRenderFence=" << completedRenderFenceValue;
                exactUploadDebugLog(stream.str());
            }
        }
    }
}

void ChunkManager::Impl::releaseChunkAllocationRange(std::uint32_t pageIndex,
                                                     std::size_t vertexOffset,
                                                     std::size_t vertexCount,
                                                     std::size_t indexOffset,
                                                     std::size_t indexCount,
                                                     bool residentAllocation)
{
    (void)vertexOffset;
    (void)vertexCount;
    (void)indexOffset;
    (void)indexCount;
    if (pageIndex == kInvalidChunkBufferPage)
    {
        return;
    }

    const UINT64 completedRenderFenceValue = (renderFence_ != nullptr)
        ? renderFence_->GetCompletedValue()
        : std::numeric_limits<UINT64>::max();

    std::lock_guard<std::mutex> lock(bufferPageMutex_);
    if (pageIndex >= bufferPages_.size())
    {
        return;
    }

    ChunkBufferPage& page = bufferPages_[pageIndex];
    if (residentAllocation)
    {
        if (page.residentChunks > 0)
        {
            --page.residentChunks;
        }

        if (page.residentChunks == 0)
        {
            if (page.pendingChunks > 0)
            {
                page.state = ChunkBufferPageState::PendingUploaded;
                if (exactUploadDebugLoggingEnabled())
                {
                    std::ostringstream stream;
                    stream << "exact upload page returned to pending_uploaded"
                           << " page=" << pageIndex
                           << " residentChunks=" << page.residentChunks
                           << " pendingChunks=" << page.pendingChunks;
                    exactUploadDebugLog(stream.str());
                }
            }
            else if (renderFence_ != nullptr &&
                     renderFenceValue_ > 0 &&
                     completedRenderFenceValue < renderFenceValue_)
            {
                page.state = ChunkBufferPageState::Retiring;
                page.retireFenceValue = renderFenceValue_;
                if (exactUploadDebugLoggingEnabled())
                {
                    std::ostringstream stream;
                    stream << "exact upload page retiring"
                           << " page=" << pageIndex
                           << " retireFence=" << page.retireFenceValue;
                    exactUploadDebugLog(stream.str());
                }
            }
            else
            {
                releaseChunkBufferPageUploadBuffers(page);
                resetChunkBufferPage(page);
                if (exactUploadDebugLoggingEnabled())
                {
                    std::ostringstream stream;
                    stream << "exact upload page released to available"
                           << " page=" << pageIndex;
                    exactUploadDebugLog(stream.str());
                }
            }
        }
        return;
    }

    if (page.pendingChunks > 0)
    {
        --page.pendingChunks;
    }

    if (page.pendingChunks == 0)
    {
        if (page.residentChunks > 0)
        {
            page.state = ChunkBufferPageState::Resident;
            page.uploadFenceValue = 0;
            page.pendingBatchId = 0;
            releaseChunkBufferPageUploadBuffers(page);
            if (exactUploadDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "exact upload page resident"
                       << " page=" << pageIndex
                       << " residentChunks=" << page.residentChunks;
                exactUploadDebugLog(stream.str());
            }
        }
        else
        {
            releaseChunkBufferPageUploadBuffers(page);
            resetChunkBufferPage(page);
            if (exactUploadDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "exact upload page pending release completed"
                       << " page=" << pageIndex;
                exactUploadDebugLog(stream.str());
            }
        }
    }
}

void ChunkManager::Impl::collectDeferredPendingChunkReleases()
{
    const UINT64 completedUploadFenceValue = uploadContext_.completedFenceValue();
    std::deque<DeferredPendingChunkRelease> stillPending;
    while (!deferredPendingChunkReleases_.empty())
    {
        DeferredPendingChunkRelease pending = deferredPendingChunkReleases_.front();
        deferredPendingChunkReleases_.pop_front();
        if (pending.uploadFenceValue != 0 && completedUploadFenceValue < pending.uploadFenceValue)
        {
            stillPending.push_back(pending);
            continue;
        }

        releaseChunkAllocationRange(pending.pageIndex,
                                    pending.vertexOffset,
                                    pending.vertexCount,
                                    pending.indexOffset,
                                    pending.indexCount,
                                    false);
    }

    deferredPendingChunkReleases_.swap(stillPending);
}

void ChunkManager::Impl::deferPendingChunkRelease(const Chunk::PendingRenderMesh& pendingMesh)
{
    if (!pendingMesh.valid())
    {
        return;
    }

    deferredPendingChunkReleases_.push_back(DeferredPendingChunkRelease{
        pendingMesh.pageIndex,
        pendingMesh.vertexOffset,
        pendingMesh.indexOffset,
        pendingMesh.vertexCount,
        pendingMesh.indexCount,
        pendingMesh.uploadFenceValue});
}

void ChunkManager::Impl::releaseChunkAllocation(Chunk& chunk)
{
    const std::uint32_t pageIndex = chunk.bufferPageIndex.load(std::memory_order_acquire);
    if (pageIndex == kInvalidChunkBufferPage)
    {
        chunk.vertexCount.store(0, std::memory_order_relaxed);
        chunk.indexCount.store(0, std::memory_order_relaxed);
        chunk.vertexOffset.store(0, std::memory_order_relaxed);
        chunk.indexOffset.store(0, std::memory_order_relaxed);
        return;
    }

    const std::size_t vertexCount = chunk.vertexCount.load(std::memory_order_acquire);
    const std::size_t indexCount = static_cast<std::size_t>(chunk.indexCount.load(std::memory_order_acquire));
    const std::size_t vertexOffset = chunk.vertexOffset.load(std::memory_order_acquire);
    const std::size_t indexOffset = chunk.indexOffset.load(std::memory_order_acquire);

    chunk.bufferPageIndex.store(kInvalidChunkBufferPage, std::memory_order_release);
    chunk.vertexCount.store(0, std::memory_order_release);
    chunk.indexCount.store(0, std::memory_order_release);
    chunk.vertexOffset.store(0, std::memory_order_release);
    chunk.indexOffset.store(0, std::memory_order_release);
    releaseChunkAllocationRange(pageIndex, vertexOffset, vertexCount, indexOffset, indexCount, true);
}

void ChunkManager::Impl::recycleChunkGPU(Chunk& chunk)
{
    Chunk::PendingRenderMesh pendingMesh;
    bool wasQueuedForUpload = false;
    std::uint8_t queuedBucket = std::numeric_limits<std::uint8_t>::max();
    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        releaseChunkAllocation(chunk);
        pendingMesh = chunk.pendingMesh;
        wasQueuedForUpload = chunk.queuedForUpload.load(std::memory_order_acquire);
        queuedBucket = chunk.queuedUploadBucket.load(std::memory_order_acquire);
        chunk.pendingMesh = {};
        chunk.meshData.clear();
        chunk.meshReady.store(false, std::memory_order_release);
        chunk.queuedForUpload.store(false, std::memory_order_release);
        chunk.queuedUploadBucket.store(std::numeric_limits<std::uint8_t>::max(), std::memory_order_release);
        chunk.uploadQueueTicket.store(0, std::memory_order_release);
        chunk.queuedForCommit.store(false, std::memory_order_release);
        chunk.commitQueueTicket.store(0, std::memory_order_release);
    }

    if (wasQueuedForUpload)
    {
        std::lock_guard<std::mutex> uploadLock(uploadQueueMutex_);
        if (queuedUploadCount_ > 0)
        {
            --queuedUploadCount_;
        }
        if (queuedBucket == static_cast<std::uint8_t>(UploadQueueBucket::InitialVisible) &&
            initialVisibleUploadCount_ > 0)
        {
            --initialVisibleUploadCount_;
        }
    }

    if (!pendingMesh.valid())
    {
        return;
    }

    const UINT64 completedUploadFenceValue = uploadContext_.completedFenceValue();
    if (pendingMesh.uploadFenceValue != 0 && completedUploadFenceValue < pendingMesh.uploadFenceValue)
    {
        deferPendingChunkRelease(pendingMesh);
        return;
    }

    releaseChunkAllocationRange(pendingMesh.pageIndex,
                                pendingMesh.vertexOffset,
                                pendingMesh.vertexCount,
                                pendingMesh.indexOffset,
                                pendingMesh.indexCount,
                                false);
}

std::size_t ChunkManager::Impl::estimateChunkRetainedBytes(const Chunk& chunk) noexcept
{
    return sizeof(Chunk) +
           chunk.blocks.capacity() * sizeof(BlockId) +
           chunk.lightLevels.capacity() * sizeof(std::uint8_t) +
           chunk.meshData.retainedBytes();
}

std::size_t ChunkManager::Impl::chunkPoolBudgetBytes() const noexcept
{
    const double horizontalPressure =
        std::max(1.0, static_cast<double>(std::max(targetViewDistance_, 1)) /
                          static_cast<double>(kDefaultNearRenderDistance));
    const double verticalPressure =
        std::max(1.0, static_cast<double>(std::max(lastVerticalRadius_, kVerticalStreamingConfig.minRadiusChunks)) /
                          static_cast<double>(std::max(kVerticalStreamingConfig.minRadiusChunks, 1)));
    const double uploadPressure =
        1.0 + std::min(kChunkPoolMaxUploadPressure,
                       static_cast<double>(pendingUploadsLastFrame_) / kChunkPoolUploadPressureDivisor);

    double pressure = horizontalPressure * std::sqrt(verticalPressure) * uploadPressure;
    const std::size_t budget = static_cast<std::size_t>(
        static_cast<double>(kChunkPoolBaseBudgetBytes) / std::max(pressure, 1.0));
    return std::clamp(budget, kChunkPoolMinBudgetBytes, kChunkPoolBaseBudgetBytes);
}

void ChunkManager::Impl::trimChunkPoolToBudgetLocked(std::size_t budgetBytes)
{
    chunkPoolBudgetBytes_ = budgetBytes;
    while (chunkPoolBytes_ > budgetBytes && !chunkPool_.empty())
    {
        const std::size_t retainedBytes = chunkPool_.front().retainedBytes;
        chunkPool_.pop_front();
        chunkPoolBytes_ = (retainedBytes >= chunkPoolBytes_) ? 0 : (chunkPoolBytes_ - retainedBytes);
    }
}

void ChunkManager::Impl::trimChunkPoolToBudget()
{
    const std::size_t budgetBytes = chunkPoolBudgetBytes();
    std::lock_guard<std::mutex> lock(chunkPoolMutex_);
    trimChunkPoolToBudgetLocked(budgetBytes);
}

void ChunkManager::Impl::recycleChunkObject(std::shared_ptr<Chunk> chunk)
{
    if (!chunk)
    {
        return;
    }

    std::size_t retainedBytes = 0;
    {
        std::lock_guard<std::mutex> meshLock(chunk->meshMutex);
        chunk->reset(chunk->coord);
        retainedBytes = estimateChunkRetainedBytes(*chunk);
    }

    const std::size_t budgetBytes = chunkPoolBudgetBytes();
    std::lock_guard<std::mutex> lock(chunkPoolMutex_);
    chunkPoolBudgetBytes_ = budgetBytes;
    if (retainedBytes <= budgetBytes)
    {
        chunkPool_.push_back(PooledChunkEntry{std::move(chunk), retainedBytes});
        chunkPoolBytes_ += retainedBytes;
    }
    trimChunkPoolToBudgetLocked(budgetBytes);
}

void ChunkManager::Impl::destroyBufferPages()
{
    std::lock_guard<std::mutex> lock(bufferPageMutex_);
    for (auto& page : bufferPages_)
    {
        page.vertexBuffer.Reset();
        page.indexBuffer.Reset();
        releaseChunkBufferPageUploadBuffers(page);
    }
    bufferPages_.clear();
}

void ChunkManager::Impl::resetColumnBudgets()
{
    jobsScheduledThisFrame_.clear();
}

int ChunkManager::Impl::baseUploadsPerColumnLimit(int verticalRadius) const noexcept
{
    const int ramp = std::max(0, verticalRadius - kVerticalStreamingConfig.minRadiusChunks);
    const int divisor = std::max(1, kVerticalStreamingConfig.uploadRampDivisor);
    const int bonus = ramp / divisor;
    const int base = kVerticalStreamingConfig.uploadBasePerColumn;
    const int maxLimit = kVerticalStreamingConfig.uploadMaxPerColumn;
    return std::clamp(base + bonus, base, maxLimit);
}

std::size_t ChunkManager::Impl::estimateUploadQueueSize()
{
    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    return queuedUploadCount_;
}

std::size_t ChunkManager::Impl::estimateInitialReadyUploadQueueSize()
{
    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    return initialVisibleUploadCount_;
}

ChunkManager::Impl::UploadBudgets ChunkManager::Impl::computeUploadBudgets(int verticalRadius)
{
    UploadBudgets budgets{};
    budgets.columnLimit = baseUploadsPerColumnLimit(verticalRadius);
    const bool exactOnly = renderSettings_.totalChunks <= renderSettings_.exactChunks;
    budgets.chunkLimit = exactOnly ? 5 : 3;
    budgets.queueSize = estimateUploadQueueSize();
    const std::size_t initialReadyUploads = estimateInitialReadyUploadQueueSize();
    const double previousQueueAgeMs = uploadQueueAgeMsLastFrame_;
    const int uploadDebtSteps = computeBacklogSteps(static_cast<int>(std::min<std::size_t>(
                                                        budgets.queueSize,
                                                        static_cast<std::size_t>(std::numeric_limits<int>::max()))),
                                                    12,
                                                    8);
    const bool exactPreload = startupEnabled_ &&
                              startupState_.preloadStarted &&
                              startupState_.phase == StreamingPhase::ExactPreload;
    const bool interactiveUploadWindow = startupEnabled_ &&
                                         startupState_.preloadStarted &&
                                         (startupState_.phase == StreamingPhase::InteractiveNearOnly ||
                                          startupState_.phase == StreamingPhase::FarRamp);
    if (startupEnabled_ && startupState_.preloadStarted)
    {
        if (exactPreload)
        {
            budgets.byteBudget = exactOnly ? 24ull * 1024ull * 1024ull : 16ull * 1024ull * 1024ull;
            budgets.columnLimit = std::min(budgets.columnLimit + (exactOnly ? 4 : 2), exactOnly ? 12 : 9);
            budgets.chunkLimit = exactOnly ? 6 : 4;
            budgets.timeBudgetMs = exactOnly ? 3.25 : 2.0;
        }
        else if (interactiveUploadWindow)
        {
            budgets.byteBudget = exactOnly ? 24ull * 1024ull * 1024ull : 16ull * 1024ull * 1024ull;
            budgets.columnLimit = std::min(budgets.columnLimit + (exactOnly ? 3 : 1), exactOnly ? 10 : 7);
            budgets.chunkLimit = exactOnly ? 5 : 3;
            budgets.timeBudgetMs = exactOnly ? 2.75 : 1.5;
        }
        else
        {
            budgets.byteBudget = exactOnly ? 24ull * 1024ull * 1024ull : 20ull * 1024ull * 1024ull;
            budgets.columnLimit = std::min(budgets.columnLimit + (exactOnly ? 2 : 0), exactOnly ? 10 : budgets.columnLimit);
            budgets.chunkLimit = exactOnly ? 5 : 3;
            budgets.timeBudgetMs = exactOnly ? 3.0 : 2.0;
        }
    }
    else
    {
        budgets.byteBudget = exactOnly ? 24ull * 1024ull * 1024ull : 20ull * 1024ull * 1024ull;
        budgets.columnLimit = std::min(budgets.columnLimit + (exactOnly ? 2 : 0), exactOnly ? 10 : budgets.columnLimit);
        budgets.chunkLimit = exactOnly ? 5 : 3;
        budgets.timeBudgetMs = exactOnly ? 3.0 : 2.0;
    }

    if (uploadDebtSteps > 0)
    {
        if (exactPreload)
        {
            const int clampedSteps = std::min(uploadDebtSteps, 2);
            budgets.byteBudget += 4ull * 1024ull * 1024ull * static_cast<std::size_t>(clampedSteps);
            budgets.chunkLimit += clampedSteps;
            budgets.columnLimit = std::min(budgets.columnLimit + 1, exactOnly ? 13 : 10);
            budgets.timeBudgetMs =
                std::min(exactOnly ? 4.5 : 2.5, budgets.timeBudgetMs + 0.35 * static_cast<double>(clampedSteps));
        }
        else if (interactiveUploadWindow)
        {
            budgets.byteBudget += 4ull * 1024ull * 1024ull;
            budgets.chunkLimit = std::min(budgets.chunkLimit + 1, exactOnly ? 6 : 4);
            budgets.columnLimit = std::min(budgets.columnLimit + 1, exactOnly ? 11 : 8);
            if (exactOnly)
            {
                budgets.timeBudgetMs = std::min(3.5, budgets.timeBudgetMs + 0.25);
            }
        }
        else
        {
            const int clampedSteps = std::min(uploadDebtSteps, 2);
            budgets.byteBudget += 4ull * 1024ull * 1024ull * static_cast<std::size_t>(clampedSteps);
            budgets.chunkLimit += clampedSteps;
            budgets.columnLimit = std::min(budgets.columnLimit + 1, exactOnly ? 12 : 10);
            budgets.timeBudgetMs =
                std::min(exactOnly ? 4.0 : 2.5, budgets.timeBudgetMs + 0.35 * static_cast<double>(clampedSteps));
        }
    }

    if (initialReadyUploads > 0)
    {
        const int urgencySteps = std::min<int>(static_cast<int>(initialReadyUploads), 2);
        budgets.byteBudget += 4ull * 1024ull * 1024ull * static_cast<std::size_t>(urgencySteps);
        budgets.chunkLimit = std::min(budgets.chunkLimit + urgencySteps, exactOnly ? 8 : 6);
        budgets.columnLimit = std::min(budgets.columnLimit + 1, exactOnly ? 12 : 10);
        budgets.timeBudgetMs =
            std::min(exactOnly ? 4.25 : 3.0, budgets.timeBudgetMs + 0.45 * static_cast<double>(urgencySteps));
    }

    const int queueAgePressureSteps = computeBacklogSteps(
        static_cast<int>(std::min(previousQueueAgeMs, 5000.0)),
        150,
        150);
    int latencyPressureSteps =
        std::min(queueAgePressureSteps + static_cast<int>(std::min<std::size_t>(initialReadyUploads, 4)), 4);
    const double frameBudgetMs = 1000.0 / 60.0;
    const double frameHeadroomMs = std::clamp(frameBudgetMs + 2.0 - smoothedFrameMs_, -4.0, 4.0);
    int latencyPressureCap = (frameHeadroomMs >= 1.5) ? 4 : ((frameHeadroomMs >= 0.0) ? 3 : 2);
    if (initialReadyUploads == 0 && frameHeadroomMs < 0.0)
    {
        latencyPressureCap = 1;
    }
    latencyPressureSteps = std::min(latencyPressureSteps, latencyPressureCap);
    if (latencyPressureSteps > 0)
    {
        const int maxChunkLimit = exactPreload ? (exactOnly ? 11 : 10)
                                               : (interactiveUploadWindow ? (exactOnly ? 9 : 8)
                                                                          : (exactOnly ? 8 : 7));
        const int maxColumnLimit = exactPreload ? (exactOnly ? 13 : 12) : (exactOnly ? 11 : 10);
        const double maxTimeBudgetMs = exactPreload ? (exactOnly ? 5.5 : 5.0)
                                                    : (interactiveUploadWindow ? (exactOnly ? 5.0 : 4.5)
                                                                               : (exactOnly ? 4.5 : 4.0));

        budgets.byteBudget += 8ull * 1024ull * 1024ull * static_cast<std::size_t>(latencyPressureSteps);
        budgets.chunkLimit = std::min(maxChunkLimit, budgets.chunkLimit + latencyPressureSteps);
        budgets.columnLimit =
            std::min(maxColumnLimit, budgets.columnLimit + (latencyPressureSteps + 1) / 2);
        budgets.timeBudgetMs =
            std::min(maxTimeBudgetMs, budgets.timeBudgetMs + 0.50 * static_cast<double>(latencyPressureSteps));
    }

    if (protectedPressureActive_)
    {
        budgets.byteBudget += 8ull * 1024ull * 1024ull;
        budgets.columnLimit = std::min(budgets.columnLimit + 2, exactOnly ? 14 : 11);
        budgets.chunkLimit = std::max(budgets.chunkLimit, exactOnly ? 8 : 5);
        budgets.timeBudgetMs = std::max(budgets.timeBudgetMs, exactOnly ? 4.5 : 3.0);
    }

    if (severeProtectedPressureActive_)
    {
        budgets.byteBudget += 8ull * 1024ull * 1024ull;
        budgets.columnLimit = std::min(budgets.columnLimit + 2, exactOnly ? 16 : 12);
        budgets.chunkLimit = std::max(budgets.chunkLimit, exactOnly ? 10 : 6);
        budgets.timeBudgetMs = std::max(budgets.timeBudgetMs, exactOnly ? 5.5 : 3.5);
    }

    return budgets;
}

int ChunkManager::Impl::computeBacklogSteps(int backlog, int threshold, int stepSize) noexcept
{
    if (backlog <= threshold)
    {
        return 0;
    }

    if (stepSize <= 0)
    {
        return 1;
    }

    const long long safeOver = static_cast<long long>(backlog) - static_cast<long long>(threshold);
    const long long safeStep = std::max(stepSize, 1);
    const long long steps = (safeOver + safeStep - 1) / safeStep;
    return static_cast<int>(std::min(steps, static_cast<long long>(std::numeric_limits<int>::max())));
}

int ChunkManager::Impl::computeGenerationBudget(int horizontalRadius, int verticalRadius, int backlogSteps) const
{
    const auto& tuning = kVerticalStreamingConfig.generationBudget;
    const int safeHorizontal = std::max(horizontalRadius, 0);
    const int safeVertical = std::max(verticalRadius, 0);

    double budget = static_cast<double>(tuning.baseJobsPerFrame);
    budget += static_cast<double>(tuning.jobsPerHorizontalRing) * static_cast<double>(safeHorizontal);
    budget += static_cast<double>(tuning.jobsPerVerticalLayer) * static_cast<double>(safeVertical);
    budget += static_cast<double>(tuning.backlogBoostPerStep)
              * static_cast<double>(std::max(backlogSteps, 0));

    long long result = static_cast<long long>(std::ceil(budget));
    if (tuning.maxJobsPerFrame > 0)
    {
        result = std::min(result, static_cast<long long>(tuning.maxJobsPerFrame));
    }

    result = std::max(result, 1ll);
    return static_cast<int>(std::min(result, static_cast<long long>(std::numeric_limits<int>::max())));
}

int ChunkManager::Impl::computeRingExpansionBudget(int backlogChunks) const
{
    const auto& tuning = kVerticalStreamingConfig.generationBudget;
    const int minRings = std::max(0, tuning.minRingExpansionsPerFrame);
    const int maxRings = std::max(minRings, tuning.maxRingExpansionsPerFrame);

    if (maxRings == 0)
    {
        return 0;
    }

    if (tuning.backlogRingStepSize <= 0)
    {
        return maxRings;
    }

    const int steps = computeBacklogSteps(backlogChunks,
                                          tuning.backlogStartThreshold,
                                          tuning.backlogRingStepSize);

    int budget = minRings + steps;
    budget = std::clamp(budget, minRings, maxRings);
    return budget;
}

int ChunkManager::Impl::computeColumnJobCap(int backlogSteps, int backlogChunks) const
{
    int baseCap = kVerticalStreamingConfig.maxGenerationJobsPerColumn;
    if (baseCap <= 0)
    {
        return std::numeric_limits<int>::max();
    }

    if (kVerticalStreamingConfig.backlogColumnCapReleaseThreshold > 0 &&
        backlogChunks >= kVerticalStreamingConfig.backlogColumnCapReleaseThreshold)
    {
        return std::numeric_limits<int>::max();
    }

    const int boostPerStep = kVerticalStreamingConfig.generationBudget.columnCapBoostPerStep;
    if (boostPerStep > 0 && backlogSteps > 0)
    {
        const long long boosted = static_cast<long long>(baseCap) +
                                  static_cast<long long>(backlogSteps) *
                                      static_cast<long long>(boostPerStep);
        baseCap = static_cast<int>(std::min(boosted, static_cast<long long>(std::numeric_limits<int>::max())));
    }

    return std::max(baseCap, 0);
}

ChunkManager::Impl::VisibleChunkCoverage ChunkManager::Impl::scanVisibleChunkCoverage(const glm::ivec3& center,
                                                                                      int horizontalRadius,
                                                                                      int verticalRadius) const
{
    VisibleChunkCoverage coverage{};
    const glm::ivec2 cameraColumn{center.x, center.z};
    const int cameraChunkY = center.y;

    std::unordered_map<glm::ivec3, ChunkState, ChunkHasher> chunkStates;
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        chunkStates.reserve(chunks_.size());
        for (const auto& [coord, chunkPtr] : chunks_)
        {
            if (!chunkPtr)
            {
                continue;
            }

            chunkStates.emplace(coord, chunkPtr->state.load(std::memory_order_acquire));
        }
    }

    for (int dx = -horizontalRadius; dx <= horizontalRadius; ++dx)
    {
        for (int dz = -horizontalRadius; dz <= horizontalRadius; ++dz)
        {
            if (std::max(std::abs(dx), std::abs(dz)) > horizontalRadius)
            {
                continue;
            }

            const int chunkX = center.x + dx;
            const int chunkZ = center.z + dz;
            const glm::ivec2 column{chunkX, chunkZ};
            const bool protectedColumn =
                movementEnvelopeBucketForColumn(column, center, horizontalRadius) != MovementEnvelopeBucket::Background;
            const int worldX = chunkX * kChunkSizeX + kChunkSizeX / 2;
            const int worldZ = chunkZ * kChunkSizeZ + kChunkSizeZ / 2;

            int columnHeight = ColumnManager::kNoHeight;
            (void)tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight);

            ColumnSlabOccupancy occupancy{};
            const bool haveOccupancy = tryGetCachedColumnSlabOccupancy(column, occupancy);
            const ColumnChunkIntervals playerBand =
                playerBandIntervalsForColumn(column, cameraColumn, cameraChunkY, verticalRadius);
            const ColumnChunkIntervals intervals = columnIntervalsForHeight(column,
                                                                            cameraColumn,
                                                                            cameraChunkY,
                                                                            verticalRadius,
                                                                            columnHeight);
            for (std::uint8_t intervalIndex = 0; intervalIndex < intervals.count; ++intervalIndex)
            {
                const ChunkYInterval interval = intervals.intervals[intervalIndex];
                for (int chunkY = interval.minChunkY; chunkY <= interval.maxChunkY; ++chunkY)
                {
                    ++coverage.required;
                    if (protectedColumn)
                    {
                        ++coverage.protectedRequired;
                    }
                    const bool withinPlayerBand = chunkYWithinIntervals(chunkY, playerBand);
                    if (haveOccupancy &&
                        !withinPlayerBand &&
                        classifyColumnSlab(occupancy, chunkY) == ColumnSlabOccupancyState::DefinitelyEmpty)
                    {
                        ++coverage.ready;
                        if (protectedColumn)
                        {
                            ++coverage.protectedReady;
                        }
                        continue;
                    }

                    const auto stateIt = chunkStates.find(glm::ivec3{chunkX, chunkY, chunkZ});
                    if (stateIt == chunkStates.end())
                    {
                        ++coverage.missing;
                        if (protectedColumn)
                        {
                            ++coverage.protectedMissing;
                        }
                        continue;
                    }

                    const ChunkState state = stateIt->second;
                    if (state == ChunkState::Uploaded || state == ChunkState::Ready || state == ChunkState::Remeshing)
                    {
                        ++coverage.ready;
                        if (protectedColumn)
                        {
                            ++coverage.protectedReady;
                        }
                    }
                }
            }
        }
    }

    return coverage;
}

int ChunkManager::Impl::estimateMissingChunks(const glm::ivec3& center,
                                              int horizontalRadius,
                                              int verticalRadius) const
{
    return scanVisibleChunkCoverage(center, horizontalRadius, verticalRadius).missing;
}

int ChunkManager::Impl::computeVerticalRadius(const glm::ivec3& center,
                                              int horizontalRadius,
                                              int cameraWorldY)
{
    int verticalRadius = kVerticalStreamingConfig.minRadiusChunks;

    const glm::ivec2 cameraColumn{center.x, center.z};
    const int cameraChunkY = center.y;
    const int cameraWorldChunk = floorDiv(cameraWorldY, kChunkSizeY);
    verticalRadius = std::max(verticalRadius,
                              std::abs(cameraWorldChunk - cameraChunkY) +
                                  kVerticalStreamingConfig.columnSlackChunks);

    const int sampleRadius = std::max(0,
                                      std::min(horizontalRadius, kVerticalStreamingConfig.sampleRadiusChunks));

    for (int dx = -sampleRadius; dx <= sampleRadius; ++dx)
    {
        for (int dz = -sampleRadius; dz <= sampleRadius; ++dz)
        {
            const glm::ivec2 column{center.x + dx, center.z + dz};
            const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
            const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
            int columnHeight = ColumnManager::kNoHeight;
            if (!tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight))
            {
                requestColumnHeightPrefetch(column, ColumnHeightPrefetchPriority::Critical);
            }

            const int radius = columnRadiusForHeight(column,
                                                     cameraColumn,
                                                     cameraChunkY,
                                                     verticalRadius,
                                                     columnHeight);
            verticalRadius = std::max(verticalRadius, radius);
        }
    }

    return std::clamp(verticalRadius,
                      kVerticalStreamingConfig.minRadiusChunks,
                      kVerticalStreamingConfig.maxRadiusChunks);
}

int ChunkManager::Impl::updateEvictionCenterChunkY(int targetChunkY) noexcept
{
    if (!evictionCenterInitialized_)
    {
        evictionCenterChunkY_ = targetChunkY;
        evictionCenterInitialized_ = true;
        return evictionCenterChunkY_;
    }

    const int deadband = std::max(0, kVerticalStreamingConfig.verticalEvictionDeadbandChunks);
    if (targetChunkY > evictionCenterChunkY_ + deadband)
    {
        evictionCenterChunkY_ = targetChunkY - deadband;
    }
    else if (targetChunkY < evictionCenterChunkY_ - deadband)
    {
        evictionCenterChunkY_ = targetChunkY + deadband;
    }

    return evictionCenterChunkY_;
}

int ChunkManager::Impl::computeEvictionBudget(std::size_t pendingEvictions) const noexcept
{
    const int baseBudget = std::max(1, kVerticalStreamingConfig.baseEvictionChunksPerFrame);
    const int maxBudget = std::max(baseBudget, kVerticalStreamingConfig.maxEvictionChunksPerFrame);
    const int divisor = std::max(1, kVerticalStreamingConfig.evictionBudgetBoostDivisor);
    const std::size_t safeSteps = pendingEvictions / static_cast<std::size_t>(divisor);
    const int boostedBudget = baseBudget + static_cast<int>(std::min<std::size_t>(
                                           safeSteps,
                                           static_cast<std::size_t>(std::max(0, maxBudget - baseBudget))));
    return std::clamp(boostedBudget, baseBudget, maxBudget);
}

bool ChunkManager::Impl::tryGetPredictedColumnHeight(const glm::ivec2& column, int& outHeight) const
{
    std::lock_guard<std::mutex> lock(predictedColumnMutex_);
    auto it = predictedColumnHeights_.find(column);
    if (it == predictedColumnHeights_.end())
    {
        return false;
    }

    outHeight = it->second;
    return true;
}

bool ChunkManager::Impl::tryGetCachedColumnHeight(const glm::ivec2& column,
                                                  int worldX,
                                                  int worldZ,
                                                  int& outHeight) const
{
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();
    const auto lookupStart = benchmarkEnabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const int highest = columnManager_.highestSolidBlock(worldX, worldZ);
    if (benchmarkEnabled)
    {
        benchmarkMetrics_.columnHeightLookupStage.recordMicros(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - lookupStart).count()));
    }

    int predictedHeight = ColumnManager::kNoHeight;
    const bool havePredictedHeight = tryGetPredictedColumnHeight(column, predictedHeight);
    if (highest != ColumnManager::kNoHeight)
    {
        outHeight = havePredictedHeight ? std::max(highest, predictedHeight) : highest;
        return true;
    }

    if (havePredictedHeight)
    {
        outHeight = predictedHeight;
        return true;
    }

    return false;
}

int ChunkManager::Impl::cacheSampledColumnHeight(const glm::ivec2& column, int worldX, int worldZ) const
{
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();
    const auto sampleStart = benchmarkEnabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const ColumnSample sample = sampleColumn(worldX, worldZ);
    if (benchmarkEnabled)
    {
        benchmarkMetrics_.columnHeightSampleStage.recordMicros(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - sampleStart).count()));
    }
    const int height = sample.surfaceY;
    mergePredictedColumnHeight(column, height);
    return height;
}

void ChunkManager::Impl::requestColumnHeightPrefetch(const glm::ivec2& column,
                                                     ColumnHeightPrefetchPriority priority) const
{
    if (shouldStop_.load(std::memory_order_acquire))
    {
        return;
    }

    constexpr std::size_t kMaxQueuedColumnHeightPrefetches = 4096u;
    const glm::ivec3 centerChunk = lastCenterChunk_;
    const std::uint32_t distance = static_cast<std::uint32_t>(std::max(std::abs(column.x - centerChunk.x),
                                                                       std::abs(column.y - centerChunk.z)));

    bool shouldNotify = false;
    {
        std::lock_guard<std::mutex> lock(columnHeightPrefetchMutex_);
        auto requestIt = pendingColumnHeightPrefetchRequests_.find(column);
        if (requestIt != pendingColumnHeightPrefetchRequests_.end())
        {
            ColumnHeightPrefetchRequestState& state = requestIt->second;
            if (state.inFlight || state.priority >= priority)
            {
                return;
            }

            state.token = nextColumnHeightPrefetchToken_++;
            state.priority = priority;
            state.inFlight = false;
            pendingColumnHeightPrefetchQueue_.push(
                ColumnHeightPrefetchRequest{column, state.token, nextColumnHeightPrefetchSequence_++, distance, priority});
            shouldNotify = true;
        }
        else
        {
            const bool latencySensitive = priority >= ColumnHeightPrefetchPriority::Critical;
            if (pendingColumnHeightPrefetchRequests_.size() >= kMaxQueuedColumnHeightPrefetches && !latencySensitive)
            {
                return;
            }

            const std::uint64_t token = nextColumnHeightPrefetchToken_++;
            pendingColumnHeightPrefetchRequests_.emplace(column,
                                                         ColumnHeightPrefetchRequestState{
                                                             token,
                                                             priority,
                                                             false});
            pendingColumnHeightPrefetchQueue_.push(
                ColumnHeightPrefetchRequest{column, token, nextColumnHeightPrefetchSequence_++, distance, priority});
            shouldNotify = true;
        }
    }

    if (shouldNotify)
    {
        columnHeightPrefetchCondition_.notify_one();
    }
}

void ChunkManager::Impl::prefetchVisibleColumnHeights(const glm::ivec3& center,
                                                      const glm::ivec3& previousCenter,
                                                      int horizontalRadius)
{
    if (horizontalRadius <= 0 || shouldStop_.load(std::memory_order_acquire))
    {
        return;
    }

    const bool havePreviousCenter = updateFrameIndex_ > 1;
    const glm::ivec2 movementDelta = havePreviousCenter
        ? glm::ivec2{center.x - previousCenter.x, center.z - previousCenter.z}
        : glm::ivec2{0, 0};
    const int horizontalShift = std::max(std::abs(movementDelta.x), std::abs(movementDelta.y));
    const int verticalShift = havePreviousCenter ? std::abs(center.y - previousCenter.y) : 0;
    const bool horizontalDominant = horizontalShift > 0 && horizontalShift >= verticalShift;
    const bool verticalDominant = verticalShift > horizontalShift;
    const bool exactOnly = renderSettings_.totalChunks <= renderSettings_.exactChunks;
    const bool needsStaticWarmup = !horizontalDominant &&
                                   !verticalDominant &&
                                   targetViewDistance_ > viewDistance_ + 1;
    const bool needsStaticDiscovery = !horizontalDominant &&
                                      !verticalDominant &&
                                      exactOnly &&
                                      lastMissingChunks_ > 0;
    if (!horizontalDominant && !needsStaticWarmup && !needsStaticDiscovery)
    {
        return;
    }

    struct PrefetchCandidate
    {
        glm::ivec2 column{0};
        ColumnHeightPrefetchPriority priority{ColumnHeightPrefetchPriority::Normal};
        float score{0.0f};
    };

    const int requestBudget = horizontalDominant
        ? (severeProtectedPressureActive_ ? 160 : (protectedPressureActive_ ? 128 : 96))
        : (needsStaticDiscovery ? 144 : 48);
    std::vector<PrefetchCandidate> candidates;
    candidates.reserve(96u);
    std::unordered_set<glm::ivec2, ColumnHasher> queuedColumns;
    queuedColumns.reserve(128u);
    const std::size_t candidateCap = needsStaticDiscovery ? 224u : 160u;

    auto addCandidate = [&](const glm::ivec2& column, ColumnHeightPrefetchPriority priority, float score)
    {
        if (candidates.size() >= candidateCap && priority < ColumnHeightPrefetchPriority::Critical)
        {
            return;
        }

        if (!queuedColumns.insert(column).second)
        {
            return;
        }

        const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
        const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
        int cachedHeight = ColumnManager::kNoHeight;
        const bool haveHeight = tryGetCachedColumnHeight(column, worldX, worldZ, cachedHeight);
        const bool needsOccupancy = priority >= ColumnHeightPrefetchPriority::Visible;
        ColumnSlabOccupancy occupancy{};
        const bool haveOccupancy = !needsOccupancy || tryGetCachedColumnSlabOccupancy(column, occupancy);
        if (haveHeight && haveOccupancy)
        {
            return;
        }

        candidates.push_back(PrefetchCandidate{column, priority, score});
    };

    if (horizontalDominant)
    {
        glm::vec2 forward = normalizePriorityForwardXZ(lastCameraForward_);
        const glm::vec2 movementVector(static_cast<float>(movementDelta.x), static_cast<float>(movementDelta.y));
        if (glm::dot(movementVector, movementVector) > kEpsilon)
        {
            forward = glm::normalize(movementVector);
        }
        const glm::vec2 side{-forward.y, forward.x};
        const int startDistance = std::max(horizontalRadius - 1, 1);
        const int lookahead = std::clamp(horizontalRadius / 4, 4, 10);
        const int endDistance = horizontalRadius + lookahead;
        const int nearHalfWidth = std::clamp(horizontalRadius / 16, 1, 2);
        const int farHalfWidth = std::clamp(horizontalRadius / 12, 2, 4);

        for (int distance = startDistance; distance <= endDistance; distance += 2)
        {
            const bool insideExact = distance <= horizontalRadius;
            const bool justOutside = distance <= horizontalRadius + 2;
            const int sideHalfWidth = insideExact ? nearHalfWidth : farHalfWidth;
            const glm::vec2 anchor = glm::vec2(static_cast<float>(center.x), static_cast<float>(center.z)) +
                                     forward * static_cast<float>(distance);

            for (int sideOffset = -sideHalfWidth; sideOffset <= sideHalfWidth; ++sideOffset)
            {
                const glm::vec2 sample = anchor + side * static_cast<float>(sideOffset);
                const glm::ivec2 column{static_cast<int>(std::lround(sample.x)),
                                        static_cast<int>(std::lround(sample.y))};

                ColumnHeightPrefetchPriority priority = ColumnHeightPrefetchPriority::Normal;
                if (insideExact && std::abs(sideOffset) <= 1)
                {
                    priority = ColumnHeightPrefetchPriority::Critical;
                }
                else if (insideExact || justOutside)
                {
                    priority = ColumnHeightPrefetchPriority::Visible;
                }

                addCandidate(column,
                             priority,
                             static_cast<float>(distance) + static_cast<float>(std::abs(sideOffset)) * 0.4f);
            }
        }
    }
    else
    {
        auto addPerimeter = [&](int ring)
        {
            const int stride = ring < 16 ? 2 : 4;
            for (int dx = -ring; dx <= ring; dx += stride)
            {
                addCandidate(glm::ivec2{center.x + dx, center.z - ring},
                             ColumnHeightPrefetchPriority::Visible,
                             static_cast<float>(ring));
                addCandidate(glm::ivec2{center.x + dx, center.z + ring},
                             ColumnHeightPrefetchPriority::Visible,
                             static_cast<float>(ring));
            }

            for (int dz = -ring + stride; dz <= ring - stride; dz += stride)
            {
                addCandidate(glm::ivec2{center.x - ring, center.z + dz},
                             ColumnHeightPrefetchPriority::Visible,
                             static_cast<float>(ring));
                addCandidate(glm::ivec2{center.x + ring, center.z + dz},
                             ColumnHeightPrefetchPriority::Visible,
                             static_cast<float>(ring));
            }
        };

        if (needsStaticDiscovery)
        {
            const int ringEnd = std::max(targetViewDistance_, 1);
            const int ringStart = std::max(1, ringEnd - 6);
            for (int ring = ringStart; ring <= ringEnd; ++ring)
            {
                addPerimeter(ring);
            }

            const int laneStep = std::max(4, ringEnd / 6);
            const int lanePhase = static_cast<int>(updateFrameIndex_ % static_cast<std::uint64_t>(laneStep));
            for (int offset = -ringEnd + lanePhase; offset <= ringEnd; offset += laneStep)
            {
                addCandidate(glm::ivec2{center.x + offset, center.z},
                             ColumnHeightPrefetchPriority::Visible,
                             static_cast<float>(std::abs(offset)));
                addCandidate(glm::ivec2{center.x, center.z + offset},
                             ColumnHeightPrefetchPriority::Visible,
                             static_cast<float>(std::abs(offset)));
            }
        }
        else
        {
            const int ringStart = std::max(viewDistance_ + 1, 1);
            const int ringEnd = std::min(targetViewDistance_, ringStart + 2);
            for (int ring = ringStart; ring <= ringEnd; ++ring)
            {
                addPerimeter(ring);
            }
        }
    }

    if (candidates.empty())
    {
        return;
    }

    std::sort(candidates.begin(),
              candidates.end(),
              [](const PrefetchCandidate& lhs, const PrefetchCandidate& rhs)
              {
                  if (lhs.priority != rhs.priority)
                  {
                      return lhs.priority > rhs.priority;
                  }
                  if (lhs.score != rhs.score)
                  {
                      return lhs.score < rhs.score;
                  }
                  if (lhs.column.x != rhs.column.x)
                  {
                      return lhs.column.x < rhs.column.x;
                  }
                  return lhs.column.y < rhs.column.y;
              });

    int requested = 0;
    for (const PrefetchCandidate& candidate : candidates)
    {
        if (requested >= requestBudget)
        {
            break;
        }

        requestColumnHeightPrefetch(candidate.column, candidate.priority);
        ++requested;
    }
}

int ChunkManager::Impl::ensureColumnHeightCached(const glm::ivec2& column,
                                                 int worldX,
                                                 int worldZ) const
{
    int highest = ColumnManager::kNoHeight;
    if (tryGetCachedColumnHeight(column, worldX, worldZ, highest))
    {
        return highest;
    }

    return cacheSampledColumnHeight(column, worldX, worldZ);
}

void ChunkManager::Impl::mergePredictedColumnHeight(const glm::ivec2& column, int height) const
{
    if (height == ColumnManager::kNoHeight)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(predictedColumnMutex_);
    auto [it, inserted] = predictedColumnHeights_.try_emplace(column, height);
    if (!inserted)
    {
        it->second = std::max(it->second, height);
    }
}

void ChunkManager::Impl::refreshPredictedColumnHeightFromLoadedData(const glm::ivec2& column) const
{
    const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
    const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
    const int highest = columnManager_.highestSolidBlock(worldX, worldZ);

    std::lock_guard<std::mutex> lock(predictedColumnMutex_);
    if (highest == ColumnManager::kNoHeight)
    {
        predictedColumnHeights_.erase(column);
    }
    else
    {
        predictedColumnHeights_[column] = highest;
    }
}

void ChunkManager::Impl::invalidatePredictedColumn(const glm::ivec2& column) const
{
    std::lock_guard<std::mutex> lock(predictedColumnMutex_);
    predictedColumnHeights_.erase(column);
}

void ChunkManager::Impl::noteRecentEdit(const char* kind,
                                        const glm::ivec3& worldPos,
                                        const glm::ivec3& chunkCoord)
{
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(recentEditDebugMutex_);
        recentEditDebug_.valid = true;
        recentEditDebug_.kind = kind ? kind : "edit";
        recentEditDebug_.worldPos = worldPos;
        recentEditDebug_.chunkCoord = chunkCoord;
        recentEditDebug_.timestamp = now;
        recentEditDebugEvents_.clear();
    }

    std::ostringstream stream;
    stream << "edit " << (kind ? kind : "edit")
           << " world=(" << worldPos.x << ", " << worldPos.y << ", " << worldPos.z << ")"
           << " chunk=(" << chunkCoord.x << ", " << chunkCoord.y << ", " << chunkCoord.z << ")";
    appendRecentEditDebugEvent(stream.str());
}

void ChunkManager::Impl::appendRecentEditDebugEvent(const std::string& event)
{
    std::lock_guard<std::mutex> lock(recentEditDebugMutex_);
    if (!recentEditDebug_.valid)
    {
        return;
    }

    const double ageMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - recentEditDebug_.timestamp).count();

    std::ostringstream stream;
    stream.setf(std::ios::fixed, std::ios::floatfield);
    stream << '[' << std::setprecision(1) << ageMs << " ms] " << event;
    recentEditDebugEvents_.push_back(stream.str());
    while (recentEditDebugEvents_.size() > 48)
    {
        recentEditDebugEvents_.pop_front();
    }
}

bool ChunkManager::Impl::shouldTrackRecentEditChunk(const glm::ivec3& coord) const
{
    std::lock_guard<std::mutex> lock(recentEditDebugMutex_);
    if (!recentEditDebug_.valid)
    {
        return false;
    }

    const double ageSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - recentEditDebug_.timestamp).count();
    if (ageSeconds > 6.0)
    {
        return false;
    }

    return std::abs(coord.x - recentEditDebug_.chunkCoord.x) <= 1 &&
           std::abs(coord.z - recentEditDebug_.chunkCoord.z) <= 1 &&
           std::abs(coord.y - recentEditDebug_.chunkCoord.y) <= 6;
}

RecentEditHoleDebugSnapshot ChunkManager::Impl::recentEditHoleDebugSnapshot(const glm::vec3& cameraPos) const
{
    RecentEditHoleDebugSnapshot snapshot{};
    RecentEditDebugState recentEdit{};
    {
        std::lock_guard<std::mutex> lock(recentEditDebugMutex_);
        recentEdit = recentEditDebug_;
        snapshot.recentEvents.assign(recentEditDebugEvents_.begin(), recentEditDebugEvents_.end());
    }

    if (!recentEdit.valid)
    {
        return snapshot;
    }

    snapshot.editKind = recentEdit.kind;
    snapshot.editWorldPos = recentEdit.worldPos;
    snapshot.editChunkCoord = recentEdit.chunkCoord;
    snapshot.ageSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - recentEdit.timestamp).count();
    snapshot.hasRecentEdit = snapshot.ageSeconds <= 6.0;

    const glm::ivec3 cameraChunk = worldToChunkCoords(static_cast<int>(std::floor(cameraPos.x)),
                                                      static_cast<int>(std::floor(cameraPos.y)),
                                                      static_cast<int>(std::floor(cameraPos.z)));
    const glm::ivec3 centerChunk = lastCenterChunk_;
    const glm::ivec2 cameraColumn{centerChunk.x, centerChunk.z};
    const int cameraChunkY = centerChunk.y;
    snapshot.cameraChunkY = cameraChunkY;
    snapshot.verticalRadius = lastVerticalRadius_;

    const int minY = std::max(0, std::min(recentEdit.chunkCoord.y, cameraChunk.y) - 4);
    const int maxY = std::max(recentEdit.chunkCoord.y + 2, cameraChunk.y + 1);
    std::vector<glm::ivec3> coords;
    coords.reserve(static_cast<std::size_t>((maxY - minY + 1) * 9));
    for (int x = recentEdit.chunkCoord.x - 1; x <= recentEdit.chunkCoord.x + 1; ++x)
    {
        for (int z = recentEdit.chunkCoord.z - 1; z <= recentEdit.chunkCoord.z + 1; ++z)
        {
            for (int y = minY; y <= maxY; ++y)
            {
                coords.push_back(glm::ivec3{x, y, z});
            }
        }
    }

    std::unordered_map<glm::ivec3, std::shared_ptr<Chunk>, ChunkHasher> localChunks;
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        for (const glm::ivec3& coord : coords)
        {
            auto it = chunks_.find(coord);
            if (it != chunks_.end())
            {
                localChunks.emplace(coord, it->second);
            }
        }
    }

    std::sort(coords.begin(),
              coords.end(),
              [](const glm::ivec3& lhs, const glm::ivec3& rhs)
              {
                  if (lhs.y != rhs.y)
                  {
                      return lhs.y > rhs.y;
                  }
                  if (lhs.x != rhs.x)
                  {
                      return lhs.x < rhs.x;
                  }
                  return lhs.z < rhs.z;
              });

    const int horizontalThreshold = targetViewDistance_ + kVerticalStreamingConfig.horizontalEvictionSlack;

    for (const glm::ivec3& coord : coords)
    {
        RecentEditHoleChunkInfo info{};
        info.coord = coord;

        const glm::ivec2 column{coord.x, coord.z};
        const int worldX = coord.x * kChunkSizeX + kChunkSizeX / 2;
        const int worldZ = coord.z * kChunkSizeZ + kChunkSizeZ / 2;

        int columnHeight = columnManager_.highestSolidBlock(worldX, worldZ);
        if (columnHeight != ColumnManager::kNoHeight)
        {
            info.heightSource = "column";
        }
        else
        {
            int predictedHeight = ColumnManager::kNoHeight;
            if (tryGetPredictedColumnHeight(column, predictedHeight))
            {
                columnHeight = predictedHeight;
                info.heightSource = "predicted";
            }
            else
            {
                requestColumnHeightPrefetch(column, ColumnHeightPrefetchPriority::Background);
                columnHeight = centerChunk.y * kChunkSizeY;
                info.heightSource = "pending";
            }
        }
        info.columnHeight = columnHeight;

        const ColumnChunkIntervals intervals =
            columnIntervalsForHeight(column, cameraColumn, cameraChunkY, lastVerticalRadius_, columnHeight);
        info.columnMinChunkY = intervals.minChunkY();
        info.columnMaxChunkY = intervals.maxChunkY();

        const int horizontalDistance = std::max(std::abs(coord.x - centerChunk.x), std::abs(coord.z - centerChunk.z));
        info.wouldEvict = coord.y < 0 ||
                          horizontalDistance > horizontalThreshold ||
                          !chunkYWithinIntervals(coord.y, intervals, kVerticalStreamingConfig.columnSlackChunks);

        auto it = localChunks.find(coord);
        if (it == localChunks.end())
        {
            snapshot.chunks.push_back(std::move(info));
            continue;
        }

        const std::shared_ptr<Chunk>& chunk = it->second;
        info.present = true;
        info.stateLabel = chunkStateLabel(chunk->state.load(std::memory_order_acquire));
        info.hasBlocks = chunk->hasBlocks.load(std::memory_order_acquire);
        info.meshReady = chunk->meshReady.load(std::memory_order_acquire);
        info.queuedForUpload = chunk->queuedForUpload.load(std::memory_order_acquire);
        info.inFlight = chunk->inFlight.load(std::memory_order_acquire);
        info.indexCount = chunk->indexCount.load(std::memory_order_acquire);
        info.bufferPageIndex = chunk->bufferPageIndex.load(std::memory_order_acquire);
        snapshot.chunks.push_back(std::move(info));
    }

    return snapshot;
}

int ChunkManager::Impl::columnRadiusFor(const glm::ivec2& column,
                                        const glm::ivec2& cameraColumn,
                                        int cameraChunkY,
                                        int verticalRadius) const
{
    const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
    const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
    int columnHeight = ColumnManager::kNoHeight;
    if (!tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight))
    {
        requestColumnHeightPrefetch(column);
    }
    return columnRadiusForHeight(column, cameraColumn, cameraChunkY, verticalRadius, columnHeight);
}

int ChunkManager::Impl::columnRadiusForHeight(const glm::ivec2& column,
                                              const glm::ivec2& cameraColumn,
                                              int cameraChunkY,
                                              int verticalRadius,
                                              int columnHeight) const
{
    int radius = std::max(verticalRadius, kVerticalStreamingConfig.minRadiusChunks);

    const int falloffStep = kVerticalStreamingConfig.verticalRadiusFalloffStep;
    if (falloffStep > 0)
    {
        const int horizontalDistance = std::max(std::abs(column.x - cameraColumn.x),
                                                std::abs(column.y - cameraColumn.y));
        if (horizontalDistance > 0)
        {
            const int reduction = horizontalDistance / falloffStep;
            if (reduction > 0)
            {
                radius = std::max(kVerticalStreamingConfig.minRadiusChunks, radius - reduction);
            }
        }
    }

    if (columnHeight != ColumnManager::kNoHeight)
    {
        const int highestChunk = floorDiv(columnHeight, kChunkSizeY);
        const int required = std::abs(highestChunk - cameraChunkY) +
                             kVerticalStreamingConfig.columnSlackChunks;
        radius = std::max(radius, required);
    }

    return std::clamp(radius,
                      kVerticalStreamingConfig.minRadiusChunks,
                      kVerticalStreamingConfig.maxRadiusChunks);
}

int ChunkManager::Impl::cameraBandRadiusForColumn(const glm::ivec2& column,
                                                  const glm::ivec2& cameraColumn,
                                                  int verticalRadius) const
{
    int radius = std::max(verticalRadius, kVerticalStreamingConfig.minRadiusChunks);

    const int falloffStep = kVerticalStreamingConfig.verticalRadiusFalloffStep;
    if (falloffStep > 0)
    {
        const int horizontalDistance = std::max(std::abs(column.x - cameraColumn.x),
                                                std::abs(column.y - cameraColumn.y));
        if (horizontalDistance > 0)
        {
            const int reduction = horizontalDistance / falloffStep;
            if (reduction > 0)
            {
                radius = std::max(kVerticalStreamingConfig.minRadiusChunks, radius - reduction);
            }
        }
    }

    return std::clamp(radius,
                      kVerticalStreamingConfig.minRadiusChunks,
                      kExactPlayerBandRadiusMax);
}

int ChunkManager::Impl::surfaceShellFloorChunkForHeight(const glm::ivec2& column, int columnHeight) const
{
    if (columnHeight == ColumnManager::kNoHeight)
    {
        return -1;
    }

    int minNeighborHeight = columnHeight;
    constexpr std::array<glm::ivec2, 4> kNeighborOffsets{
        glm::ivec2{1, 0},
        glm::ivec2{-1, 0},
        glm::ivec2{0, 1},
        glm::ivec2{0, -1},
    };

    for (const glm::ivec2& offset : kNeighborOffsets)
    {
        const glm::ivec2 neighbor = column + offset;
        const int worldX = neighbor.x * kChunkSizeX + kChunkSizeX / 2;
        const int worldZ = neighbor.y * kChunkSizeZ + kChunkSizeZ / 2;
        int neighborHeight = ColumnManager::kNoHeight;
        if (!tryGetCachedColumnHeight(neighbor, worldX, worldZ, neighborHeight))
        {
            continue;
        }

        if (neighborHeight != ColumnManager::kNoHeight)
        {
            minNeighborHeight = std::min(minNeighborHeight, neighborHeight);
        }
    }

    const int lowestExposedWorldY = std::min(columnHeight, minNeighborHeight + 1);
    const int baseChunk = floorDiv(std::max(0, lowestExposedWorldY), kChunkSizeY);
    return std::max(0, baseChunk - kExactSurfaceShellBelowSlackChunks);
}

bool ChunkManager::Impl::columnUsesPlayerBand(const glm::ivec2& column,
                                              const glm::ivec2& cameraColumn) const noexcept
{
    const int horizontalDistance = std::max(std::abs(column.x - cameraColumn.x),
                                            std::abs(column.y - cameraColumn.y));
    return horizontalDistance <= kExactPlayerBandHorizontalRadius;
}

ColumnChunkIntervals ChunkManager::Impl::playerBandIntervalsForColumn(const glm::ivec2& column,
                                                                      const glm::ivec2& cameraColumn,
                                                                      int cameraChunkY,
                                                                      int verticalRadius) const
{
    ColumnChunkIntervals intervals{};
    if (!columnUsesPlayerBand(column, cameraColumn))
    {
        return intervals;
    }

    const int cameraBandRadius = cameraBandRadiusForColumn(column, cameraColumn, verticalRadius);
    addColumnChunkInterval(intervals,
                           std::max(0, cameraChunkY - cameraBandRadius),
                           std::max(0, cameraChunkY + cameraBandRadius));
    return intervals;
}

std::pair<int, int> ChunkManager::Impl::columnSpanFor(const glm::ivec2& column,
                                                       const glm::ivec2& cameraColumn,
                                                       int cameraChunkY,
                                                       int verticalRadius) const
{
    const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
    const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
    int columnHeight = ColumnManager::kNoHeight;
    if (!tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight))
    {
        requestColumnHeightPrefetch(column);
    }
    return columnSpanForHeight(column, cameraColumn, cameraChunkY, verticalRadius, columnHeight);
}

void ChunkManager::Impl::addColumnChunkInterval(ColumnChunkIntervals& intervals,
                                                int minChunkY,
                                                int maxChunkY) noexcept
{
    if (maxChunkY < minChunkY || maxChunkY < 0)
    {
        return;
    }

    minChunkY = std::max(0, minChunkY);
    maxChunkY = std::max(minChunkY, maxChunkY);

    ChunkYInterval pending{minChunkY, maxChunkY};
    std::array<ChunkYInterval, ColumnChunkIntervals::kMaxIntervals + 1> merged{};
    std::size_t mergedCount = 0;
    bool inserted = false;

    for (std::uint8_t i = 0; i < intervals.count; ++i)
    {
        const ChunkYInterval current = intervals.intervals[i];
        if (current.maxChunkY + 1 < pending.minChunkY)
        {
            merged[mergedCount++] = current;
            continue;
        }

        if (pending.maxChunkY + 1 < current.minChunkY)
        {
            if (!inserted)
            {
                merged[mergedCount++] = pending;
                inserted = true;
            }
            merged[mergedCount++] = current;
            continue;
        }

        pending.minChunkY = std::min(pending.minChunkY, current.minChunkY);
        pending.maxChunkY = std::max(pending.maxChunkY, current.maxChunkY);
    }

    if (!inserted)
    {
        merged[mergedCount++] = pending;
    }

    if (mergedCount > ColumnChunkIntervals::kMaxIntervals)
    {
        intervals.intervals.fill(ChunkYInterval{});
        intervals.intervals[0] = ChunkYInterval{merged.front().minChunkY,
                                                merged[mergedCount - 1].maxChunkY};
        intervals.count = 1;
        return;
    }

    intervals.intervals.fill(ChunkYInterval{});
    for (std::size_t i = 0; i < mergedCount; ++i)
    {
        intervals.intervals[i] = merged[i];
    }
    intervals.count = static_cast<std::uint8_t>(mergedCount);
}

void ChunkManager::Impl::mergeColumnChunkIntervals(ColumnChunkIntervals& dst,
                                                   const ColumnChunkIntervals& src) noexcept
{
    for (std::uint8_t i = 0; i < src.count; ++i)
    {
        addColumnChunkInterval(dst, src.intervals[i].minChunkY, src.intervals[i].maxChunkY);
    }
}

bool ChunkManager::Impl::chunkYWithinIntervals(int chunkY,
                                               const ColumnChunkIntervals& intervals,
                                               int slackChunks) noexcept
{
    for (std::uint8_t i = 0; i < intervals.count; ++i)
    {
        const ChunkYInterval interval = intervals.intervals[i];
        if (chunkY >= interval.minChunkY - slackChunks && chunkY <= interval.maxChunkY + slackChunks)
        {
            return true;
        }
    }
    return false;
}

int ChunkManager::Impl::chunkYDistanceToIntervals(int chunkY,
                                                  const ColumnChunkIntervals& intervals,
                                                  int slackChunks) noexcept
{
    if (intervals.empty())
    {
        return std::numeric_limits<int>::max();
    }

    int bestDistance = std::numeric_limits<int>::max();
    for (std::uint8_t i = 0; i < intervals.count; ++i)
    {
        const ChunkYInterval interval = intervals.intervals[i];
        const int minChunkY = interval.minChunkY - slackChunks;
        const int maxChunkY = interval.maxChunkY + slackChunks;
        if (chunkY < minChunkY)
        {
            bestDistance = std::min(bestDistance, minChunkY - chunkY);
        }
        else if (chunkY > maxChunkY)
        {
            bestDistance = std::min(bestDistance, chunkY - maxChunkY);
        }
        else
        {
            return 0;
        }
    }

    return bestDistance;
}

int ChunkManager::Impl::adjustedSurfaceYForColumn(const terrain::SurfaceColumn& surfaceColumn,
                                                  float neighborAverage) const noexcept
{
    const BiomeDefinition* biome = surfaceColumn.dominantBiome;
    int adjustedSurfaceY = surfaceColumn.surfaceY;
    if (biome == nullptr)
    {
        return adjustedSurfaceY;
    }

    const auto& soilCreep = biome->terrainSettings.soilCreep;
    if (soilCreep.strength <= 0.0f || surfaceColumn.soilCreepCoefficient <= 0.0f)
    {
        return adjustedSurfaceY;
    }

    const float strength = std::clamp(surfaceColumn.soilCreepCoefficient * soilCreep.strength, 0.0f, 1.0f);
    float offset = (neighborAverage - static_cast<float>(adjustedSurfaceY)) * strength;
    if (soilCreep.maxStep > 0)
    {
        const float maxStep = static_cast<float>(soilCreep.maxStep);
        offset = std::clamp(offset, -maxStep, maxStep);
    }
    if (soilCreep.maxDepth > 0)
    {
        const float maxDepth = static_cast<float>(soilCreep.maxDepth);
        offset = std::clamp(offset, -maxDepth, maxDepth);
    }

    return static_cast<int>(std::round(static_cast<float>(adjustedSurfaceY) + offset));
}

ChunkManager::Impl::ColumnSlabOccupancy ChunkManager::Impl::buildColumnSlabOccupancy(const glm::ivec2& column) const
{
    ColumnSlabOccupancy occupancy{};
    auto noteChunkInterval = [&](int minChunkY, int maxChunkY)
    {
        occupancy.highestOccupiedChunkY = std::max(occupancy.highestOccupiedChunkY, maxChunkY);
        addColumnChunkInterval(occupancy.terrainIntervals, minChunkY, maxChunkY);
    };

    const int baseWorldX = column.x * kChunkSizeX;
    const int baseWorldZ = column.y * kChunkSizeZ;
    constexpr int kSampleExtentX = kChunkSizeX + 2;
    constexpr int kSampleExtentZ = kChunkSizeZ + 2;

    std::array<terrain::SurfaceColumn, static_cast<std::size_t>(kSampleExtentX * kSampleExtentZ)> surfaceColumns{};
    auto sampleIndex = [](int sampleX, int sampleZ) noexcept {
        return static_cast<std::size_t>(sampleZ * kSampleExtentX + sampleX);
    };

    if (surfaceMap_)
    {
        for (int sampleX = -1; sampleX <= kChunkSizeX; ++sampleX)
        {
            for (int sampleZ = -1; sampleZ <= kChunkSizeZ; ++sampleZ)
            {
                surfaceColumns[sampleIndex(sampleX + 1, sampleZ + 1)] =
                    surfaceMap_->columnValue(baseWorldX + sampleX, baseWorldZ + sampleZ);
            }
        }

        auto computeNeighborAverage = [&](int localX, int localZ) noexcept {
            float sum = 0.0f;
            int count = 0;
            for (int dx = -1; dx <= 1; ++dx)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    if (dx == 0 && dz == 0)
                    {
                        continue;
                    }
                    sum += static_cast<float>(
                        surfaceColumns[sampleIndex(localX + dx + 1, localZ + dz + 1)].surfaceY);
                    ++count;
                }
            }
            return count > 0 ? sum / static_cast<float>(count) : 0.0f;
        };

        int highestTerrainWorld = std::numeric_limits<int>::min();
        int minWaterBottomWorld = std::numeric_limits<int>::max();
        int maxWaterTopWorld = std::numeric_limits<int>::min();

        for (int localX = 0; localX < kChunkSizeX; ++localX)
        {
            for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
            {
                const terrain::SurfaceColumn& surfaceColumn = surfaceColumns[sampleIndex(localX + 1, localZ + 1)];
                const BiomeDefinition* biome = surfaceColumn.dominantBiome;
                if (biome == nullptr)
                {
                    continue;
                }

                const int adjustedSurfaceY =
                    adjustedSurfaceYForColumn(surfaceColumn, computeNeighborAverage(localX, localZ));
                highestTerrainWorld = std::max(highestTerrainWorld, adjustedSurfaceY);

                const auto& waterFill = biome->terrainSettings.waterFill;
                if (!waterFill.enabled || adjustedSurfaceY >= globalSeaLevel_)
                {
                    continue;
                }

                int waterBottomWorld = adjustedSurfaceY + 1;
                int waterTopWorld = globalSeaLevel_;
                if (waterFill.maxDepth > 0)
                {
                    waterBottomWorld = std::max(waterBottomWorld, waterTopWorld - waterFill.maxDepth + 1);
                }
                minWaterBottomWorld = std::min(minWaterBottomWorld, waterBottomWorld);
                maxWaterTopWorld = std::max(maxWaterTopWorld, waterTopWorld);
            }
        }

        if (highestTerrainWorld >= 0)
        {
            const int highestChunkY = floorDiv(highestTerrainWorld, kChunkSizeY);
            noteChunkInterval(0, highestChunkY);
            const int shellFloorChunk = surfaceShellFloorChunkForHeight(column, highestTerrainWorld);
            addColumnChunkInterval(occupancy.surfaceShellIntervals,
                                   std::max(0, shellFloorChunk),
                                   std::max(highestChunkY, highestChunkY + kExactSurfaceShellAirAboveChunks));
        }
        if (maxWaterTopWorld >= 0 && minWaterBottomWorld <= maxWaterTopWorld)
        {
            const int minChunkY = floorDiv(std::max(0, minWaterBottomWorld), kChunkSizeY);
            const int maxChunkY = floorDiv(std::max(0, maxWaterTopWorld), kChunkSizeY);
            occupancy.highestOccupiedChunkY = std::max(occupancy.highestOccupiedChunkY, maxChunkY);
            addColumnChunkInterval(occupancy.waterIntervals, minChunkY, maxChunkY);
        }
    }

    const StructureChunkColumnSpan structureSpan = structureRegistry_.queryChunkColumnSpan(column);
    if (structureSpan.valid())
    {
        occupancy.highestOccupiedChunkY = std::max(occupancy.highestOccupiedChunkY, structureSpan.maxChunkY);
        addColumnChunkInterval(occupancy.structureIntervals, structureSpan.minChunkY, structureSpan.maxChunkY);
    }

    {
        std::lock_guard<std::mutex> lock(pendingStructureMutex_);
        for (const auto& [coord, edits] : pendingStructureEdits_)
        {
            if (coord.x != column.x || coord.z != column.y)
            {
                continue;
            }

            const bool hasSolidEdit = std::any_of(edits.begin(),
                                                  edits.end(),
                                                  [](const PendingStructureEdit& edit) {
                                                      return edit.block != BlockId::Air;
                                                  });
            if (hasSolidEdit)
            {
                occupancy.highestOccupiedChunkY = std::max(occupancy.highestOccupiedChunkY, coord.y);
                addColumnChunkInterval(occupancy.editIntervals, coord.y, coord.y);
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(blockEditOverlayMutex_);
        for (const auto& [coord, overlays] : blockEditOverlays_)
        {
            if (coord.x != column.x || coord.z != column.y)
            {
                continue;
            }

            const bool hasSolidOverlay = std::any_of(overlays.begin(),
                                                     overlays.end(),
                                                     [](const BlockEditOverlayEntry& overlay) {
                                                         return overlay.block != BlockId::Air;
                                                     });
            if (hasSolidOverlay)
            {
                occupancy.highestOccupiedChunkY = std::max(occupancy.highestOccupiedChunkY, coord.y);
                addColumnChunkInterval(occupancy.editIntervals, coord.y, coord.y);
            }
        }
    }

    mergeColumnChunkIntervals(occupancy.occupiedIntervals, occupancy.terrainIntervals);
    mergeColumnChunkIntervals(occupancy.occupiedIntervals, occupancy.waterIntervals);
    mergeColumnChunkIntervals(occupancy.occupiedIntervals, occupancy.structureIntervals);
    mergeColumnChunkIntervals(occupancy.occupiedIntervals, occupancy.editIntervals);
    return occupancy;
}

ChunkManager::Impl::ColumnSlabOccupancy ChunkManager::Impl::cachedColumnSlabOccupancy(const glm::ivec2& column) const
{
    {
        std::lock_guard<std::mutex> lock(columnSlabOccupancyMutex_);
        auto it = columnSlabOccupancyCache_.find(column);
        if (it != columnSlabOccupancyCache_.end())
        {
            return it->second;
        }
    }

    ColumnSlabOccupancy built = buildColumnSlabOccupancy(column);
    {
        std::lock_guard<std::mutex> lock(columnSlabOccupancyMutex_);
        auto [it, inserted] = columnSlabOccupancyCache_.emplace(column, built);
        if (!inserted)
        {
            return it->second;
        }
    }
    return built;
}

bool ChunkManager::Impl::tryGetCachedColumnSlabOccupancy(const glm::ivec2& column,
                                                         ColumnSlabOccupancy& out) const
{
    std::lock_guard<std::mutex> lock(columnSlabOccupancyMutex_);
    auto it = columnSlabOccupancyCache_.find(column);
    if (it == columnSlabOccupancyCache_.end())
    {
        return false;
    }

    out = it->second;
    return true;
}

ChunkManager::Impl::ColumnSlabOccupancyState
ChunkManager::Impl::classifyColumnSlab(const ColumnSlabOccupancy& occupancy, int chunkY) noexcept
{
    if (chunkYWithinIntervals(chunkY, occupancy.occupiedIntervals))
    {
        return ColumnSlabOccupancyState::DefinitelyOccupied;
    }
    if (chunkYWithinIntervals(chunkY, occupancy.maybeIntervals))
    {
        return ColumnSlabOccupancyState::MaybeOccupied;
    }
    return ColumnSlabOccupancyState::DefinitelyEmpty;
}

void ChunkManager::Impl::invalidateColumnSlabOccupancy(const glm::ivec2& column) const
{
    std::lock_guard<std::mutex> lock(columnSlabOccupancyMutex_);
    columnSlabOccupancyCache_.erase(column);
}

void ChunkManager::Impl::invalidateAllColumnSlabOccupancy() const
{
    std::lock_guard<std::mutex> lock(columnSlabOccupancyMutex_);
    columnSlabOccupancyCache_.clear();
}

void ChunkManager::Impl::updateMovementEnvelopeState(const glm::ivec3& center,
                                                     const glm::ivec3& previousCenter) noexcept
{
    const bool havePreviousCenter = updateFrameIndex_ > 1;
    const glm::ivec2 movementDelta = havePreviousCenter
        ? glm::ivec2{center.x - previousCenter.x, center.z - previousCenter.z}
        : glm::ivec2{0, 0};
    lastHorizontalMovementShift_ = std::max(std::abs(movementDelta.x), std::abs(movementDelta.y));
    lastVerticalMovementShift_ = havePreviousCenter ? std::abs(center.y - previousCenter.y) : 0;

    glm::vec2 desiredForward = normalizePriorityForwardXZ(lastCameraForward_);
    if (glm::dot(desiredForward, desiredForward) <= kEpsilon)
    {
        desiredForward = movementEnvelopeForwardXZ_;
    }

    glm::vec2 blendedForward = desiredForward;
    const glm::vec2 movementVector(static_cast<float>(movementDelta.x), static_cast<float>(movementDelta.y));
    if (glm::dot(movementVector, movementVector) > kEpsilon)
    {
        const glm::vec2 movementForward = glm::normalize(movementVector);
        const float movementWeight = lastHorizontalMovementShift_ >= 2 ? 0.80f : 0.65f;
        const float facingWeight = 1.0f - movementWeight;
        blendedForward = movementForward * movementWeight + desiredForward * facingWeight;
    }
    else if (glm::dot(movementEnvelopeForwardXZ_, movementEnvelopeForwardXZ_) > kEpsilon)
    {
        blendedForward = movementEnvelopeForwardXZ_ * 0.65f + desiredForward * 0.35f;
    }

    if (glm::dot(blendedForward, blendedForward) > kEpsilon)
    {
        movementEnvelopeForwardXZ_ = glm::normalize(blendedForward);
    }
    else
    {
        movementEnvelopeForwardXZ_ = glm::vec2{0.0f, -1.0f};
    }
}

ChunkManager::Impl::MovementEnvelopeBucket ChunkManager::Impl::movementEnvelopeBucketForColumn(
    const glm::ivec2& column,
    const glm::ivec3& center,
    int horizontalRadius,
    int lookaheadChunks) const noexcept
{
    const int dx = column.x - center.x;
    const int dz = column.y - center.z;
    const int horizontalDistance = std::max(std::abs(dx), std::abs(dz));
    if (horizontalDistance == 0)
    {
        return MovementEnvelopeBucket::Core;
    }

    const int safeHorizontalRadius = std::max(horizontalRadius, 1);
    const int coreRadius = std::clamp(safeHorizontalRadius / 8,
                                      kMovementEnvelopeCoreRadiusMin,
                                      kMovementEnvelopeCoreRadiusMax);
    if (horizontalDistance <= coreRadius)
    {
        return MovementEnvelopeBucket::Core;
    }

    glm::vec2 forward = movementEnvelopeForwardXZ_;
    if (glm::dot(forward, forward) <= kEpsilon)
    {
        forward = normalizePriorityForwardXZ(lastCameraForward_);
    }
    if (glm::dot(forward, forward) <= kEpsilon)
    {
        forward = glm::vec2{0.0f, -1.0f};
    }

    const glm::vec2 side{-forward.y, forward.x};
    const glm::vec2 delta(static_cast<float>(dx), static_cast<float>(dz));
    const float forwardDistance = glm::dot(delta, forward) + static_cast<float>(lookaheadChunks);
    const float sideDistance = std::abs(glm::dot(delta, side));
    const int corridorHalfWidth =
        std::clamp(kMovementEnvelopeCorridorHalfWidthMin +
                       static_cast<int>(std::max(forwardDistance, 0.0f)) / kMovementEnvelopeCorridorWidthStep,
                   kMovementEnvelopeCorridorHalfWidthMin,
                   kMovementEnvelopeCorridorHalfWidthMax);
    const int corridorForwardReach = safeHorizontalRadius + std::clamp(safeHorizontalRadius / 4, 4, 12);
    if (forwardDistance >= -1.0f &&
        forwardDistance <= static_cast<float>(corridorForwardReach) &&
        sideDistance <= static_cast<float>(corridorHalfWidth))
    {
        return MovementEnvelopeBucket::Corridor;
    }

    const int turnReserveRadius = std::clamp(safeHorizontalRadius / 3,
                                             kMovementEnvelopeTurnReserveRadiusMin,
                                             kMovementEnvelopeTurnReserveRadiusMax);
    const int turnReserveHalfWidth = std::clamp(safeHorizontalRadius / 4,
                                                kMovementEnvelopeTurnReserveHalfWidthMin,
                                                kMovementEnvelopeTurnReserveHalfWidthMax);
    if (horizontalDistance <= turnReserveRadius &&
        forwardDistance >= -static_cast<float>(kMovementEnvelopeRearSlackChunks) &&
        sideDistance <= static_cast<float>(turnReserveHalfWidth))
    {
        return MovementEnvelopeBucket::TurnReserve;
    }

    return MovementEnvelopeBucket::Background;
}

ColumnChunkIntervals ChunkManager::Impl::columnIntervalsFor(const glm::ivec2& column,
                                                            const glm::ivec2& cameraColumn,
                                                            int cameraChunkY,
                                                            int verticalRadius) const
{
    const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
    const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
    int columnHeight = ColumnManager::kNoHeight;
    if (!tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight))
    {
        requestColumnHeightPrefetch(column);
    }
    return columnIntervalsForHeight(column, cameraColumn, cameraChunkY, verticalRadius, columnHeight);
}

ColumnChunkIntervals ChunkManager::Impl::columnIntervalsForHeight(const glm::ivec2& column,
                                                                  const glm::ivec2& cameraColumn,
                                                                  int cameraChunkY,
                                                                  int verticalRadius,
                                                                  int columnHeight) const
{
    ColumnChunkIntervals intervals{};
    const ColumnChunkIntervals playerBand =
        playerBandIntervalsForColumn(column, cameraColumn, cameraChunkY, verticalRadius);
    mergeColumnChunkIntervals(intervals, playerBand);

    ColumnSlabOccupancy occupancy{};
    if (tryGetCachedColumnSlabOccupancy(column, occupancy))
    {
        const bool keepFullOccupiedColumn = !playerBand.empty();
        if (keepFullOccupiedColumn)
        {
            mergeColumnChunkIntervals(intervals, occupancy.occupiedIntervals);
        }
        else
        {
            mergeColumnChunkIntervals(intervals, occupancy.surfaceShellIntervals);
            mergeColumnChunkIntervals(intervals, occupancy.waterIntervals);
            mergeColumnChunkIntervals(intervals, occupancy.structureIntervals);
            mergeColumnChunkIntervals(intervals, occupancy.editIntervals);
        }
        mergeColumnChunkIntervals(intervals, occupancy.maybeIntervals);
        return intervals;
    }

    if (columnHeight != ColumnManager::kNoHeight)
    {
        const int highestChunk = floorDiv(columnHeight, kChunkSizeY);
        const int shellFloorChunk = surfaceShellFloorChunkForHeight(column, columnHeight);
        addColumnChunkInterval(intervals,
                               std::max(0, shellFloorChunk),
                               std::max(highestChunk, highestChunk + kExactSurfaceShellAirAboveChunks));
    }

    return intervals;
}

std::pair<int, int> ChunkManager::Impl::columnSpanForHeight(const glm::ivec2& column,
                                                             const glm::ivec2& cameraColumn,
                                                             int cameraChunkY,
                                                             int verticalRadius,
                                                             int columnHeight) const
{
    const ColumnChunkIntervals intervals =
        columnIntervalsForHeight(column, cameraColumn, cameraChunkY, verticalRadius, columnHeight);
    return {intervals.minChunkY(), intervals.maxChunkY()};
}

ChunkManager::Impl::RingProgress ChunkManager::Impl::ensureVolume(const glm::ivec3& center,
                                                                  int horizontalRadius,
                                                                  int verticalRadius,
                                                                  int& jobBudget,
                                                                  SchedulingBand band)
{
    bool missingFound = false;

    const glm::ivec2 cameraColumn{center.x, center.z};
    const bool exactOnly = renderSettings_.totalChunks <= renderSettings_.exactChunks;
    glm::vec2 forward = movementEnvelopeForwardXZ_;
    if (glm::dot(forward, forward) <= kEpsilon)
    {
        forward = normalizePriorityForwardXZ(lastCameraForward_);
    }
    if (glm::dot(forward, forward) <= kEpsilon)
    {
        forward = glm::vec2{0.0f, -1.0f};
    }
    const glm::vec2 side{-forward.y, forward.x};

    struct Candidate
    {
        glm::ivec3 coord;
        int bucketRank{0};
        float priority{0.0f};
        bool forceResident{false};
    };

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>((verticalRadius * 2 + 1) *
                                                std::max(1, horizontalRadius * 8)));

    std::unordered_set<glm::ivec2, ColumnHasher> visitedColumns;
    visitedColumns.reserve(static_cast<std::size_t>(std::max(1, horizontalRadius * 8)));
    const int maxJobsPerColumn = generationColumnCapThisFrame_;
    const bool enforceColumnCap = maxJobsPerColumn > 0 &&
                                  maxJobsPerColumn < std::numeric_limits<int>::max();

    auto enqueueColumn = [&](int chunkX, int chunkZ) {
        const glm::ivec2 column{chunkX, chunkZ};
        if (!visitedColumns.insert(column).second)
        {
            return;
        }

        const MovementEnvelopeBucket bucket = movementEnvelopeBucketForColumn(column, center, horizontalRadius);
        const bool protectedColumn = bucket != MovementEnvelopeBucket::Background;
        const ColumnHeightPrefetchPriority occupancyPriority =
            (protectedColumn || exactOnly) ? ColumnHeightPrefetchPriority::Visible
                                           : ColumnHeightPrefetchPriority::Normal;
        if (band == SchedulingBand::Protected && !protectedColumn)
        {
            return;
        }
        if (band == SchedulingBand::Background && protectedColumn)
        {
            return;
        }

        const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
        const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
        int columnHeight = ColumnManager::kNoHeight;
        if (!tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight))
        {
            requestColumnHeightPrefetch(column, occupancyPriority);
        }
        ColumnSlabOccupancy occupancy{};
        const bool shouldPrefetchOccupancy = !columnUsesPlayerBand(column, cameraColumn);
        if (shouldPrefetchOccupancy)
        {
            requestColumnHeightPrefetch(column, occupancyPriority);
        }
        const bool haveOccupancy = tryGetCachedColumnSlabOccupancy(column, occupancy);
        const ColumnChunkIntervals playerBand =
            playerBandIntervalsForColumn(column, cameraColumn, center.y, verticalRadius);
        const ColumnChunkIntervals intervals = columnIntervalsForHeight(column,
                                                                        cameraColumn,
                                                                        center.y,
                                                                        verticalRadius,
                                                                        columnHeight);
        for (std::uint8_t intervalIndex = 0; intervalIndex < intervals.count; ++intervalIndex)
        {
            const ChunkYInterval interval = intervals.intervals[intervalIndex];
            for (int chunkY = interval.minChunkY; chunkY <= interval.maxChunkY; ++chunkY)
            {
                const bool forceResident = chunkYWithinIntervals(chunkY, playerBand);
                if (haveOccupancy &&
                    !forceResident &&
                    classifyColumnSlab(occupancy, chunkY) == ColumnSlabOccupancyState::DefinitelyEmpty)
                {
                    continue;
                }

                const glm::ivec3 coord{chunkX, chunkY, chunkZ};
                const int dx = coord.x - center.x;
                const int dy = coord.y - center.y;
                const int dz = coord.z - center.z;
                const glm::vec2 delta(static_cast<float>(dx), static_cast<float>(dz));
                const float horizontal = std::sqrt(static_cast<float>(dx * dx + dz * dz));
                const float forwardDistance = glm::dot(delta, forward);
                const float sideDistance = std::abs(glm::dot(delta, side));
                float priority = horizontal + 0.45f * static_cast<float>(std::abs(dy));
                if (bucket == MovementEnvelopeBucket::Core)
                {
                    priority += 0.10f * sideDistance;
                }
                else if (bucket == MovementEnvelopeBucket::Corridor)
                {
                    priority += 0.15f * sideDistance - 0.18f * std::max(forwardDistance, 0.0f);
                }
                else if (bucket == MovementEnvelopeBucket::TurnReserve)
                {
                    priority += 0.20f * sideDistance + 0.10f * std::max(-forwardDistance, 0.0f);
                }
                else
                {
                    priority += 0.10f * std::max(-forwardDistance, 0.0f);
                }
                candidates.push_back(Candidate{coord,
                                               static_cast<int>(bucket),
                                               priority,
                                               forceResident});
            }
        }
    };

    if (horizontalRadius == 0)
    {
        enqueueColumn(center.x, center.z);
    }
    else
    {
        for (int dx = -horizontalRadius; dx <= horizontalRadius; ++dx)
        {
            enqueueColumn(center.x + dx, center.z - horizontalRadius);
            enqueueColumn(center.x + dx, center.z + horizontalRadius);
        }
        for (int dz = -horizontalRadius + 1; dz <= horizontalRadius - 1; ++dz)
        {
            enqueueColumn(center.x - horizontalRadius, center.z + dz);
            enqueueColumn(center.x + horizontalRadius, center.z + dz);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.bucketRank != rhs.bucketRank)
        {
            return lhs.bucketRank < rhs.bucketRank;
        }
        if (lhs.priority == rhs.priority)
        {
            if (lhs.coord.y == rhs.coord.y)
            {
                if (lhs.coord.x == rhs.coord.x)
                {
                    return lhs.coord.z < rhs.coord.z;
                }
                return lhs.coord.x < rhs.coord.x;
            }
            return lhs.coord.y < rhs.coord.y;
        }
        return lhs.priority < rhs.priority;
    });

    for (const Candidate& candidate : candidates)
    {
        if (jobBudget <= 0)
        {
            break;
        }

        const glm::ivec2 columnKey{candidate.coord.x, candidate.coord.z};
        int& columnJobs = jobsScheduledThisFrame_[columnKey];

        if (auto existing = getChunkShared(candidate.coord))
        {
            continue;
        }

        missingFound = true;

        if (enforceColumnCap && columnJobs >= maxJobsPerColumn)
        {
            continue;
        }

        if (ensureChunkAsync(candidate.coord, candidate.forceResident))
        {
            --jobBudget;
            ++columnJobs;
        }
    }

    return RingProgress{!missingFound, jobBudget <= 0};
}

void ChunkManager::Impl::removeDistantChunks(const glm::ivec3& center,
                                             int horizontalThreshold,
                                             int verticalRadius)
{
    struct EvictionCandidate
    {
        glm::ivec3 coord{0};
        int horizontalExcess{0};
        int verticalExcess{0};
    };

    std::vector<EvictionCandidate> immediateRemovals;
    std::vector<EvictionCandidate> deferredVerticalRemovals;
    const glm::ivec2 cameraColumn{center.x, center.z};
    const int evictionCenterY = updateEvictionCenterChunkY(center.y);
    const bool deferVerticalEvictions = evictionCenterY != center.y;
    const int evictionSlack =
        std::max(0, kVerticalStreamingConfig.columnSlackChunks) +
        std::max(0, kVerticalStreamingConfig.verticalEvictionExtraSlackChunks);
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        immediateRemovals.reserve(chunks_.size());
        deferredVerticalRemovals.reserve(chunks_.size());
        for (const auto& [coord, chunkPtr] : chunks_)
        {
            (void)chunkPtr;
            if (coord.y < 0)
            {
                immediateRemovals.push_back(EvictionCandidate{coord, horizontalThreshold + 1, 0});
                continue;
            }

            const int dx = coord.x - center.x;
            const int dz = coord.z - center.z;
            const int horizontalDistance = std::max(std::abs(dx), std::abs(dz));
            if (horizontalDistance > horizontalThreshold)
            {
                immediateRemovals.push_back(EvictionCandidate{
                    coord,
                    horizontalDistance - horizontalThreshold,
                    0});
                continue;
            }

            const glm::ivec2 column{coord.x, coord.z};
            const int worldX = coord.x * kChunkSizeX + kChunkSizeX / 2;
            const int worldZ = coord.z * kChunkSizeZ + kChunkSizeZ / 2;
            int columnHeight = ColumnManager::kNoHeight;
            if (!tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight))
            {
                requestColumnHeightPrefetch(column, ColumnHeightPrefetchPriority::Normal);
            }
            const ColumnChunkIntervals intervals = columnIntervalsForHeight(column,
                                                                            cameraColumn,
                                                                            evictionCenterY,
                                                                            verticalRadius,
                                                                            columnHeight);
            const int verticalExcess = chunkYDistanceToIntervals(coord.y, intervals, evictionSlack);

            if (verticalExcess > 0)
            {
                if (deferVerticalEvictions)
                {
                    deferredVerticalRemovals.push_back(EvictionCandidate{coord, 0, verticalExcess});
                }
                else
                {
                    immediateRemovals.push_back(EvictionCandidate{coord, 0, verticalExcess});
                }
            }
        }
    }

    std::sort(deferredVerticalRemovals.begin(),
              deferredVerticalRemovals.end(),
              [](const EvictionCandidate& lhs, const EvictionCandidate& rhs)
              {
                  if (lhs.verticalExcess != rhs.verticalExcess)
                  {
                      return lhs.verticalExcess > rhs.verticalExcess;
                  }
                  if (lhs.coord.y != rhs.coord.y)
                  {
                      return lhs.coord.y > rhs.coord.y;
                  }
                  if (lhs.coord.x != rhs.coord.x)
                  {
                      return lhs.coord.x < rhs.coord.x;
                  }
                  return lhs.coord.z < rhs.coord.z;
              });

    const int evictionBudget = computeEvictionBudget(deferredVerticalRemovals.size());
    if (static_cast<int>(deferredVerticalRemovals.size()) > evictionBudget)
    {
        deferredVerticalRemovals.resize(static_cast<std::size_t>(evictionBudget));
    }

    std::vector<EvictionCandidate> toRemove;
    toRemove.reserve(immediateRemovals.size() + deferredVerticalRemovals.size());
    toRemove.insert(toRemove.end(), immediateRemovals.begin(), immediateRemovals.end());
    toRemove.insert(toRemove.end(), deferredVerticalRemovals.begin(), deferredVerticalRemovals.end());

    int evictedCount = 0;
    for (const EvictionCandidate& candidate : toRemove)
    {
        const glm::ivec3& coord = candidate.coord;
        std::shared_ptr<Chunk> chunk;
        {
            std::lock_guard<std::mutex> lock(chunksMutex);
            auto it = chunks_.find(coord);
            if (it == chunks_.end())
            {
                continue;
            }

            if (it->second->inFlight.load(std::memory_order_acquire) != 0)
            {
                continue;
            }

            chunk = it->second;
            chunks_.erase(it);
        }

        if (chunk)
        {
            if (shouldTrackRecentEditChunk(chunk->coord))
            {
                const glm::ivec2 evictionCameraColumn{center.x, center.z};
                const glm::ivec2 column{chunk->coord.x, chunk->coord.z};
                const int worldX = chunk->coord.x * kChunkSizeX + kChunkSizeX / 2;
                const int worldZ = chunk->coord.z * kChunkSizeZ + kChunkSizeZ / 2;
                int columnHeight = ColumnManager::kNoHeight;
                if (!tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight))
                {
                    requestColumnHeightPrefetch(column, ColumnHeightPrefetchPriority::Background);
                }

                const auto [minChunkY, maxChunkY] =
                    columnSpanForHeight(column, evictionCameraColumn, evictionCenterY, lastVerticalRadius_, columnHeight);
                const int horizontalDistance =
                    std::max(std::abs(chunk->coord.x - center.x), std::abs(chunk->coord.z - center.z));

                std::ostringstream stream;
                stream << "evict chunk=(" << chunk->coord.x << ", " << chunk->coord.y << ", " << chunk->coord.z
                       << ") hDist=" << horizontalDistance
                       << " hThreshold=" << horizontalThreshold
                       << " hExcess=" << candidate.horizontalExcess
                       << " vExcess=" << candidate.verticalExcess
                       << " evictCenterY=" << evictionCenterY
                       << " spanY=[" << minChunkY << ", " << maxChunkY << "]"
                       << " idx=" << chunk->indexCount.load(std::memory_order_acquire);
                appendRecentEditDebugEvent(stream.str());
            }

            columnManager_.removeChunk(chunk->coord);
            recycleChunkGPU(*chunk);
            recycleChunkObject(std::move(chunk));
            ++evictedCount;
        }
    }

    if (evictedCount > 0)
    {
        profilingCounters_.evictedChunks.fetch_add(evictedCount, std::memory_order_relaxed);
    }
}

bool ChunkManager::Impl::ensureChunkAsync(const glm::ivec3& coord, bool forceResident)
{
    if (coord.y < 0)
    {
        return false;
    }

    if (!forceResident)
    {
        ColumnSlabOccupancy occupancy{};
        if (tryGetCachedColumnSlabOccupancy({coord.x, coord.z}, occupancy) &&
            classifyColumnSlab(occupancy, coord.y) == ColumnSlabOccupancyState::DefinitelyEmpty)
        {
            return false;
        }
    }

    try
    {
        std::shared_ptr<Chunk> chunk;
        {
            std::lock_guard<std::mutex> lock(chunksMutex);
            auto it = chunks_.find(coord);
            if (it != chunks_.end())
            {
                return false;
            }

            chunk = acquireChunk(coord);
            chunk->state.store(ChunkState::Generating, std::memory_order_release);
            chunk->requestTimestampMicros.store(static_cast<long long>(steadyMicrosNow()), std::memory_order_release);
            chunk->initialReadyRecorded.store(false, std::memory_order_release);
            chunks_.emplace(coord, chunk);
        }

        const std::uint32_t generationEpoch =
            chunk->generationEpoch.fetch_add(1, std::memory_order_acq_rel) + 1u;
        enqueueJob(chunk, JobType::Generate, coord, generationEpoch, true);
        return true;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error creating chunk at (" << coord.x << ", " << coord.y << ", " << coord.z
                  << "): " << ex.what() << std::endl;
        return false;
    }
}

void ChunkManager::Impl::uploadReadyMeshes()
{
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();
    uploadQueueAgeMsLastFrame_ = 0.0;
    uploadAttemptsLastFrame_ = 0;
    uploadQueueScanEntriesLastFrame_ = 0;
    uploadSkippedExpiredLastFrame_ = 0;
    uploadSkippedNotReadyLastFrame_ = 0;
    uploadSkippedPendingMeshLastFrame_ = 0;
    uploadColumnLimitedLastFrame_ = 0;
    uploadBudgetDeferredLastFrame_ = 0;
    uploadRetryFailuresLastFrame_ = 0;
    uploadScanLimitHitsLastFrame_ = 0;
    uploadBeginFailuresLastFrame_ = 0;
    uploadStalePendingMeshesLastFrame_ = 0;
    uploadQueuePickMsLastFrame_ = 0.0;
    uploadPrepareMsLastFrame_ = 0.0;
    uploadContextBeginMsLastFrame_ = 0.0;
    uploadFinalizeMsLastFrame_ = 0.0;
    commitCollectMsLastFrame_ = 0.0;
    commitChunkScanMsLastFrame_ = 0.0;
    commitMeshLockWaitMsLastFrame_ = 0.0;
    commitMeshLockedMsLastFrame_ = 0.0;
    commitMeshStateMsLastFrame_ = 0.0;
    commitPageStateMsLastFrame_ = 0.0;
    commitReleaseMsLastFrame_ = 0.0;

    const auto uploadPrepareStart = std::chrono::steady_clock::now();
    commitPendingChunkUploads();
    uploadPrepareMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - uploadPrepareStart).count();

    const std::size_t initialBudget = uploadBudgetBytesThisFrame_;
    std::size_t remainingBudget = initialBudget;
    bool uploadedAnything = false;
    int uploadedChunkCount = 0;
    std::unordered_map<glm::ivec2, int, ColumnHasher> uploadsPerColumn;
    std::size_t attempts = 0;
    const UINT64 uploadBatchId = nextUploadBatchId_++;
    std::vector<std::shared_ptr<Chunk>> stagedChunks;
    const int columnUploadLimit = std::max(1, uploadColumnLimitThisFrame_);
    const int chunkUploadLimit = std::max(1, uploadChunkLimitThisFrame_);
    const auto uploadStart = std::chrono::steady_clock::now();
    const auto uploadContextBeginStart = std::chrono::steady_clock::now();
    if (uploadContext_.ready() && !uploadContext_.begin())
    {
        uploadBeginFailuresLastFrame_ = 1;
        lastUploadBytesUsed_ = 0;
        lastUploadMsUsed_ = 0.0;
        pendingUploadsLastFrame_ = estimateUploadQueueSize();
        if (benchmarkEnabled)
        {
            benchmarkMetrics_.uploadBeginFailuresPerFrame.record(1);
            benchmarkMetrics_.uploadAttemptsPerFrame.record(0);
            benchmarkMetrics_.uploadChunksPerFrame.record(0);
            benchmarkMetrics_.uploadBytesPerFrame.record(0);
            benchmarkMetrics_.uploadQueueScanEntries.record(
                static_cast<std::uint64_t>(std::max(uploadQueueScanEntriesLastFrame_, 0)));
            benchmarkMetrics_.uploadExpiredEntriesPerFrame.record(
                static_cast<std::uint64_t>(std::max(uploadSkippedExpiredLastFrame_, 0)));
            benchmarkMetrics_.uploadSkippedNotReadyPerFrame.record(0);
            benchmarkMetrics_.uploadSkippedPendingMeshPerFrame.record(0);
            benchmarkMetrics_.uploadColumnLimitedPerFrame.record(0);
            benchmarkMetrics_.uploadBudgetDeferredPerFrame.record(0);
            benchmarkMetrics_.uploadRetryFailuresPerFrame.record(0);
            benchmarkMetrics_.uploadScanLimitHitsPerFrame.record(0);
            benchmarkMetrics_.uploadStalePendingMeshesPerFrame.record(
                static_cast<std::uint64_t>(std::max(uploadStalePendingMeshesLastFrame_, 0)));
        }
        return;
    }
    uploadContextBeginMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - uploadContextBeginStart).count();

    while ((remainingBudget > 0 || !uploadedAnything) && attempts < kUploadQueueScanLimit)
    {
        if (uploadedChunkCount >= chunkUploadLimit && uploadedAnything)
        {
            break;
        }

        ++attempts;
        uploadAttemptsLastFrame_ = static_cast<int>(attempts);
        std::shared_ptr<Chunk> chunk;
        if (benchmarkEnabled)
        {
            const auto uploadPickStart = SteadyClock::now();
            chunk = popNextChunkForUpload();
            uploadQueuePickMsLastFrame_ +=
                std::chrono::duration<double, std::milli>(SteadyClock::now() - uploadPickStart).count();
        }
        else
        {
            chunk = popNextChunkForUpload();
        }
        if (!chunk)
        {
            break;
        }

        if (!chunk->meshReady.load(std::memory_order_acquire) ||
            chunk->state.load(std::memory_order_acquire) != ChunkState::Ready)
        {
            ++uploadSkippedNotReadyLastFrame_;
            continue;
        }

        {
            std::lock_guard<std::mutex> meshLock(chunk->meshMutex);
            if (chunk->pendingMesh.valid())
            {
                ++uploadSkippedPendingMeshLastFrame_;
                requeueChunkForUpload(chunk, false);
                continue;
            }
        }

        const glm::ivec2 columnKey{chunk->coord.x, chunk->coord.z};
        int& columnUploads = uploadsPerColumn[columnKey];
        if (columnUploads >= columnUploadLimit)
        {
            ++uploadColumnLimitedLastFrame_;
            requeueChunkForUpload(chunk, false);
            profilingCounters_.throttledUploads.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        std::size_t vertexBytes = 0;
        std::size_t indexBytes = 0;
        {
            std::lock_guard<std::mutex> meshLock(chunk->meshMutex);
            vertexBytes = chunk->meshData.vertices.size() * sizeof(Vertex);
            indexBytes = chunk->meshData.indices.size() * sizeof(std::uint32_t);
        }
        const std::size_t totalBytes = vertexBytes + indexBytes;

        const bool allowOversizeUpload = !uploadedAnything && pendingUploadsLastFrame_ <= 2;
        if (totalBytes > remainingBudget && totalBytes > 0 && !allowOversizeUpload)
        {
            ++uploadBudgetDeferredLastFrame_;
            requeueChunkForUpload(chunk, false);
            profilingCounters_.deferredUploads.fetch_add(1, std::memory_order_relaxed);
            if (uploadedAnything)
            {
                break;
            }
            continue;
        }

        const auto uploadChunkStart = benchmarkEnabled ? SteadyClock::now() : SteadyClock::time_point{};
        if (!uploadChunkMesh(*chunk, uploadBatchId))
        {
            ++uploadRetryFailuresLastFrame_;
            requeueChunkForUpload(chunk, true);
            if (uploadedAnything)
            {
                break;
            }
            continue;
        }
        chunk->state.store(ChunkState::Uploaded, std::memory_order_release);
        chunk->meshReady.store(false, std::memory_order_release);
        uploadedAnything = true;
        ++uploadedChunkCount;
        ++columnUploads;
        stagedChunks.push_back(chunk);

        profilingCounters_.uploadedChunks.fetch_add(1, std::memory_order_relaxed);
        profilingCounters_.uploadedBytes.fetch_add(totalBytes, std::memory_order_relaxed);
        if (benchmarkEnabled)
        {
            const auto uploadChunkEnd = SteadyClock::now();
            const auto uploadMicros =
                std::chrono::duration_cast<std::chrono::microseconds>(uploadChunkEnd - uploadChunkStart).count();
            benchmarkMetrics_.uploadStage.recordMicros(static_cast<std::uint64_t>(uploadMicros));
            benchmarkMetrics_.uploadedChunks.fetch_add(1, std::memory_order_relaxed);
            benchmarkMetrics_.uploadedBytes.fetch_add(totalBytes, std::memory_order_relaxed);
        }

        if (totalBytes >= remainingBudget)
        {
            remainingBudget = 0;
        }
        else
        {
            remainingBudget -= totalBytes;
        }

        const double elapsedMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - uploadStart).count();
        if (elapsedMs >= uploadBudgetMsThisFrame_ && uploadedAnything)
        {
            break;
        }
    }

    if (attempts >= kUploadQueueScanLimit && estimateUploadQueueSize() > 0)
    {
        uploadScanLimitHitsLastFrame_ = 1;
    }

    const auto uploadFinalizeStart = std::chrono::steady_clock::now();
    uploadContext_.flush();
    if (uploadedAnything)
    {
        const UINT64 submittedFenceValue = uploadContext_.lastSubmittedFenceValue();
        sealPendingChunkUploadPages(uploadBatchId, submittedFenceValue);
        for (const std::shared_ptr<Chunk>& chunk : stagedChunks)
        {
            if (!chunk)
            {
                continue;
            }

            UINT64 chunkFenceValue = 0;
            std::lock_guard<std::mutex> meshLock(chunk->meshMutex);
            if (chunk->pendingMesh.valid() && chunk->pendingMesh.uploadFenceValue == 0)
            {
                chunk->pendingMesh.uploadFenceValue = submittedFenceValue;
                chunkFenceValue = submittedFenceValue;
            }
            if (chunkFenceValue != 0)
            {
                queueChunkForCommit(chunk, chunkFenceValue);
            }
        }
    }

    if (initialBudget > remainingBudget)
    {
        lastUploadBytesUsed_ = initialBudget - remainingBudget;
    }
    else
    {
        lastUploadBytesUsed_ = 0;
    }
    lastUploadMsUsed_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - uploadStart).count();
    uploadFinalizeMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - uploadFinalizeStart).count();

    pendingUploadsLastFrame_ = estimateUploadQueueSize();
    if (benchmarkEnabled)
    {
        benchmarkMetrics_.uploadQueueScanEntries.record(
            static_cast<std::uint64_t>(std::max(uploadQueueScanEntriesLastFrame_, 0)));
        benchmarkMetrics_.uploadAttemptsPerFrame.record(static_cast<std::uint64_t>(std::max(uploadAttemptsLastFrame_, 0)));
        benchmarkMetrics_.uploadChunksPerFrame.record(static_cast<std::uint64_t>(std::max(uploadedChunkCount, 0)));
        benchmarkMetrics_.uploadBytesPerFrame.record(static_cast<std::uint64_t>(lastUploadBytesUsed_));
        benchmarkMetrics_.uploadExpiredEntriesPerFrame.record(
            static_cast<std::uint64_t>(std::max(uploadSkippedExpiredLastFrame_, 0)));
        benchmarkMetrics_.uploadSkippedNotReadyPerFrame.record(
            static_cast<std::uint64_t>(std::max(uploadSkippedNotReadyLastFrame_, 0)));
        benchmarkMetrics_.uploadSkippedPendingMeshPerFrame.record(
            static_cast<std::uint64_t>(std::max(uploadSkippedPendingMeshLastFrame_, 0)));
        benchmarkMetrics_.uploadColumnLimitedPerFrame.record(
            static_cast<std::uint64_t>(std::max(uploadColumnLimitedLastFrame_, 0)));
        benchmarkMetrics_.uploadBudgetDeferredPerFrame.record(
            static_cast<std::uint64_t>(std::max(uploadBudgetDeferredLastFrame_, 0)));
        benchmarkMetrics_.uploadRetryFailuresPerFrame.record(
            static_cast<std::uint64_t>(std::max(uploadRetryFailuresLastFrame_, 0)));
        benchmarkMetrics_.uploadScanLimitHitsPerFrame.record(
            static_cast<std::uint64_t>(std::max(uploadScanLimitHitsLastFrame_, 0)));
        benchmarkMetrics_.uploadBeginFailuresPerFrame.record(
            static_cast<std::uint64_t>(std::max(uploadBeginFailuresLastFrame_, 0)));
        benchmarkMetrics_.uploadStalePendingMeshesPerFrame.record(
            static_cast<std::uint64_t>(std::max(uploadStalePendingMeshesLastFrame_, 0)));
    }
}

void ChunkManager::Impl::commitPendingChunkUploads()
{
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();

    const auto commitCollectStart = benchmarkEnabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    collectReusableChunkBufferPages();
    collectDeferredPendingChunkReleases();
    const UINT64 completedUploadFenceValue = uploadContext_.completedFenceValue();
    if (benchmarkEnabled)
    {
        commitCollectMsLastFrame_ +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - commitCollectStart).count();
    }

    while (true)
    {
        const auto commitChunkScanStart =
            benchmarkEnabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        PendingCommitQueueEntry entry{};
        bool haveEntry = false;
        {
            std::lock_guard<std::mutex> lock(pendingCommitQueueMutex_);
            if (!pendingCommitQueue_.empty())
            {
                const PendingCommitQueueEntry& front = pendingCommitQueue_.front();
                if (front.uploadFenceValue == 0 || completedUploadFenceValue >= front.uploadFenceValue)
                {
                    entry = front;
                    pendingCommitQueue_.pop_front();
                    haveEntry = true;
                }
            }
        }
        if (benchmarkEnabled)
        {
            commitChunkScanMsLastFrame_ +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - commitChunkScanStart).count();
        }

        if (!haveEntry)
        {
            break;
        }

        const std::shared_ptr<Chunk> chunk = entry.chunk.lock();
        if (!chunk)
        {
            continue;
        }

        if (!chunk->queuedForCommit.load(std::memory_order_acquire) ||
            chunk->commitQueueTicket.load(std::memory_order_acquire) != entry.ticket)
        {
            continue;
        }

        Chunk::PendingRenderMesh pendingMesh{};
        std::uint32_t currentMeshVersion = 0;
        std::uint32_t oldPageIndex = kInvalidChunkBufferPage;
        std::size_t oldVertexOffset = 0;
        std::size_t oldIndexOffset = 0;
        std::size_t oldVertexCount = 0;
        std::size_t oldIndexCount = 0;
        bool stalePending = false;

        const auto commitMeshLockWaitStart =
            benchmarkEnabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        std::unique_lock<std::mutex> meshLock(chunk->meshMutex);
        if (benchmarkEnabled)
        {
            const double meshLockWaitMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          commitMeshLockWaitStart)
                    .count();
            commitMeshLockWaitMsLastFrame_ += meshLockWaitMs;
            commitMeshStateMsLastFrame_ += meshLockWaitMs;
        }

        const auto commitMeshStateStart =
            benchmarkEnabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (!chunk->pendingMesh.valid())
        {
            chunk->queuedForCommit.store(false, std::memory_order_release);
            chunk->commitQueueTicket.store(0, std::memory_order_release);
            if (benchmarkEnabled)
            {
                const double meshLockedMs =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              commitMeshStateStart)
                        .count();
                commitMeshLockedMsLastFrame_ += meshLockedMs;
                commitMeshStateMsLastFrame_ += meshLockedMs;
            }
            continue;
        }

        if (chunk->pendingMesh.uploadFenceValue != 0 &&
            completedUploadFenceValue < chunk->pendingMesh.uploadFenceValue)
        {
            const UINT64 deferredFenceValue = chunk->pendingMesh.uploadFenceValue;
            chunk->queuedForCommit.store(false, std::memory_order_release);
            chunk->commitQueueTicket.store(0, std::memory_order_release);
            if (benchmarkEnabled)
            {
                const double meshLockedMs =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              commitMeshStateStart)
                        .count();
                commitMeshLockedMsLastFrame_ += meshLockedMs;
                commitMeshStateMsLastFrame_ += meshLockedMs;
            }
            meshLock.unlock();
            queueChunkForCommit(chunk, deferredFenceValue);
            continue;
        }

        pendingMesh = chunk->pendingMesh;
        chunk->pendingMesh = {};
        chunk->queuedForCommit.store(false, std::memory_order_release);
        chunk->commitQueueTicket.store(0, std::memory_order_release);
        currentMeshVersion = chunk->meshVersion.load(std::memory_order_acquire);
        stalePending = currentMeshVersion != pendingMesh.meshVersion;
        if (!stalePending)
        {
            oldPageIndex = chunk->bufferPageIndex.load(std::memory_order_acquire);
            oldVertexOffset = chunk->vertexOffset.load(std::memory_order_acquire);
            oldIndexOffset = chunk->indexOffset.load(std::memory_order_acquire);
            oldVertexCount = chunk->vertexCount.load(std::memory_order_acquire);
            oldIndexCount = static_cast<std::size_t>(chunk->indexCount.load(std::memory_order_acquire));
            chunk->bufferPageIndex.store(pendingMesh.pageIndex, std::memory_order_release);
            chunk->vertexOffset.store(pendingMesh.vertexOffset, std::memory_order_release);
            chunk->indexOffset.store(pendingMesh.indexOffset, std::memory_order_release);
            chunk->vertexCount.store(pendingMesh.vertexCount, std::memory_order_release);
            chunk->indexCount.store(static_cast<std::uint32_t>(pendingMesh.indexCount), std::memory_order_release);
        }
        if (benchmarkEnabled)
        {
            const double meshLockedMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          commitMeshStateStart)
                    .count();
            commitMeshLockedMsLastFrame_ += meshLockedMs;
            commitMeshStateMsLastFrame_ += meshLockedMs;
        }
        meshLock.unlock();

        const auto commitPageStateStart =
            benchmarkEnabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (!stalePending && pendingMesh.pageIndex < bufferPages_.size())
        {
            std::lock_guard<std::mutex> pageLock(bufferPageMutex_);
            if (pendingMesh.pageIndex < bufferPages_.size())
            {
                ChunkBufferPage& page = bufferPages_[pendingMesh.pageIndex];
                if (page.pendingChunks > 0)
                {
                    --page.pendingChunks;
                }
                if (!stalePending)
                {
                    ++page.residentChunks;
                    page.state = ChunkBufferPageState::Resident;
                    page.uploadFenceValue = 0;
                    page.pendingBatchId = 0;
                    releaseChunkBufferPageUploadBuffers(page);
                }
            }
        }
        if (benchmarkEnabled)
        {
            commitPageStateMsLastFrame_ +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - commitPageStateStart)
                    .count();
        }

        const auto commitReleaseStart =
            benchmarkEnabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (stalePending)
        {
            ++uploadStalePendingMeshesLastFrame_;
            if (pendingMesh.pageIndex != kInvalidChunkBufferPage)
            {
                releaseChunkAllocationRange(pendingMesh.pageIndex,
                                            pendingMesh.vertexOffset,
                                            pendingMesh.vertexCount,
                                            pendingMesh.indexOffset,
                                            pendingMesh.indexCount,
                                            false);
            }
            if (chunk->meshReady.load(std::memory_order_acquire) &&
                chunk->state.load(std::memory_order_acquire) == ChunkState::Ready)
            {
                queueChunkForUpload(chunk, true);
            }
            if (benchmarkEnabled)
            {
                commitReleaseMsLastFrame_ +=
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - commitReleaseStart)
                        .count();
            }
            continue;
        }

        if (oldPageIndex != kInvalidChunkBufferPage)
        {
            releaseChunkAllocationRange(oldPageIndex,
                                        oldVertexOffset,
                                        oldVertexCount,
                                        oldIndexOffset,
                                        oldIndexCount,
                                        true);
        }

        noteChunkReadyLatency(*chunk);
        if (benchmarkEnabled)
        {
            commitReleaseMsLastFrame_ +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - commitReleaseStart)
                    .count();
        }
    }
}

bool ChunkManager::Impl::uploadChunkMesh(Chunk& chunk, UINT64 uploadBatchId)
{
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();
    const std::uint64_t uploadStartMicros = steadyMicrosNow();
    storeFirstBenchmarkTimestamp(chunk.uploadStartTimestampMicros, uploadStartMicros);
    const std::uint64_t queuedMicros = loadBenchmarkTimestamp(chunk.uploadQueuedTimestampMicros);
    if (queuedMicros > 0 && uploadStartMicros > queuedMicros)
    {
        const std::uint64_t queueAgeMicros = uploadStartMicros - queuedMicros;
        uploadQueueAgeMsLastFrame_ =
            std::max(uploadQueueAgeMsLastFrame_, static_cast<double>(queueAgeMicros) / 1000.0);
        if (benchmarkEnabled)
        {
            benchmarkMetrics_.uploadQueueAgeStage.recordMicros(queueAgeMicros);
        }
    }
    const auto meshLockStart = benchmarkEnabled ? SteadyClock::now() : SteadyClock::time_point{};
    const auto recordMeshLock = [&]()
    {
        if (!benchmarkEnabled)
        {
            return;
        }

        const auto lockMicros =
            std::chrono::duration_cast<std::chrono::microseconds>(SteadyClock::now() - meshLockStart).count();
        benchmarkMetrics_.uploadChunkMeshLockStage.recordMicros(static_cast<std::uint64_t>(lockMicros));
    };
    std::lock_guard<std::mutex> lock(chunk.meshMutex);
    const std::uint32_t oldPageIndex = chunk.bufferPageIndex.load(std::memory_order_acquire);
    const std::uint32_t oldIndexCount = chunk.indexCount.load(std::memory_order_acquire);

    if (chunk.meshData.empty())
    {
        if (shouldTrackRecentEditChunk(chunk.coord))
        {
            std::ostringstream stream;
            stream << "upload skipped empty chunk=(" << chunk.coord.x << ", " << chunk.coord.y << ", " << chunk.coord.z
                   << ") oldPage=" << oldPageIndex
                   << " oldIdx=" << oldIndexCount;
            appendRecentEditDebugEvent(stream.str());
        }
        recordMeshLock();
        return false;
    }

    const std::size_t vertexCount = chunk.meshData.vertices.size();
    const std::size_t indexCount = chunk.meshData.indices.size();
    const std::uint32_t meshVersion = chunk.meshVersion.load(std::memory_order_acquire);

    ChunkAllocation allocation = acquireChunkAllocation(vertexCount, indexCount, uploadBatchId);
    if (allocation.pageIndex == kInvalidChunkBufferPage)
    {
        if (exactUploadDebugLoggingEnabled())
        {
            std::lock_guard<std::mutex> pageLock(bufferPageMutex_);
            std::ostringstream stream;
            stream << "exact upload allocation failed"
                   << " chunk=(" << chunk.coord.x << "," << chunk.coord.y << "," << chunk.coord.z << ")"
                   << " requestVerts=" << vertexCount
                   << " requestIdx=" << indexCount
                   << " uploadBatchId=" << uploadBatchId
                   << " | " << summarizeChunkBufferPagesLocked();
            exactUploadDebugLog(stream.str());
        }
        if (shouldTrackRecentEditChunk(chunk.coord))
        {
            std::ostringstream stream;
            stream << "upload alloc failed chunk=(" << chunk.coord.x << ", " << chunk.coord.y << ", " << chunk.coord.z
                   << ") oldPage=" << oldPageIndex
                   << " verts=" << vertexCount
                   << " idx=" << indexCount;
            appendRecentEditDebugEvent(stream.str());
        }
        recordMeshLock();
        return false;
    }

    std::lock_guard<std::mutex> pageLock(bufferPageMutex_);
    if (allocation.pageIndex >= bufferPages_.size())
    {
        recordMeshLock();
        return false;
    }

    ChunkBufferPage& page = bufferPages_[allocation.pageIndex];
    if (page.mappedVertexData != nullptr && vertexCount > 0)
    {
        std::memcpy(page.mappedVertexData + allocation.vertexOffset * sizeof(Vertex),
                    chunk.meshData.vertices.data(),
                    vertexCount * sizeof(Vertex));
        if (uploadContext_.ready() && page.vertexUploadBuffer != nullptr && page.vertexBuffer != nullptr)
        {
            uploadContext_.copyBuffer(page.vertexBuffer.Get(),
                                      static_cast<std::uint64_t>(allocation.vertexOffset * sizeof(Vertex)),
                                      page.vertexUploadBuffer.Get(),
                                      static_cast<std::uint64_t>(allocation.vertexOffset * sizeof(Vertex)),
                                      static_cast<std::uint64_t>(vertexCount * sizeof(Vertex)));
        }
    }
    if (page.mappedIndexData != nullptr && indexCount > 0)
    {
        std::memcpy(page.mappedIndexData + allocation.indexOffset * sizeof(std::uint32_t),
                    chunk.meshData.indices.data(),
                    indexCount * sizeof(std::uint32_t));
        if (uploadContext_.ready() && page.indexUploadBuffer != nullptr && page.indexBuffer != nullptr)
        {
            uploadContext_.copyBuffer(page.indexBuffer.Get(),
                                      static_cast<std::uint64_t>(allocation.indexOffset * sizeof(std::uint32_t)),
                                      page.indexUploadBuffer.Get(),
                                      static_cast<std::uint64_t>(allocation.indexOffset * sizeof(std::uint32_t)),
                                      static_cast<std::uint64_t>(indexCount * sizeof(std::uint32_t)));
        }
    }

    chunk.pendingMesh.pageIndex = allocation.pageIndex;
    chunk.pendingMesh.vertexOffset = allocation.vertexOffset;
    chunk.pendingMesh.indexOffset = allocation.indexOffset;
    chunk.pendingMesh.vertexCount = vertexCount;
    chunk.pendingMesh.indexCount = indexCount;
    chunk.pendingMesh.meshVersion = meshVersion;
    chunk.pendingMesh.uploadFenceValue = 0;
    chunk.meshData.clear();
    recordMeshLock();

    if (shouldTrackRecentEditChunk(chunk.coord))
    {
        std::ostringstream stream;
        stream << "upload complete chunk=(" << chunk.coord.x << ", " << chunk.coord.y << ", " << chunk.coord.z
               << ") oldPage=" << oldPageIndex
               << " pendingPage=" << allocation.pageIndex
               << " idx=" << indexCount;
        appendRecentEditDebugEvent(stream.str());
    }
    return true;
}

ChunkManager::Impl::ChunkNeighborhoodSnapshot ChunkManager::Impl::captureChunkNeighborhoodSnapshot(
    const Chunk& chunk,
    const std::vector<BlockId>& centerBlocks,
    const std::vector<std::uint8_t>& centerLightLevels)
{
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();
    ChunkNeighborhoodSnapshot snapshot(chunk.minWorldY);

    for (int localX = 0; localX < kChunkSizeX; ++localX)
    {
        for (int localY = 0; localY < kChunkSizeY; ++localY)
        {
            for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
            {
                const std::size_t voxelIndex = blockIndex(localX, localY, localZ);
                snapshot.set(localX, localY, localZ, centerBlocks[voxelIndex], centerLightLevels[voxelIndex]);
            }
        }
    }

    std::unordered_map<glm::ivec3, std::shared_ptr<Chunk>, ChunkHasher> neighborhoodChunks;
    neighborhoodChunks.reserve(27);
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    const glm::ivec3 sampleCoord = chunk.coord + glm::ivec3(dx, dy, dz);
                    auto it = chunks_.find(sampleCoord);
                    if (it != chunks_.end())
                    {
                        neighborhoodChunks.emplace(sampleCoord, it->second);
                    }
                }
            }
        }
    }

    std::vector<std::shared_ptr<Chunk>> lockedNeighbors;
    lockedNeighbors.reserve(neighborhoodChunks.size());
    for (const auto& [coord, sampleChunk] : neighborhoodChunks)
    {
        if (!sampleChunk || sampleChunk.get() == &chunk)
        {
            continue;
        }

        lockedNeighbors.push_back(sampleChunk);
    }

    std::sort(lockedNeighbors.begin(),
              lockedNeighbors.end(),
              [](const std::shared_ptr<Chunk>& lhs, const std::shared_ptr<Chunk>& rhs)
              {
                  if (lhs->coord.x != rhs->coord.x)
                  {
                      return lhs->coord.x < rhs->coord.x;
                  }
                  if (lhs->coord.y != rhs->coord.y)
                  {
                      return lhs->coord.y < rhs->coord.y;
                  }
                  return lhs->coord.z < rhs->coord.z;
              });

    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(lockedNeighbors.size());
    const auto lockStageStart = benchmarkEnabled ? SteadyClock::now() : SteadyClock::time_point{};
    for (const auto& sampleChunk : lockedNeighbors)
    {
        if (sampleChunk && !sampleChunk->cpuDataResident)
        {
            (void)ensureChunkCpuDataResident(*sampleChunk);
        }
        locks.emplace_back(sampleChunk->meshMutex);
    }

    const int baseWorldX = chunk.coord.x * kChunkSizeX;
    const int baseWorldY = chunk.minWorldY;
    const int baseWorldZ = chunk.coord.z * kChunkSizeZ;

    for (int sampleX = -1; sampleX <= kChunkSizeX; ++sampleX)
    {
        for (int sampleY = -1; sampleY <= kChunkSizeY; ++sampleY)
        {
            for (int sampleZ = -1; sampleZ <= kChunkSizeZ; ++sampleZ)
            {
                if (sampleX >= 0 && sampleX < kChunkSizeX &&
                    sampleY >= 0 && sampleY < kChunkSizeY &&
                    sampleZ >= 0 && sampleZ < kChunkSizeZ)
                {
                    continue;
                }

                const glm::ivec3 worldPos(baseWorldX + sampleX, baseWorldY + sampleY, baseWorldZ + sampleZ);
                if (worldPos.y < 0)
                {
                    snapshot.set(sampleX, sampleY, sampleZ, BlockId::Air, packLightLevels(0, 0));
                    continue;
                }

                const glm::ivec3 sampleChunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
                auto neighborhoodIt = neighborhoodChunks.find(sampleChunkCoord);
                if (neighborhoodIt == neighborhoodChunks.end() || !neighborhoodIt->second)
                {
                    continue;
                }

                const Chunk& sampleChunk = *neighborhoodIt->second;
                if (worldPos.y < sampleChunk.minWorldY || worldPos.y > sampleChunk.maxWorldY)
                {
                    continue;
                }

                const glm::ivec3 local = localBlockCoords(worldPos, sampleChunkCoord);
                if (local.x < 0 || local.x >= kChunkSizeX ||
                    local.z < 0 || local.z >= kChunkSizeZ)
                {
                    continue;
                }

                const int localY = worldPos.y - sampleChunk.minWorldY;
                const std::size_t voxelIndex = blockIndex(local.x, localY, local.z);
                snapshot.set(sampleX,
                             sampleY,
                             sampleZ,
                             sampleChunk.blocks[voxelIndex],
                             sampleChunk.lightLevels[voxelIndex]);
            }
        }
    }

    if (benchmarkEnabled)
    {
        const auto lockMicros =
            std::chrono::duration_cast<std::chrono::microseconds>(SteadyClock::now() - lockStageStart).count();
        benchmarkMetrics_.neighborhoodSnapshotLockStage.recordMicros(static_cast<std::uint64_t>(lockMicros));
    }

    return snapshot;
}

void ChunkManager::Impl::buildChunkMeshAsync(Chunk& chunk)
{
    if (!ensureChunkCpuDataResident(chunk))
    {
        return;
    }
    std::vector<BlockId> chunkBlocks;
    std::vector<std::uint8_t> chunkLightLevels;
    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        if (!chunk.hasBlocks.load(std::memory_order_acquire))
        {
            chunk.meshData.clear();
            chunk.meshReady.store(true, std::memory_order_release);
            return;
        }

        chunkBlocks = chunk.blocks;
        chunkLightLevels = chunk.lightLevels;
    }

    const ChunkNeighborhoodSnapshot neighborhood =
        captureChunkNeighborhoodSnapshot(chunk, chunkBlocks, chunkLightLevels);

    MeshData meshData;

    const int baseWorldX = chunk.coord.x * kChunkSizeX;
    const int baseWorldY = chunk.minWorldY;
    const int baseWorldZ = chunk.coord.z * kChunkSizeZ;
    const glm::vec3 chunkOrigin(static_cast<float>(baseWorldX), static_cast<float>(baseWorldY), static_cast<float>(baseWorldZ));
    std::array<std::uint8_t, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)> grassTintByColumn{};
    for (int localX = 0; localX < kChunkSizeX; ++localX)
    {
        for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
        {
            const ColumnSample columnSample = sampleColumn(baseWorldX + localX,
                                                           baseWorldZ + localZ,
                                                           baseWorldY,
                                                           baseWorldY + kChunkSizeY - 1);
            grassTintByColumn[static_cast<std::size_t>(localZ * kChunkSizeX + localX)] =
                static_cast<std::uint8_t>(grassTintIndexForBiome(columnSample.dominantBiome));
        }
    }

    auto grassTintForLocal = [&](int localX, int localZ) noexcept -> GrassTintIndex
    {
        if (localX < 0 || localX >= kChunkSizeX || localZ < 0 || localZ >= kChunkSizeZ)
        {
            return GrassTintIndex::Default;
        }

        return static_cast<GrassTintIndex>(
            grassTintByColumn[static_cast<std::size_t>(localZ * kChunkSizeX + localX)]);
    };

    auto isInsideChunk = [](const glm::ivec3& local) noexcept
    {
        return local.x >= 0 && local.x < kChunkSizeX &&
               local.y >= 0 && local.y < kChunkSizeY &&
               local.z >= 0 && local.z < kChunkSizeZ;
    };

    auto sampleBlock = [&](int lx, int ly, int lz) -> BlockId
    {
        return neighborhood.blockAt(lx, ly, lz);
    };

    auto samplePackedLight = [&](int lx, int ly, int lz) -> std::uint8_t
    {
        return neighborhood.lightAt(lx, ly, lz);
    };

    enum class Axis : int { X = 0, Y = 1, Z = 2 };
    enum class FaceDir : int { Negative = 0, Positive = 1 };

    struct FaceMaterial
    {
        glm::vec2 uvBase{0.0f};
        glm::vec2 uvSize{1.0f};
        glm::ivec3 uAxis{1, 0, 0};
        glm::ivec3 vAxis{0, 1, 0};
        BlockFace face{BlockFace::Top};
        std::array<std::uint32_t, 4> lightingData{};
        std::uint8_t flags{0};
        bool mergeable{true};

        bool operator==(const FaceMaterial& other) const noexcept
        {
            return uvBase == other.uvBase &&
                   uvSize == other.uvSize &&
                   uAxis == other.uAxis &&
                   vAxis == other.vAxis &&
                   face == other.face &&
                   lightingData == other.lightingData &&
                   flags == other.flags &&
                   mergeable == other.mergeable;
        }
    };

    struct MaskCell
    {
        bool exists{false};
        FaceMaterial material{};
    };

	    const std::array<glm::vec3, 3> axisNormals{
	        glm::vec3{1.0f, 0.0f, 0.0f},
	        glm::vec3{0.0f, 1.0f, 0.0f},
	        glm::vec3{0.0f, 0.0f, 1.0f}
	    };

        constexpr std::array<int, 4> kCornerUSigns{-1, 1, 1, -1};
        constexpr std::array<int, 4> kCornerVSigns{-1, -1, 1, 1};

	    auto faceFromNormal = [](const glm::vec3& normal) noexcept -> BlockFace
	    {
	        if (normal.y > 0.5f) return BlockFace::Top;
	        if (normal.y < -0.5f) return BlockFace::Bottom;
	        if (normal.x > 0.5f) return BlockFace::East;
	        if (normal.x < -0.5f) return BlockFace::West;
	        if (normal.z > 0.5f) return BlockFace::South;
	        return BlockFace::North;
	    };

	    auto faceSampleAxes = [](BlockFace face, glm::ivec3& uAxis, glm::ivec3& vAxis) noexcept
	    {
	        switch (face)
	        {
	        case BlockFace::Top:
	        case BlockFace::Bottom:
	            uAxis = glm::ivec3(1, 0, 0);
	            vAxis = glm::ivec3(0, 0, 1);
	            break;
	        case BlockFace::East:
	        case BlockFace::West:
	            uAxis = glm::ivec3(0, 1, 0);
	            vAxis = glm::ivec3(0, 0, 1);
	            break;
	        case BlockFace::South:
	        case BlockFace::North:
	        default:
	            uAxis = glm::ivec3(1, 0, 0);
	            vAxis = glm::ivec3(0, 1, 0);
	            break;
	        }
	    };

	    auto buildCornerLighting = [&](BlockFace face, const glm::ivec3& owningLocal) -> std::array<std::uint32_t, 4>
	    {
	        std::array<std::uint32_t, 4> cornerLighting{};
	        const glm::ivec3 outward = faceOffset(face);
	        glm::ivec3 sideU{0};
	        glm::ivec3 sideV{0};
	        faceSampleAxes(face, sideU, sideV);

	        for (std::size_t cornerIndex = 0; cornerIndex < cornerLighting.size(); ++cornerIndex)
	        {
	            const int uSign = kCornerUSigns[cornerIndex];
	            const int vSign = kCornerVSigns[cornerIndex];
	            const glm::ivec3 fallbackSample = owningLocal + outward;
	            const std::array<glm::ivec3, 4> lightSamples{
	                fallbackSample,
	                fallbackSample + sideU * uSign,
	                fallbackSample + sideV * vSign,
	                fallbackSample + sideU * uSign + sideV * vSign
	            };

	            int skySum = 0;
	            int blockSum = 0;
	            int validSamples = 0;
	            for (const glm::ivec3& samplePos : lightSamples)
	            {
	                const BlockId sampleLightBlock = sampleBlock(samplePos.x, samplePos.y, samplePos.z);
	                if (isOpaqueForLighting(sampleLightBlock))
	                {
	                    continue;
	                }

	                const std::uint8_t packedLight = samplePackedLight(samplePos.x, samplePos.y, samplePos.z);
	                skySum += static_cast<int>(skyLightFromPacked(packedLight));
	                blockSum += static_cast<int>(blockLightFromPacked(packedLight));
	                ++validSamples;
	            }

	            std::uint8_t averagedSky = 0;
	            std::uint8_t averagedBlock = 0;
	            if (validSamples > 0)
	            {
	                averagedSky = static_cast<std::uint8_t>((skySum + validSamples / 2) / validSamples);
	                averagedBlock = static_cast<std::uint8_t>((blockSum + validSamples / 2) / validSamples);
	            }
	            else
	            {
	                const std::uint8_t fallbackPacked =
	                    samplePackedLight(fallbackSample.x, fallbackSample.y, fallbackSample.z);
	                averagedSky = skyLightFromPacked(fallbackPacked);
	                averagedBlock = blockLightFromPacked(fallbackPacked);
	            }

            const glm::ivec3 aoBase = owningLocal + outward;
            const bool side1Solid =
                isAoSolid(sampleBlock(aoBase.x + sideU.x * uSign,
                                      aoBase.y + sideU.y * uSign,
                                      aoBase.z + sideU.z * uSign));
            const bool side2Solid =
                isAoSolid(sampleBlock(aoBase.x + sideV.x * vSign,
                                      aoBase.y + sideV.y * vSign,
                                      aoBase.z + sideV.z * vSign));
            const bool cornerSolid =
                isAoSolid(sampleBlock(aoBase.x + sideU.x * uSign + sideV.x * vSign,
                                      aoBase.y + sideU.y * uSign + sideV.y * vSign,
                                      aoBase.z + sideU.z * uSign + sideV.z * vSign));
	            const std::uint8_t aoLevel =
	                (side1Solid && side2Solid)
	                    ? static_cast<std::uint8_t>(3)
	                    : static_cast<std::uint8_t>(static_cast<int>(side1Solid) +
	                                                static_cast<int>(side2Solid) +
	                                                static_cast<int>(cornerSolid));

	            cornerLighting[cornerIndex] =
	                packVertexLighting(packLightLevels(averagedSky, averagedBlock), aoLevel);
	        }

	        return cornerLighting;
	    };

	    auto makeMaterial = [&](BlockId block, const glm::vec3& normal, const glm::ivec3& owningLocal) -> FaceMaterial
	    {
	        FaceMaterial material{};
	        const BlockFace face = faceFromNormal(normal);

	        material.face = face;
            material.lightingData = buildCornerLighting(face, owningLocal);
	        material.mergeable =
	            !isAlphaCutoutBlock(block) && hasUniformCornerLighting(material.lightingData);

	        if (blockAtlasConfigured_)
	        {
	            const BlockUVSet& uvSet = blockUVTable_[toIndex(block)];
	            const FaceUV& faceUV = uvSet.faces[toIndex(face)];
            material.uvBase = faceUV.base;
            material.uvSize = faceUV.size;
        }
        else
        {
            material.uvBase = glm::vec2(0.0f);
            material.uvSize = glm::vec2(1.0f);
        }

        switch (face)
        {
        case BlockFace::Top:
            material.uAxis = glm::ivec3(1, 0, 0);
            material.vAxis = glm::ivec3(0, 0, 1);
            break;
        case BlockFace::Bottom:
            material.uAxis = glm::ivec3(1, 0, 0);
            material.vAxis = glm::ivec3(0, 0, -1);
            break;
        case BlockFace::East:
            material.uAxis = glm::ivec3(0, 0, 1);
            material.vAxis = glm::ivec3(0, 1, 0);
            break;
        case BlockFace::West:
            material.uAxis = glm::ivec3(0, 0, -1);
            material.vAxis = glm::ivec3(0, 1, 0);
            break;
        case BlockFace::South:
            material.uAxis = glm::ivec3(-1, 0, 0);
            material.vAxis = glm::ivec3(0, 1, 0);
            break;
        case BlockFace::North:
        default:
            material.uAxis = glm::ivec3(1, 0, 0);
            material.vAxis = glm::ivec3(0, 1, 0);
            break;
        }

        if (block == BlockId::Grass)
        {
            const GrassTintIndex tintIndex = grassTintForLocal(owningLocal.x, owningLocal.z);
            if (face == BlockFace::Top)
            {
                material.flags = packGrassTintFlags(tintIndex, false);
            }
            else if (face != BlockFace::Bottom)
            {
                material.flags = packGrassTintFlags(tintIndex, true);
            }
        }

        return material;
    };

    auto emitQuad = [&](Axis axis, FaceDir dir, int slice, int bStart, int cStart, int bSize, int cSize, const FaceMaterial& material)
    {
        const int a = static_cast<int>(axis);
        const int b = (a + 1) % 3;
        const int c = (a + 2) % 3;

        glm::vec3 normal = axisNormals[a];
        if (dir == FaceDir::Negative)
        {
            normal = -normal;
        }

        glm::vec3 base(0.0f);
        base[a] = static_cast<float>(slice);
        base[b] = static_cast<float>(bStart);
        base[c] = static_cast<float>(cStart);

        glm::vec3 du(0.0f);
        du[b] = static_cast<float>(bSize);

        glm::vec3 dv(0.0f);
        dv[c] = static_cast<float>(cSize);

        std::array<glm::vec3, 4> positions{
            chunkOrigin + base,
            chunkOrigin + base + du,
            chunkOrigin + base + du + dv,
            chunkOrigin + base + dv
        };

            std::array<std::uint32_t, 4> cornerLighting = material.lightingData;
	        if (dir == FaceDir::Negative)
	        {
	            std::swap(positions[1], positions[3]);
	        }

            std::array<std::uint32_t, 4> vertexLighting{};
            const glm::vec3 quadCenter = 0.25f * (positions[0] + positions[1] + positions[2] + positions[3]);
            const glm::vec3 uAxisVec = glm::vec3(material.uAxis);
            const glm::vec3 vAxisVec = glm::vec3(material.vAxis);
            for (std::size_t i = 0; i < positions.size(); ++i)
            {
                const glm::vec3 offset = positions[i] - quadCenter;
                const int uSign = glm::dot(offset, uAxisVec) >= 0.0f ? 1 : -1;
                const int vSign = glm::dot(offset, vAxisVec) >= 0.0f ? 1 : -1;
                vertexLighting[i] = cornerLighting[cornerIndexForSigns(uSign, vSign)];
            }

            const int diagonal02 =
                lightingMetricFromPackedVertex(vertexLighting[0]) +
                lightingMetricFromPackedVertex(vertexLighting[2]);
            const int diagonal13 =
                lightingMetricFromPackedVertex(vertexLighting[1]) +
                lightingMetricFromPackedVertex(vertexLighting[3]);
            const bool flipDiagonal = diagonal13 > diagonal02;

	        const std::size_t vertexStart = meshData.vertices.size();
        for (int i = 0; i < 4; ++i)
        {
            const glm::vec3& pos = positions[i];

            Vertex vertex{};
	            vertex.position = pos;
	            vertex.normal = normal;
	            vertex.tileCoord = glm::vec2(glm::dot(pos, uAxisVec), glm::dot(pos, vAxisVec));
	            vertex.atlasBase = material.uvBase;
	            vertex.atlasSize = material.uvSize;
	            vertex.lightingData = applyVertexFlags(vertexLighting[i], material.flags);
	            meshData.vertices.push_back(vertex);
	        }

            if (flipDiagonal)
            {
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 0));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 1));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 3));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 1));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 2));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 3));
            }
            else
            {
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 0));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 1));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 2));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 2));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 3));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 0));
            }
	    };

    auto greedyMeshAxis = [&](Axis axis)
    {
        const int dims[3] = {kChunkSizeX, kChunkSizeY, kChunkSizeZ};
        const int a = static_cast<int>(axis);
        const int b = (a + 1) % 3;
        const int c = (a + 2) % 3;

        const int sizeA = dims[a];
        const int sizeB = dims[b];
        const int sizeC = dims[c];

        std::vector<MaskCell> mask(static_cast<std::size_t>(sizeB * sizeC));

        auto maskIndex = [&](int bi, int ci) -> int
        {
            return bi * sizeC + ci;
        };

        for (int dirIndex = 0; dirIndex < 2; ++dirIndex)
        {
            const FaceDir dir = static_cast<FaceDir>(dirIndex);

            for (int slice = 0; slice <= sizeA; ++slice)
            {
                std::fill(mask.begin(), mask.end(), MaskCell{});

                for (int bi = 0; bi < sizeB; ++bi)
                {
                    for (int ci = 0; ci < sizeC; ++ci)
                    {
                        const int maskIdx = maskIndex(bi, ci);
                        MaskCell cell{};

                        const glm::ivec3 positiveLocal{
                            (a == 0) ? slice : ((b == 0) ? bi : ci),
                            (a == 1) ? slice : ((b == 1) ? bi : ci),
                            (a == 2) ? slice : ((b == 2) ? bi : ci)
                        };

                        const glm::ivec3 negativeLocal{
                            (a == 0) ? slice - 1 : ((b == 0) ? bi : ci),
                            (a == 1) ? slice - 1 : ((b == 1) ? bi : ci),
                            (a == 2) ? slice - 1 : ((b == 2) ? bi : ci)
                        };

                        const BlockId positiveBlock = sampleBlock(positiveLocal.x, positiveLocal.y, positiveLocal.z);
                        const BlockId negativeBlock = sampleBlock(negativeLocal.x, negativeLocal.y, negativeLocal.z);

                        glm::ivec3 owningLocal{0};
                        bool createFace = false;

                        if (dir == FaceDir::Positive)
                        {
                            if (isInsideChunk(negativeLocal) && shouldRenderBlockFace(negativeBlock, positiveBlock))
                            {
                                owningLocal = negativeLocal;
                                createFace = true;
                            }
                        }
                        else
                        {
                            if (isInsideChunk(positiveLocal) && shouldRenderBlockFace(positiveBlock, negativeBlock))
                            {
                                owningLocal = positiveLocal;
                                createFace = true;
                            }
                        }

	                        if (createFace)
	                        {
	                            const glm::vec3 normal = axisNormals[a] * ((dir == FaceDir::Positive) ? 1.0f : -1.0f);
	                            cell.exists = true;
	                            const std::size_t blockIdx = blockIndex(owningLocal.x, owningLocal.y, owningLocal.z);
	                            const BlockId owningBlock = chunkBlocks[blockIdx];
	                            cell.material = makeMaterial(
	                                owningBlock,
	                                normal,
	                                owningLocal);
	                        }

                        mask[maskIdx] = cell;
                    }
                }

                for (int bi = 0; bi < sizeB; ++bi)
                {
                    int ci = 0;
                    while (ci < sizeC)
                    {
                        const int maskIdx = maskIndex(bi, ci);
                        const MaskCell& cell = mask[maskIdx];
                        if (!cell.exists)
                        {
                            ++ci;
                            continue;
                        }

                        const FaceMaterial material = cell.material;

                        int runLengthC = 1;
                        while (material.mergeable && ci + runLengthC < sizeC)
                        {
                            const MaskCell& nextCell = mask[maskIndex(bi, ci + runLengthC)];
                            if (!nextCell.exists || !(nextCell.material == material))
                            {
                                break;
                            }
                            ++runLengthC;
                        }

                        int runHeightB = 1;
                        while (material.mergeable && bi + runHeightB < sizeB)
                        {
                            bool rowMatches = true;
                            for (int offset = 0; offset < runLengthC; ++offset)
                            {
                                const MaskCell& rowCell = mask[maskIndex(bi + runHeightB, ci + offset)];
                                if (!rowCell.exists || !(rowCell.material == material))
                                {
                                    rowMatches = false;
                                    break;
                                }
                            }

                            if (!rowMatches)
                            {
                                break;
                            }

                            ++runHeightB;
                        }

                        emitQuad(axis, dir, slice, bi, ci, runHeightB, runLengthC, material);

                        for (int bOffset = 0; bOffset < runHeightB; ++bOffset)
                        {
                            for (int cOffset = 0; cOffset < runLengthC; ++cOffset)
                            {
                                mask[maskIndex(bi + bOffset, ci + cOffset)].exists = false;
                            }
                        }

                        ci += runLengthC;
                    }
                }
            }
        }
    };

    greedyMeshAxis(Axis::X);
    greedyMeshAxis(Axis::Y);
    greedyMeshAxis(Axis::Z);

    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        // Keep the previously uploaded mesh alive until the replacement mesh is fully built.
        // Clearing meshData at the start of remeshing lets stale queued uploads observe an
        // empty mesh and momentarily punch holes in the world.
        chunk.meshData = std::move(meshData);
    }
    chunk.meshReady.store(true, std::memory_order_release);
}

glm::ivec3 ChunkManager::Impl::worldToChunkCoords(int worldX, int worldY, int worldZ) noexcept
{
    return {floorDiv(worldX, kChunkSizeX), floorDiv(worldY, kChunkSizeY), floorDiv(worldZ, kChunkSizeZ)};
}

std::shared_ptr<Chunk> ChunkManager::Impl::acquireChunk(const glm::ivec3& coord)
{
    std::shared_ptr<Chunk> chunk;
    {
        std::lock_guard<std::mutex> lock(chunkPoolMutex_);
        if (!chunkPool_.empty())
        {
            PooledChunkEntry entry = std::move(chunkPool_.back());
            chunkPool_.pop_back();
            chunkPoolBytes_ = (entry.retainedBytes >= chunkPoolBytes_) ? 0 : (chunkPoolBytes_ - entry.retainedBytes);
            chunk = std::move(entry.chunk);
        }
    }

    if (!chunk)
    {
        chunk = std::make_shared<Chunk>(coord);
    }

    chunk->reset(coord);
    return chunk;

}

std::shared_ptr<Chunk> ChunkManager::Impl::getChunkShared(const glm::ivec3& coord) noexcept
{
    std::lock_guard<std::mutex> lock(chunksMutex);
    auto it = chunks_.find(coord);
    return (it != chunks_.end()) ? it->second : nullptr;
}

std::shared_ptr<const Chunk> ChunkManager::Impl::getChunkShared(const glm::ivec3& coord) const noexcept
{
    std::lock_guard<std::mutex> lock(chunksMutex);
    auto it = chunks_.find(coord);
    if (it != chunks_.end())
    {
        return it->second;
    }
    return nullptr;
}

Chunk* ChunkManager::Impl::getChunk(const glm::ivec3& coord) noexcept
{
    return getChunkShared(coord).get();
}

const Chunk* ChunkManager::Impl::getChunk(const glm::ivec3& coord) const noexcept
{
    return getChunkShared(coord).get();
}

void ChunkManager::Impl::markNeighborsForRemeshingIfNeeded(const glm::ivec3& coord, int localX, int localY, int localZ)
{
    auto queueNeighbor = [&](const glm::ivec3& neighborCoord)
    {
        auto neighbor = getChunkShared(neighborCoord);
        if (!neighbor)
        {
            return;
        }

        requestChunkRemesh(neighbor);
    };

    if (localX == 0)
    {
        queueNeighbor(coord + glm::ivec3{-1, 0, 0});
    }

    if (localX == kChunkSizeX - 1)
    {
        queueNeighbor(coord + glm::ivec3{1, 0, 0});
    }

    if (localZ == 0)
    {
        queueNeighbor(coord + glm::ivec3{0, 0, -1});
    }

    if (localZ == kChunkSizeZ - 1)
    {
        queueNeighbor(coord + glm::ivec3{0, 0, 1});

    }

    if (localY == 0)
    {
        queueNeighbor(coord + glm::ivec3{0, -1, 0});
    }

    if (localY == kChunkSizeY - 1)
    {
        queueNeighbor(coord + glm::ivec3{0, 1, 0});
    }
}

void ChunkManager::Impl::requestChunkRemesh(const std::shared_ptr<Chunk>& chunk)
{
    if (!chunk)
    {
        return;
    }

    if (!chunk->hasBlocks.load(std::memory_order_acquire) &&
        chunk->indexCount.load(std::memory_order_acquire) == 0)
    {
        return;
    }

    const ChunkState state = chunk->state.load(std::memory_order_acquire);
    if (state == ChunkState::Generating || state == ChunkState::Meshing)
    {
        chunk->pendingMeshRefresh.store(true, std::memory_order_release);
        if (shouldTrackRecentEditChunk(chunk->coord))
        {
            std::ostringstream stream;
            stream << "remesh defer chunk=(" << chunk->coord.x << ", " << chunk->coord.y << ", " << chunk->coord.z
                   << ") state=" << chunkStateLabel(state)
                   << " inFlight=" << chunk->inFlight.load(std::memory_order_acquire);
            appendRecentEditDebugEvent(stream.str());
        }
        return;
    }

    if (state == ChunkState::Remeshing)
    {
        if (chunk->inFlight.load(std::memory_order_acquire) > 0)
        {
            chunk->pendingMeshRefresh.store(true, std::memory_order_release);
            if (shouldTrackRecentEditChunk(chunk->coord))
            {
                std::ostringstream stream;
                stream << "remesh refresh-pending chunk=(" << chunk->coord.x << ", " << chunk->coord.y << ", " << chunk->coord.z
                       << ") inFlight=" << chunk->inFlight.load(std::memory_order_acquire);
                appendRecentEditDebugEvent(stream.str());
            }
            return;
        }
    }

    if (state == ChunkState::Uploaded || state == ChunkState::Ready || state == ChunkState::Remeshing)
    {
        chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
        if (benchmarkMetrics_.isEnabled())
        {
            storeFirstBenchmarkTimestamp(chunk->meshQueuedTimestampMicros, steadyMicrosNow());
        }
        enqueueJob(chunk,
                   JobType::Mesh,
                   chunk->coord,
                   0,
                   chunkAwaitingInitialVisibleReady(*chunk));
        if (shouldTrackRecentEditChunk(chunk->coord))
        {
            std::ostringstream stream;
            stream << "remesh enqueue chunk=(" << chunk->coord.x << ", " << chunk->coord.y << ", " << chunk->coord.z
                   << ") prevState=" << chunkStateLabel(state)
                   << " idx=" << chunk->indexCount.load(std::memory_order_acquire);
            appendRecentEditDebugEvent(stream.str());
        }
    }
}

void ChunkManager::Impl::requestChunkRemeshFromRelight(const std::shared_ptr<Chunk>& chunk)
{
    if (!chunk)
    {
        return;
    }

    if (!chunk->hasBlocks.load(std::memory_order_acquire) &&
        chunk->indexCount.load(std::memory_order_acquire) == 0)
    {
        return;
    }

    const ChunkState state = chunk->state.load(std::memory_order_acquire);
    if (state == ChunkState::Generating || state == ChunkState::Meshing)
    {
        chunk->pendingMeshRefresh.store(true, std::memory_order_release);
        return;
    }

    const int inFlight = chunk->inFlight.load(std::memory_order_acquire);
    if (state == ChunkState::Remeshing && inFlight > 1)
    {
        chunk->pendingMeshRefresh.store(true, std::memory_order_release);
        return;
    }

    if (state == ChunkState::Uploaded || state == ChunkState::Ready || state == ChunkState::Remeshing)
    {
        chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
        if (benchmarkMetrics_.isEnabled())
        {
            storeFirstBenchmarkTimestamp(chunk->meshQueuedTimestampMicros, steadyMicrosNow());
        }
        enqueueJob(chunk,
                   JobType::Mesh,
                   chunk->coord,
                   0,
                   chunkAwaitingInitialVisibleReady(*chunk));
    }
}

bool ChunkManager::Impl::relightRegionsOverlap(const glm::ivec3& minA,
                                               const glm::ivec3& maxA,
                                               const glm::ivec3& minB,
                                               const glm::ivec3& maxB) noexcept
{
    return minA.x <= maxB.x && maxA.x >= minB.x &&
           minA.y <= maxB.y && maxA.y >= minB.y &&
           minA.z <= maxB.z && maxA.z >= minB.z;
}

glm::ivec3 ChunkManager::Impl::relightRegionAnchor(const PendingRelightBatch& batch) noexcept
{
    return glm::ivec3((batch.minCoord.x + batch.maxCoord.x) / 2,
                      (batch.minCoord.y + batch.maxCoord.y) / 2,
                      (batch.minCoord.z + batch.maxCoord.z) / 2);
}

void ChunkManager::Impl::recomputePendingRelightBatchBounds(PendingRelightBatch& batch) const noexcept
{
    if (batch.dirtyCoordGenerations.empty())
    {
        batch = PendingRelightBatch{};
        return;
    }

    bool first = true;
    std::uint64_t maxGeneration = 0;
    for (const auto& [coord, generation] : batch.dirtyCoordGenerations)
    {
        if (first)
        {
            batch.minCoord = coord;
            batch.maxCoord = coord;
            first = false;
        }
        else
        {
            batch.minCoord = glm::min(batch.minCoord, coord);
            batch.maxCoord = glm::max(batch.maxCoord, coord);
        }
        maxGeneration = std::max(maxGeneration, generation);
    }

    batch.reservedMinCoord = batch.minCoord - kRelightReservationPadding;
    batch.reservedMaxCoord = batch.maxCoord + kRelightReservationPadding;
    batch.maxGeneration = maxGeneration;
    batch.valid = true;
}

void ChunkManager::Impl::mergePendingRelightBatch(PendingRelightBatch& dst, PendingRelightBatch&& src) const
{
    for (auto& [coord, generation] : src.dirtyCoordGenerations)
    {
        auto [it, inserted] = dst.dirtyCoordGenerations.try_emplace(coord, generation);
        if (!inserted)
        {
            it->second = std::max(it->second, generation);
        }
    }
    dst.forceRemeshCoords.insert(src.forceRemeshCoords.begin(), src.forceRemeshCoords.end());
    dst.containsInitialReadyCoord = dst.containsInitialReadyCoord || src.containsInitialReadyCoord;
    dst.sequence = (dst.sequence == 0) ? src.sequence : std::min(dst.sequence, src.sequence);
    dst.estimatedCostUnits = 0;
    recomputePendingRelightBatchBounds(dst);
}

std::unordered_set<glm::ivec3, ChunkHasher> ChunkManager::Impl::expandRelightCoords(
    const RelightCoordGenerationMap& dirtyCoords) const
{
    std::unordered_set<glm::ivec3, ChunkHasher> expanded;
    expanded.reserve(dirtyCoords.size() * 27);

    for (const auto& [coord, _] : dirtyCoords)
    {
        for (int dx = -kRelightNeighborPadding.x; dx <= kRelightNeighborPadding.x; ++dx)
        {
            for (int dy = -kRelightNeighborPadding.y; dy <= kRelightNeighborPadding.y; ++dy)
            {
                for (int dz = -kRelightNeighborPadding.z; dz <= kRelightNeighborPadding.z; ++dz)
                {
                    expanded.insert(coord + glm::ivec3(dx, dy, dz));
                }
            }
        }
    }

    return expanded;
}

std::uint64_t ChunkManager::Impl::estimatePendingRelightBatchCost(const PendingRelightBatch& batch) const
{
    if (!batch.valid || batch.dirtyCoordGenerations.empty())
    {
        return 0;
    }

    const std::unordered_set<glm::ivec3, ChunkHasher> expandedCoords = expandRelightCoords(batch.dirtyCoordGenerations);
    std::unordered_set<glm::ivec3, ChunkHasher> loadedRegionCoords;
    std::unordered_set<glm::ivec3, ChunkHasher> externalCoords;
    std::unordered_map<glm::ivec2, int, ColumnHasher> maxLoadedChunkYByColumn;

    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        loadedRegionCoords.reserve(expandedCoords.size());

        for (const glm::ivec3& coord : expandedCoords)
        {
            auto it = chunks_.find(coord);
            if (it == chunks_.end())
            {
                continue;
            }

            loadedRegionCoords.insert(coord);
            const glm::ivec2 column(coord.x, coord.z);
            auto [columnIt, inserted] = maxLoadedChunkYByColumn.try_emplace(column, coord.y);
            if (!inserted)
            {
                columnIt->second = std::max(columnIt->second, coord.y);
            }
        }

        for (const glm::ivec3& coord : loadedRegionCoords)
        {
            for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
            {
                const glm::ivec3 neighborCoord = coord + faceOffset(face);
                if (loadedRegionCoords.find(neighborCoord) != loadedRegionCoords.end())
                {
                    continue;
                }

                if (chunks_.find(neighborCoord) != chunks_.end())
                {
                    externalCoords.insert(neighborCoord);
                }
            }
        }

        for (const auto& [coord, chunk] : chunks_)
        {
            (void)chunk;
            const glm::ivec2 column(coord.x, coord.z);
            auto maxIt = maxLoadedChunkYByColumn.find(column);
            if (maxIt != maxLoadedChunkYByColumn.end() && coord.y > maxIt->second)
            {
                externalCoords.insert(coord);
            }
        }
    }

    const std::uint64_t regionChunkCount = static_cast<std::uint64_t>(loadedRegionCoords.size());
    const std::uint64_t externalChunkCount = static_cast<std::uint64_t>(externalCoords.size());
    const std::uint64_t dirtyCoordCount = static_cast<std::uint64_t>(batch.dirtyCoordGenerations.size());
    const std::uint64_t forceRemeshCount = static_cast<std::uint64_t>(batch.forceRemeshCoords.size());

    const std::uint64_t estimatedCost =
        regionChunkCount * 2048ull +
        externalChunkCount * 4096ull +
        dirtyCoordCount * 1024ull +
        forceRemeshCount * 512ull;
    return std::max<std::uint64_t>(estimatedCost, 1ull);
}

void ChunkManager::Impl::markSkyLightColumnDirty(const glm::ivec2& column)
{
    std::lock_guard<std::mutex> lock(skyLightCacheMutex_);
    auto [it, inserted] = skyLightColumnGenerations_.try_emplace(column, 1ull);
    if (!inserted)
    {
        ++it->second;
        if (it->second == 0)
        {
            it->second = 1;
        }
    }
}

std::uint64_t ChunkManager::Impl::currentSkyLightColumnGeneration(const glm::ivec2& column)
{
    std::lock_guard<std::mutex> lock(skyLightCacheMutex_);
    auto [it, inserted] = skyLightColumnGenerations_.try_emplace(column, 1ull);
    if (inserted && it->second == 0)
    {
        it->second = 1;
    }
    return it->second;
}

void ChunkManager::Impl::ensureSkyLightColumnCacheForChunks(const std::vector<std::shared_ptr<Chunk>>& chunks)
{
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();
    if (chunks.empty())
    {
        return;
    }

    std::unordered_set<glm::ivec2, ColumnHasher> columns;
    columns.reserve(chunks.size());
    for (const auto& chunk : chunks)
    {
        if (chunk)
        {
            columns.insert(glm::ivec2(chunk->coord.x, chunk->coord.z));
        }
    }

    if (columns.empty())
    {
        return;
    }

    std::unordered_map<glm::ivec2, std::uint64_t, ColumnHasher> columnGenerations;
    columnGenerations.reserve(columns.size());
    for (const glm::ivec2& column : columns)
    {
        columnGenerations.emplace(column, currentSkyLightColumnGeneration(column));
    }

    std::unordered_map<glm::ivec2, std::vector<std::shared_ptr<Chunk>>, ColumnHasher> columnStacks;
    columnStacks.reserve(columns.size());
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        for (const auto& [coord, chunk] : chunks_)
        {
            const glm::ivec2 column(coord.x, coord.z);
            if (columns.find(column) != columns.end())
            {
                columnStacks[column].push_back(chunk);
            }
        }
    }

    std::vector<std::shared_ptr<Chunk>> chunksToLock;
    chunksToLock.reserve(chunks.size());
    for (auto& [column, stack] : columnStacks)
    {
        if (stack.empty())
        {
            continue;
        }

        std::sort(stack.begin(),
                  stack.end(),
                  [](const std::shared_ptr<Chunk>& lhs, const std::shared_ptr<Chunk>& rhs)
                  {
                      return lhs->coord.y > rhs->coord.y;
                  });
        for (auto& chunk : stack)
        {
            chunksToLock.push_back(chunk);
        }
    }

    std::sort(chunksToLock.begin(),
              chunksToLock.end(),
              [](const std::shared_ptr<Chunk>& lhs, const std::shared_ptr<Chunk>& rhs)
              {
                  if (lhs->coord.x != rhs->coord.x)
                  {
                      return lhs->coord.x < rhs->coord.x;
                  }
                  if (lhs->coord.y != rhs->coord.y)
                  {
                      return lhs->coord.y < rhs->coord.y;
                  }
                  return lhs->coord.z < rhs->coord.z;
              });
    chunksToLock.erase(std::unique(chunksToLock.begin(), chunksToLock.end()), chunksToLock.end());

    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(chunksToLock.size());
    const auto lockStageStart = benchmarkEnabled ? SteadyClock::now() : SteadyClock::time_point{};
    for (auto& chunk : chunksToLock)
    {
        if (chunk && !chunk->cpuDataResident)
        {
            (void)ensureChunkCpuDataResident(*chunk);
        }
        locks.emplace_back(chunk->meshMutex);
    }

    for (auto& [column, stack] : columnStacks)
    {
        const std::uint64_t generation = columnGenerations[column];
        bool needsRefresh = false;
        for (const auto& chunk : stack)
        {
            if (chunk->skyLightCacheGeneration.load(std::memory_order_acquire) != generation)
            {
                needsRefresh = true;
                break;
            }
        }
        if (!needsRefresh)
        {
            continue;
        }

        std::array<std::uint8_t, kChunkSizeX * kChunkSizeZ> incomingSky{};
        incomingSky.fill(kMaxLightLevel);

        for (const auto& chunk : stack)
        {
            chunk->skyLightFromAboveCache = incomingSky;
            chunk->skyLightCacheGeneration.store(generation, std::memory_order_release);

            std::array<std::uint8_t, kChunkSizeX * kChunkSizeZ> outgoingSky = incomingSky;
            for (int localX = 0; localX < kChunkSizeX; ++localX)
            {
                for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
                {
                    const std::size_t columnIndex = static_cast<std::size_t>(localZ * kChunkSizeX + localX);
                    std::uint8_t sky = incomingSky[columnIndex];
                    for (int localY = kChunkSizeY - 1; localY >= 0 && sky > 0; --localY)
                    {
                        const BlockId block = chunk->blocks[blockIndex(localX, localY, localZ)];
                        if (isOpaqueForLighting(block))
                        {
                            sky = 0;
                            break;
                        }

                        const std::uint8_t attenuation = blockLightingProperties(block).skyAttenuation;
                        sky = static_cast<std::uint8_t>(
                            std::max(0, static_cast<int>(sky) - static_cast<int>(attenuation)));
                    }
                    outgoingSky[columnIndex] = sky;
                }
            }

            incomingSky = outgoingSky;
        }
    }

    if (benchmarkEnabled)
    {
        const auto lockMicros =
            std::chrono::duration_cast<std::chrono::microseconds>(SteadyClock::now() - lockStageStart).count();
        benchmarkMetrics_.skyLightCacheLockStage.recordMicros(static_cast<std::uint64_t>(lockMicros));
    }
}

std::uint64_t ChunkManager::Impl::computeRelightBudgetUnits()
{
    const bool exactOnly = renderSettings_.totalChunks <= renderSettings_.exactChunks;
    std::size_t pendingRegionCount = 0;
    {
        std::lock_guard<std::mutex> lock(relightStateMutex_);
        pendingRegionCount = pendingRelightRegions_.size();
    }

    std::uint64_t budget = kRelightBaseBudgetUnits;
    budget += std::min<std::uint64_t>(workerThreadCount_, 4u) * kRelightPerWorkerBudgetUnits;
    budget += std::min<std::uint64_t>(pendingRegionCount * kRelightBacklogBudgetUnitsPerRegion,
                                      kRelightMaxBudgetUnits / 2);
    if (exactOnly)
    {
        budget += 1'000'000ull;
    }
    if (lastMissingChunks_ > 0)
    {
        budget += std::min<std::uint64_t>(static_cast<std::uint64_t>(lastMissingChunks_) * 2048ull,
                                          1'000'000ull);
    }
    if (protectedPressureActive_)
    {
        budget += 750'000ull;
    }
    if (severeProtectedPressureActive_)
    {
        budget += 1'250'000ull;
    }

    const std::uint64_t uploadPenalty =
        std::min<std::uint64_t>(pendingUploadsLastFrame_ * (exactOnly ? 1024ull : 2048ull),
                                kRelightBaseBudgetUnits / 2);
    budget = (budget > uploadPenalty) ? (budget - uploadPenalty / 2) : budget;
    return std::clamp<std::uint64_t>(budget, kRelightMinBudgetUnits, kRelightMaxBudgetUnits);
}

int ChunkManager::Impl::computeRelightBatchBudget()
{
    const bool exactOnly = renderSettings_.totalChunks <= renderSettings_.exactChunks;
    std::size_t pendingRegionCount = 0;
    {
        std::lock_guard<std::mutex> lock(relightStateMutex_);
        pendingRegionCount = pendingRelightRegions_.size();
    }

    int batchBudget = kRelightMinBatchBudget + static_cast<int>(std::min<std::size_t>(workerThreadCount_, 6) / 2);
    batchBudget += static_cast<int>(std::min<std::size_t>(pendingRegionCount / 12, 3));
    if (exactOnly)
    {
        batchBudget += 2;
    }
    if (lastMissingChunks_ > 0)
    {
        batchBudget += std::min(lastMissingChunks_ / 256, 2);
    }
    if (protectedPressureActive_)
    {
        batchBudget += 2;
    }
    if (severeProtectedPressureActive_)
    {
        batchBudget += 2;
    }
    if (pendingUploadsLastFrame_ > 256 && !protectedPressureActive_)
    {
        batchBudget = std::max(kRelightMinBatchBudget, batchBudget - 1);
    }
    return std::clamp(batchBudget, kRelightMinBatchBudget, kRelightMaxBatchBudget);
}

void ChunkManager::Impl::resetRelightBudgetForFrame()
{
    const std::uint64_t budgetUnits = computeRelightBudgetUnits();
    const int batchBudget = computeRelightBatchBudget();

    std::lock_guard<std::mutex> lock(relightStateMutex_);
    relightBudgetUnitsThisFrame_ = budgetUnits;
    relightBudgetUnitsRemaining_ = budgetUnits;
    relightBatchBudgetThisFrame_ = batchBudget;
    relightBatchBudgetRemaining_ = batchBudget;
}

void ChunkManager::Impl::queueRelightRequest(const glm::ivec3& centerCoord, bool forceRemesh)
{
    if (centerCoord.y < 0)
    {
        return;
    }

    std::shared_ptr<Chunk> chunk = getChunkShared(centerCoord);
    const bool initialReadyCoord =
        chunk && chunkAwaitingInitialVisibleReady(*chunk);
    PendingRelightBatch incoming{};
    std::lock_guard<std::mutex> lock(relightStateMutex_);

    if (auto pendingIt = pendingRelightCoordGenerations_.find(centerCoord);
        pendingIt != pendingRelightCoordGenerations_.end())
    {
        const std::uint64_t pendingGeneration = pendingIt->second;
        for (PendingRelightBatch& pendingBatch : pendingRelightRegions_)
        {
            auto dirtyIt = pendingBatch.dirtyCoordGenerations.find(centerCoord);
            if (dirtyIt == pendingBatch.dirtyCoordGenerations.end())
            {
                continue;
            }

            dirtyIt->second = std::max(dirtyIt->second, pendingGeneration);
            if (forceRemesh)
            {
                pendingBatch.forceRemeshCoords.insert(centerCoord);
            }
            pendingBatch.containsInitialReadyCoord =
                pendingBatch.containsInitialReadyCoord || initialReadyCoord;
            pendingBatch.maxGeneration = std::max(pendingBatch.maxGeneration, dirtyIt->second);
            pendingBatch.estimatedCostUnits = 0;
            break;
        }

        if (chunk)
        {
            chunk->lightingRevision.store(pendingGeneration, std::memory_order_release);
        }
        return;
    }

    const std::uint64_t generation = nextRelightGeneration_++;
    incoming.valid = true;
    incoming.sequence = nextPendingRelightSequence_++;
    incoming.dirtyCoordGenerations.emplace(centerCoord, generation);
    if (forceRemesh)
    {
        incoming.forceRemeshCoords.insert(centerCoord);
    }
    incoming.containsInitialReadyCoord = initialReadyCoord;
    recomputePendingRelightBatchBounds(incoming);

    if (chunk)
    {
        chunk->lightingRevision.store(generation, std::memory_order_release);
    }

    for (auto it = pendingRelightRegions_.begin(); it != pendingRelightRegions_.end();)
    {
        if (incoming.dirtyCoordGenerations.size() + it->dirtyCoordGenerations.size() >
            kRelightMaxPendingDirtyCoordsPerRegion)
        {
            ++it;
            continue;
        }

        if (relightRegionsOverlap(incoming.reservedMinCoord,
                                  incoming.reservedMaxCoord,
                                  it->reservedMinCoord,
                                  it->reservedMaxCoord))
        {
            mergePendingRelightBatch(incoming, std::move(*it));
            it = pendingRelightRegions_.erase(it);
            continue;
        }
        ++it;
    }

    for (const auto& [coord, coordGeneration] : incoming.dirtyCoordGenerations)
    {
        auto [it, inserted] = pendingRelightCoordGenerations_.try_emplace(coord, coordGeneration);
        if (!inserted)
        {
            it->second = std::max(it->second, coordGeneration);
        }
    }
    incoming.estimatedCostUnits = 0;
    pendingRelightRegions_.push_back(std::move(incoming));
}

bool ChunkManager::Impl::takePendingRelightBatch(PendingRelightBatch& batch)
{
    glm::ivec3 priorityOrigin{0};
    glm::vec3 priorityForward{0.0f, 0.0f, -1.0f};
    {
        std::lock_guard<std::mutex> priorityLock(schedulingPriorityMutex_);
        priorityOrigin = schedulingPriorityOrigin_;
        priorityForward = schedulingPriorityForward_;
    }

    std::lock_guard<std::mutex> lock(relightStateMutex_);
    if (pendingRelightRegions_.empty() || relightBatchBudgetRemaining_ <= 0)
    {
        batch = PendingRelightBatch{};
        return false;
    }

    auto overlapsActiveRegions = [&](const PendingRelightBatch& candidate) noexcept
    {
        for (const ActiveRelightRegion& region : activeRelightRegions_)
        {
            if (relightRegionsOverlap(candidate.reservedMinCoord,
                                      candidate.reservedMaxCoord,
                                      region.minCoord,
                                      region.maxCoord))
            {
                return true;
            }
        }
        return false;
    };

    const bool allowOverBudgetFallback =
        relightBudgetUnitsRemaining_ == relightBudgetUnitsThisFrame_ &&
        relightBatchBudgetRemaining_ == relightBatchBudgetThisFrame_;

    auto isHigherPriorityBatch = [&](const PendingRelightBatch& lhs, const PendingRelightBatch& rhs) noexcept
    {
        if (lhs.containsInitialReadyCoord != rhs.containsInitialReadyCoord)
        {
            return lhs.containsInitialReadyCoord;
        }

        return isChunkCoordHigherPriority(relightRegionAnchor(lhs),
                                          relightRegionAnchor(rhs),
                                          priorityOrigin,
                                          priorityForward);
    };

    auto bestFitIt = pendingRelightRegions_.end();
    auto bestFallbackIt = pendingRelightRegions_.end();
    for (auto it = pendingRelightRegions_.begin(); it != pendingRelightRegions_.end(); ++it)
    {
        if (!it->valid || overlapsActiveRegions(*it))
        {
            continue;
        }

        if (it->estimatedCostUnits == 0)
        {
            it->estimatedCostUnits = estimatePendingRelightBatchCost(*it);
        }

        if (bestFallbackIt == pendingRelightRegions_.end() ||
            isHigherPriorityBatch(*it, *bestFallbackIt))
        {
            bestFallbackIt = it;
        }

        if (it->estimatedCostUnits <= relightBudgetUnitsRemaining_)
        {
            if (bestFitIt == pendingRelightRegions_.end() ||
                isHigherPriorityBatch(*it, *bestFitIt))
            {
                bestFitIt = it;
            }
        }
    }

    auto chosenIt = bestFitIt;
    if (chosenIt == pendingRelightRegions_.end())
    {
        if (!allowOverBudgetFallback || bestFallbackIt == pendingRelightRegions_.end())
        {
            batch = PendingRelightBatch{};
            return false;
        }
        chosenIt = bestFallbackIt;
    }

    batch = std::move(*chosenIt);
    pendingRelightRegions_.erase(chosenIt);

    for (const auto& [coord, coordGeneration] : batch.dirtyCoordGenerations)
    {
        auto pendingIt = pendingRelightCoordGenerations_.find(coord);
        if (pendingIt != pendingRelightCoordGenerations_.end() && pendingIt->second <= coordGeneration)
        {
            pendingRelightCoordGenerations_.erase(pendingIt);
        }

        auto [activeIt, inserted] = activeRelightCoordGenerations_.try_emplace(coord, coordGeneration);
        if (!inserted)
        {
            activeIt->second = std::max(activeIt->second, coordGeneration);
        }
    }

    activeRelightRegions_.push_back(ActiveRelightRegion{
        batch.reservedMinCoord,
        batch.reservedMaxCoord,
        batch.dirtyCoordGenerations,
        batch.maxGeneration,
        batch.sequence});

    --relightBatchBudgetRemaining_;
    if (batch.estimatedCostUnits >= relightBudgetUnitsRemaining_)
    {
        relightBudgetUnitsRemaining_ = 0;
    }
    else
    {
        relightBudgetUnitsRemaining_ -= batch.estimatedCostUnits;
    }

    return true;
}

void ChunkManager::Impl::releasePendingRelightBatch(const PendingRelightBatch& batch)
{
    if (!batch.valid)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(relightStateMutex_);
    auto activeIt = std::find_if(activeRelightRegions_.begin(),
                                 activeRelightRegions_.end(),
                                 [&](const ActiveRelightRegion& region)
                                 {
                                     return region.sequence == batch.sequence;
                                 });
    if (activeIt != activeRelightRegions_.end())
    {
        activeRelightRegions_.erase(activeIt);
    }

    for (const auto& [coord, coordGeneration] : batch.dirtyCoordGenerations)
    {
        auto coordIt = activeRelightCoordGenerations_.find(coord);
        if (coordIt != activeRelightCoordGenerations_.end() && coordIt->second <= coordGeneration)
        {
            activeRelightCoordGenerations_.erase(coordIt);
        }
    }
}

void ChunkManager::Impl::processPendingRelightRequests(int maxBatches)
{
    if (maxBatches <= 0)
    {
        return;
    }

    const int kMaxConcurrentRelightProcessors =
        std::clamp(static_cast<int>(std::max<std::size_t>(workerThreadCount_, 1) / 2), 1, 4);
    int activeProcessors = activeRelightProcessors_.load(std::memory_order_acquire);
    while (activeProcessors < kMaxConcurrentRelightProcessors)
    {
        if (activeRelightProcessors_.compare_exchange_weak(activeProcessors,
                                                           activeProcessors + 1,
                                                           std::memory_order_acq_rel,
                                                           std::memory_order_acquire))
        {
            break;
        }
    }

    if (activeProcessors >= kMaxConcurrentRelightProcessors)
    {
        return;
    }

    struct RelightProcessorGuard
    {
        std::atomic<int>& counter;

        ~RelightProcessorGuard()
        {
            counter.fetch_sub(1, std::memory_order_release);
        }
    } guard{activeRelightProcessors_};

    for (int batchIndex = 0; batchIndex < maxBatches; ++batchIndex)
    {
        PendingRelightBatch batch{};
        if (!takePendingRelightBatch(batch) || !batch.valid)
        {
            return;
        }

        struct RelightBatchReservationGuard
        {
            ChunkManager::Impl& owner;
            PendingRelightBatch& batch;

            ~RelightBatchReservationGuard()
            {
                owner.releasePendingRelightBatch(batch);
            }
        } batchReservationGuard{*this, batch};

        profilingCounters_.relightBatches.fetch_add(1, std::memory_order_relaxed);
        relightChunkRegion(batch);
    }
}

void ChunkManager::Impl::queueChunkForLightingRemesh(const std::shared_ptr<Chunk>& chunk)
{
    requestChunkRemesh(chunk);
}

std::uint8_t ChunkManager::Impl::packedLightAtWorld(const glm::ivec3& worldPos) const noexcept
{
    if (worldPos.y < 0)
    {
        return packLightLevels(0, 0);
    }

    const glm::ivec3 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
    auto chunk = getChunkShared(chunkCoord);
    if (!chunk)
    {
        return packLightLevels(kMaxLightLevel, 0);
    }

    if (worldPos.y < chunk->minWorldY || worldPos.y > chunk->maxWorldY)
    {
        return packLightLevels(kMaxLightLevel, 0);
    }

    const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
    if (local.x < 0 || local.x >= kChunkSizeX ||
        local.z < 0 || local.z >= kChunkSizeZ)
    {
        return packLightLevels(kMaxLightLevel, 0);
    }

    if (!chunk->cpuDataResident || chunk->lightLevels.size() != static_cast<std::size_t>(kChunkBlockCount))
    {
        return packLightLevels(kMaxLightLevel, 0);
    }

    const int localY = worldPos.y - chunk->minWorldY;
    return chunk->lightLevels[blockIndex(local.x, localY, local.z)];
}

void ChunkManager::Impl::relightChunkRegion(const PendingRelightBatch& batch)
{
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();
    const SteadyClock::time_point relightStart = benchmarkEnabled ? SteadyClock::now() : SteadyClock::time_point{};
    const std::unordered_set<glm::ivec3, ChunkHasher> expandedCoords = expandRelightCoords(batch.dirtyCoordGenerations);
    std::vector<std::shared_ptr<Chunk>> regionChunks;
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        regionChunks.reserve(expandedCoords.size());
        for (const glm::ivec3& coord : expandedCoords)
        {
            auto it = chunks_.find(coord);
            if (it != chunks_.end())
            {
                regionChunks.push_back(it->second);
            }
        }
    }

    if (regionChunks.empty())
    {
        profilingCounters_.relitChunks.fetch_add(1, std::memory_order_relaxed);
        if (benchmarkEnabled)
        {
            benchmarkMetrics_.relightStage.recordMicros(0);
            benchmarkMetrics_.relightRegionChunks.record(0);
            benchmarkMetrics_.relightChangedChunks.record(0);
            benchmarkMetrics_.relightExternalSnapshotChunks.record(0);
            benchmarkMetrics_.relightSkyAboveChunkScans.record(0);
            benchmarkMetrics_.relightSkySeedNodes.record(0);
            benchmarkMetrics_.relightBlockSeedNodes.record(0);
            benchmarkMetrics_.relightSkyNodesProcessed.record(0);
            benchmarkMetrics_.relightBlockNodesProcessed.record(0);
        }
        return;
    }

    for (const auto& chunk : regionChunks)
    {
        if (chunk && !chunk->cpuDataResident)
        {
            (void)ensureChunkCpuDataResident(*chunk);
        }
    }

    std::sort(regionChunks.begin(),
              regionChunks.end(),
              [](const std::shared_ptr<Chunk>& lhs, const std::shared_ptr<Chunk>& rhs)
              {
                  if (lhs->coord.x != rhs->coord.x)
                  {
                      return lhs->coord.x < rhs->coord.x;
                  }
                  if (lhs->coord.y != rhs->coord.y)
                  {
                      return lhs->coord.y < rhs->coord.y;
                  }
                  return lhs->coord.z < rhs->coord.z;
              });
    regionChunks.erase(std::unique(regionChunks.begin(), regionChunks.end()), regionChunks.end());

    std::unordered_map<glm::ivec3, std::shared_ptr<Chunk>, ChunkHasher> regionLookup;
    regionLookup.reserve(regionChunks.size());
    for (const auto& chunk : regionChunks)
    {
        regionLookup.emplace(chunk->coord, chunk);
        chunk->inFlight.fetch_add(1, std::memory_order_relaxed);
    }

    struct RelightFlightGuard
    {
        std::vector<std::shared_ptr<Chunk>>& chunks;

        ~RelightFlightGuard()
        {
            for (const auto& chunk : chunks)
            {
                if (chunk)
                {
                    chunk->inFlight.fetch_sub(1, std::memory_order_relaxed);
                }
            }
        }
    } relightFlightGuard{regionChunks};

    ensureSkyLightColumnCacheForChunks(regionChunks);

    struct RelightChunkReadSnapshot
    {
        int minWorldY{0};
        int maxWorldY{0};
        std::vector<BlockId> blocks;
        std::vector<std::uint8_t> lightLevels;
    };

    std::unordered_map<glm::ivec3, std::shared_ptr<Chunk>, ChunkHasher> externalSnapshotSources;
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        externalSnapshotSources.reserve(regionChunks.size() * 2);

        for (const auto& chunk : regionChunks)
        {
            for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
            {
                const glm::ivec3 neighborCoord = chunk->coord + faceOffset(face);
                if (regionLookup.find(neighborCoord) != regionLookup.end())
                {
                    continue;
                }

                auto it = chunks_.find(neighborCoord);
                if (it != chunks_.end())
                {
                    externalSnapshotSources.try_emplace(neighborCoord, it->second);
                }
            }
        }

        std::unordered_map<glm::ivec2, int, ColumnHasher> maxRegionChunkYByColumn;
        maxRegionChunkYByColumn.reserve(regionChunks.size());
        for (const auto& chunk : regionChunks)
        {
            const glm::ivec2 column(chunk->coord.x, chunk->coord.z);
            auto [it, inserted] = maxRegionChunkYByColumn.try_emplace(column, chunk->coord.y);
            if (!inserted)
            {
                it->second = std::max(it->second, chunk->coord.y);
            }
        }

        for (const auto& [coord, chunk] : chunks_)
        {
            const glm::ivec2 column(coord.x, coord.z);
            auto maxIt = maxRegionChunkYByColumn.find(column);
            if (maxIt != maxRegionChunkYByColumn.end() && coord.y > maxIt->second)
            {
                externalSnapshotSources.try_emplace(coord, chunk);
            }
        }
    }

    std::unordered_map<glm::ivec3, RelightChunkReadSnapshot, ChunkHasher> externalReadSnapshots;
    externalReadSnapshots.reserve(externalSnapshotSources.size());
    for (const auto& [coord, chunk] : externalSnapshotSources)
    {
        if (!chunk)
        {
            continue;
        }

        if (!chunk->cpuDataResident)
        {
            (void)ensureChunkCpuDataResident(*chunk);
        }

        std::lock_guard<std::mutex> lock(chunk->meshMutex);
        RelightChunkReadSnapshot snapshot{};
        snapshot.minWorldY = chunk->minWorldY;
        snapshot.maxWorldY = chunk->maxWorldY;
        snapshot.blocks = chunk->blocks;
        snapshot.lightLevels = chunk->lightLevels;
        externalReadSnapshots.emplace(coord, std::move(snapshot));
    }

    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(regionChunks.size());
    for (auto& chunk : regionChunks)
    {
        locks.emplace_back(chunk->meshMutex);
    }

    std::vector<std::vector<std::uint8_t>> previousLights;
    previousLights.reserve(regionChunks.size());
    std::unordered_map<glm::ivec3, std::size_t, ChunkHasher> regionIndexByCoord;
    regionIndexByCoord.reserve(regionChunks.size());
    for (std::size_t chunkIndex = 0; chunkIndex < regionChunks.size(); ++chunkIndex)
    {
        auto& chunk = regionChunks[chunkIndex];
        previousLights.push_back(chunk->lightLevels);
        chunk->lightBoundaryDirtyMask = 0;
        regionIndexByCoord.emplace(chunk->coord, chunkIndex);
    }

    auto findExternalSnapshot = [&](const glm::ivec3& chunkCoord) -> const RelightChunkReadSnapshot*
    {
        auto it = externalReadSnapshots.find(chunkCoord);
        return (it != externalReadSnapshots.end()) ? &it->second : nullptr;
    };

    struct LightNode
    {
        glm::ivec3 worldPos{0};
        std::uint8_t level{0};
    };

    std::deque<LightNode> skyQueue;
    std::deque<LightNode> blockQueue;
    std::deque<LightNode> skyRemovalQueue;
    std::deque<LightNode> blockRemovalQueue;
    std::uint64_t skyAboveChunkScans = 0;
    std::uint64_t skySeedNodes = 0;
    std::uint64_t blockSeedNodes = 0;
    std::uint64_t skyNodesProcessed = 0;
    std::uint64_t blockNodesProcessed = 0;

    auto accessRegionVoxel = [&](const glm::ivec3& worldPos) -> std::pair<Chunk*, std::size_t>
    {
        const glm::ivec3 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
        auto it = regionLookup.find(chunkCoord);
        if (it == regionLookup.end())
        {
            return {nullptr, 0};
        }

        const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
        if (local.x < 0 || local.x >= kChunkSizeX ||
            local.z < 0 || local.z >= kChunkSizeZ ||
            worldPos.y < it->second->minWorldY ||
            worldPos.y > it->second->maxWorldY)
        {
            return {nullptr, 0};
        }

        const int localY = worldPos.y - it->second->minWorldY;
        return {it->second.get(), blockIndex(local.x, localY, local.z)};
    };

    auto channelFromPacked = [](std::uint8_t packed, bool skyChannel) noexcept -> std::uint8_t
    {
        return skyChannel ? skyLightFromPacked(packed) : blockLightFromPacked(packed);
    };

    auto setPackedChannel = [](std::uint8_t& packed, bool skyChannel, std::uint8_t level) noexcept
    {
        if (skyChannel)
        {
            setSkyLight(packed, level);
        }
        else
        {
            setBlockLight(packed, level);
        }
    };

    std::vector<std::vector<std::uint8_t>> baseLightLevels(
        regionChunks.size(),
        std::vector<std::uint8_t>(kChunkBlockCount, packLightLevels(0, 0)));
    std::unordered_set<glm::ivec2, ColumnHasher> dirtyColumns;
    dirtyColumns.reserve(batch.dirtyCoordGenerations.size());
    for (const auto& [coord, _] : batch.dirtyCoordGenerations)
    {
        dirtyColumns.insert(glm::ivec2(coord.x, coord.z));
    }

    std::unordered_set<glm::ivec3, ChunkHasher> recomputeCoords;
    recomputeCoords.reserve(regionChunks.size());
    for (const auto& chunk : regionChunks)
    {
        if (dirtyColumns.find(glm::ivec2(chunk->coord.x, chunk->coord.z)) != dirtyColumns.end())
        {
            recomputeCoords.insert(chunk->coord);
        }
    }

    auto basePackedAtRegionVoxel = [&](Chunk* chunk, std::size_t idx) -> std::uint8_t
    {
        auto it = regionIndexByCoord.find(chunk->coord);
        if (it == regionIndexByCoord.end())
        {
            return packLightLevels(0, 0);
        }
        return baseLightLevels[it->second][idx];
    };

    auto seedSkyLight = [&](const glm::ivec3& worldPos, std::uint8_t level)
    {
        auto [chunk, idx] = accessRegionVoxel(worldPos);
        if (!chunk || isOpaqueForLighting(chunk->blocks[idx]))
        {
            return;
        }

        if (level > skyLightFromPacked(chunk->lightLevels[idx]))
        {
            setSkyLight(chunk->lightLevels[idx], level);
            if (level > 1)
            {
                skyQueue.push_back({worldPos, level});
                ++skySeedNodes;
            }
        }
    };

    auto seedBlockLight = [&](const glm::ivec3& worldPos, std::uint8_t level)
    {
        auto [chunk, idx] = accessRegionVoxel(worldPos);
        if (!chunk)
        {
            return;
        }

        if (level > blockLightFromPacked(chunk->lightLevels[idx]))
        {
            setBlockLight(chunk->lightLevels[idx], level);
            if (level > 1)
            {
                blockQueue.push_back({worldPos, level});
                ++blockSeedNodes;
            }
        }
    };

    auto packedLightFromBatchSnapshot = [&](const glm::ivec3& worldPos) -> std::uint8_t
    {
        if (worldPos.y < 0)
        {
            return packLightLevels(0, 0);
        }

        const glm::ivec3 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
        if (auto regionIt = regionLookup.find(chunkCoord); regionIt != regionLookup.end())
        {
            const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
            if (local.x < 0 || local.x >= kChunkSizeX ||
                local.z < 0 || local.z >= kChunkSizeZ ||
                worldPos.y < regionIt->second->minWorldY ||
                worldPos.y > regionIt->second->maxWorldY)
            {
                return packLightLevels(kMaxLightLevel, 0);
            }

            const int localY = worldPos.y - regionIt->second->minWorldY;
            return regionIt->second->lightLevels[blockIndex(local.x, localY, local.z)];
        }

        const RelightChunkReadSnapshot* snapshot = findExternalSnapshot(chunkCoord);
        if (!snapshot)
        {
            return packLightLevels(kMaxLightLevel, 0);
        }

        const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
        if (local.x < 0 || local.x >= kChunkSizeX ||
            local.z < 0 || local.z >= kChunkSizeZ ||
            worldPos.y < snapshot->minWorldY ||
            worldPos.y > snapshot->maxWorldY)
        {
            return packLightLevels(kMaxLightLevel, 0);
        }

        const int localY = worldPos.y - snapshot->minWorldY;
        return snapshot->lightLevels[blockIndex(local.x, localY, local.z)];
    };

    std::vector<std::shared_ptr<Chunk>> verticalOrder = regionChunks;
    std::sort(verticalOrder.begin(),
              verticalOrder.end(),
              [](const std::shared_ptr<Chunk>& lhs, const std::shared_ptr<Chunk>& rhs)
              {
                  if (lhs->coord.y != rhs->coord.y)
                  {
                      return lhs->coord.y > rhs->coord.y;
                  }
                  if (lhs->coord.x != rhs->coord.x)
                  {
                      return lhs->coord.x < rhs->coord.x;
                  }
                  return lhs->coord.z < rhs->coord.z;
              });

    for (const auto& chunk : verticalOrder)
    {
        auto regionIndexIt = regionIndexByCoord.find(chunk->coord);
        if (regionIndexIt == regionIndexByCoord.end())
        {
            continue;
        }

        auto& baseLights = baseLightLevels[regionIndexIt->second];
        const int baseWorldX = chunk->coord.x * kChunkSizeX;
        const int baseWorldZ = chunk->coord.z * kChunkSizeZ;

        for (int localX = 0; localX < kChunkSizeX; ++localX)
        {
            for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
            {
                const int worldX = baseWorldX + localX;
                const int worldZ = baseWorldZ + localZ;
                const std::size_t columnIndex = static_cast<std::size_t>(localZ * kChunkSizeX + localX);
                std::uint8_t incomingSky = chunk->skyLightFromAboveCache[columnIndex];

                for (int localY = kChunkSizeY - 1; localY >= 0; --localY)
                {
                    const std::size_t idx = blockIndex(localX, localY, localZ);
                    const BlockId block = chunk->blocks[idx];
                    const glm::ivec3 worldPos(worldX, chunk->minWorldY + localY, worldZ);

                    if (isOpaqueForLighting(block))
                    {
                        incomingSky = 0;
                    }
                    else
                    {
                        const std::uint8_t attenuation = blockLightingProperties(block).skyAttenuation;
                        incomingSky = static_cast<std::uint8_t>(
                            std::max(0, static_cast<int>(incomingSky) - static_cast<int>(attenuation)));
                    }

                    setSkyLight(baseLights[idx], incomingSky);
                    const std::uint8_t emission = blockLightingProperties(block).blockEmission;
                    setBlockLight(baseLights[idx], emission);
                }
            }
        }
    }

    for (std::size_t chunkIndex = 0; chunkIndex < regionChunks.size(); ++chunkIndex)
    {
        auto& chunk = regionChunks[chunkIndex];
        if (recomputeCoords.find(chunk->coord) != recomputeCoords.end())
        {
            chunk->lightLevels = baseLightLevels[chunkIndex];
        }
        else
        {
            chunk->lightLevels = previousLights[chunkIndex];
        }
    }

    for (std::size_t chunkIndex = 0; chunkIndex < regionChunks.size(); ++chunkIndex)
    {
        const auto& chunk = regionChunks[chunkIndex];
        if (recomputeCoords.find(chunk->coord) == recomputeCoords.end())
        {
            continue;
        }

        const int baseWorldX = chunk->coord.x * kChunkSizeX;
        const int baseWorldZ = chunk->coord.z * kChunkSizeZ;
        for (int localX = 0; localX < kChunkSizeX; ++localX)
        {
            for (int localY = 0; localY < kChunkSizeY; ++localY)
            {
                for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
                {
                    const std::size_t idx = blockIndex(localX, localY, localZ);
                    const glm::ivec3 worldPos(baseWorldX + localX,
                                              chunk->minWorldY + localY,
                                              baseWorldZ + localZ);
                    const std::uint8_t oldPacked = previousLights[chunkIndex][idx];
                    const std::uint8_t newPacked = chunk->lightLevels[idx];

                    const std::uint8_t oldSky = skyLightFromPacked(oldPacked);
                    const std::uint8_t oldBlock = blockLightFromPacked(oldPacked);
                    const std::uint8_t newSky = skyLightFromPacked(newPacked);
                    const std::uint8_t newBlock = blockLightFromPacked(newPacked);

                    if (oldSky != newSky)
                    {
                        if (oldSky > newSky && oldSky > 1)
                        {
                            skyRemovalQueue.push_back({worldPos, oldSky});
                        }
                        if (newSky > 1)
                        {
                            skyQueue.push_back({worldPos, newSky});
                            ++skySeedNodes;
                        }
                    }

                    if (oldBlock != newBlock)
                    {
                        if (oldBlock > newBlock && oldBlock > 1)
                        {
                            blockRemovalQueue.push_back({worldPos, oldBlock});
                        }
                        if (newBlock > 1)
                        {
                            blockQueue.push_back({worldPos, newBlock});
                            ++blockSeedNodes;
                        }
                    }
                }
            }
        }
    }

    auto propagateLightRemoval = [&](std::deque<LightNode>& removalQueue,
                                     bool skyChannel,
                                     std::deque<LightNode>& addQueue,
                                     std::uint64_t& processedNodeCounter)
    {
        while (!removalQueue.empty())
        {
            const LightNode node = removalQueue.front();
            removalQueue.pop_front();
            ++processedNodeCounter;

            for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
            {
                const glm::ivec3 neighborPos = node.worldPos + faceOffset(face);
                auto [targetChunk, targetIdx] = accessRegionVoxel(neighborPos);
                if (!targetChunk)
                {
                    continue;
                }

                const BlockId targetBlock = targetChunk->blocks[targetIdx];
                if (isOpaqueForLighting(targetBlock))
                {
                    continue;
                }

                const std::uint8_t loss = propagationLossFor(targetBlock);
                if (node.level <= loss)
                {
                    continue;
                }

                const std::uint8_t removedLevel = static_cast<std::uint8_t>(node.level - loss);
                const std::uint8_t existingLevel =
                    channelFromPacked(targetChunk->lightLevels[targetIdx], skyChannel);
                if (existingLevel == 0)
                {
                    continue;
                }

                const std::uint8_t baseLevel =
                    channelFromPacked(basePackedAtRegionVoxel(targetChunk, targetIdx), skyChannel);
                if (existingLevel <= removedLevel)
                {
                    if (existingLevel != baseLevel)
                    {
                        setPackedChannel(targetChunk->lightLevels[targetIdx], skyChannel, baseLevel);
                        if (existingLevel > baseLevel)
                        {
                            removalQueue.push_back({neighborPos, existingLevel});
                        }
                    }

                    if (baseLevel > 1)
                    {
                        addQueue.push_back({neighborPos, baseLevel});
                    }
                }
                else if (existingLevel > 1)
                {
                    addQueue.push_back({neighborPos, existingLevel});
                }
            }
        }
    };

    propagateLightRemoval(skyRemovalQueue, true, skyQueue, skyNodesProcessed);
    propagateLightRemoval(blockRemovalQueue, false, blockQueue, blockNodesProcessed);

    for (auto& chunk : regionChunks)
    {
        for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
        {
            const glm::ivec3 neighborCoord = chunk->coord + faceOffset(face);
            if (regionLookup.find(neighborCoord) != regionLookup.end())
            {
                continue;
            }

            const RelightChunkReadSnapshot* outsideSnapshot = findExternalSnapshot(neighborCoord);
            if (!outsideSnapshot)
            {
                chunk->lightBoundaryDirtyMask |= static_cast<std::uint8_t>(1u << toIndex(face));
                continue;
            }

            const glm::ivec3 offset = faceOffset(face);
            for (int localX = 0; localX < kChunkSizeX; ++localX)
            {
                for (int localY = 0; localY < kChunkSizeY; ++localY)
                {
                    for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
                    {
                        if ((offset.x < 0 && localX != 0) ||
                            (offset.x > 0 && localX != kChunkSizeX - 1) ||
                            (offset.y < 0 && localY != 0) ||
                            (offset.y > 0 && localY != kChunkSizeY - 1) ||
                            (offset.z < 0 && localZ != 0) ||
                            (offset.z > 0 && localZ != kChunkSizeZ - 1))
                        {
                            continue;
                        }

                        const std::size_t idx = blockIndex(localX, localY, localZ);
                        const BlockId block = chunk->blocks[idx];
                        if (isOpaqueForLighting(block))
                        {
                            continue;
                        }

                        const glm::ivec3 worldPos(chunk->coord.x * kChunkSizeX + localX,
                                                  chunk->minWorldY + localY,
                                                  chunk->coord.z * kChunkSizeZ + localZ);
                        const std::uint8_t neighborPacked = packedLightFromBatchSnapshot(worldPos + offset);
                        const std::uint8_t loss = propagationLossFor(block);
                        const std::uint8_t skySeed =
                            (skyLightFromPacked(neighborPacked) > loss)
                                ? static_cast<std::uint8_t>(skyLightFromPacked(neighborPacked) - loss)
                                : 0;
                        const std::uint8_t blockSeed =
                            (blockLightFromPacked(neighborPacked) > loss)
                                ? static_cast<std::uint8_t>(blockLightFromPacked(neighborPacked) - loss)
                                : 0;
                        if (skySeed > 0)
                        {
                            seedSkyLight(worldPos, skySeed);
                        }
                        if (blockSeed > 0)
                        {
                            seedBlockLight(worldPos, blockSeed);
                        }
                    }
                }
            }
        }
    }

    auto propagateLight = [&](std::deque<LightNode>& queue, bool skyChannel, std::uint64_t& processedNodeCounter)
    {
        while (!queue.empty())
        {
            const LightNode node = queue.front();
            queue.pop_front();
            ++processedNodeCounter;

            auto [sourceChunk, sourceIdx] = accessRegionVoxel(node.worldPos);
            if (!sourceChunk)
            {
                continue;
            }

            const std::uint8_t currentLevel =
                skyChannel ? skyLightFromPacked(sourceChunk->lightLevels[sourceIdx])
                           : blockLightFromPacked(sourceChunk->lightLevels[sourceIdx]);
            if (currentLevel != node.level || currentLevel == 0)
            {
                continue;
            }

            for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
            {
                const glm::ivec3 neighborPos = node.worldPos + faceOffset(face);
                auto [targetChunk, targetIdx] = accessRegionVoxel(neighborPos);
                if (!targetChunk)
                {
                    continue;
                }

                const BlockId targetBlock = targetChunk->blocks[targetIdx];
                if (isOpaqueForLighting(targetBlock))
                {
                    continue;
                }

                const std::uint8_t loss = propagationLossFor(targetBlock);
                if (currentLevel <= loss)
                {
                    continue;
                }

                const std::uint8_t nextLevel = static_cast<std::uint8_t>(currentLevel - loss);
                const std::uint8_t existingLevel =
                    skyChannel ? skyLightFromPacked(targetChunk->lightLevels[targetIdx])
                               : blockLightFromPacked(targetChunk->lightLevels[targetIdx]);
                if (nextLevel <= existingLevel)
                {
                    continue;
                }

                if (skyChannel)
                {
                    setSkyLight(targetChunk->lightLevels[targetIdx], nextLevel);
                }
                else
                {
                    setBlockLight(targetChunk->lightLevels[targetIdx], nextLevel);
                }

                if (nextLevel > 1)
                {
                    queue.push_back({neighborPos, nextLevel});
                }
            }
        }
    };

    propagateLight(skyQueue, true, skyNodesProcessed);
    propagateLight(blockQueue, false, blockNodesProcessed);

    std::vector<std::shared_ptr<Chunk>> changedChunks;
    changedChunks.reserve(regionChunks.size());
    std::unordered_set<glm::ivec3, ChunkHasher> changedChunkCoords;
    changedChunkCoords.reserve(regionChunks.size());
    for (std::size_t i = 0; i < regionChunks.size(); ++i)
    {
        if (regionChunks[i]->lightLevels != previousLights[i])
        {
            changedChunkCoords.insert(regionChunks[i]->coord);
            changedChunks.push_back(regionChunks[i]);
        }
    }

    const std::uint64_t relightRegionChunkCount = static_cast<std::uint64_t>(regionChunks.size());
    const std::uint64_t relightChangedChunkCount = static_cast<std::uint64_t>(changedChunks.size());
    const std::uint64_t relightExternalSnapshotChunkCount =
        static_cast<std::uint64_t>(externalReadSnapshots.size());

    locks.clear();

    for (const auto& [coord, generation] : batch.dirtyCoordGenerations)
    {
        auto regionIt = regionLookup.find(coord);
        if (regionIt == regionLookup.end() || !regionIt->second)
        {
            continue;
        }

        if (regionIt->second->lightingRevision.load(std::memory_order_acquire) <= generation)
        {
            regionIt->second->appliedLightingRevision.store(generation, std::memory_order_release);
        }
    }

    for (const auto& chunk : changedChunks)
    {
        requestChunkRemeshFromRelight(chunk);
    }

    if (!batch.forceRemeshCoords.empty())
    {
        for (const glm::ivec3& coord : batch.forceRemeshCoords)
        {
            if (changedChunkCoords.find(coord) != changedChunkCoords.end())
            {
                continue;
            }

            auto chunk = getChunkShared(coord);
            if (!chunk)
            {
                continue;
            }

            requestChunkRemeshFromRelight(chunk);
        }
    }

    const auto relightEnd = SteadyClock::now();
    const auto relightMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(relightEnd - relightStart).count();
    profilingCounters_.relightMicros.fetch_add(relightMicros, std::memory_order_relaxed);
    profilingCounters_.relightRegionChunks.fetch_add(relightRegionChunkCount, std::memory_order_relaxed);
    profilingCounters_.relightChangedChunks.fetch_add(relightChangedChunkCount, std::memory_order_relaxed);
    profilingCounters_.relightExternalSnapshotChunks.fetch_add(relightExternalSnapshotChunkCount, std::memory_order_relaxed);
    profilingCounters_.relightSkyAboveChunkScans.fetch_add(skyAboveChunkScans, std::memory_order_relaxed);
    profilingCounters_.relightSkySeedNodes.fetch_add(skySeedNodes, std::memory_order_relaxed);
    profilingCounters_.relightBlockSeedNodes.fetch_add(blockSeedNodes, std::memory_order_relaxed);
    profilingCounters_.relightSkyNodesProcessed.fetch_add(skyNodesProcessed, std::memory_order_relaxed);
    profilingCounters_.relightBlockNodesProcessed.fetch_add(blockNodesProcessed, std::memory_order_relaxed);
    profilingCounters_.relitChunks.fetch_add(1, std::memory_order_relaxed);
    if (benchmarkEnabled)
    {
        benchmarkMetrics_.relightStage.recordMicros(static_cast<std::uint64_t>(relightMicros));
        benchmarkMetrics_.relightRegionChunks.record(relightRegionChunkCount);
        benchmarkMetrics_.relightChangedChunks.record(relightChangedChunkCount);
        benchmarkMetrics_.relightExternalSnapshotChunks.record(relightExternalSnapshotChunkCount);
        benchmarkMetrics_.relightSkyAboveChunkScans.record(skyAboveChunkScans);
        benchmarkMetrics_.relightSkySeedNodes.record(skySeedNodes);
        benchmarkMetrics_.relightBlockSeedNodes.record(blockSeedNodes);
        benchmarkMetrics_.relightSkyNodesProcessed.record(skyNodesProcessed);
        benchmarkMetrics_.relightBlockNodesProcessed.record(blockNodesProcessed);
    }
}

void ChunkManager::Impl::relightAroundChunk(const glm::ivec3& centerCoord)
{
    PendingRelightBatch batch{};
    batch.valid = true;
    batch.sequence = 0;
    batch.dirtyCoordGenerations.emplace(centerCoord, 1);
    recomputePendingRelightBatchBounds(batch);
    batch.estimatedCostUnits = estimatePendingRelightBatchCost(batch);
    relightChunkRegion(batch);
}

void ChunkManager::Impl::noteChunkReadyLatency(Chunk& chunk)
{
    if (!benchmarkMetrics_.isEnabled())
    {
        return;
    }

    if (chunk.initialReadyRecorded.exchange(true, std::memory_order_acq_rel))
    {
        return;
    }

    const long long requestedMicros = chunk.requestTimestampMicros.load(std::memory_order_acquire);
    if (requestedMicros <= 0)
    {
        return;
    }

    const std::uint64_t readyMicros = steadyMicrosNow();
    const std::uint64_t requestMicros = static_cast<std::uint64_t>(requestedMicros);
    if (readyMicros <= requestMicros)
    {
        benchmarkMetrics_.chunkReadyLatency.recordMicros(0);
        return;
    }

    benchmarkMetrics_.chunkReadyLatency.recordMicros(readyMicros - requestMicros);

    const std::uint64_t generateStartMicros = loadBenchmarkTimestamp(chunk.generateStartTimestampMicros);
    const std::uint64_t generateDoneMicros = loadBenchmarkTimestamp(chunk.generateDoneTimestampMicros);
    const std::uint64_t meshQueuedMicros = loadBenchmarkTimestamp(chunk.meshQueuedTimestampMicros);
    const std::uint64_t meshStartMicros = loadBenchmarkTimestamp(chunk.meshStartTimestampMicros);
    const std::uint64_t meshDoneMicros = loadBenchmarkTimestamp(chunk.meshDoneTimestampMicros);
    const std::uint64_t uploadQueuedMicros = loadBenchmarkTimestamp(chunk.uploadQueuedTimestampMicros);
    const std::uint64_t uploadStartMicros = loadBenchmarkTimestamp(chunk.uploadStartTimestampMicros);

    auto recordStage = [&](AtomicLatencyHistogram& stage, std::uint64_t beginMicros, std::uint64_t endMicros)
    {
        if (beginMicros == 0 || endMicros == 0 || endMicros <= beginMicros)
        {
            return;
        }

        stage.recordMicros(endMicros - beginMicros);
    };

    recordStage(benchmarkMetrics_.chunkReadyWaitGenerateStage, requestMicros, generateStartMicros);
    recordStage(benchmarkMetrics_.chunkReadyGenerateStage, generateStartMicros, generateDoneMicros);
    recordStage(benchmarkMetrics_.chunkReadyWaitMeshEnqueueStage, generateDoneMicros, meshQueuedMicros);
    recordStage(benchmarkMetrics_.chunkReadyWaitMeshStartStage, meshQueuedMicros, meshStartMicros);
    recordStage(benchmarkMetrics_.chunkReadyMeshStage, meshStartMicros, meshDoneMicros);
    const std::uint64_t uploadWaitStartMicros = uploadQueuedMicros != 0 ? uploadQueuedMicros : meshDoneMicros;
    recordStage(benchmarkMetrics_.chunkReadyWaitUploadStage, uploadWaitStartMicros, uploadStartMicros);
    recordStage(benchmarkMetrics_.chunkReadyUploadToReadyStage, uploadStartMicros, readyMicros);
}







ColumnSample ChunkManager::Impl::sampleColumn(
    int worldX,
    int worldZ,
    int slabMinWorldY,
    int slabMaxWorldY,
    bool includeBlendDebug) const
{
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();
    const SteadyClock::time_point sampleStart = benchmarkEnabled ? SteadyClock::now() : SteadyClock::time_point{};
    const bool usesDefaultSlabBounds =
        slabMinWorldY == std::numeric_limits<int>::min() && slabMaxWorldY == std::numeric_limits<int>::max();
    if (slabMinWorldY > slabMaxWorldY)
    {
        std::swap(slabMinWorldY, slabMaxWorldY);
    }

    if (!surfaceMap_)
    {
        throw std::runtime_error("Surface map is not initialized");
    }
    if (!climateMap_)
    {
        throw std::runtime_error("Climate map is not initialized");
    }

    ColumnSample sample{};
    const terrain::SurfaceColumn surfaceColumn = surfaceMap_->columnValue(worldX, worldZ);
    const terrain::ClimateSample climateSample = climateMap_->sample(worldX, worldZ);

    sample.dominantBiome = surfaceColumn.dominantBiome;
    sample.dominantWeight = surfaceColumn.dominantWeight;
    sample.surfaceHeight = surfaceColumn.surfaceHeight;
    sample.surfaceY = surfaceColumn.surfaceY;
    sample.originalSurfaceY = surfaceColumn.surfaceY;
    if (usesDefaultSlabBounds)
    {
        slabMinWorldY = sample.surfaceY;
        slabMaxWorldY = sample.surfaceY;
    }
    sample.minSurfaceY = std::min(sample.surfaceY, slabMinWorldY);
    sample.maxSurfaceY = std::max(sample.surfaceY, slabMaxWorldY);
    sample.soilCreepCoefficient = surfaceColumn.soilCreepCoefficient;
    sample.roughAmplitude = surfaceColumn.roughAmplitude;
    sample.hillAmplitude = surfaceColumn.hillAmplitude;
    sample.mountainAmplitude = surfaceColumn.mountainAmplitude;
    sample.dominantIsOcean = climateSample.dominantIsOcean;
    sample.distanceToCoast = climateSample.distanceToCoast;
    sample.distanceToShore = std::isfinite(climateSample.distanceToCoast)
                                 ? climateSample.distanceToCoast
                                 : std::numeric_limits<float>::infinity();
    sample.soilCreepOffset = 0.0f;

    sample.topBlendCount = includeBlendDebug ? std::min(climateSample.blendCount, sample.topBlendDebug.size()) : 0;
    const glm::vec2 columnPos(static_cast<float>(worldX), static_cast<float>(worldZ));
    for (std::size_t i = 0; i < sample.topBlendCount; ++i)
    {
        const auto& srcBlend = climateSample.blends[i];
        auto& dstBlend = sample.topBlendDebug[i];
        dstBlend.biome = srcBlend.biome;
        dstBlend.weight = srcBlend.weight;
        dstBlend.aggregatedHeight = srcBlend.height;
        dstBlend.normalizedDistance = srcBlend.normalizedDistance;
        dstBlend.seedRadius = srcBlend.falloff;
        dstBlend.worldDistance = glm::length(columnPos - srcBlend.sitePosition);
        dstBlend.isOcean = srcBlend.biome && srcBlend.biome->isOcean();
    }

    sample.slabHasSolid = surfaceColumn.surfaceY >= slabMinWorldY;
    if (sample.slabHasSolid)
    {
        sample.slabHighestSolidY = std::min(surfaceColumn.surfaceY, slabMaxWorldY);
    }

    if (!std::isfinite(sample.distanceToShore))
    {
        if (sample.dominantBiome && sample.dominantBiome->isOcean())
        {
            sample.distanceToShore = 0.0f;
            sample.distanceToCoast = 0.0f;
        }
    }

    if (benchmarkEnabled)
    {
        const auto sampleEnd = SteadyClock::now();
        const auto sampleMicros =
            std::chrono::duration_cast<std::chrono::microseconds>(sampleEnd - sampleStart).count();
        benchmarkMetrics_.sampleStage.recordMicros(static_cast<std::uint64_t>(sampleMicros));
    }

    return sample;
}



std::vector<StructureInstance> ChunkManager::Impl::queryStructureInstances(const glm::ivec3& minWorld,
                                                                           const glm::ivec3& maxWorld,
                                                                           int lodLevel) const
{
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();
    const SteadyClock::time_point queryStart = benchmarkEnabled ? SteadyClock::now() : SteadyClock::time_point{};
    std::vector<StructureInstance> instances = structureRegistry_.query(minWorld, maxWorld, lodLevel);
    if (benchmarkEnabled)
    {
        const auto queryMicros =
            std::chrono::duration_cast<std::chrono::microseconds>(SteadyClock::now() - queryStart).count();
        benchmarkMetrics_.structureQueryStage.recordMicros(
            static_cast<std::uint64_t>(std::max<std::int64_t>(queryMicros, 0)));
    }
    return instances;
}

namespace
{
struct ChunkBuildScratch
{
    explicit ChunkBuildScratch(const Chunk& chunk)
        : coord(chunk.coord),
          minWorldY(chunk.minWorldY),
          maxWorldY(chunk.maxWorldY),
          blocks(kChunkBlockCount, BlockId::Air)
    {
        highestSolidWorlds.fill(ColumnManager::kNoHeight);
    }

    void setGeneratedLocalBlock(int localX, int localY, int localZ, BlockId block)
    {
        if (localX < 0 || localX >= kChunkSizeX ||
            localZ < 0 || localZ >= kChunkSizeZ ||
            localY < 0 || localY >= kChunkSizeY)
        {
            return;
        }

        blocks[blockIndex(localX, localY, localZ)] = block;
    }

    [[nodiscard]] int scanColumnHighestWorld(int localX, int localZ) const noexcept
    {
        for (int localY = kChunkSizeY - 1; localY >= 0; --localY)
        {
            if (isSolid(blocks[blockIndex(localX, localY, localZ)]))
            {
                return minWorldY + localY;
            }
        }
        return ColumnManager::kNoHeight;
    }

    bool setWorldBlock(int worldX, int worldY, int worldZ, BlockId block, bool replaceSolid)
    {
        const glm::ivec3 worldPos{worldX, worldY, worldZ};
        const glm::ivec3 local = localBlockCoords(worldPos, coord);
        if (local.x < 0 || local.x >= kChunkSizeX ||
            local.z < 0 || local.z >= kChunkSizeZ ||
            worldY < minWorldY || worldY > maxWorldY)
        {
            return false;
        }

        const int localY = worldY - minWorldY;
        BlockId& destination = blocks[blockIndex(local.x, localY, local.z)];
        if (!replaceSolid && destination != BlockId::Air)
        {
            return false;
        }

        const BlockId previous = destination;
        if (previous == block)
        {
            return false;
        }

        destination = block;

        const std::size_t heightIndex = columnIndex(local.x, local.z);
        if (isSolid(block))
        {
            highestSolidWorlds[heightIndex] = std::max(highestSolidWorlds[heightIndex], worldY);
        }
        else if (isSolid(previous) && highestSolidWorlds[heightIndex] == worldY)
        {
            highestSolidWorlds[heightIndex] = scanColumnHighestWorld(local.x, local.z);
        }
        return true;
    }

    glm::ivec3 coord{0};
    int minWorldY{0};
    int maxWorldY{0};
    std::vector<BlockId> blocks;
    std::array<int, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)> highestSolidWorlds{};
};

template <typename WriteBlockFn>
void stampStructureInstanceIntoTarget(const StructureInstance& instance, WriteBlockFn&& writeBlock)
{
    auto setLocalBlock = [&](int worldX, int worldY, int worldZ, BlockId block, bool replaceSolid)
    {
        writeBlock(worldX, worldY, worldZ, block, replaceSolid);
    };

    if (instance.type == StructureType::TaigaSpruce)
    {
        for (int trunkX = 0; trunkX < 2; ++trunkX)
        {
            for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
            {
                for (int dy = 1; dy <= instance.trunkHeight; ++dy)
                {
                    setLocalBlock(instance.origin.x + trunkX,
                                  instance.origin.y + dy,
                                  instance.origin.z + trunkZ,
                                  BlockId::SpruceLog,
                                  true);
                }
            }
        }

        const int canopyBaseWorld = instance.origin.y + instance.bareTrunkHeight + 1;
        const int canopyTopWorld = instance.origin.y + instance.trunkHeight;
        const int totalLayers = std::max(1, canopyTopWorld - canopyBaseWorld + 1);
        for (int worldY = canopyBaseWorld; worldY <= canopyTopWorld; ++worldY)
        {
            const int layerFromBottom = worldY - canopyBaseWorld;
            const int radius = taigaSpruceLeafRadiusForLayer(layerFromBottom, totalLayers);
            if (radius <= 0)
            {
                continue;
            }

            for (int worldX = instance.origin.x - radius; worldX <= instance.origin.x + 1 + radius; ++worldX)
            {
                for (int worldZ = instance.origin.z - radius; worldZ <= instance.origin.z + 1 + radius; ++worldZ)
                {
                    if (!taigaSpruceLeafOccupiesCell(instance.origin.x,
                                                     instance.origin.z,
                                                     worldX,
                                                     worldZ,
                                                     radius,
                                                     layerFromBottom,
                                                     totalLayers))
                    {
                        continue;
                    }
                    setLocalBlock(worldX, worldY, worldZ, BlockId::SpruceLeaves, false);
                }
            }
        }

        const int crownWorldY = canopyTopWorld + 1;
        for (int trunkX = 0; trunkX < 2; ++trunkX)
        {
            for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
            {
                setLocalBlock(instance.origin.x + trunkX,
                              crownWorldY,
                              instance.origin.z + trunkZ,
                              BlockId::SpruceLeaves,
                              false);
            }
        }
        return;
    }

    if (instance.type == StructureType::DarkOak)
    {
        forEachDarkOakTreeBlock(instance.origin.x,
                                instance.origin.z,
                                instance.origin.y,
                                instance.trunkHeight,
                                instance.trunkBlock,
                                instance.leavesBlock,
                                [&](int blockX, int blockY, int blockZ, BlockId block) {
                                    setLocalBlock(blockX, blockY, blockZ, block, block == instance.trunkBlock);
                                    return false;
                                });
        return;
    }

    if (instance.type == StructureType::Acacia)
    {
        forEachAcaciaTreeBlock(instance.origin.x,
                               instance.origin.z,
                               instance.origin.y,
                               instance.trunkHeight,
                               instance.trunkBlock,
                               instance.leavesBlock,
                               [&](int blockX, int blockY, int blockZ, BlockId block) {
                                   setLocalBlock(blockX, blockY, blockZ, block, block == instance.trunkBlock);
                                   return false;
                               });
        return;
    }

    forEachDefaultTreeBlock(instance.origin.x,
                            instance.origin.z,
                            instance.origin.y,
                            instance.trunkHeight,
                            instance.trunkBlock,
                            instance.leavesBlock,
                            [&](int blockX, int blockY, int blockZ, BlockId block) {
                                setLocalBlock(blockX, blockY, blockZ, block, block == instance.trunkBlock);
                                return false;
                            });
}
} // namespace

bool ChunkManager::Impl::generateChunkBlocks(Chunk& chunk, std::uint32_t generationEpoch)
{
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();
    ChunkBuildScratch scratch(chunk);
    std::array<ColumnBuildResult, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)> columnResults{};
    std::vector<PendingStructureEdit> pendingEdits;
    buildChunkCpuBlocks(chunk, scratch, true, columnResults, &pendingEdits);

    // `hasBlocks` controls whether the chunk enters relight/mesh/upload. Water-only
    // slabs and canopy-only slabs must count here even though they do not raise the
    // column manager's "highest solid" height.
    const bool anyBlocks =
        std::any_of(scratch.blocks.begin(),
                    scratch.blocks.end(),
                    [](BlockId block) { return block != BlockId::Air; });

    const auto meshLockStart = benchmarkEnabled ? SteadyClock::now() : SteadyClock::time_point{};
    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        const bool staleGeneration =
            chunk.generationEpoch.load(std::memory_order_acquire) != generationEpoch ||
            chunk.state.load(std::memory_order_acquire) != ChunkState::Generating;
        if (staleGeneration)
        {
            if (!pendingEdits.empty())
            {
                std::lock_guard<std::mutex> pendingLock(pendingStructureMutex_);
                auto& restoredEdits = pendingStructureEdits_[chunk.coord];
                restoredEdits.insert(restoredEdits.end(),
                                     std::make_move_iterator(pendingEdits.begin()),
                                     std::make_move_iterator(pendingEdits.end()));
            }

            if (benchmarkEnabled)
            {
                const auto lockMicros =
                    std::chrono::duration_cast<std::chrono::microseconds>(SteadyClock::now() - meshLockStart).count();
                benchmarkMetrics_.generateBlocksMeshLockStage.recordMicros(static_cast<std::uint64_t>(lockMicros));
            }
            return false;
        }

        // Publish the fully built block slab in one short handoff so upload/commit only contends on swaps.
        chunk.ensureCpuDataAllocated();
        chunk.blocks = std::move(scratch.blocks);
        chunk.hasBlocks.store(anyBlocks, std::memory_order_release);
        chunk.lastDenseFrameTouched = updateFrameIndex_;
    }
    if (benchmarkEnabled)
    {
        const auto lockMicros =
            std::chrono::duration_cast<std::chrono::microseconds>(SteadyClock::now() - meshLockStart).count();
        benchmarkMetrics_.generateBlocksMeshLockStage.recordMicros(static_cast<std::uint64_t>(lockMicros));
    }

    const glm::ivec2 column{chunk.coord.x, chunk.coord.z};
    columnManager_.updateChunkHeights(chunk.coord, scratch.highestSolidWorlds);
    mergePredictedColumnHeight(column,
                               columnManager_.highestSolidBlock(column.x * kChunkSizeX + kChunkSizeX / 2,
                                                               column.y * kChunkSizeZ + kChunkSizeZ / 2));
    markSkyLightColumnDirty({chunk.coord.x, chunk.coord.z});
    return true;
}

std::vector<PendingStructureEdit> ChunkManager::Impl::takePendingStructureEdits(const glm::ivec3& coord)
{
    std::vector<PendingStructureEdit> edits;
    {
        std::lock_guard<std::mutex> lock(pendingStructureMutex_);
        auto it = pendingStructureEdits_.find(coord);
        if (it != pendingStructureEdits_.end())
        {
            edits = std::move(it->second);
            pendingStructureEdits_.erase(it);
        }
    }
    if (!edits.empty())
    {
        invalidateColumnSlabOccupancy({coord.x, coord.z});
    }
    return edits;
}

std::vector<PendingStructureEdit> ChunkManager::Impl::copyPendingStructureEdits(const glm::ivec3& coord) const
{
    std::vector<PendingStructureEdit> edits;
    std::lock_guard<std::mutex> lock(pendingStructureMutex_);
    auto it = pendingStructureEdits_.find(coord);
    if (it != pendingStructureEdits_.end())
    {
        edits = it->second;
    }
    return edits;
}

std::uint16_t ChunkManager::Impl::blockOverlayLocalIndex(int localX, int localY, int localZ) noexcept
{
    return static_cast<std::uint16_t>(blockIndex(localX, localY, localZ));
}

void ChunkManager::Impl::recordBlockEditOverlay(const glm::ivec3& worldPos, BlockId block)
{
    const glm::ivec3 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
    const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
    if (local.x < 0 || local.x >= kChunkSizeX ||
        local.y < 0 || local.y >= kChunkSizeY ||
        local.z < 0 || local.z >= kChunkSizeZ)
    {
        return;
    }

    const std::uint16_t localIndex = blockOverlayLocalIndex(local.x, local.y, local.z);
    {
        std::lock_guard<std::mutex> lock(blockEditOverlayMutex_);
        auto& overlays = blockEditOverlays_[chunkCoord];
        auto it = std::find_if(overlays.begin(),
                               overlays.end(),
                               [localIndex](const BlockEditOverlayEntry& entry) {
                                   return entry.localIndex == localIndex;
                               });
        if (it != overlays.end())
        {
            it->block = block;
        }
        else
        {
            overlays.push_back(BlockEditOverlayEntry{localIndex, block});
        }
    }
    invalidateColumnSlabOccupancy({chunkCoord.x, chunkCoord.z});
}

bool ChunkManager::Impl::tryGetBlockEditOverlay(const glm::ivec3& worldPos, BlockId& outBlock) const
{
    const glm::ivec3 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
    const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
    if (local.x < 0 || local.x >= kChunkSizeX ||
        local.y < 0 || local.y >= kChunkSizeY ||
        local.z < 0 || local.z >= kChunkSizeZ)
    {
        return false;
    }

    const std::uint16_t localIndex = blockOverlayLocalIndex(local.x, local.y, local.z);
    std::lock_guard<std::mutex> lock(blockEditOverlayMutex_);
    auto chunkIt = blockEditOverlays_.find(chunkCoord);
    if (chunkIt == blockEditOverlays_.end())
    {
        return false;
    }

    const auto& overlays = chunkIt->second;
    auto it = std::find_if(overlays.begin(),
                           overlays.end(),
                           [localIndex](const BlockEditOverlayEntry& entry) {
                               return entry.localIndex == localIndex;
                           });
    if (it == overlays.end())
    {
        return false;
    }

    outBlock = it->block;
    return true;
}

void ChunkManager::Impl::applyBlockEditOverlay(ChunkBuildScratch& scratch, const glm::ivec3& chunkCoord) const
{
    std::vector<BlockEditOverlayEntry> overlays;
    {
        std::lock_guard<std::mutex> lock(blockEditOverlayMutex_);
        auto it = blockEditOverlays_.find(chunkCoord);
        if (it == blockEditOverlays_.end())
        {
            return;
        }
        overlays = it->second;
    }

    for (const BlockEditOverlayEntry& entry : overlays)
    {
        const int localX = entry.localIndex % kChunkSizeX;
        const int localZ = (entry.localIndex / kChunkSizeX) % kChunkSizeZ;
        const int localY = entry.localIndex / (kChunkSizeX * kChunkSizeZ);
        scratch.setWorldBlock(chunkCoord.x * kChunkSizeX + localX,
                              chunkCoord.y * kChunkSizeY + localY,
                              chunkCoord.z * kChunkSizeZ + localZ,
                              entry.block,
                              true);
    }
}

bool ChunkManager::Impl::chunkHasPendingStructureEdits(const glm::ivec3& coord) const
{
    std::lock_guard<std::mutex> lock(pendingStructureMutex_);
    auto it = pendingStructureEdits_.find(coord);
    return it != pendingStructureEdits_.end() && !it->second.empty();
}

void ChunkManager::Impl::buildChunkCpuBlocks(
    const Chunk& chunk,
    ChunkBuildScratch& scratch,
    bool includePendingStructureEdits,
    std::array<ColumnBuildResult, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)>& columnResults,
    std::vector<PendingStructureEdit>* consumedPendingEdits)
{
    const int baseWorldX = chunk.coord.x * kChunkSizeX;
    const int baseWorldZ = chunk.coord.z * kChunkSizeZ;

    if (surfaceMap_)
    {
        const int fragmentSize = terrain::SurfaceFragment::kSize;
        const int minFragmentX = floorDiv(baseWorldX - 1, fragmentSize);
        const int maxFragmentX = floorDiv(baseWorldX + kChunkSizeX, fragmentSize);
        const int minFragmentZ = floorDiv(baseWorldZ - 1, fragmentSize);
        const int maxFragmentZ = floorDiv(baseWorldZ + kChunkSizeZ, fragmentSize);

        for (int fx = minFragmentX; fx <= maxFragmentX; ++fx)
        {
            for (int fz = minFragmentZ; fz <= maxFragmentZ; ++fz)
            {
                const auto& prefetchedFragment = surfaceMap_->getFragment({fx, fz});
                (void)prefetchedFragment;
            }
        }
    }

    auto setBlockDirect = [&](int localX, int localY, int localZ, BlockId block)
    {
        if (localX < 0 || localX >= kChunkSizeX || localZ < 0 || localZ >= kChunkSizeZ)
        {
            return;
        }
        if (localY < 0 || localY >= kChunkSizeY)
        {
            return;
        }
        scratch.setGeneratedLocalBlock(localX, localY, localZ, block);
    };

    if (terrainGenerator_)
    {
        terrainGenerator_->generateChunkColumns(chunk.coord,
                                                chunk.minWorldY,
                                                chunk.maxWorldY,
                                                kChunkSizeX,
                                                kChunkSizeY,
                                                kChunkSizeZ,
                                                setBlockDirect,
                                                columnResults);
    }

    for (std::size_t i = 0; i < columnResults.size(); ++i)
    {
        scratch.highestSolidWorlds[i] = columnResults[i].highestSolidWorld;
    }

    const glm::ivec3 queryMin(baseWorldX, chunk.minWorldY, baseWorldZ);
    const glm::ivec3 queryMax(baseWorldX + kChunkSizeX - 1, chunk.maxWorldY, baseWorldZ + kChunkSizeZ - 1);
    const std::vector<StructureInstance> structures = queryStructureInstances(queryMin, queryMax, 0);
    for (const StructureInstance& instance : structures)
    {
        stampStructureInstanceIntoTarget(instance,
                                        [&](int worldX, int worldY, int worldZ, BlockId block, bool replaceSolid) {
                                            scratch.setWorldBlock(worldX, worldY, worldZ, block, replaceSolid);
                                        });
    }

    std::vector<PendingStructureEdit> pendingEdits = includePendingStructureEdits
        ? takePendingStructureEdits(chunk.coord)
        : copyPendingStructureEdits(chunk.coord);
    for (const PendingStructureEdit& edit : pendingEdits)
    {
        scratch.setWorldBlock(edit.worldPos.x, edit.worldPos.y, edit.worldPos.z, edit.block, edit.replaceSolid);
    }

    applyBlockEditOverlay(scratch, chunk.coord);

    if (consumedPendingEdits != nullptr)
    {
        *consumedPendingEdits = std::move(pendingEdits);
    }
}

void ChunkManager::Impl::rebuildChunkBaseLighting(Chunk& chunk) const
{
    chunk.lightLevels.assign(kChunkBlockCount, packLightLevels(0, 0));
    for (int localX = 0; localX < kChunkSizeX; ++localX)
    {
        for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
        {
            const std::size_t localColumnIndex = static_cast<std::size_t>(localZ * kChunkSizeX + localX);
            std::uint8_t incomingSky = chunk.skyLightFromAboveCache[localColumnIndex];

            for (int localY = kChunkSizeY - 1; localY >= 0; --localY)
            {
                const std::size_t idx = blockIndex(localX, localY, localZ);
                const BlockId block = chunk.blocks[idx];
                if (isOpaqueForLighting(block))
                {
                    incomingSky = 0;
                }
                else
                {
                    const std::uint8_t attenuation = blockLightingProperties(block).skyAttenuation;
                    incomingSky = static_cast<std::uint8_t>(
                        std::max(0, static_cast<int>(incomingSky) - static_cast<int>(attenuation)));
                }

                chunk.lightLevels[idx] = packLightLevels(incomingSky, blockLightingProperties(block).blockEmission);
            }
        }
    }
}

int ChunkManager::Impl::denseCpuHorizontalRadius() const noexcept
{
    return std::clamp(2 + targetViewDistance_ / 4,
                      kDenseCpuHorizontalRadiusMin,
                      kDenseCpuHorizontalRadiusMax);
}

int ChunkManager::Impl::denseCpuVerticalRadius() const noexcept
{
    return std::clamp(3 + targetViewDistance_ / 8,
                      kDenseCpuVerticalRadiusMin,
                      kDenseCpuVerticalRadiusMax);
}

bool ChunkManager::Impl::shouldKeepChunkCpuDense(const Chunk& chunk,
                                                 const glm::ivec3& centerChunk,
                                                 int horizontalRadius,
                                                 int verticalRadius) const
{
    const ChunkState state = chunk.state.load(std::memory_order_acquire);
    if (state != ChunkState::Uploaded)
    {
        return true;
    }

    if (chunk.inFlight.load(std::memory_order_acquire) > 0 ||
        chunk.queuedForUpload.load(std::memory_order_acquire) ||
        chunk.pendingMesh.valid() ||
        chunk.pendingMeshRefresh.load(std::memory_order_acquire) ||
        chunk.meshReady.load(std::memory_order_acquire) ||
        chunkHasPendingStructureEdits(chunk.coord))
    {
        return true;
    }

    const int horizontalDistance =
        std::max(std::abs(chunk.coord.x - centerChunk.x), std::abs(chunk.coord.z - centerChunk.z));
    const int verticalDistance = std::abs(chunk.coord.y - centerChunk.y);
    return horizontalDistance <= horizontalRadius && verticalDistance <= verticalRadius;
}

bool ChunkManager::Impl::ensureChunkCpuDataResident(Chunk& chunk)
{
    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        if (chunk.cpuDataResident &&
            chunk.blocks.size() == static_cast<std::size_t>(kChunkBlockCount) &&
            chunk.lightLevels.size() == static_cast<std::size_t>(kChunkBlockCount))
        {
            chunk.lastDenseFrameTouched = updateFrameIndex_;
            return true;
        }
    }

    ChunkBuildScratch scratch(chunk);
    std::array<ColumnBuildResult, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)> columnResults{};
    buildChunkCpuBlocks(chunk, scratch, false, columnResults, nullptr);

    std::lock_guard<std::mutex> lock(chunk.meshMutex);
    chunk.ensureCpuDataAllocated();
    chunk.blocks = std::move(scratch.blocks);
    chunk.hasBlocks.store(std::any_of(chunk.blocks.begin(),
                                      chunk.blocks.end(),
                                      [](BlockId block) { return block != BlockId::Air; }),
                          std::memory_order_release);
    rebuildChunkBaseLighting(chunk);
    chunk.lastDenseFrameTouched = updateFrameIndex_;
    return true;
}

void ChunkManager::Impl::releaseChunkCpuData(Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(chunk.meshMutex);
    if (!chunk.cpuDataResident)
    {
        return;
    }

    const ChunkState state = chunk.state.load(std::memory_order_acquire);
    if (state != ChunkState::Uploaded ||
        chunk.inFlight.load(std::memory_order_acquire) > 0 ||
        chunk.queuedForUpload.load(std::memory_order_acquire) ||
        chunk.pendingMesh.valid() ||
        chunk.pendingMeshRefresh.load(std::memory_order_acquire) ||
        chunk.meshReady.load(std::memory_order_acquire))
    {
        return;
    }

    chunk.releaseCpuData();
}

void ChunkManager::Impl::updateDenseChunkResidency(const glm::ivec3& centerChunk)
{
    const int horizontalRadius = denseCpuHorizontalRadius();
    const int verticalRadius = denseCpuVerticalRadius();
    const int hydrationBudget = std::clamp(1 + horizontalRadius / 3,
                                           kDenseCpuHydrationBudgetMin,
                                           kDenseCpuHydrationBudgetMax);
    const int demotionBudget = std::clamp(hydrationBudget * 8,
                                          kDenseCpuDemotionBudgetMin,
                                          kDenseCpuDemotionBudgetMax);

    std::vector<std::shared_ptr<Chunk>> chunks;
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        chunks.reserve(chunks_.size());
        for (const auto& [coord, chunk] : chunks_)
        {
            (void)coord;
            chunks.push_back(chunk);
        }
    }

    std::vector<std::shared_ptr<Chunk>> toHydrate;
    std::vector<std::shared_ptr<Chunk>> toDemote;
    toHydrate.reserve(chunks.size());
    toDemote.reserve(chunks.size());

    const auto densePriority = [&centerChunk](const std::shared_ptr<Chunk>& chunk)
    {
        const int horizontalDistance =
            std::max(std::abs(chunk->coord.x - centerChunk.x), std::abs(chunk->coord.z - centerChunk.z));
        const int verticalDistance = std::abs(chunk->coord.y - centerChunk.y);
        return std::tuple<int, int, int, int>{
            horizontalDistance,
            verticalDistance,
            chunk->coord.y,
            chunk->coord.x};
    };

    for (const std::shared_ptr<Chunk>& chunk : chunks)
    {
        if (!chunk)
        {
            continue;
        }

        const bool keepDense = shouldKeepChunkCpuDense(*chunk, centerChunk, horizontalRadius, verticalRadius);
        if (keepDense)
        {
            if (chunk->cpuDataResident)
            {
                std::lock_guard<std::mutex> lock(chunk->meshMutex);
                chunk->lastDenseFrameTouched = updateFrameIndex_;
            }
            else
            {
                toHydrate.push_back(chunk);
            }
            continue;
        }

        if (chunk->cpuDataResident)
        {
            std::uint64_t lastTouched = 0;
            {
                std::lock_guard<std::mutex> lock(chunk->meshMutex);
                lastTouched = chunk->lastDenseFrameTouched;
            }
            if (updateFrameIndex_ > lastTouched + kDenseCpuDemotionGraceFrames)
            {
                toDemote.push_back(chunk);
            }
        }
    }

    std::sort(toHydrate.begin(),
              toHydrate.end(),
              [&densePriority](const std::shared_ptr<Chunk>& lhs, const std::shared_ptr<Chunk>& rhs)
              {
                  return densePriority(lhs) < densePriority(rhs);
              });
    std::sort(toDemote.begin(),
              toDemote.end(),
              [&densePriority](const std::shared_ptr<Chunk>& lhs, const std::shared_ptr<Chunk>& rhs)
              {
                  return densePriority(lhs) > densePriority(rhs);
              });

    // Keep dense CPU hydration incremental so altitude changes do not rebuild hundreds of chunks
    // on the main thread in a single frame.
    const std::size_t hydrateCount =
        std::min<std::size_t>(toHydrate.size(), static_cast<std::size_t>(std::max(0, hydrationBudget)));
    const std::size_t demoteCount =
        std::min<std::size_t>(toDemote.size(), static_cast<std::size_t>(std::max(0, demotionBudget)));

    for (std::size_t i = 0; i < hydrateCount; ++i)
    {
        const std::shared_ptr<Chunk>& chunk = toHydrate[i];
        if (chunk && ensureChunkCpuDataResident(*chunk))
        {
            std::lock_guard<std::mutex> lock(chunk->meshMutex);
            columnManager_.updateChunk(makeChunkBlockView(*chunk));
        }
    }

    for (std::size_t i = 0; i < demoteCount; ++i)
    {
        const std::shared_ptr<Chunk>& chunk = toDemote[i];
        if (chunk)
        {
            releaseChunkCpuData(*chunk);
        }
    }
}

bool ChunkManager::Impl::applyPendingStructureEditsLocked(Chunk& chunk)
{
    std::vector<PendingStructureEdit> edits = takePendingStructureEdits(chunk.coord);

    bool wroteSolid = false;
    for (const PendingStructureEdit& edit : edits)
    {
        const glm::ivec3 local = localBlockCoords(edit.worldPos, chunk.coord);
        if (local.x < 0 || local.x >= kChunkSizeX ||
            edit.worldPos.y < chunk.minWorldY || edit.worldPos.y > chunk.maxWorldY ||
            local.z < 0 || local.z >= kChunkSizeZ)
        {
            continue;
        }

        const int localY = edit.worldPos.y - chunk.minWorldY;
        BlockId& destination = chunk.blocks[blockIndex(local.x, localY, local.z)];
        if (!edit.replaceSolid && destination != BlockId::Air)
        {
            continue;
        }

        destination = edit.block;
        if (edit.block != BlockId::Air)
        {
            wroteSolid = true;
        }
    }

    return wroteSolid;
}

void ChunkManager::Impl::dispatchStructureEdits(const std::vector<PendingStructureEdit>& edits)
{
    if (edits.empty())
    {
        return;
    }

    std::unordered_set<glm::ivec3, ChunkHasher> touchedChunks;
    touchedChunks.reserve(edits.size());

    {
        std::lock_guard<std::mutex> lock(pendingStructureMutex_);
        for (const PendingStructureEdit& edit : edits)
        {
            pendingStructureEdits_[edit.chunkCoord].push_back(edit);
            touchedChunks.insert(edit.chunkCoord);
        }
    }

    for (const glm::ivec3& coord : touchedChunks)
    {
        invalidateColumnSlabOccupancy({coord.x, coord.z});
    }

    for (const glm::ivec3& coord : touchedChunks)
    {
        auto chunk = getChunkShared(coord);
        if (!chunk)
        {
            continue;
        }

        if (!chunk->cpuDataResident && !ensureChunkCpuDataResident(*chunk))
        {
            continue;
        }

        ChunkState state = chunk->state.load(std::memory_order_acquire);
        if (state == ChunkState::Generating)
        {
            continue;
        }

        bool wroteSolid = false;
        {
            std::lock_guard<std::mutex> lock(chunk->meshMutex);
            wroteSolid = applyPendingStructureEditsLocked(*chunk);
            if (wroteSolid)
            {
                chunk->hasBlocks.store(true, std::memory_order_release);
                columnManager_.updateChunk(makeChunkBlockView(*chunk));
                refreshPredictedColumnHeightFromLoadedData({chunk->coord.x, chunk->coord.z});
                markSkyLightColumnDirty({chunk->coord.x, chunk->coord.z});
            }
        }

        if (!wroteSolid)
        {
            continue;
        }

        queueRelightRequest(coord, true);

        (void)state;
    }
}

bool ChunkManager::Impl::chunkHasSolidBlocks(const Chunk& chunk) noexcept
{
    return std::any_of(chunk.blocks.begin(), chunk.blocks.end(), [](BlockId block) { return block != BlockId::Air; });

}

ChunkManager::ChunkManager(unsigned seed)
    : impl_(std::make_unique<Impl>(seed))
{
}

ChunkManager::~ChunkManager() = default;

void ChunkManager::initializeRendering(ID3D12Device* device)
{
    impl_->initializeRendering(device);
}

void ChunkManager::setRenderSynchronization(ID3D12Fence* graphicsFence, std::uint64_t graphicsFenceValue)
{
    impl_->setRenderSynchronization(graphicsFence, graphicsFenceValue);
}

ID3D12Fence* ChunkManager::uploadFence() const noexcept
{
    return impl_->uploadFence();
}

std::uint64_t ChunkManager::lastSubmittedUploadFenceValue() const noexcept
{
    return impl_->lastSubmittedUploadFenceValue();
}

ID3D12Fence* ChunkManager::farUploadFence() const noexcept
{
    return impl_->farUploadFence();
}

std::uint64_t ChunkManager::lastSubmittedFarUploadFenceValue() const noexcept
{
    return impl_->lastSubmittedFarUploadFenceValue();
}

void ChunkManager::setBlockTextureAtlasConfig(const BlockTextureAtlasConfig& config)
{
    impl_->setBlockTextureAtlasConfig(config);
}

void ChunkManager::update(const glm::vec3& cameraPos)
{
    impl_->update(cameraPos);
}

void ChunkManager::update(const glm::vec3& cameraPos, const glm::vec3& cameraForward)
{
    impl_->update(cameraPos, cameraForward);
}

WorldRenderData ChunkManager::buildRenderData(const Frustum& frustum) const
{
    return impl_->buildRenderData(frustum);
}

float ChunkManager::surfaceHeight(float worldX, float worldZ) const noexcept
{
    return impl_->surfaceHeight(worldX, worldZ);
}

terrain::ColumnSample ChunkManager::sampleColumnAt(const glm::vec3& worldPos,
                                                   int slabMinWorldY,
                                                   int slabMaxWorldY) const
{
    return impl_->sampleColumnAt(worldPos, slabMinWorldY, slabMaxWorldY);
}

void ChunkManager::clear()
{
    impl_->clear();
}

bool ChunkManager::destroyBlock(const glm::ivec3& worldPos)
{
    return impl_->destroyBlock(worldPos);
}

bool ChunkManager::placeBlock(const glm::ivec3& targetBlockPos,
                              const glm::ivec3& faceNormal,
                              BlockId block)
{
    return impl_->placeBlock(targetBlockPos, faceNormal, block);
}

RaycastHit ChunkManager::raycast(const glm::vec3& origin, const glm::vec3& direction) const
{
    return impl_->raycast(origin, direction);
}

void ChunkManager::updateHighlight(const glm::vec3& cameraPos, const glm::vec3& cameraDirection)
{
    impl_->updateHighlight(cameraPos, cameraDirection);
}

void ChunkManager::toggleViewDistance()
{
    impl_->toggleViewDistance();
}

int ChunkManager::viewDistance() const noexcept
{
    return impl_->viewDistance();
}

int ChunkManager::exactRenderDistanceChunks() const noexcept
{
    return impl_->exactRenderDistanceChunks();
}

int ChunkManager::totalRenderDistanceChunks() const noexcept
{
    return impl_->totalRenderDistanceChunks();
}

int ChunkManager::nearRenderDistance() const noexcept
{
    return impl_->nearRenderDistance();
}

int ChunkManager::farRenderDistanceBlocks() const noexcept
{
    return impl_->farRenderDistanceBlocks();
}

RenderDistanceSettings ChunkManager::renderDistanceSettings() const noexcept
{
    return impl_->renderDistanceSettings();
}

void ChunkManager::setRenderDistance(int distance) noexcept
{
    impl_->setRenderDistance(distance);
}

void ChunkManager::setExactRenderDistanceChunks(int chunks) noexcept
{
    impl_->setExactRenderDistanceChunks(chunks);
}

void ChunkManager::setTotalRenderDistanceChunks(int chunks) noexcept
{
    impl_->setTotalRenderDistanceChunks(chunks);
}

void ChunkManager::setNearRenderDistance(int chunks) noexcept
{
    impl_->setNearRenderDistance(chunks);
}

void ChunkManager::setFarRenderDistanceBlocks(int blocks) noexcept
{
    impl_->setFarRenderDistanceBlocks(blocks);
}

void ChunkManager::setFogStartBlocks(int blocks) noexcept
{
    impl_->setFogStartBlocks(blocks);
}

void ChunkManager::setLodEnabled(bool enabled)
{
    impl_->setLodEnabled(enabled);
}

bool ChunkManager::lodEnabled() const noexcept
{
    return impl_->lodEnabled();
}

void ChunkManager::setFarTerrainEnabled(bool enabled)
{
    impl_->setFarTerrainEnabled(enabled);
}

bool ChunkManager::farTerrainEnabled() const noexcept
{
    return impl_->farTerrainEnabled();
}

BlockId ChunkManager::blockAt(const glm::ivec3& worldPos) const noexcept
{
    return impl_->blockAt(worldPos);
}

LightSample ChunkManager::lightAt(const glm::ivec3& worldPos) const noexcept
{
    return impl_->lightAt(worldPos);
}

glm::vec3 ChunkManager::findSafeSpawnPosition(float worldX, float worldZ) const
{
    return impl_->findSafeSpawnPosition(worldX, worldZ);
}

void ChunkManager::beginSpawnPreload(const glm::vec3& spawnPos)
{
    impl_->beginSpawnPreload(spawnPos);
}

bool ChunkManager::isSpawnPreloadReady() const noexcept
{
    return impl_->isSpawnPreloadReady();
}

bool ChunkManager::playerReleaseReady() const noexcept
{
    return impl_->playerReleaseReady();
}

StreamingPhase ChunkManager::streamingPhase() const noexcept
{
    return impl_->streamingPhase();
}

void ChunkManager::setStartupEnabled(bool enabled) noexcept
{
    impl_->setStartupEnabled(enabled);
}

bool ChunkManager::startupEnabled() const noexcept
{
    return impl_->startupEnabled();
}

StreamingStatusSnapshot ChunkManager::streamingStatusSnapshot() const noexcept
{
    return impl_->streamingStatusSnapshot();
}

LodDiagnosticsSnapshot ChunkManager::lodDiagnosticsSnapshot(const glm::vec3& cameraPos) const
{
    return impl_->lodDiagnosticsSnapshot(cameraPos);
}

RecentEditHoleDebugSnapshot ChunkManager::recentEditHoleDebugSnapshot(const glm::vec3& cameraPos) const
{
    return impl_->recentEditHoleDebugSnapshot(cameraPos);
}

void ChunkManager::writeLodDebugSnapshot(const std::filesystem::path& outputPath,
                                         const glm::vec3& cameraPos) const
{
    impl_->writeLodDebugSnapshot(outputPath, cameraPos);
}

ChunkProfilingSnapshot ChunkManager::sampleProfilingSnapshot()
{
    return impl_->sampleProfilingSnapshot();
}

void ChunkManager::setBenchmarkMetricsEnabled(bool enabled) noexcept
{
    impl_->setBenchmarkMetricsEnabled(enabled);
}

bool ChunkManager::benchmarkMetricsEnabled() const noexcept
{
    return impl_->benchmarkMetricsEnabled();
}

void ChunkManager::resetBenchmarkMetrics()
{
    impl_->resetBenchmarkMetrics();
}

ChunkBenchmarkReport ChunkManager::benchmarkReport() const
{
    return impl_->benchmarkReport();
}

std::string ChunkManager::biomeNameAt(const glm::vec3& worldPos) const
{
    return impl_->biomeNameAt(worldPos);
}
