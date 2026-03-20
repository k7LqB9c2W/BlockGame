// chunk_manager.cpp
// Implements the chunk streaming, terrain generation, and GPU upload subsystem.

#include "chunk_manager.h"

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
constexpr std::uint64_t kRelightBaseBudgetUnits = 1'500'000ull;
constexpr std::uint64_t kRelightPerWorkerBudgetUnits = 250'000ull;
constexpr std::uint64_t kRelightBacklogBudgetUnitsPerRegion = 32'768ull;
constexpr std::uint64_t kRelightMaxBudgetUnits = 4'000'000ull;
constexpr std::uint64_t kRelightMinBudgetUnits = 750'000ull;
constexpr int kRelightMinBatchBudget = 2;
constexpr int kRelightMaxBatchBudget = 8;
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
        structureQueryStage.reset();
        verticalRadiusDelta.reset();
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
        report.structureQueryStage = structureQueryStage.snapshot();
        report.verticalRadiusDelta = verticalRadiusDelta.snapshot();
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
    AtomicLatencyHistogram structureQueryStage{};
    AtomicCountHistogram verticalRadiusDelta{};
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

[[nodiscard]] float horizontalDistanceSqToAabb2D(float pointX,
                                                 float pointZ,
                                                 float minX,
                                                 float minZ,
                                                 float maxX,
                                                 float maxZ) noexcept
{
    const float dx = (pointX < minX) ? (minX - pointX) : ((pointX > maxX) ? (pointX - maxX) : 0.0f);
    const float dz = (pointZ < minZ) ? (minZ - pointZ) : ((pointZ > maxZ) ? (pointZ - maxZ) : 0.0f);
    return dx * dx + dz * dz;
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

class UploadContext
{
public:
    struct FlushTimings
    {
        double waitMs{0.0};
        double submitMs{0.0};
    };

    ~UploadContext()
    {
        shutdown();
    }

    void initialize(ID3D12Device* device)
    {
        shutdown();
        if (device == nullptr)
        {
            return;
        }

        device_ = device;

        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        throwIfFailedDx(device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue_)),
                        "failed to create upload command queue");
        throwIfFailedDx(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&allocator_)),
                        "failed to create upload command allocator");
        throwIfFailedDx(device_->CreateCommandList(0,
                                                   D3D12_COMMAND_LIST_TYPE_COPY,
                                                   allocator_.Get(),
                                                   nullptr,
                                                   IID_PPV_ARGS(&commandList_)),
                        "failed to create upload command list");
        throwIfFailedDx(commandList_->Close(), "failed to close initial upload command list");
        throwIfFailedDx(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
                        "failed to create upload fence");
        fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (fenceEvent_ == nullptr)
        {
            throw std::runtime_error("failed to create upload fence event");
        }
    }

    void shutdown()
    {
        graphicsFence_ = nullptr;
        graphicsFenceValue_ = 0;
        waitForIdle();
        if (fenceEvent_ != nullptr)
        {
            CloseHandle(fenceEvent_);
            fenceEvent_ = nullptr;
        }
        commandList_.Reset();
        allocator_.Reset();
        queue_.Reset();
        fence_.Reset();
        device_.Reset();
        graphicsFence_ = nullptr;
        graphicsFenceValue_ = 0;
        lastSubmittedFenceValue_ = 0;
        open_ = false;
        hasCommands_ = false;
    }

    void setGraphicsFenceDependency(ID3D12Fence* graphicsFence, UINT64 graphicsFenceValue) noexcept
    {
        graphicsFence_ = graphicsFence;
        graphicsFenceValue_ = graphicsFenceValue;
    }

    [[nodiscard]] bool begin()
    {
        if (device_ == nullptr)
        {
            return false;
        }
        if (open_)
        {
            return true;
        }
        if (fence_ != nullptr && fenceValue_ > 0 && fence_->GetCompletedValue() < fenceValue_)
        {
            return false;
        }

        throwIfFailedDx(allocator_->Reset(), "failed to reset upload command allocator");
        throwIfFailedDx(commandList_->Reset(allocator_.Get(), nullptr), "failed to reset upload command list");
        open_ = true;
        hasCommands_ = false;
        return true;
    }

    void transition(ID3D12Resource* resource,
                    D3D12_RESOURCE_STATES before,
                    D3D12_RESOURCE_STATES after)
    {
        if (!open_ || resource == nullptr || before == after)
        {
            return;
        }

        const D3D12_RESOURCE_BARRIER barrier = transitionBarrier(resource, before, after);
        commandList_->ResourceBarrier(1, &barrier);
        hasCommands_ = true;
    }

    void copyBuffer(ID3D12Resource* destination,
                    std::uint64_t destinationOffset,
                    ID3D12Resource* source,
                    std::uint64_t sourceOffset,
                    std::uint64_t sizeInBytes)
    {
        if (!open_ || destination == nullptr || source == nullptr || sizeInBytes == 0)
        {
            return;
        }

        commandList_->CopyBufferRegion(destination, destinationOffset, source, sourceOffset, sizeInBytes);
        hasCommands_ = true;
    }

    void waitForFence(ID3D12Fence* fence, UINT64 fenceValue)
    {
        if (!open_ || queue_ == nullptr || fence == nullptr || fenceValue == 0)
        {
            return;
        }

        throwIfFailedDx(queue_->Wait(fence, fenceValue), "failed to wait for external fence before upload");
        hasCommands_ = true;
    }

    void flush(FlushTimings* timings = nullptr)
    {
        if (!open_)
        {
            return;
        }

        if (timings != nullptr)
        {
            *timings = FlushTimings{};
        }

        throwIfFailedDx(commandList_->Close(), "failed to close upload command list");
        if (hasCommands_)
        {
            // Chunk buffers are shared with the render path. Do not transition/copy them
            // until the last submitted graphics work that referenced them has completed.
            if (graphicsFence_ != nullptr && graphicsFenceValue_ > 0)
            {
                const SteadyClock::time_point waitStart = SteadyClock::now();
                throwIfFailedDx(queue_->Wait(graphicsFence_, graphicsFenceValue_),
                                "failed to wait for graphics fence before upload");
                if (timings != nullptr)
                {
                    timings->waitMs =
                        std::chrono::duration<double, std::milli>(SteadyClock::now() - waitStart).count();
                }
            }

            const SteadyClock::time_point submitStart = SteadyClock::now();
            ID3D12CommandList* lists[] = {commandList_.Get()};
            queue_->ExecuteCommandLists(static_cast<UINT>(std::size(lists)), lists);

            ++fenceValue_;
            throwIfFailedDx(queue_->Signal(fence_.Get(), fenceValue_), "failed to signal upload fence");
            lastSubmittedFenceValue_ = fenceValue_;
            if (timings != nullptr)
            {
                timings->submitMs =
                    std::chrono::duration<double, std::milli>(SteadyClock::now() - submitStart).count();
            }
        }

        open_ = false;
        hasCommands_ = false;
    }

    [[nodiscard]] bool ready() const noexcept
    {
        return device_ != nullptr && queue_ != nullptr && allocator_ != nullptr && commandList_ != nullptr;
    }

    [[nodiscard]] ID3D12Fence* fence() const noexcept
    {
        return fence_.Get();
    }

    [[nodiscard]] UINT64 lastSubmittedFenceValue() const noexcept
    {
        return lastSubmittedFenceValue_;
    }

    [[nodiscard]] UINT64 completedFenceValue() const noexcept
    {
        return (fence_ != nullptr) ? fence_->GetCompletedValue() : 0;
    }

    void waitForIdle()
    {
        if (fence_ == nullptr || lastSubmittedFenceValue_ == 0)
        {
            return;
        }
        if (fence_->GetCompletedValue() >= lastSubmittedFenceValue_)
        {
            return;
        }
        throwIfFailedDx(fence_->SetEventOnCompletion(lastSubmittedFenceValue_, fenceEvent_),
                        "failed to wait for upload fence");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }

private:
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_{nullptr};
    UINT64 fenceValue_{0};
    UINT64 lastSubmittedFenceValue_{0};
    ID3D12Fence* graphicsFence_{nullptr};
    UINT64 graphicsFenceValue_{0};
    bool open_{false};
    bool hasCommands_{false};
};

class FarLodGpuContext
{
public:
    struct ScratchAllocation
    {
        ID3D12Resource* resource{nullptr};
        std::byte* cpuPtr{nullptr};
        D3D12_GPU_VIRTUAL_ADDRESS gpuAddress{0};
        std::uint64_t offset{0};
        std::uint64_t size{0};
    };

    struct FlushResult
    {
        UINT64 fenceValue{0};
    };

    ~FarLodGpuContext()
    {
        shutdown();
    }

    void initialize(ID3D12Device* device)
    {
        shutdown();
        if (device == nullptr)
        {
            return;
        }

        device_ = device;

        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        throwIfFailedDx(device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue_)),
                        "failed to create far lod compute queue");
        throwIfFailedDx(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator_)),
                        "failed to create far lod compute allocator");
        throwIfFailedDx(device_->CreateCommandList(0,
                                                   D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                   allocator_.Get(),
                                                   nullptr,
                                                   IID_PPV_ARGS(&commandList_)),
                        "failed to create far lod compute command list");
        throwIfFailedDx(commandList_->Close(), "failed to close initial far lod compute command list");
        throwIfFailedDx(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
                        "failed to create far lod compute fence");
        setDebugObjectName(queue_.Get(), L"FarLodComputeQueue");
        setDebugObjectName(allocator_.Get(), L"FarLodComputeAllocator");
        setDebugObjectName(commandList_.Get(), L"FarLodComputeCommandList");
        setDebugObjectName(fence_.Get(), L"FarLodComputeFence");
        fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (fenceEvent_ == nullptr)
        {
            throw std::runtime_error("failed to create far lod compute fence event");
        }

        uploadScratch_ = createUploadBuffer(device_.Get(), kUploadScratchSizeBytes, uploadScratchMapped_);
        setDebugObjectName(uploadScratch_.Get(), L"FarLodComputeUploadScratch");
        compileShaders();
        createDescriptorHeap();
        createPipelines();
        chunkManagerDebugLog("Far LOD GPU context initialized");
    }

    void shutdown()
    {
        waitForIdle();
        if (readbackScratch_ != nullptr)
        {
            readbackScratch_->Unmap(0, nullptr);
        }
        if (uploadScratch_ != nullptr)
        {
            uploadScratch_->Unmap(0, nullptr);
        }
        if (fenceEvent_ != nullptr)
        {
            CloseHandle(fenceEvent_);
            fenceEvent_ = nullptr;
        }

        atlasSeedCachePipelineState_.Reset();
        atlasSampleCachePipelineState_.Reset();
        atlasUpdatePipelineState_.Reset();
        synthColumnPipelineState_.Reset();
        stampPipelineState_.Reset();
        faceCountPipelineState_.Reset();
        facePrefixGroupPipelineState_.Reset();
        facePrefixScanPipelineState_.Reset();
        facePrefixAddPipelineState_.Reset();
        faceEmitPipelineState_.Reset();
        atlasSeedRootSignature_.Reset();
        atlasSampleRootSignature_.Reset();
        atlasFinalizeRootSignature_.Reset();
        synthColumnRootSignature_.Reset();
        stampRootSignature_.Reset();
        faceCountRootSignature_.Reset();
        facePrefixGroupRootSignature_.Reset();
        facePrefixScanRootSignature_.Reset();
        facePrefixAddRootSignature_.Reset();
        faceEmitRootSignature_.Reset();
        descriptorHeap_.Reset();
        atlasSeedCacheShader_.Reset();
        atlasSampleCacheShader_.Reset();
        atlasUpdateShader_.Reset();
        synthColumnShader_.Reset();
        stampShader_.Reset();
        faceCountShader_.Reset();
        facePrefixGroupShader_.Reset();
        facePrefixScanShader_.Reset();
        facePrefixAddShader_.Reset();
        faceEmitShader_.Reset();
        readbackScratch_.Reset();
        uploadScratch_.Reset();
        commandList_.Reset();
        allocator_.Reset();
        queue_.Reset();
        fence_.Reset();
        device_.Reset();
        uploadScratchMapped_ = nullptr;
        readbackScratchMapped_ = nullptr;
        open_ = false;
        hasCommands_ = false;
        readbackEnabled_ = false;
        uploadCursor_ = 0;
        readbackCursor_ = 0;
        lastSubmittedFenceValue_ = 0;
        fenceValue_ = 0;
    }

    void setReadbackEnabled(bool enabled)
    {
        if (device_ == nullptr)
        {
            readbackEnabled_ = enabled;
            return;
        }
        if (enabled == readbackEnabled_)
        {
            return;
        }
        waitForIdle();
        if (enabled)
        {
            readbackScratch_ = createReadbackBuffer(device_.Get(), kReadbackScratchSizeBytes, readbackScratchMapped_);
            setDebugObjectName(readbackScratch_.Get(), L"FarLodComputeReadbackScratch");
        }
        else
        {
            if (readbackScratch_ != nullptr)
            {
                readbackScratch_->Unmap(0, nullptr);
            }
            readbackScratch_.Reset();
            readbackScratchMapped_ = nullptr;
        }
        readbackEnabled_ = enabled;
        readbackCursor_ = 0;
    }

    [[nodiscard]] bool begin()
    {
        if (device_ == nullptr)
        {
            return false;
        }
        if (open_)
        {
            return true;
        }
        if (fence_ != nullptr && fenceValue_ > 0 && fence_->GetCompletedValue() < fenceValue_)
        {
            return false;
        }

        throwIfFailedDx(allocator_->Reset(), "failed to reset far lod compute allocator");
        throwIfFailedDx(commandList_->Reset(allocator_.Get(), nullptr), "failed to reset far lod compute command list");
        open_ = true;
        hasCommands_ = false;
        uploadCursor_ = 0;
        readbackCursor_ = 0;
        descriptorCursor_ = 0;
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD compute begin nextFence=" << (fenceValue_ + 1)
                   << " completedFence=" << ((fence_ != nullptr) ? fence_->GetCompletedValue() : 0);
            if (fence_ != nullptr && fence_->GetCompletedValue() == std::numeric_limits<std::uint64_t>::max())
            {
                const std::string dredMessages = collectDeviceDredMessages(device_.Get());
                if (!dredMessages.empty())
                {
                    stream << dredMessages;
                }
            }
            chunkManagerDebugLog(stream.str());
        }
        return true;
    }

    [[nodiscard]] ScratchAllocation allocateUpload(std::uint64_t sizeInBytes,
                                                   std::uint64_t alignment = 16u)
    {
        ScratchAllocation allocation{};
        if (!open_ || uploadScratch_ == nullptr || sizeInBytes == 0)
        {
            return allocation;
        }

        const std::uint64_t alignedOffset = (uploadCursor_ + alignment - 1u) / alignment * alignment;
        if (alignedOffset + sizeInBytes > kUploadScratchSizeBytes)
        {
            return allocation;
        }

        allocation.resource = uploadScratch_.Get();
        allocation.cpuPtr = uploadScratchMapped_ + alignedOffset;
        allocation.gpuAddress = uploadScratch_->GetGPUVirtualAddress() + alignedOffset;
        allocation.offset = alignedOffset;
        allocation.size = sizeInBytes;
        uploadCursor_ = alignedOffset + sizeInBytes;
        return allocation;
    }

    [[nodiscard]] ScratchAllocation allocateReadback(std::uint64_t sizeInBytes,
                                                     std::uint64_t alignment = 16u)
    {
        ScratchAllocation allocation{};
        if (!open_ || readbackScratch_ == nullptr || sizeInBytes == 0)
        {
            return allocation;
        }

        const std::uint64_t alignedOffset = (readbackCursor_ + alignment - 1u) / alignment * alignment;
        if (alignedOffset + sizeInBytes > kReadbackScratchSizeBytes)
        {
            return allocation;
        }

        allocation.resource = readbackScratch_.Get();
        allocation.cpuPtr = readbackScratchMapped_ + alignedOffset;
        allocation.gpuAddress = readbackScratch_->GetGPUVirtualAddress() + alignedOffset;
        allocation.offset = alignedOffset;
        allocation.size = sizeInBytes;
        readbackCursor_ = alignedOffset + sizeInBytes;
        return allocation;
    }

    void transition(ID3D12Resource* resource,
                    D3D12_RESOURCE_STATES before,
                    D3D12_RESOURCE_STATES after)
    {
        if (!open_ || resource == nullptr || before == after)
        {
            return;
        }

        const D3D12_RESOURCE_BARRIER barrier = transitionBarrier(resource, before, after);
        commandList_->ResourceBarrier(1, &barrier);
        hasCommands_ = true;
    }

    void uavBarrier(ID3D12Resource* resource)
    {
        if (!open_ || resource == nullptr)
        {
            return;
        }

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = resource;
        commandList_->ResourceBarrier(1, &barrier);
        hasCommands_ = true;
    }

    void copyBuffer(ID3D12Resource* destination,
                    std::uint64_t destinationOffset,
                    ID3D12Resource* source,
                    std::uint64_t sourceOffset,
                    std::uint64_t sizeInBytes)
    {
        if (!open_ || destination == nullptr || source == nullptr || sizeInBytes == 0)
        {
            return;
        }
        commandList_->CopyBufferRegion(destination, destinationOffset, source, sourceOffset, sizeInBytes);
        hasCommands_ = true;
    }

    void setAtlasBindings(ID3D12RootSignature* rootSignature,
                          ID3D12PipelineState* pipelineState,
                          const std::array<std::uint32_t, 14>& constants,
                          UINT srvDescriptorIndex,
                          UINT uavDescriptorIndex)
    {
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(pipelineState);
        commandList_->SetComputeRootSignature(rootSignature);
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<UINT64>(srvDescriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, srvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        uavHandle.ptr += static_cast<UINT64>(uavDescriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(2, uavHandle);
    }

    void bindAtlasSeedDescriptors(UINT srvDescriptorIndex,
                                  UINT uavDescriptorIndex,
                                  ID3D12Resource* worldgenHeaderBuffer,
                                  ID3D12Resource* worldgenBiomeBuffer,
                                  std::uint32_t biomeCount,
                                  ID3D12Resource* biomeSelectionBuffer,
                                  std::uint32_t biomeSelectionCount,
                                  ID3D12Resource* oceanSelectionBuffer,
                                  std::uint32_t oceanSelectionCount,
                                  ID3D12Resource* subBiomeBuffer,
                                  std::uint32_t subBiomeCount,
                                  ID3D12Resource* seedHeaderUavBuffer,
                                  std::uint32_t seedHeaderElementCount,
                                  ID3D12Resource* seedDataUavBuffer,
                                  std::uint32_t seedDataElementCount)
    {
        writeStructuredSrvDescriptor(srvDescriptorIndex,
                                     worldgenHeaderBuffer,
                                     0,
                                     1,
                                     static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuWorldgenHeader)));
        writeStructuredSrvDescriptor(srvDescriptorIndex + 1u,
                                     worldgenBiomeBuffer,
                                     0,
                                     biomeCount,
                                     static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuBiome)));
        writeStructuredSrvDescriptor(srvDescriptorIndex + 2u,
                                     biomeSelectionBuffer,
                                     0,
                                     biomeSelectionCount,
                                     kGpuContextBiomeSelectionStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 3u,
                                     oceanSelectionBuffer,
                                     0,
                                     oceanSelectionCount,
                                     kGpuContextBiomeSelectionStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 4u,
                                     subBiomeBuffer,
                                     0,
                                     subBiomeCount,
                                     kGpuContextSubBiomeStrideBytes);
        writeStructuredUavDescriptor(uavDescriptorIndex,
                                     seedHeaderUavBuffer,
                                     0,
                                     seedHeaderElementCount,
                                     kGpuContextChunkSeedCacheHeaderStrideBytes);
        writeStructuredUavDescriptor(uavDescriptorIndex + 1u,
                                     seedDataUavBuffer,
                                     0,
                                     seedDataElementCount,
                                     kGpuContextChunkSeedStrideBytes);
    }

    void bindAtlasSampleDescriptors(UINT srvDescriptorIndex,
                                    UINT uavDescriptorIndex,
                                    ID3D12Resource* worldgenHeaderBuffer,
                                    ID3D12Resource* worldgenBiomeBuffer,
                                    std::uint32_t biomeCount,
                                    ID3D12Resource* biomeSelectionBuffer,
                                    std::uint32_t biomeSelectionCount,
                                    ID3D12Resource* oceanSelectionBuffer,
                                    std::uint32_t oceanSelectionCount,
                                    ID3D12Resource* subBiomeBuffer,
                                    std::uint32_t subBiomeCount,
                                    ID3D12Resource* permutationBuffer,
                                    std::uint32_t permutationCount,
                                    ID3D12Resource* seedHeaderSrvBuffer,
                                    std::uint32_t seedHeaderElementCount,
                                    ID3D12Resource* seedDataSrvBuffer,
                                    std::uint32_t seedDataElementCount,
                                    ID3D12Resource* sampleUavBuffer,
                                    std::uint32_t sampleUavElementCount)
    {
        const auto logDescriptorWrite = [&](const char* kind,
                                            UINT descriptorIndexValue,
                                            const char* label,
                                            ID3D12Resource* resource,
                                            std::uint32_t elementCount,
                                            std::uint32_t strideBytes)
        {
            if (!chunkManagerDebugLoggingEnabled())
            {
                return;
            }

            std::ostringstream stream;
            stream << "Far LOD sample descriptor write"
                   << " kind=" << kind
                   << " descriptor=" << descriptorIndexValue
                   << " label=" << label
                   << " resource=" << hexPtr(resource)
                   << " elements=" << elementCount
                   << " stride=" << strideBytes;
            chunkManagerDebugLog(stream.str());
        };

        logDescriptorWrite("srv",
                           srvDescriptorIndex,
                           "worldgen_header",
                           worldgenHeaderBuffer,
                           1u,
                           static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuWorldgenHeader)));
        writeStructuredSrvDescriptor(srvDescriptorIndex,
                                     worldgenHeaderBuffer,
                                     0,
                                     1,
                                     static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuWorldgenHeader)));
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 1u,
                           "biomes",
                           worldgenBiomeBuffer,
                           biomeCount,
                           static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuBiome)));
        writeStructuredSrvDescriptor(srvDescriptorIndex + 1u,
                                     worldgenBiomeBuffer,
                                     0,
                                     biomeCount,
                                     static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuBiome)));
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 2u,
                           "biome_selections",
                           biomeSelectionBuffer,
                           biomeSelectionCount,
                           kGpuContextBiomeSelectionStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 2u,
                                     biomeSelectionBuffer,
                                     0,
                                     biomeSelectionCount,
                                     kGpuContextBiomeSelectionStrideBytes);
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 3u,
                           "ocean_selections",
                           oceanSelectionBuffer,
                           oceanSelectionCount,
                           kGpuContextBiomeSelectionStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 3u,
                                     oceanSelectionBuffer,
                                     0,
                                     oceanSelectionCount,
                                     kGpuContextBiomeSelectionStrideBytes);
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 4u,
                           "sub_biomes",
                           subBiomeBuffer,
                           subBiomeCount,
                           kGpuContextSubBiomeStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 4u,
                                     subBiomeBuffer,
                                     0,
                                     subBiomeCount,
                                     kGpuContextSubBiomeStrideBytes);
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 5u,
                           "surface_permutation",
                           permutationBuffer,
                           permutationCount,
                           kGpuContextPermutationStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 5u,
                                     permutationBuffer,
                                     0,
                                     permutationCount,
                                     kGpuContextPermutationStrideBytes);
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 6u,
                           "seed_headers",
                           seedHeaderSrvBuffer,
                           seedHeaderElementCount,
                           kGpuContextChunkSeedCacheHeaderStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 6u,
                                     seedHeaderSrvBuffer,
                                     0,
                                     seedHeaderElementCount,
                                     kGpuContextChunkSeedCacheHeaderStrideBytes);
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 7u,
                           "seed_data",
                           seedDataSrvBuffer,
                           seedDataElementCount,
                           kGpuContextChunkSeedStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 7u,
                                     seedDataSrvBuffer,
                                     0,
                                     seedDataElementCount,
                                     kGpuContextChunkSeedStrideBytes);
        logDescriptorWrite("uav",
                           uavDescriptorIndex,
                           "sample_cache_out",
                           sampleUavBuffer,
                           sampleUavElementCount,
                           kGpuContextAtlasSamplePointCacheStrideBytes);
        writeStructuredUavDescriptor(uavDescriptorIndex,
                                     sampleUavBuffer,
                                     0,
                                     sampleUavElementCount,
                                     kGpuContextAtlasSamplePointCacheStrideBytes);
    }

    void bindAtlasFinalizeDescriptors(UINT srvDescriptorIndex,
                                      UINT uavDescriptorIndex,
                                      ID3D12Resource* worldgenHeaderBuffer,
                                      ID3D12Resource* worldgenBiomeBuffer,
                                      std::uint32_t biomeCount,
                                      ID3D12Resource* permutationBuffer,
                                      std::uint32_t permutationCount,
                                      ID3D12Resource* sampleSrvBuffer,
                                      std::uint32_t sampleSrvElementCount,
                                      ID3D12Resource* atlasUavBuffer,
                                      std::uint32_t atlasUavElementCount)
    {
        writeStructuredSrvDescriptor(srvDescriptorIndex,
                                     worldgenHeaderBuffer,
                                     0,
                                     1,
                                     static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuWorldgenHeader)));
        writeStructuredSrvDescriptor(srvDescriptorIndex + 1u,
                                     worldgenBiomeBuffer,
                                     0,
                                     biomeCount,
                                     static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuBiome)));
        writeStructuredSrvDescriptor(srvDescriptorIndex + 2u,
                                     permutationBuffer,
                                     0,
                                     permutationCount,
                                     kGpuContextPermutationStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 3u,
                                     sampleSrvBuffer,
                                     0,
                                     sampleSrvElementCount,
                                     kGpuContextAtlasSamplePointCacheStrideBytes);
        writeStructuredUavDescriptor(uavDescriptorIndex,
                                     atlasUavBuffer,
                                     0,
                                     atlasUavElementCount,
                                     kGpuContextAtlasSampleStrideBytes);
    }

    void dispatchSeedCacheUpdate(const glm::ivec2& updateOriginChunk,
                                 const glm::ivec2& updateSizeChunks,
                                 const glm::ivec2& seedOriginChunk,
                                 const glm::ivec2& seedSizeChunks,
                                 ID3D12Resource* worldgenHeaderBuffer,
                                 ID3D12Resource* worldgenBiomeBuffer,
                                 std::uint32_t biomeCount,
                                 ID3D12Resource* biomeSelectionBuffer,
                                 std::uint32_t biomeSelectionCount,
                                 ID3D12Resource* oceanSelectionBuffer,
                                 std::uint32_t oceanSelectionCount,
                                 ID3D12Resource* subBiomeBuffer,
                                 std::uint32_t subBiomeCount,
                                 ID3D12Resource* seedHeaderBuffer,
                                 std::uint32_t seedHeaderElementCount,
                                 ID3D12Resource* seedDataBuffer,
                                 std::uint32_t seedDataElementCount)
    {
        if (!open_ || worldgenHeaderBuffer == nullptr || worldgenBiomeBuffer == nullptr ||
            biomeSelectionBuffer == nullptr || oceanSelectionBuffer == nullptr ||
            subBiomeBuffer == nullptr || seedHeaderBuffer == nullptr || seedDataBuffer == nullptr ||
            updateSizeChunks.x <= 0 || updateSizeChunks.y <= 0)
        {
            return;
        }
        if (seedHeaderElementCount == 0 || seedDataElementCount == 0)
        {
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD seed dispatch skipped due to empty seed buffers"
                       << " seedHeader=" << seedHeaderElementCount
                       << " seedData=" << seedDataElementCount;
                chunkManagerDebugLog(stream.str());
            }
            return;
        }
        if (seedSizeChunks.x <= 0 || seedSizeChunks.y <= 0)
        {
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD seed dispatch skipped due to invalid seed cache dimensions"
                       << " seedOrigin=[" << seedOriginChunk.x << "," << seedOriginChunk.y << "]"
                       << " seedSize=[" << seedSizeChunks.x << "," << seedSizeChunks.y << "]";
                chunkManagerDebugLog(stream.str());
            }
            return;
        }
        const std::uint64_t expectedSeedHeaderElementCount =
            static_cast<std::uint64_t>(seedSizeChunks.x) * static_cast<std::uint64_t>(seedSizeChunks.y);
        const std::uint64_t expectedSeedDataElementCount =
            expectedSeedHeaderElementCount * static_cast<std::uint64_t>(kGpuContextChunkSeedCountPerCacheEntry);
        if (static_cast<std::uint64_t>(seedHeaderElementCount) < expectedSeedHeaderElementCount ||
            static_cast<std::uint64_t>(seedDataElementCount) < expectedSeedDataElementCount)
        {
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD seed dispatch skipped due to undersized seed buffers"
                       << " expectedHeader=" << expectedSeedHeaderElementCount
                       << " expectedData=" << expectedSeedDataElementCount
                       << " actualHeader=" << seedHeaderElementCount
                       << " actualData=" << seedDataElementCount;
                chunkManagerDebugLog(stream.str());
            }
            return;
        }

        std::array<std::uint32_t, 14> constants{
            0u, 0u, 0u, 0u,
            static_cast<std::uint32_t>(updateOriginChunk.x),
            static_cast<std::uint32_t>(updateOriginChunk.y),
            static_cast<std::uint32_t>(updateSizeChunks.x),
            static_cast<std::uint32_t>(updateSizeChunks.y),
            0u,
            0u,
            static_cast<std::uint32_t>(seedOriginChunk.x),
            static_cast<std::uint32_t>(seedOriginChunk.y),
            static_cast<std::uint32_t>(seedSizeChunks.x),
            static_cast<std::uint32_t>(seedSizeChunks.y)};
        const UINT srvDescriptorIndex = allocateDescriptorRange(5);
        const UINT uavDescriptorIndex = allocateDescriptorRange(2);
        bindAtlasSeedDescriptors(srvDescriptorIndex,
                                 uavDescriptorIndex,
                                 worldgenHeaderBuffer,
                                 worldgenBiomeBuffer,
                                 biomeCount,
                                 biomeSelectionBuffer,
                                 biomeSelectionCount,
                                 oceanSelectionBuffer,
                                 oceanSelectionCount,
                                 subBiomeBuffer,
                                 subBiomeCount,
                                 seedHeaderBuffer,
                                 seedHeaderElementCount,
                                 seedDataBuffer,
                                 seedDataElementCount);
        setAtlasBindings(atlasSeedRootSignature_.Get(),
                         atlasSeedCachePipelineState_.Get(),
                         constants,
                         srvDescriptorIndex,
                         uavDescriptorIndex);
        commandList_->Dispatch(static_cast<UINT>((updateSizeChunks.x + 7) / 8),
                               static_cast<UINT>((updateSizeChunks.y + 7) / 8),
                               1);
        hasCommands_ = true;
    }

    void dispatchAtlasSampleCacheUpdate(const glm::ivec2& atlasOriginCell,
                                        const glm::ivec2& atlasSizeCells,
                                        const glm::ivec2& updateOriginCell,
                                        const glm::ivec2& updateSizeCells,
                                        int blockScale,
                                        const glm::ivec2& seedOriginChunk,
                                        const glm::ivec2& seedSizeChunks,
                                        ID3D12Resource* worldgenHeaderBuffer,
                                        ID3D12Resource* worldgenBiomeBuffer,
                                        std::uint32_t biomeCount,
                                        ID3D12Resource* biomeSelectionBuffer,
                                        std::uint32_t biomeSelectionCount,
                                        ID3D12Resource* oceanSelectionBuffer,
                                        std::uint32_t oceanSelectionCount,
                                        ID3D12Resource* subBiomeBuffer,
                                        std::uint32_t subBiomeCount,
                                        ID3D12Resource* permutationBuffer,
                                        std::uint32_t permutationCount,
                                        ID3D12Resource* seedHeaderBuffer,
                                        std::uint32_t seedHeaderElementCount,
                                        ID3D12Resource* seedDataBuffer,
                                        std::uint32_t seedDataElementCount,
                                        ID3D12Resource* sampleBuffer,
                                        std::uint32_t sampleElementCount)
    {
        static std::uint64_t sampleDispatchSequence = 0u;
        const std::uint64_t dispatchId = ++sampleDispatchSequence;
        const bool logEnabled = chunkManagerDebugLoggingEnabled();
        if (!open_ || worldgenHeaderBuffer == nullptr || worldgenBiomeBuffer == nullptr ||
            biomeSelectionBuffer == nullptr || oceanSelectionBuffer == nullptr || subBiomeBuffer == nullptr ||
            permutationBuffer == nullptr || seedHeaderBuffer == nullptr || seedDataBuffer == nullptr ||
            sampleBuffer == nullptr || updateSizeCells.x <= 0 || updateSizeCells.y <= 0)
        {
            return;
        }
        if (seedSizeChunks.x <= 0 || seedSizeChunks.y <= 0)
        {
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD sample dispatch skipped due to invalid seed cache dimensions"
                       << " seedOrigin=[" << seedOriginChunk.x << "," << seedOriginChunk.y << "]"
                       << " seedSize=[" << seedSizeChunks.x << "," << seedSizeChunks.y << "]";
                chunkManagerDebugLog(stream.str());
            }
            return;
        }
        const std::uint64_t expectedSeedHeaderElementCount =
            static_cast<std::uint64_t>(seedSizeChunks.x) * static_cast<std::uint64_t>(seedSizeChunks.y);
        const std::uint64_t expectedSeedDataElementCount =
            expectedSeedHeaderElementCount * static_cast<std::uint64_t>(kGpuContextChunkSeedCountPerCacheEntry);
        if (static_cast<std::uint64_t>(seedHeaderElementCount) < expectedSeedHeaderElementCount ||
            static_cast<std::uint64_t>(seedDataElementCount) < expectedSeedDataElementCount)
        {
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD sample dispatch skipped due to undersized seed buffers"
                       << " expectedHeader=" << expectedSeedHeaderElementCount
                       << " expectedData=" << expectedSeedDataElementCount
                       << " actualHeader=" << seedHeaderElementCount
                       << " actualData=" << seedDataElementCount;
                chunkManagerDebugLog(stream.str());
            }
            return;
        }

        if (logEnabled)
        {
            std::ostringstream stream;
            stream << "Far LOD sample dispatch begin"
                   << " id=" << dispatchId
                   << " atlasOrigin=[" << atlasOriginCell.x << "," << atlasOriginCell.y << "]"
                   << " atlasSize=[" << atlasSizeCells.x << "," << atlasSizeCells.y << "]"
                   << " updateOrigin=[" << updateOriginCell.x << "," << updateOriginCell.y << "]"
                   << " updateSize=[" << updateSizeCells.x << "," << updateSizeCells.y << "]"
                   << " blockScale=" << blockScale
                   << " seedOrigin=[" << seedOriginChunk.x << "," << seedOriginChunk.y << "]"
                   << " seedSize=[" << seedSizeChunks.x << "," << seedSizeChunks.y << "]"
                   << " biomeCount=" << biomeCount
                   << " biomeSelCount=" << biomeSelectionCount
                   << " oceanSelCount=" << oceanSelectionCount
                   << " subBiomeCount=" << subBiomeCount
                   << " permutationCount=" << permutationCount
                   << " seedHeaderCount=" << seedHeaderElementCount
                   << " seedDataCount=" << seedDataElementCount
                   << " sampleCount=" << sampleElementCount
                   << " descriptorCursor=" << descriptorCursor_
                   << " resources=[header=" << hexPtr(worldgenHeaderBuffer)
                   << ",biomes=" << hexPtr(worldgenBiomeBuffer)
                   << ",biomeSel=" << hexPtr(biomeSelectionBuffer)
                   << ",oceanSel=" << hexPtr(oceanSelectionBuffer)
                   << ",subBiome=" << hexPtr(subBiomeBuffer)
                   << ",perm=" << hexPtr(permutationBuffer)
                   << ",seedHeader=" << hexPtr(seedHeaderBuffer)
                   << ",seedData=" << hexPtr(seedDataBuffer)
                   << ",sample=" << hexPtr(sampleBuffer) << "]";
            chunkManagerDebugLog(stream.str());
        }

        std::array<std::uint32_t, 14> constants{
            static_cast<std::uint32_t>(atlasOriginCell.x),
            static_cast<std::uint32_t>(atlasOriginCell.y),
            static_cast<std::uint32_t>(atlasSizeCells.x),
            static_cast<std::uint32_t>(atlasSizeCells.y),
            static_cast<std::uint32_t>(updateOriginCell.x),
            static_cast<std::uint32_t>(updateOriginCell.y),
            static_cast<std::uint32_t>(updateSizeCells.x),
            static_cast<std::uint32_t>(updateSizeCells.y),
            static_cast<std::uint32_t>(blockScale),
            0u,
            static_cast<std::uint32_t>(seedOriginChunk.x),
            static_cast<std::uint32_t>(seedOriginChunk.y),
            static_cast<std::uint32_t>(seedSizeChunks.x),
            static_cast<std::uint32_t>(seedSizeChunks.y)};
        const UINT srvDescriptorIndex = allocateDescriptorRange(8);
        const UINT uavDescriptorIndex = allocateDescriptorRange(1);
        if (logEnabled)
        {
            std::ostringstream stream;
            stream << "Far LOD sample dispatch descriptors"
                   << " id=" << dispatchId
                   << " srvBase=" << srvDescriptorIndex
                   << " uavBase=" << uavDescriptorIndex
                   << " descriptorCursor=" << descriptorCursor_;
            chunkManagerDebugLog(stream.str());
        }

        if (logEnabled)
        {
            chunkManagerDebugLog("Far LOD sample dispatch bind begin id=" + std::to_string(dispatchId));
        }
        bindAtlasSampleDescriptors(srvDescriptorIndex,
                                   uavDescriptorIndex,
                                   worldgenHeaderBuffer,
                                   worldgenBiomeBuffer,
                                   biomeCount,
                                   biomeSelectionBuffer,
                                   biomeSelectionCount,
                                   oceanSelectionBuffer,
                                   oceanSelectionCount,
                                   subBiomeBuffer,
                                   subBiomeCount,
                                   permutationBuffer,
                                   permutationCount,
                                   seedHeaderBuffer,
                                   seedHeaderElementCount,
                                   seedDataBuffer,
                                   seedDataElementCount,
                                   sampleBuffer,
                                   sampleElementCount);
        if (logEnabled)
        {
            chunkManagerDebugLog("Far LOD sample dispatch bind end id=" + std::to_string(dispatchId));
            chunkManagerDebugLog("Far LOD sample dispatch set_bindings begin id=" + std::to_string(dispatchId));
        }
        setAtlasBindings(atlasSampleRootSignature_.Get(),
                         atlasSampleCachePipelineState_.Get(),
                         constants,
                         srvDescriptorIndex,
                         uavDescriptorIndex);
        if (logEnabled)
        {
            chunkManagerDebugLog("Far LOD sample dispatch set_bindings end id=" + std::to_string(dispatchId));
            chunkManagerDebugLog("Far LOD sample dispatch dispatch begin id=" + std::to_string(dispatchId));
        }
        commandList_->Dispatch(static_cast<UINT>((updateSizeCells.x + 7) / 8),
                               static_cast<UINT>((updateSizeCells.y + 7) / 8),
                               1);
        if (logEnabled)
        {
            chunkManagerDebugLog("Far LOD sample dispatch dispatch end id=" + std::to_string(dispatchId));
        }
        hasCommands_ = true;
        if (logEnabled)
        {
            chunkManagerDebugLog("Far LOD sample dispatch complete id=" + std::to_string(dispatchId));
        }
    }

    void dispatchAtlasUpdate(const glm::ivec2& atlasOriginCell,
                             const glm::ivec2& atlasSizeCells,
                             const glm::ivec2& updateOriginCell,
                             const glm::ivec2& updateSizeCells,
                             int blockScale,
                             int seaLevel,
                             ID3D12Resource* worldgenHeaderBuffer,
                             ID3D12Resource* worldgenBiomeBuffer,
                             std::uint32_t biomeCount,
                             ID3D12Resource* permutationBuffer,
                             std::uint32_t permutationCount,
                             ID3D12Resource* sampleBuffer,
                             std::uint32_t sampleElementCount,
                             ID3D12Resource* atlasBuffer,
                             std::uint32_t atlasElementCount)
    {
        if (!open_ || worldgenHeaderBuffer == nullptr || worldgenBiomeBuffer == nullptr ||
            permutationBuffer == nullptr || sampleBuffer == nullptr ||
            atlasBuffer == nullptr || updateSizeCells.x <= 0 || updateSizeCells.y <= 0)
        {
            return;
        }

        std::array<std::uint32_t, 14> constants{
            static_cast<std::uint32_t>(atlasOriginCell.x),
            static_cast<std::uint32_t>(atlasOriginCell.y),
            static_cast<std::uint32_t>(atlasSizeCells.x),
            static_cast<std::uint32_t>(atlasSizeCells.y),
            static_cast<std::uint32_t>(updateOriginCell.x),
            static_cast<std::uint32_t>(updateOriginCell.y),
            static_cast<std::uint32_t>(updateSizeCells.x),
            static_cast<std::uint32_t>(updateSizeCells.y),
            static_cast<std::uint32_t>(blockScale),
            static_cast<std::uint32_t>(seaLevel),
            0u,
            0u,
            0u,
            0u};
        const UINT srvDescriptorIndex = allocateDescriptorRange(4);
        const UINT uavDescriptorIndex = allocateDescriptorRange(1);
        bindAtlasFinalizeDescriptors(srvDescriptorIndex,
                                     uavDescriptorIndex,
                                     worldgenHeaderBuffer,
                                     worldgenBiomeBuffer,
                                     biomeCount,
                                     permutationBuffer,
                                     permutationCount,
                                     sampleBuffer,
                                     sampleElementCount,
                                     atlasBuffer,
                                     atlasElementCount);
        setAtlasBindings(atlasFinalizeRootSignature_.Get(),
                         atlasUpdatePipelineState_.Get(),
                         constants,
                         srvDescriptorIndex,
                         uavDescriptorIndex);
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD GPU dispatch atlas-finalize originCell=[" << updateOriginCell.x << "," << updateOriginCell.y
                   << "] size=[" << updateSizeCells.x << "," << updateSizeCells.y << "]"
                   << " atlasOrigin=[" << atlasOriginCell.x << "," << atlasOriginCell.y << "]"
                   << " atlasSize=[" << atlasSizeCells.x << "," << atlasSizeCells.y << "]"
                   << " blockScale=" << blockScale
                   << " descriptorBase=" << srvDescriptorIndex;
            chunkManagerDebugLog(stream.str());
        }
        commandList_->Dispatch(static_cast<UINT>((updateSizeCells.x + 7) / 8),
                               static_cast<UINT>((updateSizeCells.y + 7) / 8),
                               1);
        hasCommands_ = true;
    }

    void dispatchSynth(const glm::ivec3& worldMin,
                       int blockScale,
                       int seaLevel,
                       const glm::ivec2& atlasOriginCell,
                       const glm::ivec2& atlasSizeCells,
                       ID3D12Resource* atlasBuffer,
                       std::uint32_t atlasElementCount,
                       ID3D12Resource* columnBuffer)
    {
        if (!open_ || atlasBuffer == nullptr || columnBuffer == nullptr)
        {
            return;
        }

        std::array<std::uint32_t, 9> constants{
            static_cast<std::uint32_t>(worldMin.x),
            static_cast<std::uint32_t>(worldMin.y),
            static_cast<std::uint32_t>(worldMin.z),
            static_cast<std::uint32_t>(blockScale),
            static_cast<std::uint32_t>(seaLevel),
            static_cast<std::uint32_t>(atlasOriginCell.x),
            static_cast<std::uint32_t>(atlasOriginCell.y),
            static_cast<std::uint32_t>(atlasSizeCells.x),
            static_cast<std::uint32_t>(atlasSizeCells.y)};
        const UINT descriptorIndex = allocateDescriptorRange(2);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     atlasBuffer,
                                     0,
                                     atlasElementCount,
                                     kGpuContextAtlasSampleStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 1u,
                                     columnBuffer,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(synthColumnPipelineState_.Get());
        commandList_->SetComputeRootSignature(synthColumnRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, srvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE columnUavHandle = srvHandle;
        columnUavHandle.ptr += descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(2, columnUavHandle);
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD GPU dispatch synth-columns worldMin=[" << worldMin.x << "," << worldMin.y << "," << worldMin.z
                   << "] blockScale=" << blockScale
                   << " atlasSize=[" << atlasSizeCells.x << "," << atlasSizeCells.y << "]"
                   << " descriptorBase=" << descriptorIndex;
            chunkManagerDebugLog(stream.str());
        }
        commandList_->Dispatch(4, 4, 1);
        hasCommands_ = true;
    }

    void dispatchStamp(const glm::ivec3& worldMin,
                       int blockScale,
                       int lodLevel,
                       std::uint32_t structureCount,
                       ID3D12Resource* structureBuffer,
                       ID3D12Resource* voxelBuffer)
    {
        if (!open_ || structureCount == 0 || structureBuffer == nullptr || voxelBuffer == nullptr)
        {
            return;
        }

        std::array<std::uint32_t, 6> constants{
            static_cast<std::uint32_t>(worldMin.x),
            static_cast<std::uint32_t>(worldMin.y),
            static_cast<std::uint32_t>(worldMin.z),
            static_cast<std::uint32_t>(blockScale),
            static_cast<std::uint32_t>(lodLevel),
            structureCount};
        const UINT descriptorIndex = allocateDescriptorRange(2);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     structureBuffer,
                                     0,
                                     structureCount,
                                     kGpuContextStructureInstanceStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 1u,
                                     voxelBuffer,
                                     0,
                                     kGpuContextVoxelCount,
                                     kGpuContextPackedVoxelStrideBytes);
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(stampPipelineState_.Get());
        commandList_->SetComputeRootSignature(stampRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, srvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = srvHandle;
        uavHandle.ptr += descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(2, uavHandle);
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD GPU dispatch structure-stamp structureCount=" << structureCount
                   << " worldMin=[" << worldMin.x << "," << worldMin.y << "," << worldMin.z << "]"
                   << " blockScale=" << blockScale
                   << " lodLevel=" << lodLevel
                   << " descriptorBase=" << descriptorIndex;
            chunkManagerDebugLog(stream.str());
        }
        commandList_->Dispatch((kGpuContextVoxelCount + 63u) / 64u, 1, 1);
        hasCommands_ = true;
    }

    void dispatchFaceCount(int worldMinY,
                           int blockScale,
                           std::uint32_t reservedFlags,
                           std::uint32_t maxMergeExtent,
                           ID3D12Resource* voxelBuffer,
                           ID3D12Resource* neighborPosX,
                           ID3D12Resource* neighborNegX,
                           ID3D12Resource* neighborPosZ,
                           ID3D12Resource* neighborNegZ,
                           ID3D12Resource* faceCountBuffer,
                           ID3D12Resource* faceAnalysisBuffer,
                           ID3D12Resource* faceDescriptorBuffer)
    {
        if (!open_ || voxelBuffer == nullptr || neighborPosX == nullptr ||
            neighborNegX == nullptr || neighborPosZ == nullptr || neighborNegZ == nullptr ||
            faceCountBuffer == nullptr || faceAnalysisBuffer == nullptr || faceDescriptorBuffer == nullptr)
        {
            return;
        }

        std::array<std::uint32_t, 4> constants{
            static_cast<std::uint32_t>(worldMinY),
            static_cast<std::uint32_t>(blockScale),
            reservedFlags,
            maxMergeExtent};
        const UINT descriptorIndex = allocateDescriptorRange(8);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     voxelBuffer,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 1u,
                                     neighborPosX,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 2u,
                                     neighborNegX,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 3u,
                                     neighborPosZ,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 4u,
                                     neighborNegZ,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 5u,
                                     faceCountBuffer,
                                     0,
                                     kGpuContextPlaneCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 6u,
                                     faceAnalysisBuffer,
                                     0,
                                     kGpuContextFaceMetadataEntryCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 7u,
                                     faceDescriptorBuffer,
                                     0,
                                     kGpuContextFaceDescriptorCount,
                                     kGpuContextFaceDescriptorStrideBytes);
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(faceCountPipelineState_.Get());
        commandList_->SetComputeRootSignature(faceCountRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, srvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = srvHandle;
        uavHandle.ptr += descriptorSize_ * 5u;
        commandList_->SetComputeRootDescriptorTable(2, uavHandle);
        commandList_->Dispatch(kGpuContextPlaneDispatchGroupCount, 1, 1);
        hasCommands_ = true;
    }

    void dispatchFacePrefix(ID3D12Resource* faceCountBuffer,
                            ID3D12Resource* facePrefixBuffer,
                            ID3D12Resource* faceGroupSumBuffer)
    {
        if (!open_ || faceCountBuffer == nullptr || facePrefixBuffer == nullptr || faceGroupSumBuffer == nullptr)
        {
            return;
        }

        UINT descriptorIndex = allocateDescriptorRange(3);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     faceCountBuffer,
                                     0,
                                     kGpuContextPlaneCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 1u,
                                     facePrefixBuffer,
                                     0,
                                     kGpuContextPlaneCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 2u,
                                     faceGroupSumBuffer,
                                     0,
                                     kGpuContextFacePrefixGroupCount,
                                     kGpuContextPackedVoxelStrideBytes);
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(facePrefixGroupPipelineState_.Get());
        commandList_->SetComputeRootSignature(facePrefixGroupRootSignature_.Get());
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(0, srvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = srvHandle;
        uavHandle.ptr += descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, uavHandle);
        commandList_->Dispatch(kGpuContextFacePrefixGroupCount, 1, 1);
        uavBarrier(facePrefixBuffer);
        uavBarrier(faceGroupSumBuffer);

        descriptorIndex = allocateDescriptorRange(1);
        writeStructuredUavDescriptor(descriptorIndex,
                                     faceGroupSumBuffer,
                                     0,
                                     kGpuContextFacePrefixGroupCount,
                                     kGpuContextPackedVoxelStrideBytes);
        D3D12_GPU_DESCRIPTOR_HANDLE scanUavHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        scanUavHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetPipelineState(facePrefixScanPipelineState_.Get());
        commandList_->SetComputeRootSignature(facePrefixScanRootSignature_.Get());
        commandList_->SetComputeRootDescriptorTable(0, scanUavHandle);
        commandList_->Dispatch(1, 1, 1);
        uavBarrier(faceGroupSumBuffer);

        descriptorIndex = allocateDescriptorRange(2);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     faceGroupSumBuffer,
                                     0,
                                     kGpuContextFacePrefixGroupCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 1u,
                                     facePrefixBuffer,
                                     0,
                                     kGpuContextPlaneCount,
                                     kGpuContextPackedVoxelStrideBytes);
        D3D12_GPU_DESCRIPTOR_HANDLE addSrvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        addSrvHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetPipelineState(facePrefixAddPipelineState_.Get());
        commandList_->SetComputeRootSignature(facePrefixAddRootSignature_.Get());
        commandList_->SetComputeRootDescriptorTable(0, addSrvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE addUavHandle = addSrvHandle;
        addUavHandle.ptr += descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, addUavHandle);
        commandList_->Dispatch(kGpuContextFacePrefixGroupCount, 1, 1);
        uavBarrier(facePrefixBuffer);
        hasCommands_ = true;
    }

    void dispatchFaceEmit(const glm::ivec3& worldMin,
                          int blockScale,
                          std::uint32_t maxMergeExtent,
                          std::uint32_t vertexBase,
                          std::uint32_t indexBase,
                          std::uint32_t recordIndex,
                          std::uint32_t reservedFaceCapacity,
                          ID3D12Resource* voxelBuffer,
                          ID3D12Resource* faceCountBuffer,
                          ID3D12Resource* faceAnalysisBuffer,
                          ID3D12Resource* faceDescriptorBuffer,
                          ID3D12Resource* facePrefixBuffer,
                          ID3D12Resource* blockUvBuffer,
                          std::uint32_t blockUvCount,
                          std::uint32_t blockUvStrideBytes,
                          ID3D12Resource* vertexBuffer,
                          std::uint32_t vertexBufferCount,
                          ID3D12Resource* indexBuffer,
                          std::uint32_t indexBufferCount,
                          ID3D12Resource* drawRecordBuffer,
                          std::uint32_t drawRecordCount)
    {
        if (!open_ || voxelBuffer == nullptr || faceCountBuffer == nullptr ||
            faceAnalysisBuffer == nullptr || faceDescriptorBuffer == nullptr || facePrefixBuffer == nullptr ||
            blockUvBuffer == nullptr || vertexBuffer == nullptr || indexBuffer == nullptr || drawRecordBuffer == nullptr)
        {
            return;
        }

        std::array<std::uint32_t, 9> constants{
            static_cast<std::uint32_t>(worldMin.x),
            static_cast<std::uint32_t>(worldMin.y),
            static_cast<std::uint32_t>(worldMin.z),
            static_cast<std::uint32_t>(blockScale),
            maxMergeExtent,
            vertexBase,
            indexBase,
            recordIndex,
            reservedFaceCapacity};
        const UINT descriptorIndex = allocateDescriptorRange(9);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     voxelBuffer,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 1u,
                                     faceCountBuffer,
                                     0,
                                     kGpuContextPlaneCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 2u,
                                     faceAnalysisBuffer,
                                     0,
                                     kGpuContextFaceMetadataEntryCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 3u,
                                     faceDescriptorBuffer,
                                     0,
                                     kGpuContextFaceDescriptorCount,
                                     kGpuContextFaceDescriptorStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 4u,
                                     facePrefixBuffer,
                                     0,
                                     kGpuContextPlaneCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 5u,
                                     blockUvBuffer,
                                     0,
                                     blockUvCount,
                                     blockUvStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 6u,
                                     vertexBuffer,
                                     0,
                                     vertexBufferCount,
                                     static_cast<std::uint32_t>(sizeof(Vertex)));
        writeStructuredUavDescriptor(descriptorIndex + 7u,
                                     indexBuffer,
                                     0,
                                     indexBufferCount,
                                     static_cast<std::uint32_t>(sizeof(std::uint32_t)));
        writeStructuredUavDescriptor(descriptorIndex + 8u,
                                     drawRecordBuffer,
                                     0,
                                     drawRecordCount,
                                     static_cast<std::uint32_t>(sizeof(ChunkRenderBatch::GpuCullRecord)));
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(faceEmitPipelineState_.Get());
        commandList_->SetComputeRootSignature(faceEmitRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, srvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = srvHandle;
        uavHandle.ptr += descriptorSize_ * 6u;
        commandList_->SetComputeRootDescriptorTable(2, uavHandle);
        commandList_->Dispatch(kGpuContextPlaneDispatchGroupCount, 1, 1);
        hasCommands_ = true;
    }

    [[nodiscard]] FlushResult flush()
    {
        FlushResult result{};
        if (!open_)
        {
            return result;
        }

        throwIfFailedDx(commandList_->Close(), "failed to close far lod compute command list");
        if (hasCommands_)
        {
            ID3D12CommandList* lists[] = {commandList_.Get()};
            queue_->ExecuteCommandLists(static_cast<UINT>(std::size(lists)), lists);
            ++fenceValue_;
            throwIfFailedDx(queue_->Signal(fence_.Get(), fenceValue_), "failed to signal far lod compute fence");
            lastSubmittedFenceValue_ = fenceValue_;
            result.fenceValue = fenceValue_;
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD GPU flush submitted fence=" << fenceValue_;
                chunkManagerDebugLog(stream.str());
            }
        }
        else if (chunkManagerDebugLoggingEnabled())
        {
            chunkManagerDebugLog("Far LOD GPU flush skipped (no commands)");
        }

        open_ = false;
        hasCommands_ = false;
        return result;
    }

    void waitForIdle()
    {
        if (fence_ == nullptr || lastSubmittedFenceValue_ == 0)
        {
            return;
        }
        if (fence_->GetCompletedValue() >= lastSubmittedFenceValue_)
        {
            return;
        }
        throwIfFailedDx(fence_->SetEventOnCompletion(lastSubmittedFenceValue_, fenceEvent_),
                        "failed to wait for far lod compute fence");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }

    void waitForFence(UINT64 fenceValue)
    {
        if (fence_ == nullptr || fenceValue == 0)
        {
            return;
        }
        if (fence_->GetCompletedValue() >= fenceValue)
        {
            return;
        }
        throwIfFailedDx(fence_->SetEventOnCompletion(fenceValue, fenceEvent_),
                        "failed to wait for far lod compute fence value");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }

    [[nodiscard]] bool ready() const noexcept
    {
        return device_ != nullptr && queue_ != nullptr && allocator_ != nullptr && commandList_ != nullptr &&
               descriptorHeap_ != nullptr && descriptorSize_ > 0 &&
               atlasSeedRootSignature_ != nullptr && atlasSampleRootSignature_ != nullptr &&
               atlasFinalizeRootSignature_ != nullptr &&
               atlasSeedCachePipelineState_ != nullptr && atlasSampleCachePipelineState_ != nullptr &&
                atlasUpdatePipelineState_ != nullptr && synthColumnPipelineState_ != nullptr &&
                stampPipelineState_ != nullptr &&
               faceCountPipelineState_ != nullptr &&
               facePrefixGroupPipelineState_ != nullptr &&
               facePrefixScanPipelineState_ != nullptr &&
               facePrefixAddPipelineState_ != nullptr &&
               faceEmitPipelineState_ != nullptr;
    }

    [[nodiscard]] UINT64 completedFenceValue() const noexcept
    {
        return (fence_ != nullptr) ? fence_->GetCompletedValue() : 0;
    }

    [[nodiscard]] UINT64 lastSubmittedFenceValue() const noexcept
    {
        return lastSubmittedFenceValue_;
    }

    [[nodiscard]] ID3D12Fence* fence() const noexcept
    {
        return fence_.Get();
    }

    [[nodiscard]] const std::byte* readbackMappedData() const noexcept
    {
        return readbackScratchMapped_;
    }

private:
    static constexpr std::uint32_t kGpuContextLogicalSize = 16u;
    static constexpr std::uint32_t kGpuContextColumnCount = kGpuContextLogicalSize * kGpuContextLogicalSize;
    static constexpr std::uint32_t kGpuContextVoxelCount = kGpuContextLogicalSize * kGpuContextLogicalSize * kGpuContextLogicalSize;
    static constexpr std::uint32_t kGpuContextFacePrefixGroupSize = 256u;
    static constexpr std::uint32_t kGpuContextTopPlaneCount = 3u;
    static constexpr std::uint32_t kGpuContextSideSlicesPerLayer = 64u;
    static constexpr std::uint32_t kGpuContextPlaneCount =
        kGpuContextTopPlaneCount + 3u * kGpuContextSideSlicesPerLayer;
    static constexpr std::uint32_t kGpuContextPlaneDispatchGroupCount = (kGpuContextPlaneCount + 63u) / 64u;
    static constexpr std::uint32_t kGpuContextFacePrefixGroupCount =
        (kGpuContextPlaneCount + kGpuContextFacePrefixGroupSize - 1u) / kGpuContextFacePrefixGroupSize;
    static constexpr std::uint32_t kGpuContextFaceMetadataEntryCount = 1u;
    static constexpr std::uint32_t kGpuContextMaxTopDescriptorsPerPlane = kGpuContextColumnCount;
    static constexpr std::uint32_t kGpuContextMaxSideDescriptorsPerPlane = kGpuContextLogicalSize;
    static constexpr std::uint32_t kGpuContextFaceDescriptorCount =
        kGpuContextTopPlaneCount * kGpuContextMaxTopDescriptorsPerPlane +
        (kGpuContextPlaneCount - kGpuContextTopPlaneCount) * kGpuContextMaxSideDescriptorsPerPlane;
    static constexpr std::uint32_t kGpuContextColumnDescriptorStrideBytes = 48u;
    static constexpr std::uint32_t kGpuContextAtlasSampleStrideBytes = 48u;
    static constexpr std::uint32_t kGpuContextAtlasSamplePointCacheStrideBytes = 144u;
    static constexpr std::uint32_t kGpuContextBiomeSelectionStrideBytes = 16u;
    static constexpr std::uint32_t kGpuContextTransitionBiomeStrideBytes = 16u;
    static constexpr std::uint32_t kGpuContextSubBiomeStrideBytes = 16u;
    static constexpr std::uint32_t kGpuContextFaceDescriptorStrideBytes = 16u;
    static constexpr std::uint32_t kGpuContextChunkSeedCacheHeaderStrideBytes = 16u;
    static constexpr std::uint32_t kGpuContextChunkSeedCountPerCacheEntry = 64u;
    static constexpr std::uint32_t kGpuContextChunkSeedStrideBytes = 20u;
    static constexpr std::uint32_t kGpuContextPermutationStrideBytes = 4u;
    static constexpr std::uint32_t kGpuContextStructureInstanceStrideBytes = 64u;
    static constexpr std::uint32_t kGpuContextPackedVoxelStrideBytes = 4u;
    static constexpr UINT kDescriptorHeapDescriptorCount = 2048u;

    void createDescriptorHeap()
    {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = kDescriptorHeapDescriptorCount;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        throwIfFailedDx(device_->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&descriptorHeap_)),
                        "failed to create far lod compute descriptor heap");
        setDebugObjectName(descriptorHeap_.Get(), L"FarLodComputeDescriptorHeap");
        descriptorSize_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    [[nodiscard]] UINT allocateDescriptorRange(UINT descriptorCount)
    {
        if (descriptorCursor_ + descriptorCount > kDescriptorHeapDescriptorCount)
        {
            throw std::runtime_error("far lod compute descriptor heap exhausted");
        }
        const UINT baseIndex = descriptorCursor_;
        descriptorCursor_ += descriptorCount;
        return baseIndex;
    }

    void writeStructuredSrvDescriptor(std::uint32_t descriptorIndex,
                                      ID3D12Resource* resource,
                                      std::uint64_t byteOffset,
                                      std::uint32_t elementCount,
                                      std::uint32_t strideBytes)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.Buffer.FirstElement = byteOffset / strideBytes;
        srvDesc.Buffer.NumElements = std::max(elementCount, 1u);
        srvDesc.Buffer.StructureByteStride = strideBytes;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        D3D12_CPU_DESCRIPTOR_HANDLE handle = descriptorHeap_->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<SIZE_T>(descriptorIndex) * descriptorSize_;
        device_->CreateShaderResourceView(resource, &srvDesc, handle);
    }

    void writeStructuredUavDescriptor(std::uint32_t descriptorIndex,
                                      ID3D12Resource* resource,
                                      std::uint64_t byteOffset,
                                      std::uint32_t elementCount,
                                      std::uint32_t strideBytes)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.Buffer.FirstElement = byteOffset / strideBytes;
        uavDesc.Buffer.NumElements = std::max(elementCount, 1u);
        uavDesc.Buffer.StructureByteStride = strideBytes;
        uavDesc.Buffer.CounterOffsetInBytes = 0;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
        D3D12_CPU_DESCRIPTOR_HANDLE handle = descriptorHeap_->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<SIZE_T>(descriptorIndex) * descriptorSize_;
        device_->CreateUnorderedAccessView(resource, nullptr, &uavDesc, handle);
    }

    void compileShaders()
    {
        const std::filesystem::path shaderRoot = std::filesystem::current_path() / "assets" / "shaders";
        atlasSeedCacheShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_column_atlas_update_canonical_cs.hlsl").string(), "FarLodChunkSeedCacheMain", "cs_5_0");
        atlasSampleCacheShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_column_atlas_update_canonical_cs.hlsl").string(), "FarLodColumnSampleCacheMain", "cs_5_0");
        atlasUpdateShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_column_atlas_update_canonical_cs.hlsl").string(), "FarLodColumnAtlasUpdateMain", "cs_5_0");
        synthColumnShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_synth_cs.hlsl").string(), "FarLodChunkSynthMain", "cs_5_0");
        stampShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_structure_stamp_cs.hlsl").string(), "FarLodChunkStructureStampMain", "cs_5_0");
        faceCountShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_face_count_cs.hlsl").string(), "FarLodChunkFaceCountMain", "cs_5_0");
        facePrefixGroupShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_face_prefix_cs.hlsl").string(), "FarLodChunkFacePrefixGroupMain", "cs_5_0");
        facePrefixScanShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_face_prefix_cs.hlsl").string(), "FarLodChunkFacePrefixScanMain", "cs_5_0");
        facePrefixAddShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_face_prefix_cs.hlsl").string(), "FarLodChunkFacePrefixAddMain", "cs_5_0");
        faceEmitShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_face_emit_cs.hlsl").string(), "FarLodChunkFaceEmitMain", "cs_5_0");
    }

    void createPipelines()
    {
        auto createRootSignature = [this](const D3D12_ROOT_SIGNATURE_DESC& desc,
                                          Microsoft::WRL::ComPtr<ID3D12RootSignature>& rootSignature,
                                          const char* label)
        {
            Microsoft::WRL::ComPtr<ID3DBlob> serialized;
            Microsoft::WRL::ComPtr<ID3DBlob> rootErrors;
            const std::string serializeMessage = std::string("failed to serialize ") + label;
            const std::string createMessage = std::string("failed to create ") + label;
            throwIfFailedDx(D3D12SerializeRootSignature(&desc,
                                                        D3D_ROOT_SIGNATURE_VERSION_1,
                                                        &serialized,
                                                        &rootErrors),
                            serializeMessage.c_str());
            throwIfFailedDx(device_->CreateRootSignature(0,
                                                         serialized->GetBufferPointer(),
                                                         serialized->GetBufferSize(),
                                                         IID_PPV_ARGS(&rootSignature)),
                            createMessage.c_str());
        };

        D3D12_DESCRIPTOR_RANGE atlasSeedSrvRanges[2]{};
        atlasSeedSrvRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasSeedSrvRanges[0].NumDescriptors = 4;
        atlasSeedSrvRanges[0].BaseShaderRegister = 0;
        atlasSeedSrvRanges[0].OffsetInDescriptorsFromTableStart = 0;
        atlasSeedSrvRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasSeedSrvRanges[1].NumDescriptors = 1;
        atlasSeedSrvRanges[1].BaseShaderRegister = 5;
        atlasSeedSrvRanges[1].OffsetInDescriptorsFromTableStart = 4;
        D3D12_DESCRIPTOR_RANGE atlasSeedUavRange{};
        atlasSeedUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        atlasSeedUavRange.NumDescriptors = 2;
        atlasSeedUavRange.BaseShaderRegister = 0;
        atlasSeedUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> atlasSeedParams{};
        atlasSeedParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        atlasSeedParams[0].Constants.ShaderRegister = 0;
        atlasSeedParams[0].Constants.Num32BitValues = 14;
        atlasSeedParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        atlasSeedParams[1].DescriptorTable.NumDescriptorRanges = static_cast<UINT>(std::size(atlasSeedSrvRanges));
        atlasSeedParams[1].DescriptorTable.pDescriptorRanges = atlasSeedSrvRanges;
        atlasSeedParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        atlasSeedParams[2].DescriptorTable.NumDescriptorRanges = 1;
        atlasSeedParams[2].DescriptorTable.pDescriptorRanges = &atlasSeedUavRange;
        D3D12_ROOT_SIGNATURE_DESC atlasSeedDesc{};
        atlasSeedDesc.NumParameters = static_cast<UINT>(atlasSeedParams.size());
        atlasSeedDesc.pParameters = atlasSeedParams.data();
        createRootSignature(atlasSeedDesc, atlasSeedRootSignature_, "far lod atlas seed root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC atlasSeedPso{};
        atlasSeedPso.pRootSignature = atlasSeedRootSignature_.Get();
        atlasSeedPso.CS = {atlasSeedCacheShader_->GetBufferPointer(), atlasSeedCacheShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&atlasSeedPso, IID_PPV_ARGS(&atlasSeedCachePipelineState_)),
                        "failed to create far lod atlas seed cache pipeline");

        D3D12_DESCRIPTOR_RANGE atlasSampleSrvRanges[2]{};
        atlasSampleSrvRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasSampleSrvRanges[0].NumDescriptors = 4;
        atlasSampleSrvRanges[0].BaseShaderRegister = 0;
        atlasSampleSrvRanges[0].OffsetInDescriptorsFromTableStart = 0;
        atlasSampleSrvRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasSampleSrvRanges[1].NumDescriptors = 4;
        atlasSampleSrvRanges[1].BaseShaderRegister = 5;
        atlasSampleSrvRanges[1].OffsetInDescriptorsFromTableStart = 4;
        D3D12_DESCRIPTOR_RANGE atlasSampleUavRange{};
        atlasSampleUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        atlasSampleUavRange.NumDescriptors = 1;
        atlasSampleUavRange.BaseShaderRegister = 2;
        atlasSampleUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> atlasSampleParams{};
        atlasSampleParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        atlasSampleParams[0].Constants.ShaderRegister = 0;
        atlasSampleParams[0].Constants.Num32BitValues = 14;
        atlasSampleParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        atlasSampleParams[1].DescriptorTable.NumDescriptorRanges = static_cast<UINT>(std::size(atlasSampleSrvRanges));
        atlasSampleParams[1].DescriptorTable.pDescriptorRanges = atlasSampleSrvRanges;
        atlasSampleParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        atlasSampleParams[2].DescriptorTable.NumDescriptorRanges = 1;
        atlasSampleParams[2].DescriptorTable.pDescriptorRanges = &atlasSampleUavRange;
        D3D12_ROOT_SIGNATURE_DESC atlasSampleDesc{};
        atlasSampleDesc.NumParameters = static_cast<UINT>(atlasSampleParams.size());
        atlasSampleDesc.pParameters = atlasSampleParams.data();
        createRootSignature(atlasSampleDesc, atlasSampleRootSignature_, "far lod atlas sample root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC atlasSamplePso{};
        atlasSamplePso.pRootSignature = atlasSampleRootSignature_.Get();
        atlasSamplePso.CS = {atlasSampleCacheShader_->GetBufferPointer(), atlasSampleCacheShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&atlasSamplePso, IID_PPV_ARGS(&atlasSampleCachePipelineState_)),
                        "failed to create far lod atlas sample cache pipeline");

        D3D12_DESCRIPTOR_RANGE atlasFinalizeSrvRanges[3]{};
        atlasFinalizeSrvRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasFinalizeSrvRanges[0].NumDescriptors = 2;
        atlasFinalizeSrvRanges[0].BaseShaderRegister = 0;
        atlasFinalizeSrvRanges[0].OffsetInDescriptorsFromTableStart = 0;
        atlasFinalizeSrvRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasFinalizeSrvRanges[1].NumDescriptors = 1;
        atlasFinalizeSrvRanges[1].BaseShaderRegister = 6;
        atlasFinalizeSrvRanges[1].OffsetInDescriptorsFromTableStart = 2;
        atlasFinalizeSrvRanges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasFinalizeSrvRanges[2].NumDescriptors = 1;
        atlasFinalizeSrvRanges[2].BaseShaderRegister = 9;
        atlasFinalizeSrvRanges[2].OffsetInDescriptorsFromTableStart = 3;
        D3D12_DESCRIPTOR_RANGE atlasFinalizeUavRange{};
        atlasFinalizeUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        atlasFinalizeUavRange.NumDescriptors = 1;
        atlasFinalizeUavRange.BaseShaderRegister = 3;
        atlasFinalizeUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> atlasFinalizeParams{};
        atlasFinalizeParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        atlasFinalizeParams[0].Constants.ShaderRegister = 0;
        atlasFinalizeParams[0].Constants.Num32BitValues = 14;
        atlasFinalizeParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        atlasFinalizeParams[1].DescriptorTable.NumDescriptorRanges = static_cast<UINT>(std::size(atlasFinalizeSrvRanges));
        atlasFinalizeParams[1].DescriptorTable.pDescriptorRanges = atlasFinalizeSrvRanges;
        atlasFinalizeParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        atlasFinalizeParams[2].DescriptorTable.NumDescriptorRanges = 1;
        atlasFinalizeParams[2].DescriptorTable.pDescriptorRanges = &atlasFinalizeUavRange;
        D3D12_ROOT_SIGNATURE_DESC atlasFinalizeDesc{};
        atlasFinalizeDesc.NumParameters = static_cast<UINT>(atlasFinalizeParams.size());
        atlasFinalizeDesc.pParameters = atlasFinalizeParams.data();
        createRootSignature(atlasFinalizeDesc, atlasFinalizeRootSignature_, "far lod atlas finalize root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC atlasPso{};
        atlasPso.pRootSignature = atlasFinalizeRootSignature_.Get();
        atlasPso.CS = {atlasUpdateShader_->GetBufferPointer(), atlasUpdateShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&atlasPso, IID_PPV_ARGS(&atlasUpdatePipelineState_)),
                        "failed to create far lod atlas update pipeline");

        D3D12_DESCRIPTOR_RANGE synthSrvRange{};
        synthSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        synthSrvRange.NumDescriptors = 1;
        synthSrvRange.BaseShaderRegister = 0;
        synthSrvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE synthUavRange{};
        synthUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        synthUavRange.NumDescriptors = 1;
        synthUavRange.BaseShaderRegister = 0;
        synthUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> synthParams{};
        synthParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        synthParams[0].Constants.ShaderRegister = 0;
        synthParams[0].Constants.Num32BitValues = 9;
        synthParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        synthParams[1].DescriptorTable.NumDescriptorRanges = 1;
        synthParams[1].DescriptorTable.pDescriptorRanges = &synthSrvRange;
        synthParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        synthParams[2].DescriptorTable.NumDescriptorRanges = 1;
        synthParams[2].DescriptorTable.pDescriptorRanges = &synthUavRange;
        D3D12_ROOT_SIGNATURE_DESC synthDesc{};
        synthDesc.NumParameters = static_cast<UINT>(synthParams.size());
        synthDesc.pParameters = synthParams.data();
        createRootSignature(synthDesc, synthColumnRootSignature_, "far lod synth column root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC synthColumnPso{};
        synthColumnPso.pRootSignature = synthColumnRootSignature_.Get();
        synthColumnPso.CS = {synthColumnShader_->GetBufferPointer(), synthColumnShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&synthColumnPso, IID_PPV_ARGS(&synthColumnPipelineState_)),
                        "failed to create far lod synth column pipeline");

        D3D12_DESCRIPTOR_RANGE stampSrvRange{};
        stampSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        stampSrvRange.NumDescriptors = 1;
        stampSrvRange.BaseShaderRegister = 0;
        stampSrvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE stampUavRange{};
        stampUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        stampUavRange.NumDescriptors = 1;
        stampUavRange.BaseShaderRegister = 0;
        stampUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> stampParams{};
        stampParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        stampParams[0].Constants.ShaderRegister = 0;
        stampParams[0].Constants.Num32BitValues = 6;
        stampParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        stampParams[1].DescriptorTable.NumDescriptorRanges = 1;
        stampParams[1].DescriptorTable.pDescriptorRanges = &stampSrvRange;
        stampParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        stampParams[2].DescriptorTable.NumDescriptorRanges = 1;
        stampParams[2].DescriptorTable.pDescriptorRanges = &stampUavRange;
        D3D12_ROOT_SIGNATURE_DESC stampDesc{};
        stampDesc.NumParameters = static_cast<UINT>(stampParams.size());
        stampDesc.pParameters = stampParams.data();
        createRootSignature(stampDesc, stampRootSignature_, "far lod stamp root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC stampPso{};
        stampPso.pRootSignature = stampRootSignature_.Get();
        stampPso.CS = {stampShader_->GetBufferPointer(), stampShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&stampPso, IID_PPV_ARGS(&stampPipelineState_)),
                        "failed to create far lod stamp pipeline");

        D3D12_DESCRIPTOR_RANGE faceCountSrvRange{};
        faceCountSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        faceCountSrvRange.NumDescriptors = 5;
        faceCountSrvRange.BaseShaderRegister = 0;
        faceCountSrvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE faceCountUavRange{};
        faceCountUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        faceCountUavRange.NumDescriptors = 3;
        faceCountUavRange.BaseShaderRegister = 0;
        faceCountUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> faceCountParams{};
        faceCountParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        faceCountParams[0].Constants.ShaderRegister = 0;
        faceCountParams[0].Constants.Num32BitValues = 4;
        faceCountParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        faceCountParams[1].DescriptorTable.NumDescriptorRanges = 1;
        faceCountParams[1].DescriptorTable.pDescriptorRanges = &faceCountSrvRange;
        faceCountParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        faceCountParams[2].DescriptorTable.NumDescriptorRanges = 1;
        faceCountParams[2].DescriptorTable.pDescriptorRanges = &faceCountUavRange;
        D3D12_ROOT_SIGNATURE_DESC faceCountDesc{};
        faceCountDesc.NumParameters = static_cast<UINT>(faceCountParams.size());
        faceCountDesc.pParameters = faceCountParams.data();
        createRootSignature(faceCountDesc, faceCountRootSignature_, "far lod face count root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC faceCountPso{};
        faceCountPso.pRootSignature = faceCountRootSignature_.Get();
        faceCountPso.CS = {faceCountShader_->GetBufferPointer(), faceCountShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&faceCountPso, IID_PPV_ARGS(&faceCountPipelineState_)),
                        "failed to create far lod face count pipeline");

        D3D12_DESCRIPTOR_RANGE prefixSrvRange{};
        prefixSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        prefixSrvRange.NumDescriptors = 1;
        prefixSrvRange.BaseShaderRegister = 0;
        prefixSrvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE prefixUavRange{};
        prefixUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        prefixUavRange.NumDescriptors = 2;
        prefixUavRange.BaseShaderRegister = 0;
        prefixUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 2> prefixParams{};
        prefixParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        prefixParams[0].DescriptorTable.NumDescriptorRanges = 1;
        prefixParams[0].DescriptorTable.pDescriptorRanges = &prefixSrvRange;
        prefixParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        prefixParams[1].DescriptorTable.NumDescriptorRanges = 1;
        prefixParams[1].DescriptorTable.pDescriptorRanges = &prefixUavRange;
        D3D12_ROOT_SIGNATURE_DESC prefixDesc{};
        prefixDesc.NumParameters = static_cast<UINT>(prefixParams.size());
        prefixDesc.pParameters = prefixParams.data();
        createRootSignature(prefixDesc, facePrefixGroupRootSignature_, "far lod face prefix group root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC prefixGroupPso{};
        prefixGroupPso.pRootSignature = facePrefixGroupRootSignature_.Get();
        prefixGroupPso.CS = {facePrefixGroupShader_->GetBufferPointer(), facePrefixGroupShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&prefixGroupPso, IID_PPV_ARGS(&facePrefixGroupPipelineState_)),
                        "failed to create far lod face prefix group pipeline");

        D3D12_DESCRIPTOR_RANGE prefixScanUavRange{};
        prefixScanUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        prefixScanUavRange.NumDescriptors = 1;
        prefixScanUavRange.BaseShaderRegister = 0;
        prefixScanUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 1> prefixScanParams{};
        prefixScanParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        prefixScanParams[0].DescriptorTable.NumDescriptorRanges = 1;
        prefixScanParams[0].DescriptorTable.pDescriptorRanges = &prefixScanUavRange;
        D3D12_ROOT_SIGNATURE_DESC prefixScanDesc{};
        prefixScanDesc.NumParameters = static_cast<UINT>(prefixScanParams.size());
        prefixScanDesc.pParameters = prefixScanParams.data();
        createRootSignature(prefixScanDesc, facePrefixScanRootSignature_, "far lod face prefix scan root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC prefixScanPso{};
        prefixScanPso.pRootSignature = facePrefixScanRootSignature_.Get();
        prefixScanPso.CS = {facePrefixScanShader_->GetBufferPointer(), facePrefixScanShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&prefixScanPso, IID_PPV_ARGS(&facePrefixScanPipelineState_)),
                        "failed to create far lod face prefix scan pipeline");

        D3D12_DESCRIPTOR_RANGE prefixAddSrvRange{};
        prefixAddSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        prefixAddSrvRange.NumDescriptors = 1;
        prefixAddSrvRange.BaseShaderRegister = 0;
        prefixAddSrvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE prefixAddUavRange{};
        prefixAddUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        prefixAddUavRange.NumDescriptors = 1;
        prefixAddUavRange.BaseShaderRegister = 0;
        prefixAddUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 2> prefixAddParams{};
        prefixAddParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        prefixAddParams[0].DescriptorTable.NumDescriptorRanges = 1;
        prefixAddParams[0].DescriptorTable.pDescriptorRanges = &prefixAddSrvRange;
        prefixAddParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        prefixAddParams[1].DescriptorTable.NumDescriptorRanges = 1;
        prefixAddParams[1].DescriptorTable.pDescriptorRanges = &prefixAddUavRange;
        D3D12_ROOT_SIGNATURE_DESC prefixAddDesc{};
        prefixAddDesc.NumParameters = static_cast<UINT>(prefixAddParams.size());
        prefixAddDesc.pParameters = prefixAddParams.data();
        createRootSignature(prefixAddDesc, facePrefixAddRootSignature_, "far lod face prefix add root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC prefixAddPso{};
        prefixAddPso.pRootSignature = facePrefixAddRootSignature_.Get();
        prefixAddPso.CS = {facePrefixAddShader_->GetBufferPointer(), facePrefixAddShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&prefixAddPso, IID_PPV_ARGS(&facePrefixAddPipelineState_)),
                        "failed to create far lod face prefix add pipeline");

        D3D12_DESCRIPTOR_RANGE faceEmitSrvRange{};
        faceEmitSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        faceEmitSrvRange.NumDescriptors = 6;
        faceEmitSrvRange.BaseShaderRegister = 0;
        faceEmitSrvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE faceEmitUavRange{};
        faceEmitUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        faceEmitUavRange.NumDescriptors = 3;
        faceEmitUavRange.BaseShaderRegister = 0;
        faceEmitUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> faceEmitParams{};
        faceEmitParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        faceEmitParams[0].Constants.ShaderRegister = 0;
        faceEmitParams[0].Constants.Num32BitValues = 9;
        faceEmitParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        faceEmitParams[1].DescriptorTable.NumDescriptorRanges = 1;
        faceEmitParams[1].DescriptorTable.pDescriptorRanges = &faceEmitSrvRange;
        faceEmitParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        faceEmitParams[2].DescriptorTable.NumDescriptorRanges = 1;
        faceEmitParams[2].DescriptorTable.pDescriptorRanges = &faceEmitUavRange;
        D3D12_ROOT_SIGNATURE_DESC faceEmitDesc{};
        faceEmitDesc.NumParameters = static_cast<UINT>(faceEmitParams.size());
        faceEmitDesc.pParameters = faceEmitParams.data();
        createRootSignature(faceEmitDesc, faceEmitRootSignature_, "far lod face emit root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC faceEmitPso{};
        faceEmitPso.pRootSignature = faceEmitRootSignature_.Get();
        faceEmitPso.CS = {faceEmitShader_->GetBufferPointer(), faceEmitShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&faceEmitPso, IID_PPV_ARGS(&faceEmitPipelineState_)),
                        "failed to create far lod face emit pipeline");
    }

    static constexpr std::uint64_t kUploadScratchSizeBytes = 16ull * 1024ull * 1024ull;
    static constexpr std::uint64_t kReadbackScratchSizeBytes = 4ull * 1024ull * 1024ull;

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_{nullptr};
    UINT64 fenceValue_{0};
    UINT64 lastSubmittedFenceValue_{0};
    Microsoft::WRL::ComPtr<ID3DBlob> atlasSeedCacheShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> atlasSampleCacheShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> synthColumnShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> stampShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> atlasUpdateShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> faceCountShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> facePrefixGroupShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> facePrefixScanShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> facePrefixAddShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> faceEmitShader_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> atlasSeedRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> atlasSampleRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> atlasFinalizeRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> synthColumnRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> stampRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> faceCountRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> facePrefixGroupRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> facePrefixScanRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> facePrefixAddRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> faceEmitRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> atlasSeedCachePipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> atlasSampleCachePipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> atlasUpdatePipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> synthColumnPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> stampPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> faceCountPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> facePrefixGroupPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> facePrefixScanPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> facePrefixAddPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> faceEmitPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptorHeap_;
    Microsoft::WRL::ComPtr<ID3D12Resource> uploadScratch_;
    std::byte* uploadScratchMapped_{nullptr};
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackScratch_;
    std::byte* readbackScratchMapped_{nullptr};
    std::uint64_t uploadCursor_{0};
    std::uint64_t readbackCursor_{0};
    UINT descriptorSize_{0};
    UINT descriptorCursor_{0};
    bool open_{false};
    bool hasCommands_{false};
    bool readbackEnabled_{false};
};

inline int floorDiv(int value, int divisor) noexcept
{
    int quotient = value / divisor;
    int remainder = value % divisor;
    if ((remainder != 0) && ((remainder < 0) != (divisor < 0)))
    {
        --quotient;
    }
    return quotient;
}

inline int wrapIndex(int value, int modulus) noexcept
{
    int result = value % modulus;
    if (result < 0)
    {
        result += modulus;
    }
    return result;
}

inline glm::ivec3 localBlockCoords(const glm::ivec3& worldPos, const glm::ivec3& chunkCoord) noexcept
{
    return {
        worldPos.x - chunkCoord.x * kChunkSizeX,
        worldPos.y - chunkCoord.y * kChunkSizeY,
        worldPos.z - chunkCoord.z * kChunkSizeZ
    };
}

enum class BlockFace : std::uint8_t
{
    Top = 0,
    Bottom,
    North,
    South,
    East,
    West,
    Count
};

constexpr std::size_t toIndex(BlockFace face) noexcept
{
    return static_cast<std::size_t>(face);
}

constexpr std::size_t kBlockFaceCount = toIndex(BlockFace::Count);
constexpr std::uint8_t kMaxLightLevel = 15;

inline glm::ivec3 faceOffset(BlockFace face) noexcept
{
    switch (face)
    {
    case BlockFace::Top:
        return glm::ivec3(0, 1, 0);
    case BlockFace::Bottom:
        return glm::ivec3(0, -1, 0);
    case BlockFace::North:
        return glm::ivec3(0, 0, -1);
    case BlockFace::South:
        return glm::ivec3(0, 0, 1);
    case BlockFace::East:
        return glm::ivec3(1, 0, 0);
    case BlockFace::West:
    default:
        return glm::ivec3(-1, 0, 0);
    }
}

inline BlockFace oppositeFace(BlockFace face) noexcept
{
    switch (face)
    {
    case BlockFace::Top:
        return BlockFace::Bottom;
    case BlockFace::Bottom:
        return BlockFace::Top;
    case BlockFace::North:
        return BlockFace::South;
    case BlockFace::South:
        return BlockFace::North;
    case BlockFace::East:
        return BlockFace::West;
    case BlockFace::West:
    default:
        return BlockFace::East;
    }
}

inline std::size_t blockIndex(int x, int y, int z) noexcept
{
    return static_cast<std::size_t>(y) * (kChunkSizeX * kChunkSizeZ) + static_cast<std::size_t>(z) * kChunkSizeX + static_cast<std::size_t>(x);
}

inline std::size_t columnIndex(int x, int z) noexcept
{
    return static_cast<std::size_t>(z) * kChunkSizeX + static_cast<std::size_t>(x);
}

inline int ceilToIntPositive(float value)
{
    const int truncated = static_cast<int>(value);
    return (static_cast<float>(truncated) < value) ? truncated + 1 : truncated;
}

using terrain::BiomeDefinition;
using terrain::ColumnBuildResult;
using terrain::ColumnSample;

// To introduce a new biome:
// 1. Create a new TOML file under assets/biomes describing the biome parameters.
// 2. Provide textures for any new blocks in setBlockTextureAtlasConfig.

inline float hashToUnitFloat(int x, int y, int z) noexcept
{
    constexpr std::uint64_t kMulX = 374761393ull;
    constexpr std::uint64_t kMulY = 668265263ull;
    constexpr std::uint64_t kMulZ = 2147483647ull;
    constexpr std::uint64_t kMixMul = 1274126177ull;
    constexpr std::uint64_t kMask24 = 0xFFFFFFull;

    const auto widen = [](int value) noexcept -> std::uint64_t {
        return static_cast<std::uint64_t>(static_cast<std::uint32_t>(value));
    };

    std::uint64_t h = widen(x) * kMulX + widen(y) * kMulY + widen(z) * kMulZ;
    h = (h ^ (h >> 13)) * kMixMul;
    h ^= (h >> 16);
    return static_cast<float>(h & kMask24) / static_cast<float>(kMask24);
}

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

inline std::uint8_t aoLevelFromPackedVertexLighting(std::uint32_t packed) noexcept
{
    return static_cast<std::uint8_t>((packed >> 8) & 0x03u);
}

inline std::uint8_t vertexFlagsFromPackedLighting(std::uint32_t packed) noexcept
{
    return static_cast<std::uint8_t>((packed >> 10) & 0x3Fu);
}

inline int lightingMetricFromPackedVertex(std::uint32_t packed) noexcept
{
    const std::uint8_t packedLight = static_cast<std::uint8_t>(packed & 0xFFu);
    const int sky = static_cast<int>(skyLightFromPacked(packedLight));
    const int block = static_cast<int>(blockLightFromPacked(packedLight));
    const int ao = static_cast<int>(aoLevelFromPackedVertexLighting(packed));
    return sky * 24 + block * 18 + (3 - ao) * 20;
}

inline bool isAlphaCutoutBlock(BlockId block) noexcept
{
    return block == BlockId::Leaves ||
           block == BlockId::SpruceLeaves ||
           block == BlockId::DarkOakLeaves;
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

struct DefaultTreeCandidate
{
    int originX{0};
    int originZ{0};
    int groundWorldY{0};
    int trunkHeight{0};
    float priority{0.0f};
};

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

inline int defaultTreeTrunkHeight(int worldX, int groundWorldY, int worldZ) noexcept
{
    int height = kDefaultTreeMinHeight +
                 static_cast<int>(hashToUnitFloat(worldX, groundWorldY + 1, worldZ) *
                                  static_cast<float>(kDefaultTreeMaxHeight - kDefaultTreeMinHeight + 1));
    return std::clamp(height, kDefaultTreeMinHeight, kDefaultTreeMaxHeight);
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
                                    Callback&& callback)
{
    for (int dy = 0; dy < trunkHeight; ++dy)
    {
        if (callback(originX, groundWorldY + dy, originZ, BlockId::Wood))
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

                if (callback(originX + dx, worldY, originZ + dz, BlockId::Leaves))
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
                                   [&](int ax, int ay, int az, BlockId) {
                                       return forEachDefaultTreeBlock(b.originX,
                                                                      b.originZ,
                                                                      b.groundWorldY,
                                                                      b.trunkHeight,
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
    if (terrain::isTaigaBiome(biome))
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

constexpr int kStructureRegionSize = 128;
constexpr int kMaxStructureHorizontalRadius = kTaigaSpruceMaxLeafRadius + 1;

struct StructureAabb
{
    glm::ivec3 min{0};
    glm::ivec3 max{0};

    [[nodiscard]] bool intersects(const glm::ivec3& otherMin, const glm::ivec3& otherMax) const noexcept
    {
        return min.x <= otherMax.x && max.x >= otherMin.x &&
               min.y <= otherMax.y && max.y >= otherMin.y &&
               min.z <= otherMax.z && max.z >= otherMin.z;
    }
};

enum class StructureType : std::uint8_t
{
    DefaultTree = 0,
    TaigaSpruce = 1,
};

struct StructureInstance
{
    StructureType type{StructureType::DefaultTree};
    glm::ivec3 origin{0};
    StructureAabb bounds{};
    int trunkHeight{0};
    int bareTrunkHeight{0};
    float priority{0.0f};
    int maxLodLevel{3};
};

template <typename Callback>
inline bool forEachStructureVoxel(const StructureInstance& instance, Callback&& callback)
{
    if (instance.type == StructureType::TaigaSpruce)
    {
        for (int trunkX = 0; trunkX < 2; ++trunkX)
        {
            for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
            {
                for (int dy = 1; dy <= instance.trunkHeight; ++dy)
                {
                    if (callback(instance.origin.x + trunkX,
                                 instance.origin.y + dy,
                                 instance.origin.z + trunkZ,
                                 BlockId::SpruceLog))
                    {
                        return true;
                    }
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
                    if (callback(worldX, worldY, worldZ, BlockId::SpruceLeaves))
                    {
                        return true;
                    }
                }
            }
        }

        const int crownWorldY = canopyTopWorld + 1;
        for (int trunkX = 0; trunkX < 2; ++trunkX)
        {
            for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
            {
                if (callback(instance.origin.x + trunkX,
                             crownWorldY,
                             instance.origin.z + trunkZ,
                             BlockId::SpruceLeaves))
                {
                    return true;
                }
            }
        }
        return false;
    }

    return forEachDefaultTreeBlock(instance.origin.x,
                                   instance.origin.z,
                                   instance.origin.y,
                                   instance.trunkHeight,
                                   std::forward<Callback>(callback));
}

struct StructureRegionKey
{
    int regionX{0};
    int regionZ{0};

    bool operator==(const StructureRegionKey& other) const noexcept
    {
        return regionX == other.regionX && regionZ == other.regionZ;
    }
};

struct StructureRegionKeyHasher
{
    std::size_t operator()(const StructureRegionKey& key) const noexcept
    {
        std::size_t hash = static_cast<std::size_t>(key.regionX) * 73856093u;
        hash ^= static_cast<std::size_t>(key.regionZ) * 19349663u;
        return hash;
    }
};

struct StructureBvhNode
{
    StructureAabb bounds{};
    int leftFirst{0};
    int rightChild{0};
    int primitiveCount{0};
};

struct StructureBvh
{
    static constexpr int kLeafSize = 4;

    void build(const std::vector<StructureInstance>& instances)
    {
        nodes.clear();
        indices.resize(instances.size());
        std::iota(indices.begin(), indices.end(), 0);
        if (instances.empty())
        {
            return;
        }

        buildNode(instances, 0, static_cast<int>(indices.size()));
    }

    void query(const glm::ivec3& queryMin,
               const glm::ivec3& queryMax,
               const std::vector<StructureInstance>& instances,
               std::vector<const StructureInstance*>& out) const
    {
        if (nodes.empty())
        {
            return;
        }

        std::vector<int> stack;
        stack.push_back(0);
        while (!stack.empty())
        {
            const int nodeIndex = stack.back();
            stack.pop_back();
            const StructureBvhNode& node = nodes[static_cast<std::size_t>(nodeIndex)];
            if (!node.bounds.intersects(queryMin, queryMax))
            {
                continue;
            }

            if (node.primitiveCount > 0)
            {
                for (int i = 0; i < node.primitiveCount; ++i)
                {
                    const int instanceIndex = indices[static_cast<std::size_t>(node.leftFirst + i)];
                    const StructureInstance& instance = instances[static_cast<std::size_t>(instanceIndex)];
                    if (instance.bounds.intersects(queryMin, queryMax))
                    {
                        out.push_back(&instance);
                    }
                }
                continue;
            }

            stack.push_back(node.leftFirst);
            stack.push_back(node.rightChild);
        }
    }

    std::vector<StructureBvhNode> nodes;
    std::vector<int> indices;

private:
    int buildNode(const std::vector<StructureInstance>& instances, int begin, int end)
    {
        const int nodeIndex = static_cast<int>(nodes.size());
        nodes.push_back(StructureBvhNode{});
        nodes[static_cast<std::size_t>(nodeIndex)].bounds = boundsForRange(instances, begin, end);

        const int count = end - begin;
        if (count <= kLeafSize)
        {
            nodes[static_cast<std::size_t>(nodeIndex)].leftFirst = begin;
            nodes[static_cast<std::size_t>(nodeIndex)].rightChild = -1;
            nodes[static_cast<std::size_t>(nodeIndex)].primitiveCount = count;
            return nodeIndex;
        }

        const glm::ivec3 extent = nodes[static_cast<std::size_t>(nodeIndex)].bounds.max -
                                  nodes[static_cast<std::size_t>(nodeIndex)].bounds.min;
        int axis = 0;
        if (extent.y > extent.x && extent.y >= extent.z)
        {
            axis = 1;
        }
        else if (extent.z > extent.x)
        {
            axis = 2;
        }

        const int mid = begin + count / 2;
        std::nth_element(indices.begin() + begin,
                         indices.begin() + mid,
                         indices.begin() + end,
                         [&](int lhs, int rhs)
                         {
                             return centerComponent(instances[static_cast<std::size_t>(lhs)].bounds, axis) <
                                    centerComponent(instances[static_cast<std::size_t>(rhs)].bounds, axis);
                         });

        const int leftNode = buildNode(instances, begin, mid);
        const int rightNode = buildNode(instances, mid, end);
        nodes[static_cast<std::size_t>(nodeIndex)].leftFirst = leftNode;
        nodes[static_cast<std::size_t>(nodeIndex)].rightChild = rightNode;
        nodes[static_cast<std::size_t>(nodeIndex)].primitiveCount = 0;
        return nodeIndex;
    }

    [[nodiscard]] StructureAabb boundsForRange(const std::vector<StructureInstance>& instances, int begin, int end) const
    {
        StructureAabb bounds{};
        bounds.min = glm::ivec3(std::numeric_limits<int>::max());
        bounds.max = glm::ivec3(std::numeric_limits<int>::min());
        for (int i = begin; i < end; ++i)
        {
            const StructureAabb& instanceBounds = instances[static_cast<std::size_t>(indices[static_cast<std::size_t>(i)])].bounds;
            bounds.min = glm::min(bounds.min, instanceBounds.min);
            bounds.max = glm::max(bounds.max, instanceBounds.max);
        }
        return bounds;
    }

    [[nodiscard]] static float centerComponent(const StructureAabb& bounds, int axis) noexcept
    {
        const glm::ivec3 center = bounds.min + ((bounds.max - bounds.min) / 2);
        if (axis == 1)
        {
            return static_cast<float>(center.y);
        }
        if (axis == 2)
        {
            return static_cast<float>(center.z);
        }
        return static_cast<float>(center.x);
    }
};

struct StructureRegion
{
    StructureRegionKey key{};
    glm::ivec2 worldMin{0};
    glm::ivec2 worldMax{0};
    std::vector<StructureInstance> instances;
    StructureBvh bvh{};
};

using StructureSampleColumnFn = std::function<ColumnSample(int worldX, int worldZ)>;
using StructureSurfaceBlockFn = std::function<BlockId(int worldX, int worldZ, const ColumnSample&)>;
using StructureDensityFn = std::function<float(int worldX, int worldZ)>;

[[nodiscard]] StructureRegion buildStructureRegionData(const StructureRegionKey& key,
                                                       const StructureSampleColumnFn& sampleColumnFn,
                                                       const StructureSurfaceBlockFn& surfaceBlockFn,
                                                       const StructureDensityFn& densityFn)
{
    StructureRegion region{};
    region.key = key;
    region.worldMin = glm::ivec2(key.regionX * kStructureRegionSize, key.regionZ * kStructureRegionSize);
    region.worldMax = region.worldMin + glm::ivec2(kStructureRegionSize - 1);
    region.instances.reserve(64);

    auto sampleColumn = [&](int worldX, int worldZ) -> ColumnSample {
        return sampleColumnFn(worldX, worldZ);
    };

    auto resolvedSurfaceBlockAt = [&](int worldX, int worldZ, const ColumnSample& sample) -> BlockId {
        return surfaceBlockFn(worldX, worldZ, sample);
    };

    auto densityAt = [&](int worldX, int worldZ) noexcept {
        return densityFn(worldX, worldZ);
    };

    auto canAnchorTaigaSpruce = [&](int originX, int originZ, int& outGroundWorldY) -> bool
    {
        int groundWorldY = std::numeric_limits<int>::min();

        for (int trunkX = 0; trunkX < 2; ++trunkX)
        {
            for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
            {
                const ColumnSample baseSample = sampleColumn(originX + trunkX, originZ + trunkZ);
                if (!baseSample.dominantBiome || !terrain::isTaigaBiome(*baseSample.dominantBiome))
                {
                    return false;
                }
                if (baseSample.dominantWeight < kTreeBiomeWeightThreshold)
                {
                    return false;
                }

                const BlockId surfaceBlock = resolvedSurfaceBlockAt(originX + trunkX, originZ + trunkZ, baseSample);
                if (surfaceBlock != BlockId::Grass && surfaceBlock != BlockId::Podzol)
                {
                    return false;
                }

                if (groundWorldY == std::numeric_limits<int>::min())
                {
                    groundWorldY = baseSample.surfaceY;
                }
                else if (baseSample.surfaceY != groundWorldY)
                {
                    return false;
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
                    return false;
                }
                if (std::abs(neighborSample.surfaceY - groundWorldY) > 1)
                {
                    return false;
                }
            }
        }

        outGroundWorldY = groundWorldY;
        return groundWorldY > 2;
    };

    for (int worldX = region.worldMin.x; worldX <= region.worldMax.x; ++worldX)
    {
        for (int worldZ = region.worldMin.y; worldZ <= region.worldMax.y; ++worldZ)
        {
            const ColumnSample columnSample = sampleColumn(worldX, worldZ);
            if (!columnSample.dominantBiome)
            {
                continue;
            }

            const BiomeDefinition& biome = *columnSample.dominantBiome;
            if (!biome.generatesTrees || columnSample.dominantWeight < kTreeBiomeWeightThreshold)
            {
                continue;
            }

            const int groundWorldY = columnSample.surfaceY;
            if (groundWorldY <= 2)
            {
                continue;
            }

            if (terrain::isTaigaBiome(biome))
            {
                if (!shouldSpawnTaigaSpruce(biome, worldX, groundWorldY, worldZ))
                {
                    continue;
                }

                int taigaGroundWorldY = std::numeric_limits<int>::min();
                if (!canAnchorTaigaSpruce(worldX, worldZ, taigaGroundWorldY))
                {
                    continue;
                }

                StructureInstance instance{};
                instance.type = StructureType::TaigaSpruce;
                instance.origin = glm::ivec3(worldX, taigaGroundWorldY, worldZ);
                instance.trunkHeight = taigaSpruceTrunkHeight(worldX, taigaGroundWorldY, worldZ);
                instance.bareTrunkHeight = taigaSpruceBareTrunkHeight(worldX, taigaGroundWorldY, worldZ);
                instance.maxLodLevel = 4;
                instance.bounds.min = glm::ivec3(worldX - kTaigaSpruceMaxLeafRadius,
                                                 taigaGroundWorldY + 1,
                                                 worldZ - kTaigaSpruceMaxLeafRadius);
                instance.bounds.max = glm::ivec3(worldX + 1 + kTaigaSpruceMaxLeafRadius,
                                                 taigaGroundWorldY + instance.trunkHeight + 1,
                                                 worldZ + 1 + kTaigaSpruceMaxLeafRadius);
                region.instances.push_back(instance);
                continue;
            }

            DefaultTreeCandidate candidate{};
            if (!tryBuildDefaultTreeCandidate(worldX,
                                              worldZ,
                                              columnSample,
                                              sampleColumn,
                                              densityAt,
                                              candidate))
            {
                continue;
            }

            if (defaultTreeHasSpacingConflict(candidate, sampleColumn, densityAt))
            {
                continue;
            }

            StructureInstance instance{};
            instance.type = StructureType::DefaultTree;
            instance.origin = glm::ivec3(candidate.originX, candidate.groundWorldY, candidate.originZ);
            instance.trunkHeight = candidate.trunkHeight;
            instance.priority = candidate.priority;
            instance.maxLodLevel = 3;
            instance.bounds.min = glm::ivec3(candidate.originX - kDefaultTreeMaxRadius,
                                             candidate.groundWorldY,
                                             candidate.originZ - kDefaultTreeMaxRadius);
            instance.bounds.max = glm::ivec3(candidate.originX + kDefaultTreeMaxRadius,
                                             candidate.groundWorldY + candidate.trunkHeight,
                                             candidate.originZ + kDefaultTreeMaxRadius);
            region.instances.push_back(instance);
        }
    }

    region.bvh.build(region.instances);
    return region;
}

struct StructureRegistryProfilingSnapshot
{
    std::uint64_t cacheHits{0};
    std::uint64_t cacheMisses{0};
    std::uint64_t regionsBuilt{0};
    std::uint64_t queryCount{0};
    std::uint64_t totalQueryMicros{0};
    double cacheHitRate{0.0};
    double averageQueryMs{0.0};
};

class StructureRegistry
{
public:
    using SampleColumnFn = std::function<ColumnSample(int worldX, int worldZ)>;
    using SurfaceBlockFn = std::function<BlockId(int worldX, int worldZ, const ColumnSample&)>;
    using DensityFn = std::function<float(int worldX, int worldZ)>;

    StructureRegistry() = default;

    StructureRegistry(SampleColumnFn sampleColumnFn,
                      SurfaceBlockFn surfaceBlockFn,
                      DensityFn densityFn)
        : sampleColumnFn_(std::move(sampleColumnFn)),
          surfaceBlockFn_(std::move(surfaceBlockFn)),
          densityFn_(std::move(densityFn))
    {
    }

    void configure(SampleColumnFn sampleColumnFn,
                   SurfaceBlockFn surfaceBlockFn,
                   DensityFn densityFn)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        sampleColumnFn_ = std::move(sampleColumnFn);
        surfaceBlockFn_ = std::move(surfaceBlockFn);
        densityFn_ = std::move(densityFn);
        regions_.clear();
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        regions_.clear();
    }

    void setProfilingEnabled(bool enabled) noexcept
    {
        profilingEnabled_.store(enabled, std::memory_order_release);
    }

    void resetProfiling() noexcept
    {
        cacheHits_.store(0, std::memory_order_relaxed);
        cacheMisses_.store(0, std::memory_order_relaxed);
        regionsBuilt_.store(0, std::memory_order_relaxed);
        queryCount_.store(0, std::memory_order_relaxed);
        totalQueryMicros_.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] StructureRegistryProfilingSnapshot profilingSnapshot() const noexcept
    {
        StructureRegistryProfilingSnapshot snapshot{};
        snapshot.cacheHits = cacheHits_.load(std::memory_order_relaxed);
        snapshot.cacheMisses = cacheMisses_.load(std::memory_order_relaxed);
        snapshot.regionsBuilt = regionsBuilt_.load(std::memory_order_relaxed);
        snapshot.queryCount = queryCount_.load(std::memory_order_relaxed);
        snapshot.totalQueryMicros = totalQueryMicros_.load(std::memory_order_relaxed);
        const std::uint64_t totalLookups = snapshot.cacheHits + snapshot.cacheMisses;
        if (totalLookups > 0)
        {
            snapshot.cacheHitRate = static_cast<double>(snapshot.cacheHits) / static_cast<double>(totalLookups);
        }
        if (snapshot.queryCount > 0)
        {
            snapshot.averageQueryMs =
                static_cast<double>(snapshot.totalQueryMicros) / (1000.0 * static_cast<double>(snapshot.queryCount));
        }
        return snapshot;
    }

    [[nodiscard]] std::vector<StructureInstance> query(const glm::ivec3& queryMin,
                                                       const glm::ivec3& queryMax,
                                                       int lodLevel = 0) const
    {
        const SteadyClock::time_point queryStart = SteadyClock::now();
        std::vector<StructureInstance> result;
        const int minRegionX = floorDiv(queryMin.x - kMaxStructureHorizontalRadius, kStructureRegionSize);
        const int maxRegionX = floorDiv(queryMax.x + kMaxStructureHorizontalRadius, kStructureRegionSize);
        const int minRegionZ = floorDiv(queryMin.z - kMaxStructureHorizontalRadius, kStructureRegionSize);
        const int maxRegionZ = floorDiv(queryMax.z + kMaxStructureHorizontalRadius, kStructureRegionSize);

        for (int regionZ = minRegionZ; regionZ <= maxRegionZ; ++regionZ)
        {
            for (int regionX = minRegionX; regionX <= maxRegionX; ++regionX)
            {
                const std::shared_ptr<const StructureRegion> region =
                    getOrBuildRegion(StructureRegionKey{regionX, regionZ});
                if (!region)
                {
                    continue;
                }

                std::vector<const StructureInstance*> candidates;
                region->bvh.query(queryMin, queryMax, region->instances, candidates);
                for (const StructureInstance* candidate : candidates)
                {
                    if (candidate == nullptr)
                    {
                        continue;
                    }
                    if (lodLevel > 0 && candidate->maxLodLevel < lodLevel)
                    {
                        continue;
                    }
                    result.push_back(*candidate);
                }
            }
        }

        if (profilingEnabled_.load(std::memory_order_acquire))
        {
            queryCount_.fetch_add(1, std::memory_order_relaxed);
            const auto elapsedMicros =
                std::chrono::duration_cast<std::chrono::microseconds>(SteadyClock::now() - queryStart).count();
            totalQueryMicros_.fetch_add(static_cast<std::uint64_t>(std::max<std::int64_t>(elapsedMicros, 0)),
                                        std::memory_order_relaxed);
        }

        return result;
    }

    [[nodiscard]] std::vector<StructureInstance> copyRegionInstances(const StructureRegionKey& key) const
    {
        const std::shared_ptr<const StructureRegion> region = getOrBuildRegion(key);
        if (!region)
        {
            return {};
        }
        return region->instances;
    }

private:
    [[nodiscard]] std::shared_ptr<const StructureRegion> getOrBuildRegion(const StructureRegionKey& key) const
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = regions_.find(key);
            if (it != regions_.end())
            {
                if (profilingEnabled_.load(std::memory_order_acquire))
                {
                    cacheHits_.fetch_add(1, std::memory_order_relaxed);
                }
                return it->second;
            }
        }

        if (!sampleColumnFn_ || !surfaceBlockFn_ || !densityFn_)
        {
            return {};
        }

        if (profilingEnabled_.load(std::memory_order_acquire))
        {
            cacheMisses_.fetch_add(1, std::memory_order_relaxed);
        }

        auto builtRegion = std::make_shared<StructureRegion>(buildRegion(key));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto [it, inserted] = regions_.emplace(key, builtRegion);
            if (inserted && profilingEnabled_.load(std::memory_order_acquire))
            {
                regionsBuilt_.fetch_add(1, std::memory_order_relaxed);
            }
            return it->second;
        }
    }

    [[nodiscard]] StructureRegion buildRegion(const StructureRegionKey& key) const
    {
        return buildStructureRegionData(key, sampleColumnFn_, surfaceBlockFn_, densityFn_);
    }

    mutable std::mutex mutex_;
    mutable std::unordered_map<StructureRegionKey, std::shared_ptr<StructureRegion>, StructureRegionKeyHasher> regions_{};
    SampleColumnFn sampleColumnFn_{};
    SurfaceBlockFn surfaceBlockFn_{};
    DensityFn densityFn_{};
    std::atomic<bool> profilingEnabled_{false};
    mutable std::atomic<std::uint64_t> cacheHits_{0};
    mutable std::atomic<std::uint64_t> cacheMisses_{0};
    mutable std::atomic<std::uint64_t> regionsBuilt_{0};
    mutable std::atomic<std::uint64_t> queryCount_{0};
    mutable std::atomic<std::uint64_t> totalQueryMicros_{0};
};

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

enum class JobType : std::uint8_t
{
    Generate = 0,
    Mesh = 1
};

constexpr std::uint32_t kInvalidChunkBufferPage = std::numeric_limits<std::uint32_t>::max();

struct Chunk
{
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
        pendingMesh = {};
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
    PendingRenderMesh pendingMesh{};
};

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

struct ChunkHasher
{
    std::size_t operator()(const glm::ivec3& v) const noexcept
    {
        std::size_t hash = static_cast<std::size_t>(v.x) * 73856093u;
        hash ^= static_cast<std::size_t>(v.y) * 19349663u;
        hash ^= static_cast<std::size_t>(v.z) * 83492791u;
        return hash;
    }
};

struct ColumnHasher
{
    std::size_t operator()(const glm::ivec2& v) const noexcept
    {
        std::size_t hash = static_cast<std::size_t>(v.x) * 73856093u;
        hash ^= static_cast<std::size_t>(v.y) * 19349663u;

        return hash;
    }
};

struct PendingStructureEdit
{
    glm::ivec3 chunkCoord{0};
    glm::ivec3 worldPos{0};
    BlockId block{BlockId::Air};
    bool replaceSolid{false};
};

struct Job
{
    JobType type;
    glm::ivec3 chunkCoord;
    std::shared_ptr<Chunk> chunk;
    std::uint32_t generationEpoch{0};
    bool initialReadyPriority{false};

    Job(JobType t,
        const glm::ivec3& coord,
        std::shared_ptr<Chunk> c,
        std::uint32_t epoch = 0,
        bool initialPriority = false)
        : type(t),
          chunkCoord(coord),
          chunk(std::move(c)),
          generationEpoch(epoch),
          initialReadyPriority(initialPriority)
    {
    }
};

constexpr std::size_t kJobTypeCount = 2;

[[nodiscard]] constexpr std::size_t jobTypeIndex(JobType type) noexcept
{
    return static_cast<std::size_t>(type);
}

struct ChunkPriorityKey
{
    int supportBucket{3};
    int horizontalDistance{0};
    int forwardBucket{2};
    int verticalDistance{0};
    int axisDistance{0};
};

[[nodiscard]] glm::vec2 normalizePriorityForwardXZ(const glm::vec3& forward) noexcept
{
    glm::vec2 forwardXZ(forward.x, forward.z);
    if (glm::dot(forwardXZ, forwardXZ) <= kEpsilon)
    {
        return {0.0f, -1.0f};
    }

    return glm::normalize(forwardXZ);
}

[[nodiscard]] ChunkPriorityKey buildChunkPriorityKey(const glm::ivec3& coord,
                                                     const glm::ivec3& origin,
                                                     const glm::vec2& forwardXZ) noexcept
{
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

[[nodiscard]] int compareChunkPriorityKeys(const ChunkPriorityKey& lhs,
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

[[nodiscard]] bool isChunkCoordHigherPriority(const glm::ivec3& lhs,
                                              const glm::ivec3& rhs,
                                              const glm::ivec3& origin,
                                              const glm::vec3& forward) noexcept
{
    const glm::vec2 forwardXZ = normalizePriorityForwardXZ(forward);
    return compareChunkPriorityKeys(buildChunkPriorityKey(lhs, origin, forwardXZ),
                                    buildChunkPriorityKey(rhs, origin, forwardXZ)) < 0;
}

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

class JobQueue
{
public:
    bool push(const Job& job);
    bool tryPop(Job& job);
    Job waitAndPop();
    std::vector<Job> stop();
    bool empty() const;
    std::size_t size() const;
    std::size_t size(JobType type) const;
    std::size_t outstanding(JobType type) const;
    void updatePriorityState(const glm::ivec3& origin, const glm::vec3& forward);
    bool tryUpdatePriorityState(const glm::ivec3& origin, const glm::vec3& forward);
    void setWorkerConcurrency(std::size_t workerCount) noexcept;
    void jobCompleted(JobType type) noexcept;

private:
    struct PrioritizedJob
    {
        Job job;
        ChunkPriorityKey priority{};
        int lifecycleBias{0};
        int stageBias{0};
        std::uint64_t sequence{0};
    };

    struct JobComparer
    {
        bool operator()(const PrioritizedJob& lhs, const PrioritizedJob& rhs) const;
    };

    PrioritizedJob wrap(const Job& job);
    [[nodiscard]] static int comparePrioritizedJobs(const PrioritizedJob& lhs,
                                                    const PrioritizedJob& rhs) noexcept;
    [[nodiscard]] bool hasQueuedJobsLocked() const noexcept;
    [[nodiscard]] std::array<std::size_t, kJobTypeCount> computeStageTargetsLocked() const noexcept;
    [[nodiscard]] std::size_t pickNextQueueIndexLocked() const noexcept;
    void rebuildLocked();

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> shouldStop_{false};
    glm::ivec3 priorityOrigin_{0, 0, 0};
    glm::vec2 priorityForwardXZ_{0.0f, -1.0f};
    std::array<std::priority_queue<PrioritizedJob, std::vector<PrioritizedJob>, JobComparer>, kJobTypeCount> queues_{};
    std::array<std::size_t, kJobTypeCount> activeCounts_{};
    std::atomic<std::size_t> queuedJobCount_{0};
    std::size_t workerConcurrency_{1};
    std::uint64_t nextSequence_{0};
};

class ColumnManager
{
public:
    static constexpr int kNoHeight = std::numeric_limits<int>::min();

    void updateChunk(const Chunk& chunk);
    void updateChunkHeights(const glm::ivec3& chunkCoord,
                            const std::array<int, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)>& highestWorlds);
    void updateColumn(const Chunk& chunk, int localX, int localZ);
    void removeChunk(const Chunk& chunk);
    void clear();

    int highestSolidBlock(int worldX, int worldZ) const noexcept;

private:
    struct ColumnData
    {
        std::unordered_map<int, int> slabHeights;
        int highestWorldY{kNoHeight};
    };

    static glm::ivec2 columnKey(const glm::ivec3& chunkCoord, int localX, int localZ) noexcept;
    static int scanColumnHighestWorld(const Chunk& chunk, int localX, int localZ) noexcept;
    static int computeHighest(const ColumnData& data) noexcept;
    void applyHeightLocked(const glm::ivec2& key, int chunkY, int highestWorldY);

    mutable std::mutex mutex_;
    std::unordered_map<glm::ivec2, ColumnData, ColumnHasher> columns_;
};

class FarTerrainManager
{
public:
    static constexpr int kLogicalSize = 16;
    static constexpr std::size_t kVoxelCount =
        static_cast<std::size_t>(kLogicalSize) *
        static_cast<std::size_t>(kLogicalSize) *
        static_cast<std::size_t>(kLogicalSize);
    static constexpr std::size_t kMaxFarFacesPerTile = 8192u;
    static constexpr std::size_t kMaxFarVerticesPerTile = kMaxFarFacesPerTile * 4u;
    static constexpr std::size_t kMaxFarIndicesPerTile = kMaxFarFacesPerTile * 6u;
    static constexpr std::uint32_t kInvalidFarDrawRecordIndex = std::numeric_limits<std::uint32_t>::max();
    static constexpr std::uint32_t kTopPlaneCount = 3u;
    static constexpr std::uint32_t kSideSlicesPerLayer = 64u;
    static constexpr std::uint32_t kPlaneCount = kTopPlaneCount + 3u * kSideSlicesPerLayer;
    static constexpr std::uint32_t kFacePrefixGroupSize = 256u;
    static constexpr std::uint32_t kFacePrefixGroupCount =
        static_cast<std::uint32_t>((kPlaneCount + kFacePrefixGroupSize - 1u) / kFacePrefixGroupSize);
    static constexpr std::uint32_t kFaceMetadataEntryCount = 1u;
    static constexpr std::uint32_t kMaxTopDescriptorsPerPlane =
        static_cast<std::uint32_t>(kLogicalSize * kLogicalSize);
    static constexpr std::uint32_t kMaxSideDescriptorsPerPlane = static_cast<std::uint32_t>(kLogicalSize);
    static constexpr std::uint32_t kFaceDescriptorCount =
        kTopPlaneCount * kMaxTopDescriptorsPerPlane +
        (kPlaneCount - kTopPlaneCount) * kMaxSideDescriptorsPerPlane;
    static constexpr std::uint32_t kGpuDrawRecordOverflowFlag = 0x80000000u;
    static constexpr std::uint32_t kGpuDrawRecordFaceCountMask = 0x7fffffffu;

    struct FarLodChunkKey
    {
        int level{0};
        glm::ivec3 coord{0};

        bool operator==(const FarLodChunkKey& other) const noexcept
        {
            return level == other.level && coord == other.coord;
        }
    };

    struct FarLodLevelConfig
    {
        int level{0};
        int blockScale{1};
        int innerRadiusChunks{0};
        int outerRadiusChunks{0};

        [[nodiscard]] int chunkSpanBlocks() const noexcept
        {
            return kLogicalSize * blockScale;
        }
    };

    struct FarLodVoxel
    {
        std::uint8_t occupied{0};
        BlockId material{BlockId::Air};
        std::uint8_t flags{0};
    };

    using PackedFarLodVoxelGpu = std::uint32_t;

    struct GpuBlockFaceUv
    {
        glm::vec2 base{0.0f};
        glm::vec2 size{0.0f};
    };
    static_assert(sizeof(GpuBlockFaceUv) == 16u);

    struct GpuStructureInstance
    {
        std::uint32_t type{0};
        std::int32_t originX{0};
        std::int32_t originY{0};
        std::int32_t originZ{0};
        std::int32_t boundsMinX{0};
        std::int32_t boundsMinY{0};
        std::int32_t boundsMinZ{0};
        std::int32_t boundsMaxX{0};
        std::int32_t boundsMaxY{0};
        std::int32_t boundsMaxZ{0};
        std::uint32_t trunkHeight{0};
        std::uint32_t bareTrunkHeight{0};
        std::uint32_t maxLodLevel{0};
        std::uint32_t reserved0{0};
        std::uint32_t reserved1{0};
        std::uint32_t reserved2{0};
    };
    static_assert(sizeof(GpuStructureInstance) == 64u);

    struct GpuStructureRegionState
    {
        StructureRegionKey key{};
        std::uint32_t instanceCount{0};
        Microsoft::WRL::ComPtr<ID3D12Resource> instanceBuffer;
        D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};

        [[nodiscard]] bool valid() const noexcept
        {
            return instanceBuffer != nullptr && instanceCount > 0;
        }
    };

    struct FarLodChunkCpu
    {
        static constexpr int logicalSize = kLogicalSize;

        FarLodChunkKey key{};
        glm::ivec3 worldMin{0};
        int blockScale{1};
        std::array<FarLodVoxel, kVoxelCount> voxels{};
        glm::vec3 boundsMin{0.0f};
        glm::vec3 boundsMax{0.0f};
        int minOccupiedLocalY{logicalSize};
        int maxOccupiedLocalY{-1};
        bool terrainReady{false};
        bool structuresReady{false};
        bool meshReady{false};
    };

    struct FarLodChunkGpuState
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> columnBuffer;
        D3D12_RESOURCE_STATES columnState{D3D12_RESOURCE_STATE_COMMON};
        Microsoft::WRL::ComPtr<ID3D12Resource> voxelBuffer;
        D3D12_RESOURCE_STATES voxelState{D3D12_RESOURCE_STATE_COMMON};
        Microsoft::WRL::ComPtr<ID3D12Resource> faceCountBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> faceAnalysisBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> faceDescriptorBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> facePrefixBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> faceGroupSumBuffer;
        D3D12_RESOURCE_STATES faceCountState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES faceAnalysisState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES faceDescriptorState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES facePrefixState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES faceGroupSumState{D3D12_RESOURCE_STATE_COMMON};
        UINT64 voxelFenceValue{0};
        UINT64 readbackFenceValue{0};
        bool voxelReady{false};
        std::uint32_t parityMismatchCount{0};
        bool parityValidated{false};
        std::uint32_t pageIndex{kInvalidChunkBufferPage};
        std::size_t vertexOffset{0};
        std::size_t indexOffset{0};
        std::size_t vertexCount{0};
        std::uint32_t indexCount{0};
        std::size_t reservedVertexCount{0};
        std::size_t reservedIndexCount{0};
        std::uint32_t faceCapacity{0};
        std::uint32_t recordIndex{kInvalidFarDrawRecordIndex};
        bool resident{false};
    };

    using ColumnSampleFn =
        std::function<ColumnSample(int worldX, int worldZ, int slabMinWorldY, int slabMaxWorldY)>;

    FarTerrainManager()
        : levels_{
              FarLodLevelConfig{1, 2, kDefaultNearRenderDistance, 32},
              FarLodLevelConfig{2, 4, 32, 80},
              FarLodLevelConfig{3, 8, 128, 192},
              FarLodLevelConfig{4, 16, 192, 320},
              FarLodLevelConfig{5, 32, 320, kMaxTotalRenderDistanceChunks}}
    {
        const char* parityEnv = std::getenv("BLOCKGAME_ENABLE_LOD_GPU_PARITY");
        gpuParityEnabled_ =
            (parityEnv != nullptr && std::string_view(parityEnv) != "0" && std::string_view(parityEnv) != "false");
    }

    ~FarTerrainManager()
    {
        shutdown();
    }

    void shutdown()
    {
        setRenderSynchronization(nullptr, 0);
        stopWorkers();
        clear();
        uploadContext_.shutdown();
        gpuContext_.shutdown();
        worldgenHeaderBuffer_.Reset();
        worldgenBiomeBuffer_.Reset();
        blockUvBuffer_.Reset();
        blockUvCount_ = 0;
        emptyVoxelBuffer_.Reset();
        levelAtlases_.clear();
        {
            std::lock_guard<std::mutex> lock(structureRegionMutex_);
            destroyGpuStructureRegions();
        }
        if (parityReadbackScratch_ != nullptr)
        {
            parityReadbackScratch_->Unmap(0, nullptr);
        }
        parityReadbackScratch_.Reset();
        parityReadbackMapped_ = nullptr;
    }

    void setEnabled(bool enabled)
    {
        if (enabled_ == enabled)
        {
            return;
        }

        enabled_ = enabled;
        if (!enabled_)
        {
            clear();
        }
    }

    [[nodiscard]] bool enabled() const noexcept
    {
        return enabled_;
    }

    void setWorkerCount(std::size_t count)
    {
        const std::size_t clamped = std::max<std::size_t>(count, 1);
        if (workerCount_ == clamped)
        {
            return;
        }

        stopWorkers();
        workerCount_ = clamped;
    }

    void setDistanceBlocks(int blocks)
    {
        farDistanceTargetBlocks_ = blocks <= 0 ? 0 : std::max(blocks, 256);
        if (farDistanceTargetBlocks_ <= 0)
        {
            farDistanceBlocks_ = 0;
        }
        else if (farDistanceBlocks_ > farDistanceTargetBlocks_)
        {
            farDistanceBlocks_ = farDistanceTargetBlocks_;
        }
    }

    [[nodiscard]] int distanceBlocks() const noexcept
    {
        return farDistanceTargetBlocks_;
    }

    void setFogStartBlocks(int blocks) noexcept
    {
        fogStartBlocks_ = std::max(blocks, 0);
    }

    [[nodiscard]] int fogStartBlocks() const noexcept
    {
        return fogStartBlocks_;
    }

    void setDevice(ID3D12Device* device)
    {
        device_ = device;
        uploadContext_.initialize(device_.Get());
        gpuContext_.initialize(device_.Get());
        gpuContext_.setReadbackEnabled(false);
        levelAtlases_.clear();
        uploadWorldgenTables();
        emptyVoxelBuffer_.Reset();
        {
            const std::uint64_t bufferBytes =
                static_cast<std::uint64_t>(kLogicalSize * kLogicalSize * sizeof(GpuTerrainColumnDescriptor));
            emptyVoxelBuffer_ = createDefaultBuffer(device_.Get(), bufferBytes, D3D12_RESOURCE_STATE_COMMON);
            if (emptyVoxelBuffer_ != nullptr)
            {
                std::byte* uploadMapped = nullptr;
                Microsoft::WRL::ComPtr<ID3D12Resource> upload = createUploadBuffer(device_.Get(), bufferBytes, uploadMapped);
                if (upload != nullptr && uploadMapped != nullptr)
                {
                    std::memset(uploadMapped, 0, static_cast<std::size_t>(bufferBytes));
                    if (uploadContext_.ready() && uploadContext_.begin())
                    {
                        uploadContext_.transition(emptyVoxelBuffer_.Get(),
                                                  D3D12_RESOURCE_STATE_COMMON,
                                                  D3D12_RESOURCE_STATE_COPY_DEST);
                        uploadContext_.copyBuffer(emptyVoxelBuffer_.Get(), 0, upload.Get(), 0, bufferBytes);
                        uploadContext_.transition(emptyVoxelBuffer_.Get(),
                                                  D3D12_RESOURCE_STATE_COPY_DEST,
                                                  D3D12_RESOURCE_STATE_COMMON);
                        uploadContext_.flush(nullptr);
                        uploadContext_.waitForIdle();
                    }
                    upload->Unmap(0, nullptr);
                }
            }
        }
        if (parityReadbackScratch_ != nullptr)
        {
            parityReadbackScratch_->Unmap(0, nullptr);
        }
        parityReadbackScratch_.Reset();
        parityReadbackMapped_ = nullptr;
        if (gpuParityEnabled_)
        {
            parityReadbackScratch_ = createReadbackBuffer(device_.Get(),
                                                          static_cast<std::uint64_t>(kVoxelCount * sizeof(PackedFarLodVoxelGpu)),
                                                          parityReadbackMapped_);
            setDebugObjectName(parityReadbackScratch_.Get(), L"FarLodParityReadbackScratch");
        }
        clear();
    }

    void setRenderSynchronization(ID3D12Fence* graphicsFence, UINT64 graphicsFenceValue)
    {
        uploadContext_.setGraphicsFenceDependency(graphicsFence, graphicsFenceValue);
    }

    [[nodiscard]] ID3D12Fence* uploadFence() const noexcept
    {
        return uploadContext_.fence();
    }

    [[nodiscard]] UINT64 lastSubmittedUploadFenceValue() const noexcept
    {
        return uploadContext_.lastSubmittedFenceValue();
    }

    void setBenchmarkMetrics(ChunkBenchmarkMetrics* metrics) noexcept
    {
        benchmarkMetrics_ = metrics;
    }

    void setSeaLevel(int seaLevel) noexcept
    {
        seaLevel_ = seaLevel;
    }

    void setWorldgenTables(const terrain::FarLodWorldgenTables& tables)
    {
        worldgenTables_ = tables;
        uploadWorldgenTables();
        for (auto& [levelId, atlas] : levelAtlases_)
        {
            (void)levelId;
            atlas.initialized = false;
        }
    }

    void setStructureFieldSources(StructureSampleColumnFn sampleColumnFn,
                                  StructureSurfaceBlockFn surfaceBlockFn,
                                  StructureDensityFn densityFn)
    {
        std::lock_guard<std::mutex> lock(structureRegionMutex_);
        structureSampleColumnFn_ = std::move(sampleColumnFn);
        structureSurfaceBlockFn_ = std::move(surfaceBlockFn);
        structureDensityFn_ = std::move(densityFn);
        destroyGpuStructureRegions();
    }

    void setBlockUvTable(const std::vector<GpuBlockFaceUv>& table)
    {
        blockUvBuffer_.Reset();
        blockUvCount_ = 0;
        if (device_ == nullptr || table.empty())
        {
            return;
        }

        const std::uint64_t bufferBytes = static_cast<std::uint64_t>(table.size() * sizeof(GpuBlockFaceUv));
        blockUvBuffer_ = createDefaultBuffer(device_.Get(), bufferBytes, D3D12_RESOURCE_STATE_COMMON);
        if (blockUvBuffer_ == nullptr)
        {
            return;
        }

        std::byte* uploadMapped = nullptr;
        Microsoft::WRL::ComPtr<ID3D12Resource> upload = createUploadBuffer(device_.Get(), bufferBytes, uploadMapped);
        if (upload == nullptr || uploadMapped == nullptr)
        {
            return;
        }
        std::memcpy(uploadMapped, table.data(), static_cast<std::size_t>(bufferBytes));

        if (uploadContext_.ready() && uploadContext_.begin())
        {
            uploadContext_.transition(blockUvBuffer_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            uploadContext_.copyBuffer(blockUvBuffer_.Get(), 0, upload.Get(), 0, bufferBytes);
            uploadContext_.transition(blockUvBuffer_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            uploadContext_.flush(nullptr);
            uploadContext_.waitForIdle();
        }

        upload->Unmap(0, nullptr);
        blockUvCount_ = static_cast<std::uint32_t>(table.size());
    }

    void setVisibility(const Frustum& frustum, const glm::vec3& cameraWorldPos) const
    {
        std::lock_guard<std::mutex> lock(configMutex_);
        lastVisibilityFrustum_ = frustum;
        lastVisibilityCameraPos_ = cameraWorldPos;
        hasVisibilityFrustum_ = true;
    }

    void setBacklogPressure(int exactMissingChunks, std::size_t exactPendingUploads) noexcept
    {
        exactMissingChunks_ = std::max(exactMissingChunks, 0);
        exactPendingUploads_ = exactPendingUploads;
    }

    void update(const glm::ivec3& cameraChunk,
                const glm::vec3& cameraForward,
                int nearRadiusChunks,
                int realDistanceBlocks,
                double uploadBudgetMs,
                const ColumnSampleFn& columnSampleFn)
    {
        builtTilesLastUpdate_ = 0;
        if (!enabled_ || realDistanceBlocks <= 0)
        {
            clear();
            return;
        }
        if (workerThreads_.empty())
        {
            startWorkers();
        }

        commitPendingMeshUploads();
        ++updateStamp_;
        cameraChunk_ = cameraChunk;
        if (glm::dot(cameraForward, cameraForward) > kEpsilon)
        {
            cameraForward_ = glm::normalize(cameraForward);
        }

        {
            std::lock_guard<std::mutex> lock(configMutex_);
            columnSampleFn_ = columnSampleFn;
        }

        FarLodWorkBudget workBudget = computeWorkBudget();
        const int requestedDistanceBlocks = realDistanceBlocks <= 0 ? 0 : std::max(realDistanceBlocks, 256);
        if (farDistanceTargetBlocks_ <= 0)
        {
            farDistanceTargetBlocks_ = requestedDistanceBlocks;
        }

        const int minimumActiveRadiusChunks = nearRadiusChunks + 1;
        const int targetRadiusChunks =
            farDistanceTargetBlocks_ > 0
                ? std::max(minimumActiveRadiusChunks,
                           ceilToIntPositive(static_cast<float>(farDistanceTargetBlocks_) /
                                             static_cast<float>(kChunkSizeX)))
                : 0;
        int currentRadiusChunks =
            farDistanceBlocks_ > 0
                ? std::max(minimumActiveRadiusChunks,
                           ceilToIntPositive(static_cast<float>(farDistanceBlocks_) /
                                             static_cast<float>(kChunkSizeX)))
                : minimumActiveRadiusChunks;

        if (targetRadiusChunks <= 0)
        {
            farDistanceBlocks_ = 0;
        }
        else if (currentRadiusChunks > targetRadiusChunks)
        {
            currentRadiusChunks = targetRadiusChunks;
            farDistanceBlocks_ = chunksToBlocks(currentRadiusChunks);
        }
        else if (currentRadiusChunks < targetRadiusChunks)
        {
            currentRadiusChunks = std::min(targetRadiusChunks,
                                           currentRadiusChunks + std::max(workBudget.distanceRampStepChunks, 0));
            farDistanceBlocks_ = chunksToBlocks(currentRadiusChunks);
        }
        else
        {
            farDistanceBlocks_ = chunksToBlocks(currentRadiusChunks);
        }

        const int realRadiusChunks = std::max(minimumActiveRadiusChunks, currentRadiusChunks);
        refreshLevels(nearRadiusChunks, realRadiusChunks);
        std::size_t remainingNewChunkActivations = workBudget.newChunkActivations;
        std::size_t remainingNewFallbackActivations = workBudget.newFallbackActivations;

        for (const FarLodLevelConfig& level : levels_)
        {
            if (level.outerRadiusChunks > level.innerRadiusChunks)
            {
                int& activationOuterRadiusChunks = levelActivationOuterRadiusChunks_[level.level];
                activationOuterRadiusChunks = std::max(activationOuterRadiusChunks, level.innerRadiusChunks);
                activationOuterRadiusChunks =
                    std::min(level.outerRadiusChunks, activationOuterRadiusChunks + workBudget.activationStepChunks);
            }
        }

        consumeReadyTouchLevelPlan();
        for (const FarLodLevelConfig& level : levels_)
        {
            if (level.outerRadiusChunks > level.innerRadiusChunks)
            {
                applyTouchLevelCache(level, remainingNewChunkActivations, remainingNewFallbackActivations);
            }
        }
        requestTouchLevelPlan(cameraChunk_);

        static constexpr std::uint64_t kFarTileUntouchedGraceUpdates = 12u;
        static constexpr std::uint64_t kFarFallbackUntouchedGraceUpdates = 2u;
        std::vector<FarLodChunkKey> staleKeys;
        staleKeys.reserve(chunks_.size());
        for (auto& [key, chunk] : chunks_)
        {
            if (chunk.lastTouchedStamp == updateStamp_)
            {
                continue;
            }

            if (chunk.fallbackOnly)
            {
                // Fallback parents are temporary: stop drawing them as soon as they are no longer needed,
                // then retire their residency quickly so they don't hang around as redundant coverage.
                chunk.active = false;
                if (chunk.gpu.resident && chunk.gpu.indexCount > 0)
                {
                    releaseChunkRenderAllocation(chunk);
                }
            }

            const std::uint64_t untouchedUpdates = updateStamp_ - chunk.lastTouchedStamp;
            const std::uint64_t graceUpdates = chunk.fallbackOnly ? kFarFallbackUntouchedGraceUpdates : kFarTileUntouchedGraceUpdates;
            if (untouchedUpdates > graceUpdates)
            {
                staleKeys.push_back(key);
            }
        }

        if (staleKeys.size() > workBudget.staleReleaseCount)
        {
            staleKeys.resize(workBudget.staleReleaseCount);
        }

        for (const FarLodChunkKey& key : staleKeys)
        {
            auto it = chunks_.find(key);
            if (it == chunks_.end())
            {
                continue;
            }

            releaseChunkGpu(it->second);
            chunks_.erase(it);
        }

        if (lodVisibilityDebugLoggingEnabled())
        {
            std::size_t touchedCount = 0;
            std::size_t activeCount = 0;
            std::size_t residentCount = 0;
            std::size_t dirtyCount = 0;
            for (const auto& [key, chunk] : chunks_)
            {
                (void)key;
                if (chunk.lastTouchedStamp == updateStamp_)
                {
                    ++touchedCount;
                }
                if (chunk.active)
                {
                    ++activeCount;
                }
                if (chunk.gpu.resident && chunk.gpu.indexCount > 0)
                {
                    ++residentCount;
                }
                if (chunk.dirty)
                {
                    ++dirtyCount;
                }
            }

            std::ostringstream stream;
            stream << "lodvis update stamp=" << updateStamp_
                   << " chunks=" << chunks_.size()
                   << " touched=" << touchedCount
                   << " active=" << activeCount
                   << " resident=" << residentCount
                   << " dirty=" << dirtyCount
                   << " stale_erased=" << staleKeys.size();
            lodVisibilityDebugLog(stream.str());
        }

        scheduleDirtyBuilds();
        submitGpuSynthesisRequests(uploadBudgetMs);
        collectCompletedBuilds(uploadBudgetMs);
        pollGpuParityResults();
    }

    [[nodiscard]] std::vector<ChunkRenderBatch> buildRenderBatches(const Frustum& frustum) const
    {
        (void)frustum;
        std::lock_guard<std::mutex> lock(configMutex_);
        std::vector<ChunkRenderBatch> batches(bufferPages_.size());
        lastRenderedFaceCount_ = 0;
        lastRenderedVertexCount_ = 0;
        std::size_t emittedPages = 0;
        bool loggedFirstRecord = false;
        for (std::size_t pageIndex = 0; pageIndex < bufferPages_.size(); ++pageIndex)
        {
            const BufferPage& page = bufferPages_[pageIndex];
            ChunkRenderBatch& batch = batches[pageIndex];
            batch.vertexBufferView = page.vertexView;
            batch.indexBufferView = page.indexView;
            batch.gpuCullRecordBuffer = page.drawRecordBuffer.Get();
            batch.gpuCullRecordCount = static_cast<std::uint32_t>(page.recordActiveCount);
            batch.supportsGpuCull = (batch.gpuCullRecordBuffer != nullptr && batch.gpuCullRecordCount > 0);
            batch.debugPageIndex = static_cast<std::uint32_t>(pageIndex);
            if (batch.supportsGpuCull)
            {
                ++emittedPages;
                if (lodVisibilityDebugLoggingEnabled() && !loggedFirstRecord)
                {
                    for (const auto& [key, chunk] : chunks_)
                    {
                        (void)key;
                        if (!chunk.active ||
                            !chunk.gpu.resident ||
                            chunk.gpu.pageIndex != pageIndex ||
                            chunk.gpu.indexCount == 0)
                        {
                            continue;
                        }

                        std::ostringstream recordStream;
                        recordStream << "lodvis first_record page=" << pageIndex
                                     << " level=" << chunk.key.level
                                     << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                                     << " boundsMin=[" << chunk.residentBoundsMin.x << "," << chunk.residentBoundsMin.y << "," << chunk.residentBoundsMin.z << "]"
                                     << " boundsMax=[" << chunk.residentBoundsMax.x << "," << chunk.residentBoundsMax.y << "," << chunk.residentBoundsMax.z << "]"
                                     << " indexCount=" << chunk.gpu.indexCount
                                     << " firstIndex=" << chunk.gpu.indexOffset
                                     << " baseVertex=" << chunk.gpu.vertexOffset
                                     << " recordIndex=" << chunk.gpu.recordIndex
                                     << " pageRecordCount=" << page.recordActiveCount;
                        lodVisibilityDebugLog(recordStream.str());
                        loggedFirstRecord = true;
                        break;
                    }
                }
            }
        }

        auto emptyIt = std::remove_if(batches.begin(),
                                      batches.end(),
                                      [](const ChunkRenderBatch& batch)
                                      {
                                          return batch.gpuCullRecordCount == 0;
                                      });
        batches.erase(emptyIt, batches.end());
        if (lodVisibilityDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "lodvis batches emitted_pages=" << emittedPages
                   << " batch_count=" << batches.size()
                   << " skipped_inactive=0"
                   << " skipped_nonresident=0"
                   << " skipped_noindices=0"
                   << " skipped_invalidpage=0"
                   << " skipped_frustum=0";
            lodVisibilityDebugLog(stream.str());
        }
        return batches;
    }

    void invalidateWorldBlock(const glm::ivec3& worldPos)
    {
        std::lock_guard<std::mutex> lock(configMutex_);
        for (auto& [levelId, atlas] : levelAtlases_)
        {
            (void)levelId;
            const int scale = std::max(atlas.blockScale, 1);
            const glm::ivec2 cellCoord(floorDiv(worldPos.x, scale), floorDiv(worldPos.z, scale));
            appendClippedAtlasUpdateRect(atlas.pendingDirtyRects,
                                         AtlasUpdateRect{cellCoord, glm::ivec2(1, 1)},
                                         atlas.originCell,
                                         atlas.atlasSizeCells);
        }

        for (auto& [key, chunk] : chunks_)
        {
            (void)key;
            const int span = chunk.level.chunkSpanBlocks();
            const glm::ivec3 minWorld = chunk.cpu.worldMin;
            const glm::ivec3 maxWorld = minWorld + glm::ivec3(span - 1);
            if (worldPos.x < minWorld.x - span || worldPos.x > maxWorld.x + span ||
                worldPos.y < minWorld.y - span || worldPos.y > maxWorld.y + span ||
                worldPos.z < minWorld.z - span || worldPos.z > maxWorld.z + span)
            {
                continue;
            }

            markDirty(chunk);
        }
    }

    void clear()
    {
        buildEpoch_.fetch_add(1, std::memory_order_acq_rel);
        {
            std::lock_guard<std::mutex> lock(buildQueueMutex_);
            buildQueue_.clear();
            queuedKeys_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(completedMutex_);
            completedBuilds_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            gpuSynthesisRequests_.clear();
            pendingGpuParityReadbacks_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(dirtyBuildPlanMutex_);
            hasPendingDirtyBuildPlanRequest_ = false;
            hasReadyDirtyBuildPlanResult_ = false;
            pendingDirtyBuildPlanRequest_ = {};
            readyDirtyBuildPlanResult_ = {};
        }
        {
            std::lock_guard<std::mutex> lock(touchLevelPlanMutex_);
            hasPendingTouchLevelPlanRequest_ = false;
            hasReadyTouchLevelPlanResult_ = false;
            touchLevelPlannerBusy_ = false;
            pendingTouchLevelPlanRequest_ = {};
            readyTouchLevelPlanResult_ = {};
            touchLevelCaches_.clear();
            hasLastTouchLevelPlanRequest_ = false;
            lastTouchLevelPlanLevels_.clear();
        }
        std::lock_guard<std::mutex> lock(configMutex_);
        for (auto& [key, chunk] : chunks_)
        {
            (void)key;
            releaseChunkGpu(chunk);
            chunk.inFlight = false;
        }
        chunks_.clear();
        destroyBufferPages();
        builtTilesLastUpdate_ = 0;
        skippedTilesLastUpdate_ = 0;
        lastAverageBuildMs_ = 0.0;
        lastAverageGpuSynthesisMs_ = 0.0;
        lastAverageGpuStampMs_ = 0.0;
        lastAverageGpuFaceBuildMs_ = 0.0;
        lastAverageCpuTerrainSynthesisMs_ = 0.0;
        lastAverageCpuStructureStampMs_ = 0.0;
        lastAverageCpuMeshMs_ = 0.0;
        lastAverageUploadWaitMs_ = 0.0;
        lastAverageUploadCopyMs_ = 0.0;
        lastCollectMs_ = 0.0;
        lastUploadMs_ = 0.0;
        lastBuiltFaceCount_ = 0;
        lastBuiltVertexCount_ = 0;
        lastBuiltIndexCount_ = 0;
        lastRenderedFaceCount_ = 0;
        lastRenderedVertexCount_ = 0;
        rollingGpuParityMismatchCount_ = 0;
        levelActivationOuterRadiusChunks_.clear();
        {
            std::lock_guard<std::mutex> structureLock(structureRegionMutex_);
            destroyGpuStructureRegions();
        }
    }

    [[nodiscard]] int activeTileCount() const noexcept
    {
        std::lock_guard<std::mutex> lock(configMutex_);
        return static_cast<int>(chunks_.size());
    }

    [[nodiscard]] int dirtyTileCount() const noexcept
    {
        int dirty = 0;
        std::lock_guard<std::mutex> lock(configMutex_);
        for (const auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (chunk.active && chunk.dirty)
            {
                ++dirty;
            }
        }
        return dirty;
    }

    [[nodiscard]] int builtTilesLastUpdate() const noexcept
    {
        return builtTilesLastUpdate_;
    }

    [[nodiscard]] int skippedTilesLastUpdate() const noexcept
    {
        return skippedTilesLastUpdate_;
    }

    [[nodiscard]] double averageCpuTerrainSynthesisMs() const noexcept
    {
        return lastAverageCpuTerrainSynthesisMs_;
    }

    [[nodiscard]] double averageCpuStructureStampMs() const noexcept
    {
        return lastAverageCpuStructureStampMs_;
    }

    [[nodiscard]] double averageCpuMeshMs() const noexcept
    {
        return lastAverageCpuMeshMs_;
    }

    [[nodiscard]] double averageUploadWaitMs() const noexcept
    {
        return lastAverageUploadWaitMs_;
    }

    [[nodiscard]] double averageUploadCopyMs() const noexcept
    {
        return lastAverageUploadCopyMs_;
    }

    [[nodiscard]] std::size_t renderedFaceCount() const noexcept
    {
        return lastRenderedFaceCount_;
    }

    [[nodiscard]] std::size_t renderedVertexCount() const noexcept
    {
        return lastRenderedVertexCount_;
    }

    [[nodiscard]] double lastCollectMs() const noexcept
    {
        return lastCollectMs_;
    }

    [[nodiscard]] double lastUploadMs() const noexcept
    {
        return lastUploadMs_;
    }

private:
    void refreshLevels(int nearRadiusChunks, int totalRadiusChunks)
    {
        struct LevelTemplate
        {
            int level;
            int blockScale;
            int outerRadiusChunks;
        };

        static constexpr std::array<LevelTemplate, 8> kLevelTemplates{{
            {1, 2, 32},
            {2, 4, 80},
            {3, 8, 128},
            {4, 16, 192},
            {5, 32, 288},
            {6, 64, 384},
            {7, 128, 480},
            {8, 256, kMaxTotalRenderDistanceChunks},
        }};

        levels_.clear();
        if (totalRadiusChunks <= nearRadiusChunks)
        {
            return;
        }

        constexpr int kExactOverlapChunks = 4;
        int previousOuterRadiusChunks = nearRadiusChunks;
        for (const LevelTemplate& levelTemplate : kLevelTemplates)
        {
            const int outerRadiusChunks = std::min(totalRadiusChunks, levelTemplate.outerRadiusChunks);
            const int innerRadiusChunks = levels_.empty()
                                              ? std::max(0, nearRadiusChunks - kExactOverlapChunks)
                                              : std::max(0, previousOuterRadiusChunks - levelTemplate.blockScale);
            if (outerRadiusChunks <= innerRadiusChunks)
            {
                continue;
            }

            levels_.push_back(FarLodLevelConfig{
                levelTemplate.level,
                levelTemplate.blockScale,
                innerRadiusChunks,
                outerRadiusChunks});
            previousOuterRadiusChunks = outerRadiusChunks;
            if (outerRadiusChunks == totalRadiusChunks)
            {
                break;
            }
        }

        std::unordered_set<int> activeLevelIds;
        activeLevelIds.reserve(levels_.size());
        for (const FarLodLevelConfig& level : levels_)
        {
            activeLevelIds.insert(level.level);
            int& progress = levelActivationOuterRadiusChunks_[level.level];
            if (progress < level.innerRadiusChunks)
            {
                progress = level.innerRadiusChunks;
            }
            progress = std::min(progress, level.outerRadiusChunks);
        }

        for (auto it = levelActivationOuterRadiusChunks_.begin(); it != levelActivationOuterRadiusChunks_.end();)
        {
            if (!activeLevelIds.contains(it->first))
            {
                it = levelActivationOuterRadiusChunks_.erase(it);
            }
            else
            {
                ++it;
            }
        }

        for (auto it = touchLevelCaches_.begin(); it != touchLevelCaches_.end();)
        {
            if (!activeLevelIds.contains(it->first))
            {
                it = touchLevelCaches_.erase(it);
            }
            else
            {
                ++it;
            }
        }
        if (!activeLevelIds.size())
        {
            hasLastTouchLevelPlanRequest_ = false;
            lastTouchLevelPlanLevels_.clear();
        }
    }

    struct FarLodChunkKeyHasher
    {
        std::size_t operator()(const FarLodChunkKey& key) const noexcept
        {
            std::size_t hash = static_cast<std::size_t>(key.level) * 73856093u;
            hash ^= static_cast<std::size_t>(key.coord.x) * 19349663u;
            hash ^= static_cast<std::size_t>(key.coord.y) * 83492791u;
            hash ^= static_cast<std::size_t>(key.coord.z) * 2654435761u;
            return hash;
        }
    };

    struct BufferPage
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
        Microsoft::WRL::ComPtr<ID3D12Resource> drawRecordBuffer;
        D3D12_VERTEX_BUFFER_VIEW vertexView{};
        D3D12_INDEX_BUFFER_VIEW indexView{};
        std::byte* mappedVertexData{nullptr};
        std::byte* mappedIndexData{nullptr};
        D3D12_RESOURCE_STATES vertexState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES indexState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES drawRecordState{D3D12_RESOURCE_STATE_COMMON};
        std::size_t vertexCapacity{0};
        std::size_t indexCapacity{0};
        std::size_t recordCapacity{0};
        std::size_t vertexCursor{0};
        std::size_t indexCursor{0};
        std::size_t recordCursor{0};
        std::size_t recordActiveCount{0};
        std::vector<Range> freeVertices;
        std::vector<Range> freeIndices;
        std::vector<Range> freeRecords;
    };

    struct Allocation
    {
        std::uint32_t pageIndex{kInvalidChunkBufferPage};
        std::size_t vertexOffset{0};
        std::size_t indexOffset{0};
        std::uint32_t recordIndex{kInvalidFarDrawRecordIndex};
    };

    struct FarLodChunkRecord
    {
        struct PendingRenderMesh
        {
            std::uint32_t pageIndex{kInvalidChunkBufferPage};
            std::size_t vertexOffset{0};
            std::size_t indexOffset{0};
            std::size_t vertexCount{0};
            std::uint32_t indexCount{0};
            std::uint32_t faceCapacity{0};
            std::uint32_t recordIndex{kInvalidFarDrawRecordIndex};
            glm::vec3 boundsMin{0.0f};
            glm::vec3 boundsMax{1.0f};
            UINT64 uploadFenceValue{0};
            UINT64 gpuFenceValue{0};
            std::uint32_t buildVersion{0};
            std::uint64_t epoch{0};
            bool gpuGenerated{false};
            Microsoft::WRL::ComPtr<ID3D12Resource> drawRecordReadbackBuffer;
            ChunkRenderBatch::GpuCullRecord* mappedDrawRecord{nullptr};

            [[nodiscard]] bool valid() const noexcept
            {
                return pageIndex != kInvalidChunkBufferPage;
            }
        };

        FarLodChunkKey key{};
        FarLodLevelConfig level{};
        FarLodChunkCpu cpu{};
        FarLodChunkGpuState gpu{};
        glm::vec3 residentBoundsMin{0.0f};
        glm::vec3 residentBoundsMax{1.0f};
        PendingRenderMesh pendingMesh{};
        bool fallbackOnly{false};
        std::uint64_t seamSignature{0};
        std::uint64_t lastBuiltAtlasDependencyRevision{0};
        std::uint64_t lastTouchedStamp{0};
        std::uint32_t buildVersion{1};
        std::uint32_t requestedFaceCapacityHint{0};
        bool active{false};
        bool dirty{true};
        bool initialized{false};
        bool inFlight{false};
    };

    struct ChunkMesh
    {
        std::vector<Vertex> vertices;
        std::vector<std::uint32_t> indices;
        glm::vec3 boundsMin{0.0f};
        glm::vec3 boundsMax{1.0f};
    };

    struct BuildResult
    {
        FarLodChunkKey key{};
        std::uint32_t buildVersion{0};
        std::uint64_t epoch{0};
        FarLodChunkCpu cpu{};
        ChunkMesh mesh{};
        double buildMs{0.0};
        double cpuTerrainSynthesisMs{0.0};
        double cpuStructureStampMs{0.0};
        double cpuMeshMs{0.0};
        double uploadWaitMs{0.0};
        double uploadCopyMs{0.0};
        std::size_t faceCount{0};
        std::size_t vertexCount{0};
        std::size_t indexCount{0};
        bool skippedByRelevance{false};
    };

    struct GpuTerrainColumnDescriptor
    {
        std::uint32_t flags{0};
        std::int32_t terrainTopY{0};
        std::int32_t terrainBaseY{0};
        std::int32_t waterTopY{0};
        std::int32_t waterBottomY{0};
        std::int32_t canopyTopY{0};
        std::int32_t canopyBottomY{0};
        std::uint32_t terrainTopBlock{0};
        std::uint32_t terrainSideBlock{0};
        std::uint32_t waterBlock{0};
        std::uint32_t canopyBlock{0};
        std::uint32_t reserved{0};
    };
    static_assert(sizeof(GpuTerrainColumnDescriptor) == 48u);

    struct GpuTerrainAtlasSample
    {
        std::uint32_t hasSolid{0};
        std::uint32_t waterEnabled{0}; // Aggregated water presence votes within this cell (0..N).
        std::int32_t surfaceY{0};
        std::int32_t waterBottomY{0};
        std::int32_t minSurfaceY{0};
        std::int32_t maxSurfaceY{0};
        std::uint32_t surfaceBlock{0};
        std::uint32_t fillerBlock{0};
        std::int32_t canopyBottomY{0};
        std::int32_t canopyTopY{0};
        std::uint32_t canopyBlock{0};
        std::uint32_t canopyStrength{0};
    };
    static_assert(sizeof(GpuTerrainAtlasSample) == 48u);

    struct GpuAtlasSamplePoint
    {
        std::uint32_t biomeIndex{0};
        std::uint32_t biomeFlags{0};
        std::int32_t surfaceY{0};
        float distanceToShore{0.0f};
    };
    static_assert(sizeof(GpuAtlasSamplePoint) == 16u);

    struct GpuAtlasSampleCacheEntry
    {
        std::array<GpuAtlasSamplePoint, 9> points{};
    };
    static_assert(sizeof(GpuAtlasSampleCacheEntry) == 144u);

    struct GpuChunkSeedCacheSeed
    {
        std::uint32_t biomeIndex{0};
        std::int32_t positionX{0};
        std::int32_t positionZ{0};
        float radius{0.0f};
        float baseHeight{0.0f};
    };
    static_assert(sizeof(GpuChunkSeedCacheSeed) == 20u);

    struct GpuChunkSeedCacheHeader
    {
        std::uint32_t seedCount{0};
        std::uint32_t baseSeedIndex{0};
        std::uint32_t reserved0{0};
        std::uint32_t reserved1{0};
    };
    static_assert(sizeof(GpuChunkSeedCacheHeader) == 16u);

    struct GpuSynthesisRequest
    {
        FarLodChunkKey key{};
        std::uint32_t buildVersion{0};
        std::uint64_t epoch{0};
        glm::ivec3 worldMin{0};
        int blockScale{1};
        int lodLevel{0};
        std::vector<StructureRegionKey> structureRegionKeys;
        std::shared_ptr<std::array<PackedFarLodVoxelGpu, kVoxelCount>> cpuPackedParityVoxels;
        std::shared_ptr<std::array<GpuTerrainColumnDescriptor, kLogicalSize * kLogicalSize>> cpuColumnParityDescriptors;
        bool parityRequested{false};
    };

    struct PendingGpuParityReadback
    {
        FarLodChunkKey key{};
        std::uint32_t buildVersion{0};
        std::uint64_t epoch{0};
        UINT64 computeFenceValue{0};
        UINT64 copyFenceValue{0};
        std::shared_ptr<std::array<PackedFarLodVoxelGpu, kVoxelCount>> cpuPackedParityVoxels;
        std::shared_ptr<std::array<GpuTerrainColumnDescriptor, kLogicalSize * kLogicalSize>> cpuColumnParityDescriptors;
        bool parityRequested{false};
        bool copySubmitted{false};
    };

    struct BuildJob
    {
        FarLodChunkKey key{};
        FarLodLevelConfig level{};
        std::uint32_t buildVersion{0};
        std::uint64_t epoch{0};
        int ringDistanceChunks{0};
        bool hadResidentMesh{false};
        bool fallbackOnly{false};
    };

    struct DirtyBuildPlanRequest
    {
        std::uint64_t sequence{0};
        std::uint64_t epoch{0};
        glm::ivec3 cameraChunk{0};
        glm::vec3 cameraForward{0.0f, 0.0f, -1.0f};
        Frustum visibilityFrustum{};
        glm::vec3 visibilityCameraPos{0.0f};
        bool hasVisibilityFrustum{false};
        int exactMissingChunks{0};
        std::size_t exactPendingUploads{0};
        std::size_t workerBudget{1};
    };

    struct DirtyBuildPlanResult
    {
        std::uint64_t sequence{0};
        std::uint64_t epoch{0};
        std::vector<BuildJob> jobs;
    };

    struct HorizontalRingSpan
    {
        int chunkZ{0};
        int minChunkX{0};
        int maxChunkX{0};
    };

    struct TouchLevelPlanLevelRequest
    {
        FarLodLevelConfig level{};
        int activeOuterRadiusChunks{0};
    };

    struct TouchLevelPlanLevelResult
    {
        FarLodLevelConfig level{};
        int activeOuterRadiusChunks{0};
        std::vector<FarLodChunkKey> activeKeys;
    };

    struct TouchLevelPlanRequest
    {
        std::uint64_t sequence{0};
        std::uint64_t epoch{0};
        glm::ivec3 cameraChunk{0};
        std::size_t workerBudget{1};
        std::vector<TouchLevelPlanLevelRequest> levels;
    };

    struct TouchLevelPlanResult
    {
        std::uint64_t sequence{0};
        std::uint64_t epoch{0};
        std::vector<TouchLevelPlanLevelResult> levels;
    };

    struct TouchLevelCacheState
    {
        FarLodLevelConfig level{};
        int activeOuterRadiusChunks{0};
        glm::ivec3 cameraChunk{0};
        std::vector<FarLodChunkKey> activeKeys;
    };

    struct AtlasUpdateRect
    {
        glm::ivec2 originCell{0};
        glm::ivec2 sizeCells{0};
    };

    struct FarLodWorkBudget
    {
        int activationStepChunks{1};
        int distanceRampStepChunks{0};
        std::size_t newChunkActivations{0};
        std::size_t newFallbackActivations{0};
        std::size_t staleReleaseCount{0};
        std::size_t atlasUpdateCells{0};
        std::size_t gpuDispatchBudgetUnits{0};
        std::size_t maxGpuSubmissions{0};
    };

    struct FarLodLevelAtlasState
    {
        int level{0};
        int blockScale{1};
        glm::ivec2 atlasSizeCells{0};
        glm::ivec2 originCell{0};
        Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> sampleBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> seedHeaderBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> seedDataBuffer;
        D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES sampleState{D3D12_RESOURCE_STATE_COMMON};
        glm::ivec2 seedOriginChunk{0};
        glm::ivec2 seedSizeChunks{0};
        D3D12_RESOURCE_STATES seedHeaderState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES seedDataState{D3D12_RESOURCE_STATE_COMMON};
        bool initialized{false};
        bool seedInitialized{false};
        std::vector<AtlasUpdateRect> pendingDirtyRects;
        std::vector<AtlasUpdateRect> pendingSeedDirtyRects;
        std::unordered_map<std::uint64_t, std::uint64_t> cellRevisions;

        [[nodiscard]] std::uint32_t elementCount() const noexcept
        {
            return atlasSizeCells.x > 0 && atlasSizeCells.y > 0
                       ? static_cast<std::uint32_t>(atlasSizeCells.x * atlasSizeCells.y)
                       : 0u;
        }

        [[nodiscard]] std::uint32_t seedHeaderElementCount() const noexcept
        {
            return seedSizeChunks.x > 0 && seedSizeChunks.y > 0
                       ? static_cast<std::uint32_t>(seedSizeChunks.x * seedSizeChunks.y)
                       : 0u;
        }

        [[nodiscard]] std::uint32_t seedDataElementCount() const noexcept
        {
            return seedHeaderElementCount() * kFarLodChunkSeedCountPerCacheEntry;
        }
    };

    struct VoxelFootprintClassification
    {
        FarLodVoxel voxel{};
        int minSampledSurfaceY{std::numeric_limits<int>::max()};
        int maxSampledSurfaceY{std::numeric_limits<int>::min()};
    };

    struct GreedyMaskCell
    {
        bool visible{false};
        BlockId material{BlockId::Air};
        std::uint8_t flags{0};

        [[nodiscard]] bool mergeEquals(const GreedyMaskCell& other) const noexcept
        {
            return visible == other.visible && material == other.material && flags == other.flags;
        }
    };

    static glm::vec2 projectTileCoord(const glm::vec3& position, const glm::vec3& normal) noexcept
    {
        const glm::vec3 absNormal = glm::abs(normal);
        if (absNormal.y >= absNormal.x && absNormal.y >= absNormal.z)
        {
            return glm::vec2(position.x, position.z);
        }
        if (absNormal.x >= absNormal.z)
        {
            return glm::vec2(position.z, position.y);
        }
        return glm::vec2(position.x, position.y);
    }

    static void appendQuad(std::vector<Vertex>& vertices,
                           std::vector<std::uint32_t>& indices,
                           const glm::vec3& p0,
                           const glm::vec3& p1,
                           const glm::vec3& p2,
                           const glm::vec3& p3,
                           const glm::vec3& normal,
                           const std::pair<glm::vec2, glm::vec2>& uv,
                           std::uint8_t flags = 0)
    {
        const std::uint32_t baseIndex = static_cast<std::uint32_t>(vertices.size());
        const std::uint32_t lightingData = packVertexLighting(packLightLevels(kMaxLightLevel, 0), 0, flags);
        vertices.push_back(Vertex{p0, normal, projectTileCoord(p0, normal), uv.first, uv.second, lightingData});
        vertices.push_back(Vertex{p1, normal, projectTileCoord(p1, normal), uv.first, uv.second, lightingData});
        vertices.push_back(Vertex{p2, normal, projectTileCoord(p2, normal), uv.first, uv.second, lightingData});
        vertices.push_back(Vertex{p3, normal, projectTileCoord(p3, normal), uv.first, uv.second, lightingData});

        indices.push_back(baseIndex + 0);
        indices.push_back(baseIndex + 1);
        indices.push_back(baseIndex + 2);
        indices.push_back(baseIndex + 0);
        indices.push_back(baseIndex + 2);
        indices.push_back(baseIndex + 3);
    }

    static std::size_t nextPowerOfTwo(std::size_t value) noexcept
    {
        if (value <= 1)
        {
            return 1;
        }

        --value;
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

    [[nodiscard]] static std::uint32_t roundReservedFaceCapacity(std::uint32_t faceCount) noexcept
    {
        constexpr std::uint32_t kMinFaceCapacity = 96u;
        faceCount = std::clamp(faceCount, kMinFaceCapacity, kFaceDescriptorCount);
        std::uint32_t rounded = 1u;
        while (rounded < faceCount && rounded < kFaceDescriptorCount)
        {
            rounded <<= 1u;
        }
        return std::min<std::uint32_t>(rounded, kFaceDescriptorCount);
    }

    [[nodiscard]] static std::uint32_t initialReservedFaceCapacity(const FarLodLevelConfig& level) noexcept
    {
        if (level.blockScale <= 4)
        {
            return 1024u;
        }
        if (level.blockScale <= 8)
        {
            return 768u;
        }
        if (level.blockScale <= 16)
        {
            return 512u;
        }
        if (level.blockScale <= 32)
        {
            return 320u;
        }
        if (level.blockScale <= 64)
        {
            return 192u;
        }
        return 128u;
    }

    [[nodiscard]] static std::uint32_t growReservedFaceCapacity(std::uint32_t requiredFaces) noexcept
    {
        const std::uint32_t paddedFaces =
            requiredFaces + std::max<std::uint32_t>(requiredFaces / 2u, 32u);
        return roundReservedFaceCapacity(paddedFaces);
    }

    [[nodiscard]] static std::size_t voxelIndex(int localX, int localY, int localZ) noexcept
    {
        return (static_cast<std::size_t>(localY) * kLogicalSize + static_cast<std::size_t>(localZ)) *
                   kLogicalSize +
               static_cast<std::size_t>(localX);
    }

    [[nodiscard]] static std::uint64_t computeSeamSignature(const FarLodChunkCpu& cpu) noexcept
    {
        constexpr std::uint64_t kOffsetBasis = 1469598103934665603ull;
        constexpr std::uint64_t kPrime = 1099511628211ull;

        auto hashVoxel = [](std::uint64_t hash, const FarLodVoxel& voxel, std::uint8_t faceId) noexcept
        {
            const std::uint32_t packed =
                static_cast<std::uint32_t>(voxel.occupied) |
                (static_cast<std::uint32_t>(voxel.flags) << 8u) |
                (static_cast<std::uint32_t>(toIndex(voxel.material)) << 16u) |
                (static_cast<std::uint32_t>(faceId) << 24u);
            hash ^= static_cast<std::uint64_t>(packed);
            hash *= kPrime;
            return hash;
        };

        std::uint64_t hash = kOffsetBasis;
        for (int localY = 0; localY < kLogicalSize; ++localY)
        {
            for (int localZ = 0; localZ < kLogicalSize; ++localZ)
            {
                hash = hashVoxel(hash, cpu.voxels[voxelIndex(0, localY, localZ)], 0u);
                hash = hashVoxel(hash, cpu.voxels[voxelIndex(kLogicalSize - 1, localY, localZ)], 1u);
            }
        }
        for (int localY = 0; localY < kLogicalSize; ++localY)
        {
            for (int localX = 0; localX < kLogicalSize; ++localX)
            {
                hash = hashVoxel(hash, cpu.voxels[voxelIndex(localX, localY, 0)], 2u);
                hash = hashVoxel(hash, cpu.voxels[voxelIndex(localX, localY, kLogicalSize - 1)], 3u);
            }
        }
        for (int localZ = 0; localZ < kLogicalSize; ++localZ)
        {
            for (int localX = 0; localX < kLogicalSize; ++localX)
            {
                hash = hashVoxel(hash, cpu.voxels[voxelIndex(localX, 0, localZ)], 4u);
                hash = hashVoxel(hash, cpu.voxels[voxelIndex(localX, kLogicalSize - 1, localZ)], 5u);
            }
        }
        return hash;
    }

    [[nodiscard]] static std::uint64_t packAtlasCellKey(int cellX, int cellZ) noexcept
    {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cellX)) << 32u) |
               static_cast<std::uint32_t>(cellZ);
    }

    [[nodiscard]] static std::uint64_t staggeredChunkUpdateHash(const FarLodChunkKey& key) noexcept
    {
        std::uint64_t hash = 1469598103934665603ull;
        const auto mix = [&hash](std::uint64_t value) noexcept
        {
            hash ^= value;
            hash *= 1099511628211ull;
        };
        mix(static_cast<std::uint32_t>(key.level));
        mix(static_cast<std::uint32_t>(key.coord.x));
        mix(static_cast<std::uint32_t>(key.coord.y));
        mix(static_cast<std::uint32_t>(key.coord.z));
        return hash;
    }

    [[nodiscard]] static int atlasOriginSnapStrideChunks(const FarLodLevelConfig& level) noexcept
    {
        return level.level >= 3 ? 2 : 1;
    }

    [[nodiscard]] static std::uint64_t residentRefreshCadenceUpdates(const FarLodLevelConfig& level) noexcept
    {
        if (level.level >= 5)
        {
            return 4u;
        }
        if (level.level >= 4)
        {
            return 3u;
        }
        if (level.level >= 3)
        {
            return 2u;
        }
        return 1u;
    }

    [[nodiscard]] static PackedFarLodVoxelGpu packGpuVoxel(const FarLodVoxel& voxel) noexcept
    {
        PackedFarLodVoxelGpu packed = 0u;
        if (voxel.occupied)
        {
            packed |= 0x1u;
        }
        if ((voxel.flags & 0x01u) != 0u)
        {
            packed |= 0x2u;
        }
        if ((voxel.flags & 0x02u) != 0u)
        {
            packed |= 0x4u;
        }
        if ((voxel.flags & 0x04u) != 0u)
        {
            packed |= 0x8u;
        }
        if ((voxel.flags & 0x08u) != 0u)
        {
            packed |= 0x10u;
        }
        packed |= (static_cast<PackedFarLodVoxelGpu>(toIndex(voxel.material) & 0xffu) << 8u);
        return packed;
    }

    [[nodiscard]] static FarLodVoxel unpackGpuVoxel(PackedFarLodVoxelGpu packed) noexcept
    {
        FarLodVoxel voxel{};
        voxel.occupied = (packed & 0x1u) ? 1u : 0u;
        voxel.material = static_cast<BlockId>((packed >> 8u) & 0xffu);
        if ((packed & 0x2u) != 0u)
        {
            voxel.flags |= 0x01u;
        }
        if ((packed & 0x4u) != 0u)
        {
            voxel.flags |= 0x02u;
        }
        if ((packed & 0x8u) != 0u)
        {
            voxel.flags |= 0x04u;
        }
        if ((packed & 0x10u) != 0u)
        {
            voxel.flags |= 0x08u;
        }
        return voxel;
    }

    [[nodiscard]] static bool intersectsRange(int minA, int maxA, int minB, int maxB) noexcept
    {
        return minA <= maxB && minB <= maxA;
    }

    [[nodiscard]] static BlockFace blockFaceForNormal(const glm::ivec3& normal) noexcept
    {
        if (normal.x > 0) return BlockFace::East;
        if (normal.x < 0) return BlockFace::West;
        if (normal.y > 0) return BlockFace::Top;
        if (normal.y < 0) return BlockFace::Bottom;
        if (normal.z > 0) return BlockFace::South;
        return BlockFace::North;
    }

    [[nodiscard]] static int structureMaterialPriority(BlockId block) noexcept
    {
        if (block == BlockId::SpruceLog || block == BlockId::Wood || block == BlockId::DarkOakLog)
        {
            return 4;
        }
        if (block == BlockId::SpruceLeaves || block == BlockId::Leaves || block == BlockId::DarkOakLeaves)
        {
            return 3;
        }
        if (block == BlockId::Water)
        {
            return 1;
        }
        if (block != BlockId::Air)
        {
            return 2;
        }
        return 0;
    }

    [[nodiscard]] static int chunkMinHorizontalRingDistanceChunks(const FarLodLevelConfig& level,
                                                                  const glm::ivec3& cameraChunk,
                                                                  const glm::ivec3& chunkCoord) noexcept
    {
        const glm::ivec2 minCoord(chunkCoord.x * level.blockScale, chunkCoord.z * level.blockScale);
        const glm::ivec2 maxCoord = minCoord + glm::ivec2(level.blockScale - 1);
        const auto minAxisDistance = [](int center, int minValue, int maxValue) noexcept
        {
            if (center < minValue) return minValue - center;
            if (center > maxValue) return center - maxValue;
            return 0;
        };

        const int minDistance = std::max(minAxisDistance(cameraChunk.x, minCoord.x, maxCoord.x),
                                         minAxisDistance(cameraChunk.z, minCoord.y, maxCoord.y));
        return std::max(0, minDistance - level.innerRadiusChunks);
    }

    [[nodiscard]] static int ceilDivPositive(int value, int divisor) noexcept
    {
        return -floorDiv(-value, divisor);
    }

    [[nodiscard]] static std::vector<HorizontalRingSpan> buildHorizontalRingSpans(const FarLodLevelConfig& level,
                                                                                   const glm::ivec3& cameraChunk,
                                                                                   int outerRadiusChunks)
    {
        std::vector<HorizontalRingSpan> spans;
        if (outerRadiusChunks <= level.innerRadiusChunks || level.blockScale <= 0)
        {
            return spans;
        }

        const int blockScale = level.blockScale;
        const int outerMinChunkX = floorDiv(cameraChunk.x - outerRadiusChunks, blockScale);
        const int outerMaxChunkX = floorDiv(cameraChunk.x + outerRadiusChunks, blockScale);
        const int outerMinChunkZ = floorDiv(cameraChunk.z - outerRadiusChunks, blockScale);
        const int outerMaxChunkZ = floorDiv(cameraChunk.z + outerRadiusChunks, blockScale);

        spans.reserve(static_cast<std::size_t>(std::max(outerMaxChunkZ - outerMinChunkZ + 1, 0) * 2));

        const auto minAxisDistance = [](int center, int minValue, int maxValue) noexcept
        {
            if (center < minValue) return minValue - center;
            if (center > maxValue) return center - maxValue;
            return 0;
        };
        const auto maxAxisDistance = [](int center, int minValue, int maxValue) noexcept
        {
            return std::max(std::abs(center - minValue), std::abs(center - maxValue));
        };

        for (int chunkZ = outerMinChunkZ; chunkZ <= outerMaxChunkZ; ++chunkZ)
        {
            const int zMin = chunkZ * blockScale;
            const int zMax = zMin + blockScale - 1;
            const int zMinDistance = minAxisDistance(cameraChunk.z, zMin, zMax);
            if (zMinDistance > outerRadiusChunks)
            {
                continue;
            }

            const int zMaxDistance = maxAxisDistance(cameraChunk.z, zMin, zMax);
            if (zMaxDistance > level.innerRadiusChunks)
            {
                spans.push_back(HorizontalRingSpan{chunkZ, outerMinChunkX, outerMaxChunkX});
                continue;
            }

            const int fullyInsideMinChunkX =
                ceilDivPositive(cameraChunk.x - level.innerRadiusChunks, blockScale);
            const int fullyInsideMaxChunkX =
                floorDiv(cameraChunk.x + level.innerRadiusChunks - (blockScale - 1), blockScale);
            if (fullyInsideMinChunkX > fullyInsideMaxChunkX)
            {
                spans.push_back(HorizontalRingSpan{chunkZ, outerMinChunkX, outerMaxChunkX});
                continue;
            }

            if (outerMinChunkX < fullyInsideMinChunkX)
            {
                spans.push_back(HorizontalRingSpan{
                    chunkZ,
                    outerMinChunkX,
                    std::min(outerMaxChunkX, fullyInsideMinChunkX - 1)});
            }
            if (fullyInsideMaxChunkX < outerMaxChunkX)
            {
                spans.push_back(HorizontalRingSpan{
                    chunkZ,
                    std::max(outerMinChunkX, fullyInsideMaxChunkX + 1),
                    outerMaxChunkX});
            }
        }

        return spans;
    }

    [[nodiscard]] std::vector<FarLodChunkKey> buildTouchLevelActiveKeys(const TouchLevelPlanLevelRequest& levelRequest,
                                                                        const glm::ivec3& cameraChunk,
                                                                        std::size_t workerBudget) const
    {
        const std::vector<HorizontalRingSpan> spans =
            buildHorizontalRingSpans(levelRequest.level, cameraChunk, levelRequest.activeOuterRadiusChunks);
        if (spans.empty())
        {
            return {};
        }

        std::size_t totalKeyCount = 0;
        for (const HorizontalRingSpan& span : spans)
        {
            totalKeyCount += static_cast<std::size_t>(std::max(span.maxChunkX - span.minChunkX + 1, 0));
        }

        const std::size_t threadCount =
            std::min<std::size_t>(std::max<std::size_t>(workerBudget, 1), spans.size());
        std::vector<std::vector<FarLodChunkKey>> threadKeys(threadCount);
        std::vector<std::thread> workers;
        workers.reserve(threadCount > 0 ? threadCount - 1 : 0);

        auto buildRange = [&](std::size_t threadIndex, std::size_t beginSpan, std::size_t endSpan)
        {
            std::size_t localCount = 0;
            for (std::size_t spanIndex = beginSpan; spanIndex < endSpan; ++spanIndex)
            {
                localCount += static_cast<std::size_t>(
                    std::max(spans[spanIndex].maxChunkX - spans[spanIndex].minChunkX + 1, 0));
            }

            std::vector<FarLodChunkKey>& out = threadKeys[threadIndex];
            out.reserve(localCount);
            for (std::size_t spanIndex = beginSpan; spanIndex < endSpan; ++spanIndex)
            {
                const HorizontalRingSpan& span = spans[spanIndex];
                for (int chunkX = span.minChunkX; chunkX <= span.maxChunkX; ++chunkX)
                {
                    out.push_back(FarLodChunkKey{
                        levelRequest.level.level,
                        glm::ivec3(chunkX, 0, span.chunkZ)});
                }
            }
        };

        const std::size_t spansPerThread = (spans.size() + threadCount - 1) / threadCount;
        for (std::size_t threadIndex = 1; threadIndex < threadCount; ++threadIndex)
        {
            const std::size_t beginSpan = threadIndex * spansPerThread;
            const std::size_t endSpan = std::min(spans.size(), beginSpan + spansPerThread);
            workers.emplace_back(buildRange, threadIndex, beginSpan, endSpan);
        }

        buildRange(0u, 0u, std::min(spans.size(), spansPerThread));
        for (std::thread& worker : workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        std::vector<FarLodChunkKey> keys;
        keys.reserve(totalKeyCount);
        for (std::vector<FarLodChunkKey>& local : threadKeys)
        {
            keys.insert(keys.end(),
                        std::make_move_iterator(local.begin()),
                        std::make_move_iterator(local.end()));
        }
        return keys;
    }

    [[nodiscard]] TouchLevelPlanResult buildTouchLevelPlan(const TouchLevelPlanRequest& request) const
    {
        TouchLevelPlanResult result{};
        result.sequence = request.sequence;
        result.epoch = request.epoch;
        result.levels.reserve(request.levels.size());
        for (const TouchLevelPlanLevelRequest& levelRequest : request.levels)
        {
            TouchLevelPlanLevelResult levelResult{};
            levelResult.level = levelRequest.level;
            levelResult.activeOuterRadiusChunks = levelRequest.activeOuterRadiusChunks;
            levelResult.activeKeys = buildTouchLevelActiveKeys(levelRequest, request.cameraChunk, request.workerBudget);
            result.levels.push_back(std::move(levelResult));
        }
        return result;
    }

    [[nodiscard]] const FarLodLevelConfig* nextCoarserLevel(int levelId) const noexcept
    {
        for (std::size_t i = 0; i < levels_.size(); ++i)
        {
            if (levels_[i].level == levelId)
            {
                return (i + 1 < levels_.size()) ? &levels_[i + 1] : nullptr;
            }
        }
        return nullptr;
    }

    void updateChunkConservativeBounds(FarLodChunkRecord& chunk,
                                       const FarLodLevelConfig& level,
                                       int verticalBandMinWorldY,
                                       int verticalBandMaxWorldY) const
    {
        constexpr int kFarStructureHeadroomBlocks = 48;
        const int conservativeMinWorldY =
            std::min(verticalBandMinWorldY - level.blockScale,
                     seaLevel_ - (level.chunkSpanBlocks() + level.blockScale));
        const int conservativeMaxWorldY =
            std::max(verticalBandMaxWorldY + level.blockScale + kFarStructureHeadroomBlocks,
                     seaLevel_ + level.blockScale);
        chunk.cpu.boundsMin = glm::vec3(chunk.cpu.worldMin.x,
                                        static_cast<float>(conservativeMinWorldY),
                                        chunk.cpu.worldMin.z);
        chunk.cpu.boundsMax = glm::vec3(chunk.cpu.worldMin.x + level.chunkSpanBlocks(),
                                        static_cast<float>(conservativeMaxWorldY + 1),
                                        chunk.cpu.worldMin.z + level.chunkSpanBlocks());
    }

    void requestTouchLevelPlan(const glm::ivec3& cameraChunk)
    {
        std::vector<std::pair<int, int>> levelState;
        levelState.reserve(levels_.size());
        TouchLevelPlanRequest request{};
        request.sequence = touchLevelPlanSequence_.fetch_add(1, std::memory_order_acq_rel) + 1u;
        request.epoch = buildEpoch_.load(std::memory_order_acquire);
        request.cameraChunk = cameraChunk;
        request.workerBudget = std::max<std::size_t>(workerCount_, 1);
        request.levels.reserve(levels_.size());

        for (const FarLodLevelConfig& level : levels_)
        {
            if (level.outerRadiusChunks <= level.innerRadiusChunks)
            {
                continue;
            }
            const auto it = levelActivationOuterRadiusChunks_.find(level.level);
            const int activeOuterRadiusChunks =
                (it != levelActivationOuterRadiusChunks_.end()) ? it->second : level.innerRadiusChunks;
            levelState.emplace_back(level.level, activeOuterRadiusChunks);
            request.levels.push_back(TouchLevelPlanLevelRequest{level, activeOuterRadiusChunks});
        }

        if (hasLastTouchLevelPlanRequest_ &&
            lastTouchLevelPlanCameraChunk_.x == cameraChunk.x &&
            lastTouchLevelPlanCameraChunk_.z == cameraChunk.z &&
            lastTouchLevelPlanLevels_ == levelState)
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(touchLevelPlanMutex_);
            pendingTouchLevelPlanRequest_ = std::move(request);
            hasPendingTouchLevelPlanRequest_ = true;
        }
        touchLevelPlanCv_.notify_one();
        lastTouchLevelPlanCameraChunk_ = cameraChunk;
        lastTouchLevelPlanLevels_ = std::move(levelState);
        hasLastTouchLevelPlanRequest_ = true;
    }

    void consumeReadyTouchLevelPlan()
    {
        TouchLevelPlanResult result{};
        {
            std::lock_guard<std::mutex> lock(touchLevelPlanMutex_);
            if (!hasReadyTouchLevelPlanResult_)
            {
                return;
            }
            result = std::move(readyTouchLevelPlanResult_);
            readyTouchLevelPlanResult_ = {};
            hasReadyTouchLevelPlanResult_ = false;
        }

        if (result.epoch != buildEpoch_.load(std::memory_order_acquire))
        {
            return;
        }

        for (TouchLevelPlanLevelResult& levelResult : result.levels)
        {
            TouchLevelCacheState& cache = touchLevelCaches_[levelResult.level.level];
            cache.level = levelResult.level;
            cache.activeOuterRadiusChunks = levelResult.activeOuterRadiusChunks;
            cache.cameraChunk = lastTouchLevelPlanCameraChunk_;
            cache.activeKeys = std::move(levelResult.activeKeys);
        }
    }

    void applyTouchLevelCache(const FarLodLevelConfig& level,
                              std::size_t& remainingNewChunkActivations,
                              std::size_t& remainingNewFallbackActivations)
    {
        const auto cacheIt = touchLevelCaches_.find(level.level);
        if (cacheIt == touchLevelCaches_.end())
        {
            return;
        }

        constexpr int kFarVerticalHeadroomChunks = 6;
        const int verticalRadiusChunks =
            std::max(gActiveVerticalRadius.load(std::memory_order_relaxed) + kFarVerticalHeadroomChunks, 8);
        const int chunkMinY = floorDiv(cameraChunk_.y - verticalRadiusChunks, level.blockScale);
        const int chunkMaxY = floorDiv(cameraChunk_.y + verticalRadiusChunks, level.blockScale);
        const int span = level.chunkSpanBlocks();
        const int verticalBandMinWorldY = chunkMinY * span;
        const int verticalBandMaxWorldY = ((chunkMaxY + 1) * span) - 1;

        std::vector<FarLodChunkKey> fallbackParents;
        fallbackParents.reserve(cacheIt->second.activeKeys.size() / 2 + 1);
        std::unordered_set<FarLodChunkKey, FarLodChunkKeyHasher> uniqueFallbackParents;

        for (const FarLodChunkKey& key : cacheIt->second.activeKeys)
        {
            auto chunkIt = chunks_.find(key);
            const bool chunkMissing = (chunkIt == chunks_.end());
            if (chunkMissing && remainingNewChunkActivations == 0)
            {
                continue;
            }

            FarLodChunkRecord& chunk = chunkMissing ? chunks_[key] : chunkIt->second;
            if (chunkMissing)
            {
                --remainingNewChunkActivations;
            }

            chunk.key = key;
            chunk.level = level;
            chunk.lastTouchedStamp = updateStamp_;
            chunk.active = true;
            chunk.fallbackOnly = false;
            if (!chunk.initialized)
            {
                initializeChunk(chunk, level);
                chunk.dirty = true;
            }
            updateChunkConservativeBounds(chunk, level, verticalBandMinWorldY, verticalBandMaxWorldY);
            if (!(chunk.gpu.resident && chunk.gpu.indexCount > 0) && !chunk.dirty && !chunk.inFlight)
            {
                markDirty(chunk);
            }

            if (chunk.gpu.resident && chunk.gpu.indexCount > 0)
            {
                continue;
            }

            const FarLodLevelConfig* parentLevel = nextCoarserLevel(level.level);
            if (parentLevel == nullptr)
            {
                continue;
            }

            const glm::ivec3 baseMin = key.coord * level.blockScale;
            const glm::ivec3 parentCoord{
                floorDiv(baseMin.x, parentLevel->blockScale),
                floorDiv(baseMin.y, parentLevel->blockScale),
                floorDiv(baseMin.z, parentLevel->blockScale)};
            FarLodChunkKey parentKey{parentLevel->level, parentCoord};
            if (uniqueFallbackParents.insert(parentKey).second)
            {
                fallbackParents.push_back(parentKey);
            }
        }

        for (const FarLodChunkKey& parentKey : fallbackParents)
        {
            auto parentIt = chunks_.find(parentKey);
            const bool parentMissing = (parentIt == chunks_.end());
            if (parentMissing && remainingNewFallbackActivations == 0)
            {
                continue;
            }

            const FarLodLevelConfig* parentLevel = nextCoarserLevel(level.level);
            if (parentLevel == nullptr)
            {
                continue;
            }

            FarLodChunkRecord& parent = parentMissing ? chunks_[parentKey] : parentIt->second;
            if (parentMissing)
            {
                --remainingNewFallbackActivations;
            }

            const bool alreadyTouchedThisUpdate = (parent.lastTouchedStamp == updateStamp_);
            parent.key = parentKey;
            parent.level = *parentLevel;
            parent.lastTouchedStamp = updateStamp_;
            parent.active = true;
            if (!(alreadyTouchedThisUpdate && !parent.fallbackOnly))
            {
                parent.fallbackOnly = true;
            }
            if (!parent.initialized)
            {
                initializeChunk(parent, *parentLevel);
                parent.dirty = true;
            }
            else if (!(parent.gpu.resident && parent.gpu.indexCount > 0) && !parent.dirty)
            {
                markDirty(parent);
            }
        }
    }

    void initializeChunk(FarLodChunkRecord& chunk, const FarLodLevelConfig& level)
    {
        chunk.level = level;
        chunk.cpu.key = chunk.key;
        chunk.cpu.blockScale = level.blockScale;
        chunk.cpu.worldMin = chunk.key.coord * level.chunkSpanBlocks();
        chunk.cpu.boundsMin = glm::vec3(chunk.cpu.worldMin);
        chunk.cpu.boundsMax = glm::vec3(chunk.cpu.worldMin + glm::ivec3(level.chunkSpanBlocks()));
        chunk.residentBoundsMin = chunk.cpu.boundsMin;
        chunk.residentBoundsMax = chunk.cpu.boundsMax;
        chunk.pendingMesh = {};
        chunk.fallbackOnly = false;
        chunk.lastBuiltAtlasDependencyRevision = 0;
        chunk.gpu.columnBuffer = createDefaultBuffer(device_.Get(),
                                                     static_cast<std::uint64_t>(kLogicalSize * kLogicalSize *
                                                                                sizeof(GpuTerrainColumnDescriptor)),
                                                     D3D12_RESOURCE_STATE_COMMON,
                                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        chunk.gpu.faceCountBuffer = createDefaultBuffer(device_.Get(),
                                                        static_cast<std::uint64_t>(kPlaneCount * sizeof(std::uint32_t)),
                                                        D3D12_RESOURCE_STATE_COMMON,
                                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        chunk.gpu.faceAnalysisBuffer = createDefaultBuffer(device_.Get(),
                                                           static_cast<std::uint64_t>(kFaceMetadataEntryCount *
                                                                                      sizeof(std::uint32_t)),
                                                           D3D12_RESOURCE_STATE_COMMON,
                                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        chunk.gpu.faceDescriptorBuffer = createDefaultBuffer(device_.Get(),
                                                             static_cast<std::uint64_t>(kFaceDescriptorCount * sizeof(std::uint32_t) * 4u),
                                                             D3D12_RESOURCE_STATE_COMMON,
                                                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        chunk.gpu.facePrefixBuffer = createDefaultBuffer(device_.Get(),
                                                         static_cast<std::uint64_t>(kPlaneCount * sizeof(std::uint32_t)),
                                                         D3D12_RESOURCE_STATE_COMMON,
                                                         D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        chunk.gpu.faceGroupSumBuffer = createDefaultBuffer(device_.Get(),
                                                           static_cast<std::uint64_t>(kFacePrefixGroupCount * sizeof(std::uint32_t)),
                                                           D3D12_RESOURCE_STATE_COMMON,
                                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (chunk.gpu.columnBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodColumns_L" << chunk.key.level
                 << L"_" << chunk.key.coord.x
                 << L"_" << chunk.key.coord.y
                 << L"_" << chunk.key.coord.z;
            setDebugObjectName(chunk.gpu.columnBuffer.Get(), name.str());
        }
        if (chunk.gpu.faceCountBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodFaceCount_L" << chunk.key.level
                 << L"_" << chunk.key.coord.x
                 << L"_" << chunk.key.coord.y
                 << L"_" << chunk.key.coord.z;
            setDebugObjectName(chunk.gpu.faceCountBuffer.Get(), name.str());
        }
        if (chunk.gpu.faceAnalysisBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodFaceAnalysis_L" << chunk.key.level
                 << L"_" << chunk.key.coord.x
                 << L"_" << chunk.key.coord.y
                 << L"_" << chunk.key.coord.z;
            setDebugObjectName(chunk.gpu.faceAnalysisBuffer.Get(), name.str());
        }
        if (chunk.gpu.faceDescriptorBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodFaceDescriptor_L" << chunk.key.level
                 << L"_" << chunk.key.coord.x
                 << L"_" << chunk.key.coord.y
                 << L"_" << chunk.key.coord.z;
            setDebugObjectName(chunk.gpu.faceDescriptorBuffer.Get(), name.str());
        }
        if (chunk.gpu.facePrefixBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodFacePrefix_L" << chunk.key.level
                 << L"_" << chunk.key.coord.x
                 << L"_" << chunk.key.coord.y
                 << L"_" << chunk.key.coord.z;
            setDebugObjectName(chunk.gpu.facePrefixBuffer.Get(), name.str());
        }
        if (chunk.gpu.faceGroupSumBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodFaceGroupSum_L" << chunk.key.level
                 << L"_" << chunk.key.coord.x
                 << L"_" << chunk.key.coord.y
                 << L"_" << chunk.key.coord.z;
            setDebugObjectName(chunk.gpu.faceGroupSumBuffer.Get(), name.str());
        }
        chunk.gpu.columnState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceCountState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceAnalysisState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceDescriptorState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.facePrefixState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceGroupSumState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.voxelFenceValue = 0;
        chunk.gpu.readbackFenceValue = 0;
        chunk.gpu.voxelReady = false;
        chunk.gpu.parityMismatchCount = 0;
        chunk.gpu.parityValidated = false;
        chunk.gpu.reservedVertexCount = 0;
        chunk.gpu.reservedIndexCount = 0;
        chunk.gpu.faceCapacity = 0;
        chunk.gpu.recordIndex = kInvalidFarDrawRecordIndex;
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD chunk initialized level=" << chunk.key.level
                   << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                   << " blockScale=" << level.blockScale
                   << " span=" << level.chunkSpanBlocks();
            chunkManagerDebugLog(stream.str());
        }
        chunk.initialized = true;
    }

    void markDirty(FarLodChunkRecord& chunk)
    {
        chunk.dirty = true;
        ++chunk.buildVersion;
        if (chunk.buildVersion == 0)
        {
            chunk.buildVersion = 1;
        }
    }

    [[nodiscard]] static std::uint32_t committedFaceCount(const FarLodChunkGpuState& gpu) noexcept
    {
        return (gpu.indexCount > 0u) ? (gpu.indexCount / 6u) : 0u;
    }

    [[nodiscard]] static std::uint32_t requestedFaceCapacity(const FarLodChunkRecord& chunk) noexcept
    {
        std::uint32_t requested = initialReservedFaceCapacity(chunk.level);
        requested = std::max(requested, chunk.gpu.faceCapacity);
        requested = std::max(requested, chunk.requestedFaceCapacityHint);
        const std::uint32_t actualFaces = committedFaceCount(chunk.gpu);
        if (actualFaces > 0u)
        {
            requested = std::max(requested, growReservedFaceCapacity(actualFaces));
        }
        return std::min<std::uint32_t>(roundReservedFaceCapacity(requested), kFaceDescriptorCount);
    }

    void markSameLevelNeighborSeamsDirty(const FarLodChunkKey& key)
    {
        static constexpr std::array<glm::ivec3, 6> kNeighborOffsets{{
            glm::ivec3( 1,  0,  0),
            glm::ivec3(-1,  0,  0),
            glm::ivec3( 0,  1,  0),
            glm::ivec3( 0, -1,  0),
            glm::ivec3( 0,  0,  1),
            glm::ivec3( 0,  0, -1),
        }};

        for (const glm::ivec3& offset : kNeighborOffsets)
        {
            const FarLodChunkKey neighborKey{key.level, key.coord + offset};
            const auto it = chunks_.find(neighborKey);
            if (it == chunks_.end())
            {
                continue;
            }

            FarLodChunkRecord& neighbor = it->second;
            if (!neighbor.initialized || !neighbor.active)
            {
                continue;
            }

            if (!neighbor.dirty)
            {
                markDirty(neighbor);
            }
        }
    }

    BufferPage createBufferPage(std::size_t vertexCount, std::size_t indexCount)
    {
        static constexpr std::size_t kDefaultVertexCapacity = 131072;
        static constexpr std::size_t kDefaultIndexCapacity = 196608;

        BufferPage page;
        page.vertexCapacity = std::max(nextPowerOfTwo(vertexCount), kDefaultVertexCapacity);
        page.indexCapacity = std::max(nextPowerOfTwo(indexCount), kDefaultIndexCapacity);
        page.recordCapacity = std::max<std::size_t>(
            1,
            std::min(page.vertexCapacity / 4u, page.indexCapacity / 6u));
        page.vertexBuffer = createDefaultBuffer(device_.Get(),
                                                static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex)),
                                                D3D12_RESOURCE_STATE_COMMON,
                                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        page.indexBuffer = createDefaultBuffer(device_.Get(),
                                               static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t)),
                                               D3D12_RESOURCE_STATE_COMMON,
                                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        page.vertexUploadBuffer = createUploadBuffer(device_.Get(),
                                                     static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex)),
                                                     page.mappedVertexData);
        page.indexUploadBuffer = createUploadBuffer(device_.Get(),
                                                    static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t)),
                                                    page.mappedIndexData);
        page.drawRecordBuffer = createDefaultBuffer(device_.Get(),
                                                    static_cast<std::uint64_t>(page.recordCapacity * sizeof(ChunkRenderBatch::GpuCullRecord)),
                                                    D3D12_RESOURCE_STATE_COMMON,
                                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        page.vertexView.BufferLocation = page.vertexBuffer ? page.vertexBuffer->GetGPUVirtualAddress() : 0;
        page.vertexView.StrideInBytes = sizeof(Vertex);
        page.vertexView.SizeInBytes = static_cast<UINT>(page.vertexCapacity * sizeof(Vertex));
        page.indexView.BufferLocation = page.indexBuffer ? page.indexBuffer->GetGPUVirtualAddress() : 0;
        page.indexView.SizeInBytes = static_cast<UINT>(page.indexCapacity * sizeof(std::uint32_t));
        page.indexView.Format = DXGI_FORMAT_R32_UINT;
        return page;
    }

    static bool tryAllocateRange(std::vector<BufferPage::Range>& ranges,
                                 std::size_t& cursor,
                                 std::size_t capacity,
                                 std::size_t count,
                                 std::size_t& outOffset)
    {
        if (count == 0)
        {
            outOffset = cursor;
            return true;
        }

        for (auto it = ranges.begin(); it != ranges.end(); ++it)
        {
            if (it->size >= count)
            {
                outOffset = it->offset;
                it->offset += count;
                it->size -= count;
                if (it->size == 0)
                {
                    ranges.erase(it);
                }
                return true;
            }
        }

        if (cursor + count <= capacity)
        {
            outOffset = cursor;
            cursor += count;
            return true;
        }

        return false;
    }

    static void mergeRange(std::vector<BufferPage::Range>& ranges, std::size_t offset, std::size_t size)
    {
        if (size == 0)
        {
            return;
        }

        BufferPage::Range range{offset, size};
        auto it = std::lower_bound(ranges.begin(), ranges.end(), range.offset,
                                   [](const BufferPage::Range& lhs, std::size_t value)
                                   {
                                       return lhs.offset < value;
                                   });
        it = ranges.insert(it, range);

        if (it != ranges.begin())
        {
            auto prev = std::prev(it);
            if (prev->offset + prev->size == it->offset)
            {
                prev->size += it->size;
                it = ranges.erase(it);
                it = prev;
            }
        }

        auto next = std::next(it);
        if (next != ranges.end() && it->offset + it->size == next->offset)
        {
            it->size += next->size;
            ranges.erase(next);
        }
    }

    Allocation acquireAllocation(std::size_t vertexCount, std::size_t indexCount)
    {
        Allocation allocation{};
        if (vertexCount == 0 || indexCount == 0)
        {
            return allocation;
        }

        for (std::uint32_t pageIndex = 0; pageIndex < bufferPages_.size(); ++pageIndex)
        {
            BufferPage& page = bufferPages_[pageIndex];
            std::size_t vertexOffset = 0;
            if (!tryAllocateRange(page.freeVertices, page.vertexCursor, page.vertexCapacity, vertexCount, vertexOffset))
            {
                continue;
            }

            std::size_t indexOffset = 0;
            if (!tryAllocateRange(page.freeIndices, page.indexCursor, page.indexCapacity, indexCount, indexOffset))
            {
                mergeRange(page.freeVertices, vertexOffset, vertexCount);
                continue;
            }

            std::size_t recordOffset = 0;
            if (!tryAllocateRange(page.freeRecords, page.recordCursor, page.recordCapacity, 1, recordOffset))
            {
                mergeRange(page.freeVertices, vertexOffset, vertexCount);
                mergeRange(page.freeIndices, indexOffset, indexCount);
                continue;
            }

            allocation.pageIndex = pageIndex;
            allocation.vertexOffset = vertexOffset;
            allocation.indexOffset = indexOffset;
            allocation.recordIndex = static_cast<std::uint32_t>(recordOffset);
            return allocation;
        }

        BufferPage newPage = createBufferPage(vertexCount, indexCount);
        bufferPages_.push_back(std::move(newPage));
        BufferPage& page = bufferPages_.back();
        allocation.pageIndex = static_cast<std::uint32_t>(bufferPages_.size() - 1);
        tryAllocateRange(page.freeVertices, page.vertexCursor, page.vertexCapacity, vertexCount, allocation.vertexOffset);
        tryAllocateRange(page.freeIndices, page.indexCursor, page.indexCapacity, indexCount, allocation.indexOffset);
        std::size_t recordOffset = 0;
        tryAllocateRange(page.freeRecords, page.recordCursor, page.recordCapacity, 1, recordOffset);
        allocation.recordIndex = static_cast<std::uint32_t>(recordOffset);
        return allocation;
    }

    void releaseAllocationRange(std::uint32_t pageIndex,
                                std::size_t vertexOffset,
                                std::size_t vertexCount,
                                std::size_t indexOffset,
                                std::uint32_t indexCount,
                                std::uint32_t recordIndex) noexcept
    {
        if (pageIndex == kInvalidChunkBufferPage)
        {
            return;
        }
        if (pageIndex >= bufferPages_.size())
        {
            return;
        }
        BufferPage& page = bufferPages_[pageIndex];
        mergeRange(page.freeVertices, vertexOffset, vertexCount);
        mergeRange(page.freeIndices, indexOffset, static_cast<std::size_t>(indexCount));
        if (recordIndex != kInvalidFarDrawRecordIndex)
        {
            mergeRange(page.freeRecords, recordIndex, 1);
        }
    }

    void clearDrawRecord(BufferPage& page, std::uint32_t recordIndex)
    {
        if (recordIndex == kInvalidFarDrawRecordIndex || page.drawRecordBuffer == nullptr || device_ == nullptr)
        {
            return;
        }
        if (!uploadContext_.ready())
        {
            return;
        }

        std::byte* mapped = nullptr;
        Microsoft::WRL::ComPtr<ID3D12Resource> upload = createUploadBuffer(
            device_.Get(),
            static_cast<std::uint64_t>(sizeof(ChunkRenderBatch::GpuCullRecord)),
            mapped);
        if (upload == nullptr || mapped == nullptr)
        {
            return;
        }
        std::memset(mapped, 0, sizeof(ChunkRenderBatch::GpuCullRecord));

        if (!uploadContext_.begin())
        {
            upload->Unmap(0, nullptr);
            return;
        }

        uploadContext_.copyBuffer(page.drawRecordBuffer.Get(),
                                  static_cast<std::uint64_t>(recordIndex * sizeof(ChunkRenderBatch::GpuCullRecord)),
                                  upload.Get(),
                                  0,
                                  static_cast<std::uint64_t>(sizeof(ChunkRenderBatch::GpuCullRecord)));
        uploadContext_.flush(nullptr);
        uploadContext_.waitForIdle();
        upload->Unmap(0, nullptr);
    }

    static void releasePendingDrawRecordReadback(FarLodChunkRecord::PendingRenderMesh& pendingMesh) noexcept
    {
        if (pendingMesh.drawRecordReadbackBuffer != nullptr && pendingMesh.mappedDrawRecord != nullptr)
        {
            pendingMesh.drawRecordReadbackBuffer->Unmap(0, nullptr);
        }
        pendingMesh.drawRecordReadbackBuffer.Reset();
        pendingMesh.mappedDrawRecord = nullptr;
    }

    void releasePendingMeshAllocation(FarLodChunkRecord& chunk)
    {
        if (!chunk.pendingMesh.valid())
        {
            return;
        }

        if (chunk.pendingMesh.uploadFenceValue != 0)
        {
            const UINT64 completedUploadFenceValue = uploadContext_.completedFenceValue();
            if (completedUploadFenceValue < chunk.pendingMesh.uploadFenceValue)
            {
                uploadContext_.waitForIdle();
            }
        }
        if (chunk.pendingMesh.gpuGenerated && chunk.pendingMesh.gpuFenceValue != 0)
        {
            const UINT64 completedGpuFenceValue = gpuContext_.completedFenceValue();
            if (completedGpuFenceValue < chunk.pendingMesh.gpuFenceValue)
            {
                gpuContext_.waitForFence(chunk.pendingMesh.gpuFenceValue);
            }
        }

        if (chunk.pendingMesh.pageIndex < bufferPages_.size() && chunk.pendingMesh.recordIndex != kInvalidFarDrawRecordIndex)
        {
            BufferPage& page = bufferPages_[chunk.pendingMesh.pageIndex];
            clearDrawRecord(page, chunk.pendingMesh.recordIndex);
        }

        releaseAllocationRange(chunk.pendingMesh.pageIndex,
                               chunk.pendingMesh.vertexOffset,
                               chunk.pendingMesh.vertexCount,
                               chunk.pendingMesh.indexOffset,
                               chunk.pendingMesh.indexCount,
                               chunk.pendingMesh.recordIndex);
        releasePendingDrawRecordReadback(chunk.pendingMesh);
        chunk.pendingMesh = {};
    }

    void releaseChunkRenderAllocation(FarLodChunkRecord& chunk)
    {
        if (chunk.gpu.pageIndex == kInvalidChunkBufferPage)
        {
            chunk.gpu.vertexOffset = 0;
            chunk.gpu.indexOffset = 0;
            chunk.gpu.vertexCount = 0;
            chunk.gpu.indexCount = 0;
            chunk.gpu.reservedVertexCount = 0;
            chunk.gpu.reservedIndexCount = 0;
            chunk.gpu.resident = false;
            return;
        }

        const UINT64 completedUploadFenceValue = uploadContext_.completedFenceValue();
        const UINT64 submittedUploadFenceValue = uploadContext_.lastSubmittedFenceValue();
        if (submittedUploadFenceValue != 0 && completedUploadFenceValue < submittedUploadFenceValue)
        {
            uploadContext_.waitForIdle();
        }

        if (chunk.gpu.pageIndex < bufferPages_.size())
        {
            BufferPage& page = bufferPages_[chunk.gpu.pageIndex];
            mergeRange(page.freeVertices, chunk.gpu.vertexOffset, chunk.gpu.reservedVertexCount);
            mergeRange(page.freeIndices, chunk.gpu.indexOffset, chunk.gpu.reservedIndexCount);
            if (chunk.gpu.recordIndex != kInvalidFarDrawRecordIndex)
            {
                clearDrawRecord(page, chunk.gpu.recordIndex);
                mergeRange(page.freeRecords, chunk.gpu.recordIndex, 1);
            }
        }

        chunk.gpu.pageIndex = kInvalidChunkBufferPage;
        chunk.gpu.vertexOffset = 0;
        chunk.gpu.indexOffset = 0;
        chunk.gpu.vertexCount = 0;
        chunk.gpu.indexCount = 0;
        chunk.gpu.reservedVertexCount = 0;
        chunk.gpu.reservedIndexCount = 0;
        chunk.gpu.faceCapacity = 0;
        chunk.gpu.recordIndex = kInvalidFarDrawRecordIndex;
        chunk.gpu.resident = false;
    }

    void releaseChunkGpu(FarLodChunkRecord& chunk)
    {
        releasePendingMeshAllocation(chunk);
        if ((chunk.gpu.voxelBuffer != nullptr || chunk.gpu.columnBuffer != nullptr) && chunk.gpu.voxelFenceValue != 0)
        {
            const UINT64 completedFenceValue = gpuContext_.completedFenceValue();
            if (completedFenceValue < chunk.gpu.voxelFenceValue)
            {
                gpuContext_.waitForFence(chunk.gpu.voxelFenceValue);
            }
        }
        if (chunk.gpu.readbackFenceValue != 0)
        {
            const UINT64 completedUploadFenceValue = uploadContext_.completedFenceValue();
            if (completedUploadFenceValue < chunk.gpu.readbackFenceValue)
            {
                uploadContext_.waitForIdle();
            }
        }
        chunk.gpu.columnBuffer.Reset();
        chunk.gpu.columnState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.voxelBuffer.Reset();
        chunk.gpu.voxelState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceCountBuffer.Reset();
        chunk.gpu.faceAnalysisBuffer.Reset();
        chunk.gpu.faceDescriptorBuffer.Reset();
        chunk.gpu.facePrefixBuffer.Reset();
        chunk.gpu.faceGroupSumBuffer.Reset();
        chunk.gpu.faceCountState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceAnalysisState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceDescriptorState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.facePrefixState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceGroupSumState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.voxelFenceValue = 0;
        chunk.gpu.readbackFenceValue = 0;
        chunk.gpu.voxelReady = false;
        chunk.gpu.parityMismatchCount = 0;
        chunk.gpu.parityValidated = false;
        releaseChunkRenderAllocation(chunk);
        chunk.requestedFaceCapacityHint = 0;
    }

    [[nodiscard]] glm::ivec2 computeAtlasSizeCells(const FarLodLevelConfig& level) const noexcept
    {
        const auto it = levelActivationOuterRadiusChunks_.find(level.level);
        const int activeOuterRadiusChunks =
            (it != levelActivationOuterRadiusChunks_.end()) ? it->second : level.innerRadiusChunks;
        const int minChunk = floorDiv(-activeOuterRadiusChunks, level.blockScale);
        const int maxChunk = floorDiv(activeOuterRadiusChunks, level.blockScale);
        const int chunkCount = (maxChunk - minChunk + 1) + 2;
        const int cells = std::max(chunkCount * kLogicalSize, kLogicalSize * 4);
        return glm::ivec2(cells, cells);
    }

    [[nodiscard]] glm::ivec2 computeDesiredAtlasOriginCell(const FarLodLevelConfig& level) const noexcept
    {
        const auto it = levelActivationOuterRadiusChunks_.find(level.level);
        const int activeOuterRadiusChunks =
            (it != levelActivationOuterRadiusChunks_.end()) ? it->second : level.innerRadiusChunks;
        int chunkMinX = floorDiv(cameraChunk_.x - activeOuterRadiusChunks, level.blockScale) - 1;
        int chunkMinZ = floorDiv(cameraChunk_.z - activeOuterRadiusChunks, level.blockScale) - 1;
        const int snapStrideChunks = atlasOriginSnapStrideChunks(level);
        if (snapStrideChunks > 1)
        {
            chunkMinX = floorDiv(chunkMinX, snapStrideChunks) * snapStrideChunks;
            chunkMinZ = floorDiv(chunkMinZ, snapStrideChunks) * snapStrideChunks;
        }
        return glm::ivec2(chunkMinX * kLogicalSize, chunkMinZ * kLogicalSize);
    }

    [[nodiscard]] glm::ivec2 computeSeedCacheSizeChunks(const FarLodLevelConfig& level) const noexcept
    {
        const terrain::FarLodGpuWorldgenHeader& header = worldgenTables_.header;
        if (header.chunkSpan <= 0)
        {
            return glm::ivec2(0, 0);
        }

        const glm::ivec2 atlasSizeCells = computeAtlasSizeCells(level);
        const int coveredChunksX = ((atlasSizeCells.x * level.blockScale) + header.chunkSpan - 1) / header.chunkSpan + 1;
        const int coveredChunksZ = ((atlasSizeCells.y * level.blockScale) + header.chunkSpan - 1) / header.chunkSpan + 1;
        return glm::ivec2(coveredChunksX + header.neighborRadius * 2,
                          coveredChunksZ + header.neighborRadius * 2);
    }

    [[nodiscard]] glm::ivec2 computeDesiredSeedOriginChunk(const FarLodLevelConfig& level,
                                                           const glm::ivec2& atlasOriginCell) const noexcept
    {
        const terrain::FarLodGpuWorldgenHeader& header = worldgenTables_.header;
        if (header.chunkSpan <= 0)
        {
            return glm::ivec2(0, 0);
        }

        const int minWorldX = atlasOriginCell.x * level.blockScale;
        const int minWorldZ = atlasOriginCell.y * level.blockScale;
        return glm::ivec2(floorDiv(minWorldX, header.chunkSpan) - header.neighborRadius,
                          floorDiv(minWorldZ, header.chunkSpan) - header.neighborRadius);
    }

    [[nodiscard]] FarLodWorkBudget computeWorkBudget() const
    {
        std::size_t buildBacklog = 0;
        {
            std::lock_guard<std::mutex> lock(buildQueueMutex_);
            buildBacklog = buildQueue_.size() + queuedKeys_.size();
        }

        std::size_t gpuBacklog = 0;
        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            gpuBacklog = gpuSynthesisRequests_.size() +
                         pendingGpuParityReadbacks_.size();
        }

        const std::size_t totalBacklog = buildBacklog + gpuBacklog;
        const int exactMissing = exactMissingChunks_.load(std::memory_order_relaxed);
        const std::size_t exactPendingUploads = exactPendingUploads_.load(std::memory_order_relaxed);
        const std::size_t workerBudget = std::max<std::size_t>(workerCount_, 1);

        FarLodWorkBudget budget{};
        if (totalBacklog > 512 || exactMissing > 48 || exactPendingUploads > 48)
        {
            budget.activationStepChunks = 1;
            budget.distanceRampStepChunks = 0;
            budget.newChunkActivations = std::max<std::size_t>(workerBudget * 2, 4);
            budget.newFallbackActivations = 2;
            budget.staleReleaseCount = 4;
            budget.atlasUpdateCells = 8u * 1024u;
            budget.gpuDispatchBudgetUnits = 8;
            budget.maxGpuSubmissions = 2;
        }
        else if (totalBacklog > 256 || exactMissing > 24 || exactPendingUploads > 24)
        {
            budget.activationStepChunks = 2;
            budget.distanceRampStepChunks = 1;
            budget.newChunkActivations = std::max<std::size_t>(workerBudget * 4, 8);
            budget.newFallbackActivations = 4;
            budget.staleReleaseCount = 8;
            budget.atlasUpdateCells = 16u * 1024u;
            budget.gpuDispatchBudgetUnits = 12;
            budget.maxGpuSubmissions = 3;
        }
        else if (totalBacklog > 96 || exactMissing > 8 || exactPendingUploads > 8)
        {
            budget.activationStepChunks = 4;
            budget.distanceRampStepChunks = 2;
            budget.newChunkActivations = std::max<std::size_t>(workerBudget * 8, 16);
            budget.newFallbackActivations = 6;
            budget.staleReleaseCount = 16;
            budget.atlasUpdateCells = 48u * 1024u;
            budget.gpuDispatchBudgetUnits = 24;
            budget.maxGpuSubmissions = 5;
        }
        else
        {
            budget.activationStepChunks = 6;
            budget.distanceRampStepChunks = 4;
            budget.newChunkActivations = std::max<std::size_t>(workerBudget * 12, 32);
            budget.newFallbackActivations = 8;
            budget.staleReleaseCount = 24;
            budget.atlasUpdateCells = 160u * 1024u;
            budget.gpuDispatchBudgetUnits = 36;
            budget.maxGpuSubmissions = 8;
        }
        return budget;
    }

    void ensureAtlasState(const FarLodLevelConfig& level)
    {
        if (device_ == nullptr)
        {
            return;
        }

        FarLodLevelAtlasState& atlas = levelAtlases_[level.level];
        const glm::ivec2 requiredSize = computeAtlasSizeCells(level);
        const glm::ivec2 requiredSeedSize = computeSeedCacheSizeChunks(level);
        if (atlas.buffer != nullptr &&
            atlas.sampleBuffer != nullptr &&
            atlas.seedHeaderBuffer != nullptr &&
            atlas.seedDataBuffer != nullptr &&
            atlas.blockScale == level.blockScale &&
            atlas.atlasSizeCells == requiredSize &&
            atlas.seedSizeChunks == requiredSeedSize)
        {
            return;
        }

        atlas = FarLodLevelAtlasState{};
        atlas.level = level.level;
        atlas.blockScale = level.blockScale;
        atlas.atlasSizeCells = requiredSize;
        atlas.seedSizeChunks = requiredSeedSize;
        const std::uint64_t bufferBytes =
            static_cast<std::uint64_t>(requiredSize.x) * static_cast<std::uint64_t>(requiredSize.y) *
            sizeof(GpuTerrainAtlasSample);
        const std::uint64_t sampleBufferBytes =
            static_cast<std::uint64_t>(requiredSize.x) * static_cast<std::uint64_t>(requiredSize.y) *
            sizeof(GpuAtlasSampleCacheEntry);
        const std::uint64_t seedHeaderBufferBytes =
            static_cast<std::uint64_t>(requiredSeedSize.x) * static_cast<std::uint64_t>(requiredSeedSize.y) *
            sizeof(GpuChunkSeedCacheHeader);
        const std::uint64_t seedDataBufferBytes =
            static_cast<std::uint64_t>(requiredSeedSize.x) * static_cast<std::uint64_t>(requiredSeedSize.y) *
            kFarLodChunkSeedCountPerCacheEntry * sizeof(GpuChunkSeedCacheSeed);
        atlas.buffer = createDefaultBuffer(device_.Get(),
                                           bufferBytes,
                                           D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        atlas.sampleBuffer = createDefaultBuffer(device_.Get(),
                                                 sampleBufferBytes,
                                                 D3D12_RESOURCE_STATE_COMMON,
                                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        atlas.seedHeaderBuffer = createDefaultBuffer(device_.Get(),
                                                     seedHeaderBufferBytes,
                                                     D3D12_RESOURCE_STATE_COMMON,
                                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        atlas.seedDataBuffer = createDefaultBuffer(device_.Get(),
                                                   seedDataBufferBytes,
                                                   D3D12_RESOURCE_STATE_COMMON,
                                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (atlas.buffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodAtlas_L" << level.level;
            setDebugObjectName(atlas.buffer.Get(), name.str());
        }
        if (atlas.sampleBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodAtlasSamples_L" << level.level;
            setDebugObjectName(atlas.sampleBuffer.Get(), name.str());
        }
        if (atlas.seedHeaderBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodChunkSeedHeaders_L" << level.level;
            setDebugObjectName(atlas.seedHeaderBuffer.Get(), name.str());
        }
        if (atlas.seedDataBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodChunkSeedData_L" << level.level;
            setDebugObjectName(atlas.seedDataBuffer.Get(), name.str());
        }
        atlas.state = D3D12_RESOURCE_STATE_COMMON;
        atlas.sampleState = D3D12_RESOURCE_STATE_COMMON;
        atlas.seedHeaderState = D3D12_RESOURCE_STATE_COMMON;
        atlas.seedDataState = D3D12_RESOURCE_STATE_COMMON;
        atlas.originCell = computeDesiredAtlasOriginCell(level);
        atlas.seedOriginChunk = computeDesiredSeedOriginChunk(level, atlas.originCell);
        atlas.initialized = false;
        atlas.seedInitialized = false;
        atlas.pendingDirtyRects.clear();
        atlas.pendingSeedDirtyRects.clear();
    }

    [[nodiscard]] std::vector<AtlasUpdateRect> prepareAtlasUpdateRects(FarLodLevelAtlasState& atlas,
                                                                       const FarLodLevelConfig& level)
    {
        std::vector<AtlasUpdateRect> rects;
        const glm::ivec2 desiredOriginCell = computeDesiredAtlasOriginCell(level);
        if (!atlas.initialized)
        {
            atlas.originCell = desiredOriginCell;
            atlas.initialized = true;
            rects.push_back(AtlasUpdateRect{atlas.originCell, atlas.atlasSizeCells});
            return rects;
        }

        const glm::ivec2 delta = desiredOriginCell - atlas.originCell;
        if (delta.x == 0 && delta.y == 0)
        {
            return rects;
        }

        if (std::abs(delta.x) >= atlas.atlasSizeCells.x || std::abs(delta.y) >= atlas.atlasSizeCells.y)
        {
            atlas.originCell = desiredOriginCell;
            rects.push_back(AtlasUpdateRect{atlas.originCell, atlas.atlasSizeCells});
            return rects;
        }

        atlas.originCell = desiredOriginCell;
        const int absDeltaX = std::abs(delta.x);
        if (delta.x > 0)
        {
            rects.push_back(AtlasUpdateRect{
                {atlas.originCell.x + atlas.atlasSizeCells.x - delta.x, atlas.originCell.y},
                {delta.x, atlas.atlasSizeCells.y}});
        }
        else if (delta.x < 0)
        {
            rects.push_back(AtlasUpdateRect{
                {atlas.originCell.x, atlas.originCell.y},
                {-delta.x, atlas.atlasSizeCells.y}});
        }

        if (delta.y > 0)
        {
            const int originX = atlas.originCell.x + (delta.x < 0 ? absDeltaX : 0);
            const int width = atlas.atlasSizeCells.x - absDeltaX;
            if (width > 0)
            {
            rects.push_back(AtlasUpdateRect{
                    {originX, atlas.originCell.y + atlas.atlasSizeCells.y - delta.y},
                    {width, delta.y}});
            }
        }
        else if (delta.y < 0)
        {
            const int originX = atlas.originCell.x + (delta.x < 0 ? absDeltaX : 0);
            const int width = atlas.atlasSizeCells.x - absDeltaX;
            if (width > 0)
            {
            rects.push_back(AtlasUpdateRect{
                    {originX, atlas.originCell.y},
                    {width, -delta.y}});
            }
        }
        return rects;
    }

    [[nodiscard]] std::vector<AtlasUpdateRect> prepareSeedCacheUpdateRects(FarLodLevelAtlasState& atlas,
                                                                           const FarLodLevelConfig& level)
    {
        std::vector<AtlasUpdateRect> rects;
        const glm::ivec2 desiredOriginChunk = computeDesiredSeedOriginChunk(level, atlas.originCell);
        if (!atlas.seedInitialized)
        {
            atlas.seedOriginChunk = desiredOriginChunk;
            atlas.seedInitialized = true;
            rects.push_back(AtlasUpdateRect{atlas.seedOriginChunk, atlas.seedSizeChunks});
            return rects;
        }

        const glm::ivec2 delta = desiredOriginChunk - atlas.seedOriginChunk;
        if (delta.x == 0 && delta.y == 0)
        {
            return rects;
        }

        if (std::abs(delta.x) >= atlas.seedSizeChunks.x || std::abs(delta.y) >= atlas.seedSizeChunks.y)
        {
            atlas.seedOriginChunk = desiredOriginChunk;
            rects.push_back(AtlasUpdateRect{atlas.seedOriginChunk, atlas.seedSizeChunks});
            return rects;
        }

        atlas.seedOriginChunk = desiredOriginChunk;
        const int absDeltaX = std::abs(delta.x);
        if (delta.x > 0)
        {
            rects.push_back(AtlasUpdateRect{
                {atlas.seedOriginChunk.x + atlas.seedSizeChunks.x - delta.x, atlas.seedOriginChunk.y},
                {delta.x, atlas.seedSizeChunks.y}});
        }
        else if (delta.x < 0)
        {
            rects.push_back(AtlasUpdateRect{
                {atlas.seedOriginChunk.x, atlas.seedOriginChunk.y},
                {-delta.x, atlas.seedSizeChunks.y}});
        }

        if (delta.y > 0)
        {
            const int originX = atlas.seedOriginChunk.x + (delta.x < 0 ? absDeltaX : 0);
            const int width = atlas.seedSizeChunks.x - absDeltaX;
            if (width > 0)
            {
                rects.push_back(AtlasUpdateRect{
                    {originX, atlas.seedOriginChunk.y + atlas.seedSizeChunks.y - delta.y},
                    {width, delta.y}});
            }
        }
        else if (delta.y < 0)
        {
            const int originX = atlas.seedOriginChunk.x + (delta.x < 0 ? absDeltaX : 0);
            const int width = atlas.seedSizeChunks.x - absDeltaX;
            if (width > 0)
            {
                rects.push_back(AtlasUpdateRect{
                    {originX, atlas.seedOriginChunk.y},
                    {width, -delta.y}});
            }
        }
        return rects;
    }

    [[nodiscard]] static bool atlasRectValid(const AtlasUpdateRect& rect) noexcept
    {
        return rect.sizeCells.x > 0 && rect.sizeCells.y > 0;
    }

    [[nodiscard]] static std::optional<AtlasUpdateRect> clipAtlasUpdateRect(const AtlasUpdateRect& rect,
                                                                            const glm::ivec2& originCell,
                                                                            const glm::ivec2& atlasSizeCells) noexcept
    {
        const int minX = std::max(rect.originCell.x, originCell.x);
        const int minZ = std::max(rect.originCell.y, originCell.y);
        const int maxX = std::min(rect.originCell.x + rect.sizeCells.x, originCell.x + atlasSizeCells.x);
        const int maxZ = std::min(rect.originCell.y + rect.sizeCells.y, originCell.y + atlasSizeCells.y);
        if (maxX <= minX || maxZ <= minZ)
        {
            return std::nullopt;
        }
        return AtlasUpdateRect{{minX, minZ}, {maxX - minX, maxZ - minZ}};
    }

    [[nodiscard]] static std::optional<AtlasUpdateRect> intersectAtlasUpdateRect(const AtlasUpdateRect& a,
                                                                                 const AtlasUpdateRect& b) noexcept
    {
        const int minX = std::max(a.originCell.x, b.originCell.x);
        const int minZ = std::max(a.originCell.y, b.originCell.y);
        const int maxX = std::min(a.originCell.x + a.sizeCells.x, b.originCell.x + b.sizeCells.x);
        const int maxZ = std::min(a.originCell.y + a.sizeCells.y, b.originCell.y + b.sizeCells.y);
        if (maxX <= minX || maxZ <= minZ)
        {
            return std::nullopt;
        }
        return AtlasUpdateRect{{minX, minZ}, {maxX - minX, maxZ - minZ}};
    }

    static void subtractAtlasUpdateRect(const AtlasUpdateRect& rect,
                                        const AtlasUpdateRect& cut,
                                        std::vector<AtlasUpdateRect>& leftovers)
    {
        const std::optional<AtlasUpdateRect> intersection = intersectAtlasUpdateRect(rect, cut);
        if (!intersection.has_value())
        {
            leftovers.push_back(rect);
            return;
        }

        const AtlasUpdateRect& overlap = *intersection;
        const int rectMinX = rect.originCell.x;
        const int rectMinZ = rect.originCell.y;
        const int rectMaxX = rect.originCell.x + rect.sizeCells.x;
        const int rectMaxZ = rect.originCell.y + rect.sizeCells.y;
        const int cutMinX = overlap.originCell.x;
        const int cutMinZ = overlap.originCell.y;
        const int cutMaxX = overlap.originCell.x + overlap.sizeCells.x;
        const int cutMaxZ = overlap.originCell.y + overlap.sizeCells.y;

        if (cutMinZ > rectMinZ)
        {
            leftovers.push_back(AtlasUpdateRect{{rectMinX, rectMinZ}, {rect.sizeCells.x, cutMinZ - rectMinZ}});
        }
        if (cutMaxZ < rectMaxZ)
        {
            leftovers.push_back(AtlasUpdateRect{{rectMinX, cutMaxZ}, {rect.sizeCells.x, rectMaxZ - cutMaxZ}});
        }
        if (cutMinX > rectMinX)
        {
            leftovers.push_back(AtlasUpdateRect{{rectMinX, cutMinZ}, {cutMinX - rectMinX, overlap.sizeCells.y}});
        }
        if (cutMaxX < rectMaxX)
        {
            leftovers.push_back(AtlasUpdateRect{{cutMaxX, cutMinZ}, {rectMaxX - cutMaxX, overlap.sizeCells.y}});
        }
    }

    static void mergeAtlasUpdateRect(std::vector<AtlasUpdateRect>& rects, const AtlasUpdateRect& rect)
    {
        if (!atlasRectValid(rect))
        {
            return;
        }

        AtlasUpdateRect merged = rect;
        bool mergedAny = true;
        while (mergedAny)
        {
            mergedAny = false;
            for (auto it = rects.begin(); it != rects.end();)
            {
                const AtlasUpdateRect& existing = *it;
                const bool overlapOrTouch =
                    merged.originCell.x <= existing.originCell.x + existing.sizeCells.x &&
                    merged.originCell.x + merged.sizeCells.x >= existing.originCell.x &&
                    merged.originCell.y <= existing.originCell.y + existing.sizeCells.y &&
                    merged.originCell.y + merged.sizeCells.y >= existing.originCell.y;
                if (!overlapOrTouch)
                {
                    ++it;
                    continue;
                }

                const int minX = std::min(merged.originCell.x, existing.originCell.x);
                const int minZ = std::min(merged.originCell.y, existing.originCell.y);
                const int maxX = std::max(merged.originCell.x + merged.sizeCells.x,
                                          existing.originCell.x + existing.sizeCells.x);
                const int maxZ = std::max(merged.originCell.y + merged.sizeCells.y,
                                          existing.originCell.y + existing.sizeCells.y);
                merged.originCell = glm::ivec2(minX, minZ);
                merged.sizeCells = glm::ivec2(maxX - minX, maxZ - minZ);
                it = rects.erase(it);
                mergedAny = true;
            }
        }

        rects.push_back(merged);
    }

    static void appendClippedAtlasUpdateRect(std::vector<AtlasUpdateRect>& rects,
                                             const AtlasUpdateRect& rect,
                                             const glm::ivec2& originCell,
                                             const glm::ivec2& atlasSizeCells)
    {
        const std::optional<AtlasUpdateRect> clipped = clipAtlasUpdateRect(rect, originCell, atlasSizeCells);
        if (!clipped.has_value())
        {
            return;
        }
        mergeAtlasUpdateRect(rects, *clipped);
    }

    void markAtlasCellsUpdated(FarLodLevelAtlasState& atlas, const AtlasUpdateRect& rect)
    {
        if (!atlasRectValid(rect))
        {
            return;
        }

        const std::uint64_t revision = ++atlasRevisionCounter_;
        const int maxCellZ = rect.originCell.y + rect.sizeCells.y;
        const int maxCellX = rect.originCell.x + rect.sizeCells.x;
        for (int cellZ = rect.originCell.y; cellZ < maxCellZ; ++cellZ)
        {
            for (int cellX = rect.originCell.x; cellX < maxCellX; ++cellX)
            {
                atlas.cellRevisions[packAtlasCellKey(cellX, cellZ)] = revision;
            }
        }
    }

    [[nodiscard]] std::uint64_t currentAtlasDependencyRevision(const FarLodChunkRecord& chunk) const
    {
        const auto atlasIt = levelAtlases_.find(chunk.level.level);
        if (atlasIt == levelAtlases_.end())
        {
            return 0;
        }

        const FarLodLevelAtlasState& atlas = atlasIt->second;
        const int cellOriginX = (chunk.key.coord.x - 1) * kLogicalSize;
        const int cellOriginZ = (chunk.key.coord.z - 1) * kLogicalSize;
        const int cellMaxX = cellOriginX + (kLogicalSize * 3);
        const int cellMaxZ = cellOriginZ + (kLogicalSize * 3);
        std::uint64_t revision = 0;
        for (int cellZ = cellOriginZ; cellZ < cellMaxZ; ++cellZ)
        {
            for (int cellX = cellOriginX; cellX < cellMaxX; ++cellX)
            {
                const auto revisionIt = atlas.cellRevisions.find(packAtlasCellKey(cellX, cellZ));
                if (revisionIt != atlas.cellRevisions.end())
                {
                    revision = std::max(revision, revisionIt->second);
                }
            }
        }
        return revision;
    }

    [[nodiscard]] AtlasUpdateRect computeSeedRectForAtlasUpdate(const FarLodLevelConfig& level,
                                                                const AtlasUpdateRect& rect) const
    {
        const terrain::FarLodGpuWorldgenHeader& header = worldgenTables_.header;
        const int blockScale = std::max(level.blockScale, 1);
        const int worldMinX = rect.originCell.x * blockScale;
        const int worldMinZ = rect.originCell.y * blockScale;
        const int worldMaxX = (rect.originCell.x + rect.sizeCells.x - 1) * blockScale + (blockScale - 1);
        const int worldMaxZ = (rect.originCell.y + rect.sizeCells.y - 1) * blockScale + (blockScale - 1);
        const int chunkMinX = floorDiv(worldMinX, header.chunkSpan) - header.neighborRadius;
        const int chunkMinZ = floorDiv(worldMinZ, header.chunkSpan) - header.neighborRadius;
        const int chunkMaxX = floorDiv(worldMaxX, header.chunkSpan) + header.neighborRadius;
        const int chunkMaxZ = floorDiv(worldMaxZ, header.chunkSpan) + header.neighborRadius;
        return AtlasUpdateRect{
            {chunkMinX, chunkMinZ},
            {chunkMaxX - chunkMinX + 1, chunkMaxZ - chunkMinZ + 1}};
    }

    void appendAtlasUpdates(const std::deque<GpuSynthesisRequest>& requests, std::size_t atlasCellBudget)
    {
        atlasCellBudget = std::min(atlasCellBudget, kMaxFarLodAtlasUpdateCellsPerSubmission);

        std::unordered_set<int> requestedLevels;
        requestedLevels.reserve(requests.size());
        for (const GpuSynthesisRequest& request : requests)
        {
            requestedLevels.insert(request.key.level);
        }

        for (const FarLodLevelConfig& level : levels_)
        {
            if (level.outerRadiusChunks <= level.innerRadiusChunks)
            {
                continue;
            }
            if (!requestedLevels.contains(level.level))
            {
                continue;
            }

            ensureAtlasState(level);
            auto atlasIt = levelAtlases_.find(level.level);
            if (atlasIt == levelAtlases_.end())
            {
                continue;
            }

            FarLodLevelAtlasState& atlas = atlasIt->second;
            if (atlas.buffer == nullptr || atlas.sampleBuffer == nullptr ||
                atlas.seedHeaderBuffer == nullptr || atlas.seedDataBuffer == nullptr)
            {
                continue;
            }

            std::vector<AtlasUpdateRect> rects;
            for (const AtlasUpdateRect& rect : prepareAtlasUpdateRects(atlas, level))
            {
                appendClippedAtlasUpdateRect(rects, rect, atlas.originCell, atlas.atlasSizeCells);
            }
            for (const AtlasUpdateRect& rect : atlas.pendingDirtyRects)
            {
                appendClippedAtlasUpdateRect(rects, rect, atlas.originCell, atlas.atlasSizeCells);
            }
            atlas.pendingDirtyRects.clear();

            std::vector<AtlasUpdateRect> seedRects;
            for (const AtlasUpdateRect& rect : prepareSeedCacheUpdateRects(atlas, level))
            {
                appendClippedAtlasUpdateRect(seedRects, rect, atlas.seedOriginChunk, atlas.seedSizeChunks);
            }
            for (const AtlasUpdateRect& rect : atlas.pendingSeedDirtyRects)
            {
                appendClippedAtlasUpdateRect(seedRects, rect, atlas.seedOriginChunk, atlas.seedSizeChunks);
            }
            atlas.pendingSeedDirtyRects.clear();

            if (atlasCellBudget == 0 || rects.empty())
            {
                for (const AtlasUpdateRect& rect : rects)
                {
                    mergeAtlasUpdateRect(atlas.pendingDirtyRects, rect);
                }
                for (const AtlasUpdateRect& rect : seedRects)
                {
                    mergeAtlasUpdateRect(atlas.pendingSeedDirtyRects, rect);
                }
                continue;
            }

            std::vector<AtlasUpdateRect> uploadRects;
            for (const AtlasUpdateRect& rect : rects)
            {
                if (atlasCellBudget == 0)
                {
                    mergeAtlasUpdateRect(atlas.pendingDirtyRects, rect);
                    continue;
                }

                const std::size_t rectCellCount =
                    static_cast<std::size_t>(rect.sizeCells.x) * static_cast<std::size_t>(rect.sizeCells.y);
                int uploadHeight = rect.sizeCells.y;
                if (rectCellCount > atlasCellBudget)
                {
                    const std::size_t maxRows = std::max<std::size_t>(1, atlasCellBudget / static_cast<std::size_t>(rect.sizeCells.x));
                    uploadHeight = static_cast<int>(std::min<std::size_t>(maxRows, static_cast<std::size_t>(rect.sizeCells.y)));
                }

                const AtlasUpdateRect uploadRect{rect.originCell, glm::ivec2(rect.sizeCells.x, uploadHeight)};
                const int remainingHeight = rect.sizeCells.y - uploadHeight;
                if (remainingHeight > 0)
                {
                    mergeAtlasUpdateRect(atlas.pendingDirtyRects,
                                         AtlasUpdateRect{
                                             glm::ivec2(rect.originCell.x, rect.originCell.y + uploadHeight),
                                             glm::ivec2(rect.sizeCells.x, remainingHeight)});
                }

                uploadRects.push_back(uploadRect);
                const std::size_t uploadedCells =
                    static_cast<std::size_t>(uploadRect.sizeCells.x) * static_cast<std::size_t>(uploadRect.sizeCells.y);
                atlasCellBudget = (uploadedCells >= atlasCellBudget) ? 0 : (atlasCellBudget - uploadedCells);
            }

            if (uploadRects.empty())
            {
                for (const AtlasUpdateRect& rect : seedRects)
                {
                    mergeAtlasUpdateRect(atlas.pendingSeedDirtyRects, rect);
                }
                continue;
            }

            std::vector<AtlasUpdateRect> requiredSeedRects;
            for (const AtlasUpdateRect& uploadRect : uploadRects)
            {
                appendClippedAtlasUpdateRect(requiredSeedRects,
                                            computeSeedRectForAtlasUpdate(level, uploadRect),
                                            atlas.seedOriginChunk,
                                            atlas.seedSizeChunks);
            }

            std::vector<AtlasUpdateRect> seedUploadRects;
            for (const AtlasUpdateRect& seedRect : seedRects)
            {
                std::vector<AtlasUpdateRect> remainingParts{seedRect};
                for (const AtlasUpdateRect& requiredRect : requiredSeedRects)
                {
                    std::vector<AtlasUpdateRect> nextParts;
                    for (const AtlasUpdateRect& part : remainingParts)
                    {
                        const std::optional<AtlasUpdateRect> overlap = intersectAtlasUpdateRect(part, requiredRect);
                        if (!overlap.has_value())
                        {
                            nextParts.push_back(part);
                            continue;
                        }

                        mergeAtlasUpdateRect(seedUploadRects, *overlap);
                        subtractAtlasUpdateRect(part, *overlap, nextParts);
                    }
                    remainingParts = std::move(nextParts);
                    if (remainingParts.empty())
                    {
                        break;
                    }
                }

                for (const AtlasUpdateRect& remainingRect : remainingParts)
                {
                    mergeAtlasUpdateRect(atlas.pendingSeedDirtyRects, remainingRect);
                }
            }

            if (!seedUploadRects.empty())
            {
                if (chunkManagerDebugLoggingEnabled())
                {
                    std::ostringstream stream;
                    stream << "Far LOD seed dispatch: biome=" << worldgenTables_.header.biomeCount
                           << " biomeSel=" << worldgenTables_.header.biomeSelectionCount
                           << " oceanSel=" << worldgenTables_.header.oceanSelectionCount
                           << " subBiome=" << worldgenTables_.header.subBiomeCount
                           << " seedHeader=" << atlas.seedHeaderElementCount()
                           << " seedData=" << atlas.seedDataElementCount()
                           << " rects=" << seedUploadRects.size();
                    chunkManagerDebugLog(stream.str());
                }

                gpuContext_.transition(atlas.seedHeaderBuffer.Get(), atlas.seedHeaderState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                atlas.seedHeaderState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                gpuContext_.transition(atlas.seedDataBuffer.Get(), atlas.seedDataState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                atlas.seedDataState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                for (const AtlasUpdateRect& seedRect : seedUploadRects)
                {
                    gpuContext_.dispatchSeedCacheUpdate(seedRect.originCell,
                                                        seedRect.sizeCells,
                                                        atlas.seedOriginChunk,
                                                        atlas.seedSizeChunks,
                                                        worldgenHeaderBuffer_.Get(),
                                                        worldgenBiomeBuffer_.Get(),
                                                        worldgenTables_.header.biomeCount,
                                                        worldgenBiomeSelectionBuffer_.Get(),
                                                        worldgenTables_.header.biomeSelectionCount,
                                                        worldgenOceanSelectionBuffer_.Get(),
                                                        worldgenTables_.header.oceanSelectionCount,
                                                        worldgenSubBiomeBuffer_.Get(),
                                                        worldgenTables_.header.subBiomeCount,
                                                        atlas.seedHeaderBuffer.Get(),
                                                        atlas.seedHeaderElementCount(),
                                                        atlas.seedDataBuffer.Get(),
                                                        atlas.seedDataElementCount());
                }
                gpuContext_.uavBarrier(atlas.seedHeaderBuffer.Get());
                gpuContext_.uavBarrier(atlas.seedDataBuffer.Get());
                gpuContext_.transition(atlas.seedHeaderBuffer.Get(),
                                       atlas.seedHeaderState,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                atlas.seedHeaderState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                gpuContext_.transition(atlas.seedDataBuffer.Get(),
                                       atlas.seedDataState,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                atlas.seedDataState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            }

            gpuContext_.transition(atlas.sampleBuffer.Get(), atlas.sampleState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            atlas.sampleState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            for (const AtlasUpdateRect& uploadRect : uploadRects)
            {
                gpuContext_.dispatchAtlasSampleCacheUpdate(atlas.originCell,
                                                           atlas.atlasSizeCells,
                                                           uploadRect.originCell,
                                                           uploadRect.sizeCells,
                                                           level.blockScale,
                                                           atlas.seedOriginChunk,
                                                           atlas.seedSizeChunks,
                                                           worldgenHeaderBuffer_.Get(),
                                                           worldgenBiomeBuffer_.Get(),
                                                           worldgenTables_.header.biomeCount,
                                                           worldgenBiomeSelectionBuffer_.Get(),
                                                           worldgenTables_.header.biomeSelectionCount,
                                                           worldgenOceanSelectionBuffer_.Get(),
                                                           worldgenTables_.header.oceanSelectionCount,
                                                           worldgenSubBiomeBuffer_.Get(),
                                                           worldgenTables_.header.subBiomeCount,
                                                           worldgenPermutationBuffer_.Get(),
                                                           static_cast<std::uint32_t>(worldgenTables_.surfacePermutation.size()),
                                                           atlas.seedHeaderBuffer.Get(),
                                                           atlas.seedHeaderElementCount(),
                                                           atlas.seedDataBuffer.Get(),
                                                           atlas.seedDataElementCount(),
                                                           atlas.sampleBuffer.Get(),
                                                           atlas.elementCount());
            }
            gpuContext_.uavBarrier(atlas.sampleBuffer.Get());
            gpuContext_.transition(atlas.sampleBuffer.Get(),
                                   atlas.sampleState,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            atlas.sampleState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

            gpuContext_.transition(atlas.buffer.Get(), atlas.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            atlas.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            for (const AtlasUpdateRect& uploadRect : uploadRects)
            {
                gpuContext_.dispatchAtlasUpdate(atlas.originCell,
                                                atlas.atlasSizeCells,
                                                uploadRect.originCell,
                                                uploadRect.sizeCells,
                                                level.blockScale,
                                                worldgenTables_.header.seaLevel,
                                                worldgenHeaderBuffer_.Get(),
                                                worldgenBiomeBuffer_.Get(),
                                                worldgenTables_.header.biomeCount,
                                                worldgenPermutationBuffer_.Get(),
                                                static_cast<std::uint32_t>(worldgenTables_.surfacePermutation.size()),
                                                atlas.sampleBuffer.Get(),
                                                atlas.elementCount(),
                                                atlas.buffer.Get(),
                                                atlas.elementCount());
                markAtlasCellsUpdated(atlas, uploadRect);
            }
            gpuContext_.uavBarrier(atlas.buffer.Get());
            gpuContext_.transition(atlas.buffer.Get(),
                                   atlas.state,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            atlas.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
    }

    [[nodiscard]] static GpuStructureInstance packGpuStructureInstance(const StructureInstance& instance) noexcept
    {
        GpuStructureInstance gpuInstance{};
        gpuInstance.type = static_cast<std::uint32_t>(instance.type);
        gpuInstance.originX = instance.origin.x;
        gpuInstance.originY = instance.origin.y;
        gpuInstance.originZ = instance.origin.z;
        gpuInstance.boundsMinX = instance.bounds.min.x;
        gpuInstance.boundsMinY = instance.bounds.min.y;
        gpuInstance.boundsMinZ = instance.bounds.min.z;
        gpuInstance.boundsMaxX = instance.bounds.max.x;
        gpuInstance.boundsMaxY = instance.bounds.max.y;
        gpuInstance.boundsMaxZ = instance.bounds.max.z;
        gpuInstance.trunkHeight = static_cast<std::uint32_t>(std::max(instance.trunkHeight, 0));
        gpuInstance.bareTrunkHeight = static_cast<std::uint32_t>(std::max(instance.bareTrunkHeight, 0));
        gpuInstance.maxLodLevel = static_cast<std::uint32_t>(std::max(instance.maxLodLevel, 0));
        return gpuInstance;
    }

    [[nodiscard]] static std::vector<StructureRegionKey> computeStructureRegionKeysForTile(const glm::ivec3& worldMin,
                                                                                            int chunkSpanBlocks)
    {
        std::vector<StructureRegionKey> keys;
        if (chunkSpanBlocks <= 0)
        {
            return keys;
        }

        const int minRegionX = floorDiv(worldMin.x - kMaxStructureHorizontalRadius, kStructureRegionSize);
        const int maxRegionX = floorDiv(worldMin.x + chunkSpanBlocks - 1 + kMaxStructureHorizontalRadius, kStructureRegionSize);
        const int minRegionZ = floorDiv(worldMin.z - kMaxStructureHorizontalRadius, kStructureRegionSize);
        const int maxRegionZ = floorDiv(worldMin.z + chunkSpanBlocks - 1 + kMaxStructureHorizontalRadius, kStructureRegionSize);
        keys.reserve(static_cast<std::size_t>(std::max(maxRegionX - minRegionX + 1, 0) *
                                              std::max(maxRegionZ - minRegionZ + 1, 0)));
        for (int regionZ = minRegionZ; regionZ <= maxRegionZ; ++regionZ)
        {
            for (int regionX = minRegionX; regionX <= maxRegionX; ++regionX)
            {
                keys.push_back(StructureRegionKey{regionX, regionZ});
            }
        }
        return keys;
    }

    void destroyGpuStructureRegions()
    {
        for (auto& [key, region] : gpuStructureRegions_)
        {
            (void)key;
            region.instanceBuffer.Reset();
        }
        gpuStructureRegions_.clear();
    }

    void ensureGpuStructureRegions(const std::deque<GpuSynthesisRequest>& requests)
    {
        if (device_ == nullptr)
        {
            return;
        }

        StructureSampleColumnFn sampleColumnFn;
        StructureSurfaceBlockFn surfaceBlockFn;
        StructureDensityFn densityFn;
        {
            std::lock_guard<std::mutex> lock(structureRegionMutex_);
            sampleColumnFn = structureSampleColumnFn_;
            surfaceBlockFn = structureSurfaceBlockFn_;
            densityFn = structureDensityFn_;
        }
        if (!sampleColumnFn || !surfaceBlockFn || !densityFn)
        {
            return;
        }

        std::vector<StructureRegionKey> missingKeys;
        {
            std::lock_guard<std::mutex> lock(structureRegionMutex_);
            std::unordered_set<StructureRegionKey, StructureRegionKeyHasher> requestedKeys;
            for (const GpuSynthesisRequest& request : requests)
            {
                for (const StructureRegionKey& key : request.structureRegionKeys)
                {
                    if (!requestedKeys.insert(key).second)
                    {
                        continue;
                    }
                    if (!gpuStructureRegions_.contains(key))
                    {
                        missingKeys.push_back(key);
                    }
                }
            }
        }

        if (missingKeys.empty())
        {
            return;
        }

        struct PendingRegionUpload
        {
            StructureRegionKey key{};
            std::uint32_t instanceCount{0};
            Microsoft::WRL::ComPtr<ID3D12Resource> defaultBuffer;
            Microsoft::WRL::ComPtr<ID3D12Resource> uploadBuffer;
            D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
            bool uploadReady{false};
        };

        std::vector<PendingRegionUpload> uploads;
        uploads.reserve(missingKeys.size());
        for (const StructureRegionKey& key : missingKeys)
        {
            const StructureRegion region = buildStructureRegionData(key, sampleColumnFn, surfaceBlockFn, densityFn);
            const std::vector<StructureInstance>& instances = region.instances;
            if (instances.empty())
            {
                PendingRegionUpload upload{};
                upload.key = key;
                uploads.push_back(std::move(upload));
                continue;
            }

            std::vector<GpuStructureInstance> gpuInstances;
            gpuInstances.reserve(instances.size());
            for (const StructureInstance& instance : instances)
            {
                gpuInstances.push_back(packGpuStructureInstance(instance));
            }

            PendingRegionUpload upload{};
            upload.key = key;
            upload.instanceCount = static_cast<std::uint32_t>(gpuInstances.size());
            const std::uint64_t bufferBytes =
                static_cast<std::uint64_t>(gpuInstances.size() * sizeof(GpuStructureInstance));
            upload.defaultBuffer = createDefaultBuffer(device_.Get(), bufferBytes, D3D12_RESOURCE_STATE_COMMON);
            if (upload.defaultBuffer == nullptr)
            {
                continue;
            }

            std::byte* uploadMapped = nullptr;
            upload.uploadBuffer = createUploadBuffer(device_.Get(), bufferBytes, uploadMapped);
            if (upload.uploadBuffer == nullptr || uploadMapped == nullptr)
            {
                continue;
            }

            std::memcpy(uploadMapped, gpuInstances.data(), static_cast<std::size_t>(bufferBytes));
            upload.uploadBuffer->Unmap(0, nullptr);
            uploadMapped = nullptr;
            upload.state = D3D12_RESOURCE_STATE_COMMON;
            upload.uploadReady = true;

            std::wostringstream name;
            name << L"FarLodStructureRegion_" << key.regionX << L"_" << key.regionZ;
            setDebugObjectName(upload.defaultBuffer.Get(), name.str());
            uploads.push_back(std::move(upload));
        }

        bool uploadedAnyBuffers = false;
        bool uploadedBuffersCommitted = false;
        if (uploadContext_.ready() && uploadContext_.begin())
        {
            for (PendingRegionUpload& upload : uploads)
            {
                if (!upload.uploadReady || upload.defaultBuffer == nullptr || upload.uploadBuffer == nullptr)
                {
                    continue;
                }
                uploadedAnyBuffers = true;
                uploadContext_.transition(upload.defaultBuffer.Get(),
                                          D3D12_RESOURCE_STATE_COMMON,
                                          D3D12_RESOURCE_STATE_COPY_DEST);
                uploadContext_.copyBuffer(upload.defaultBuffer.Get(), 0, upload.uploadBuffer.Get(), 0,
                                          static_cast<std::uint64_t>(upload.instanceCount) * sizeof(GpuStructureInstance));
                uploadContext_.transition(upload.defaultBuffer.Get(),
                                          D3D12_RESOURCE_STATE_COPY_DEST,
                                          D3D12_RESOURCE_STATE_COMMON);
            }
            if (uploadedAnyBuffers)
            {
                uploadContext_.flush(nullptr);
                uploadContext_.waitForIdle();
            }
            uploadedBuffersCommitted = true;
        }

        std::lock_guard<std::mutex> lock(structureRegionMutex_);
        for (PendingRegionUpload& upload : uploads)
        {
            if (gpuStructureRegions_.contains(upload.key))
            {
                continue;
            }
            if (upload.instanceCount > 0 && !uploadedBuffersCommitted)
            {
                continue;
            }

            GpuStructureRegionState region{};
            region.key = upload.key;
            region.instanceCount = upload.instanceCount;
            if (upload.uploadReady)
            {
                region.instanceBuffer = std::move(upload.defaultBuffer);
                region.state = upload.state;
            }
            gpuStructureRegions_.emplace(upload.key, std::move(region));
        }
    }

    void destroyBufferPages()
    {
        for (BufferPage& page : bufferPages_)
        {
            page.vertexBuffer.Reset();
            page.indexBuffer.Reset();
            page.vertexUploadBuffer.Reset();
            page.indexUploadBuffer.Reset();
            page.drawRecordBuffer.Reset();
            page.mappedVertexData = nullptr;
            page.mappedIndexData = nullptr;
        }
        bufferPages_.clear();
    }

public:
    [[nodiscard]] int readyTileCount() const noexcept
    {
        int ready = 0;
        std::lock_guard<std::mutex> lock(configMutex_);
        for (const auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (chunk.active && chunk.gpu.resident)
            {
                ++ready;
            }
        }
        return ready;
    }

    [[nodiscard]] int queuedTileCount() const noexcept
    {
        int queued = 0;
        std::lock_guard<std::mutex> lock(configMutex_);
        for (const auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (chunk.active && chunk.inFlight)
            {
                ++queued;
            }
        }
        return queued;
    }

    [[nodiscard]] int pendingUploadTileCount() const noexcept
    {
        int pending = 0;
        std::lock_guard<std::mutex> lock(configMutex_);
        for (const auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (chunk.active && chunk.pendingMesh.valid())
            {
                ++pending;
            }
        }
        return pending;
    }

    [[nodiscard]] int buildQueueDepth() const noexcept
    {
        std::lock_guard<std::mutex> lock(buildQueueMutex_);
        return static_cast<int>(buildQueue_.size());
    }

    [[nodiscard]] double averageBuildMs() const noexcept
    {
        return lastAverageBuildMs_;
    }

    [[nodiscard]] double averageGpuSynthesisMs() const noexcept
    {
        return lastAverageGpuSynthesisMs_;
    }

    [[nodiscard]] double averageGpuStampMs() const noexcept
    {
        return lastAverageGpuStampMs_;
    }

    [[nodiscard]] double averageGpuFaceBuildMs() const noexcept
    {
        return lastAverageGpuFaceBuildMs_;
    }

    [[nodiscard]] LodDiagnosticsSnapshot diagnosticsSnapshot(const glm::vec3& cameraPos) const
    {
        struct OrderedRegion
        {
            const FarLodChunkRecord* chunk{nullptr};
            float distanceSq{0.0f};
        };

        constexpr std::size_t kMaxSnapshotTiles = 8;
        LodDiagnosticsSnapshot snapshot{};

        std::lock_guard<std::mutex> lock(configMutex_);
        std::vector<OrderedRegion> orderedRegions;
        orderedRegions.reserve(chunks_.size());
        for (const auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (!chunk.active)
            {
                continue;
            }

            ++snapshot.activeTiles;
            if (chunk.gpu.resident)
            {
                ++snapshot.readyTiles;
            }
            if (chunk.dirty)
            {
                ++snapshot.dirtyTiles;
            }
            if (chunk.inFlight)
            {
                ++snapshot.inFlightTiles;
            }

            const glm::vec3 center = (chunkDrawBoundsMin(chunk) + chunkDrawBoundsMax(chunk)) * 0.5f;
            const glm::vec3 delta = center - cameraPos;
            orderedRegions.push_back(OrderedRegion{&chunk, glm::dot(delta, delta)});
        }

        snapshot.averageBuildMs = lastAverageBuildMs_;
        snapshot.averageGpuSynthesisMs = lastAverageGpuSynthesisMs_;
        snapshot.averageGpuStampMs = lastAverageGpuStampMs_;
        snapshot.averageGpuFaceBuildMs = lastAverageGpuFaceBuildMs_;

        std::sort(orderedRegions.begin(),
                  orderedRegions.end(),
                  [](const OrderedRegion& lhs, const OrderedRegion& rhs)
                  {
                      return lhs.distanceSq < rhs.distanceSq;
                  });

        snapshot.tiles.reserve(std::min<std::size_t>(orderedRegions.size(), kMaxSnapshotTiles));
        for (std::size_t tileIndex = 0; tileIndex < orderedRegions.size() && tileIndex < kMaxSnapshotTiles; ++tileIndex)
        {
            const FarLodChunkRecord& chunk = *orderedRegions[tileIndex].chunk;
            LodDiagnosticsTileSnapshot tileSnapshot{};
            tileSnapshot.level = chunk.key.level;
            tileSnapshot.tileCoord = glm::ivec2(chunk.key.coord.x, chunk.key.coord.z);
            tileSnapshot.distanceSq = orderedRegions[tileIndex].distanceSq;
            tileSnapshot.active = chunk.active;
            tileSnapshot.dirty = chunk.dirty;
            tileSnapshot.inFlight = chunk.inFlight;
            tileSnapshot.indexCount = chunk.gpu.indexCount;
            tileSnapshot.blockScaleBlocks = chunk.level.blockScale;
            tileSnapshot.chunkSpanBlocks = chunk.level.chunkSpanBlocks();
            tileSnapshot.worldMin = chunk.cpu.worldMin;
            tileSnapshot.worldMax = chunk.cpu.worldMin + glm::ivec3(chunk.level.chunkSpanBlocks());

            snapshot.tiles.push_back(tileSnapshot);
        }

        return snapshot;
    }

    void writeDebugSnapshot(const std::filesystem::path& outputPath, const glm::vec3& cameraPos) const
    {
        struct OrderedRegion
        {
            const FarLodChunkRecord* chunk{nullptr};
            float distanceSq{0.0f};
        };

        constexpr std::size_t kMaxSnapshotTiles = 12;
        auto writeBool = [](std::ostream& out, bool value)
        {
            out << (value ? "true" : "false");
        };
        auto writeVec3 = [](std::ostream& out, const glm::vec3& value)
        {
            out << '[' << value.x << ',' << value.y << ',' << value.z << ']';
        };
        std::lock_guard<std::mutex> lock(configMutex_);
        std::vector<OrderedRegion> orderedRegions;
        orderedRegions.reserve(chunks_.size());
        int activeTiles = 0;
        int readyTiles = 0;
        int dirtyTiles = 0;
        int inFlightTiles = 0;
        for (const auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (!chunk.active)
            {
                continue;
            }

            ++activeTiles;
            if (chunk.gpu.resident)
            {
                ++readyTiles;
            }
            if (chunk.dirty)
            {
                ++dirtyTiles;
            }
            if (chunk.inFlight)
            {
                ++inFlightTiles;
            }

            const glm::vec3 center = (chunkDrawBoundsMin(chunk) + chunkDrawBoundsMax(chunk)) * 0.5f;
            const glm::vec3 delta = center - cameraPos;
            orderedRegions.push_back(OrderedRegion{
                &chunk,
                glm::dot(delta, delta)});
        }

        std::sort(orderedRegions.begin(),
                  orderedRegions.end(),
                  [](const OrderedRegion& lhs, const OrderedRegion& rhs)
                  {
                      return lhs.distanceSq < rhs.distanceSq;
                  });

        std::error_code ec;
        const std::filesystem::path parentPath = outputPath.parent_path();
        if (!parentPath.empty())
        {
            std::filesystem::create_directories(parentPath, ec);
        }

        std::ofstream out(outputPath, std::ios::trunc);
        if (!out)
        {
            std::cerr << "Failed to open LOD debug snapshot: " << outputPath << std::endl;
            return;
        }

        out << "{\n";
        out << "  \"camera\":";
        writeVec3(out, cameraPos);
        out << ",\n";
        out << "  \"active_tiles\":" << activeTiles
            << ",\n  \"ready_tiles\":" << readyTiles
            << ",\n  \"dirty_tiles\":" << dirtyTiles
            << ",\n  \"in_flight_tiles\":" << inFlightTiles
            << ",\n  \"built_tiles_last_update\":" << builtTilesLastUpdate_
            << ",\n  \"skipped_tiles_last_update\":" << skippedTilesLastUpdate_
            << ",\n  \"average_cpu_terrain_synthesis_ms\":" << lastAverageCpuTerrainSynthesisMs_
            << ",\n  \"average_cpu_structure_stamp_ms\":" << lastAverageCpuStructureStampMs_
            << ",\n  \"average_cpu_mesh_ms\":" << lastAverageCpuMeshMs_
            << ",\n  \"average_upload_wait_ms\":" << lastAverageUploadWaitMs_
            << ",\n  \"average_upload_copy_ms\":" << lastAverageUploadCopyMs_
            << ",\n  \"rendered_face_count\":" << lastRenderedFaceCount_
            << ",\n  \"rendered_vertex_count\":" << lastRenderedVertexCount_
            << ",\n  \"tiles\":[\n";

        const std::size_t tileCount = std::min<std::size_t>(orderedRegions.size(), kMaxSnapshotTiles);
        for (std::size_t tileIndex = 0; tileIndex < tileCount; ++tileIndex)
        {
            const FarLodChunkRecord& chunk = *orderedRegions[tileIndex].chunk;
            if (tileIndex > 0)
            {
                out << ",\n";
            }

            out << "    {\n";
            out << "      \"level\":" << chunk.key.level
                << ",\n      \"chunk_x\":" << chunk.key.coord.x
                << ",\n      \"chunk_y\":" << chunk.key.coord.y
                << ",\n      \"chunk_z\":" << chunk.key.coord.z
                << ",\n      \"distance_sq\":" << orderedRegions[tileIndex].distanceSq
                << ",\n      \"active\":";
            writeBool(out, chunk.active);
            out << ",\n      \"dirty\":";
            writeBool(out, chunk.dirty);
            out << ",\n      \"in_flight\":";
            writeBool(out, chunk.inFlight);
            out << ",\n      \"vertex_count\":" << chunk.gpu.vertexCount
                << ",\n      \"index_count\":" << chunk.gpu.indexCount
                << ",\n      \"block_scale\":" << chunk.level.blockScale
                << ",\n      \"chunk_span_blocks\":" << chunk.level.chunkSpanBlocks()
                << ",\n      \"bounds_min\":";
            writeVec3(out, chunkDrawBoundsMin(chunk));
            out << ",\n      \"bounds_max\":";
            writeVec3(out, chunkDrawBoundsMax(chunk));

            out << "\n    }";
        }

        out << "\n  ]\n}\n";
    }

private:
    void startWorkers()
    {
        stopWorkers_.store(false, std::memory_order_release);
        const std::size_t desired = std::max<std::size_t>(workerCount_, 1);
        workerThreads_.reserve(desired);
        for (std::size_t i = 0; i < desired; ++i)
        {
            workerThreads_.emplace_back(&FarTerrainManager::workerThreadLoop, this);
        }
        if (!dirtyBuildPlannerThread_.joinable())
        {
            dirtyBuildPlannerThread_ = std::thread(&FarTerrainManager::dirtyBuildPlannerLoop, this);
        }
        if (!touchLevelPlannerThread_.joinable())
        {
            touchLevelPlannerThread_ = std::thread(&FarTerrainManager::touchLevelPlannerLoop, this);
        }
    }

    void stopWorkers()
    {
        stopWorkers_.store(true, std::memory_order_release);
        buildQueueCv_.notify_all();
        for (std::thread& worker : workerThreads_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        workerThreads_.clear();
        {
            std::lock_guard<std::mutex> lock(dirtyBuildPlanMutex_);
            hasPendingDirtyBuildPlanRequest_ = false;
            hasReadyDirtyBuildPlanResult_ = false;
            dirtyBuildPlannerBusy_ = false;
            pendingDirtyBuildPlanRequest_ = {};
            readyDirtyBuildPlanResult_ = {};
        }
        dirtyBuildPlanCv_.notify_all();
        if (dirtyBuildPlannerThread_.joinable())
        {
            dirtyBuildPlannerThread_.join();
        }
        {
            std::lock_guard<std::mutex> lock(touchLevelPlanMutex_);
            hasPendingTouchLevelPlanRequest_ = false;
            hasReadyTouchLevelPlanResult_ = false;
            touchLevelPlannerBusy_ = false;
            pendingTouchLevelPlanRequest_ = {};
            readyTouchLevelPlanResult_ = {};
            touchLevelCaches_.clear();
            hasLastTouchLevelPlanRequest_ = false;
            lastTouchLevelPlanLevels_.clear();
        }
        touchLevelPlanCv_.notify_all();
        if (touchLevelPlannerThread_.joinable())
        {
            touchLevelPlannerThread_.join();
        }
        {
            std::lock_guard<std::mutex> lock(buildQueueMutex_);
            buildQueue_.clear();
            queuedKeys_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(completedMutex_);
            completedBuilds_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            gpuSynthesisRequests_.clear();
            pendingGpuParityReadbacks_.clear();
        }
    }

    void dirtyBuildPlannerLoop()
    {
        for (;;)
        {
            DirtyBuildPlanRequest request{};
            {
                std::unique_lock<std::mutex> lock(dirtyBuildPlanMutex_);
                dirtyBuildPlanCv_.wait(lock,
                                       [this]
                                       {
                                           return stopWorkers_.load(std::memory_order_acquire) ||
                                                  hasPendingDirtyBuildPlanRequest_;
                                       });
                if (stopWorkers_.load(std::memory_order_acquire) && !hasPendingDirtyBuildPlanRequest_)
                {
                    return;
                }

                request = std::move(pendingDirtyBuildPlanRequest_);
                pendingDirtyBuildPlanRequest_ = {};
                hasPendingDirtyBuildPlanRequest_ = false;
                dirtyBuildPlannerBusy_ = true;
            }

            DirtyBuildPlanResult result = buildDirtyPlanFromSnapshot(request);

            {
                std::lock_guard<std::mutex> lock(dirtyBuildPlanMutex_);
                readyDirtyBuildPlanResult_ = std::move(result);
                hasReadyDirtyBuildPlanResult_ = true;
                dirtyBuildPlannerBusy_ = false;
            }
        }
    }

    void touchLevelPlannerLoop()
    {
        for (;;)
        {
            TouchLevelPlanRequest request{};
            {
                std::unique_lock<std::mutex> lock(touchLevelPlanMutex_);
                touchLevelPlanCv_.wait(lock,
                                       [this]
                                       {
                                           return stopWorkers_.load(std::memory_order_acquire) ||
                                                  hasPendingTouchLevelPlanRequest_;
                                       });
                if (stopWorkers_.load(std::memory_order_acquire) && !hasPendingTouchLevelPlanRequest_)
                {
                    return;
                }

                request = std::move(pendingTouchLevelPlanRequest_);
                pendingTouchLevelPlanRequest_ = {};
                hasPendingTouchLevelPlanRequest_ = false;
                touchLevelPlannerBusy_ = true;
            }

            TouchLevelPlanResult result = buildTouchLevelPlan(request);

            {
                std::lock_guard<std::mutex> lock(touchLevelPlanMutex_);
                readyTouchLevelPlanResult_ = std::move(result);
                hasReadyTouchLevelPlanResult_ = true;
                touchLevelPlannerBusy_ = false;
            }
        }
    }

    void workerThreadLoop()
    {
        for (;;)
        {
            BuildJob job{};
            {
                std::unique_lock<std::mutex> lock(buildQueueMutex_);
                buildQueueCv_.wait(lock,
                                   [this]
                                   {
                                       return stopWorkers_.load(std::memory_order_acquire) || !buildQueue_.empty();
                                   });
                if (stopWorkers_.load(std::memory_order_acquire) && buildQueue_.empty())
                {
                    return;
                }

                job = buildQueue_.front();
                buildQueue_.pop_front();
                queuedKeys_.erase(job.key);
            }

            ColumnSampleFn columnSampleFn;
            {
                std::lock_guard<std::mutex> lock(configMutex_);
                columnSampleFn = columnSampleFn_;
            }
            if (!columnSampleFn)
            {
                continue;
            }

            const SteadyClock::time_point buildStart = SteadyClock::now();
            BuildResult result{};
            result.key = job.key;
            result.buildVersion = job.buildVersion;
            result.epoch = job.epoch;
            {
                const SteadyClock::time_point terrainStart = SteadyClock::now();
                result.cpu.key = job.key;
                result.cpu.blockScale = job.level.blockScale;
                result.cpu.worldMin = job.key.coord * job.level.chunkSpanBlocks();
                result.cpu.boundsMin = glm::vec3(result.cpu.worldMin);
                result.cpu.boundsMax = glm::vec3(result.cpu.worldMin + glm::ivec3(job.level.chunkSpanBlocks()));
                result.cpuTerrainSynthesisMs =
                    std::chrono::duration<double, std::milli>(SteadyClock::now() - terrainStart).count();

                GpuSynthesisRequest gpuRequest{};
                gpuRequest.key = job.key;
                gpuRequest.buildVersion = job.buildVersion;
                gpuRequest.epoch = job.epoch;
                gpuRequest.worldMin = result.cpu.worldMin;
                gpuRequest.blockScale = result.cpu.blockScale;
                gpuRequest.lodLevel = job.level.level;
                gpuRequest.structureRegionKeys =
                    computeStructureRegionKeysForTile(result.cpu.worldMin, job.level.chunkSpanBlocks());
                gpuRequest.parityRequested = gpuParityEnabled_;
                if (gpuRequest.parityRequested)
                {
                    gpuRequest.cpuPackedParityVoxels =
                        std::make_shared<std::array<PackedFarLodVoxelGpu, kVoxelCount>>();
                    for (std::size_t voxelIdx = 0; voxelIdx < kVoxelCount; ++voxelIdx)
                    {
                        (*gpuRequest.cpuPackedParityVoxels)[voxelIdx] = packGpuVoxel(result.cpu.voxels[voxelIdx]);
                    }
                }
                {
                    std::lock_guard<std::mutex> gpuLock(gpuRequestMutex_);
                    gpuSynthesisRequests_.push_back(std::move(gpuRequest));
                }

            }
            result.buildMs =
                std::chrono::duration<double, std::milli>(SteadyClock::now() - buildStart).count();

            std::lock_guard<std::mutex> lock(completedMutex_);
            completedBuilds_.push_back(std::move(result));
        }
    }

    [[nodiscard]] static glm::vec3 chunkDrawBoundsMin(const FarLodChunkRecord& chunk) noexcept
    {
        return (chunk.gpu.resident && chunk.gpu.indexCount > 0) ? chunk.residentBoundsMin : chunk.cpu.boundsMin;
    }

    [[nodiscard]] static glm::vec3 chunkDrawBoundsMax(const FarLodChunkRecord& chunk) noexcept
    {
        return (chunk.gpu.resident && chunk.gpu.indexCount > 0) ? chunk.residentBoundsMax : chunk.cpu.boundsMax;
    }

    [[nodiscard]] static bool dirtyBuildJobPriorityLess(const BuildJob& lhs, const BuildJob& rhs) noexcept
    {
        const int lhsTier = !lhs.hadResidentMesh ? (lhs.fallbackOnly ? 1 : 0) : 2;
        const int rhsTier = !rhs.hadResidentMesh ? (rhs.fallbackOnly ? 1 : 0) : 2;
        if (lhsTier != rhsTier)
        {
            return lhsTier < rhsTier;
        }
        if (lhs.hadResidentMesh != rhs.hadResidentMesh)
        {
            return rhs.hadResidentMesh;
        }
        if (!lhs.hadResidentMesh && !rhs.hadResidentMesh)
        {
            if (lhs.level.level != rhs.level.level)
            {
                return lhs.level.level > rhs.level.level;
            }
        }
        else if (lhs.level.level != rhs.level.level)
        {
            return lhs.level.level < rhs.level.level;
        }
        return lhs.ringDistanceChunks < rhs.ringDistanceChunks;
    }

    static void trimDirtyBuildJobs(std::vector<BuildJob>& jobs,
                                   std::size_t workerBudget,
                                   int exactMissingChunks,
                                   std::size_t exactPendingUploads)
    {
        static constexpr std::size_t kFallbackReservation = 2;
        const std::size_t exactBacklogPenalty =
            (exactMissingChunks > 32 || exactPendingUploads > 24)
                ? 8
                : ((exactMissingChunks > 8 || exactPendingUploads > 8) ? 16 : 32);
        const std::size_t maxQueuedPerUpdate = std::max<std::size_t>(workerBudget * 12, exactBacklogPenalty);
        if (jobs.size() <= maxQueuedPerUpdate)
        {
            return;
        }

        std::vector<BuildJob> trimmed;
        trimmed.reserve(maxQueuedPerUpdate);
        std::size_t fallbackAdded = 0;
        const std::size_t desiredFallback = std::min(kFallbackReservation, maxQueuedPerUpdate);
        for (const BuildJob& job : jobs)
        {
            if (trimmed.size() >= maxQueuedPerUpdate)
            {
                break;
            }
            if (job.fallbackOnly)
            {
                continue;
            }
            const std::size_t remaining = maxQueuedPerUpdate - trimmed.size();
            if (remaining <= (desiredFallback - fallbackAdded))
            {
                break;
            }
            trimmed.push_back(job);
        }
        if (trimmed.size() < maxQueuedPerUpdate && fallbackAdded < desiredFallback)
        {
            for (const BuildJob& job : jobs)
            {
                if (!job.fallbackOnly)
                {
                    continue;
                }
                if (trimmed.size() >= maxQueuedPerUpdate)
                {
                    break;
                }
                trimmed.push_back(job);
                ++fallbackAdded;
                if (fallbackAdded >= desiredFallback)
                {
                    break;
                }
            }
        }
        jobs.swap(trimmed);
    }

    [[nodiscard]] DirtyBuildPlanResult buildDirtyPlanFromSnapshot(const DirtyBuildPlanRequest& request)
    {
        DirtyBuildPlanResult result{};
        result.sequence = request.sequence;
        result.epoch = request.epoch;

        std::unordered_set<FarLodChunkKey, FarLodChunkKeyHasher> queuedKeysSnapshot;
        {
            std::lock_guard<std::mutex> lock(buildQueueMutex_);
            queuedKeysSnapshot = queuedKeys_;
        }

        constexpr float kForwardDotThreshold = -0.35f;
        const glm::vec2 forwardXZ = normalizePriorityForwardXZ(request.cameraForward);

        std::lock_guard<std::mutex> lock(configMutex_);
        for (const auto& [key, chunk] : chunks_)
        {
            if (!chunk.active || !chunk.dirty || chunk.inFlight || queuedKeysSnapshot.contains(key))
            {
                continue;
            }

            const bool keepResidentStable = chunk.gpu.resident && chunk.gpu.indexCount > 0;
            const int ringDistanceChunks =
                chunkMinHorizontalRingDistanceChunks(chunk.level, request.cameraChunk, key.coord);
            const int nearBuildGraceChunks = std::max(2, chunk.level.blockScale * 2);
            const bool keepNearbyStable = ringDistanceChunks <= nearBuildGraceChunks;

            if (keepResidentStable)
            {
                const std::uint64_t cadence = residentRefreshCadenceUpdates(chunk.level);
                if (cadence > 1u && ringDistanceChunks > nearBuildGraceChunks)
                {
                    const std::uint64_t stagger = staggeredChunkUpdateHash(key) % cadence;
                    if ((updateStamp_ + stagger) % cadence != 0u)
                    {
                        continue;
                    }
                }
            }

            if (!keepResidentStable && !keepNearbyStable)
            {
                const glm::vec3 boundsMin = chunkDrawBoundsMin(chunk);
                const glm::vec3 boundsMax = chunkDrawBoundsMax(chunk);
                const glm::vec3 center = (boundsMin + boundsMax) * 0.5f;
                const glm::vec2 toChunk(center.x - static_cast<float>(request.cameraChunk.x * kChunkSizeX),
                                        center.z - static_cast<float>(request.cameraChunk.z * kChunkSizeZ));
                if (glm::dot(toChunk, toChunk) > kEpsilon)
                {
                    const float facing = glm::dot(glm::normalize(toChunk), forwardXZ);
                    if (facing < kForwardDotThreshold)
                    {
                        continue;
                    }
                }
                if (request.hasVisibilityFrustum && !request.visibilityFrustum.intersectsAABB(boundsMin, boundsMax))
                {
                    const glm::vec3 toChunk3 = center - request.visibilityCameraPos;
                    if (glm::dot(toChunk3, toChunk3) >
                        static_cast<float>(chunk.level.chunkSpanBlocks() * chunk.level.chunkSpanBlocks() * 4))
                    {
                        continue;
                    }
                }
            }

            result.jobs.push_back(BuildJob{
                key,
                chunk.level,
                chunk.buildVersion,
                request.epoch,
                ringDistanceChunks,
                keepResidentStable,
                chunk.fallbackOnly});
        }

        std::sort(result.jobs.begin(), result.jobs.end(), dirtyBuildJobPriorityLess);
        trimDirtyBuildJobs(result.jobs,
                           request.workerBudget,
                           request.exactMissingChunks,
                           request.exactPendingUploads);
        return result;
    }

    void requestDirtyBuildPlan()
    {
        DirtyBuildPlanRequest request{};
        request.sequence = dirtyBuildPlanSequence_.fetch_add(1, std::memory_order_acq_rel) + 1u;
        request.epoch = buildEpoch_.load(std::memory_order_acquire);
        request.cameraChunk = cameraChunk_;
        request.cameraForward = cameraForward_;
        request.exactMissingChunks = exactMissingChunks_.load(std::memory_order_relaxed);
        request.exactPendingUploads = exactPendingUploads_.load(std::memory_order_relaxed);
        request.workerBudget = std::max<std::size_t>(workerCount_, 1);
        {
            std::lock_guard<std::mutex> lock(configMutex_);
            request.visibilityFrustum = lastVisibilityFrustum_;
            request.visibilityCameraPos = lastVisibilityCameraPos_;
            request.hasVisibilityFrustum = hasVisibilityFrustum_;
        }

        std::lock_guard<std::mutex> lock(dirtyBuildPlanMutex_);
        if (dirtyBuildPlannerBusy_ || hasPendingDirtyBuildPlanRequest_)
        {
            return;
        }
        pendingDirtyBuildPlanRequest_ = std::move(request);
        hasPendingDirtyBuildPlanRequest_ = true;
        dirtyBuildPlanCv_.notify_one();
    }

    void consumeReadyDirtyBuildPlan()
    {
        DirtyBuildPlanResult result{};
        {
            std::lock_guard<std::mutex> lock(dirtyBuildPlanMutex_);
            if (!hasReadyDirtyBuildPlanResult_)
            {
                return;
            }
            result = std::move(readyDirtyBuildPlanResult_);
            readyDirtyBuildPlanResult_ = {};
            hasReadyDirtyBuildPlanResult_ = false;
        }

        if (result.jobs.empty())
        {
            return;
        }

        const std::uint64_t currentEpoch = buildEpoch_.load(std::memory_order_acquire);
        std::vector<BuildJob> readyJobs;
        readyJobs.reserve(result.jobs.size());
        {
            std::lock_guard<std::mutex> lock(configMutex_);
            if (result.epoch != currentEpoch)
            {
                return;
            }

            for (const BuildJob& job : result.jobs)
            {
                auto it = chunks_.find(job.key);
                if (it == chunks_.end())
                {
                    continue;
                }

                FarLodChunkRecord& chunk = it->second;
                if (!chunk.active || !chunk.dirty || chunk.inFlight || chunk.buildVersion != job.buildVersion)
                {
                    continue;
                }

                chunk.inFlight = true;
                readyJobs.push_back(job);
            }
        }

        if (readyJobs.empty())
        {
            return;
        }

        std::vector<FarLodChunkKey> duplicateKeys;
        duplicateKeys.reserve(readyJobs.size());
        {
            std::lock_guard<std::mutex> lock(buildQueueMutex_);
            for (const BuildJob& job : readyJobs)
            {
                if (!queuedKeys_.insert(job.key).second)
                {
                    duplicateKeys.push_back(job.key);
                    continue;
                }
                buildQueue_.push_back(job);
            }
        }

        if (!duplicateKeys.empty())
        {
            std::lock_guard<std::mutex> lock(configMutex_);
            for (const FarLodChunkKey& key : duplicateKeys)
            {
                auto it = chunks_.find(key);
                if (it != chunks_.end())
                {
                    it->second.inFlight = false;
                }
            }
        }

        if (readyJobs.size() > duplicateKeys.size())
        {
            buildQueueCv_.notify_all();
        }
    }

    void scheduleDirtyBuilds()
    {
        consumeReadyDirtyBuildPlan();
        requestDirtyBuildPlan();
    }

    void submitGpuSynthesisRequests(double budgetMs)
    {
        std::deque<GpuSynthesisRequest> requests;
        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            requests.swap(gpuSynthesisRequests_);
        }

        lastAverageGpuSynthesisMs_ = 0.0;
        lastAverageGpuStampMs_ = 0.0;
        lastAverageGpuFaceBuildMs_ = 0.0;
        if (requests.empty() || !gpuContext_.ready())
        {
            if (!requests.empty())
            {
                std::lock_guard<std::mutex> lock(gpuRequestMutex_);
                for (GpuSynthesisRequest& request : requests)
                {
                    gpuSynthesisRequests_.push_back(std::move(request));
                }
            }
            return;
        }

        gpuContext_.setReadbackEnabled(false);
        if (!gpuContext_.begin())
        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            for (GpuSynthesisRequest& request : requests)
            {
                gpuSynthesisRequests_.push_back(std::move(request));
            }
            return;
        }

        const FarLodWorkBudget workBudget = computeWorkBudget();
        appendAtlasUpdates(requests, workBudget.atlasUpdateCells);

        std::vector<PendingGpuParityReadback> pendingReadbacks;
        std::vector<FarLodChunkKey> submittedKeys;
        std::vector<FarLodChunkKey> stagedGpuMeshes;
        const double budgetLimit = budgetMs <= 0.0 ? std::numeric_limits<double>::max() : budgetMs;
        const SteadyClock::time_point submitStart = SteadyClock::now();
        double totalGpuSynthesisMs = 0.0;
        double totalGpuStampMs = 0.0;
        double totalGpuFaceBuildMs = 0.0;
        std::size_t submittedCount = 0;
        std::size_t synthSubmittedCount = 0;
        std::size_t faceBuildSamples = 0;
        std::size_t remainingGpuBudgetUnits = std::max<std::size_t>(workBudget.gpuDispatchBudgetUnits, 1);
        const std::size_t maxGpuSubmissions = std::max<std::size_t>(workBudget.maxGpuSubmissions, 1);
        auto computeMaxMergeExtent = [](int blockScale) noexcept
        {
            return (blockScale <= 2)   ? 2u
                 : (blockScale <= 4)   ? 4u
                 : (blockScale <= 8)   ? 8u
                                       : 16u;
        };

        while (!requests.empty())
        {
            if (submittedCount >= maxGpuSubmissions)
            {
                break;
            }
            if (std::chrono::duration<double, std::milli>(SteadyClock::now() - submitStart).count() >= budgetLimit)
            {
                break;
            }

            GpuSynthesisRequest request = std::move(requests.front());
            requests.pop_front();

            std::lock_guard<std::mutex> lock(configMutex_);
            auto it = chunks_.find(request.key);
            if (it == chunks_.end())
            {
                continue;
            }

            FarLodChunkRecord& chunk = it->second;
            if (!chunk.active || request.epoch != buildEpoch_.load(std::memory_order_acquire) ||
                request.buildVersion != chunk.buildVersion ||
                chunk.gpu.columnBuffer == nullptr)
            {
                continue;
            }

            auto atlasIt = levelAtlases_.find(request.key.level);
            if (atlasIt == levelAtlases_.end() || atlasIt->second.buffer == nullptr)
            {
                continue;
            }
            FarLodLevelAtlasState& atlas = atlasIt->second;

            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD GPU submit chunk level=" << request.key.level
                       << " coord=[" << request.key.coord.x << "," << request.key.coord.y << "," << request.key.coord.z << "]"
                       << " buildVersion=" << request.buildVersion
                       << " epoch=" << request.epoch
                       << " blockScale=" << request.blockScale
                       << " lodLevel=" << request.lodLevel
                       << " parity=" << (request.parityRequested ? "on" : "off")
                       << " columnStateBefore=" << resourceStateName(chunk.gpu.columnState);
                chunkManagerDebugLog(stream.str());
            }

            const SteadyClock::time_point synthStart = SteadyClock::now();
            gpuContext_.transition(chunk.gpu.columnBuffer.Get(),
                                   chunk.gpu.columnState,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            chunk.gpu.columnState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            gpuContext_.dispatchSynth(request.worldMin,
                                      request.blockScale,
                                      worldgenTables_.header.seaLevel,
                                      atlas.originCell,
                                      atlas.atlasSizeCells,
                                      atlas.buffer.Get(),
                                      atlas.elementCount(),
                                      chunk.gpu.columnBuffer.Get());
            gpuContext_.uavBarrier(chunk.gpu.columnBuffer.Get());
            chunk.gpu.columnState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD GPU synth dispatched chunk level=" << request.key.level
                       << " coord=[" << request.key.coord.x << "," << request.key.coord.y << "," << request.key.coord.z << "]";
                chunkManagerDebugLog(stream.str());
            }
            totalGpuSynthesisMs +=
                std::chrono::duration<double, std::milli>(SteadyClock::now() - synthStart).count();

            const bool canBuildGpuMesh =
                blockUvBuffer_ != nullptr &&
                blockUvCount_ > 0 &&
                emptyVoxelBuffer_ != nullptr &&
                chunk.gpu.faceCountBuffer != nullptr &&
                chunk.gpu.faceAnalysisBuffer != nullptr &&
                chunk.gpu.faceDescriptorBuffer != nullptr &&
                chunk.gpu.facePrefixBuffer != nullptr &&
                chunk.gpu.faceGroupSumBuffer != nullptr;
            const std::size_t synthCostUnits = canBuildGpuMesh ? 8u : 2u;
            if (remainingGpuBudgetUnits < synthCostUnits)
            {
                requests.push_front(std::move(request));
                break;
            }
            if (canBuildGpuMesh)
            {
                ID3D12Resource* neighborPosX = emptyVoxelBuffer_.Get();
                ID3D12Resource* neighborNegX = emptyVoxelBuffer_.Get();
                ID3D12Resource* neighborPosZ = emptyVoxelBuffer_.Get();
                ID3D12Resource* neighborNegZ = emptyVoxelBuffer_.Get();
                auto bindNeighbor = [&](const glm::ivec3& offset,
                                        ID3D12Resource*& outBuffer)
                {
                    const FarLodChunkKey neighborKey{request.key.level, request.key.coord + offset};
                    auto neighborIt = chunks_.find(neighborKey);
                    if (neighborIt == chunks_.end())
                    {
                        return;
                    }
                    FarLodChunkRecord& neighbor = neighborIt->second;
                    if (!neighbor.gpu.voxelReady || neighbor.gpu.columnBuffer == nullptr)
                    {
                        return;
                    }
                    if (gpuContext_.completedFenceValue() < neighbor.gpu.voxelFenceValue)
                    {
                        return;
                    }
                    outBuffer = neighbor.gpu.columnBuffer.Get();
                    if (neighbor.gpu.columnState != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
                    {
                        gpuContext_.transition(neighbor.gpu.columnBuffer.Get(),
                                               neighbor.gpu.columnState,
                                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                        neighbor.gpu.columnState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    }
                };

                bindNeighbor(glm::ivec3(1, 0, 0), neighborPosX);
                bindNeighbor(glm::ivec3(-1, 0, 0), neighborNegX);
                bindNeighbor(glm::ivec3(0, 0, 1), neighborPosZ);
                bindNeighbor(glm::ivec3(0, 0, -1), neighborNegZ);

                const std::uint32_t faceCapacity = requestedFaceCapacity(chunk);
                const std::size_t reservedVertexCount = static_cast<std::size_t>(faceCapacity) * 4u;
                const std::uint32_t reservedIndexCount = faceCapacity * 6u;
                Allocation allocation = acquireAllocation(reservedVertexCount, reservedIndexCount);
                if (allocation.pageIndex == kInvalidChunkBufferPage || allocation.pageIndex >= bufferPages_.size())
                {
                    requests.push_front(std::move(request));
                    break;
                }
                if (!stageGpuMeshForCommit(chunk,
                                           allocation,
                                           faceCapacity,
                                           reservedVertexCount,
                                           reservedIndexCount,
                                           request.buildVersion,
                                           request.epoch))
                {
                    releaseAllocationRange(allocation.pageIndex,
                                           allocation.vertexOffset,
                                           reservedVertexCount,
                                           allocation.indexOffset,
                                           reservedIndexCount,
                                           allocation.recordIndex);
                    requests.push_front(std::move(request));
                    break;
                }

                if (chunk.gpu.columnState != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
                {
                    gpuContext_.transition(chunk.gpu.columnBuffer.Get(),
                                           chunk.gpu.columnState,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    chunk.gpu.columnState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                }
                gpuContext_.transition(chunk.gpu.faceCountBuffer.Get(),
                                       chunk.gpu.faceCountState,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chunk.gpu.faceCountState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                gpuContext_.transition(chunk.gpu.faceAnalysisBuffer.Get(),
                                       chunk.gpu.faceAnalysisState,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chunk.gpu.faceAnalysisState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                gpuContext_.transition(chunk.gpu.faceDescriptorBuffer.Get(),
                                       chunk.gpu.faceDescriptorState,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chunk.gpu.faceDescriptorState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                gpuContext_.transition(chunk.gpu.facePrefixBuffer.Get(),
                                       chunk.gpu.facePrefixState,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chunk.gpu.facePrefixState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                gpuContext_.transition(chunk.gpu.faceGroupSumBuffer.Get(),
                                       chunk.gpu.faceGroupSumState,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chunk.gpu.faceGroupSumState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

                BufferPage& page = bufferPages_[allocation.pageIndex];
                if (page.vertexBuffer != nullptr)
                {
                    gpuContext_.transition(page.vertexBuffer.Get(),
                                           page.vertexState,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    page.vertexState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                }
                if (page.indexBuffer != nullptr)
                {
                    gpuContext_.transition(page.indexBuffer.Get(),
                                           page.indexState,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    page.indexState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                }
                if (page.drawRecordBuffer != nullptr)
                {
                    gpuContext_.transition(page.drawRecordBuffer.Get(),
                                           page.drawRecordState,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    page.drawRecordState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                }

                const SteadyClock::time_point faceBuildStart = SteadyClock::now();
                const std::uint32_t maxMergeExtent = computeMaxMergeExtent(request.blockScale);
                gpuContext_.dispatchFaceCount(static_cast<int>(std::floor(chunk.cpu.boundsMin.y)),
                                              request.blockScale,
                                              0u,
                                              maxMergeExtent,
                                              chunk.gpu.columnBuffer.Get(),
                                              neighborPosX,
                                              neighborNegX,
                                              neighborPosZ,
                                              neighborNegZ,
                                              chunk.gpu.faceCountBuffer.Get(),
                                              chunk.gpu.faceAnalysisBuffer.Get(),
                                              chunk.gpu.faceDescriptorBuffer.Get());
                gpuContext_.uavBarrier(chunk.gpu.faceCountBuffer.Get());
                gpuContext_.uavBarrier(chunk.gpu.faceAnalysisBuffer.Get());
                gpuContext_.uavBarrier(chunk.gpu.faceDescriptorBuffer.Get());
                gpuContext_.dispatchFacePrefix(chunk.gpu.faceCountBuffer.Get(),
                                               chunk.gpu.facePrefixBuffer.Get(),
                                               chunk.gpu.faceGroupSumBuffer.Get());
                gpuContext_.uavBarrier(chunk.gpu.facePrefixBuffer.Get());
                gpuContext_.dispatchFaceEmit(glm::ivec3(request.worldMin.x,
                                                        static_cast<int>(std::floor(chunk.cpu.boundsMin.y)),
                                                        request.worldMin.z),
                                             request.blockScale,
                                             maxMergeExtent,
                                             static_cast<std::uint32_t>(allocation.vertexOffset),
                                             static_cast<std::uint32_t>(allocation.indexOffset),
                                             allocation.recordIndex,
                                             faceCapacity,
                                             chunk.gpu.columnBuffer.Get(),
                                             chunk.gpu.faceCountBuffer.Get(),
                                             chunk.gpu.faceAnalysisBuffer.Get(),
                                             chunk.gpu.faceDescriptorBuffer.Get(),
                                             chunk.gpu.facePrefixBuffer.Get(),
                                             blockUvBuffer_.Get(),
                                             blockUvCount_,
                                             static_cast<std::uint32_t>(sizeof(GpuBlockFaceUv)),
                                             page.vertexBuffer.Get(),
                                             static_cast<std::uint32_t>(page.vertexCapacity),
                                             page.indexBuffer.Get(),
                                             static_cast<std::uint32_t>(page.indexCapacity),
                                             page.drawRecordBuffer.Get(),
                                             static_cast<std::uint32_t>(page.recordCapacity));
                if (page.vertexBuffer != nullptr)
                {
                    gpuContext_.uavBarrier(page.vertexBuffer.Get());
                }
                if (page.indexBuffer != nullptr)
                {
                    gpuContext_.uavBarrier(page.indexBuffer.Get());
                }
                if (page.drawRecordBuffer != nullptr)
                {
                    gpuContext_.uavBarrier(page.drawRecordBuffer.Get());
                    gpuContext_.transition(page.drawRecordBuffer.Get(),
                                           page.drawRecordState,
                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
                    page.drawRecordState = D3D12_RESOURCE_STATE_COPY_SOURCE;
                    gpuContext_.copyBuffer(chunk.pendingMesh.drawRecordReadbackBuffer.Get(),
                                           0,
                                           page.drawRecordBuffer.Get(),
                                           static_cast<std::uint64_t>(allocation.recordIndex) *
                                               sizeof(ChunkRenderBatch::GpuCullRecord),
                                           static_cast<std::uint64_t>(sizeof(ChunkRenderBatch::GpuCullRecord)));
                }
                gpuContext_.transition(chunk.gpu.faceCountBuffer.Get(),
                                       chunk.gpu.faceCountState,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                chunk.gpu.faceCountState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                gpuContext_.transition(chunk.gpu.faceAnalysisBuffer.Get(),
                                       chunk.gpu.faceAnalysisState,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                chunk.gpu.faceAnalysisState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                gpuContext_.transition(chunk.gpu.faceDescriptorBuffer.Get(),
                                       chunk.gpu.faceDescriptorState,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                chunk.gpu.faceDescriptorState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                gpuContext_.transition(chunk.gpu.facePrefixBuffer.Get(),
                                       chunk.gpu.facePrefixState,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                chunk.gpu.facePrefixState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                gpuContext_.transition(chunk.gpu.faceGroupSumBuffer.Get(),
                                       chunk.gpu.faceGroupSumState,
                                       D3D12_RESOURCE_STATE_COMMON);
                chunk.gpu.faceGroupSumState = D3D12_RESOURCE_STATE_COMMON;
                if (page.vertexBuffer != nullptr)
                {
                    gpuContext_.transition(page.vertexBuffer.Get(),
                                           page.vertexState,
                                           D3D12_RESOURCE_STATE_COMMON);
                    page.vertexState = D3D12_RESOURCE_STATE_COMMON;
                }
                if (page.indexBuffer != nullptr)
                {
                    gpuContext_.transition(page.indexBuffer.Get(),
                                           page.indexState,
                                           D3D12_RESOURCE_STATE_COMMON);
                    page.indexState = D3D12_RESOURCE_STATE_COMMON;
                }
                if (page.drawRecordBuffer != nullptr)
                {
                    gpuContext_.transition(page.drawRecordBuffer.Get(),
                                           page.drawRecordState,
                                           D3D12_RESOURCE_STATE_COMMON);
                    page.drawRecordState = D3D12_RESOURCE_STATE_COMMON;
                }

                totalGpuFaceBuildMs +=
                    std::chrono::duration<double, std::milli>(SteadyClock::now() - faceBuildStart).count();
                ++faceBuildSamples;
                stagedGpuMeshes.push_back(request.key);
            }

            if (request.parityRequested)
            {
                gpuContext_.transition(chunk.gpu.columnBuffer.Get(),
                                       chunk.gpu.columnState,
                                       D3D12_RESOURCE_STATE_COMMON);
                chunk.gpu.columnState = D3D12_RESOURCE_STATE_COMMON;
                PendingGpuParityReadback pending{};
                pending.key = request.key;
                pending.buildVersion = request.buildVersion;
                pending.epoch = request.epoch;
                pending.cpuPackedParityVoxels = std::move(request.cpuPackedParityVoxels);
                pending.cpuColumnParityDescriptors = std::move(request.cpuColumnParityDescriptors);
                pending.parityRequested = true;
                pending.copySubmitted = false;
                pendingReadbacks.push_back(std::move(pending));
            }
            else
            {
                gpuContext_.transition(chunk.gpu.columnBuffer.Get(),
                                       chunk.gpu.columnState,
                                       D3D12_RESOURCE_STATE_COMMON);
                chunk.gpu.columnState = D3D12_RESOURCE_STATE_COMMON;
            }

            submittedKeys.push_back(request.key);
            remainingGpuBudgetUnits -= synthCostUnits;
            ++synthSubmittedCount;
            ++submittedCount;
        }

        const FarLodGpuContext::FlushResult flushResult = gpuContext_.flush();
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD GPU submit batch flushed submittedCount=" << submittedCount
                   << " stagedGpuMeshes=" << stagedGpuMeshes.size()
                   << " pendingParity=" << pendingReadbacks.size()
                   << " fence=" << flushResult.fenceValue;
            chunkManagerDebugLog(stream.str());
        }
        for (PendingGpuParityReadback& pending : pendingReadbacks)
        {
            pending.computeFenceValue = flushResult.fenceValue;
        }

        {
            std::lock_guard<std::mutex> lock(configMutex_);
            for (const PendingGpuParityReadback& pending : pendingReadbacks)
            {
                auto it = chunks_.find(pending.key);
                if (it == chunks_.end())
                {
                    continue;
                }
                it->second.gpu.voxelFenceValue = flushResult.fenceValue;
                it->second.gpu.voxelReady = (flushResult.fenceValue == 0);
                it->second.gpu.parityValidated = false;
            }
            if (!gpuParityEnabled_)
            {
                for (const FarLodChunkKey& key : submittedKeys)
                {
                    auto it = chunks_.find(key);
                    if (it == chunks_.end())
                    {
                        continue;
                    }
                    it->second.gpu.voxelFenceValue = flushResult.fenceValue;
                    it->second.gpu.voxelReady = true;
                    it->second.gpu.parityValidated = false;
                }
            }
            for (const FarLodChunkKey& key : stagedGpuMeshes)
            {
                auto it = chunks_.find(key);
                if (it == chunks_.end())
                {
                    continue;
                }
                FarLodChunkRecord& chunk = it->second;
                if (chunk.pendingMesh.valid() && chunk.pendingMesh.gpuGenerated &&
                    chunk.pendingMesh.buildVersion == chunk.buildVersion &&
                    chunk.pendingMesh.epoch == buildEpoch_.load(std::memory_order_acquire))
                {
                    chunk.pendingMesh.gpuFenceValue = flushResult.fenceValue;
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            for (GpuSynthesisRequest& request : requests)
            {
                gpuSynthesisRequests_.push_back(std::move(request));
            }
            for (PendingGpuParityReadback& pending : pendingReadbacks)
            {
                pendingGpuParityReadbacks_.push_back(std::move(pending));
            }
        }

        if (submittedCount > 0)
        {
            if (synthSubmittedCount > 0)
            {
                lastAverageGpuSynthesisMs_ = totalGpuSynthesisMs / static_cast<double>(synthSubmittedCount);
                lastAverageGpuStampMs_ = totalGpuStampMs / static_cast<double>(synthSubmittedCount);
            }
            else
            {
                lastAverageGpuSynthesisMs_ = 0.0;
                lastAverageGpuStampMs_ = 0.0;
            }
            if (faceBuildSamples > 0)
            {
                lastAverageGpuFaceBuildMs_ = totalGpuFaceBuildMs / static_cast<double>(faceBuildSamples);
                if (benchmarkMetrics_ != nullptr)
                {
                    benchmarkMetrics_->lodGpuFaceBuildStage.recordMicros(
                        static_cast<std::uint64_t>(std::max(lastAverageGpuFaceBuildMs_, 0.0) * 1000.0));
                }
            }
            if (benchmarkMetrics_ != nullptr)
            {
                benchmarkMetrics_->lodGpuSynthesisStage.recordMicros(
                    static_cast<std::uint64_t>(std::max(lastAverageGpuSynthesisMs_, 0.0) * 1000.0));
                benchmarkMetrics_->lodGpuStampStage.recordMicros(
                    static_cast<std::uint64_t>(std::max(lastAverageGpuStampMs_, 0.0) * 1000.0));
            }
        }
    }

    void collectCompletedBuilds(double uploadBudgetMs)
    {
        const SteadyClock::time_point collectStart = SteadyClock::now();
        std::deque<BuildResult> completed;
        {
            std::lock_guard<std::mutex> lock(completedMutex_);
            completed.swap(completedBuilds_);
        }

        builtTilesLastUpdate_ = 0;
        skippedTilesLastUpdate_ = 0;
        lastBuiltFaceCount_ = 0;
        lastBuiltVertexCount_ = 0;
        lastBuiltIndexCount_ = 0;
        lastAverageBuildMs_ = 0.0;
        lastAverageCpuTerrainSynthesisMs_ = 0.0;
        lastAverageCpuStructureStampMs_ = 0.0;
        lastAverageCpuMeshMs_ = 0.0;
        lastAverageUploadWaitMs_ = 0.0;
        lastAverageUploadCopyMs_ = 0.0;
        if (completed.empty())
        {
            lastCollectMs_ =
                std::chrono::duration<double, std::milli>(SteadyClock::now() - collectStart).count();
            lastUploadMs_ = 0.0;
            return;
        }

        const double uploadBudgetLimit = uploadBudgetMs <= 0.0 ? std::numeric_limits<double>::max() : uploadBudgetMs;
        const SteadyClock::time_point uploadStart = SteadyClock::now();
        double totalBuildMs = 0.0;
        double totalCpuTerrainSynthesisMs = 0.0;
        double totalCpuStructureStampMs = 0.0;
        double totalCpuMeshMs = 0.0;
        double totalUploadCopyMs = 0.0;
        std::size_t applied = 0;
        std::deque<BuildResult> stillPendingCompleted;

        while (!completed.empty())
        {
            if (std::chrono::duration<double, std::milli>(SteadyClock::now() - uploadStart).count() >= uploadBudgetLimit)
            {
                break;
            }

            BuildResult result = std::move(completed.front());
            completed.pop_front();

            std::lock_guard<std::mutex> lock(configMutex_);
            auto it = chunks_.find(result.key);
            if (it == chunks_.end())
            {
                continue;
            }

            FarLodChunkRecord& chunk = it->second;
            chunk.inFlight = false;
            if (result.epoch != buildEpoch_.load(std::memory_order_acquire) || result.buildVersion != chunk.buildVersion)
            {
                continue;
            }

            if (result.skippedByRelevance)
            {
                if (lodVisibilityDebugLoggingEnabled())
                {
                    std::ostringstream stream;
                    stream << "lodvis skipped_by_relevance_preserve_resident level=" << chunk.key.level
                           << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                           << " resident=" << (chunk.gpu.resident ? "y" : "n")
                           << " indexCount=" << chunk.gpu.indexCount;
                    lodVisibilityDebugLog(stream.str());
                }
                chunk.dirty = true;
                ++skippedTilesLastUpdate_;
                continue;
            }

            const std::uint64_t currentDependencyRevision = currentAtlasDependencyRevision(chunk);
            const bool staged =
                chunk.pendingMesh.valid() &&
                chunk.pendingMesh.gpuGenerated &&
                chunk.pendingMesh.epoch == result.epoch &&
                chunk.pendingMesh.buildVersion == result.buildVersion;
            const bool canPreserveResidentMesh =
                chunk.gpu.resident &&
                chunk.gpu.indexCount > 0 &&
                chunk.lastBuiltAtlasDependencyRevision != 0 &&
                currentDependencyRevision == chunk.lastBuiltAtlasDependencyRevision;
            if (!staged)
            {
                if (canPreserveResidentMesh)
                {
                    chunk.dirty = false;
                    ++skippedTilesLastUpdate_;
                    if (lodVisibilityDebugLoggingEnabled())
                    {
                        std::ostringstream stream;
                        stream << "lodvis apply_skip_unchanged_gpu_mesh level=" << chunk.key.level
                               << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                               << " resident=y"
                               << " indexCount=" << chunk.gpu.indexCount
                               << " dependencyRevision=" << currentDependencyRevision;
                        lodVisibilityDebugLog(stream.str());
                    }
                    continue;
                }
                // Keep the old resident allocation alive; this chunk remains dirty until a replacement commits.
                chunk.dirty = true;
                ++skippedTilesLastUpdate_;
                if (lodVisibilityDebugLoggingEnabled())
                {
                    std::ostringstream stream;
                    stream << "lodvis apply_deferred_no_gpu_mesh level=" << chunk.key.level
                           << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                           << " resident=" << (chunk.gpu.resident ? "y" : "n")
                           << " indexCount=" << chunk.gpu.indexCount;
                    lodVisibilityDebugLog(stream.str());
                }
                continue;
            }

            chunk.dirty = false;
            chunk.lastBuiltAtlasDependencyRevision = currentDependencyRevision;
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis apply_gpu_mesh_staged level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " resident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount
                       << " pendingReservedIndexCount=" << chunk.pendingMesh.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            ++builtTilesLastUpdate_;
            ++applied;
            totalBuildMs += result.buildMs;
            totalCpuTerrainSynthesisMs += result.cpuTerrainSynthesisMs;
            totalCpuStructureStampMs += result.cpuStructureStampMs;
            totalCpuMeshMs += result.cpuMeshMs;
            totalUploadCopyMs += result.uploadCopyMs;
        }

        if (applied > 0)
        {
            lastAverageBuildMs_ = totalBuildMs / static_cast<double>(applied);
            lastAverageCpuTerrainSynthesisMs_ = totalCpuTerrainSynthesisMs / static_cast<double>(applied);
            lastAverageCpuStructureStampMs_ = totalCpuStructureStampMs / static_cast<double>(applied);
            lastAverageCpuMeshMs_ = totalCpuMeshMs / static_cast<double>(applied);
            lastAverageUploadCopyMs_ = totalUploadCopyMs / static_cast<double>(applied);
            if (benchmarkMetrics_ != nullptr)
            {
                benchmarkMetrics_->farBuildStage.recordMicros(
                    static_cast<std::uint64_t>(std::max(lastAverageBuildMs_, 0.0) * 1000.0));
            }
        }

        {
            std::lock_guard<std::mutex> lock(completedMutex_);
            for (BuildResult& result : completed)
            {
                completedBuilds_.push_back(std::move(result));
            }
            for (BuildResult& result : stillPendingCompleted)
            {
                completedBuilds_.push_back(std::move(result));
            }
        }
        lastCollectMs_ = std::chrono::duration<double, std::milli>(SteadyClock::now() - collectStart).count();
        lastUploadMs_ = std::chrono::duration<double, std::milli>(SteadyClock::now() - uploadStart).count();
    }

    void pollGpuParityResults()
    {
        std::deque<PendingGpuParityReadback> pending;
        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            pending.swap(pendingGpuParityReadbacks_);
        }

        if (pending.empty())
        {
            return;
        }

        std::deque<PendingGpuParityReadback> stillPending;
        for (PendingGpuParityReadback& parity : pending)
        {
            if (!parity.copySubmitted)
            {
                bool copyAlreadyPending = false;
                for (const PendingGpuParityReadback& queuedParity : stillPending)
                {
                    if (queuedParity.copySubmitted)
                    {
                        copyAlreadyPending = true;
                        break;
                    }
                }
                if (copyAlreadyPending)
                {
                    stillPending.push_back(std::move(parity));
                    continue;
                }

                if (parity.computeFenceValue == 0)
                {
                    stillPending.push_back(std::move(parity));
                    continue;
                }

                if (parityReadbackScratch_ == nullptr || parityReadbackMapped_ == nullptr || !uploadContext_.ready())
                {
                    if (chunkManagerDebugLoggingEnabled())
                    {
                        std::ostringstream stream;
                        stream << "Far LOD GPU parity waiting for readback resources chunk level=" << parity.key.level
                               << " coord=[" << parity.key.coord.x << "," << parity.key.coord.y << "," << parity.key.coord.z << "]";
                        chunkManagerDebugLog(stream.str());
                    }
                    stillPending.push_back(std::move(parity));
                    continue;
                }

                if (!uploadContext_.begin())
                {
                    if (chunkManagerDebugLoggingEnabled())
                    {
                        std::ostringstream stream;
                        stream << "Far LOD GPU parity waiting for upload context chunk level=" << parity.key.level
                               << " coord=[" << parity.key.coord.x << "," << parity.key.coord.y << "," << parity.key.coord.z << "]";
                        chunkManagerDebugLog(stream.str());
                    }
                    stillPending.push_back(std::move(parity));
                    continue;
                }

                bool submittedCopy = false;
                {
                    std::lock_guard<std::mutex> configLock(configMutex_);
                    auto it = chunks_.find(parity.key);
                    if (it == chunks_.end() ||
                        parity.epoch != buildEpoch_.load(std::memory_order_acquire) ||
                        parity.buildVersion != it->second.buildVersion ||
                        it->second.gpu.columnBuffer == nullptr)
                    {
                        // No longer relevant; drop it.
                    }
                    else
                    {
                        FarLodChunkRecord& chunk = it->second;
                        if (chunkManagerDebugLoggingEnabled())
                        {
                            std::ostringstream stream;
                            stream << "Far LOD GPU atlas parity copy submit chunk level=" << parity.key.level
                                   << " coord=[" << parity.key.coord.x << "," << parity.key.coord.y << "," << parity.key.coord.z << "]"
                                   << " computeFence=" << parity.computeFenceValue
                                   << " columnState=" << resourceStateName(chunk.gpu.columnState);
                            chunkManagerDebugLog(stream.str());
                        }
                        uploadContext_.waitForFence(gpuContext_.fence(), parity.computeFenceValue);
                        uploadContext_.copyBuffer(parityReadbackScratch_.Get(),
                                                  0,
                                                  chunk.gpu.columnBuffer.Get(),
                                                  0,
                                                  static_cast<std::uint64_t>(kLogicalSize * kLogicalSize * sizeof(GpuTerrainColumnDescriptor)));
                        submittedCopy = true;
                    }
                }

                if (!submittedCopy)
                {
                    uploadContext_.flush(nullptr);
                    continue;
                }

                uploadContext_.flush(nullptr);
                parity.copyFenceValue = uploadContext_.lastSubmittedFenceValue();
                parity.copySubmitted = true;
                if (chunkManagerDebugLoggingEnabled())
                {
                    std::ostringstream stream;
                    stream << "Far LOD GPU atlas parity copy queued chunk level=" << parity.key.level
                           << " coord=[" << parity.key.coord.x << "," << parity.key.coord.y << "," << parity.key.coord.z << "]"
                           << " copyFence=" << parity.copyFenceValue;
                    chunkManagerDebugLog(stream.str());
                }
                stillPending.push_back(std::move(parity));
                continue;
            }

            if (parity.copyFenceValue == 0 || parity.copyFenceValue > uploadContext_.completedFenceValue())
            {
                stillPending.push_back(std::move(parity));
                continue;
            }

            const auto* gpuReadback = reinterpret_cast<const GpuTerrainColumnDescriptor*>(parityReadbackMapped_);
            std::uint32_t mismatchCount = 0;
            std::size_t firstMismatchIndex = kLogicalSize * kLogicalSize;
            auto descriptorsMatchForParity = [](const GpuTerrainColumnDescriptor& cpu,
                                                const GpuTerrainColumnDescriptor& gpu) noexcept
            {
                constexpr std::uint32_t kParityRelevantFlags = 0x0Bu;
                return (cpu.flags & kParityRelevantFlags) == (gpu.flags & kParityRelevantFlags) &&
                       cpu.terrainTopY == gpu.terrainTopY &&
                       cpu.terrainBaseY == gpu.terrainBaseY &&
                       cpu.waterTopY == gpu.waterTopY &&
                       cpu.waterBottomY == gpu.waterBottomY &&
                       cpu.terrainTopBlock == gpu.terrainTopBlock &&
                       cpu.terrainSideBlock == gpu.terrainSideBlock &&
                       cpu.waterBlock == gpu.waterBlock;
            };

            GpuTerrainColumnDescriptor cpuValue{};
            GpuTerrainColumnDescriptor gpuValue{};
            if (parity.cpuColumnParityDescriptors)
            {
                for (std::size_t columnIdx = 0; columnIdx < static_cast<std::size_t>(kLogicalSize * kLogicalSize); ++columnIdx)
                {
                    if (descriptorsMatchForParity((*parity.cpuColumnParityDescriptors)[columnIdx],
                                                  gpuReadback[columnIdx]))
                    {
                        continue;
                    }

                    if (firstMismatchIndex == static_cast<std::size_t>(kLogicalSize * kLogicalSize))
                    {
                        firstMismatchIndex = columnIdx;
                        cpuValue = (*parity.cpuColumnParityDescriptors)[columnIdx];
                        gpuValue = gpuReadback[columnIdx];
                    }
                    ++mismatchCount;
                }
            }

            std::lock_guard<std::mutex> configLock(configMutex_);
            auto it = chunks_.find(parity.key);
            if (it == chunks_.end())
            {
                continue;
            }

            FarLodChunkRecord& chunk = it->second;
            if (parity.epoch != buildEpoch_.load(std::memory_order_acquire) || parity.buildVersion != chunk.buildVersion)
            {
                continue;
            }

            chunk.gpu.voxelReady = true;
            chunk.gpu.parityValidated = parity.parityRequested;
            chunk.gpu.parityMismatchCount = mismatchCount;
            rollingGpuParityMismatchCount_ += mismatchCount;
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD GPU atlas parity complete chunk level=" << parity.key.level
                       << " coord=[" << parity.key.coord.x << "," << parity.key.coord.y << "," << parity.key.coord.z << "]"
                       << " mismatches=" << mismatchCount
                       << " copyFence=" << parity.copyFenceValue;
                chunkManagerDebugLog(stream.str());
            }
            if (mismatchCount > 0 && firstMismatchIndex < static_cast<std::size_t>(kLogicalSize * kLogicalSize))
            {
                std::cerr << "Far LOD GPU atlas parity mismatch chunk level=" << parity.key.level
                          << " coord=[" << parity.key.coord.x << "," << parity.key.coord.y << "," << parity.key.coord.z
                          << "] firstColumn=" << firstMismatchIndex
                          << " cpu={flags=0x" << std::hex << cpuValue.flags << std::dec
                          << ", terrainTopY=" << cpuValue.terrainTopY
                          << ", terrainBaseY=" << cpuValue.terrainBaseY
                          << ", waterTopY=" << cpuValue.waterTopY
                          << ", waterBottomY=" << cpuValue.waterBottomY
                          << ", canopyTopY=" << cpuValue.canopyTopY
                          << ", canopyBottomY=" << cpuValue.canopyBottomY
                          << ", terrainTopBlock=" << cpuValue.terrainTopBlock
                          << ", terrainSideBlock=" << cpuValue.terrainSideBlock
                          << ", waterBlock=" << cpuValue.waterBlock
                          << ", canopyBlock=" << cpuValue.canopyBlock
                          << "}"
                          << " gpu={flags=0x" << std::hex << gpuValue.flags << std::dec
                          << ", terrainTopY=" << gpuValue.terrainTopY
                          << ", terrainBaseY=" << gpuValue.terrainBaseY
                          << ", waterTopY=" << gpuValue.waterTopY
                          << ", waterBottomY=" << gpuValue.waterBottomY
                          << ", canopyTopY=" << gpuValue.canopyTopY
                          << ", canopyBottomY=" << gpuValue.canopyBottomY
                          << ", terrainTopBlock=" << gpuValue.terrainTopBlock
                          << ", terrainSideBlock=" << gpuValue.terrainSideBlock
                          << ", waterBlock=" << gpuValue.waterBlock
                          << ", canopyBlock=" << gpuValue.canopyBlock
                          << "}" << std::endl;
            }
        }

        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            for (PendingGpuParityReadback& parity : stillPending)
            {
                pendingGpuParityReadbacks_.push_back(std::move(parity));
            }
        }
    }

    [[nodiscard]] ChunkMesh buildChunkMesh(const FarLodChunkCpu& cpu,
                                           int lodLevel) const;
    void uploadWorldgenTables()
    {
        worldgenHeaderBuffer_.Reset();
        worldgenBiomeBuffer_.Reset();
        worldgenBiomeSelectionBuffer_.Reset();
        worldgenOceanSelectionBuffer_.Reset();
        worldgenTransitionBuffer_.Reset();
        worldgenSubBiomeBuffer_.Reset();
        worldgenPermutationBuffer_.Reset();
        if (device_ == nullptr)
        {
            return;
        }

        struct PendingUpload
        {
            ID3D12Resource* destination{nullptr};
            Microsoft::WRL::ComPtr<ID3D12Resource> upload;
            std::uint64_t bytes{0};
        };

        std::vector<PendingUpload> pending;
        pending.reserve(6);

        auto stageRawBuffer = [&](Microsoft::WRL::ComPtr<ID3D12Resource>& destination,
                                  const wchar_t* debugName,
                                  const void* data,
                                  std::uint64_t bytes,
                                  std::uint64_t minimumBytes) {
            const std::uint64_t uploadBytes = std::max(bytes, minimumBytes);
            destination = createDefaultBuffer(device_.Get(), uploadBytes, D3D12_RESOURCE_STATE_COMMON);
            if (destination == nullptr)
            {
                throw std::runtime_error("failed to allocate far lod worldgen buffer");
            }
            setDebugObjectName(destination.Get(), debugName);

            std::byte* uploadMapped = nullptr;
            Microsoft::WRL::ComPtr<ID3D12Resource> upload =
                createUploadBuffer(device_.Get(), uploadBytes, uploadMapped);
            if (upload == nullptr || uploadMapped == nullptr)
            {
                throw std::runtime_error("failed to allocate far lod worldgen upload buffer");
            }

            std::memset(uploadMapped, 0, static_cast<std::size_t>(uploadBytes));
            if (data != nullptr && bytes > 0u)
            {
                std::memcpy(uploadMapped, data, static_cast<std::size_t>(bytes));
            }
            pending.push_back(PendingUpload{destination.Get(), upload, uploadBytes});
        };

        stageRawBuffer(worldgenHeaderBuffer_,
                       L"FarLodWorldgenHeader",
                       &worldgenTables_.header,
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuWorldgenHeader)),
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuWorldgenHeader)));
        stageRawBuffer(worldgenBiomeBuffer_,
                       L"FarLodWorldgenBiomes",
                       worldgenTables_.biomes.empty() ? nullptr : worldgenTables_.biomes.data(),
                       static_cast<std::uint64_t>(worldgenTables_.biomes.size() * sizeof(terrain::FarLodGpuBiome)),
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuBiome)));
        stageRawBuffer(worldgenBiomeSelectionBuffer_,
                       L"FarLodWorldgenBiomeSelections",
                       worldgenTables_.biomeSelections.empty() ? nullptr : worldgenTables_.biomeSelections.data(),
                       static_cast<std::uint64_t>(worldgenTables_.biomeSelections.size() *
                                                  sizeof(terrain::FarLodGpuBiomeSelection)),
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuBiomeSelection)));
        stageRawBuffer(worldgenOceanSelectionBuffer_,
                       L"FarLodWorldgenOceanSelections",
                       worldgenTables_.oceanSelections.empty() ? nullptr : worldgenTables_.oceanSelections.data(),
                       static_cast<std::uint64_t>(worldgenTables_.oceanSelections.size() *
                                                  sizeof(terrain::FarLodGpuBiomeSelection)),
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuBiomeSelection)));
        stageRawBuffer(worldgenTransitionBuffer_,
                       L"FarLodWorldgenTransitions",
                       worldgenTables_.transitionBiomes.empty() ? nullptr : worldgenTables_.transitionBiomes.data(),
                       static_cast<std::uint64_t>(worldgenTables_.transitionBiomes.size() *
                                                  sizeof(terrain::FarLodGpuTransitionBiome)),
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuTransitionBiome)));
        stageRawBuffer(worldgenSubBiomeBuffer_,
                       L"FarLodWorldgenSubBiomes",
                       worldgenTables_.subBiomes.empty() ? nullptr : worldgenTables_.subBiomes.data(),
                       static_cast<std::uint64_t>(worldgenTables_.subBiomes.size() *
                                                  sizeof(terrain::FarLodGpuSubBiome)),
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuSubBiome)));
        stageRawBuffer(worldgenPermutationBuffer_,
                       L"FarLodWorldgenSurfacePermutation",
                       worldgenTables_.surfacePermutation.empty() ? nullptr : worldgenTables_.surfacePermutation.data(),
                       static_cast<std::uint64_t>(worldgenTables_.surfacePermutation.size() * sizeof(std::uint32_t)),
                       static_cast<std::uint64_t>(sizeof(std::uint32_t)));

        if (!uploadContext_.begin())
        {
            throw std::runtime_error("failed to begin upload for far lod worldgen tables");
        }
        for (const PendingUpload& upload : pending)
        {
            uploadContext_.transition(upload.destination, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            uploadContext_.copyBuffer(upload.destination, 0, upload.upload.Get(), 0, upload.bytes);
            uploadContext_.transition(upload.destination, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        }
        uploadContext_.flush(nullptr);
        uploadContext_.waitForIdle();

        for (PendingUpload& upload : pending)
        {
            if (upload.upload != nullptr)
            {
                upload.upload->Unmap(0, nullptr);
            }
        }
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD worldgen tables uploaded biomeCount=" << worldgenTables_.biomes.size()
                   << " biomeSelectionCount=" << worldgenTables_.biomeSelections.size()
                   << " oceanSelectionCount=" << worldgenTables_.oceanSelections.size()
                   << " transitionCount=" << worldgenTables_.transitionBiomes.size()
                   << " subBiomeCount=" << worldgenTables_.subBiomes.size()
                   << " seaLevel=" << worldgenTables_.header.seaLevel
                   << " seed=" << worldgenTables_.header.seed;
            chunkManagerDebugLog(stream.str());
        }
    }

    static constexpr std::uint8_t kFarLodVoxelWater = 0x01u;
    static constexpr std::uint8_t kFarLodVoxelStructure = 0x02u;
    static constexpr std::uint8_t kFarLodVoxelCutout = 0x04u;
    static constexpr std::uint8_t kFarLodVoxelTerrain = 0x08u;
    static constexpr std::uint32_t kFarColumnFlagTerrain = 0x01u;
    static constexpr std::uint32_t kFarColumnFlagWater = 0x02u;
    static constexpr std::uint32_t kFarColumnFlagCanopy = 0x04u;
    static constexpr std::uint32_t kFarColumnFlagSteep = 0x08u;

    [[nodiscard]] bool stageBuiltChunkForCommit(FarLodChunkRecord& chunk,
                                                const ChunkMesh& mesh,
                                                std::uint32_t buildVersion,
                                                std::uint64_t epoch,
                                                double& outCopyMs)
    {
        outCopyMs = 0.0;
        if (chunk.pendingMesh.valid())
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_pending_replacement level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }
        if (mesh.vertices.empty() || mesh.indices.empty())
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_empty_mesh level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        if (!uploadContext_.ready())
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_upload_not_ready level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        Allocation allocation = acquireAllocation(mesh.vertices.size(), mesh.indices.size());
        if (allocation.pageIndex == kInvalidChunkBufferPage || allocation.pageIndex >= bufferPages_.size())
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_allocation_failed level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount
                       << " requestedVertices=" << mesh.vertices.size()
                       << " requestedIndices=" << mesh.indices.size();
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        BufferPage& page = bufferPages_[allocation.pageIndex];
        if (page.mappedVertexData == nullptr || page.mappedIndexData == nullptr ||
            page.vertexUploadBuffer == nullptr || page.indexUploadBuffer == nullptr ||
            page.vertexBuffer == nullptr || page.indexBuffer == nullptr)
        {
            releaseAllocationRange(allocation.pageIndex,
                                   allocation.vertexOffset,
                                   mesh.vertices.size(),
                                   allocation.indexOffset,
                                   static_cast<std::uint32_t>(mesh.indices.size()),
                                   allocation.recordIndex);
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_upload_buffers_missing level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        const SteadyClock::time_point copyStart = SteadyClock::now();
        if (!mesh.vertices.empty())
        {
            std::memcpy(page.mappedVertexData + allocation.vertexOffset * sizeof(Vertex),
                        mesh.vertices.data(),
                        mesh.vertices.size() * sizeof(Vertex));
            uploadContext_.copyBuffer(page.vertexBuffer.Get(),
                                      static_cast<std::uint64_t>(allocation.vertexOffset * sizeof(Vertex)),
                                      page.vertexUploadBuffer.Get(),
                                      static_cast<std::uint64_t>(allocation.vertexOffset * sizeof(Vertex)),
                                      static_cast<std::uint64_t>(mesh.vertices.size() * sizeof(Vertex)));
        }
        if (!mesh.indices.empty())
        {
            std::memcpy(page.mappedIndexData + allocation.indexOffset * sizeof(std::uint32_t),
                        mesh.indices.data(),
                        mesh.indices.size() * sizeof(std::uint32_t));
            uploadContext_.copyBuffer(page.indexBuffer.Get(),
                                      static_cast<std::uint64_t>(allocation.indexOffset * sizeof(std::uint32_t)),
                                      page.indexUploadBuffer.Get(),
                                      static_cast<std::uint64_t>(allocation.indexOffset * sizeof(std::uint32_t)),
                                      static_cast<std::uint64_t>(mesh.indices.size() * sizeof(std::uint32_t)));
        }
        outCopyMs = std::chrono::duration<double, std::milli>(SteadyClock::now() - copyStart).count();

        chunk.pendingMesh.pageIndex = allocation.pageIndex;
        chunk.pendingMesh.vertexOffset = allocation.vertexOffset;
        chunk.pendingMesh.indexOffset = allocation.indexOffset;
        chunk.pendingMesh.vertexCount = mesh.vertices.size();
        chunk.pendingMesh.indexCount = static_cast<std::uint32_t>(mesh.indices.size());
        chunk.pendingMesh.boundsMin = mesh.boundsMin;
        chunk.pendingMesh.boundsMax = mesh.boundsMax;
        chunk.pendingMesh.uploadFenceValue = 0;
        chunk.pendingMesh.buildVersion = buildVersion;
        chunk.pendingMesh.epoch = epoch;

        if (lodVisibilityDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "lodvis staged_replacement level=" << chunk.key.level
                   << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                   << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                   << " oldIndexCount=" << chunk.gpu.indexCount
                   << " reservedIndexCount=" << chunk.pendingMesh.indexCount;
            lodVisibilityDebugLog(stream.str());
        }
        return true;
    }

    [[nodiscard]] bool stageGpuMeshForCommit(FarLodChunkRecord& chunk,
                                             const Allocation& allocation,
                                             std::uint32_t faceCapacity,
                                             std::size_t vertexCount,
                                             std::uint32_t indexCount,
                                             std::uint32_t buildVersion,
                                             std::uint64_t epoch)
    {
        if (chunk.pendingMesh.valid())
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_pending_gpu_replacement level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        if (allocation.pageIndex == kInvalidChunkBufferPage || allocation.pageIndex >= bufferPages_.size())
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_gpu_allocation_failed level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        std::byte* mappedDrawRecordBytes = nullptr;
        Microsoft::WRL::ComPtr<ID3D12Resource> readbackBuffer =
            createReadbackBuffer(device_.Get(),
                                 static_cast<std::uint64_t>(sizeof(ChunkRenderBatch::GpuCullRecord)),
                                 mappedDrawRecordBytes);
        ChunkRenderBatch::GpuCullRecord* mappedDrawRecord =
            reinterpret_cast<ChunkRenderBatch::GpuCullRecord*>(mappedDrawRecordBytes);
        if (readbackBuffer == nullptr || mappedDrawRecord == nullptr)
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_gpu_readback_failed level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        chunk.pendingMesh.pageIndex = allocation.pageIndex;
        chunk.pendingMesh.vertexOffset = allocation.vertexOffset;
        chunk.pendingMesh.indexOffset = allocation.indexOffset;
        chunk.pendingMesh.vertexCount = vertexCount;
        chunk.pendingMesh.indexCount = indexCount;
        chunk.pendingMesh.faceCapacity = faceCapacity;
        chunk.pendingMesh.recordIndex = allocation.recordIndex;
        chunk.pendingMesh.boundsMin = chunk.cpu.boundsMin;
        chunk.pendingMesh.boundsMax = chunk.cpu.boundsMax;
        chunk.pendingMesh.uploadFenceValue = 0;
        chunk.pendingMesh.gpuFenceValue = 0;
        chunk.pendingMesh.gpuGenerated = true;
        chunk.pendingMesh.buildVersion = buildVersion;
        chunk.pendingMesh.epoch = epoch;
        chunk.pendingMesh.drawRecordReadbackBuffer = std::move(readbackBuffer);
        chunk.pendingMesh.mappedDrawRecord = mappedDrawRecord;

        if (lodVisibilityDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "lodvis staged_gpu_replacement level=" << chunk.key.level
                   << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                   << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                   << " recordIndex=" << chunk.pendingMesh.recordIndex
                   << " reservedFaces=" << chunk.pendingMesh.faceCapacity;
            lodVisibilityDebugLog(stream.str());
        }
        return true;
    }

    void commitPendingMeshUploads()
    {
        const UINT64 completedUploadFenceValue = uploadContext_.completedFenceValue();
        const UINT64 completedGpuFenceValue = gpuContext_.completedFenceValue();
        const std::uint64_t epoch = buildEpoch_.load(std::memory_order_acquire);

        std::lock_guard<std::mutex> lock(configMutex_);
        for (auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (!chunk.pendingMesh.valid())
            {
                continue;
            }
            if (chunk.pendingMesh.gpuGenerated)
            {
                if (chunk.pendingMesh.gpuFenceValue != 0 && completedGpuFenceValue < chunk.pendingMesh.gpuFenceValue)
                {
                    continue;
                }
            }
            else
            {
                if (chunk.pendingMesh.uploadFenceValue == 0 || completedUploadFenceValue < chunk.pendingMesh.uploadFenceValue)
                {
                    continue;
                }
            }

            if (chunk.pendingMesh.epoch != epoch || chunk.pendingMesh.buildVersion != chunk.buildVersion)
            {
                releasePendingMeshAllocation(chunk);
                continue;
            }

            if (chunk.pendingMesh.gpuGenerated && chunk.pendingMesh.mappedDrawRecord == nullptr)
            {
                releasePendingMeshAllocation(chunk);
                if (!chunk.dirty)
                {
                    markDirty(chunk);
                }
                continue;
            }

            ChunkRenderBatch::GpuCullRecord pendingRecord{};
            std::uint32_t actualFaceCount = 0u;
            bool overflowed = false;
            if (chunk.pendingMesh.gpuGenerated)
            {
                pendingRecord = *chunk.pendingMesh.mappedDrawRecord;
                actualFaceCount = pendingRecord.reserved & kGpuDrawRecordFaceCountMask;
                overflowed = (pendingRecord.reserved & kGpuDrawRecordOverflowFlag) != 0u;
                if (overflowed)
                {
                    chunk.requestedFaceCapacityHint =
                        std::max(chunk.requestedFaceCapacityHint, growReservedFaceCapacity(actualFaceCount));
                    releasePendingMeshAllocation(chunk);
                    if (!chunk.dirty)
                    {
                        markDirty(chunk);
                    }
                    continue;
                }
            }
            else
            {
                pendingRecord.indexCount = chunk.pendingMesh.indexCount;
                actualFaceCount = chunk.pendingMesh.indexCount / 6u;
                pendingRecord.boundsMin = glm::vec4(chunk.pendingMesh.boundsMin, 1.0f);
                pendingRecord.boundsMax = glm::vec4(chunk.pendingMesh.boundsMax, 1.0f);
            }

            const std::uint32_t oldPageIndex = chunk.gpu.pageIndex;
            const std::size_t oldVertexOffset = chunk.gpu.vertexOffset;
            const std::size_t oldIndexOffset = chunk.gpu.indexOffset;
            const std::size_t oldReservedVertexCount = chunk.gpu.reservedVertexCount;
            const std::size_t oldReservedIndexCount = chunk.gpu.reservedIndexCount;
            const std::uint32_t oldRecordIndex = chunk.gpu.recordIndex;
            const std::uint32_t pendingPageIndex = chunk.pendingMesh.pageIndex;
            const std::size_t pendingVertexOffset = chunk.pendingMesh.vertexOffset;
            const std::size_t pendingIndexOffset = chunk.pendingMesh.indexOffset;
            const std::size_t pendingReservedVertexCount = chunk.pendingMesh.vertexCount;
            const std::size_t pendingReservedIndexCount = chunk.pendingMesh.indexCount;
            const std::uint32_t pendingRecordIndex = chunk.pendingMesh.recordIndex;
            const std::uint32_t pendingFaceCapacity = chunk.pendingMesh.faceCapacity;

            const bool hasResidentGeometry = !chunk.pendingMesh.gpuGenerated || pendingRecord.indexCount > 0u;
            if (hasResidentGeometry)
            {
                chunk.gpu.pageIndex = pendingPageIndex;
                chunk.gpu.vertexOffset = pendingVertexOffset;
                chunk.gpu.indexOffset = pendingIndexOffset;
                chunk.gpu.vertexCount = static_cast<std::size_t>(actualFaceCount) * 4u;
                chunk.gpu.indexCount = pendingRecord.indexCount;
                chunk.gpu.reservedVertexCount = pendingReservedVertexCount;
                chunk.gpu.reservedIndexCount = pendingReservedIndexCount;
                chunk.gpu.faceCapacity = pendingFaceCapacity;
                chunk.gpu.recordIndex = pendingRecordIndex;
                chunk.gpu.resident = (chunk.gpu.indexCount > 0u);
                chunk.residentBoundsMin = glm::vec3(pendingRecord.boundsMin);
                chunk.residentBoundsMax = glm::vec3(pendingRecord.boundsMax);
                if (chunk.gpu.pageIndex != kInvalidChunkBufferPage && chunk.gpu.pageIndex < bufferPages_.size())
                {
                    BufferPage& page = bufferPages_[chunk.gpu.pageIndex];
                    page.recordActiveCount = std::max(page.recordActiveCount,
                                                      static_cast<std::size_t>(chunk.gpu.recordIndex + 1u));
                }
            }
            else
            {
                releasePendingMeshAllocation(chunk);
                chunk.gpu.pageIndex = kInvalidChunkBufferPage;
                chunk.gpu.vertexOffset = 0;
                chunk.gpu.indexOffset = 0;
                chunk.gpu.vertexCount = 0;
                chunk.gpu.indexCount = 0;
                chunk.gpu.reservedVertexCount = 0;
                chunk.gpu.reservedIndexCount = 0;
                chunk.gpu.faceCapacity = pendingFaceCapacity;
                chunk.gpu.recordIndex = kInvalidFarDrawRecordIndex;
                chunk.gpu.resident = false;
            }
            releasePendingDrawRecordReadback(chunk.pendingMesh);
            chunk.pendingMesh = {};

            if (oldPageIndex != kInvalidChunkBufferPage && oldPageIndex < bufferPages_.size())
            {
                BufferPage& oldPage = bufferPages_[oldPageIndex];
                if (oldRecordIndex != kInvalidFarDrawRecordIndex)
                {
                    clearDrawRecord(oldPage, oldRecordIndex);
                }
                releaseAllocationRange(oldPageIndex,
                                       oldVertexOffset,
                                       oldReservedVertexCount,
                                       oldIndexOffset,
                                       static_cast<std::uint32_t>(oldReservedIndexCount),
                                       oldRecordIndex);
            }
            chunk.requestedFaceCapacityHint = 0;
            chunk.cpu.meshReady = chunk.gpu.resident;
        }
    }

    bool enabled_{true};
    int farDistanceBlocks_{chunksToBlocks(kDefaultTotalRenderDistanceChunks)};
    int farDistanceTargetBlocks_{chunksToBlocks(kDefaultTotalRenderDistanceChunks)};
    int fogStartBlocks_{kDefaultFarFogStartBlocks};
    int seaLevel_{20};
    glm::ivec3 cameraChunk_{0};
    glm::vec3 cameraForward_{0.0f, 0.0f, -1.0f};
    mutable Frustum lastVisibilityFrustum_{};
    mutable glm::vec3 lastVisibilityCameraPos_{0.0f};
    mutable bool hasVisibilityFrustum_{false};
    std::uint64_t updateStamp_{0};
    int builtTilesLastUpdate_{0};
    int skippedTilesLastUpdate_{0};
    double lastAverageBuildMs_{0.0};
    double lastAverageGpuSynthesisMs_{0.0};
    double lastAverageGpuStampMs_{0.0};
    double lastAverageGpuFaceBuildMs_{0.0};
    double lastAverageCpuTerrainSynthesisMs_{0.0};
    double lastAverageCpuStructureStampMs_{0.0};
    double lastAverageCpuMeshMs_{0.0};
    double lastAverageUploadWaitMs_{0.0};
    double lastAverageUploadCopyMs_{0.0};
    double lastCollectMs_{0.0};
    double lastUploadMs_{0.0};
    std::size_t lastBuiltFaceCount_{0};
    std::size_t lastBuiltVertexCount_{0};
    std::size_t lastBuiltIndexCount_{0};
    mutable std::size_t lastRenderedFaceCount_{0};
    mutable std::size_t lastRenderedVertexCount_{0};
    std::vector<FarLodLevelConfig> levels_;
    std::vector<BufferPage> bufferPages_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    UploadContext uploadContext_{};
    FarLodGpuContext gpuContext_{};
    Microsoft::WRL::ComPtr<ID3D12Resource> parityReadbackScratch_;
    std::byte* parityReadbackMapped_{nullptr};
    terrain::FarLodWorldgenTables worldgenTables_{};
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenHeaderBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenBiomeBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenBiomeSelectionBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenOceanSelectionBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenTransitionBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenSubBiomeBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenPermutationBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> blockUvBuffer_;
    std::uint32_t blockUvCount_{0};
    Microsoft::WRL::ComPtr<ID3D12Resource> emptyVoxelBuffer_;
    std::unordered_map<int, FarLodLevelAtlasState> levelAtlases_;
    std::uint64_t atlasRevisionCounter_{1};
    mutable std::mutex structureRegionMutex_;
    StructureSampleColumnFn structureSampleColumnFn_{};
    StructureSurfaceBlockFn structureSurfaceBlockFn_{};
    StructureDensityFn structureDensityFn_{};
    std::unordered_map<StructureRegionKey, GpuStructureRegionState, StructureRegionKeyHasher> gpuStructureRegions_;
    std::unordered_map<FarLodChunkKey, FarLodChunkRecord, FarLodChunkKeyHasher> chunks_;
    mutable std::mutex configMutex_;
    ColumnSampleFn columnSampleFn_{};
    mutable std::mutex buildQueueMutex_;
    std::condition_variable buildQueueCv_;
    std::deque<BuildJob> buildQueue_;
    std::unordered_set<FarLodChunkKey, FarLodChunkKeyHasher> queuedKeys_;
    mutable std::mutex completedMutex_;
    std::deque<BuildResult> completedBuilds_;
    std::deque<GpuSynthesisRequest> gpuSynthesisRequests_;
    std::deque<PendingGpuParityReadback> pendingGpuParityReadbacks_;
    std::vector<std::thread> workerThreads_;
    std::thread dirtyBuildPlannerThread_;
    std::thread touchLevelPlannerThread_;
    std::atomic<bool> stopWorkers_{false};
    std::atomic<std::uint64_t> buildEpoch_{1};
    std::atomic<std::uint64_t> dirtyBuildPlanSequence_{0};
    std::atomic<std::uint64_t> touchLevelPlanSequence_{0};
    std::size_t workerCount_{1};
    std::atomic<int> exactMissingChunks_{0};
    std::atomic<std::size_t> exactPendingUploads_{0};
    ChunkBenchmarkMetrics* benchmarkMetrics_{nullptr};
    std::unordered_map<int, int> levelActivationOuterRadiusChunks_;
    mutable std::mutex gpuRequestMutex_;
    mutable std::mutex dirtyBuildPlanMutex_;
    std::condition_variable dirtyBuildPlanCv_;
    DirtyBuildPlanRequest pendingDirtyBuildPlanRequest_{};
    DirtyBuildPlanResult readyDirtyBuildPlanResult_{};
    bool hasPendingDirtyBuildPlanRequest_{false};
    bool hasReadyDirtyBuildPlanResult_{false};
    bool dirtyBuildPlannerBusy_{false};
    mutable std::mutex touchLevelPlanMutex_;
    std::condition_variable touchLevelPlanCv_;
    TouchLevelPlanRequest pendingTouchLevelPlanRequest_{};
    TouchLevelPlanResult readyTouchLevelPlanResult_{};
    bool hasPendingTouchLevelPlanRequest_{false};
    bool hasReadyTouchLevelPlanResult_{false};
    bool touchLevelPlannerBusy_{false};
    bool hasLastTouchLevelPlanRequest_{false};
    glm::ivec3 lastTouchLevelPlanCameraChunk_{0};
    std::vector<std::pair<int, int>> lastTouchLevelPlanLevels_;
    std::unordered_map<int, TouchLevelCacheState> touchLevelCaches_;
    bool gpuParityEnabled_{false};
    std::uint32_t rollingGpuParityMismatchCount_{0};
};

FarTerrainManager::ChunkMesh FarTerrainManager::buildChunkMesh(const FarLodChunkCpu& cpu,
                                                              int lodLevel) const
{
    (void)lodLevel;
    ChunkMesh mesh{};
    mesh.boundsMin = cpu.boundsMin;
    mesh.boundsMax = cpu.boundsMax;

    auto isInsideChunk = [](const glm::ivec3& localCoord) noexcept
    {
        return localCoord.x >= 0 && localCoord.x < kLogicalSize &&
               localCoord.y >= 0 && localCoord.y < kLogicalSize &&
               localCoord.z >= 0 && localCoord.z < kLogicalSize;
    };

    struct LocalCoordLess
    {
        bool operator()(const glm::ivec3& lhs, const glm::ivec3& rhs) const noexcept
        {
            if (lhs.x != rhs.x) return lhs.x < rhs.x;
            if (lhs.y != rhs.y) return lhs.y < rhs.y;
            return lhs.z < rhs.z;
        }
    };

    std::map<glm::ivec3, FarLodVoxel, LocalCoordLess> neighborVoxelCache;
    auto sampleVoxel = [&](const glm::ivec3& localCoord) -> FarLodVoxel
    {
        if (isInsideChunk(localCoord))
        {
            return cpu.voxels[voxelIndex(localCoord.x, localCoord.y, localCoord.z)];
        }

        const auto cached = neighborVoxelCache.find(localCoord);
        if (cached != neighborVoxelCache.end())
        {
            return cached->second;
        }

        const glm::ivec3 chunkOffset{
            floorDiv(localCoord.x, kLogicalSize),
            floorDiv(localCoord.y, kLogicalSize),
            floorDiv(localCoord.z, kLogicalSize)
        };

        const glm::ivec3 wrappedLocal{
            wrapIndex(localCoord.x, kLogicalSize),
            wrapIndex(localCoord.y, kLogicalSize),
            wrapIndex(localCoord.z, kLogicalSize)
        };

        FarLodVoxel sampledVoxel{};

        const FarLodChunkKey neighborKey{
            cpu.key.level,
            cpu.key.coord + chunkOffset
        };

        const auto neighborIt = chunks_.find(neighborKey);
        if (neighborIt != chunks_.end())
        {
            const FarLodChunkRecord& neighborChunk = neighborIt->second;
            if (neighborChunk.cpu.blockScale == cpu.blockScale)
            {
                sampledVoxel = neighborChunk.cpu.voxels[
                    voxelIndex(wrappedLocal.x, wrappedLocal.y, wrappedLocal.z)];
            }
        }

        neighborVoxelCache.emplace(localCoord, sampledVoxel);
        return sampledVoxel;
    };

    auto emitGreedyQuad = [&](int axis, bool positiveFace, int slice, int bStart, int cStart, int bSize, int cSize, const GreedyMaskCell& cell)
    {
        glm::ivec3 normal{0};
        normal[axis] = positiveFace ? 1 : -1;

        glm::vec3 base = glm::vec3(cpu.worldMin);
        base[axis] += static_cast<float>(slice * cpu.blockScale);
        base[(axis + 1) % 3] += static_cast<float>(bStart * cpu.blockScale);
        base[(axis + 2) % 3] += static_cast<float>(cStart * cpu.blockScale);

        glm::vec3 du(0.0f);
        du[(axis + 1) % 3] = static_cast<float>(bSize * cpu.blockScale);
        glm::vec3 dv(0.0f);
        dv[(axis + 2) % 3] = static_cast<float>(cSize * cpu.blockScale);

        glm::vec3 p0 = base;
        glm::vec3 p1 = base + du;
        glm::vec3 p2 = base + du + dv;
        glm::vec3 p3 = base + dv;
        if (!positiveFace)
        {
            std::swap(p1, p3);
        }

        const std::pair<glm::vec2, glm::vec2> uv{glm::vec2(0.0f), glm::vec2(1.0f)};
        const std::uint8_t flags =
            static_cast<std::uint8_t>(kVertexFlagFarLod | ((cell.flags & kFarLodVoxelWater) ? kVertexFlagWater : 0));
        appendQuad(mesh.vertices, mesh.indices, p0, p1, p2, p3, glm::vec3(normal), uv, flags);
    };

    for (int axis = 0; axis < 3; ++axis)
    {
        const int sizeA = kLogicalSize;
        const int sizeB = kLogicalSize;
        const int sizeC = kLogicalSize;
        std::vector<GreedyMaskCell> mask(static_cast<std::size_t>(sizeB * sizeC));
        auto maskIndex = [sizeC](int bIndex, int cIndex) noexcept
        {
            return static_cast<std::size_t>(bIndex * sizeC + cIndex);
        };

        for (int dirIndex = 0; dirIndex < 2; ++dirIndex)
        {
            const bool positiveFace = (dirIndex == 0);
            for (int slice = 0; slice <= sizeA; ++slice)
            {
                std::fill(mask.begin(), mask.end(), GreedyMaskCell{});

                for (int bi = 0; bi < sizeB; ++bi)
                {
                    for (int ci = 0; ci < sizeC; ++ci)
                    {
                        glm::ivec3 positiveLocal{0};
                        glm::ivec3 negativeLocal{0};
                        positiveLocal[axis] = slice;
                        negativeLocal[axis] = slice - 1;
                        positiveLocal[(axis + 1) % 3] = bi;
                        positiveLocal[(axis + 2) % 3] = ci;
                        negativeLocal[(axis + 1) % 3] = bi;
                        negativeLocal[(axis + 2) % 3] = ci;

                        const glm::ivec3 owningLocal = positiveFace ? negativeLocal : positiveLocal;
                        if (!isInsideChunk(owningLocal))
                        {
                            continue;
                        }

                        const FarLodVoxel owningVoxel = sampleVoxel(owningLocal);
                        if (!owningVoxel.occupied || owningVoxel.material == BlockId::Air)
                        {
                            continue;
                        }

                        const FarLodVoxel neighborVoxel = sampleVoxel(positiveFace ? positiveLocal : negativeLocal);
                        if (!shouldRenderBlockFace(owningVoxel.material, neighborVoxel.material))
                        {
                            continue;
                        }

                        GreedyMaskCell& cell = mask[maskIndex(bi, ci)];
                        cell.visible = true;
                        cell.material = owningVoxel.material;
                        cell.flags = static_cast<std::uint8_t>(owningVoxel.flags & (kFarLodVoxelWater |
                                                                                   kFarLodVoxelStructure |
                                                                                   kFarLodVoxelCutout));
                    }
                }

                for (int bi = 0; bi < sizeB; ++bi)
                {
                    int ci = 0;
                    while (ci < sizeC)
                    {
                        GreedyMaskCell& startCell = mask[maskIndex(bi, ci)];
                        if (!startCell.visible)
                        {
                            ++ci;
                            continue;
                        }

                        int runC = 1;
                        while (ci + runC < sizeC)
                        {
                            const GreedyMaskCell& nextCell = mask[maskIndex(bi, ci + runC)];
                            if (!nextCell.visible || !nextCell.mergeEquals(startCell))
                            {
                                break;
                            }
                            ++runC;
                        }

                        int runB = 1;
                        while (bi + runB < sizeB)
                        {
                            bool rowMatches = true;
                            for (int offset = 0; offset < runC; ++offset)
                            {
                                const GreedyMaskCell& rowCell = mask[maskIndex(bi + runB, ci + offset)];
                                if (!rowCell.visible || !rowCell.mergeEquals(startCell))
                                {
                                    rowMatches = false;
                                    break;
                                }
                            }
                            if (!rowMatches)
                            {
                                break;
                            }
                            ++runB;
                        }

                        emitGreedyQuad(axis, positiveFace, slice, bi, ci, runB, runC, startCell);

                        for (int bOffset = 0; bOffset < runB; ++bOffset)
                        {
                            for (int cOffset = 0; cOffset < runC; ++cOffset)
                            {
                                mask[maskIndex(bi + bOffset, ci + cOffset)].visible = false;
                            }
                        }

                        ci += runC;
                    }
                }
            }
        }
    }

    return mesh;
}


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
    void enqueueJob(const std::shared_ptr<Chunk>& chunk,
                    JobType type,
                    const glm::ivec3& coord,
                    std::uint32_t generationEpoch = 0,
                    bool initialReadyPriority = false);
    void processJob(const Job& job);
    std::shared_ptr<Chunk> popNextChunkForUpload();
    void queueChunkForUpload(const std::shared_ptr<Chunk>& chunk);
    void requeueChunkForUpload(const std::shared_ptr<Chunk>& chunk, bool toFront);

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
    std::pair<int, int> columnSpanFor(const glm::ivec2& column,
                                      const glm::ivec2& cameraColumn,
                                      int cameraChunkY,
                                      int verticalRadius) const;
    std::pair<int, int> columnSpanForHeight(const glm::ivec2& column,
                                            const glm::ivec2& cameraColumn,
                                            int cameraChunkY,
                                            int verticalRadius,
                                            int columnHeight) const;
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

    RingProgress ensureVolume(const glm::ivec3& center, int horizontalRadius, int verticalRadius, int& jobBudget);
    void removeDistantChunks(const glm::ivec3& center, int horizontalThreshold, int verticalThreshold);
    bool ensureChunkAsync(const glm::ivec3& coord);
    void uploadReadyMeshes();
    bool uploadChunkMesh(Chunk& chunk, UINT64 uploadBatchId);
    void buildChunkMeshAsync(Chunk& chunk);
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
    ColumnSample sampleColumn(int worldX,
                              int worldZ,
                              int slabMinWorldY = std::numeric_limits<int>::min(),
                              int slabMaxWorldY = std::numeric_limits<int>::max()) const;
    int ensureColumnHeightCached(const glm::ivec2& column, int worldX, int worldZ) const;
    bool tryGetCachedColumnHeight(const glm::ivec2& column, int worldX, int worldZ, int& outHeight) const;
    bool tryGetPredictedColumnHeight(const glm::ivec2& column, int& outHeight) const;
    int cacheSampledColumnHeight(const glm::ivec2& column, int worldX, int worldZ) const;
    void invalidatePredictedColumn(const glm::ivec2& column) const;
    std::vector<PendingStructureEdit> takePendingStructureEdits(const glm::ivec3& coord);
    bool applyPendingStructureEditsLocked(Chunk& chunk);
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
        const std::vector<std::uint8_t>& centerLightLevels) const;

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

    std::deque<std::weak_ptr<Chunk>> uploadQueue_;
    std::mutex uploadQueueMutex_;
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
    std::unordered_map<glm::ivec3, std::vector<PendingStructureEdit>, ChunkHasher> pendingStructureEdits_;
    mutable std::mutex pendingStructureMutex_;

    std::vector<std::thread> workerThreads_;
    std::size_t workerThreadCount_{0};
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
    int generationColumnCapThisFrame_{kVerticalStreamingConfig.maxGenerationJobsPerColumn};
    int lastGenerationBudget_{kVerticalStreamingConfig.generationBudget.baseJobsPerFrame};
    int lastGenerationJobsIssued_{0};
    int lastRingBudget_{kVerticalStreamingConfig.generationBudget.minRingExpansionsPerFrame};
    int lastRingExpansionsUsed_{0};
    int lastMissingChunks_{0};
    int cachedExactReadyChunks_{0};
    int cachedExactRequiredChunks_{0};
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
    std::chrono::steady_clock::time_point lastUpdateTime_{};
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

// JobQueue implementations

bool JobQueue::push(const Job& job)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (shouldStop_.load(std::memory_order_acquire))
    {
        return false;
    }

    queues_[jobTypeIndex(job.type)].push(wrap(job));
    queuedJobCount_.fetch_add(1, std::memory_order_relaxed);
    condition_.notify_one();
    return true;
}

bool JobQueue::tryPop(Job& job)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (!hasQueuedJobsLocked())
    {
        return false;
    }

    const std::size_t queueIndex = pickNextQueueIndexLocked();
    job = queues_[queueIndex].top().job;
    queues_[queueIndex].pop();
    queuedJobCount_.fetch_sub(1, std::memory_order_relaxed);
    ++activeCounts_[queueIndex];
    return true;
}

Job JobQueue::waitAndPop()
{
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this]
    {
        return hasQueuedJobsLocked() || shouldStop_.load(std::memory_order_acquire);
    });

    while (true)
    {
        if (shouldStop_.load(std::memory_order_acquire) && !hasQueuedJobsLocked())
        {
            throw std::runtime_error("Job queue stopped");
        }

        if (!hasQueuedJobsLocked())
        {
            condition_.wait(lock, [this]
            {
                return hasQueuedJobsLocked() || shouldStop_.load(std::memory_order_acquire);
            });
            continue;
        }

        const std::size_t queueIndex = pickNextQueueIndexLocked();
        if (queues_[queueIndex].empty())
        {
            condition_.wait(lock, [this]
            {
                return hasQueuedJobsLocked() || shouldStop_.load(std::memory_order_acquire);
            });
            continue;
        }

        Job job = queues_[queueIndex].top().job;
        queues_[queueIndex].pop();
        queuedJobCount_.fetch_sub(1, std::memory_order_relaxed);
        ++activeCounts_[queueIndex];
        return job;
    }
}

std::vector<Job> JobQueue::stop()
{
    std::vector<Job> cancelledJobs;
    std::lock_guard<std::mutex> lock(mutex_);
    shouldStop_.store(true, std::memory_order_release);
    std::size_t pendingCount = 0;
    for (const auto& queue : queues_)
    {
        pendingCount += queue.size();
    }
    cancelledJobs.reserve(pendingCount);
    for (auto& queue : queues_)
    {
        while (!queue.empty())
        {
            cancelledJobs.push_back(queue.top().job);
            queue.pop();
        }
    }
    queuedJobCount_.store(0, std::memory_order_relaxed);
    condition_.notify_all();
    return cancelledJobs;
}

bool JobQueue::empty() const
{
    return queuedJobCount_.load(std::memory_order_relaxed) == 0;
}

std::size_t JobQueue::size() const
{
    return queuedJobCount_.load(std::memory_order_relaxed);
}

std::size_t JobQueue::size(JobType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return queues_[jobTypeIndex(type)].size();
}

std::size_t JobQueue::outstanding(JobType type) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t index = jobTypeIndex(type);
    return queues_[index].size() + activeCounts_[index];
}

void JobQueue::updatePriorityState(const glm::ivec3& origin, const glm::vec3& forward)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const glm::vec2 forwardXZ = normalizePriorityForwardXZ(forward);
    const float facingDot = glm::dot(priorityForwardXZ_, forwardXZ);
    if (origin == priorityOrigin_ && facingDot >= 0.995f)
    {
        return;
    }

    const glm::ivec3 delta = origin - priorityOrigin_;
    const int horizontalShift = std::max(std::abs(delta.x), std::abs(delta.z));
    const int verticalShift = std::abs(delta.y);

    priorityOrigin_ = origin;
    priorityForwardXZ_ = forwardXZ;

    constexpr int kPriorityRebuildChunkShiftThreshold = 2;
    constexpr int kPriorityRebuildForceShiftThreshold = 6;
    constexpr float kPriorityRebuildFacingDotThreshold = 0.85f;
    constexpr std::size_t kPriorityRebuildQueueThreshold = 128;

    const bool forceRebuild = horizontalShift >= kPriorityRebuildForceShiftThreshold ||
                              verticalShift >= kPriorityRebuildForceShiftThreshold;
    const bool significantMove = horizontalShift >= kPriorityRebuildChunkShiftThreshold ||
                                 verticalShift >= kPriorityRebuildChunkShiftThreshold;
    const bool significantTurn = facingDot <= kPriorityRebuildFacingDotThreshold;
    if (!forceRebuild && !significantMove && !significantTurn)
    {
        return;
    }

    std::size_t queuedJobs = 0;
    for (const auto& queue : queues_)
    {
        queuedJobs += queue.size();
    }

    if (!forceRebuild && queuedJobs > kPriorityRebuildQueueThreshold)
    {
        return;
    }

    rebuildLocked();
}

bool JobQueue::tryUpdatePriorityState(const glm::ivec3& origin, const glm::vec3& forward)
{
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock())
    {
        return false;
    }

    const glm::vec2 forwardXZ = normalizePriorityForwardXZ(forward);
    const float facingDot = glm::dot(priorityForwardXZ_, forwardXZ);
    if (origin == priorityOrigin_ && facingDot >= 0.995f)
    {
        return true;
    }

    const glm::ivec3 delta = origin - priorityOrigin_;
    const int horizontalShift = std::max(std::abs(delta.x), std::abs(delta.z));
    const int verticalShift = std::abs(delta.y);

    priorityOrigin_ = origin;
    priorityForwardXZ_ = forwardXZ;

    constexpr int kPriorityRebuildChunkShiftThreshold = 2;
    constexpr int kPriorityRebuildForceShiftThreshold = 6;
    constexpr float kPriorityRebuildFacingDotThreshold = 0.85f;
    constexpr std::size_t kPriorityRebuildQueueThreshold = 128;

    const bool forceRebuild = horizontalShift >= kPriorityRebuildForceShiftThreshold ||
                              verticalShift >= kPriorityRebuildForceShiftThreshold;
    const bool significantMove = horizontalShift >= kPriorityRebuildChunkShiftThreshold ||
                                 verticalShift >= kPriorityRebuildChunkShiftThreshold;
    const bool significantTurn = facingDot <= kPriorityRebuildFacingDotThreshold;
    if (!forceRebuild && !significantMove && !significantTurn)
    {
        return true;
    }

    std::size_t queuedJobs = 0;
    for (const auto& queue : queues_)
    {
        queuedJobs += queue.size();
    }

    if (!forceRebuild && queuedJobs > kPriorityRebuildQueueThreshold)
    {
        return true;
    }

    rebuildLocked();
    return true;
}

bool JobQueue::JobComparer::operator()(const PrioritizedJob& lhs, const PrioritizedJob& rhs) const
{
    return JobQueue::comparePrioritizedJobs(lhs, rhs) > 0;
}

JobQueue::PrioritizedJob JobQueue::wrap(const Job& job)
{
    const ChunkPriorityKey priority = buildChunkPriorityKey(job.chunkCoord, priorityOrigin_, priorityForwardXZ_);
    const int lifecycleBias = job.initialReadyPriority ? 0 : 1;
    const int bias = (job.type == JobType::Mesh) ? 0 : 1;
    const std::uint64_t sequence = nextSequence_++;
    return PrioritizedJob{job, priority, lifecycleBias, bias, sequence};
}

int JobQueue::comparePrioritizedJobs(const PrioritizedJob& lhs,
                                     const PrioritizedJob& rhs) noexcept
{
    if (lhs.lifecycleBias != rhs.lifecycleBias)
    {
        return lhs.lifecycleBias < rhs.lifecycleBias ? -1 : 1;
    }
    const int priorityComparison = compareChunkPriorityKeys(lhs.priority, rhs.priority);
    if (priorityComparison != 0)
    {
        return priorityComparison;
    }
    if (lhs.stageBias != rhs.stageBias)
    {
        return lhs.stageBias < rhs.stageBias ? -1 : 1;
    }
    if (lhs.sequence != rhs.sequence)
    {
        return lhs.sequence < rhs.sequence ? -1 : 1;
    }
    return 0;
}

bool JobQueue::hasQueuedJobsLocked() const noexcept
{
    for (const auto& queue : queues_)
    {
        if (!queue.empty())
        {
            return true;
        }
    }
    return false;
}

std::array<std::size_t, kJobTypeCount> JobQueue::computeStageTargetsLocked() const noexcept
{
    std::array<std::size_t, kJobTypeCount> targets{1, 1};
    const std::size_t totalWorkers = std::max<std::size_t>(workerConcurrency_, 1);
    if (totalWorkers <= 1)
    {
        return targets;
    }

    const std::size_t generateBacklog = queues_[jobTypeIndex(JobType::Generate)].size();
    const std::size_t meshBacklog = queues_[jobTypeIndex(JobType::Mesh)].size();
    const bool generateInitialReadyTop =
        generateBacklog > 0 && queues_[jobTypeIndex(JobType::Generate)].top().lifecycleBias == 0;
    const bool meshInitialReadyTop =
        meshBacklog > 0 && queues_[jobTypeIndex(JobType::Mesh)].top().lifecycleBias == 0;

    double meshShare = 0.5;
    if (generateInitialReadyTop && meshInitialReadyTop)
    {
        if (generateBacklog > meshBacklog * 2)
        {
            meshShare = 0.35;
        }
        else if (generateBacklog > meshBacklog)
        {
            meshShare = 0.40;
        }
        else if (meshBacklog > generateBacklog * 2)
        {
            meshShare = 0.60;
        }
        else if (meshBacklog > generateBacklog)
        {
            meshShare = 0.50;
        }
        else
        {
            meshShare = 0.45;
        }
    }
    else if (meshBacklog == 0 && generateBacklog > 0)
    {
        meshShare = 0.4;
    }
    else if (generateBacklog == 0 && meshBacklog > 0)
    {
        meshShare = 0.8;
    }
    else if (meshBacklog > generateBacklog * 2)
    {
        meshShare = 0.8;
    }
    else if (generateBacklog > meshBacklog * 2)
    {
        meshShare = 0.45;
    }
    else if (meshBacklog > generateBacklog)
    {
        meshShare = 0.65;
    }
    else if (generateBacklog > meshBacklog)
    {
        meshShare = 0.5;
    }

    std::size_t meshTarget = static_cast<std::size_t>(std::llround(static_cast<double>(totalWorkers) * meshShare));
    meshTarget = std::clamp<std::size_t>(meshTarget, 1, totalWorkers - 1);
    const std::size_t generateTarget = std::max<std::size_t>(1, totalWorkers - meshTarget);
    targets[jobTypeIndex(JobType::Generate)] = generateTarget;
    targets[jobTypeIndex(JobType::Mesh)] = meshTarget;
    return targets;
}

std::size_t JobQueue::pickNextQueueIndexLocked() const noexcept
{
    const std::size_t generateIndex = jobTypeIndex(JobType::Generate);
    const std::size_t meshIndex = jobTypeIndex(JobType::Mesh);
    const bool generateReady = !queues_[generateIndex].empty();
    const bool meshReady = !queues_[meshIndex].empty();

    if (!meshReady)
    {
        return generateIndex;
    }
    if (!generateReady)
    {
        return meshIndex;
    }

    const std::array<std::size_t, kJobTypeCount> targets = computeStageTargetsLocked();
    const bool generateUnderTarget = activeCounts_[generateIndex] < targets[generateIndex];
    const bool meshUnderTarget = activeCounts_[meshIndex] < targets[meshIndex];

    if (meshUnderTarget != generateUnderTarget)
    {
        return meshUnderTarget ? meshIndex : generateIndex;
    }

    const PrioritizedJob& generateTop = queues_[generateIndex].top();
    const PrioritizedJob& meshTop = queues_[meshIndex].top();
    if (generateTop.lifecycleBias != meshTop.lifecycleBias)
    {
        return (generateTop.lifecycleBias < meshTop.lifecycleBias) ? generateIndex : meshIndex;
    }
    return comparePrioritizedJobs(meshTop, generateTop) <= 0 ? meshIndex : generateIndex;
}

void JobQueue::rebuildLocked()
{
    if (!hasQueuedJobsLocked())
    {
        return;
    }

    for (auto& queue : queues_)
    {
        if (queue.empty())
        {
            continue;
        }

        std::vector<PrioritizedJob> jobs;
        jobs.reserve(queue.size());
        while (!queue.empty())
        {
            jobs.push_back(queue.top());
            queue.pop();
        }

        for (auto& prioritized : jobs)
        {
            prioritized.priority = buildChunkPriorityKey(prioritized.job.chunkCoord, priorityOrigin_, priorityForwardXZ_);
            queue.push(std::move(prioritized));
        }
    }
}

void JobQueue::setWorkerConcurrency(std::size_t workerCount) noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    workerConcurrency_ = std::max<std::size_t>(workerCount, 1);
    condition_.notify_all();
}

void JobQueue::jobCompleted(JobType type) noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t index = jobTypeIndex(type);
    if (activeCounts_[index] > 0)
    {
        --activeCounts_[index];
    }
    condition_.notify_one();
}

glm::ivec2 ColumnManager::columnKey(const glm::ivec3& chunkCoord, int localX, int localZ) noexcept
{
    return {chunkCoord.x * kChunkSizeX + localX, chunkCoord.z * kChunkSizeZ + localZ};
}

int ColumnManager::scanColumnHighestWorld(const Chunk& chunk, int localX, int localZ) noexcept
{
    for (int y = kChunkSizeY - 1; y >= 0; --y)
    {
        if (isSolid(chunk.blocks[blockIndex(localX, y, localZ)]))
        {
            return chunk.minWorldY + y;
        }
    }
    return kNoHeight;
}

int ColumnManager::computeHighest(const ColumnData& data) noexcept
{
    int highest = kNoHeight;
    for (const auto& entry : data.slabHeights)
    {
        highest = std::max(highest, entry.second);
    }
    return highest;
}

void ColumnManager::applyHeightLocked(const glm::ivec2& key, int chunkY, int highestWorldY)
{
    if (highestWorldY == kNoHeight)
    {
        auto it = columns_.find(key);
        if (it == columns_.end())
        {
            return;
        }

        it->second.slabHeights.erase(chunkY);
        if (it->second.slabHeights.empty())
        {
            columns_.erase(it);
        }
        else
        {
            it->second.highestWorldY = computeHighest(it->second);
        }
        return;
    }

    auto [it, inserted] = columns_.try_emplace(key);
    it->second.slabHeights[chunkY] = highestWorldY;
    it->second.highestWorldY = computeHighest(it->second);
}

void ColumnManager::updateChunk(const Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (int x = 0; x < kChunkSizeX; ++x)
    {
        for (int z = 0; z < kChunkSizeZ; ++z)
        {
            const glm::ivec2 key = columnKey(chunk.coord, x, z);
            const int highestWorld = scanColumnHighestWorld(chunk, x, z);
            applyHeightLocked(key, chunk.coord.y, highestWorld);
        }
    }
}

inline void storeFirstBenchmarkTimestamp(std::atomic<long long>& current, std::uint64_t micros) noexcept
{
    long long expected = 0;
    const long long value = static_cast<long long>(micros);
    current.compare_exchange_strong(expected,
                                    value,
                                    std::memory_order_relaxed,
                                    std::memory_order_relaxed);
}

[[nodiscard]] inline std::uint64_t loadBenchmarkTimestamp(const std::atomic<long long>& current) noexcept
{
    const long long value = current.load(std::memory_order_relaxed);
    return value > 0 ? static_cast<std::uint64_t>(value) : 0u;
}

void ColumnManager::updateChunkHeights(
    const glm::ivec3& chunkCoord,
    const std::array<int, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)>& highestWorlds)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (int x = 0; x < kChunkSizeX; ++x)
    {
        for (int z = 0; z < kChunkSizeZ; ++z)
        {
            applyHeightLocked(columnKey(chunkCoord, x, z), chunkCoord.y, highestWorlds[columnIndex(x, z)]);
        }
    }
}

void ColumnManager::updateColumn(const Chunk& chunk, int localX, int localZ)
{
    const int highestWorld = scanColumnHighestWorld(chunk, localX, localZ);
    std::lock_guard<std::mutex> lock(mutex_);
    applyHeightLocked(columnKey(chunk.coord, localX, localZ), chunk.coord.y, highestWorld);
}

void ColumnManager::removeChunk(const Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (int x = 0; x < kChunkSizeX; ++x)
    {
        for (int z = 0; z < kChunkSizeZ; ++z)
        {
            applyHeightLocked(columnKey(chunk.coord, x, z), chunk.coord.y, kNoHeight);
        }
    }
}

void ColumnManager::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    columns_.clear();
}

int ColumnManager::highestSolidBlock(int worldX, int worldZ) const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    const glm::ivec2 key{worldX, worldZ};
    auto it = columns_.find(key);
    if (it == columns_.end())
    {
        return kNoHeight;
    }
    return it->second.highestWorldY;
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

    climateMap_ = std::make_unique<terrain::ClimateMap>(std::move(climateGenerator), 256);

    surfaceMap_ = std::make_unique<terrain::SurfaceMap>(
        std::make_unique<terrain::MapGenV1>(biomeDatabase_, *climateMap_, worldgenProfile_, effectiveSeed),
        256);

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

    const auto now = std::chrono::steady_clock::now();
    double frameSeconds = 1.0 / 60.0;
    if (lastUpdateTime_.time_since_epoch().count() != 0)
    {
        frameSeconds = std::chrono::duration<double>(now - lastUpdateTime_).count();
    }
    lastUpdateTime_ = now;
    frameSeconds = std::clamp(frameSeconds, 1.0 / 240.0, 0.25);
    smoothedFrameMs_ = smoothedFrameMs_ * 0.90 + frameSeconds * 1000.0 * 0.10;

    const int worldX = static_cast<int>(std::floor(cameraPos.x));
    const int worldY = static_cast<int>(std::floor(cameraPos.y));
    const int worldZ = static_cast<int>(std::floor(cameraPos.z));
    const int clampedWorldY = std::max(worldY, 0);
    const glm::ivec3 centerChunk = worldToChunkCoords(worldX, clampedWorldY, worldZ);
    lastCenterChunk_ = centerChunk;
    {
        std::lock_guard<std::mutex> lock(schedulingPriorityMutex_);
        schedulingPriorityOrigin_ = centerChunk;
        schedulingPriorityForward_ = lastCameraForward_;
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

    const glm::vec2 desiredPriorityForwardXZ = normalizePriorityForwardXZ(lastCameraForward_);
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
        const bool updatedPriority = jobQueue_.tryUpdatePriorityState(centerChunk, lastCameraForward_);
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

    const auto missingScanStart = std::chrono::steady_clock::now();
    const VisibleChunkCoverage visibleCoverage =
        scanVisibleChunkCoverage(centerChunk, targetViewDistance_, verticalRadius);
    const int missingChunks = visibleCoverage.missing;
    missingScanMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - missingScanStart).count();
    const int backlogSteps = computeBacklogSteps(missingChunks,
                                                 kVerticalStreamingConfig.generationBudget.backlogStartThreshold,
                                                 kVerticalStreamingConfig.generationBudget.backlogStepSize);
    int columnCap = computeColumnJobCap(backlogSteps, missingChunks);
    if (columnCap <= 0)
    {
        columnCap = std::numeric_limits<int>::max();
    }

    generationColumnCapThisFrame_ = columnCap;

    const int generationBudgetTarget =
        computeGenerationBudget(targetViewDistance_, verticalRadius, backlogSteps);
    const int ringBudget = computeRingExpansionBudget(missingChunks);

    const std::size_t workerSlots = std::max<std::size_t>(workerThreadCount_, 1);
    const std::size_t outstandingGenerateCap = std::clamp<std::size_t>(
        64u + workerSlots * 24u + static_cast<std::size_t>(std::max(verticalRadius, 0)) * 4u,
        64u,
        384u);
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
    const auto timedEnsureVolume = [&](int horizontalRadius)
    {
        if (!benchmarkEnabled)
        {
            return ensureVolume(centerChunk, horizontalRadius, verticalRadius, jobBudget);
        }

        const auto ensureVolumeStart = std::chrono::steady_clock::now();
        RingProgress progress = ensureVolume(centerChunk, horizontalRadius, verticalRadius, jobBudget);
        ensureVolumeMsLastFrame_ +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ensureVolumeStart).count();
        return progress;
    };

    const auto schedulingStart = std::chrono::steady_clock::now();
    for (int ring = 0; ring <= viewDistance_ && jobBudget > 0; ++ring)
    {
        RingProgress progress = timedEnsureVolume(ring);
        if (progress.budgetExhausted)
        {
            break;
        }
    }

    int ringsExpanded = 0;
    while (jobBudget > 0 && viewDistance_ < preloadTargetViewDistance && ringsExpanded < ringBudget)
    {
        const int nextRing = viewDistance_ + 1;
        RingProgress progress = timedEnsureVolume(nextRing);

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
    relightMsLastFrame_ = 0.0;
    if (allowMainThreadRelightWindow)
    {
        const auto relightStart = std::chrono::steady_clock::now();
        const bool allowMainThreadRelightNow =
            workerThreadCount_ == 0 ||
            activeRelightProcessors_.load(std::memory_order_acquire) == 0;
        if (allowMainThreadRelightNow)
        {
            processPendingRelightRequests(1);
        }
        relightMsLastFrame_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - relightStart).count();
    }
    uploadReadyMeshes();
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
        const int farWorkerBudget =
            (missingChunks > 32 || exactPendingUploads > 24) ? 1 :
            ((missingChunks > 8 || exactPendingUploads > 8) ? std::min(farWorkerCount_, 2) : farWorkerCount_);
        farTerrainManager_.setEnabled(true);
        farTerrainManager_.setDistanceBlocks(chunksToBlocks(renderSettings_.totalChunks));
        farTerrainManager_.setSeaLevel(globalSeaLevel_);
        farTerrainManager_.setWorkerCount(static_cast<std::size_t>(std::max(farWorkerBudget, 1)));
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
                startupState_.phase = StreamingPhase::InteractiveNearOnly;
                startupState_.phaseTimeSeconds = 0.0;
                startupState_.healthyTimeSeconds = 0.0;
                startupState_.exactNearCurrentChunks = std::min(renderSettings_.exactChunks, 6);
                startupState_.playerReleaseReady = true;
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
        uploadContextBeginMsLastFrame_ +
        uploadFinalizeMsLastFrame_ +
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
    const float exactDrawRadiusBlocks = static_cast<float>(chunksToBlocks(std::max(exactDrawRadiusChunks, 0)));
    const float exactDrawRadiusSq = exactDrawRadiusBlocks * exactDrawRadiusBlocks;

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

        // Keep exact terrain alive a little past the user-facing horizon so fog hides the
        // cutoff instead of the cutoff defining the visible edge.
        if (horizontalDistanceSqToAabb2D(lastCameraPosition_.x,
                                         lastCameraPosition_.z,
                                         minCorner.x,
                                         minCorner.z,
                                         maxCorner.x,
                                         maxCorner.z) > exactDrawRadiusSq)
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
    return sampleColumn(worldX, worldZ, slabMinWorldY, slabMaxWorldY);
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
            columnManager_.removeChunk(*chunk);
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
        uploadQueue_.clear();
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
    {
        std::lock_guard<std::mutex> lock(pendingStructureMutex_);
        pendingStructureEdits_.clear();
    }

    uploadBudgetBytesThisFrame_ = kUploadBudgetBytesPerFrame;
    uploadColumnLimitThisFrame_ = kVerticalStreamingConfig.uploadBasePerColumn;
    lastUploadBytesUsed_ = 0;
    pendingUploadsLastFrame_ = 0;
    lastMissingChunks_ = 0;
    cachedExactReadyChunks_ = 0;
    cachedExactRequiredChunks_ = 0;
    lastJobQueuePriorityOrigin_ = glm::ivec3{0};
    lastJobQueuePriorityForwardXZ_ = glm::vec2{0.0f, -1.0f};
    lastJobQueuePriorityRefreshTime_ = SteadyClock::time_point{};

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


    {
        std::lock_guard<std::mutex> lock(chunk->meshMutex);
        if (!isSolid(chunk->blocks[blockIdx]))
        {
            return false;
        }

        chunk->blocks[blockIdx] = BlockId::Air;
        if (chunk->hasBlocks.load(std::memory_order_relaxed))
        {
            chunk->hasBlocks.store(chunkHasSolidBlocks(*chunk), std::memory_order_relaxed);
        }

        columnManager_.updateColumn(*chunk, local.x, local.z);
    }

    invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});
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


    {
        std::lock_guard<std::mutex> lock(chunk->meshMutex);
        if (isSolid(chunk->blocks[blockIdx]))
        {
            return false;
        }

        chunk->blocks[blockIdx] = block;
        chunk->hasBlocks.store(true, std::memory_order_relaxed);

        columnManager_.updateColumn(*chunk, local.x, local.z);
    }

    invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});
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
    snapshot.uploadColumnLimit = uploadColumnLimitThisFrame_;
    snapshot.updateMsLastFrame = updateMsLastFrame_;
    snapshot.updateResidualMsLastFrame = updateResidualMsLastFrame_;
    snapshot.verticalRadiusMsLastFrame = verticalRadiusMsLastFrame_;
    snapshot.priorityUpdateMsLastFrame = priorityUpdateMsLastFrame_;
    snapshot.uploadBudgetMsLastFrame = uploadBudgetPrepMsLastFrame_;
    snapshot.missingScanMsLastFrame = missingScanMsLastFrame_;
    snapshot.ensureVolumeMsLastFrame = ensureVolumeMsLastFrame_;
    snapshot.schedulingMsLastFrame = schedulingMsLastFrame_;
    snapshot.evictionMsLastFrame = evictionMsLastFrame_;
    snapshot.relightMsLastFrame = relightMsLastFrame_;
    snapshot.uploadMsLastFrame = lastUploadMsUsed_;
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
    if (concurrency >= 12)
    {
        desired = 4u;
    }
    else if (concurrency >= 8)
    {
        desired = 3u;
    }
    else
    {
        desired = std::max(1u, concurrency > 3 ? concurrency - 3 : 1u);
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
}

void ChunkManager::Impl::stopWorkerThreads()
{
    shouldStop_.store(true, std::memory_order_release);
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
    workerThreadCount_ = 0;
}

void ChunkManager::Impl::workerThreadFunction()
{
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
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
    if (uploadQueue_.empty())
    {
        return nullptr;
    }

    constexpr std::size_t kUploadPriorityScanLimit = 48;
    auto bestIt = uploadQueue_.end();
    std::shared_ptr<Chunk> bestChunk;
    std::size_t liveEntriesScanned = 0;
    for (auto it = uploadQueue_.begin();
         it != uploadQueue_.end() && liveEntriesScanned < kUploadPriorityScanLimit;)
    {
        std::shared_ptr<Chunk> chunk = it->lock();
        if (!chunk)
        {
            it = uploadQueue_.erase(it);
            continue;
        }

        const bool chunkInitialReady = chunkAwaitingInitialVisibleReady(*chunk);
        const bool bestInitialReady = bestChunk && chunkAwaitingInitialVisibleReady(*bestChunk);
        if (!bestChunk ||
            (chunkInitialReady != bestInitialReady
                 ? chunkInitialReady
                 : isChunkCoordHigherPriority(chunk->coord, bestChunk->coord, priorityOrigin, priorityForward)))
        {
            bestIt = it;
            bestChunk = chunk;
        }

        ++liveEntriesScanned;
        ++it;
    }

    if (!bestChunk || bestIt == uploadQueue_.end())
    {
        return nullptr;
    }

    uploadQueue_.erase(bestIt);
    bestChunk->queuedForUpload.store(false, std::memory_order_release);
    return bestChunk;
}

void ChunkManager::Impl::queueChunkForUpload(const std::shared_ptr<Chunk>& chunk)
{
    if (!chunk)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    if (chunk->queuedForUpload.load(std::memory_order_acquire))
    {
        return;
    }

    if (chunkAwaitingInitialVisibleReady(*chunk))
    {
        uploadQueue_.emplace_front(chunk);
    }
    else
    {
        uploadQueue_.emplace_back(chunk);
    }
    chunk->queuedForUpload.store(true, std::memory_order_release);
    if (benchmarkMetrics_.isEnabled())
    {
        storeFirstBenchmarkTimestamp(chunk->uploadQueuedTimestampMicros, steadyMicrosNow());
    }

    if (shouldTrackRecentEditChunk(chunk->coord))
    {
        std::ostringstream stream;
        stream << "queue upload chunk=(" << chunk->coord.x << ", " << chunk->coord.y << ", " << chunk->coord.z
               << ") idx=" << chunk->indexCount.load(std::memory_order_acquire);
        appendRecentEditDebugEvent(stream.str());
    }
}

void ChunkManager::Impl::requeueChunkForUpload(const std::shared_ptr<Chunk>& chunk, bool toFront)
{
    if (!chunk)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    if (chunk->queuedForUpload.load(std::memory_order_acquire))
    {
        return;
    }

    if (toFront || chunkAwaitingInitialVisibleReady(*chunk))
    {
        uploadQueue_.emplace_front(chunk);
    }
    else
    {
        uploadQueue_.emplace_back(chunk);
    }
    chunk->queuedForUpload.store(true, std::memory_order_release);
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
    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        releaseChunkAllocation(chunk);
        pendingMesh = chunk.pendingMesh;
        chunk.pendingMesh = {};
        chunk.meshData.clear();
        chunk.meshReady.store(false, std::memory_order_release);
        chunk.queuedForUpload.store(false, std::memory_order_release);
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
    return uploadQueue_.size();
}

std::size_t ChunkManager::Impl::estimateInitialReadyUploadQueueSize()
{
    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    std::size_t readyCount = 0;
    for (const std::weak_ptr<Chunk>& weakChunk : uploadQueue_)
    {
        const std::shared_ptr<Chunk> chunk = weakChunk.lock();
        if (chunk && chunkAwaitingInitialVisibleReady(*chunk))
        {
            ++readyCount;
        }
    }
    return readyCount;
}

ChunkManager::Impl::UploadBudgets ChunkManager::Impl::computeUploadBudgets(int verticalRadius)
{
    UploadBudgets budgets{};
    budgets.columnLimit = baseUploadsPerColumnLimit(verticalRadius);
    budgets.chunkLimit = 3;
    budgets.queueSize = estimateUploadQueueSize();
    const std::size_t initialReadyUploads = estimateInitialReadyUploadQueueSize();
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
            budgets.byteBudget = 16ull * 1024ull * 1024ull;
            budgets.columnLimit = std::min(budgets.columnLimit + 2, 8);
            budgets.chunkLimit = 4;
            budgets.timeBudgetMs = 2.0;
        }
        else if (interactiveUploadWindow)
        {
            budgets.byteBudget = 16ull * 1024ull * 1024ull;
            budgets.columnLimit = std::min(budgets.columnLimit + 1, 7);
            budgets.chunkLimit = 3;
            budgets.timeBudgetMs = 1.5;
        }
        else
        {
            budgets.byteBudget = 20ull * 1024ull * 1024ull;
            budgets.chunkLimit = 3;
            budgets.timeBudgetMs = 2.0;
        }
    }
    else
    {
        budgets.byteBudget = 20ull * 1024ull * 1024ull;
        budgets.chunkLimit = 3;
        budgets.timeBudgetMs = 2.0;
    }

    if (uploadDebtSteps > 0)
    {
        if (exactPreload)
        {
            const int clampedSteps = std::min(uploadDebtSteps, 2);
            budgets.byteBudget += 4ull * 1024ull * 1024ull * static_cast<std::size_t>(clampedSteps);
            budgets.chunkLimit += clampedSteps;
            budgets.columnLimit = std::min(budgets.columnLimit + 1, 10);
            budgets.timeBudgetMs = std::min(2.5, budgets.timeBudgetMs + 0.25 * static_cast<double>(clampedSteps));
        }
        else if (interactiveUploadWindow)
        {
            budgets.byteBudget += 4ull * 1024ull * 1024ull;
            budgets.chunkLimit = std::min(budgets.chunkLimit + 1, 4);
            budgets.columnLimit = std::min(budgets.columnLimit + 1, 8);
        }
        else
        {
            const int clampedSteps = std::min(uploadDebtSteps, 2);
            budgets.byteBudget += 4ull * 1024ull * 1024ull * static_cast<std::size_t>(clampedSteps);
            budgets.chunkLimit += clampedSteps;
            budgets.columnLimit = std::min(budgets.columnLimit + 1, 10);
            budgets.timeBudgetMs = std::min(2.5, budgets.timeBudgetMs + 0.25 * static_cast<double>(clampedSteps));
        }
    }

    if (initialReadyUploads > 0)
    {
        const int urgencySteps = std::min<int>(static_cast<int>(initialReadyUploads), 2);
        budgets.byteBudget += 4ull * 1024ull * 1024ull * static_cast<std::size_t>(urgencySteps);
        budgets.chunkLimit = std::min(budgets.chunkLimit + urgencySteps, 6);
        budgets.columnLimit = std::min(budgets.columnLimit + 1, 10);
        budgets.timeBudgetMs = std::min(3.0, budgets.timeBudgetMs + 0.35 * static_cast<double>(urgencySteps));
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
            const int worldX = chunkX * kChunkSizeX + kChunkSizeX / 2;
            const int worldZ = chunkZ * kChunkSizeZ + kChunkSizeZ / 2;

            int columnHeight = ColumnManager::kNoHeight;
            tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight);

            const auto [minChunkY, maxChunkY] = columnSpanForHeight(column,
                                                                    cameraColumn,
                                                                    cameraChunkY,
                                                                    verticalRadius,
                                                                    columnHeight);
            for (int chunkY = minChunkY; chunkY <= maxChunkY; ++chunkY)
            {
                ++coverage.required;
                const auto stateIt = chunkStates.find(glm::ivec3{chunkX, chunkY, chunkZ});
                if (stateIt == chunkStates.end())
                {
                    ++coverage.missing;
                    continue;
                }

                const ChunkState state = stateIt->second;
                if (state == ChunkState::Uploaded || state == ChunkState::Ready || state == ChunkState::Remeshing)
                {
                    ++coverage.ready;
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
            tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight);

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
    if (highest != ColumnManager::kNoHeight)
    {
        outHeight = highest;
        return true;
    }

    return tryGetPredictedColumnHeight(column, outHeight);
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
    {
        std::lock_guard<std::mutex> lock(predictedColumnMutex_);
        predictedColumnHeights_[column] = height;
    }
    return height;
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
                columnHeight = sampleColumn(worldX, worldZ).surfaceY;
                info.heightSource = "sample";
            }
        }
        info.columnHeight = columnHeight;

        const auto [minChunkY, maxChunkY] =
            columnSpanForHeight(column, cameraColumn, cameraChunkY, lastVerticalRadius_, columnHeight);
        info.columnMinChunkY = minChunkY;
        info.columnMaxChunkY = maxChunkY;

        const int horizontalDistance = std::max(std::abs(coord.x - centerChunk.x), std::abs(coord.z - centerChunk.z));
        info.wouldEvict = coord.y < 0 ||
                          horizontalDistance > horizontalThreshold ||
                          coord.y < (minChunkY - kVerticalStreamingConfig.columnSlackChunks) ||
                          coord.y > (maxChunkY + kVerticalStreamingConfig.columnSlackChunks);

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
    const int columnHeight = ensureColumnHeightCached(column, worldX, worldZ);
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

std::pair<int, int> ChunkManager::Impl::columnSpanFor(const glm::ivec2& column,
                                                       const glm::ivec2& cameraColumn,
                                                       int cameraChunkY,
                                                       int verticalRadius) const
{
    const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
    const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
    const int columnHeight = ensureColumnHeightCached(column, worldX, worldZ);
    return columnSpanForHeight(column, cameraColumn, cameraChunkY, verticalRadius, columnHeight);
}

std::pair<int, int> ChunkManager::Impl::columnSpanForHeight(const glm::ivec2& column,
                                                             const glm::ivec2& cameraColumn,
                                                             int cameraChunkY,
                                                             int verticalRadius,
                                                             int columnHeight) const
{
    const int radius = columnRadiusForHeight(column, cameraColumn, cameraChunkY, verticalRadius, columnHeight);
    const int minChunk = std::max(0, cameraChunkY - radius);
    const int maxChunk = std::max(minChunk, cameraChunkY + radius);
    return {minChunk, maxChunk};
}

ChunkManager::Impl::RingProgress ChunkManager::Impl::ensureVolume(const glm::ivec3& center,
                                                                  int horizontalRadius,
                                                                  int verticalRadius,
                                                                  int& jobBudget)
{
    bool missingFound = false;

    const glm::ivec2 cameraColumn{center.x, center.z};

    struct Candidate
    {
        glm::ivec3 coord;
        float priority{0.0f};
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
        glm::ivec2 column{chunkX, chunkZ};
        if (!visitedColumns.insert(column).second)
        {
            return;
        }

        const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
        const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
        int columnHeight = ColumnManager::kNoHeight;
        tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight);
        const auto [minChunkY, maxChunkY] = columnSpanForHeight(column,
                                                                cameraColumn,
                                                                center.y,
                                                                verticalRadius,
                                                                columnHeight);
        for (int chunkY = minChunkY; chunkY <= maxChunkY; ++chunkY)
        {
            const glm::ivec3 coord{chunkX, chunkY, chunkZ};
            const int dx = coord.x - center.x;
            const int dy = coord.y - center.y;
            const int dz = coord.z - center.z;
            const float horizontal = std::sqrt(static_cast<float>(dx * dx + dz * dz));
            const float priority = horizontal + 0.5f * static_cast<float>(std::abs(dy));
            candidates.push_back(Candidate{coord, priority});
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

        if (ensureChunkAsync(candidate.coord))
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
            tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight);
            const auto [minChunkY, maxChunkY] = columnSpanForHeight(column,
                                                                    cameraColumn,
                                                                    evictionCenterY,
                                                                    verticalRadius,
                                                                    columnHeight);
            int verticalExcess = 0;
            if (coord.y < (minChunkY - evictionSlack))
            {
                verticalExcess = (minChunkY - evictionSlack) - coord.y;
            }
            else if (coord.y > (maxChunkY + evictionSlack))
            {
                verticalExcess = coord.y - (maxChunkY + evictionSlack);
            }

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
                tryGetCachedColumnHeight(column, worldX, worldZ, columnHeight);

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

            columnManager_.removeChunk(*chunk);
            invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});
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

bool ChunkManager::Impl::ensureChunkAsync(const glm::ivec3& coord)
{
    if (coord.y < 0)
    {
        return false;
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
        lastUploadBytesUsed_ = 0;
        lastUploadMsUsed_ = 0.0;
        pendingUploadsLastFrame_ = estimateUploadQueueSize();
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
            continue;
        }

        {
            std::lock_guard<std::mutex> meshLock(chunk->meshMutex);
            if (chunk->pendingMesh.valid())
            {
                requeueChunkForUpload(chunk, false);
                continue;
            }
        }

        const glm::ivec2 columnKey{chunk->coord.x, chunk->coord.z};
        int& columnUploads = uploadsPerColumn[columnKey];
        if (columnUploads >= columnUploadLimit)
        {
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

            std::lock_guard<std::mutex> meshLock(chunk->meshMutex);
            if (chunk->pendingMesh.valid() && chunk->pendingMesh.uploadFenceValue == 0)
            {
                chunk->pendingMesh.uploadFenceValue = submittedFenceValue;
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

    const auto commitChunkScanStart =
        benchmarkEnabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
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
    if (benchmarkEnabled)
    {
        commitChunkScanMsLastFrame_ +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - commitChunkScanStart).count();
    }

    for (const std::shared_ptr<Chunk>& chunk : chunks)
    {
        if (!chunk)
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

        pendingMesh = chunk->pendingMesh;
        chunk->pendingMesh = {};
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
                queueChunkForUpload(chunk);
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
    if (benchmarkEnabled)
    {
        storeFirstBenchmarkTimestamp(chunk.uploadStartTimestampMicros, steadyMicrosNow());
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
    const std::vector<std::uint8_t>& centerLightLevels) const
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

    std::unordered_map<glm::ivec3, std::shared_ptr<const Chunk>, ChunkHasher> neighborhoodChunks;
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

    std::vector<std::shared_ptr<const Chunk>> lockedNeighbors;
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
              [](const std::shared_ptr<const Chunk>& lhs, const std::shared_ptr<const Chunk>& rhs)
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
        bool mergeable{true};

        bool operator==(const FaceMaterial& other) const noexcept
        {
            return uvBase == other.uvBase &&
                   uvSize == other.uvSize &&
                   uAxis == other.uAxis &&
                   vAxis == other.vAxis &&
                   face == other.face &&
                   lightingData == other.lightingData &&
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
	        material.mergeable = !isAlphaCutoutBlock(block);
	        const BlockFace face = faceFromNormal(normal);

	        material.face = face;
            material.lightingData = buildCornerLighting(face, owningLocal);

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
                std::swap(cornerLighting[1], cornerLighting[3]);
	        }

            const int diagonal02 =
                lightingMetricFromPackedVertex(cornerLighting[0]) +
                lightingMetricFromPackedVertex(cornerLighting[2]);
            const int diagonal13 =
                lightingMetricFromPackedVertex(cornerLighting[1]) +
                lightingMetricFromPackedVertex(cornerLighting[3]);
            const bool flipDiagonal = diagonal13 > diagonal02;

	        const glm::vec3 uAxisVec = glm::vec3(material.uAxis);
	        const glm::vec3 vAxisVec = glm::vec3(material.vAxis);

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
	            vertex.lightingData = cornerLighting[i];
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

        for (const auto& [column, maxChunkY] : maxLoadedChunkYByColumn)
        {
            for (int chunkY = maxChunkY + 1;; ++chunkY)
            {
                const glm::ivec3 coord(column.x, chunkY, column.y);
                if (chunks_.find(coord) == chunks_.end())
                {
                    break;
                }

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
    std::size_t pendingRegionCount = 0;
    {
        std::lock_guard<std::mutex> lock(relightStateMutex_);
        pendingRegionCount = pendingRelightRegions_.size();
    }

    std::uint64_t budget = kRelightBaseBudgetUnits;
    budget += std::min<std::uint64_t>(workerThreadCount_, 4u) * kRelightPerWorkerBudgetUnits;
    budget += std::min<std::uint64_t>(pendingRegionCount * kRelightBacklogBudgetUnitsPerRegion,
                                      kRelightMaxBudgetUnits / 2);

    const std::uint64_t uploadPenalty =
        std::min<std::uint64_t>(pendingUploadsLastFrame_ * 2048ull, kRelightBaseBudgetUnits / 2);
    budget = (budget > uploadPenalty) ? (budget - uploadPenalty / 2) : budget;
    return std::clamp<std::uint64_t>(budget, kRelightMinBudgetUnits, kRelightMaxBudgetUnits);
}

int ChunkManager::Impl::computeRelightBatchBudget()
{
    std::size_t pendingRegionCount = 0;
    {
        std::lock_guard<std::mutex> lock(relightStateMutex_);
        pendingRegionCount = pendingRelightRegions_.size();
    }

    int batchBudget = kRelightMinBatchBudget + static_cast<int>(std::min<std::size_t>(workerThreadCount_, 4) / 2);
    batchBudget += static_cast<int>(std::min<std::size_t>(pendingRegionCount / 12, 3));
    if (pendingUploadsLastFrame_ > 256)
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
        std::clamp(static_cast<int>(std::max<std::size_t>(workerThreadCount_, 1) / 2), 1, 2);
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

        for (const auto& [column, maxChunkY] : maxRegionChunkYByColumn)
        {
            for (int chunkY = maxChunkY + 1;; ++chunkY)
            {
                const glm::ivec3 coord(column.x, chunkY, column.y);
                auto it = chunks_.find(coord);
                if (it == chunks_.end())
                {
                    break;
                }

                externalSnapshotSources.try_emplace(coord, it->second);
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







ColumnSample ChunkManager::Impl::sampleColumn(int worldX, int worldZ, int slabMinWorldY, int slabMaxWorldY) const
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
    const terrain::SurfaceColumn& surfaceColumn = surfaceMap_->column(worldX, worldZ);
    const terrain::ClimateSample& climateSample = climateMap_->sample(worldX, worldZ);

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

    sample.topBlendCount = std::min(climateSample.blendCount, sample.topBlendDebug.size());
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

    forEachDefaultTreeBlock(instance.origin.x,
                            instance.origin.z,
                            instance.origin.y,
                            instance.trunkHeight,
                            [&](int blockX, int blockY, int blockZ, BlockId block) {
                                setLocalBlock(blockX, blockY, blockZ, block, block == BlockId::Wood);
                                return false;
                            });
}
} // namespace

bool ChunkManager::Impl::generateChunkBlocks(Chunk& chunk, std::uint32_t generationEpoch)
{
    const bool benchmarkEnabled = benchmarkMetrics_.isEnabled();
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

    ChunkBuildScratch scratch(chunk);
    std::array<ColumnBuildResult, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)> columnResults{};

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

    // Structure-only slabs above the ground still need canopy/trunk blocks.
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

    std::vector<PendingStructureEdit> pendingEdits = takePendingStructureEdits(chunk.coord);
    for (const PendingStructureEdit& edit : pendingEdits)
    {
        scratch.setWorldBlock(edit.worldPos.x, edit.worldPos.y, edit.worldPos.z, edit.block, edit.replaceSolid);
    }

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
        chunk.blocks = std::move(scratch.blocks);
        chunk.hasBlocks.store(anyBlocks, std::memory_order_release);
    }
    if (benchmarkEnabled)
    {
        const auto lockMicros =
            std::chrono::duration_cast<std::chrono::microseconds>(SteadyClock::now() - meshLockStart).count();
        benchmarkMetrics_.generateBlocksMeshLockStage.recordMicros(static_cast<std::uint64_t>(lockMicros));
    }

    columnManager_.updateChunkHeights(chunk.coord, scratch.highestSolidWorlds);
    invalidatePredictedColumn({chunk.coord.x, chunk.coord.z});
    markSkyLightColumnDirty({chunk.coord.x, chunk.coord.z});
    return true;
}

std::vector<PendingStructureEdit> ChunkManager::Impl::takePendingStructureEdits(const glm::ivec3& coord)
{
    std::vector<PendingStructureEdit> edits;
    std::lock_guard<std::mutex> lock(pendingStructureMutex_);
    auto it = pendingStructureEdits_.find(coord);
    if (it != pendingStructureEdits_.end())
    {
        edits = std::move(it->second);
        pendingStructureEdits_.erase(it);
    }
    return edits;
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
        auto chunk = getChunkShared(coord);
        if (!chunk)
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
                columnManager_.updateChunk(*chunk);
                invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});
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

