// chunk_manager_support.h
// Declares internal job scheduling and column-height helpers used by chunk_manager.cpp.

#pragma once

#include "chunk_manager.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace chunk_manager_detail
{
struct Chunk;
}

using chunk_manager_detail::Chunk;

struct ChunkHasher
{
    std::size_t operator()(const glm::ivec3& value) const noexcept
    {
        std::size_t hash = static_cast<std::size_t>(value.x) * 73856093u;
        hash ^= static_cast<std::size_t>(value.y) * 19349663u;
        hash ^= static_cast<std::size_t>(value.z) * 83492791u;
        return hash;
    }
};

struct ColumnHasher
{
    std::size_t operator()(const glm::ivec2& value) const noexcept
    {
        std::size_t hash = static_cast<std::size_t>(value.x) * 73856093u;
        hash ^= static_cast<std::size_t>(value.y) * 19349663u;
        return hash;
    }
};

enum class JobType : std::uint8_t
{
    Generate = 0,
    Mesh = 1
};

inline constexpr std::size_t kJobTypeCount = 2;

[[nodiscard]] constexpr std::size_t jobTypeIndex(JobType type) noexcept
{
    return static_cast<std::size_t>(type);
}

struct Job
{
    JobType type;
    glm::ivec3 chunkCoord;
    std::shared_ptr<Chunk> chunk;
    std::uint32_t generationEpoch{0};
    bool initialReadyPriority{false};

    Job(JobType jobType,
        const glm::ivec3& coord,
        std::shared_ptr<Chunk> chunkRef,
        std::uint32_t epoch = 0,
        bool initialPriority = false)
        : type(jobType),
          chunkCoord(coord),
          chunk(std::move(chunkRef)),
          generationEpoch(epoch),
          initialReadyPriority(initialPriority)
    {
    }
};

struct ChunkPriorityKey
{
    int supportBucket{3};
    int horizontalDistance{0};
    int forwardBucket{2};
    int verticalDistance{0};
    int axisDistance{0};
};

[[nodiscard]] glm::vec2 normalizePriorityForwardXZ(const glm::vec3& forward) noexcept;
[[nodiscard]] bool isChunkCoordHigherPriority(const glm::ivec3& lhs,
                                              const glm::ivec3& rhs,
                                              const glm::ivec3& origin,
                                              const glm::vec3& forward) noexcept;

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

struct ChunkBlockView
{
    glm::ivec3 coord{0};
    int minWorldY{0};
    std::span<const BlockId> blocks{};
};

class ColumnManager
{
public:
    static constexpr int kNoHeight = std::numeric_limits<int>::min();

    void updateChunk(const ChunkBlockView& chunk);
    void updateChunkHeights(const glm::ivec3& chunkCoord,
                            const std::array<int, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)>& highestWorlds);
    void updateColumn(const ChunkBlockView& chunk, int localX, int localZ);
    void removeChunk(const glm::ivec3& chunkCoord);
    void clear();

    int highestSolidBlock(int worldX, int worldZ) const noexcept;

private:
    struct ColumnData
    {
        std::unordered_map<int, int> slabHeights;
        int highestWorldY{kNoHeight};
    };

    static glm::ivec2 columnKey(const glm::ivec3& chunkCoord, int localX, int localZ) noexcept;
    static int scanColumnHighestWorld(const ChunkBlockView& chunk, int localX, int localZ) noexcept;
    static int computeHighest(const ColumnData& data) noexcept;
    void applyHeightLocked(const glm::ivec2& key, int chunkY, int highestWorldY);

    mutable std::mutex mutex_;
    std::unordered_map<glm::ivec2, ColumnData, ColumnHasher> columns_;
};
