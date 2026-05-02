// chunk_manager_job_queue.cpp
// Implements the internal chunk job scheduler used by ChunkManager worker threads.

#include "chunk_manager_support.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <glm/geometric.hpp>

namespace
{
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
} // namespace

glm::vec2 normalizePriorityForwardXZ(const glm::vec3& forward) noexcept
{
    glm::vec2 forwardXZ(forward.x, forward.z);
    if (glm::dot(forwardXZ, forwardXZ) <= kEpsilon)
    {
        return {0.0f, 0.0f};
    }

    return glm::normalize(forwardXZ);
}

bool isChunkCoordHigherPriority(const glm::ivec3& lhs,
                                const glm::ivec3& rhs,
                                const glm::ivec3& origin,
                                const glm::vec3& forward) noexcept
{
    const glm::vec2 forwardXZ = normalizePriorityForwardXZ(forward);
    return compareChunkPriorityKeys(buildChunkPriorityKey(lhs, origin, forwardXZ),
                                    buildChunkPriorityKey(rhs, origin, forwardXZ)) < 0;
}

bool JobQueue::push(const Job& job)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (shouldStop_.load(std::memory_order_acquire))
    {
        return false;
    }

    queues_[jobTypeIndex(job.type)].push(wrap(job));
    queuedJobCount_.fetch_add(1, std::memory_order_relaxed);
    ++queuedServiceCounts_[jobServiceClassIndex(job.serviceClass)];
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
    --queuedServiceCounts_[jobServiceClassIndex(job.serviceClass)];
    ++activeServiceCounts_[jobServiceClassIndex(job.serviceClass)];
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
        --queuedServiceCounts_[jobServiceClassIndex(job.serviceClass)];
        ++activeServiceCounts_[jobServiceClassIndex(job.serviceClass)];
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
            const Job job = queue.top().job;
            --queuedServiceCounts_[jobServiceClassIndex(job.serviceClass)];
            cancelledJobs.push_back(job);
            queue.pop();
        }
    }
    queuedServiceCounts_.fill(0);
    queuedJobCount_.store(0, std::memory_order_relaxed);
    condition_.notify_all();
    return cancelledJobs;
}

void JobQueue::restart() noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    shouldStop_.store(false, std::memory_order_release);
    condition_.notify_all();
}

bool JobQueue::stopped() const noexcept
{
    return shouldStop_.load(std::memory_order_acquire);
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

std::size_t JobQueue::outstanding(JobServiceClass serviceClass) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t index = jobServiceClassIndex(serviceClass);
    return queuedServiceCounts_[index] + activeServiceCounts_[index];
}

JobQueueSnapshot JobQueue::snapshot() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    JobQueueSnapshot snapshot{};
    snapshot.totalQueued = queuedJobCount_.load(std::memory_order_relaxed);
    snapshot.queuedByService = queuedServiceCounts_;
    snapshot.activeByService = activeServiceCounts_;
    for (std::size_t index = 0; index < queues_.size(); ++index)
    {
        snapshot.queuedByType[index] = queues_[index].size();
        snapshot.activeByType[index] = activeCounts_[index];
    }
    return snapshot;
}

void JobQueue::updatePriorityState(const glm::ivec3& origin, const glm::vec3& forward)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const glm::vec2 forwardXZ = normalizePriorityForwardXZ(forward);
    const bool hadForward = glm::dot(priorityForwardXZ_, priorityForwardXZ_) > kEpsilon;
    const bool haveForward = glm::dot(forwardXZ, forwardXZ) > kEpsilon;
    float facingDot = 1.0f;
    if (hadForward && haveForward)
    {
        facingDot = glm::dot(priorityForwardXZ_, forwardXZ);
    }
    else if (hadForward != haveForward)
    {
        facingDot = 0.0f;
    }
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
    const bool hadForward = glm::dot(priorityForwardXZ_, priorityForwardXZ_) > kEpsilon;
    const bool haveForward = glm::dot(forwardXZ, forwardXZ) > kEpsilon;
    float facingDot = 1.0f;
    if (hadForward && haveForward)
    {
        facingDot = glm::dot(priorityForwardXZ_, forwardXZ);
    }
    else if (hadForward != haveForward)
    {
        facingDot = 0.0f;
    }
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
    const int serviceBias = static_cast<int>(job.serviceClass);
    int bias = 3;
    switch (job.type)
    {
    case JobType::Mesh:
        bias = 0;
        break;
    case JobType::Generate:
        bias = 1;
        break;
    case JobType::WorldgenPageDependency:
        bias = 2;
        break;
    case JobType::StructureRegionDependency:
        bias = 3;
        break;
    case JobType::BulkShellOracle:
        bias = 4;
        break;
    case JobType::ExactFillBatchPrepare:
        bias = 3;
        break;
    }
    const std::uint64_t sequence = nextSequence_++;
    return PrioritizedJob{job, priority, lifecycleBias, serviceBias, bias, sequence};
}

int JobQueue::comparePrioritizedJobs(const PrioritizedJob& lhs,
                                     const PrioritizedJob& rhs) noexcept
{
    if (lhs.lifecycleBias != rhs.lifecycleBias)
    {
        return lhs.lifecycleBias < rhs.lifecycleBias ? -1 : 1;
    }
    if (lhs.serviceBias != rhs.serviceBias)
    {
        return lhs.serviceBias < rhs.serviceBias ? -1 : 1;
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
    std::array<std::size_t, kJobTypeCount> targets{};
    const std::size_t totalWorkers = std::max<std::size_t>(workerConcurrency_, 1);
    const std::size_t generateIndex = jobTypeIndex(JobType::Generate);
    const std::size_t meshIndex = jobTypeIndex(JobType::Mesh);
    const std::size_t worldgenDependencyIndex = jobTypeIndex(JobType::WorldgenPageDependency);
    const std::size_t structureDependencyIndex = jobTypeIndex(JobType::StructureRegionDependency);
    const std::size_t bulkShellIndex = jobTypeIndex(JobType::BulkShellOracle);
    const std::size_t exactFillBatchIndex = jobTypeIndex(JobType::ExactFillBatchPrepare);
    const std::size_t generateBacklog = queues_[generateIndex].size();
    const std::size_t meshBacklog = queues_[meshIndex].size();
    const std::size_t worldgenDependencyBacklog = queues_[worldgenDependencyIndex].size();
    const std::size_t structureDependencyBacklog = queues_[structureDependencyIndex].size();
    const std::size_t bulkShellBacklog = queues_[bulkShellIndex].size();
    const std::size_t exactFillBatchBacklog = queues_[exactFillBatchIndex].size();
    const bool generateInitialReadyTop =
        generateBacklog > 0 && queues_[generateIndex].top().lifecycleBias == 0;
    const bool meshInitialReadyTop =
        meshBacklog > 0 && queues_[meshIndex].top().lifecycleBias == 0;
    const std::size_t latencySensitivePressure =
        queuedServiceCounts_[jobServiceClassIndex(JobServiceClass::InitialVisible)] +
        queuedServiceCounts_[jobServiceClassIndex(JobServiceClass::LocalInteraction)] +
        activeServiceCounts_[jobServiceClassIndex(JobServiceClass::InitialVisible)] +
        activeServiceCounts_[jobServiceClassIndex(JobServiceClass::LocalInteraction)];
    const bool playableBacklog = generateBacklog > 0 || meshBacklog > 0;

    std::size_t worldgenDependencyTarget = 0;
    if (worldgenDependencyBacklog > 0)
    {
        worldgenDependencyTarget = 1;
        if (!playableBacklog)
        {
            worldgenDependencyTarget =
                std::min<std::size_t>(worldgenDependencyBacklog, std::min<std::size_t>(totalWorkers, 4));
        }
        else if (totalWorkers >= 12 && worldgenDependencyBacklog > 512)
        {
            worldgenDependencyTarget = 4;
        }
        else if (totalWorkers >= 10 && worldgenDependencyBacklog > 128)
        {
            worldgenDependencyTarget = 3;
        }
        else if (totalWorkers >= 8 && worldgenDependencyBacklog > 256)
        {
            worldgenDependencyTarget = 2;
        }
        else if (latencySensitivePressure == 0 && totalWorkers >= 8 && worldgenDependencyBacklog > 1)
        {
            worldgenDependencyTarget = 2;
        }
    }

    std::size_t structureDependencyTarget = 0;
    if (structureDependencyBacklog > 0)
    {
        structureDependencyTarget = 1;
        if (!playableBacklog)
        {
            const std::size_t remainingCapacity =
                (totalWorkers > worldgenDependencyTarget) ? (totalWorkers - worldgenDependencyTarget) : 0;
            structureDependencyTarget =
                std::min<std::size_t>(structureDependencyBacklog, std::min<std::size_t>(remainingCapacity, 2));
        }
        else if (latencySensitivePressure == 0 && totalWorkers >= 10 && structureDependencyBacklog > 1)
        {
            structureDependencyTarget = 2;
        }
    }

    std::size_t bulkShellTarget = 0;
    if (bulkShellBacklog > 0)
    {
        if (!playableBacklog)
        {
            const std::size_t remainingCapacity =
                (totalWorkers > worldgenDependencyTarget + structureDependencyTarget)
                    ? (totalWorkers - worldgenDependencyTarget - structureDependencyTarget)
                : 0;
            bulkShellTarget = std::min<std::size_t>(bulkShellBacklog, std::min<std::size_t>(remainingCapacity, 2));
        }
        else if (totalWorkers >= 12 && worldgenDependencyBacklog < 256 && latencySensitivePressure == 0)
        {
            bulkShellTarget = 1;
        }
        else if (latencySensitivePressure == 0 && totalWorkers >= 10)
        {
            bulkShellTarget = 1;
        }
    }

    std::size_t exactFillBatchTarget = 0;
    if (exactFillBatchBacklog > 0)
    {
        const std::size_t supportBacklog = worldgenDependencyBacklog + structureDependencyBacklog;
        if (!playableBacklog && supportBacklog == 0)
        {
            const std::size_t supportTargets = worldgenDependencyTarget + structureDependencyTarget + bulkShellTarget;
            const std::size_t remainingCapacity =
                totalWorkers > supportTargets ? (totalWorkers - supportTargets) : 0;
            exactFillBatchTarget =
                std::min<std::size_t>(exactFillBatchBacklog, std::min<std::size_t>(remainingCapacity, 4));
        }
        else if (latencySensitivePressure == 0 && supportBacklog < 128 && totalWorkers >= 8)
        {
            exactFillBatchTarget = 1;
        }
    }

    std::size_t playableReserve = 0;
    if (generateBacklog > 0)
    {
        ++playableReserve;
    }
    if (meshBacklog > 0)
    {
        ++playableReserve;
    }
    playableReserve = std::min(playableReserve, totalWorkers);

    std::size_t supportTargetTotal =
        worldgenDependencyTarget + structureDependencyTarget + bulkShellTarget + exactFillBatchTarget;
    const std::size_t maxSupportWorkers = (totalWorkers > playableReserve)
        ? (totalWorkers - playableReserve)
        : 0;
    if (supportTargetTotal > maxSupportWorkers)
    {
        std::size_t overflow = supportTargetTotal - maxSupportWorkers;
        const std::size_t exactFillTrim = std::min(exactFillBatchTarget, overflow);
        exactFillBatchTarget -= exactFillTrim;
        overflow -= exactFillTrim;
        const std::size_t bulkTrim = std::min(bulkShellTarget, overflow);
        bulkShellTarget -= bulkTrim;
        overflow -= bulkTrim;
        if (overflow > 0)
        {
            const std::size_t structureTrim = std::min(structureDependencyTarget, overflow);
            structureDependencyTarget -= structureTrim;
            overflow -= structureTrim;
        }
        if (overflow > 0)
        {
            worldgenDependencyTarget -= std::min(worldgenDependencyTarget, overflow);
        }
        supportTargetTotal = worldgenDependencyTarget + structureDependencyTarget + bulkShellTarget + exactFillBatchTarget;
    }

    targets[worldgenDependencyIndex] = worldgenDependencyTarget;
    targets[structureDependencyIndex] = structureDependencyTarget;
    targets[bulkShellIndex] = bulkShellTarget;
    targets[exactFillBatchIndex] = exactFillBatchTarget;

    const std::size_t playableWorkers = totalWorkers - supportTargetTotal;
    if (playableWorkers == 0)
    {
        return targets;
    }
    if (playableWorkers == 1)
    {
        if (generateBacklog > 0 && meshBacklog == 0)
        {
            targets[generateIndex] = 1;
        }
        else if (meshBacklog > 0 && generateBacklog == 0)
        {
            targets[meshIndex] = 1;
        }
        return targets;
    }

    double meshShare = 0.25;
    if (generateInitialReadyTop && meshInitialReadyTop)
    {
        if (generateBacklog > meshBacklog * 2)
        {
            meshShare = 0.18;
        }
        else if (generateBacklog > meshBacklog)
        {
            meshShare = 0.22;
        }
        else if (meshBacklog > generateBacklog * 2)
        {
            meshShare = 0.35;
        }
        else if (meshBacklog > generateBacklog)
        {
            meshShare = 0.30;
        }
        else
        {
            meshShare = 0.25;
        }
    }
    else if (meshBacklog == 0 && generateBacklog > 0)
    {
        meshShare = 0.15;
    }
    else if (generateBacklog == 0 && meshBacklog > 0)
    {
        meshShare = 0.8;
    }
    else if (meshBacklog > generateBacklog * 2)
    {
        meshShare = 0.40;
    }
    else if (generateBacklog > meshBacklog * 2)
    {
        meshShare = 0.18;
    }

    std::size_t meshTarget = static_cast<std::size_t>(std::round(meshShare * static_cast<double>(playableWorkers)));
    meshTarget = std::clamp<std::size_t>(meshTarget, 1, playableWorkers - 1);
    std::size_t generateTarget = playableWorkers - meshTarget;
    if (generateBacklog == 0)
    {
        generateTarget = 0;
        meshTarget = playableWorkers;
    }
    else if (meshBacklog == 0)
    {
        meshTarget = 0;
        generateTarget = playableWorkers;
    }

    targets[generateIndex] = generateTarget;
    targets[meshIndex] = meshTarget;
    return targets;
}

std::size_t JobQueue::pickNextQueueIndexLocked() const noexcept
{
    const std::size_t generateIndex = jobTypeIndex(JobType::Generate);
    const std::size_t meshIndex = jobTypeIndex(JobType::Mesh);
    const bool generateReady = !queues_[generateIndex].empty();
    const bool meshReady = !queues_[meshIndex].empty();
    if (generateReady && meshReady && workerConcurrency_ > 1)
    {
        const PrioritizedJob& generateTop = queues_[generateIndex].top();
        const PrioritizedJob& meshTop = queues_[meshIndex].top();
        const bool generateLatencySensitive =
            generateTop.serviceBias <= static_cast<int>(JobServiceClass::LocalInteraction);
        const bool meshLatencySensitive =
            meshTop.serviceBias <= static_cast<int>(JobServiceClass::LocalInteraction);
        const bool reserveGenerateLane = generateLatencySensitive && activeCounts_[generateIndex] == 0;
        const bool reserveMeshLane = meshLatencySensitive && activeCounts_[meshIndex] == 0;
        if (reserveGenerateLane != reserveMeshLane)
        {
            return reserveGenerateLane ? generateIndex : meshIndex;
        }
        if (reserveGenerateLane && reserveMeshLane)
        {
            return comparePrioritizedJobs(meshTop, generateTop) <= 0 ? meshIndex : generateIndex;
        }
    }

    const std::array<std::size_t, kJobTypeCount> targets = computeStageTargetsLocked();
    std::size_t bestUnderTarget = kJobTypeCount;
    std::size_t bestUnderTargetDeficit = 0;
    std::size_t bestReady = kJobTypeCount;
    for (std::size_t queueIndex = 0; queueIndex < queues_.size(); ++queueIndex)
    {
        if (queues_[queueIndex].empty())
        {
            continue;
        }

        const PrioritizedJob& candidate = queues_[queueIndex].top();
        if (bestReady == kJobTypeCount ||
            comparePrioritizedJobs(candidate, queues_[bestReady].top()) < 0)
        {
            bestReady = queueIndex;
        }

        if (activeCounts_[queueIndex] >= targets[queueIndex])
        {
            continue;
        }

        const std::size_t deficit = targets[queueIndex] - activeCounts_[queueIndex];
        if (bestUnderTarget == kJobTypeCount ||
            deficit > bestUnderTargetDeficit ||
            (deficit == bestUnderTargetDeficit &&
             comparePrioritizedJobs(candidate, queues_[bestUnderTarget].top()) < 0))
        {
            bestUnderTarget = queueIndex;
            bestUnderTargetDeficit = deficit;
        }
    }

    if (bestUnderTarget != kJobTypeCount)
    {
        return bestUnderTarget;
    }
    if (bestReady != kJobTypeCount)
    {
        return bestReady;
    }
    return generateIndex;
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

void JobQueue::jobCompleted(JobType type, JobServiceClass serviceClass) noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t index = jobTypeIndex(type);
    if (activeCounts_[index] > 0)
    {
        --activeCounts_[index];
    }
    const std::size_t serviceIndex = jobServiceClassIndex(serviceClass);
    if (activeServiceCounts_[serviceIndex] > 0)
    {
        --activeServiceCounts_[serviceIndex];
    }
    condition_.notify_one();
}
