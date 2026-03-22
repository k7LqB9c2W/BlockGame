// chunk_manager_structure_registry.inl
// Defines the internal structure generation and query cache used by ChunkManager and far terrain.

constexpr int kStructureRegionSize = 128;
constexpr int kMaxStructureHorizontalRadius =
    std::max(std::max(kTaigaSpruceMaxLeafRadius + 1, kDarkOakMaxHorizontalReach), kAcaciaMaxHorizontalReach);

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
    DarkOak = 2,
    Acacia = 3,
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
    BlockId trunkBlock{BlockId::Wood};
    BlockId leavesBlock{BlockId::Leaves};
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

    if (instance.type == StructureType::Acacia)
    {
        return forEachAcaciaTreeBlock(instance.origin.x,
                                      instance.origin.z,
                                      instance.origin.y,
                                      instance.trunkHeight,
                                      instance.trunkBlock,
                                      instance.leavesBlock,
                                      std::forward<Callback>(callback));
    }

    if (instance.type == StructureType::DarkOak)
    {
        return forEachDarkOakTreeBlock(instance.origin.x,
                                       instance.origin.z,
                                       instance.origin.y,
                                       instance.trunkHeight,
                                       instance.trunkBlock,
                                       instance.leavesBlock,
                                       std::forward<Callback>(callback));
    }

    return forEachDefaultTreeBlock(instance.origin.x,
                                   instance.origin.z,
                                   instance.origin.y,
                                   instance.trunkHeight,
                                   instance.trunkBlock,
                                   instance.leavesBlock,
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

            if (biome.id == "dark_forest")
            {
                DarkOakTreeCandidate candidate{};
                if (!tryBuildDarkOakCandidate(worldX,
                                              worldZ,
                                              columnSample,
                                              sampleColumn,
                                              resolvedSurfaceBlockAt,
                                              densityAt,
                                              candidate))
                {
                    continue;
                }

                if (darkOakHasSpacingConflict(candidate, sampleColumn, resolvedSurfaceBlockAt, densityAt))
                {
                    continue;
                }

                StructureInstance instance{};
                instance.type = StructureType::DarkOak;
                instance.origin = glm::ivec3(candidate.originX, candidate.groundWorldY, candidate.originZ);
                instance.trunkHeight = candidate.trunkHeight;
                instance.priority = candidate.priority;
                instance.maxLodLevel = 4;
                instance.trunkBlock = BlockId::DarkOakLog;
                instance.leavesBlock = BlockId::DarkOakLeaves;
                instance.bounds.min = glm::ivec3(candidate.originX - kDarkOakBranchMaxLength,
                                                 candidate.groundWorldY,
                                                 candidate.originZ - kDarkOakBranchMaxLength);
                instance.bounds.max = glm::ivec3(candidate.originX + 1 + kDarkOakBranchMaxLength,
                                                 candidate.groundWorldY + candidate.trunkHeight + kDarkOakCanopyTopOffset,
                                                 candidate.originZ + 1 + kDarkOakBranchMaxLength);
                region.instances.push_back(instance);
                continue;
            }

            if (biome.id == "savanna")
            {
                AcaciaTreeCandidate candidate{};
                if (!tryBuildAcaciaCandidate(worldX,
                                             worldZ,
                                             columnSample,
                                             sampleColumn,
                                             resolvedSurfaceBlockAt,
                                             densityAt,
                                             candidate))
                {
                    continue;
                }

                if (acaciaHasSpacingConflict(candidate, sampleColumn, resolvedSurfaceBlockAt, densityAt))
                {
                    continue;
                }

                StructureInstance instance{};
                instance.type = StructureType::Acacia;
                instance.origin = glm::ivec3(candidate.originX, candidate.groundWorldY, candidate.originZ);
                instance.trunkHeight = candidate.trunkHeight;
                instance.priority = candidate.priority;
                instance.maxLodLevel = 4;
                instance.trunkBlock = BlockId::AcaciaLog;
                instance.leavesBlock = BlockId::AcaciaLeaves;
                instance.bounds.min = glm::ivec3(candidate.originX - kAcaciaMaxHorizontalReach,
                                                 candidate.groundWorldY,
                                                 candidate.originZ - kAcaciaMaxHorizontalReach);
                instance.bounds.max = glm::ivec3(candidate.originX + kAcaciaMaxHorizontalReach,
                                                 candidate.groundWorldY + candidate.trunkHeight + 1,
                                                 candidate.originZ + kAcaciaMaxHorizontalReach);
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
            instance.trunkBlock = candidate.trunkBlock;
            instance.leavesBlock = candidate.leavesBlock;
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



