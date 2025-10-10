// chunk_manager.cpp
// Implements the chunk streaming, terrain generation, and GPU upload subsystem.

#include "chunk_manager.h"

#include "terrain/biome_database.h"
#include "terrain/climate_map.h"
#include "terrain/surface_map.h"
#include "terrain/terrain_generator.h"
#include "terrain/worldgen_profile.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
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

namespace
{
std::atomic<int> gActiveVerticalRadius{kVerticalStreamingConfig.minRadiusChunks};
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

float kFarPlane = computeFarPlaneForViewDistance(kDefaultViewDistance);

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
struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 tileCoord;
    glm::vec2 atlasBase;
    glm::vec2 atlasSize;
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
    std::uint32_t h = static_cast<std::uint32_t>(x * 374761393 + y * 668265263 + z * 2147483647);
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
}

struct MeshData
{
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

    bool empty() const
    {
        return vertices.empty() || indices.empty();
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

enum class JobType : std::uint8_t
{
    Generate = 0,
    Mesh = 1
};

struct FarChunk
{
    static constexpr int kColumnStep = 4;
    static constexpr int kColumnsX = kChunkSizeX / kColumnStep;
    static constexpr int kColumnsZ = kChunkSizeZ / kColumnStep;

    struct SurfaceCell
    {
        int worldY{std::numeric_limits<int>::min()};
        BlockId block{BlockId::Air};
    };

    glm::vec3 origin{0.0f};
    glm::ivec3 size{kChunkSizeX, kChunkSizeY, kChunkSizeZ};
    int lodStep{kColumnStep};
    int thickness{1};
    std::array<SurfaceCell, kColumnsX * kColumnsZ> strata{};
    GLuint opaqueVao{0};
    GLuint opaqueVbo{0};
    GLuint opaqueIbo{0};
    GLuint cutoutVao{0};
    GLuint cutoutVbo{0};
    GLuint cutoutIbo{0};

    static constexpr std::size_t index(int x, int z) noexcept
    {
        return static_cast<std::size_t>(z) * static_cast<std::size_t>(kColumnsX) +
               static_cast<std::size_t>(x);
    }
};

constexpr std::uint32_t kInvalidChunkBufferPage = std::numeric_limits<std::uint32_t>::max();

struct Chunk
{
    explicit Chunk(const glm::ivec3& c)
        : coord(c),
          minWorldY(c.y * kChunkSizeY),
          maxWorldY(minWorldY + kChunkSizeY - 1),
          blocks(kChunkBlockCount, BlockId::Air),
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
        state.store(ChunkState::Empty, std::memory_order_relaxed);
        meshData.clear();
        meshReady = false;
        hasBlocks = false;
        queuedForUpload = false;
        indexCount = 0;
        vertexCount = 0;
        bufferPageIndex = kInvalidChunkBufferPage;
        vertexOffset = 0;
        indexOffset = 0;
        inFlight.store(0, std::memory_order_relaxed);
        surfaceOnly = false;
        lodData.reset();
    }


    glm::ivec3 coord;
    int minWorldY{0};
    int maxWorldY{0};
    std::vector<BlockId> blocks;
    std::atomic<ChunkState> state;

    GLsizei indexCount{0};
    std::size_t vertexCount{0};
    std::uint32_t bufferPageIndex{kInvalidChunkBufferPage};
    std::size_t vertexOffset{0};
    std::size_t indexOffset{0};
    bool queuedForUpload{false};

    mutable std::mutex meshMutex;
    MeshData meshData;
    bool meshReady{false};
    bool hasBlocks{false};
    std::atomic<int> inFlight{0};
    bool surfaceOnly{false};
    std::unique_ptr<FarChunk> lodData;
};

struct ProfilingCounters
{
    std::atomic<long long> generationMicros{0};
    std::atomic<long long> meshingMicros{0};
    std::atomic<std::size_t> uploadedBytes{0};
    std::atomic<int> generatedChunks{0};
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

    Job(JobType t, const glm::ivec3& coord, std::shared_ptr<Chunk> c)
        : type(t), chunkCoord(coord), chunk(std::move(c)) {}
};

class JobQueue
{
public:
    void push(const Job& job);
    bool tryPop(Job& job);
    Job waitAndPop();
    void stop();
    bool empty() const;
    void updatePriorityOrigin(const glm::ivec3& origin);

private:
    struct PrioritizedJob
    {
        Job job;
        int distance{0};
        int priorityBias{0};
        std::uint64_t sequence{0};
    };

    struct JobComparer
    {
        bool operator()(const PrioritizedJob& lhs, const PrioritizedJob& rhs) const;
    };

    PrioritizedJob wrap(const Job& job);
    static int manhattanDistance(const glm::ivec3& a, const glm::ivec3& b) noexcept;
    void rebuildLocked();

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> shouldStop_{false};
    glm::ivec3 priorityOrigin_{0, 0, 0};
    std::priority_queue<PrioritizedJob, std::vector<PrioritizedJob>, JobComparer> priorityQueue_;
    std::uint64_t nextSequence_{0};
};

class ColumnManager
{
public:
    static constexpr int kNoHeight = std::numeric_limits<int>::min();

    void updateChunk(const Chunk& chunk);
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


} // namespace

struct ChunkManager::Impl
{
    explicit Impl(unsigned seed);
    ~Impl();

    void setAtlasTexture(GLuint texture) noexcept;
    void setBlockTextureAtlasConfig(const glm::ivec2& textureSizePixels, int tileSizePixels);
    void update(const glm::vec3& cameraPos);
    ChunkRenderData buildRenderData(const Frustum& frustum) const;

    float surfaceHeight(float worldX, float worldZ) const noexcept;
    void clear();

    bool destroyBlock(const glm::ivec3& worldPos);
    bool placeBlock(const glm::ivec3& targetBlockPos, const glm::ivec3& faceNormal);

    RaycastHit raycast(const glm::vec3& origin, const glm::vec3& direction) const;
    void updateHighlight(const glm::vec3& cameraPos, const glm::vec3& cameraDirection);

    void toggleViewDistance();
    int viewDistance() const noexcept;
    void setRenderDistance(int distance) noexcept;
    void setLodEnabled(bool enabled);
    bool lodEnabled() const noexcept;

    BlockId blockAt(const glm::ivec3& worldPos) const noexcept;
    glm::vec3 findSafeSpawnPosition(float worldX, float worldZ) const;
    ChunkProfilingSnapshot sampleProfilingSnapshot();

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

    void startWorkerThreads();
    void stopWorkerThreads();
    void workerThreadFunction();
    void enqueueJob(const std::shared_ptr<Chunk>& chunk, JobType type, const glm::ivec3& coord);
    void processJob(const Job& job);
    std::shared_ptr<Chunk> popNextChunkForUpload();
    void queueChunkForUpload(const std::shared_ptr<Chunk>& chunk);
    void requeueChunkForUpload(const std::shared_ptr<Chunk>& chunk, bool toFront);

    struct ChunkBufferPage
    {
        struct Range
        {
            std::size_t offset{0};
            std::size_t size{0};
        };

        GLuint vao{0};
        GLuint vbo{0};
        GLuint ibo{0};
        std::size_t vertexCapacity{0};
        std::size_t indexCapacity{0};
        std::size_t vertexCursor{0};
        std::size_t indexCursor{0};
        std::vector<Range> freeVertices;
        std::vector<Range> freeIndices;
        std::size_t activeChunks{0};
    };

    struct ChunkAllocation
    {
        std::uint32_t pageIndex{kInvalidChunkBufferPage};
        std::size_t vertexOffset{0};
        std::size_t indexOffset{0};
    };

    static std::size_t nextPowerOfTwo(std::size_t value) noexcept;
    ChunkBufferPage createBufferPage(std::size_t vertexCount, std::size_t indexCount);
    ChunkAllocation acquireChunkAllocation(std::size_t vertexCount, std::size_t indexCount);
    void releaseChunkAllocation(Chunk& chunk);
    void recycleChunkGPU(Chunk& chunk);
    void destroyBufferPages();
    int computeVerticalRadius(const glm::ivec3& center, int horizontalRadius, int cameraWorldY);
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
    struct UploadBudgets
    {
        std::size_t byteBudget{kUploadBudgetBytesPerFrame};
        int columnLimit{kVerticalStreamingConfig.uploadBasePerColumn};
        std::size_t queueSize{0};
    };
    UploadBudgets computeUploadBudgets(int verticalRadius);
    static int computeBacklogSteps(int backlog, int threshold, int stepSize) noexcept;
    int computeGenerationBudget(int horizontalRadius, int verticalRadius, int backlogSteps) const;
    int computeRingExpansionBudget(int backlogChunks) const;
    int computeColumnJobCap(int backlogSteps, int backlogChunks) const;
    int estimateMissingChunks(const glm::ivec3& center, int horizontalRadius, int verticalRadius) const;

    struct RingProgress
    {
        bool fullyLoaded{false};
        bool budgetExhausted{false};
    };

    RingProgress ensureVolume(const glm::ivec3& center, int horizontalRadius, int verticalRadius, int& jobBudget);
    void removeDistantChunks(const glm::ivec3& center, int horizontalThreshold, int verticalThreshold);
    bool ensureChunkAsync(const glm::ivec3& coord, bool surfaceOnly);
    void uploadReadyMeshes();
    void uploadChunkMesh(Chunk& chunk);
    void buildChunkMeshAsync(Chunk& chunk);
    static glm::ivec3 worldToChunkCoords(int worldX, int worldY, int worldZ) noexcept;
    std::shared_ptr<Chunk> acquireChunk(const glm::ivec3& coord);

    std::shared_ptr<Chunk> getChunkShared(const glm::ivec3& coord) noexcept;
    std::shared_ptr<const Chunk> getChunkShared(const glm::ivec3& coord) const noexcept;
    Chunk* getChunk(const glm::ivec3& coord) noexcept;
    const Chunk* getChunk(const glm::ivec3& coord) const noexcept;
    void markNeighborsForRemeshingIfNeeded(const glm::ivec3& coord, int localX, int localY, int localZ);
    void generateChunkBlocks(Chunk& chunk);
    void generateSurfaceOnlyChunk(Chunk& chunk);
    ColumnSample sampleColumn(int worldX,
                              int worldZ,
                              int slabMinWorldY = std::numeric_limits<int>::min(),
                              int slabMaxWorldY = std::numeric_limits<int>::max()) const;
    int ensureColumnHeightCached(const glm::ivec2& column, int worldX, int worldZ) const;
    bool tryGetPredictedColumnHeight(const glm::ivec2& column, int& outHeight) const;
    int cacheSampledColumnHeight(const glm::ivec2& column, int worldX, int worldZ) const;
    void invalidatePredictedColumn(const glm::ivec2& column) const;
    bool applyPendingStructureEditsLocked(Chunk& chunk);
    void dispatchStructureEdits(const std::vector<PendingStructureEdit>& edits);
    static bool chunkHasSolidBlocks(const Chunk& chunk) noexcept;
    void recycleChunkObject(std::shared_ptr<Chunk> chunk);
    void buildSurfaceOnlyMesh(Chunk& chunk);
    bool shouldUseSurfaceOnly(const glm::ivec3& center, const glm::ivec3& coord) const noexcept;

    glm::vec2 atlasTileScale_{1.0f, 1.0f};
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
    bool lodEnabled_{false};
    int lodNearRadius_{8};
    bool lodModeDirty_{false};

    std::deque<std::weak_ptr<Chunk>> uploadQueue_;
    std::mutex uploadQueueMutex_;
    std::vector<ChunkBufferPage> bufferPages_;
    mutable std::mutex bufferPageMutex_;

    std::unordered_map<glm::ivec3, std::shared_ptr<Chunk>, ChunkHasher> chunks_;
    mutable std::mutex chunksMutex;
    const glm::vec3 lightDirection_{glm::normalize(glm::vec3(0.5f, -1.0f, 0.2f))};
    GLuint atlasTexture_{0};
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
    std::vector<std::shared_ptr<Chunk>> chunkPool_;
    std::mutex chunkPoolMutex_;
    ProfilingCounters profilingCounters_{};
    std::unordered_map<glm::ivec2, int, ColumnHasher> jobsScheduledThisFrame_{};
    int lastVerticalRadius_{kVerticalStreamingConfig.minRadiusChunks};
    int uploadColumnLimitThisFrame_{kVerticalStreamingConfig.uploadBasePerColumn};
    std::size_t uploadBudgetBytesThisFrame_{kUploadBudgetBytesPerFrame};
    std::size_t lastUploadBytesUsed_{0};
    std::size_t pendingUploadsLastFrame_{0};
    int generationColumnCapThisFrame_{kVerticalStreamingConfig.maxGenerationJobsPerColumn};
    int lastGenerationBudget_{kVerticalStreamingConfig.generationBudget.baseJobsPerFrame};
    int lastGenerationJobsIssued_{0};
    int lastRingBudget_{kVerticalStreamingConfig.generationBudget.minRingExpansionsPerFrame};
    int lastRingExpansionsUsed_{0};
    int lastMissingChunks_{0};
    int lastColumnCap_{kVerticalStreamingConfig.maxGenerationJobsPerColumn};
    int lastBacklogSteps_{0};
    int lastLoggedGenerationBudget_{-1};
    int lastLoggedRingBudget_{-1};
    int lastLoggedColumnCap_{-1};
};

// JobQueue implementations

void JobQueue::push(const Job& job)
{
    std::lock_guard<std::mutex> lock(mutex_);
    priorityQueue_.push(wrap(job));
    condition_.notify_one();
}

bool JobQueue::tryPop(Job& job)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (priorityQueue_.empty())
    {
        return false;
    }
    job = priorityQueue_.top().job;
    priorityQueue_.pop();
    return true;
}

Job JobQueue::waitAndPop()
{
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !priorityQueue_.empty() || shouldStop_.load(std::memory_order_acquire); });

    if (shouldStop_.load(std::memory_order_acquire) && priorityQueue_.empty())
    {
        throw std::runtime_error("Job queue stopped");
    }

    Job job = priorityQueue_.top().job;
    priorityQueue_.pop();
    return job;
}

void JobQueue::stop()
{
    std::lock_guard<std::mutex> lock(mutex_);
    shouldStop_.store(true, std::memory_order_release);
    condition_.notify_all();
}

bool JobQueue::empty() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return priorityQueue_.empty();
}

void JobQueue::updatePriorityOrigin(const glm::ivec3& origin)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (origin == priorityOrigin_)
    {
        return;
    }

    priorityOrigin_ = origin;
    rebuildLocked();
}

bool JobQueue::JobComparer::operator()(const PrioritizedJob& lhs, const PrioritizedJob& rhs) const
{
    if (lhs.distance != rhs.distance)
    {
        return lhs.distance > rhs.distance;
    }
    if (lhs.priorityBias != rhs.priorityBias)
    {
        return lhs.priorityBias > rhs.priorityBias;
    }
    return lhs.sequence > rhs.sequence;
}

JobQueue::PrioritizedJob JobQueue::wrap(const Job& job)
{
    const int distance = manhattanDistance(job.chunkCoord, priorityOrigin_);
    const int bias = (job.type == JobType::Mesh) ? 0 : 1;
    const std::uint64_t sequence = nextSequence_++;
    return PrioritizedJob{job, distance, bias, sequence};
}

int JobQueue::manhattanDistance(const glm::ivec3& a, const glm::ivec3& b) noexcept
{
    return std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z);
}

void JobQueue::rebuildLocked()
{
    if (priorityQueue_.empty())
    {
        return;
    }

    std::vector<PrioritizedJob> jobs;
    jobs.reserve(priorityQueue_.size());
    while (!priorityQueue_.empty())
    {
        jobs.push_back(priorityQueue_.top());
        priorityQueue_.pop();
    }

    for (auto& prioritized : jobs)
    {
        prioritized.distance = manhattanDistance(prioritized.job.chunkCoord, priorityOrigin_);
        priorityQueue_.push(std::move(prioritized));
    }
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
      viewDistance_(kDefaultViewDistance),
      targetViewDistance_(kDefaultViewDistance)
{
    const unsigned effectiveSeed = worldgenProfile_.effectiveSeed(seed);

    noise_.reseed(effectiveSeed);

    if (biomeDatabase_.biomeCount() == 0)
    {
        throw std::runtime_error("Biome database is empty");
    }

    climateMap_ = std::make_unique<terrain::ClimateMap>(
        std::make_unique<terrain::NoiseVoronoiClimateGenerator>(biomeDatabase_, worldgenProfile_, effectiveSeed,
                                                                kChunkSizeX, kBiomeSizeInChunks),
        64);

    surfaceMap_ = std::make_unique<terrain::SurfaceMap>(
        std::make_unique<terrain::MapGenV1>(biomeDatabase_, *climateMap_, worldgenProfile_, effectiveSeed),
        64);

    terrainGenerator_ = std::make_unique<terrain::TerrainGenerator>(
        *climateMap_,
        *surfaceMap_,
        biomeDatabase_,
        globalSeaLevel_,
        [this](int worldX, int worldZ, int slabMin, int slabMax) {
            return this->sampleColumn(worldX, worldZ, slabMin, slabMax);
        });

    gActiveVerticalRadius.store(kVerticalStreamingConfig.minRadiusChunks, std::memory_order_relaxed);
    kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);
    startWorkerThreads();
}

ChunkManager::Impl::~Impl()
{
    stopWorkerThreads();
    clear();
    destroyBufferPages();
}

void ChunkManager::Impl::setAtlasTexture(GLuint texture) noexcept
{
    atlasTexture_ = texture;
}

void ChunkManager::Impl::setBlockTextureAtlasConfig(const glm::ivec2& textureSizePixels, int tileSizePixels)
{
    if (tileSizePixels <= 0 || textureSizePixels.x <= 0 || textureSizePixels.y <= 0)
    {
        std::cerr << "Invalid block atlas dimensions provided" << std::endl;
        blockAtlasConfigured_ = false;
        return;
    }

    atlasTileScale_ = glm::vec2(
        static_cast<float>(tileSizePixels) / static_cast<float>(textureSizePixels.x),
        static_cast<float>(tileSizePixels) / static_cast<float>(textureSizePixels.y));

    for (auto& blockEntry : blockUVTable_)
    {
        for (auto& face : blockEntry.faces)
        {
            face.base = glm::vec2(0.0f);
            face.size = atlasTileScale_;
        }
    }

    auto assignFace = [&](BlockId block, BlockFace face, const glm::ivec2& tile)
    {
        const glm::vec2 base = glm::vec2(static_cast<float>(tile.x), static_cast<float>(tile.y)) * atlasTileScale_;
        auto& faceUV = blockUVTable_[toIndex(block)].faces[toIndex(face)];
        faceUV.base = base;
        faceUV.size = atlasTileScale_;
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
        assignFace(BlockId::Stone, face, {1, 9});
    }

    blockAtlasConfigured_ = true;
}

void ChunkManager::Impl::update(const glm::vec3& cameraPos)
{
    const int worldX = static_cast<int>(std::floor(cameraPos.x));
    const int worldY = static_cast<int>(std::floor(cameraPos.y));
    const int worldZ = static_cast<int>(std::floor(cameraPos.z));
    const int clampedWorldY = std::max(worldY, 0);
    const glm::ivec3 centerChunk = worldToChunkCoords(worldX, clampedWorldY, worldZ);

    if (lodModeDirty_)
    {
        clear();
        lodModeDirty_ = false;
    }

    if (lodEnabled_)
    {
        lodNearRadius_ = std::max(4, targetViewDistance_ / 2);
    }

    resetColumnBudgets();
    const int verticalRadius = computeVerticalRadius(centerChunk, targetViewDistance_, clampedWorldY);
    lastVerticalRadius_ = verticalRadius;
    gActiveVerticalRadius.store(verticalRadius, std::memory_order_relaxed);
    kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);

    UploadBudgets uploadBudgets = computeUploadBudgets(verticalRadius);
    uploadBudgetBytesThisFrame_ = uploadBudgets.byteBudget;
    uploadColumnLimitThisFrame_ = uploadBudgets.columnLimit;
    pendingUploadsLastFrame_ = uploadBudgets.queueSize;

    jobQueue_.updatePriorityOrigin(centerChunk);

    if (viewDistance_ > targetViewDistance_)
    {
        viewDistance_ = targetViewDistance_;
    }

    const int missingChunks = estimateMissingChunks(centerChunk, targetViewDistance_, verticalRadius);
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

    lastGenerationBudget_ = generationBudgetTarget;
    lastRingBudget_ = ringBudget;
    lastMissingChunks_ = missingChunks;
    lastColumnCap_ = generationColumnCapThisFrame_;
    lastBacklogSteps_ = backlogSteps;

    if (generationBudgetTarget != lastLoggedGenerationBudget_ ||
        ringBudget != lastLoggedRingBudget_ ||
        lastColumnCap_ != lastLoggedColumnCap_)
    {
        std::cout << "[ChunkManager] Adaptive streaming budgets -- backlog: " << missingChunks
                  << " chunks, steps: " << backlogSteps
                  << ", jobBudget: " << generationBudgetTarget
                  << ", ringBudget: " << ringBudget
                  << ", columnCap: ";
        if (generationColumnCapThisFrame_ >= std::numeric_limits<int>::max())
        {
            std::cout << "unlimited";
        }
        else
        {
            std::cout << generationColumnCapThisFrame_;
        }
        std::cout << ", verticalRadius: " << verticalRadius << std::endl;

        lastLoggedGenerationBudget_ = generationBudgetTarget;
        lastLoggedRingBudget_ = ringBudget;
        lastLoggedColumnCap_ = lastColumnCap_;
    }

    int jobBudget = generationBudgetTarget;

    for (int ring = 0; ring <= viewDistance_ && jobBudget > 0; ++ring)
    {
        RingProgress progress = ensureVolume(centerChunk, ring, verticalRadius, jobBudget);
        if (progress.budgetExhausted)
        {
            break;
        }
    }

    int ringsExpanded = 0;
    while (jobBudget > 0 && viewDistance_ < targetViewDistance_ && ringsExpanded < ringBudget)
    {
        const int nextRing = viewDistance_ + 1;
        RingProgress progress = ensureVolume(centerChunk, nextRing, verticalRadius, jobBudget);

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

    lastGenerationJobsIssued_ = std::clamp(generationBudgetTarget - jobBudget, 0, generationBudgetTarget);
    lastRingExpansionsUsed_ = ringsExpanded;

    removeDistantChunks(centerChunk,
                        targetViewDistance_ + kVerticalStreamingConfig.horizontalEvictionSlack,
                        verticalRadius);

    uploadReadyMeshes();
}

ChunkRenderData ChunkManager::Impl::buildRenderData(const Frustum& frustum) const
{
    ChunkRenderData renderData;
    renderData.lightDirection = lightDirection_;
    renderData.highlightedBlock = highlightedBlock_;
    renderData.hasHighlight = hasHighlight_;
    renderData.atlasTexture = atlasTexture_;

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
        renderData.batches.resize(pageCount);
        for (std::size_t i = 0; i < pageCount; ++i)
        {
            renderData.batches[i].vao = bufferPages_[i].vao;
        }
    }

    for (const auto& [coord, chunkPtr] : snapshot)
    {
        if (!chunkPtr)
        {
            continue;
        }

        ChunkState state = chunkPtr->state.load();
        if ((state != ChunkState::Uploaded && state != ChunkState::Remeshing) || chunkPtr->indexCount == 0)
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

        const std::uint32_t pageIndex = chunkPtr->bufferPageIndex;
        if (pageIndex == kInvalidChunkBufferPage || pageIndex >= renderData.batches.size())
        {
            continue;
        }

        if (chunkPtr->vertexOffset > static_cast<std::size_t>(std::numeric_limits<GLint>::max()))
        {
            continue;
        }

        ChunkRenderBatch& batch = renderData.batches[pageIndex];
        batch.counts.push_back(chunkPtr->indexCount);
        batch.offsets.push_back(reinterpret_cast<const void*>(chunkPtr->indexOffset * sizeof(std::uint32_t)));
        batch.baseVertices.push_back(static_cast<GLint>(chunkPtr->vertexOffset));
    }

    auto emptyIt = std::remove_if(renderData.batches.begin(),
                                  renderData.batches.end(),
                                  [](const ChunkRenderBatch& batch)
                                  {
                                      return batch.counts.empty();
                                  });
    renderData.batches.erase(emptyIt, renderData.batches.end());

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
    columnManager_.clear();
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

    if (climateMap_)
    {
        climateMap_->clear();
    }

    if (surfaceMap_)
    {
        surfaceMap_->clear();
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
        if (chunk->hasBlocks)
        {
            chunk->hasBlocks = chunkHasSolidBlocks(*chunk);
        }

        columnManager_.updateColumn(*chunk, local.x, local.z);
        chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
    }

    invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});

    enqueueJob(chunk, JobType::Mesh, chunkCoord);
    markNeighborsForRemeshingIfNeeded(chunkCoord, local.x, localY, local.z);

    return true;
}

bool ChunkManager::Impl::placeBlock(const glm::ivec3& targetBlockPos, const glm::ivec3& faceNormal)
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

        chunk->blocks[blockIdx] = BlockId::Grass;
        chunk->hasBlocks = true;

        columnManager_.updateColumn(*chunk, local.x, local.z);
        chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
    }

    invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});

    enqueueJob(chunk, JobType::Mesh, chunkCoord);
    markNeighborsForRemeshingIfNeeded(chunkCoord, local.x, localY, local.z);

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
        if (targetViewDistance_ == kDefaultViewDistance)
        {
            std::cout << "Switching to extended render distance..." << std::endl;

            targetViewDistance_ = kExtendedViewDistance;
            kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);
            const long long width = static_cast<long long>(targetViewDistance_) * 2ll + 1ll;
            const long long totalColumns = width * width;
            std::cout << "Extended render distance target: " << targetViewDistance_ << " chunks (total: "
                      << totalColumns << " chunks)" << std::endl;
        }
        else
        {
            std::cout << "Switching to default render distance..." << std::endl;

            targetViewDistance_ = kDefaultViewDistance;
            kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);
            const long long width = static_cast<long long>(targetViewDistance_) * 2ll + 1ll;
            const long long totalColumns = width * width;
            std::cout << "Default render distance target: " << targetViewDistance_
                      << " chunks (total: " << totalColumns << " chunks)" << std::endl;
        }

        if (viewDistance_ > targetViewDistance_)
        {
            viewDistance_ = targetViewDistance_;
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error toggling view distance: " << ex.what() << std::endl;
        targetViewDistance_ = kDefaultViewDistance;
        viewDistance_ = std::min(viewDistance_, targetViewDistance_);
        kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);
    }
}

int ChunkManager::Impl::viewDistance() const noexcept
{
    return targetViewDistance_;
}

void ChunkManager::Impl::setRenderDistance(int distance) noexcept
{
    try
    {
        const int clampedDistance = std::max(distance, 1);
        targetViewDistance_ = clampedDistance;
        kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);

        if (viewDistance_ > targetViewDistance_)
        {
            viewDistance_ = targetViewDistance_;
        }

        const long long width = static_cast<long long>(targetViewDistance_) * 2ll + 1ll;
        const long long totalColumns = width * width;
        std::cout << "Render distance set to: " << targetViewDistance_ << " chunks (total: "
                  << totalColumns << " chunks)" << std::endl;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error setting render distance: " << ex.what() << std::endl;
    }
}

void ChunkManager::Impl::setLodEnabled(bool enabled)
{
    if (lodEnabled_ == enabled)
    {
        return;
    }

    lodEnabled_ = enabled;
    lodNearRadius_ = enabled ? std::max(4, targetViewDistance_ / 2) : 0;
    lodModeDirty_ = true;

    std::cout << "[ChunkManager] Surface LOD " << (enabled ? "enabled" : "disabled")
              << " via F3 toggle" << std::endl;
}

bool ChunkManager::Impl::lodEnabled() const noexcept
{
    return lodEnabled_;
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

glm::vec3 ChunkManager::Impl::findSafeSpawnPosition(float worldX, float worldZ) const
{
    const float halfWidth = kPlayerWidth * 0.5f;
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

    auto predictTreeCanopyTop = [&](const ColumnSample& columnSample) -> int
    {
        if (!columnSample.dominantBiome || !columnSample.dominantBiome->generatesTrees)
        {
            return ColumnManager::kNoHeight;
        }

        constexpr float kTreeBiomeWeightThreshold = 0.55f;
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

        const float density = noise_.fbm(static_cast<float>(baseX) * 0.05f,
                                         static_cast<float>(baseZ) * 0.05f,
                                         4,
                                         0.55f,
                                         2.0f);
        const float normalizedDensity = std::clamp((density + 1.0f) * 0.5f, 0.0f, 1.0f);
        const float randomValue = hashToUnitFloat(baseX, groundWorldY, baseZ);
        const float spawnThresholdBase = 0.015f + normalizedDensity * 0.02f;
        const float spawnThreshold =
            std::clamp(spawnThresholdBase * std::max(biome.treeDensityMultiplier, 0.0f), 0.0f, 1.0f);
        if (randomValue > spawnThreshold)
        {
            return ColumnManager::kNoHeight;
        }

        bool terrainSuitable = true;
        for (int dx = -1; dx <= 1 && terrainSuitable; ++dx)
        {
            for (int dz = -1; dz <= 1; ++dz)
            {
                if (dx == 0 && dz == 0)
                {
                    continue;
                }

                const ColumnSample neighborSample = sampleColumn(baseX + dx, baseZ + dz);
                if (std::abs(neighborSample.surfaceY - groundWorldY) > 1)
                {
                    terrainSuitable = false;
                    break;
                }
            }
        }

        if (!terrainSuitable)
        {
            return ColumnManager::kNoHeight;
        }

        constexpr int kTreeMinHeight = 6;
        constexpr int kTreeMaxHeight = 8;

        int trunkHeight = kTreeMinHeight +
                          static_cast<int>(hashToUnitFloat(baseX, groundWorldY + 1, baseZ) *
                                           static_cast<float>(kTreeMaxHeight - kTreeMinHeight + 1));
        trunkHeight = std::clamp(trunkHeight, kTreeMinHeight, kTreeMaxHeight);

        return groundWorldY + trunkHeight;
    };

    int predictedHighest = ColumnManager::kNoHeight;
    if (baseSample.dominantBiome)
    {
        predictedHighest = mergeHeight(predictedHighest, baseSample.surfaceY);
        predictedHighest = mergeHeight(predictedHighest, predictTreeCanopyTop(baseSample));
    }

    highestSolid = mergeHeight(highestSolid, predictedHighest);
    if (highestSolid == ColumnManager::kNoHeight)
    {
        highestSolid = 0;
    }

    const int clearanceHeight = static_cast<int>(std::ceil(kPlayerHeight)) + 1;
    const int searchTop = highestSolid + clearanceHeight + 2;
    int searchBottom = highestSolid - kChunkSizeY;
    if (searchBottom > searchTop)
    {
        searchBottom = searchTop - 1;
    }
    searchBottom = std::max(searchBottom, highestSolid - 2 * kChunkSizeY);
    searchBottom = std::max(searchBottom, 0);

    for (int y = searchTop; y >= searchBottom; --y)
    {
        bool hasGround = false;
        for (int dx = -1; dx <= 1 && !hasGround; ++dx)
        {
            for (int dz = -1; dz <= 1; ++dz)
            {
                const int checkX = static_cast<int>(std::floor(worldX + dx * halfWidth));
                const int checkZ = static_cast<int>(std::floor(worldZ + dz * halfWidth));
                if (isSolid(blockAt(glm::ivec3(checkX, y - 1, checkZ))))
                {
                    hasGround = true;
                    break;
                }
            }
        }

        if (!hasGround)
        {
            continue;
        }

        bool canFit = true;
        for (int dy = 0; dy < clearanceHeight && canFit; ++dy)
        {
            const int checkY = y + dy;
            for (int dx = -1; dx <= 1 && canFit; ++dx)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    const int checkX = static_cast<int>(std::floor(worldX + dx * halfWidth));
                    const int checkZ = static_cast<int>(std::floor(worldZ + dz * halfWidth));
                    if (isSolid(blockAt(glm::ivec3(checkX, checkY, checkZ))))
                    {
                        canFit = false;
                        break;
                    }
                }
            }
        }

        if (canFit)
        {
            const float safeY = static_cast<float>(y) + kCameraEyeHeight;
            std::cout << "Safe spawn found at height: " << safeY << " (feet at: " << y << ")" << std::endl;
            return glm::vec3(worldX, safeY, worldZ);
        }
    }

    std::cout << "Warning: No safe spawn found, spawning above highest predicted block" << std::endl;
    const int spawnFeetY = highestSolid + 1;
    const float fallbackY = static_cast<float>(spawnFeetY) + kCameraEyeHeight;
    return glm::vec3(worldX, fallbackY, worldZ);
}

ChunkProfilingSnapshot ChunkManager::Impl::sampleProfilingSnapshot()
{
    ChunkProfilingSnapshot snapshot{};

    const int generated = profilingCounters_.generatedChunks.exchange(0, std::memory_order_relaxed);
    const int meshed = profilingCounters_.meshedChunks.exchange(0, std::memory_order_relaxed);
    const int uploaded = profilingCounters_.uploadedChunks.exchange(0, std::memory_order_relaxed);

    snapshot.generatedChunks = generated;
    snapshot.meshedChunks = meshed;
    snapshot.uploadedChunks = uploaded;
    snapshot.uploadedBytes = profilingCounters_.uploadedBytes.exchange(0, std::memory_order_relaxed);
    snapshot.throttledUploads = profilingCounters_.throttledUploads.exchange(0, std::memory_order_relaxed);
    snapshot.deferredUploads = profilingCounters_.deferredUploads.exchange(0, std::memory_order_relaxed);
    snapshot.evictedChunks = profilingCounters_.evictedChunks.exchange(0, std::memory_order_relaxed);
    snapshot.verticalRadius = lastVerticalRadius_;
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

    snapshot.uploadBudgetBytes = uploadBudgetBytesThisFrame_;
    snapshot.uploadColumnLimit = uploadColumnLimitThisFrame_;
    const std::size_t pendingUploads = pendingUploadsLastFrame_;
    snapshot.pendingUploadChunks = static_cast<int>(
        std::min<std::size_t>(pendingUploads, static_cast<std::size_t>(std::numeric_limits<int>::max())));

    return snapshot;
}

void ChunkManager::Impl::startWorkerThreads()
{
    shouldStop_.store(false, std::memory_order_release);

    unsigned concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0)
    {
        concurrency = 2;
    }

    const unsigned minimum = 2u;
    unsigned desired = std::max(minimum, concurrency);

    if (kVerticalStreamingConfig.maxWorkerThreads > 0)
    {
        desired = std::min(desired, static_cast<unsigned>(kVerticalStreamingConfig.maxWorkerThreads));
    }

    desired = std::max(minimum, desired);

    workerThreadCount_ = static_cast<std::size_t>(desired);
    workerThreads_.reserve(workerThreadCount_);

    for (std::size_t i = 0; i < workerThreadCount_; ++i)
    {
        workerThreads_.emplace_back(&ChunkManager::Impl::workerThreadFunction, this);
    }
}

void ChunkManager::Impl::stopWorkerThreads()
{
    shouldStop_.store(true, std::memory_order_release);
    jobQueue_.stop();

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
    while (!shouldStop_.load(std::memory_order_acquire))
    {
        try
        {
            Job job = jobQueue_.waitAndPop();
            processJob(job);
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

void ChunkManager::Impl::enqueueJob(const std::shared_ptr<Chunk>& chunk, JobType type, const glm::ivec3& coord)
{
    if (!chunk)
    {
        return;
    }

    chunk->inFlight.fetch_add(1, std::memory_order_relaxed);
    try
    {
        jobQueue_.push(Job(type, coord, chunk));
    }
    catch (...)
    {
        chunk->inFlight.fetch_sub(1, std::memory_order_relaxed);
        throw;
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
        const auto start = std::chrono::steady_clock::now();
        generateChunkBlocks(*chunk);
        const auto end = std::chrono::steady_clock::now();
        const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        profilingCounters_.generationMicros.fetch_add(micros, std::memory_order_relaxed);
        profilingCounters_.generatedChunks.fetch_add(1, std::memory_order_relaxed);

        if (chunk->hasBlocks)
        {
            chunk->state.store(ChunkState::Meshing, std::memory_order_release);
            enqueueJob(chunk, JobType::Mesh, job.chunkCoord);
        }
        else
        {
            chunk->state.store(ChunkState::Uploaded, std::memory_order_release);
            chunk->meshReady = false;
            chunk->indexCount = 0;
        }
    }
    else if (job.type == JobType::Mesh)
    {
        const auto start = std::chrono::steady_clock::now();
        buildChunkMeshAsync(*chunk);
        const auto end = std::chrono::steady_clock::now();
        const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        profilingCounters_.meshingMicros.fetch_add(micros, std::memory_order_relaxed);
        profilingCounters_.meshedChunks.fetch_add(1, std::memory_order_relaxed);

        if (chunk->meshData.empty())
        {
            recycleChunkGPU(*chunk);
            chunk->meshReady = false;
            chunk->state.store(ChunkState::Uploaded, std::memory_order_release);
            chunk->indexCount = 0;
            return;
        }

        chunk->state.store(ChunkState::Ready, std::memory_order_release);
        queueChunkForUpload(chunk);
    }
}

std::shared_ptr<Chunk> ChunkManager::Impl::popNextChunkForUpload()
{
    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    while (!uploadQueue_.empty())
    {
        std::shared_ptr<Chunk> chunk = uploadQueue_.front().lock();
        uploadQueue_.pop_front();
        if (!chunk)
        {
            continue;
        }

        chunk->queuedForUpload = false;
        return chunk;
    }
    return nullptr;
}

void ChunkManager::Impl::queueChunkForUpload(const std::shared_ptr<Chunk>& chunk)
{
    if (!chunk)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    if (chunk->queuedForUpload)
    {
        return;
    }

    uploadQueue_.emplace_back(chunk);
    chunk->queuedForUpload = true;
}

void ChunkManager::Impl::requeueChunkForUpload(const std::shared_ptr<Chunk>& chunk, bool toFront)
{
    if (!chunk)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    if (chunk->queuedForUpload)
    {
        return;
    }

    if (toFront)
    {
        uploadQueue_.emplace_front(chunk);
    }
    else
    {
        uploadQueue_.emplace_back(chunk);
    }
    chunk->queuedForUpload = true;
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

ChunkManager::Impl::ChunkBufferPage ChunkManager::Impl::createBufferPage(std::size_t vertexCount, std::size_t indexCount)
{
    static constexpr std::size_t kDefaultVertexCapacity = 262144;
    static constexpr std::size_t kDefaultIndexCapacity = 393216;

    ChunkBufferPage page;
    page.vertexCapacity = std::max(nextPowerOfTwo(vertexCount), kDefaultVertexCapacity);
    page.indexCapacity = std::max(nextPowerOfTwo(indexCount), kDefaultIndexCapacity);

    glGenVertexArrays(1, &page.vao);
    glGenBuffers(1, &page.vbo);
    glGenBuffers(1, &page.ibo);

    glBindVertexArray(page.vao);

    glBindBuffer(GL_ARRAY_BUFFER, page.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(page.vertexCapacity * sizeof(Vertex)),
                 nullptr,
                 GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, tileCoord)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, atlasBase)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, atlasSize)));
    glEnableVertexAttribArray(4);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, page.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(page.indexCapacity * sizeof(std::uint32_t)),
                 nullptr,
                 GL_DYNAMIC_DRAW);

    glBindVertexArray(0);

    return page;
}

ChunkManager::Impl::ChunkAllocation ChunkManager::Impl::acquireChunkAllocation(std::size_t vertexCount,
                                                                               std::size_t indexCount)
{
    ChunkAllocation allocation{};
    if (vertexCount == 0 || indexCount == 0)
    {
        return allocation;
    }

    auto tryAllocateRange = [](std::vector<ChunkBufferPage::Range>& ranges,
                               std::size_t& cursor,
                               std::size_t capacity,
                               std::size_t count,
                               std::size_t& outOffset) -> bool
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
    };

    auto mergeRange = [](std::vector<ChunkBufferPage::Range>& ranges,
                         std::size_t offset,
                         std::size_t size)
    {
        if (size == 0)
        {
            return;
        }

        ChunkBufferPage::Range range{offset, size};
        auto it = std::lower_bound(ranges.begin(), ranges.end(), range.offset,
                                   [](const ChunkBufferPage::Range& lhs, std::size_t value)
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
    };

    std::lock_guard<std::mutex> lock(bufferPageMutex_);
    for (std::uint32_t pageIndex = 0; pageIndex < bufferPages_.size(); ++pageIndex)
    {
        ChunkBufferPage& page = bufferPages_[pageIndex];
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

        ++page.activeChunks;
        allocation.pageIndex = pageIndex;
        allocation.vertexOffset = vertexOffset;
        allocation.indexOffset = indexOffset;
        return allocation;
    }

    ChunkBufferPage newPage = createBufferPage(vertexCount, indexCount);
    bufferPages_.push_back(std::move(newPage));
    const std::uint32_t newIndex = static_cast<std::uint32_t>(bufferPages_.size() - 1);
    ChunkBufferPage& page = bufferPages_.back();

    std::size_t vertexOffset = 0;
    std::size_t indexOffset = 0;
    const bool vertexSuccess = tryAllocateRange(page.freeVertices, page.vertexCursor, page.vertexCapacity, vertexCount, vertexOffset);
    const bool indexSuccess = tryAllocateRange(page.freeIndices, page.indexCursor, page.indexCapacity, indexCount, indexOffset);
    (void)vertexSuccess;
    (void)indexSuccess;

    ++page.activeChunks;
    allocation.pageIndex = newIndex;
    allocation.vertexOffset = vertexOffset;
    allocation.indexOffset = indexOffset;
    return allocation;
}

void ChunkManager::Impl::releaseChunkAllocation(Chunk& chunk)
{
    const std::uint32_t pageIndex = chunk.bufferPageIndex;
    if (pageIndex == kInvalidChunkBufferPage)
    {
        chunk.vertexCount = 0;
        chunk.indexCount = 0;
        chunk.vertexOffset = 0;
        chunk.indexOffset = 0;
        return;
    }

    const std::size_t vertexCount = chunk.vertexCount;
    const std::size_t indexCount = static_cast<std::size_t>(chunk.indexCount);
    const std::size_t vertexOffset = chunk.vertexOffset;
    const std::size_t indexOffset = chunk.indexOffset;

    chunk.bufferPageIndex = kInvalidChunkBufferPage;
    chunk.vertexCount = 0;
    chunk.indexCount = 0;
    chunk.vertexOffset = 0;
    chunk.indexOffset = 0;

    auto mergeRange = [](std::vector<ChunkBufferPage::Range>& ranges,
                         std::size_t offset,
                         std::size_t size)
    {
        if (size == 0)
        {
            return;
        }

        ChunkBufferPage::Range range{offset, size};
        auto it = std::lower_bound(ranges.begin(), ranges.end(), range.offset,
                                   [](const ChunkBufferPage::Range& lhs, std::size_t value)
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
    };

    std::lock_guard<std::mutex> lock(bufferPageMutex_);
    if (pageIndex >= bufferPages_.size())
    {
        return;
    }

    ChunkBufferPage& page = bufferPages_[pageIndex];
    mergeRange(page.freeVertices, vertexOffset, vertexCount);
    mergeRange(page.freeIndices, indexOffset, indexCount);
    if (page.activeChunks > 0)
    {
        --page.activeChunks;
    }
}

void ChunkManager::Impl::recycleChunkGPU(Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(chunk.meshMutex);
    releaseChunkAllocation(chunk);
    chunk.meshData.clear();
    chunk.meshReady = false;
    chunk.queuedForUpload = false;
}

void ChunkManager::Impl::recycleChunkObject(std::shared_ptr<Chunk> chunk)
{
    if (!chunk)
    {
        return;
    }

    {
        std::lock_guard<std::mutex> meshLock(chunk->meshMutex);
        chunk->reset(chunk->coord);
    }

    std::lock_guard<std::mutex> lock(chunkPoolMutex_);
    if (chunkPool_.size() < kChunkPoolSoftCap)
    {
        chunkPool_.push_back(std::move(chunk));
    }
}

void ChunkManager::Impl::destroyBufferPages()
{
    std::lock_guard<std::mutex> lock(bufferPageMutex_);
    for (auto& page : bufferPages_)
    {
        if (page.ibo != 0)
        {
            glDeleteBuffers(1, &page.ibo);
        }
        if (page.vbo != 0)
        {
            glDeleteBuffers(1, &page.vbo);
        }
        if (page.vao != 0)
        {
            glDeleteVertexArrays(1, &page.vao);
        }
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
    std::size_t count = 0;
    for (const auto& entry : uploadQueue_)
    {
        if (!entry.expired())
        {
            ++count;
        }
    }
    return count;
}

ChunkManager::Impl::UploadBudgets ChunkManager::Impl::computeUploadBudgets(int verticalRadius)
{
    UploadBudgets budgets{};
    budgets.columnLimit = baseUploadsPerColumnLimit(verticalRadius);
    budgets.byteBudget = kUploadBudgetBytesPerFrame;
    budgets.queueSize = estimateUploadQueueSize();

    const std::size_t queueSize = budgets.queueSize;
    const std::size_t baseBudget = kUploadBudgetBytesPerFrame;

    constexpr std::size_t kQueueThreshold = 12;
    constexpr std::size_t kQueueStep = 8;
    constexpr int kMaxBurstSteps = 3;
    constexpr int kBurstMaxPerColumn = 12;

    int backlogSteps = 0;
    if (queueSize > kQueueThreshold)
    {
        backlogSteps = static_cast<int>((queueSize - kQueueThreshold + kQueueStep - 1) / kQueueStep);
    }

    if (queueSize > 0)
    {
        const bool hasFrameHeadroom = lastUploadBytesUsed_ + (baseBudget / 4) <= baseBudget;
        if (hasFrameHeadroom)
        {
            backlogSteps = std::max(backlogSteps, 1);
        }
    }

    backlogSteps = std::clamp(backlogSteps, 0, kMaxBurstSteps);

    const int multiplier = std::clamp(1 + backlogSteps, 1, kMaxBurstSteps + 1);
    budgets.byteBudget = baseBudget * static_cast<std::size_t>(multiplier);

    if (backlogSteps > 0)
    {
        const int boostedLimit = budgets.columnLimit + backlogSteps;
        budgets.columnLimit = std::min(boostedLimit, kBurstMaxPerColumn);
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

int ChunkManager::Impl::estimateMissingChunks(const glm::ivec3& center,
                                              int horizontalRadius,
                                              int verticalRadius) const
{
    const glm::ivec2 cameraColumn{center.x, center.z};
    const int cameraChunkY = center.y;

    int missing = 0;
    std::lock_guard<std::mutex> lock(chunksMutex);
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
            const int columnHeight = ensureColumnHeightCached(column, worldX, worldZ);
            const int columnRadius = columnRadiusForHeight(column,
                                                           cameraColumn,
                                                           cameraChunkY,
                                                           verticalRadius,
                                                           columnHeight);
            const int minChunkY = std::max(0, cameraChunkY - columnRadius);
            const int maxChunkY = std::max(minChunkY, cameraChunkY + columnRadius);
            for (int chunkY = minChunkY; chunkY <= maxChunkY; ++chunkY)
            {
                const glm::ivec3 coord{chunkX, chunkY, chunkZ};
                if (chunks_.find(coord) == chunks_.end())
                {
                    ++missing;
                }
            }
        }
    }

    return missing;
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
            const int radius = columnRadiusFor(column,
                                               cameraColumn,
                                               cameraChunkY,
                                               verticalRadius);
            verticalRadius = std::max(verticalRadius, radius);
        }
    }

    return std::clamp(verticalRadius,
                      kVerticalStreamingConfig.minRadiusChunks,
                      kVerticalStreamingConfig.maxRadiusChunks);
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

int ChunkManager::Impl::cacheSampledColumnHeight(const glm::ivec2& column, int worldX, int worldZ) const
{
    const ColumnSample sample = sampleColumn(worldX, worldZ);
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
    int highest = columnManager_.highestSolidBlock(worldX, worldZ);
    if (highest != ColumnManager::kNoHeight)
    {
        return highest;
    }

    int cachedHeight = ColumnManager::kNoHeight;
    if (tryGetPredictedColumnHeight(column, cachedHeight))
    {
        return cachedHeight;
    }

    return cacheSampledColumnHeight(column, worldX, worldZ);
}

void ChunkManager::Impl::invalidatePredictedColumn(const glm::ivec2& column) const
{
    std::lock_guard<std::mutex> lock(predictedColumnMutex_);
    predictedColumnHeights_.erase(column);
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
        const int columnHeight = ensureColumnHeightCached(column, worldX, worldZ);
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
        const bool surfaceOnly = shouldUseSurfaceOnly(center, candidate.coord);

        if (auto existing = getChunkShared(candidate.coord))
        {
            if (existing->surfaceOnly != surfaceOnly && existing->inFlight.load(std::memory_order_acquire) == 0)
            {
                if (enforceColumnCap && columnJobs >= maxJobsPerColumn)
                {
                    continue;
                }

                {
                    std::lock_guard<std::mutex> meshLock(existing->meshMutex);
                    std::fill(existing->blocks.begin(), existing->blocks.end(), BlockId::Air);
                    existing->meshData.clear();
                    existing->hasBlocks = false;
                    if (surfaceOnly)
                    {
                        if (!existing->lodData)
                        {
                            existing->lodData = std::make_unique<FarChunk>();
                        }
                    }
                    else
                    {
                        existing->lodData.reset();
                    }
                }

                existing->surfaceOnly = surfaceOnly;
                existing->state.store(ChunkState::Generating, std::memory_order_release);
                enqueueJob(existing, JobType::Generate, candidate.coord);
                --jobBudget;
                ++columnJobs;
                missingFound = true;
            }
            continue;
        }

        missingFound = true;

        if (enforceColumnCap && columnJobs >= maxJobsPerColumn)
        {
            continue;
        }

        if (ensureChunkAsync(candidate.coord, surfaceOnly))
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
    std::vector<glm::ivec3> toRemove;
    const glm::ivec2 cameraColumn{center.x, center.z};
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        toRemove.reserve(chunks_.size());
        for (const auto& [coord, chunkPtr] : chunks_)
        {
            if (coord.y < 0)
            {
                toRemove.push_back(coord);
                continue;
            }

            const int dx = coord.x - center.x;
            const int dz = coord.z - center.z;
            const int horizontalDistance = std::max(std::abs(dx), std::abs(dz));
            if (horizontalDistance > horizontalThreshold)
            {
                toRemove.push_back(coord);
                continue;
            }

            const glm::ivec2 column{coord.x, coord.z};
            const auto [minChunkY, maxChunkY] = columnSpanFor(column,
                                                              cameraColumn,
                                                              center.y,
                                                              verticalRadius);
            const int slack = kVerticalStreamingConfig.columnSlackChunks;
            if (coord.y < (minChunkY - slack) || coord.y > (maxChunkY + slack))
            {
                toRemove.push_back(coord);
            }
        }
    }

    int evictedCount = 0;
    for (const glm::ivec3& coord : toRemove)
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
        }

        if (chunk)
        {
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

bool ChunkManager::Impl::shouldUseSurfaceOnly(const glm::ivec3& center, const glm::ivec3& coord) const noexcept
{
    if (!lodEnabled_)
    {
        return false;
    }

    const int horizontalDistance = std::max(std::abs(coord.x - center.x), std::abs(coord.z - center.z));
    return horizontalDistance > lodNearRadius_;
}

bool ChunkManager::Impl::ensureChunkAsync(const glm::ivec3& coord, bool surfaceOnly)
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
            chunk->surfaceOnly = surfaceOnly;
            if (surfaceOnly)
            {
                if (!chunk->lodData)
                {
                    chunk->lodData = std::make_unique<FarChunk>();
                }
            }
            else
            {
                chunk->lodData.reset();
            }
            chunks_.emplace(coord, chunk);
        }

        enqueueJob(chunk, JobType::Generate, coord);
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
    const std::size_t initialBudget = uploadBudgetBytesThisFrame_;
    std::size_t remainingBudget = initialBudget;
    bool uploadedAnything = false;
    std::unordered_map<glm::ivec2, int, ColumnHasher> uploadsPerColumn;
    std::size_t attempts = 0;
    const int columnUploadLimit = std::max(1, uploadColumnLimitThisFrame_);

    while ((remainingBudget > 0 || !uploadedAnything) && attempts < kUploadQueueScanLimit)
    {
        ++attempts;
        std::shared_ptr<Chunk> chunk = popNextChunkForUpload();
        if (!chunk)
        {
            break;
        }

        if (!chunk->meshReady || chunk->state.load() != ChunkState::Ready)
        {
            continue;
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

        if (uploadedAnything && totalBytes > remainingBudget && totalBytes > 0)
        {
            requeueChunkForUpload(chunk, true);
            profilingCounters_.deferredUploads.fetch_add(1, std::memory_order_relaxed);
            break;
        }

        uploadChunkMesh(*chunk);
        chunk->state.store(ChunkState::Uploaded, std::memory_order_release);
        chunk->meshReady = false;
        uploadedAnything = true;
        ++columnUploads;

        profilingCounters_.uploadedChunks.fetch_add(1, std::memory_order_relaxed);
        profilingCounters_.uploadedBytes.fetch_add(totalBytes, std::memory_order_relaxed);

        if (totalBytes >= remainingBudget)
        {
            remainingBudget = 0;
        }
        else
        {
            remainingBudget -= totalBytes;
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

    pendingUploadsLastFrame_ = estimateUploadQueueSize();
}

void ChunkManager::Impl::uploadChunkMesh(Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(chunk.meshMutex);

    if (chunk.meshData.empty())
    {
        releaseChunkAllocation(chunk);
        chunk.meshData.clear();
        return;
    }

    const std::size_t vertexCount = chunk.meshData.vertices.size();
    const std::size_t indexCount = chunk.meshData.indices.size();

    releaseChunkAllocation(chunk);
    ChunkAllocation allocation = acquireChunkAllocation(vertexCount, indexCount);
    if (allocation.pageIndex == kInvalidChunkBufferPage)
    {
        chunk.meshData.clear();
        return;
    }

    chunk.bufferPageIndex = allocation.pageIndex;
    chunk.vertexOffset = allocation.vertexOffset;
    chunk.indexOffset = allocation.indexOffset;
    chunk.vertexCount = vertexCount;

    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ibo = 0;
    {
        std::lock_guard<std::mutex> pageLock(bufferPageMutex_);
        if (allocation.pageIndex < bufferPages_.size())
        {
            ChunkBufferPage& page = bufferPages_[allocation.pageIndex];
            vao = page.vao;
            vbo = page.vbo;
            ibo = page.ibo;
        }
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    if (vertexCount > 0)
    {
        glBufferSubData(GL_ARRAY_BUFFER,
                        static_cast<GLintptr>(chunk.vertexOffset * sizeof(Vertex)),
                        static_cast<GLsizeiptr>(vertexCount * sizeof(Vertex)),
                        chunk.meshData.vertices.data());
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    if (indexCount > 0)
    {
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER,
                        static_cast<GLintptr>(chunk.indexOffset * sizeof(std::uint32_t)),
                        static_cast<GLsizeiptr>(indexCount * sizeof(std::uint32_t)),
                        chunk.meshData.indices.data());
    }

    chunk.indexCount = static_cast<GLsizei>(indexCount);
    glBindVertexArray(0);

    chunk.meshData.clear();
}

void ChunkManager::Impl::buildSurfaceOnlyMesh(Chunk& chunk)
{
    if (!chunk.lodData)
    {
        chunk.meshData.clear();
        return;
    }

    FarChunk& lod = *chunk.lodData;
    chunk.meshData.clear();

    const glm::vec3 baseOrigin = lod.origin;
    const glm::vec3 normal{0.0f, 1.0f, 0.0f};
    const float step = static_cast<float>(lod.lodStep);

    for (int rx = 0; rx < FarChunk::kColumnsX; ++rx)
    {
        for (int rz = 0; rz < FarChunk::kColumnsZ; ++rz)
        {
            const FarChunk::SurfaceCell& cell = lod.strata[FarChunk::index(rx, rz)];
            if (cell.block == BlockId::Air || cell.worldY == std::numeric_limits<int>::min())
            {
                continue;
            }

            const float worldY = static_cast<float>(cell.worldY + 1);
            const float minX = baseOrigin.x + static_cast<float>(rx) * step;
            const float maxX = baseOrigin.x + static_cast<float>(rx + 1) * step;
            const float minZ = baseOrigin.z + static_cast<float>(rz) * step;
            const float maxZ = baseOrigin.z + static_cast<float>(rz + 1) * step;

            const glm::vec3 p0{minX, worldY, minZ};
            const glm::vec3 p1{maxX, worldY, minZ};
            const glm::vec3 p2{maxX, worldY, maxZ};
            const glm::vec3 p3{minX, worldY, maxZ};

            const auto& uv = blockUVTable_[toIndex(cell.block)].faces[toIndex(BlockFace::Top)];

            const std::uint32_t baseIndex = static_cast<std::uint32_t>(chunk.meshData.vertices.size());
            chunk.meshData.vertices.push_back(Vertex{p0, normal, glm::vec2{0.0f, 0.0f}, uv.base, uv.size});
            chunk.meshData.vertices.push_back(Vertex{p1, normal, glm::vec2{1.0f, 0.0f}, uv.base, uv.size});
            chunk.meshData.vertices.push_back(Vertex{p2, normal, glm::vec2{1.0f, 1.0f}, uv.base, uv.size});
            chunk.meshData.vertices.push_back(Vertex{p3, normal, glm::vec2{0.0f, 1.0f}, uv.base, uv.size});

            chunk.meshData.indices.push_back(baseIndex + 0);
            chunk.meshData.indices.push_back(baseIndex + 1);
            chunk.meshData.indices.push_back(baseIndex + 2);
            chunk.meshData.indices.push_back(baseIndex + 0);
            chunk.meshData.indices.push_back(baseIndex + 2);
            chunk.meshData.indices.push_back(baseIndex + 3);
        }
    }
}

void ChunkManager::Impl::buildChunkMeshAsync(Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(chunk.meshMutex);
    chunk.meshData.clear();

    if (!chunk.hasBlocks)
    {
        chunk.meshReady = true;
        return;
    }

    if (chunk.surfaceOnly)
    {
        buildSurfaceOnlyMesh(chunk);
        chunk.meshReady = true;
        return;
    }

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

    auto localToWorld = [&](int lx, int ly, int lz) -> glm::ivec3
    {
        return glm::ivec3(baseWorldX + lx, baseWorldY + ly, baseWorldZ + lz);
    };

    auto sampleBlock = [&](int lx, int ly, int lz) -> BlockId
    {
        if (lx >= 0 && lx < kChunkSizeX && ly >= 0 && ly < kChunkSizeY && lz >= 0 && lz < kChunkSizeZ)
        {
            return chunk.blocks[blockIndex(lx, ly, lz)];
        }

        return blockAt(localToWorld(lx, ly, lz));
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

        bool operator==(const FaceMaterial& other) const noexcept
        {
            return uvBase == other.uvBase &&
                   uvSize == other.uvSize &&
                   uAxis == other.uAxis &&
                   vAxis == other.vAxis &&
                   face == other.face;
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

    auto makeMaterial = [&](BlockId block, const glm::vec3& normal) -> FaceMaterial
    {
        FaceMaterial material{};

        const BlockFace face = [&]() -> BlockFace
        {
            if (normal.y > 0.5f) return BlockFace::Top;
            if (normal.y < -0.5f) return BlockFace::Bottom;
            if (normal.x > 0.5f) return BlockFace::East;
            if (normal.x < -0.5f) return BlockFace::West;
            if (normal.z > 0.5f) return BlockFace::South;
            return BlockFace::North;
        }();

        material.face = face;

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

        if (dir == FaceDir::Negative)
        {
            std::swap(positions[1], positions[3]);
        }

        const glm::vec3 uAxisVec = glm::vec3(material.uAxis);
        const glm::vec3 vAxisVec = glm::vec3(material.vAxis);

        const std::size_t vertexStart = chunk.meshData.vertices.size();
        for (int i = 0; i < 4; ++i)
        {
            const glm::vec3& pos = positions[i];

            Vertex vertex{};
            vertex.position = pos;
            vertex.normal = normal;
            vertex.tileCoord = glm::vec2(glm::dot(pos, uAxisVec), glm::dot(pos, vAxisVec));
            vertex.atlasBase = material.uvBase;
            vertex.atlasSize = material.uvSize;
            chunk.meshData.vertices.push_back(vertex);
        }

        chunk.meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 0));
        chunk.meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 1));
        chunk.meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 2));
        chunk.meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 2));
        chunk.meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 3));
        chunk.meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 0));
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

                        const bool positiveSolid = isSolid(sampleBlock(positiveLocal.x, positiveLocal.y, positiveLocal.z));
                        const bool negativeSolid = isSolid(sampleBlock(negativeLocal.x, negativeLocal.y, negativeLocal.z));

                        glm::ivec3 owningLocal{0};
                        bool createFace = false;

                        if (dir == FaceDir::Positive)
                        {
                            if (negativeSolid && !positiveSolid && isInsideChunk(negativeLocal))
                            {
                                owningLocal = negativeLocal;
                                createFace = true;
                            }
                        }
                        else
                        {
                            if (positiveSolid && !negativeSolid && isInsideChunk(positiveLocal))
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
                            const BlockId owningBlock = chunk.blocks[blockIdx];
                            cell.material = makeMaterial(owningBlock, normal);
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
                        while (ci + runLengthC < sizeC)
                        {
                            const MaskCell& nextCell = mask[maskIndex(bi, ci + runLengthC)];
                            if (!nextCell.exists || !(nextCell.material == material))
                            {
                                break;
                            }
                            ++runLengthC;
                        }

                        int runHeightB = 1;
                        while (bi + runHeightB < sizeB)
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

    chunk.meshReady = true;
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
            chunk = std::move(chunkPool_.back());
            chunkPool_.pop_back();
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

        ChunkState neighborState = neighbor->state.load();
        if (neighborState != ChunkState::Uploaded && neighborState != ChunkState::Remeshing)
        {
            return;
        }

        neighbor->state.store(ChunkState::Remeshing, std::memory_order_release);
        try
        {
            enqueueJob(neighbor, JobType::Mesh, neighborCoord);
        }
        catch (const std::exception& ex)
        {
            std::cerr << "Failed to queue remesh for neighbor (" << neighborCoord.x << ", " << neighborCoord.y
                      << ", " << neighborCoord.z << "): " << ex.what() << std::endl;
        }
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







ColumnSample ChunkManager::Impl::sampleColumn(int worldX, int worldZ, int slabMinWorldY, int slabMaxWorldY) const
{
    if (slabMinWorldY > slabMaxWorldY)
    {
        std::swap(slabMinWorldY, slabMaxWorldY);
    }

    if (!surfaceMap_)
    {
        throw std::runtime_error("Surface map is not initialized");
    }

    ColumnSample sample{};
    const terrain::SurfaceColumn& surfaceColumn = surfaceMap_->column(worldX, worldZ);

    sample.dominantBiome = surfaceColumn.dominantBiome;
    sample.dominantWeight = surfaceColumn.dominantWeight;
    sample.surfaceY = surfaceColumn.surfaceY;
    sample.minSurfaceY = std::min(sample.surfaceY, slabMinWorldY);
    sample.maxSurfaceY = std::max(sample.surfaceY, slabMaxWorldY);
    sample.soilCreepCoefficient = surfaceColumn.soilCreepCoefficient;
    sample.roughAmplitude = surfaceColumn.roughAmplitude;
    sample.hillAmplitude = surfaceColumn.hillAmplitude;
    sample.mountainAmplitude = surfaceColumn.mountainAmplitude;
    sample.distanceToShore = std::abs(static_cast<float>(surfaceColumn.surfaceY - globalSeaLevel_));

    sample.slabHasSolid = surfaceColumn.surfaceY >= slabMinWorldY;
    if (sample.slabHasSolid)
    {
        sample.slabHighestSolidY = std::min(surfaceColumn.surfaceY, slabMaxWorldY);
    }

    return sample;
}



void ChunkManager::Impl::generateSurfaceOnlyChunk(Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(chunk.meshMutex);
    std::fill(chunk.blocks.begin(), chunk.blocks.end(), BlockId::Air);

    if (!chunk.lodData)
    {
        chunk.lodData = std::make_unique<FarChunk>();
    }

    FarChunk& lod = *chunk.lodData;
    lod.origin = glm::vec3(static_cast<float>(chunk.coord.x * kChunkSizeX),
                           static_cast<float>(chunk.minWorldY),
                           static_cast<float>(chunk.coord.z * kChunkSizeZ));
    lod.size = glm::ivec3{kChunkSizeX, kChunkSizeY, kChunkSizeZ};
    lod.lodStep = FarChunk::kColumnStep;
    lod.thickness = 1;

    const int baseWorldX = chunk.coord.x * kChunkSizeX;
    const int baseWorldZ = chunk.coord.z * kChunkSizeZ;
    const int slabMinWorldY = chunk.minWorldY;
    const int slabMaxWorldY = chunk.maxWorldY;

    bool anySolid = false;

    for (int rx = 0; rx < FarChunk::kColumnsX; ++rx)
    {
        for (int rz = 0; rz < FarChunk::kColumnsZ; ++rz)
        {
            int bestWorldY = std::numeric_limits<int>::min();
            BlockId bestBlock = BlockId::Air;
            int bestLocalX = -1;
            int bestLocalZ = -1;

            for (int localX = rx * FarChunk::kColumnStep;
                 localX < (rx + 1) * FarChunk::kColumnStep && localX < kChunkSizeX;
                 ++localX)
            {
                for (int localZ = rz * FarChunk::kColumnStep;
                     localZ < (rz + 1) * FarChunk::kColumnStep && localZ < kChunkSizeZ;
                     ++localZ)
                {
                    const int worldX = baseWorldX + localX;
                    const int worldZ = baseWorldZ + localZ;

                    ColumnSample sample = sampleColumn(worldX, worldZ, slabMinWorldY, slabMaxWorldY);
                    if (!sample.dominantBiome || !sample.slabHasSolid)
                    {
                        continue;
                    }

                    BlockId surfaceBlock = sample.dominantBiome->surfaceBlock;
                    if (!sample.dominantBiome->isOcean())
                    {
                        constexpr float kBeachDistanceRange = 6.0f;
                        constexpr int kBeachHeightBand = 2;
                        const bool nearSeaLevel = std::abs(sample.surfaceY - globalSeaLevel_) <= kBeachHeightBand;
                        if (nearSeaLevel && sample.distanceToShore <= kBeachDistanceRange)
                        {
                            const float beachNoise = hashToUnitFloat(worldX, sample.surfaceY, worldZ);
                            surfaceBlock = beachNoise < 0.55f ? BlockId::Sand : BlockId::Grass;
                        }
                    }

                    const int highestSolidWorld = sample.slabHighestSolidY;
                    if (highestSolidWorld < slabMinWorldY || highestSolidWorld > slabMaxWorldY)
                    {
                        continue;
                    }

                    if (highestSolidWorld > bestWorldY)
                    {
                        bestWorldY = highestSolidWorld;
                        bestBlock = surfaceBlock;
                        bestLocalX = localX;
                        bestLocalZ = localZ;
                    }
                }
            }

            FarChunk::SurfaceCell cell{};

            if (bestLocalX >= 0 && bestBlock != BlockId::Air)
            {
                cell.worldY = bestWorldY;
                cell.block = bestBlock;

                const int localY = bestWorldY - chunk.minWorldY;
                if (localY >= 0 && localY < kChunkSizeY)
                {
                    chunk.blocks[blockIndex(bestLocalX, localY, bestLocalZ)] = bestBlock;
                    anySolid = true;
                }
            }

            lod.strata[FarChunk::index(rx, rz)] = cell;
        }
    }

    chunk.hasBlocks = anySolid;
    if (anySolid)
    {
        columnManager_.updateChunk(chunk);
    }
    else
    {
        columnManager_.removeChunk(chunk);
    }
    invalidatePredictedColumn({chunk.coord.x, chunk.coord.z});
}

void ChunkManager::Impl::generateChunkBlocks(Chunk& chunk)
{
    std::vector<PendingStructureEdit> externalEdits;
    bool anySolid = false;

    if (chunk.surfaceOnly)
    {
        generateSurfaceOnlyChunk(chunk);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        std::fill(chunk.blocks.begin(), chunk.blocks.end(), BlockId::Air);

        const int baseWorldX = chunk.coord.x * kChunkSizeX;
        const int baseWorldZ = chunk.coord.z * kChunkSizeZ;
        const int slabMinWorldY = chunk.minWorldY;
        const int slabMaxWorldY = chunk.maxWorldY;

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
                    surfaceMap_->getFragment({fx, fz});
                }
            }
        }

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
            chunk.blocks[blockIndex(localX, localY, localZ)] = block;
            if (block != BlockId::Air)
            {
                anySolid = true;
            }
        };

        terrain::ChunkGenerationSummary summary{};
        if (terrainGenerator_)
        {
            summary = terrainGenerator_->generateChunkColumns(chunk.coord,
                                                              chunk.minWorldY,
                                                              chunk.maxWorldY,
                                                              kChunkSizeX,
                                                              kChunkSizeY,
                                                              kChunkSizeZ,
                                                              setBlockDirect,
                                                              columnResults);
            anySolid = anySolid || summary.anySolid;
        }

        const bool slabContainsTerrain = summary.slabContainsTerrain;

        auto setOrQueueBlock = [&](int worldX, int worldY, int worldZ, BlockId block, bool replaceSolid)
        {
            const glm::ivec3 worldPos{worldX, worldY, worldZ};
            const glm::ivec3 targetChunk = worldToChunkCoords(worldX, worldY, worldZ);
            if (targetChunk == chunk.coord)
            {
                if (worldY < chunk.minWorldY || worldY > chunk.maxWorldY)
                {
                    return;
                }

                const glm::ivec3 local = localBlockCoords(worldPos, targetChunk);
                const int localY = worldY - chunk.minWorldY;
                BlockId& destination = chunk.blocks[blockIndex(local.x, localY, local.z)];
                if (!replaceSolid && destination != BlockId::Air)
                {
                    return;
                }

                destination = block;
                if (block != BlockId::Air)
                {
                    anySolid = true;
                }
            }
            else
            {
                if (block == BlockId::Air)
                {
                    return;
                }

                externalEdits.push_back(PendingStructureEdit{targetChunk, worldPos, block, replaceSolid});
            }
        };

        auto getLocalColumnSample = [&](int worldX, int worldZ) -> ColumnSample
        {
            if (worldX >= baseWorldX && worldX < baseWorldX + kChunkSizeX && worldZ >= baseWorldZ
                && worldZ < baseWorldZ + kChunkSizeZ)
            {
                const int localX = worldX - baseWorldX;
                const int localZ = worldZ - baseWorldZ;
                return columnResults[columnIndex(localX, localZ)].sample;
            }
            return sampleColumn(worldX, worldZ);
        };

        if (slabContainsTerrain)
        {
            constexpr int kTreeMinHeight = 6;
            constexpr int kTreeMaxHeight = 8;
            constexpr int kTreeMaxRadius = 2;

            const int minWorldX = baseWorldX - kTreeMaxRadius;
            const int maxWorldX = baseWorldX + kChunkSizeX + kTreeMaxRadius - 1;
            const int minWorldZ = baseWorldZ - kTreeMaxRadius;
            const int maxWorldZ = baseWorldZ + kChunkSizeZ + kTreeMaxRadius - 1;

            for (int worldX = minWorldX; worldX <= maxWorldX; ++worldX)
            {
                for (int worldZ = minWorldZ; worldZ <= maxWorldZ; ++worldZ)
                {
                    const ColumnSample columnSample = getLocalColumnSample(worldX, worldZ);
                    const BiomeDefinition& biome = *columnSample.dominantBiome;
                    if (!biome.generatesTrees)
                    {
                        continue;
                    }

                    constexpr float kTreeBiomeWeightThreshold = 0.55f;
                    if (columnSample.dominantWeight < kTreeBiomeWeightThreshold)
                    {
                        continue;
                    }

                    const int groundWorldY = columnSample.surfaceY;
                    const int groundLocalY = groundWorldY - chunk.minWorldY;
                    if (groundLocalY < 0 || groundLocalY >= kChunkSizeY)
                    {
                        continue;
                    }

                    if (groundLocalY <= 2)
                    {
                        continue;
                    }

                    const int localX = worldX - baseWorldX;
                    const int localZ = worldZ - baseWorldZ;
                    if (localX >= 0 && localX < kChunkSizeX && localZ >= 0 && localZ < kChunkSizeZ)
                    {
                        const std::size_t blockIdx = blockIndex(localX, groundLocalY, localZ);
                        if (chunk.blocks[blockIdx] != biome.surfaceBlock)
                        {
                            continue;
                        }
                    }

                    const float density = noise_.fbm(static_cast<float>(worldX) * 0.05f,
                                                     static_cast<float>(worldZ) * 0.05f,
                                                     4,
                                                     0.55f,
                                                     2.0f);
                    const float normalizedDensity = std::clamp((density + 1.0f) * 0.5f, 0.0f, 1.0f);
                    const float randomValue = hashToUnitFloat(worldX, groundWorldY, worldZ);
                    const float spawnThresholdBase = 0.015f + normalizedDensity * 0.02f;
                    const float spawnThreshold =
                        std::clamp(spawnThresholdBase * std::max(biome.treeDensityMultiplier, 0.0f), 0.0f, 1.0f);
                    if (randomValue > spawnThreshold)
                    {
                        continue;
                    }

                    bool terrainSuitable = true;
                    for (int dx = -1; dx <= 1 && terrainSuitable; ++dx)
                    {
                        for (int dz = -1; dz <= 1; ++dz)
                        {
                            if (dx == 0 && dz == 0)
                            {
                                continue;
                            }

                            const ColumnSample neighborSample = getLocalColumnSample(worldX + dx, worldZ + dz);
                            const int neighborHeight = neighborSample.surfaceY;
                            if (std::abs(neighborHeight - groundWorldY) > 1)
                            {
                                terrainSuitable = false;
                                break;
                            }
                        }
                    }

                    if (!terrainSuitable)
                    {
                        continue;
                    }

                    int trunkHeight = kTreeMinHeight +
                                      static_cast<int>(hashToUnitFloat(worldX, groundWorldY + 1, worldZ) *
                                                       static_cast<float>(kTreeMaxHeight - kTreeMinHeight + 1));
                    trunkHeight = std::clamp(trunkHeight, kTreeMinHeight, kTreeMaxHeight);

                    for (int dy = 0; dy < trunkHeight; ++dy)
                    {
                        setOrQueueBlock(worldX, groundWorldY + dy, worldZ, BlockId::Wood, true);
                    }

                    const int canopyBaseWorld = groundWorldY + trunkHeight - 3;
                    const int canopyTopWorld = groundWorldY + trunkHeight;
                    for (int worldY = canopyBaseWorld; worldY <= canopyTopWorld; ++worldY)
                    {
                        const int layer = worldY - canopyBaseWorld;
                        int radius = 2;
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

                                setOrQueueBlock(worldX + dx, worldY, worldZ + dz, BlockId::Leaves, false);
                            }
                        }
                    }
                }
            }
        }

        const bool appliedPending = applyPendingStructureEditsLocked(chunk);
        if (appliedPending)
        {
            anySolid = true;
        }

        chunk.hasBlocks = anySolid;
        columnManager_.updateChunk(chunk);
    }

    if (!externalEdits.empty())
    {
        dispatchStructureEdits(externalEdits);
    }

    invalidatePredictedColumn({chunk.coord.x, chunk.coord.z});
}


bool ChunkManager::Impl::applyPendingStructureEditsLocked(Chunk& chunk)
{
    std::vector<PendingStructureEdit> edits;
    {
        std::lock_guard<std::mutex> lock(pendingStructureMutex_);
        auto it = pendingStructureEdits_.find(chunk.coord);
        if (it != pendingStructureEdits_.end())
        {
            edits = std::move(it->second);
            pendingStructureEdits_.erase(it);
        }
    }

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
                chunk->hasBlocks = true;
                columnManager_.updateChunk(*chunk);
                invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});
            }
        }

        if (!wroteSolid)
        {
            continue;
        }

        if (state == ChunkState::Uploaded || state == ChunkState::Ready || state == ChunkState::Remeshing)
        {
            chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
            enqueueJob(chunk, JobType::Mesh, coord);
        }
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

void ChunkManager::setAtlasTexture(GLuint texture) noexcept
{
    impl_->setAtlasTexture(texture);
}

void ChunkManager::setBlockTextureAtlasConfig(const glm::ivec2& textureSizePixels, int tileSizePixels)
{
    impl_->setBlockTextureAtlasConfig(textureSizePixels, tileSizePixels);
}

void ChunkManager::update(const glm::vec3& cameraPos)
{
    impl_->update(cameraPos);
}

ChunkRenderData ChunkManager::buildRenderData(const Frustum& frustum) const
{
    return impl_->buildRenderData(frustum);
}

float ChunkManager::surfaceHeight(float worldX, float worldZ) const noexcept
{
    return impl_->surfaceHeight(worldX, worldZ);
}

void ChunkManager::clear()
{
    impl_->clear();
}

bool ChunkManager::destroyBlock(const glm::ivec3& worldPos)
{
    return impl_->destroyBlock(worldPos);
}

bool ChunkManager::placeBlock(const glm::ivec3& targetBlockPos, const glm::ivec3& faceNormal)
{
    return impl_->placeBlock(targetBlockPos, faceNormal);
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

void ChunkManager::setRenderDistance(int distance) noexcept
{
    impl_->setRenderDistance(distance);
}

void ChunkManager::setLodEnabled(bool enabled)
{
    impl_->setLodEnabled(enabled);
}

bool ChunkManager::lodEnabled() const noexcept
{
    return impl_->lodEnabled();
}

BlockId ChunkManager::blockAt(const glm::ivec3& worldPos) const noexcept
{
    return impl_->blockAt(worldPos);
}

glm::vec3 ChunkManager::findSafeSpawnPosition(float worldX, float worldZ) const
{
    return impl_->findSafeSpawnPosition(worldX, worldZ);
}

ChunkProfilingSnapshot ChunkManager::sampleProfilingSnapshot()
{
    return impl_->sampleProfilingSnapshot();
}

