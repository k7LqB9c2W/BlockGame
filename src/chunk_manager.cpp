// chunk_manager.cpp
// Implements the chunk streaming, terrain generation, and GPU upload subsystem.

#include "chunk_manager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
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
#include <glm/gtc/type_ptr.hpp>

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

enum class BiomeId : std::uint8_t
{
    Grasslands = 0,
    Forest,
    Desert,
    LittleMountains,
    Ocean,
    Count
};

constexpr std::size_t toIndex(BiomeId biome) noexcept
{
    return static_cast<std::size_t>(biome);
}

struct BiomeDefinition
{
    BiomeId id;
    const char* name;
    BlockId surfaceBlock;
    BlockId fillerBlock;
    bool generatesTrees;
    float treeDensityMultiplier;
    float heightOffset;
    float heightScale;
    int minHeight;
    int maxHeight;
    float baseSlopeBias; // Bias toward flattening macro terrain (0 = legacy shaping, 1 = offset-driven)
    float maxGradient;   // Maximum deviation from the local base height before clamping (blocks)
    float footprintMultiplier{1.0f};
};

constexpr std::size_t kBiomeCount = toIndex(BiomeId::Count);

// The biome height ranges are expressed in absolute world Y units so that terrain
// generation remains stable regardless of the chunk edge length. These values
// mirror the pre-cubic-chunk tuning that produced varied hills for each biome,
// while oceans clamp to a fixed sea level so their surfaces stay flat.
constexpr int kGrasslandsMaxSurfaceHeight = 61;
constexpr int kForestMaxSurfaceHeight = 61;
constexpr int kDesertMaxSurfaceHeight = 60;
constexpr int kLittleMountainsMinSurfaceHeight = 30;
constexpr int kLittleMountainsMaxSurfaceHeight = 820;
constexpr int kGlobalSeaLevel = 20;
constexpr int kOceanMaxSurfaceHeight = kGlobalSeaLevel;

constexpr std::array<BiomeDefinition, kBiomeCount> kBiomeDefinitions{ {
    {BiomeId::Grasslands,
     "Grasslands",
     BlockId::Grass,
     BlockId::Grass,
     false,
     0.0f,
     16.0f,
     0.35f,
     2,
     kGrasslandsMaxSurfaceHeight,
     0.65f,
     6.0f,
     1.0f},
    {BiomeId::Forest,
     "Forest",
     BlockId::Grass,
     BlockId::Grass,
     true,
     3.5f,
     19.0f,
     0.45f,
     3,
     kForestMaxSurfaceHeight,
     0.45f,
     9.0f,
     1.0f},
    {BiomeId::Desert,
     "Desert",
     BlockId::Sand,
     BlockId::Sand,
     false,
     0.0f,
     19.0f,
     0.85f,
     1,
     kDesertMaxSurfaceHeight,
     0.30f,
     14.0f,
     1.0f},
    {BiomeId::LittleMountains,
     "Little Mountains",
     BlockId::Grass,
     BlockId::Stone,
     false,
     0.0f,
     270.0f,
     44.0f,
     kLittleMountainsMinSurfaceHeight,
     kLittleMountainsMaxSurfaceHeight,
     0.1f,
     320.0f,
     3.3f},
    {BiomeId::Ocean,
     "Ocean",
     BlockId::Water,
     BlockId::Water,
     false,
     0.0f,
     static_cast<float>(kGlobalSeaLevel),
     0.0f,
     kGlobalSeaLevel,
     kOceanMaxSurfaceHeight,
     1.0f,
     1.0f,
     1.0f},
} };

constexpr float computeMaxFootprintMultiplier()
{
    float maxMultiplier = 0.0f;
    for (const auto& definition : kBiomeDefinitions)
    {
        maxMultiplier = (definition.footprintMultiplier > maxMultiplier) ? definition.footprintMultiplier : maxMultiplier;
    }
    return maxMultiplier;
}

constexpr int ceilToIntPositive(float value)
{
    const int truncated = static_cast<int>(value);
    return (static_cast<float>(truncated) < value) ? truncated + 1 : truncated;
}

constexpr float kMaxBiomeFootprintMultiplier = computeMaxFootprintMultiplier();
constexpr int kBiomeRegionSearchRadius = std::max(1, ceilToIntPositive(kMaxBiomeFootprintMultiplier * 0.5f));
constexpr std::size_t kBiomeRegionCandidateCapacity =
    static_cast<std::size_t>((kBiomeRegionSearchRadius * 2 + 1) * (kBiomeRegionSearchRadius * 2 + 1));

struct ColumnSample
{
    const BiomeDefinition* dominantBiome{nullptr};
    float dominantWeight{0.0f};
    int surfaceY{0};
    int minSurfaceY{0};
    int maxSurfaceY{0};
    int slabHighestSolidY{std::numeric_limits<int>::min()};
    float continentMask{0.0f};
    float baseElevation{0.0f};
    float oceanContribution{0.0f};
    float landContribution{0.0f};
    float oceanShare{0.0f};
    float landShare{0.0f};
    float shorelineBlend{0.0f};
    float distanceToShore{0.0f};
    bool slabHasSolid{false};
};

// To introduce a new biome:
// 1. Extend BiomeId before Count.
// 2. Append a definition to kBiomeDefinitions with the desired blocks and tuning parameters.
// 3. Provide textures for any new blocks in setBlockTextureAtlasConfig.

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

class OpenSimplexNoise
{
public:
    explicit OpenSimplexNoise(unsigned seed = 2025u);

    float noise(float x, float y) const noexcept;
    float fbm(float x, float y, int octaves, float persistence, float lacunarity) const noexcept;
    float ridge(float x, float y, int octaves, float lacunarity, float gain) const noexcept;
    glm::vec2 sampleGradient(float x, float y) const noexcept;

private:
    std::array<int, 512> permutation_{};
    std::array<int, 512> permutationMod8_{};

    static const std::array<glm::vec2, 8> kGradients;
};

class PerlinNoise
{
public:
    explicit PerlinNoise(unsigned seed = 2025u);

    float noise(float x, float y) const noexcept;
    float fbm(float x, float y, int octaves, float persistence, float lacunarity) const noexcept;
    float ridge(float x, float y, int octaves, float lacunarity, float gain) const noexcept;

private:
    std::array<int, 512> permutation_{};

    static float fade(float t) noexcept;
    static float lerp(float a, float b, float t) noexcept;
    static float grad(int hash, float x, float y) noexcept;
};

} // namespace

struct ChunkManager::Impl
{
    explicit Impl(unsigned seed);
    ~Impl();

    void setAtlasTexture(GLuint texture) noexcept;
    void setBlockTextureAtlasConfig(const glm::ivec2& textureSizePixels, int tileSizePixels);
    void update(const glm::vec3& cameraPos);
    void render(GLuint shaderProgram,
                const glm::mat4& viewProj,
                const glm::vec3& cameraPos,
                const Frustum& frustum,
                const ChunkShaderUniformLocations& uniforms) const;

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

    struct BiomeSite
    {
        glm::vec2 worldPosXZ{0.0f};
        glm::vec2 halfExtents{0.0f};
    };

    struct BiomeRegionInfo
    {
        const BiomeDefinition* definition{nullptr};
        BiomeSite site{};
    };

    struct WeightedBiome
    {
        const BiomeDefinition* biome{nullptr};
        float weight{0.0f};
    };

    struct TerrainBasisSample
    {
        float continentMask{0.0f};
        float mainTerrain{0.0f};
        float mountainNoise{0.0f};
        float mediumNoise{0.0f};
        float detailNoise{0.0f};
        float combinedNoise{0.0f};
        float baseElevation{0.0f};
    };

    struct BiomePerturbationSample
    {
        float blendedOffset{0.0f};
        float blendedScale{0.0f};
        float blendedMinHeight{0.0f};
        float blendedMaxHeight{0.0f};
        float blendedSlopeBias{0.0f};
        float blendedMaxGradient{0.0f};
        float oceanWeight{0.0f};
        float oceanOffset{0.0f};
        float oceanScale{0.0f};
        float oceanMinHeight{0.0f};
        float oceanMaxHeight{0.0f};
        float oceanSlopeBias{0.0f};
        float oceanMaxGradient{0.0f};
        float landWeight{0.0f};
        float landOffset{0.0f};
        float landScale{0.0f};
        float landMinHeight{0.0f};
        float landMaxHeight{0.0f};
        float landSlopeBias{0.0f};
        float landMaxGradient{0.0f};
        const BiomeDefinition* dominantBiome{nullptr};
        float dominantWeight{0.0f};
    };

    TerrainBasisSample computeTerrainBasis(int worldX, int worldZ) const;
    float computeLittleMountainsNormalized(float worldX, float worldZ) const;

    struct LittleMountainSample
    {
        float height{0.0f};
        float entryFloor{0.0f};
        float interiorMask{0.0f};
    };

    float computeBaselineSurfaceHeight(const BiomePerturbationSample& perturbations,
                                       const TerrainBasisSample& basis) const;

    LittleMountainSample computeLittleMountainsHeight(int worldX,
                                                      int worldZ,
                                                      const BiomeDefinition& definition,
                                                      float interiorMask,
                                                      bool hasBorderAnchor,
                                                      float borderAnchorHeight) const;
    BiomePerturbationSample applyBiomePerturbations(const std::array<WeightedBiome, 5>& weightedBiomes,
                                                    std::size_t weightCount,
                                                    int biomeRegionX,
                                                    int biomeRegionZ) const;
    static BiomeSite computeBiomeSite(const BiomeDefinition& definition, int regionX, int regionZ) noexcept;
    const BiomeRegionInfo& biomeRegionInfo(int regionX, int regionZ) const;
    const BiomeDefinition& biomeForRegion(int regionX, int regionZ) const;


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

    OpenSimplexNoise littleMountainsNoise_;
    OpenSimplexNoise littleMountainsWarpNoise_;
    OpenSimplexNoise littleMountainsOrientationNoise_;
    PerlinNoise noise_;
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
    mutable std::unordered_map<glm::ivec2, BiomeRegionInfo, ColumnHasher> biomeRegionCache_;
    mutable std::mutex biomeRegionCacheMutex_;

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

// OpenSimplexNoise implementations
const std::array<glm::vec2, 8> OpenSimplexNoise::kGradients = {
    glm::vec2(1.0f, 0.0f),
    glm::vec2(-1.0f, 0.0f),
    glm::vec2(0.0f, 1.0f),
    glm::vec2(0.0f, -1.0f),
    glm::vec2(0.70710678f, 0.70710678f),
    glm::vec2(-0.70710678f, 0.70710678f),
    glm::vec2(0.70710678f, -0.70710678f),
    glm::vec2(-0.70710678f, -0.70710678f)
};

OpenSimplexNoise::OpenSimplexNoise(unsigned seed)
{
    std::array<int, 256> temp;
    std::iota(temp.begin(), temp.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(temp.begin(), temp.end(), rng);

    for (int i = 0; i < 256; ++i)
    {
        const int value = temp[static_cast<std::size_t>(i)];
        permutation_[i] = permutation_[i + 256] = value;
        permutationMod8_[i] = permutationMod8_[i + 256] = value & 7;
    }
}

float OpenSimplexNoise::noise(float x, float y) const noexcept
{
    constexpr float F2 = 0.3660254037844386f;
    constexpr float G2 = 0.21132486540518713f;

    const float s = (x + y) * F2;
    const int i = static_cast<int>(std::floor(x + s));
    const int j = static_cast<int>(std::floor(y + s));
    const float t = static_cast<float>(i + j) * G2;
    const float X0 = static_cast<float>(i) - t;
    const float Y0 = static_cast<float>(j) - t;
    const float x0 = x - X0;
    const float y0 = y - Y0;

    const int i1 = x0 > y0 ? 1 : 0;
    const int j1 = x0 > y0 ? 0 : 1;

    const float x1 = x0 - static_cast<float>(i1) + G2;
    const float y1 = y0 - static_cast<float>(j1) + G2;
    const float x2 = x0 - 1.0f + 2.0f * G2;
    const float y2 = y0 - 1.0f + 2.0f * G2;

    const int ii = i & 255;
    const int jj = j & 255;

    const int gi0 = permutationMod8_[ii + permutation_[jj]];
    const int gi1 = permutationMod8_[ii + i1 + permutation_[jj + j1]];
    const int gi2 = permutationMod8_[ii + 1 + permutation_[jj + 1]];

    float n0 = 0.0f;
    float n1 = 0.0f;
    float n2 = 0.0f;

    float t0 = 0.5f - x0 * x0 - y0 * y0;
    if (t0 > 0.0f)
    {
        const float t0Sq = t0 * t0;
        const float t0Pow4 = t0Sq * t0Sq;
        n0 = t0Pow4 * glm::dot(kGradients[static_cast<std::size_t>(gi0)], glm::vec2(x0, y0));
    }

    float t1 = 0.5f - x1 * x1 - y1 * y1;
    if (t1 > 0.0f)
    {
        const float t1Sq = t1 * t1;
        const float t1Pow4 = t1Sq * t1Sq;
        n1 = t1Pow4 * glm::dot(kGradients[static_cast<std::size_t>(gi1)], glm::vec2(x1, y1));
    }

    float t2 = 0.5f - x2 * x2 - y2 * y2;
    if (t2 > 0.0f)
    {
        const float t2Sq = t2 * t2;
        const float t2Pow4 = t2Sq * t2Sq;
        n2 = t2Pow4 * glm::dot(kGradients[static_cast<std::size_t>(gi2)], glm::vec2(x2, y2));
    }

    return 70.0f * (n0 + n1 + n2);
}

float OpenSimplexNoise::fbm(float x, float y, int octaves, float persistence, float lacunarity) const noexcept
{
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float sum = 0.0f;
    float maxValue = 0.0f;

    for (int i = 0; i < octaves; ++i)
    {
        sum += noise(x * frequency, y * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    if (maxValue > 0.0f)
    {
        sum /= maxValue;
    }

    return sum;
}

float OpenSimplexNoise::ridge(float x, float y, int octaves, float lacunarity, float gain) const noexcept
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    float prev = 1.0f;

    for (int i = 0; i < octaves; ++i)
    {
        float n = noise(x * frequency, y * frequency);
        n = 1.0f - std::abs(n);
        n *= n;
        sum += n * amplitude * prev;
        prev = n;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return sum;
}

glm::vec2 OpenSimplexNoise::sampleGradient(float x, float y) const noexcept
{
    constexpr float F2 = 0.3660254037844386f;
    constexpr float G2 = 0.21132486540518713f;

    const float s = (x + y) * F2;
    const int i = static_cast<int>(std::floor(x + s));
    const int j = static_cast<int>(std::floor(y + s));
    const float t = static_cast<float>(i + j) * G2;
    const float X0 = static_cast<float>(i) - t;
    const float Y0 = static_cast<float>(j) - t;
    const float x0 = x - X0;
    const float y0 = y - Y0;

    const int i1 = x0 > y0 ? 1 : 0;
    const int j1 = x0 > y0 ? 0 : 1;

    const float x1 = x0 - static_cast<float>(i1) + G2;
    const float y1 = y0 - static_cast<float>(j1) + G2;
    const float x2 = x0 - 1.0f + 2.0f * G2;
    const float y2 = y0 - 1.0f + 2.0f * G2;

    const int ii = i & 255;
    const int jj = j & 255;

    const int gi0 = permutationMod8_[ii + permutation_[jj]];
    const int gi1 = permutationMod8_[ii + i1 + permutation_[jj + j1]];
    const int gi2 = permutationMod8_[ii + 1 + permutation_[jj + 1]];

    glm::vec2 gradient{0.0f, 0.0f};

    float t0 = 0.5f - x0 * x0 - y0 * y0;
    if (t0 > 0.0f)
    {
        const float t0Sq = t0 * t0;
        const float t0Pow3 = t0Sq * t0;
        const float t0Pow4 = t0Sq * t0Sq;
        const glm::vec2 grad = kGradients[static_cast<std::size_t>(gi0)];
        const float dot = grad.x * x0 + grad.y * y0;
        const float influence = -8.0f * t0Pow3 * dot;
        gradient.x += influence * x0 + t0Pow4 * grad.x;
        gradient.y += influence * y0 + t0Pow4 * grad.y;
    }

    float t1 = 0.5f - x1 * x1 - y1 * y1;
    if (t1 > 0.0f)
    {
        const float t1Sq = t1 * t1;
        const float t1Pow3 = t1Sq * t1;
        const float t1Pow4 = t1Sq * t1Sq;
        const glm::vec2 grad = kGradients[static_cast<std::size_t>(gi1)];
        const float dot = grad.x * x1 + grad.y * y1;
        const float influence = -8.0f * t1Pow3 * dot;
        gradient.x += influence * x1 + t1Pow4 * grad.x;
        gradient.y += influence * y1 + t1Pow4 * grad.y;
    }

    float t2 = 0.5f - x2 * x2 - y2 * y2;
    if (t2 > 0.0f)
    {
        const float t2Sq = t2 * t2;
        const float t2Pow3 = t2Sq * t2;
        const float t2Pow4 = t2Sq * t2Sq;
        const glm::vec2 grad = kGradients[static_cast<std::size_t>(gi2)];
        const float dot = grad.x * x2 + grad.y * y2;
        const float influence = -8.0f * t2Pow3 * dot;
        gradient.x += influence * x2 + t2Pow4 * grad.x;
        gradient.y += influence * y2 + t2Pow4 * grad.y;
    }

    return gradient * 70.0f;
}

// PerlinNoise implementations

PerlinNoise::PerlinNoise(unsigned seed)
{
    std::array<int, 256> temp;
    std::iota(temp.begin(), temp.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(temp.begin(), temp.end(), rng);

    for (int i = 0; i < 256; ++i)
    {
        permutation_[i] = permutation_[i + 256] = temp[static_cast<std::size_t>(i)];
    }
}

float PerlinNoise::noise(float x, float y) const noexcept
{
    const int xi = static_cast<int>(std::floor(x)) & 255;
    const int yi = static_cast<int>(std::floor(y)) & 255;

    const float xf = x - std::floor(x);
    const float yf = y - std::floor(y);

    const float u = fade(xf);
    const float v = fade(yf);

    const int aa = permutation_[permutation_[xi] + yi];
    const int ab = permutation_[permutation_[xi] + yi + 1];
    const int ba = permutation_[permutation_[xi + 1] + yi];
    const int bb = permutation_[permutation_[xi + 1] + yi + 1];

    const float x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1.0f, yf), u);
    const float x2 = lerp(grad(ab, xf, yf - 1.0f), grad(bb, xf - 1.0f, yf - 1.0f), u);
    return lerp(x1, x2, v);
}

float PerlinNoise::fbm(float x, float y, int octaves, float persistence, float lacunarity) const noexcept
{
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float sum = 0.0f;
    float maxValue = 0.0f;

    for (int i = 0; i < octaves; ++i)
    {
        sum += noise(x * frequency, y * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    if (maxValue > 0.0f)
    {
        sum /= maxValue;
    }

    return sum;
}

float PerlinNoise::ridge(float x, float y, int octaves, float lacunarity, float gain) const noexcept
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    float prev = 1.0f;

    for (int i = 0; i < octaves; ++i)
    {
        float n = noise(x * frequency, y * frequency);
        n = 1.0f - std::abs(n);
        n *= n;
        sum += n * amplitude * prev;
        prev = n;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return sum;
}

float PerlinNoise::fade(float t) noexcept
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float PerlinNoise::lerp(float a, float b, float t) noexcept
{
    return a + t * (b - a);
}

float PerlinNoise::grad(int hash, float x, float y) noexcept
{
    const int h = hash & 7;
    const float u = h < 4 ? x : y;
    const float v = h < 4 ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

// ChunkManager::Impl methods (to be filled)

ChunkManager::Impl::Impl(unsigned seed)
    : littleMountainsNoise_(seed ^ 0x9E3779B9u),
      littleMountainsWarpNoise_(seed ^ 0x7F4A7C15u),
      littleMountainsOrientationNoise_(seed ^ 0xDD62BBA1u),
      noise_(seed),
      shouldStop_(false),
      viewDistance_(kDefaultViewDistance),
      targetViewDistance_(kDefaultViewDistance)
{
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

void ChunkManager::Impl::render(GLuint shaderProgram,
                                const glm::mat4& viewProj,
                                const glm::vec3& cameraPos,
                                const Frustum& frustum,
                                const ChunkShaderUniformLocations& uniforms) const
{
    glUseProgram(shaderProgram);
    if (uniforms.uViewProj >= 0)
    {
        glUniformMatrix4fv(uniforms.uViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
    }
    if (uniforms.uLightDir >= 0)
    {
        glUniform3fv(uniforms.uLightDir, 1, glm::value_ptr(lightDirection_));
    }
    if (uniforms.uCameraPos >= 0)
    {
        glUniform3fv(uniforms.uCameraPos, 1, glm::value_ptr(cameraPos));
    }

    if (atlasTexture_ != 0)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, atlasTexture_);
        if (uniforms.uAtlas >= 0)
        {
            glUniform1i(uniforms.uAtlas, 0);
        }
    }

    if (uniforms.uHighlightedBlock >= 0)
    {
        glUniform3f(uniforms.uHighlightedBlock,
                    static_cast<float>(highlightedBlock_.x),
                    static_cast<float>(highlightedBlock_.y),
                    static_cast<float>(highlightedBlock_.z));
    }
    if (uniforms.uHasHighlight >= 0)
    {
        glUniform1i(uniforms.uHasHighlight, hasHighlight_ ? 1 : 0);
    }

    std::vector<std::pair<glm::ivec3, std::shared_ptr<Chunk>>> snapshot;
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        snapshot.reserve(chunks_.size());
        for (const auto& entry : chunks_)
        {
            snapshot.push_back(entry);
        }
    }

    struct DrawBatch
    {
        std::vector<GLsizei> counts;
        std::vector<const void*> offsets;
        std::vector<GLint> baseVertices;
    };

    std::vector<DrawBatch> drawBatches;
    std::vector<GLuint> pageVaos;
    {
        std::lock_guard<std::mutex> pageLock(bufferPageMutex_);
        const std::size_t pageCount = bufferPages_.size();
        drawBatches.resize(pageCount);
        pageVaos.resize(pageCount, 0);
        for (std::size_t i = 0; i < pageCount; ++i)
        {
            pageVaos[i] = bufferPages_[i].vao;
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
        if (pageIndex == kInvalidChunkBufferPage || pageIndex >= drawBatches.size())
        {
            continue;
        }

        if (chunkPtr->vertexOffset > static_cast<std::size_t>(std::numeric_limits<GLint>::max()))
        {
            continue;
        }

        DrawBatch& batch = drawBatches[pageIndex];
        batch.counts.push_back(chunkPtr->indexCount);
        batch.offsets.push_back(reinterpret_cast<const void*>(chunkPtr->indexOffset * sizeof(std::uint32_t)));
        batch.baseVertices.push_back(static_cast<GLint>(chunkPtr->vertexOffset));
    }

    for (std::size_t pageIndex = 0; pageIndex < drawBatches.size(); ++pageIndex)
    {
        const DrawBatch& batch = drawBatches[pageIndex];
        if (batch.counts.empty())
        {
            continue;
        }

        glBindVertexArray(pageVaos[pageIndex]);
        glMultiDrawElementsBaseVertex(GL_TRIANGLES,
                                      batch.counts.data(),
                                      GL_UNSIGNED_INT,
                                      batch.offsets.data(),
                                      static_cast<GLsizei>(batch.counts.size()),
                                      batch.baseVertices.data());
    }

    glBindVertexArray(0);
    if (atlasTexture_ != 0)
    {
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glUseProgram(0);
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

ChunkManager::Impl::BiomeSite ChunkManager::Impl::computeBiomeSite(const BiomeDefinition& definition,
                                                                   int regionX,
                                                                   int regionZ) noexcept
{
    constexpr float kMarginRatio = 0.2f;
    const float footprintMultiplier = std::max(definition.footprintMultiplier, 0.001f);
    const float baseRegionWidth = static_cast<float>(kChunkSizeX * kBiomeSizeInChunks);
    const float baseRegionDepth = static_cast<float>(kChunkSizeZ * kBiomeSizeInChunks);
    const float scaledBiomeSizeInChunks = static_cast<float>(kBiomeSizeInChunks) * footprintMultiplier;
    const float regionWidth = static_cast<float>(kChunkSizeX) * scaledBiomeSizeInChunks;
    const float regionDepth = static_cast<float>(kChunkSizeZ) * scaledBiomeSizeInChunks;
    const float marginX = regionWidth * kMarginRatio;
    const float marginZ = regionDepth * kMarginRatio;
    const float jitterX = hashToUnitFloat(regionX, 137, regionZ);
    const float jitterZ = hashToUnitFloat(regionX, 613, regionZ);
    const float availableWidth = std::max(regionWidth - marginX * 2.0f, 0.0f);
    const float availableDepth = std::max(regionDepth - marginZ * 2.0f, 0.0f);
    const float baseX = static_cast<float>(regionX) * baseRegionWidth;
    const float baseZ = static_cast<float>(regionZ) * baseRegionDepth;

    BiomeSite site{};
    site.worldPosXZ.x = baseX + marginX + availableWidth * jitterX;
    site.worldPosXZ.y = baseZ + marginZ + availableDepth * jitterZ;
    site.halfExtents = glm::vec2(regionWidth * 0.5f, regionDepth * 0.5f);
    return site;
}

const ChunkManager::Impl::BiomeRegionInfo& ChunkManager::Impl::biomeRegionInfo(int regionX, int regionZ) const
{
    const glm::ivec2 key{regionX, regionZ};
    std::lock_guard<std::mutex> lock(biomeRegionCacheMutex_);
    auto it = biomeRegionCache_.find(key);
    if (it != biomeRegionCache_.end())
    {
        return it->second;
    }

    BiomeRegionInfo info{};
    const float selector = hashToUnitFloat(regionX, 31, regionZ);
    const std::size_t maxIndex = kBiomeDefinitions.size() - 1;
    const std::size_t biomeIndex =
        std::min(static_cast<std::size_t>(selector * static_cast<float>(kBiomeDefinitions.size())), maxIndex);
    info.definition = &kBiomeDefinitions[biomeIndex];
    info.site = computeBiomeSite(*info.definition, regionX, regionZ);

    auto [insertedIt, inserted] = biomeRegionCache_.emplace(key, info);
    return insertedIt->second;
}

const BiomeDefinition& ChunkManager::Impl::biomeForRegion(int regionX, int regionZ) const
{
    return *biomeRegionInfo(regionX, regionZ).definition;
}

ChunkManager::Impl::TerrainBasisSample ChunkManager::Impl::computeTerrainBasis(int worldX, int worldZ) const
{
    TerrainBasisSample basis{};

    const float nx = static_cast<float>(worldX) * 0.01f;
    const float nz = static_cast<float>(worldZ) * 0.01f;

    basis.mainTerrain = noise_.fbm(nx, nz, 6, 0.5f, 2.0f);
    basis.mountainNoise = noise_.ridge(nx * 0.4f, nz * 0.4f, 5, 2.1f, 0.5f);
    basis.detailNoise = noise_.fbm(nx * 4.0f, nz * 4.0f, 8, 0.45f, 2.2f);
    basis.mediumNoise = noise_.fbm(nx * 0.8f, nz * 0.8f, 7, 0.5f, 2.0f);

    basis.combinedNoise = basis.mainTerrain * 12.0f +
                          basis.mountainNoise * 8.0f +
                          basis.mediumNoise * 4.0f +
                          basis.detailNoise * 2.0f;

    basis.baseElevation = basis.combinedNoise;
    basis.continentMask = 1.0f;

    return basis;
}

float ChunkManager::Impl::computeLittleMountainsNormalized(float worldX, float worldZ) const
{
    const glm::vec2 worldPos{worldX, worldZ};

    const glm::vec2 kilometerField = worldPos * 0.001f;
    glm::vec2 orientationWarp{
        littleMountainsWarpNoise_.fbm(worldPos.x * 0.00035f + 103.0f,
                                      worldPos.y * 0.00035f - 77.0f,
                                      3,
                                      0.55f,
                                      2.15f),
        littleMountainsWarpNoise_.fbm(worldPos.x * 0.00035f - 59.0f,
                                      worldPos.y * 0.00035f + 43.0f,
                                      3,
                                      0.55f,
                                      2.15f)};
    orientationWarp *= 0.35f;
    const glm::vec2 orientationSample = kilometerField + orientationWarp;

    glm::vec2 orientationGradient =
        littleMountainsOrientationNoise_.sampleGradient(orientationSample.x, orientationSample.y);
    if (!std::isfinite(orientationGradient.x) || !std::isfinite(orientationGradient.y)
        || glm::dot(orientationGradient, orientationGradient) < 1e-6f)
    {
        orientationGradient = glm::vec2(1.0f, 0.0f);
    }
    glm::vec2 ridgeDirection = glm::normalize(orientationGradient);
    const glm::vec2 ridgePerpendicular{-ridgeDirection.y, ridgeDirection.x};

    glm::vec2 warpPrimary{
        littleMountainsWarpNoise_.fbm(worldPos.x * 0.0011f + 19.0f,
                                      worldPos.y * 0.0011f + 87.0f,
                                      4,
                                      0.6f,
                                      2.15f),
        littleMountainsWarpNoise_.fbm(worldPos.x * 0.0011f - 71.0f,
                                      worldPos.y * 0.0011f - 29.0f,
                                      4,
                                      0.6f,
                                      2.15f)};
    glm::vec2 warpDetail{
        littleMountainsWarpNoise_.fbm(worldPos.x * 0.0045f - 11.0f,
                                      worldPos.y * 0.0045f + 53.0f,
                                      3,
                                      0.5f,
                                      2.3f),
        littleMountainsWarpNoise_.fbm(worldPos.x * 0.0045f + 67.0f,
                                      worldPos.y * 0.0045f - 41.0f,
                                      3,
                                      0.5f,
                                      2.3f)};
    glm::vec2 warped = worldPos + warpPrimary * 180.0f + warpDetail * 28.0f;

    const float alongRidge = glm::dot(warped, ridgeDirection);
    const float acrossRidge = glm::dot(warped, ridgePerpendicular);

    const float ridgePrimary = littleMountainsNoise_.ridge(alongRidge * 0.016f,
                                                           acrossRidge * 0.016f,
                                                           5,
                                                           2.05f,
                                                           0.55f);
    const float ridgeSecondary = littleMountainsNoise_.ridge(alongRidge * 0.028f + 57.0f,
                                                             acrossRidge * 0.028f - 113.0f,
                                                             4,
                                                             2.2f,
                                                             0.6f);
    const float ridgeMicro = littleMountainsNoise_.ridge(alongRidge * 0.043f - 211.0f,
                                                         acrossRidge * 0.021f + 167.0f,
                                                         3,
                                                         2.1f,
                                                         0.5f);

    float ridgeStack = ridgePrimary * 0.6f + ridgeSecondary * 0.3f + ridgeMicro * 0.25f;
    ridgeStack = std::clamp(ridgeStack, 0.0f, 1.3f);
    ridgeStack = std::clamp(ridgeStack, 0.0f, 1.0f);

    const float valleyFill = littleMountainsNoise_.fbm(warped.x * 0.0032f - 401.0f,
                                                       warped.y * 0.0032f + 245.0f,
                                                       4,
                                                       0.5f,
                                                       2.1f)
                             * 0.5f
                             + 0.5f;

    const float macroRamps = littleMountainsNoise_.fbm(worldPos.x * 0.00042f + 11.0f,
                                                       worldPos.y * 0.00042f - 37.0f,
                                                       5,
                                                       0.55f,
                                                       2.05f)
                             * 0.5f
                             + 0.5f;

    const float uplift = littleMountainsNoise_.fbm(worldPos.x * 0.00078f - 91.0f,
                                                   worldPos.y * 0.00078f + 133.0f,
                                                   4,
                                                   0.6f,
                                                   2.15f)
                         * 0.5f
                         + 0.5f;

    const float ridgeMask = glm::smoothstep(0.3f, 0.72f, macroRamps);
    const float valleyMask = 1.0f - glm::smoothstep(0.1f, 0.4f, macroRamps);

    float ridged = glm::mix(valleyFill, ridgeStack, ridgeMask);
    ridged = std::clamp(ridged, 0.0f, 1.0f);

    const float terraces = littleMountainsNoise_.fbm(warped.x * 0.008f + 211.0f,
                                                     warped.y * 0.008f - 157.0f,
                                                     3,
                                                     0.5f,
                                                     2.3f)
                        * 0.5f
                        + 0.5f;

    float combined = macroRamps * 0.35f + ridged * 0.45f + uplift * 0.15f + terraces * 0.05f;
    combined = std::clamp(combined, 0.0f, 1.0f);

    const float peakBlendControl = glm::smoothstep(0.55f, 0.9f, combined);
    const float peakShaped = 1.0f - std::pow(1.0f - combined, 3.0f);
    float finalValue = glm::mix(combined, peakShaped, peakBlendControl);
    finalValue = std::clamp(finalValue, 0.0f, 1.0f);

    const float valleyBlend = glm::smoothstep(0.2f, 0.6f, valleyFill);
    const float valleyBase = macroRamps * 0.5f + valleyFill * 0.5f;
    finalValue = glm::mix(valleyBase, finalValue, valleyBlend);

    finalValue = std::clamp(finalValue + (uplift - 0.5f) * 0.08f, 0.0f, 1.0f);
    finalValue = glm::mix(finalValue, macroRamps, valleyMask * 0.25f);

    return std::clamp(finalValue, 0.0f, 1.0f);
}

float ChunkManager::Impl::computeBaselineSurfaceHeight(const BiomePerturbationSample& perturbations,
                                                       const TerrainBasisSample& basis) const
{
    constexpr float kMinIntAsFloat = static_cast<float>(std::numeric_limits<int>::min());
    constexpr float kMaxIntAsFloat = static_cast<float>(std::numeric_limits<int>::max());

    float minHeight = perturbations.blendedMinHeight;
    float maxHeight = perturbations.blendedMaxHeight;
    if (minHeight > maxHeight)
    {
        std::swap(minHeight, maxHeight);
    }

    minHeight = std::clamp(minHeight, kMinIntAsFloat, kMaxIntAsFloat);
    maxHeight = std::clamp(maxHeight, kMinIntAsFloat, kMaxIntAsFloat);

    const float slopeBias = std::clamp(perturbations.blendedSlopeBias, 0.0f, 1.0f);
    const float lowAmplitudeCombined = basis.mainTerrain * 3.0f + basis.mountainNoise * 1.5f +
                                       basis.mediumNoise * 1.0f + basis.detailNoise * 0.5f;
    const float blendedTerrain = std::lerp(basis.combinedNoise, lowAmplitudeCombined, slopeBias);
    const float unclampedMacroHeight = perturbations.blendedOffset + blendedTerrain * perturbations.blendedScale;

    float macroStageHeight = std::clamp(unclampedMacroHeight, minHeight, maxHeight);
    float targetHeight = macroStageHeight;

    const bool hasOceanContribution = perturbations.oceanWeight > 0.0f;
    const bool hasLandContribution = perturbations.landWeight > 0.0f;

    const float oceanSlopeBias =
        std::clamp(hasOceanContribution ? perturbations.oceanSlopeBias : slopeBias, 0.0f, 1.0f);
    const float landSlopeBias =
        std::clamp(hasLandContribution ? perturbations.landSlopeBias : slopeBias, 0.0f, 1.0f);
    const float oceanGradientWindow =
        std::max(hasOceanContribution ? perturbations.oceanMaxGradient : perturbations.blendedMaxGradient, 0.0f);
    const float landGradientWindow =
        std::max(hasLandContribution ? perturbations.landMaxGradient : perturbations.blendedMaxGradient, 0.0f);

    float oceanTarget = targetHeight;
    if (hasOceanContribution)
    {
        const float oceanVariation = std::lerp(basis.combinedNoise, lowAmplitudeCombined, oceanSlopeBias);
        float rawOceanTarget = perturbations.oceanOffset + oceanVariation * perturbations.oceanScale;
        if (oceanGradientWindow > 0.0f)
        {
            const float oceanGradientMin = perturbations.oceanOffset - oceanGradientWindow;
            const float oceanGradientMax = perturbations.oceanOffset + oceanGradientWindow;
            rawOceanTarget = std::clamp(rawOceanTarget, oceanGradientMin, oceanGradientMax);
        }

        oceanTarget = std::clamp(rawOceanTarget, perturbations.oceanMinHeight, perturbations.oceanMaxHeight);
        minHeight = std::min(minHeight, oceanTarget);
        maxHeight = std::max(maxHeight, oceanTarget);
    }

    float landTarget = targetHeight;
    if (hasLandContribution)
    {
        float landMin = perturbations.landMinHeight;
        float landMax = perturbations.landMaxHeight;
        if (landMin > landMax)
        {
            std::swap(landMin, landMax);
        }

        landMin = std::clamp(landMin, kMinIntAsFloat, kMaxIntAsFloat);
        landMax = std::clamp(landMax, kMinIntAsFloat, kMaxIntAsFloat);

        const float lowFrequencyNoise = basis.mainTerrain * 0.3f + basis.mediumNoise * 0.4f + basis.detailNoise * 0.3f;
        const float slopeNoise = basis.mountainNoise * 0.15f;
        const float landBaseHeight = std::lerp(macroStageHeight, perturbations.landOffset, landSlopeBias);
        float rawLandTarget = landBaseHeight + (lowFrequencyNoise + slopeNoise) * perturbations.landScale;

        if (landGradientWindow > 0.0f)
        {
            const float landGradientMin = landBaseHeight - landGradientWindow;
            const float landGradientMax = landBaseHeight + landGradientWindow;
            rawLandTarget = std::clamp(rawLandTarget, landGradientMin, landGradientMax);
        }

        landTarget = std::clamp(rawLandTarget, landMin, landMax);
        minHeight = std::min(minHeight, landTarget);
        maxHeight = std::max(maxHeight, landTarget);
    }

    const float globalSeaLevelF = static_cast<float>(kGlobalSeaLevel);
    float shorelineDistance = std::abs(macroStageHeight - globalSeaLevelF);

    if (hasLandContribution && landTarget > globalSeaLevelF)
    {
        constexpr float kShorelineEaseRange = 8.0f;
        if (shorelineDistance < kShorelineEaseRange)
        {
            const float normalized = 1.0f - std::clamp(shorelineDistance / kShorelineEaseRange, 0.0f, 1.0f);
            const float easing = normalized * normalized * normalized;
            const float easedLandHeight = globalSeaLevelF + (landTarget - globalSeaLevelF) * (1.0f - easing);
            const float loweredLandHeight = std::min(landTarget, easedLandHeight);
            landTarget = loweredLandHeight;
            minHeight = std::min(minHeight, landTarget);
            maxHeight = std::max(maxHeight, landTarget);
        }
    }

    float shorelineBlend = 0.0f;
    float oceanShare = 0.0f;
    float landShare = 0.0f;

    if (hasOceanContribution && hasLandContribution)
    {
        const float totalCategoryWeight = perturbations.oceanWeight + perturbations.landWeight;
        if (totalCategoryWeight > std::numeric_limits<float>::epsilon())
        {
            oceanShare = perturbations.oceanWeight / totalCategoryWeight;
            landShare = perturbations.landWeight / totalCategoryWeight;

            auto shorelineRamp = [](float share)
            {
                const float t = std::clamp((share - 0.3f) / 0.2f, 0.0f, 1.0f);
                return t * t * (3.0f - 2.0f * t);
            };

            shorelineBlend = shorelineRamp(oceanShare) * shorelineRamp(landShare);
            if (shorelineBlend > 0.0f)
            {
                const float easedBlend = shorelineBlend * shorelineBlend * (3.0f - 2.0f * shorelineBlend);
                const float shorelineLandHeight = oceanTarget + (landTarget - oceanTarget) * (1.0f - easedBlend);
                const float clampedShoreline = std::max(shorelineLandHeight, oceanTarget);

                landTarget = clampedShoreline;
                minHeight = std::min(minHeight, clampedShoreline);
                maxHeight = std::max(maxHeight, clampedShoreline);
            }
        }
    }

    if (perturbations.dominantBiome && perturbations.dominantBiome->id == BiomeId::Ocean && hasOceanContribution)
    {
        targetHeight = oceanTarget;
    }
    else if (hasLandContribution)
    {
        targetHeight = landTarget;
    }
    else if (hasOceanContribution)
    {
        targetHeight = oceanTarget;
    }

    return targetHeight;
}

ChunkManager::Impl::LittleMountainSample ChunkManager::Impl::computeLittleMountainsHeight(
    int worldX,
    int worldZ,
    const BiomeDefinition& definition,
    float interiorMask,
    bool hasBorderAnchor,
    float borderAnchorHeight) const
{
    const float minHeight = static_cast<float>(definition.minHeight);
    const float maxHeight = static_cast<float>(definition.maxHeight);
    const float range = std::max(maxHeight - minHeight, 1.0f);
    const float floorRange = std::clamp(range * 0.12f, 24.0f, 110.0f);

    auto sampleColumn = [&](float sampleX, float sampleZ) -> LittleMountainSample {
        const float normalized = computeLittleMountainsNormalized(sampleX, sampleZ);
        float height = minHeight + normalized * range;
        const float floorNoise = littleMountainsNoise_.fbm(sampleX * 0.0013f + 311.0f,
                                                           sampleZ * 0.0013f - 173.0f,
                                                           4,
                                                           0.5f,
                                                           2.0f);
        const float floorT = std::clamp(floorNoise * 0.5f + 0.5f, 0.0f, 1.0f);
        const float entryFloor = minHeight + floorT * floorRange;
        if (height < entryFloor)
        {
            height = entryFloor;
        }
        return LittleMountainSample{height, entryFloor, 1.0f};
    };

    const auto baseSample = sampleColumn(static_cast<float>(worldX), static_cast<float>(worldZ));
    float baseHeight = baseSample.height;
    const float entryFloor = baseSample.entryFloor;

    const float sampleStep = 12.0f;
    const float highSlopeStart = minHeight + range * 0.65f;
    const float highSlopeEnd = minHeight + range * 0.90f;
    const float altitudeT = std::clamp((baseHeight - highSlopeStart) / (highSlopeEnd - highSlopeStart), 0.0f, 1.0f);
    const float normalizedAltitude = std::clamp((baseHeight - minHeight) / range, 0.0f, 1.0f);
    const float lowTalusDeg = 1.5f;
    const float midTalusDeg = 4.0f;
    const float highTalusDeg = 7.0f;
    const float foothillBlend = glm::smoothstep(0.25f, 0.65f, normalizedAltitude);
    float talusDeg = std::lerp(lowTalusDeg, midTalusDeg, foothillBlend);
    talusDeg = std::lerp(talusDeg, highTalusDeg, glm::smoothstep(0.0f, 1.0f, altitudeT));
    const float talusAngle = glm::radians(talusDeg);
    const float rawMaxDiff = std::tan(talusAngle) * sampleStep;
    const float maxTalusDiff = std::tan(glm::radians(highTalusDeg)) * sampleStep;
    const float maxDiff = std::clamp(rawMaxDiff, 0.2f, maxTalusDiff);

    auto sampleNeighbor = [&](float offsetX, float offsetZ) {
        const auto neighborSample =
            sampleColumn(static_cast<float>(worldX) + offsetX, static_cast<float>(worldZ) + offsetZ);
        return neighborSample.height;
    };

    std::array<float, 4> neighbors{
        sampleNeighbor(sampleStep, 0.0f),
        sampleNeighbor(-sampleStep, 0.0f),
        sampleNeighbor(0.0f, sampleStep),
        sampleNeighbor(0.0f, -sampleStep),
    };

    std::array<float, 4> diagonalNeighbors{
        sampleNeighbor(sampleStep, sampleStep),
        sampleNeighbor(sampleStep, -sampleStep),
        sampleNeighbor(-sampleStep, sampleStep),
        sampleNeighbor(-sampleStep, -sampleStep),
    };

    const float neighborAverage =
        (neighbors[0] + neighbors[1] + neighbors[2] + neighbors[3]) * 0.25f;
    const float diagonalAverage =
        (diagonalNeighbors[0] + diagonalNeighbors[1] + diagonalNeighbors[2] + diagonalNeighbors[3]) * 0.25f;
    const float convexity = std::max(baseHeight - neighborAverage, 0.0f);
    const float diagonalConvexity = std::max(baseHeight - diagonalAverage, 0.0f);
    const float curvatureMagnitude = std::max(convexity, diagonalConvexity);
    const float lowSlopeMask = 1.0f - glm::smoothstep(0.25f, 0.6f, normalizedAltitude);
    const float curvatureSuppression = lowSlopeMask * glm::smoothstep(1.5f, 10.0f, curvatureMagnitude) * 0.55f;
    const float curvatureFactor = std::clamp(1.0f - curvatureSuppression, 0.45f, 1.0f);
    const float adjustedMaxDiff = maxDiff * curvatureFactor;

    float relaxedHeight = baseHeight;
    auto relaxWithNeighbors = [&](const std::array<float, 4>& neighborHeights, float allowedDiff) {
        for (float neighborHeight : neighborHeights)
        {
            const float diff = relaxedHeight - neighborHeight;
            if (diff > allowedDiff)
            {
                relaxedHeight -= (diff - allowedDiff) * 0.5f;
            }
            else if (diff < -allowedDiff)
            {
                relaxedHeight += (-allowedDiff - diff) * 0.5f;
            }
        }
    };

    relaxWithNeighbors(neighbors, adjustedMaxDiff);

    const float diagonalStep = sampleStep * std::sqrt(2.0f);
    const float diagonalStepFactor = diagonalStep / sampleStep;
    const float diagonalRawDiff = adjustedMaxDiff * diagonalStepFactor;
    const float maxDiagonalDiff = maxDiff * diagonalStepFactor;
    const float diagonalDiff = std::clamp(diagonalRawDiff, 0.25f, maxDiagonalDiff);
    relaxWithNeighbors(diagonalNeighbors, diagonalDiff);

    relaxedHeight = std::clamp(relaxedHeight, entryFloor, maxHeight);

    baseHeight = std::lerp(baseHeight, relaxedHeight, 0.9f);

    const float clampedHeight = std::clamp(baseHeight, entryFloor, maxHeight);

    const float maskedInterior = std::clamp(interiorMask, 0.0f, 1.0f);
    const float interiorBlend = std::pow(maskedInterior, 1.75f);

    const float relaxedEntryFloor = std::clamp(entryFloor, minHeight, clampedHeight);
    float borderBaseline = relaxedEntryFloor;
    if (hasBorderAnchor)
    {
        const float anchorMin = relaxedEntryFloor - floorRange;
        const float anchorMax = relaxedEntryFloor + floorRange;
        borderBaseline = std::clamp(borderAnchorHeight, anchorMin, anchorMax);
        borderBaseline = std::clamp(borderBaseline, minHeight, maxHeight);
    }
    else
    {
        const float lowInteriorBlend = 0.35f;
        if (interiorBlend < lowInteriorBlend)
        {
            const float ringAverage = (neighborAverage + diagonalAverage) * 0.5f;
            borderBaseline = std::clamp(ringAverage, entryFloor, maxHeight);
        }
    }

    const float baselineEntryFloor = glm::mix(borderBaseline, relaxedEntryFloor, interiorBlend);
    const float baselineMinHeight = glm::mix(borderBaseline, minHeight, interiorBlend);

    const float interiorFoothillLift = interiorBlend * floorRange * 0.65f;
    const float raisedEntryFloor = std::min(baselineEntryFloor + interiorFoothillLift, clampedHeight);
    const float maskedEntryFloor = glm::mix(baselineMinHeight, raisedEntryFloor, interiorBlend);
    float maskedHeight = glm::mix(baselineMinHeight, clampedHeight, interiorBlend);
    maskedHeight = std::max(maskedHeight, maskedEntryFloor);

    return LittleMountainSample{maskedHeight, maskedEntryFloor, maskedInterior};
}

ChunkManager::Impl::BiomePerturbationSample ChunkManager::Impl::applyBiomePerturbations(
    const std::array<WeightedBiome, 5>& weightedBiomes,
    std::size_t weightCount,
    int biomeRegionX,
    int biomeRegionZ) const
{
    BiomePerturbationSample result{};
    result.dominantWeight = -1.0f;
    float totalBlendWeight = 0.0f;

    for (std::size_t i = 0; i < weightCount; ++i)
    {
        const WeightedBiome& weightedBiome = weightedBiomes[i];
        if (weightedBiome.biome == nullptr || weightedBiome.weight <= 0.0f)
        {
            continue;
        }

        result.blendedOffset += weightedBiome.biome->heightOffset * weightedBiome.weight;
        result.blendedScale += weightedBiome.biome->heightScale * weightedBiome.weight;
        result.blendedMinHeight += static_cast<float>(weightedBiome.biome->minHeight) * weightedBiome.weight;
        result.blendedMaxHeight += static_cast<float>(weightedBiome.biome->maxHeight) * weightedBiome.weight;
        result.blendedSlopeBias += weightedBiome.biome->baseSlopeBias * weightedBiome.weight;
        result.blendedMaxGradient += weightedBiome.biome->maxGradient * weightedBiome.weight;
        totalBlendWeight += weightedBiome.weight;

        if (weightedBiome.biome->id == BiomeId::Ocean)
        {
            result.oceanWeight += weightedBiome.weight;
            result.oceanOffset += weightedBiome.biome->heightOffset * weightedBiome.weight;
            result.oceanScale += weightedBiome.biome->heightScale * weightedBiome.weight;
            result.oceanMinHeight += static_cast<float>(weightedBiome.biome->minHeight) * weightedBiome.weight;
            result.oceanMaxHeight += static_cast<float>(weightedBiome.biome->maxHeight) * weightedBiome.weight;
            result.oceanSlopeBias += weightedBiome.biome->baseSlopeBias * weightedBiome.weight;
            result.oceanMaxGradient += weightedBiome.biome->maxGradient * weightedBiome.weight;
        }
        else
        {
            result.landWeight += weightedBiome.weight;
            result.landOffset += weightedBiome.biome->heightOffset * weightedBiome.weight;
            result.landScale += weightedBiome.biome->heightScale * weightedBiome.weight;
            result.landMinHeight += static_cast<float>(weightedBiome.biome->minHeight) * weightedBiome.weight;
            result.landMaxHeight += static_cast<float>(weightedBiome.biome->maxHeight) * weightedBiome.weight;
            result.landSlopeBias += weightedBiome.biome->baseSlopeBias * weightedBiome.weight;
            result.landMaxGradient += weightedBiome.biome->maxGradient * weightedBiome.weight;
        }

        if (weightedBiome.weight > result.dominantWeight)
        {
            result.dominantWeight = weightedBiome.weight;
            result.dominantBiome = weightedBiome.biome;
        }
    }

    auto normalizeCategory = [](float weight, float& offset, float& scale, float& minHeight, float& maxHeight)
    {
        if (weight <= 0.0f)
        {
            return;
        }

        const float invWeight = 1.0f / weight;
        offset *= invWeight;
        scale *= invWeight;
        minHeight *= invWeight;
        maxHeight *= invWeight;
    };

    normalizeCategory(result.oceanWeight, result.oceanOffset, result.oceanScale, result.oceanMinHeight, result.oceanMaxHeight);
    normalizeCategory(result.landWeight, result.landOffset, result.landScale, result.landMinHeight, result.landMaxHeight);

    auto normalizeSlope = [](float weight, float& slopeBias, float& maxGradient)
    {
        if (weight <= 0.0f)
        {
            slopeBias = 0.0f;
            maxGradient = 0.0f;
            return;
        }

        const float invWeight = 1.0f / weight;
        slopeBias = std::clamp(slopeBias * invWeight, 0.0f, 1.0f);
        maxGradient = std::max(maxGradient * invWeight, 0.0f);
    };

    normalizeSlope(totalBlendWeight, result.blendedSlopeBias, result.blendedMaxGradient);
    normalizeSlope(result.oceanWeight, result.oceanSlopeBias, result.oceanMaxGradient);
    normalizeSlope(result.landWeight, result.landSlopeBias, result.landMaxGradient);

    result.dominantWeight = std::max(result.dominantWeight, 0.0f);
    if (result.dominantBiome == nullptr)
    {
        result.dominantBiome = &biomeForRegion(biomeRegionX, biomeRegionZ);
    }

    return result;
}

ColumnSample ChunkManager::Impl::sampleColumn(int worldX, int worldZ, int slabMinWorldY, int slabMaxWorldY) const
{
    if (slabMinWorldY > slabMaxWorldY)
    {
        std::swap(slabMinWorldY, slabMaxWorldY);
    }

    const int chunkX = floorDiv(worldX, kChunkSizeX);
    const int chunkZ = floorDiv(worldZ, kChunkSizeZ);
    const int biomeRegionX = floorDiv(chunkX, kBiomeSizeInChunks);
    const int biomeRegionZ = floorDiv(chunkZ, kBiomeSizeInChunks);

    const glm::vec2 columnPosition{static_cast<float>(worldX) + 0.5f, static_cast<float>(worldZ) + 0.5f};

    struct CandidateSite
    {
        const BiomeDefinition* biome{nullptr};
        glm::vec2 positionXZ{0.0f};
        float distanceSquared{std::numeric_limits<float>::max()};
        float normalizedDistance{1.0f};
    };

    constexpr int regionRadius = kBiomeRegionSearchRadius;
    std::array<CandidateSite, kBiomeRegionCandidateCapacity> candidateSites{};
    std::size_t candidateCount = 0;
    auto littleMountainInfluence = [](float normalizedDistance) {
        const float clamped = std::clamp(normalizedDistance, 0.0f, 1.0f);
        const float tapered = 1.0f - glm::smoothstep(0.35f, 0.85f, clamped);
        return std::pow(std::clamp(tapered, 0.0f, 1.0f), 1.75f);
    };
    for (int regionOffsetZ = -regionRadius; regionOffsetZ <= regionRadius; ++regionOffsetZ)
    {
        for (int regionOffsetX = -regionRadius; regionOffsetX <= regionRadius; ++regionOffsetX)
        {
            const BiomeRegionInfo& info =
                biomeRegionInfo(biomeRegionX + regionOffsetX, biomeRegionZ + regionOffsetZ);

            CandidateSite site{};
            site.biome = info.definition;
            site.positionXZ = info.site.worldPosXZ;
            const glm::vec2 delta = columnPosition - site.positionXZ;
            const glm::vec2 halfExtents = glm::max(info.site.halfExtents, glm::vec2(1.0f));
            const glm::vec2 normalizedDelta = delta / halfExtents;
            site.distanceSquared = glm::dot(delta, delta);
            site.normalizedDistance = glm::length(normalizedDelta);
            candidateSites[candidateCount++] = site;
        }
    }

    constexpr std::size_t kMaxConsideredSites = 4;
    std::size_t sitesToConsider = std::min<std::size_t>(kMaxConsideredSites, candidateCount);
    if (sitesToConsider > 0)
    {
        auto candidateLess = [](const CandidateSite& lhs, const CandidateSite& rhs)
        {
            const bool lhsValid = std::isfinite(lhs.normalizedDistance);
            const bool rhsValid = std::isfinite(rhs.normalizedDistance);
            if (lhsValid && rhsValid)
            {
                if (lhs.normalizedDistance == rhs.normalizedDistance)
                {
                    return lhs.distanceSquared < rhs.distanceSquared;
                }
                return lhs.normalizedDistance < rhs.normalizedDistance;
            }

            if (lhsValid != rhsValid)
            {
                return lhsValid;
            }

            return lhs.distanceSquared < rhs.distanceSquared;
        };

        std::partial_sort(candidateSites.begin(), candidateSites.begin() + sitesToConsider,
                          candidateSites.begin() + candidateCount, candidateLess);

        bool allLittleMountains = true;
        for (std::size_t i = 0; i < sitesToConsider; ++i)
        {
            if (candidateSites[i].biome && candidateSites[i].biome->id != BiomeId::LittleMountains)
            {
                allLittleMountains = false;
                break;
            }
        }

        if (allLittleMountains)
        {
            std::size_t bestIndex = candidateCount;
            CandidateSite bestSite{};
            bool hasBestSite = false;
            for (std::size_t i = sitesToConsider; i < candidateCount; ++i)
            {
                const CandidateSite& site = candidateSites[i];
                if (!site.biome || site.biome->id == BiomeId::LittleMountains)
                {
                    continue;
                }

                if (!hasBestSite || candidateLess(site, bestSite))
                {
                    bestIndex = i;
                    bestSite = site;
                    hasBestSite = true;
                }
            }

            if (hasBestSite)
            {
                if (sitesToConsider < kMaxConsideredSites)
                {
                    std::swap(candidateSites[sitesToConsider], candidateSites[bestIndex]);
                    ++sitesToConsider;
                }
                else if (sitesToConsider > 0)
                {
                    std::swap(candidateSites[sitesToConsider - 1], candidateSites[bestIndex]);
                }

                std::partial_sort(candidateSites.begin(), candidateSites.begin() + sitesToConsider,
                                  candidateSites.begin() + candidateCount, candidateLess);
            }
        }
    }

    std::array<WeightedBiome, 5> weightedBiomes{};
    std::size_t weightCount = 0;

    if (sitesToConsider == 0)
    {
        const BiomeDefinition& fallbackBiome = biomeForRegion(biomeRegionX, biomeRegionZ);
        weightedBiomes[weightCount++] = WeightedBiome{&fallbackBiome, 1.0f};
    }
    else
    {
        std::array<float, kMaxConsideredSites> rawWeights{};
        std::array<float, kMaxConsideredSites> distances{};
        for (std::size_t i = 0; i < sitesToConsider; ++i)
        {
            distances[i] = std::sqrt(candidateSites[i].distanceSquared);
        }

        constexpr float kDistanceBias = 1e-3f;
        for (std::size_t i = 0; i < sitesToConsider; ++i)
        {
            const float biasedDistance = distances[i] + kDistanceBias;
            if (!std::isfinite(biasedDistance))
            {
                rawWeights[i] = 1.0f / kDistanceBias;
            }
            else
            {
                const float safeDistance = std::max(biasedDistance, kDistanceBias);
                rawWeights[i] = 1.0f / safeDistance;
            }
        }

        for (std::size_t i = 0; i < sitesToConsider; ++i)
        {
            if (candidateSites[i].biome && candidateSites[i].biome->id == BiomeId::LittleMountains)
            {
                const float influence = littleMountainInfluence(candidateSites[i].normalizedDistance);
                rawWeights[i] *= influence;
                rawWeights[i] *= 1.15f;
            }
        }

        float totalWeight = 0.0f;
        for (std::size_t i = 0; i < sitesToConsider; ++i)
        {
            totalWeight += rawWeights[i];
        }

        if (totalWeight <= std::numeric_limits<float>::epsilon())
        {
            rawWeights.fill(0.0f);
            rawWeights[0] = 1.0f;
            sitesToConsider = 1;
            totalWeight = 1.0f;
        }

#if defined(_DEBUG)
        // Update kDebugWorldX/Z to the column you want to inspect before building.
        constexpr int kDebugWorldX = std::numeric_limits<int>::min();
        constexpr int kDebugWorldZ = std::numeric_limits<int>::min();
#endif
        for (std::size_t i = 0; i < sitesToConsider; ++i)
        {
            const float normalizedWeight = rawWeights[i] / totalWeight;
#if defined(_DEBUG)
            if (worldX == kDebugWorldX && worldZ == kDebugWorldZ)
            {
                const CandidateSite& site = candidateSites[i];
                const char* biomeName = site.biome ? site.biome->name : "<null>";
                std::cout << "[BiomeBlendDebug] column(" << worldX << ", " << worldZ << ") candidate[" << i << "] "
                          << biomeName << " normDist=" << site.normalizedDistance
                          << " weight=" << normalizedWeight << std::endl;
            }
#endif
            if (normalizedWeight <= 0.0f || candidateSites[i].biome == nullptr)
            {
                continue;
            }

            weightedBiomes[weightCount++] = WeightedBiome{candidateSites[i].biome, normalizedWeight};
        }
    }

    if (weightCount == 0)
    {
        const BiomeDefinition& fallbackBiome = biomeForRegion(biomeRegionX, biomeRegionZ);
        weightedBiomes[weightCount++] = WeightedBiome{&fallbackBiome, 1.0f};
    }

    const TerrainBasisSample basis = computeTerrainBasis(worldX, worldZ);
    BiomePerturbationSample perturbations =
        applyBiomePerturbations(weightedBiomes, weightCount, biomeRegionX, biomeRegionZ);

    const BiomeDefinition* clampBiomePtr =
        perturbations.dominantBiome ? perturbations.dominantBiome : &biomeForRegion(biomeRegionX, biomeRegionZ);
    const BiomeDefinition& clampBiome = *clampBiomePtr;

    const BiomeDefinition* littleMountainsDefinition{nullptr};
    float littleMountainsWeight = 0.0f;
    bool hasNonLittleMountainsBiome = false;
    for (std::size_t i = 0; i < weightCount; ++i)
    {
        const WeightedBiome& weightedBiome = weightedBiomes[i];
        if (!weightedBiome.biome || weightedBiome.weight <= 0.0f)
        {
            continue;
        }

        if (weightedBiome.biome->id == BiomeId::LittleMountains)
        {
            littleMountainsWeight += weightedBiome.weight;
            if (!littleMountainsDefinition)
            {
                littleMountainsDefinition = weightedBiome.biome;
            }
        }
        else if (weightedBiome.weight > 0.0f)
        {
            hasNonLittleMountainsBiome = true;
        }
    }
    littleMountainsWeight = std::clamp(littleMountainsWeight, 0.0f, 1.0f);

    float borderAnchorHeight = std::numeric_limits<float>::quiet_NaN();
    bool hasBorderAnchor = false;
    BiomePerturbationSample borderPerturbations{};
    bool hasBorderPerturbations = false;

    if (hasNonLittleMountainsBiome)
    {
        std::array<WeightedBiome, 5> nonMountainBiomes{};
        std::size_t nonMountainCount = 0;
        float nonMountainWeight = 0.0f;
        for (std::size_t i = 0; i < weightCount; ++i)
        {
            const WeightedBiome& weightedBiome = weightedBiomes[i];
            if (!weightedBiome.biome || weightedBiome.weight <= 0.0f)
            {
                continue;
            }

            if (weightedBiome.biome->id == BiomeId::LittleMountains)
            {
                continue;
            }

            nonMountainBiomes[nonMountainCount++] = weightedBiome;
            nonMountainWeight += weightedBiome.weight;
        }

        if (nonMountainCount > 0 && nonMountainWeight > std::numeric_limits<float>::epsilon())
        {
            const float invWeight = 1.0f / nonMountainWeight;
            for (std::size_t i = 0; i < nonMountainCount; ++i)
            {
                nonMountainBiomes[i].weight *= invWeight;
            }

            borderPerturbations =
                applyBiomePerturbations(nonMountainBiomes, nonMountainCount, biomeRegionX, biomeRegionZ);
            borderAnchorHeight = computeBaselineSurfaceHeight(borderPerturbations, basis);
            hasBorderAnchor = std::isfinite(borderAnchorHeight);
            hasBorderPerturbations = true;
        }
    }

    float littleMountainInteriorMask = 0.0f;
    if (littleMountainsDefinition)
    {
        float closestNormalizedDistance = std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < candidateCount; ++i)
        {
            const CandidateSite& site = candidateSites[i];
            if (site.biome != littleMountainsDefinition)
            {
                continue;
            }

            closestNormalizedDistance = std::min(closestNormalizedDistance, site.normalizedDistance);
        }

        if (std::isfinite(closestNormalizedDistance))
        {
            littleMountainInteriorMask = littleMountainInfluence(closestNormalizedDistance);
        }
    }

    if (littleMountainsDefinition && hasNonLittleMountainsBiome && hasBorderPerturbations)
    {
        const float maskedInterior = std::clamp(littleMountainInteriorMask, 0.0f, 1.0f);
        const float interiorBlend = glm::smoothstep(0.2f, 0.75f, maskedInterior);
        const float weightBlend = glm::smoothstep(0.15f, 0.85f, littleMountainsWeight);
        const float transitionBlend = std::clamp(interiorBlend * weightBlend, 0.0f, 1.0f);

        if (transitionBlend < 1.0f)
        {
            auto mixField = [&](float& target, float borderValue)
            {
                target = std::lerp(borderValue, target, transitionBlend);
            };

            mixField(perturbations.blendedOffset, borderPerturbations.blendedOffset);
            mixField(perturbations.blendedScale, borderPerturbations.blendedScale);
            mixField(perturbations.blendedMinHeight, borderPerturbations.blendedMinHeight);
            mixField(perturbations.blendedMaxHeight, borderPerturbations.blendedMaxHeight);
            mixField(perturbations.blendedSlopeBias, borderPerturbations.blendedSlopeBias);
            mixField(perturbations.blendedMaxGradient, borderPerturbations.blendedMaxGradient);

            mixField(perturbations.landWeight, borderPerturbations.landWeight);
            mixField(perturbations.landOffset, borderPerturbations.landOffset);
            mixField(perturbations.landScale, borderPerturbations.landScale);
            mixField(perturbations.landMinHeight, borderPerturbations.landMinHeight);
            mixField(perturbations.landMaxHeight, borderPerturbations.landMaxHeight);
            mixField(perturbations.landSlopeBias, borderPerturbations.landSlopeBias);
            mixField(perturbations.landMaxGradient, borderPerturbations.landMaxGradient);

            mixField(perturbations.oceanWeight, borderPerturbations.oceanWeight);
            mixField(perturbations.oceanOffset, borderPerturbations.oceanOffset);
            mixField(perturbations.oceanScale, borderPerturbations.oceanScale);
            mixField(perturbations.oceanMinHeight, borderPerturbations.oceanMinHeight);
            mixField(perturbations.oceanMaxHeight, borderPerturbations.oceanMaxHeight);
            mixField(perturbations.oceanSlopeBias, borderPerturbations.oceanSlopeBias);
            mixField(perturbations.oceanMaxGradient, borderPerturbations.oceanMaxGradient);
        }
    }

    float enforcedLittleMountainFloor = std::numeric_limits<float>::lowest();
    bool enforceLittleMountainFloor = false;

    auto logHeightClamp = [&](const char* stage, float candidate, float minBound, float maxBound)
    {
        const float localMin = std::min(minBound, maxBound);
        const float localMax = std::max(minBound, maxBound);
        if (candidate >= localMin && candidate <= localMax)
        {
            return;
        }

        static std::atomic<int> clampWarnings{0};
        const int warningIndex = clampWarnings.fetch_add(1, std::memory_order_relaxed);
        if (warningIndex < 8)
        {
            std::cerr << "[BiomeHeightWarning] " << clampBiome.name << ' ' << stage << " candidate " << candidate
                      << " outside [" << localMin << ", " << localMax << ']' << std::endl;
            if (warningIndex == 7)
            {
                std::cerr << "[BiomeHeightWarning] Further biome height warnings suppressed" << std::endl;
            }
        }
    };

    float minHeight = perturbations.blendedMinHeight;
    float maxHeight = perturbations.blendedMaxHeight;
    if (minHeight > maxHeight)
    {
        std::swap(minHeight, maxHeight);
    }

    constexpr float kMinIntAsFloat = static_cast<float>(std::numeric_limits<int>::min());
    constexpr float kMaxIntAsFloat = static_cast<float>(std::numeric_limits<int>::max());

    minHeight = std::clamp(minHeight, kMinIntAsFloat, kMaxIntAsFloat);
    maxHeight = std::clamp(maxHeight, kMinIntAsFloat, kMaxIntAsFloat);

    const float slopeBias = std::clamp(perturbations.blendedSlopeBias, 0.0f, 1.0f);
    const float lowAmplitudeCombined = basis.mainTerrain * 3.0f + basis.mountainNoise * 1.5f + basis.mediumNoise * 1.0f +
                                       basis.detailNoise * 0.5f;
    const float blendedTerrain = std::lerp(basis.combinedNoise, lowAmplitudeCombined, slopeBias);
    const float unclampedMacroHeight = perturbations.blendedOffset + blendedTerrain * perturbations.blendedScale;
    logHeightClamp("macro", unclampedMacroHeight, minHeight, maxHeight);
    float macroStageHeight = std::clamp(unclampedMacroHeight, minHeight, maxHeight);
    float targetHeight = macroStageHeight;

    const bool hasOceanContribution = perturbations.oceanWeight > 0.0f;
    const bool hasLandContribution = perturbations.landWeight > 0.0f;

    const float oceanSlopeBias = std::clamp(hasOceanContribution ? perturbations.oceanSlopeBias : slopeBias, 0.0f, 1.0f);
    const float landSlopeBias = std::clamp(hasLandContribution ? perturbations.landSlopeBias : slopeBias, 0.0f, 1.0f);
    const float oceanGradientWindow =
        std::max(hasOceanContribution ? perturbations.oceanMaxGradient : perturbations.blendedMaxGradient, 0.0f);
    const float landGradientWindow =
        std::max(hasLandContribution ? perturbations.landMaxGradient : perturbations.blendedMaxGradient, 0.0f);

    float oceanTarget = targetHeight;
    if (hasOceanContribution)
    {
        const float oceanVariation = std::lerp(basis.combinedNoise, lowAmplitudeCombined, oceanSlopeBias);
        float rawOceanTarget = perturbations.oceanOffset + oceanVariation * perturbations.oceanScale;
        if (oceanGradientWindow > 0.0f)
        {
            const float oceanGradientMin = perturbations.oceanOffset - oceanGradientWindow;
            const float oceanGradientMax = perturbations.oceanOffset + oceanGradientWindow;
            logHeightClamp("ocean-gradient", rawOceanTarget, oceanGradientMin, oceanGradientMax);
            rawOceanTarget = std::clamp(rawOceanTarget, oceanGradientMin, oceanGradientMax);
        }

        logHeightClamp("ocean", rawOceanTarget, perturbations.oceanMinHeight, perturbations.oceanMaxHeight);
        oceanTarget = std::clamp(rawOceanTarget, perturbations.oceanMinHeight, perturbations.oceanMaxHeight);
        minHeight = std::min(minHeight, oceanTarget);
        maxHeight = std::max(maxHeight, oceanTarget);
    }

    float landTarget = targetHeight;
    if (hasLandContribution)
    {
        float landMin = perturbations.landMinHeight;
        float landMax = perturbations.landMaxHeight;
        if (landMin > landMax)
        {
            std::swap(landMin, landMax);
        }

        landMin = std::clamp(landMin, kMinIntAsFloat, kMaxIntAsFloat);
        landMax = std::clamp(landMax, kMinIntAsFloat, kMaxIntAsFloat);

        const float lowFrequencyNoise = basis.mainTerrain * 0.3f + basis.mediumNoise * 0.4f + basis.detailNoise * 0.3f;
        const float slopeNoise = basis.mountainNoise * 0.15f;
        const float landBaseHeight = std::lerp(macroStageHeight, perturbations.landOffset, landSlopeBias);
        float rawLandTarget = landBaseHeight + (lowFrequencyNoise + slopeNoise) * perturbations.landScale;

        if (landGradientWindow > 0.0f)
        {
            const float landGradientMin = landBaseHeight - landGradientWindow;
            const float landGradientMax = landBaseHeight + landGradientWindow;
            logHeightClamp("land-gradient", rawLandTarget, landGradientMin, landGradientMax);
            rawLandTarget = std::clamp(rawLandTarget, landGradientMin, landGradientMax);
        }

        logHeightClamp("land", rawLandTarget, landMin, landMax);
        landTarget = std::clamp(rawLandTarget, landMin, landMax);
        minHeight = std::min(minHeight, landTarget);
        maxHeight = std::max(maxHeight, landTarget);
    }

    if (littleMountainsDefinition && littleMountainsWeight > 0.0f)
    {
        const float mountainBlend = std::clamp(std::pow(littleMountainsWeight, 1.35f), 0.0f, 1.0f);
        if (mountainBlend > 0.0f)
        {
            const LittleMountainSample mountainSample = computeLittleMountainsHeight(
                worldX, worldZ, *littleMountainsDefinition, littleMountainInteriorMask, hasBorderAnchor, borderAnchorHeight);
            const float mountainHeight = mountainSample.height;
            const float entryFloor = mountainSample.entryFloor;
            const float interiorMask = mountainSample.interiorMask;
            const float maskedMountainBlend = mountainBlend * interiorMask;
            if (maskedMountainBlend > 0.0f)
            {
                constexpr float kMacroEntryExponent = 2.0f;
                constexpr float kTargetEntryExponent = 1.8f;
                constexpr float kLandEntryExponent = 1.5f;

                const float macroBlend =
                    std::clamp(std::pow(mountainBlend, kMacroEntryExponent) * interiorMask, 0.0f, 1.0f);
                const float targetBlend =
                    std::clamp(std::pow(mountainBlend, kTargetEntryExponent) * interiorMask, 0.0f, 1.0f);
                const float landBlend =
                    std::clamp(std::pow(mountainBlend, kLandEntryExponent) * interiorMask, 0.0f, 1.0f);

                const float macroEntryHeight = std::lerp(entryFloor, mountainHeight, macroBlend);
                const float targetEntryHeight = std::lerp(entryFloor, mountainHeight, targetBlend);
                macroStageHeight = std::lerp(macroStageHeight, macroEntryHeight, maskedMountainBlend * 0.45f);
                targetHeight = std::lerp(targetHeight, targetEntryHeight, maskedMountainBlend * 0.6f);
                if (hasLandContribution)
                {
                    const float landEntryHeight = std::lerp(entryFloor, mountainHeight, landBlend);
                    landTarget = std::lerp(landTarget, landEntryHeight, maskedMountainBlend);
                    minHeight = std::min(minHeight, landEntryHeight);
                    maxHeight = std::max(maxHeight, landEntryHeight);
                }
                minHeight = std::min(minHeight, entryFloor);
                minHeight = std::min(minHeight, macroEntryHeight);
                minHeight = std::min(minHeight, targetEntryHeight);
                minHeight = std::min(minHeight, mountainHeight);
                maxHeight = std::max(maxHeight, entryFloor);
                maxHeight = std::max(maxHeight, macroEntryHeight);
                maxHeight = std::max(maxHeight, targetEntryHeight);
                maxHeight = std::max(maxHeight, mountainHeight);
                macroStageHeight = std::clamp(macroStageHeight, minHeight, maxHeight);
                targetHeight = std::clamp(targetHeight, minHeight, maxHeight);

                if (hasNonLittleMountainsBiome && interiorMask > 0.0f)
                {
                    enforcedLittleMountainFloor = entryFloor;
                    enforceLittleMountainFloor = true;
                }
            }
            else if (hasNonLittleMountainsBiome && interiorMask > 0.0f)
            {
                enforcedLittleMountainFloor = entryFloor;
                enforceLittleMountainFloor = true;
            }
        }
    }

    const float globalSeaLevelF = static_cast<float>(kGlobalSeaLevel);
    float shorelineDistance = std::abs(macroStageHeight - globalSeaLevelF);

    if (hasLandContribution && landTarget > globalSeaLevelF)
    {
        constexpr float kShorelineEaseRange = 8.0f;
        if (shorelineDistance < kShorelineEaseRange)
        {
            const float normalized = 1.0f - std::clamp(shorelineDistance / kShorelineEaseRange, 0.0f, 1.0f);
            const float easing = normalized * normalized * normalized;
            const float easedLandHeight = globalSeaLevelF + (landTarget - globalSeaLevelF) * (1.0f - easing);
            const float loweredLandHeight = std::min(landTarget, easedLandHeight);
            landTarget = loweredLandHeight;
            minHeight = std::min(minHeight, landTarget);
            maxHeight = std::max(maxHeight, landTarget);
        }
    }

    float shorelineBlend = 0.0f;
    float oceanShare = 0.0f;
    float landShare = 0.0f;

    if (hasOceanContribution && hasLandContribution)
    {
        const float totalCategoryWeight = perturbations.oceanWeight + perturbations.landWeight;
        if (totalCategoryWeight > std::numeric_limits<float>::epsilon())
        {
            oceanShare = perturbations.oceanWeight / totalCategoryWeight;
            landShare = perturbations.landWeight / totalCategoryWeight;

            auto shorelineRamp = [](float share)
            {
                const float t = std::clamp((share - 0.3f) / 0.2f, 0.0f, 1.0f);
                return t * t * (3.0f - 2.0f * t);
            };

            shorelineBlend = shorelineRamp(oceanShare) * shorelineRamp(landShare);
            if (shorelineBlend > 0.0f)
            {
                const float easedBlend = shorelineBlend * shorelineBlend * (3.0f - 2.0f * shorelineBlend);
                const float shorelineLandHeight = oceanTarget + (landTarget - oceanTarget) * (1.0f - easedBlend);
                const float clampedShoreline = std::max(shorelineLandHeight, oceanTarget);

                landTarget = clampedShoreline;
                minHeight = std::min(minHeight, clampedShoreline);
                maxHeight = std::max(maxHeight, clampedShoreline);
            }
        }
    }

    float distanceToShore = std::numeric_limits<float>::infinity();

    if (hasOceanContribution && hasLandContribution)
    {
        distanceToShore = shorelineDistance;
        if (shorelineBlend > std::numeric_limits<float>::epsilon())
        {
            const float blendScale = std::clamp(shorelineBlend, 0.0f, 1.0f);
            // Areas with stronger ocean/land mixing should be considered closer to the coast.
            distanceToShore /= std::max(blendScale, 0.0001f);
        }
    }

    if (enforceLittleMountainFloor)
    {
        const float floorHeight = enforcedLittleMountainFloor;
        minHeight = std::max(minHeight, floorHeight);
        maxHeight = std::max(maxHeight, minHeight);
        macroStageHeight = std::max(macroStageHeight, floorHeight);
        targetHeight = std::max(targetHeight, floorHeight);
        if (hasLandContribution)
        {
            landTarget = std::max(landTarget, floorHeight);
            maxHeight = std::max(maxHeight, landTarget);
        }
        if (hasOceanContribution)
        {
            oceanTarget = std::max(oceanTarget, floorHeight);
            maxHeight = std::max(maxHeight, oceanTarget);
        }
        macroStageHeight = std::clamp(macroStageHeight, minHeight, maxHeight);
        targetHeight = std::clamp(targetHeight, minHeight, maxHeight);
        if (hasLandContribution)
        {
            landTarget = std::clamp(landTarget, minHeight, maxHeight);
        }
        if (hasOceanContribution)
        {
            oceanTarget = std::clamp(oceanTarget, minHeight, maxHeight);
        }
    }

    if (perturbations.dominantBiome && perturbations.dominantBiome->id == BiomeId::Ocean && hasOceanContribution)
    {
        targetHeight = oceanTarget;
    }
    else if (hasLandContribution)
    {
        targetHeight = landTarget;
    }
    else if (hasOceanContribution)
    {
        targetHeight = oceanTarget;
    }

    ColumnSample sample;
    sample.dominantBiome = perturbations.dominantBiome;
    sample.dominantWeight = perturbations.dominantWeight;
    sample.minSurfaceY = static_cast<int>(std::floor(minHeight));
    sample.maxSurfaceY = static_cast<int>(std::ceil(maxHeight));

    const float roundedSurface = std::round(targetHeight);
    const float clampedSurface = std::clamp(roundedSurface, kMinIntAsFloat, kMaxIntAsFloat);
    sample.surfaceY = static_cast<int>(clampedSurface);

    sample.continentMask = basis.continentMask;
    sample.baseElevation = macroStageHeight;
    sample.oceanContribution = perturbations.oceanWeight;
    sample.landContribution = perturbations.landWeight;
    sample.oceanShare = oceanShare;
    sample.landShare = landShare;
    sample.shorelineBlend = shorelineBlend;
    sample.distanceToShore = distanceToShore;

    if (sample.dominantBiome)
    {
        sample.slabHasSolid = slabMinWorldY <= sample.surfaceY;
        if (sample.slabHasSolid)
        {
            sample.slabHighestSolidY = std::min(sample.surfaceY, slabMaxWorldY);
        }
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
                    if (sample.dominantBiome->id != BiomeId::Ocean)
                    {
                        constexpr float kBeachDistanceRange = 6.0f;
                        constexpr int kBeachHeightBand = 2;
                        const bool nearSeaLevel = std::abs(sample.surfaceY - kGlobalSeaLevel) <= kBeachHeightBand;
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

        std::array<ColumnSample, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)> columnSamples{};
        bool slabContainsTerrain = false;

        for (int x = 0; x < kChunkSizeX; ++x)
        {
            for (int z = 0; z < kChunkSizeZ; ++z)
            {
                const int worldX = baseWorldX + x;
                const int worldZ = baseWorldZ + z;
                ColumnSample sample = sampleColumn(worldX, worldZ, slabMinWorldY, slabMaxWorldY);
                slabContainsTerrain = slabContainsTerrain || sample.slabHasSolid;
                columnSamples[columnIndex(x, z)] = sample;
            }
        }

        if (slabContainsTerrain)
        {
            for (int x = 0; x < kChunkSizeX; ++x)
            {
                for (int z = 0; z < kChunkSizeZ; ++z)
                {
                    const int worldX = baseWorldX + x;
                    const int worldZ = baseWorldZ + z;
                    const ColumnSample& columnSample = columnSamples[columnIndex(x, z)];
                    if (!columnSample.dominantBiome || !columnSample.slabHasSolid)
                    {
                        continue;
                    }

                    const BiomeDefinition& biome = *columnSample.dominantBiome;
                    BlockId surfaceBlock = biome.surfaceBlock;
                    BlockId fillerBlock = biome.fillerBlock;

                    if (biome.id != BiomeId::Ocean)
                    {
                        constexpr float kBeachDistanceRange = 6.0f;
                        constexpr int kBeachHeightBand = 2;
                        const bool nearSeaLevel = std::abs(columnSample.surfaceY - kGlobalSeaLevel) <= kBeachHeightBand;
                        if (nearSeaLevel && columnSample.distanceToShore <= kBeachDistanceRange)
                        {
                            const float beachNoise = hashToUnitFloat(worldX, columnSample.surfaceY, worldZ);
                            surfaceBlock = beachNoise < 0.55f ? BlockId::Sand : BlockId::Grass;
                            fillerBlock = BlockId::Sand;
                        }
                    }

                    if (biome.id == BiomeId::LittleMountains)
                    {
                        const int surfaceHeight = columnSample.surfaceY;
                        if (surfaceHeight < 140)
                        {
                            fillerBlock = BlockId::Grass;
                            surfaceBlock = BlockId::Grass;
                        }
                        else
                        {
                            fillerBlock = BlockId::Stone;
                            if (surfaceHeight >= 200)
                            {
                                surfaceBlock = BlockId::Stone;
                            }
                        }
                    }

                    const int highestSolidWorld = columnSample.slabHighestSolidY;
                    if (highestSolidWorld < chunk.minWorldY)
                    {
                        continue;
                    }

                    const int highestSolidLocal = std::min(highestSolidWorld - chunk.minWorldY, kChunkSizeY - 1);

                    for (int localY = 0; localY <= highestSolidLocal; ++localY)
                    {
                        const int worldY = chunk.minWorldY + localY;
                        BlockId block = BlockId::Air;
                        if (worldY < columnSample.surfaceY)
                        {
                            block = fillerBlock;
                        }
                        else if (worldY == columnSample.surfaceY)
                        {
                            block = surfaceBlock;
                        }

                        if (block == BlockId::Air)
                        {
                            continue;
                        }

                        if (biome.id == BiomeId::LittleMountains && block != BlockId::Air)
                        {
                            const int surfaceHeight = columnSample.surfaceY;
                            if (surfaceHeight >= 160)
                            {
                                const int depthFromSurface = surfaceHeight - worldY;
                                if (surfaceHeight >= 220 || depthFromSurface <= 2)
                                {
                                    if (surfaceHeight >= 220)
                                    {
                                        block = BlockId::Stone;
                                    }
                                    else
                                    {
                                        const float stoneNoise = hashToUnitFloat(worldX, worldY, worldZ);
                                        const float blend = glm::smoothstep(0.0f, 3.0f,
                                                                             static_cast<float>(2 - depthFromSurface));
                                        if (stoneNoise < blend)
                                        {
                                            block = BlockId::Stone;
                                        }
                                    }
                                }
                            }
                        }

                        chunk.blocks[blockIndex(x, localY, z)] = block;
                        anySolid = true;
                    }
                }
            }
        }

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
                    const ColumnSample columnSample = sampleColumn(worldX, worldZ);
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

                            const ColumnSample neighborSample = sampleColumn(worldX + dx, worldZ + dz);
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

void ChunkManager::render(GLuint shaderProgram,
                          const glm::mat4& viewProj,
                          const glm::vec3& cameraPos,
                          const Frustum& frustum,
                          const ChunkShaderUniformLocations& uniforms) const
{
    impl_->render(shaderProgram, viewProj, cameraPos, frustum, uniforms);
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

