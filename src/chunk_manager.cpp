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
#include <utility>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

float computeFarPlaneForViewDistance(int viewDistance) noexcept
{
    const float horizontalSpan = static_cast<float>(viewDistance + 1) * static_cast<float>(std::max(kChunkSizeX, kChunkSizeZ));
    const float diagonal = std::sqrt(2.0f) * horizontalSpan;
    return std::max(diagonal + kFarPlanePadding, kDefaultFarPlane);
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
};

constexpr std::size_t kBiomeCount = toIndex(BiomeId::Count);

constexpr std::array<BiomeDefinition, kBiomeCount> kBiomeDefinitions{ {
    {BiomeId::Grasslands, "Grasslands", BlockId::Grass, BlockId::Grass, false, 0.0f, 16.0f, 1.0f, 2, kChunkSizeY - 3},
    {BiomeId::Forest, "Forest", BlockId::Grass, BlockId::Grass, true, 3.5f, 18.0f, 1.1f, 3, kChunkSizeY - 3},
    {BiomeId::Desert, "Desert", BlockId::Sand, BlockId::Sand, false, 0.0f, 12.0f, 0.5f, 1, kChunkSizeY - 4},
    {BiomeId::Ocean, "Ocean", BlockId::Water, BlockId::Water, false, 0.0f, 20.0f, 0.0f, 6, kChunkSizeY - 2},
} };

struct ColumnSample
{
    const BiomeDefinition* dominantBiome{nullptr};
    float dominantWeight{0.0f};
    int surfaceY{0};
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

    glm::ivec3 coord;
    int minWorldY{0};
    int maxWorldY{0};
    std::vector<BlockId> blocks;
    std::atomic<ChunkState> state;

    GLuint vao{0};
    GLuint vbo{0};
    GLuint ibo{0};
    GLsizei indexCount{0};
    std::size_t vertexCapacity{0};
    std::size_t indexCapacity{0};
    bool queuedForUpload{false};

    mutable std::mutex meshMutex;
    MeshData meshData;
    bool meshReady{false};
    std::atomic<int> inFlight{0};
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

    BlockId blockAt(const glm::ivec3& worldPos) const noexcept;
    glm::vec3 findSafeSpawnPosition(float worldX, float worldZ) const;

private:
    void startWorkerThreads();
    void stopWorkerThreads();
    void workerThreadFunction();
    void enqueueJob(const std::shared_ptr<Chunk>& chunk, JobType type, const glm::ivec3& coord);
    void processJob(const Job& job);
    std::shared_ptr<Chunk> popNextChunkForUpload();
    void queueChunkForUpload(const std::shared_ptr<Chunk>& chunk);
    void requeueChunkForUpload(const std::shared_ptr<Chunk>& chunk, bool toFront);

    struct BufferEntry
    {
        GLuint vao{0};
        GLuint vbo{0};
        GLuint ibo{0};
        std::size_t vertexCapacity{0};
        std::size_t indexCapacity{0};
    };

    static std::size_t bucketForSize(std::size_t bytes) noexcept;
    BufferEntry acquireBufferEntry(std::size_t vertexBytes, std::size_t indexBytes);
    void releaseChunkBuffers(Chunk& chunk);
    void ensureChunkBuffers(Chunk& chunk, std::size_t vertexBytes, std::size_t indexBytes);
    void recycleChunkGPU(Chunk& chunk);
    void destroyBufferPool();

    struct RingProgress
    {
        bool fullyLoaded{false};
        bool budgetExhausted{false};
    };

    RingProgress ensureVolume(const glm::ivec3& center, int horizontalRadius, int verticalRadius, int& jobBudget);
    void removeDistantChunks(const glm::ivec3& center, int horizontalThreshold, int verticalThreshold);
    bool ensureChunkAsync(const glm::ivec3& coord);
    void uploadReadyMeshes();
    void uploadChunkMesh(Chunk& chunk);
    void buildChunkMeshAsync(Chunk& chunk);
    static glm::ivec3 worldToChunkCoords(int worldX, int worldY, int worldZ) noexcept;
    std::shared_ptr<Chunk> getChunkShared(const glm::ivec3& coord) noexcept;
    std::shared_ptr<const Chunk> getChunkShared(const glm::ivec3& coord) const noexcept;
    Chunk* getChunk(const glm::ivec3& coord) noexcept;
    const Chunk* getChunk(const glm::ivec3& coord) const noexcept;
    void markNeighborsForRemeshingIfNeeded(const glm::ivec3& coord, int localX, int localY, int localZ);

    void generateChunkBlocks(Chunk& chunk);
    ColumnSample sampleColumn(int worldX, int worldZ) const;

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

    std::deque<std::weak_ptr<Chunk>> uploadQueue_;
    std::mutex uploadQueueMutex_;
    std::map<std::size_t, std::vector<BufferEntry>> bufferPool_;
    std::mutex bufferPoolMutex_;
    PerlinNoise noise_;
    std::unordered_map<glm::ivec3, std::shared_ptr<Chunk>, ChunkHasher> chunks_;
    mutable std::mutex chunksMutex;
    const glm::vec3 lightDirection_{glm::normalize(glm::vec3(0.5f, -1.0f, 0.2f))};
    GLuint atlasTexture_{0};
    JobQueue jobQueue_;
    ColumnManager columnManager_;
    std::vector<std::thread> workerThreads_;
    std::atomic<bool> shouldStop_;

    glm::ivec3 highlightedBlock_{0};
    bool hasHighlight_{false};

    int viewDistance_;
    int targetViewDistance_;
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
    return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
}

// ChunkManager::Impl methods (to be filled)

ChunkManager::Impl::Impl(unsigned seed)
    : noise_(seed),
      shouldStop_(false),
      viewDistance_(kDefaultViewDistance),
      targetViewDistance_(kDefaultViewDistance)
{
    kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);
    startWorkerThreads();
}

ChunkManager::Impl::~Impl()
{
    stopWorkerThreads();
    clear();
    destroyBufferPool();
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

    blockAtlasConfigured_ = true;
}

void ChunkManager::Impl::update(const glm::vec3& cameraPos)
{
    const int worldX = static_cast<int>(std::floor(cameraPos.x));
    const int worldY = static_cast<int>(std::floor(cameraPos.y));
    const int worldZ = static_cast<int>(std::floor(cameraPos.z));
    const glm::ivec3 centerChunk = worldToChunkCoords(worldX, worldY, worldZ);

    const int verticalRadius = kVerticalViewDistance;

    jobQueue_.updatePriorityOrigin(centerChunk);

    if (viewDistance_ > targetViewDistance_)
    {
        viewDistance_ = targetViewDistance_;
    }

    int jobBudget = kMaxChunkJobsPerFrame;

    for (int ring = 0; ring <= viewDistance_ && jobBudget > 0; ++ring)
    {
        RingProgress progress = ensureVolume(centerChunk, ring, verticalRadius, jobBudget);
        if (progress.budgetExhausted)
        {
            break;
        }
    }

    int ringsExpanded = 0;
    while (jobBudget > 0 && viewDistance_ < targetViewDistance_ && ringsExpanded < kMaxRingsPerFrame)
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

    removeDistantChunks(centerChunk, targetViewDistance_ + 1, verticalRadius + 1);

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

        glBindVertexArray(chunkPtr->vao);
        glDrawElements(GL_TRIANGLES, chunkPtr->indexCount, GL_UNSIGNED_INT, nullptr);
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
            recycleChunkGPU(*chunk);
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

    const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
    if (local.y < 0 || local.y >= kChunkSizeY)
    {
        return false;
    }
    const std::size_t blockIdx = blockIndex(local.x, local.y, local.z);


    {
        std::lock_guard<std::mutex> lock(chunk->meshMutex);
        if (!isSolid(chunk->blocks[blockIdx]))
        {
            return false;
        }

        chunk->blocks[blockIdx] = BlockId::Air;
        columnManager_.updateColumn(*chunk, local.x, local.z);
        chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
    }

    enqueueJob(chunk, JobType::Mesh, chunkCoord);
    markNeighborsForRemeshingIfNeeded(chunkCoord, local.x, local.y, local.z);

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

    const glm::ivec3 local = localBlockCoords(placePos, chunkCoord);
    if (local.y < 0 || local.y >= kChunkSizeY)
    {
        return false;
    }
    const std::size_t blockIdx = blockIndex(local.x, local.y, local.z);


    {
        std::lock_guard<std::mutex> lock(chunk->meshMutex);
        if (isSolid(chunk->blocks[blockIdx]))
        {
            return false;
        }

        chunk->blocks[blockIdx] = BlockId::Grass;
        columnManager_.updateColumn(*chunk, local.x, local.z);
        chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
    }

    enqueueJob(chunk, JobType::Mesh, chunkCoord);
    markNeighborsForRemeshingIfNeeded(chunkCoord, local.x, local.y, local.z);

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
            std::cout << "Extended render distance target: " << targetViewDistance_ << " chunks (total: "
                      << (2 * targetViewDistance_ + 1) * (2 * targetViewDistance_ + 1) << " chunks)" << std::endl;
        }
        else
        {
            std::cout << "Switching to default render distance..." << std::endl;

            targetViewDistance_ = kDefaultViewDistance;
            kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);
            std::cout << "Default render distance target: " << targetViewDistance_ << " chunks" << std::endl;
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
        const int clampedDistance = std::max(1, std::min(distance, 200));
        targetViewDistance_ = clampedDistance;
        kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);

        if (viewDistance_ > targetViewDistance_)
        {
            viewDistance_ = targetViewDistance_;
        }

        std::cout << "Render distance set to: " << targetViewDistance_ << " chunks (total: "
                  << (2 * targetViewDistance_ + 1) * (2 * targetViewDistance_ + 1) << " chunks)" << std::endl;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error setting render distance: " << ex.what() << std::endl;
    }
}

BlockId ChunkManager::Impl::blockAt(const glm::ivec3& worldPos) const noexcept
{
    const glm::ivec3 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
    auto chunk = getChunkShared(chunkCoord);
    if (!chunk)
    {
        return BlockId::Air;
    }

    const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
    if (local.y < 0 || local.y >= kChunkSizeY)
    {
        return BlockId::Air;
    }
    return chunk->blocks[blockIndex(local.x, local.y, local.z)];

}

glm::vec3 ChunkManager::Impl::findSafeSpawnPosition(float worldX, float worldZ) const
{
    const float halfWidth = kPlayerWidth * 0.5f;
    const int baseX = static_cast<int>(std::floor(worldX));
    const int baseZ = static_cast<int>(std::floor(worldZ));
    int highestSolid = columnManager_.highestSolidBlock(baseX, baseZ);
    if (highestSolid == ColumnManager::kNoHeight)
    {
        highestSolid = sampleColumn(baseX, baseZ).surfaceY;
    }

    const int clearanceHeight = static_cast<int>(std::ceil(kPlayerHeight)) + 1;
    const int searchTop = highestSolid + clearanceHeight + 2;
    int searchBottom = highestSolid - 64;
    if (searchBottom > searchTop)
    {
        searchBottom = searchTop - 1;
    }
    searchBottom = std::max(searchBottom, highestSolid - 128);
    searchBottom = std::max(searchBottom, -256);

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

    std::cout << "Warning: No safe spawn found, spawning above terrain" << std::endl;
    const float fallbackY = static_cast<float>(searchTop) + kCameraEyeHeight;
    return glm::vec3(worldX, fallbackY, worldZ);
}

void ChunkManager::Impl::startWorkerThreads()
{
    const unsigned numThreads = std::max(1u, std::thread::hardware_concurrency() / 2);
    workerThreads_.reserve(numThreads);

    for (unsigned i = 0; i < numThreads; ++i)
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
        generateChunkBlocks(*chunk);
        chunk->state.store(ChunkState::Meshing, std::memory_order_release);

        enqueueJob(chunk, JobType::Mesh, job.chunkCoord);
    }
    else if (job.type == JobType::Mesh)
    {
        buildChunkMeshAsync(*chunk);
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

std::size_t ChunkManager::Impl::bucketForSize(std::size_t bytes) noexcept
{
    bytes = std::max(bytes, kMinBufferSizeBytes);
    bytes -= 1;
    bytes |= bytes >> 1;
    bytes |= bytes >> 2;
    bytes |= bytes >> 4;
    bytes |= bytes >> 8;
    bytes |= bytes >> 16;
#if SIZE_MAX > 0xffffffffu
    bytes |= bytes >> 32;
#endif
    return bytes + 1;
}

ChunkManager::Impl::BufferEntry ChunkManager::Impl::acquireBufferEntry(std::size_t vertexBytes, std::size_t indexBytes)
{
    const std::size_t vertexBucket = bucketForSize(vertexBytes);
    const std::size_t indexBucket = bucketForSize(indexBytes);

    {
        std::lock_guard<std::mutex> lock(bufferPoolMutex_);
        auto it = bufferPool_.lower_bound(vertexBucket);
        while (it != bufferPool_.end())
        {
            auto& pool = it->second;
            for (std::size_t i = 0; i < pool.size(); ++i)
            {
                if (pool[i].indexCapacity >= indexBucket)
                {
                    auto itEntry = pool.begin() + static_cast<std::ptrdiff_t>(i);
                    BufferEntry entry = *itEntry;
                    pool.erase(itEntry);
                    if (pool.empty())
                    {
                        bufferPool_.erase(it);
                    }
                    return entry;
                }
            }
            ++it;
        }
    }

    BufferEntry entry;
    entry.vertexCapacity = vertexBucket;
    entry.indexCapacity = indexBucket;

    glGenVertexArrays(1, &entry.vao);
    glGenBuffers(1, &entry.vbo);
    glGenBuffers(1, &entry.ibo);

    glBindVertexArray(entry.vao);

    glBindBuffer(GL_ARRAY_BUFFER, entry.vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(entry.vertexCapacity), nullptr, GL_DYNAMIC_DRAW);
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

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, entry.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(entry.indexCapacity), nullptr, GL_DYNAMIC_DRAW);

    glBindVertexArray(0);

    return entry;
}

void ChunkManager::Impl::releaseChunkBuffers(Chunk& chunk)
{
    if (chunk.vao == 0)
    {
        chunk.vertexCapacity = 0;
        chunk.indexCapacity = 0;
        return;
    }

    BufferEntry entry{};
    entry.vao = chunk.vao;
    entry.vbo = chunk.vbo;
    entry.ibo = chunk.ibo;
    entry.vertexCapacity = chunk.vertexCapacity;
    entry.indexCapacity = chunk.indexCapacity;

    chunk.vao = 0;
    chunk.vbo = 0;
    chunk.ibo = 0;
    chunk.vertexCapacity = 0;
    chunk.indexCapacity = 0;
    chunk.indexCount = 0;

    std::lock_guard<std::mutex> lock(bufferPoolMutex_);
    bufferPool_[entry.vertexCapacity].push_back(entry);
}

void ChunkManager::Impl::ensureChunkBuffers(Chunk& chunk, std::size_t vertexBytes, std::size_t indexBytes)
{
    const std::size_t requiredVertex = std::max(vertexBytes, static_cast<std::size_t>(sizeof(Vertex)));
    const std::size_t requiredIndex = std::max(indexBytes, static_cast<std::size_t>(sizeof(std::uint32_t)));

    if (chunk.vao != 0 &&
        chunk.vertexCapacity >= requiredVertex &&
        chunk.indexCapacity >= requiredIndex)
    {
        return;
    }

    if (chunk.vao != 0)
    {
        releaseChunkBuffers(chunk);
    }

    BufferEntry entry = acquireBufferEntry(requiredVertex, requiredIndex);
    chunk.vao = entry.vao;
    chunk.vbo = entry.vbo;
    chunk.ibo = entry.ibo;
    chunk.vertexCapacity = entry.vertexCapacity;
    chunk.indexCapacity = entry.indexCapacity;
}

void ChunkManager::Impl::recycleChunkGPU(Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(chunk.meshMutex);
    releaseChunkBuffers(chunk);
    chunk.meshData.clear();
    chunk.meshReady = false;
    chunk.queuedForUpload = false;
    chunk.indexCount = 0;
}

void ChunkManager::Impl::destroyBufferPool()
{
    std::lock_guard<std::mutex> lock(bufferPoolMutex_);
    for (auto& [_, entries] : bufferPool_)
    {
        for (auto& entry : entries)
        {
            if (entry.ibo != 0)
            {
                glDeleteBuffers(1, &entry.ibo);
            }
            if (entry.vbo != 0)
            {
                glDeleteBuffers(1, &entry.vbo);
            }
            if (entry.vao != 0)
            {
                glDeleteVertexArrays(1, &entry.vao);
            }
        }
    }
    bufferPool_.clear();
}

ChunkManager::Impl::RingProgress ChunkManager::Impl::ensureVolume(const glm::ivec3& center,
                                                                  int horizontalRadius,
                                                                  int verticalRadius,
                                                                  int& jobBudget)
{
    bool missingFound = false;

    auto visitCoordinate = [&](const glm::ivec3& coord) -> bool {
        if (jobBudget <= 0)
        {
            return true;
        }

        if (ensureChunkAsync(coord))
        {
            --jobBudget;
            missingFound = true;
        }

        return jobBudget <= 0;
    };

    for (int dy = -verticalRadius; dy <= verticalRadius; ++dy)
    {
        const glm::ivec3 base = center + glm::ivec3(0, dy, 0);

        if (horizontalRadius == 0)
        {
            if (visitCoordinate(base))
            {
                return RingProgress{false, true};
            }
            continue;
        }

        for (int dx = -horizontalRadius; dx <= horizontalRadius; ++dx)
        {
            if (visitCoordinate(base + glm::ivec3(dx, 0, -horizontalRadius)))
            {
                return RingProgress{false, true};
            }
            if (visitCoordinate(base + glm::ivec3(dx, 0, horizontalRadius)))
            {
                return RingProgress{false, true};
            }
        }

        for (int dz = -horizontalRadius + 1; dz <= horizontalRadius - 1; ++dz)
        {
            if (visitCoordinate(base + glm::ivec3(-horizontalRadius, 0, dz)))
            {
                return RingProgress{false, true};
            }
            if (visitCoordinate(base + glm::ivec3(horizontalRadius, 0, dz)))
            {
                return RingProgress{false, true};
            }
        }
    }

    return RingProgress{!missingFound, false};
}

void ChunkManager::Impl::removeDistantChunks(const glm::ivec3& center,
                                             int horizontalThreshold,
                                             int verticalThreshold)
{
    std::vector<glm::ivec3> toRemove;
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        toRemove.reserve(chunks_.size());
        for (const auto& [coord, chunkPtr] : chunks_)
        {
            const int dx = coord.x - center.x;
            const int dz = coord.z - center.z;
            const int dy = coord.y - center.y;
            const int horizontalDistance = std::max(std::abs(dx), std::abs(dz));
            if (horizontalDistance > horizontalThreshold || std::abs(dy) > verticalThreshold)
            {
                toRemove.push_back(coord);
            }
        }
    }

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
            recycleChunkGPU(*chunk);
        }
    }
}

bool ChunkManager::Impl::ensureChunkAsync(const glm::ivec3& coord)
{
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

            chunk = std::make_shared<Chunk>(coord);
            chunk->state.store(ChunkState::Generating, std::memory_order_release);
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
    std::size_t remainingBudget = kUploadBudgetBytesPerFrame;
    bool uploadedAnything = false;

    while (remainingBudget > 0 || !uploadedAnything)
    {
        std::shared_ptr<Chunk> chunk = popNextChunkForUpload();
        if (!chunk)
        {
            break;
        }

        if (!chunk->meshReady || chunk->state.load() != ChunkState::Ready)
        {
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
            break;
        }

        uploadChunkMesh(*chunk);
        chunk->state.store(ChunkState::Uploaded, std::memory_order_release);
        chunk->meshReady = false;
        uploadedAnything = true;

        if (totalBytes >= remainingBudget)
        {
            remainingBudget = 0;
        }
        else
        {
            remainingBudget -= totalBytes;
        }
    }
}

void ChunkManager::Impl::uploadChunkMesh(Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(chunk.meshMutex);

    if (chunk.meshData.empty())
    {
        releaseChunkBuffers(chunk);
        chunk.meshData.clear();
        chunk.indexCount = 0;
        return;
    }

    const std::size_t vertexBytes = chunk.meshData.vertices.size() * sizeof(Vertex);
    const std::size_t indexBytes = chunk.meshData.indices.size() * sizeof(std::uint32_t);

    ensureChunkBuffers(chunk, vertexBytes, indexBytes);

    glBindVertexArray(chunk.vao);
    glBindBuffer(GL_ARRAY_BUFFER, chunk.vbo);
    if (vertexBytes > 0)
    {
        glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(vertexBytes), chunk.meshData.vertices.data());
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, chunk.ibo);
    if (indexBytes > 0)
    {
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(indexBytes), chunk.meshData.indices.data());
    }

    chunk.indexCount = static_cast<GLsizei>(chunk.meshData.indices.size());
    glBindVertexArray(0);

    chunk.meshData.clear();
}

void ChunkManager::Impl::buildChunkMeshAsync(Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(chunk.meshMutex);
    chunk.meshData.clear();

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

ColumnSample ChunkManager::Impl::sampleColumn(int worldX, int worldZ) const
{

    auto biomeForRegion = [&](int regionX, int regionZ) -> const BiomeDefinition&
    {
        const float selector = hashToUnitFloat(regionX, 31, regionZ);
        const std::size_t maxIndex = kBiomeDefinitions.size() - 1;
        const std::size_t biomeIndex = std::min(static_cast<std::size_t>(selector * static_cast<float>(kBiomeDefinitions.size())), maxIndex);
        return kBiomeDefinitions[biomeIndex];
    };

        struct WeightedBiome
        {
            const BiomeDefinition* biome;
            float weight;
        };

        const int chunkX = floorDiv(worldX, kChunkSizeX);
        const int chunkZ = floorDiv(worldZ, kChunkSizeZ);
        const int biomeRegionX = floorDiv(chunkX, kBiomeSizeInChunks);
        const int biomeRegionZ = floorDiv(chunkZ, kBiomeSizeInChunks);

        const int regionBaseChunkX = biomeRegionX * kBiomeSizeInChunks;
        const int regionBaseChunkZ = biomeRegionZ * kBiomeSizeInChunks;
        const int regionBaseBlockX = regionBaseChunkX * kChunkSizeX;
        const int regionBaseBlockZ = regionBaseChunkZ * kChunkSizeZ;

        const int regionSizeBlocksX = kBiomeSizeInChunks * kChunkSizeX;
        const int regionSizeBlocksZ = kBiomeSizeInChunks * kChunkSizeZ;

        const int localBlockX = worldX - regionBaseBlockX;
        const int localBlockZ = worldZ - regionBaseBlockZ;

        constexpr float kBiomeBlendRangeBlocks = 4.0f;

        auto smooth01 = [](float t)
        {
            t = std::clamp(t, 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
        };

        auto edgeInfluence = [&](float distance)
        {
            if (distance >= kBiomeBlendRangeBlocks)
            {
                return 0.0f;
            }

            const float normalized = 1.0f - (distance / kBiomeBlendRangeBlocks);
            return smooth01(normalized);
        };

        const float distanceLeft = static_cast<float>(localBlockX);
        const float distanceRight = static_cast<float>((regionSizeBlocksX - 1) - localBlockX);
        const float distanceNorth = static_cast<float>(localBlockZ);
        const float distanceSouth = static_cast<float>((regionSizeBlocksZ - 1) - localBlockZ);

        auto edgeVariation = [&](int offsetX, int offsetZ)
        {
            const float sampleX = static_cast<float>(worldX + offsetX * 31);
            const float sampleZ = static_cast<float>(worldZ + offsetZ * 31);
            const float variationNoise = noise_.fbm(sampleX * 0.07f, sampleZ * 0.07f, 3, 0.55f, 2.0f);
            return std::clamp(variationNoise * 0.5f + 0.5f, 0.0f, 1.0f);
        };

        float leftWeightAxis = edgeInfluence(distanceLeft);
        float rightWeightAxis = edgeInfluence(distanceRight);
        float northWeightAxis = edgeInfluence(distanceNorth);
        float southWeightAxis = edgeInfluence(distanceSouth);

        leftWeightAxis *= 0.3f + edgeVariation(-1, 0) * 0.7f;
        rightWeightAxis *= 0.3f + edgeVariation(1, 0) * 0.7f;
        northWeightAxis *= 0.3f + edgeVariation(0, -1) * 0.7f;
        southWeightAxis *= 0.3f + edgeVariation(0, 1) * 0.7f;

        float centerWeightAxisX = 1.0f - (leftWeightAxis + rightWeightAxis);
        float centerWeightAxisZ = 1.0f - (northWeightAxis + southWeightAxis);
        centerWeightAxisX = std::clamp(centerWeightAxisX, 0.0f, 1.0f);
        centerWeightAxisZ = std::clamp(centerWeightAxisZ, 0.0f, 1.0f);

        const float axisSumX = leftWeightAxis + rightWeightAxis + centerWeightAxisX;
        if (axisSumX > std::numeric_limits<float>::epsilon())
        {
            leftWeightAxis /= axisSumX;
            rightWeightAxis /= axisSumX;
            centerWeightAxisX /= axisSumX;
        }

        const float axisSumZ = northWeightAxis + southWeightAxis + centerWeightAxisZ;
        if (axisSumZ > std::numeric_limits<float>::epsilon())
        {
            northWeightAxis /= axisSumZ;
            southWeightAxis /= axisSumZ;
            centerWeightAxisZ /= axisSumZ;
        }

        std::array<WeightedBiome, 5> weightedBiomes{};

        std::size_t weightCount = 0;

        auto addBiomeWeight = [&](int regionOffsetX, int regionOffsetZ, float weight)
        {
            if (weight <= 0.0f)
            {
                return;
            }

            const float scatterNoise = hashToUnitFloat(worldX + regionOffsetX * 53,
                                                       157 + regionOffsetX * 31 + regionOffsetZ * 17,
                                                       worldZ + regionOffsetZ * 71);
            const bool isCenterRegion = (regionOffsetX == 0) && (regionOffsetZ == 0);
            const float scatterMin = isCenterRegion ? 0.85f : 0.4f;
            const float scatterMax = isCenterRegion ? 1.1f : 1.25f;
            const float scatter = scatterMin + (scatterMax - scatterMin) * scatterNoise;
            weight *= scatter;

            if (weight <= 0.0f)
            {
                return;
            }

            const BiomeDefinition& biome = biomeForRegion(biomeRegionX + regionOffsetX, biomeRegionZ + regionOffsetZ);
            weightedBiomes[weightCount++] = WeightedBiome{&biome, weight};
        };

        addBiomeWeight(0, 0, centerWeightAxisX * centerWeightAxisZ);
        addBiomeWeight(-1, 0, leftWeightAxis * centerWeightAxisZ);
        addBiomeWeight(1, 0, rightWeightAxis * centerWeightAxisZ);
        addBiomeWeight(0, -1, centerWeightAxisX * northWeightAxis);
        addBiomeWeight(0, 1, centerWeightAxisX * southWeightAxis);

        if (weightCount == 0)
        {
            addBiomeWeight(0, 0, 1.0f);
        }

        float totalWeight = 0.0f;
        for (std::size_t i = 0; i < weightCount; ++i)
        {
            totalWeight += weightedBiomes[i].weight;
        }

        if (totalWeight <= std::numeric_limits<float>::epsilon())
        {
            totalWeight = 1.0f;
        }

        for (std::size_t i = 0; i < weightCount; ++i)
        {
            weightedBiomes[i].weight /= totalWeight;
        }

        const float nx = static_cast<float>(worldX) * 0.01f;
        const float nz = static_cast<float>(worldZ) * 0.01f;

        const float mainTerrain = noise_.fbm(nx, nz, 6, 0.5f, 2.0f);
        const float mountainNoise = noise_.ridge(nx * 0.4f, nz * 0.4f, 5, 2.1f, 0.5f);
        const float detailNoise = noise_.fbm(nx * 4.0f, nz * 4.0f, 8, 0.45f, 2.2f);
        const float mediumNoise = noise_.fbm(nx * 0.8f, nz * 0.8f, 7, 0.5f, 2.0f);

        const float combined = mainTerrain * 12.0f +
                               mountainNoise * 8.0f +
                               mediumNoise * 4.0f +
                               detailNoise * 2.0f;

        float blendedOffset = 0.0f;
        float blendedScale = 0.0f;
        float blendedMinHeight = 0.0f;
        float blendedMaxHeight = 0.0f;
        const BiomeDefinition* dominantBiome = nullptr;
        float dominantWeight = -1.0f;

        for (std::size_t i = 0; i < weightCount; ++i)
        {
            const auto& weightedBiome = weightedBiomes[i];
            blendedOffset += weightedBiome.biome->heightOffset * weightedBiome.weight;
            blendedScale += weightedBiome.biome->heightScale * weightedBiome.weight;
            blendedMinHeight += static_cast<float>(weightedBiome.biome->minHeight) * weightedBiome.weight;
            blendedMaxHeight += static_cast<float>(weightedBiome.biome->maxHeight) * weightedBiome.weight;

            if (weightedBiome.weight > dominantWeight)
            {
                dominantWeight = weightedBiome.weight;
                dominantBiome = weightedBiome.biome;
            }
        }

        dominantWeight = std::max(dominantWeight, 0.0f);
        if (dominantBiome == nullptr)
        {
            dominantBiome = &biomeForRegion(biomeRegionX, biomeRegionZ);
        }

        const float minHeight = std::clamp(blendedMinHeight, 0.0f, static_cast<float>(kChunkSizeY - 1));
        const float maxHeight = std::clamp(blendedMaxHeight, 0.0f, static_cast<float>(kChunkSizeY - 1));

        float targetHeight = blendedOffset + combined * blendedScale;
        targetHeight = std::clamp(targetHeight, minHeight, maxHeight);

        ColumnSample sample;
        sample.dominantBiome = dominantBiome;
        sample.dominantWeight = dominantWeight;
        sample.surfaceY = std::clamp(static_cast<int>(std::round(targetHeight)), 0, kChunkSizeY - 1);
        return sample;
    }
}

void ChunkManager::Impl::generateChunkBlocks(Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(chunk.meshMutex);
    std::fill(chunk.blocks.begin(), chunk.blocks.end(), BlockId::Air);

    const int baseWorldX = chunk.coord.x * kChunkSizeX;
    const int baseWorldZ = chunk.coord.z * kChunkSizeZ;

    std::array<ColumnSample, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)> columnSamples{};
    for (int x = 0; x < kChunkSizeX; ++x)
    {
        for (int z = 0; z < kChunkSizeZ; ++z)
        {
            const int worldX = baseWorldX + x;
            const int worldZ = baseWorldZ + z;
            columnSamples[columnIndex(x, z)] = sampleColumn(worldX, worldZ);
        }
    }

    bool anySolid = false;

    for (int x = 0; x < kChunkSizeX; ++x)
    {
        for (int z = 0; z < kChunkSizeZ; ++z)
        {
            const ColumnSample& columnSample = columnSamples[columnIndex(x, z)];
            if (!columnSample.dominantBiome)
            {
                continue;
            }

            const BiomeDefinition& biome = *columnSample.dominantBiome;
            const int surfaceY = columnSample.surfaceY;

            if (surfaceY < chunk.minWorldY)
            {
                continue;
            }

            const int localSurface = surfaceY - chunk.minWorldY;
            const int columnFillTop = std::min(localSurface, kChunkSizeY - 1);

            for (int localY = 0; localY <= columnFillTop; ++localY)
            {
                const int worldY = chunk.minWorldY + localY;
                BlockId block = BlockId::Air;
                if (worldY < surfaceY)
                {
                    block = biome.fillerBlock;
                }
                else if (worldY == surfaceY)
                {
                    block = biome.surfaceBlock;
                }

                if (block != BlockId::Air)
                {
                    anySolid = true;
                }

                chunk.blocks[blockIndex(x, localY, z)] = block;
            }

            if (surfaceY > chunk.maxWorldY)
            {
                for (int localY = columnFillTop + 1; localY < kChunkSizeY; ++localY)
                {
                    chunk.blocks[blockIndex(x, localY, z)] = biome.fillerBlock;
                    anySolid = true;
                }
            }
        }
    }

    constexpr int kTreeMinHeight = 6;
    constexpr int kTreeMaxHeight = 8;
    constexpr int kTreeMaxRadius = 2;

    auto trySetBlock = [&](int worldX, int worldY, int worldZ, BlockId block, bool replaceSolid)
    {
        const int localX = worldX - baseWorldX;
        const int localZ = worldZ - baseWorldZ;
        if (localX < 0 || localX >= kChunkSizeX || localZ < 0 || localZ >= kChunkSizeZ)
        {
            return;
        }
        if (worldY < chunk.minWorldY || worldY > chunk.maxWorldY)
        {
            return;
        }

        const int localY = worldY - chunk.minWorldY;
        const std::size_t idx = blockIndex(localX, localY, localZ);
        BlockId& destination = chunk.blocks[idx];
        if (!replaceSolid && destination != BlockId::Air)
        {
            return;
        }

        destination = block;
        if (block != BlockId::Air)
        {
            anySolid = true;

        }
    };

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

            if (groundLocalY <= 2 || groundLocalY >= kChunkSizeY - (kTreeMaxHeight + 1))
            {
                continue;
            }

            const int groundWorldY = chunk.minWorldY + groundY;

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
            const float spawnThreshold = std::clamp(spawnThresholdBase * std::max(biome.treeDensityMultiplier, 0.0f), 0.0f, 1.0f);
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

            const int canopyTopLocal = groundLocalY + trunkHeight;
            if (canopyTopLocal >= kChunkSizeY)
            {
                continue;
            }

            for (int dy = 0; dy < trunkHeight; ++dy)
            {
                trySetBlock(worldX, groundWorldY + dy, worldZ, BlockId::Wood, true);
            }

            const int canopyBaseLocal = groundLocalY + trunkHeight - 3;
            for (int y = canopyBaseLocal; y <= canopyTopLocal; ++y)
            {
                const int worldY = chunk.minWorldY + y;
                const int layer = y - canopyBaseLocal;
                int radius = 2;
                if (y >= canopyTopLocal - 1)
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

                        if (dx == 0 && dz == 0 && y <= groundLocalY + trunkHeight - 1)
                        {
                            continue;
                        }

                        if (layer == 0 && std::abs(dx) + std::abs(dz) > 3)
                        {
                            continue;
                        }

                        trySetBlock(worldX + dx, worldY, worldZ + dz, BlockId::Leaves, false);

                    }
                }
            }
        }
    }

    if (!anySolid)
    {
        columnManager_.updateChunk(chunk);
        return;
    }

    columnManager_.updateChunk(chunk);
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

BlockId ChunkManager::blockAt(const glm::ivec3& worldPos) const noexcept
{
    return impl_->blockAt(worldPos);
}

glm::vec3 ChunkManager::findSafeSpawnPosition(float worldX, float worldZ) const
{
    return impl_->findSafeSpawnPosition(worldX, worldZ);
}

