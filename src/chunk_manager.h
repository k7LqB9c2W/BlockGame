#pragma once
// chunk_manager.h
// Declares the chunk streaming, terrain meshing, and GPU upload subsystem used by BlockGame.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <glad/glad.h>
#include <glm/glm.hpp>

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
inline constexpr int kDefaultViewDistance = 4;
inline constexpr int kExtendedViewDistance = 12;
struct VerticalStreamingConfig
{
    int minRadiusChunks{2};
    int maxRadiusChunks{8};
    int columnSlackChunks{1};
    int sampleRadiusChunks{3};
    int horizontalEvictionSlack{1};
    int uploadBasePerColumn{2};
    int uploadRampDivisor{2};
    int uploadMaxPerColumn{6};
    int maxGenerationJobsPerColumn{3};
    int backlogColumnCapReleaseThreshold{96};

    struct GenerationBudgetSettings
    {
        int baseJobsPerFrame{18};
        float jobsPerHorizontalRing{1.5f};
        float jobsPerVerticalLayer{1.0f};
        int backlogStartThreshold{12};
        int backlogStepSize{16};
        int backlogBoostPerStep{6};
        int maxJobsPerFrame{96};
        int minRingExpansionsPerFrame{1};
        int maxRingExpansionsPerFrame{4};
        int backlogRingStepSize{24};
        int columnCapBoostPerStep{1};
    } generationBudget{};

    int maxWorkerThreads{0};
};

inline constexpr VerticalStreamingConfig kVerticalStreamingConfig{};
inline constexpr std::size_t kUploadBudgetBytesPerFrame = 6ull * 1024ull * 1024ull;

inline constexpr std::size_t kMinBufferSizeBytes = 4ull * 1024ull;
inline constexpr std::size_t kChunkPoolSoftCap = 512ull;
inline constexpr std::size_t kUploadQueueScanLimit = 128ull;
inline constexpr int kBiomeSizeInChunks = 30; // Controls the width/height of each biome in chunks.

float computeFarPlaneForViewDistance(int viewDistance) noexcept;
extern float kFarPlane;

enum class BlockId : std::uint8_t
{
    Air = 0,
    Grass = 1,
    Wood = 2,
    Leaves = 3,
    Sand = 4,
    Water = 5,
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

struct ChunkShaderUniformLocations
{
    GLint uViewProj{-1};
    GLint uLightDir{-1};
    GLint uCameraPos{-1};
    GLint uAtlas{-1};
    GLint uHighlightedBlock{-1};
    GLint uHasHighlight{-1};
};

struct ChunkProfilingSnapshot
{
    double averageGenerationMs{0.0};
    double averageMeshingMs{0.0};
    std::size_t uploadedBytes{0};
    int generatedChunks{0};
    int meshedChunks{0};
    int uploadedChunks{0};
    int throttledUploads{0};
    int deferredUploads{0};
    int evictedChunks{0};
    int verticalRadius{0};
    int generationBudget{0};
    int generationJobsIssued{0};
    int ringExpansionBudget{0};
    int ringExpansionsUsed{0};
    int missingChunks{0};
    int generationColumnCap{0};
    int generationBacklogSteps{0};
    int workerThreads{0};
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

    ChunkProfilingSnapshot sampleProfilingSnapshot();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

