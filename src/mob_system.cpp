// mob_system.cpp
// Implements the lightweight passive-mob runtime, including chunk-radius despawn and a shared idle/wander update loop.

#include "mob_system.h"
#include "terrain/terrain_generator.h"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>

namespace
{
struct PassiveMobProfile
{
    float walkSpeed{1.35f};
    float idleMinSeconds{2.0f};
    float idleMaxSeconds{5.0f};
    float wanderMinDistance{2.0f};
    float wanderMaxDistance{6.0f};
    float arrivalDistance{0.2f};
    float maxStepHeight{1.15f};
    int wanderCandidateAttempts{6};
};

[[nodiscard]] PassiveMobProfile passiveMobProfileFor(std::string_view modelId) noexcept
{
    PassiveMobProfile profile{};
    if (modelId == "chicken")
    {
        profile.walkSpeed = 1.6f;
        profile.wanderMaxDistance = 5.0f;
    }
    else if (modelId == "cow" || modelId == "sheep")
    {
        profile.walkSpeed = 1.15f;
        profile.wanderMaxDistance = 7.0f;
    }
    else if (modelId == "pig")
    {
        profile.walkSpeed = 1.25f;
    }
    return profile;
}

[[nodiscard]] std::array<glm::vec3, 8> aabbCorners(const glm::vec3& minCorner,
                                                   const glm::vec3& maxCorner) noexcept
{
    return {{
        {minCorner.x, minCorner.y, minCorner.z},
        {maxCorner.x, minCorner.y, minCorner.z},
        {maxCorner.x, maxCorner.y, minCorner.z},
        {minCorner.x, maxCorner.y, minCorner.z},
        {minCorner.x, minCorner.y, maxCorner.z},
        {maxCorner.x, minCorner.y, maxCorner.z},
        {maxCorner.x, maxCorner.y, maxCorner.z},
        {minCorner.x, maxCorner.y, maxCorner.z},
    }};
}

[[nodiscard]] void transformedBounds(const MobModel& model,
                                     const glm::mat4& transform,
                                     glm::vec3& outMin,
                                     glm::vec3& outMax)
{
    const std::array<glm::vec3, 8> corners = aabbCorners(model.localBoundsMin, model.localBoundsMax);
    outMin = glm::vec3(transform * glm::vec4(corners.front(), 1.0f));
    outMax = outMin;
    for (const glm::vec3& corner : corners)
    {
        const glm::vec3 worldCorner = glm::vec3(transform * glm::vec4(corner, 1.0f));
        outMin = glm::min(outMin, worldCorner);
        outMax = glm::max(outMax, worldCorner);
    }
}

[[nodiscard]] float randomRange(std::mt19937& rng, float minValue, float maxValue)
{
    std::uniform_real_distribution<float> distribution(minValue, maxValue);
    return distribution(rng);
}

[[nodiscard]] int floorDiv(int value, int divisor) noexcept
{
    const int quotient = value / divisor;
    const int remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0)))
    {
        return quotient - 1;
    }
    return quotient;
}

[[nodiscard]] glm::ivec2 worldToChunkXZ(const glm::vec3& worldPosition) noexcept
{
    return {
        floorDiv(static_cast<int>(std::floor(worldPosition.x)), kChunkSizeX),
        floorDiv(static_cast<int>(std::floor(worldPosition.z)), kChunkSizeZ)
    };
}

[[nodiscard]] int chunkRadiusDistance(const glm::ivec2& a, const glm::ivec2& b) noexcept
{
    return std::max(std::abs(a.x - b.x), std::abs(a.y - b.y));
}

[[nodiscard]] float yawFromDirection(const glm::vec2& direction) noexcept
{
    return std::atan2(-direction.x, -direction.y);
}

void beginIdle(MobSystem::MobInstance& instance,
               std::mt19937& rng,
               const PassiveMobProfile& profile)
{
    instance.state = MobSystem::PassiveState::Idle;
    instance.stateTimerSeconds = randomRange(rng, profile.idleMinSeconds, profile.idleMaxSeconds);
    instance.targetWorldPosition = instance.worldPosition;
}

[[nodiscard]] bool isWalkableTarget(const ChunkManager& chunkManager,
                                    const glm::vec3& currentPosition,
                                    const glm::vec3& candidatePosition,
                                    float maxStepHeight,
                                    float& outGroundY)
{
    const terrain::ColumnSample column = chunkManager.sampleColumnAt(candidatePosition);
    outGroundY = column.surfaceHeight;
    if (std::abs(outGroundY - currentPosition.y) > maxStepHeight)
    {
        return false;
    }

    const glm::ivec3 bodyBlock(static_cast<int>(std::floor(candidatePosition.x)),
                               column.surfaceY + 1,
                               static_cast<int>(std::floor(candidatePosition.z)));
    const glm::ivec3 headBlock(bodyBlock.x, bodyBlock.y + 1, bodyBlock.z);
    if (isSolid(chunkManager.blockAt(bodyBlock)) || isSolid(chunkManager.blockAt(headBlock)))
    {
        return false;
    }

    return true;
}

[[nodiscard]] bool chooseWanderTarget(MobSystem::MobInstance& instance,
                                      const ChunkManager& chunkManager,
                                      std::mt19937& rng,
                                      const PassiveMobProfile& profile)
{
    constexpr float kTwoPi = glm::two_pi<float>();
    for (int attempt = 0; attempt < profile.wanderCandidateAttempts; ++attempt)
    {
        const float angle = randomRange(rng, 0.0f, kTwoPi);
        const float distance = randomRange(rng, profile.wanderMinDistance, profile.wanderMaxDistance);
        const glm::vec2 direction(std::sin(angle), std::cos(angle));
        glm::vec3 candidatePosition = instance.worldPosition;
        candidatePosition.x += direction.x * distance;
        candidatePosition.z += direction.y * distance;

        float groundY = candidatePosition.y;
        if (!isWalkableTarget(chunkManager,
                              instance.worldPosition,
                              candidatePosition,
                              profile.maxStepHeight,
                              groundY))
        {
            continue;
        }

        candidatePosition.y = groundY;
        instance.state = MobSystem::PassiveState::Walk;
        instance.stateTimerSeconds = std::max(distance / std::max(profile.walkSpeed, 0.1f), 0.5f) + 0.75f;
        instance.targetWorldPosition = candidatePosition;
        instance.yawRadians = yawFromDirection(direction);
        return true;
    }

    beginIdle(instance, rng, profile);
    return false;
}

void advanceWalk(MobSystem::MobInstance& instance,
                 const ChunkManager& chunkManager,
                 std::mt19937& rng,
                 const PassiveMobProfile& profile,
                 float deltaSeconds)
{
    const glm::vec2 toTarget(instance.targetWorldPosition.x - instance.worldPosition.x,
                             instance.targetWorldPosition.z - instance.worldPosition.z);
    const float distanceSquared = glm::dot(toTarget, toTarget);
    if (distanceSquared <= profile.arrivalDistance * profile.arrivalDistance)
    {
        instance.worldPosition = instance.targetWorldPosition;
        beginIdle(instance, rng, profile);
        return;
    }

    const float distance = std::sqrt(distanceSquared);
    if (distance <= 1e-4f)
    {
        beginIdle(instance, rng, profile);
        return;
    }

    const glm::vec2 direction = toTarget / distance;
    const float stepDistance = std::min(profile.walkSpeed * deltaSeconds, distance);

    glm::vec3 candidatePosition = instance.worldPosition;
    candidatePosition.x += direction.x * stepDistance;
    candidatePosition.z += direction.y * stepDistance;

    float groundY = candidatePosition.y;
    if (!isWalkableTarget(chunkManager,
                          instance.worldPosition,
                          candidatePosition,
                          profile.maxStepHeight,
                          groundY))
    {
        beginIdle(instance, rng, profile);
        return;
    }

    candidatePosition.y = groundY;
    instance.worldPosition = candidatePosition;
    instance.yawRadians = yawFromDirection(direction);

    if (instance.stateTimerSeconds <= 0.0f)
    {
        beginIdle(instance, rng, profile);
    }
}
} // namespace

bool MobSystem::loadDefinitions(const std::filesystem::path& directory)
{
    instances_.clear();
    return library_.loadDirectory(directory);
}

const MobModel* MobSystem::findModel(std::string_view id) const noexcept
{
    return library_.find(id);
}

std::vector<const MobModel*> MobSystem::allModels() const
{
    return library_.all();
}

std::size_t MobSystem::definitionCount() const noexcept
{
    return library_.size();
}

bool MobSystem::spawn(std::string_view id, const glm::vec3& worldPosition, float yawRadians)
{
    const MobModel* model = library_.find(id);
    if (model == nullptr)
    {
        return false;
    }

    static thread_local std::mt19937 rng(std::random_device{}());
    const PassiveMobProfile profile = passiveMobProfileFor(id);
    MobInstance instance;
    instance.model = model;
    instance.worldPosition = worldPosition;
    instance.yawRadians = yawRadians;
    beginIdle(instance, rng, profile);
    instances_.push_back(instance);
    return true;
}

void MobSystem::clearInstances() noexcept
{
    instances_.clear();
}

std::size_t MobSystem::instanceCount() const noexcept
{
    return instances_.size();
}

void MobSystem::update(const glm::vec3& playerWorldPosition,
                       const ChunkManager& chunkManager,
                       float deltaSeconds)
{
    if (instances_.empty() || deltaSeconds <= 0.0f)
    {
        return;
    }

    static thread_local std::mt19937 rng(std::random_device{}());
    const glm::ivec2 playerChunk = worldToChunkXZ(playerWorldPosition);
    const int despawnRadiusChunks = std::max(chunkManager.exactRenderDistanceChunks(), 0);

    instances_.erase(
        std::remove_if(instances_.begin(),
                       instances_.end(),
                       [&](const MobInstance& instance)
                       {
                           if (instance.model == nullptr)
                           {
                               return true;
                           }

                           const glm::ivec2 mobChunk = worldToChunkXZ(instance.worldPosition);
                           return chunkRadiusDistance(mobChunk, playerChunk) > despawnRadiusChunks;
                       }),
        instances_.end());

    for (MobInstance& instance : instances_)
    {
        if (instance.model == nullptr)
        {
            continue;
        }

        const PassiveMobProfile profile = passiveMobProfileFor(instance.model->id);
        instance.stateTimerSeconds -= deltaSeconds;

        if (instance.state == PassiveState::Idle)
        {
            if (instance.stateTimerSeconds <= 0.0f)
            {
                const bool choseTarget = chooseWanderTarget(instance, chunkManager, rng, profile);
                (void)choseTarget;
            }
        }
        else
        {
            advanceWalk(instance, chunkManager, rng, profile, deltaSeconds);
        }
    }
}

void MobSystem::appendRenderBatches(WorldRenderData& renderData,
                                    const Frustum& frustum,
                                    const std::function<MobTextureBinding(const MobModel&)>& resolveTexture) const
{
    for (const MobInstance& instance : instances_)
    {
        if (instance.model == nullptr || instance.model->empty())
        {
            continue;
        }

        glm::mat4 transform(1.0f);
        transform = glm::translate(transform, instance.worldPosition);
        transform = glm::rotate(transform, instance.yawRadians, glm::vec3(0.0f, 1.0f, 0.0f));

        glm::vec3 worldMin(0.0f);
        glm::vec3 worldMax(0.0f);
        transformedBounds(*instance.model, transform, worldMin, worldMax);
        if (!frustum.intersectsAABB(worldMin, worldMax))
        {
            continue;
        }

        const MobTextureBinding binding = resolveTexture(*instance.model);
        MobRenderBatch batch;
        batch.hasTexture = binding.hasTexture;
        batch.textureSrv = binding.srv;
        batch.vertices.reserve(instance.model->vertices.size());
        batch.indices = instance.model->indices;

        const glm::mat3 normalMatrix(transform);
        const glm::vec4 baseColor = binding.hasTexture ? glm::vec4(1.0f) : instance.model->fallbackColor;
        for (const MobVertex& sourceVertex : instance.model->vertices)
        {
            MobVertex vertex = sourceVertex;
            vertex.position = glm::vec3(transform * glm::vec4(sourceVertex.position, 1.0f));
            vertex.normal = glm::normalize(normalMatrix * sourceVertex.normal);
            vertex.color = baseColor;
            batch.vertices.push_back(vertex);
        }

        renderData.mobBatches.push_back(std::move(batch));
    }
}
