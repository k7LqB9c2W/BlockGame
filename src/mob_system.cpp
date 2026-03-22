// mob_system.cpp
// Implements the lightweight passive-mob runtime, including chunk-radius despawn, simple locomotion, and per-part walk animation.

#include "mob_system.h"
#include "terrain/terrain_generator.h"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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
    float colliderWidth{0.6f};
    float colliderHeight{1.0f};
    float jumpVelocity{6.5f};
    float jumpCooldownSeconds{0.3f};
    float gravity{kGravity};
    float terminalVelocity{kTerminalVelocity};
    float autoJumpHeight{1.15f};
    int wanderCandidateAttempts{6};
};

struct AABB
{
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
};

struct AxisMoveResult
{
    float actualMove{0.0f};
    bool collided{false};
};

constexpr float kWalkCycleRadiansPerBlock = glm::two_pi<float>() * 1.2f;
constexpr float kWalkCycleDampPerSecond = 8.0f;
constexpr float kLegSwingAmplitudeRadians = glm::radians(26.0f);

[[nodiscard]] PassiveMobProfile passiveMobProfileFor(std::string_view modelId) noexcept
{
    PassiveMobProfile profile{};
    if (modelId == "chicken")
    {
        profile.walkSpeed = 1.6f;
        profile.wanderMaxDistance = 5.0f;
        profile.colliderWidth = 0.4f;
        profile.colliderHeight = 0.9f;
        profile.jumpVelocity = 5.8f;
    }
    else if (modelId == "cow" || modelId == "sheep")
    {
        profile.walkSpeed = 1.15f;
        profile.wanderMaxDistance = 7.0f;
        profile.colliderWidth = 0.8f;
        profile.colliderHeight = 1.4f;
    }
    else if (modelId == "pig")
    {
        profile.walkSpeed = 1.25f;
        profile.colliderWidth = 0.72f;
        profile.colliderHeight = 1.1f;
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

[[nodiscard]] AABB makeMobAABB(const glm::vec3& worldPosition, const PassiveMobProfile& profile) noexcept
{
    const float halfWidth = profile.colliderWidth * 0.5f;
    const glm::vec3 minCorner(worldPosition.x - halfWidth,
                              worldPosition.y,
                              worldPosition.z - halfWidth);
    return AABB{minCorner, minCorner + glm::vec3(profile.colliderWidth,
                                                 profile.colliderHeight,
                                                 profile.colliderWidth)};
}

[[nodiscard]] bool overlaps1D(float minA, float maxA, float minB, float maxB) noexcept
{
    return (minA < maxB - kAxisCollisionEpsilon) && (maxA > minB + kAxisCollisionEpsilon);
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

[[nodiscard]] float legSwingAngleRadians(const MobSystem::MobInstance& instance,
                                         MobPartAnimationRole role) noexcept
{
    const float phaseValue = std::sin(instance.walkCyclePhaseRadians) *
                             kLegSwingAmplitudeRadians *
                             std::clamp(instance.walkCycleStrength, 0.0f, 1.0f);
    switch (role)
    {
    case MobPartAnimationRole::FrontLeftLeg:
    case MobPartAnimationRole::BackRightLeg:
        return phaseValue;
    case MobPartAnimationRole::FrontRightLeg:
    case MobPartAnimationRole::BackLeftLeg:
        return -phaseValue;
    case MobPartAnimationRole::Static:
    default:
        return 0.0f;
    }
}

[[nodiscard]] glm::mat4 modelPartTransform(const MobSystem::MobInstance& instance,
                                           const MobModelPart& part) noexcept
{
    const float legSwing = legSwingAngleRadians(instance, part.animationRole);
    if (std::abs(legSwing) <= 1e-5f)
    {
        return glm::mat4(1.0f);
    }

    glm::mat4 transform(1.0f);
    transform = glm::translate(transform, part.pivot);
    transform = glm::rotate(transform, legSwing, glm::vec3(1.0f, 0.0f, 0.0f));
    transform = glm::translate(transform, -part.pivot);
    return transform;
}

[[nodiscard]] AxisMoveResult sweepMobAABB(AABB& box,
                                          glm::vec3& position,
                                          float move,
                                          int axis,
                                          const ChunkManager& chunkManager)
{
    AxisMoveResult result{move, false};
    if (std::abs(move) <= kAxisCollisionEpsilon)
    {
        if (move != 0.0f)
        {
            position[axis] += move;
            box.min[axis] += move;
            box.max[axis] += move;
        }
        return result;
    }

    const int otherAxis0 = (axis + 1) % 3;
    const int otherAxis1 = (axis + 2) % 3;
    const float minOther0 = box.min[otherAxis0];
    const float maxOther0 = box.max[otherAxis0];
    const float minOther1 = box.min[otherAxis1];
    const float maxOther1 = box.max[otherAxis1];

    int other0Min = static_cast<int>(std::floor(minOther0));
    int other0Max = static_cast<int>(std::floor(maxOther0));
    if (other0Max < other0Min)
    {
        other0Max = other0Min;
    }

    int other1Min = static_cast<int>(std::floor(minOther1));
    int other1Max = static_cast<int>(std::floor(maxOther1));
    if (other1Max < other1Min)
    {
        other1Max = other1Min;
    }

    auto layerHasCollision = [&](int primaryIndex) -> bool
    {
        for (int idx0 = other0Min; idx0 <= other0Max; ++idx0)
        {
            const float blockMin0 = static_cast<float>(idx0);
            const float blockMax0 = blockMin0 + 1.0f;
            if (!overlaps1D(minOther0, maxOther0, blockMin0, blockMax0))
            {
                continue;
            }

            for (int idx1 = other1Min; idx1 <= other1Max; ++idx1)
            {
                const float blockMin1 = static_cast<float>(idx1);
                const float blockMax1 = blockMin1 + 1.0f;
                if (!overlaps1D(minOther1, maxOther1, blockMin1, blockMax1))
                {
                    continue;
                }

                glm::ivec3 blockCoord(0);
                blockCoord[axis] = primaryIndex;
                blockCoord[otherAxis0] = idx0;
                blockCoord[otherAxis1] = idx1;
                if (isSolid(chunkManager.blockAt(blockCoord)))
                {
                    return true;
                }
            }
        }
        return false;
    };

    float allowed = move;
    if (move > 0.0f)
    {
        const float face = box.max[axis];
        const int firstBlock = static_cast<int>(std::floor(face - kAxisCollisionEpsilon)) + 1;
        const int lastBlock = static_cast<int>(std::floor(face + move + kAxisCollisionEpsilon));
        if (firstBlock <= lastBlock)
        {
            for (int primary = firstBlock; primary <= lastBlock; ++primary)
            {
                const float blockMin = static_cast<float>(primary);
                const float distance = blockMin - face;
                if (distance > allowed + kAxisCollisionEpsilon)
                {
                    break;
                }

                if (layerHasCollision(primary))
                {
                    allowed = std::min(allowed, std::max(distance - kAxisCollisionEpsilon, 0.0f));
                    result.collided = true;
                    break;
                }
            }
        }
        allowed = std::clamp(allowed, 0.0f, move);
    }
    else
    {
        const float face = box.min[axis];
        const int firstBlock = static_cast<int>(std::floor(face - kAxisCollisionEpsilon));
        const int lastBlock = static_cast<int>(std::floor(face + move - kAxisCollisionEpsilon));
        if (firstBlock >= lastBlock)
        {
            for (int primary = firstBlock; primary >= lastBlock; --primary)
            {
                const float blockMax = static_cast<float>(primary + 1);
                const float distance = blockMax - face;
                if (distance < allowed - kAxisCollisionEpsilon)
                {
                    break;
                }

                if (layerHasCollision(primary))
                {
                    allowed = std::max(allowed, std::min(distance + kAxisCollisionEpsilon, 0.0f));
                    result.collided = true;
                    break;
                }
            }
        }
        allowed = std::clamp(allowed, move, 0.0f);
    }

    position[axis] += allowed;
    box.min[axis] += allowed;
    box.max[axis] += allowed;
    result.actualMove = allowed;
    return result;
}

[[nodiscard]] bool intersectsSolidBlocks(const AABB& box, const ChunkManager& chunkManager)
{
    const int minX = static_cast<int>(std::floor(box.min.x));
    const int maxX = static_cast<int>(std::floor(box.max.x));
    const int minY = static_cast<int>(std::floor(box.min.y));
    const int maxY = static_cast<int>(std::floor(box.max.y));
    const int minZ = static_cast<int>(std::floor(box.min.z));
    const int maxZ = static_cast<int>(std::floor(box.max.z));

    for (int x = minX; x <= maxX; ++x)
    {
        const float blockMinX = static_cast<float>(x);
        const float blockMaxX = blockMinX + 1.0f;
        if (!overlaps1D(box.min.x, box.max.x, blockMinX, blockMaxX))
        {
            continue;
        }

        for (int y = minY; y <= maxY; ++y)
        {
            const float blockMinY = static_cast<float>(y);
            const float blockMaxY = blockMinY + 1.0f;
            if (!overlaps1D(box.min.y, box.max.y, blockMinY, blockMaxY))
            {
                continue;
            }

            for (int z = minZ; z <= maxZ; ++z)
            {
                const float blockMinZ = static_cast<float>(z);
                const float blockMaxZ = blockMinZ + 1.0f;
                if (!overlaps1D(box.min.z, box.max.z, blockMinZ, blockMaxZ))
                {
                    continue;
                }

                if (isSolid(chunkManager.blockAt(glm::ivec3(x, y, z))))
                {
                    return true;
                }
            }
        }
    }

    return false;
}

[[nodiscard]] float highestSurfaceUnderFootprint(const glm::vec3& worldPosition,
                                                 const PassiveMobProfile& profile,
                                                 const ChunkManager& chunkManager)
{
    const float halfWidth = profile.colliderWidth * 0.5f;
    const std::array<glm::vec2, 4> sampleOffsets = {
        glm::vec2{-halfWidth, -halfWidth},
        glm::vec2{halfWidth, -halfWidth},
        glm::vec2{-halfWidth, halfWidth},
        glm::vec2{halfWidth, halfWidth}
    };

    float highestSurface = -std::numeric_limits<float>::infinity();
    for (const glm::vec2& offset : sampleOffsets)
    {
        highestSurface = std::max(highestSurface,
                                  chunkManager.surfaceHeight(worldPosition.x + offset.x,
                                                             worldPosition.z + offset.y));
    }
    return highestSurface;
}

void applyMobGroundSnap(MobSystem::MobInstance& instance,
                        const PassiveMobProfile& profile,
                        const ChunkManager& chunkManager)
{
    const float desiredY = highestSurfaceUnderFootprint(instance.worldPosition, profile, chunkManager);
    if (desiredY > -std::numeric_limits<float>::infinity() &&
        desiredY <= instance.worldPosition.y + kGroundSnapTolerance &&
        instance.verticalVelocity <= 0.0f)
    {
        instance.worldPosition.y = desiredY;
        instance.verticalVelocity = 0.0f;
        instance.onGround = true;
    }
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
                                    const PassiveMobProfile& profile,
                                    float& outGroundY)
{
    outGroundY = highestSurfaceUnderFootprint(candidatePosition, profile, chunkManager);
    if (std::abs(outGroundY - currentPosition.y) > profile.maxStepHeight)
    {
        return false;
    }

    glm::vec3 standingPosition = candidatePosition;
    standingPosition.y = outGroundY;
    return !intersectsSolidBlocks(makeMobAABB(standingPosition, profile), chunkManager);
}

[[nodiscard]] bool canAutoJumpLedge(const MobSystem::MobInstance& instance,
                                    const ChunkManager& chunkManager,
                                    const PassiveMobProfile& profile,
                                    const glm::vec2& moveDirection)
{
    if (glm::dot(moveDirection, moveDirection) <= 1e-6f)
    {
        return false;
    }

    const float currentGroundY = highestSurfaceUnderFootprint(instance.worldPosition, profile, chunkManager);
    if (!std::isfinite(currentGroundY))
    {
        return false;
    }

    const float forwardProbeDistance = profile.colliderWidth * 0.5f + 0.2f;
    glm::vec3 stepProbePosition = instance.worldPosition;
    stepProbePosition.x += moveDirection.x * forwardProbeDistance;
    stepProbePosition.z += moveDirection.y * forwardProbeDistance;

    const float nextGroundY = highestSurfaceUnderFootprint(stepProbePosition, profile, chunkManager);
    const float climbHeight = nextGroundY - currentGroundY;
    if (climbHeight <= 0.05f || climbHeight > profile.autoJumpHeight)
    {
        return false;
    }

    glm::vec3 landingPosition = instance.worldPosition;
    landingPosition.x += moveDirection.x * (profile.colliderWidth * 0.5f + 0.55f);
    landingPosition.z += moveDirection.y * (profile.colliderWidth * 0.5f + 0.55f);
    landingPosition.y = nextGroundY;
    return !intersectsSolidBlocks(makeMobAABB(landingPosition, profile), chunkManager);
}

void simulateMobMotion(MobSystem::MobInstance& instance,
                       const ChunkManager& chunkManager,
                       const PassiveMobProfile& profile,
                       const glm::vec2& desiredHorizontalVelocity,
                       float deltaSeconds)
{
    const glm::vec2 previousHorizontalPosition(instance.worldPosition.x, instance.worldPosition.z);
    AABB box = makeMobAABB(instance.worldPosition, profile);
    const glm::vec3 desiredHorizontalMove(desiredHorizontalVelocity.x * deltaSeconds,
                                          0.0f,
                                          desiredHorizontalVelocity.y * deltaSeconds);

    const AxisMoveResult moveX = sweepMobAABB(box, instance.worldPosition, desiredHorizontalMove.x, 0, chunkManager);
    const AxisMoveResult moveZ = sweepMobAABB(box, instance.worldPosition, desiredHorizontalMove.z, 2, chunkManager);
    const bool horizontalBlocked =
        (std::abs(moveX.actualMove - desiredHorizontalMove.x) > kAxisCollisionEpsilon) ||
        (std::abs(moveZ.actualMove - desiredHorizontalMove.z) > kAxisCollisionEpsilon);

    if (horizontalBlocked &&
        instance.onGround &&
        instance.jumpCooldownSeconds <= 0.0f &&
        glm::dot(desiredHorizontalVelocity, desiredHorizontalVelocity) > 1e-6f)
    {
        const glm::vec2 moveDirection = glm::normalize(desiredHorizontalVelocity);
        if (canAutoJumpLedge(instance, chunkManager, profile, moveDirection))
        {
            instance.verticalVelocity = profile.jumpVelocity;
            instance.onGround = false;
            instance.jumpCooldownSeconds = profile.jumpCooldownSeconds;
        }
    }

    instance.verticalVelocity += profile.gravity * deltaSeconds;
    if (instance.verticalVelocity < profile.terminalVelocity)
    {
        instance.verticalVelocity = profile.terminalVelocity;
    }

    bool groundedThisStep = false;
    const float desiredVerticalMove = instance.verticalVelocity * deltaSeconds;
    const AxisMoveResult moveY = sweepMobAABB(box, instance.worldPosition, desiredVerticalMove, 1, chunkManager);
    if (std::abs(moveY.actualMove - desiredVerticalMove) > kAxisCollisionEpsilon)
    {
        if (desiredVerticalMove < 0.0f && moveY.actualMove > desiredVerticalMove)
        {
            groundedThisStep = true;
        }
        instance.verticalVelocity = 0.0f;
    }

    instance.onGround = groundedThisStep;
    if (instance.onGround)
    {
        applyMobGroundSnap(instance, profile, chunkManager);
    }

    const glm::vec2 currentHorizontalPosition(instance.worldPosition.x, instance.worldPosition.z);
    const float horizontalDistanceMoved = glm::length(currentHorizontalPosition - previousHorizontalPosition);
    if (horizontalDistanceMoved > 1e-4f)
    {
        instance.walkCyclePhaseRadians = std::fmod(instance.walkCyclePhaseRadians +
                                                   horizontalDistanceMoved * kWalkCycleRadiansPerBlock,
                                                   glm::two_pi<float>());
        const float expectedDistance = std::max(glm::length(desiredHorizontalVelocity) * deltaSeconds, 1e-4f);
        instance.walkCycleStrength = std::clamp(horizontalDistanceMoved / expectedDistance, 0.0f, 1.0f);
    }
    else
    {
        instance.walkCycleStrength = std::max(instance.walkCycleStrength - deltaSeconds * kWalkCycleDampPerSecond,
                                              0.0f);
    }
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
                              profile,
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
    const float verticalDelta = std::abs(instance.targetWorldPosition.y - instance.worldPosition.y);
    if (distanceSquared <= profile.arrivalDistance * profile.arrivalDistance && verticalDelta <= profile.maxStepHeight)
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
    instance.yawRadians = yawFromDirection(direction);
    simulateMobMotion(instance,
                      chunkManager,
                      profile,
                      direction * profile.walkSpeed,
                      deltaSeconds);

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
    instance.verticalVelocity = 0.0f;
    instance.jumpCooldownSeconds = 0.0f;
    instance.walkCyclePhaseRadians = 0.0f;
    instance.walkCycleStrength = 0.0f;
    instance.onGround = true;
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
        instance.jumpCooldownSeconds = std::max(instance.jumpCooldownSeconds - deltaSeconds, 0.0f);

        if (instance.state == PassiveState::Idle)
        {
            if (instance.stateTimerSeconds <= 0.0f)
            {
                const bool choseTarget = chooseWanderTarget(instance, chunkManager, rng, profile);
                (void)choseTarget;
            }

            simulateMobMotion(instance,
                              chunkManager,
                              profile,
                              glm::vec2(0.0f),
                              deltaSeconds);
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
        batch.vertices.resize(instance.model->vertices.size());
        batch.indices = instance.model->indices;

        const glm::vec4 baseColor = binding.hasTexture ? glm::vec4(1.0f) : instance.model->fallbackColor;
        if (instance.model->parts.empty())
        {
            const glm::mat3 normalMatrix(transform);
            for (std::size_t vertexIndex = 0; vertexIndex < instance.model->vertices.size(); ++vertexIndex)
            {
                const MobVertex& sourceVertex = instance.model->vertices[vertexIndex];
                MobVertex vertex = sourceVertex;
                vertex.position = glm::vec3(transform * glm::vec4(sourceVertex.position, 1.0f));
                vertex.normal = glm::normalize(normalMatrix * sourceVertex.normal);
                vertex.color = baseColor;
                batch.vertices[vertexIndex] = vertex;
            }
        }
        else
        {
            for (const MobModelPart& part : instance.model->parts)
            {
                const glm::mat4 combinedTransform = transform * modelPartTransform(instance, part);
                const glm::mat3 normalMatrix(combinedTransform);
                const std::size_t vertexEnd = std::min(part.vertexOffset + part.vertexCount,
                                                       instance.model->vertices.size());
                for (std::size_t vertexIndex = part.vertexOffset; vertexIndex < vertexEnd; ++vertexIndex)
                {
                    const MobVertex& sourceVertex = instance.model->vertices[vertexIndex];
                    MobVertex vertex = sourceVertex;
                    vertex.position = glm::vec3(combinedTransform * glm::vec4(sourceVertex.position, 1.0f));
                    vertex.normal = glm::normalize(normalMatrix * sourceVertex.normal);
                    vertex.color = baseColor;
                    batch.vertices[vertexIndex] = vertex;
                }
            }
        }

        renderData.mobBatches.push_back(std::move(batch));
    }
}
