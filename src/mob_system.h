// mob_system.h
// Declares the lightweight passive-mob runtime that owns spawned mob instances, simple wander AI, and animated render-batch emission.

#pragma once

#include "mob_model.h"

#include <functional>
#include <string_view>
#include <vector>

struct MobTextureBinding
{
    D3D12_GPU_DESCRIPTOR_HANDLE srv{};
    bool hasTexture{false};
};

class MobSystem
{
public:
    enum class PassiveState
    {
        Idle,
        Walk
    };

    struct MobInstance
    {
        const MobModel* model{nullptr};
        glm::vec3 worldPosition{0.0f};
        float yawRadians{0.0f};
        float desiredYawRadians{0.0f};
        PassiveState state{PassiveState::Idle};
        float stateTimerSeconds{0.0f};
        glm::vec3 targetWorldPosition{0.0f};
        float verticalVelocity{0.0f};
        float jumpCooldownSeconds{0.0f};
        float walkCyclePhaseRadians{0.0f};
        float walkCycleStrength{0.0f};
        float headYawRadians{0.0f};
        float headPitchRadians{0.0f};
        float desiredHeadYawRadians{0.0f};
        float desiredHeadPitchRadians{0.0f};
        float headLookTimerSeconds{0.0f};
        bool onGround{true};
    };

    bool loadDefinitions(const std::filesystem::path& directory);
    [[nodiscard]] const MobModel* findModel(std::string_view id) const noexcept;
    [[nodiscard]] std::vector<const MobModel*> allModels() const;
    [[nodiscard]] std::size_t definitionCount() const noexcept;

    bool spawn(std::string_view id, const glm::vec3& worldPosition, float yawRadians = 0.0f);
    void clearInstances() noexcept;
    [[nodiscard]] std::size_t instanceCount() const noexcept;
    void update(const glm::vec3& playerWorldPosition, const ChunkManager& chunkManager, float deltaSeconds);

    void appendRenderBatches(WorldRenderData& renderData,
                             const Frustum& frustum,
                             const std::function<MobTextureBinding(const MobModel&)>& resolveTexture) const;

private:
    MobModelLibrary library_;
    std::vector<MobInstance> instances_;
};
