// mob_model.h
// Declares Bedrock-style mob geometry loading, including per-bone baked parts for simple runtime mob animation.

#pragma once

#include "chunk_manager.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

enum class MobPartAnimationRole
{
    Static,
    Head,
    FrontLeftLeg,
    FrontRightLeg,
    BackLeftLeg,
    BackRightLeg
};

struct MobModelPart
{
    std::string name;
    glm::vec3 pivot{0.0f};
    std::size_t vertexOffset{0};
    std::size_t vertexCount{0};
    MobPartAnimationRole animationRole{MobPartAnimationRole::Static};
};

struct MobModel
{
    std::string id;
    std::filesystem::path sourcePath;
    std::filesystem::path texturePath;
    glm::ivec2 textureSize{64, 32};
    glm::vec3 localBoundsMin{0.0f};
    glm::vec3 localBoundsMax{0.0f};
    glm::vec4 fallbackColor{1.0f, 0.72f, 0.78f, 1.0f};
    std::vector<MobVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<MobModelPart> parts;

    [[nodiscard]] bool empty() const noexcept
    {
        return vertices.empty() || indices.empty();
    }
};

class MobModelLibrary
{
public:
    void clear() noexcept;
    bool loadDirectory(const std::filesystem::path& directory);
    [[nodiscard]] const MobModel* find(std::string_view id) const noexcept;
    [[nodiscard]] std::vector<const MobModel*> all() const;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    std::unordered_map<std::string, MobModel> models_;
};
