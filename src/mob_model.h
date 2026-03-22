// mob_model.h
// Declares Bedrock-style mob geometry loading and bind-pose mesh baking for BlockGame's first mob-content path.

#pragma once

#include "chunk_manager.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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
