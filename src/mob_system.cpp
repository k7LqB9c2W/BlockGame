// mob_system.cpp
// Implements the temporary static-mob runtime used to spawn debug mobs before AI and gameplay systems exist.

#include "mob_system.h"

#include <glm/gtc/matrix_transform.hpp>

#include <array>

namespace
{
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

    instances_.push_back(MobInstance{model, worldPosition, yawRadians});
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
