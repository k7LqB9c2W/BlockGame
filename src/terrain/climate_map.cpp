#include "terrain/climate_map.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <glm/common.hpp>
#include <glm/geometric.hpp>

namespace terrain
{
namespace
{
unsigned hashCombine(unsigned a, unsigned b) noexcept
{
    a ^= b + 0x9E3779B9u + (a << 6) + (a >> 2);
    return a;
}

float hashToUnitFloat(int x, int y, int z) noexcept
{
    std::uint32_t h = static_cast<std::uint32_t>(x);
    h ^= static_cast<std::uint32_t>(y) * 374761393u;
    h ^= static_cast<std::uint32_t>(z) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
}

std::array<float, 2> axisInterpolationWeights(float t, BiomeDefinition::InterpolationCurve curve) noexcept
{
    t = std::clamp(t, 0.0f, 1.0f);
    switch (curve)
    {
        case BiomeDefinition::InterpolationCurve::Step:
            return t < 0.5f ? std::array<float, 2>{1.0f, 0.0f} : std::array<float, 2>{0.0f, 1.0f};
        case BiomeDefinition::InterpolationCurve::Linear:
            return {1.0f - t, t};
        case BiomeDefinition::InterpolationCurve::Square:
        {
            if (t < 0.5f)
            {
                const float tsqr = 2.0f * t * t;
                return {1.0f - tsqr, tsqr};
            }
            const float inv = 1.0f - t;
            const float tsqr = 2.0f * inv * inv;
            return {tsqr, 1.0f - tsqr};
        }
    }
    return {1.0f - t, t};
}

} // namespace

ClimateFragment::ClimateFragment(const glm::ivec2& fragmentCoord) noexcept
    : fragmentCoord_(fragmentCoord),
      baseWorld_(fragmentCoord * kSize)
{
}

const ClimateSample& ClimateFragment::sample(int localX, int localZ) const noexcept
{
    const int clampedX = std::clamp(localX, 0, kSize - 1);
    const int clampedZ = std::clamp(localZ, 0, kSize - 1);
    const std::size_t index = static_cast<std::size_t>(clampedZ) * kSize + static_cast<std::size_t>(clampedX);
    return samples_[index];
}

ClimateSample& ClimateFragment::sample(int localX, int localZ) noexcept
{
    const int clampedX = std::clamp(localX, 0, kSize - 1);
    const int clampedZ = std::clamp(localZ, 0, kSize - 1);
    const std::size_t index = static_cast<std::size_t>(clampedZ) * kSize + static_cast<std::size_t>(clampedX);
    return samples_[index];
}

NoiseVoronoiClimateGenerator::NoiseVoronoiClimateGenerator(const BiomeDatabase& database,
                                                           const WorldgenProfile& profile,
                                                           unsigned seed,
                                                           int chunkSize,
                                                           int biomeSizeInChunks)
    : biomeDatabase_(database),
      profile_(profile),
      chunkSize_(std::max(chunkSize, 1)),
      biomeSizeInChunks_(std::max(biomeSizeInChunks, 1)),
      cellSize_(static_cast<float>(chunkSize_ * biomeSizeInChunks_)),
      baseSeed_(seed)
{
}

int NoiseVoronoiClimateGenerator::floorDiv(int value, int divisor) noexcept
{
    int quotient = value / divisor;
    int remainder = value % divisor;
    if ((remainder != 0) && ((remainder < 0) != (divisor < 0)))
    {
        --quotient;
    }
    return quotient;
}

std::array<float, 2> NoiseVoronoiClimateGenerator::axisInterpolationWeights(
    float t,
    BiomeDefinition::InterpolationCurve curve) noexcept
{
    return ::terrain::axisInterpolationWeights(t, curve);
}

const BiomeDefinition& NoiseVoronoiClimateGenerator::biomeForCell(int cellX, int cellZ) const
{
    const float selector = hashToUnitFloat(cellX, 31, cellZ);
    const std::size_t biomeCount = biomeDatabase_.biomeCount();
    if (biomeCount == 0)
    {
        throw std::runtime_error("Biome database is empty");
    }
    const std::size_t maxIndex = biomeCount - 1;
    const std::size_t biomeIndex =
        std::min(static_cast<std::size_t>(selector * static_cast<float>(biomeCount)), maxIndex);
    return biomeDatabase_.definitionByIndex(biomeIndex);
}

float NoiseVoronoiClimateGenerator::sampleBaseHeight(const BiomeDefinition& definition,
                                                     int cellX,
                                                     int cellZ) const noexcept
{
    const float minHeight = static_cast<float>(definition.minHeight);
    const float maxHeight = static_cast<float>(definition.maxHeight);
    const float center = 0.5f * (minHeight + maxHeight);
    const float range = maxHeight - minHeight;
    unsigned seed = baseSeed_;
    seed = hashCombine(seed, static_cast<unsigned>(cellX * 73856093));
    seed = hashCombine(seed, static_cast<unsigned>(cellZ * 19349663));
    seed = hashCombine(seed, static_cast<unsigned>(std::hash<std::string>{}(definition.id) & 0xFFFFFFFFu));
    const float noise = hashToUnitFloat(static_cast<int>(seed), 17, 91) * 2.0f - 1.0f;
    const float variation = range * 0.35f;
    return std::clamp(center + noise * variation, minHeight, maxHeight);
}

void NoiseVoronoiClimateGenerator::populateBlends(int worldX, int worldZ, ClimateSample& outSample) const
{
    outSample = ClimateSample{};

    const float cellSize = cellSize_;
    const float invCellSize = 1.0f / cellSize;
    const float fx = (static_cast<float>(worldX) + 0.5f) * invCellSize;
    const float fz = (static_cast<float>(worldZ) + 0.5f) * invCellSize;

    const int baseCellX = static_cast<int>(std::floor(fx));
    const int baseCellZ = static_cast<int>(std::floor(fz));
    const float relX = fx - static_cast<float>(baseCellX);
    const float relZ = fz - static_cast<float>(baseCellZ);

    std::array<BiomeBlend, 4> localBlends{};
    std::array<float, 4> rawWeights{};
    std::array<glm::vec2, 4> centers{};
    std::size_t blendCount = 0;
    float weightSum = 0.0f;

    for (int dz = 0; dz < 2; ++dz)
    {
        for (int dx = 0; dx < 2; ++dx)
        {
            const int cellX = baseCellX + dx;
            const int cellZ = baseCellZ + dz;
            const BiomeDefinition& biome = biomeForCell(cellX, cellZ);

            const auto wx = axisInterpolationWeights(relX, biome.interpolationCurve);
            const auto wz = axisInterpolationWeights(relZ, biome.interpolationCurve);
            float weight = wx[dx] * wz[dz];
            weight *= std::max(biome.interpolationWeight, 1e-3f);
            if (weight <= std::numeric_limits<float>::epsilon())
            {
                continue;
            }

            BiomeBlend blend{};
            blend.biome = &biome;
            blend.weight = weight;
            blend.height = sampleBaseHeight(biome, cellX, cellZ);
            blend.roughness = biome.roughness;
            blend.hills = biome.hills;
            blend.mountains = biome.mountains;

            const glm::vec2 cellCenter = glm::vec2((static_cast<float>(cellX) + 0.5f) * cellSize,
                                                   (static_cast<float>(cellZ) + 0.5f) * cellSize);
            const glm::vec2 worldPos = glm::vec2(static_cast<float>(worldX) + 0.5f,
                                                 static_cast<float>(worldZ) + 0.5f);
            const glm::vec2 offset = worldPos - cellCenter;
            const float normalizedDistance =
                glm::length(offset) / std::max(cellSize * 0.5f, 1.0f);
            blend.normalizedDistance = normalizedDistance;
            blend.seed = hashCombine(baseSeed_, static_cast<unsigned>(cellX * 73856093));

            localBlends[blendCount] = blend;
            rawWeights[blendCount] = weight;
            centers[blendCount] = cellCenter;
            weightSum += weight;
            ++blendCount;
        }
    }

    if (blendCount == 0)
    {
        outSample.blendCount = 0;
        outSample.aggregatedHeight = 0.0f;
        return;
    }

    for (std::size_t i = 0; i < blendCount; ++i)
    {
        localBlends[i].weight = rawWeights[i] / weightSum;
    }

    for (std::size_t i = 0; i < blendCount; ++i)
    {
        std::size_t best = i;
        for (std::size_t j = i + 1; j < blendCount; ++j)
        {
            if (localBlends[j].weight > localBlends[best].weight)
            {
                best = j;
            }
        }
        if (best != i)
        {
            std::swap(localBlends[i], localBlends[best]);
            std::swap(centers[i], centers[best]);
        }
    }

    outSample.blendCount = blendCount;
    for (std::size_t i = 0; i < blendCount; ++i)
    {
        outSample.blends[i] = localBlends[i];
    }

    float aggregatedHeight = 0.0f;
    float aggregatedRoughness = 0.0f;
    float aggregatedHills = 0.0f;
    float aggregatedMountains = 0.0f;
    float keepOriginal = 0.0f;

    for (std::size_t i = 0; i < blendCount; ++i)
    {
        const BiomeBlend& blend = localBlends[i];
        aggregatedHeight += blend.height * blend.weight;
        aggregatedRoughness += std::max(blend.roughness, 0.0f) * blend.weight;
        aggregatedHills += std::max(blend.hills, 0.0f) * blend.weight;
        aggregatedMountains += std::max(blend.mountains, 0.0f) * blend.weight;
        if (blend.biome)
        {
            keepOriginal += std::clamp(blend.biome->keepOriginalTerrain, 0.0f, 1.0f) * blend.weight;
        }
    }

    outSample.aggregatedHeight = aggregatedHeight;
    outSample.aggregatedRoughness = aggregatedRoughness;
    outSample.aggregatedHills = aggregatedHills;
    outSample.aggregatedMountains = aggregatedMountains;
    outSample.keepOriginalMix = std::clamp(keepOriginal, 0.0f, 1.0f);

    outSample.dominantSitePos = outSample.blendCount > 0 ? centers[0] : glm::vec2(0.0f);
    outSample.dominantSiteHalfExtents = glm::vec2(cellSize * 0.5f);
}

void NoiseVoronoiClimateGenerator::generate(ClimateFragment& fragment)
{
    const glm::ivec2 baseWorld = fragment.baseWorld();
    for (int localZ = 0; localZ < ClimateFragment::kSize; ++localZ)
    {
        for (int localX = 0; localX < ClimateFragment::kSize; ++localX)
        {
            const int worldX = baseWorld.x + localX;
            const int worldZ = baseWorld.y + localZ;
            ClimateSample& sample = fragment.sample(localX, localZ);
            populateBlends(worldX, worldZ, sample);
        }
    }
}

ClimateMap::ClimateMap(std::unique_ptr<ClimateGenerator> generator, std::size_t maxFragments)
    : generator_(std::move(generator)),
      maxFragments_(std::max<std::size_t>(maxFragments, 1))
{
    if (!generator_)
    {
        throw std::invalid_argument("ClimateMap requires a generator");
    }
}

const ClimateSample& ClimateMap::sample(int worldX, int worldZ) const
{
    const ClimateFragment& fragment = fragmentForColumn(worldX, worldZ);
    const glm::ivec2 baseWorld = fragment.baseWorld();
    const int localX = worldX - baseWorld.x;
    const int localZ = worldZ - baseWorld.y;
    return fragment.sample(localX, localZ);
}

void ClimateMap::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    fragments_.clear();
    lru_.clear();
}

int ClimateMap::floorDiv(int value, int divisor) noexcept
{
    int quotient = value / divisor;
    int remainder = value % divisor;
    if ((remainder != 0) && ((remainder < 0) != (divisor < 0)))
    {
        --quotient;
    }
    return quotient;
}

const ClimateFragment& ClimateMap::fragmentForColumn(int worldX, int worldZ) const
{
    const int fragmentX = floorDiv(worldX, ClimateFragment::kSize);
    const int fragmentZ = floorDiv(worldZ, ClimateFragment::kSize);
    const glm::ivec2 key{fragmentX, fragmentZ};

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = fragments_.find(key);
        if (it != fragments_.end())
        {
            touch(it->second);
            return *it->second.fragment;
        }
    }

    auto fragment = std::make_unique<ClimateFragment>(key);
    generator_->generate(*fragment);

    const ClimateFragment* result = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto [it, inserted] = fragments_.emplace(key, FragmentCacheEntry{std::move(fragment), lru_.end()});
        (void)inserted;
        it->second.lruIt = lru_.emplace(lru_.begin(), key);
        touch(it->second);
        evictIfNeeded();
        result = it->second.fragment.get();
    }

    return *result;
}

void ClimateMap::touch(FragmentCacheEntry& entry) const
{
    if (entry.lruIt != lru_.end())
    {
        lru_.erase(entry.lruIt);
    }
    entry.lruIt = lru_.emplace(lru_.begin(), entry.fragment->fragmentCoord());
}

void ClimateMap::evictIfNeeded() const
{
    while (!lru_.empty() && fragments_.size() > maxFragments_)
    {
        const glm::ivec2 key = lru_.back();
        auto it = fragments_.find(key);
        if (it != fragments_.end())
        {
            fragments_.erase(it);
        }
        lru_.pop_back();
    }
}

} // namespace terrain
