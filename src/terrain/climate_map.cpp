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
#include <glm/gtc/constants.hpp>

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
      baseSeed_(seed)
{
    (void)chunkSize;
    (void)biomeSizeInChunks;

    chunkSpan_ = std::max(64, static_cast<int>(std::ceil(biomeDatabase_.maxBiomeRadius() * 1.75f)));
    const int alignment = 32;
    chunkSpan_ = std::max(alignment, ((chunkSpan_ + alignment - 1) / alignment) * alignment);
    neighborRadius_ =
        std::max(2, static_cast<int>(std::ceil(biomeDatabase_.maxBiomeRadius() / static_cast<float>(chunkSpan_))) + 1);

    const auto& defs = biomeDatabase_.definitions();
    biomeSelection_.reserve(defs.size());
    biomeWeightPrefix_.reserve(defs.size());
    for (const BiomeDefinition& def : defs)
    {
        if (def.spawnChance <= 0.0f)
        {
            continue;
        }
        const float weight = std::max(def.spawnChance * def.footprintMultiplier, 0.0f);
        if (weight <= 0.0f)
        {
            continue;
        }
        biomeSelection_.push_back(&def);
        totalSpawnWeight_ += weight;
        biomeWeightPrefix_.push_back(totalSpawnWeight_);
    }

    if (biomeSelection_.empty())
    {
        throw std::runtime_error("No suitable biomes for radius-aware climate generation");
    }
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

float NoiseVoronoiClimateGenerator::smoothStep(float t) noexcept
{
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float NoiseVoronoiClimateGenerator::lengthSquared(const glm::ivec2& a, const glm::ivec2& b) noexcept
{
    const glm::ivec2 d = a - b;
    return static_cast<float>(d.x * d.x + d.y * d.y);
}

const NoiseVoronoiClimateGenerator::ChunkSeeds&
NoiseVoronoiClimateGenerator::chunkSeeds(int chunkX, int chunkZ) const
{
    const glm::ivec2 key{chunkX, chunkZ};
    std::lock_guard<std::mutex> lock(chunkMutex_);
    auto it = chunkCache_.find(key);
    if (it != chunkCache_.end())
    {
        return it->second;
    }
    ChunkSeeds seeds = buildChunkSeeds(chunkX, chunkZ);
    auto [insertedIt, inserted] = chunkCache_.emplace(key, std::move(seeds));
    (void)inserted;
    return insertedIt->second;
}

NoiseVoronoiClimateGenerator::ChunkSeeds
NoiseVoronoiClimateGenerator::buildChunkSeeds(int chunkX, int chunkZ) const
{
    ChunkSeeds result{};
    const int baseX = chunkX * chunkSpan_;
    const int baseZ = chunkZ * chunkSpan_;

    std::uint64_t seedValue = baseSeed_;
    seedValue = hashCombine(seedValue, static_cast<unsigned>(chunkX * 73856093));
    seedValue = hashCombine(seedValue, static_cast<unsigned>(chunkZ * 19349663));
    Random rng(seedValue);

    constexpr int kMaxSeedsPerChunk = 48;
    constexpr int kMaxRejections = 96;
    int rejections = 0;

    while (static_cast<int>(result.seeds.size()) < kMaxSeedsPerChunk && rejections < kMaxRejections)
    {
        const int worldX = baseX + rng.nextInt(0, chunkSpan_ - 1);
        const int worldZ = baseZ + rng.nextInt(0, chunkSpan_ - 1);

        BiomeSeed seed = createSeed(rng, worldX, worldZ);
        if (!seed.biome)
        {
            ++rejections;
            continue;
        }

        if (!isValidPlacement(seed.position, seed.radius, result.seeds))
        {
            ++rejections;
            continue;
        }

        result.maxRadius = std::max(result.maxRadius, static_cast<int>(std::ceil(seed.radius)));
        result.seeds.push_back(seed);
        rejections = 0;
    }

    if (result.seeds.empty())
    {
        BiomeSeed fallback = createSeed(rng, baseX + chunkSpan_ / 2, baseZ + chunkSpan_ / 2);
        result.maxRadius = static_cast<int>(std::ceil(fallback.radius));
        result.seeds.push_back(fallback);
    }

    return result;
}

NoiseVoronoiClimateGenerator::BiomeSeed
NoiseVoronoiClimateGenerator::createSeed(Random& rng, int worldX, int worldZ) const
{
    BiomeSeed seed{};
    const BiomeDefinition& biome = chooseBiome(rng);
    seed.biome = &biome;
    const float radius = std::clamp(biome.radius + biome.radiusVariation * rng.nextFloatSigned(),
                                    biome.minRadius(),
                                    biome.maxRadius());
    seed.radius = std::max(radius, 1.0f);
    seed.weight = 1.0f / std::max(seed.radius * std::sqrt(glm::pi<float>()), 1.0f);
    seed.baseHeight = randomizedHeight(rng, biome);
    seed.position = {worldX, worldZ};
    return seed;
}

const BiomeDefinition& NoiseVoronoiClimateGenerator::chooseBiome(Random& rng) const
{
    if (biomeSelection_.empty())
    {
        throw std::runtime_error("Biome selection table is empty");
    }

    const float pick = rng.nextFloat() * totalSpawnWeight_;
    auto it = std::lower_bound(biomeWeightPrefix_.begin(), biomeWeightPrefix_.end(), pick);
    std::size_t index = 0;
    if (it == biomeWeightPrefix_.end())
    {
        index = biomeWeightPrefix_.size() - 1;
    }
    else
    {
        index = static_cast<std::size_t>(std::distance(biomeWeightPrefix_.begin(), it));
    }
    return *biomeSelection_[index];
}

float NoiseVoronoiClimateGenerator::randomizedHeight(Random& rng, const BiomeDefinition& biome) const noexcept
{
    const float minHeight = static_cast<float>(biome.minHeight);
    const float maxHeight = static_cast<float>(biome.maxHeight);
    if (maxHeight <= minHeight)
    {
        return minHeight;
    }
    return glm::mix(minHeight, maxHeight, rng.nextFloat());
}

bool NoiseVoronoiClimateGenerator::isValidPlacement(const glm::ivec2& position,
                                                    float radius,
                                                    const std::vector<BiomeSeed>& seeds) const noexcept
{
    for (const BiomeSeed& other : seeds)
    {
        const float combined = (radius + other.radius) * 0.85f;
        const float distSq = lengthSquared(position, other.position);
        if (distSq < combined * combined)
        {
            return false;
        }
    }
    return true;
}

void NoiseVoronoiClimateGenerator::gatherCandidateSeeds(const glm::ivec2& worldPos,
                                                        std::vector<const BiomeSeed*>& outCandidates) const
{
    const int chunkX = floorDiv(worldPos.x, chunkSpan_);
    const int chunkZ = floorDiv(worldPos.y, chunkSpan_);
    for (int dz = -neighborRadius_; dz <= neighborRadius_; ++dz)
    {
        for (int dx = -neighborRadius_; dx <= neighborRadius_; ++dx)
        {
            const ChunkSeeds& chunk = chunkSeeds(chunkX + dx, chunkZ + dz);
            for (const BiomeSeed& seed : chunk.seeds)
            {
                outCandidates.push_back(&seed);
            }
        }
    }
}

void NoiseVoronoiClimateGenerator::accumulateSample(const glm::ivec2& worldPos, ClimateSample& outSample) const
{
    std::vector<const BiomeSeed*> candidates;
    candidates.reserve(128);
    gatherCandidateSeeds(worldPos, candidates);

    struct WeightedSeed
    {
        const BiomeSeed* seed{nullptr};
        float weight{0.0f};
        float normalizedDistance{0.0f};
        float distance{0.0f};
    };

    std::vector<WeightedSeed> weighted;
    weighted.reserve(candidates.size());

    for (const BiomeSeed* candidate : candidates)
    {
        const float distSq = lengthSquared(worldPos, candidate->position);
        const float distance = std::sqrt(distSq);
        const float normalized = distance / std::max(candidate->radius, 1.0f);
        const float blended = std::clamp(1.0f - normalized, 0.0f, 1.0f);
        float influence = smoothStep(blended);
        if (influence <= std::numeric_limits<float>::epsilon())
        {
            continue;
        }
        weighted.push_back(WeightedSeed{candidate, influence, normalized, distance});
    }

    if (weighted.empty())
    {
        ClimateSample fallback{};
        fallback.blendCount = 1;
        const BiomeDefinition& biome = biomeDatabase_.definitionByIndex(0);
        BiomeBlend blend{};
        blend.biome = &biome;
        blend.weight = 1.0f;
        blend.height = static_cast<float>(biome.minHeight);
        blend.roughness = biome.roughness;
        blend.hills = biome.hills;
        blend.mountains = biome.mountains;
        blend.normalizedDistance = 0.0f;
        blend.seed = hashCombine(baseSeed_, static_cast<unsigned>(biome.minHeight));
        blend.falloff = biome.maxRadius();
        blend.sitePosition = glm::vec2(static_cast<float>(worldPos.x), static_cast<float>(worldPos.y));
        fallback.blends[0] = blend;
        fallback.aggregatedHeight = blend.height;
        fallback.aggregatedRoughness = blend.roughness;
        fallback.aggregatedHills = blend.hills;
        fallback.aggregatedMountains = blend.mountains;
        fallback.keepOriginalMix = std::clamp(biome.keepOriginalTerrain, 0.0f, 1.0f);
        fallback.dominantSitePos = glm::vec2(static_cast<float>(worldPos.x), static_cast<float>(worldPos.y));
        fallback.dominantSiteHalfExtents = glm::vec2(biome.radius);
        outSample = fallback;
        return;
    }

    std::sort(weighted.begin(), weighted.end(), [](const WeightedSeed& a, const WeightedSeed& b) {
        return a.weight > b.weight;
    });

    const std::size_t blendCount = std::min<std::size_t>(weighted.size(), outSample.blends.size());
    float totalWeight = 0.0f;
    for (std::size_t i = 0; i < blendCount; ++i)
    {
        totalWeight += weighted[i].weight;
    }
    if (totalWeight <= std::numeric_limits<float>::epsilon())
    {
        totalWeight = 1.0f;
    }

    outSample = ClimateSample{};
    outSample.blendCount = blendCount;

    float aggregatedHeight = 0.0f;
    float aggregatedRoughness = 0.0f;
    float aggregatedHills = 0.0f;
    float aggregatedMountains = 0.0f;
    float keepOriginal = 0.0f;

    for (std::size_t i = 0; i < blendCount; ++i)
    {
        const WeightedSeed& entry = weighted[i];
        const BiomeDefinition& biome = *entry.seed->biome;
        const float normalizedWeight = entry.weight / totalWeight;

        BiomeBlend blend{};
        blend.biome = &biome;
        blend.weight = normalizedWeight;
        blend.height = entry.seed->baseHeight;
        blend.roughness = biome.roughness;
        blend.hills = biome.hills;
        blend.mountains = biome.mountains;
        blend.normalizedDistance = entry.normalizedDistance;
        blend.falloff = std::max(entry.seed->radius, 1.0f);
        const unsigned seedHash =
            hashCombine(baseSeed_, hashCombine(static_cast<unsigned>(entry.seed->position.x),
                                               static_cast<unsigned>(entry.seed->position.y)));
        blend.seed = seedHash;
        blend.sitePosition = glm::vec2(static_cast<float>(entry.seed->position.x),
                                       static_cast<float>(entry.seed->position.y));

        outSample.blends[i] = blend;

        aggregatedHeight += blend.height * normalizedWeight;
        aggregatedRoughness += blend.roughness * normalizedWeight;
        aggregatedHills += blend.hills * normalizedWeight;
        aggregatedMountains += blend.mountains * normalizedWeight;
        keepOriginal += std::clamp(biome.keepOriginalTerrain, 0.0f, 1.0f) * normalizedWeight;
    }

    outSample.aggregatedHeight = aggregatedHeight;
    outSample.aggregatedRoughness = aggregatedRoughness;
    outSample.aggregatedHills = aggregatedHills;
    outSample.aggregatedMountains = aggregatedMountains;
    outSample.keepOriginalMix = std::clamp(keepOriginal, 0.0f, 1.0f);

    const WeightedSeed& dominant = weighted.front();
    outSample.dominantSitePos = glm::vec2(static_cast<float>(dominant.seed->position.x),
                                          static_cast<float>(dominant.seed->position.y));
    outSample.dominantSiteHalfExtents = glm::vec2(std::max(dominant.seed->radius, 1.0f));
}

void NoiseVoronoiClimateGenerator::generate(ClimateFragment& fragment)
{
    const glm::ivec2 baseWorld = fragment.baseWorld();
    for (int localZ = 0; localZ < ClimateFragment::kSize; ++localZ)
    {
        for (int localX = 0; localX < ClimateFragment::kSize; ++localX)
        {
            ClimateSample& sample = fragment.sample(localX, localZ);
            const glm::ivec2 worldPos{baseWorld.x + localX, baseWorld.y + localZ};
            accumulateSample(worldPos, sample);
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

        auto existing = fragments_.find(key);
        if (existing != fragments_.end())
        {
            // Another thread populated this fragment while we were generating ours.
            touch(existing->second);
            result = existing->second.fragment.get();
        }
        else
        {
            FragmentCacheEntry entry{};
            entry.fragment = std::move(fragment);
            entry.lruIt = lru_.emplace(lru_.begin(), key);
            entry.inLru = true;

            auto [it, inserted] = fragments_.emplace(key, std::move(entry));
            (void)inserted;
            evictIfNeeded();
            result = it->second.fragment.get();
        }
    }

    return *result;
}

void ClimateMap::touch(FragmentCacheEntry& entry) const
{
    if (entry.inLru)
    {
        lru_.erase(entry.lruIt);
    }
    entry.lruIt = lru_.emplace(lru_.begin(), entry.fragment->fragmentCoord());
    entry.inLru = true;
}

void ClimateMap::evictIfNeeded() const
{
    while (fragments_.size() > maxFragments_ && !lru_.empty())
    {
        auto lruIt = std::prev(lru_.end());
        const glm::ivec2 key = *lruIt;

        auto fragIt = fragments_.find(key);
        if (fragIt != fragments_.end())
        {
            fragIt->second.inLru = false;
            fragments_.erase(fragIt);
        }

        lru_.erase(lruIt);
    }
}

} // namespace terrain
