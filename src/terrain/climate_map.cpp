#include "terrain/climate_map.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <functional>
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

float evaluateInterpolationCurve(float t, BiomeDefinition::InterpolationCurve curve) noexcept
{
    t = std::clamp(t, 0.0f, 1.0f);
    switch (curve)
    {
    case BiomeDefinition::InterpolationCurve::Step:
        return t >= 0.5f ? 1.0f : 0.0f;
    case BiomeDefinition::InterpolationCurve::Linear:
        return t;
    case BiomeDefinition::InterpolationCurve::Square:
    default:
        if (t < 0.5f)
        {
            return std::clamp(2.0f * t * t, 0.0f, 1.0f);
        }
        const float inv = 1.0f - t;
        return std::clamp(1.0f - 2.0f * inv * inv, 0.0f, 1.0f);
    }
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

std::uint16_t groupPresenceMask(std::uint16_t bits) noexcept
{
    constexpr int kGroupSize = 3;
    constexpr int kGroupCount = 5;

    std::uint16_t mask = 0;
    for (int group = 0; group < kGroupCount; ++group)
    {
        const std::uint16_t groupBits =
            static_cast<std::uint16_t>((bits >> (group * kGroupSize)) & 0x7u);
        if (groupBits != 0)
        {
            mask |= static_cast<std::uint16_t>(1u << (group * kGroupSize));
        }
    }
    return mask;
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
    oceanBiomes_.reserve(defs.size());
    oceanWeightPrefix_.reserve(defs.size());
    maxTransitionWidth_ = 0;
    for (const BiomeDefinition& def : defs)
    {
        if (def.spawnChance <= 0.0f)
        {
            continue;
        }
        const float radiusScale = std::max(def.radius, 1.0f);
        float weight = std::max(def.spawnChance * def.footprintMultiplier, 0.0f);
        weight /= std::max(radiusScale, 1.0f);
        const auto& props = def.generationProperties();
        if (props.has(BiomeDefinition::GenerationProperties::kOcean))
        {
            weight *= 1.25f;
        }
        if (props.has(BiomeDefinition::GenerationProperties::kMountain))
        {
            weight *= 0.85f;
        }
        if (props.has(BiomeDefinition::GenerationProperties::kLowTerrain))
        {
            weight *= 1.1f;
        }
        if (weight <= 0.0f)
        {
            continue;
        }
        biomeSelection_.push_back(&def);
        totalSpawnWeight_ += weight;
        biomeWeightPrefix_.push_back(totalSpawnWeight_);

        if (def.isOcean())
        {
            oceanBiomes_.push_back(&def);
            totalOceanWeight_ += weight;
            oceanWeightPrefix_.push_back(totalOceanWeight_);
        }

        for (const auto& transition : def.transitionBiomes)
        {
            maxTransitionWidth_ = std::max(maxTransitionWidth_, transition.width);
        }
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
    bool hasOceanSeed = false;

    const auto pushSeed = [&](BiomeSeed&& seed) {
        if (!seed.biome)
        {
            return;
        }
        result.maxRadius = std::max(result.maxRadius, static_cast<int>(std::ceil(seed.radius)));
        if (seed.biome->isOcean())
        {
            hasOceanSeed = true;
        }
        result.seeds.push_back(std::move(seed));
    };

    const auto updateNewSeeds = [&](std::size_t startIndex) {
        for (std::size_t i = startIndex; i < result.seeds.size(); ++i)
        {
            result.maxRadius =
                std::max(result.maxRadius, static_cast<int>(std::ceil(result.seeds[i].radius)));
            if (result.seeds[i].biome && result.seeds[i].biome->isOcean())
            {
                hasOceanSeed = true;
            }
        }
    };

    const auto tryAddOceanSeed = [&](int attempts, float spacingScale) -> bool {
        if (oceanBiomes_.empty() || static_cast<int>(result.seeds.size()) >= kMaxSeedsPerChunk)
        {
            return false;
        }
        for (int attempt = 0; attempt < attempts; ++attempt)
        {
            const BiomeDefinition& oceanBiome = chooseOceanBiome(rng);
            const int worldX = baseX + rng.nextInt(0, chunkSpan_ - 1);
            const int worldZ = baseZ + rng.nextInt(0, chunkSpan_ - 1);

            BiomeSeed seed = createSeed(rng, worldX, worldZ, oceanBiome);
            if (!seed.biome)
            {
                continue;
            }
            if (!isValidPlacement(seed.position, seed.radius, result.seeds, spacingScale))
            {
                continue;
            }
            pushSeed(std::move(seed));
            return true;
        }
        return false;
    };

    if (totalOceanWeight_ > 0.0f && totalSpawnWeight_ > 0.0f)
    {
        const float expectedShare = std::clamp(totalOceanWeight_ / totalSpawnWeight_, 0.05f, 0.35f);
        if (rng.nextFloat() < expectedShare)
        {
            tryAddOceanSeed(24, 1.0f);
        }
    }

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

        pushSeed(std::move(seed));
        rejections = 0;
        const std::size_t beforeSub = result.seeds.size();
        spawnSubBiomeSeeds(result.seeds.back(), result.seeds, rng);
        updateNewSeeds(beforeSub);
    }

    if (result.seeds.empty())
    {
        BiomeSeed fallback = createSeed(rng, baseX + chunkSpan_ / 2, baseZ + chunkSpan_ / 2);
        pushSeed(std::move(fallback));
        const std::size_t beforeSub = result.seeds.size();
        spawnSubBiomeSeeds(result.seeds.back(), result.seeds, rng);
        updateNewSeeds(beforeSub);
    }

    if (!hasOceanSeed)
    {
        if (!tryAddOceanSeed(32, 1.0f))
        {
            tryAddOceanSeed(48, 0.75f);
        }
    }

    return result;
}

NoiseVoronoiClimateGenerator::BiomeSeed
NoiseVoronoiClimateGenerator::createSeed(Random& rng, int worldX, int worldZ) const
{
    const BiomeDefinition& biome = chooseBiome(rng);
    return createSeed(rng, worldX, worldZ, biome);
}

NoiseVoronoiClimateGenerator::BiomeSeed
NoiseVoronoiClimateGenerator::createSeed(Random& rng,
                                         int worldX,
                                         int worldZ,
                                         const BiomeDefinition& biome) const
{
    BiomeSeed seed{};
    seed.biome = &biome;
    float radius = biome.radius;
    if (!(biome.fixedRadius || biome.isOcean()))
    {
        radius = std::clamp(biome.radius + biome.radiusVariation * rng.nextFloatSigned(),
                             biome.minRadius(),
                             biome.maxRadius());
    }
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

const BiomeDefinition& NoiseVoronoiClimateGenerator::chooseOceanBiome(Random& rng) const
{
    if (oceanBiomes_.empty())
    {
        return chooseBiome(rng);
    }

    const float pick = rng.nextFloat() * totalOceanWeight_;
    auto it = std::lower_bound(oceanWeightPrefix_.begin(), oceanWeightPrefix_.end(), pick);
    std::size_t index = 0;
    if (it == oceanWeightPrefix_.end())
    {
        index = oceanWeightPrefix_.size() - 1;
    }
    else
    {
        index = static_cast<std::size_t>(std::distance(oceanWeightPrefix_.begin(), it));
    }
    return *oceanBiomes_[index];
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
    return isValidPlacement(position, radius, seeds, 1.0f);
}

bool NoiseVoronoiClimateGenerator::isValidPlacement(const glm::ivec2& position,
                                                    float radius,
                                                    const std::vector<BiomeSeed>& seeds,
                                                    float spacingScale) const noexcept
{
    for (const BiomeSeed& other : seeds)
    {
        const float largestRadius = std::max(radius, other.radius);
        const float baseSpacing = std::clamp(0.85f - 0.0005f * largestRadius, 0.6f, 0.85f);
        const float spacingFactor = std::clamp(baseSpacing * spacingScale, 0.4f, 0.85f);
        const float combined = (radius + other.radius) * spacingFactor;
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
    std::vector<const BiomeSeed*> rawCandidates;
    rawCandidates.reserve(128);
    gatherCandidateSeeds(worldPos, rawCandidates);

    struct CandidateInfo
    {
        const BiomeSeed* seed{nullptr};
        float distance{0.0f};
        float radius{1.0f};
        float normalized{0.0f};
        float influence{0.0f};
    };

    std::vector<CandidateInfo> candidates;
    candidates.reserve(rawCandidates.size());

    for (const BiomeSeed* candidate : rawCandidates)
    {
        const float distSq = lengthSquared(worldPos, candidate->position);
        const float distance = std::sqrt(distSq);
        const float normalized = distance / std::max(candidate->radius, 1.0f);
        const float blended = std::clamp(1.0f - normalized, 0.0f, 1.0f);
        float influence = smoothStep(blended);
        candidates.push_back(CandidateInfo{candidate, distance, std::max(candidate->radius, 1.0f), normalized, influence});
    }

    struct WeightedSeed
    {
        const BiomeSeed* seed{nullptr};
        float weight{0.0f};
        float normalizedDistance{0.0f};
        float distance{0.0f};
        float radius{1.0f};
    };

    std::vector<WeightedSeed> weighted;
    weighted.reserve(candidates.size());

    for (const CandidateInfo& candidate : candidates)
    {
        if (candidate.influence <= std::numeric_limits<float>::epsilon())
        {
            continue;
        }
        if (!candidate.seed || !candidate.seed->biome)
        {
            continue;
        }

        const BiomeDefinition& biome = *candidate.seed->biome;
        const float blendFactor =
            evaluateInterpolationCurve(1.0f - candidate.normalized, biome.interpolationCurve);
        const float adjustedWeight = candidate.influence * blendFactor * biome.interpolationWeight;
        if (adjustedWeight <= std::numeric_limits<float>::epsilon())
        {
            continue;
        }

        weighted.push_back(WeightedSeed{candidate.seed,
                                        adjustedWeight,
                                        candidate.normalized,
                                        candidate.distance,
                                        candidate.radius});
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
        blend.height = biome.applyHeightLimits(blend.height, 0.0f);
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
        blend.height = biome.applyHeightLimits(blend.height, entry.normalizedDistance);
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
    outSample.dominantSiteHalfExtents = glm::vec2(dominant.radius);
    outSample.dominantIsOcean = dominant.seed->biome && dominant.seed->biome->isOcean();

    float bestBoundary = std::numeric_limits<float>::infinity();
    for (const CandidateInfo& entry : candidates)
    {
        if (!entry.seed || !entry.seed->biome)
        {
            continue;
        }
        const bool isOceanSeed = entry.seed->biome->isOcean();
        if (isOceanSeed == outSample.dominantIsOcean)
        {
            continue;
        }
        float boundaryDistance = outSample.dominantIsOcean ? std::max(0.0f, entry.radius - entry.distance)
                                                           : std::max(0.0f, entry.distance - entry.radius);
        bestBoundary = std::min(bestBoundary, boundaryDistance);
    }
    if (std::isfinite(bestBoundary))
    {
        outSample.distanceToCoast = bestBoundary;
    }
    else
    {
        outSample.distanceToCoast = outSample.dominantIsOcean ? 0.0f : std::numeric_limits<float>::infinity();
    }
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

    applyTransitionBiomes(baseWorld, fragment);
}

glm::vec2 NoiseVoronoiClimateGenerator::randomInUnitCircle(Random& rng) noexcept
{
    glm::vec2 v{0.0f};
    do
    {
        v.x = rng.nextFloatSigned();
        v.y = rng.nextFloatSigned();
    } while (glm::dot(v, v) > 1.0f);
    return v;
}

void NoiseVoronoiClimateGenerator::spawnSubBiomeSeeds(const BiomeSeed& parent,
                                                      std::vector<BiomeSeed>& seeds,
                                                      Random& rng) const
{
    if (!parent.biome || parent.biome->subBiomes.empty())
    {
        return;
    }

    const int maxCount = parent.biome->maxSubBiomeCount > 0.0f
                             ? static_cast<int>(std::ceil(parent.biome->maxSubBiomeCount))
                             : std::numeric_limits<int>::max();
    int spawned = 0;

    for (const auto& sub : parent.biome->subBiomes)
    {
        if (!sub.biome)
        {
            continue;
        }
        if (spawned >= maxCount)
        {
            break;
        }

        const float probability = std::clamp(sub.chance, 0.0f, 1.0f);
        if (probability <= std::numeric_limits<float>::epsilon())
        {
            continue;
        }
        if (rng.nextFloat() > probability)
        {
            continue;
        }

        const glm::vec2 offset = randomInUnitCircle(rng);
        const float parentRadius = std::max(parent.radius, 1.0f);
        const float distance = parentRadius * 0.6f * std::sqrt(rng.nextFloat());
        const glm::ivec2 candidatePos = parent.position + glm::ivec2(static_cast<int>(offset.x * distance),
                                                                     static_cast<int>(offset.y * distance));

        const float radiusNoise = rng.nextFloat();
        float radius = sub.sampleRadius(parentRadius * 0.75f, radiusNoise);
        radius = std::clamp(radius, 4.0f, parentRadius);

        if (!isValidPlacement(candidatePos, radius, seeds))
        {
            continue;
        }

        BiomeSeed child{};
        child.biome = sub.biome;
        child.position = candidatePos;
        child.radius = radius;
        child.weight = 1.0f / std::max(child.radius * std::sqrt(glm::pi<float>()), 1.0f);
        child.baseHeight = randomizedHeight(rng, *child.biome);

        seeds.push_back(child);
        ++spawned;
    }
}

void NoiseVoronoiClimateGenerator::applyTransitionBiomes(const glm::ivec2& baseWorld,
                                                         ClimateFragment& fragment) const
{
    if (maxTransitionWidth_ <= 0)
    {
        return;
    }

    const int size = ClimateFragment::kSize;
    const std::size_t area = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
    const int maxWidth = std::max(1, maxTransitionWidth_);

    const auto indexFor = [size](int x, int z) -> std::size_t {
        return static_cast<std::size_t>(z) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x);
    };

    std::vector<std::uint16_t> propertyGrid(area, 0);
    std::vector<std::uint8_t> oceanSnapshot(area, 0);
    const auto refreshProperties = [&]() {
        for (int z = 0; z < size; ++z)
        {
            for (int x = 0; x < size; ++x)
            {
                const std::size_t idx = indexFor(x, z);
                const ClimateSample& sample = fragment.sample(x, z);
                const BiomeDefinition* biome = sample.dominantBiome();
                propertyGrid[idx] = biome ? biome->generationProperties().value() : 0;
            }
        }
    };

    refreshProperties();

    std::vector<std::uint16_t> neighborLayers(static_cast<std::size_t>(maxWidth + 1) * area, 0);

    const auto layerPtr = [&](int distance) -> std::uint16_t* {
        return neighborLayers.data() + static_cast<std::size_t>(distance) * area;
    };

    const auto rebuildNeighborLayers = [&]() {
        std::uint16_t* baseLayer = layerPtr(0);
        std::copy(propertyGrid.begin(), propertyGrid.end(), baseLayer);

        for (int distance = 1; distance <= maxWidth; ++distance)
        {
            const std::uint16_t* prev = layerPtr(distance - 1);
            std::uint16_t* curr = layerPtr(distance);
            for (int z = 0; z < size; ++z)
            {
                for (int x = 0; x < size; ++x)
                {
                    const std::size_t idx = indexFor(x, z);
                    std::uint16_t value = prev[idx];
                    if (x > 0)
                    {
                        value |= prev[idx - 1];
                    }
                    if (x + 1 < size)
                    {
                        value |= prev[idx + 1];
                    }
                    if (z > 0)
                    {
                        value |= prev[idx - size];
                    }
                    if (x > 0 && z > 0)
                    {
                        value |= prev[idx - size - 1];
                    }
                    if (x + 1 < size && z > 0)
                    {
                        value |= prev[idx - size + 1];
                    }
                    if (z + 1 < size)
                    {
                        value |= prev[idx + size];
                    }
                    if (x > 0 && z + 1 < size)
                    {
                        value |= prev[idx + size - 1];
                    }
                    if (x + 1 < size && z + 1 < size)
                    {
                        value |= prev[idx + size + 1];
                    }
                    curr[idx] = value;
                }
            }
        }
    };

    constexpr int kMaxIterations = 4;
    for (int iteration = 0; iteration < kMaxIterations; ++iteration)
    {
        rebuildNeighborLayers();
        for (int z = 0; z < size; ++z)
        {
            for (int x = 0; x < size; ++x)
            {
                const std::size_t idx = indexFor(x, z);
                const ClimateSample& sample = fragment.sample(x, z);
                oceanSnapshot[idx] = sample.dominantIsOcean ? 1 : 0;
            }
        }
        bool anyChange = false;

        for (int z = 0; z < size; ++z)
        {
            for (int x = 0; x < size; ++x)
            {
                ClimateSample& sample = fragment.sample(x, z);
                const BiomeDefinition* baseBiome = sample.dominantBiome();
                if (!baseBiome || baseBiome->transitionBiomes.empty())
                {
                    continue;
                }

                const std::size_t idx = indexFor(x, z);
                const int worldX = baseWorld.x + x;
                const int worldZ = baseWorld.y + z;

                for (const BiomeDefinition::TransitionBiomeDefinition& transition : baseBiome->transitionBiomes)
                {
                    if (!transition.biome)
                    {
                        continue;
                    }

                    const int radius = std::clamp(transition.width, 0, maxWidth);
                    const std::uint16_t neighborMask = layerPtr(radius)[idx];
                    const std::uint16_t requiredBits = transition.propertyMask.value();
                    const std::uint16_t matched = static_cast<std::uint16_t>(neighborMask & requiredBits);
                    const std::uint16_t spread =
                        static_cast<std::uint16_t>(matched | (matched >> 1) | (matched >> 2));
                    const std::uint16_t requiredGroups = groupPresenceMask(requiredBits);
                    const std::uint16_t availableGroups = groupPresenceMask(spread);
                    if ((availableGroups & requiredGroups) != requiredGroups)
                    {
                        continue;
                    }

                    const unsigned hashSeed = hashCombine(
                        baseSeed_,
                        hashCombine(static_cast<unsigned>(worldX),
                                    hashCombine(static_cast<unsigned>(worldZ),
                                                static_cast<unsigned>(transition.width))));
                    const float threshold = std::clamp(transition.chance, 0.0f, 1.0f);
                    const float roll = hashToUnitFloat(worldX, worldZ, static_cast<int>(hashSeed));
                    if (roll > threshold)
                    {
                        continue;
                    }

                    const BiomeDefinition& target = *transition.biome;
                    const bool targetIsCoast = target.generationProperties().isCoastal();
                    const bool targetIsBeach = target.hasFlag("beach");
                    const float seaLevelF = static_cast<float>(profile_.seaLevel);
                    const bool baseIsOcean = baseBiome->isOcean();

                    if (targetIsBeach && !baseIsOcean)
                    {
                        continue;
                    }

                    ClimateSample newSample{};
                    newSample.blendCount = 1;

                    const float prevHeight = sample.aggregatedHeight;
                    const float prevRoughness = sample.aggregatedRoughness;
                    const float prevHills = sample.aggregatedHills;
                    const float prevMountains = sample.aggregatedMountains;
                    const float prevDistance = sample.distanceToCoast;

                    const bool requiresCoastline = targetIsCoast || targetIsBeach;
                    if (requiresCoastline)
                    {
                        const float transitionWidth = static_cast<float>(std::max(transition.width, 0));
                        if (!std::isfinite(prevDistance) || prevDistance > transitionWidth)
                        {
                            continue;
                        }

                        bool hasOceanNeighbor = false;
                        for (int dz = -radius; dz <= radius && !hasOceanNeighbor; ++dz)
                        {
                            const int nz = z + dz;
                            if (nz < 0 || nz >= size)
                            {
                                continue;
                            }
                            for (int dx = -radius; dx <= radius; ++dx)
                            {
                                const int nx = x + dx;
                                if (nx < 0 || nx >= size)
                                {
                                    continue;
                                }
                                if (oceanSnapshot[indexFor(nx, nz)] != 0)
                                {
                                    hasOceanNeighbor = true;
                                    break;
                                }
                            }
                        }
                        if (!hasOceanNeighbor)
                        {
                            continue;
                        }
                    }

                    if (targetIsCoast && !baseIsOcean)
                    {
                        constexpr float kMaxElevationAboveSea = 12.0f;
                        if (prevHeight > seaLevelF + kMaxElevationAboveSea)
                        {
                            continue;
                        }
                    }

                    BiomeBlend blend{};
                    blend.biome = &target;
                    blend.weight = 1.0f;
                    blend.height = glm::mix(static_cast<float>(target.minHeight),
                                            static_cast<float>(target.maxHeight),
                                            hashToUnitFloat(worldX, worldZ, static_cast<int>(hashSeed ^ 0x45AFC123u)));
                    blend.height = target.applyHeightLimits(blend.height, 1.0f);
                    blend.roughness = target.roughness;
                    blend.hills = target.hills;
                    blend.mountains = target.mountains;
                    blend.normalizedDistance = 1.0f;
                    blend.falloff = target.maxRadius();
                    blend.seed = hashCombine(hashSeed, static_cast<unsigned>(std::hash<std::string>{}(target.id)));
                    blend.sitePosition = glm::vec2(static_cast<float>(worldX), static_cast<float>(worldZ));
                    newSample.blends[0] = blend;

                    float keepOriginal = std::clamp(target.keepOriginalTerrain, 0.0f, 1.0f);
                    float heightBlend = keepOriginal;
                    float roughBlend = keepOriginal;

                    float newHeight = glm::mix(blend.height, prevHeight, heightBlend);
                    newHeight = target.applyHeightLimits(newHeight, 1.0f);
                    float newRoughness = glm::mix(target.roughness, prevRoughness, roughBlend);
                    float newHills = glm::mix(target.hills, prevHills, roughBlend);
                    float newMountains = glm::mix(target.mountains, prevMountains, roughBlend);

                    if (targetIsCoast)
                    {
                        const float coastNoise =
                            hashToUnitFloat(worldX, worldZ, static_cast<int>(hashSeed ^ 0x17D4A5B3u));
                        const float targetCoastHeight = glm::mix(seaLevelF - 1.5f, seaLevelF + 1.5f, coastNoise);

                        heightBlend = std::min(heightBlend, 0.15f);
                        newHeight = glm::mix(targetCoastHeight, prevHeight, heightBlend);
                        newHeight = target.applyHeightLimits(newHeight, 1.0f);
                        newHeight = std::clamp(newHeight, seaLevelF - 2.5f, seaLevelF + 2.5f);

                        roughBlend = std::min(roughBlend, 0.12f);
                        newRoughness = glm::mix(target.roughness, prevRoughness, roughBlend);
                        newHills = glm::mix(target.hills, prevHills, roughBlend);
                        newMountains = glm::mix(target.mountains, prevMountains, roughBlend);

                        newRoughness = std::min(newRoughness, 0.18f);
                        newHills = std::min(newHills, 0.18f);
                        newMountains = std::min(newMountains, 0.12f);

                        keepOriginal = std::min(keepOriginal, roughBlend);
                    }

                    newSample.aggregatedHeight = newHeight;
                    newSample.aggregatedRoughness = newRoughness;
                    newSample.aggregatedHills = newHills;
                    newSample.aggregatedMountains = newMountains;
                    newSample.keepOriginalMix = keepOriginal;
                    newSample.dominantSitePos = glm::vec2(static_cast<float>(worldX), static_cast<float>(worldZ));
                    newSample.dominantSiteHalfExtents = glm::vec2(target.maxRadius());
                    newSample.dominantIsOcean = target.isOcean();
                    if (targetIsCoast)
                    {
                        newSample.distanceToCoast = std::abs(newSample.aggregatedHeight - seaLevelF);
                    }
                    else
                    {
                        newSample.distanceToCoast = target.isOcean() ? 0.0f : prevDistance;
                    }

                    sample = newSample;
                    propertyGrid[idx] = target.generationProperties().value();
                    anyChange = true;
                    break;
                }
            }
        }

        if (!anyChange)
        {
            break;
        }

        refreshProperties();
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
