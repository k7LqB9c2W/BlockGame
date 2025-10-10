#include "terrain/climate_map.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
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
constexpr float kDistanceBias = 1e-3f;
}

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

NoiseVoronoiClimateGenerator::PerlinNoise::PerlinNoise(unsigned seed)
{
    std::array<int, 256> p;
    for (int i = 0; i < 256; ++i)
    {
        p[i] = i;
    }

    std::mt19937 rng(seed);
    std::shuffle(p.begin(), p.end(), rng);

    for (int i = 0; i < 256; ++i)
    {
        permutation_[i] = p[i];
        permutation_[256 + i] = p[i];
    }
}

float NoiseVoronoiClimateGenerator::PerlinNoise::noise(float x, float y) const noexcept
{
    const int xi = static_cast<int>(std::floor(x)) & 255;
    const int yi = static_cast<int>(std::floor(y)) & 255;
    const float xf = x - std::floor(x);
    const float yf = y - std::floor(y);

    const float u = fade(xf);
    const float v = fade(yf);

    const int aa = permutation_[xi] + yi;
    const int ab = permutation_[xi] + yi + 1;
    const int ba = permutation_[xi + 1] + yi;
    const int bb = permutation_[xi + 1] + yi + 1;

    const float x1 = lerp(grad(permutation_[aa], xf, yf), grad(permutation_[ba], xf - 1.0f, yf), u);
    const float x2 = lerp(grad(permutation_[ab], xf, yf - 1.0f), grad(permutation_[bb], xf - 1.0f, yf - 1.0f), u);
    return lerp(x1, x2, v);
}

float NoiseVoronoiClimateGenerator::PerlinNoise::fbm(float x,
                                                      float y,
                                                      int octaves,
                                                      float persistence,
                                                      float lacunarity) const noexcept
{
    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;

    for (int i = 0; i < octaves; ++i)
    {
        total += noise(x * frequency, y * frequency) * amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    return total;
}

float NoiseVoronoiClimateGenerator::PerlinNoise::ridge(float x,
                                                        float y,
                                                        int octaves,
                                                        float lacunarity,
                                                        float gain) const noexcept
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    float prev = 1.0f;

    for (int i = 0; i < octaves; ++i)
    {
        float n = noise(x * frequency, y * frequency);
        n = 1.0f - std::abs(n);
        n *= n;
        sum += n * amplitude * prev;
        prev = n;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return sum;
}

float NoiseVoronoiClimateGenerator::PerlinNoise::fade(float t) noexcept
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float NoiseVoronoiClimateGenerator::PerlinNoise::lerp(float a, float b, float t) noexcept
{
    return a + t * (b - a);
}

float NoiseVoronoiClimateGenerator::PerlinNoise::grad(int hash, float x, float y) noexcept
{
    const int h = hash & 7;
    const float u = h < 4 ? x : y;
    const float v = h < 4 ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
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
      noise_(seed)
{
    const float maxFootprint = biomeDatabase_.maxFootprintMultiplier();
    biomeRegionSearchRadius_ = std::max(1, ceilToIntPositive(maxFootprint * 0.5f));
    biomeRegionCandidateCapacity_ = static_cast<std::size_t>((biomeRegionSearchRadius_ * 2 + 1)
                                                             * (biomeRegionSearchRadius_ * 2 + 1));
    biomeRegionCandidateCapacity_ = std::max<std::size_t>(biomeRegionCandidateCapacity_, 1);
}

int NoiseVoronoiClimateGenerator::ceilToIntPositive(float value) noexcept
{
    const int truncated = static_cast<int>(value);
    return (static_cast<float>(truncated) < value) ? truncated + 1 : truncated;
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

float NoiseVoronoiClimateGenerator::hashToUnitFloat(int x, int y, int z) noexcept
{
    std::uint32_t h = static_cast<std::uint32_t>(x * 374761393 + y * 668265263 + z * 2147483647);
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
}

float NoiseVoronoiClimateGenerator::littleMountainInfluence(float normalizedDistance) noexcept
{
    const float clamped = std::clamp(normalizedDistance, 0.0f, 1.0f);
    const float tapered = 1.0f - glm::smoothstep(0.35f, 0.85f, clamped);
    return std::pow(std::clamp(tapered, 0.0f, 1.0f), 1.75f);
}

TerrainBasisSample NoiseVoronoiClimateGenerator::computeTerrainBasis(int worldX, int worldZ) const
{
    TerrainBasisSample basis{};

    const auto& noiseSettings = profile_.noise;
    const float worldXF = static_cast<float>(worldX);
    const float worldZF = static_cast<float>(worldZ);

    basis.mainTerrain = noise_.fbm(worldXF * noiseSettings.main.frequency,
                                   worldZF * noiseSettings.main.frequency,
                                   noiseSettings.main.octaves,
                                   noiseSettings.main.gain,
                                   noiseSettings.main.lacunarity);
    basis.mountainNoise = noise_.ridge(worldXF * noiseSettings.mountain.frequency,
                                       worldZF * noiseSettings.mountain.frequency,
                                       noiseSettings.mountain.octaves,
                                       noiseSettings.mountain.lacunarity,
                                       noiseSettings.mountain.gain);
    basis.detailNoise = noise_.fbm(worldXF * noiseSettings.detail.frequency,
                                   worldZF * noiseSettings.detail.frequency,
                                   noiseSettings.detail.octaves,
                                   noiseSettings.detail.gain,
                                   noiseSettings.detail.lacunarity);
    basis.mediumNoise = noise_.fbm(worldXF * noiseSettings.medium.frequency,
                                   worldZF * noiseSettings.medium.frequency,
                                   noiseSettings.medium.octaves,
                                   noiseSettings.medium.gain,
                                   noiseSettings.medium.lacunarity);

    basis.combinedNoise = basis.mainTerrain * 12.0f + basis.mountainNoise * 8.0f + basis.mediumNoise * 4.0f
                          + basis.detailNoise * 2.0f;

    basis.baseElevation = basis.combinedNoise;
    basis.continentMask = 1.0f;

    return basis;
}

NoiseVoronoiClimateGenerator::BiomeSite NoiseVoronoiClimateGenerator::computeBiomeSite(const BiomeDefinition& definition,
                                                                                       int regionX,
                                                                                       int regionZ) const noexcept
{
    const float baseRegionWidth = static_cast<float>(biomeSizeInChunks_) * static_cast<float>(chunkSize_);
    const float baseRegionDepth = static_cast<float>(biomeSizeInChunks_) * static_cast<float>(chunkSize_);
    const float footprint = std::max(definition.footprintMultiplier, 0.1f);
    const float regionWidth = baseRegionWidth * footprint;
    const float regionDepth = baseRegionDepth * footprint;
    const float marginX = regionWidth * 0.25f;
    const float marginZ = regionDepth * 0.25f;
    const float availableWidth = std::max(regionWidth - marginX * 2.0f, 0.0f);
    const float availableDepth = std::max(regionDepth - marginZ * 2.0f, 0.0f);
    const float baseX = static_cast<float>(regionX) * baseRegionWidth;
    const float baseZ = static_cast<float>(regionZ) * baseRegionDepth;

    const float jitterX = hashToUnitFloat(regionX, 137, regionZ);
    const float jitterZ = hashToUnitFloat(regionX, 613, regionZ);

    BiomeSite site{};
    site.worldPosXZ.x = baseX + marginX + availableWidth * jitterX;
    site.worldPosXZ.y = baseZ + marginZ + availableDepth * jitterZ;
    site.halfExtents = glm::vec2(regionWidth * 0.5f, regionDepth * 0.5f);
    return site;
}

const BiomeDefinition& NoiseVoronoiClimateGenerator::biomeForRegion(int regionX, int regionZ) const
{
    const float selector = hashToUnitFloat(regionX, 31, regionZ);
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

BiomePerturbationSample NoiseVoronoiClimateGenerator::applyBiomePerturbations(
    const std::array<WeightedBiome, 5>& weightedBiomes,
    std::size_t weightCount,
    int biomeRegionX,
    int biomeRegionZ) const
{
    BiomePerturbationSample result{};

    if (weightCount == 0)
    {
        const BiomeDefinition& fallbackBiome = biomeForRegion(biomeRegionX, biomeRegionZ);
        result.dominantBiome = &fallbackBiome;
        result.dominantWeight = 1.0f;
        result.blendedOffset = fallbackBiome.heightOffset;
        result.blendedScale = fallbackBiome.heightScale;
        result.blendedMinHeight = static_cast<float>(fallbackBiome.minHeight);
        result.blendedMaxHeight = static_cast<float>(fallbackBiome.maxHeight);
        result.blendedSlopeBias = fallbackBiome.baseSlopeBias;
        result.blendedMaxGradient = fallbackBiome.maxGradient;
        return result;
    }

    float totalBlendWeight = 0.0f;
    float totalOceanWeight = 0.0f;
    float totalLandWeight = 0.0f;

    for (std::size_t i = 0; i < weightCount; ++i)
    {
        const WeightedBiome& weightedBiome = weightedBiomes[i];
        if (!weightedBiome.biome || weightedBiome.weight <= 0.0f)
        {
            continue;
        }

        const BiomeDefinition& biome = *weightedBiome.biome;
        const float weight = weightedBiome.weight;

        if (!result.dominantBiome || weight > result.dominantWeight)
        {
            result.dominantBiome = &biome;
            result.dominantWeight = weight;
        }

        result.blendedOffset += biome.heightOffset * weight;
        result.blendedScale += biome.heightScale * weight;
        result.blendedMinHeight += static_cast<float>(biome.minHeight) * weight;
        result.blendedMaxHeight += static_cast<float>(biome.maxHeight) * weight;
        result.blendedSlopeBias += biome.baseSlopeBias * weight;
        result.blendedMaxGradient += biome.maxGradient * weight;

        totalBlendWeight += weight;

        if (biome.isOcean())
        {
            result.oceanWeight += weight;
            result.oceanOffset += biome.heightOffset * weight;
            result.oceanScale += biome.heightScale * weight;
            result.oceanMinHeight += static_cast<float>(biome.minHeight) * weight;
            result.oceanMaxHeight += static_cast<float>(biome.maxHeight) * weight;
            result.oceanSlopeBias += biome.baseSlopeBias * weight;
            result.oceanMaxGradient += biome.maxGradient * weight;
            totalOceanWeight += weight;
        }
        else
        {
            result.landWeight += weight;
            result.landOffset += biome.heightOffset * weight;
            result.landScale += biome.heightScale * weight;
            result.landMinHeight += static_cast<float>(biome.minHeight) * weight;
            result.landMaxHeight += static_cast<float>(biome.maxHeight) * weight;
            result.landSlopeBias += biome.baseSlopeBias * weight;
            result.landMaxGradient += biome.maxGradient * weight;
            totalLandWeight += weight;
        }
    }

    auto normalizeSlope = [](float weight, float& slopeBias, float& maxGradient)
    {
        if (weight <= 0.0f)
        {
            slopeBias = 0.0f;
            maxGradient = 0.0f;
            return;
        }

        const float invWeight = 1.0f / weight;
        slopeBias = std::clamp(slopeBias * invWeight, 0.0f, 1.0f);
        maxGradient = std::max(maxGradient * invWeight, 0.0f);
    };

    normalizeSlope(totalBlendWeight, result.blendedSlopeBias, result.blendedMaxGradient);
    normalizeSlope(result.oceanWeight, result.oceanSlopeBias, result.oceanMaxGradient);
    normalizeSlope(result.landWeight, result.landSlopeBias, result.landMaxGradient);

    if (totalBlendWeight > std::numeric_limits<float>::epsilon())
    {
        const float invWeight = 1.0f / totalBlendWeight;
        result.blendedOffset *= invWeight;
        result.blendedScale *= invWeight;
        result.blendedMinHeight *= invWeight;
        result.blendedMaxHeight *= invWeight;
    }

    if (result.oceanWeight > std::numeric_limits<float>::epsilon())
    {
        const float invOcean = 1.0f / result.oceanWeight;
        result.oceanOffset *= invOcean;
        result.oceanScale *= invOcean;
        result.oceanMinHeight *= invOcean;
        result.oceanMaxHeight *= invOcean;
    }

    if (result.landWeight > std::numeric_limits<float>::epsilon())
    {
        const float invLand = 1.0f / result.landWeight;
        result.landOffset *= invLand;
        result.landScale *= invLand;
        result.landMinHeight *= invLand;
        result.landMaxHeight *= invLand;
    }

    if (!result.dominantBiome)
    {
        result.dominantBiome = &biomeForRegion(biomeRegionX, biomeRegionZ);
        result.dominantWeight = 1.0f;
    }

    return result;
}

void NoiseVoronoiClimateGenerator::generate(ClimateFragment& fragment)
{
    const glm::ivec2 fragmentCoord = fragment.fragmentCoord();
    const glm::ivec2 baseWorld = fragment.baseWorld();

    for (int localZ = 0; localZ < ClimateFragment::kSize; ++localZ)
    {
        for (int localX = 0; localX < ClimateFragment::kSize; ++localX)
        {
            const int worldX = baseWorld.x + localX;
            const int worldZ = baseWorld.y + localZ;

            const int chunkX = floorDiv(worldX, chunkSize_);
            const int chunkZ = floorDiv(worldZ, chunkSize_);
            const int biomeRegionX = floorDiv(chunkX, biomeSizeInChunks_);
            const int biomeRegionZ = floorDiv(chunkZ, biomeSizeInChunks_);

            candidateSites.clear();

            struct CandidateSite
            {
                const BiomeDefinition* biome{nullptr};
                glm::vec2 positionXZ{0.0f};
                glm::vec2 halfExtents{0.0f};
                float distanceSquared{std::numeric_limits<float>::max()};
                float normalizedDistance{1.0f};
            };

            std::vector<CandidateSite> candidates;
            candidates.reserve(biomeRegionCandidateCapacity_);

            const glm::vec2 columnPosition{static_cast<float>(worldX) + 0.5f, static_cast<float>(worldZ) + 0.5f};

            for (int regionOffsetZ = -biomeRegionSearchRadius_; regionOffsetZ <= biomeRegionSearchRadius_;
                 ++regionOffsetZ)
            {
                for (int regionOffsetX = -biomeRegionSearchRadius_; regionOffsetX <= biomeRegionSearchRadius_;
                     ++regionOffsetX)
                {
                    const int regionX = biomeRegionX + regionOffsetX;
                    const int regionZ = biomeRegionZ + regionOffsetZ;
                    const BiomeDefinition& definition = biomeForRegion(regionX, regionZ);
                    const BiomeSite site = computeBiomeSite(definition, regionX, regionZ);

                    CandidateSite candidate{};
                    candidate.biome = &definition;
                    candidate.positionXZ = site.worldPosXZ;
                    candidate.halfExtents = site.halfExtents;
                    const glm::vec2 delta = columnPosition - candidate.positionXZ;
                    const glm::vec2 halfExtents = glm::max(candidate.halfExtents, glm::vec2(1.0f));
                    const glm::vec2 normalizedDelta = delta / halfExtents;
                    candidate.distanceSquared = glm::dot(delta, delta);
                    candidate.normalizedDistance = glm::length(normalizedDelta);
                    candidates.push_back(candidate);
                }
            }

            constexpr std::size_t kMaxConsideredSites = 4;
            const std::size_t candidateCount = candidates.size();
            std::size_t sitesToConsider = std::min<std::size_t>(kMaxConsideredSites, candidateCount);

            auto candidateLess = [](const CandidateSite& lhs, const CandidateSite& rhs)
            {
                const bool lhsValid = std::isfinite(lhs.normalizedDistance);
                const bool rhsValid = std::isfinite(rhs.normalizedDistance);
                if (lhsValid && rhsValid)
                {
                    if (lhs.normalizedDistance == rhs.normalizedDistance)
                    {
                        return lhs.distanceSquared < rhs.distanceSquared;
                    }
                    return lhs.normalizedDistance < rhs.normalizedDistance;
                }

                if (lhsValid != rhsValid)
                {
                    return lhsValid;
                }

                return lhs.distanceSquared < rhs.distanceSquared;
            };

            if (sitesToConsider > 0)
            {
                std::partial_sort(candidates.begin(), candidates.begin() + sitesToConsider,
                                  candidates.begin() + candidateCount, candidateLess);

                bool allLittleMountains = true;
                for (std::size_t i = 0; i < sitesToConsider; ++i)
                {
                    if (candidates[i].biome && !candidates[i].biome->isLittleMountains())
                    {
                        allLittleMountains = false;
                        break;
                    }
                }

                if (allLittleMountains)
                {
                    std::size_t bestIndex = candidateCount;
                    CandidateSite bestSite{};
                    bool hasBestSite = false;
                    for (std::size_t i = sitesToConsider; i < candidateCount; ++i)
                    {
                        const CandidateSite& site = candidates[i];
                        if (!site.biome || site.biome->isLittleMountains())
                        {
                            continue;
                        }

                        if (!hasBestSite || candidateLess(site, bestSite))
                        {
                            bestIndex = i;
                            bestSite = site;
                            hasBestSite = true;
                        }
                    }

                    if (hasBestSite)
                    {
                        if (sitesToConsider < kMaxConsideredSites)
                        {
                            std::swap(candidates[sitesToConsider], candidates[bestIndex]);
                            ++sitesToConsider;
                        }
                        else if (sitesToConsider > 0)
                        {
                            std::swap(candidates[sitesToConsider - 1], candidates[bestIndex]);
                        }

                        std::partial_sort(candidates.begin(), candidates.begin() + sitesToConsider,
                                          candidates.begin() + candidateCount, candidateLess);
                    }
                }
            }

            std::array<WeightedBiome, 5> weightedBiomes{};
            std::size_t weightCount = 0;

            if (sitesToConsider == 0)
            {
                const BiomeDefinition& fallbackBiome = biomeForRegion(biomeRegionX, biomeRegionZ);
                weightedBiomes[weightCount++] = WeightedBiome{&fallbackBiome, 1.0f};
            }
            else
            {
                std::array<float, kMaxConsideredSites> rawWeights{};
                std::array<float, kMaxConsideredSites> distances{};
                for (std::size_t i = 0; i < sitesToConsider; ++i)
                {
                    distances[i] = std::sqrt(candidates[i].distanceSquared);
                }

                for (std::size_t i = 0; i < sitesToConsider; ++i)
                {
                    const float biasedDistance = distances[i] + kDistanceBias;
                    if (!std::isfinite(biasedDistance))
                    {
                        rawWeights[i] = 1.0f / kDistanceBias;
                    }
                    else
                    {
                        const float safeDistance = std::max(biasedDistance, kDistanceBias);
                        rawWeights[i] = 1.0f / safeDistance;
                    }
                }

                for (std::size_t i = 0; i < sitesToConsider; ++i)
                {
                    if (candidates[i].biome && candidates[i].biome->isLittleMountains())
                    {
                        const float influence = littleMountainInfluence(candidates[i].normalizedDistance);
                        rawWeights[i] *= influence;
                        rawWeights[i] *= 1.15f;
                    }
                }

                float totalWeight = 0.0f;
                for (std::size_t i = 0; i < sitesToConsider; ++i)
                {
                    totalWeight += rawWeights[i];
                }

                if (totalWeight <= std::numeric_limits<float>::epsilon())
                {
                    rawWeights.fill(0.0f);
                    rawWeights[0] = 1.0f;
                    sitesToConsider = 1;
                    totalWeight = 1.0f;
                }

                for (std::size_t i = 0; i < sitesToConsider; ++i)
                {
                    const float normalizedWeight = rawWeights[i] / totalWeight;
                    if (normalizedWeight <= 0.0f || candidates[i].biome == nullptr)
                    {
                        continue;
                    }

                    weightedBiomes[weightCount++] = WeightedBiome{candidates[i].biome, normalizedWeight};
                }
            }

            if (weightCount == 0)
            {
                const BiomeDefinition& fallbackBiome = biomeForRegion(biomeRegionX, biomeRegionZ);
                weightedBiomes[weightCount++] = WeightedBiome{&fallbackBiome, 1.0f};
            }

            ClimateSample& outSample = fragment.sample(localX, localZ);
            outSample.fallbackBiome = &biomeForRegion(biomeRegionX, biomeRegionZ);
            outSample.perturbations = applyBiomePerturbations(weightedBiomes, weightCount, biomeRegionX, biomeRegionZ);
            outSample.basis = computeTerrainBasis(worldX, worldZ);

            outSample.hasNonLittleMountainsBiome = false;
            outSample.littleMountainsWeight = 0.0f;
            outSample.littleMountainsDefinition = nullptr;
            for (std::size_t i = 0; i < weightCount; ++i)
            {
                const WeightedBiome& weightedBiome = weightedBiomes[i];
                if (!weightedBiome.biome || weightedBiome.weight <= 0.0f)
                {
                    continue;
                }

                if (weightedBiome.biome->isLittleMountains())
                {
                    outSample.littleMountainsWeight += weightedBiome.weight;
                    if (!outSample.littleMountainsDefinition)
                    {
                        outSample.littleMountainsDefinition = weightedBiome.biome;
                    }
                }
                else if (weightedBiome.weight > 0.0f)
                {
                    outSample.hasNonLittleMountainsBiome = true;
                }
            }
            outSample.littleMountainsWeight = std::clamp(outSample.littleMountainsWeight, 0.0f, 1.0f);

            float closestNormalizedDistance = std::numeric_limits<float>::infinity();
            if (outSample.littleMountainsDefinition)
            {
                for (const CandidateSite& candidate : candidates)
                {
                    if (candidate.biome != outSample.littleMountainsDefinition)
                    {
                        continue;
                    }

                    closestNormalizedDistance = std::min(closestNormalizedDistance, candidate.normalizedDistance);
                }
            }
            if (std::isfinite(closestNormalizedDistance))
            {
                outSample.littleMountainInteriorMask = littleMountainInfluence(closestNormalizedDistance);
            }
            else
            {
                outSample.littleMountainInteriorMask = 0.0f;
            }

            outSample.hasBorderPerturbations = false;
            if (outSample.hasNonLittleMountainsBiome)
            {
                std::array<WeightedBiome, 5> nonMountainBiomes{};
                std::size_t nonMountainCount = 0;
                float nonMountainWeight = 0.0f;
                for (std::size_t i = 0; i < weightCount; ++i)
                {
                    const WeightedBiome& weightedBiome = weightedBiomes[i];
                    if (!weightedBiome.biome || weightedBiome.weight <= 0.0f)
                    {
                        continue;
                    }

                    if (weightedBiome.biome->isLittleMountains())
                    {
                        continue;
                    }

                    nonMountainBiomes[nonMountainCount++] = weightedBiome;
                    nonMountainWeight += weightedBiome.weight;
                }

                if (nonMountainCount > 0 && nonMountainWeight > std::numeric_limits<float>::epsilon())
                {
                    const float invWeight = 1.0f / nonMountainWeight;
                    for (std::size_t i = 0; i < nonMountainCount; ++i)
                    {
                        nonMountainBiomes[i].weight *= invWeight;
                    }

                    outSample.borderPerturbations =
                        applyBiomePerturbations(nonMountainBiomes, nonMountainCount, biomeRegionX, biomeRegionZ);
                    outSample.hasBorderPerturbations = true;
                }
            }
        }
    }
}

ClimateMap::ClimateMap(std::unique_ptr<ClimateGenerator> generator, std::size_t maxFragments)
    : generator_(std::move(generator)),
      maxFragments_(std::max<std::size_t>(maxFragments, 1))
{
    if (!generator_)
    {
        throw std::invalid_argument("ClimateMap requires a valid generator");
    }
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

    std::unique_lock<std::mutex> lock(mutex_);
    auto it = fragments_.find(key);
    if (it != fragments_.end())
    {
        touch(it->second);
        return *it->second.fragment;
    }

    lock.unlock();

    auto fragment = std::make_unique<ClimateFragment>(key);
    generator_->generate(*fragment);

    lock.lock();
    auto [insertIt, inserted] = fragments_.try_emplace(key);
    if (!inserted)
    {
        touch(insertIt->second);
        return *insertIt->second.fragment;
    }

    lru_.push_front(key);
    insertIt->second.fragment = std::move(fragment);
    insertIt->second.lruIt = lru_.begin();
    evictIfNeeded();
    return *insertIt->second.fragment;
}

void ClimateMap::touch(FragmentCacheEntry& entry) const
{
    lru_.splice(lru_.begin(), lru_, entry.lruIt);
    entry.lruIt = lru_.begin();
}

void ClimateMap::evictIfNeeded() const
{
    while (fragments_.size() > maxFragments_)
    {
        const glm::ivec2& leastUsed = lru_.back();
        auto it = fragments_.find(leastUsed);
        if (it != fragments_.end())
        {
            fragments_.erase(it);
        }
        lru_.pop_back();
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

} // namespace terrain

