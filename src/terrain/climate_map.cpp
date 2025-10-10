#include "terrain/climate_map.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <functional>

#include <glm/common.hpp>
#include <glm/geometric.hpp>

namespace terrain
{
namespace
{
constexpr float kDistanceBias = 1e-3f;

unsigned hashCombine(unsigned a, unsigned b) noexcept
{
    a ^= b + 0x9E3779B9u + (a << 6) + (a >> 2);
    return a;
}

float randomUnit(unsigned seed, int saltX, int saltY, int saltZ) noexcept
{
    std::uint32_t h = static_cast<std::uint32_t>(seed);
    h ^= static_cast<std::uint32_t>(saltX) * 374761393u;
    h ^= static_cast<std::uint32_t>(saltY) * 668265263u;
    h ^= static_cast<std::uint32_t>(saltZ) * 2147483647u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
}

float applyInterpolationCurve(float closeness, BiomeDefinition::InterpolationCurve curve) noexcept
{
    closeness = std::clamp(closeness, 0.0f, 1.0f);
    switch (curve)
    {
        case BiomeDefinition::InterpolationCurve::Step:
            return glm::smoothstep(0.2f, 0.8f, closeness);
        case BiomeDefinition::InterpolationCurve::Linear:
            return closeness;
        case BiomeDefinition::InterpolationCurve::Square:
            return closeness * closeness;
    }
    return closeness;
}

constexpr float kFalloffExponent = 2.0f;
constexpr float kInterpolationStrengthEpsilon = 1e-3f;
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
      baseSeed_(seed)
{
    const float maxFootprint = biomeDatabase_.maxFootprintMultiplier();
    biomeRegionSearchRadius_ = std::max(1, static_cast<int>(std::ceil(maxFootprint * 0.5f)));
    const int diameter = biomeRegionSearchRadius_ * 2 + 1;
    biomeRegionCandidateCapacity_ = static_cast<std::size_t>(diameter * diameter);
    biomeRegionCandidateCapacity_ = std::max<std::size_t>(biomeRegionCandidateCapacity_, 1);
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
    constexpr std::uint64_t kMulX = 374761393ull;
    constexpr std::uint64_t kMulY = 668265263ull;
    constexpr std::uint64_t kMulZ = 2147483647ull;
    constexpr std::uint64_t kMixMul = 1274126177ull;
    constexpr std::uint64_t kMask24 = 0xFFFFFFull;

    const auto widen = [](int value) noexcept -> std::uint64_t {
        return static_cast<std::uint64_t>(static_cast<std::uint32_t>(value));
    };

    std::uint64_t h = widen(x) * kMulX + widen(y) * kMulY + widen(z) * kMulZ;
    h = (h ^ (h >> 13)) * kMixMul;
    h ^= (h >> 16);
    return static_cast<float>(h & kMask24) / static_cast<float>(kMask24);
}

NoiseVoronoiClimateGenerator::BiomeSite NoiseVoronoiClimateGenerator::computeBiomeSite(const BiomeDefinition& definition,
                                                                                       int regionX,
                                                                                       int regionZ) const noexcept
{
    const float baseRegionWidth = static_cast<float>(biomeSizeInChunks_) * static_cast<float>(chunkSize_);
    const float baseRegionDepth = static_cast<float>(biomeSizeInChunks_) * static_cast<float>(chunkSize_);
    const float footprint = BiomeDefinition::clampFootprintMultiplier(definition.footprintMultiplier);
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

unsigned NoiseVoronoiClimateGenerator::computeSiteSeed(const BiomeDefinition& definition,
                                                       int regionX,
                                                       int regionZ,
                                                       std::size_t siteIndex) const noexcept
{
    unsigned seed = baseSeed_;
    seed = hashCombine(seed, static_cast<unsigned>(regionX * 73856093));
    seed = hashCombine(seed, static_cast<unsigned>(regionZ * 19349663));
    seed = hashCombine(seed, static_cast<unsigned>(siteIndex));
    seed = hashCombine(seed,
                       static_cast<unsigned>(std::hash<std::string>{}(definition.id) & 0xFFFFFFFFu));
    return seed;
}

float NoiseVoronoiClimateGenerator::computeSiteBaseHeight(const BiomeDefinition& definition,
                                                          unsigned siteSeed) const noexcept
{
    const float minHeight = static_cast<float>(definition.minHeight);
    const float maxHeight = static_cast<float>(definition.maxHeight);
    const float t = randomUnit(siteSeed, 0, 0, 0);
    return std::lerp(minHeight, maxHeight, t);
}

void NoiseVoronoiClimateGenerator::applyPostProcessing(ClimateFragment& fragment, int stride) const
{
    const int size = ClimateFragment::kSize;
    const glm::ivec2 baseWorld = fragment.baseWorld();
    const std::size_t totalCells = static_cast<std::size_t>(size * size);

    auto withinBounds = [](int value, int limit) noexcept { return value >= 0 && value < limit; };
    auto cellIndex = [size](int x, int z) noexcept -> std::size_t {
        return static_cast<std::size_t>(z) * size + static_cast<std::size_t>(x);
    };

    std::vector<float> originalHeight(totalCells, 0.0f);
    std::vector<float> originalRoughness(totalCells, 0.0f);
    std::vector<float> originalHills(totalCells, 0.0f);
    std::vector<float> originalMountains(totalCells, 0.0f);
    std::vector<float> originalKeep(totalCells, 0.0f);
    std::vector<glm::vec2> originalSitePos(totalCells, glm::vec2(0.0f));
    std::vector<glm::vec2> originalSiteHalf(totalCells, glm::vec2(1.0f));
    std::vector<BiomeBlend> originalPrimary(totalCells);

    for (int localZ = 0; localZ < size; ++localZ)
    {
        for (int localX = 0; localX < size; ++localX)
        {
            ClimateSample& sample = fragment.sample(localX, localZ);
            const std::size_t idx = cellIndex(localX, localZ);
            originalHeight[idx] = sample.aggregatedHeight;
            originalRoughness[idx] = sample.aggregatedRoughness;
            originalHills[idx] = sample.aggregatedHills;
            originalMountains[idx] = sample.aggregatedMountains;
            originalKeep[idx] = sample.keepOriginalMix;
            originalSitePos[idx] = sample.dominantSitePos;
            originalSiteHalf[idx] = sample.dominantSiteHalfExtents;
            if (sample.blendCount > 0)
            {
                originalPrimary[idx] = sample.blends[0];
            }
            else
            {
                originalPrimary[idx] = BiomeBlend{};
            }
        }
    }

    const std::array<glm::ivec2, 4> kNeighborOffsets{glm::ivec2{-1, 0}, glm::ivec2{1, 0}, glm::ivec2{0, -1},
                                                     glm::ivec2{0, 1}};

    for (int localZ = 0; localZ < size; ++localZ)
    {
        for (int localX = 0; localX < size; ++localX)
        {
            ClimateSample& sample = fragment.sample(localX, localZ);
            const std::size_t idx = cellIndex(localX, localZ);

            if (sample.blendCount == 0)
            {
                continue;
            }

            const BiomeBlend& primary = originalPrimary[idx];
            const BiomeDefinition* dominantBiome = primary.biome ? primary.biome : sample.dominantBiome();
            if (!dominantBiome)
            {
                continue;
            }

            float neighborHeightSum = 0.0f;
            int neighborCount = 0;
            bool hasForeignNeighbor = false;
            float lowestNeighborHeight = std::numeric_limits<float>::infinity();
            std::size_t lowestNeighborIdx = idx;

            for (const glm::ivec2& offset : kNeighborOffsets)
            {
                const int nx = localX + offset.x;
                const int nz = localZ + offset.y;
                if (!withinBounds(nx, size) || !withinBounds(nz, size))
                {
                    continue;
                }

                const std::size_t nidx = cellIndex(nx, nz);
                neighborHeightSum += originalHeight[nidx];
                neighborCount++;

                const BiomeBlend& neighborPrimary = originalPrimary[nidx];
                if (neighborPrimary.biome && neighborPrimary.biome != dominantBiome)
                {
                    hasForeignNeighbor = true;
                    if (originalHeight[nidx] < lowestNeighborHeight)
                    {
                        lowestNeighborHeight = originalHeight[nidx];
                        lowestNeighborIdx = nidx;
                    }
                }
            }

            const float baseHeight = originalHeight[idx];
            if (neighborCount > 0)
            {
                const float neighborAvg = neighborHeightSum / static_cast<float>(neighborCount);
                const float heightDelta = std::abs(baseHeight - neighborAvg);
                const float blendFactor = std::clamp(heightDelta / 112.0f, 0.0f, 0.55f);
                sample.aggregatedHeight = glm::mix(baseHeight, neighborAvg, blendFactor);
                if (sample.blendCount > 0)
                {
                    sample.blends[0].height = glm::mix(primary.height, neighborAvg, blendFactor);
                }
            }

            if (hasForeignNeighbor && lowestNeighborIdx != idx && lowestNeighborHeight < std::numeric_limits<float>::infinity())
            {
                const float originalDelta = baseHeight - lowestNeighborHeight;
                if (originalDelta > 18.0f)
                {
                    const BiomeBlend& neighborBlend = originalPrimary[lowestNeighborIdx];
                    if (neighborBlend.biome)
                    {
                        sample.blends[0] = neighborBlend;
                        sample.blends[0].weight = std::max(neighborBlend.weight, 0.9f);
                        sample.blendCount = std::max<std::size_t>(sample.blendCount, 1);
                        sample.aggregatedHeight = glm::mix(sample.aggregatedHeight, lowestNeighborHeight, 0.75f);
                        sample.aggregatedRoughness =
                            glm::mix(originalRoughness[idx], originalRoughness[lowestNeighborIdx], 0.6f);
                        sample.aggregatedHills =
                            glm::mix(originalHills[idx], originalHills[lowestNeighborIdx], 0.6f);
                        sample.aggregatedMountains =
                            glm::mix(originalMountains[idx], originalMountains[lowestNeighborIdx], 0.6f);
                        sample.keepOriginalMix = originalKeep[lowestNeighborIdx];
                        sample.dominantSitePos = originalSitePos[lowestNeighborIdx];
                        sample.dominantSiteHalfExtents = originalSiteHalf[lowestNeighborIdx];
                        dominantBiome = neighborBlend.biome;
                        const float clampedWeight = std::clamp(sample.blends[0].weight, 0.0f, 1.0f);
                        sample.blends[0].weight = clampedWeight;
                    }
                }
            }

            const glm::vec2 siteHalf = glm::max(sample.dominantSiteHalfExtents, glm::vec2(1.0f));
            const float radius = std::max(siteHalf.x, siteHalf.y);
            if (radius > 0.01f)
            {
                const float worldX = static_cast<float>(baseWorld.x) + static_cast<float>(localX * stride) + 0.5f;
                const float worldZ = static_cast<float>(baseWorld.y) + static_cast<float>(localZ * stride) + 0.5f;
                const glm::vec2 delta = glm::vec2(worldX, worldZ) - sample.dominantSitePos;
                const float distance = glm::length(delta);
                const float centerFactor =
                    std::clamp(1.0f - distance / (radius + static_cast<float>(stride)), 0.0f, 1.0f);
                if (centerFactor > 0.0f && sample.blendCount > 0)
                {
                    const float centerBias = centerFactor * centerFactor;
                    const float targetHeight =
                        originalPrimary[idx].biome == sample.blends[0].biome ? originalPrimary[idx].height
                                                                             : sample.blends[0].height;
                    sample.aggregatedHeight =
                        glm::mix(sample.aggregatedHeight, targetHeight, centerBias * 0.45f);
                    sample.blends[0].height = glm::mix(sample.blends[0].height, targetHeight, centerBias * 0.45f);
                }
            }

            const float keepMix = std::clamp(sample.keepOriginalMix, 0.0f, 1.0f);
            if (keepMix > 0.0f)
            {
                const float preserved = originalHeight[idx];
                sample.aggregatedHeight = glm::mix(sample.aggregatedHeight, preserved, keepMix * 0.5f);
                if (sample.blendCount > 0)
                {
                    sample.blends[0].height = glm::mix(sample.blends[0].height, preserved, keepMix * 0.5f);
                }
            }

            const float patchChance =
                std::clamp(1.0f - sample.keepOriginalMix, 0.0f, 1.0f) * 0.08f;
            const int patchSalt = static_cast<int>(baseWorld.x + localX * stride);
            if (hashToUnitFloat(static_cast<int>(sample.aggregatedHeight * 31.0f), patchSalt,
                                static_cast<int>(baseWorld.y + localZ * stride)) < patchChance)
            {
                sample.aggregatedHeight =
                    glm::mix(sample.aggregatedHeight, originalHeight[idx], 0.35f);
                if (sample.blendCount > 0)
                {
                    sample.blends[0].height =
                        glm::mix(sample.blends[0].height, originalHeight[idx], 0.35f);
                }
            }

            sample.keepOriginalMix = std::clamp(sample.keepOriginalMix, 0.0f, 1.0f);
        }
    }
}

void NoiseVoronoiClimateGenerator::populateBlends(int worldX, int worldZ, ClimateSample& outSample)
{
    const int chunkX = floorDiv(worldX, chunkSize_);
    const int chunkZ = floorDiv(worldZ, chunkSize_);
    const int biomeRegionX = floorDiv(chunkX, biomeSizeInChunks_);
    const int biomeRegionZ = floorDiv(chunkZ, biomeSizeInChunks_);

    std::vector<CandidateSite> candidates;
    candidates.reserve(biomeRegionCandidateCapacity_);

    const glm::vec2 columnPosition{static_cast<float>(worldX) + 0.5f, static_cast<float>(worldZ) + 0.5f};

    for (int regionOffsetZ = -biomeRegionSearchRadius_; regionOffsetZ <= biomeRegionSearchRadius_; ++regionOffsetZ)
    {
        for (int regionOffsetX = -biomeRegionSearchRadius_; regionOffsetX <= biomeRegionSearchRadius_; ++regionOffsetX)
        {
            const int regionX = biomeRegionX + regionOffsetX;
            const int regionZ = biomeRegionZ + regionOffsetZ;
            const BiomeDefinition& definition = biomeForRegion(regionX, regionZ);
            const BiomeSite site = computeBiomeSite(definition, regionX, regionZ);

            const std::size_t siteIndex = candidates.size();
            CandidateSite candidate{};
            candidate.biome = &definition;
            candidate.positionXZ = site.worldPosXZ;
            candidate.halfExtents = site.halfExtents;
            const glm::vec2 delta = columnPosition - candidate.positionXZ;
            const glm::vec2 halfExtents = glm::max(candidate.halfExtents, glm::vec2(1.0f));
            const glm::vec2 normalizedDelta = delta / halfExtents;
            candidate.offsetXZ = delta;
            candidate.distanceSquared = glm::dot(delta, delta);
            candidate.normalizedDistance = glm::length(normalizedDelta);
            candidate.siteSeed = computeSiteSeed(definition, regionX, regionZ, siteIndex);
            candidate.baseHeight = computeSiteBaseHeight(definition, candidate.siteSeed);
            candidates.push_back(candidate);
        }
    }

    if (candidates.empty())
    {
        outSample.blendCount = 0;
        return;
    }

    constexpr std::size_t kMaxConsidered = 4;
    const std::size_t candidateCount = candidates.size();
    std::size_t sitesToConsider = std::min<std::size_t>(kMaxConsidered, candidateCount);

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

    std::partial_sort(candidates.begin(), candidates.begin() + sitesToConsider,
                      candidates.begin() + candidateCount, candidateLess);

    std::array<float, kMaxConsidered> weights{};
    std::array<float, kMaxConsidered> falloffFactors{};
    float totalWeight = 0.0f;
    for (std::size_t i = 0; i < sitesToConsider; ++i)
    {
        const CandidateSite& candidate = candidates[i];
        const glm::vec2 halfExtents = glm::max(candidate.halfExtents, glm::vec2(1.0f));
        const glm::vec2 axisRatio = glm::abs(candidate.offsetXZ) / halfExtents;
        const glm::vec2 axisClamped = glm::clamp(axisRatio, glm::vec2(0.0f), glm::vec2(1.0f));

        const float closenessX = 1.0f - axisClamped.x;
        const float closenessZ = 1.0f - axisClamped.y;

        float axisWeight = applyInterpolationCurve(closenessX, candidate.biome->interpolationCurve);
        axisWeight *= applyInterpolationCurve(closenessZ, candidate.biome->interpolationCurve);
        axisWeight = std::clamp(axisWeight, 0.0f, 1.0f);

        const float radial = std::pow(std::clamp(1.0f - candidate.normalizedDistance, 0.0f, 1.0f), kFalloffExponent);

        float combinedFalloff = axisWeight * radial;
        combinedFalloff *= std::max(candidate.biome->interpolationWeight, kInterpolationStrengthEpsilon);
        combinedFalloff = std::clamp(combinedFalloff, 0.0f, 1.0f);

        if (combinedFalloff <= std::numeric_limits<float>::epsilon())
        {
            weights[i] = 0.0f;
            falloffFactors[i] = 0.0f;
            continue;
        }

        const float distance = std::sqrt(std::max(candidate.distanceSquared, 0.0f));
        float weight = combinedFalloff / std::max(distance, kDistanceBias);
        weights[i] = weight;
        falloffFactors[i] = combinedFalloff;
        totalWeight += weight;
    }

    if (totalWeight <= std::numeric_limits<float>::epsilon())
    {
        weights.fill(0.0f);
        falloffFactors.fill(0.0f);
        weights[0] = 1.0f;
        falloffFactors[0] = 1.0f;
        sitesToConsider = std::min<std::size_t>(1, sitesToConsider);
        totalWeight = 1.0f;
    }

    outSample.blendCount = sitesToConsider;
    float aggregatedHeight = 0.0f;
    float aggregatedRoughness = 0.0f;
    float aggregatedHills = 0.0f;
    float aggregatedMountains = 0.0f;
    float keepOriginalMix = 0.0f;
    float amplitudeWeightSum = 0.0f;

    for (std::size_t i = 0; i < sitesToConsider; ++i)
    {
        const CandidateSite& candidate = candidates[i];
        const float normalizedWeight = weights[i] / totalWeight;
        const float amplitudeScale = std::clamp(falloffFactors[i], 0.0f, 1.0f);

        BiomeBlend blend{};
        blend.biome = candidate.biome;
        blend.weight = normalizedWeight;
        unsigned seed = hashCombine(candidate.siteSeed, static_cast<unsigned>(i));
        blend.seed = seed;
        blend.height = candidate.baseHeight;
        blend.roughness = candidate.biome->roughness * amplitudeScale;
        blend.hills = candidate.biome->hills * amplitudeScale;
        blend.mountains = candidate.biome->mountains * amplitudeScale;
        blend.normalizedDistance = candidate.normalizedDistance;

        outSample.blends[i] = blend;

        aggregatedHeight += candidate.baseHeight * amplitudeScale * normalizedWeight;
        aggregatedRoughness += blend.roughness * normalizedWeight;
        aggregatedHills += blend.hills * normalizedWeight;
        aggregatedMountains += blend.mountains * normalizedWeight;
        if (blend.biome)
        {
            keepOriginalMix += blend.biome->keepOriginalTerrain * amplitudeScale * normalizedWeight;
        }
        amplitudeWeightSum += amplitudeScale * normalizedWeight;
    }

    std::sort(outSample.blends.begin(), outSample.blends.begin() + outSample.blendCount,
              [](const BiomeBlend& lhs, const BiomeBlend& rhs) {
                  return lhs.weight > rhs.weight;
              });

    if (amplitudeWeightSum > std::numeric_limits<float>::epsilon())
    {
        aggregatedHeight /= amplitudeWeightSum;
        keepOriginalMix /= amplitudeWeightSum;
    }

    outSample.aggregatedHeight = aggregatedHeight;
    outSample.aggregatedRoughness = aggregatedRoughness;
    outSample.aggregatedHills = aggregatedHills;
    outSample.aggregatedMountains = aggregatedMountains;
    outSample.keepOriginalMix = keepOriginalMix;
    if (sitesToConsider > 0)
    {
        const CandidateSite& dominantSite = candidates[0];
        outSample.dominantSitePos = dominantSite.positionXZ;
        outSample.dominantSiteHalfExtents = dominantSite.halfExtents;
    }
    else
    {
        outSample.dominantSitePos = glm::vec2(0.0f);
        outSample.dominantSiteHalfExtents = glm::vec2(0.0f);
    }
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

    applyPostProcessing(fragment, 1);
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
        it->second.lruIt = lru_.emplace(lru_.begin(), key);
        touch(it->second);
        evictIfNeeded();
        result = it->second.fragment.get();
    }

    return *result;
}

void ClimateMap::touch(FragmentCacheEntry& entry) const
{
    lru_.erase(entry.lruIt);
    entry.lruIt = lru_.emplace(lru_.begin(), entry.fragment->fragmentCoord());
}

void ClimateMap::evictIfNeeded() const
{
    while (fragments_.size() > maxFragments_)
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

