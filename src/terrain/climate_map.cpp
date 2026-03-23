#include "terrain/climate_map.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/noise.hpp>

namespace terrain
{
namespace
{
constexpr bool kLogCoastTransitions = false;
constexpr float kCoastDistanceFieldRange = 96.0f;
constexpr float kEpsilon = 1e-6f;
constexpr float kDiagonalStep = 1.41421356237f;

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

float unitPerlinNoise(int worldX, int worldZ, unsigned seed, float frequency) noexcept
{
    const float offsetX = static_cast<float>(seed & 0xFFFFu) * 0.013f;
    const float offsetZ = static_cast<float>((seed >> 16) & 0xFFFFu) * 0.017f;
    const glm::vec2 sample{(static_cast<float>(worldX) + offsetX) * frequency,
                           (static_cast<float>(worldZ) + offsetZ) * frequency};
    return glm::clamp(glm::perlin(sample) * 0.5f + 0.5f, 0.0f, 1.0f);
}

float smoothFactorFromDistance(float distance, float range) noexcept
{
    if (!std::isfinite(distance))
    {
        return 0.0f;
    }
    if (range <= 0.0f)
    {
        return 0.0f;
    }

    const float t = glm::clamp(distance / range, 0.0f, 1.0f);
    return 1.0f - (t * t * (3.0f - 2.0f * t));
}

void smoothDomainMask(std::vector<std::uint8_t>& mask, int width, int height)
{
    std::vector<std::uint8_t> original = mask;
    for (int z = 0; z < height; ++z)
    {
        for (int x = 0; x < width; ++x)
        {
            int oceanCount = 0;
            int samples = 0;
            for (int dz = -1; dz <= 1; ++dz)
            {
                const int nz = z + dz;
                if (nz < 0 || nz >= height)
                {
                    continue;
                }
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int nx = x + dx;
                    if (nx < 0 || nx >= width)
                    {
                        continue;
                    }
                    oceanCount += original[static_cast<std::size_t>(nz) * static_cast<std::size_t>(width)
                                           + static_cast<std::size_t>(nx)] != 0
                                      ? 1
                                      : 0;
                    ++samples;
                }
            }

            std::uint8_t& cell = mask[static_cast<std::size_t>(z) * static_cast<std::size_t>(width)
                                      + static_cast<std::size_t>(x)];
            if (oceanCount >= 6)
            {
                cell = 1;
            }
            else if (oceanCount <= 3)
            {
                cell = 0;
            }
            else if (samples == 0)
            {
                cell = 0;
            }
        }
    }
}

std::vector<float> computeDistanceField(const std::vector<std::uint8_t>& mask,
                                        int width,
                                        int height,
                                        std::uint8_t targetValue)
{
    const std::size_t area = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<float> distances(area, std::numeric_limits<float>::infinity());

    using QueueEntry = std::pair<float, int>;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> frontier;

    for (int z = 0; z < height; ++z)
    {
        for (int x = 0; x < width; ++x)
        {
            const std::size_t idx = static_cast<std::size_t>(z) * static_cast<std::size_t>(width)
                                    + static_cast<std::size_t>(x);
            if (mask[idx] == targetValue)
            {
                distances[idx] = 0.0f;
                frontier.emplace(0.0f, static_cast<int>(idx));
            }
        }
    }

    constexpr std::array<int, 8> kDx{1, -1, 0, 0, 1, 1, -1, -1};
    constexpr std::array<int, 8> kDz{0, 0, 1, -1, 1, -1, 1, -1};

    while (!frontier.empty())
    {
        const auto [distance, flatIndex] = frontier.top();
        frontier.pop();

        if (distance > distances[static_cast<std::size_t>(flatIndex)] + kEpsilon)
        {
            continue;
        }

        const int z = flatIndex / width;
        const int x = flatIndex % width;

        for (std::size_t i = 0; i < kDx.size(); ++i)
        {
            const int nx = x + kDx[i];
            const int nz = z + kDz[i];
            if (nx < 0 || nx >= width || nz < 0 || nz >= height)
            {
                continue;
            }

            const float step = (kDx[i] == 0 || kDz[i] == 0) ? 1.0f : kDiagonalStep;
            const float nextDistance = distance + step;
            const std::size_t nextIdx =
                static_cast<std::size_t>(nz) * static_cast<std::size_t>(width) + static_cast<std::size_t>(nx);
            if (nextDistance + kEpsilon < distances[nextIdx])
            {
                distances[nextIdx] = nextDistance;
                frontier.emplace(nextDistance, static_cast<int>(nextIdx));
            }
        }
    }

    return distances;
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
    maxChunkCacheEntries_ = std::clamp<std::size_t>(
        static_cast<std::size_t>((neighborRadius_ * 2 + 1) * (neighborRadius_ * 2 + 1) * 64),
        512u,
        4096u);

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

void NoiseVoronoiClimateGenerator::touchChunkSeedCacheEntry(ChunkSeedCacheEntry& entry) const
{
    if (!entry.inLru)
    {
        return;
    }
    chunkCacheLru_.splice(chunkCacheLru_.begin(), chunkCacheLru_, entry.lruIt);
}

void NoiseVoronoiClimateGenerator::evictChunkSeedCacheIfNeeded() const
{
    while (chunkCache_.size() > maxChunkCacheEntries_ && !chunkCacheLru_.empty())
    {
        const glm::ivec2 key = chunkCacheLru_.back();
        chunkCacheLru_.pop_back();
        chunkCache_.erase(key);
    }
}

NoiseVoronoiClimateGenerator::ChunkSeeds
NoiseVoronoiClimateGenerator::buildChunkSeeds(int chunkX, int chunkZ) const
{
    ChunkSeeds result{};
    const int baseX = chunkX * chunkSpan_;
    const int baseZ = chunkZ * chunkSpan_;

    unsigned seedValue = baseSeed_;
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
                                                        std::vector<BiomeSeed>& outCandidates) const
{
    const int chunkX = floorDiv(worldPos.x, chunkSpan_);
    const int chunkZ = floorDiv(worldPos.y, chunkSpan_);
    for (int dz = -neighborRadius_; dz <= neighborRadius_; ++dz)
    {
        for (int dx = -neighborRadius_; dx <= neighborRadius_; ++dx)
        {
            const glm::ivec2 key{chunkX + dx, chunkZ + dz};
            std::lock_guard<std::mutex> lock(chunkMutex_);
            auto it = chunkCache_.find(key);
            if (it == chunkCache_.end())
            {
                ChunkSeedCacheEntry entry{};
                entry.seeds = buildChunkSeeds(key.x, key.y);
                entry.lruIt = chunkCacheLru_.emplace(chunkCacheLru_.begin(), key);
                entry.inLru = true;
                auto [insertedIt, inserted] = chunkCache_.emplace(key, std::move(entry));
                (void)inserted;
                it = insertedIt;
                evictChunkSeedCacheIfNeeded();
            }

            touchChunkSeedCacheEntry(it->second);
            const std::size_t previousSize = outCandidates.size();
            outCandidates.resize(previousSize + it->second.seeds.seeds.size());
            std::copy(it->second.seeds.seeds.begin(),
                      it->second.seeds.seeds.end(),
                      outCandidates.begin() + static_cast<std::ptrdiff_t>(previousSize));
        }
    }
}

void NoiseVoronoiClimateGenerator::accumulateSample(const glm::ivec2& worldPos,
                                                    ClimateSample& outSample,
                                                    SampleComposition* outComposition) const
{
    std::vector<BiomeSeed> rawCandidates;
    rawCandidates.reserve(static_cast<std::size_t>((neighborRadius_ * 2 + 1) * (neighborRadius_ * 2 + 1) * 48));
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

    for (const BiomeSeed& candidate : rawCandidates)
    {
        const float distSq = lengthSquared(worldPos, candidate.position);
        const float distance = std::sqrt(distSq);
        const float normalized = distance / std::max(candidate.radius, 1.0f);
        const float blended = std::clamp(1.0f - normalized, 0.0f, 1.0f);
        float influence = smoothStep(blended);
        candidates.push_back(CandidateInfo{&candidate,
                                           distance,
                                           std::max(candidate.radius, 1.0f),
                                           normalized,
                                           influence});
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

    SampleComposition scratch{};

    if (weighted.empty())
    {
        const CandidateInfo* nearestCandidate = nullptr;
        for (const CandidateInfo& candidate : candidates)
        {
            if (!candidate.seed || !candidate.seed->biome)
            {
                continue;
            }

            if (nearestCandidate == nullptr ||
                candidate.normalized < nearestCandidate->normalized ||
                (candidate.normalized == nearestCandidate->normalized &&
                 candidate.distance < nearestCandidate->distance))
            {
                nearestCandidate = &candidate;
            }
        }

        ClimateSample fallback{};
        fallback.blendCount = 1;
        const BiomeDefinition* fallbackBiome =
            (nearestCandidate && nearestCandidate->seed && nearestCandidate->seed->biome)
                ? nearestCandidate->seed->biome
                : (biomeSelection_.empty() ? &biomeDatabase_.definitionByIndex(0) : biomeSelection_.front());
        const BiomeDefinition& biome = *fallbackBiome;
        const float normalizedDistance = nearestCandidate ? nearestCandidate->normalized : 0.0f;
        const float falloff = nearestCandidate ? nearestCandidate->radius : biome.maxRadius();
        const glm::vec2 sitePosition =
            nearestCandidate
                ? glm::vec2(static_cast<float>(nearestCandidate->seed->position.x),
                            static_cast<float>(nearestCandidate->seed->position.y))
                : glm::vec2(static_cast<float>(worldPos.x), static_cast<float>(worldPos.y));
        const float baseHeight =
            nearestCandidate ? nearestCandidate->seed->baseHeight : static_cast<float>(biome.minHeight);
        BiomeBlend blend{};
        blend.biome = &biome;
        blend.weight = 1.0f;
        blend.height = biome.applyHeightLimits(baseHeight, normalizedDistance);
        blend.roughness = biome.roughness;
        blend.hills = biome.hills;
        blend.mountains = biome.mountains;
        blend.normalizedDistance = normalizedDistance;
        blend.seed = nearestCandidate
                         ? hashCombine(baseSeed_,
                                       hashCombine(static_cast<unsigned>(nearestCandidate->seed->position.x),
                                                   static_cast<unsigned>(nearestCandidate->seed->position.y)))
                         : hashCombine(baseSeed_, static_cast<unsigned>(biome.minHeight));
        blend.falloff = falloff;
        blend.sitePosition = sitePosition;
        fallback.blends[0] = blend;
        fallback.representativeBiome = &biome;
        fallback.representativeWeight = 1.0f;
        fallback.aggregatedHeight = blend.height;
        fallback.aggregatedRoughness = blend.roughness;
        fallback.aggregatedHills = blend.hills;
        fallback.aggregatedMountains = blend.mountains;
        fallback.keepOriginalMix = std::clamp(biome.keepOriginalTerrain, 0.0f, 1.0f);
        fallback.dominantSitePos = sitePosition;
        fallback.dominantSiteHalfExtents = glm::vec2(falloff);
        fallback.dominantIsOcean = biome.isOcean();
        fallback.distanceToCoast = std::numeric_limits<float>::infinity();
        fallback.signedDistanceToCoast =
            biome.isOcean() ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        fallback.landBaseHeight = blend.height;
        fallback.oceanBaseHeight = blend.height;

        if (biome.isOcean())
        {
            scratch.oceanWeight = 1.0f;
            scratch.oceanHeight = blend.height;
            scratch.oceanRoughness = blend.roughness;
            scratch.oceanHills = blend.hills;
            scratch.oceanMountains = blend.mountains;
            scratch.oceanKeepOriginal = fallback.keepOriginalMix;
            scratch.oceanRepresentativeBiome = &biome;
            scratch.oceanRepresentativeWeight = 1.0f;
            scratch.oceanSitePos = blend.sitePosition;
            scratch.oceanSiteRadius = blend.falloff;
            scratch.prefersOcean = true;
        }
        else
        {
            scratch.landWeight = 1.0f;
            scratch.landHeight = blend.height;
            scratch.landRoughness = blend.roughness;
            scratch.landHills = blend.hills;
            scratch.landMountains = blend.mountains;
            scratch.landKeepOriginal = fallback.keepOriginalMix;
            scratch.landRepresentativeBiome = &biome;
            scratch.landRepresentativeWeight = 1.0f;
            scratch.landSitePos = blend.sitePosition;
            scratch.landSiteRadius = blend.falloff;
        }

        outSample = fallback;
        if (outComposition)
        {
            *outComposition = scratch;
        }
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

        if (biome.isOcean())
        {
            scratch.oceanWeight += normalizedWeight;
            scratch.oceanHeight += blend.height * normalizedWeight;
            scratch.oceanRoughness += blend.roughness * normalizedWeight;
            scratch.oceanHills += blend.hills * normalizedWeight;
            scratch.oceanMountains += blend.mountains * normalizedWeight;
            scratch.oceanKeepOriginal += std::clamp(biome.keepOriginalTerrain, 0.0f, 1.0f) * normalizedWeight;

            if (normalizedWeight > scratch.oceanRepresentativeWeight)
            {
                scratch.oceanRepresentativeBiome = &biome;
                scratch.oceanRepresentativeWeight = normalizedWeight;
                scratch.oceanSitePos = blend.sitePosition;
                scratch.oceanSiteRadius = blend.falloff;
            }
        }
        else
        {
            scratch.landWeight += normalizedWeight;
            scratch.landHeight += blend.height * normalizedWeight;
            scratch.landRoughness += blend.roughness * normalizedWeight;
            scratch.landHills += blend.hills * normalizedWeight;
            scratch.landMountains += blend.mountains * normalizedWeight;
            scratch.landKeepOriginal += std::clamp(biome.keepOriginalTerrain, 0.0f, 1.0f) * normalizedWeight;

            if (normalizedWeight > scratch.landRepresentativeWeight)
            {
                scratch.landRepresentativeBiome = &biome;
                scratch.landRepresentativeWeight = normalizedWeight;
                scratch.landSitePos = blend.sitePosition;
                scratch.landSiteRadius = blend.falloff;
            }
        }
    }

    outSample.aggregatedHeight = aggregatedHeight;
    outSample.aggregatedRoughness = aggregatedRoughness;
    outSample.aggregatedHills = aggregatedHills;
    outSample.aggregatedMountains = aggregatedMountains;
    outSample.keepOriginalMix = std::clamp(keepOriginal, 0.0f, 1.0f);

    if (scratch.landWeight > kEpsilon)
    {
        scratch.landHeight /= scratch.landWeight;
        scratch.landRoughness /= scratch.landWeight;
        scratch.landHills /= scratch.landWeight;
        scratch.landMountains /= scratch.landWeight;
        scratch.landKeepOriginal /= scratch.landWeight;
    }

    if (scratch.oceanWeight > kEpsilon)
    {
        scratch.oceanHeight /= scratch.oceanWeight;
        scratch.oceanRoughness /= scratch.oceanWeight;
        scratch.oceanHills /= scratch.oceanWeight;
        scratch.oceanMountains /= scratch.oceanWeight;
        scratch.oceanKeepOriginal /= scratch.oceanWeight;
    }

    scratch.prefersOcean = scratch.oceanWeight > scratch.landWeight;
    const bool representativeOcean = scratch.prefersOcean && scratch.oceanRepresentativeBiome;

    if (representativeOcean)
    {
        outSample.representativeBiome = scratch.oceanRepresentativeBiome;
        outSample.representativeWeight = scratch.oceanRepresentativeWeight;
        outSample.dominantSitePos = scratch.oceanSitePos;
        outSample.dominantSiteHalfExtents = glm::vec2(scratch.oceanSiteRadius);
        outSample.dominantIsOcean = true;
    }
    else
    {
        outSample.representativeBiome = scratch.landRepresentativeBiome ? scratch.landRepresentativeBiome
                                                                        : scratch.oceanRepresentativeBiome;
        outSample.representativeWeight = scratch.landRepresentativeWeight > 0.0f
                                             ? scratch.landRepresentativeWeight
                                             : scratch.oceanRepresentativeWeight;
        const glm::vec2 sitePos = scratch.landRepresentativeBiome ? scratch.landSitePos : scratch.oceanSitePos;
        const float siteRadius = scratch.landRepresentativeBiome ? scratch.landSiteRadius : scratch.oceanSiteRadius;
        outSample.dominantSitePos = sitePos;
        outSample.dominantSiteHalfExtents = glm::vec2(siteRadius);
        outSample.dominantIsOcean = scratch.landRepresentativeBiome == nullptr
                                    && scratch.oceanRepresentativeBiome != nullptr;
    }

    outSample.distanceToCoast = std::numeric_limits<float>::infinity();

    if (outComposition)
    {
        *outComposition = scratch;
    }
}

void NoiseVoronoiClimateGenerator::generate(ClimateFragment& fragment)
{
    const glm::ivec2 baseWorld = fragment.baseWorld();
    constexpr int kCoreSize = ClimateFragment::kSize;
    const int halo = std::max(maxTransitionWidth_ + 8, static_cast<int>(std::ceil(kCoastDistanceFieldRange)));
    const int extendedSize = kCoreSize + halo * 2;

    std::vector<SampleComposition> compositions(static_cast<std::size_t>(kCoreSize) * static_cast<std::size_t>(kCoreSize));

    for (int localZ = 0; localZ < ClimateFragment::kSize; ++localZ)
    {
        for (int localX = 0; localX < ClimateFragment::kSize; ++localX)
        {
            ClimateSample& sample = fragment.sample(localX, localZ);
            const glm::ivec2 worldPos{baseWorld.x + localX, baseWorld.y + localZ};
            const std::size_t coreIndex =
                static_cast<std::size_t>(localZ) * static_cast<std::size_t>(kCoreSize) + static_cast<std::size_t>(localX);
            accumulateSample(worldPos, sample, &compositions[coreIndex]);
        }
    }

    std::vector<std::uint8_t> domainMask(static_cast<std::size_t>(extendedSize) * static_cast<std::size_t>(extendedSize), 0);
    const auto extendedIndex = [extendedSize](int x, int z) -> std::size_t {
        return static_cast<std::size_t>(z) * static_cast<std::size_t>(extendedSize) + static_cast<std::size_t>(x);
    };

    for (int localZ = -halo; localZ < kCoreSize + halo; ++localZ)
    {
        for (int localX = -halo; localX < kCoreSize + halo; ++localX)
        {
            bool prefersOcean = false;
            if (localX >= 0 && localX < kCoreSize && localZ >= 0 && localZ < kCoreSize)
            {
                const std::size_t coreIndex =
                    static_cast<std::size_t>(localZ) * static_cast<std::size_t>(kCoreSize) + static_cast<std::size_t>(localX);
                prefersOcean = compositions[coreIndex].prefersOcean;
            }
            else
            {
                ClimateSample haloSample{};
                SampleComposition haloComposition{};
                const glm::ivec2 worldPos{baseWorld.x + localX, baseWorld.y + localZ};
                accumulateSample(worldPos, haloSample, &haloComposition);
                prefersOcean = haloComposition.prefersOcean;
            }

            domainMask[extendedIndex(localX + halo, localZ + halo)] = prefersOcean ? 1u : 0u;
        }
    }

    smoothDomainMask(domainMask, extendedSize, extendedSize);
    const std::vector<float> distanceToLand = computeDistanceField(domainMask, extendedSize, extendedSize, 0u);
    const std::vector<float> distanceToOcean = computeDistanceField(domainMask, extendedSize, extendedSize, 1u);

    for (int localZ = 0; localZ < kCoreSize; ++localZ)
    {
        for (int localX = 0; localX < kCoreSize; ++localX)
        {
            ClimateSample& sample = fragment.sample(localX, localZ);
            const std::size_t coreIndex =
                static_cast<std::size_t>(localZ) * static_cast<std::size_t>(kCoreSize) + static_cast<std::size_t>(localX);
            const SampleComposition& composition = compositions[coreIndex];

            bool useOceanDomain = domainMask[extendedIndex(localX + halo, localZ + halo)] != 0;
            if ((useOceanDomain && composition.oceanWeight <= kEpsilon)
                || (!useOceanDomain && composition.landWeight <= kEpsilon))
            {
                useOceanDomain = composition.prefersOcean;
            }

            const float rawDistance =
                useOceanDomain ? distanceToLand[extendedIndex(localX + halo, localZ + halo)]
                               : distanceToOcean[extendedIndex(localX + halo, localZ + halo)];
            const float coastDistance =
                (std::isfinite(rawDistance) && rawDistance <= static_cast<float>(halo)) ? rawDistance
                                                                                        : std::numeric_limits<float>::infinity();
            const float signedCoastDistance = std::isfinite(coastDistance)
                                                  ? (useOceanDomain ? -coastDistance : coastDistance)
                                                  : (useOceanDomain ? -std::numeric_limits<float>::infinity()
                                                                    : std::numeric_limits<float>::infinity());

            const bool hasRequestedGroup =
                useOceanDomain ? (composition.oceanWeight > kEpsilon) : (composition.landWeight > kEpsilon);
            const bool fallbackToOcean = !hasRequestedGroup && composition.oceanWeight > kEpsilon;

            const BiomeDefinition* representativeBiome = nullptr;
            float representativeWeight = 0.0f;
            float groupHeight = sample.aggregatedHeight;
            float groupRoughness = sample.aggregatedRoughness;
            float groupHills = sample.aggregatedHills;
            float groupMountains = sample.aggregatedMountains;
            float groupKeepOriginal = sample.keepOriginalMix;
            glm::vec2 dominantSitePos = sample.dominantSitePos;
            float dominantSiteRadius = sample.dominantSiteHalfExtents.x;

            if ((useOceanDomain && !fallbackToOcean) || fallbackToOcean)
            {
                representativeBiome = composition.oceanRepresentativeBiome;
                representativeWeight = composition.oceanRepresentativeWeight;
                groupHeight = composition.oceanWeight > kEpsilon ? composition.oceanHeight : groupHeight;
                groupRoughness = composition.oceanWeight > kEpsilon ? composition.oceanRoughness : groupRoughness;
                groupHills = composition.oceanWeight > kEpsilon ? composition.oceanHills : groupHills;
                groupMountains = composition.oceanWeight > kEpsilon ? composition.oceanMountains : groupMountains;
                groupKeepOriginal = composition.oceanWeight > kEpsilon ? composition.oceanKeepOriginal : groupKeepOriginal;
                dominantSitePos = composition.oceanRepresentativeBiome ? composition.oceanSitePos : dominantSitePos;
                dominantSiteRadius = composition.oceanRepresentativeBiome ? composition.oceanSiteRadius : dominantSiteRadius;
                useOceanDomain = true;
            }
            else if (composition.landWeight > kEpsilon)
            {
                representativeBiome = composition.landRepresentativeBiome;
                representativeWeight = composition.landRepresentativeWeight;
                groupHeight = composition.landHeight;
                groupRoughness = composition.landRoughness;
                groupHills = composition.landHills;
                groupMountains = composition.landMountains;
                groupKeepOriginal = composition.landKeepOriginal;
                dominantSitePos = composition.landRepresentativeBiome ? composition.landSitePos : dominantSitePos;
                dominantSiteRadius = composition.landRepresentativeBiome ? composition.landSiteRadius : dominantSiteRadius;
                useOceanDomain = false;
            }

            if (!representativeBiome)
            {
                representativeBiome = sample.blendCount > 0 ? sample.blends[0].biome : nullptr;
                representativeWeight = sample.blendCount > 0 ? sample.blends[0].weight : 0.0f;
            }

            sample.representativeBiome = representativeBiome;
            sample.representativeWeight = representativeWeight;
            sample.aggregatedHeight = groupHeight;
            sample.aggregatedRoughness = groupRoughness;
            sample.aggregatedHills = groupHills;
            sample.aggregatedMountains = groupMountains;
            sample.keepOriginalMix = glm::clamp(groupKeepOriginal, 0.0f, 1.0f);
            sample.dominantSitePos = dominantSitePos;
            sample.dominantSiteHalfExtents = glm::vec2(dominantSiteRadius);
            sample.dominantIsOcean = useOceanDomain;
            sample.distanceToCoast = coastDistance;
            sample.signedDistanceToCoast = signedCoastDistance;
            sample.landBaseHeight = composition.landWeight > kEpsilon ? composition.landHeight : sample.aggregatedHeight;
            sample.oceanBaseHeight = composition.oceanWeight > kEpsilon ? composition.oceanHeight : sample.aggregatedHeight;
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

        const bool requiresOceanNeighbor =
            sub.biome->generationProperties().isCoastal() || sub.biome->hasFlag("beach");
        if (requiresOceanNeighbor)
        {
            constexpr float kOceanProximityFactor = 2.0f;
            bool hasNearbyOcean = false;
            for (const BiomeSeed& oceanSeed : seeds)
            {
                if (!oceanSeed.biome || !oceanSeed.biome->isOcean())
                {
                    continue;
                }

                const float distanceToOcean =
                    glm::length(glm::vec2(candidatePos - oceanSeed.position));
                if (distanceToOcean <= radius * kOceanProximityFactor)
                {
                    hasNearbyOcean = true;
                    break;
                }
            }

            if (!hasNearbyOcean)
            {
                continue;
            }
        }

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
                oceanSnapshot[idx] = sample.dominantIsOcean ? 1u : 0u;
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

    rebuildNeighborLayers();

    for (int z = 0; z < size; ++z)
    {
        for (int x = 0; x < size; ++x)
        {
            const std::size_t idx = indexFor(x, z);
            const ClimateSample originalSample = fragment.sample(x, z);
            const BiomeDefinition* baseBiome = originalSample.dominantBiome();
            if (!baseBiome || baseBiome->transitionBiomes.empty())
            {
                continue;
            }

            ClimateSample updatedSample = originalSample;
            float strongestTransition = 0.0f;
            const int worldX = baseWorld.x + x;
            const int worldZ = baseWorld.y + z;

            for (const BiomeDefinition::TransitionBiomeDefinition& transition : baseBiome->transitionBiomes)
            {
                if (!transition.biome)
                {
                    continue;
                }

                const BiomeDefinition& target = *transition.biome;
                if (target.hasFlag("beach"))
                {
                    // Beach material placement is handled from shore distance during terrain fill.
                    continue;
                }

                const int radius = std::clamp(transition.width, 0, maxWidth);
                const std::uint16_t neighborMask = layerPtr(radius)[idx];
                const std::uint16_t requiredBits = transition.propertyMask.value();
                const std::uint16_t matched = static_cast<std::uint16_t>(neighborMask & requiredBits);
                const std::uint16_t spread = static_cast<std::uint16_t>(matched | (matched >> 1) | (matched >> 2));
                const std::uint16_t requiredGroups = groupPresenceMask(requiredBits);
                const std::uint16_t availableGroups = groupPresenceMask(spread);
                if ((availableGroups & requiredGroups) != requiredGroups)
                {
                    continue;
                }

                const bool targetIsCoast = target.generationProperties().isCoastal();
                const bool targetIsMountainCoast =
                    targetIsCoast
                    && target.generationProperties().has(BiomeDefinition::GenerationProperties::kMountain)
                    && !target.isOcean();
                const float coastDistance = originalSample.distanceToCoast;
                if (!std::isfinite(coastDistance))
                {
                    continue;
                }

                bool hasOceanNeighbor = false;
                if (targetIsCoast || target.isOcean())
                {
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

                if (target.isOcean() && !originalSample.dominantIsOcean)
                {
                    continue;
                }
                if (targetIsMountainCoast && originalSample.dominantIsOcean)
                {
                    continue;
                }

                const unsigned hashSeed = hashCombine(
                    baseSeed_,
                    hashCombine(static_cast<unsigned>(worldX),
                                hashCombine(static_cast<unsigned>(worldZ),
                                            static_cast<unsigned>(transition.width))));

                const float transitionWidth = static_cast<float>(std::max(transition.width, 1));
                float range = transitionWidth * 6.0f;
                if (target.isOcean())
                {
                    range = std::max(range, 32.0f);
                }
                else if (targetIsMountainCoast)
                {
                    range = std::max(range, 26.0f);
                }
                else if (targetIsCoast)
                {
                    range = std::max(range, 18.0f);
                }
                else
                {
                    range = std::max(range, 12.0f);
                }

                const float edgeNoise = unitPerlinNoise(worldX, worldZ, hashSeed ^ 0xA53C9E21u, 0.03f);
                const float effectiveRange = range * glm::mix(0.85f, 1.15f, edgeNoise);
                float transitionStrength = smoothFactorFromDistance(coastDistance, effectiveRange);
                transitionStrength *= glm::clamp(transition.chance, 0.0f, 1.0f);
                if (transitionStrength <= 0.01f)
                {
                    continue;
                }

                if (transitionStrength > strongestTransition)
                {
                    updatedSample.representativeBiome = &target;
                    updatedSample.representativeWeight =
                        std::max(updatedSample.representativeWeight, transitionStrength);
                    updatedSample.dominantSitePos = glm::vec2(static_cast<float>(worldX), static_cast<float>(worldZ));
                    updatedSample.dominantSiteHalfExtents = glm::vec2(target.maxRadius());
                    strongestTransition = transitionStrength;
                }

                if (kLogCoastTransitions)
                {
                    std::cout << "[CoastTransition] world=(" << worldX << ',' << worldZ << ")"
                              << " base=" << (baseBiome ? baseBiome->id : "<none>")
                              << " target=" << target.id
                              << " coastDistance=" << coastDistance
                              << " strength=" << transitionStrength
                              << '\n';
                }
            }

            fragment.sample(x, z) = updatedSample;
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

ClimateSample ClimateMap::sample(int worldX, int worldZ) const
{
    const int fragmentX = floorDiv(worldX, ClimateFragment::kSize);
    const int fragmentZ = floorDiv(worldZ, ClimateFragment::kSize);
    const glm::ivec2 key{fragmentX, fragmentZ};

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = fragments_.find(key);
        if (it != fragments_.end())
        {
            if (profilingEnabled())
            {
                cacheHits_.fetch_add(1, std::memory_order_relaxed);
            }
            touch(it->second);
            const glm::ivec2 baseWorld = it->second.fragment->baseWorld();
            const int localX = worldX - baseWorld.x;
            const int localZ = worldZ - baseWorld.y;
            return it->second.fragment->sample(localX, localZ);
        }
    }

    if (profilingEnabled())
    {
        cacheMisses_.fetch_add(1, std::memory_order_relaxed);
    }

    auto fragment = std::make_unique<ClimateFragment>(key);
    generator_->generate(*fragment);

    std::lock_guard<std::mutex> lock(mutex_);
    auto existing = fragments_.find(key);
    if (existing != fragments_.end())
    {
        touch(existing->second);
        const glm::ivec2 baseWorld = existing->second.fragment->baseWorld();
        const int localX = worldX - baseWorld.x;
        const int localZ = worldZ - baseWorld.y;
        return existing->second.fragment->sample(localX, localZ);
    }

    FragmentCacheEntry entry{};
    entry.fragment = std::move(fragment);
    entry.lruIt = lru_.emplace(lru_.begin(), key);
    entry.inLru = true;

    auto [it, inserted] = fragments_.emplace(key, std::move(entry));
    (void)inserted;
    if (profilingEnabled())
    {
        cacheFills_.fetch_add(1, std::memory_order_relaxed);
    }
    evictIfNeeded();

    const glm::ivec2 baseWorld = it->second.fragment->baseWorld();
    const int localX = worldX - baseWorld.x;
    const int localZ = worldZ - baseWorld.y;
    return it->second.fragment->sample(localX, localZ);
}

void ClimateMap::setProfilingEnabled(bool enabled) noexcept
{
    profilingEnabled_.store(enabled, std::memory_order_release);
}

bool ClimateMap::profilingEnabled() const noexcept
{
    return profilingEnabled_.load(std::memory_order_acquire);
}

ClimateMap::CacheProfilingSnapshot ClimateMap::profilingSnapshot() const noexcept
{
    CacheProfilingSnapshot snapshot{};
    snapshot.hits = cacheHits_.load(std::memory_order_relaxed);
    snapshot.misses = cacheMisses_.load(std::memory_order_relaxed);
    snapshot.fills = cacheFills_.load(std::memory_order_relaxed);
    return snapshot;
}

void ClimateMap::resetProfiling() noexcept
{
    cacheHits_.store(0, std::memory_order_relaxed);
    cacheMisses_.store(0, std::memory_order_relaxed);
    cacheFills_.store(0, std::memory_order_relaxed);
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

