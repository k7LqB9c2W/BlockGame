#include "terrain/far_lod_worldgen.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include <glm/common.hpp>

#include "chunk_manager.h"

namespace terrain
{
namespace
{
constexpr float kCoastDistanceFieldRange = 96.0f;
constexpr float kOceanThreshold = -0.08f;
constexpr float kCoastDistanceScale = 72.0f;
constexpr float kTemperatureScale = 0.11f;
constexpr float kMoistureScale = 0.09f;
constexpr float kFertilityScale = 0.05f;
constexpr float kContinentalScale = 0.065f;

float smoothStep(float t) noexcept
{
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float fade(float t) noexcept
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float grad(std::uint32_t hash, float x, float y) noexcept
{
    switch (hash & 7u)
    {
    case 0u: return x + y;
    case 1u: return -x + y;
    case 2u: return x - y;
    case 3u: return -x - y;
    case 4u: return x;
    case 5u: return -x;
    case 6u: return y;
    default: return -y;
    }
}

std::uint32_t latticeHash(int x, int y, std::uint32_t seed) noexcept
{
    std::uint32_t h = static_cast<std::uint32_t>(x);
    h ^= static_cast<std::uint32_t>(y) * 374761393u;
    h ^= seed * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return h;
}

float perlin2(float x, float y, std::uint32_t seed) noexcept
{
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float u = fade(fx);
    const float v = fade(fy);

    const float a = grad(latticeHash(x0, y0, seed), fx, fy);
    const float b = grad(latticeHash(x1, y0, seed), fx - 1.0f, fy);
    const float c = grad(latticeHash(x0, y1, seed), fx, fy - 1.0f);
    const float d = grad(latticeHash(x1, y1, seed), fx - 1.0f, fy - 1.0f);

    const float ix0 = std::lerp(a, b, u);
    const float ix1 = std::lerp(c, d, u);
    return std::lerp(ix0, ix1, v);
}

float fbm2(float x,
           float y,
           const FbmSettings& settings,
           std::uint32_t seed,
           float frequencyScale = 1.0f) noexcept
{
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = settings.frequency * frequencyScale;
    float amplitudeSum = 0.0f;
    for (int octave = 0; octave < settings.octaves; ++octave)
    {
        value += perlin2(x * frequency, y * frequency, seed + static_cast<std::uint32_t>(octave) * 17u) * amplitude;
        amplitudeSum += amplitude;
        amplitude *= settings.gain;
        frequency *= settings.lacunarity;
    }
    return amplitudeSum > 0.0f ? value / amplitudeSum : 0.0f;
}

float ridge2(float x, float y, const FbmSettings& settings, std::uint32_t seed) noexcept
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float frequency = settings.frequency;
    float prev = 1.0f;
    for (int octave = 0; octave < settings.octaves; ++octave)
    {
        float n = 1.0f - std::abs(perlin2(x * frequency, y * frequency, seed + 97u + static_cast<std::uint32_t>(octave) * 29u));
        n *= n;
        sum += n * amplitude * prev;
        prev = n * settings.gain;
        frequency *= settings.lacunarity;
        amplitude *= 0.5f;
    }
    return sum;
}

struct CoastProfileSettings
{
    float inlandBlendDistance{48.0f};
    float offshoreBlendDistance{48.0f};
    float shorelineRise{2.0f};
    float nearshoreDepth{3.0f};
    float roughFadeDistance{32.0f};
    float hillFadeDistance{40.0f};
    float mountainFadeDistance{48.0f};
    float roughFloor{0.08f};
    float hillFloor{0.05f};
    float mountainFloor{0.02f};
};

const CoastProfileSettings& coastProfileSettings(FarLodCoastProfile profile) noexcept
{
    static const CoastProfileSettings kGentleBeach{
        56.0f, 64.0f, 1.5f, 2.5f, 36.0f, 44.0f, 56.0f, 0.08f, 0.05f, 0.02f};
    static const CoastProfileSettings kDunes{
        72.0f, 56.0f, 2.5f, 2.0f, 42.0f, 54.0f, 68.0f, 0.16f, 0.08f, 0.02f};
    static const CoastProfileSettings kRockyShore{
        40.0f, 36.0f, 4.5f, 4.0f, 28.0f, 34.0f, 40.0f, 0.35f, 0.24f, 0.12f};
    static const CoastProfileSettings kCliffCoast{
        18.0f, 28.0f, 12.0f, 6.0f, 18.0f, 22.0f, 28.0f, 0.60f, 0.52f, 0.46f};
    static const CoastProfileSettings kMarsh{
        84.0f, 52.0f, 0.75f, 1.5f, 54.0f, 66.0f, 80.0f, 0.04f, 0.02f, 0.00f};

    switch (profile)
    {
    case FarLodCoastProfile::Dunes:
        return kDunes;
    case FarLodCoastProfile::RockyShore:
        return kRockyShore;
    case FarLodCoastProfile::CliffCoast:
        return kCliffCoast;
    case FarLodCoastProfile::Marsh:
        return kMarsh;
    case FarLodCoastProfile::GentleBeach:
    default:
        return kGentleBeach;
    }
}

float shorelineNoiseFactor(float distance, float fadeDistance, float floorValue) noexcept
{
    if (!std::isfinite(distance))
    {
        return 1.0f;
    }
    return std::lerp(floorValue, 1.0f, smoothStep(distance / std::max(fadeDistance, 1.0f)));
}

float solveShorelineBaseHeight(float signedDistance,
                               float landBaseHeight,
                               float oceanBaseHeight,
                               float seaLevel,
                               const CoastProfileSettings& settings) noexcept
{
    if (!std::isfinite(signedDistance))
    {
        return signedDistance < 0.0f ? oceanBaseHeight : landBaseHeight;
    }

    const float shorelineLandHeight = seaLevel + settings.shorelineRise;
    const float nearshoreFloor = seaLevel - settings.nearshoreDepth;
    const float safeLandBase = std::max(landBaseHeight, shorelineLandHeight);
    const float safeOceanBase = std::min(oceanBaseHeight, nearshoreFloor);

    if (signedDistance >= 0.0f)
    {
        const float inlandFactor = smoothStep(signedDistance / std::max(settings.inlandBlendDistance, 1.0f));
        return std::lerp(shorelineLandHeight, safeLandBase, inlandFactor);
    }

    const float offshoreFactor = smoothStep((-signedDistance) / std::max(settings.offshoreBlendDistance, 1.0f));
    return std::lerp(nearshoreFloor, safeOceanBase, offshoreFactor);
}

float categoryScore(float value01, std::uint16_t bits, std::uint16_t lowMask, std::uint16_t midMask, std::uint16_t highMask) noexcept
{
    if ((bits & (lowMask | midMask | highMask)) == 0)
    {
        return 1.0f;
    }

    const float low = std::clamp((0.6f - value01) / 0.6f, 0.0f, 1.0f);
    const float high = std::clamp((value01 - 0.4f) / 0.6f, 0.0f, 1.0f);
    const float mid = 1.0f - std::min(1.0f, std::abs(value01 - 0.5f) * 2.2f);

    float score = 0.0f;
    if ((bits & lowMask) != 0) score = std::max(score, low);
    if ((bits & midMask) != 0) score = std::max(score, mid);
    if ((bits & highMask) != 0) score = std::max(score, high);
    return score;
}

float scoreBiome(const FarLodGpuBiome& biome,
                 float oceaniness,
                 float temperature01,
                 float moisture01,
                 float fertility01,
                 float mountain01,
                 float inland01) noexcept
{
    using GP = BiomeDefinition::GenerationProperties;

    float score = std::max(biome.spawnChance, 0.01f);
    const bool isOcean = (biome.flags & kFarLodBiomeOcean) != 0;
    const float oceanScore = isOcean ? std::clamp((0.5f - oceaniness) * 2.0f + 0.5f, 0.0f, 1.0f)
                                     : std::clamp((oceaniness + 0.5f) * 2.0f, 0.0f, 1.0f);
    score *= std::max(oceanScore, 0.05f);
    score *= std::max(categoryScore(temperature01, static_cast<std::uint16_t>(biome.propertyBits),
                                    GP::kCold, GP::kTemperate, GP::kHot), 0.05f);
    score *= std::max(categoryScore(moisture01, static_cast<std::uint16_t>(biome.propertyBits),
                                    GP::kDry, GP::kNeutralHydration, GP::kWet), 0.05f);
    score *= std::max(categoryScore(fertility01, static_cast<std::uint16_t>(biome.propertyBits),
                                    GP::kBarren, GP::kBalanced, GP::kOvergrown), 0.05f);
    score *= std::max(categoryScore(mountain01, static_cast<std::uint16_t>(biome.propertyBits),
                                    GP::kLowTerrain, GP::kAntiMountain, GP::kMountain), 0.05f);

    const std::uint16_t propertyBits = static_cast<std::uint16_t>(biome.propertyBits);
    if ((propertyBits & GP::kInland) != 0)
    {
        score *= std::max(inland01, 0.05f);
    }

    score *= std::lerp(0.85f, 1.15f, std::clamp(biome.interpolationWeight, 0.0f, 1.0f));
    return score;
}

FarLodCoastProfile mapCoastProfile(BiomeDefinition::TerrainSettings::CoastProfile profile) noexcept
{
    using CpuProfile = BiomeDefinition::TerrainSettings::CoastProfile;
    switch (profile)
    {
    case CpuProfile::Dunes: return FarLodCoastProfile::Dunes;
    case CpuProfile::RockyShore: return FarLodCoastProfile::RockyShore;
    case CpuProfile::CliffCoast: return FarLodCoastProfile::CliffCoast;
    case CpuProfile::Marsh: return FarLodCoastProfile::Marsh;
    case CpuProfile::Auto:
    case CpuProfile::GentleBeach:
    default:
        return FarLodCoastProfile::GentleBeach;
    }
}

std::uint32_t mapInterpolationCurve(BiomeDefinition::InterpolationCurve curve) noexcept
{
    switch (curve)
    {
    case BiomeDefinition::InterpolationCurve::Step:
        return 0u;
    case BiomeDefinition::InterpolationCurve::Linear:
        return 1u;
    case BiomeDefinition::InterpolationCurve::Square:
    default:
        return 2u;
    }
}

std::array<float, 2> warpPosition(const FarLodGpuWorldgenHeader& header, int worldX, int worldZ) noexcept
{
    const float worldXF = static_cast<float>(worldX);
    const float worldZF = static_cast<float>(worldZ);
    const float warpSample =
        fbm2(worldXF, worldZF, header.mainNoise, header.seed + 11u, header.warpFrequency / std::max(header.mainNoise.frequency, 1.0e-6f));
    return {worldXF + warpSample * header.warpAmplitude, worldZF + warpSample * header.warpAmplitude};
}

} // namespace

FarLodWorldgenTables buildFarLodWorldgenTables(const BiomeDatabase& biomeDatabase,
                                               const WorldgenProfile& worldgenProfile,
                                               unsigned seed)
{
    FarLodWorldgenTables tables{};
    tables.header.seaLevel = worldgenProfile.seaLevel;
    tables.header.seed = worldgenProfile.effectiveSeed(seed);
    tables.header.warpFrequency = 0.0025f;
    tables.header.warpAmplitude = 18.0f;
    tables.header.mainNoise = worldgenProfile.noise.main;
    tables.header.mediumNoise = worldgenProfile.noise.medium;
    tables.header.detailNoise = worldgenProfile.noise.detail;
    tables.header.mountainNoise = worldgenProfile.noise.mountain;
    tables.header.treeDensityFrequency = 0.05f;
    tables.header.treeDensityOctaves = 4u;
    tables.header.treeDensityGain = 0.55f;
    tables.header.treeDensityLacunarity = 2.0f;
    tables.header.coastDistanceFieldRange = kCoastDistanceFieldRange;

    const int chunkSpan =
        std::max(64, static_cast<int>(std::ceil(biomeDatabase.maxBiomeRadius() * 1.75f)));
    const int alignment = 32;
    tables.header.chunkSpan = std::max(alignment, ((chunkSpan + alignment - 1) / alignment) * alignment);
    tables.header.neighborRadius =
        std::max(2, static_cast<int>(std::ceil(biomeDatabase.maxBiomeRadius() /
                                               static_cast<float>(tables.header.chunkSpan))) + 1);

    std::unordered_map<const BiomeDefinition*, std::uint32_t> biomeIndexByPtr;
    biomeIndexByPtr.reserve(biomeDatabase.biomeCount());

    tables.biomes.reserve(biomeDatabase.biomeCount());
    tables.transitionBiomes.reserve(biomeDatabase.biomeCount() * 2u);
    tables.subBiomes.reserve(biomeDatabase.biomeCount() * 2u);
    std::uint32_t biomeIndex = 0u;
    for (const BiomeDefinition& biome : biomeDatabase.definitions())
    {
        FarLodGpuBiome packed{};
        packed.surfaceBlock = static_cast<std::uint32_t>(biome.surfaceBlock);
        packed.fillerBlock = static_cast<std::uint32_t>(biome.fillerBlock);
        packed.canopyBlock = 0u;
        packed.secondaryCanopyBlock = 0u;
        packed.flags = 0u;
        packed.secondaryCanopyChance = 0.0f;
        if (biome.isOcean())
        {
            packed.flags |= kFarLodBiomeOcean;
        }
        if (biome.terrainSettings.smoothBeaches)
        {
            packed.flags |= kFarLodBiomeSmoothBeaches;
        }
        if (biome.terrainSettings.waterFill.enabled)
        {
            packed.flags |= kFarLodBiomeWaterFill;
        }
        if (biome.id == "taiga")
        {
            packed.flags |= kFarLodBiomeTaiga;
            packed.canopyBlock = static_cast<std::uint32_t>(BlockId::SpruceLeaves);
        }
        if (biome.generatesTrees)
        {
            packed.flags |= kFarLodBiomeGeneratesTrees;
            if (packed.canopyBlock == 0u)
            {
                packed.canopyBlock = static_cast<std::uint32_t>(BlockId::Leaves);
            }
        }
        if (biome.id == "birch_forest")
        {
            packed.canopyBlock = static_cast<std::uint32_t>(BlockId::BirchLeaves);
        }
        if (biome.id == "dark_forest")
        {
            packed.canopyBlock = static_cast<std::uint32_t>(BlockId::DarkOakLeaves);
        }
        if (biome.id == "forest")
        {
            packed.secondaryCanopyBlock = static_cast<std::uint32_t>(BlockId::BirchLeaves);
            packed.secondaryCanopyChance = 0.30f;
        }
        if (biome.hasFlag("beach"))
        {
            packed.flags |= kFarLodBiomeBeach;
        }
        if (biome.hasFlag("coastal"))
        {
            packed.flags |= kFarLodBiomeCoastal;
        }
        packed.coastProfile = static_cast<std::uint32_t>(mapCoastProfile(biome.effectiveCoastProfile()));
        packed.propertyBits = biome.properties.value();
        packed.waterMaxDepth = biome.terrainSettings.waterFill.maxDepth;
        packed.spawnChance = biome.spawnChance;
        packed.minHeight = static_cast<float>(biome.minHeight);
        packed.maxHeight = static_cast<float>(biome.maxHeight);
        packed.heightOffset = biome.heightOffset;
        packed.heightScale = biome.heightScale;
        packed.roughness = biome.roughness;
        packed.hills = biome.hills;
        packed.mountains = biome.mountains;
        packed.keepOriginalTerrain = biome.keepOriginalTerrain;
        packed.interpolationWeight = biome.interpolationWeight;
        packed.interpolationCurve = mapInterpolationCurve(biome.interpolationCurve);
        packed.radius = biome.radius;
        packed.radiusVariation = biome.radiusVariation;
        packed.fixedRadius = biome.fixedRadius ? 1u : 0u;
        packed.treeDensityMultiplier = biome.treeDensityMultiplier;
        packed.maxSubBiomeCount = biome.maxSubBiomeCount;
        packed.subBiomeTotalChance = biome.subBiomeTotalChance;
        packed.minHeightLimit = biome.minHeightLimit.value_or(0);
        packed.maxHeightLimit = biome.maxHeightLimit.value_or(0);
        packed.hasMinHeightLimit = biome.minHeightLimit.has_value() ? 1u : 0u;
        packed.hasMaxHeightLimit = biome.maxHeightLimit.has_value() ? 1u : 0u;
        packed.baseSlopeBias = biome.baseSlopeBias;
        packed.maxGradient = biome.maxGradient;
        packed.footprintMultiplier = biome.footprintMultiplier;
        for (const auto& transition : biome.transitionBiomes)
        {
            tables.header.maxTransitionWidth =
                std::max(tables.header.maxTransitionWidth, static_cast<std::int32_t>(transition.width));
        }
        tables.biomes.push_back(packed);
        biomeIndexByPtr.emplace(&biome, biomeIndex++);
    }

    for (std::size_t index = 0; index < biomeDatabase.definitions().size(); ++index)
    {
        const BiomeDefinition& biome = biomeDatabase.definitions()[index];
        FarLodGpuBiome& packed = tables.biomes[index];
        packed.transitionOffset = static_cast<std::uint32_t>(tables.transitionBiomes.size());
        packed.subBiomeOffset = static_cast<std::uint32_t>(tables.subBiomes.size());

        for (const auto& transition : biome.transitionBiomes)
        {
            if (!transition.biome)
            {
                continue;
            }

            const auto it = biomeIndexByPtr.find(transition.biome);
            if (it == biomeIndexByPtr.end())
            {
                continue;
            }

            FarLodGpuTransitionBiome packedTransition{};
            packedTransition.biomeIndex = it->second;
            packedTransition.chance = transition.chance;
            packedTransition.width = transition.width;
            packedTransition.propertyBits = transition.propertyMask.value();
            tables.transitionBiomes.push_back(packedTransition);
        }
        packed.transitionCount =
            static_cast<std::uint32_t>(tables.transitionBiomes.size()) - packed.transitionOffset;

        for (const auto& sub : biome.subBiomes)
        {
            if (!sub.biome)
            {
                continue;
            }

            const auto it = biomeIndexByPtr.find(sub.biome);
            if (it == biomeIndexByPtr.end())
            {
                continue;
            }

            FarLodGpuSubBiome packedSub{};
            packedSub.biomeIndex = it->second;
            packedSub.chance = sub.chance;
            packedSub.minRadius = sub.minRadius;
            packedSub.maxRadius = sub.maxRadius;
            tables.subBiomes.push_back(packedSub);
        }
        packed.subBiomeCount =
            static_cast<std::uint32_t>(tables.subBiomes.size()) - packed.subBiomeOffset;
    }

    for (std::uint32_t index = 0; index < tables.biomes.size(); ++index)
    {
        const BiomeDefinition& biome = biomeDatabase.definitionByIndex(index);
        if (biome.spawnChance <= 0.0f)
        {
            continue;
        }

        const float radiusScale = std::max(biome.radius, 1.0f);
        float weight = std::max(biome.spawnChance * biome.footprintMultiplier, 0.0f);
        weight /= std::max(radiusScale, 1.0f);
        const auto& props = biome.generationProperties();
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

        tables.header.totalSpawnWeight += weight;
        tables.biomeSelections.push_back(FarLodGpuBiomeSelection{
            index,
            tables.header.totalSpawnWeight,
            0u,
            0u});

        if (biome.isOcean())
        {
            tables.header.totalOceanWeight += weight;
            tables.oceanSelections.push_back(FarLodGpuBiomeSelection{
                index,
                tables.header.totalOceanWeight,
                0u,
                0u});
        }
    }

    tables.surfacePermutation.resize(512u);
    std::array<std::uint32_t, 256> permutation{};
    for (std::uint32_t i = 0; i < permutation.size(); ++i)
    {
        permutation[i] = i;
    }
    std::mt19937 rng(tables.header.seed ^ 0xA511E9B7u);
    std::shuffle(permutation.begin(), permutation.end(), rng);
    for (std::size_t i = 0; i < permutation.size(); ++i)
    {
        tables.surfacePermutation[i] = permutation[i];
        tables.surfacePermutation[256u + i] = permutation[i];
    }

    tables.header.biomeCount = static_cast<std::uint32_t>(tables.biomes.size());
    tables.header.biomeSelectionCount = static_cast<std::uint32_t>(tables.biomeSelections.size());
    tables.header.oceanSelectionCount = static_cast<std::uint32_t>(tables.oceanSelections.size());
    tables.header.transitionCount = static_cast<std::uint32_t>(tables.transitionBiomes.size());
    tables.header.subBiomeCount = static_cast<std::uint32_t>(tables.subBiomes.size());
    return tables;
}

} // namespace terrain
