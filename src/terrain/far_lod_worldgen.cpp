#include "terrain/far_lod_worldgen.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <glm/common.hpp>

#include "chunk_manager.h"

namespace terrain
{
namespace
{
constexpr float kOceanThreshold = -0.08f;
constexpr float kCoastDistanceScale = 72.0f;
constexpr float kTemperatureScale = 0.11f;
constexpr float kMoistureScale = 0.09f;
constexpr float kFertilityScale = 0.05f;
constexpr float kContinentalScale = 0.065f;

float hashToUnitFloat(int x, int y, int z) noexcept
{
    std::uint32_t h = static_cast<std::uint32_t>(x);
    h ^= static_cast<std::uint32_t>(y) * 374761393u;
    h ^= static_cast<std::uint32_t>(z) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
}

float smoothStep(float t) noexcept
{
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float valueNoise2D(float x, float z, float frequency, int seed) noexcept
{
    const float sampleX = x * frequency;
    const float sampleZ = z * frequency;
    const int x0 = static_cast<int>(std::floor(sampleX));
    const int z0 = static_cast<int>(std::floor(sampleZ));
    const int x1 = x0 + 1;
    const int z1 = z0 + 1;

    const float tx = smoothStep(sampleX - static_cast<float>(x0));
    const float tz = smoothStep(sampleZ - static_cast<float>(z0));

    const float v00 = hashToUnitFloat(x0 + seed * 17, seed * 31, z0 - seed * 13);
    const float v10 = hashToUnitFloat(x1 + seed * 17, seed * 31, z0 - seed * 13);
    const float v01 = hashToUnitFloat(x0 + seed * 17, seed * 31, z1 - seed * 13);
    const float v11 = hashToUnitFloat(x1 + seed * 17, seed * 31, z1 - seed * 13);

    const float ix0 = std::lerp(v00, v10, tx);
    const float ix1 = std::lerp(v01, v11, tx);
    return std::lerp(ix0, ix1, tz);
}

float taigaPodzolNoise(int worldX, int worldZ) noexcept
{
    const float broad = valueNoise2D(static_cast<float>(worldX), static_cast<float>(worldZ), 1.0f / 16.0f, 19);
    const float medium = valueNoise2D(static_cast<float>(worldX), static_cast<float>(worldZ), 1.0f / 8.0f, 37);
    const float detail = valueNoise2D(static_cast<float>(worldX), static_cast<float>(worldZ), 1.0f / 4.0f, 73);
    return broad * 0.55f + medium * 0.30f + detail * 0.15f;
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

std::array<float, 2> warpPosition(const FarLodGpuWorldgenHeader& header, int worldX, int worldZ) noexcept
{
    const float worldXF = static_cast<float>(worldX);
    const float worldZF = static_cast<float>(worldZ);
    const float warpSample =
        fbm2(worldXF, worldZF, header.mainNoise, header.seed + 11u, header.warpFrequency / std::max(header.mainNoise.frequency, 1.0e-6f));
    return {worldXF + warpSample * header.warpAmplitude, worldZF + warpSample * header.warpAmplitude};
}

void resolveFarLodColumnBlocks(const FarLodGpuBiome& biome,
                               int surfaceY,
                               float distanceToShore,
                               int seaLevel,
                               int worldX,
                               int worldZ,
                               BlockId& surfaceBlock,
                               BlockId& fillerBlock) noexcept
{
    surfaceBlock = static_cast<BlockId>(biome.surfaceBlock);
    fillerBlock = static_cast<BlockId>(biome.fillerBlock);

    const bool nearSeaLevel = std::abs(surfaceY - seaLevel) <= 2;
    constexpr float kBeachDistanceRange = 6.0f;
    if ((biome.flags & kFarLodBiomeOcean) == 0u &&
        nearSeaLevel &&
        std::isfinite(distanceToShore) &&
        distanceToShore <= kBeachDistanceRange)
    {
        const float noise = hashToUnitFloat(worldX, surfaceY, worldZ);
        if ((biome.flags & kFarLodBiomeSmoothBeaches) != 0u)
        {
            const float shorelineWeight = 1.0f - std::clamp(distanceToShore / kBeachDistanceRange, 0.0f, 1.0f);
            const float sandProbability = std::lerp(0.4f, 0.95f, shorelineWeight);
            if (noise <= sandProbability)
            {
                surfaceBlock = BlockId::Sand;
                fillerBlock = BlockId::Sand;
            }
            else if (noise < sandProbability + 0.1f)
            {
                fillerBlock = BlockId::Sand;
            }
        }
        else
        {
            surfaceBlock = noise < 0.55f ? BlockId::Sand : surfaceBlock;
            fillerBlock = BlockId::Sand;
        }
    }

    if ((biome.flags & kFarLodBiomeTaiga) != 0u && surfaceBlock != BlockId::Sand)
    {
        const float patchNoise = taigaPodzolNoise(worldX, worldZ);
        const float patchSelector = hashToUnitFloat(worldX, surfaceY * 23 + 11, worldZ);
        const bool usePodzol = patchNoise > 0.67f || (patchNoise > 0.59f && patchSelector > 0.45f);
        if (usePodzol)
        {
            surfaceBlock = BlockId::Podzol;
            fillerBlock = BlockId::Podzol;
        }
    }
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

    tables.biomes.reserve(biomeDatabase.biomeCount());
    for (const BiomeDefinition& biome : biomeDatabase.definitions())
    {
        FarLodGpuBiome packed{};
        packed.surfaceBlock = static_cast<std::uint32_t>(biome.surfaceBlock);
        packed.fillerBlock = static_cast<std::uint32_t>(biome.fillerBlock);
        packed.flags = 0u;
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
        packed.baseSlopeBias = biome.baseSlopeBias;
        packed.maxGradient = biome.maxGradient;
        packed.footprintMultiplier = biome.footprintMultiplier;
        tables.biomes.push_back(packed);
    }

    tables.header.biomeCount = static_cast<std::uint32_t>(tables.biomes.size());
    return tables;
}

} // namespace terrain
