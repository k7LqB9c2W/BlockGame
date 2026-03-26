#include "terrain/terrain_generator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <glm/common.hpp>

#include "chunk_manager.h"
#include "terrain/climate_map.h"
#include "terrain/surface_map.h"

namespace terrain
{
namespace
{
enum class GrassTintIndex : std::uint8_t
{
    None = 0,
    Default = 1,
    DarkForest = 2,
    Taiga = 3,
    Warm = 4,
};

[[nodiscard]] bool terrainDebugLoggingEnabled() noexcept
{
    static const bool enabled = []()
    {
        const char* value = std::getenv("BLOCKGAME_TERRAIN_DEBUG_LOG");
        return value != nullptr && std::string_view(value) != "0" && std::string_view(value) != "false";
    }();
    return enabled;
}

[[nodiscard]] const char* terrainDebugLogPath() noexcept
{
    const char* value = std::getenv("BLOCKGAME_TERRAIN_DEBUG_LOG_FILE");
    return (value != nullptr && *value != '\0') ? value : "debug_terrain.log";
}

inline std::size_t columnIndex(int x, int z, int strideX) noexcept
{
    return static_cast<std::size_t>(z) * static_cast<std::size_t>(strideX) + static_cast<std::size_t>(x);
}

[[nodiscard]] std::uint8_t grassTintIndexForBiome(const BiomeDefinition* biome) noexcept
{
    if (!biome)
    {
        return static_cast<std::uint8_t>(GrassTintIndex::Default);
    }
    if (biome->id == "dark_forest")
    {
        return static_cast<std::uint8_t>(GrassTintIndex::DarkForest);
    }
    if (isTaigaBiome(*biome))
    {
        return static_cast<std::uint8_t>(GrassTintIndex::Taiga);
    }
    if (biome->id == "savanna" || biome->id == "desert")
    {
        return static_cast<std::uint8_t>(GrassTintIndex::Warm);
    }
    return static_cast<std::uint8_t>(GrassTintIndex::Default);
}

float hashToUnitFloat(int x, int y, int z) noexcept
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

void logTerrainAnomaly(const char* tag,
                       int worldX,
                       int worldZ,
                       int surfaceY,
                       float neighborAverage,
                       const ColumnSample& sample,
                       int seaLevel)
{
    if (!terrainDebugLoggingEnabled())
    {
        return;
    }

    static std::mutex s_logMutex;
    static std::ofstream s_logFile(terrainDebugLogPath(), std::ios::app);
    static int s_logCount = 0;
    if (s_logCount >= 500 || !s_logFile.is_open())
    {
        return;
    }

    std::lock_guard<std::mutex> lock(s_logMutex);
    ++s_logCount;
    const float seaLevelDelta = static_cast<float>(surfaceY - seaLevel);
    const float netDelta = static_cast<float>(surfaceY - sample.originalSurfaceY);
    s_logFile << tag << " world=(" << worldX << ',' << worldZ << ")"
              << " finalSurfaceY=" << surfaceY
              << " originalSurfaceY=" << sample.originalSurfaceY
              << " rawHeight=" << sample.surfaceHeight
              << " creepOffset=" << sample.soilCreepOffset
              << " netDelta=" << netDelta
              << " neighborAvg=" << neighborAverage
              << " seaLevelDelta=" << seaLevelDelta
              << " distanceToShore=" << sample.distanceToShore
              << " dominantIsOcean=" << (sample.dominantIsOcean ? "true" : "false")
              << " coastDistance=";
    if (std::isfinite(sample.distanceToCoast))
    {
        s_logFile << sample.distanceToCoast;
    }
    else
    {
        s_logFile << "inf";
    }
    s_logFile
              << " dominantBiome=" << (sample.dominantBiome ? sample.dominantBiome->id : "<none>")
              << " weight=" << sample.dominantWeight
              << " roughAmp=" << sample.roughAmplitude
              << " hillAmp=" << sample.hillAmplitude
              << " mountainAmp=" << sample.mountainAmplitude;

    if (sample.topBlendCount > 0)
    {
        s_logFile << " blends=";
        for (std::size_t i = 0; i < sample.topBlendCount; ++i)
        {
            const auto& blend = sample.topBlendDebug[i];
            s_logFile << " [" << i << "] biome=" << (blend.biome ? blend.biome->id : "<none>")
                      << " weight=" << blend.weight << " aggHeight=" << blend.aggregatedHeight
                      << " normDist=" << blend.normalizedDistance
                      << " radius=" << blend.seedRadius
                      << " worldDist=" << blend.worldDistance
                      << " isOcean=" << (blend.isOcean ? "true" : "false");
        }
    }

    s_logFile << '\n';
}

} // namespace

bool isTaigaBiome(const BiomeDefinition& biome) noexcept
{
    return biome.id == "taiga";
}

TerrainColumnBlocks resolveTerrainColumnBlocks(const BiomeDefinition& biome,
                                               const ColumnSample& sample,
                                               int worldX,
                                               int worldZ,
                                               int seaLevel) noexcept
{
    TerrainColumnBlocks result{biome.surfaceBlock, biome.fillerBlock};

    const bool nearSeaLevel = std::abs(sample.surfaceY - seaLevel) <= 2;
    constexpr float kBeachDistanceRange = 6.0f;
    if (!biome.isOcean() && nearSeaLevel && std::isfinite(sample.distanceToShore)
        && sample.distanceToShore <= kBeachDistanceRange)
    {
        const float noise = hashToUnitFloat(worldX, sample.surfaceY, worldZ);
        if (biome.terrainSettings.smoothBeaches)
        {
            const float shorelineWeight = 1.0f - std::clamp(sample.distanceToShore / kBeachDistanceRange, 0.0f, 1.0f);
            const float sandProbability = glm::mix(0.4f, 0.95f, shorelineWeight);
            if (noise <= sandProbability)
            {
                result.surfaceBlock = BlockId::Sand;
                result.fillerBlock = BlockId::Sand;
            }
            else if (noise < sandProbability + 0.1f)
            {
                result.fillerBlock = BlockId::Sand;
            }
        }
        else
        {
            result.surfaceBlock = noise < 0.55f ? BlockId::Sand : result.surfaceBlock;
            result.fillerBlock = BlockId::Sand;
        }
    }

    if (isTaigaBiome(biome) && result.surfaceBlock != BlockId::Sand)
    {
        const float patchNoise = taigaPodzolNoise(worldX, worldZ);
        const float patchSelector = hashToUnitFloat(worldX, sample.surfaceY * 23 + 11, worldZ);
        const bool usePodzol =
            patchNoise > 0.67f || (patchNoise > 0.59f && patchSelector > 0.45f);
        if (usePodzol)
        {
            result.surfaceBlock = BlockId::Podzol;
            result.fillerBlock = BlockId::Podzol;
        }
    }

    return result;
}

TerrainGenerator::TerrainGenerator(const ClimateMap& climateMap,
                                   const SurfaceMap& surfaceMap,
                                   const BiomeDatabase& biomeDatabase,
                                   int seaLevel,
                                   SampleColumnFn sampler)
    : climateMap_(climateMap),
      surfaceMap_(surfaceMap),
      biomeDatabase_(biomeDatabase),
      seaLevel_(seaLevel),
      sampler_(std::move(sampler))
{
    if (!sampler_)
    {
        throw std::invalid_argument("TerrainGenerator requires a column sampler");
    }
}

ChunkGenerationSummary TerrainGenerator::describeChunkColumns(const glm::ivec3& chunkCoord,
                                                              int minWorldY,
                                                              int maxWorldY,
                                                              int chunkSizeX,
                                                              int chunkSizeY,
                                                              int chunkSizeZ,
                                                              std::span<ExactChunkColumnDescriptor> outDescriptors,
                                                              std::span<ColumnBuildResult> outColumns) const
{
    if (outColumns.size() < static_cast<std::size_t>(chunkSizeX * chunkSizeZ))
    {
        throw std::invalid_argument("outColumns span is smaller than the chunk column count");
    }
    if (outDescriptors.size() < static_cast<std::size_t>(chunkSizeX * chunkSizeZ))
    {
        throw std::invalid_argument("descriptor span is smaller than the chunk column count");
    }

    ChunkGenerationSummary summary{};

    const int baseWorldX = chunkCoord.x * chunkSizeX;
    const int baseWorldZ = chunkCoord.z * chunkSizeZ;

    const int neighborSizeX = chunkSizeX + 2;
    const int neighborSizeZ = chunkSizeZ + 2;
    std::vector<int> neighborHeights(static_cast<std::size_t>(neighborSizeX * neighborSizeZ), 0);

    for (int dx = -1; dx <= chunkSizeX; ++dx)
    {
        for (int dz = -1; dz <= chunkSizeZ; ++dz)
        {
            const int worldX = baseWorldX + dx;
            const int worldZ = baseWorldZ + dz;
            const ColumnSample neighborSample = sampler_(worldX, worldZ, minWorldY, maxWorldY);
            const std::size_t idx = columnIndex(dx + 1, dz + 1, neighborSizeX);
            neighborHeights[idx] = neighborSample.surfaceY;
        }
    }

    const auto computeNeighborAverage = [&](int localX, int localZ) -> float
    {
        float sum = 0.0f;
        int count = 0;
        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dz = -1; dz <= 1; ++dz)
            {
                if (dx == 0 && dz == 0)
                {
                    continue;
                }
                const int nx = localX + dx + 1;
                const int nz = localZ + dz + 1;
                const std::size_t idx = columnIndex(nx, nz, neighborSizeX);
                sum += static_cast<float>(neighborHeights[idx]);
                ++count;
            }
        }
        if (count == 0)
        {
            return 0.0f;
        }
        return sum / static_cast<float>(count);
    };

    for (int localX = 0; localX < chunkSizeX; ++localX)
    {
        for (int localZ = 0; localZ < chunkSizeZ; ++localZ)
        {
            const int worldX = baseWorldX + localX;
            const int worldZ = baseWorldZ + localZ;
            ColumnBuildResult result{};
            ExactChunkColumnDescriptor descriptor{};
            result.sample = sampler_(worldX, worldZ, minWorldY, maxWorldY);
            ColumnSample& sample = result.sample;

            const std::size_t columnIdx = columnIndex(localX, localZ, chunkSizeX);
            outColumns[columnIdx] = result;
            outDescriptors[columnIdx] = descriptor;

            if (!sample.dominantBiome)
            {
                continue;
            }

            const BiomeDefinition& biome = *sample.dominantBiome;
            descriptor.flags |= ExactChunkColumnDescriptor::kFlagHasBiome;
            descriptor.biomeIndex = static_cast<std::uint32_t>(biomeDatabase_.definitionIndex(biome));
            descriptor.grassTintIndex = grassTintIndexForBiome(sample.dominantBiome);

            const float neighborAverage = computeNeighborAverage(localX, localZ);
            int adjustedSurfaceY = sample.surfaceY;
            float creepOffset = 0.0f;
            if (biome.terrainSettings.soilCreep.strength > 0.0f && sample.soilCreepCoefficient > 0.0f)
            {
                const float strength = std::clamp(sample.soilCreepCoefficient * biome.terrainSettings.soilCreep.strength,
                                                  0.0f,
                                                  1.0f);
                const float delta = neighborAverage - static_cast<float>(adjustedSurfaceY);
                float offset = delta * strength;
                if (biome.terrainSettings.soilCreep.maxStep > 0)
                {
                    const float maxStep = static_cast<float>(biome.terrainSettings.soilCreep.maxStep);
                    offset = std::clamp(offset, -maxStep, maxStep);
                }
                if (biome.terrainSettings.soilCreep.maxDepth > 0)
                {
                    const float maxDepth = static_cast<float>(biome.terrainSettings.soilCreep.maxDepth);
                    offset = std::clamp(offset, -maxDepth, maxDepth);
                }
                creepOffset = offset;
                adjustedSurfaceY = static_cast<int>(std::round(static_cast<float>(adjustedSurfaceY) + offset));
                adjustedSurfaceY = std::clamp(adjustedSurfaceY, sample.minSurfaceY, sample.maxSurfaceY);
            }

            sample.soilCreepOffset = creepOffset;
            sample.surfaceY = adjustedSurfaceY;
            sample.slabHasSolid = minWorldY <= adjustedSurfaceY;
            sample.slabHighestSolidY = sample.slabHasSolid ? std::min(adjustedSurfaceY, maxWorldY)
                                                           : std::numeric_limits<int>::min();

            const auto& waterFill = biome.terrainSettings.waterFill;
            int waterTopWorld = std::numeric_limits<int>::min();
            int waterBottomWorld = std::numeric_limits<int>::max();
            bool slabHasWater = false;
            if (waterFill.enabled && adjustedSurfaceY < seaLevel_)
            {
                waterTopWorld = std::min(seaLevel_, maxWorldY);
                waterBottomWorld = std::max(adjustedSurfaceY + 1, minWorldY);
                if (waterFill.maxDepth > 0)
                {
                    waterBottomWorld = std::max(waterBottomWorld, waterTopWorld - waterFill.maxDepth + 1);
                }
                slabHasWater = waterBottomWorld <= waterTopWorld;
            }

            descriptor.surfaceY = adjustedSurfaceY;
            descriptor.originalSurfaceY = sample.originalSurfaceY;
            descriptor.minSurfaceY = sample.minSurfaceY;
            descriptor.maxSurfaceY = sample.maxSurfaceY;
            descriptor.waterTopWorld = waterTopWorld;
            descriptor.waterBottomWorld = waterBottomWorld;
            descriptor.waterBlock = waterFill.block;
            if (sample.slabHasSolid)
            {
                descriptor.flags |= ExactChunkColumnDescriptor::kFlagHasSolid;
            }
            if (slabHasWater)
            {
                descriptor.flags |= ExactChunkColumnDescriptor::kFlagHasWater;
            }
            if (sample.dominantIsOcean)
            {
                descriptor.flags |= ExactChunkColumnDescriptor::kFlagDominantIsOcean;
            }

            outColumns[columnIdx].sample = sample;
            if (!sample.slabHasSolid && !slabHasWater)
            {
                outDescriptors[columnIdx] = descriptor;
                continue;
            }

            if (sample.slabHasSolid)
            {
                summary.slabContainsTerrain = true;
            }

            if (terrainDebugLoggingEnabled())
            {
                const float diff = std::abs(static_cast<float>(adjustedSurfaceY) - neighborAverage);
                if (sample.slabHasSolid && (adjustedSurfaceY <= minWorldY + 4 || diff > 48.0f))
                {
                    logTerrainAnomaly("[HeightDebug]", worldX, worldZ, adjustedSurfaceY, neighborAverage, sample, seaLevel_);
                }
            }

            sample.surfaceY = adjustedSurfaceY;
            const TerrainColumnBlocks resolvedBlocks =
                resolveTerrainColumnBlocks(biome, sample, worldX, worldZ, seaLevel_);
            const BlockId surfaceBlock = resolvedBlocks.surfaceBlock;
            const BlockId fillerBlock = resolvedBlocks.fillerBlock;
            descriptor.surfaceBlock = surfaceBlock;
            descriptor.fillerBlock = fillerBlock;


            const int highestSolidWorld = std::min(sample.slabHighestSolidY, maxWorldY);
            if (sample.slabHasSolid && highestSolidWorld >= minWorldY)
            {
                const int highestLocalY = std::min(highestSolidWorld - minWorldY, chunkSizeY - 1);

                const auto& stripes = biome.terrainSettings.stripes;
                const bool stripesEnabled = stripes.enabled && stripes.period > 0 && stripes.thickness > 0;
                const bool columnHasStripes = stripesEnabled
                                              && hashToUnitFloat(worldX, adjustedSurfaceY * 17 + 3, worldZ)
                                                     > stripes.noiseThreshold;
                const int stripePeriod = std::max(stripes.period, stripes.thickness);
                const int stripeOffset = stripesEnabled
                                             ? static_cast<int>(hashToUnitFloat(worldX, adjustedSurfaceY * 31 + 7, worldZ)
                                                               * static_cast<float>(stripePeriod))
                                             : 0;
                if (stripesEnabled)
                {
                    descriptor.flags |= ExactChunkColumnDescriptor::kFlagStripesEnabled;
                }
                if (columnHasStripes)
                {
                    descriptor.flags |= ExactChunkColumnDescriptor::kFlagColumnHasStripes;
                }
                descriptor.stripePeriod = static_cast<std::uint16_t>(stripePeriod);
                descriptor.stripeThickness = static_cast<std::uint16_t>(std::max(0, stripes.thickness));
                descriptor.stripeOffset = stripeOffset;
                descriptor.stripeBlock = stripes.block;
                outColumns[columnIdx].highestSolidWorld = highestSolidWorld;
                outColumns[columnIdx].wroteSolid = true;
                descriptor.highestSolidWorld = highestSolidWorld;
                summary.anySolid = true;
                (void)highestLocalY;
            }

            if (slabHasWater)
            {
                outColumns[columnIdx].waterTopWorld = waterTopWorld;
                outColumns[columnIdx].wroteSolid = true;
                summary.anySolid = true;
            }

            outDescriptors[columnIdx] = descriptor;
        }
    }

    return summary;
}

void TerrainGenerator::materializeChunkColumns(int minWorldY,
                                               int maxWorldY,
                                               int chunkSizeX,
                                               int chunkSizeY,
                                               int chunkSizeZ,
                                               std::span<const ExactChunkColumnDescriptor> descriptors,
                                               const BlockSetter& setBlock) const
{
    if (descriptors.size() < static_cast<std::size_t>(chunkSizeX * chunkSizeZ))
    {
        throw std::invalid_argument("descriptor span is smaller than the chunk column count");
    }
    if (!setBlock)
    {
        throw std::invalid_argument("TerrainGenerator requires a block setter callback");
    }

    for (int localX = 0; localX < chunkSizeX; ++localX)
    {
        for (int localZ = 0; localZ < chunkSizeZ; ++localZ)
        {
            const ExactChunkColumnDescriptor& descriptor =
                descriptors[columnIndex(localX, localZ, chunkSizeX)];
            if (!descriptor.hasBiome())
            {
                continue;
            }

            if (descriptor.hasSolid() && descriptor.highestSolidWorld >= minWorldY)
            {
                const int highestLocalY = std::min(descriptor.highestSolidWorld - minWorldY, chunkSizeY - 1);
                for (int localY = 0; localY <= highestLocalY; ++localY)
                {
                    const int worldY = minWorldY + localY;
                    BlockId block = BlockId::Air;
                    if (worldY < descriptor.surfaceY)
                    {
                        block = descriptor.fillerBlock;
                        if (descriptor.columnHasStripes() && descriptor.stripePeriod > 0 &&
                            descriptor.stripeThickness > 0)
                        {
                            const int pattern = (worldY + descriptor.stripeOffset) %
                                                static_cast<int>(descriptor.stripePeriod);
                            if (pattern < static_cast<int>(descriptor.stripeThickness))
                            {
                                block = descriptor.stripeBlock;
                            }
                        }
                    }
                    else if (worldY == descriptor.surfaceY)
                    {
                        block = descriptor.surfaceBlock;
                    }

                    if (block != BlockId::Air)
                    {
                        setBlock(localX, localY, localZ, block);
                    }
                }
            }

            if (descriptor.hasWater())
            {
                for (int worldY = descriptor.waterBottomWorld; worldY <= descriptor.waterTopWorld; ++worldY)
                {
                    const int localY = worldY - minWorldY;
                    if (localY < 0 || localY >= chunkSizeY || worldY < minWorldY || worldY > maxWorldY)
                    {
                        continue;
                    }
                    setBlock(localX, localY, localZ, descriptor.waterBlock);
                }
            }
        }
    }
}

ChunkGenerationSummary TerrainGenerator::generateChunkColumns(const glm::ivec3& chunkCoord,
                                                              int minWorldY,
                                                              int maxWorldY,
                                                              int chunkSizeX,
                                                              int chunkSizeY,
                                                              int chunkSizeZ,
                                                              const BlockSetter& setBlock,
                                                              std::span<ExactChunkColumnDescriptor> outDescriptors,
                                                              std::span<ColumnBuildResult> outColumns) const
{
    const ChunkGenerationSummary summary = describeChunkColumns(chunkCoord,
                                                                minWorldY,
                                                                maxWorldY,
                                                                chunkSizeX,
                                                                chunkSizeY,
                                                                chunkSizeZ,
                                                                outDescriptors,
                                                                outColumns);
    materializeChunkColumns(minWorldY,
                            maxWorldY,
                            chunkSizeX,
                            chunkSizeY,
                            chunkSizeZ,
                            outDescriptors,
                            setBlock);
    return summary;
}

} // namespace terrain

