#include "terrain/terrain_generator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <stdexcept>
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
inline std::size_t columnIndex(int x, int z, int strideX) noexcept
{
    return static_cast<std::size_t>(z) * static_cast<std::size_t>(strideX) + static_cast<std::size_t>(x);
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

constexpr bool kEnableTerrainDebugLogs = true;

void logTerrainAnomaly(const char* tag,
                       int worldX,
                       int worldZ,
                       int surfaceY,
                       float neighborAverage,
                       const ColumnSample& sample)
{
    if (!kEnableTerrainDebugLogs)
    {
        return;
    }

    static std::mutex s_logMutex;
    static std::ofstream s_logFile("debug_terrain.log", std::ios::app);
    static int s_logCount = 0;
    if (s_logCount >= 500 || !s_logFile.is_open())
    {
        return;
    }

    std::lock_guard<std::mutex> lock(s_logMutex);
    ++s_logCount;
    s_logFile << tag << " world=(" << worldX << ',' << worldZ << ") surfaceY=" << surfaceY
              << " neighborAvg=" << neighborAverage
              << " dominantBiome=" << (sample.dominantBiome ? sample.dominantBiome->id : "<none>")
              << " weight=" << sample.dominantWeight << " roughAmp=" << sample.roughAmplitude
              << " hillAmp=" << sample.hillAmplitude << " mountainAmp=" << sample.mountainAmplitude << '\n';
}

} // namespace

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

ChunkGenerationSummary TerrainGenerator::generateChunkColumns(const glm::ivec3& chunkCoord,
                                                              int minWorldY,
                                                              int maxWorldY,
                                                              int chunkSizeX,
                                                              int chunkSizeY,
                                                              int chunkSizeZ,
                                                              const BlockSetter& setBlock,
                                                              std::span<ColumnBuildResult> outColumns) const
{
    if (outColumns.size() < static_cast<std::size_t>(chunkSizeX * chunkSizeZ))
    {
        throw std::invalid_argument("outColumns span is smaller than the chunk column count");
    }
    if (!setBlock)
    {
        throw std::invalid_argument("TerrainGenerator requires a block setter callback");
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
            const SurfaceColumn& surfaceColumn = surfaceMap_.column(worldX, worldZ);
            const std::size_t idx = columnIndex(dx + 1, dz + 1, neighborSizeX);
            neighborHeights[idx] = surfaceColumn.surfaceY;
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
            result.sample = sampler_(worldX, worldZ, minWorldY, maxWorldY);
            ColumnSample& sample = result.sample;

            const std::size_t columnIdx = columnIndex(localX, localZ, chunkSizeX);
            outColumns[columnIdx] = result;

            if (!sample.dominantBiome)
            {
                continue;
            }

            const BiomeDefinition& biome = *sample.dominantBiome;

            if (!sample.slabHasSolid)
            {
                continue;
            }

            summary.slabContainsTerrain = true;

            const float neighborAverage = computeNeighborAverage(localX, localZ);
            int adjustedSurfaceY = sample.surfaceY;
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
                adjustedSurfaceY = static_cast<int>(std::round(static_cast<float>(adjustedSurfaceY) + offset));
                adjustedSurfaceY = std::clamp(adjustedSurfaceY, sample.minSurfaceY, sample.maxSurfaceY);
            }

            sample.surfaceY = adjustedSurfaceY;
            sample.slabHasSolid = minWorldY <= adjustedSurfaceY;
            sample.slabHighestSolidY = sample.slabHasSolid ? std::min(adjustedSurfaceY, maxWorldY)
                                                           : std::numeric_limits<int>::min();
            outColumns[columnIdx].sample = sample;
            if (kEnableTerrainDebugLogs)
            {
                const float diff = std::abs(static_cast<float>(adjustedSurfaceY) - neighborAverage);
                if (adjustedSurfaceY <= minWorldY + 4 || diff > 48.0f)
                {
                    logTerrainAnomaly("[HeightDebug]", worldX, worldZ, adjustedSurfaceY, neighborAverage, sample);
                }
            }

            BlockId surfaceBlock = biome.surfaceBlock;
            BlockId fillerBlock = biome.fillerBlock;

            const bool nearSeaLevel = std::abs(adjustedSurfaceY - seaLevel_) <= 2;
            constexpr float kBeachDistanceRange = 6.0f;
            if (!biome.isOcean() && nearSeaLevel && sample.distanceToShore <= kBeachDistanceRange)
            {
                const float noise = hashToUnitFloat(worldX, adjustedSurfaceY, worldZ);
                if (biome.terrainSettings.smoothBeaches)
                {
                    const float shorelineWeight = 1.0f - std::clamp(sample.distanceToShore / kBeachDistanceRange, 0.0f, 1.0f);
                    const float sandProbability = glm::mix(0.4f, 0.95f, shorelineWeight);
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


            const int highestSolidWorld = std::min(sample.slabHighestSolidY, maxWorldY);
            if (highestSolidWorld < minWorldY)
            {
                continue;
            }

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

            for (int localY = 0; localY <= highestLocalY; ++localY)
            {
                const int worldY = minWorldY + localY;
                BlockId block = BlockId::Air;
                if (worldY < adjustedSurfaceY)
                {
                    block = fillerBlock;
                    if (columnHasStripes)
                    {
                        const int pattern = (worldY + stripeOffset) % stripePeriod;
                        if (pattern < stripes.thickness)
                        {
                            block = stripes.block;
                        }
                    }
                }
                else if (worldY == adjustedSurfaceY)
                {
                    block = surfaceBlock;
                }

                if (block == BlockId::Air)
                {
                    continue;
                }


                setBlock(localX, localY, localZ, block);
                outColumns[columnIdx].wroteSolid = true;
                summary.anySolid = true;
            }

            outColumns[columnIdx].highestSolidWorld = highestSolidWorld;

            const auto& waterFill = biome.terrainSettings.waterFill;
            if (waterFill.enabled && adjustedSurfaceY < seaLevel_)
            {
                const int waterTop = std::min(seaLevel_, maxWorldY);
                int waterBottom = std::max(highestSolidWorld + 1, minWorldY);
                if (waterFill.maxDepth > 0)
                {
                    waterBottom = std::max(waterBottom, waterTop - waterFill.maxDepth + 1);
                }

                if (waterBottom <= waterTop)
                {
                    for (int worldY = waterBottom; worldY <= waterTop; ++worldY)
                    {
                        const int localY = worldY - minWorldY;
                        if (localY < 0 || localY >= chunkSizeY)
                        {
                            continue;
                        }
                        setBlock(localX, localY, localZ, waterFill.block);
                        outColumns[columnIdx].wroteSolid = true;
                        summary.anySolid = true;
                    }
                    outColumns[columnIdx].waterTopWorld = waterTop;
                }
            }
        }
    }

    return summary;
}

} // namespace terrain

