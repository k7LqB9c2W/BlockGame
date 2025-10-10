#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "terrain/biome_database.h"

enum class BlockId : std::uint8_t;

namespace terrain
{

class ClimateMap;
class SurfaceMap;
struct SurfaceColumn;
struct ClimateSample;
struct BiomePerturbationSample;
struct TerrainBasisSample;
struct WorldgenProfile;

struct ColumnSample
{
    const BiomeDefinition* dominantBiome{nullptr};
    float dominantWeight{0.0f};
    int surfaceY{0};
    int minSurfaceY{0};
    int maxSurfaceY{0};
    int slabHighestSolidY{std::numeric_limits<int>::min()};
    bool slabHasSolid{false};
    float soilCreepCoefficient{0.0f};
    float roughAmplitude{0.0f};
    float hillAmplitude{0.0f};
    float mountainAmplitude{0.0f};
    float distanceToShore{0.0f};
};

struct ColumnBuildResult
{
    ColumnSample sample{};
    int highestSolidWorld{std::numeric_limits<int>::min()};
    int waterTopWorld{std::numeric_limits<int>::min()};
    bool wroteSolid{false};
};

struct ChunkGenerationSummary
{
    bool slabContainsTerrain{false};
    bool anySolid{false};
};

class TerrainGenerator
{
public:
    using SampleColumnFn = std::function<ColumnSample(int worldX, int worldZ, int slabMinWorldY, int slabMaxWorldY)>;
    using BlockSetter = std::function<void(int localX, int localY, int localZ, BlockId block)>;

    TerrainGenerator(const ClimateMap& climateMap,
                     const SurfaceMap& surfaceMap,
                     const BiomeDatabase& biomeDatabase,
                     int seaLevel,
                     SampleColumnFn sampler);

    ChunkGenerationSummary generateChunkColumns(const glm::ivec3& chunkCoord,
                                                int minWorldY,
                                                int maxWorldY,
                                                int chunkSizeX,
                                                int chunkSizeY,
                                                int chunkSizeZ,
                                                const BlockSetter& setBlock,
                                                std::span<ColumnBuildResult> outColumns) const;

private:
    const ClimateMap& climateMap_;
    const SurfaceMap& surfaceMap_;
    const BiomeDatabase& biomeDatabase_;
    int seaLevel_{0};
    SampleColumnFn sampler_;
};

} // namespace terrain

