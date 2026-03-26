#pragma once

#include <array>
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
    struct BlendDebugInfo
    {
        const BiomeDefinition* biome{nullptr};
        float weight{0.0f};
        float aggregatedHeight{0.0f};
        float normalizedDistance{0.0f};
        float seedRadius{0.0f};
        float worldDistance{0.0f};
        bool isOcean{false};
    };

    const BiomeDefinition* dominantBiome{nullptr};
    float dominantWeight{0.0f};
    float surfaceHeight{0.0f};
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
    int originalSurfaceY{0};
    float soilCreepOffset{0.0f};
    bool dominantIsOcean{false};
    float distanceToCoast{0.0f};
    std::array<BlendDebugInfo, 4> topBlendDebug{};
    std::size_t topBlendCount{0};
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

struct ExactChunkColumnDescriptor
{
    static constexpr std::uint32_t kInvalidBiomeIndex = std::numeric_limits<std::uint32_t>::max();
    static constexpr std::uint32_t kFlagHasBiome = 1u << 0;
    static constexpr std::uint32_t kFlagHasSolid = 1u << 1;
    static constexpr std::uint32_t kFlagHasWater = 1u << 2;
    static constexpr std::uint32_t kFlagStripesEnabled = 1u << 3;
    static constexpr std::uint32_t kFlagColumnHasStripes = 1u << 4;
    static constexpr std::uint32_t kFlagDominantIsOcean = 1u << 5;

    [[nodiscard]] bool hasBiome() const noexcept { return (flags & kFlagHasBiome) != 0; }
    [[nodiscard]] bool hasSolid() const noexcept { return (flags & kFlagHasSolid) != 0; }
    [[nodiscard]] bool hasWater() const noexcept { return (flags & kFlagHasWater) != 0; }
    [[nodiscard]] bool stripesEnabled() const noexcept { return (flags & kFlagStripesEnabled) != 0; }
    [[nodiscard]] bool columnHasStripes() const noexcept { return (flags & kFlagColumnHasStripes) != 0; }

    std::uint32_t flags{0};
    std::uint32_t biomeIndex{kInvalidBiomeIndex};
    std::int32_t surfaceY{0};
    std::int32_t originalSurfaceY{0};
    std::int32_t minSurfaceY{0};
    std::int32_t maxSurfaceY{0};
    std::int32_t highestSolidWorld{std::numeric_limits<int>::min()};
    std::int32_t waterTopWorld{std::numeric_limits<int>::min()};
    std::int32_t waterBottomWorld{std::numeric_limits<int>::max()};
    std::int32_t stripeOffset{0};
    std::uint16_t stripePeriod{0};
    std::uint16_t stripeThickness{0};
    BlockId surfaceBlock{};
    BlockId fillerBlock{};
    BlockId waterBlock{};
    BlockId stripeBlock{};
};

struct TerrainColumnBlocks
{
    BlockId surfaceBlock{};
    BlockId fillerBlock{};
};

[[nodiscard]] bool isTaigaBiome(const BiomeDefinition& biome) noexcept;
[[nodiscard]] TerrainColumnBlocks resolveTerrainColumnBlocks(const BiomeDefinition& biome,
                                                            const ColumnSample& sample,
                                                            int worldX,
                                                            int worldZ,
                                                            int seaLevel) noexcept;

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

    ChunkGenerationSummary describeChunkColumns(const glm::ivec3& chunkCoord,
                                                int minWorldY,
                                                int maxWorldY,
                                                int chunkSizeX,
                                                int chunkSizeY,
                                                int chunkSizeZ,
                                                std::span<ExactChunkColumnDescriptor> outDescriptors,
                                                std::span<ColumnBuildResult> outColumns) const;
    void materializeChunkColumns(int minWorldY,
                                 int maxWorldY,
                                 int chunkSizeX,
                                 int chunkSizeY,
                                 int chunkSizeZ,
                                 std::span<const ExactChunkColumnDescriptor> descriptors,
                                 const BlockSetter& setBlock) const;
    ChunkGenerationSummary generateChunkColumns(const glm::ivec3& chunkCoord,
                                                int minWorldY,
                                                int maxWorldY,
                                                int chunkSizeX,
                                                int chunkSizeY,
                                                int chunkSizeZ,
                                                const BlockSetter& setBlock,
                                                std::span<ExactChunkColumnDescriptor> outDescriptors,
                                                std::span<ColumnBuildResult> outColumns) const;

private:
    const ClimateMap& climateMap_;
    const SurfaceMap& surfaceMap_;
    const BiomeDatabase& biomeDatabase_;
    int seaLevel_{0};
    SampleColumnFn sampler_;
};

} // namespace terrain

