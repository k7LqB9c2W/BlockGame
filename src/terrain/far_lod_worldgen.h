#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "terrain/biome_database.h"
#include "terrain/worldgen_profile.h"

enum class BlockId : std::uint8_t;

namespace terrain
{

enum class FarLodCoastProfile : std::uint32_t
{
    GentleBeach = 0,
    Dunes = 1,
    RockyShore = 2,
    CliffCoast = 3,
    Marsh = 4
};

enum FarLodBiomeFlags : std::uint32_t
{
    kFarLodBiomeOcean = 1u << 0,
    kFarLodBiomeSmoothBeaches = 1u << 1,
    kFarLodBiomeWaterFill = 1u << 2,
    kFarLodBiomeTaiga = 1u << 3,
    kFarLodBiomeGeneratesTrees = 1u << 4,
    kFarLodBiomeBeach = 1u << 5,
    kFarLodBiomeCoastal = 1u << 6,
};

struct FarLodGpuFloat2
{
    float x{0.0f};
    float y{0.0f};
};

struct FarLodGpuWorldgenHeader
{
    std::int32_t seaLevel{20};
    std::uint32_t seed{0};
    std::uint32_t biomeCount{0};
    std::uint32_t biomeSelectionCount{0};
    std::uint32_t oceanSelectionCount{0};
    std::uint32_t transitionCount{0};
    std::uint32_t subBiomeCount{0};
    std::int32_t chunkSpan{0};
    std::int32_t neighborRadius{0};
    std::int32_t maxTransitionWidth{0};
    float totalSpawnWeight{0.0f};
    float totalOceanWeight{0.0f};
    float coastDistanceFieldRange{96.0f};
    float warpFrequency{0.0025f};
    float warpAmplitude{18.0f};
    FbmSettings mainNoise{};
    FbmSettings mediumNoise{};
    FbmSettings detailNoise{};
    FbmSettings mountainNoise{};
    float treeDensityFrequency{0.05f};
    std::uint32_t treeDensityOctaves{4};
    float treeDensityGain{0.55f};
    float treeDensityLacunarity{2.0f};
    std::array<FarLodGpuFloat2, 4> treeDensityOctaveOffsets{};
};

struct FarLodGpuBiome
{
    std::uint32_t surfaceBlock{0};
    std::uint32_t fillerBlock{0};
    std::uint32_t flags{0};
    std::uint32_t coastProfile{0};
    std::uint32_t propertyBits{0};
    std::int32_t waterMaxDepth{0};
    float spawnChance{1.0f};
    float minHeight{0.0f};
    float maxHeight{0.0f};
    float heightOffset{0.0f};
    float heightScale{0.0f};
    float roughness{0.0f};
    float hills{0.0f};
    float mountains{0.0f};
    float keepOriginalTerrain{0.0f};
    float interpolationWeight{1.0f};
    std::uint32_t interpolationCurve{0};
    float radius{256.0f};
    float radiusVariation{0.0f};
    std::uint32_t fixedRadius{0};
    float treeDensityMultiplier{0.0f};
    float maxSubBiomeCount{0.0f};
    float subBiomeTotalChance{0.0f};
    std::int32_t minHeightLimit{0};
    std::int32_t maxHeightLimit{0};
    std::uint32_t hasMinHeightLimit{0};
    std::uint32_t hasMaxHeightLimit{0};
    float baseSlopeBias{0.0f};
    float maxGradient{0.0f};
    float footprintMultiplier{1.0f};
    std::uint32_t transitionOffset{0};
    std::uint32_t transitionCount{0};
    std::uint32_t subBiomeOffset{0};
    std::uint32_t subBiomeCount{0};
};

struct FarLodGpuBiomeSelection
{
    std::uint32_t biomeIndex{0};
    float prefixWeight{0.0f};
    std::uint32_t reserved0{0};
    std::uint32_t reserved1{0};
};

struct FarLodGpuTransitionBiome
{
    std::uint32_t biomeIndex{0};
    float chance{1.0f};
    std::int32_t width{0};
    std::uint32_t propertyBits{0};
};

struct FarLodGpuSubBiome
{
    std::uint32_t biomeIndex{0};
    float chance{0.0f};
    float minRadius{0.0f};
    float maxRadius{0.0f};
};

struct FarLodWorldgenTables
{
    FarLodGpuWorldgenHeader header{};
    std::vector<FarLodGpuBiome> biomes;
    std::vector<FarLodGpuBiomeSelection> biomeSelections;
    std::vector<FarLodGpuBiomeSelection> oceanSelections;
    std::vector<FarLodGpuTransitionBiome> transitionBiomes;
    std::vector<FarLodGpuSubBiome> subBiomes;
    std::vector<std::uint32_t> surfacePermutation;
};

struct FarLodColumnSample
{
    std::uint32_t biomeIndex{0};
    bool hasSolid{false};
    bool waterEnabled{false};
    bool dominantIsOcean{false};
    int surfaceY{0};
    int waterBottomY{0};
    float distanceToShore{0.0f};
    BlockId surfaceBlock{};
    BlockId fillerBlock{};
};

[[nodiscard]] FarLodWorldgenTables buildFarLodWorldgenTables(const BiomeDatabase& biomeDatabase,
                                                             const WorldgenProfile& worldgenProfile,
                                                             unsigned seed);

} // namespace terrain
