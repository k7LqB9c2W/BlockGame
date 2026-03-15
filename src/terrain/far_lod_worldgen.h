#pragma once

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
};

struct FarLodGpuWorldgenHeader
{
    std::int32_t seaLevel{20};
    std::uint32_t seed{0};
    std::uint32_t biomeCount{0};
    float warpFrequency{0.0025f};
    float warpAmplitude{18.0f};
    FbmSettings mainNoise{};
    FbmSettings mediumNoise{};
    FbmSettings detailNoise{};
    FbmSettings mountainNoise{};
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
    float baseSlopeBias{0.0f};
    float maxGradient{0.0f};
    float footprintMultiplier{1.0f};
};

struct FarLodWorldgenTables
{
    FarLodGpuWorldgenHeader header{};
    std::vector<FarLodGpuBiome> biomes;
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
