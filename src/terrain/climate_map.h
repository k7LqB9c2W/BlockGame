#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <glm/vec2.hpp>

#include "terrain/biome_database.h"
#include "terrain/worldgen_profile.h"

namespace terrain
{

struct TerrainBasisSample
{
    float continentMask{0.0f};
    float mainTerrain{0.0f};
    float mountainNoise{0.0f};
    float mediumNoise{0.0f};
    float detailNoise{0.0f};
    float combinedNoise{0.0f};
    float baseElevation{0.0f};
};

struct BiomePerturbationSample
{
    float blendedOffset{0.0f};
    float blendedScale{0.0f};
    float blendedMinHeight{0.0f};
    float blendedMaxHeight{0.0f};
    float blendedSlopeBias{0.0f};
    float blendedMaxGradient{0.0f};
    float oceanWeight{0.0f};
    float oceanOffset{0.0f};
    float oceanScale{0.0f};
    float oceanMinHeight{0.0f};
    float oceanMaxHeight{0.0f};
    float oceanSlopeBias{0.0f};
    float oceanMaxGradient{0.0f};
    float landWeight{0.0f};
    float landOffset{0.0f};
    float landScale{0.0f};
    float landMinHeight{0.0f};
    float landMaxHeight{0.0f};
    float landSlopeBias{0.0f};
    float landMaxGradient{0.0f};
    const BiomeDefinition* dominantBiome{nullptr};
    float dominantWeight{0.0f};
};

struct ClimateSample
{
    TerrainBasisSample basis{};
    BiomePerturbationSample perturbations{};
    BiomePerturbationSample borderPerturbations{};
    bool hasBorderPerturbations{false};
    float littleMountainsWeight{0.0f};
    const BiomeDefinition* littleMountainsDefinition{nullptr};
    float littleMountainInteriorMask{0.0f};
    bool hasNonLittleMountainsBiome{false};
    const BiomeDefinition* fallbackBiome{nullptr};
};

class ClimateFragment
{
public:
    static constexpr int kSize = 256;

    explicit ClimateFragment(const glm::ivec2& fragmentCoord) noexcept;

    [[nodiscard]] const glm::ivec2& fragmentCoord() const noexcept { return fragmentCoord_; }
    [[nodiscard]] glm::ivec2 baseWorld() const noexcept { return baseWorld_; }

    [[nodiscard]] const ClimateSample& sample(int localX, int localZ) const noexcept;
    ClimateSample& sample(int localX, int localZ) noexcept;

private:
    glm::ivec2 fragmentCoord_{0};
    glm::ivec2 baseWorld_{0};
    std::array<ClimateSample, kSize * kSize> samples_{};
};

class ClimateGenerator
{
public:
    virtual ~ClimateGenerator() = default;

    virtual void generate(ClimateFragment& fragment) = 0;
};

class NoiseVoronoiClimateGenerator final : public ClimateGenerator
{
public:
    NoiseVoronoiClimateGenerator(const BiomeDatabase& database,
                                 const WorldgenProfile& profile,
                                 unsigned seed,
                                 int chunkSize,
                                 int biomeSizeInChunks);

    void generate(ClimateFragment& fragment) override;

private:
    struct WeightedBiome
    {
        const BiomeDefinition* biome{nullptr};
        float weight{0.0f};
    };

    struct BiomeSite
    {
        glm::vec2 worldPosXZ{0.0f};
        glm::vec2 halfExtents{0.0f};
    };

    class PerlinNoise
    {
    public:
        explicit PerlinNoise(unsigned seed = 2025u);

        float noise(float x, float y) const noexcept;
        float fbm(float x, float y, int octaves, float persistence, float lacunarity) const noexcept;
        float ridge(float x, float y, int octaves, float lacunarity, float gain) const noexcept;

    private:
        std::array<int, 512> permutation_{};

        static float fade(float t) noexcept;
        static float lerp(float a, float b, float t) noexcept;
        static float grad(int hash, float x, float y) noexcept;
    };

    static int ceilToIntPositive(float value) noexcept;
    static int floorDiv(int value, int divisor) noexcept;
    static float hashToUnitFloat(int x, int y, int z) noexcept;
    static float littleMountainInfluence(float normalizedDistance) noexcept;

    TerrainBasisSample computeTerrainBasis(int worldX, int worldZ) const;
    BiomePerturbationSample applyBiomePerturbations(const std::array<WeightedBiome, 5>& weightedBiomes,
                                                    std::size_t weightCount,
                                                    int biomeRegionX,
                                                    int biomeRegionZ) const;
    BiomeSite computeBiomeSite(const BiomeDefinition& definition, int regionX, int regionZ) const noexcept;
    const BiomeDefinition& biomeForRegion(int regionX, int regionZ) const;

    const BiomeDatabase& biomeDatabase_;
    const WorldgenProfile& profile_;
    int chunkSize_{16};
    int biomeSizeInChunks_{1};
    int biomeRegionSearchRadius_{1};
    std::size_t biomeRegionCandidateCapacity_{1};
    PerlinNoise noise_;
};

class ClimateMap
{
public:
    ClimateMap(std::unique_ptr<ClimateGenerator> generator, std::size_t maxFragments = 32);

    [[nodiscard]] const ClimateSample& sample(int worldX, int worldZ) const;
    void clear();

private:
    struct IVec2Hasher
    {
        std::size_t operator()(const glm::ivec2& value) const noexcept
        {
            std::size_t h1 = std::hash<int>{}(value.x);
            std::size_t h2 = std::hash<int>{}(value.y);
            return h1 ^ (h2 + 0x9E3779B97f4A7C15ull + (h1 << 6) + (h1 >> 2));
        }
    };

    struct FragmentCacheEntry
    {
        std::unique_ptr<ClimateFragment> fragment;
        std::list<glm::ivec2>::iterator lruIt;
    };

    static int floorDiv(int value, int divisor) noexcept;

    [[nodiscard]] const ClimateFragment& fragmentForColumn(int worldX, int worldZ) const;
    void touch(FragmentCacheEntry& entry) const;
    void evictIfNeeded() const;

    std::unique_ptr<ClimateGenerator> generator_;
    std::size_t maxFragments_{32};
    mutable std::mutex mutex_;
    mutable std::unordered_map<glm::ivec2, FragmentCacheEntry, IVec2Hasher> fragments_{};
    mutable std::list<glm::ivec2> lru_{};
};

} // namespace terrain

