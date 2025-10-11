#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <glm/vec2.hpp>

#include "terrain/biome_database.h"
#include "terrain/worldgen_profile.h"

namespace terrain
{

struct BiomeBlend
{
    const BiomeDefinition* biome{nullptr};
    float weight{0.0f};
    float height{0.0f};
    float roughness{0.0f};
    float hills{0.0f};
    float mountains{0.0f};
    float normalizedDistance{0.0f};
    unsigned seed{0};
    float falloff{1.0f};
};

struct ClimateSample
{
    std::array<BiomeBlend, 4> blends{};
    std::size_t blendCount{0};
    float aggregatedHeight{0.0f};
    float aggregatedRoughness{0.0f};
    float aggregatedHills{0.0f};
    float aggregatedMountains{0.0f};
    float keepOriginalMix{0.0f};
    glm::vec2 dominantSitePos{0.0f};
    glm::vec2 dominantSiteHalfExtents{0.0f};

    [[nodiscard]] const BiomeDefinition* dominantBiome() const noexcept
    {
        return blendCount > 0 ? blends[0].biome : nullptr;
    }

    [[nodiscard]] float dominantWeight() const noexcept
    {
        return blendCount > 0 ? blends[0].weight : 0.0f;
    }
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
    static int floorDiv(int value, int divisor) noexcept;
    static std::array<float, 2> axisInterpolationWeights(float t, BiomeDefinition::InterpolationCurve curve) noexcept;

    const BiomeDefinition& biomeForCell(int cellX, int cellZ) const;
    float sampleBaseHeight(const BiomeDefinition& definition, int cellX, int cellZ) const noexcept;
    void populateBlends(int worldX, int worldZ, ClimateSample& outSample) const;

    const BiomeDatabase& biomeDatabase_;
    const WorldgenProfile& profile_;
    int chunkSize_{16};
    int biomeSizeInChunks_{1};
    float cellSize_{16.0f};
    unsigned baseSeed_{0};
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
        bool inLru{false};
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

