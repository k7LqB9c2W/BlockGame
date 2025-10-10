#pragma once

#include <array>
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <glm/vec2.hpp>

#include "terrain/biome_database.h"

namespace terrain
{

class ClimateMap;
struct ClimateSample;
struct BiomePerturbationSample;
struct TerrainBasisSample;
struct WorldgenProfile;

struct SurfaceColumn
{
    const BiomeDefinition* dominantBiome{nullptr};
    float dominantWeight{0.0f};
    float surfaceHeight{0.0f};
    int surfaceY{0};
    float roughAmplitude{0.0f};
    float hillAmplitude{0.0f};
    float mountainAmplitude{0.0f};
    float soilCreepCoefficient{0.0f};
};

class SurfaceFragment
{
public:
    static constexpr int kSize = 64;

    SurfaceFragment(const glm::ivec2& fragmentCoord, int lodLevel) noexcept;

    [[nodiscard]] const glm::ivec2& fragmentCoord() const noexcept { return fragmentCoord_; }
    [[nodiscard]] glm::ivec2 baseWorld() const noexcept { return baseWorld_; }
    [[nodiscard]] int lodLevel() const noexcept { return lodLevel_; }
    [[nodiscard]] int stride() const noexcept { return stride_; }

    [[nodiscard]] const SurfaceColumn& column(int localX, int localZ) const noexcept;
    SurfaceColumn& column(int localX, int localZ) noexcept;

private:
    glm::ivec2 fragmentCoord_{0};
    glm::ivec2 baseWorld_{0};
    int lodLevel_{0};
    int stride_{1};
    std::array<SurfaceColumn, kSize * kSize> columns_{};
};

class SurfaceGenerator
{
public:
    virtual ~SurfaceGenerator() = default;

    virtual void generate(SurfaceFragment& fragment, int lodLevel) = 0;
};

class MapGenV1 final : public SurfaceGenerator
{
public:
    MapGenV1(const BiomeDatabase& database,
             const ClimateMap& climateMap,
             const WorldgenProfile& profile,
             unsigned seed);

    void generate(SurfaceFragment& fragment, int lodLevel) override;

private:
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

    const BiomeDatabase& biomeDatabase_;
    const ClimateMap* climateMap_;
    const WorldgenProfile& profile_;
    PerlinNoise offsetNoise_;
    float warpFrequency_{0.0025f};
    float warpAmplitude_{18.0f};
};

class SurfaceMap
{
public:
    SurfaceMap(std::unique_ptr<SurfaceGenerator> generator, std::size_t maxFragments = 32);

    [[nodiscard]] const SurfaceFragment& getFragment(const glm::ivec2& fragmentCoord, int lodLevel = 0) const;
    [[nodiscard]] const SurfaceColumn& column(int worldX, int worldZ, int lodLevel = 0) const;
    void clear();

private:
    struct FragmentKey
    {
        glm::ivec2 coord{0};
        int lod{0};

        bool operator==(const FragmentKey& other) const noexcept
        {
            return coord == other.coord && lod == other.lod;
        }
    };

    struct FragmentKeyHasher
    {
        std::size_t operator()(const FragmentKey& key) const noexcept;
    };

    struct FragmentCacheEntry
    {
        std::unique_ptr<SurfaceFragment> fragment;
        std::list<FragmentKey>::iterator lruIt;
    };

    static int floorDiv(int value, int divisor) noexcept;
    void touch(FragmentCacheEntry& entry) const;
    void evictIfNeeded() const;

    std::unique_ptr<SurfaceGenerator> generator_;
    std::size_t maxFragments_{32};
    mutable std::mutex mutex_;
    mutable std::unordered_map<FragmentKey, FragmentCacheEntry, FragmentKeyHasher> fragments_{};
    mutable std::list<FragmentKey> lru_{};
};

} // namespace terrain

