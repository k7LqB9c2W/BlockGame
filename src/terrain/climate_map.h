#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
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
    glm::vec2 sitePosition{0.0f};
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
    float distanceToCoast{std::numeric_limits<float>::infinity()};
    bool dominantIsOcean{false};

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
    struct BiomeSeed
    {
        const BiomeDefinition* biome{nullptr};
        glm::ivec2 position{0};
        float radius{1.0f};
        float weight{1.0f};
        float baseHeight{0.0f};
    };

    struct ChunkSeeds
    {
        std::vector<BiomeSeed> seeds{};
        int maxRadius{0};
    };

    struct ChunkKeyHasher
    {
        std::size_t operator()(const glm::ivec2& value) const noexcept
        {
            std::size_t h1 = std::hash<int>{}(value.x);
            std::size_t h2 = std::hash<int>{}(value.y);
            return h1 ^ (h2 + 0x9E3779B97f4A7C15ull + (h1 << 6) + (h1 >> 2));
        }
    };

    class Random
    {
    public:
        explicit Random(std::uint64_t seed) noexcept : state_(seed) {}

        std::uint32_t next() noexcept
        {
            state_ ^= state_ >> 12;
            state_ ^= state_ << 25;
            state_ ^= state_ >> 27;
            return static_cast<std::uint32_t>((state_ * 2685821657736338717ull) >> 32);
        }

        float nextFloat() noexcept
        {
            return static_cast<float>(next()) / static_cast<float>(std::numeric_limits<std::uint32_t>::max());
        }

        int nextInt(int minInclusive, int maxInclusive) noexcept
        {
            if (maxInclusive <= minInclusive)
            {
                return minInclusive;
            }
            const std::uint32_t range = static_cast<std::uint32_t>(maxInclusive - minInclusive + 1);
            return minInclusive + static_cast<int>(next() % range);
        }

        float nextFloatSigned() noexcept
        {
            return nextFloat() * 2.0f - 1.0f;
        }

    private:
        std::uint64_t state_;
    };

    static int floorDiv(int value, int divisor) noexcept;
    static float smoothStep(float t) noexcept;
    static float lengthSquared(const glm::ivec2& a, const glm::ivec2& b) noexcept;

    const BiomeDatabase& biomeDatabase_;
    const WorldgenProfile& profile_;
    unsigned baseSeed_{0};
    int chunkSpan_{512};
    int neighborRadius_{2};
    int maxTransitionWidth_{0};

    std::vector<const BiomeDefinition*> biomeSelection_{};
    std::vector<float> biomeWeightPrefix_{};
    float totalSpawnWeight_{0.0f};
    std::vector<const BiomeDefinition*> oceanBiomes_{};
    std::vector<float> oceanWeightPrefix_{};
    float totalOceanWeight_{0.0f};

    mutable std::unordered_map<glm::ivec2, ChunkSeeds, ChunkKeyHasher> chunkCache_{};
    mutable std::mutex chunkMutex_;

    const ChunkSeeds& chunkSeeds(int chunkX, int chunkZ) const;
    ChunkSeeds buildChunkSeeds(int chunkX, int chunkZ) const;
    BiomeSeed createSeed(Random& rng, int worldX, int worldZ) const;
    BiomeSeed createSeed(Random& rng, int worldX, int worldZ, const BiomeDefinition& biome) const;
    const BiomeDefinition& chooseBiome(Random& rng) const;
    const BiomeDefinition& chooseOceanBiome(Random& rng) const;
    float randomizedHeight(Random& rng, const BiomeDefinition& biome) const noexcept;
    bool isValidPlacement(const glm::ivec2& position,
                          float radius,
                          const std::vector<BiomeSeed>& seeds) const noexcept;
    bool isValidPlacement(const glm::ivec2& position,
                          float radius,
                          const std::vector<BiomeSeed>& seeds,
                          float spacingScale) const noexcept;
    void gatherCandidateSeeds(const glm::ivec2& worldPos,
                              std::vector<const BiomeSeed*>& outCandidates) const;
    void accumulateSample(const glm::ivec2& worldPos, ClimateSample& outSample) const;
    void applyTransitionBiomes(const glm::ivec2& baseWorld, ClimateFragment& fragment) const;
    void spawnSubBiomeSeeds(const BiomeSeed& parent,
                            std::vector<BiomeSeed>& seeds,
                            Random& rng) const;
    static glm::vec2 randomInUnitCircle(Random& rng) noexcept;
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

