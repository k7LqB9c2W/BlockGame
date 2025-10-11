#include "terrain/surface_map.h"

#include <algorithm>
#include <array>
#include <limits>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <glm/common.hpp>
#include <glm/gtc/constants.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/compatibility.hpp>
#undef GLM_ENABLE_EXPERIMENTAL

#include "terrain/climate_map.h"
#include "terrain/worldgen_profile.h"

namespace terrain
{
namespace
{
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
} // namespace

SurfaceFragment::SurfaceFragment(const glm::ivec2& fragmentCoord, int lodLevel) noexcept
    : fragmentCoord_(fragmentCoord),
      baseWorld_(fragmentCoord * (kSize << lodLevel)),
      lodLevel_(lodLevel),
      stride_(1 << lodLevel)
{
}

const SurfaceColumn& SurfaceFragment::column(int localX, int localZ) const noexcept
{
    const int clampedX = std::clamp(localX, 0, kSize - 1);
    const int clampedZ = std::clamp(localZ, 0, kSize - 1);
    const std::size_t index = static_cast<std::size_t>(clampedZ) * kSize + static_cast<std::size_t>(clampedX);
    return columns_[index];
}

SurfaceColumn& SurfaceFragment::column(int localX, int localZ) noexcept
{
    const int clampedX = std::clamp(localX, 0, kSize - 1);
    const int clampedZ = std::clamp(localZ, 0, kSize - 1);
    const std::size_t index = static_cast<std::size_t>(clampedZ) * kSize + static_cast<std::size_t>(clampedX);
    return columns_[index];
}

MapGenV1::PerlinNoise::PerlinNoise(unsigned seed)
{
    std::array<int, 256> p{};
    for (int i = 0; i < 256; ++i)
    {
        p[i] = i;
    }

    std::mt19937 rng(seed);
    std::shuffle(p.begin(), p.end(), rng);

    for (int i = 0; i < 256; ++i)
    {
        permutation_[i] = p[i];
        permutation_[256 + i] = p[i];
    }
}

float MapGenV1::PerlinNoise::noise(float x, float y) const noexcept
{
    const int xi = static_cast<int>(std::floor(x)) & 255;
    const int yi = static_cast<int>(std::floor(y)) & 255;
    const float xf = x - std::floor(x);
    const float yf = y - std::floor(y);

    const float u = fade(xf);
    const float v = fade(yf);

    const int aa = permutation_[xi] + yi;
    const int ab = permutation_[xi] + yi + 1;
    const int ba = permutation_[xi + 1] + yi;
    const int bb = permutation_[xi + 1] + yi + 1;

    const float x1 = lerp(grad(permutation_[aa], xf, yf), grad(permutation_[ba], xf - 1.0f, yf), u);
    const float x2 = lerp(grad(permutation_[ab], xf, yf - 1.0f), grad(permutation_[bb], xf - 1.0f, yf - 1.0f), u);
    return lerp(x1, x2, v);
}

float MapGenV1::PerlinNoise::fbm(float x,
                                 float y,
                                 int octaves,
                                 float persistence,
                                 float lacunarity) const noexcept
{
    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;

    for (int i = 0; i < octaves; ++i)
    {
        total += noise(x * frequency, y * frequency) * amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    return total;
}

float MapGenV1::PerlinNoise::ridge(float x,
                                   float y,
                                   int octaves,
                                   float lacunarity,
                                   float gain) const noexcept
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    float prev = 1.0f;

    for (int i = 0; i < octaves; ++i)
    {
        float n = noise(x * frequency, y * frequency);
        n = 1.0f - std::abs(n);
        n *= n;
        sum += n * amplitude * prev;
        prev = n;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return sum;
}

float MapGenV1::PerlinNoise::fade(float t) noexcept
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float MapGenV1::PerlinNoise::lerp(float a, float b, float t) noexcept
{
    return a + t * (b - a);
}

float MapGenV1::PerlinNoise::grad(int hash, float x, float y) noexcept
{
    const int h = hash & 7;
    const float u = h < 4 ? x : y;
    const float v = h < 4 ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

MapGenV1::MapGenV1(const BiomeDatabase& database,
                   const ClimateMap& climateMap,
                   const WorldgenProfile& profile,
                   unsigned seed)
    : biomeDatabase_(database),
      climateMap_(&climateMap),
      profile_(profile),
      offsetNoise_(seed ^ 0xA511E9B7u)
{
}

void MapGenV1::generate(SurfaceFragment& fragment, int lodLevel)
{
    if (!climateMap_)
    {
        throw std::runtime_error("Surface generator requires a climate map");
    }

    const int stride = std::max(1, fragment.stride());
    const glm::ivec2 baseWorld = fragment.baseWorld();
    const auto& noiseProfile = profile_.noise;

    const float roughFrequency = noiseProfile.detail.frequency;
    const float hillFrequency = noiseProfile.medium.frequency;
    const float mountainFrequency = noiseProfile.mountain.frequency;

    for (int localZ = 0; localZ < SurfaceFragment::kSize; ++localZ)
    {
        for (int localX = 0; localX < SurfaceFragment::kSize; ++localX)
        {
            const int worldX = baseWorld.x + localX * stride;
            const int worldZ = baseWorld.y + localZ * stride;

            const ClimateSample& climateSample = climateMap_->sample(worldX, worldZ);
            if (climateSample.blendCount == 0)
            {
                continue;
            }

            SurfaceColumn& outColumn = fragment.column(localX, localZ);
            outColumn = SurfaceColumn{};

            const BiomeBlend& dominantBlend = climateSample.blends[0];
            const BiomeDefinition* dominantBiome = dominantBlend.biome;
            if (!dominantBiome && biomeDatabase_.biomeCount() > 0)
            {
                dominantBiome = &biomeDatabase_.definitionByIndex(0);
            }
            if (!dominantBiome)
            {
                continue;
            }

            float roughStrength = std::max(climateSample.aggregatedRoughness, 0.0f);
            float hillStrength = std::max(climateSample.aggregatedHills, 0.0f);
            float mountainStrength = std::max(climateSample.aggregatedMountains, 0.0f);
            const float keepOriginal = std::clamp(climateSample.keepOriginalMix, 0.0f, 1.0f);
            float baseHeight = climateSample.aggregatedHeight;

            if (std::isfinite(climateSample.distanceToCoast))
            {
                constexpr float kBeachSmoothingRange = 16.0f;
                const float t = std::clamp(climateSample.distanceToCoast / kBeachSmoothingRange, 0.0f, 1.0f);
                const bool biomePrefersBeach = dominantBiome->terrainSettings.smoothBeaches || dominantBiome->isOcean();
                const float minFactor = biomePrefersBeach ? 0.1f : 0.25f;
                const float smoothFactor = glm::mix(minFactor, 1.0f, t);
                roughStrength *= smoothFactor;
                hillStrength *= smoothFactor;
                mountainStrength *= smoothFactor;
                const float heightBlend = glm::mix(0.0f, 1.0f, t);
                baseHeight = glm::mix(static_cast<float>(profile_.seaLevel), baseHeight, heightBlend);
            }

            const float worldXF = static_cast<float>(worldX);
            const float worldZF = static_cast<float>(worldZ);

            const float warpSample = offsetNoise_.fbm(worldXF * warpFrequency_,
                                                      worldZF * warpFrequency_,
                                                      4,
                                                      0.5f,
                                                      2.03f);
            const float warpedX = worldXF + warpSample * warpAmplitude_;
            const float warpedZ = worldZF + warpSample * warpAmplitude_;

            const float roughNoise = offsetNoise_.fbm(warpedX * roughFrequency,
                                                      warpedZ * roughFrequency,
                                                      noiseProfile.detail.octaves,
                                                      noiseProfile.detail.gain,
                                                      noiseProfile.detail.lacunarity);
            const float hillNoise = offsetNoise_.fbm(warpedX * hillFrequency,
                                                     warpedZ * hillFrequency,
                                                     noiseProfile.medium.octaves,
                                                     noiseProfile.medium.gain,
                                                     noiseProfile.medium.lacunarity);
            const float mountainNoise = offsetNoise_.ridge(warpedX * mountainFrequency,
                                                            warpedZ * mountainFrequency,
                                                            noiseProfile.mountain.octaves,
                                                            noiseProfile.mountain.lacunarity,
                                                            noiseProfile.mountain.gain);

            float surfaceHeight = baseHeight;
            surfaceHeight += (roughNoise - 0.5f) * 4.0f * roughStrength;
            surfaceHeight += (hillNoise - 0.5f) * 6.0f * hillStrength;
            surfaceHeight += mountainNoise * 12.0f * mountainStrength;

            outColumn.dominantBiome = dominantBiome;
            outColumn.dominantWeight = dominantBlend.weight;
            outColumn.surfaceHeight = surfaceHeight;
            outColumn.surfaceY = static_cast<int>(std::round(surfaceHeight));
            outColumn.roughAmplitude = roughStrength;
            outColumn.hillAmplitude = hillStrength;
            outColumn.mountainAmplitude = mountainStrength;
            outColumn.soilCreepCoefficient = std::clamp(1.0f - keepOriginal, 0.0f, 1.0f);
        }
    }
}

SurfaceMap::SurfaceMap(std::unique_ptr<SurfaceGenerator> generator, std::size_t maxFragments)
    : generator_(std::move(generator)),
      maxFragments_(std::max<std::size_t>(maxFragments, 1))
{
    if (!generator_)
    {
        throw std::invalid_argument("SurfaceMap requires a generator");
    }
}

const SurfaceFragment& SurfaceMap::getFragment(const glm::ivec2& fragmentCoord, int lodLevel) const
{
    const FragmentKey key{fragmentCoord, lodLevel};

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = fragments_.find(key);
        if (it != fragments_.end())
        {
            touch(it->second);
            return *it->second.fragment;
        }
    }

    auto fragment = std::make_unique<SurfaceFragment>(fragmentCoord, lodLevel);
    generator_->generate(*fragment, lodLevel);

    const SurfaceFragment* result = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto existing = fragments_.find(key);
        if (existing != fragments_.end())
        {
            touch(existing->second);
            result = existing->second.fragment.get();
        }
        else
        {
            FragmentCacheEntry entry{};
            entry.fragment = std::move(fragment);
            entry.lruIt = lru_.emplace(lru_.begin(), key);
            entry.inLru = true;

            auto [it, inserted] = fragments_.emplace(key, std::move(entry));
            (void)inserted;

            evictIfNeeded();
            result = it->second.fragment.get();
        }
    }

    return *result;
}

const SurfaceColumn& SurfaceMap::column(int worldX, int worldZ, int lodLevel) const
{
    const int stride = 1 << lodLevel;
    const int fragmentSpan = SurfaceFragment::kSize * stride;
    const glm::ivec2 fragmentCoord{floorDiv(worldX, fragmentSpan), floorDiv(worldZ, fragmentSpan)};
    const SurfaceFragment& fragment = getFragment(fragmentCoord, lodLevel);
    const glm::ivec2 baseWorld = fragment.baseWorld();
    const int localX = (worldX - baseWorld.x) / stride;
    const int localZ = (worldZ - baseWorld.y) / stride;
    return fragment.column(localX, localZ);
}

void SurfaceMap::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    fragments_.clear();
    lru_.clear();
}

std::size_t SurfaceMap::FragmentKeyHasher::operator()(const FragmentKey& key) const noexcept
{
    std::size_t h1 = std::hash<int>{}(key.coord.x);
    std::size_t h2 = std::hash<int>{}(key.coord.y);
    std::size_t h3 = std::hash<int>{}(key.lod);
    std::size_t combined = h1 ^ (h2 + 0x9E3779B97f4A7C15ull + (h1 << 6) + (h1 >> 2));
    combined ^= h3 + 0x517CC1B727220A95ull + (combined << 6) + (combined >> 2);
    return combined;
}

int SurfaceMap::floorDiv(int value, int divisor) noexcept
{
    int quotient = value / divisor;
    int remainder = value % divisor;
    if ((remainder != 0) && ((remainder < 0) != (divisor < 0)))
    {
        --quotient;
    }
    return quotient;
}

void SurfaceMap::touch(FragmentCacheEntry& entry) const
{
    if (entry.inLru)
    {
        lru_.erase(entry.lruIt);
    }

    FragmentKey key{};
    key.coord = entry.fragment->fragmentCoord();
    key.lod = entry.fragment->lodLevel();
    entry.lruIt = lru_.emplace(lru_.begin(), key);
    entry.inLru = true;
}

void SurfaceMap::evictIfNeeded() const
{
    while (fragments_.size() > maxFragments_ && !lru_.empty())
    {
        auto lruIt = std::prev(lru_.end());
        const FragmentKey key = *lruIt;

        auto fragIt = fragments_.find(key);
        if (fragIt != fragments_.end())
        {
            fragIt->second.inLru = false;
            fragments_.erase(fragIt);
        }

        lru_.erase(lruIt);
    }
}

} // namespace terrain

