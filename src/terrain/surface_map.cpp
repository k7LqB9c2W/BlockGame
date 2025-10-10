#include "terrain/surface_map.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <glm/common.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/compatibility.hpp>

#include "terrain/climate_map.h"
#include "terrain/worldgen_profile.h"

namespace terrain
{

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
            const TerrainBasisSample& basis = climateSample.basis;
            const BiomePerturbationSample& perturbations = climateSample.perturbations;

            const BiomeDefinition* dominantBiome = perturbations.dominantBiome;
            if (!dominantBiome)
            {
                dominantBiome = climateSample.fallbackBiome;
            }
            if (!dominantBiome && biomeDatabase_.biomeCount() > 0)
            {
                dominantBiome = &biomeDatabase_.definitionByIndex(0);
            }

            SurfaceColumn& outColumn = fragment.column(localX, localZ);
            outColumn = SurfaceColumn{};

            if (!dominantBiome)
            {
                continue;
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

            const float roughAmplitude = glm::clamp((roughNoise + 1.0f) * 0.5f, 0.0f, 1.0f);
            const float hillAmplitude = glm::clamp((hillNoise + 1.0f) * 0.5f, 0.0f, 1.0f);
            const float mountainAmplitude = glm::clamp(mountainNoise, 0.0f, 1.0f);

            float blendedHeight = perturbations.blendedOffset
                                   + (basis.mainTerrain * 12.0f + basis.mediumNoise * 6.0f + basis.detailNoise * 3.0f)
                                         * perturbations.blendedScale;
            blendedHeight += (roughAmplitude - 0.5f) * 8.0f * perturbations.blendedScale;
            blendedHeight += (hillAmplitude - 0.5f) * 12.0f * perturbations.blendedScale;
            blendedHeight += mountainAmplitude * 24.0f * perturbations.blendedScale;

            const bool hasLand = perturbations.landWeight > 0.0f;
            const bool hasOcean = perturbations.oceanWeight > 0.0f;

            if (hasLand)
            {
                const float landHeight = perturbations.landOffset
                                          + (basis.mainTerrain * 10.0f + hillAmplitude * 8.0f + roughAmplitude * 4.0f)
                                                * perturbations.landScale;
                blendedHeight = glm::mix(blendedHeight, landHeight, glm::clamp(perturbations.landWeight, 0.0f, 1.0f));
            }

            if (hasOcean)
            {
                const float oceanHeight = perturbations.oceanOffset
                                           + (basis.mainTerrain * 6.0f + mountainAmplitude * 4.0f)
                                                 * perturbations.oceanScale;
                blendedHeight = glm::mix(blendedHeight,
                                         oceanHeight,
                                         glm::clamp(perturbations.oceanWeight, 0.0f, 1.0f));
            }

            const float minHeight = std::min(perturbations.blendedMinHeight, perturbations.blendedMaxHeight);
            const float maxHeight = std::max(perturbations.blendedMinHeight, perturbations.blendedMaxHeight);
            const float clampedHeight = glm::clamp(blendedHeight, minHeight, maxHeight);

            outColumn.dominantBiome = dominantBiome;
            outColumn.dominantWeight = perturbations.dominantWeight;
            outColumn.surfaceHeight = clampedHeight;
            outColumn.surfaceY = static_cast<int>(std::round(clampedHeight));
            outColumn.blendedOffset = perturbations.blendedOffset;
            outColumn.blendedScale = perturbations.blendedScale;
            outColumn.roughAmplitude = roughAmplitude;
            outColumn.hillAmplitude = hillAmplitude;
            outColumn.mountainAmplitude = mountainAmplitude;
            outColumn.soilCreepCoefficient = 1.0f - glm::clamp(perturbations.blendedSlopeBias, 0.0f, 1.0f);
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
        auto [it, inserted] = fragments_.emplace(key, FragmentCacheEntry{std::move(fragment), lru_.end()});
        it->second.lruIt = lru_.emplace(lru_.begin(), key);
        touch(it->second);
        evictIfNeeded();
        result = it->second.fragment.get();
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
    if (entry.lruIt != lru_.end())
    {
        lru_.erase(entry.lruIt);
    }

    FragmentKey key{};
    key.coord = entry.fragment->fragmentCoord();
    key.lod = entry.fragment->lodLevel();
    entry.lruIt = lru_.emplace(lru_.begin(), key);
}

void SurfaceMap::evictIfNeeded() const
{
    while (!lru_.empty() && fragments_.size() > maxFragments_)
    {
        const FragmentKey key = lru_.back();
        auto it = fragments_.find(key);
        if (it != fragments_.end())
        {
            fragments_.erase(it);
        }
        lru_.pop_back();
    }
}

} // namespace terrain

