#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace
{

struct OpenSimplexNoise
{
    explicit OpenSimplexNoise(unsigned seed = 2025u)
    {
        std::array<int, 256> temp;
        std::iota(temp.begin(), temp.end(), 0);

        std::mt19937 rng(seed);
        std::shuffle(temp.begin(), temp.end(), rng);

        for (int i = 0; i < 256; ++i)
        {
            const int value = temp[static_cast<std::size_t>(i)];
            permutation_[i] = permutation_[i + 256] = value;
            permutationMod8_[i] = permutationMod8_[i + 256] = value & 7;
        }
    }

    float noise(float x, float y) const noexcept
    {
        constexpr float F2 = 0.3660254037844386f;
        constexpr float G2 = 0.21132486540518713f;

        const float s = (x + y) * F2;
        const int i = static_cast<int>(std::floor(x + s));
        const int j = static_cast<int>(std::floor(y + s));
        const float t = static_cast<float>(i + j) * G2;
        const float X0 = static_cast<float>(i) - t;
        const float Y0 = static_cast<float>(j) - t;
        const float x0 = x - X0;
        const float y0 = y - Y0;

        const int i1 = x0 > y0 ? 1 : 0;
        const int j1 = x0 > y0 ? 0 : 1;

        const float x1 = x0 - static_cast<float>(i1) + G2;
        const float y1 = y0 - static_cast<float>(j1) + G2;
        const float x2 = x0 - 1.0f + 2.0f * G2;
        const float y2 = y0 - 1.0f + 2.0f * G2;

        const int ii = i & 255;
        const int jj = j & 255;

        const int gi0 = permutationMod8_[ii + permutation_[jj]];
        const int gi1 = permutationMod8_[ii + i1 + permutation_[jj + j1]];
        const int gi2 = permutationMod8_[ii + 1 + permutation_[jj + 1]];

        float n0 = 0.0f;
        float n1 = 0.0f;
        float n2 = 0.0f;

        float t0 = 0.5f - x0 * x0 - y0 * y0;
        if (t0 > 0.0f)
        {
            const float t0Sq = t0 * t0;
            const float t0Pow4 = t0Sq * t0Sq;
            n0 = t0Pow4 * glm::dot(kGradients[static_cast<std::size_t>(gi0)], glm::vec2(x0, y0));
        }

        float t1 = 0.5f - x1 * x1 - y1 * y1;
        if (t1 > 0.0f)
        {
            const float t1Sq = t1 * t1;
            const float t1Pow4 = t1Sq * t1Sq;
            n1 = t1Pow4 * glm::dot(kGradients[static_cast<std::size_t>(gi1)], glm::vec2(x1, y1));
        }

        float t2 = 0.5f - x2 * x2 - y2 * y2;
        if (t2 > 0.0f)
        {
            const float t2Sq = t2 * t2;
            const float t2Pow4 = t2Sq * t2Sq;
            n2 = t2Pow4 * glm::dot(kGradients[static_cast<std::size_t>(gi2)], glm::vec2(x2, y2));
        }

        return 70.0f * (n0 + n1 + n2);
    }

    float fbm(float x, float y, int octaves, float persistence, float lacunarity) const noexcept
    {
        float amplitude = 1.0f;
        float frequency = 1.0f;
        float sum = 0.0f;
        float maxValue = 0.0f;

        for (int i = 0; i < octaves; ++i)
        {
            sum += noise(x * frequency, y * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        if (maxValue > 0.0f)
        {
            sum /= maxValue;
        }

        return sum;
    }

    float ridge(float x, float y, int octaves, float lacunarity, float gain) const noexcept
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

    glm::vec2 sampleGradient(float x, float y) const noexcept
    {
        constexpr float F2 = 0.3660254037844386f;
        constexpr float G2 = 0.21132486540518713f;

        const float s = (x + y) * F2;
        const int i = static_cast<int>(std::floor(x + s));
        const int j = static_cast<int>(std::floor(y + s));
        const float t = static_cast<float>(i + j) * G2;
        const float X0 = static_cast<float>(i) - t;
        const float Y0 = static_cast<float>(j) - t;
        const float x0 = x - X0;
        const float y0 = y - Y0;

        const int i1 = x0 > y0 ? 1 : 0;
        const int j1 = x0 > y0 ? 0 : 1;

        const float x1 = x0 - static_cast<float>(i1) + G2;
        const float y1 = y0 - static_cast<float>(j1) + G2;
        const float x2 = x0 - 1.0f + 2.0f * G2;
        const float y2 = y0 - 1.0f + 2.0f * G2;

        const int ii = i & 255;
        const int jj = j & 255;

        const int gi0 = permutationMod8_[ii + permutation_[jj]];
        const int gi1 = permutationMod8_[ii + i1 + permutation_[jj + j1]];
        const int gi2 = permutationMod8_[ii + 1 + permutation_[jj + 1]];

        glm::vec2 gradient{0.0f, 0.0f};

        float t0 = 0.5f - x0 * x0 - y0 * y0;
        if (t0 > 0.0f)
        {
            const float t0Sq = t0 * t0;
            const float t0Pow3 = t0Sq * t0;
            const float t0Pow4 = t0Sq * t0Sq;
            const glm::vec2 grad = kGradients[static_cast<std::size_t>(gi0)];
            const float dot = grad.x * x0 + grad.y * y0;
            const float influence = -8.0f * t0Pow3 * dot;
            gradient.x += influence * x0 + t0Pow4 * grad.x;
            gradient.y += influence * y0 + t0Pow4 * grad.y;
        }

        float t1 = 0.5f - x1 * x1 - y1 * y1;
        if (t1 > 0.0f)
        {
            const float t1Sq = t1 * t1;
            const float t1Pow3 = t1Sq * t1;
            const float t1Pow4 = t1Sq * t1Sq;
            const glm::vec2 grad = kGradients[static_cast<std::size_t>(gi1)];
            const float dot = grad.x * x1 + grad.y * y1;
            const float influence = -8.0f * t1Pow3 * dot;
            gradient.x += influence * x1 + t1Pow4 * grad.x;
            gradient.y += influence * y1 + t1Pow4 * grad.y;
        }

        float t2 = 0.5f - x2 * x2 - y2 * y2;
        if (t2 > 0.0f)
        {
            const float t2Sq = t2 * t2;
            const float t2Pow3 = t2Sq * t2;
            const float t2Pow4 = t2Sq * t2Sq;
            const glm::vec2 grad = kGradients[static_cast<std::size_t>(gi2)];
            const float dot = grad.x * x2 + grad.y * y2;
            const float influence = -8.0f * t2Pow3 * dot;
            gradient.x += influence * x2 + t2Pow4 * grad.x;
            gradient.y += influence * y2 + t2Pow4 * grad.y;
        }

        return gradient * 70.0f;
    }

private:
    std::array<int, 512> permutation_{};
    std::array<int, 512> permutationMod8_{};

    static const std::array<glm::vec2, 8> kGradients;
};

const std::array<glm::vec2, 8> OpenSimplexNoise::kGradients = {
    glm::vec2(1.0f, 0.0f),
    glm::vec2(-1.0f, 0.0f),
    glm::vec2(0.0f, 1.0f),
    glm::vec2(0.0f, -1.0f),
    glm::vec2(0.70710678f, 0.70710678f),
    glm::vec2(-0.70710678f, 0.70710678f),
    glm::vec2(0.70710678f, -0.70710678f),
    glm::vec2(-0.70710678f, -0.70710678f)};

struct BiomeDefinition
{
    int minHeight{30};
    int maxHeight{820};
};

struct LittleMountainSample
{
    float height{0.0f};
    float entryFloor{0.0f};
    float interiorMask{0.0f};
};

struct LittleMountainSampler
{
    explicit LittleMountainSampler(unsigned seed)
        : littleMountainsNoise(seed ^ 0x9E3779B9u),
          littleMountainsWarpNoise(seed ^ 0x7F4A7C15u),
          littleMountainsOrientationNoise(seed ^ 0xDD62BBA1u)
    {
    }

    float computeNormalized(float worldX, float worldZ) const
    {
        const glm::vec2 worldPos{worldX, worldZ};
        const glm::vec2 kilometerField = worldPos * 0.001f;

        glm::vec2 orientationWarp{
            littleMountainsWarpNoise.fbm(worldPos.x * 0.00035f + 103.0f,
                                         worldPos.y * 0.00035f - 77.0f,
                                         3,
                                         0.55f,
                                         2.15f),
            littleMountainsWarpNoise.fbm(worldPos.x * 0.00035f - 59.0f,
                                         worldPos.y * 0.00035f + 43.0f,
                                         3,
                                         0.55f,
                                         2.15f)};
        orientationWarp *= 0.35f;
        const glm::vec2 orientationSample = kilometerField + orientationWarp;

        glm::vec2 orientationGradient =
            littleMountainsOrientationNoise.sampleGradient(orientationSample.x, orientationSample.y);
        if (!std::isfinite(orientationGradient.x) || !std::isfinite(orientationGradient.y)
            || glm::dot(orientationGradient, orientationGradient) < 1e-6f)
        {
            orientationGradient = glm::vec2(1.0f, 0.0f);
        }
        glm::vec2 ridgeDirection = glm::normalize(orientationGradient);
        const glm::vec2 ridgePerpendicular{-ridgeDirection.y, ridgeDirection.x};

        glm::vec2 warpPrimary{
            littleMountainsWarpNoise.fbm(worldPos.x * 0.0011f + 19.0f,
                                         worldPos.y * 0.0011f + 87.0f,
                                         4,
                                         0.6f,
                                         2.15f),
            littleMountainsWarpNoise.fbm(worldPos.x * 0.0011f - 71.0f,
                                         worldPos.y * 0.0011f - 29.0f,
                                         4,
                                         0.6f,
                                         2.15f)};
        glm::vec2 warpDetail{
            littleMountainsWarpNoise.fbm(worldPos.x * 0.0045f - 11.0f,
                                         worldPos.y * 0.0045f + 53.0f,
                                         3,
                                         0.5f,
                                         2.3f),
            littleMountainsWarpNoise.fbm(worldPos.x * 0.0045f + 67.0f,
                                         worldPos.y * 0.0045f - 41.0f,
                                         3,
                                         0.5f,
                                         2.3f)};
        glm::vec2 warped = worldPos + warpPrimary * 180.0f + warpDetail * 28.0f;

        const float alongRidge = glm::dot(warped, ridgeDirection);
        const float acrossRidge = glm::dot(warped, ridgePerpendicular);

        const float ridgePrimary = littleMountainsNoise.ridge(alongRidge * 0.016f,
                                                              acrossRidge * 0.016f,
                                                              5,
                                                              2.05f,
                                                              0.55f);
        const float ridgeSecondary = littleMountainsNoise.ridge(alongRidge * 0.028f + 57.0f,
                                                                acrossRidge * 0.028f - 113.0f,
                                                                4,
                                                                2.2f,
                                                                0.6f);
        const float ridgeMicro = littleMountainsNoise.ridge(alongRidge * 0.043f - 211.0f,
                                                            acrossRidge * 0.021f + 167.0f,
                                                            3,
                                                            2.1f,
                                                            0.5f);

        float ridgeStack = ridgePrimary * 0.6f + ridgeSecondary * 0.3f + ridgeMicro * 0.25f;
        ridgeStack = std::clamp(ridgeStack, 0.0f, 1.0f);

        const float valleyFill = littleMountainsNoise.fbm(warped.x * 0.0032f - 401.0f,
                                                          warped.y * 0.0032f + 245.0f,
                                                          4,
                                                          0.5f,
                                                          2.1f)
                                 * 0.5f
                                 + 0.5f;

        const float macroRamps = littleMountainsNoise.fbm(worldPos.x * 0.00042f + 11.0f,
                                                          worldPos.y * 0.00042f - 37.0f,
                                                          5,
                                                          0.55f,
                                                          2.05f)
                                 * 0.5f
                                 + 0.5f;

        const float uplift = littleMountainsNoise.fbm(worldPos.x * 0.00078f - 91.0f,
                                                      worldPos.y * 0.00078f + 133.0f,
                                                      4,
                                                      0.6f,
                                                      2.15f)
                               * 0.5f
                               + 0.5f;

        const float ridgeMask = glm::smoothstep(0.3f, 0.72f, macroRamps);
        const float valleyMask = 1.0f - glm::smoothstep(0.1f, 0.4f, macroRamps);

        float ridged = glm::mix(valleyFill, ridgeStack, ridgeMask);
        ridged = std::clamp(ridged, 0.0f, 1.0f);

        const float terraces = littleMountainsNoise.fbm(warped.x * 0.008f + 211.0f,
                                                        warped.y * 0.008f - 157.0f,
                                                        3,
                                                        0.5f,
                                                        2.3f)
                              * 0.5f
                              + 0.5f;

        float combined = macroRamps * 0.35f + ridged * 0.45f + uplift * 0.15f + terraces * 0.05f;
        combined = std::clamp(combined, 0.0f, 1.0f);

        const float peakBlendControl = glm::smoothstep(0.55f, 0.9f, combined);
        const float peakShaped = 1.0f - std::pow(1.0f - combined, 3.0f);
        float finalValue = glm::mix(combined, peakShaped, peakBlendControl);
        finalValue = std::clamp(finalValue, 0.0f, 1.0f);

        const float valleyBlend = glm::smoothstep(0.2f, 0.6f, valleyFill);
        const float valleyBase = macroRamps * 0.5f + valleyFill * 0.5f;
        finalValue = glm::mix(valleyBase, finalValue, valleyBlend);

        finalValue = std::clamp(finalValue + (uplift - 0.5f) * 0.08f, 0.0f, 1.0f);
        finalValue = glm::mix(finalValue, macroRamps, valleyMask * 0.25f);

        return std::clamp(finalValue, 0.0f, 1.0f);
    }

    LittleMountainSample computeHeight(int worldX,
                                       int worldZ,
                                       const BiomeDefinition& definition,
                                       float interiorMask,
                                       bool hasBorderAnchor,
                                       float borderAnchorHeight) const
    {
        const float minHeight = static_cast<float>(definition.minHeight);
        const float maxHeight = static_cast<float>(definition.maxHeight);
        const float range = std::max(maxHeight - minHeight, 1.0f);
        const float floorRange = std::clamp(range * 0.12f, 24.0f, 110.0f);

        auto sampleColumn = [&](float sampleX, float sampleZ) -> LittleMountainSample {
            const float normalized = computeNormalized(sampleX, sampleZ);
            float height = minHeight + normalized * range;
            const float floorNoise = littleMountainsNoise.fbm(sampleX * 0.0013f + 311.0f,
                                                              sampleZ * 0.0013f - 173.0f,
                                                              4,
                                                              0.5f,
                                                              2.0f);
            const float floorT = std::clamp(floorNoise * 0.5f + 0.5f, 0.0f, 1.0f);
            const float entryFloor = minHeight + floorT * floorRange;
            if (height < entryFloor)
            {
                height = entryFloor;
            }
            return LittleMountainSample{height, entryFloor, 1.0f};
        };

        const auto baseSample = sampleColumn(static_cast<float>(worldX), static_cast<float>(worldZ));
        float baseHeight = baseSample.height;
        const float entryFloor = baseSample.entryFloor;

        const float sampleStep = 12.0f;
        const float highSlopeStart = minHeight + range * 0.65f;
        const float highSlopeEnd = minHeight + range * 0.90f;
        const float altitudeT = std::clamp((baseHeight - highSlopeStart) / (highSlopeEnd - highSlopeStart), 0.0f, 1.0f);
        const float normalizedAltitude = std::clamp((baseHeight - minHeight) / range, 0.0f, 1.0f);
        const float lowTalusDeg = 1.5f;
        const float midTalusDeg = 4.0f;
        const float highTalusDeg = 7.0f;
        const float foothillBlend = glm::smoothstep(0.25f, 0.65f, normalizedAltitude);
        float talusDeg = std::lerp(lowTalusDeg, midTalusDeg, foothillBlend);
        talusDeg = std::lerp(talusDeg, highTalusDeg, glm::smoothstep(0.0f, 1.0f, altitudeT));
        const float talusAngle = glm::radians(talusDeg);
        const float rawMaxDiff = std::tan(talusAngle) * sampleStep;
        const float maxTalusDiff = std::tan(glm::radians(highTalusDeg)) * sampleStep;
        const float maxDiff = std::clamp(rawMaxDiff, 0.2f, maxTalusDiff);

        auto sampleNeighbor = [&](float offsetX, float offsetZ) {
            const auto neighborSample =
                sampleColumn(static_cast<float>(worldX) + offsetX, static_cast<float>(worldZ) + offsetZ);
            return neighborSample.height;
        };

        std::array<float, 4> neighbors{
            sampleNeighbor(sampleStep, 0.0f),
            sampleNeighbor(-sampleStep, 0.0f),
            sampleNeighbor(0.0f, sampleStep),
            sampleNeighbor(0.0f, -sampleStep),
        };

        std::array<float, 4> diagonalNeighbors{
            sampleNeighbor(sampleStep, sampleStep),
            sampleNeighbor(sampleStep, -sampleStep),
            sampleNeighbor(-sampleStep, sampleStep),
            sampleNeighbor(-sampleStep, -sampleStep),
        };

        const float neighborAverage =
            (neighbors[0] + neighbors[1] + neighbors[2] + neighbors[3]) * 0.25f;
        const float diagonalAverage =
            (diagonalNeighbors[0] + diagonalNeighbors[1] + diagonalNeighbors[2] + diagonalNeighbors[3]) * 0.25f;
        const float convexity = std::max(baseHeight - neighborAverage, 0.0f);
        const float diagonalConvexity = std::max(baseHeight - diagonalAverage, 0.0f);
        const float curvatureMagnitude = std::max(convexity, diagonalConvexity);
        const float lowSlopeMask = 1.0f - glm::smoothstep(0.25f, 0.6f, normalizedAltitude);
        const float curvatureSuppression = lowSlopeMask * glm::smoothstep(1.5f, 10.0f, curvatureMagnitude) * 0.55f;
        const float curvatureFactor = std::clamp(1.0f - curvatureSuppression, 0.45f, 1.0f);
        const float adjustedMaxDiff = maxDiff * curvatureFactor;

        float relaxedHeight = baseHeight;
        auto relaxWithNeighbors = [&](const std::array<float, 4>& neighborHeights, float allowedDiff) {
            for (float neighborHeight : neighborHeights)
            {
                const float diff = relaxedHeight - neighborHeight;
                if (diff > allowedDiff)
                {
                    relaxedHeight -= (diff - allowedDiff) * 0.5f;
                }
                else if (diff < -allowedDiff)
                {
                    relaxedHeight += (-allowedDiff - diff) * 0.5f;
                }
            }
        };

        relaxWithNeighbors(neighbors, adjustedMaxDiff);

        const float diagonalStep = sampleStep * std::sqrt(2.0f);
        const float diagonalStepFactor = diagonalStep / sampleStep;
        const float diagonalRawDiff = adjustedMaxDiff * diagonalStepFactor;
        const float maxDiagonalDiff = maxDiff * diagonalStepFactor;
        const float diagonalDiff = std::clamp(diagonalRawDiff, 0.25f, maxDiagonalDiff);
        relaxWithNeighbors(diagonalNeighbors, diagonalDiff);

        relaxedHeight = std::clamp(relaxedHeight, entryFloor, maxHeight);

        baseHeight = std::lerp(baseHeight, relaxedHeight, 0.9f);

        const float clampedHeight = std::clamp(baseHeight, entryFloor, maxHeight);

        const float maskedInterior = std::clamp(interiorMask, 0.0f, 1.0f);

        const float relaxedEntryFloor = std::clamp(entryFloor, minHeight, clampedHeight);
        float borderBaseline = relaxedEntryFloor;
        if (hasBorderAnchor)
        {
            const float anchorMin = relaxedEntryFloor - floorRange;
            const float anchorMax = relaxedEntryFloor + floorRange;
            borderBaseline = std::clamp(borderAnchorHeight, anchorMin, anchorMax);
            borderBaseline = std::clamp(borderBaseline, minHeight, maxHeight);
        }

        const float interiorBlend = maskedInterior * maskedInterior;
        const float baselineEntryFloor = glm::mix(borderBaseline, relaxedEntryFloor, interiorBlend);
        const float baselineMinHeight = glm::mix(borderBaseline, minHeight, interiorBlend);

        const float interiorFoothillLift = maskedInterior * maskedInterior * floorRange * 0.65f;
        const float raisedEntryFloor = std::min(baselineEntryFloor + interiorFoothillLift, clampedHeight);
        const float maskedEntryFloor = glm::mix(baselineMinHeight, raisedEntryFloor, maskedInterior);
        float maskedHeight = glm::mix(baselineMinHeight, clampedHeight, maskedInterior);
        maskedHeight = std::max(maskedHeight, maskedEntryFloor);

        return LittleMountainSample{maskedHeight, maskedEntryFloor, maskedInterior};
    }

    OpenSimplexNoise littleMountainsNoise;
    OpenSimplexNoise littleMountainsWarpNoise;
    OpenSimplexNoise littleMountainsOrientationNoise;
};

constexpr int kChunkSize = 16;
constexpr int kBiomeSizeInChunks = 30;
constexpr float kLittleMountainsFootprint = 2.4f;
constexpr float kMarginRatio = 0.2f;
constexpr float kLittleMountainsInfluenceCutoff = 0.95f;

struct Biome
{
    const char* name;
    float footprintMultiplier;
};

const std::array<Biome, 5> kBiomes{{
    {"Grasslands", 1.0f},
    {"Forest", 1.0f},
    {"Desert", 1.0f},
    {"Little Mountains", kLittleMountainsFootprint},
    {"Ocean", 1.0f},
}};

float hashToUnitFloat(int x, int y, int z)
{
    std::uint32_t h = static_cast<std::uint32_t>(x * 374761393 + y * 668265263 + z * 2147483647);
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
}

int biomeIndexForRegion(int regionX, int regionZ)
{
    const float selector = hashToUnitFloat(regionX, 31, regionZ);
    const int count = static_cast<int>(kBiomes.size());
    const int index = static_cast<int>(selector * count);
    return std::min(index, count - 1);
}

struct BiomeSite
{
    int biomeIndex{0};
    float centerX{0.0f};
    float centerZ{0.0f};
    float halfExtentX{0.0f};
    float halfExtentZ{0.0f};

    float radius() const
    {
        return std::min(halfExtentX, halfExtentZ) * kLittleMountainsInfluenceCutoff;
    }
};

BiomeSite computeSite(int regionX, int regionZ, const Biome& biome)
{
    const float scaledBiomeSize = kBiomeSizeInChunks * biome.footprintMultiplier;
    const float regionWidth = kChunkSize * scaledBiomeSize;
    const float regionDepth = kChunkSize * scaledBiomeSize;
    const float marginX = regionWidth * kMarginRatio;
    const float marginZ = regionDepth * kMarginRatio;
    const float jitterX = hashToUnitFloat(regionX, 137, regionZ);
    const float jitterZ = hashToUnitFloat(regionX, 613, regionZ);
    const float availableWidth = std::max(regionWidth - 2.0f * marginX, 0.0f);
    const float availableDepth = std::max(regionDepth - 2.0f * marginZ, 0.0f);
    const float baseX = regionX * regionWidth;
    const float baseZ = regionZ * regionDepth;

    BiomeSite site;
    site.biomeIndex = biomeIndexForRegion(regionX, regionZ);
    site.centerX = baseX + marginX + availableWidth * jitterX;
    site.centerZ = baseZ + marginZ + availableDepth * jitterZ;
    site.halfExtentX = regionWidth * 0.5f;
    site.halfExtentZ = regionDepth * 0.5f;
    return site;
}

struct LittleMountainColumnResult
{
    int worldX{0};
    int worldZ{0};
    float normalizedDistance{0.0f};
    float interiorMask{0.0f};
    LittleMountainSample sample{};
};

float littleMountainInfluence(float normalizedDistance)
{
    const float clamped = std::clamp(normalizedDistance, 0.0f, 1.0f);
    return 1.0f - glm::smoothstep(0.55f, 0.95f, clamped);
}

} // namespace

int main()
{
    constexpr unsigned kSeed = 1337u;
    constexpr int kSearchRadius = 12;

    std::pair<int, int> bestColumn{0, 0};
    float bestDistance = std::numeric_limits<float>::infinity();
    BiomeSite bestSite{};
    bool found = false;

    for (int regionZ = -kSearchRadius; regionZ <= kSearchRadius; ++regionZ)
    {
        for (int regionX = -kSearchRadius; regionX <= kSearchRadius; ++regionX)
        {
            const int biomeIndex = biomeIndexForRegion(regionX, regionZ);
            if (biomeIndex != 3)
            {
                continue;
            }

            const Biome& biome = kBiomes[static_cast<std::size_t>(biomeIndex)];
            const BiomeSite site = computeSite(regionX, regionZ, biome);

            const float centerDistance = std::hypot(site.centerX, site.centerZ);
            const float influenceRadius = site.radius();
            float candidateCenterX = 0.0f;
            float candidateCenterZ = 0.0f;
            if (centerDistance > influenceRadius)
            {
                const float scale = 1.0f - influenceRadius / centerDistance;
                candidateCenterX = site.centerX * scale;
                candidateCenterZ = site.centerZ * scale;
            }

            const int baseX = static_cast<int>(std::round(candidateCenterX - 0.5f));
            const int baseZ = static_cast<int>(std::round(candidateCenterZ - 0.5f));

            for (int offsetX = -3; offsetX <= 3; ++offsetX)
            {
                for (int offsetZ = -3; offsetZ <= 3; ++offsetZ)
                {
                    const int columnX = baseX + offsetX;
                    const int columnZ = baseZ + offsetZ;
                    const float dx = static_cast<float>(columnX) + 0.5f - site.centerX;
                    const float dz = static_cast<float>(columnZ) + 0.5f - site.centerZ;
                    const float normalizedDistance =
                        std::sqrt((dx / site.halfExtentX) * (dx / site.halfExtentX)
                                  + (dz / site.halfExtentZ) * (dz / site.halfExtentZ));
                    if (normalizedDistance >= kLittleMountainsInfluenceCutoff)
                    {
                        continue;
                    }

                    const float planarDistance = std::hypot(static_cast<float>(columnX), static_cast<float>(columnZ));
                    if (planarDistance < bestDistance)
                    {
                        bestDistance = planarDistance;
                        bestColumn = {columnX, columnZ};
                        bestSite = site;
                        found = true;
                    }
                }
            }
        }
    }

    if (!found)
    {
        std::cerr << "Failed to locate Little Mountains site" << std::endl;
        return EXIT_FAILURE;
    }

    LittleMountainSampler sampler(kSeed);
    BiomeDefinition definition{};

    // Sample columns moving from the biome edge toward the core along the X axis.
    const float baseZ = bestSite.centerZ;
    const std::vector<float> normalizedTargets{1.05f, 0.95f, 0.85f, 0.75f, 0.65f, 0.55f, 0.45f, 0.35f, 0.25f, 0.15f, 0.05f};

    std::vector<LittleMountainColumnResult> results;
    results.reserve(normalizedTargets.size());

    for (float normalized : normalizedTargets)
    {
        const float sampleCenterX = bestSite.centerX + normalized * bestSite.halfExtentX;
        const int worldX = static_cast<int>(std::floor(sampleCenterX));
        const int worldZ = static_cast<int>(std::floor(baseZ));
        const float columnCenterX = static_cast<float>(worldX) + 0.5f;
        const float columnCenterZ = static_cast<float>(worldZ) + 0.5f;
        const float dx = columnCenterX - bestSite.centerX;
        const float dz = columnCenterZ - bestSite.centerZ;
        const float normalizedDistance =
            std::sqrt((dx / bestSite.halfExtentX) * (dx / bestSite.halfExtentX)
                      + (dz / bestSite.halfExtentZ) * (dz / bestSite.halfExtentZ));
        const float interiorMask = littleMountainInfluence(normalizedDistance);

        LittleMountainColumnResult result;
        result.worldX = worldX;
        result.worldZ = worldZ;
        result.normalizedDistance = normalizedDistance;
        result.interiorMask = interiorMask;
        result.sample = sampler.computeHeight(worldX, worldZ, definition, interiorMask, false, 0.0f);
        results.push_back(result);
    }

    std::ofstream out("tools/little_mountains_columns.csv", std::ios::trunc);
    out << "world_x,world_z,normalized_distance,interior_mask,height,entry_floor\n";
    out << std::fixed << std::setprecision(6);

    for (const auto& result : results)
    {
        out << result.worldX << ',' << result.worldZ << ',' << result.normalizedDistance << ',' << result.interiorMask << ','
            << result.sample.height << ',' << result.sample.entryFloor << '\n';
    }

    std::cout << "Wrote " << results.size() << " samples to tools/little_mountains_columns.csv" << std::endl;
    return 0;
}

