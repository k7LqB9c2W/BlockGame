#include "base_game_sky_common.hlsli"

struct DecodedVertexLighting
{
    float sky;
    float block;
    float ao;
    float aoDebug;
    uint flags;
    float farVoxelScale;
};

static const uint kMaterialFlagWater = 0x01u;
static const uint kMaterialFlagFarLod = 0x02u;
static const uint kMaterialFlagGrassTintShift = 2u;
static const uint kMaterialFlagGrassTintMask = 0x1Cu;
static const uint kMaterialFlagGrassSideTint = 0x20u;

struct FogBlendResult
{
    float transmittance;
    float inscatter;
};

static const float kVanillaLightLut[16] = {
    0.000f, 0.018f, 0.025f, 0.035f,
    0.048f, 0.065f, 0.088f, 0.118f,
    0.159f, 0.214f, 0.288f, 0.387f,
    0.520f, 0.690f, 0.845f, 1.000f
};

// Lean a bit more into corner darkening so terrain AO reads clearly without
// returning to the old overly harsh fast-lighting look.
static const float kAoFactors[4] = {1.00f, 0.86f, 0.73f, 0.60f};

float decodeNonLinearLightLevel(uint level)
{
    return kVanillaLightLut[min(level, 15u)];
}

float decodeAoFactor(uint aoLevel)
{
    return kAoFactors[min(aoLevel, 3u)];
}

DecodedVertexLighting decodeVertexLighting(uint packedLighting)
{
    DecodedVertexLighting decoded;
    const uint packedLight = packedLighting & 0xFFu;
    const uint skyLevel = (packedLight >> 4) & 0xFu;
    const uint blockLevel = packedLight & 0xFu;
    const uint aoLevel = (packedLighting >> 8) & 0x3u;
    decoded.sky = decodeNonLinearLightLevel(skyLevel);
    decoded.block = decodeNonLinearLightLevel(blockLevel);
    decoded.ao = decodeAoFactor(aoLevel);
    decoded.aoDebug = decoded.ao;
    decoded.flags = (packedLighting >> 10) & 0x3Fu;
    const uint scale = (packedLighting >> 16) & 0xFFu;
    decoded.farVoxelScale = (scale > 0u) ? (float)scale : 1.0f;
    return decoded;
}

float faceShadeMultiplier(float3 normal)
{
    const float3 absNormal = abs(normal);
    if (absNormal.y >= absNormal.x && absNormal.y >= absNormal.z)
    {
        return normal.y >= 0.0f ? 1.00f : 0.82f;
    }
    if (absNormal.z >= absNormal.x)
    {
        return 0.94f;
    }
    return 0.90f;
}

uint decodeGrassTintIndex(uint materialFlags)
{
    return (materialFlags & kMaterialFlagGrassTintMask) >> kMaterialFlagGrassTintShift;
}

bool isGrassSideTint(uint materialFlags)
{
    return (materialFlags & kMaterialFlagGrassSideTint) != 0u;
}

float3 biomeGrassTint(uint tintIndex)
{
    if (tintIndex == 2u)
    {
        return float3(80.0f, 122.0f, 50.0f) / 255.0f;
    }
    if (tintIndex == 3u)
    {
        return float3(134.0f, 183.0f, 131.0f) / 255.0f;
    }
    if (tintIndex == 4u)
    {
        return float3(191.0f, 183.0f, 85.0f) / 255.0f;
    }
    return float3(121.0f, 192.0f, 90.0f) / 255.0f;
}

float grassSideTintMask(float2 wrappedTileUv)
{
    return smoothstep(0.70f, 0.84f, wrappedTileUv.y);
}

float3 computeTerrainFogColor(float3 viewDir, float3 topSkyColorSrgb, float3 horizonSkyColorSrgb)
{
    return computeSkyGradientFromViewY(max(viewDir.y, 0.0f), topSkyColorSrgb, horizonSkyColorSrgb);
}

FogBlendResult computeLayeredFog(float distanceBlocks,
                                 float horizontalDistanceBlocks,
                                 float3 viewDir,
                                 float worldY,
                                 float cameraY,
                                 float fogStartBlocks,
                                 float farDistanceBlocks,
                                 float concealStrength,
                                 float maxFog)
{
    FogBlendResult result;
    result.transmittance = 1.0f;
    result.inscatter = 0.0f;

    if (farDistanceBlocks <= 1.0f)
    {
        return result;
    }

    const float safeFogStart = min(max(fogStartBlocks, 0.0f), farDistanceBlocks - 1.0f);
    const float horizon = pow(saturate(1.0f - abs(viewDir.y)), 1.10f);
    const float lowerTerrain = saturate((cameraY - worldY + 24.0f) / max(farDistanceBlocks * 0.20f, 56.0f));
    const float radialDistanceBlocks = max(horizontalDistanceBlocks, distanceBlocks * 0.35f);

    const float hazeStart = min(safeFogStart * 0.34f, farDistanceBlocks * 0.42f);
    const float hazeEnd = min(max(hazeStart + 1.0f, safeFogStart * 0.98f), farDistanceBlocks - 1.0f);
    const float hazeRange = max(hazeEnd - hazeStart, 1.0f);
    const float haze01 = saturate((radialDistanceBlocks - hazeStart) / hazeRange);
    const float hazeCurve = 1.0f - exp(-pow(haze01, 1.12f) * 1.30f);
    const float haze = hazeCurve * (0.10f + horizon * 0.04f + lowerTerrain * 0.05f);

    const float concealStart = min(max(safeFogStart * 0.96f, farDistanceBlocks * 0.74f), farDistanceBlocks - 1.0f);
    const float concealRange = max(farDistanceBlocks - concealStart, 1.0f);
    const float conceal01 = saturate((radialDistanceBlocks - concealStart) / concealRange);
    const float concealCurve = conceal01 * conceal01 * (3.0f - 2.0f * conceal01);
    const float conceal = concealCurve * (0.88f + horizon * 0.06f + lowerTerrain * 0.08f) * concealStrength;

    const float fog = min(saturate(haze + conceal), maxFog);
    result.transmittance = exp(-fog * 2.35f);
    result.inscatter = saturate(1.0f - result.transmittance);
    return result;
}
