struct DecodedVertexLighting
{
    float sky;
    float block;
    float ao;
    float aoDebug;
};

struct FogBlendResult
{
    float transmittance;
    float inscatter;
};

static const float3 kBaseGameSkyColor = float3(120.0f / 255.0f, 167.0f / 255.0f, 255.0f / 255.0f);
static const float3 kBaseGameHorizonColor = float3(187.0f / 255.0f, 212.0f / 255.0f, 255.0f / 255.0f);

static const float kVanillaLightLut[16] = {
    0.000f, 0.018f, 0.025f, 0.035f,
    0.048f, 0.065f, 0.088f, 0.118f,
    0.159f, 0.214f, 0.288f, 0.387f,
    0.520f, 0.690f, 0.845f, 1.000f
};

static const float kAoFactors[4] = {1.00f, 0.80f, 0.62f, 0.45f};

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
    return decoded;
}

float faceShadeMultiplier(float3 normal)
{
    const float3 absNormal = abs(normal);
    if (absNormal.y >= absNormal.x && absNormal.y >= absNormal.z)
    {
        return normal.y >= 0.0f ? 1.00f : 0.50f;
    }
    if (absNormal.z >= absNormal.x)
    {
        return 0.80f;
    }
    return 0.60f;
}

float3 computeTerrainFogColor(float3 viewDir,
                              float3 lightDir,
                              float3 sunColor,
                              float3 skyAmbient,
                              float3 groundAmbient)
{
    const float horizon = pow(saturate(1.0f - abs(viewDir.y) * 2.2f), 0.60f);
    const float upness = saturate(viewDir.y * 0.5f + 0.5f);
    const float sunFacing = pow(saturate(dot(normalize(float3(viewDir.x, max(viewDir.y, -0.12f), viewDir.z)), lightDir)), 5.0f);
    const float3 zenithTint = kBaseGameSkyColor * 0.98f + skyAmbient * 0.92f;
    const float3 horizonBase = kBaseGameHorizonColor * 0.92f + skyAmbient * 0.55f + groundAmbient * 0.06f;
    const float3 horizonSunTint = horizonBase + sunColor * 0.035f;
    const float3 horizonTint = lerp(horizonBase, horizonSunTint, sunFacing);
    const float3 lowerTint = lerp(kBaseGameHorizonColor * 0.96f, horizonTint * 0.995f, horizon);
    return lerp(lowerTint, lerp(horizonTint, zenithTint, upness), saturate(viewDir.y * 1.35f + 0.52f));
}

float computeFarLodHaze(float distanceBlocks, float fogStartBlocks, float farDistanceBlocks)
{
    const float hazeStart = fogStartBlocks * 0.72f;
    const float hazeRange = max(farDistanceBlocks - hazeStart, 1.0f);
    const float haze = saturate((distanceBlocks - hazeStart) / hazeRange);
    return haze * haze * (3.0f - 2.0f * haze);
}

FogBlendResult computeRoundedFog(float distanceBlocks,
                                 float3 viewDir,
                                 float worldY,
                                 float cameraY,
                                 float fogStartBlocks,
                                 float farDistanceBlocks,
                                 float horizonBoostScale,
                                 float maxFog)
{
    FogBlendResult result;
    result.transmittance = 1.0f;
    result.inscatter = 0.0f;

    if (farDistanceBlocks <= 1.0f)
    {
        return result;
    }

    const float clampedFogStart = min(fogStartBlocks, farDistanceBlocks - 1.0f);
    const float fogSpan = max(farDistanceBlocks - clampedFogStart, 1.0f);
    const float distance01 = saturate((distanceBlocks - clampedFogStart) / fogSpan);
    const float curvedDistance = 1.0f - exp(-pow(distance01, 1.35f) * 4.0f);
    const float horizon = pow(saturate(1.0f - abs(viewDir.y)), 1.65f);
    const float horizonBoost = 1.0f + horizon * horizonBoostScale;
    const float lowerTerrain = saturate((cameraY - worldY + 20.0f) / max(farDistanceBlocks * 0.18f, 48.0f));
    const float altitudeBoost = lerp(0.94f, 1.18f, lowerTerrain);
    const float fog = min(saturate(curvedDistance * horizonBoost * altitudeBoost), maxFog);

    result.transmittance = saturate(1.0f - fog * 0.88f);
    result.inscatter = fog;
    return result;
}
