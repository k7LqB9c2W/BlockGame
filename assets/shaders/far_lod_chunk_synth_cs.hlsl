cbuffer ChunkParams : register(b0)
{
    int3 gWorldMin;
    int gBlockScale;
    int gSeaLevel;
};

struct FarLodGpuWorldgenHeader
{
    int seaLevel;
    uint seed;
    uint biomeCount;
    float warpFrequency;
    float warpAmplitude;
    float mainFrequency;
    uint mainOctaves;
    float mainGain;
    float mainLacunarity;
    float mediumFrequency;
    uint mediumOctaves;
    float mediumGain;
    float mediumLacunarity;
    float detailFrequency;
    uint detailOctaves;
    float detailGain;
    float detailLacunarity;
    float mountainFrequency;
    uint mountainOctaves;
    float mountainGain;
    float mountainLacunarity;
};

struct FarLodGpuBiome
{
    uint surfaceBlock;
    uint fillerBlock;
    uint flags;
    uint coastProfile;
    uint propertyBits;
    int waterMaxDepth;
    float spawnChance;
    float minHeight;
    float maxHeight;
    float heightOffset;
    float heightScale;
    float roughness;
    float hills;
    float mountains;
    float keepOriginalTerrain;
    float interpolationWeight;
    float baseSlopeBias;
    float maxGradient;
    float footprintMultiplier;
};

struct GpuTerrainColumnDescriptor
{
    uint centerHasSolid;
    uint centerWaterEnabled;
    int centerSurfaceY;
    int centerWaterBottomY;
    uint centerSurfaceBlock;
    uint centerFillerBlock;
    int minSurfaceY;
    int maxSurfaceY;
    uint waterHitCount;
    int minWaterBottomY;
};

StructuredBuffer<FarLodGpuWorldgenHeader> gWorldgenHeader : register(t0);
StructuredBuffer<FarLodGpuBiome> gBiomes : register(t1);
RWStructuredBuffer<GpuTerrainColumnDescriptor> gColumnBuffer : register(u0);

static const uint kLogicalSize = 16u;
static const uint kFarLodBiomeOcean = 1u << 0;
static const uint kFarLodBiomeSmoothBeaches = 1u << 1;
static const uint kFarLodBiomeWaterFill = 1u << 2;
static const uint kFarLodBiomeTaiga = 1u << 3;
static const uint kPropHot = 1u << 0;
static const uint kPropTemperate = 1u << 1;
static const uint kPropCold = 1u << 2;
static const uint kPropInland = 1u << 3;
static const uint kPropLand = 1u << 4;
static const uint kPropOcean = 1u << 5;
static const uint kPropWet = 1u << 6;
static const uint kPropNeutralHydration = 1u << 7;
static const uint kPropDry = 1u << 8;
static const uint kPropBarren = 1u << 9;
static const uint kPropBalanced = 1u << 10;
static const uint kPropOvergrown = 1u << 11;
static const uint kPropMountain = 1u << 12;
static const uint kPropLowTerrain = 1u << 13;
static const uint kPropAntiMountain = 1u << 14;
static const float kOceanThreshold = -0.08f;
static const float kCoastDistanceScale = 72.0f;
static const float kTemperatureScale = 0.11f;
static const float kMoistureScale = 0.09f;
static const float kFertilityScale = 0.05f;
static const float kContinentalScale = 0.065f;

uint columnIndex(uint localX, uint localZ)
{
    return localZ * kLogicalSize + localX;
}

float hashToUnitFloat(int x, int y, int z)
{
    uint h = (uint)x;
    h ^= (uint)y * 374761393u;
    h ^= (uint)z * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return (float)(h & 0xFFFFFFu) / 16777215.0f;
}

float smoothStep(float t)
{
    t = saturate(t);
    return t * t * (3.0f - 2.0f * t);
}

float valueNoise2D(float x, float z, float frequency, int seed)
{
    const float sampleX = x * frequency;
    const float sampleZ = z * frequency;
    const int x0 = (int)floor(sampleX);
    const int z0 = (int)floor(sampleZ);
    const int x1 = x0 + 1;
    const int z1 = z0 + 1;

    const float tx = smoothStep(sampleX - (float)x0);
    const float tz = smoothStep(sampleZ - (float)z0);

    const float v00 = hashToUnitFloat(x0 + seed * 17, seed * 31, z0 - seed * 13);
    const float v10 = hashToUnitFloat(x1 + seed * 17, seed * 31, z0 - seed * 13);
    const float v01 = hashToUnitFloat(x0 + seed * 17, seed * 31, z1 - seed * 13);
    const float v11 = hashToUnitFloat(x1 + seed * 17, seed * 31, z1 - seed * 13);

    const float ix0 = lerp(v00, v10, tx);
    const float ix1 = lerp(v01, v11, tx);
    return lerp(ix0, ix1, tz);
}

float taigaPodzolNoise(int worldX, int worldZ)
{
    const float broad = valueNoise2D((float)worldX, (float)worldZ, 1.0f / 16.0f, 19);
    const float medium = valueNoise2D((float)worldX, (float)worldZ, 1.0f / 8.0f, 37);
    const float detail = valueNoise2D((float)worldX, (float)worldZ, 1.0f / 4.0f, 73);
    return broad * 0.55f + medium * 0.30f + detail * 0.15f;
}

float fade(float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float grad(uint hash, float x, float y)
{
    switch (hash & 7u)
    {
    case 0u: return x + y;
    case 1u: return -x + y;
    case 2u: return x - y;
    case 3u: return -x - y;
    case 4u: return x;
    case 5u: return -x;
    case 6u: return y;
    default: return -y;
    }
}

uint latticeHash(int x, int y, uint seed)
{
    uint h = (uint)x;
    h ^= (uint)y * 374761393u;
    h ^= seed * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return h;
}

float perlin2(float x, float y, uint seed)
{
    const int x0 = (int)floor(x);
    const int y0 = (int)floor(y);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float fx = x - (float)x0;
    const float fy = y - (float)y0;
    const float u = fade(fx);
    const float v = fade(fy);

    const float a = grad(latticeHash(x0, y0, seed), fx, fy);
    const float b = grad(latticeHash(x1, y0, seed), fx - 1.0f, fy);
    const float c = grad(latticeHash(x0, y1, seed), fx, fy - 1.0f);
    const float d = grad(latticeHash(x1, y1, seed), fx - 1.0f, fy - 1.0f);

    const float ix0 = lerp(a, b, u);
    const float ix1 = lerp(c, d, u);
    return lerp(ix0, ix1, v);
}

float fbm2(float x, float y, float frequency, uint octaves, float gain, float lacunarity, uint seed, float frequencyScale)
{
    float value = 0.0f;
    float amplitude = 1.0f;
    float sampleFrequency = frequency * frequencyScale;
    float amplitudeSum = 0.0f;
    [loop]
    for (uint octave = 0u; octave < octaves; ++octave)
    {
        value += perlin2(x * sampleFrequency, y * sampleFrequency, seed + octave * 17u) * amplitude;
        amplitudeSum += amplitude;
        amplitude *= gain;
        sampleFrequency *= lacunarity;
    }
    return amplitudeSum > 0.0f ? value / amplitudeSum : 0.0f;
}

float ridge2(float x, float y, float frequency, uint octaves, float gain, float lacunarity, uint seed)
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float sampleFrequency = frequency;
    float prev = 1.0f;
    [loop]
    for (uint octave = 0u; octave < octaves; ++octave)
    {
        float n = 1.0f - abs(perlin2(x * sampleFrequency, y * sampleFrequency, seed + 97u + octave * 29u));
        n *= n;
        sum += n * amplitude * prev;
        prev = n * gain;
        sampleFrequency *= lacunarity;
        amplitude *= 0.5f;
    }
    return sum;
}

struct CoastProfileSettings
{
    float inlandBlendDistance;
    float offshoreBlendDistance;
    float shorelineRise;
    float nearshoreDepth;
    float roughFadeDistance;
    float hillFadeDistance;
    float mountainFadeDistance;
    float roughFloor;
    float hillFloor;
    float mountainFloor;
};

CoastProfileSettings coastProfileSettings(uint profile)
{
    CoastProfileSettings settings;
    settings.inlandBlendDistance = 0.0f;
    settings.offshoreBlendDistance = 0.0f;
    settings.shorelineRise = 0.0f;
    settings.nearshoreDepth = 0.0f;
    settings.roughFadeDistance = 0.0f;
    settings.hillFadeDistance = 0.0f;
    settings.mountainFadeDistance = 0.0f;
    settings.roughFloor = 0.0f;
    settings.hillFloor = 0.0f;
    settings.mountainFloor = 0.0f;
    if (profile == 1u)
    {
        settings.inlandBlendDistance = 72.0f;
        settings.offshoreBlendDistance = 56.0f;
        settings.shorelineRise = 2.5f;
        settings.nearshoreDepth = 2.0f;
        settings.roughFadeDistance = 42.0f;
        settings.hillFadeDistance = 54.0f;
        settings.mountainFadeDistance = 68.0f;
        settings.roughFloor = 0.16f;
        settings.hillFloor = 0.08f;
        settings.mountainFloor = 0.02f;
        return settings;
    }
    if (profile == 2u)
    {
        settings.inlandBlendDistance = 40.0f;
        settings.offshoreBlendDistance = 36.0f;
        settings.shorelineRise = 4.5f;
        settings.nearshoreDepth = 4.0f;
        settings.roughFadeDistance = 28.0f;
        settings.hillFadeDistance = 34.0f;
        settings.mountainFadeDistance = 40.0f;
        settings.roughFloor = 0.35f;
        settings.hillFloor = 0.24f;
        settings.mountainFloor = 0.12f;
        return settings;
    }
    if (profile == 3u)
    {
        settings.inlandBlendDistance = 18.0f;
        settings.offshoreBlendDistance = 28.0f;
        settings.shorelineRise = 12.0f;
        settings.nearshoreDepth = 6.0f;
        settings.roughFadeDistance = 18.0f;
        settings.hillFadeDistance = 22.0f;
        settings.mountainFadeDistance = 28.0f;
        settings.roughFloor = 0.60f;
        settings.hillFloor = 0.52f;
        settings.mountainFloor = 0.46f;
        return settings;
    }
    if (profile == 4u)
    {
        settings.inlandBlendDistance = 84.0f;
        settings.offshoreBlendDistance = 52.0f;
        settings.shorelineRise = 0.75f;
        settings.nearshoreDepth = 1.5f;
        settings.roughFadeDistance = 54.0f;
        settings.hillFadeDistance = 66.0f;
        settings.mountainFadeDistance = 80.0f;
        settings.roughFloor = 0.04f;
        settings.hillFloor = 0.02f;
        settings.mountainFloor = 0.00f;
        return settings;
    }

    settings.inlandBlendDistance = 56.0f;
    settings.offshoreBlendDistance = 64.0f;
    settings.shorelineRise = 1.5f;
    settings.nearshoreDepth = 2.5f;
    settings.roughFadeDistance = 36.0f;
    settings.hillFadeDistance = 44.0f;
    settings.mountainFadeDistance = 56.0f;
    settings.roughFloor = 0.08f;
    settings.hillFloor = 0.05f;
    settings.mountainFloor = 0.02f;
    return settings;
}

float shorelineNoiseFactor(float distance, float fadeDistance, float floorValue)
{
    return lerp(floorValue, 1.0f, smoothStep(distance / max(fadeDistance, 1.0f)));
}

float solveShorelineBaseHeight(float signedDistance,
                               float landBaseHeight,
                               float oceanBaseHeight,
                               float seaLevel,
                               CoastProfileSettings settings)
{
    const float shorelineLandHeight = seaLevel + settings.shorelineRise;
    const float nearshoreFloor = seaLevel - settings.nearshoreDepth;
    const float safeLandBase = max(landBaseHeight, shorelineLandHeight);
    const float safeOceanBase = min(oceanBaseHeight, nearshoreFloor);

    if (signedDistance >= 0.0f)
    {
        const float inlandFactor = smoothStep(signedDistance / max(settings.inlandBlendDistance, 1.0f));
        return lerp(shorelineLandHeight, safeLandBase, inlandFactor);
    }

    const float offshoreFactor = smoothStep((-signedDistance) / max(settings.offshoreBlendDistance, 1.0f));
    return lerp(nearshoreFloor, safeOceanBase, offshoreFactor);
}

float categoryScore(float value01, uint bits, uint lowMask, uint midMask, uint highMask)
{
    if ((bits & (lowMask | midMask | highMask)) == 0u)
    {
        return 1.0f;
    }

    const float low = saturate((0.6f - value01) / 0.6f);
    const float high = saturate((value01 - 0.4f) / 0.6f);
    const float mid = 1.0f - min(1.0f, abs(value01 - 0.5f) * 2.2f);

    float score = 0.0f;
    if ((bits & lowMask) != 0u) score = max(score, low);
    if ((bits & midMask) != 0u) score = max(score, mid);
    if ((bits & highMask) != 0u) score = max(score, high);
    return score;
}

float scoreBiome(FarLodGpuBiome biome,
                 float oceaniness,
                 float temperature01,
                 float moisture01,
                 float fertility01,
                 float mountain01,
                 float inland01)
{
    float score = max(biome.spawnChance, 0.01f);
    const bool isOcean = (biome.flags & kFarLodBiomeOcean) != 0u;
    const float oceanScore = isOcean ? saturate((0.5f - oceaniness) * 2.0f + 0.5f)
                                     : saturate((oceaniness + 0.5f) * 2.0f);
    score *= max(oceanScore, 0.05f);
    score *= max(categoryScore(temperature01, biome.propertyBits, kPropCold, kPropTemperate, kPropHot), 0.05f);
    score *= max(categoryScore(moisture01, biome.propertyBits, kPropDry, kPropNeutralHydration, kPropWet), 0.05f);
    score *= max(categoryScore(fertility01, biome.propertyBits, kPropBarren, kPropBalanced, kPropOvergrown), 0.05f);
    score *= max(categoryScore(mountain01, biome.propertyBits, kPropLowTerrain, kPropAntiMountain, kPropMountain), 0.05f);
    if ((biome.propertyBits & kPropInland) != 0u)
    {
        score *= max(inland01, 0.05f);
    }
    score *= lerp(0.85f, 1.15f, saturate(biome.interpolationWeight));
    return score;
}

void resolveFarLodColumnBlocks(FarLodGpuBiome biome,
                               int surfaceY,
                               float distanceToShore,
                               int seaLevel,
                               int worldX,
                               int worldZ,
                               out uint surfaceBlock,
                               out uint fillerBlock)
{
    surfaceBlock = biome.surfaceBlock;
    fillerBlock = biome.fillerBlock;

    const bool nearSeaLevel = abs(surfaceY - seaLevel) <= 2;
    const float kBeachDistanceRange = 6.0f;
    if ((biome.flags & kFarLodBiomeOcean) == 0u &&
        nearSeaLevel &&
        distanceToShore <= kBeachDistanceRange)
    {
        const float noise = hashToUnitFloat(worldX, surfaceY, worldZ);
        if ((biome.flags & kFarLodBiomeSmoothBeaches) != 0u)
        {
            const float shorelineWeight = 1.0f - saturate(distanceToShore / kBeachDistanceRange);
            const float sandProbability = lerp(0.4f, 0.95f, shorelineWeight);
            if (noise <= sandProbability)
            {
                surfaceBlock = 4u;
                fillerBlock = 4u;
            }
            else if (noise < sandProbability + 0.1f)
            {
                fillerBlock = 4u;
            }
        }
        else
        {
            surfaceBlock = noise < 0.55f ? 4u : surfaceBlock;
            fillerBlock = 4u;
        }
    }

    if ((biome.flags & kFarLodBiomeTaiga) != 0u && surfaceBlock != 4u)
    {
        const float patchNoise = taigaPodzolNoise(worldX, worldZ);
        const float patchSelector = hashToUnitFloat(worldX, surfaceY * 23 + 11, worldZ);
        const bool usePodzol = patchNoise > 0.67f || (patchNoise > 0.59f && patchSelector > 0.45f);
        if (usePodzol)
        {
            surfaceBlock = 9u;
            fillerBlock = 9u;
        }
    }
}

struct TerrainPointSample
{
    bool hasSolid;
    bool waterEnabled;
    int surfaceY;
    int waterBottomY;
    uint surfaceBlock;
    uint fillerBlock;
};

TerrainPointSample sampleTerrainPoint(int worldX, int worldZ)
{
    TerrainPointSample sample = (TerrainPointSample)0;
    const FarLodGpuWorldgenHeader header = gWorldgenHeader[0];
    if (header.biomeCount == 0u)
    {
        return sample;
    }

    // Temporary lightweight GPU worldgen path for stability under debug/TDR pressure.
    // Keep the same overall contract shape, but drastically reduce noise cost.
    const float warpSample =
        valueNoise2D((float)worldX, (float)worldZ, header.warpFrequency, int(header.seed + 11u)) * 2.0f - 1.0f;
    const float warpedX = (float)worldX + warpSample * header.warpAmplitude;
    const float warpedZ = (float)worldZ + warpSample * header.warpAmplitude;

    const float continental =
        valueNoise2D(warpedX, warpedZ, header.mainFrequency * kContinentalScale, int(header.seed + 23u)) * 2.0f - 1.0f;
    const float temperature01 = saturate(
        valueNoise2D(warpedX + 101.7f, warpedZ - 73.1f, header.mediumFrequency * kTemperatureScale, int(header.seed + 41u)));
    const float moisture01 = saturate(
        valueNoise2D(warpedX - 211.4f, warpedZ + 39.5f, header.mediumFrequency * kMoistureScale, int(header.seed + 59u)));
    const float fertility01 = saturate(
        valueNoise2D(warpedX + 19.5f, warpedZ + 311.9f, header.detailFrequency * kFertilityScale, int(header.seed + 71u)));
    const float mountain01 = saturate(
        valueNoise2D(warpedX, warpedZ, header.mountainFrequency, int(header.seed + 83u)));

    const float oceaniness = continental - mountain01 * 0.35f;
    const float signedDistanceToCoast = (oceaniness - kOceanThreshold) * kCoastDistanceScale;
    const float inland01 = saturate(abs(signedDistanceToCoast) / 96.0f);

    uint bestBiomeIndex = 0u;
    float bestScore = -1.0f;
    [loop]
    for (uint biomeIndex = 0u; biomeIndex < header.biomeCount; ++biomeIndex)
    {
        const float score = scoreBiome(gBiomes[biomeIndex],
                                       oceaniness,
                                       temperature01,
                                       moisture01,
                                       fertility01,
                                       mountain01,
                                       inland01);
        if (score > bestScore)
        {
            bestScore = score;
            bestBiomeIndex = biomeIndex;
        }
    }

    const FarLodGpuBiome biome = gBiomes[bestBiomeIndex];
    const CoastProfileSettings coastSettings = coastProfileSettings(biome.coastProfile);
    const float continental01 = saturate(continental * 0.5f + 0.5f);
    const float landBaseHeight = lerp(biome.minHeight, biome.maxHeight, continental01) + biome.heightOffset + continental * biome.heightScale;
    const float oceanBaseHeight = (float)header.seaLevel - 28.0f + oceaniness * 12.0f;
    float baseHeight = solveShorelineBaseHeight(signedDistanceToCoast,
                                                landBaseHeight,
                                                oceanBaseHeight,
                                                (float)header.seaLevel,
                                                coastSettings);

    float roughStrength = max(biome.roughness, 0.0f);
    float hillStrength = max(biome.hills, 0.0f);
    float mountainStrength = max(biome.mountains, 0.0f);
    const float absCoastDistance = abs(signedDistanceToCoast);
    roughStrength *= shorelineNoiseFactor(absCoastDistance, coastSettings.roughFadeDistance, coastSettings.roughFloor);
    hillStrength *= shorelineNoiseFactor(absCoastDistance, coastSettings.hillFadeDistance, coastSettings.hillFloor);
    mountainStrength *= shorelineNoiseFactor(absCoastDistance, coastSettings.mountainFadeDistance, coastSettings.mountainFloor);

    const float roughNoise =
        valueNoise2D(warpedX, warpedZ, header.detailFrequency, int(header.seed + 97u)) * 2.0f - 1.0f;
    const float hillNoise =
        valueNoise2D(warpedX, warpedZ, header.mediumFrequency, int(header.seed + 113u)) * 2.0f - 1.0f;
    const float mountainNoise = valueNoise2D(warpedX, warpedZ, header.mountainFrequency, int(header.seed + 131u));

    float surfaceHeight = baseHeight;
    surfaceHeight += roughNoise * 4.0f * roughStrength;
    surfaceHeight += hillNoise * 6.0f * hillStrength;
    surfaceHeight += mountainNoise * 12.0f * mountainStrength;
    sample.surfaceY = (int)round(surfaceHeight);

    const float distanceToShore = ((biome.flags & kFarLodBiomeOcean) != 0u) ? abs(signedDistanceToCoast)
                                                                             : max(signedDistanceToCoast, 0.0f);
    resolveFarLodColumnBlocks(biome,
                              sample.surfaceY,
                              distanceToShore,
                              header.seaLevel,
                              worldX,
                              worldZ,
                              sample.surfaceBlock,
                              sample.fillerBlock);

    sample.hasSolid = true;
    sample.waterEnabled = (biome.flags & kFarLodBiomeWaterFill) != 0u;
    sample.waterBottomY = max(sample.surfaceY + 1, header.seaLevel - biome.waterMaxDepth + 1);
    return sample;
}

// Temporary stability pass: classify from the center sample only.
// This keeps the GPU synth pipeline alive while we incrementally add
// the full 5-point footprint logic back without triggering TDRs.
[numthreads(4, 4, 1)]
void FarLodChunkSynthMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kLogicalSize || dispatchThreadId.y >= kLogicalSize)
    {
        return;
    }

    const uint localX = dispatchThreadId.x;
    const uint localZ = dispatchThreadId.y;
    const int minX = gWorldMin.x + int(localX) * gBlockScale;
    const int maxX = minX + (gBlockScale - 1);
    const int minZ = gWorldMin.z + int(localZ) * gBlockScale;
    const int maxZ = minZ + (gBlockScale - 1);
    const int centerX = (minX + maxX) / 2;
    const int centerZ = (minZ + maxZ) / 2;

    GpuTerrainColumnDescriptor descriptor = (GpuTerrainColumnDescriptor)0;
    const TerrainPointSample centerSample = sampleTerrainPoint(centerX, centerZ);
    descriptor.centerHasSolid = centerSample.hasSolid ? 1u : 0u;
    descriptor.centerWaterEnabled = centerSample.waterEnabled ? 1u : 0u;
    descriptor.centerSurfaceY = centerSample.surfaceY;
    descriptor.centerWaterBottomY = centerSample.waterBottomY;
    descriptor.centerSurfaceBlock = centerSample.surfaceBlock;
    descriptor.centerFillerBlock = centerSample.fillerBlock;
    descriptor.minSurfaceY = centerSample.surfaceY;
    descriptor.maxSurfaceY = centerSample.surfaceY;
    descriptor.minWaterBottomY = centerSample.waterBottomY;
    descriptor.waterHitCount = (centerSample.waterEnabled && centerSample.surfaceY < gSeaLevel) ? 5u : 0u;

    gColumnBuffer[columnIndex(localX, localZ)] = descriptor;
}
