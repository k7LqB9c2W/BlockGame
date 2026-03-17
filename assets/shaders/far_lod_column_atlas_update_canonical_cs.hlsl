cbuffer AtlasUpdateParams : register(b0)
{
    int gAtlasOriginCellX;
    int gAtlasOriginCellZ;
    int gAtlasSizeX;
    int gAtlasSizeZ;
    int gUpdateOriginCellX;
    int gUpdateOriginCellZ;
    int gUpdateSizeX;
    int gUpdateSizeZ;
    int gBlockScale;
    int gSeaLevel;
};

struct FarLodGpuFloat2
{
    float x;
    float y;
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
    float treeDensityFrequency;
    uint treeDensityOctaves;
    float treeDensityGain;
    float treeDensityLacunarity;
    FarLodGpuFloat2 treeDensityOctaveOffsets[4];
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

struct GpuTerrainAtlasSample
{
    uint hasSolid;
    uint waterEnabled;
    int surfaceY;
    int waterBottomY;
    int minSurfaceY;
    int maxSurfaceY;
    uint surfaceBlock;
    uint fillerBlock;
    int canopyBottomY;
    int canopyTopY;
    uint canopyBlock;
    uint canopyStrength;
};

struct GpuTerrainAtlasReductionInput
{
    int surfaceHeights[9];
    uint validCount;
};

struct CanopyPointSample
{
    uint biomeFlags;
    int surfaceY;
};

StructuredBuffer<FarLodGpuWorldgenHeader> gWorldgenHeader : register(t0);
StructuredBuffer<FarLodGpuBiome> gBiomes : register(t1);
StructuredBuffer<GpuTerrainAtlasSample> gCanonicalSamples : register(t2);
StructuredBuffer<GpuTerrainAtlasReductionInput> gCanonicalReductionInputs : register(t3);
RWStructuredBuffer<GpuTerrainAtlasSample> gAtlasSamples : register(u0);

static const uint kFarLodBiomeOcean = 1u << 0;
static const uint kFarLodBiomeWaterFill = 1u << 2;
static const uint kFarLodBiomeTaiga = 1u << 3;
static const uint kFarLodBiomeGeneratesTrees = 1u << 4;
static const uint kPropHot = 1u << 0;
static const uint kPropTemperate = 1u << 1;
static const uint kPropCold = 1u << 2;
static const uint kPropInland = 1u << 3;
static const uint kPropWet = 1u << 6;
static const uint kPropNeutralHydration = 1u << 7;
static const uint kPropDry = 1u << 8;
static const uint kPropBarren = 1u << 9;
static const uint kPropBalanced = 1u << 10;
static const uint kPropOvergrown = 1u << 11;
static const uint kPropLowTerrain = 1u << 12;
static const uint kPropAntiMountain = 1u << 13;
static const uint kPropMountain = 1u << 14;
static const float kOceanThreshold = -0.08f;
static const float kCoastDistanceScale = 72.0f;
static const float kTemperatureScale = 0.11f;
static const float kMoistureScale = 0.09f;
static const float kFertilityScale = 0.05f;
static const float kContinentalScale = 0.065f;
static const uint kBlockLeaves = 3u;
static const uint kBlockSpruceLeaves = 8u;

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

int positiveModulo(int value, int divisor)
{
    const int result = value % divisor;
    return result < 0 ? result + divisor : result;
}

uint atlasIndex(int2 cellCoord)
{
    const int atlasX = positiveModulo(cellCoord.x - gAtlasOriginCellX, gAtlasSizeX);
    const int atlasZ = positiveModulo(cellCoord.y - gAtlasOriginCellZ, gAtlasSizeZ);
    return (uint)(atlasZ * gAtlasSizeX + atlasX);
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

CoastProfileSettings coastProfileSettings(uint profile)
{
    CoastProfileSettings settings;
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
    }
    else if (profile == 2u)
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
    }
    else if (profile == 3u)
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
    }
    else if (profile == 4u)
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
    }
    return settings;
}

float shorelineNoiseFactor(float distance, float fadeDistance, float floorValue)
{
    if (!isfinite(distance))
    {
        return 1.0f;
    }
    return lerp(floorValue, 1.0f, smoothStep(distance / max(fadeDistance, 1.0f)));
}

float solveShorelineBaseHeight(float signedDistance, float landBaseHeight, float oceanBaseHeight, float seaLevel, CoastProfileSettings settings)
{
    if (!isfinite(signedDistance))
    {
        return signedDistance < 0.0f ? oceanBaseHeight : landBaseHeight;
    }

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

float scoreBiome(FarLodGpuBiome biome, float oceaniness, float temperature01, float moisture01, float fertility01, float mountain01, float inland01)
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

CanopyPointSample sampleCanopyPoint(FarLodGpuWorldgenHeader header, int worldX, int worldZ)
{
    CanopyPointSample result = (CanopyPointSample)0;
    if (header.biomeCount == 0u)
    {
        return result;
    }

    const float warpSample =
        fbm2((float)worldX,
             (float)worldZ,
             header.mainFrequency,
             header.mainOctaves,
             header.mainGain,
             header.mainLacunarity,
             header.seed + 11u,
             header.warpFrequency / max(header.mainFrequency, 1.0e-6f));
    const float warpedX = (float)worldX + warpSample * header.warpAmplitude;
    const float warpedZ = (float)worldZ + warpSample * header.warpAmplitude;

    const float continental =
        fbm2(warpedX, warpedZ, header.mainFrequency, header.mainOctaves, header.mainGain, header.mainLacunarity, header.seed + 23u, kContinentalScale);
    const float temperature01 =
        saturate(fbm2(warpedX + 101.7f, warpedZ - 73.1f, header.mediumFrequency, header.mediumOctaves, header.mediumGain, header.mediumLacunarity, header.seed + 41u, kTemperatureScale) * 0.5f + 0.5f);
    const float moisture01 =
        saturate(fbm2(warpedX - 211.4f, warpedZ + 39.5f, header.mediumFrequency, header.mediumOctaves, header.mediumGain, header.mediumLacunarity, header.seed + 59u, kMoistureScale) * 0.5f + 0.5f);
    const float fertility01 =
        saturate(fbm2(warpedX + 19.5f, warpedZ + 311.9f, header.detailFrequency, header.detailOctaves, header.detailGain, header.detailLacunarity, header.seed + 71u, kFertilityScale) * 0.5f + 0.5f);
    const float mountain01 =
        saturate(ridge2(warpedX, warpedZ, header.mountainFrequency, header.mountainOctaves, header.mountainGain, header.mountainLacunarity, header.seed + 83u));

    const float oceaniness = continental - mountain01 * 0.35f;
    const float signedDistanceToCoast = (oceaniness - kOceanThreshold) * kCoastDistanceScale;
    const float inland01 = saturate(abs(signedDistanceToCoast) / 96.0f);

    uint bestBiomeIndex = 0u;
    float bestScore = -1.0f;
    [loop]
    for (uint biomeIndex = 0u; biomeIndex < header.biomeCount; ++biomeIndex)
    {
        const float score = scoreBiome(gBiomes[biomeIndex], oceaniness, temperature01, moisture01, fertility01, mountain01, inland01);
        if (score > bestScore)
        {
            bestScore = score;
            bestBiomeIndex = biomeIndex;
        }
    }

    const FarLodGpuBiome biome = gBiomes[bestBiomeIndex];
    const bool isOcean = (biome.flags & kFarLodBiomeOcean) != 0u;
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
    const float absCoastDistance = isfinite(signedDistanceToCoast) ? abs(signedDistanceToCoast) : 3.402823466e+38F;
    roughStrength *= shorelineNoiseFactor(absCoastDistance, coastSettings.roughFadeDistance, coastSettings.roughFloor);
    hillStrength *= shorelineNoiseFactor(absCoastDistance, coastSettings.hillFadeDistance, coastSettings.hillFloor);
    mountainStrength *= shorelineNoiseFactor(absCoastDistance, coastSettings.mountainFadeDistance, coastSettings.mountainFloor);

    const float roughNoise =
        fbm2(warpedX, warpedZ, header.detailFrequency, header.detailOctaves, header.detailGain, header.detailLacunarity, header.seed + 97u, 1.0f);
    const float hillNoise =
        fbm2(warpedX, warpedZ, header.mediumFrequency, header.mediumOctaves, header.mediumGain, header.mediumLacunarity, header.seed + 113u, 1.0f);
    const float mountainNoise =
        ridge2(warpedX, warpedZ, header.mountainFrequency, header.mountainOctaves, header.mountainGain, header.mountainLacunarity, header.seed + 131u);

    float surfaceHeight = baseHeight;
    surfaceHeight += roughNoise * 4.0f * roughStrength;
    surfaceHeight += hillNoise * 6.0f * hillStrength;
    surfaceHeight += mountainNoise * 12.0f * mountainStrength;

    result.biomeFlags = biome.flags;
    result.surfaceY = (int)round(surfaceHeight);
    return result;
}

float structureDensity(FarLodGpuWorldgenHeader header, int worldX, int worldZ)
{
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float value = 0.0f;
    float normalization = 0.0f;
    const uint octaveCount = min(header.treeDensityOctaves, 4u);
    [loop]
    for (uint octave = 0u; octave < octaveCount; ++octave)
    {
        const float sampleX =
            (float)worldX * header.treeDensityFrequency * frequency + header.treeDensityOctaveOffsets[octave].x;
        const float sampleZ =
            (float)worldZ * header.treeDensityFrequency * frequency + header.treeDensityOctaveOffsets[octave].y;
        value += perlin2(sampleX, sampleZ, 0u) * amplitude;
        normalization += amplitude;
        amplitude *= header.treeDensityGain;
        frequency *= header.treeDensityLacunarity;
    }

    if (normalization > 0.0f)
    {
        value /= normalization;
    }
    return value;
}

[numthreads(8, 8, 1)]
void FarLodColumnAtlasUpdateMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= (uint)gUpdateSizeX || dispatchThreadId.y >= (uint)gUpdateSizeZ)
    {
        return;
    }

    const uint updateIndex = dispatchThreadId.y * (uint)gUpdateSizeX + dispatchThreadId.x;
    const int2 cellCoord = int2(gUpdateOriginCellX + (int)dispatchThreadId.x,
                                gUpdateOriginCellZ + (int)dispatchThreadId.y);

    GpuTerrainAtlasSample sample = gCanonicalSamples[updateIndex];
    sample.canopyBottomY = 0;
    sample.canopyTopY = 0;
    sample.canopyBlock = 0u;
    sample.canopyStrength = 0u;

    if (sample.hasSolid != 0u)
    {
        const FarLodGpuWorldgenHeader header = gWorldgenHeader[0];
        const int worldX = cellCoord.x * gBlockScale;
        const int worldZ = cellCoord.y * gBlockScale;
        const int footprint = max(gBlockScale, 1);
        const int maxSampleX = worldX + (footprint - 1);
        const int maxSampleZ = worldZ + (footprint - 1);
        const int centerX = worldX + footprint / 2;
        const int centerZ = worldZ + footprint / 2;
        const int midX = worldX + footprint / 2;
        const int midZ = worldZ + footprint / 2;
        const int2 points[9] = {
            int2(worldX, worldZ),
            int2(maxSampleX, worldZ),
            int2(worldX, maxSampleZ),
            int2(maxSampleX, maxSampleZ),
            int2(centerX, centerZ),
            int2(midX, worldZ),
            int2(midX, maxSampleZ),
            int2(worldX, midZ),
            int2(maxSampleX, midZ),
        };

        const GpuTerrainAtlasReductionInput reduction = gCanonicalReductionInputs[updateIndex];
        if (reduction.validCount > 0u)
        {
            int surfaceHeights[9];
            [unroll]
            for (uint i = 0u; i < 9u; ++i)
            {
                surfaceHeights[i] = reduction.surfaceHeights[i];
            }
            [unroll]
            for (uint i = 1u; i < reduction.validCount; ++i)
            {
                const int value = surfaceHeights[i];
                int insertIndex = (int)i - 1;
                while (insertIndex >= 0 && surfaceHeights[insertIndex] > value)
                {
                    surfaceHeights[insertIndex + 1] = surfaceHeights[insertIndex];
                    insertIndex -= 1;
                }
                surfaceHeights[insertIndex + 1] = value;
            }

            sample.minSurfaceY = surfaceHeights[0];
            sample.maxSurfaceY = surfaceHeights[reduction.validCount - 1u];
            if ((reduction.validCount & 1u) != 0u)
            {
                sample.surfaceY = surfaceHeights[reduction.validCount / 2u];
            }
            else
            {
                sample.surfaceY =
                    (surfaceHeights[reduction.validCount / 2u - 1u] + surfaceHeights[reduction.validCount / 2u]) / 2;
            }
        }

        CanopyPointSample pointSamples[9];
        [unroll]
        for (uint i = 0u; i < 9u; ++i)
        {
            pointSamples[i] = sampleCanopyPoint(header, points[i].x, points[i].y);
        }
        const CanopyPointSample centerPoint = pointSamples[4];
        if ((centerPoint.biomeFlags & kFarLodBiomeGeneratesTrees) != 0u &&
            (centerPoint.biomeFlags & kFarLodBiomeOcean) == 0u)
        {
            float canopyDensitySum = 0.0f;
            uint canopyDensitySamples = 0u;
            uint taigaCanopyVotes = 0u;
            [unroll]
            for (uint i = 0u; i < 9u; ++i)
            {
                const CanopyPointSample pointSample = pointSamples[i];
                if ((pointSample.biomeFlags & kFarLodBiomeGeneratesTrees) == 0u ||
                    (pointSample.biomeFlags & kFarLodBiomeOcean) != 0u ||
                    pointSample.surfaceY < gSeaLevel - 1)
                {
                    continue;
                }

                canopyDensitySum += saturate(structureDensity(header, points[i].x, points[i].y) * 0.5f + 0.5f);
                ++canopyDensitySamples;
                if ((pointSample.biomeFlags & kFarLodBiomeTaiga) != 0u)
                {
                    ++taigaCanopyVotes;
                }
            }

            if (canopyDensitySamples > 0u)
            {
                const float densityLerp = (float)(min(gBlockScale, 32) - 1) / 31.0f;
                const float averageCanopyDensity = canopyDensitySum / (float)canopyDensitySamples;
                const float densityThreshold = clamp(lerp(0.48f, 0.28f, densityLerp), 0.24f, 0.48f);
                if (averageCanopyDensity >= densityThreshold)
                {
                    const bool taigaCanopy = taigaCanopyVotes * 2u >= canopyDensitySamples;
                    const int canopyLift = taigaCanopy ? 3 : 2;
                    const int canopyThickness =
                        (taigaCanopy ? 6 : 4) + (int)round(averageCanopyDensity * (taigaCanopy ? 4.0f : 3.0f));
                    sample.canopyBottomY = max(sample.surfaceY + canopyLift, sample.maxSurfaceY + 1);
                    sample.canopyTopY = sample.canopyBottomY + max(canopyThickness, 2);
                    sample.canopyBlock = taigaCanopy ? kBlockSpruceLeaves : kBlockLeaves;
                    sample.canopyStrength = (uint)clamp(round(averageCanopyDensity * 255.0f), 0.0f, 255.0f);
                }
            }
        }
    }

    gAtlasSamples[atlasIndex(cellCoord)] = sample;
}
