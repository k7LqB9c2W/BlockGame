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
    uint biomeSelectionCount;
    uint oceanSelectionCount;
    uint transitionCount;
    uint subBiomeCount;
    int chunkSpan;
    int neighborRadius;
    int maxTransitionWidth;
    float totalSpawnWeight;
    float totalOceanWeight;
    float coastDistanceFieldRange;
    float warpFrequency;
    float warpAmplitude;
    float mainFrequency;
    int mainOctaves;
    float mainGain;
    float mainLacunarity;
    float mediumFrequency;
    int mediumOctaves;
    float mediumGain;
    float mediumLacunarity;
    float detailFrequency;
    int detailOctaves;
    float detailGain;
    float detailLacunarity;
    float mountainFrequency;
    int mountainOctaves;
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
    uint interpolationCurve;
    float radius;
    float radiusVariation;
    uint fixedRadius;
    float treeDensityMultiplier;
    float maxSubBiomeCount;
    float subBiomeTotalChance;
    int minHeightLimit;
    int maxHeightLimit;
    uint hasMinHeightLimit;
    uint hasMaxHeightLimit;
    float baseSlopeBias;
    float maxGradient;
    float footprintMultiplier;
    uint transitionOffset;
    uint transitionCount;
    uint subBiomeOffset;
    uint subBiomeCount;
};

struct FarLodGpuBiomeSelection
{
    uint biomeIndex;
    float prefixWeight;
    uint reserved0;
    uint reserved1;
};

struct FarLodGpuTransitionBiome
{
    uint biomeIndex;
    float chance;
    int width;
    uint propertyBits;
};

struct FarLodGpuSubBiome
{
    uint biomeIndex;
    float chance;
    float minRadius;
    float maxRadius;
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

struct SamplePoint
{
    uint biomeIndex;
    uint biomeFlags;
    int surfaceY;
    float distanceToShore;
};

struct ClimateComposition
{
    float aggregatedHeight;
    float aggregatedRoughness;
    float aggregatedHills;
    float aggregatedMountains;
    float keepOriginalMix;
    float landWeight;
    float oceanWeight;
    float landHeight;
    float oceanHeight;
    float landRoughness;
    float oceanRoughness;
    float landHills;
    float oceanHills;
    float landMountains;
    float oceanMountains;
    float landKeepOriginal;
    float oceanKeepOriginal;
    uint landRepresentativeBiome;
    uint oceanRepresentativeBiome;
    float landRepresentativeWeight;
    float oceanRepresentativeWeight;
    float2 landSitePos;
    float2 oceanSitePos;
    float landSiteRadius;
    float oceanSiteRadius;
    uint prefersOcean;
};

struct ClimateResolvedPoint
{
    uint biomeIndex;
    uint biomeFlags;
    uint biomePropertyBits;
    float representativeWeight;
    float aggregatedHeight;
    float aggregatedRoughness;
    float aggregatedHills;
    float aggregatedMountains;
    float keepOriginalMix;
    float2 dominantSitePos;
    float dominantSiteRadius;
    float distanceToCoast;
    float signedDistanceToCoast;
    float landBaseHeight;
    float oceanBaseHeight;
    uint dominantIsOcean;
};

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

StructuredBuffer<FarLodGpuWorldgenHeader> gWorldgenHeader : register(t0);
StructuredBuffer<FarLodGpuBiome> gBiomes : register(t1);
StructuredBuffer<FarLodGpuBiomeSelection> gBiomeSelections : register(t2);
StructuredBuffer<FarLodGpuBiomeSelection> gOceanSelections : register(t3);
StructuredBuffer<FarLodGpuTransitionBiome> gTransitionBiomes : register(t4);
StructuredBuffer<FarLodGpuSubBiome> gSubBiomes : register(t5);
StructuredBuffer<uint> gSurfacePermutation : register(t6);
RWStructuredBuffer<GpuTerrainAtlasSample> gAtlasSamples : register(u0);

static const uint kFarLodBiomeOcean = 1u << 0;
static const uint kFarLodBiomeSmoothBeaches = 1u << 1;
static const uint kFarLodBiomeWaterFill = 1u << 2;
static const uint kFarLodBiomeTaiga = 1u << 3;
static const uint kFarLodBiomeGeneratesTrees = 1u << 4;
static const uint kFarLodBiomeBeach = 1u << 5;
static const uint kFarLodBiomeCoastal = 1u << 6;
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
static const uint kBlockSand = 4u;
static const uint kBlockPodzol = 9u;
static const uint kBlockLeaves = 3u;
static const uint kBlockSpruceLeaves = 8u;
static const uint kMaxChunkSeeds = 64u;
static const uint kMaxWeightedSeeds = 4u;
static const float kClimateEpsilon = 1.0e-6f;
static const float kDiagonalStep = 1.41421356237f;
static const float kHugeFloat = 3.402823466e+38F;

struct ExactBiomeSeed
{
    uint biomeIndex;
    int2 position;
    float radius;
    float baseHeight;
};

struct WeightedSeed
{
    ExactBiomeSeed seed;
    float weight;
    float normalizedDistance;
};

uint2 xor64(uint2 a, uint2 b)
{
    return uint2(a.x ^ b.x, a.y ^ b.y);
}

uint2 shr64(uint2 value, uint shiftAmount)
{
    if (shiftAmount == 0u)
    {
        return value;
    }
    if (shiftAmount >= 64u)
    {
        return uint2(0u, 0u);
    }
    if (shiftAmount >= 32u)
    {
        return uint2(value.y >> (shiftAmount - 32u), 0u);
    }
    return uint2((value.x >> shiftAmount) | (value.y << (32u - shiftAmount)),
                 value.y >> shiftAmount);
}

uint2 shl64(uint2 value, uint shiftAmount)
{
    if (shiftAmount == 0u)
    {
        return value;
    }
    if (shiftAmount >= 64u)
    {
        return uint2(0u, 0u);
    }
    if (shiftAmount >= 32u)
    {
        return uint2(0u, value.x << (shiftAmount - 32u));
    }
    return uint2(value.x << shiftAmount,
                 (value.y << shiftAmount) | (value.x >> (32u - shiftAmount)));
}

void mul32x32(uint a, uint b, out uint lowBits, out uint highBits)
{
    const uint a0 = a & 0xFFFFu;
    const uint a1 = a >> 16u;
    const uint b0 = b & 0xFFFFu;
    const uint b1 = b >> 16u;

    const uint p00 = a0 * b0;
    const uint p01 = a0 * b1;
    const uint p10 = a1 * b0;
    const uint p11 = a1 * b1;

    const uint mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    lowBits = (p00 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
    highBits = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
}

uint rngNext(inout uint2 state)
{
    state = xor64(state, shr64(state, 12u));
    state = xor64(state, shl64(state, 25u));
    state = xor64(state, shr64(state, 27u));

    uint lowLL = 0u;
    uint highLL = 0u;
    mul32x32(state.x, 0x4F6CDD1Du, lowLL, highLL);

    uint lowLH = 0u;
    uint highLH = 0u;
    mul32x32(state.x, 0x2545F491u, lowLH, highLH);

    uint lowHL = 0u;
    uint highHL = 0u;
    mul32x32(state.y, 0x4F6CDD1Du, lowHL, highHL);

    return highLL + lowLH + lowHL;
}

float rngNextFloat(inout uint2 state)
{
    return (float)rngNext(state) / 4294967295.0f;
}

int rngNextInt(inout uint2 state, int minInclusive, int maxInclusive)
{
    if (maxInclusive <= minInclusive)
    {
        return minInclusive;
    }
    const uint range = (uint)(maxInclusive - minInclusive + 1);
    return minInclusive + (int)(rngNext(state) % range);
}

float rngNextFloatSigned(inout uint2 state)
{
    return rngNextFloat(state) * 2.0f - 1.0f;
}

uint hashCombine(uint a, uint b)
{
    a ^= b + 0x9E3779B9u + (a << 6) + (a >> 2);
    return a;
}

int floorDivExact(int value, int divisor)
{
    int quotient = value / divisor;
    const int remainder = value % divisor;
    if ((remainder != 0) && ((remainder < 0) != (divisor < 0)))
    {
        quotient -= 1;
    }
    return quotient;
}

float evaluateInterpolationCurve(float t, uint curve)
{
    t = saturate(t);
    if (curve == 0u)
    {
        return t >= 0.5f ? 1.0f : 0.0f;
    }
    if (curve == 1u)
    {
        return t;
    }
    if (t < 0.5f)
    {
        return saturate(2.0f * t * t);
    }
    const float inv = 1.0f - t;
    return saturate(1.0f - 2.0f * inv * inv);
}

float applyHeightLimits(FarLodGpuBiome biome, float height, float normalizedDistance)
{
    if (biome.hasMinHeightLimit == 0u && biome.hasMaxHeightLimit == 0u)
    {
        return height;
    }

    const float t = saturate(normalizedDistance);
    const float fadeValue = smoothstep(0.35f, 0.95f, t);
    if (fadeValue <= 0.0f)
    {
        return height;
    }

    const float original = height;
    float result = height;
    if (biome.hasMinHeightLimit != 0u)
    {
        const float target = lerp(original, (float)biome.minHeightLimit, fadeValue);
        result = max(result, target);
    }
    if (biome.hasMaxHeightLimit != 0u)
    {
        const float target = lerp(original, (float)biome.maxHeightLimit, fadeValue);
        result = min(result, target);
    }
    return result;
}

float2 randomInUnitCircle(inout uint2 rngState)
{
    float2 value = 0.0.xx;
    [loop]
    do
    {
        value.x = rngNextFloatSigned(rngState);
        value.y = rngNextFloatSigned(rngState);
    } while (dot(value, value) > 1.0f);
    return value;
}

float sampleSubBiomeRadius(FarLodGpuSubBiome subBiomeEntry, float defaultRadius, float noiseValue)
{
    const float low = subBiomeEntry.minRadius > 0.0f ? subBiomeEntry.minRadius : defaultRadius * 0.25f;
    const float high = subBiomeEntry.maxRadius > 0.0f ? subBiomeEntry.maxRadius : defaultRadius * 0.75f;
    return lerp(low, high, saturate(noiseValue));
}

uint chooseBiomeIndexFromSelections(inout uint2 rngState, bool oceanOnly)
{
    const FarLodGpuWorldgenHeader header = gWorldgenHeader[0];
    uint selectedBiomeIndex = 0u;
    if (oceanOnly)
    {
        if (header.oceanSelectionCount == 0u || header.totalOceanWeight <= 0.0f)
        {
            oceanOnly = false;
        }
    }

    if (!oceanOnly)
    {
        if (header.biomeSelectionCount == 0u || header.totalSpawnWeight <= 0.0f)
        {
            return 0u;
        }

        const float pick = rngNextFloat(rngState) * header.totalSpawnWeight;
        [loop]
        for (uint index = 0u; index < header.biomeSelectionCount; ++index)
        {
            if (pick <= gBiomeSelections[index].prefixWeight)
            {
                selectedBiomeIndex = gBiomeSelections[index].biomeIndex;
                return selectedBiomeIndex;
            }
        }
        selectedBiomeIndex = gBiomeSelections[header.biomeSelectionCount - 1u].biomeIndex;
        return selectedBiomeIndex;
    }

    const float pick = rngNextFloat(rngState) * header.totalOceanWeight;
    [loop]
    for (uint index = 0u; index < header.oceanSelectionCount; ++index)
    {
        if (pick <= gOceanSelections[index].prefixWeight)
        {
            selectedBiomeIndex = gOceanSelections[index].biomeIndex;
            return selectedBiomeIndex;
        }
    }
    selectedBiomeIndex = gOceanSelections[header.oceanSelectionCount - 1u].biomeIndex;
    return selectedBiomeIndex;
}

ExactBiomeSeed createBiomeSeed(inout uint2 rngState, int worldX, int worldZ, uint biomeIndex)
{
    const FarLodGpuBiome biome = gBiomes[biomeIndex];
    ExactBiomeSeed seed;
    seed.biomeIndex = biomeIndex;
    seed.position = int2(worldX, worldZ);
    float radius = biome.radius;
    if (biome.fixedRadius == 0u && (biome.flags & kFarLodBiomeOcean) == 0u)
    {
        radius = clamp(biome.radius + biome.radiusVariation * rngNextFloatSigned(rngState),
                       max(biome.radius - biome.radiusVariation, 1.0f),
                       max(biome.radius + biome.radiusVariation, 1.0f));
    }
    seed.radius = max(radius, 1.0f);
    if (biome.maxHeight <= biome.minHeight)
    {
        seed.baseHeight = biome.minHeight;
    }
    else
    {
        seed.baseHeight = lerp(biome.minHeight, biome.maxHeight, rngNextFloat(rngState));
    }
    return seed;
}

bool isValidPlacement(int2 position,
                      float radius,
                      ExactBiomeSeed seeds[kMaxChunkSeeds],
                      uint seedCount,
                      float spacingScale)
{
    for (uint i = 0u; i < seedCount; ++i)
    {
        const float largestRadius = max(radius, seeds[i].radius);
        const float baseSpacing = clamp(0.85f - 0.0005f * largestRadius, 0.6f, 0.85f);
        const float spacingFactor = clamp(baseSpacing * spacingScale, 0.4f, 0.85f);
        const float combined = (radius + seeds[i].radius) * spacingFactor;
        const float2 delta = float2(position - seeds[i].position);
        if (dot(delta, delta) < combined * combined)
        {
            return false;
        }
    }
    return true;
}

void spawnSubBiomeSeeds(ExactBiomeSeed parent,
                        inout ExactBiomeSeed seeds[kMaxChunkSeeds],
                        inout uint seedCount,
                        inout uint2 rngState)
{
    const FarLodGpuBiome parentBiome = gBiomes[parent.biomeIndex];
    if (parentBiome.subBiomeCount == 0u || seedCount >= kMaxChunkSeeds)
    {
        return;
    }

    const int maxCount =
        parentBiome.maxSubBiomeCount > 0.0f ? (int)ceil(parentBiome.maxSubBiomeCount) : 2147483647;
    int spawned = 0;

    [loop]
    for (uint subIndex = 0u; subIndex < parentBiome.subBiomeCount; ++subIndex)
    {
        if (seedCount >= kMaxChunkSeeds || spawned >= maxCount)
        {
            break;
        }

        const FarLodGpuSubBiome subBiomeEntry = gSubBiomes[parentBiome.subBiomeOffset + subIndex];
        const uint childBiomeIndex = subBiomeEntry.biomeIndex;
        const float chance = subBiomeEntry.chance;
        if (chance <= 1.192092896e-07f)
        {
            continue;
        }
        if (rngNextFloat(rngState) > saturate(chance))
        {
            continue;
        }

        const float2 offset = randomInUnitCircle(rngState);
        const float parentRadius = max(parent.radius, 1.0f);
        const float distance = parentRadius * 0.6f * sqrt(rngNextFloat(rngState));
        const int2 candidatePos = parent.position + int2((int)(offset.x * distance),
                                                         (int)(offset.y * distance));

        float radius = sampleSubBiomeRadius(subBiomeEntry, parentRadius * 0.75f, rngNextFloat(rngState));
        radius = clamp(radius, 4.0f, parentRadius);

        const FarLodGpuBiome childBiome = gBiomes[childBiomeIndex];
        const bool requiresOceanNeighbor =
            ((childBiome.propertyBits & kPropLand) != 0u && (childBiome.propertyBits & kPropOcean) != 0u) ||
            ((childBiome.flags & kFarLodBiomeBeach) != 0u);
        if (requiresOceanNeighbor)
        {
            bool hasNearbyOcean = false;
            [loop]
            for (uint i = 0u; i < seedCount; ++i)
            {
                if ((gBiomes[seeds[i].biomeIndex].flags & kFarLodBiomeOcean) == 0u)
                {
                    continue;
                }
                if (length(float2(candidatePos - seeds[i].position)) <= radius * 2.0f)
                {
                    hasNearbyOcean = true;
                    break;
                }
            }
            if (!hasNearbyOcean)
            {
                continue;
            }
        }

        if (!isValidPlacement(candidatePos, radius, seeds, seedCount, 1.0f))
        {
            continue;
        }

        ExactBiomeSeed child = createBiomeSeed(rngState, candidatePos.x, candidatePos.y, childBiomeIndex);
        child.radius = radius;
        seeds[seedCount++] = child;
        spawned += 1;
    }
}

bool tryAddOceanSeed(int attempts,
                     float spacingScale,
                     int baseX,
                     int baseZ,
                     inout ExactBiomeSeed seeds[kMaxChunkSeeds],
                     inout uint seedCount,
                     inout bool hasOceanSeed,
                     inout uint2 rngState)
{
    const FarLodGpuWorldgenHeader header = gWorldgenHeader[0];
    if (header.oceanSelectionCount == 0u || seedCount >= 48u)
    {
        return false;
    }

    for (int attempt = 0; attempt < attempts; ++attempt)
    {
        const uint biomeIndex = chooseBiomeIndexFromSelections(rngState, true);
        const int worldX = baseX + rngNextInt(rngState, 0, header.chunkSpan - 1);
        const int worldZ = baseZ + rngNextInt(rngState, 0, header.chunkSpan - 1);
        ExactBiomeSeed seed = createBiomeSeed(rngState, worldX, worldZ, biomeIndex);
        if (!isValidPlacement(seed.position, seed.radius, seeds, seedCount, spacingScale))
        {
            continue;
        }
        seeds[seedCount++] = seed;
        hasOceanSeed = true;
        return true;
    }
    return false;
}

void buildChunkSeeds(int chunkX, int chunkZ, out ExactBiomeSeed seeds[kMaxChunkSeeds], out uint seedCount)
{
    const FarLodGpuWorldgenHeader header = gWorldgenHeader[0];
    const ExactBiomeSeed emptySeed = (ExactBiomeSeed)0;
    [unroll]
    for (uint initIndex = 0u; initIndex < kMaxChunkSeeds; ++initIndex)
    {
        seeds[initIndex] = emptySeed;
    }
    seedCount = 0u;

    const int baseX = chunkX * header.chunkSpan;
    const int baseZ = chunkZ * header.chunkSpan;

    uint seedValue = header.seed;
    seedValue = hashCombine(seedValue, (uint)(chunkX * 73856093));
    seedValue = hashCombine(seedValue, (uint)(chunkZ * 19349663));
    uint2 rngState = uint2(seedValue, 0u);

    const int maxRejections = 96;
    int rejections = 0;
    bool hasOceanSeed = false;

    if (header.totalOceanWeight > 0.0f && header.totalSpawnWeight > 0.0f)
    {
        const float expectedShare = clamp(header.totalOceanWeight / header.totalSpawnWeight, 0.05f, 0.35f);
        if (rngNextFloat(rngState) < expectedShare)
        {
            tryAddOceanSeed(24, 1.0f, baseX, baseZ, seeds, seedCount, hasOceanSeed, rngState);
        }
    }

    [loop]
    while (seedCount < 48u && rejections < maxRejections)
    {
        const int worldX = baseX + rngNextInt(rngState, 0, header.chunkSpan - 1);
        const int worldZ = baseZ + rngNextInt(rngState, 0, header.chunkSpan - 1);
        const uint biomeIndex = chooseBiomeIndexFromSelections(rngState, false);
        ExactBiomeSeed seed = createBiomeSeed(rngState, worldX, worldZ, biomeIndex);
        if (!isValidPlacement(seed.position, seed.radius, seeds, seedCount, 1.0f))
        {
            rejections += 1;
            continue;
        }

        seeds[seedCount++] = seed;
        if ((gBiomes[biomeIndex].flags & kFarLodBiomeOcean) != 0u)
        {
            hasOceanSeed = true;
        }
        rejections = 0;
        spawnSubBiomeSeeds(seed, seeds, seedCount, rngState);
    }

    if (seedCount == 0u)
    {
        const uint biomeIndex = chooseBiomeIndexFromSelections(rngState, false);
        ExactBiomeSeed fallback = createBiomeSeed(rngState, baseX + header.chunkSpan / 2, baseZ + header.chunkSpan / 2, biomeIndex);
        seeds[seedCount++] = fallback;
        if ((gBiomes[biomeIndex].flags & kFarLodBiomeOcean) != 0u)
        {
            hasOceanSeed = true;
        }
        spawnSubBiomeSeeds(fallback, seeds, seedCount, rngState);
    }

    if (!hasOceanSeed)
    {
        if (!tryAddOceanSeed(32, 1.0f, baseX, baseZ, seeds, seedCount, hasOceanSeed, rngState))
        {
            tryAddOceanSeed(48, 0.75f, baseX, baseZ, seeds, seedCount, hasOceanSeed, rngState);
        }
    }
}

void insertWeightedSeed(inout WeightedSeed weighted[kMaxWeightedSeeds], inout uint count, WeightedSeed candidate)
{
    if (count < kMaxWeightedSeeds)
    {
        weighted[count++] = candidate;
    }
    else if (candidate.weight <= weighted[count - 1u].weight)
    {
        return;
    }
    else
    {
        weighted[count - 1u] = candidate;
    }

    int insertIndex = (int)count - 1;
    while (insertIndex > 0 && weighted[insertIndex].weight > weighted[insertIndex - 1].weight)
    {
        const WeightedSeed tmp = weighted[insertIndex - 1];
        weighted[insertIndex - 1] = weighted[insertIndex];
        weighted[insertIndex] = tmp;
        insertIndex -= 1;
    }
}

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

uint surfacePermutationAt(uint index)
{
    return gSurfacePermutation[index & 511u];
}

float grad(uint hashValue, float x, float y)
{
    const uint h = hashValue & 7u;
    const float u = h < 4u ? x : y;
    const float v = h < 4u ? y : x;
    return ((h & 1u) != 0u ? -u : u) + ((h & 2u) != 0u ? -v : v);
}

float perlin2(float x, float y)
{
    const int xi = ((int)floor(x)) & 255;
    const int yi = ((int)floor(y)) & 255;
    const float xf = x - floor(x);
    const float yf = y - floor(y);

    const float u = fade(xf);
    const float v = fade(yf);

    const uint aa = surfacePermutationAt((uint)xi) + (uint)yi;
    const uint ab = surfacePermutationAt((uint)xi) + (uint)yi + 1u;
    const uint ba = surfacePermutationAt((uint)(xi + 1)) + (uint)yi;
    const uint bb = surfacePermutationAt((uint)(xi + 1)) + (uint)yi + 1u;

    const float x1 = lerp(grad(surfacePermutationAt(aa), xf, yf),
                          grad(surfacePermutationAt(ba), xf - 1.0f, yf),
                          u);
    const float x2 = lerp(grad(surfacePermutationAt(ab), xf, yf - 1.0f),
                          grad(surfacePermutationAt(bb), xf - 1.0f, yf - 1.0f),
                          u);
    return lerp(x1, x2, v);
}

float fbm2(float x, float y, float frequency, int octaves, float gain, float lacunarity, float frequencyScale)
{
    float value = 0.0f;
    float amplitude = 1.0f;
    float sampleFrequency = frequency * frequencyScale;
    [loop]
    for (int octave = 0; octave < octaves; ++octave)
    {
        value += perlin2(x * sampleFrequency, y * sampleFrequency) * amplitude;
        amplitude *= gain;
        sampleFrequency *= lacunarity;
    }
    return value;
}

float ridge2(float x, float y, float frequency, int octaves, float lacunarity, float gain)
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float sampleFrequency = frequency;
    float prev = 1.0f;
    [loop]
    for (int octave = 0; octave < octaves; ++octave)
    {
        float n = perlin2(x * sampleFrequency, y * sampleFrequency);
        n = 1.0f - abs(n);
        n *= n;
        sum += n * amplitude * prev;
        prev = n;
        sampleFrequency *= lacunarity;
        amplitude *= gain;
    }
    return sum;
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
    return lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), tz);
}

float taigaPodzolNoise(int worldX, int worldZ)
{
    const float broad = valueNoise2D((float)worldX, (float)worldZ, 1.0f / 16.0f, 19);
    const float medium = valueNoise2D((float)worldX, (float)worldZ, 1.0f / 8.0f, 37);
    const float detail = valueNoise2D((float)worldX, (float)worldZ, 1.0f / 4.0f, 73);
    return broad * 0.55f + medium * 0.30f + detail * 0.15f;
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

float solveShorelineBaseHeight(float signedDistance,
                               float landBaseHeight,
                               float oceanBaseHeight,
                               float seaLevel,
                               CoastProfileSettings settings)
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

uint selectBiomeIndex(float temperature01, float moisture01, float fertility01, float mountain01, float oceaniness, float inland01)
{
    const FarLodGpuWorldgenHeader header = gWorldgenHeader[0];
    uint bestBiomeIndex = 0u;
    float bestScore = -1.0f;
    [loop]
    for (uint biomeIndex = 0u; biomeIndex < header.biomeCount; ++biomeIndex)
    {
        const FarLodGpuBiome biome = gBiomes[biomeIndex];
        const float score = scoreBiome(biome, oceaniness, temperature01, moisture01, fertility01, mountain01, inland01);
        if (score > bestScore)
        {
            bestScore = score;
            bestBiomeIndex = biomeIndex;
        }
    }
    return bestBiomeIndex;
}

SamplePoint samplePointLegacy(int worldX, int worldZ)
{
    const FarLodGpuWorldgenHeader header = gWorldgenHeader[0];
    const int2 worldPos = int2(worldX, worldZ);
    const int chunkX = floorDivExact(worldX, header.chunkSpan);
    const int chunkZ = floorDivExact(worldZ, header.chunkSpan);

    WeightedSeed weightedSeeds[kMaxWeightedSeeds];
    uint weightedCount = 0u;
    float nearestLandEdge = 3.402823466e+38F;
    float nearestOceanEdge = 3.402823466e+38F;

    [loop]
    for (int dz = -header.neighborRadius; dz <= header.neighborRadius; ++dz)
    {
        [loop]
        for (int dx = -header.neighborRadius; dx <= header.neighborRadius; ++dx)
        {
            ExactBiomeSeed chunkSeeds[kMaxChunkSeeds];
            uint chunkSeedCount = 0u;
            buildChunkSeeds(chunkX + dx, chunkZ + dz, chunkSeeds, chunkSeedCount);

            [loop]
            for (uint seedIndex = 0u; seedIndex < chunkSeedCount; ++seedIndex)
            {
                const ExactBiomeSeed seed = chunkSeeds[seedIndex];
                const FarLodGpuBiome seedBiome = gBiomes[seed.biomeIndex];
                const float2 delta = float2(worldPos - seed.position);
                const float distance = length(delta);
                const float normalized = distance / max(seed.radius, 1.0f);
                const float blended = saturate(1.0f - normalized);
                const float influence = smoothStep(blended);

                const float edgeDistance = abs(distance - seed.radius);
                if ((seedBiome.flags & kFarLodBiomeOcean) != 0u)
                {
                    nearestOceanEdge = min(nearestOceanEdge, edgeDistance);
                }
                else
                {
                    nearestLandEdge = min(nearestLandEdge, edgeDistance);
                }

                if (influence <= 1.192092896e-07f)
                {
                    continue;
                }

                const float blendFactor = evaluateInterpolationCurve(1.0f - normalized, seedBiome.interpolationCurve);
                const float adjustedWeight = influence * blendFactor * seedBiome.interpolationWeight;
                if (adjustedWeight <= 1.192092896e-07f)
                {
                    continue;
                }

                WeightedSeed candidate;
                candidate.seed = seed;
                candidate.weight = adjustedWeight;
                candidate.normalizedDistance = normalized;
                insertWeightedSeed(weightedSeeds, weightedCount, candidate);
            }
        }
    }

    uint biomeIndex = 0u;
    float aggregatedHeight = 0.0f;
    float aggregatedRoughness = 0.0f;
    float aggregatedHills = 0.0f;
    float aggregatedMountains = 0.0f;
    float keepOriginal = 0.0f;
    float landWeight = 0.0f;
    float oceanWeight = 0.0f;
    float landHeight = 0.0f;
    float oceanHeight = 0.0f;
    float landRoughness = 0.0f;
    float oceanRoughness = 0.0f;
    float landHills = 0.0f;
    float oceanHills = 0.0f;
    float landMountains = 0.0f;
    float oceanMountains = 0.0f;
    float landKeepOriginal = 0.0f;
    float oceanKeepOriginal = 0.0f;
    uint landRepresentativeBiome = 0xFFFFFFFFu;
    uint oceanRepresentativeBiome = 0xFFFFFFFFu;
    float landRepresentativeWeight = 0.0f;
    float oceanRepresentativeWeight = 0.0f;

    if (weightedCount == 0u)
    {
        const FarLodGpuBiome fallbackBiome = gBiomes[0];
        biomeIndex = 0u;
        aggregatedHeight = applyHeightLimits(fallbackBiome, fallbackBiome.minHeight, 0.0f);
        aggregatedRoughness = fallbackBiome.roughness;
        aggregatedHills = fallbackBiome.hills;
        aggregatedMountains = fallbackBiome.mountains;
        keepOriginal = saturate(fallbackBiome.keepOriginalTerrain);
        if ((fallbackBiome.flags & kFarLodBiomeOcean) != 0u)
        {
            oceanWeight = 1.0f;
            oceanHeight = aggregatedHeight;
            oceanRoughness = aggregatedRoughness;
            oceanHills = aggregatedHills;
            oceanMountains = aggregatedMountains;
            oceanKeepOriginal = keepOriginal;
            oceanRepresentativeBiome = 0u;
            oceanRepresentativeWeight = 1.0f;
        }
        else
        {
            landWeight = 1.0f;
            landHeight = aggregatedHeight;
            landRoughness = aggregatedRoughness;
            landHills = aggregatedHills;
            landMountains = aggregatedMountains;
            landKeepOriginal = keepOriginal;
            landRepresentativeBiome = 0u;
            landRepresentativeWeight = 1.0f;
        }
    }
    else
    {
        float totalWeight = 0.0f;
        [unroll]
        for (uint weightIndex = 0u; weightIndex < kMaxWeightedSeeds; ++weightIndex)
        {
            if (weightIndex >= weightedCount)
            {
                break;
            }
            totalWeight += weightedSeeds[weightIndex].weight;
        }
        totalWeight = max(totalWeight, 1.0e-06f);

        [unroll]
        for (uint aggregateIndex = 0u; aggregateIndex < kMaxWeightedSeeds; ++aggregateIndex)
        {
            if (aggregateIndex >= weightedCount)
            {
                break;
            }

            const WeightedSeed entry = weightedSeeds[aggregateIndex];
            const FarLodGpuBiome entryBiome = gBiomes[entry.seed.biomeIndex];
            const float normalizedWeight = entry.weight / totalWeight;
            const float height = applyHeightLimits(entryBiome, entry.seed.baseHeight, entry.normalizedDistance);

            aggregatedHeight += height * normalizedWeight;
            aggregatedRoughness += entryBiome.roughness * normalizedWeight;
            aggregatedHills += entryBiome.hills * normalizedWeight;
            aggregatedMountains += entryBiome.mountains * normalizedWeight;
            keepOriginal += saturate(entryBiome.keepOriginalTerrain) * normalizedWeight;

            if ((entryBiome.flags & kFarLodBiomeOcean) != 0u)
            {
                oceanWeight += normalizedWeight;
                oceanHeight += height * normalizedWeight;
                oceanRoughness += entryBiome.roughness * normalizedWeight;
                oceanHills += entryBiome.hills * normalizedWeight;
                oceanMountains += entryBiome.mountains * normalizedWeight;
                oceanKeepOriginal += saturate(entryBiome.keepOriginalTerrain) * normalizedWeight;
                if (normalizedWeight > oceanRepresentativeWeight)
                {
                    oceanRepresentativeBiome = entry.seed.biomeIndex;
                    oceanRepresentativeWeight = normalizedWeight;
                }
            }
            else
            {
                landWeight += normalizedWeight;
                landHeight += height * normalizedWeight;
                landRoughness += entryBiome.roughness * normalizedWeight;
                landHills += entryBiome.hills * normalizedWeight;
                landMountains += entryBiome.mountains * normalizedWeight;
                landKeepOriginal += saturate(entryBiome.keepOriginalTerrain) * normalizedWeight;
                if (normalizedWeight > landRepresentativeWeight)
                {
                    landRepresentativeBiome = entry.seed.biomeIndex;
                    landRepresentativeWeight = normalizedWeight;
                }
            }
        }

        if (landWeight > 1.192092896e-07f)
        {
            landHeight /= landWeight;
            landRoughness /= landWeight;
            landHills /= landWeight;
            landMountains /= landWeight;
            landKeepOriginal /= landWeight;
        }
        if (oceanWeight > 1.192092896e-07f)
        {
            oceanHeight /= oceanWeight;
            oceanRoughness /= oceanWeight;
            oceanHills /= oceanWeight;
            oceanMountains /= oceanWeight;
            oceanKeepOriginal /= oceanWeight;
        }
    }

    const bool dominantIsOcean = oceanWeight > landWeight;
    if (dominantIsOcean && oceanRepresentativeBiome != 0xFFFFFFFFu)
    {
        biomeIndex = oceanRepresentativeBiome;
    }
    else if (landRepresentativeBiome != 0xFFFFFFFFu)
    {
        biomeIndex = landRepresentativeBiome;
    }
    else if (oceanRepresentativeBiome != 0xFFFFFFFFu)
    {
        biomeIndex = oceanRepresentativeBiome;
    }

    const FarLodGpuBiome biome = gBiomes[biomeIndex];
    const float groupHeight = dominantIsOcean && oceanWeight > 1.192092896e-07f ? oceanHeight :
                              (landWeight > 1.192092896e-07f ? landHeight : aggregatedHeight);
    const float groupRoughness = dominantIsOcean && oceanWeight > 1.192092896e-07f ? oceanRoughness :
                                 (landWeight > 1.192092896e-07f ? landRoughness : aggregatedRoughness);
    const float groupHills = dominantIsOcean && oceanWeight > 1.192092896e-07f ? oceanHills :
                             (landWeight > 1.192092896e-07f ? landHills : aggregatedHills);
    const float groupMountains = dominantIsOcean && oceanWeight > 1.192092896e-07f ? oceanMountains :
                                 (landWeight > 1.192092896e-07f ? landMountains : aggregatedMountains);
    const float landBaseHeight = landWeight > 1.192092896e-07f ? landHeight : groupHeight;
    const float oceanBaseHeight = oceanWeight > 1.192092896e-07f ? oceanHeight : groupHeight;
    const float signedDistanceToCoast = dominantIsOcean ? -nearestLandEdge : nearestOceanEdge;
    const float distanceToCoast = abs(signedDistanceToCoast);

    float roughStrength = max(groupRoughness, 0.0f);
    float hillStrength = max(groupHills, 0.0f);
    float mountainStrength = max(groupMountains, 0.0f);
    const CoastProfileSettings coastSettings = coastProfileSettings(biome.coastProfile);
    const float absCoastDistance = isfinite(signedDistanceToCoast) ? abs(signedDistanceToCoast) : 3.402823466e+38F;
    float baseHeight = solveShorelineBaseHeight(signedDistanceToCoast,
                                                landBaseHeight,
                                                oceanBaseHeight,
                                                (float)header.seaLevel,
                                                coastSettings);
    roughStrength *= shorelineNoiseFactor(absCoastDistance, coastSettings.roughFadeDistance, coastSettings.roughFloor);
    hillStrength *= shorelineNoiseFactor(absCoastDistance, coastSettings.hillFadeDistance, coastSettings.hillFloor);
    mountainStrength *= shorelineNoiseFactor(absCoastDistance, coastSettings.mountainFadeDistance, coastSettings.mountainFloor);

    const float worldXF = (float)worldX;
    const float worldZF = (float)worldZ;
    const float warpSample =
        fbm2(worldXF,
             worldZF,
             header.mainFrequency,
             header.mainOctaves,
             header.mainGain,
             header.mainLacunarity,
             header.warpFrequency / max(header.mainFrequency, 1.0e-6f));
    const float warpedX = worldXF + warpSample * header.warpAmplitude;
    const float warpedZ = worldZF + warpSample * header.warpAmplitude;

    const float roughNoise = fbm2(warpedX, warpedZ, header.detailFrequency, header.detailOctaves, header.detailGain, header.detailLacunarity, 1.0f);
    const float hillNoise = fbm2(warpedX, warpedZ, header.mediumFrequency, header.mediumOctaves, header.mediumGain, header.mediumLacunarity, 1.0f);
    const float mountainNoise = ridge2(warpedX, warpedZ, header.mountainFrequency, header.mountainOctaves, header.mountainLacunarity, header.mountainGain);

    float surfaceHeight = baseHeight;
    surfaceHeight += (roughNoise - 0.5f) * 4.0f * roughStrength;
    surfaceHeight += (hillNoise - 0.5f) * 6.0f * hillStrength;
    surfaceHeight += mountainNoise * 12.0f * mountainStrength;

    SamplePoint result;
    result.biomeIndex = biomeIndex;
    result.biomeFlags = biome.flags;
    result.surfaceY = (int)round(surfaceHeight);
    result.distanceToShore = distanceToCoast;
    return result;
}

float biomeMaxRadius(FarLodGpuBiome biome)
{
    return max(biome.radius + biome.radiusVariation, 1.0f);
}

uint groupPresenceMask(uint bits)
{
    uint mask = 0u;
    for (int group = 0; group < 5; ++group)
    {
        const uint groupBits = (bits >> (group * 3)) & 0x7u;
        if (groupBits != 0u)
        {
            mask |= 1u << (group * 3);
        }
    }
    return mask;
}

float4 mod289(float4 value)
{
    return value - floor(value * (1.0f / 289.0f)) * 289.0f;
}

float4 permute289(float4 value)
{
    return mod289(((value * 34.0f) + 1.0f) * value);
}

float4 taylorInvSqrt4(float4 value)
{
    return 1.79284291400159f - 0.85373472095314f * value;
}

float2 glmFade2(float2 value)
{
    return (value * value * value) * (value * (value * 6.0f - 15.0f) + 10.0f);
}

float glmPerlin2(float2 position)
{
    float4 Pi = floor(float4(position.x, position.y, position.x, position.y)) + float4(0.0f, 0.0f, 1.0f, 1.0f);
    float4 Pf = frac(float4(position.x, position.y, position.x, position.y)) - float4(0.0f, 0.0f, 1.0f, 1.0f);
    Pi = mod289(Pi);

    const float4 ix = float4(Pi.x, Pi.z, Pi.x, Pi.z);
    const float4 iy = float4(Pi.y, Pi.y, Pi.w, Pi.w);
    const float4 fx = float4(Pf.x, Pf.z, Pf.x, Pf.z);
    const float4 fy = float4(Pf.y, Pf.y, Pf.w, Pf.w);

    const float4 i = permute289(permute289(ix) + iy);

    float4 gx = 2.0f * frac(i * (1.0f / 41.0f)) - 1.0f;
    float4 gy = abs(gx) - 0.5f;
    const float4 tx = floor(gx + 0.5f);
    gx = gx - tx;

    float2 g00 = float2(gx.x, gy.x);
    float2 g10 = float2(gx.y, gy.y);
    float2 g01 = float2(gx.z, gy.z);
    float2 g11 = float2(gx.w, gy.w);

    const float4 norm = taylorInvSqrt4(float4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
    g00 *= norm.x;
    g01 *= norm.y;
    g10 *= norm.z;
    g11 *= norm.w;

    const float n00 = dot(g00, float2(fx.x, fy.x));
    const float n10 = dot(g10, float2(fx.y, fy.y));
    const float n01 = dot(g01, float2(fx.z, fy.z));
    const float n11 = dot(g11, float2(fx.w, fy.w));

    const float2 fadeXY = glmFade2(float2(Pf.x, Pf.y));
    const float2 nX = lerp(float2(n00, n01), float2(n10, n11), fadeXY.x);
    return 2.3f * lerp(nX.x, nX.y, fadeXY.y);
}

float unitPerlinNoise(int worldX, int worldZ, uint seed, float frequency)
{
    const float offsetX = (float)(seed & 0xFFFFu) * 0.013f;
    const float offsetZ = (float)((seed >> 16u) & 0xFFFFu) * 0.017f;
    const float2 sample = float2((float)worldX + offsetX, (float)worldZ + offsetZ) * frequency;
    return saturate(glmPerlin2(sample) * 0.5f + 0.5f);
}

float smoothFactorFromDistance(float distance, float range)
{
    if (!isfinite(distance) || range <= 0.0f)
    {
        return 0.0f;
    }

    const float t = saturate(distance / range);
    return 1.0f - (t * t * (3.0f - 2.0f * t));
}

float gridDistance(int dx, int dz)
{
    const float absDx = abs((float)dx);
    const float absDz = abs((float)dz);
    const float diagonal = min(absDx, absDz);
    const float straight = max(absDx, absDz) - diagonal;
    return diagonal * kDiagonalStep + straight;
}

void resolveRepresentativeFromComposition(ClimateComposition composition, bool useOceanDomain, out ClimateResolvedPoint resolvedPoint)
{
    resolvedPoint = (ClimateResolvedPoint)0;

    const bool hasRequestedGroup = useOceanDomain ? (composition.oceanWeight > kClimateEpsilon)
                                                  : (composition.landWeight > kClimateEpsilon);
    const bool fallbackToOcean = !hasRequestedGroup && composition.oceanWeight > kClimateEpsilon;

    uint representativeBiome = 0xFFFFFFFFu;
    float representativeWeight = 0.0f;
    float groupHeight = composition.aggregatedHeight;
    float groupRoughness = composition.aggregatedRoughness;
    float groupHills = composition.aggregatedHills;
    float groupMountains = composition.aggregatedMountains;
    float groupKeepOriginal = composition.keepOriginalMix;
    float2 dominantSitePos = float2(0.0f, 0.0f);
    float dominantSiteRadius = 0.0f;

    if ((useOceanDomain && !fallbackToOcean) || fallbackToOcean)
    {
        representativeBiome = composition.oceanRepresentativeBiome;
        representativeWeight = composition.oceanRepresentativeWeight;
        if (composition.oceanWeight > kClimateEpsilon)
        {
            groupHeight = composition.oceanHeight;
            groupRoughness = composition.oceanRoughness;
            groupHills = composition.oceanHills;
            groupMountains = composition.oceanMountains;
            groupKeepOriginal = composition.oceanKeepOriginal;
        }
        if (composition.oceanRepresentativeBiome != 0xFFFFFFFFu)
        {
            dominantSitePos = composition.oceanSitePos;
            dominantSiteRadius = composition.oceanSiteRadius;
        }
        useOceanDomain = true;
    }
    else if (composition.landWeight > kClimateEpsilon)
    {
        representativeBiome = composition.landRepresentativeBiome;
        representativeWeight = composition.landRepresentativeWeight;
        groupHeight = composition.landHeight;
        groupRoughness = composition.landRoughness;
        groupHills = composition.landHills;
        groupMountains = composition.landMountains;
        groupKeepOriginal = composition.landKeepOriginal;
        if (composition.landRepresentativeBiome != 0xFFFFFFFFu)
        {
            dominantSitePos = composition.landSitePos;
            dominantSiteRadius = composition.landSiteRadius;
        }
        useOceanDomain = false;
    }

    if (representativeBiome == 0xFFFFFFFFu)
    {
        representativeBiome = composition.landRepresentativeBiome != 0xFFFFFFFFu
            ? composition.landRepresentativeBiome
            : composition.oceanRepresentativeBiome;
        representativeWeight = composition.landRepresentativeWeight > 0.0f
            ? composition.landRepresentativeWeight
            : composition.oceanRepresentativeWeight;
        if (composition.landRepresentativeBiome != 0xFFFFFFFFu)
        {
            dominantSitePos = composition.landSitePos;
            dominantSiteRadius = composition.landSiteRadius;
        }
        else
        {
            dominantSitePos = composition.oceanSitePos;
            dominantSiteRadius = composition.oceanSiteRadius;
        }
    }

    if (representativeBiome == 0xFFFFFFFFu)
    {
        representativeBiome = 0u;
    }

    const FarLodGpuBiome representative = gBiomes[representativeBiome];
    resolvedPoint.biomeIndex = representativeBiome;
    resolvedPoint.biomeFlags = representative.flags;
    resolvedPoint.biomePropertyBits = representative.propertyBits;
    resolvedPoint.representativeWeight = representativeWeight;
    resolvedPoint.aggregatedHeight = groupHeight;
    resolvedPoint.aggregatedRoughness = groupRoughness;
    resolvedPoint.aggregatedHills = groupHills;
    resolvedPoint.aggregatedMountains = groupMountains;
    resolvedPoint.keepOriginalMix = saturate(groupKeepOriginal);
    resolvedPoint.dominantSitePos = dominantSitePos;
    resolvedPoint.dominantSiteRadius = dominantSiteRadius;
    resolvedPoint.landBaseHeight = composition.landWeight > kClimateEpsilon ? composition.landHeight : groupHeight;
    resolvedPoint.oceanBaseHeight = composition.oceanWeight > kClimateEpsilon ? composition.oceanHeight : groupHeight;
    resolvedPoint.dominantIsOcean = useOceanDomain ? 1u : 0u;
    resolvedPoint.distanceToCoast = kHugeFloat;
    resolvedPoint.signedDistanceToCoast = useOceanDomain ? -kHugeFloat : kHugeFloat;
}

void accumulateClimateComposition(int worldX, int worldZ, out ClimateComposition composition)
{
    const FarLodGpuWorldgenHeader header = gWorldgenHeader[0];
    const int2 worldPos = int2(worldX, worldZ);
    const int chunkX = floorDivExact(worldX, header.chunkSpan);
    const int chunkZ = floorDivExact(worldZ, header.chunkSpan);

    composition = (ClimateComposition)0;
    composition.landRepresentativeBiome = 0xFFFFFFFFu;
    composition.oceanRepresentativeBiome = 0xFFFFFFFFu;

    WeightedSeed weightedSeeds[kMaxWeightedSeeds];
    uint weightedCount = 0u;

    for (int dz = -header.neighborRadius; dz <= header.neighborRadius; ++dz)
    {
        for (int dx = -header.neighborRadius; dx <= header.neighborRadius; ++dx)
        {
            ExactBiomeSeed chunkSeeds[kMaxChunkSeeds];
            uint chunkSeedCount = 0u;
            buildChunkSeeds(chunkX + dx, chunkZ + dz, chunkSeeds, chunkSeedCount);

            for (uint seedIndex = 0u; seedIndex < chunkSeedCount; ++seedIndex)
            {
                const ExactBiomeSeed seed = chunkSeeds[seedIndex];
                const FarLodGpuBiome seedBiome = gBiomes[seed.biomeIndex];
                const float distance = length(float2(worldPos - seed.position));
                const float normalized = distance / max(seed.radius, 1.0f);
                const float blended = saturate(1.0f - normalized);
                const float influence = smoothStep(blended);
                if (influence <= kClimateEpsilon)
                {
                    continue;
                }

                const float blendFactor = evaluateInterpolationCurve(1.0f - normalized, seedBiome.interpolationCurve);
                const float adjustedWeight = influence * blendFactor * seedBiome.interpolationWeight;
                if (adjustedWeight <= kClimateEpsilon)
                {
                    continue;
                }

                WeightedSeed candidate;
                candidate.seed = seed;
                candidate.weight = adjustedWeight;
                candidate.normalizedDistance = normalized;
                insertWeightedSeed(weightedSeeds, weightedCount, candidate);
            }
        }
    }

    if (weightedCount == 0u)
    {
        const FarLodGpuBiome fallbackBiome = gBiomes[0];
        const float fallbackHeight = applyHeightLimits(fallbackBiome, fallbackBiome.minHeight, 0.0f);

        composition.aggregatedHeight = fallbackHeight;
        composition.aggregatedRoughness = fallbackBiome.roughness;
        composition.aggregatedHills = fallbackBiome.hills;
        composition.aggregatedMountains = fallbackBiome.mountains;
        composition.keepOriginalMix = saturate(fallbackBiome.keepOriginalTerrain);

        if ((fallbackBiome.flags & kFarLodBiomeOcean) != 0u)
        {
            composition.oceanWeight = 1.0f;
            composition.oceanHeight = fallbackHeight;
            composition.oceanRoughness = fallbackBiome.roughness;
            composition.oceanHills = fallbackBiome.hills;
            composition.oceanMountains = fallbackBiome.mountains;
            composition.oceanKeepOriginal = composition.keepOriginalMix;
            composition.oceanRepresentativeBiome = 0u;
            composition.oceanRepresentativeWeight = 1.0f;
            composition.oceanSitePos = float2((float)worldX, (float)worldZ);
            composition.oceanSiteRadius = biomeMaxRadius(fallbackBiome);
            composition.prefersOcean = 1u;
        }
        else
        {
            composition.landWeight = 1.0f;
            composition.landHeight = fallbackHeight;
            composition.landRoughness = fallbackBiome.roughness;
            composition.landHills = fallbackBiome.hills;
            composition.landMountains = fallbackBiome.mountains;
            composition.landKeepOriginal = composition.keepOriginalMix;
            composition.landRepresentativeBiome = 0u;
            composition.landRepresentativeWeight = 1.0f;
            composition.landSitePos = float2((float)worldX, (float)worldZ);
            composition.landSiteRadius = biomeMaxRadius(fallbackBiome);
            composition.prefersOcean = 0u;
        }
        return;
    }

    float totalWeight = 0.0f;
    for (uint weightIndex = 0u; weightIndex < weightedCount; ++weightIndex)
    {
        totalWeight += weightedSeeds[weightIndex].weight;
    }
    totalWeight = max(totalWeight, kClimateEpsilon);

    for (uint aggregateIndex = 0u; aggregateIndex < weightedCount; ++aggregateIndex)
    {
        const WeightedSeed entry = weightedSeeds[aggregateIndex];
        const FarLodGpuBiome entryBiome = gBiomes[entry.seed.biomeIndex];
        const float normalizedWeight = entry.weight / totalWeight;
        const float height = applyHeightLimits(entryBiome, entry.seed.baseHeight, entry.normalizedDistance);
        const float keepOriginal = saturate(entryBiome.keepOriginalTerrain);
        const float2 sitePos = float2((float)entry.seed.position.x, (float)entry.seed.position.y);

        composition.aggregatedHeight += height * normalizedWeight;
        composition.aggregatedRoughness += entryBiome.roughness * normalizedWeight;
        composition.aggregatedHills += entryBiome.hills * normalizedWeight;
        composition.aggregatedMountains += entryBiome.mountains * normalizedWeight;
        composition.keepOriginalMix += keepOriginal * normalizedWeight;

        if ((entryBiome.flags & kFarLodBiomeOcean) != 0u)
        {
            composition.oceanWeight += normalizedWeight;
            composition.oceanHeight += height * normalizedWeight;
            composition.oceanRoughness += entryBiome.roughness * normalizedWeight;
            composition.oceanHills += entryBiome.hills * normalizedWeight;
            composition.oceanMountains += entryBiome.mountains * normalizedWeight;
            composition.oceanKeepOriginal += keepOriginal * normalizedWeight;
            if (normalizedWeight > composition.oceanRepresentativeWeight)
            {
                composition.oceanRepresentativeBiome = entry.seed.biomeIndex;
                composition.oceanRepresentativeWeight = normalizedWeight;
                composition.oceanSitePos = sitePos;
                composition.oceanSiteRadius = max(entry.seed.radius, 1.0f);
            }
        }
        else
        {
            composition.landWeight += normalizedWeight;
            composition.landHeight += height * normalizedWeight;
            composition.landRoughness += entryBiome.roughness * normalizedWeight;
            composition.landHills += entryBiome.hills * normalizedWeight;
            composition.landMountains += entryBiome.mountains * normalizedWeight;
            composition.landKeepOriginal += keepOriginal * normalizedWeight;
            if (normalizedWeight > composition.landRepresentativeWeight)
            {
                composition.landRepresentativeBiome = entry.seed.biomeIndex;
                composition.landRepresentativeWeight = normalizedWeight;
                composition.landSitePos = sitePos;
                composition.landSiteRadius = max(entry.seed.radius, 1.0f);
            }
        }
    }

    if (composition.landWeight > kClimateEpsilon)
    {
        composition.landHeight /= composition.landWeight;
        composition.landRoughness /= composition.landWeight;
        composition.landHills /= composition.landWeight;
        composition.landMountains /= composition.landWeight;
        composition.landKeepOriginal /= composition.landWeight;
    }
    if (composition.oceanWeight > kClimateEpsilon)
    {
        composition.oceanHeight /= composition.oceanWeight;
        composition.oceanRoughness /= composition.oceanWeight;
        composition.oceanHills /= composition.oceanWeight;
        composition.oceanMountains /= composition.oceanWeight;
        composition.oceanKeepOriginal /= composition.oceanWeight;
    }

    composition.prefersOcean = composition.oceanWeight > composition.landWeight ? 1u : 0u;
}

uint sampleSmoothedDomain(int worldX, int worldZ)
{
    int oceanCount = 0;
    uint centerRaw = 0u;

    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            ClimateComposition composition;
            accumulateClimateComposition(worldX + dx, worldZ + dz, composition);
            if (dx == 0 && dz == 0)
            {
                centerRaw = composition.prefersOcean;
            }
            oceanCount += composition.prefersOcean != 0u ? 1 : 0;
        }
    }

    if (oceanCount >= 6)
    {
        return 1u;
    }
    if (oceanCount <= 3)
    {
        return 0u;
    }
    return centerRaw;
}

void resolveClimatePointNoTransition(int worldX, int worldZ, out ClimateResolvedPoint resolvedPoint)
{
    const FarLodGpuWorldgenHeader header = gWorldgenHeader[0];

    ClimateComposition composition;
    accumulateClimateComposition(worldX, worldZ, composition);

    bool useOceanDomain = sampleSmoothedDomain(worldX, worldZ) != 0u;
    if ((useOceanDomain && composition.oceanWeight <= kClimateEpsilon)
        || (!useOceanDomain && composition.landWeight <= kClimateEpsilon))
    {
        useOceanDomain = composition.prefersOcean != 0u;
    }

    resolveRepresentativeFromComposition(composition, useOceanDomain, resolvedPoint);

    const int halo = max(header.maxTransitionWidth + 8, (int)ceil(header.coastDistanceFieldRange));
    const uint targetValue = resolvedPoint.dominantIsOcean != 0u ? 0u : 1u;
    float coastDistance = kHugeFloat;

    for (int dz = -halo; dz <= halo; ++dz)
    {
        for (int dx = -halo; dx <= halo; ++dx)
        {
            if (sampleSmoothedDomain(worldX + dx, worldZ + dz) != targetValue)
            {
                continue;
            }

            const float candidateDistance = gridDistance(dx, dz);
            if (candidateDistance < coastDistance)
            {
                coastDistance = candidateDistance;
            }
        }
    }

    if (!(coastDistance < kHugeFloat) || coastDistance > (float)halo)
    {
        resolvedPoint.distanceToCoast = kHugeFloat;
        resolvedPoint.signedDistanceToCoast = resolvedPoint.dominantIsOcean != 0u ? -kHugeFloat : kHugeFloat;
    }
    else
    {
        resolvedPoint.distanceToCoast = coastDistance;
        resolvedPoint.signedDistanceToCoast = resolvedPoint.dominantIsOcean != 0u ? -coastDistance : coastDistance;
    }
}

void applyTransitionBiomeAtPoint(int worldX, int worldZ, inout ClimateResolvedPoint resolvedPoint)
{
    const FarLodGpuWorldgenHeader header = gWorldgenHeader[0];
    if (header.maxTransitionWidth <= 0)
    {
        return;
    }

    const FarLodGpuBiome baseBiome = gBiomes[resolvedPoint.biomeIndex];
    if (baseBiome.transitionCount == 0u)
    {
        return;
    }

    float strongestTransition = 0.0f;
    const float coastDistance = resolvedPoint.distanceToCoast;

    for (uint transitionIndex = 0u; transitionIndex < baseBiome.transitionCount; ++transitionIndex)
    {
        const FarLodGpuTransitionBiome transition = gTransitionBiomes[baseBiome.transitionOffset + transitionIndex];
        const FarLodGpuBiome target = gBiomes[transition.biomeIndex];
        if ((target.flags & kFarLodBiomeBeach) != 0u)
        {
            continue;
        }

        const int radius = clamp(transition.width, 0, header.maxTransitionWidth);
        uint neighborMask = 0u;
        bool hasOceanNeighbor = false;

        for (int dz = -radius; dz <= radius; ++dz)
        {
            for (int dx = -radius; dx <= radius; ++dx)
            {
                ClimateResolvedPoint neighborPoint;
                resolveClimatePointNoTransition(worldX + dx, worldZ + dz, neighborPoint);
                neighborMask |= neighborPoint.biomePropertyBits;
                if (neighborPoint.dominantIsOcean != 0u)
                {
                    hasOceanNeighbor = true;
                }
            }
        }

        const uint requiredBits = transition.propertyBits;
        const uint matched = neighborMask & requiredBits;
        const uint spread = matched | (matched >> 1u) | (matched >> 2u);
        const uint requiredGroups = groupPresenceMask(requiredBits);
        const uint availableGroups = groupPresenceMask(spread);
        if ((availableGroups & requiredGroups) != requiredGroups)
        {
            continue;
        }

        const bool targetIsCoast = (target.propertyBits & kPropLand) != 0u && (target.propertyBits & kPropOcean) != 0u;
        const bool targetIsMountainCoast = targetIsCoast
            && (target.propertyBits & kPropMountain) != 0u
            && (target.flags & kFarLodBiomeOcean) == 0u;
        if (!isfinite(coastDistance))
        {
            continue;
        }
        if ((targetIsCoast || (target.flags & kFarLodBiomeOcean) != 0u) && !hasOceanNeighbor)
        {
            continue;
        }
        if ((target.flags & kFarLodBiomeOcean) != 0u && resolvedPoint.dominantIsOcean == 0u)
        {
            continue;
        }
        if (targetIsMountainCoast && resolvedPoint.dominantIsOcean != 0u)
        {
            continue;
        }

        uint hashSeed = hashCombine(
            header.seed,
            hashCombine((uint)worldX, hashCombine((uint)worldZ, (uint)transition.width)));

        const float transitionWidth = (float)max(transition.width, 1);
        float range = transitionWidth * 6.0f;
        if ((target.flags & kFarLodBiomeOcean) != 0u)
        {
            range = max(range, 32.0f);
        }
        else if (targetIsMountainCoast)
        {
            range = max(range, 26.0f);
        }
        else if (targetIsCoast)
        {
            range = max(range, 18.0f);
        }
        else
        {
            range = max(range, 12.0f);
        }

        const float edgeNoise = unitPerlinNoise(worldX, worldZ, hashSeed ^ 0xA53C9E21u, 0.03f);
        const float effectiveRange = range * lerp(0.85f, 1.15f, edgeNoise);
        float transitionStrength = smoothFactorFromDistance(coastDistance, effectiveRange);
        transitionStrength *= saturate(transition.chance);
        if (transitionStrength <= 0.01f)
        {
            continue;
        }

        if (transitionStrength > strongestTransition)
        {
            resolvedPoint.biomeIndex = transition.biomeIndex;
            resolvedPoint.biomeFlags = target.flags;
            resolvedPoint.biomePropertyBits = target.propertyBits;
            resolvedPoint.representativeWeight = max(resolvedPoint.representativeWeight, transitionStrength);
            resolvedPoint.dominantSitePos = float2((float)worldX, (float)worldZ);
            resolvedPoint.dominantSiteRadius = biomeMaxRadius(target);
            strongestTransition = transitionStrength;
        }
    }
}

SamplePoint samplePoint(int worldX, int worldZ)
{
    return samplePointLegacy(worldX, worldZ);
}

void resolveCenterMaterialAndWater(FarLodGpuBiome biome,
                                   int centerSurfaceY,
                                   float distanceToShore,
                                   int centerX,
                                   int centerZ,
                                   out uint surfaceBlock,
                                   out uint fillerBlock,
                                   out uint waterEnabled,
                                   out int waterBottomY)
{
    surfaceBlock = biome.surfaceBlock;
    fillerBlock = biome.fillerBlock;
    waterEnabled = 0u;
    waterBottomY = centerSurfaceY + 1;

    if ((biome.flags & kFarLodBiomeOcean) == 0u && abs(centerSurfaceY - gSeaLevel) <= 2 && distanceToShore <= 6.0f)
    {
        const float noise = hashToUnitFloat(centerX, centerSurfaceY, centerZ);
        if ((biome.flags & kFarLodBiomeSmoothBeaches) != 0u)
        {
            const float sandProbability = lerp(0.4f, 0.95f, 1.0f - saturate(distanceToShore / 6.0f));
            if (noise <= sandProbability)
            {
                surfaceBlock = kBlockSand;
                fillerBlock = kBlockSand;
            }
            else if (noise < sandProbability + 0.1f)
            {
                fillerBlock = kBlockSand;
            }
        }
        else
        {
            if (noise < 0.55f) surfaceBlock = kBlockSand;
            fillerBlock = kBlockSand;
        }
    }

    if ((biome.flags & kFarLodBiomeTaiga) != 0u && surfaceBlock != kBlockSand)
    {
        const float patchNoise = taigaPodzolNoise(centerX, centerZ);
        const float patchSelector = hashToUnitFloat(centerX, centerSurfaceY * 23 + 11, centerZ);
        if (patchNoise > 0.67f || (patchNoise > 0.59f && patchSelector > 0.45f))
        {
            surfaceBlock = kBlockPodzol;
            fillerBlock = kBlockPodzol;
        }
    }

    if ((biome.flags & kFarLodBiomeWaterFill) != 0u && centerSurfaceY < gSeaLevel)
    {
        waterEnabled = 1u;
        waterBottomY = centerSurfaceY + 1;
        if (biome.waterMaxDepth > 0)
        {
            waterBottomY = max(waterBottomY, gSeaLevel - biome.waterMaxDepth + 1);
        }
    }
}

[numthreads(8, 8, 1)]
void FarLodColumnAtlasUpdateMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= (uint)gUpdateSizeX || dispatchThreadId.y >= (uint)gUpdateSizeZ)
    {
        return;
    }

    const int2 cellCoord = int2(gUpdateOriginCellX + (int)dispatchThreadId.x,
                                gUpdateOriginCellZ + (int)dispatchThreadId.y);
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

    SamplePoint samples[9];
    for (uint sampleIndex = 0u; sampleIndex < 9u; ++sampleIndex)
    {
        samples[sampleIndex] = samplePoint(points[sampleIndex].x, points[sampleIndex].y);
    }

    int heights[9];
    for (uint heightIndex = 0u; heightIndex < 9u; ++heightIndex)
    {
        heights[heightIndex] = samples[heightIndex].surfaceY;
    }
    for (uint passIndex = 0u; passIndex < 8u; ++passIndex)
    {
        for (uint sortIndex = 0u; sortIndex < 8u - passIndex; ++sortIndex)
        {
            const int left = heights[sortIndex];
            const int right = heights[sortIndex + 1u];
            if (left > right)
            {
                heights[sortIndex] = right;
                heights[sortIndex + 1u] = left;
            }
        }
    }

    GpuTerrainAtlasSample sample = (GpuTerrainAtlasSample)0;
    sample.hasSolid = 1u;
    sample.minSurfaceY = heights[0];
    sample.maxSurfaceY = heights[8];
    sample.surfaceY = heights[4];
    sample.waterBottomY = sample.surfaceY + 1;

    const SamplePoint center = samples[4];
    const FarLodGpuBiome centerBiome = gBiomes[center.biomeIndex];
    resolveCenterMaterialAndWater(centerBiome,
                                  center.surfaceY,
                                  center.distanceToShore,
                                  centerX,
                                  centerZ,
                                  sample.surfaceBlock,
                                  sample.fillerBlock,
                                  sample.waterEnabled,
                                  sample.waterBottomY);

    uint taigaVotes = 0u;
    float densitySum = 0.0f;
    uint densityCount = 0u;
    [unroll]
    for (uint canopyIndex = 0u; canopyIndex < 9u; ++canopyIndex)
    {
        if ((samples[canopyIndex].biomeFlags & kFarLodBiomeGeneratesTrees) == 0u ||
            (samples[canopyIndex].biomeFlags & kFarLodBiomeOcean) != 0u ||
            samples[canopyIndex].surfaceY < gSeaLevel - 1)
        {
            continue;
        }
        densitySum += saturate(perlin2((float)points[canopyIndex].x * 0.05f,
                                       (float)points[canopyIndex].y * 0.05f) * 0.5f + 0.5f);
        densityCount += 1u;
        if ((samples[canopyIndex].biomeFlags & kFarLodBiomeTaiga) != 0u)
        {
            taigaVotes += 1u;
        }
    }

    if (densityCount > 0u && (center.biomeFlags & kFarLodBiomeGeneratesTrees) != 0u)
    {
        const bool taigaCanopy = taigaVotes * 2u >= densityCount;
        const float averageDensity = densitySum / (float)densityCount;
        if (averageDensity >= 0.32f)
        {
            sample.canopyBottomY = sample.maxSurfaceY + (taigaCanopy ? 4 : 3);
            sample.canopyTopY = sample.canopyBottomY + (taigaCanopy ? 8 : 6);
            sample.canopyBlock = taigaCanopy ? kBlockSpruceLeaves : kBlockLeaves;
            sample.canopyStrength = (uint)clamp(round(averageDensity * 255.0f), 0.0f, 255.0f);
        }
    }

    gAtlasSamples[atlasIndex(cellCoord)] = sample;
}
