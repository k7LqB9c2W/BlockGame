cbuffer ChunkParams : register(b0)
{
    int3 gWorldMin;
    int gBlockScale;
    int gSeaLevel;
};

static const uint kLogicalSize = 16u;
static const uint kVoxelCount = kLogicalSize * kLogicalSize * kLogicalSize;
static const uint kBlockAir = 0u;
static const uint kBlockGrass = 1u;
static const uint kBlockSand = 4u;
static const uint kBlockWater = 5u;
static const uint kBlockStone = 6u;
static const uint kFlagWater = 0x01u;
static const uint kFlagTerrain = 0x08u;

RWStructuredBuffer<uint> gVoxelBuffer : register(u0);

uint voxelIndex(uint3 localCoord)
{
    return (localCoord.y * kLogicalSize + localCoord.z) * kLogicalSize + localCoord.x;
}

uint packVoxel(bool occupied, uint material, uint flags)
{
    uint packed = occupied ? 1u : 0u;
    if ((flags & 0x01u) != 0u) packed |= 0x2u;
    if ((flags & 0x02u) != 0u) packed |= 0x4u;
    if ((flags & 0x04u) != 0u) packed |= 0x8u;
    if ((flags & 0x08u) != 0u) packed |= 0x10u;
    packed |= ((material & 0xffu) << 8u);
    return packed;
}

float hash11(float n)
{
    return frac(sin(n) * 43758.5453123f);
}

float hash21(float2 p)
{
    return frac(sin(dot(p, float2(127.1f, 311.7f))) * 43758.5453123f);
}

float2 hash22(float2 p)
{
    return float2(
        hash21(p + float2(17.0f, 3.0f)),
        hash21(p + float2(29.0f, 11.0f)));
}

float fade2(float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float grad2(float2 lattice, float2 f)
{
    const float angle = hash21(lattice) * 6.28318530718f;
    const float2 g = float2(cos(angle), sin(angle));
    return dot(g, f);
}

float perlin2(float2 p)
{
    const float2 i = floor(p);
    const float2 f = frac(p);

    const float a = grad2(i + float2(0.0f, 0.0f), f - float2(0.0f, 0.0f));
    const float b = grad2(i + float2(1.0f, 0.0f), f - float2(1.0f, 0.0f));
    const float c = grad2(i + float2(0.0f, 1.0f), f - float2(0.0f, 1.0f));
    const float d = grad2(i + float2(1.0f, 1.0f), f - float2(1.0f, 1.0f));

    const float2 u = float2(fade2(f.x), fade2(f.y));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

float fbm2(float2 p, int octaves, float persistence, float lacunarity)
{
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float amplitudeSum = 0.0f;
    [unroll]
    for (int i = 0; i < 6; ++i)
    {
        if (i >= octaves)
        {
            break;
        }
        value += perlin2(p * frequency) * amplitude;
        amplitudeSum += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return amplitudeSum > 0.0f ? value / amplitudeSum : 0.0f;
}

float ridge2(float2 p, int octaves, float lacunarity, float gain)
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    float prev = 1.0f;
    [unroll]
    for (int i = 0; i < 5; ++i)
    {
        if (i >= octaves)
        {
            break;
        }
        float n = 1.0f - abs(perlin2(p * frequency));
        n *= n;
        sum += n * amplitude * prev;
        prev = n * gain;
        frequency *= lacunarity;
        amplitude *= 0.5f;
    }
    return sum;
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

    float2 pos = float2((float)worldX, (float)worldZ);
    const float2 warp = (hash22(pos * 0.0021f) * 2.0f - 1.0f) * 18.0f;
    pos += warp;

    const float continental = fbm2(pos * 0.00065f, 5, 0.5f, 2.0f);
    const float hills = fbm2(pos * 0.0024f, 4, 0.5f, 2.0f);
    const float detail = fbm2(pos * 0.0105f, 3, 0.55f, 2.0f);
    const float mountains = ridge2(pos * 0.00135f, 4, 2.1f, 1.8f);
    const float biomeNoise = fbm2(pos * 0.0011f + 17.0f, 3, 0.5f, 2.0f);

    const float baseHeight = (float)gSeaLevel +
                             continental * 44.0f +
                             hills * 14.0f +
                             detail * 5.0f +
                             mountains * 36.0f;
    sample.surfaceY = (int)floor(baseHeight);
    sample.waterEnabled = true;
    sample.waterBottomY = gSeaLevel - 31;
    sample.hasSolid = true;

    if (sample.surfaceY <= gSeaLevel + 1)
    {
        sample.surfaceBlock = kBlockSand;
        sample.fillerBlock = kBlockSand;
    }
    else if (sample.surfaceY >= gSeaLevel + 34 || biomeNoise < -0.18f)
    {
        sample.surfaceBlock = kBlockStone;
        sample.fillerBlock = kBlockStone;
    }
    else
    {
        sample.surfaceBlock = kBlockGrass;
        sample.fillerBlock = kBlockStone;
    }

    return sample;
}

[numthreads(4, 4, 4)]
void FarLodChunkSynthMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kLogicalSize || dispatchThreadId.y >= kLogicalSize || dispatchThreadId.z >= kLogicalSize)
    {
        return;
    }

    const uint flatVoxelIndex = voxelIndex(dispatchThreadId);
    const int3 voxelMin = gWorldMin + int3(dispatchThreadId) * gBlockScale;
    const int3 voxelMax = voxelMin + (gBlockScale - 1);

    const int minX = voxelMin.x;
    const int maxX = voxelMax.x;
    const int minZ = voxelMin.z;
    const int maxZ = voxelMax.z;
    const int centerX = (minX + maxX) / 2;
    const int centerZ = (minZ + maxZ) / 2;

    const int2 samplePoints[5] = {
        int2(minX, minZ),
        int2(maxX, minZ),
        int2(minX, maxZ),
        int2(maxX, maxZ),
        int2(centerX, centerZ)
    };

    TerrainPointSample centerSample = (TerrainPointSample)0;
    int minSurfaceY = 2147483647;
    int maxSurfaceY = -2147483647;
    int minWaterBottomY = 2147483647;
    uint solidHitCount = 0u;
    uint waterHitCount = 0u;

    [unroll]
    for (uint sampleIndex = 0u; sampleIndex < 5u; ++sampleIndex)
    {
        const TerrainPointSample sample = sampleTerrainPoint(samplePoints[sampleIndex].x, samplePoints[sampleIndex].y);
        minSurfaceY = min(minSurfaceY, sample.surfaceY);
        maxSurfaceY = max(maxSurfaceY, sample.surfaceY);
        if (sample.hasSolid && sample.surfaceY >= voxelMin.y)
        {
            solidHitCount += 1u;
        }
        if (sample.waterEnabled && sample.surfaceY < gSeaLevel)
        {
            waterHitCount += 1u;
            minWaterBottomY = min(minWaterBottomY, sample.waterBottomY);
        }
        if (sampleIndex == 4u)
        {
            centerSample = sample;
        }
    }

    uint packed = packVoxel(false, kBlockAir, 0u);
    if (centerSample.hasSolid && solidHitCount >= 3u)
    {
        const uint material = (minSurfaceY > voxelMax.y) ? centerSample.fillerBlock : centerSample.surfaceBlock;
        packed = packVoxel(true, material, kFlagTerrain);
    }
    else if (centerSample.waterEnabled &&
             waterHitCount >= 3u &&
             minWaterBottomY <= voxelMax.y &&
             gSeaLevel >= voxelMin.y)
    {
        packed = packVoxel(true, kBlockWater, kFlagWater);
    }

    gVoxelBuffer[flatVoxelIndex] = packed;
}
