#include "exact_chunk_common.hlsli"

cbuffer ExactChunkDescriptorGenParams : register(b0)
{
    int gSeaLevel;
    uint gBuildCount;
    uint gReserved0;
    uint gReserved1;
};

StructuredBuffer<GpuWorldgenPageColumn> gWorldgenPages : register(t0);
StructuredBuffer<GpuExactDescriptorBuildParams> gBuilds : register(t1);
StructuredBuffer<uint> gSkyLightFromAbove : register(t2);
RWStructuredBuffer<GpuExactColumnDescriptor> gOutColumns : register(u0);

static const uint kWorldgenFlagHasBiome = 1u << 0u;
static const uint kWorldgenFlagDominantIsOcean = 1u << 1u;
static const uint kWorldgenFlagSmoothBeaches = 1u << 2u;
static const uint kWorldgenFlagTaigaBiome = 1u << 3u;
static const uint kWorldgenFlagWaterFillEnabled = 1u << 4u;
static const uint kWorldgenFlagStripesEnabled = 1u << 5u;
static const uint kWorldgenFlagBiomeIsOcean = 1u << 6u;

float hashToUnitFloatExact(int x, int y, int z)
{
    const uint kMulX = 374761393u;
    const uint kMulY = 668265263u;
    const uint kMulZ = 2147483647u;
    const uint kMixMul = 1274126177u;
    const uint kMask24 = 0xFFFFFFu;

    uint h = uint(x) * kMulX + uint(y) * kMulY + uint(z) * kMulZ;
    h = (h ^ (h >> 13u)) * kMixMul;
    h ^= (h >> 16u);
    return float(h & kMask24) / float(kMask24);
}

float smoothStepExact(float t)
{
    t = clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float valueNoise2DExact(float x, float z, float frequency, int seed)
{
    const float sampleX = x * frequency;
    const float sampleZ = z * frequency;
    const int x0 = int(floor(sampleX));
    const int z0 = int(floor(sampleZ));
    const int x1 = x0 + 1;
    const int z1 = z0 + 1;

    const float tx = smoothStepExact(sampleX - float(x0));
    const float tz = smoothStepExact(sampleZ - float(z0));

    const float v00 = hashToUnitFloatExact(x0 + seed * 17, seed * 31, z0 - seed * 13);
    const float v10 = hashToUnitFloatExact(x1 + seed * 17, seed * 31, z0 - seed * 13);
    const float v01 = hashToUnitFloatExact(x0 + seed * 17, seed * 31, z1 - seed * 13);
    const float v11 = hashToUnitFloatExact(x1 + seed * 17, seed * 31, z1 - seed * 13);

    const float ix0 = lerp(v00, v10, tx);
    const float ix1 = lerp(v01, v11, tx);
    return lerp(ix0, ix1, tz);
}

float taigaPodzolNoiseExact(int worldX, int worldZ)
{
    const float broad = valueNoise2DExact((float)worldX, (float)worldZ, 1.0f / 16.0f, 19);
    const float medium = valueNoise2DExact((float)worldX, (float)worldZ, 1.0f / 8.0f, 37);
    const float detail = valueNoise2DExact((float)worldX, (float)worldZ, 1.0f / 4.0f, 73);
    return broad * 0.55f + medium * 0.30f + detail * 0.15f;
}

bool isFiniteFloat(float value)
{
    return !isinf(value) && !isnan(value);
}

int roundToIntExact(float value)
{
    return value >= 0.0f ? (int)floor(value + 0.5f) : (int)ceil(value - 0.5f);
}

uint pageColumnIndex(uint pageIndex, uint localX, uint localZ)
{
    return pageIndex * kWorldgenPageColumnCount + localZ * kWorldgenPageSize + localX;
}

GpuWorldgenPageColumn sampleWorldgenColumn(GpuExactDescriptorBuildParams build, int worldX, int worldZ)
{
    const int splitWorldX = build.sampleMinPageBaseWorldX + int(kWorldgenPageSize);
    const int splitWorldZ = build.sampleMinPageBaseWorldZ + int(kWorldgenPageSize);
    const bool useHighPageX = worldX >= splitWorldX;
    const bool useHighPageZ = worldZ >= splitWorldZ;

    uint pageIndex = build.pageIndex00;
    int pageBaseWorldX = build.sampleMinPageBaseWorldX;
    int pageBaseWorldZ = build.sampleMinPageBaseWorldZ;
    if (useHighPageX && useHighPageZ)
    {
        pageIndex = build.pageIndex11;
        pageBaseWorldX += int(kWorldgenPageSize);
        pageBaseWorldZ += int(kWorldgenPageSize);
    }
    else if (useHighPageX)
    {
        pageIndex = build.pageIndex10;
        pageBaseWorldX += int(kWorldgenPageSize);
    }
    else if (useHighPageZ)
    {
        pageIndex = build.pageIndex01;
        pageBaseWorldZ += int(kWorldgenPageSize);
    }

    const uint localX = (uint)(worldX - pageBaseWorldX);
    const uint localZ = (uint)(worldZ - pageBaseWorldZ);
    return gWorldgenPages[pageColumnIndex(pageIndex, localX, localZ)];
}

uint unpackFlags(GpuWorldgenPageColumn column)
{
    return column.packedFlagsTintWaterDepth & 0xFFu;
}

uint unpackGrassTint(GpuWorldgenPageColumn column)
{
    return (column.packedFlagsTintWaterDepth >> 8u) & 0xFFu;
}

uint unpackWaterFillMaxDepth(GpuWorldgenPageColumn column)
{
    return (column.packedFlagsTintWaterDepth >> 16u) & 0xFFFFu;
}

uint unpackSoilCreepMaxStep(GpuWorldgenPageColumn column)
{
    return column.packedSoilDepths & 0xFFFFu;
}

uint unpackSoilCreepMaxDepth(GpuWorldgenPageColumn column)
{
    return (column.packedSoilDepths >> 16u) & 0xFFFFu;
}

uint unpackStripePeriod(GpuWorldgenPageColumn column)
{
    return column.packedStripes & 0xFFFFu;
}

uint unpackStripeThickness(GpuWorldgenPageColumn column)
{
    return (column.packedStripes >> 16u) & 0xFFFFu;
}

uint unpackSurfaceBlock(GpuWorldgenPageColumn column)
{
    return column.packedBlocks & 0xFFu;
}

uint unpackFillerBlock(GpuWorldgenPageColumn column)
{
    return (column.packedBlocks >> 8u) & 0xFFu;
}

uint unpackWaterBlock(GpuWorldgenPageColumn column)
{
    return (column.packedBlocks >> 16u) & 0xFFu;
}

uint unpackStripeBlock(GpuWorldgenPageColumn column)
{
    return (column.packedBlocks >> 24u) & 0xFFu;
}

[numthreads(8, 8, 1)]
void ExactChunkDescriptorGenMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kExactChunkSize ||
        dispatchThreadId.y >= kExactChunkSize ||
        dispatchThreadId.z >= gBuildCount)
    {
        return;
    }

    const uint localX = dispatchThreadId.x;
    const uint localZ = dispatchThreadId.y;
    const uint buildIndex = dispatchThreadId.z;
    const GpuExactDescriptorBuildParams build = gBuilds[buildIndex];

    const int worldX = build.chunkBaseWorldX + int(localX);
    const int worldZ = build.chunkBaseWorldZ + int(localZ);
    const GpuWorldgenPageColumn centerColumn = sampleWorldgenColumn(build, worldX, worldZ);
    const uint worldgenFlags = unpackFlags(centerColumn);

    GpuExactColumnDescriptor descriptor;
    descriptor.surfaceY = centerColumn.surfaceY;
    descriptor.highestSolidWorld = -2147483647 - 1;
    descriptor.waterTopWorld = -2147483647 - 1;
    descriptor.waterBottomWorld = 2147483647;
    descriptor.stripeOffset = 0;
    descriptor.flags = 0u;
    descriptor.stripePeriod = 0u;
    descriptor.stripeThickness = 0u;
    descriptor.grassTintIndex = unpackGrassTint(centerColumn);
    descriptor.surfaceBlock = unpackSurfaceBlock(centerColumn);
    descriptor.fillerBlock = unpackFillerBlock(centerColumn);
    descriptor.waterBlock = unpackWaterBlock(centerColumn);
    descriptor.stripeBlock = unpackStripeBlock(centerColumn);
    descriptor.skyLightFromAbove = gSkyLightFromAbove[build.skyLightOffset + columnIndex(localX, localZ)];
    descriptor.reserved1 = 0u;
    descriptor.reserved2 = 0u;

    if ((worldgenFlags & kWorldgenFlagHasBiome) == 0u)
    {
        gOutColumns[build.descriptorOffset + columnIndex(localX, localZ)] = descriptor;
        return;
    }

    descriptor.flags |= 0x01u;
    if ((worldgenFlags & kWorldgenFlagDominantIsOcean) != 0u)
    {
        descriptor.flags |= 0x20u;
    }

    float neighborSum = 0.0f;
    uint neighborCount = 0u;
    [unroll]
    for (int dx = -1; dx <= 1; ++dx)
    {
        [unroll]
        for (int dz = -1; dz <= 1; ++dz)
        {
            if (dx == 0 && dz == 0)
            {
                continue;
            }

            neighborSum += (float)sampleWorldgenColumn(build, worldX + dx, worldZ + dz).surfaceY;
            ++neighborCount;
        }
    }
    const float neighborAverage = neighborCount > 0u ? neighborSum / (float)neighborCount : 0.0f;

    int adjustedSurfaceY = centerColumn.surfaceY;
    if (centerColumn.soilCreepStrength > 0.0f)
    {
        float offset = (neighborAverage - (float)adjustedSurfaceY) * centerColumn.soilCreepStrength;
        const uint maxStep = unpackSoilCreepMaxStep(centerColumn);
        if (maxStep > 0u)
        {
            offset = clamp(offset, -(float)maxStep, (float)maxStep);
        }
        const uint maxDepth = unpackSoilCreepMaxDepth(centerColumn);
        if (maxDepth > 0u)
        {
            offset = clamp(offset, -(float)maxDepth, (float)maxDepth);
        }
        adjustedSurfaceY = roundToIntExact((float)adjustedSurfaceY + offset);
        adjustedSurfaceY = clamp(adjustedSurfaceY,
                                 min(centerColumn.surfaceY, build.chunkMinWorldY),
                                 max(centerColumn.surfaceY, build.chunkMinWorldY + int(kExactChunkSize) - 1));
    }

    descriptor.surfaceY = adjustedSurfaceY;

    const int chunkMaxWorldY = build.chunkMinWorldY + int(kExactChunkSize) - 1;
    const bool slabHasSolid = adjustedSurfaceY >= build.chunkMinWorldY;
    if (slabHasSolid)
    {
        descriptor.flags |= 0x02u;
        descriptor.highestSolidWorld = min(adjustedSurfaceY, chunkMaxWorldY);
    }

    if ((worldgenFlags & kWorldgenFlagWaterFillEnabled) != 0u && adjustedSurfaceY < gSeaLevel)
    {
        int waterBottomWorld = max(adjustedSurfaceY + 1, build.chunkMinWorldY);
        const int waterTopWorld = min(gSeaLevel, chunkMaxWorldY);
        const uint maxDepth = unpackWaterFillMaxDepth(centerColumn);
        if (maxDepth > 0u)
        {
            waterBottomWorld = max(waterBottomWorld, waterTopWorld - int(maxDepth) + 1);
        }
        if (waterBottomWorld <= waterTopWorld)
        {
            descriptor.flags |= 0x04u;
            descriptor.waterTopWorld = waterTopWorld;
            descriptor.waterBottomWorld = waterBottomWorld;
        }
    }

    if ((descriptor.flags & 0x06u) != 0u)
    {
        float distanceToShore = centerColumn.distanceToCoast;
        if (!isFiniteFloat(distanceToShore) && (worldgenFlags & kWorldgenFlagBiomeIsOcean) != 0u)
        {
            distanceToShore = 0.0f;
        }

        const bool nearSeaLevel = abs(adjustedSurfaceY - gSeaLevel) <= 2;
        if ((worldgenFlags & kWorldgenFlagBiomeIsOcean) == 0u &&
            nearSeaLevel &&
            isFiniteFloat(distanceToShore) &&
            distanceToShore <= 6.0f)
        {
            const float noise = hashToUnitFloatExact(worldX, adjustedSurfaceY, worldZ);
            if ((worldgenFlags & kWorldgenFlagSmoothBeaches) != 0u)
            {
                const float shorelineWeight = 1.0f - clamp(distanceToShore / 6.0f, 0.0f, 1.0f);
                const float sandProbability = lerp(0.4f, 0.95f, shorelineWeight);
                if (noise <= sandProbability)
                {
                    descriptor.surfaceBlock = kBlockSand;
                    descriptor.fillerBlock = kBlockSand;
                }
                else if (noise < sandProbability + 0.1f)
                {
                    descriptor.fillerBlock = kBlockSand;
                }
            }
            else
            {
                if (noise < 0.55f)
                {
                    descriptor.surfaceBlock = kBlockSand;
                }
                descriptor.fillerBlock = kBlockSand;
            }
        }

        if ((worldgenFlags & kWorldgenFlagTaigaBiome) != 0u && descriptor.surfaceBlock != kBlockSand)
        {
            const float patchNoise = taigaPodzolNoiseExact(worldX, worldZ);
            const float patchSelector = hashToUnitFloatExact(worldX, adjustedSurfaceY * 23 + 11, worldZ);
            const bool usePodzol = patchNoise > 0.67f || (patchNoise > 0.59f && patchSelector > 0.45f);
            if (usePodzol)
            {
                descriptor.surfaceBlock = kBlockPodzol;
                descriptor.fillerBlock = kBlockPodzol;
            }
        }
    }

    const uint stripePeriodSetting = unpackStripePeriod(centerColumn);
    const uint stripeThicknessSetting = unpackStripeThickness(centerColumn);
    const bool stripesEnabled =
        slabHasSolid &&
        (worldgenFlags & kWorldgenFlagStripesEnabled) != 0u &&
        stripePeriodSetting > 0u &&
        stripeThicknessSetting > 0u;
    if (stripesEnabled)
    {
        descriptor.flags |= 0x08u;
        descriptor.stripePeriod = max(stripePeriodSetting, stripeThicknessSetting);
        descriptor.stripeThickness = stripeThicknessSetting;
        descriptor.stripeOffset =
            (int)(hashToUnitFloatExact(worldX, adjustedSurfaceY * 31 + 7, worldZ) * (float)descriptor.stripePeriod);
        if (hashToUnitFloatExact(worldX, adjustedSurfaceY * 17 + 3, worldZ) > centerColumn.stripeNoiseThreshold)
        {
            descriptor.flags |= 0x10u;
        }
    }

    gOutColumns[build.descriptorOffset + columnIndex(localX, localZ)] = descriptor;
}
