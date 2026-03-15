cbuffer ChunkParams : register(b0)
{
    int3 gWorldMin;
    int gBlockScale;
    int gSeaLevel;
    int gAtlasOriginCellX;
    int gAtlasOriginCellZ;
    int gAtlasSizeX;
    int gAtlasSizeZ;
};

struct GpuTerrainAtlasSample
{
    uint hasSolid;
    uint waterEnabled;
    int surfaceY;
    int waterBottomY;
    uint surfaceBlock;
    uint fillerBlock;
};

struct GpuTerrainColumnDescriptor
{
    uint centerHasSolid;
    uint centerWaterEnabled;
    int centerSurfaceY;
    int centerWaterBottomY;
    uint centerSurfaceBlock;
    uint centerFillerBlock;
    uint terrainSurfaceMask;
    uint terrainFillerMask;
    uint waterMask;
    uint reserved;
};

StructuredBuffer<GpuTerrainAtlasSample> gAtlasSamples : register(t0);
RWStructuredBuffer<GpuTerrainColumnDescriptor> gColumnBuffer : register(u0);

static const uint kLogicalSize = 16u;

uint columnIndex(uint localX, uint localZ)
{
    return localZ * kLogicalSize + localX;
}

int positiveModulo(int value, int divisor)
{
    const int result = value % divisor;
    return result < 0 ? result + divisor : result;
}

int floorDiv(int value, int divisor)
{
    const int quotient = value / divisor;
    const int remainder = value % divisor;
    return (remainder != 0 && ((remainder < 0) != (divisor < 0))) ? (quotient - 1) : quotient;
}

bool atlasContainsCell(int2 cellCoord)
{
    return cellCoord.x >= gAtlasOriginCellX &&
           cellCoord.x < gAtlasOriginCellX + gAtlasSizeX &&
           cellCoord.y >= gAtlasOriginCellZ &&
           cellCoord.y < gAtlasOriginCellZ + gAtlasSizeZ;
}

uint atlasIndex(int2 cellCoord)
{
    const int atlasX = positiveModulo(cellCoord.x - gAtlasOriginCellX, gAtlasSizeX);
    const int atlasZ = positiveModulo(cellCoord.y - gAtlasOriginCellZ, gAtlasSizeZ);
    return (uint)(atlasZ * gAtlasSizeX + atlasX);
}

GpuTerrainAtlasSample sampleAtlas(int worldX, int worldZ)
{
    GpuTerrainAtlasSample sample = (GpuTerrainAtlasSample)0;
    const int2 cellCoord = int2(floorDiv(worldX, gBlockScale), floorDiv(worldZ, gBlockScale));
    if (!atlasContainsCell(cellCoord))
    {
        return sample;
    }
    return gAtlasSamples[atlasIndex(cellCoord)];
}

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
    const int2 samplePoints[5] = {
        int2(minX, minZ),
        int2(maxX, minZ),
        int2(minX, maxZ),
        int2(maxX, maxZ),
        int2(centerX, centerZ)
    };

    GpuTerrainColumnDescriptor descriptor = (GpuTerrainColumnDescriptor)0;
    int minSurfaceY = 2147483647;
    int minWaterBottomY = 2147483647;
    uint waterHitCount = 0u;
    GpuTerrainAtlasSample samples[5];

    [unroll]
    for (uint sampleIndex = 0u; sampleIndex < 5u; ++sampleIndex)
    {
        const GpuTerrainAtlasSample sample = sampleAtlas(samplePoints[sampleIndex].x, samplePoints[sampleIndex].y);
        samples[sampleIndex] = sample;
        minSurfaceY = min(minSurfaceY, sample.surfaceY);
        if (sample.waterEnabled != 0u && sample.surfaceY < gSeaLevel)
        {
            waterHitCount += 1u;
            minWaterBottomY = min(minWaterBottomY, sample.waterBottomY);
        }
        if (sampleIndex == 4u)
        {
            descriptor.centerHasSolid = sample.hasSolid;
            descriptor.centerWaterEnabled = sample.waterEnabled;
            descriptor.centerSurfaceY = sample.surfaceY;
            descriptor.centerWaterBottomY = sample.waterBottomY;
            descriptor.centerSurfaceBlock = sample.surfaceBlock;
            descriptor.centerFillerBlock = sample.fillerBlock;
        }
    }

    const GpuTerrainAtlasSample centerSample = samples[4];
    [unroll]
    for (uint localY = 0u; localY < kLogicalSize; ++localY)
    {
        const int voxelMinY = gWorldMin.y + int(localY) * gBlockScale;
        const int voxelMaxY = voxelMinY + (gBlockScale - 1);
        uint solidHitCount = 0u;
        [unroll]
        for (uint sampleIndex = 0u; sampleIndex < 5u; ++sampleIndex)
        {
            if (samples[sampleIndex].hasSolid != 0u && samples[sampleIndex].surfaceY >= voxelMinY)
            {
                solidHitCount += 1u;
            }
        }

        const uint bit = (1u << localY);
        if (centerSample.hasSolid != 0u && solidHitCount >= 3u)
        {
            if (minSurfaceY > voxelMaxY)
            {
                descriptor.terrainFillerMask |= bit;
            }
            else
            {
                descriptor.terrainSurfaceMask |= bit;
            }
        }
        else if (centerSample.waterEnabled != 0u &&
                 waterHitCount >= 3u &&
                 minWaterBottomY != 2147483647 &&
                 minWaterBottomY <= voxelMaxY &&
                 gSeaLevel >= voxelMinY)
        {
            descriptor.waterMask |= bit;
        }
    }

    gColumnBuffer[columnIndex(localX, localZ)] = descriptor;
}
