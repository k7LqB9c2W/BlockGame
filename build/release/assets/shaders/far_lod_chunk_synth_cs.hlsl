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
    uint waterEnabled; // Aggregated water presence votes within this cell (0..N).
    int surfaceY;
    int waterBottomY;
    int minSurfaceY;
    int maxSurfaceY;
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
    const int minZ = gWorldMin.z + int(localZ) * gBlockScale;
    const int centerX = minX + gBlockScale / 2;
    const int centerZ = minZ + gBlockScale / 2;

    const GpuTerrainAtlasSample cellSample = sampleAtlas(centerX, centerZ);

    GpuTerrainColumnDescriptor descriptor = (GpuTerrainColumnDescriptor)0;
    descriptor.centerHasSolid = cellSample.hasSolid;
    descriptor.centerWaterEnabled = cellSample.waterEnabled;
    descriptor.centerSurfaceY = cellSample.surfaceY;
    descriptor.centerWaterBottomY = cellSample.waterBottomY;
    descriptor.centerSurfaceBlock = cellSample.surfaceBlock;
    descriptor.centerFillerBlock = cellSample.fillerBlock;

    const int minSurfaceY = cellSample.minSurfaceY;
    const int maxSurfaceY = cellSample.maxSurfaceY;
    const int waterBottomY = cellSample.waterBottomY;
    const uint waterVotes = cellSample.waterEnabled;

    [unroll]
    for (uint localY = 0u; localY < kLogicalSize; ++localY)
    {
        const int voxelMinY = gWorldMin.y + int(localY) * gBlockScale;
        const int voxelMaxY = voxelMinY + (gBlockScale - 1);
        const uint bit = (1u << localY);

        // Conservative solid coverage: if any part of the represented footprint reaches into this band,
        // treat the far voxel as occupied to avoid undercut holes on slopes.
        if (cellSample.hasSolid != 0u && maxSurfaceY >= voxelMinY)
        {
            // If even the minimum surface is above this band, it is definitely interior filler.
            if (minSurfaceY > voxelMaxY || cellSample.surfaceY > voxelMaxY)
            {
                descriptor.terrainFillerMask |= bit;
            }
            else
            {
                descriptor.terrainSurfaceMask |= bit;
            }
            continue;
        }

        // Aggregated water presence: require a majority of sampled footprint points to be below sea level.
        if (waterVotes >= 3u &&
            waterBottomY <= voxelMaxY &&
            gSeaLevel >= voxelMinY)
        {
            descriptor.waterMask |= bit;
        }
    }

    gColumnBuffer[columnIndex(localX, localZ)] = descriptor;
}
