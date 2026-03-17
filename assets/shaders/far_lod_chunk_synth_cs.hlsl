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
    int minSurfaceY;
    int maxSurfaceY;
    uint surfaceBlock;
    uint fillerBlock;
    int canopyBottomY;
    int canopyTopY;
    uint canopyBlock;
    uint canopyStrength;
};

struct GpuTerrainColumnDescriptor
{
    uint flags;
    int terrainTopY;
    int terrainBaseY;
    int waterTopY;
    int waterBottomY;
    int canopyTopY;
    int canopyBottomY;
    uint terrainTopBlock;
    uint terrainSideBlock;
    uint waterBlock;
    uint canopyBlock;
    uint reserved;
};

StructuredBuffer<GpuTerrainAtlasSample> gAtlasSamples : register(t0);
RWStructuredBuffer<GpuTerrainColumnDescriptor> gColumnBuffer : register(u0);

static const uint kLogicalSize = 16u;
static const uint kColumnFlagTerrain = 0x01u;
static const uint kColumnFlagWater = 0x02u;
static const uint kColumnFlagCanopy = 0x04u;
static const uint kColumnFlagSteep = 0x08u;
static const uint kBlockAir = 0u;
static const uint kBlockWater = 5u;

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

GpuTerrainAtlasSample sampleAtlasCell(int2 cellCoord)
{
    GpuTerrainAtlasSample sample = (GpuTerrainAtlasSample)0;
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
    const int2 centerCell = int2(floorDiv(centerX, gBlockScale), floorDiv(centerZ, gBlockScale));

    const GpuTerrainAtlasSample center = sampleAtlasCell(centerCell);
    const GpuTerrainAtlasSample east = sampleAtlasCell(centerCell + int2(1, 0));
    const GpuTerrainAtlasSample west = sampleAtlasCell(centerCell + int2(-1, 0));
    const GpuTerrainAtlasSample south = sampleAtlasCell(centerCell + int2(0, 1));
    const GpuTerrainAtlasSample north = sampleAtlasCell(centerCell + int2(0, -1));

    GpuTerrainColumnDescriptor descriptor = (GpuTerrainColumnDescriptor)0;
    descriptor.terrainTopBlock = kBlockAir;
    descriptor.terrainSideBlock = kBlockAir;
    descriptor.waterBlock = kBlockWater;
    descriptor.canopyBlock = kBlockAir;

    if (center.hasSolid != 0u)
    {
        const int localRelief = max(center.maxSurfaceY - center.minSurfaceY, 0);
        int neighborDelta = 0;
        if (east.hasSolid != 0u)
        {
            neighborDelta = max(neighborDelta, max(abs(center.maxSurfaceY - east.maxSurfaceY), abs(center.surfaceY - east.surfaceY)));
        }
        if (west.hasSolid != 0u)
        {
            neighborDelta = max(neighborDelta, max(abs(center.maxSurfaceY - west.maxSurfaceY), abs(center.surfaceY - west.surfaceY)));
        }
        if (south.hasSolid != 0u)
        {
            neighborDelta = max(neighborDelta, max(abs(center.maxSurfaceY - south.maxSurfaceY), abs(center.surfaceY - south.surfaceY)));
        }
        if (north.hasSolid != 0u)
        {
            neighborDelta = max(neighborDelta, max(abs(center.maxSurfaceY - north.maxSurfaceY), abs(center.surfaceY - north.surfaceY)));
        }

        const int steepMetric = max(localRelief, neighborDelta);
        const float preserveT = saturate((float)(steepMetric - gBlockScale) / (float)max(gBlockScale * 4, 1));
        descriptor.flags |= kColumnFlagTerrain;
        if (steepMetric >= gBlockScale)
        {
            descriptor.flags |= kColumnFlagSteep;
        }
        descriptor.terrainTopY = max(center.surfaceY, (int)round(lerp((float)center.surfaceY, (float)center.maxSurfaceY, preserveT)));
        descriptor.terrainBaseY = (steepMetric >= gBlockScale) ? min(center.minSurfaceY, center.surfaceY) : center.surfaceY;
        descriptor.terrainTopBlock = center.surfaceBlock;
        descriptor.terrainSideBlock = (center.fillerBlock != 0u) ? center.fillerBlock : center.surfaceBlock;
    }

    if ((center.waterEnabled > 0u) &&
        ((descriptor.flags & kColumnFlagTerrain) != 0u) &&
        (gSeaLevel > descriptor.terrainTopY))
    {
        descriptor.flags |= kColumnFlagWater;
        descriptor.waterTopY = gSeaLevel;
        descriptor.waterBottomY = min(center.waterBottomY, descriptor.waterTopY);
    }

    if (center.canopyStrength >= 64u && center.canopyTopY > center.canopyBottomY)
    {
        const int terrainCap = ((descriptor.flags & kColumnFlagTerrain) != 0u) ? descriptor.terrainTopY : center.surfaceY;
        const int canopyBottom = max(center.canopyBottomY, terrainCap + 1);
        if (center.canopyTopY > canopyBottom)
        {
            descriptor.flags |= kColumnFlagCanopy;
            descriptor.canopyBottomY = canopyBottom;
            descriptor.canopyTopY = center.canopyTopY;
            descriptor.canopyBlock = center.canopyBlock;
        }
    }

    gColumnBuffer[columnIndex(localX, localZ)] = descriptor;
}
