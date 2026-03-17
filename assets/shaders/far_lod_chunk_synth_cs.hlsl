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

int quantizeHeight(int y, int step)
{
    if (step <= 1)
    {
        return y;
    }

    const int halfStep = step / 2;
    if (y >= 0)
    {
        return ((y + halfStep) / step) * step;
    }
    return ((y - halfStep) / step) * step;
}

int terrainSurfaceSnapStep()
{
    return max(1, gBlockScale / 4);
}

int terrainColumnBaseY(int topY)
{
    return topY - terrainSurfaceSnapStep() + 1;
}

int stableSurfaceHeight(GpuTerrainAtlasSample center,
                        GpuTerrainAtlasSample east,
                        GpuTerrainAtlasSample west,
                        GpuTerrainAtlasSample south,
                        GpuTerrainAtlasSample north)
{
    int heights[5];
    uint count = 0u;
    heights[count++] = center.surfaceY;
    if (east.hasSolid != 0u) heights[count++] = east.surfaceY;
    if (west.hasSolid != 0u) heights[count++] = west.surfaceY;
    if (south.hasSolid != 0u) heights[count++] = south.surfaceY;
    if (north.hasSolid != 0u) heights[count++] = north.surfaceY;

    for (uint i = 1u; i < count; ++i)
    {
        const int value = heights[i];
        int insertIndex = (int)i - 1;
        while (insertIndex >= 0 && heights[insertIndex] > value)
        {
            heights[insertIndex + 1] = heights[insertIndex];
            insertIndex -= 1;
        }
        heights[insertIndex + 1] = value;
    }

    if (count == 1u)
    {
        return heights[0];
    }

    if ((count & 1u) != 0u)
    {
        return heights[count / 2u];
    }
    return (heights[count / 2u - 1u] + heights[count / 2u]) / 2;
}

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
        const int snapStep = terrainSurfaceSnapStep();
        const int quantizedMinTop = quantizeHeight(center.minSurfaceY, snapStep);
        const int quantizedMaxTop = max(quantizedMinTop, quantizeHeight(center.maxSurfaceY, snapStep));
        int sourceTopY = center.surfaceY;
        if (localRelief < snapStep * 2 && steepMetric < snapStep * 2)
        {
            sourceTopY = stableSurfaceHeight(center, east, west, south, north);
        }
        int chosenTop = clamp(quantizeHeight(sourceTopY, snapStep),
                              quantizedMinTop,
                              quantizedMaxTop);
        if (localRelief >= snapStep || steepMetric >= snapStep * 2)
        {
            chosenTop = max(chosenTop, quantizedMaxTop);
        }
        descriptor.flags |= kColumnFlagTerrain;
        if (localRelief >= snapStep || steepMetric >= snapStep * 2)
        {
            descriptor.flags |= kColumnFlagSteep;
        }
        descriptor.terrainTopY = chosenTop;
        descriptor.terrainBaseY = terrainColumnBaseY(chosenTop);
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
