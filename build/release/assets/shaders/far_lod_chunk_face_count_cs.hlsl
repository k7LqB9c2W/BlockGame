cbuffer FaceCountParams : register(b0)
{
    int gWorldMinY;
    int gBlockScale;
    uint gNegativeNeighborMask;
    uint gMaxMergeExtent;
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

StructuredBuffer<GpuTerrainColumnDescriptor> gColumnBuffer : register(t0);
StructuredBuffer<GpuTerrainColumnDescriptor> gNeighborPosX : register(t1);
StructuredBuffer<GpuTerrainColumnDescriptor> gNeighborPosY : register(t2);
StructuredBuffer<GpuTerrainColumnDescriptor> gNeighborPosZ : register(t3);
RWStructuredBuffer<uint> gFaceCounts : register(u0);

static const uint kLogicalSize = 16u;
static const uint kNegativeNeighborX = 0x1u;
static const uint kNegativeNeighborZ = 0x4u;
static const uint kColumnFlagTerrain = 0x01u;
static const uint kColumnFlagWater = 0x02u;
static const uint kColumnFlagCanopy = 0x04u;
static const uint kMaterialFlagWater = 0x01u;
static const uint kMaterialFlagFarLod = 0x02u;
static const uint kLayerTerrain = 0u;
static const uint kLayerWater = 1u;
static const uint kLayerCanopy = 2u;
static const uint kTopPlaneCount = 3u;
static const uint kSideSlicesPerLayer = 64u;
static const uint kPlaneCount = kTopPlaneCount + 3u * kSideSlicesPerLayer;

uint columnIndex(uint x, uint z)
{
    return z * kLogicalSize + x;
}

GpuTerrainColumnDescriptor sampleLocal(uint x, uint z)
{
    return gColumnBuffer[columnIndex(x, z)];
}

bool layerActive(GpuTerrainColumnDescriptor column, uint layerId)
{
    if (layerId == kLayerTerrain)
    {
        return (column.flags & kColumnFlagTerrain) != 0u;
    }
    if (layerId == kLayerWater)
    {
        return (column.flags & kColumnFlagWater) != 0u;
    }
    return (column.flags & kColumnFlagCanopy) != 0u;
}

uint layerTopBlock(GpuTerrainColumnDescriptor column, uint layerId)
{
    if (layerId == kLayerTerrain) return column.terrainTopBlock;
    if (layerId == kLayerWater) return column.waterBlock;
    return column.canopyBlock;
}

uint layerSideBlock(GpuTerrainColumnDescriptor column, uint layerId)
{
    if (layerId == kLayerTerrain) return column.terrainSideBlock;
    if (layerId == kLayerWater) return column.waterBlock;
    return column.canopyBlock;
}

int layerTopY(GpuTerrainColumnDescriptor column, uint layerId)
{
    if (layerId == kLayerTerrain) return column.terrainTopY;
    if (layerId == kLayerWater) return column.waterTopY;
    return column.canopyTopY;
}

int layerBottomY(GpuTerrainColumnDescriptor column, uint layerId)
{
    if (layerId == kLayerTerrain) return column.terrainBaseY;
    if (layerId == kLayerWater) return column.waterBottomY;
    return column.canopyBottomY;
}

uint layerMaterialFlags(uint layerId)
{
    return kMaterialFlagFarLod | ((layerId == kLayerWater) ? kMaterialFlagWater : 0u);
}

bool topFaceVisible(GpuTerrainColumnDescriptor column, uint layerId)
{
    if (!layerActive(column, layerId))
    {
        return false;
    }
    if (layerId == kLayerTerrain &&
        (column.flags & kColumnFlagWater) != 0u &&
        column.waterTopY > column.terrainTopY)
    {
        return false;
    }
    const int topY = layerTopY(column, layerId);
    const int chunkMaxY = gWorldMinY + int(kLogicalSize) * gBlockScale;
    return topY >= gWorldMinY && topY < chunkMaxY;
}

uint hashTopKey(GpuTerrainColumnDescriptor column, uint layerId)
{
    if (!topFaceVisible(column, layerId))
    {
        return 0u;
    }

    uint h = 2166136261u;
    h = (h ^ layerTopBlock(column, layerId)) * 16777619u;
    h = (h ^ layerMaterialFlags(layerId)) * 16777619u;
    h = (h ^ asuint(layerTopY(column, layerId))) * 16777619u;
    return h | 1u;
}

bool sideSegment(GpuTerrainColumnDescriptor current,
                 GpuTerrainColumnDescriptor neighbor,
                 uint layerId,
                 out int segmentBottom,
                 out int segmentTopExclusive)
{
    segmentBottom = 0;
    segmentTopExclusive = 0;
    if (!layerActive(current, layerId))
    {
        return false;
    }

    const int currentTop = layerTopY(current, layerId);
    const int currentBottom = layerBottomY(current, layerId);
    int occluderTop = currentBottom - 1;
    if (layerActive(neighbor, layerId))
    {
        occluderTop = layerTopY(neighbor, layerId);
    }

    segmentBottom = max(currentBottom, occluderTop + 1);
    segmentTopExclusive = currentTop + 1;
    segmentBottom = max(segmentBottom, gWorldMinY);
    segmentTopExclusive = min(segmentTopExclusive, gWorldMinY + int(kLogicalSize) * gBlockScale);
    return segmentTopExclusive > segmentBottom;
}

uint hashSideKey(GpuTerrainColumnDescriptor current,
                 GpuTerrainColumnDescriptor neighbor,
                 uint layerId,
                 out bool visible)
{
    visible = false;
    int segmentBottom = 0;
    int segmentTopExclusive = 0;
    if (!sideSegment(current, neighbor, layerId, segmentBottom, segmentTopExclusive))
    {
        return 0u;
    }

    uint h = 2166136261u;
    h = (h ^ layerSideBlock(current, layerId)) * 16777619u;
    h = (h ^ layerMaterialFlags(layerId)) * 16777619u;
    h = (h ^ asuint(segmentBottom)) * 16777619u;
    h = (h ^ asuint(segmentTopExclusive)) * 16777619u;
    visible = true;
    return h | 1u;
}

void decodePlane(uint planeIndex, out uint layerId, out bool isTopPlane, out uint dirId, out uint slice)
{
    if (planeIndex < kTopPlaneCount)
    {
        layerId = planeIndex;
        isTopPlane = true;
        dirId = 0u;
        slice = 0u;
        return;
    }

    const uint rem = planeIndex - kTopPlaneCount;
    layerId = rem / kSideSlicesPerLayer;
    const uint sideIndex = rem - layerId * kSideSlicesPerLayer;
    dirId = sideIndex / kLogicalSize;
    slice = sideIndex - dirId * kLogicalSize;
    isTopPlane = false;
}

uint countTopQuads(uint layerId)
{
    const uint maxExtent = clamp(gMaxMergeExtent, 1u, 16u);
    uint keys[256];
    uint visitedMask[16];
    for (uint row = 0u; row < 16u; ++row)
    {
        visitedMask[row] = 0u;
    }

    for (uint fillZ = 0u; fillZ < 16u; ++fillZ)
    {
        for (uint x = 0u; x < 16u; ++x)
        {
            const GpuTerrainColumnDescriptor column = sampleLocal(x, fillZ);
            keys[fillZ * 16u + x] = hashTopKey(column, layerId);
        }
    }

    uint quadCount = 0u;
    for (uint z = 0u; z < 16u; ++z)
    {
        for (uint x = 0u; x < 16u; ++x)
        {
            const uint bit = 1u << x;
            if ((visitedMask[z] & bit) != 0u)
            {
                continue;
            }

            const uint key = keys[z * 16u + x];
            if (key == 0u)
            {
                visitedMask[z] |= bit;
                continue;
            }

            uint width = 1u;
            for (uint nx = x + 1u; nx < 16u && width < maxExtent; ++nx)
            {
                const uint testBit = 1u << nx;
                if ((visitedMask[z] & testBit) != 0u || keys[z * 16u + nx] != key)
                {
                    break;
                }
                width += 1u;
            }

            uint height = 1u;
            for (uint nz = z + 1u; nz < 16u && height < maxExtent; ++nz)
            {
                bool rowOk = true;
                for (uint dx = 0u; dx < width; ++dx)
                {
                    const uint xTest = x + dx;
                    const uint testBit = 1u << xTest;
                    if ((visitedMask[nz] & testBit) != 0u || keys[nz * 16u + xTest] != key)
                    {
                        rowOk = false;
                        break;
                    }
                }
                if (!rowOk)
                {
                    break;
                }
                height += 1u;
            }

            const uint rowMask = ((1u << width) - 1u) << x;
            for (uint dz = 0u; dz < height; ++dz)
            {
                visitedMask[z + dz] |= rowMask;
            }

            quadCount += 1u;
        }
    }
    return quadCount;
}

uint countSideRuns(uint layerId, uint dirId, uint slice)
{
    uint keys[16];
    [unroll]
    for (uint i = 0u; i < 16u; ++i)
    {
        GpuTerrainColumnDescriptor current = (GpuTerrainColumnDescriptor)0;
        GpuTerrainColumnDescriptor neighbor = (GpuTerrainColumnDescriptor)0;
        bool blockedByNegativeNeighbor = false;

        if (dirId == 0u)
        {
            current = sampleLocal(slice, i);
            if (slice + 1u < 16u)
            {
                neighbor = sampleLocal(slice + 1u, i);
            }
            else
            {
                neighbor = gNeighborPosX[columnIndex(0u, i)];
            }
        }
        else if (dirId == 1u)
        {
            current = sampleLocal(slice, i);
            blockedByNegativeNeighbor = (slice == 0u) && ((gNegativeNeighborMask & kNegativeNeighborX) != 0u);
            if (slice > 0u)
            {
                neighbor = sampleLocal(slice - 1u, i);
            }
        }
        else if (dirId == 2u)
        {
            current = sampleLocal(i, slice);
            if (slice + 1u < 16u)
            {
                neighbor = sampleLocal(i, slice + 1u);
            }
            else
            {
                neighbor = gNeighborPosZ[columnIndex(i, 0u)];
            }
        }
        else
        {
            current = sampleLocal(i, slice);
            blockedByNegativeNeighbor = (slice == 0u) && ((gNegativeNeighborMask & kNegativeNeighborZ) != 0u);
            if (slice > 0u)
            {
                neighbor = sampleLocal(i, slice - 1u);
            }
        }

        bool visible = false;
        keys[i] = blockedByNegativeNeighbor ? 0u : hashSideKey(current, neighbor, layerId, visible);
    }

    uint runCount = 0u;
    uint scanIndex = 0u;
    while (scanIndex < 16u)
    {
        const uint key = keys[scanIndex];
        if (key == 0u)
        {
            scanIndex += 1u;
            continue;
        }

        uint runLength = 1u;
        while ((scanIndex + runLength) < 16u &&
               runLength < clamp(gMaxMergeExtent, 1u, 16u) &&
               keys[scanIndex + runLength] == key)
        {
            runLength += 1u;
        }

        runCount += 1u;
        scanIndex += runLength;
    }
    return runCount;
}

[numthreads(64, 1, 1)]
void FarLodChunkFaceCountMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint linearIndex = dispatchThreadId.x;
    if (linearIndex >= kLogicalSize * kLogicalSize * kLogicalSize)
    {
        return;
    }

    if (linearIndex >= kPlaneCount)
    {
        gFaceCounts[linearIndex] = 0u;
        return;
    }

    uint layerId;
    bool isTopPlane;
    uint dirId;
    uint slice;
    decodePlane(linearIndex, layerId, isTopPlane, dirId, slice);
    gFaceCounts[linearIndex] = isTopPlane ? countTopQuads(layerId) : countSideRuns(layerId, dirId, slice);
}
