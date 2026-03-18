cbuffer FaceCountParams : register(b0)
{
    int gWorldMinY;
    int gBlockScale;
    uint gReserved0;
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

struct GpuFaceMergeDescriptor
{
    uint value0;
    uint value1;
    uint value2;
    uint value3;
};

StructuredBuffer<GpuTerrainColumnDescriptor> gColumnBuffer : register(t0);
StructuredBuffer<GpuTerrainColumnDescriptor> gNeighborPosX : register(t1);
StructuredBuffer<GpuTerrainColumnDescriptor> gNeighborNegX : register(t2);
StructuredBuffer<GpuTerrainColumnDescriptor> gNeighborPosZ : register(t3);
StructuredBuffer<GpuTerrainColumnDescriptor> gNeighborNegZ : register(t4);
RWStructuredBuffer<uint> gFaceCounts : register(u0);
RWStructuredBuffer<uint> gFaceMetadata : register(u1);
RWStructuredBuffer<GpuFaceMergeDescriptor> gFaceDescriptors : register(u2);

static const uint kLogicalSize = 16u;
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
static const uint kFaceMetadataClosureFloorOffset = 0u;
static const uint kMaxTopDescriptorsPerPlane = kLogicalSize * kLogicalSize;
static const uint kMaxSideDescriptorsPerPlane = kLogicalSize;

groupshared GpuTerrainColumnDescriptor gSharedColumns[kLogicalSize * kLogicalSize];
groupshared int gSharedTileClosureFloorY;

uint columnIndex(uint x, uint z)
{
    return z * kLogicalSize + x;
}

GpuTerrainColumnDescriptor sampleLocal(uint x, uint z)
{
    return gSharedColumns[columnIndex(x, z)];
}

uint topDescriptorBase(uint layerId)
{
    return layerId * kMaxTopDescriptorsPerPlane;
}

uint sideDescriptorBase(uint planeIndex)
{
    return kTopPlaneCount * kMaxTopDescriptorsPerPlane +
           (planeIndex - kTopPlaneCount) * kMaxSideDescriptorsPerPlane;
}

void storeDescriptor(uint descriptorIndex, uint value0, uint value1, uint value2, uint value3)
{
    GpuFaceMergeDescriptor descriptor;
    descriptor.value0 = value0;
    descriptor.value1 = value1;
    descriptor.value2 = value2;
    descriptor.value3 = value3;
    gFaceDescriptors[descriptorIndex] = descriptor;
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
    return true;
}

int computeTileClosureFloorY()
{
    return gSharedTileClosureFloorY;
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
                 int tileClosureFloorY,
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
    if (layerId == kLayerTerrain)
    {
        const bool neighborIsActive = layerActive(neighbor, layerId);
        int neighborTop = tileClosureFloorY - 1;
        if (neighborIsActive)
        {
            neighborTop = layerTopY(neighbor, layerId);
        }

        if (currentTop <= neighborTop)
        {
            return false;
        }

        segmentBottom = neighborTop + 1;
        segmentTopExclusive = currentTop + 1;
        return segmentTopExclusive > segmentBottom;
    }

    int occluderTop = currentBottom - 1;
    const bool neighborIsActive = layerActive(neighbor, layerId);
    if (neighborIsActive)
    {
        occluderTop = layerTopY(neighbor, layerId);
    }
    else if (layerId == kLayerTerrain)
    {
        occluderTop = currentBottom - 1;
    }
    else
    {
        return false;
    }

    if (currentTop <= occluderTop)
    {
        return false;
    }

    segmentBottom = max(currentBottom, occluderTop + 1);
    segmentTopExclusive = currentTop + 1;
    return segmentTopExclusive > segmentBottom;
}

uint hashSideDescriptorKey(GpuTerrainColumnDescriptor current,
                           uint layerId,
                           int segmentBottom,
                           int segmentTopExclusive)
{
    uint h = 2166136261u;
    h = (h ^ layerSideBlock(current, layerId)) * 16777619u;
    h = (h ^ layerMaterialFlags(layerId)) * 16777619u;
    h = (h ^ asuint(segmentBottom)) * 16777619u;
    h = (h ^ asuint(segmentTopExclusive)) * 16777619u;
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
    const uint descriptorBase = topDescriptorBase(layerId);
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

            storeDescriptor(descriptorBase + quadCount, x, z, width, height);
            quadCount += 1u;
        }
    }
    return quadCount;
}

uint countSideRuns(uint planeIndex, uint layerId, uint dirId, uint slice)
{
    const uint maxExtent = clamp(gMaxMergeExtent, 1u, 16u);
    const int tileClosureFloorY = computeTileClosureFloorY();
    const uint descriptorBase = sideDescriptorBase(planeIndex);
    uint keys[16];
    int segmentBottoms[16];
    int segmentTops[16];
    [unroll]
    for (uint i = 0u; i < 16u; ++i)
    {
        GpuTerrainColumnDescriptor current = (GpuTerrainColumnDescriptor)0;
        GpuTerrainColumnDescriptor neighbor = (GpuTerrainColumnDescriptor)0;
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
            if (slice > 0u)
            {
                neighbor = sampleLocal(slice - 1u, i);
            }
            else
            {
                neighbor = gNeighborNegX[columnIndex(kLogicalSize - 1u, i)];
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
            if (slice > 0u)
            {
                neighbor = sampleLocal(i, slice - 1u);
            }
            else
            {
                neighbor = gNeighborNegZ[columnIndex(i, kLogicalSize - 1u)];
            }
        }

        int segmentBottom = 0;
        int segmentTopExclusive = 0;
        if (sideSegment(current, neighbor, layerId, tileClosureFloorY, segmentBottom, segmentTopExclusive))
        {
            keys[i] = hashSideDescriptorKey(current, layerId, segmentBottom, segmentTopExclusive);
            segmentBottoms[i] = segmentBottom;
            segmentTops[i] = segmentTopExclusive;
        }
        else
        {
            keys[i] = 0u;
            segmentBottoms[i] = 0;
            segmentTops[i] = 0;
        }
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
               runLength < maxExtent &&
               keys[scanIndex + runLength] == key)
        {
            runLength += 1u;
        }

        storeDescriptor(descriptorBase + runCount,
                        scanIndex,
                        runLength,
                        asuint(segmentBottoms[scanIndex]),
                        asuint(segmentTops[scanIndex]));
        runCount += 1u;
        scanIndex += runLength;
    }
    return runCount;
}

[numthreads(64, 1, 1)]
void FarLodChunkFaceCountMain(uint3 dispatchThreadId : SV_DispatchThreadID,
                              uint3 groupThreadId : SV_GroupThreadID)
{
    const uint linearIndex = dispatchThreadId.x;
    const uint groupIndex = groupThreadId.x;
    for (uint loadIndex = groupIndex; loadIndex < (kLogicalSize * kLogicalSize); loadIndex += 64u)
    {
        const uint x = loadIndex & 15u;
        const uint z = loadIndex >> 4u;
        gSharedColumns[loadIndex] = gColumnBuffer[columnIndex(x, z)];
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupIndex == 0u)
    {
        int floorY = 2147483647;
        [loop]
        for (uint z = 0u; z < kLogicalSize; ++z)
        {
            [loop]
            for (uint x = 0u; x < kLogicalSize; ++x)
            {
                const GpuTerrainColumnDescriptor column = sampleLocal(x, z);
                if ((column.flags & kColumnFlagTerrain) != 0u)
                {
                    floorY = min(floorY, column.terrainTopY - gBlockScale);
                }
                if ((column.flags & kColumnFlagWater) != 0u)
                {
                    floorY = min(floorY, column.waterBottomY);
                }
                if ((column.flags & kColumnFlagCanopy) != 0u)
                {
                    floorY = min(floorY, column.canopyBottomY);
                }
            }
        }
        gSharedTileClosureFloorY = (floorY == 2147483647) ? gWorldMinY : floorY;
        if (linearIndex == 0u)
        {
            gFaceMetadata[kFaceMetadataClosureFloorOffset] = asuint(gSharedTileClosureFloorY);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (linearIndex >= kPlaneCount)
    {
        return;
    }

    uint layerId;
    bool isTopPlane;
    uint dirId;
    uint slice;
    decodePlane(linearIndex, layerId, isTopPlane, dirId, slice);
    gFaceCounts[linearIndex] = isTopPlane ? countTopQuads(layerId) : countSideRuns(linearIndex, layerId, dirId, slice);
}
