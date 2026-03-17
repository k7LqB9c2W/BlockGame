cbuffer FaceEmitParams : register(b0)
{
    int gWorldMinX;
    int gWorldMinY;
    int gWorldMinZ;
    int gBlockScale;
    uint gMaxMergeExtent;
    uint gVertexBase;
    uint gIndexBase;
    uint gRecordIndex;
    uint gNegativeNeighborMask;
};

struct GpuBlockFaceUv
{
    float2 base;
    float2 size;
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

struct WorldVertex
{
    float3 position;
    float3 normal;
    float2 tileCoord;
    float2 atlasBase;
    float2 atlasSize;
    uint lightingData;
};

struct GpuCullRecord
{
    float4 boundsMin;
    float4 boundsMax;
    uint indexCount;
    uint firstIndexLocation;
    int baseVertex;
    uint reserved;
};

StructuredBuffer<GpuTerrainColumnDescriptor> gColumnBuffer : register(t0);
StructuredBuffer<GpuTerrainColumnDescriptor> gNeighborPosX : register(t1);
StructuredBuffer<GpuTerrainColumnDescriptor> gNeighborPosY : register(t2);
StructuredBuffer<GpuTerrainColumnDescriptor> gNeighborPosZ : register(t3);
StructuredBuffer<uint> gFaceCounts : register(t4);
StructuredBuffer<uint> gFacePrefixes : register(t5);
StructuredBuffer<GpuBlockFaceUv> gBlockFaceUvs : register(t6);
RWStructuredBuffer<WorldVertex> gVertices : register(u0);
RWStructuredBuffer<uint> gIndices : register(u1);
RWStructuredBuffer<GpuCullRecord> gDrawRecords : register(u2);

static const uint kLogicalSize = 16u;
static const uint kVoxelCount = 4096u;
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
static const uint kFaceTop = 0u;
static const uint kFaceNorth = 2u;
static const uint kFaceSouth = 3u;
static const uint kFaceEast = 4u;
static const uint kFaceWest = 5u;

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
    if (layerId == kLayerTerrain) return (column.flags & kColumnFlagTerrain) != 0u;
    if (layerId == kLayerWater) return (column.flags & kColumnFlagWater) != 0u;
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
    if (layerId == kLayerTerrain)
    {
        const bool neighborIsActive = layerActive(neighbor, layerId);
        int neighborTop = gWorldMinY - 1;
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

float2 projectTileCoord(uint faceId, float3 position)
{
    if (faceId == kFaceTop)
    {
        return float2(position.x, position.z);
    }
    if (faceId == kFaceEast || faceId == kFaceWest)
    {
        return float2(position.z, position.y);
    }
    return float2(position.x, position.y);
}

uint packLightingData(uint layerId)
{
    uint flags = kMaterialFlagFarLod;
    if (layerId == kLayerWater)
    {
        flags |= kMaterialFlagWater;
    }
    return 0xF0u | (flags << 10u) | ((gBlockScale & 0xFFu) << 16u);
}

void emitQuad(uint emittedFaceIndex,
              uint material,
              uint layerId,
              uint faceId,
              float3 p0,
              float3 p1,
              float3 p2,
              float3 p3,
              float3 normal)
{
    const uint localVertexOffset = emittedFaceIndex * 4u;
    const uint vertexOffset = gVertexBase + localVertexOffset;
    const uint indexOffset = gIndexBase + emittedFaceIndex * 6u;
    const GpuBlockFaceUv uv = gBlockFaceUvs[material * 6u + faceId];
    const uint lightingData = packLightingData(layerId);

    WorldVertex v0;
    v0.position = p0;
    v0.normal = normal;
    v0.tileCoord = projectTileCoord(faceId, p0);
    v0.atlasBase = uv.base;
    v0.atlasSize = uv.size;
    v0.lightingData = lightingData;

    WorldVertex v1 = v0;
    v1.position = p1;
    v1.tileCoord = projectTileCoord(faceId, p1);

    WorldVertex v2 = v0;
    v2.position = p2;
    v2.tileCoord = projectTileCoord(faceId, p2);

    WorldVertex v3 = v0;
    v3.position = p3;
    v3.tileCoord = projectTileCoord(faceId, p3);

    gVertices[vertexOffset + 0u] = v0;
    gVertices[vertexOffset + 1u] = v1;
    gVertices[vertexOffset + 2u] = v2;
    gVertices[vertexOffset + 3u] = v3;

    gIndices[indexOffset + 0u] = localVertexOffset + 0u;
    gIndices[indexOffset + 1u] = localVertexOffset + 1u;
    gIndices[indexOffset + 2u] = localVertexOffset + 2u;
    gIndices[indexOffset + 3u] = localVertexOffset + 0u;
    gIndices[indexOffset + 4u] = localVertexOffset + 2u;
    gIndices[indexOffset + 5u] = localVertexOffset + 3u;
}

void emitTopPlane(uint planeIndex, uint layerId)
{
    const uint sliceCount = gFaceCounts[planeIndex];
    if (sliceCount == 0u)
    {
        return;
    }

    const uint maxExtent = clamp(gMaxMergeExtent, 1u, 16u);
    const uint faceBase = gFacePrefixes[planeIndex];
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
            keys[fillZ * 16u + x] = hashTopKey(sampleLocal(x, fillZ), layerId);
        }
    }

    uint emitted = 0u;
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

            const GpuTerrainColumnDescriptor column = sampleLocal(x, z);
            const float y = (float)(layerTopY(column, layerId) + 1);
            const float x0 = (float)gWorldMinX + (float)(x * gBlockScale);
            const float z0 = (float)gWorldMinZ + (float)(z * gBlockScale);
            const float x1 = x0 + (float)(width * gBlockScale);
            const float z1 = z0 + (float)(height * gBlockScale);
            emitQuad(faceBase + emitted,
                     layerTopBlock(column, layerId),
                     layerId,
                     kFaceTop,
                     float3(x0, y, z0),
                     float3(x0, y, z1),
                     float3(x1, y, z1),
                     float3(x1, y, z0),
                     float3(0.0f, 1.0f, 0.0f));
            emitted += 1u;
        }
    }
}

void emitSidePlane(uint planeIndex, uint layerId, uint dirId, uint slice)
{
    const uint runCount = gFaceCounts[planeIndex];
    if (runCount == 0u)
    {
        return;
    }

    const uint maxExtent = clamp(gMaxMergeExtent, 1u, 16u);
    const uint faceBase = gFacePrefixes[planeIndex];
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

    uint emitted = 0u;
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

        GpuTerrainColumnDescriptor current = (GpuTerrainColumnDescriptor)0;
        GpuTerrainColumnDescriptor neighbor = (GpuTerrainColumnDescriptor)0;
        if (dirId == 0u)
        {
            current = sampleLocal(slice, scanIndex);
            if (slice + 1u < 16u)
            {
                neighbor = sampleLocal(slice + 1u, scanIndex);
            }
            else
            {
                neighbor = gNeighborPosX[columnIndex(0u, scanIndex)];
            }
        }
        else if (dirId == 1u)
        {
            current = sampleLocal(slice, scanIndex);
            if (slice > 0u)
            {
                neighbor = sampleLocal(slice - 1u, scanIndex);
            }
        }
        else if (dirId == 2u)
        {
            current = sampleLocal(scanIndex, slice);
            if (slice + 1u < 16u)
            {
                neighbor = sampleLocal(scanIndex, slice + 1u);
            }
            else
            {
                neighbor = gNeighborPosZ[columnIndex(scanIndex, 0u)];
            }
        }
        else
        {
            current = sampleLocal(scanIndex, slice);
            if (slice > 0u)
            {
                neighbor = sampleLocal(scanIndex, slice - 1u);
            }
        }

        int segmentBottom = 0;
        int segmentTopExclusive = 0;
        if (sideSegment(current, neighbor, layerId, segmentBottom, segmentTopExclusive))
        {
            float3 p0;
            float3 p1;
            float3 p2;
            float3 p3;
            float3 normal;

            if (dirId == 0u)
            {
                const float x = (float)gWorldMinX + (float)((slice + 1u) * gBlockScale);
                const float z0 = (float)gWorldMinZ + (float)(scanIndex * gBlockScale);
                const float z1 = z0 + (float)(runLength * gBlockScale);
                p0 = float3(x, (float)segmentBottom, z0);
                p1 = float3(x, (float)segmentBottom, z1);
                p2 = float3(x, (float)segmentTopExclusive, z1);
                p3 = float3(x, (float)segmentTopExclusive, z0);
                normal = float3(1.0f, 0.0f, 0.0f);
                emitQuad(faceBase + emitted, layerSideBlock(current, layerId), layerId, kFaceEast, p0, p3, p2, p1, normal);
            }
            else if (dirId == 1u)
            {
                const float x = (float)gWorldMinX + (float)(slice * gBlockScale);
                const float z0 = (float)gWorldMinZ + (float)(scanIndex * gBlockScale);
                const float z1 = z0 + (float)(runLength * gBlockScale);
                p0 = float3(x, (float)segmentBottom, z0);
                p1 = float3(x, (float)segmentTopExclusive, z0);
                p2 = float3(x, (float)segmentTopExclusive, z1);
                p3 = float3(x, (float)segmentBottom, z1);
                normal = float3(-1.0f, 0.0f, 0.0f);
                emitQuad(faceBase + emitted, layerSideBlock(current, layerId), layerId, kFaceWest, p0, p3, p2, p1, normal);
            }
            else if (dirId == 2u)
            {
                const float z = (float)gWorldMinZ + (float)((slice + 1u) * gBlockScale);
                const float x0 = (float)gWorldMinX + (float)(scanIndex * gBlockScale);
                const float x1 = x0 + (float)(runLength * gBlockScale);
                p0 = float3(x0, (float)segmentBottom, z);
                p1 = float3(x0, (float)segmentTopExclusive, z);
                p2 = float3(x1, (float)segmentTopExclusive, z);
                p3 = float3(x1, (float)segmentBottom, z);
                normal = float3(0.0f, 0.0f, 1.0f);
                emitQuad(faceBase + emitted, layerSideBlock(current, layerId), layerId, kFaceSouth, p0, p3, p2, p1, normal);
            }
            else
            {
                const float z = (float)gWorldMinZ + (float)(slice * gBlockScale);
                const float x0 = (float)gWorldMinX + (float)(scanIndex * gBlockScale);
                const float x1 = x0 + (float)(runLength * gBlockScale);
                p0 = float3(x0, (float)segmentBottom, z);
                p1 = float3(x1, (float)segmentBottom, z);
                p2 = float3(x1, (float)segmentTopExclusive, z);
                p3 = float3(x0, (float)segmentTopExclusive, z);
                normal = float3(0.0f, 0.0f, -1.0f);
                emitQuad(faceBase + emitted, layerSideBlock(current, layerId), layerId, kFaceNorth, p0, p3, p2, p1, normal);
            }

            emitted += 1u;
        }

        scanIndex += runLength;
    }
}

[numthreads(64, 1, 1)]
void FarLodChunkFaceEmitMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint linearIndex = dispatchThreadId.x;
    if (linearIndex >= kVoxelCount)
    {
        return;
    }

    if (linearIndex < kPlaneCount)
    {
        uint layerId;
        bool isTopPlane;
        uint dirId;
        uint slice;
        decodePlane(linearIndex, layerId, isTopPlane, dirId, slice);
        if (isTopPlane)
        {
            emitTopPlane(linearIndex, layerId);
        }
        else
        {
            emitSidePlane(linearIndex, layerId, dirId, slice);
        }
    }

    if (linearIndex == (kVoxelCount - 1u))
    {
        const uint totalFaces = gFacePrefixes[linearIndex] + gFaceCounts[linearIndex];
        int boundsMinY = 2147483647;
        int boundsMaxY = -2147483647;
        [loop]
        for (uint z = 0u; z < kLogicalSize; ++z)
        {
            [loop]
            for (uint x = 0u; x < kLogicalSize; ++x)
            {
                const GpuTerrainColumnDescriptor column = sampleLocal(x, z);
                if ((column.flags & kColumnFlagTerrain) != 0u)
                {
                    // Terrain side walls can conservatively close down to gWorldMinY - 1 for
                    // missing neighbors, so the cull bounds must include that full span.
                    boundsMinY = min(boundsMinY, gWorldMinY - 1);
                    boundsMaxY = max(boundsMaxY, column.terrainTopY + 1);
                }
                if ((column.flags & kColumnFlagWater) != 0u)
                {
                    boundsMinY = min(boundsMinY, column.waterBottomY);
                    boundsMaxY = max(boundsMaxY, column.waterTopY + 1);
                }
                if ((column.flags & kColumnFlagCanopy) != 0u)
                {
                    boundsMinY = min(boundsMinY, column.canopyBottomY);
                    boundsMaxY = max(boundsMaxY, column.canopyTopY + 1);
                }
            }
        }
        if (boundsMinY > boundsMaxY)
        {
            boundsMinY = gWorldMinY;
            boundsMaxY = gWorldMinY + gBlockScale;
        }

        GpuCullRecord record;
        record.boundsMin = float4((float)gWorldMinX, (float)boundsMinY, (float)gWorldMinZ, 1.0f);
        record.boundsMax = float4((float)(gWorldMinX + int(kLogicalSize) * gBlockScale),
                                  (float)boundsMaxY,
                                  (float)(gWorldMinZ + int(kLogicalSize) * gBlockScale),
                                  1.0f);
        record.indexCount = totalFaces * 6u;
        record.firstIndexLocation = gIndexBase;
        record.baseVertex = (int)gVertexBase;
        record.reserved = 0u;
        gDrawRecords[gRecordIndex] = record;
    }
}
