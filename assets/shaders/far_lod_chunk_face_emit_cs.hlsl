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
    uint gReserved0;
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

struct GpuFaceMergeDescriptor
{
    uint value0;
    uint value1;
    uint value2;
    uint value3;
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
StructuredBuffer<uint> gFaceCounts : register(t1);
StructuredBuffer<uint> gFaceMetadata : register(t2);
StructuredBuffer<GpuFaceMergeDescriptor> gFaceDescriptors : register(t3);
StructuredBuffer<uint> gFacePrefixes : register(t4);
StructuredBuffer<GpuBlockFaceUv> gBlockFaceUvs : register(t5);
RWStructuredBuffer<WorldVertex> gVertices : register(u0);
RWStructuredBuffer<uint> gIndices : register(u1);
RWStructuredBuffer<GpuCullRecord> gDrawRecords : register(u2);

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
static const uint kFaceTop = 0u;
static const uint kFaceNorth = 2u;
static const uint kFaceSouth = 3u;
static const uint kFaceEast = 4u;
static const uint kFaceWest = 5u;

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

int computeTileClosureFloorY()
{
    return gSharedTileClosureFloorY;
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
    const uint quadCount = gFaceCounts[planeIndex];
    if (quadCount == 0u)
    {
        return;
    }

    const uint faceBase = gFacePrefixes[planeIndex];
    const uint descriptorBase = topDescriptorBase(layerId);
    for (uint descriptorIndex = 0u; descriptorIndex < quadCount; ++descriptorIndex)
    {
        const GpuFaceMergeDescriptor descriptor = gFaceDescriptors[descriptorBase + descriptorIndex];
        const uint x = descriptor.value0;
        const uint z = descriptor.value1;
        const uint width = descriptor.value2;
        const uint height = descriptor.value3;
        const GpuTerrainColumnDescriptor column = sampleLocal(x, z);
        const float y = (float)(layerTopY(column, layerId) + 1);
        const float x0 = (float)gWorldMinX + (float)(x * gBlockScale);
        const float z0 = (float)gWorldMinZ + (float)(z * gBlockScale);
        const float x1 = x0 + (float)(width * gBlockScale);
        const float z1 = z0 + (float)(height * gBlockScale);
        emitQuad(faceBase + descriptorIndex,
                 layerTopBlock(column, layerId),
                 layerId,
                 kFaceTop,
                 float3(x0, y, z0),
                 float3(x0, y, z1),
                 float3(x1, y, z1),
                 float3(x1, y, z0),
                 float3(0.0f, 1.0f, 0.0f));
    }
}

void emitSidePlane(uint planeIndex, uint layerId, uint dirId, uint slice)
{
    const uint runCount = gFaceCounts[planeIndex];
    if (runCount == 0u)
    {
        return;
    }

    const uint faceBase = gFacePrefixes[planeIndex];
    const uint descriptorBase = sideDescriptorBase(planeIndex);
    for (uint descriptorIndex = 0u; descriptorIndex < runCount; ++descriptorIndex)
    {
        const GpuFaceMergeDescriptor descriptor = gFaceDescriptors[descriptorBase + descriptorIndex];
        const uint scanIndex = descriptor.value0;
        const uint runLength = descriptor.value1;
        const int segmentBottom = asint(descriptor.value2);
        const int segmentTopExclusive = asint(descriptor.value3);

        GpuTerrainColumnDescriptor current = (GpuTerrainColumnDescriptor)0;
        if (dirId < 2u)
        {
            current = sampleLocal(slice, scanIndex);
        }
        else
        {
            current = sampleLocal(scanIndex, slice);
        }
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
            emitQuad(faceBase + descriptorIndex, layerSideBlock(current, layerId), layerId, kFaceEast, p0, p3, p2, p1, normal);
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
            emitQuad(faceBase + descriptorIndex, layerSideBlock(current, layerId), layerId, kFaceWest, p0, p3, p2, p1, normal);
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
            emitQuad(faceBase + descriptorIndex, layerSideBlock(current, layerId), layerId, kFaceSouth, p0, p3, p2, p1, normal);
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
            emitQuad(faceBase + descriptorIndex, layerSideBlock(current, layerId), layerId, kFaceNorth, p0, p3, p2, p1, normal);
        }
    }
}

[numthreads(64, 1, 1)]
void FarLodChunkFaceEmitMain(uint3 dispatchThreadId : SV_DispatchThreadID,
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
        gSharedTileClosureFloorY = asint(gFaceMetadata[kFaceMetadataClosureFloorOffset]);
    }
    GroupMemoryBarrierWithGroupSync();

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

    if (linearIndex == 0u)
    {
        const uint totalFaces = gFacePrefixes[kPlaneCount - 1u] + gFaceCounts[kPlaneCount - 1u];
        const int tileClosureFloorY = computeTileClosureFloorY();
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
                    boundsMinY = min(boundsMinY, tileClosureFloorY - 1);
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
            boundsMinY = tileClosureFloorY;
            boundsMaxY = tileClosureFloorY + gBlockScale;
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
