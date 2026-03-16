cbuffer FaceEmitParams : register(b0)
{
    uint gWorldMinX;
    uint gWorldMinY;
    uint gWorldMinZ;
    uint gBlockScale;
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

StructuredBuffer<uint> gVoxelBuffer : register(t0);
StructuredBuffer<uint> gNeighborPosX : register(t1);
StructuredBuffer<uint> gNeighborPosY : register(t2);
StructuredBuffer<uint> gNeighborPosZ : register(t3);
StructuredBuffer<uint> gFaceCounts : register(t4);
StructuredBuffer<uint> gFacePrefixes : register(t5);
StructuredBuffer<GpuBlockFaceUv> gBlockFaceUvs : register(t6);
RWStructuredBuffer<WorldVertex> gVertices : register(u0);
RWStructuredBuffer<uint> gIndices : register(u1);
RWStructuredBuffer<GpuCullRecord> gDrawRecords : register(u2);

static const uint kLogicalSize = 16u;
static const uint kVoxelCount = 4096u;
static const uint kNegativeNeighborX = 0x1u;
static const uint kNegativeNeighborY = 0x2u;
static const uint kNegativeNeighborZ = 0x4u;

static const uint kBlockAir = 0u;
static const uint kBlockLeaves = 3u;
static const uint kBlockWater = 5u;
static const uint kBlockSpruceLeaves = 8u;

static const uint kFaceTop = 0u;
static const uint kFaceBottom = 1u;
static const uint kFaceNorth = 2u;
static const uint kFaceSouth = 3u;
static const uint kFaceEast = 4u;
static const uint kFaceWest = 5u;

uint voxelIndex(uint x, uint y, uint z)
{
    return (y * kLogicalSize + z) * kLogicalSize + x;
}

bool isOccupied(uint packedVoxel)
{
    return (packedVoxel & 0x1u) != 0u;
}

uint voxelMaterial(uint packedVoxel)
{
    return (packedVoxel >> 8u) & 0xffu;
}

bool isWater(uint packedVoxel)
{
    return (packedVoxel & 0x2u) != 0u;
}

bool isAlphaCutoutBlock(uint blockId)
{
    return blockId == kBlockLeaves || blockId == kBlockSpruceLeaves;
}

bool isNonOpaqueBlock(uint blockId)
{
    return blockId == kBlockAir || blockId == kBlockWater || isAlphaCutoutBlock(blockId);
}

bool shouldRenderBlockFace(uint owningBlock, uint neighborBlock)
{
    if (owningBlock == kBlockAir)
    {
        return false;
    }

    if (neighborBlock == kBlockAir)
    {
        return true;
    }

    if (isAlphaCutoutBlock(owningBlock))
    {
        if (isAlphaCutoutBlock(neighborBlock))
        {
            return owningBlock != neighborBlock;
        }

        return neighborBlock == kBlockWater;
    }

    if (owningBlock == kBlockWater)
    {
        return neighborBlock == kBlockAir;
    }

    return isNonOpaqueBlock(neighborBlock);
}

float3 chunkWorldMin()
{
    return float3((float)gWorldMinX, (float)gWorldMinY, (float)gWorldMinZ);
}

float3 voxelMinCorner(uint x, uint y, uint z)
{
    return chunkWorldMin() + float3((float)(x * gBlockScale),
                                    (float)(y * gBlockScale),
                                    (float)(z * gBlockScale));
}

float2 projectTileCoord(uint faceId, float3 position)
{
    if (faceId == kFaceTop || faceId == kFaceBottom)
    {
        return float2(position.x, position.z);
    }
    if (faceId == kFaceEast || faceId == kFaceWest)
    {
        return float2(position.z, position.y);
    }
    return float2(position.x, position.y);
}

uint packLightingData(uint packedVoxel)
{
    uint flags = 0x02u;
    if (isWater(packedVoxel))
    {
        flags |= 0x01u;
    }
    return 0xF0u | (flags << 10u);
}

uint samplePositiveX(uint x, uint y, uint z)
{
    return (x + 1u < kLogicalSize) ? gVoxelBuffer[voxelIndex(x + 1u, y, z)]
                                   : gNeighborPosX[voxelIndex(0u, y, z)];
}

uint samplePositiveY(uint x, uint y, uint z)
{
    return (y + 1u < kLogicalSize) ? gVoxelBuffer[voxelIndex(x, y + 1u, z)]
                                   : gNeighborPosY[voxelIndex(x, 0u, z)];
}

uint samplePositiveZ(uint x, uint y, uint z)
{
    return (z + 1u < kLogicalSize) ? gVoxelBuffer[voxelIndex(x, y, z + 1u)]
                                   : gNeighborPosZ[voxelIndex(x, y, 0u)];
}

bool faceVisible(uint faceId, uint x, uint y, uint z, uint packedVoxel)
{
    const uint material = voxelMaterial(packedVoxel);
    if (faceId == kFaceTop)
    {
        return shouldRenderBlockFace(material, samplePositiveY(x, y, z));
    }
    if (faceId == kFaceBottom)
    {
        if (y == 0u && (gNegativeNeighborMask & kNegativeNeighborY) != 0u)
        {
            return false;
        }
        return shouldRenderBlockFace(material, (y > 0u) ? gVoxelBuffer[voxelIndex(x, y - 1u, z)] : 0u);
    }
    if (faceId == kFaceNorth)
    {
        if (z == 0u && (gNegativeNeighborMask & kNegativeNeighborZ) != 0u)
        {
            return false;
        }
        return shouldRenderBlockFace(material, (z > 0u) ? gVoxelBuffer[voxelIndex(x, y, z - 1u)] : 0u);
    }
    if (faceId == kFaceSouth)
    {
        return shouldRenderBlockFace(material, samplePositiveZ(x, y, z));
    }
    if (faceId == kFaceEast)
    {
        return shouldRenderBlockFace(material, samplePositiveX(x, y, z));
    }
    if (x == 0u && (gNegativeNeighborMask & kNegativeNeighborX) != 0u)
    {
        return false;
    }
    return shouldRenderBlockFace(material, (x > 0u) ? gVoxelBuffer[voxelIndex(x - 1u, y, z)] : 0u);
}

void faceVertices(uint faceId,
                  float3 minCorner,
                  out float3 p0,
                  out float3 p1,
                  out float3 p2,
                  out float3 p3,
                  out float3 normal)
{
    const float scale = (float)gBlockScale;
    const float3 maxCorner = minCorner + float3(scale, scale, scale);

    if (faceId == kFaceTop)
    {
        p0 = float3(minCorner.x, maxCorner.y, minCorner.z);
        p1 = float3(maxCorner.x, maxCorner.y, minCorner.z);
        p2 = float3(maxCorner.x, maxCorner.y, maxCorner.z);
        p3 = float3(minCorner.x, maxCorner.y, maxCorner.z);
        normal = float3(0.0f, 1.0f, 0.0f);
        return;
    }
    if (faceId == kFaceBottom)
    {
        p0 = float3(minCorner.x, minCorner.y, minCorner.z);
        p1 = float3(minCorner.x, minCorner.y, maxCorner.z);
        p2 = float3(maxCorner.x, minCorner.y, maxCorner.z);
        p3 = float3(maxCorner.x, minCorner.y, minCorner.z);
        normal = float3(0.0f, -1.0f, 0.0f);
        return;
    }
    if (faceId == kFaceNorth)
    {
        p0 = float3(minCorner.x, minCorner.y, minCorner.z);
        p1 = float3(maxCorner.x, minCorner.y, minCorner.z);
        p2 = float3(maxCorner.x, maxCorner.y, minCorner.z);
        p3 = float3(minCorner.x, maxCorner.y, minCorner.z);
        normal = float3(0.0f, 0.0f, -1.0f);
        return;
    }
    if (faceId == kFaceSouth)
    {
        p0 = float3(minCorner.x, minCorner.y, maxCorner.z);
        p1 = float3(minCorner.x, maxCorner.y, maxCorner.z);
        p2 = float3(maxCorner.x, maxCorner.y, maxCorner.z);
        p3 = float3(maxCorner.x, minCorner.y, maxCorner.z);
        normal = float3(0.0f, 0.0f, 1.0f);
        return;
    }
    if (faceId == kFaceEast)
    {
        p0 = float3(maxCorner.x, minCorner.y, minCorner.z);
        p1 = float3(maxCorner.x, minCorner.y, maxCorner.z);
        p2 = float3(maxCorner.x, maxCorner.y, maxCorner.z);
        p3 = float3(maxCorner.x, maxCorner.y, minCorner.z);
        normal = float3(1.0f, 0.0f, 0.0f);
        return;
    }

    p0 = float3(minCorner.x, minCorner.y, minCorner.z);
    p1 = float3(minCorner.x, maxCorner.y, minCorner.z);
    p2 = float3(minCorner.x, maxCorner.y, maxCorner.z);
    p3 = float3(minCorner.x, minCorner.y, maxCorner.z);
    normal = float3(-1.0f, 0.0f, 0.0f);
}

void emitFace(uint faceId, uint packedVoxel, uint x, uint y, uint z, uint emittedFaceIndex)
{
    const uint material = voxelMaterial(packedVoxel);
    const uint localVertexOffset = emittedFaceIndex * 4u;
    const uint vertexOffset = gVertexBase + localVertexOffset;
    const uint indexOffset = gIndexBase + emittedFaceIndex * 6u;
    const uint uvIndex = material * 6u + faceId;
    const GpuBlockFaceUv uv = gBlockFaceUvs[uvIndex];
    const float3 minCorner = voxelMinCorner(x, y, z);

    float3 p0;
    float3 p1;
    float3 p2;
    float3 p3;
    float3 normal;
    faceVertices(faceId, minCorner, p0, p1, p2, p3, normal);

    const uint lightingData = packLightingData(packedVoxel);

    WorldVertex vertex0;
    vertex0.position = p0;
    vertex0.normal = normal;
    vertex0.tileCoord = projectTileCoord(faceId, p0);
    vertex0.atlasBase = uv.base;
    vertex0.atlasSize = uv.size;
    vertex0.lightingData = lightingData;

    WorldVertex vertex1 = vertex0;
    vertex1.position = p1;
    vertex1.tileCoord = projectTileCoord(faceId, p1);

    WorldVertex vertex2 = vertex0;
    vertex2.position = p2;
    vertex2.tileCoord = projectTileCoord(faceId, p2);

    WorldVertex vertex3 = vertex0;
    vertex3.position = p3;
    vertex3.tileCoord = projectTileCoord(faceId, p3);

    gVertices[vertexOffset + 0u] = vertex0;
    gVertices[vertexOffset + 1u] = vertex1;
    gVertices[vertexOffset + 2u] = vertex2;
    gVertices[vertexOffset + 3u] = vertex3;

    gIndices[indexOffset + 0u] = localVertexOffset + 0u;
    gIndices[indexOffset + 1u] = localVertexOffset + 1u;
    gIndices[indexOffset + 2u] = localVertexOffset + 2u;
    gIndices[indexOffset + 3u] = localVertexOffset + 0u;
    gIndices[indexOffset + 4u] = localVertexOffset + 2u;
    gIndices[indexOffset + 5u] = localVertexOffset + 3u;
}

[numthreads(64, 1, 1)]
void FarLodChunkFaceEmitMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint linearIndex = dispatchThreadId.x;
    if (linearIndex >= kVoxelCount)
    {
        return;
    }

    const uint packedVoxel = gVoxelBuffer[linearIndex];
    const uint faceCount = gFaceCounts[linearIndex];
    const uint faceBase = gFacePrefixes[linearIndex];
    const uint x = linearIndex % kLogicalSize;
    const uint y = (linearIndex / kLogicalSize) % kLogicalSize;
    const uint z = linearIndex / (kLogicalSize * kLogicalSize);

    if (isOccupied(packedVoxel) && faceCount > 0u)
    {
        uint localFace = 0u;
        [unroll]
        for (uint faceId = 0u; faceId < 6u; ++faceId)
        {
            if (faceVisible(faceId, x, y, z, packedVoxel))
            {
                emitFace(faceId, packedVoxel, x, y, z, faceBase + localFace);
                localFace += 1u;
            }
        }
    }

    if (linearIndex == (kVoxelCount - 1u))
    {
        const uint totalFaces = faceBase + faceCount;
        GpuCullRecord record;
        record.boundsMin = float4(chunkWorldMin(), 1.0f);
        record.boundsMax = float4(chunkWorldMin() + float3((float)(kLogicalSize * gBlockScale),
                                                           (float)(kLogicalSize * gBlockScale),
                                                           (float)(kLogicalSize * gBlockScale)),
                                  1.0f);
        record.indexCount = totalFaces * 6u;
        record.firstIndexLocation = gIndexBase;
        record.baseVertex = (int)gVertexBase;
        record.reserved = 0u;
        gDrawRecords[gRecordIndex] = record;
    }
}
