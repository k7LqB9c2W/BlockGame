#include "exact_chunk_common.hlsli"

cbuffer ExactChunkHaloCacheParams : register(b0)
{
    uint gResolvedNeighborMask;
    uint gClosedNeighborMask;
    int gChunkMinWorldY;
    uint gReserved1;
}

StructuredBuffer<uint> gNeighborPosX : register(t0);
StructuredBuffer<uint> gNeighborNegX : register(t1);
StructuredBuffer<uint> gNeighborPosY : register(t2);
StructuredBuffer<uint> gNeighborNegY : register(t3);
StructuredBuffer<uint> gNeighborPosZ : register(t4);
StructuredBuffer<uint> gNeighborNegZ : register(t5);
RWStructuredBuffer<uint> gHaloVoxels : register(u0);

uint sampleVoxel(StructuredBuffer<uint> bufferRef, uint x, uint y, uint z)
{
    return bufferRef[voxelIndex(x, y, z)];
}

uint defaultHaloVoxel(uint faceIndex, uint u, uint v)
{
    int localY = 0;
    if (faceIndex == kExactHaloFacePosY)
    {
        localY = int(kExactChunkSize);
    }
    else if (faceIndex == kExactHaloFaceNegY)
    {
        localY = -1;
    }
    else
    {
        (void)u;
        localY = int(v);
    }

    const int worldY = gChunkMinWorldY + localY;
    const uint defaultSky = (worldY < 0) ? 0u : 15u;
    return encodeVoxel(kBlockAir, defaultSky, 0u);
}

[numthreads(8, 8, 1)]
void ExactChunkHaloCacheMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kExactChunkSize ||
        dispatchThreadId.y >= kExactChunkSize ||
        dispatchThreadId.z >= kExactChunkHaloFaceCount)
    {
        return;
    }

    const uint faceIndex = dispatchThreadId.z;
    const uint u = dispatchThreadId.x;
    const uint v = dispatchThreadId.y;
    uint seamBit = 0u;
    uint packedVoxel = defaultHaloVoxel(faceIndex, u, v);

    if (faceIndex == kExactHaloFacePosX)
    {
        seamBit = kExactNeighborPosXBit;
        if ((gResolvedNeighborMask & seamBit) != 0u)
        {
            packedVoxel = sampleVoxel(gNeighborPosX, 0u, v, u);
        }
    }
    else if (faceIndex == kExactHaloFaceNegX)
    {
        seamBit = kExactNeighborNegXBit;
        if ((gResolvedNeighborMask & seamBit) != 0u)
        {
            packedVoxel = sampleVoxel(gNeighborNegX, kExactChunkSize - 1u, v, u);
        }
    }
    else if (faceIndex == kExactHaloFacePosY)
    {
        seamBit = kExactNeighborPosYBit;
        if ((gResolvedNeighborMask & seamBit) != 0u)
        {
            packedVoxel = sampleVoxel(gNeighborPosY, u, 0u, v);
        }
    }
    else if (faceIndex == kExactHaloFaceNegY)
    {
        seamBit = kExactNeighborNegYBit;
        if ((gResolvedNeighborMask & seamBit) != 0u)
        {
            packedVoxel = sampleVoxel(gNeighborNegY, u, kExactChunkSize - 1u, v);
        }
    }
    else if (faceIndex == kExactHaloFacePosZ)
    {
        seamBit = kExactNeighborPosZBit;
        if ((gResolvedNeighborMask & seamBit) != 0u)
        {
            packedVoxel = sampleVoxel(gNeighborPosZ, u, v, 0u);
        }
    }
    else
    {
        seamBit = kExactNeighborNegZBit;
        if ((gResolvedNeighborMask & seamBit) != 0u)
        {
            packedVoxel = sampleVoxel(gNeighborNegZ, u, v, kExactChunkSize - 1u);
        }
    }

    if ((gResolvedNeighborMask & seamBit) == 0u &&
        (gClosedNeighborMask & seamBit) != 0u)
    {
        packedVoxel = encodeVoxel(kBlockNeighborSolidSentinel, 0u, 0u);
    }

    gHaloVoxels[haloFaceVoxelIndex(faceIndex, u, v)] = packedVoxel;
}
