#include "exact_chunk_common.hlsli"

cbuffer ExactChunkHaloCacheParams : register(b0)
{
    uint gBatchBuildCount;
    uint gReserved0;
    uint gReserved1;
    uint gReserved2;
};

StructuredBuffer<GpuExactPrepassRecord> gPrepassRecords : register(t0);
StructuredBuffer<GpuExactColumnDescriptor> gDescriptorScratch : register(t1);
RWStructuredBuffer<uint> gFaceCountScratch : register(u0);
RWStructuredBuffer<GpuExactFaceDescriptor> gFaceDescriptorScratch : register(u1);
RWStructuredBuffer<uint> gFacePrefixScratch : register(u2);
RWStructuredBuffer<uint> gFaceTotalScratch : register(u3);

uint sampleSeamVoxelFromDescriptor(uint descriptorIndex, uint faceIndex, uint u, uint v)
{
    StructuredBuffer<uint> seamVoxels = ResourceDescriptorHeap[NonUniformResourceIndex(descriptorIndex)];
    return seamVoxels[haloFaceVoxelIndex(faceIndex, u, v)];
}

uint neighborDescriptorIndexForFace(GpuExactPrepassRecord build, uint faceIndex)
{
    if (faceIndex == kExactHaloFacePosX) return build.neighborPosXSeamSrvDescriptorIndex;
    if (faceIndex == kExactHaloFaceNegX) return build.neighborNegXSeamSrvDescriptorIndex;
    if (faceIndex == kExactHaloFacePosY) return build.neighborPosYSeamSrvDescriptorIndex;
    if (faceIndex == kExactHaloFaceNegY) return build.neighborNegYSeamSrvDescriptorIndex;
    if (faceIndex == kExactHaloFacePosZ) return build.neighborPosZSeamSrvDescriptorIndex;
    return build.neighborNegZSeamSrvDescriptorIndex;
}

uint seamBitForFace(uint faceIndex)
{
    if (faceIndex == kExactHaloFacePosX) return kExactNeighborPosXBit;
    if (faceIndex == kExactHaloFaceNegX) return kExactNeighborNegXBit;
    if (faceIndex == kExactHaloFacePosY) return kExactNeighborPosYBit;
    if (faceIndex == kExactHaloFaceNegY) return kExactNeighborNegYBit;
    if (faceIndex == kExactHaloFacePosZ) return kExactNeighborPosZBit;
    return kExactNeighborNegZBit;
}

uint oppositeSeamFaceIndex(uint faceIndex)
{
    if (faceIndex == kExactHaloFacePosX) return kExactHaloFaceNegX;
    if (faceIndex == kExactHaloFaceNegX) return kExactHaloFacePosX;
    if (faceIndex == kExactHaloFacePosY) return kExactHaloFaceNegY;
    if (faceIndex == kExactHaloFaceNegY) return kExactHaloFacePosY;
    if (faceIndex == kExactHaloFacePosZ) return kExactHaloFaceNegZ;
    return kExactHaloFacePosZ;
}

uint defaultHaloVoxel(GpuExactPrepassRecord build, uint faceIndex, uint u, uint v)
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

    const int worldY = build.chunkWorldMinY + localY;
    const uint defaultSky = (worldY < 0) ? 0u : 15u;
    return encodeVoxel(kBlockAir, defaultSky, 0u);
}

[numthreads(8, 8, 1)]
void ExactChunkHaloCacheMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kExactChunkSize || dispatchThreadId.y >= kExactChunkSize)
    {
        return;
    }

    const uint buildIndex = dispatchThreadId.z / kExactChunkHaloFaceCount;
    const uint faceIndex = dispatchThreadId.z % kExactChunkHaloFaceCount;
    if (buildIndex >= gBatchBuildCount || faceIndex >= kExactChunkHaloFaceCount)
    {
        return;
    }

    const GpuExactPrepassRecord build = gPrepassRecords[buildIndex];
    RWStructuredBuffer<uint> haloVoxels =
        ResourceDescriptorHeap[NonUniformResourceIndex(build.haloVoxelUavDescriptorIndex)];

    const uint u = dispatchThreadId.x;
    const uint v = dispatchThreadId.y;
    const uint seamBit = seamBitForFace(faceIndex);
    uint packedVoxel = defaultHaloVoxel(build, faceIndex, u, v);

    if ((build.resolvedNeighborMask & seamBit) != 0u)
    {
        const uint neighborDescriptorIndex = neighborDescriptorIndexForFace(build, faceIndex);
        packedVoxel = sampleSeamVoxelFromDescriptor(neighborDescriptorIndex, oppositeSeamFaceIndex(faceIndex), u, v);
    }
    else if ((build.closedNeighborMask & seamBit) != 0u)
    {
        packedVoxel = encodeVoxel(kBlockNeighborSolidSentinel, 0u, 0u);
    }

    haloVoxels[haloFaceVoxelIndex(faceIndex, u, v)] = packedVoxel;
}
