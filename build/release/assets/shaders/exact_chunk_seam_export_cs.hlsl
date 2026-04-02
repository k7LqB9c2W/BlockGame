#include "exact_chunk_common.hlsli"

cbuffer ExactChunkSeamExportParams : register(b0)
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

uint sourceVoxelIndexForFace(uint faceIndex, uint u, uint v)
{
    if (faceIndex == kExactHaloFacePosX)
    {
        return voxelIndex(kExactChunkSize - 1u, v, u);
    }
    if (faceIndex == kExactHaloFaceNegX)
    {
        return voxelIndex(0u, v, u);
    }
    if (faceIndex == kExactHaloFacePosY)
    {
        return voxelIndex(u, kExactChunkSize - 1u, v);
    }
    if (faceIndex == kExactHaloFaceNegY)
    {
        return voxelIndex(u, 0u, v);
    }
    if (faceIndex == kExactHaloFacePosZ)
    {
        return voxelIndex(u, v, kExactChunkSize - 1u);
    }
    return voxelIndex(u, v, 0u);
}

[numthreads(8, 8, 1)]
void ExactChunkSeamExportMain(uint3 dispatchThreadId : SV_DispatchThreadID)
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
    StructuredBuffer<uint> litVoxels =
        ResourceDescriptorHeap[NonUniformResourceIndex(build.lightScratchVoxelSrvDescriptorIndex)];
    RWStructuredBuffer<uint> seamVoxels =
        ResourceDescriptorHeap[NonUniformResourceIndex(build.seamVoxelUavDescriptorIndex)];

    const uint u = dispatchThreadId.x;
    const uint v = dispatchThreadId.y;
    seamVoxels[haloFaceVoxelIndex(faceIndex, u, v)] = litVoxels[sourceVoxelIndexForFace(faceIndex, u, v)];
}
