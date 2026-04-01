#include "exact_chunk_common.hlsli"

cbuffer ExactChunkStructureStampParams : register(b0)
{
    uint gBatchBuildCount;
    uint gMaxSparseVoxelGroups;
    uint gReserved0;
    uint gReserved1;
};

StructuredBuffer<GpuExactPrepassRecord> gPrepassRecords : register(t0);
StructuredBuffer<GpuExactColumnDescriptor> gDescriptorScratch : register(t1);
RWStructuredBuffer<uint> gFaceCountScratch : register(u0);
RWStructuredBuffer<GpuExactFaceDescriptor> gFaceDescriptorScratch : register(u1);
RWStructuredBuffer<uint> gFacePrefixScratch : register(u2);
RWStructuredBuffer<uint> gFaceTotalScratch : register(u3);

[numthreads(64, 1, 1)]
void ExactChunkStructureStampMain(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint buildIndex = groupId.y;
    if (buildIndex >= gBatchBuildCount || groupId.x >= gMaxSparseVoxelGroups)
    {
        return;
    }

    const GpuExactPrepassRecord build = gPrepassRecords[buildIndex];
    if (build.rebuildVoxelInputs == 0u || build.sparseVoxelCount == 0u)
    {
        return;
    }

    const uint sparseIndex = groupId.x * 64u + groupThreadId.x;
    if (sparseIndex >= build.sparseVoxelCount)
    {
        return;
    }

    StructuredBuffer<GpuExactSparseVoxel> sparseVoxels =
        ResourceDescriptorHeap[NonUniformResourceIndex(build.sparseVoxelSrvDescriptorIndex)];
    RWStructuredBuffer<uint> voxels =
        ResourceDescriptorHeap[NonUniformResourceIndex(build.centerVoxelUavDescriptorIndex)];

    const GpuExactSparseVoxel edit = sparseVoxels[sparseIndex];
    const uint localX = decodeLocalX(edit.packedLocalPos);
    const uint localY = decodeLocalY(edit.packedLocalPos);
    const uint localZ = decodeLocalZ(edit.packedLocalPos);
    if (localX >= kExactChunkSize || localY >= kExactChunkSize || localZ >= kExactChunkSize)
    {
        return;
    }

    const uint index = voxelIndex(localX, localY, localZ);
    const uint current = voxels[index];
    if ((edit.flags & 0x01u) != 0u || voxelBlock(current) == kBlockAir)
    {
        voxels[index] = encodeVoxel(edit.block, voxelSkyLight(current), voxelBlockLight(current));
    }
}
