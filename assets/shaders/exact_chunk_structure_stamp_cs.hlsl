#include "exact_chunk_common.hlsli"

cbuffer ExactChunkStructureStampParams : register(b0)
{
    uint gSparseVoxelCount;
    uint gVoxelCount;
    uint gReserved0;
    uint gReserved1;
};

StructuredBuffer<GpuExactSparseVoxel> gSparseVoxels : register(t0);
RWStructuredBuffer<uint> gVoxels : register(u0);

[numthreads(64, 1, 1)]
void ExactChunkStructureStampMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint sparseIndex = dispatchThreadId.x;
    if (sparseIndex >= gSparseVoxelCount)
    {
        return;
    }

    const GpuExactSparseVoxel edit = gSparseVoxels[sparseIndex];
    const uint localX = decodeLocalX(edit.packedLocalPos);
    const uint localY = decodeLocalY(edit.packedLocalPos);
    const uint localZ = decodeLocalZ(edit.packedLocalPos);
    if (localX >= kExactChunkSize || localY >= kExactChunkSize || localZ >= kExactChunkSize)
    {
        return;
    }

    const uint index = voxelIndex(localX, localY, localZ);
    if (index >= gVoxelCount)
    {
        return;
    }

    const uint current = gVoxels[index];
    if ((edit.flags & 0x01u) != 0u || voxelBlock(current) == kBlockAir)
    {
        gVoxels[index] = encodeVoxel(edit.block, voxelSkyLight(current), voxelBlockLight(current));
    }
}
