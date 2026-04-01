#include "exact_chunk_common.hlsli"

cbuffer ExactChunkSynthParams : register(b0)
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

[numthreads(8, 8, 1)]
void ExactChunkSynthMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kExactChunkSize ||
        dispatchThreadId.y >= kExactChunkSize ||
        dispatchThreadId.z >= gBatchBuildCount)
    {
        return;
    }

    const GpuExactPrepassRecord build = gPrepassRecords[dispatchThreadId.z];
    if (build.rebuildVoxelInputs == 0u ||
        build.centerVoxelUavDescriptorIndex == 0xffffffffu)
    {
        return;
    }

    RWStructuredBuffer<uint> voxels =
        ResourceDescriptorHeap[NonUniformResourceIndex(build.centerVoxelUavDescriptorIndex)];

    const uint localX = dispatchThreadId.x;
    const uint localZ = dispatchThreadId.y;
    const uint descriptorIndex = build.descriptorOffset + columnIndex(localX, localZ);
    const GpuExactColumnDescriptor column = gDescriptorScratch[descriptorIndex];

    [loop]
    for (uint localY = 0u; localY < kExactChunkSize; ++localY)
    {
        const int worldY = build.chunkWorldMinY + int(localY);
        uint blockId = kBlockAir;

        if ((column.flags & 0x02u) != 0u && worldY <= column.highestSolidWorld)
        {
            if (worldY < column.surfaceY)
            {
                blockId = column.fillerBlock;
                if ((column.flags & 0x10u) != 0u && column.stripePeriod > 0u && column.stripeThickness > 0u)
                {
                    const int pattern = (worldY + column.stripeOffset) % int(column.stripePeriod);
                    if (pattern >= 0 && pattern < int(column.stripeThickness))
                    {
                        blockId = column.stripeBlock;
                    }
                }
            }
            else
            {
                blockId = column.surfaceBlock;
            }
        }

        if ((column.flags & 0x04u) != 0u &&
            worldY >= column.waterBottomWorld &&
            worldY <= column.waterTopWorld &&
            blockId == kBlockAir)
        {
            blockId = column.waterBlock;
        }

        voxels[voxelIndex(localX, localY, localZ)] = encodeVoxel(blockId, 0u, 0u);
    }
}
