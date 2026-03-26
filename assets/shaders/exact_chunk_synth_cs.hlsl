#include "exact_chunk_common.hlsli"

cbuffer ExactChunkSynthParams : register(b0)
{
    int gChunkMinWorldY;
    uint gColumnCount;
    uint gVoxelCount;
    uint gReserved0;
};

StructuredBuffer<GpuExactColumnDescriptor> gColumns : register(t0);
RWStructuredBuffer<uint> gVoxels : register(u0);

[numthreads(8, 8, 1)]
void ExactChunkSynthMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kExactChunkSize || dispatchThreadId.y >= kExactChunkSize)
    {
        return;
    }

    const uint localX = dispatchThreadId.x;
    const uint localZ = dispatchThreadId.y;
    const uint descriptorIndex = columnIndex(localX, localZ);
    if (descriptorIndex >= gColumnCount)
    {
        return;
    }

    const GpuExactColumnDescriptor column = gColumns[descriptorIndex];
    [loop]
    for (uint localY = 0u; localY < kExactChunkSize; ++localY)
    {
        const int worldY = gChunkMinWorldY + int(localY);
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

        gVoxels[voxelIndex(localX, localY, localZ)] = encodeVoxel(blockId, 0u, 0u);
    }
}
