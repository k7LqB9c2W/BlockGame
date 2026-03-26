#include "exact_chunk_common.hlsli"

cbuffer ExactChunkLightParams : register(b0)
{
    uint gColumnCount;
    uint gVoxelCount;
    uint gReserved0;
    uint gReserved1;
};

StructuredBuffer<GpuExactColumnDescriptor> gColumns : register(t0);
RWStructuredBuffer<uint> gVoxels : register(u0);

[numthreads(8, 8, 1)]
void ExactChunkLightMain(uint3 dispatchThreadId : SV_DispatchThreadID)
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
    uint incomingSky = min(column.incomingSky, 15u);

    [loop]
    for (int localY = int(kExactChunkSize) - 1; localY >= 0; --localY)
    {
        const uint index = voxelIndex(localX, uint(localY), localZ);
        if (index >= gVoxelCount)
        {
            continue;
        }

        const uint packed = gVoxels[index];
        const uint blockId = voxelBlock(packed);
        if (isOpaqueForLighting(blockId))
        {
            incomingSky = 0u;
        }
        else
        {
            const uint attenuation = skyAttenuationForBlock(blockId);
            incomingSky = (incomingSky > attenuation) ? (incomingSky - attenuation) : 0u;
        }

        gVoxels[index] = encodeVoxel(blockId, incomingSky, blockEmissionForBlock(blockId));
    }
}
