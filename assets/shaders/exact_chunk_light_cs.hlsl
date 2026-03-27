#include "exact_chunk_common.hlsli"

cbuffer ExactChunkLightParams : register(b0)
{
    uint gVoxelCount;
    uint gReserved0;
    uint gReserved1;
    uint gReserved2;
};

StructuredBuffer<uint> gCenterVoxels : register(t0);
StructuredBuffer<uint> gNeighborPosX : register(t1);
StructuredBuffer<uint> gNeighborNegX : register(t2);
StructuredBuffer<uint> gNeighborPosY : register(t3);
StructuredBuffer<uint> gNeighborNegY : register(t4);
StructuredBuffer<uint> gNeighborPosZ : register(t5);
StructuredBuffer<uint> gNeighborNegZ : register(t6);
RWStructuredBuffer<uint> gDestVoxels : register(u0);

uint sampleVoxel(StructuredBuffer<uint> bufferRef, int x, int y, int z)
{
    if (x < 0 || y < 0 || z < 0 ||
        x >= int(kExactChunkSize) || y >= int(kExactChunkSize) || z >= int(kExactChunkSize))
    {
        return encodeVoxel(kBlockAir, 0u, 0u);
    }

    return bufferRef[voxelIndex(uint(x), uint(y), uint(z))];
}

uint sampleVoxelWithNeighbors(int x, int y, int z)
{
    if (x >= 0 && y >= 0 && z >= 0 &&
        x < int(kExactChunkSize) && y < int(kExactChunkSize) && z < int(kExactChunkSize))
    {
        return sampleVoxel(gCenterVoxels, x, y, z);
    }

    if (x == int(kExactChunkSize) && y >= 0 && y < int(kExactChunkSize) && z >= 0 && z < int(kExactChunkSize))
    {
        return sampleVoxel(gNeighborPosX, 0, y, z);
    }
    if (x == -1 && y >= 0 && y < int(kExactChunkSize) && z >= 0 && z < int(kExactChunkSize))
    {
        return sampleVoxel(gNeighborNegX, int(kExactChunkSize) - 1, y, z);
    }
    if (y == int(kExactChunkSize) && x >= 0 && x < int(kExactChunkSize) && z >= 0 && z < int(kExactChunkSize))
    {
        return sampleVoxel(gNeighborPosY, x, 0, z);
    }
    if (y == -1 && x >= 0 && x < int(kExactChunkSize) && z >= 0 && z < int(kExactChunkSize))
    {
        return sampleVoxel(gNeighborNegY, x, int(kExactChunkSize) - 1, z);
    }
    if (z == int(kExactChunkSize) && x >= 0 && x < int(kExactChunkSize) && y >= 0 && y < int(kExactChunkSize))
    {
        return sampleVoxel(gNeighborPosZ, x, y, 0);
    }
    if (z == -1 && x >= 0 && x < int(kExactChunkSize) && y >= 0 && y < int(kExactChunkSize))
    {
        return sampleVoxel(gNeighborNegZ, x, y, int(kExactChunkSize) - 1);
    }

    return encodeVoxel(kBlockAir, 0u, 0u);
}

[numthreads(8, 8, 1)]
void ExactChunkLightSeedMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kExactChunkSize || dispatchThreadId.y >= kExactChunkSize)
    {
        return;
    }

    const uint localX = dispatchThreadId.x;
    const uint localZ = dispatchThreadId.y;
    uint incomingSky =
        voxelSkyLight(sampleVoxelWithNeighbors(int(localX), int(kExactChunkSize), int(localZ)));

    [loop]
    for (int localY = int(kExactChunkSize) - 1; localY >= 0; --localY)
    {
        const uint index = voxelIndex(localX, uint(localY), localZ);
        if (index >= gVoxelCount)
        {
            continue;
        }

        const uint packed = gCenterVoxels[index];
        const uint blockId = voxelBlock(packed);
        const uint emission = blockEmissionForBlock(blockId);
        if (isOpaqueForLighting(blockId))
        {
            incomingSky = 0u;
            gDestVoxels[index] = encodeVoxel(blockId, 0u, emission);
            continue;
        }

        incomingSky = attenuateLight(incomingSky, propagationLossForBlock(blockId));
        gDestVoxels[index] = encodeVoxel(blockId, incomingSky, emission);
    }
}

[numthreads(4, 4, 4)]
void ExactChunkLightPropagateMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kExactChunkSize ||
        dispatchThreadId.y >= kExactChunkSize ||
        dispatchThreadId.z >= kExactChunkSize)
    {
        return;
    }

    const uint index = voxelIndex(dispatchThreadId.x, dispatchThreadId.y, dispatchThreadId.z);
    if (index >= gVoxelCount)
    {
        return;
    }

    const int3 local = int3(dispatchThreadId.xyz);
    const uint packedCenter = gCenterVoxels[index];
    const uint blockId = voxelBlock(packedCenter);
    const uint emission = blockEmissionForBlock(blockId);
    if (isOpaqueForLighting(blockId))
    {
        gDestVoxels[index] = encodeVoxel(blockId, 0u, emission);
        return;
    }

    const uint loss = propagationLossForBlock(blockId);
    uint skyLight = voxelSkyLight(packedCenter);
    uint blockLight = max(voxelBlockLight(packedCenter), emission);

    const uint neighborSamples[6] = {
        sampleVoxelWithNeighbors(local.x + 1, local.y, local.z),
        sampleVoxelWithNeighbors(local.x - 1, local.y, local.z),
        sampleVoxelWithNeighbors(local.x, local.y + 1, local.z),
        sampleVoxelWithNeighbors(local.x, local.y - 1, local.z),
        sampleVoxelWithNeighbors(local.x, local.y, local.z + 1),
        sampleVoxelWithNeighbors(local.x, local.y, local.z - 1)
    };

    [unroll]
    for (uint i = 0u; i < 6u; ++i)
    {
        skyLight = max(skyLight, attenuateLight(voxelSkyLight(neighborSamples[i]), loss));
        blockLight = max(blockLight, attenuateLight(voxelBlockLight(neighborSamples[i]), loss));
    }

    gDestVoxels[index] = encodeVoxel(blockId, skyLight, blockLight);
}
