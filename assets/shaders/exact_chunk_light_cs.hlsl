#include "exact_chunk_common.hlsli"

cbuffer ExactChunkLightParams : register(b0)
{
    uint gVoxelCount;
    uint gResolvedNeighborMask;
    uint gClosedNeighborMask;
    uint gPropagationPassCount;
};

StructuredBuffer<GpuExactColumnDescriptor> gColumns : register(t0);
StructuredBuffer<uint> gCenterVoxels : register(t1);
StructuredBuffer<uint> gNeighborPosX : register(t2);
StructuredBuffer<uint> gNeighborNegX : register(t3);
StructuredBuffer<uint> gNeighborPosY : register(t4);
StructuredBuffer<uint> gNeighborNegY : register(t5);
StructuredBuffer<uint> gNeighborPosZ : register(t6);
StructuredBuffer<uint> gNeighborNegZ : register(t7);
RWStructuredBuffer<uint> gDestVoxels : register(u0);

groupshared uint sSeededVoxels[kExactChunkVoxelCount];
groupshared uint sPropagatedVoxels[kExactChunkVoxelCount];

uint sampleVoxel(StructuredBuffer<uint> bufferRef, int x, int y, int z)
{
    if (x < 0 || y < 0 || z < 0 ||
        x >= int(kExactChunkSize) || y >= int(kExactChunkSize) || z >= int(kExactChunkSize))
    {
        return encodeVoxel(kBlockAir, 0u, 0u);
    }

    return bufferRef[voxelIndex(uint(x), uint(y), uint(z))];
}

uint sampleResolvedNeighbor(uint seamBit, int x, int y, int z)
{
    if (seamBit == kExactNeighborPosXBit)
    {
        return sampleVoxel(gNeighborPosX, x, y, z);
    }
    if (seamBit == kExactNeighborNegXBit)
    {
        return sampleVoxel(gNeighborNegX, x, y, z);
    }
    if (seamBit == kExactNeighborPosYBit)
    {
        return sampleVoxel(gNeighborPosY, x, y, z);
    }
    if (seamBit == kExactNeighborNegYBit)
    {
        return sampleVoxel(gNeighborNegY, x, y, z);
    }
    if (seamBit == kExactNeighborPosZBit)
    {
        return sampleVoxel(gNeighborPosZ, x, y, z);
    }
    if (seamBit == kExactNeighborNegZBit)
    {
        return sampleVoxel(gNeighborNegZ, x, y, z);
    }

    return encodeVoxel(kBlockAir, 0u, 0u);
}

uint sampleCurrentFallback(uint seamBit, int x, int y, int z)
{
    if ((gClosedNeighborMask & seamBit) != 0u)
    {
        return encodeVoxel(kBlockNeighborSolidSentinel, 0u, 0u);
    }
    if (seamBit == kExactNeighborPosYBit)
    {
        return encodeVoxel(kBlockAir, 15u, 0u);
    }

    const uint index = voxelIndex(uint(x), uint(y), uint(z));
    return sPropagatedVoxels[index];
}

uint sampleCurrentOrNeighbor(int x, int y, int z)
{
    if (x >= 0 && y >= 0 && z >= 0 &&
        x < int(kExactChunkSize) && y < int(kExactChunkSize) && z < int(kExactChunkSize))
    {
        return sPropagatedVoxels[voxelIndex(uint(x), uint(y), uint(z))];
    }

    uint seamBit = 0u;
    if (x == int(kExactChunkSize) && y >= 0 && y < int(kExactChunkSize) && z >= 0 && z < int(kExactChunkSize))
    {
        seamBit = kExactNeighborPosXBit;
        x = 0;
    }
    else if (x == -1 && y >= 0 && y < int(kExactChunkSize) && z >= 0 && z < int(kExactChunkSize))
    {
        seamBit = kExactNeighborNegXBit;
        x = int(kExactChunkSize) - 1;
    }
    else if (y == int(kExactChunkSize) && x >= 0 && x < int(kExactChunkSize) && z >= 0 && z < int(kExactChunkSize))
    {
        seamBit = kExactNeighborPosYBit;
        y = 0;
    }
    else if (y == -1 && x >= 0 && x < int(kExactChunkSize) && z >= 0 && z < int(kExactChunkSize))
    {
        seamBit = kExactNeighborNegYBit;
        y = int(kExactChunkSize) - 1;
    }
    else if (z == int(kExactChunkSize) && x >= 0 && x < int(kExactChunkSize) && y >= 0 && y < int(kExactChunkSize))
    {
        seamBit = kExactNeighborPosZBit;
        z = 0;
    }
    else if (z == -1 && x >= 0 && x < int(kExactChunkSize) && y >= 0 && y < int(kExactChunkSize))
    {
        seamBit = kExactNeighborNegZBit;
        z = int(kExactChunkSize) - 1;
    }

    if (seamBit == 0u)
    {
        return encodeVoxel(kBlockAir, 0u, 0u);
    }
    if ((gResolvedNeighborMask & seamBit) != 0u)
    {
        return sampleResolvedNeighbor(seamBit, x, y, z);
    }
    return sampleCurrentFallback(seamBit, x, y, z);
}

uint relaxVoxelFromBase(uint packedBase, int3 local)
{
    const uint blockId = voxelBlock(packedBase);
    const uint emission = blockEmissionForBlock(blockId);
    if (isOpaqueForLighting(blockId))
    {
        return encodeVoxel(blockId, 0u, emission);
    }

    const uint loss = propagationLossForBlock(blockId);
    uint skyLight = voxelSkyLight(packedBase);
    uint blockLight = max(voxelBlockLight(packedBase), emission);
    const uint neighborSamples[6] = {
        sampleCurrentOrNeighbor(local.x + 1, local.y, local.z),
        sampleCurrentOrNeighbor(local.x - 1, local.y, local.z),
        sampleCurrentOrNeighbor(local.x, local.y + 1, local.z),
        sampleCurrentOrNeighbor(local.x, local.y - 1, local.z),
        sampleCurrentOrNeighbor(local.x, local.y, local.z + 1),
        sampleCurrentOrNeighbor(local.x, local.y, local.z - 1)
    };

    [unroll]
    for (uint i = 0u; i < 6u; ++i)
    {
        skyLight = max(skyLight, attenuateLight(voxelSkyLight(neighborSamples[i]), loss));
        blockLight = max(blockLight, attenuateLight(voxelBlockLight(neighborSamples[i]), loss));
    }

    return encodeVoxel(blockId, skyLight, blockLight);
}

[numthreads(16, 16, 1)]
void ExactChunkLightMain(uint3 groupThreadId : SV_GroupThreadID)
{
    if (groupThreadId.x >= kExactChunkSize || groupThreadId.y >= kExactChunkSize)
    {
        return;
    }

    const uint localX = groupThreadId.x;
    const uint localZ = groupThreadId.y;
    const uint descriptorIndex = columnIndex(localX, localZ);
    const GpuExactColumnDescriptor column = gColumns[descriptorIndex];
    uint incomingSky = min(column.skyLightFromAbove, 15u);

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
        uint seededVoxel = encodeVoxel(blockId, 0u, emission);
        if (isOpaqueForLighting(blockId))
        {
            incomingSky = 0u;
        }
        else
        {
            incomingSky = attenuateLight(incomingSky, propagationLossForBlock(blockId));
            seededVoxel = encodeVoxel(blockId, incomingSky, emission);
        }

        sSeededVoxels[index] = seededVoxel;
        sPropagatedVoxels[index] = seededVoxel;
    }

    GroupMemoryBarrierWithGroupSync();

    [loop]
    for (uint passIndex = 0u; passIndex < gPropagationPassCount; ++passIndex)
    {
        [loop]
        for (uint localY = 0u; localY < kExactChunkSize; ++localY)
        {
            const uint index = voxelIndex(localX, localY, localZ);
            if (index >= gVoxelCount)
            {
                continue;
            }

            gDestVoxels[index] = relaxVoxelFromBase(sSeededVoxels[index],
                                                    int3(int(localX), int(localY), int(localZ)));
        }

        AllMemoryBarrierWithGroupSync();

        [loop]
        for (uint localY = 0u; localY < kExactChunkSize; ++localY)
        {
            const uint index = voxelIndex(localX, localY, localZ);
            if (index >= gVoxelCount)
            {
                continue;
            }

            sPropagatedVoxels[index] = gDestVoxels[index];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [loop]
    for (uint localY = 0u; localY < kExactChunkSize; ++localY)
    {
        const uint index = voxelIndex(localX, localY, localZ);
        if (index >= gVoxelCount)
        {
            continue;
        }

        gDestVoxels[index] = sPropagatedVoxels[index];
    }
}
