#include "exact_chunk_common.hlsli"

cbuffer ExactChunkLightParams : register(b0)
{
    uint gVoxelCount;
    uint gResolvedNeighborMask;
    uint gClosedNeighborMask;
    uint gPropagationPassCount;
};

StructuredBuffer<uint> gCenterVoxels : register(t0);
StructuredBuffer<uint> gNeighborPosX : register(t1);
StructuredBuffer<uint> gNeighborNegX : register(t2);
StructuredBuffer<uint> gNeighborPosY : register(t3);
StructuredBuffer<uint> gNeighborNegY : register(t4);
StructuredBuffer<uint> gNeighborPosZ : register(t5);
StructuredBuffer<uint> gNeighborNegZ : register(t6);
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

uint sampleLightingFallback(bool readPropagated, uint seamBit, int x, int y, int z)
{
    if (seamBit == kExactNeighborPosYBit)
    {
        return encodeVoxel(kBlockAir, 15u, 0u);
    }

    // Mirror the chunk-edge lighting sample until the real neighbor is ready.
    // This keeps border lighting stable without pushing seam handling back to CPU.
    const uint index = voxelIndex(uint(x), uint(y), uint(z));
    return readPropagated ? sPropagatedVoxels[index] : sSeededVoxels[index];
}

uint sampleSeededOrNeighbor(bool readPropagated, int x, int y, int z)
{
    if (x >= 0 && y >= 0 && z >= 0 &&
        x < int(kExactChunkSize) && y < int(kExactChunkSize) && z < int(kExactChunkSize))
    {
        const uint index = voxelIndex(uint(x), uint(y), uint(z));
        return readPropagated ? sPropagatedVoxels[index] : sSeededVoxels[index];
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
    return sampleLightingFallback(readPropagated, seamBit, x, y, z);
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
    uint incomingSky = voxelSkyLight(sampleSeededOrNeighbor(false, int(localX), int(kExactChunkSize), int(localZ)));

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
            sSeededVoxels[index] = encodeVoxel(blockId, 0u, emission);
            continue;
        }

        incomingSky = attenuateLight(incomingSky, propagationLossForBlock(blockId));
        sSeededVoxels[index] = encodeVoxel(blockId, incomingSky, emission);
    }

    GroupMemoryBarrierWithGroupSync();

    [loop]
    for (uint passIndex = 0u; passIndex < gPropagationPassCount; ++passIndex)
    {
        const bool readPropagated = (passIndex & 1u) != 0u;
        [loop]
        for (uint localY = 0u; localY < kExactChunkSize; ++localY)
        {
            const uint index = voxelIndex(localX, localY, localZ);
            if (index >= gVoxelCount)
            {
                continue;
            }

            const uint packedCenter = readPropagated ? sPropagatedVoxels[index] : sSeededVoxels[index];
            const uint blockId = voxelBlock(packedCenter);
            const uint emission = blockEmissionForBlock(blockId);
            if (isOpaqueForLighting(blockId))
            {
                if (readPropagated)
                {
                    sSeededVoxels[index] = encodeVoxel(blockId, 0u, emission);
                }
                else
                {
                    sPropagatedVoxels[index] = encodeVoxel(blockId, 0u, emission);
                }
                continue;
            }

            const uint loss = propagationLossForBlock(blockId);
            uint skyLight = voxelSkyLight(packedCenter);
            uint blockLight = max(voxelBlockLight(packedCenter), emission);
            const int3 local = int3(int(localX), int(localY), int(localZ));
            const uint neighborSamples[6] = {
                sampleSeededOrNeighbor(readPropagated, local.x + 1, local.y, local.z),
                sampleSeededOrNeighbor(readPropagated, local.x - 1, local.y, local.z),
                sampleSeededOrNeighbor(readPropagated, local.x, local.y + 1, local.z),
                sampleSeededOrNeighbor(readPropagated, local.x, local.y - 1, local.z),
                sampleSeededOrNeighbor(readPropagated, local.x, local.y, local.z + 1),
                sampleSeededOrNeighbor(readPropagated, local.x, local.y, local.z - 1)
            };

            [unroll]
            for (uint i = 0u; i < 6u; ++i)
            {
                skyLight = max(skyLight, attenuateLight(voxelSkyLight(neighborSamples[i]), loss));
                blockLight = max(blockLight, attenuateLight(voxelBlockLight(neighborSamples[i]), loss));
            }

            const uint litVoxel = encodeVoxel(blockId, skyLight, blockLight);
            if (readPropagated)
            {
                sSeededVoxels[index] = litVoxel;
            }
            else
            {
                sPropagatedVoxels[index] = litVoxel;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    const bool finalInSeeded = (gPropagationPassCount & 1u) == 0u;
    [loop]
    for (uint localY = 0u; localY < kExactChunkSize; ++localY)
    {
        const uint index = voxelIndex(localX, localY, localZ);
        if (index >= gVoxelCount)
        {
            continue;
        }

        gDestVoxels[index] = finalInSeeded ? sSeededVoxels[index] : sPropagatedVoxels[index];
    }
}
