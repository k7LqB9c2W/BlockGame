#include "exact_chunk_common.hlsli"

cbuffer ExactChunkFaceCountParams : register(b0)
{
    uint gPlaneCount;
    uint gVoxelCount;
    uint gDescriptorCount;
    uint gResolvedNeighborMask;
    uint gClosedNeighborMask;
    uint gReserved0;
};

StructuredBuffer<uint> gCenterVoxels : register(t0);
StructuredBuffer<uint> gNeighborPosX : register(t1);
StructuredBuffer<uint> gNeighborNegX : register(t2);
StructuredBuffer<uint> gNeighborPosY : register(t3);
StructuredBuffer<uint> gNeighborNegY : register(t4);
StructuredBuffer<uint> gNeighborPosZ : register(t5);
StructuredBuffer<uint> gNeighborNegZ : register(t6);
RWStructuredBuffer<uint> gFaceCounts : register(u0);
RWStructuredBuffer<GpuExactFaceDescriptor> gFaceDescriptors : register(u1);

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
        if (seamBit == kExactNeighborPosXBit) return sampleVoxel(gNeighborPosX, x, y, z);
        if (seamBit == kExactNeighborNegXBit) return sampleVoxel(gNeighborNegX, x, y, z);
        if (seamBit == kExactNeighborPosYBit) return sampleVoxel(gNeighborPosY, x, y, z);
        if (seamBit == kExactNeighborNegYBit) return sampleVoxel(gNeighborNegY, x, y, z);
        if (seamBit == kExactNeighborPosZBit) return sampleVoxel(gNeighborPosZ, x, y, z);
        if (seamBit == kExactNeighborNegZBit) return sampleVoxel(gNeighborNegZ, x, y, z);
    }
    if ((gClosedNeighborMask & seamBit) != 0u)
    {
        return encodeVoxel(kBlockNeighborSolidSentinel, 0u, 0u);
    }
    return encodeVoxel(kBlockAir, 0u, 0u);
}

void decodePlane(uint planeIndex, out uint axis, out bool positiveFace, out uint slice)
{
    axis = planeIndex / 34u;
    const uint rem = planeIndex - axis * 34u;
    positiveFace = rem < 17u;
    slice = rem % 17u;
}

uint faceIdForAxis(uint axis, bool positiveFace)
{
    if (axis == 0u) return positiveFace ? 4u : 5u;
    if (axis == 1u) return positiveFace ? 0u : 1u;
    return positiveFace ? 3u : 2u;
}

[numthreads(64, 1, 1)]
void ExactChunkFaceCountMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint planeIndex = dispatchThreadId.x;
    if (planeIndex >= gPlaneCount)
    {
        return;
    }

    uint axis = 0u;
    bool positiveFace = true;
    uint slice = 0u;
    decodePlane(planeIndex, axis, positiveFace, slice);
    const uint faceId = faceIdForAxis(axis, positiveFace);
    uint descriptorCount = 0u;
    const uint descriptorBase = planeIndex * kExactChunkMaxDescriptorsPerPlane;

    [loop]
    for (uint b = 0u; b < kExactChunkSize; ++b)
    {
        [loop]
        for (uint c = 0u; c < kExactChunkSize; ++c)
        {
            int positiveX = 0;
            int positiveY = 0;
            int positiveZ = 0;
            int negativeX = 0;
            int negativeY = 0;
            int negativeZ = 0;

            if (axis == 0u)
            {
                positiveX = int(slice);
                negativeX = int(slice) - 1;
                positiveY = int(b);
                negativeY = int(b);
                positiveZ = int(c);
                negativeZ = int(c);
            }
            else if (axis == 1u)
            {
                positiveX = int(b);
                negativeX = int(b);
                positiveY = int(slice);
                negativeY = int(slice) - 1;
                positiveZ = int(c);
                negativeZ = int(c);
            }
            else
            {
                positiveX = int(b);
                negativeX = int(b);
                positiveY = int(c);
                negativeY = int(c);
                positiveZ = int(slice);
                negativeZ = int(slice) - 1;
            }

            const int owningX = positiveFace ? negativeX : positiveX;
            const int owningY = positiveFace ? negativeY : positiveY;
            const int owningZ = positiveFace ? negativeZ : positiveZ;
            if (owningX < 0 || owningY < 0 || owningZ < 0 ||
                owningX >= int(kExactChunkSize) || owningY >= int(kExactChunkSize) || owningZ >= int(kExactChunkSize))
            {
                continue;
            }

            const uint owningBlock = voxelBlock(sampleVoxelWithNeighbors(owningX, owningY, owningZ));
            const uint neighborBlock = positiveFace
                                           ? voxelBlock(sampleVoxelWithNeighbors(positiveX, positiveY, positiveZ))
                                           : voxelBlock(sampleVoxelWithNeighbors(negativeX, negativeY, negativeZ));
            if (!shouldRenderBlockFace(owningBlock, neighborBlock))
            {
                continue;
            }

            const uint descriptorIndex = descriptorBase + descriptorCount;
            if (descriptorIndex < gDescriptorCount)
            {
                GpuExactFaceDescriptor descriptor;
                descriptor.packedLocal = packFaceLocal(uint(owningX), uint(owningY), uint(owningZ), faceId);
                descriptor.reserved0 = 0u;
                descriptor.reserved1 = 0u;
                descriptor.reserved2 = 0u;
                gFaceDescriptors[descriptorIndex] = descriptor;
            }
            descriptorCount += 1u;
        }
    }

    gFaceCounts[planeIndex] = descriptorCount;
}
