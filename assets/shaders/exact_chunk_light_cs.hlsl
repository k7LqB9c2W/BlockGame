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
StructuredBuffer<uint> gHaloVoxels : register(t2);
RWStructuredBuffer<uint> gDestVoxels : register(u0);

static const uint kVoxelPayloadMask = 0x0000FFFFu;
static const uint kQueueFlag = 0x80000000u;
static const uint kFrontierStateIndex = 0u;
static const uint kFrontierStorageOffset = 1u;
static const uint kFrontierCapacity = kExactChunkVoxelCount - kFrontierStorageOffset;

groupshared uint sLitVoxels[kExactChunkVoxelCount];
groupshared uint sFrontierQueue[kExactChunkVoxelCount];

uint sampleVoxel(StructuredBuffer<uint> bufferRef, int x, int y, int z)
{
    if (x < 0 || y < 0 || z < 0 ||
        x >= int(kExactChunkSize) || y >= int(kExactChunkSize) || z >= int(kExactChunkSize))
    {
        return encodeVoxel(kBlockAir, 0u, 0u);
    }

    return bufferRef[voxelIndex(uint(x), uint(y), uint(z))];
}

uint sampleNeighborForSeed(uint seamBit, int x, int y, int z)
{
    return sampleHaloVoxel(gHaloVoxels, seamBit, x, y, z);
}

uint loadLitVoxel(uint index)
{
    return sLitVoxels[index] & kVoxelPayloadMask;
}

void storeLitVoxel(uint index, uint packedVoxel)
{
    sLitVoxels[index] = (sLitVoxels[index] & ~kVoxelPayloadMask) | (packedVoxel & kVoxelPayloadMask);
}

void clearQueuedFlag(uint index)
{
    uint oldValue = 0u;
    InterlockedAnd(sLitVoxels[index], ~kQueueFlag, oldValue);
}

uint queueHead(uint packedState)
{
    return packedState & 0xFFFFu;
}

uint queueTail(uint packedState)
{
    return (packedState >> 16u) & 0xFFFFu;
}

uint queueSlotIndex(uint slot)
{
    return kFrontierStorageOffset + (slot % kFrontierCapacity);
}

void setQueueHead(uint head)
{
    while (true)
    {
        const uint currentState = sFrontierQueue[kFrontierStateIndex];
        const uint desiredState = (currentState & 0xFFFF0000u) | (head & 0xFFFFu);
        uint previousState = 0u;
        InterlockedCompareExchange(sFrontierQueue[kFrontierStateIndex],
                                   currentState,
                                   desiredState,
                                   previousState);
        if (previousState == currentState)
        {
            return;
        }
    }
}

void enqueueVoxel(uint index)
{
    while (true)
    {
        const uint currentRaw = sLitVoxels[index];
        if ((currentRaw & kQueueFlag) != 0u)
        {
            return;
        }

        const uint desiredRaw = currentRaw | kQueueFlag;
        uint previousRaw = 0u;
        InterlockedCompareExchange(sLitVoxels[index], currentRaw, desiredRaw, previousRaw);
        if (previousRaw == currentRaw)
        {
            uint previousState = 0u;
            InterlockedAdd(sFrontierQueue[kFrontierStateIndex], 1u << 16u, previousState);
            const uint slot = queueTail(previousState);
            sFrontierQueue[queueSlotIndex(slot)] = index;
            return;
        }
    }
}

uint seedBoundaryLight(uint3 localPos, uint packedBase)
{
    const uint blockId = voxelBlock(packedBase);
    if (isOpaqueForLighting(blockId))
    {
        return packedBase;
    }

    const uint loss = propagationLossForBlock(blockId);
    uint skyLight = voxelSkyLight(packedBase);
    uint blockLight = voxelBlockLight(packedBase);

    if (localPos.x == 0u)
    {
        const uint neighbor = sampleNeighborForSeed(kExactNeighborNegXBit,
                                                    int(kExactChunkSize) - 1,
                                                    int(localPos.y),
                                                    int(localPos.z));
        skyLight = max(skyLight, attenuateLight(voxelSkyLight(neighbor), loss));
        blockLight = max(blockLight, attenuateLight(voxelBlockLight(neighbor), loss));
    }
    if (localPos.x + 1u == kExactChunkSize)
    {
        const uint neighbor = sampleNeighborForSeed(kExactNeighborPosXBit,
                                                    0,
                                                    int(localPos.y),
                                                    int(localPos.z));
        skyLight = max(skyLight, attenuateLight(voxelSkyLight(neighbor), loss));
        blockLight = max(blockLight, attenuateLight(voxelBlockLight(neighbor), loss));
    }
    if (localPos.y == 0u)
    {
        const uint neighbor = sampleNeighborForSeed(kExactNeighborNegYBit,
                                                    int(localPos.x),
                                                    int(kExactChunkSize) - 1,
                                                    int(localPos.z));
        skyLight = max(skyLight, attenuateLight(voxelSkyLight(neighbor), loss));
        blockLight = max(blockLight, attenuateLight(voxelBlockLight(neighbor), loss));
    }
    if (localPos.y + 1u == kExactChunkSize)
    {
        const uint neighbor = sampleNeighborForSeed(kExactNeighborPosYBit,
                                                    int(localPos.x),
                                                    0,
                                                    int(localPos.z));
        skyLight = max(skyLight, attenuateLight(voxelSkyLight(neighbor), loss));
        blockLight = max(blockLight, attenuateLight(voxelBlockLight(neighbor), loss));
    }
    if (localPos.z == 0u)
    {
        const uint neighbor = sampleNeighborForSeed(kExactNeighborNegZBit,
                                                    int(localPos.x),
                                                    int(localPos.y),
                                                    int(kExactChunkSize) - 1);
        skyLight = max(skyLight, attenuateLight(voxelSkyLight(neighbor), loss));
        blockLight = max(blockLight, attenuateLight(voxelBlockLight(neighbor), loss));
    }
    if (localPos.z + 1u == kExactChunkSize)
    {
        const uint neighbor = sampleNeighborForSeed(kExactNeighborPosZBit,
                                                    int(localPos.x),
                                                    int(localPos.y),
                                                    0);
        skyLight = max(skyLight, attenuateLight(voxelSkyLight(neighbor), loss));
        blockLight = max(blockLight, attenuateLight(voxelBlockLight(neighbor), loss));
    }

    return encodeVoxel(blockId, skyLight, blockLight);
}

bool tryRaiseNeighborLight(uint neighborIndex, uint sourcePacked)
{
    while (true)
    {
        const uint currentRaw = sLitVoxels[neighborIndex];
        const uint currentPacked = currentRaw & kVoxelPayloadMask;
        const uint blockId = voxelBlock(currentPacked);
        if (isOpaqueForLighting(blockId))
        {
            return false;
        }

        const uint loss = propagationLossForBlock(blockId);
        if (loss >= 15u &&
            attenuateLight(voxelSkyLight(sourcePacked), loss) == 0u &&
            attenuateLight(voxelBlockLight(sourcePacked), loss) == 0u)
        {
            return false;
        }

        const uint nextSky = max(voxelSkyLight(currentPacked),
                                 attenuateLight(voxelSkyLight(sourcePacked), loss));
        const uint nextBlock = max(voxelBlockLight(currentPacked),
                                   attenuateLight(voxelBlockLight(sourcePacked), loss));
        if (nextSky == voxelSkyLight(currentPacked) &&
            nextBlock == voxelBlockLight(currentPacked))
        {
            return false;
        }

        const uint updatedPacked = encodeVoxel(blockId, nextSky, nextBlock);
        const uint updatedRaw = (currentRaw & ~kVoxelPayloadMask) | updatedPacked;
        uint previousRaw = 0u;
        InterlockedCompareExchange(sLitVoxels[neighborIndex], currentRaw, updatedRaw, previousRaw);
        if (previousRaw == currentRaw)
        {
            return true;
        }
    }
}

void propagateFromVoxel(uint sourceIndex)
{
    const uint sourcePacked = loadLitVoxel(sourceIndex);
    if (voxelSkyLight(sourcePacked) <= 1u && voxelBlockLight(sourcePacked) <= 1u)
    {
        return;
    }

    const uint localX = sourceIndex & 0x0Fu;
    const uint localZ = (sourceIndex >> 4u) & 0x0Fu;
    const uint localY = sourceIndex >> 8u;

    if (localX > 0u)
    {
        const uint neighborIndex = sourceIndex - 1u;
        if (tryRaiseNeighborLight(neighborIndex, sourcePacked))
        {
            enqueueVoxel(neighborIndex);
        }
    }
    if (localX + 1u < kExactChunkSize)
    {
        const uint neighborIndex = sourceIndex + 1u;
        if (tryRaiseNeighborLight(neighborIndex, sourcePacked))
        {
            enqueueVoxel(neighborIndex);
        }
    }
    if (localZ > 0u)
    {
        const uint neighborIndex = sourceIndex - kExactChunkSize;
        if (tryRaiseNeighborLight(neighborIndex, sourcePacked))
        {
            enqueueVoxel(neighborIndex);
        }
    }
    if (localZ + 1u < kExactChunkSize)
    {
        const uint neighborIndex = sourceIndex + kExactChunkSize;
        if (tryRaiseNeighborLight(neighborIndex, sourcePacked))
        {
            enqueueVoxel(neighborIndex);
        }
    }
    if (localY > 0u)
    {
        const uint neighborIndex = sourceIndex - (kExactChunkSize * kExactChunkSize);
        if (tryRaiseNeighborLight(neighborIndex, sourcePacked))
        {
            enqueueVoxel(neighborIndex);
        }
    }
    if (localY + 1u < kExactChunkSize)
    {
        const uint neighborIndex = sourceIndex + (kExactChunkSize * kExactChunkSize);
        if (tryRaiseNeighborLight(neighborIndex, sourcePacked))
        {
            enqueueVoxel(neighborIndex);
        }
    }
}

[numthreads(16, 16, 1)]
void ExactChunkLightMain(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint linearThreadIndex = groupThreadId.y * kExactChunkSize + groupThreadId.x;
    if (groupThreadId.x >= kExactChunkSize || groupThreadId.y >= kExactChunkSize)
    {
        return;
    }

    if (linearThreadIndex == 0u)
    {
        sFrontierQueue[kFrontierStateIndex] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

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
            seededVoxel = seedBoundaryLight(uint3(localX, uint(localY), localZ), seededVoxel);
        }

        sLitVoxels[index] = seededVoxel;
        if (!isOpaqueForLighting(blockId) &&
            (voxelSkyLight(seededVoxel) > 1u || voxelBlockLight(seededVoxel) > 1u))
        {
            sLitVoxels[index] |= kQueueFlag;
            uint previousState = 0u;
            InterlockedAdd(sFrontierQueue[kFrontierStateIndex], 1u << 16u, previousState);
            const uint slot = queueTail(previousState);
            sFrontierQueue[queueSlotIndex(slot)] = index;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    while (true)
    {
        GroupMemoryBarrierWithGroupSync();
        const uint queueState = sFrontierQueue[kFrontierStateIndex];
        const uint roundBegin = queueHead(queueState);
        const uint roundEnd = queueTail(queueState);

        if (roundBegin >= roundEnd)
        {
            break;
        }

        for (uint slot = roundBegin + linearThreadIndex; slot < roundEnd; slot += kExactChunkColumnCount)
        {
            const uint sourceIndex = sFrontierQueue[queueSlotIndex(slot)];
            clearQueuedFlag(sourceIndex);
            propagateFromVoxel(sourceIndex);
        }

        GroupMemoryBarrierWithGroupSync();
        if (linearThreadIndex == 0u)
        {
            setQueueHead(roundEnd);
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

        gDestVoxels[index] = loadLitVoxel(index);
    }
}
