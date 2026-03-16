StructuredBuffer<uint> gInput : register(t0);
RWStructuredBuffer<uint> gOutput0 : register(u0);
RWStructuredBuffer<uint> gOutput1 : register(u1);

static const uint kVoxelCount = 4096u;
static const uint kGroupSize = 256u;
static const uint kGroupCount = kVoxelCount / kGroupSize;

groupshared uint sScan[kGroupSize];
groupshared uint sOriginal[kGroupSize];

[numthreads(256, 1, 1)]
void FarLodChunkFacePrefixGroupMain(uint3 groupId : SV_GroupID,
                                    uint3 groupThreadId : SV_GroupThreadID,
                                    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint linearIndex = dispatchThreadId.x;
    const uint localIndex = groupThreadId.x;
    const uint count = (linearIndex < kVoxelCount) ? gInput[linearIndex] : 0u;
    sScan[localIndex] = count;
    sOriginal[localIndex] = count;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint offset = 1u; offset < kGroupSize; offset <<= 1u)
    {
        uint addend = 0u;
        if (localIndex >= offset)
        {
            addend = sScan[localIndex - offset];
        }
        GroupMemoryBarrierWithGroupSync();
        sScan[localIndex] += addend;
        GroupMemoryBarrierWithGroupSync();
    }

    if (linearIndex < kVoxelCount)
    {
        gOutput0[linearIndex] = sScan[localIndex] - sOriginal[localIndex];
    }

    if (localIndex == (kGroupSize - 1u))
    {
        gOutput1[groupId.x] = sScan[localIndex];
    }
}

[numthreads(256, 1, 1)]
void FarLodChunkFacePrefixScanMain(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint localIndex = groupThreadId.x;
    sScan[localIndex] = (localIndex < kGroupCount) ? gOutput0[localIndex] : 0u;
    sOriginal[localIndex] = sScan[localIndex];
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint offset = 1u; offset < kGroupSize; offset <<= 1u)
    {
        uint addend = 0u;
        if (localIndex >= offset)
        {
            addend = sScan[localIndex - offset];
        }
        GroupMemoryBarrierWithGroupSync();
        sScan[localIndex] += addend;
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex < kGroupCount)
    {
        gOutput0[localIndex] = sScan[localIndex] - sOriginal[localIndex];
    }
}

[numthreads(256, 1, 1)]
void FarLodChunkFacePrefixAddMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint linearIndex = dispatchThreadId.x;
    if (linearIndex >= kVoxelCount)
    {
        return;
    }

    const uint groupIndex = linearIndex / kGroupSize;
    gOutput0[linearIndex] += gInput[groupIndex];
}
