#include "exact_chunk_common.hlsli"

cbuffer ExactChunkFacePrefixParams : register(b0)
{
    uint gPlaneCount;
    uint gReserved0;
    uint gReserved1;
    uint gReserved2;
};

RWStructuredBuffer<uint> gFaceCounts : register(u0);
RWStructuredBuffer<uint> gFacePrefixes : register(u1);
RWStructuredBuffer<uint> gFaceTotals : register(u2);

[numthreads(1, 1, 1)]
void ExactChunkFacePrefixMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x != 0u)
    {
        return;
    }

    uint running = 0u;
    [loop]
    for (uint planeIndex = 0u; planeIndex < gPlaneCount; ++planeIndex)
    {
        gFacePrefixes[planeIndex] = running;
        running += gFaceCounts[planeIndex];
    }
    gFaceTotals[0] = running;
}
