#include "exact_chunk_common.hlsli"

cbuffer ExactChunkFacePrefixParams : register(b0)
{
    uint gBatchBuildCount;
    uint gPlaneCount;
    uint gReserved0;
    uint gReserved1;
};

StructuredBuffer<GpuExactPrepassRecord> gPrepassRecords : register(t0);
StructuredBuffer<GpuExactColumnDescriptor> gDescriptorScratch : register(t1);
RWStructuredBuffer<uint> gFaceCounts : register(u0);
RWStructuredBuffer<GpuExactFaceDescriptor> gFaceDescriptorScratch : register(u1);
RWStructuredBuffer<uint> gFacePrefixes : register(u2);
RWStructuredBuffer<uint> gFaceTotals : register(u3);

static const uint kExactIndirectRootBufferAlignment = 256u;
static const uint kExactFaceCountScratchStride =
    (((kExactChunkPlaneCount * 4u) + kExactIndirectRootBufferAlignment - 1u) / kExactIndirectRootBufferAlignment) *
    (kExactIndirectRootBufferAlignment / 4u);
static const uint kExactFacePrefixScratchStride = kExactFaceCountScratchStride;
static const uint kExactFaceTotalScratchStride =
    ((4u + kExactIndirectRootBufferAlignment - 1u) / kExactIndirectRootBufferAlignment) *
    (kExactIndirectRootBufferAlignment / 4u);

[numthreads(1, 1, 1)]
void ExactChunkFacePrefixMain(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    if (groupThreadId.x != 0u || groupThreadId.y != 0u || groupThreadId.z != 0u)
    {
        return;
    }

    const uint buildIndex = groupId.y;
    if (buildIndex >= gBatchBuildCount)
    {
        return;
    }

    const GpuExactPrepassRecord build = gPrepassRecords[buildIndex];
    const uint countBase = build.scratchSliceIndex * kExactFaceCountScratchStride;
    const uint prefixBase = build.scratchSliceIndex * kExactFacePrefixScratchStride;
    const uint totalBase = build.scratchSliceIndex * kExactFaceTotalScratchStride;

    uint running = 0u;
    [loop]
    for (uint planeIndex = 0u; planeIndex < gPlaneCount; ++planeIndex)
    {
        gFacePrefixes[prefixBase + planeIndex] = running;
        running += gFaceCounts[countBase + planeIndex];
    }
    gFaceTotals[totalBase] = running;
}
