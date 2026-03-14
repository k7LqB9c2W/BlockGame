cbuffer StampParams : register(b0)
{
    uint gStampCount;
};

struct GpuStructureStampEntry
{
    uint voxelIndex;
    uint packedVoxel;
};

StructuredBuffer<GpuStructureStampEntry> gStampEntries : register(t0);
RWStructuredBuffer<uint> gVoxelBuffer : register(u0);

[numthreads(64, 1, 1)]
void FarLodChunkStampMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    if (index >= gStampCount)
    {
        return;
    }

    const GpuStructureStampEntry entry = gStampEntries[index];
    gVoxelBuffer[entry.voxelIndex] = entry.packedVoxel;
}
