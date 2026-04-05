// exact_chunk_draw_record_clear_cs.hlsl
// Clears the ExactChunkRenderBatch::GpuCullRecord.reserved field for a list of record indices.

struct ExactGpuCullRecord
{
    float4 boundsMin;
    float4 boundsMax;
    uint faceCount;
    uint faceOffset;
    uint reserved0;
    uint reserved;
};

cbuffer ClearConstants : register(b0)
{
    uint gClearCount;
    uint gPad0;
    uint gPad1;
    uint gPad2;
};

StructuredBuffer<uint> gRecordIndices : register(t0);
RWStructuredBuffer<ExactGpuCullRecord> gDrawRecords : register(u0);

[numthreads(64, 1, 1)]
void ExactChunkDrawRecordClearMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gClearCount)
    {
        return;
    }

    const uint recordIndex = gRecordIndices[i];
    gDrawRecords[recordIndex].reserved = 0u;
}

