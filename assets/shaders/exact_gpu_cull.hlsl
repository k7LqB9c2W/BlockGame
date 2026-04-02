cbuffer CullParams : register(b0)
{
    float4x4 gViewProj;
    float4 gFrustumPlanes[6];
    uint gRecordCount;
    uint gDepthWidth;
    uint gDepthHeight;
    uint gDepthMipCount;
};

static const uint kExactDrawRecordActiveBit = 0x40000000u;

struct GpuExactDrawRecord
{
    float4 boundsMin;
    float4 boundsMax;
    uint faceCount;
    uint faceOffset;
    uint reserved0;
    uint reserved;
};

struct DrawInstancedArgsGpu
{
    uint vertexCountPerInstance;
    uint instanceCount;
    uint startVertexLocation;
    uint startInstanceLocation;
};

struct ExactIndirectCommand
{
    uint drawRecordIndex;
    DrawInstancedArgsGpu args;
};

Texture2D<float> gDepthPyramid : register(t0);
StructuredBuffer<GpuExactDrawRecord> gRecords : register(t1);
StructuredBuffer<uint> gVisibleIndices : register(t2);
RWStructuredBuffer<uint> gVisibleIndicesUav : register(u0);
RWStructuredBuffer<uint> gVisibleCount : register(u1);
RWStructuredBuffer<ExactIndirectCommand> gIndirectArgs : register(u2);

bool intersectsFrustum(float3 boundsMin, float3 boundsMax)
{
    [unroll]
    for (uint planeIndex = 0u; planeIndex < 6u; ++planeIndex)
    {
        const float3 normal = gFrustumPlanes[planeIndex].xyz;
        const float3 positiveVertex = float3(
            (normal.x >= 0.0f) ? boundsMax.x : boundsMin.x,
            (normal.y >= 0.0f) ? boundsMax.y : boundsMin.y,
            (normal.z >= 0.0f) ? boundsMax.z : boundsMin.z);
        if (dot(normal, positiveVertex) + gFrustumPlanes[planeIndex].w < 0.0f)
        {
            return false;
        }
    }
    return true;
}

bool passesOcclusion(float3 boundsMin, float3 boundsMax)
{
    (void)boundsMin;
    (void)boundsMax;
    return true;
}

[numthreads(64, 1, 1)]
void ExactCullMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint recordIndex = dispatchThreadId.x;
    if (recordIndex >= gRecordCount)
    {
        return;
    }

    const GpuExactDrawRecord record = gRecords[recordIndex];
    if ((record.reserved & kExactDrawRecordActiveBit) == 0u || record.faceCount == 0u)
    {
        return;
    }
    if (!intersectsFrustum(record.boundsMin.xyz, record.boundsMax.xyz))
    {
        return;
    }
    if (!passesOcclusion(record.boundsMin.xyz, record.boundsMax.xyz))
    {
        return;
    }

    uint visibleIndex = 0u;
    InterlockedAdd(gVisibleCount[0], 1u, visibleIndex);
    gVisibleIndicesUav[visibleIndex] = recordIndex;
}

[numthreads(64, 1, 1)]
void ExactIndirectBuildMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint visibleIndex = dispatchThreadId.x;
    const uint visibleCount = gVisibleCount[0];
    if (visibleIndex >= visibleCount)
    {
        return;
    }

    const uint recordIndex = gVisibleIndices[visibleIndex];
    const GpuExactDrawRecord record = gRecords[recordIndex];
    ExactIndirectCommand command;
    command.drawRecordIndex = recordIndex;
    command.args.vertexCountPerInstance = 6u;
    command.args.instanceCount = record.faceCount;
    command.args.startVertexLocation = 0u;
    command.args.startInstanceLocation = record.faceOffset;
    gIndirectArgs[visibleIndex] = command;
}
