cbuffer CullParams : register(b0)
{
    float4x4 gViewProj;
    float4 gFrustumPlanes[6];
    uint gRecordCount;
    uint gDepthWidth;
    uint gDepthHeight;
    uint gDepthMipCount;
};

struct GpuCullRecord
{
    float4 boundsMin;
    float4 boundsMax;
    uint indexCount;
    uint firstIndexLocation;
    int baseVertex;
    uint reserved;
};

struct DrawIndexedArgsGpu
{
    uint indexCountPerInstance;
    uint instanceCount;
    uint startIndexLocation;
    int baseVertexLocation;
    uint startInstanceLocation;
};

Texture2D<float> gDepthPyramid : register(t0);
StructuredBuffer<GpuCullRecord> gRecords : register(t1);
StructuredBuffer<uint> gVisibleIndices : register(t2);
RWStructuredBuffer<uint> gVisibleIndicesUav : register(u0);
RWStructuredBuffer<uint> gVisibleCount : register(u1);
RWStructuredBuffer<DrawIndexedArgsGpu> gIndirectArgs : register(u2);

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
    if (gDepthMipCount == 0u)
    {
        return true;
    }

    const float3 corners[8] = {
        float3(boundsMin.x, boundsMin.y, boundsMin.z),
        float3(boundsMax.x, boundsMin.y, boundsMin.z),
        float3(boundsMin.x, boundsMax.y, boundsMin.z),
        float3(boundsMax.x, boundsMax.y, boundsMin.z),
        float3(boundsMin.x, boundsMin.y, boundsMax.z),
        float3(boundsMax.x, boundsMin.y, boundsMax.z),
        float3(boundsMin.x, boundsMax.y, boundsMax.z),
        float3(boundsMax.x, boundsMax.y, boundsMax.z)};

    float2 pixelMin = float2((float)gDepthWidth, (float)gDepthHeight);
    float2 pixelMax = float2(0.0f, 0.0f);
    float minDepth = 1.0f;

    [unroll]
    for (uint cornerIndex = 0u; cornerIndex < 8u; ++cornerIndex)
    {
        const float4 clip = mul(gViewProj, float4(corners[cornerIndex], 1.0f));
        if (clip.w <= 0.0f)
        {
            return true;
        }

        const float invW = 1.0f / clip.w;
        const float3 ndc = clip.xyz * invW;
        const float2 pixel = (ndc.xy * float2(0.5f, -0.5f) + 0.5f) * float2((float)gDepthWidth, (float)gDepthHeight);
        pixelMin = min(pixelMin, pixel);
        pixelMax = max(pixelMax, pixel);
        minDepth = min(minDepth, saturate(ndc.z));
    }

    pixelMin = clamp(pixelMin, float2(0.0f, 0.0f), float2((float)gDepthWidth, (float)gDepthHeight));
    pixelMax = clamp(pixelMax, float2(0.0f, 0.0f), float2((float)gDepthWidth, (float)gDepthHeight));

    const float2 span = max(pixelMax - pixelMin, float2(1.0f, 1.0f));
    const float maxSpan = max(span.x, span.y);
    uint mipLevel = 0u;
    if (maxSpan > 1.0f)
    {
        mipLevel = min(gDepthMipCount - 1u, (uint)floor(log2(maxSpan)));
    }

    const uint mipWidth = max(1u, gDepthWidth >> mipLevel);
    const uint mipHeight = max(1u, gDepthHeight >> mipLevel);
    const uint minSampleX = min(mipWidth - 1u, (uint)floor(pixelMin.x * (float)mipWidth / max((float)gDepthWidth, 1.0f)));
    const uint minSampleY = min(mipHeight - 1u, (uint)floor(pixelMin.y * (float)mipHeight / max((float)gDepthHeight, 1.0f)));
    const uint maxSampleX = min(mipWidth - 1u, (uint)floor(max(pixelMax.x - 1.0f, pixelMin.x) * (float)mipWidth / max((float)gDepthWidth, 1.0f)));
    const uint maxSampleY = min(mipHeight - 1u, (uint)floor(max(pixelMax.y - 1.0f, pixelMin.y) * (float)mipHeight / max((float)gDepthHeight, 1.0f)));

    float occluderDepth = 0.0f;
    [loop]
    for (uint sampleY = minSampleY; sampleY <= maxSampleY; ++sampleY)
    {
        [loop]
        for (uint sampleX = minSampleX; sampleX <= maxSampleX; ++sampleX)
        {
            occluderDepth = max(occluderDepth, gDepthPyramid.Load(int3(sampleX, sampleY, mipLevel)));
        }
    }

    return minDepth <= occluderDepth + 1.0e-4f;
}

[numthreads(64, 1, 1)]
void LodCullMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint recordIndex = dispatchThreadId.x;
    if (recordIndex >= gRecordCount)
    {
        return;
    }

    const GpuCullRecord record = gRecords[recordIndex];
    if (record.indexCount == 0u)
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
void LodIndirectBuildMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint visibleIndex = dispatchThreadId.x;
    const uint visibleCount = gVisibleCount[0];
    if (visibleIndex >= visibleCount)
    {
        return;
    }

    const GpuCullRecord record = gRecords[gVisibleIndices[visibleIndex]];
    DrawIndexedArgsGpu args;
    args.indexCountPerInstance = record.indexCount;
    args.instanceCount = 1u;
    args.startIndexLocation = record.firstIndexLocation;
    args.baseVertexLocation = record.baseVertex;
    args.startInstanceLocation = 0u;
    gIndirectArgs[visibleIndex] = args;
}
