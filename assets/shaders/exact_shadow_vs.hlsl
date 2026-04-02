#include "exact_chunk_common.hlsli"

cbuffer ShadowConstants : register(b0)
{
    float4x4 uLightViewProj;
};

cbuffer ExactDrawRecordConstants : register(b1)
{
    uint gDrawRecordIndex;
};

StructuredBuffer<GpuExactFaceDescriptor> gFaceDescriptors : register(t0);
StructuredBuffer<GpuExactDrawRecordMetadata> gDrawRecordMetadata : register(t1);
StructuredBuffer<GpuBlockFaceUv> gBlockFaceUvs : register(t2);

struct VSOutput
{
    float4 position : SV_POSITION;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
};

uint faceCornerIndex(uint vertexId, bool flipDiagonal)
{
    if (flipDiagonal)
    {
        static const uint kFlipCorners[6] = {0u, 1u, 3u, 1u, 2u, 3u};
        return kFlipCorners[min(vertexId, 5u)];
    }

    static const uint kCorners[6] = {0u, 1u, 2u, 0u, 2u, 3u};
    return kCorners[min(vertexId, 5u)];
}

float3 faceCornerPosition(float3 base, uint faceId, uint cornerIndex)
{
    if (faceId == 0u)
    {
        static const float3 kCorners[4] = {
            float3(0.0f, 1.0f, 0.0f),
            float3(0.0f, 1.0f, 1.0f),
            float3(1.0f, 1.0f, 1.0f),
            float3(1.0f, 1.0f, 0.0f)};
        return base + kCorners[min(cornerIndex, 3u)];
    }
    if (faceId == 1u)
    {
        static const float3 kCorners[4] = {
            float3(0.0f, 0.0f, 0.0f),
            float3(1.0f, 0.0f, 0.0f),
            float3(1.0f, 0.0f, 1.0f),
            float3(0.0f, 0.0f, 1.0f)};
        return base + kCorners[min(cornerIndex, 3u)];
    }
    if (faceId == 2u)
    {
        static const float3 kCorners[4] = {
            float3(0.0f, 0.0f, 0.0f),
            float3(0.0f, 1.0f, 0.0f),
            float3(1.0f, 1.0f, 0.0f),
            float3(1.0f, 0.0f, 0.0f)};
        return base + kCorners[min(cornerIndex, 3u)];
    }
    if (faceId == 3u)
    {
        static const float3 kCorners[4] = {
            float3(0.0f, 0.0f, 1.0f),
            float3(1.0f, 0.0f, 1.0f),
            float3(1.0f, 1.0f, 1.0f),
            float3(0.0f, 1.0f, 1.0f)};
        return base + kCorners[min(cornerIndex, 3u)];
    }
    if (faceId == 4u)
    {
        static const float3 kCorners[4] = {
            float3(1.0f, 0.0f, 0.0f),
            float3(1.0f, 1.0f, 0.0f),
            float3(1.0f, 1.0f, 1.0f),
            float3(1.0f, 0.0f, 1.0f)};
        return base + kCorners[min(cornerIndex, 3u)];
    }

    static const float3 kCorners[4] = {
        float3(0.0f, 0.0f, 0.0f),
        float3(0.0f, 0.0f, 1.0f),
        float3(0.0f, 1.0f, 1.0f),
        float3(0.0f, 1.0f, 0.0f)};
    return base + kCorners[min(cornerIndex, 3u)];
}

VSOutput main(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    const GpuExactFaceDescriptor descriptor = gFaceDescriptors[instanceId];
    const GpuExactDrawRecordMetadata metadata = gDrawRecordMetadata[gDrawRecordIndex];
    const uint localX = faceLocalX(descriptor.packedLocal);
    const uint localY = faceLocalY(descriptor.packedLocal);
    const uint localZ = faceLocalZ(descriptor.packedLocal);
    const uint faceId = faceLocalFaceId(descriptor.packedLocal);
    const bool flipDiagonal = (descriptor.packedLocal & (1u << 31u)) != 0u;
    const uint cornerIndex = faceCornerIndex(vertexId, flipDiagonal);
    const float3 base = float3(float(metadata.chunkWorldMinX + int(localX)),
                               float(metadata.chunkWorldMinY + int(localY)),
                               float(metadata.chunkWorldMinZ + int(localZ)));
    const float3 worldPos = faceCornerPosition(base, faceId, cornerIndex);
    const GpuBlockFaceUv uv = gBlockFaceUvs[descriptor.blockFaceUvIndex];

    VSOutput output;
    output.position = mul(uLightViewProj, float4(worldPos, 1.0f));
    output.tileCoord = projectTileCoord(faceId, worldPos);
    output.atlasBase = uv.base;
    output.atlasSize = uv.size;
    return output;
}
