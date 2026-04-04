#include "exact_chunk_common.hlsli"

struct DecodedVertexLighting
{
    float sky;
    float block;
    float ao;
    float aoDebug;
    uint flags;
    uint alphaCutout;
    float farVoxelScale;
};

static const float kVanillaLightLut[16] = {
    0.000f, 0.018f, 0.025f, 0.035f,
    0.048f, 0.065f, 0.088f, 0.118f,
    0.159f, 0.214f, 0.288f, 0.387f,
    0.520f, 0.690f, 0.845f, 1.000f
};

static const float kAoFactors[4] = {1.00f, 0.86f, 0.73f, 0.60f};

float decodeNonLinearLightLevel(uint level)
{
    return kVanillaLightLut[min(level, 15u)];
}

float decodeAoFactor(uint aoLevel)
{
    return kAoFactors[min(aoLevel, 3u)];
}

DecodedVertexLighting decodeVertexLighting(uint packedLighting)
{
    DecodedVertexLighting decoded;
    const uint packedLight = packedLighting & 0xFFu;
    const uint skyLevel = (packedLight >> 4) & 0xFu;
    const uint blockLevel = packedLight & 0xFu;
    const uint aoLevel = (packedLighting >> 8) & 0x3u;
    decoded.sky = decodeNonLinearLightLevel(skyLevel);
    decoded.block = decodeNonLinearLightLevel(blockLevel);
    decoded.ao = decodeAoFactor(aoLevel);
    decoded.aoDebug = decoded.ao;
    decoded.flags = (packedLighting >> 10) & 0x3Fu;
    decoded.alphaCutout = (packedLighting >> 24) & 0x1u;
    const uint scale = (packedLighting >> 16) & 0xFFu;
    decoded.farVoxelScale = (scale > 0u) ? (float)scale : 1.0f;
    return decoded;
}

cbuffer WorldConstants : register(b0)
{
    float4x4 uViewProj;
    float4x4 uShadowViewProj;
    float4 uLightDirection;
    float4 uCameraPos;
    float4 uHighlightedBlock;
    float4 uParams0;
    float4 uParams1;
    float4 uSunColor;
    float4 uSkyAmbient;
    float4 uGroundAmbient;
    float4 uSkyTopColor;
    float4 uSkyHorizonColor;
    float4 uShadowParams;
    float4 uTerrainDebug;
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
    float3 worldPos : POSITION0;
    float3 normal : NORMAL0;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
    float2 lightChannels : TEXCOORD3;
    float ao : TEXCOORD4;
    uint materialFlags : TEXCOORD5;
    float farVoxelScale : TEXCOORD6;
    uint alphaCutout : TEXCOORD7;
    uint blockId : TEXCOORD8;
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

float3 faceNormal(uint faceId)
{
    if (faceId == 0u) return float3(0.0f, 1.0f, 0.0f);
    if (faceId == 1u) return float3(0.0f, -1.0f, 0.0f);
    if (faceId == 2u) return float3(0.0f, 0.0f, -1.0f);
    if (faceId == 3u) return float3(0.0f, 0.0f, 1.0f);
    if (faceId == 4u) return float3(1.0f, 0.0f, 0.0f);
    return float3(-1.0f, 0.0f, 0.0f);
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

uint faceLighting(uint cornerIndex, GpuExactFaceDescriptor descriptor)
{
    if (cornerIndex == 0u) return descriptor.packedLighting0;
    if (cornerIndex == 1u) return descriptor.packedLighting1;
    if (cornerIndex == 2u) return descriptor.packedLighting2;
    return descriptor.packedLighting3;
}

VSOutput main(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    const GpuExactDrawRecordMetadata metadata = gDrawRecordMetadata[gDrawRecordIndex];
    const GpuExactFaceDescriptor descriptor = gFaceDescriptors[metadata.faceBase + instanceId];
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
    const uint packedLighting = faceLighting(cornerIndex, descriptor);
    const DecodedVertexLighting decodedLighting = decodeVertexLighting(packedLighting);
    const GpuBlockFaceUv uv = gBlockFaceUvs[descriptor.blockFaceUvIndex];

    VSOutput output;
    output.worldPos = worldPos;
    output.normal = faceNormal(faceId);
    output.tileCoord = projectTileCoord(faceId, worldPos);
    output.atlasBase = uv.base;
    output.atlasSize = uv.size;
    output.lightChannels = float2(decodedLighting.sky, decodedLighting.block);
    output.ao = decodedLighting.ao;
    output.materialFlags = decodedLighting.flags;
    output.farVoxelScale = 1.0f;
    output.alphaCutout = decodedLighting.alphaCutout;
    output.blockId = descriptor.blockId;
    output.position = mul(uViewProj, float4(worldPos, 1.0f));
    return output;
}
