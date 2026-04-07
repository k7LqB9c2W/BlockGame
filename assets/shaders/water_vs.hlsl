#include "world_lighting_common.hlsli"

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
    float4 uTranslucencyParams;
};

struct GpuWaterQuadDescriptor
{
    float4 originTop;
    float4 axisUBottom;
    float4 axisVFaceKind;
    float4 normalAtlasBaseX;
    float4 atlasRest;
};

StructuredBuffer<GpuWaterQuadDescriptor> gWaterQuads : register(t0);

struct VSOutput
{
    float4 position : SV_POSITION;
    float3 worldPos : POSITION0;
    float3 normal : NORMAL0;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
    float topY : TEXCOORD3;
    float bottomY : TEXCOORD4;
    uint faceKind : TEXCOORD5;
};

uint quadCornerIndex(uint vertexId)
{
    static const uint kCorners[6] = {0u, 1u, 2u, 0u, 2u, 3u};
    return kCorners[min(vertexId, 5u)];
}

float2 projectWaterTileCoord(float3 worldPos, float3 normal)
{
    const float3 absNormal = abs(normal);
    if (absNormal.y >= absNormal.x && absNormal.y >= absNormal.z)
    {
        return worldPos.xz;
    }
    if (absNormal.x >= absNormal.z)
    {
        return float2(worldPos.z, worldPos.y);
    }
    return float2(worldPos.x, worldPos.y);
}

VSOutput main(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    const GpuWaterQuadDescriptor descriptor = gWaterQuads[instanceId];
    const uint cornerIndex = quadCornerIndex(vertexId);
    const float3 origin = descriptor.originTop.xyz;
    const float3 axisU = descriptor.axisUBottom.xyz;
    const float3 axisV = descriptor.axisVFaceKind.xyz;

    float3 worldPos = origin;
    if (cornerIndex == 1u)
    {
        worldPos += axisU;
    }
    else if (cornerIndex == 2u)
    {
        worldPos += axisU + axisV;
    }
    else if (cornerIndex == 3u)
    {
        worldPos += axisV;
    }

    VSOutput output;
    output.position = mul(uViewProj, float4(worldPos, 1.0f));
    output.worldPos = worldPos;
    output.normal = normalize(descriptor.normalAtlasBaseX.xyz);
    output.tileCoord = projectWaterTileCoord(worldPos, output.normal);
    output.atlasBase = float2(descriptor.normalAtlasBaseX.w, descriptor.atlasRest.x);
    output.atlasSize = descriptor.atlasRest.yz;
    output.topY = descriptor.originTop.w;
    output.bottomY = descriptor.axisUBottom.w;
    output.faceKind = (uint)descriptor.axisVFaceKind.w;
    return output;
}
