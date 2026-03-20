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
};

struct VSOutput
{
    float4 position : SV_POSITION;
};

static const float3 kCubeCorners[8] = {
    float3(0.0f, 0.0f, 0.0f),
    float3(1.0f, 0.0f, 0.0f),
    float3(1.0f, 1.0f, 0.0f),
    float3(0.0f, 1.0f, 0.0f),
    float3(0.0f, 0.0f, 1.0f),
    float3(1.0f, 0.0f, 1.0f),
    float3(1.0f, 1.0f, 1.0f),
    float3(0.0f, 1.0f, 1.0f),
};

static const uint2 kCubeEdges[12] = {
    uint2(0u, 1u),
    uint2(1u, 2u),
    uint2(2u, 3u),
    uint2(3u, 0u),
    uint2(4u, 5u),
    uint2(5u, 6u),
    uint2(6u, 7u),
    uint2(7u, 4u),
    uint2(0u, 4u),
    uint2(1u, 5u),
    uint2(2u, 6u),
    uint2(3u, 7u),
};

VSOutput main(uint vertexId : SV_VertexID)
{
    VSOutput output;

    const uint edgeIndex = vertexId / 2u;
    const uint cornerInEdge = vertexId % 2u;
    const uint cornerIndex = (cornerInEdge == 0u) ? kCubeEdges[edgeIndex].x : kCubeEdges[edgeIndex].y;

    const float outlineExpand = 0.0025f;
    const float3 cubeCorner = kCubeCorners[cornerIndex];
    const float3 offset = lerp(float3(-outlineExpand, -outlineExpand, -outlineExpand),
                               float3(outlineExpand, outlineExpand, outlineExpand),
                               cubeCorner);
    const float3 worldPos = uHighlightedBlock.xyz + cubeCorner + offset;

    output.position = mul(uViewProj, float4(worldPos, 1.0f));
    return output;
}
