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

struct VSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
    uint lighting : COLOR0;
    uint blockId : COLOR1;
};

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

VSOutput main(VSInput input)
{
    const DecodedVertexLighting decodedLighting = decodeVertexLighting(input.lighting);

    VSOutput output;
    output.worldPos = input.position;
    output.normal = input.normal;
    output.tileCoord = input.tileCoord;
    output.atlasBase = input.atlasBase;
    output.atlasSize = input.atlasSize;
    output.lightChannels = float2(decodedLighting.sky, decodedLighting.block);
    output.ao = decodedLighting.ao;
    output.materialFlags = decodedLighting.flags;
    output.farVoxelScale = decodedLighting.farVoxelScale;
    output.alphaCutout = decodedLighting.alphaCutout;
    output.blockId = input.blockId;
    output.position = mul(uViewProj, float4(input.position, 1.0f));
    return output;
}
