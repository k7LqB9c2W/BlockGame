// Terrain shadow-map vertex shader. It forwards atlas UV data so cutout blocks
// like leaves can clip in the shadow pass instead of casting solid cube shadows.
cbuffer ShadowConstants : register(b0)
{
    float4x4 uLightViewProj;
};

struct VSInput
{
    float3 position : POSITION;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
};

struct VSOutput
{
    float4 position : SV_POSITION;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
};

VSOutput main(VSInput input)
{
    VSOutput output;
    output.position = mul(uLightViewProj, float4(input.position, 1.0f));
    output.tileCoord = input.tileCoord;
    output.atlasBase = input.atlasBase;
    output.atlasSize = input.atlasSize;
    return output;
}
