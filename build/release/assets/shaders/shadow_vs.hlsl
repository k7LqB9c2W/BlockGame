cbuffer ShadowConstants : register(b0)
{
    float4x4 uLightViewProj;
};

struct VSInput
{
    float3 position : POSITION;
};

float4 main(VSInput input) : SV_POSITION
{
    return mul(uLightViewProj, float4(input.position, 1.0f));
}
