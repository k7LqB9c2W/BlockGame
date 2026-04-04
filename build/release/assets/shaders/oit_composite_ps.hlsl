Texture2D gSceneColor : register(t0);
Texture2D gAccumTexture : register(t1);
Texture2D gRevealTexture : register(t2);

SamplerState gLinearClamp : register(s0);

cbuffer ToneMapConstants : register(b0)
{
    float4 uUnused0;
}

struct PSInput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET
{
    const float3 sceneColor = gSceneColor.Sample(gLinearClamp, input.uv).rgb;
    const float4 accum = gAccumTexture.Sample(gLinearClamp, input.uv);
    const float reveal = gRevealTexture.Sample(gLinearClamp, input.uv).r;
    const float transAlpha = saturate(1.0f - reveal);
    const float3 transColor = (accum.a > 1e-4f) ? (accum.rgb / accum.a) : 0.0f.xxx;
    const float3 color = lerp(sceneColor, transColor, transAlpha);
    return float4(color, 1.0f);
}
