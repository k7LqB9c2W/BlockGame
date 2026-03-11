cbuffer BaseSkyConstants : register(b0)
{
    float4 uTopSkyColor;
    float4 uHorizonSkyColor;
    float4 uParams;
    float4 uSunColor;
}

struct PSInput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET
{
    const float horizonMask = pow(saturate(1.0f - abs(input.uv.y * 2.0f - 1.0f) * 1.15f), 2.0f);
    const float horizonBlend = saturate(1.0f - pow(input.uv.y, 1.85f));
    const float3 skyColor = lerp(uTopSkyColor.rgb, uHorizonSkyColor.rgb, horizonBlend);
    const float3 softenedSky = lerp(skyColor, uHorizonSkyColor.rgb, horizonMask * 0.22f);
    return float4(softenedSky, 1.0f);
}
