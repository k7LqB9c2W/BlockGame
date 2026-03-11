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

float3 srgbToLinear(float3 color)
{
    return pow(color, 2.2f);
}

float4 main(PSInput input) : SV_TARGET
{
    const float horizonBlend = pow(saturate(input.uv.y), 1.75f);
    const float horizonBand = pow(saturate(input.uv.y), 3.8f);
    const float3 topSky = srgbToLinear(uTopSkyColor.rgb) * 1.10f;
    const float3 horizonSky = srgbToLinear(uHorizonSkyColor.rgb) * 1.05f;
    float3 skyColor = lerp(topSky, horizonSky, horizonBlend);
    skyColor = lerp(skyColor, horizonSky, horizonBand * 0.30f);
    return float4(skyColor, 1.0f);
}
