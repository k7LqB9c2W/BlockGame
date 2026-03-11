cbuffer CloudConstants : register(b0)
{
    float4x4 uViewProj;
    float4 uCameraPosTime;
    float4 uLayerParams;
    float4 uShapeParams;
    float4 uTopColor;
    float4 uBottomColor;
}

struct PSInput
{
    float4 position : SV_POSITION;
    float3 worldPos : POSITION0;
    float3 localPos : TEXCOORD0;
    float3 normal : NORMAL0;
    float coverage : TEXCOORD1;
};

float3 srgbToLinear(float3 color)
{
    return pow(color, 2.2f);
}

float hash(float2 p)
{
    return frac(sin(dot(p, float2(91.7f, 217.3f))) * 28411.28125f);
}

float4 main(PSInput input) : SV_TARGET
{
    clip(input.coverage - 0.01f);

    const float topMask = saturate(input.normal.y * 0.5f + 0.5f);
    const float sideShade = saturate(0.72f + input.normal.y * 0.22f);
    const float edgeNoise = hash(floor(input.worldPos.xz * 0.125f));
    const float density = lerp(0.74f, 0.88f, edgeNoise) * input.coverage;
    const float alpha = lerp(uBottomColor.a, uTopColor.a, topMask) * density;

    float3 topColor = srgbToLinear(uTopColor.rgb);
    float3 bottomColor = srgbToLinear(uBottomColor.rgb);
    float3 cloudColor = lerp(bottomColor, topColor, topMask);
    cloudColor *= sideShade;

    return float4(cloudColor, alpha);
}
