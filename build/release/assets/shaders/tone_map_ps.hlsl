Texture2D gSceneColor : register(t0);
Texture2D gUnused1 : register(t1);
Texture2D gUnused2 : register(t2);

SamplerState gLinearClamp : register(s0);

cbuffer ToneMapConstants : register(b0)
{
    float4 uExposureWhitePoint;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float3 acesApprox(float3 color)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

float4 main(PSInput input) : SV_TARGET
{
    float3 hdr = gSceneColor.Sample(gLinearClamp, input.uv).rgb;
    hdr *= uExposureWhitePoint.x;
    const float whitePoint = max(uExposureWhitePoint.y, 1.0f);
    const float whiteReference = max(acesApprox(whitePoint.xxx).r, 1e-4f);
    float3 mapped = acesApprox(hdr) / whiteReference;
    mapped = saturate(mapped);
    mapped = pow(mapped, 1.0f / 2.2f);
    return float4(mapped, 1.0f);
}
