#include "base_game_sky_common.hlsli"

cbuffer BaseSkyConstants : register(b0)
{
    float4 uTopSkyColor;
    float4 uHorizonSkyColor;
    float4x4 uInvViewProj;
    float4 uCameraPos;
}

struct PSInput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET
{
    const float2 ndc = float2(input.uv.x * 2.0f - 1.0f, (1.0f - input.uv.y) * 2.0f - 1.0f);
    const float4 farWorldH = mul(uInvViewProj, float4(ndc, 1.0f, 1.0f));
    const float3 farWorld = farWorldH.xyz / max(farWorldH.w, 1e-5f);
    const float3 viewDir = normalize(farWorld - uCameraPos.xyz);
    return float4(computeSkyGradientFromViewDir(viewDir, uTopSkyColor.rgb, uHorizonSkyColor.rgb), 1.0f);
}
