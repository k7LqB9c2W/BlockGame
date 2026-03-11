#include "base_game_sky_common.hlsli"

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
    return float4(computeBaseGameSkyGradientFromScreenUv(input.uv.y), 1.0f);
}
