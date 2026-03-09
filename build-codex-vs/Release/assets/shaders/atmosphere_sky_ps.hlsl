#include "atmosphere_common.hlsli"

Texture2D gSkyViewLut : register(t0);
Texture2D gUnused1 : register(t1);
Texture2D gUnused2 : register(t2);

SamplerState gLinearClamp : register(s0);

static const float SKY_LUT_HEIGHT = 128.0f;
static const float SKY_LUT_SEAM_PADDING = 1.5f / SKY_LUT_HEIGHT;

float2 directionToSkyUv(float3 dir)
{
    const float azimuth = atan2(dir.x, dir.z);
    const float latitude = asin(clamp(dir.y, -1.0f, 1.0f));
    const float u = azimuth / (2.0f * PI) + 0.5f;
    const float v = 0.5f + 0.5f * sign(latitude) * sqrt(abs(latitude) / (0.5f * PI));
    return float2(u, v);
}

float blendRange(float value, float start, float end)
{
    return saturate((start - value) / max(start - end, 1e-4f));
}

float3 worldToSkyViewLocalDirection(float3 worldDir)
{
    return normalize(mul((float3x3)uView, normalize(worldDir)));
}

float3 sampleSkyViewSeamSafe(float3 worldDir)
{
    const float3 localDir = worldToSkyViewLocalDirection(worldDir);
    float2 uv = directionToSkyUv(localDir);
    if (localDir.y > -0.12f)
    {
        uv.y = max(uv.y, 0.5f + SKY_LUT_SEAM_PADDING);
    }
    return gSkyViewLut.SampleLevel(gLinearClamp, uv, 0.0f).rgb;
}

float3 sampleSkyUpperHemisphere(float3 worldDir)
{
    const float3 localDir = worldToSkyViewLocalDirection(worldDir);
    float2 uv = directionToSkyUv(localDir);
    uv.y = max(uv.y, 0.5f + SKY_LUT_SEAM_PADDING);
    return gSkyViewLut.SampleLevel(gLinearClamp, uv, 0.0f).rgb;
}

float4 main(PSInput input) : SV_TARGET
{
    const float3 dir = reconstructWorldDirection(input.uv);
    float3 color = sampleSkyViewSeamSafe(dir);

    const float3 horizonProbeDir = normalize(float3(dir.x, 0.18f, dir.z));
    const float3 horizonHighProbeDir = normalize(float3(dir.x, 0.40f, dir.z));
    const float3 horizonHighProbe = sampleSkyUpperHemisphere(horizonHighProbeDir);
    const float3 horizonProbe = sampleSkyUpperHemisphere(horizonProbeDir);
    const float3 horizonFill = lerp(horizonProbe, horizonHighProbe, 0.90f);
    const float seamBlend = blendRange(dir.y, 0.22f, -0.12f);
    color = lerp(color, horizonFill, seamBlend * 0.65f);

    if (dir.y < 0.04f)
    {
        const float lowerBlend = blendRange(dir.y, 0.04f, -0.34f);
        color = lerp(color, horizonFill * 0.94f, lowerBlend);
    }

    const float sunAmount = saturate(dot(dir, normalize(uSunDirection.xyz)));
    const float sunDisk = smoothstep(0.9992f, 0.99995f, sunAmount);
    color += uSunIlluminance.rgb * sunDisk * 0.015f;
    return float4(color, 1.0f);
}
