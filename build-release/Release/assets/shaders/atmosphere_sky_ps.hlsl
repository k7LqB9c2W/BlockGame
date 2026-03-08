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

float3 sampleSkyView(float3 worldDir)
{
    const float3 localDir = worldToSkyViewLocalDirection(worldDir);
    const float2 uv = directionToSkyUv(localDir);
    return gSkyViewLut.SampleLevel(gLinearClamp, uv, 0.0f).rgb;
}

float3 sampleSkyUpperHemisphere(float3 worldDir)
{
    const float3 localDir = worldToSkyViewLocalDirection(worldDir);
    const float3 safeDir = normalize(float3(localDir.x, max(localDir.y, 0.035f), localDir.z));
    float2 uv = directionToSkyUv(safeDir);
    uv.y = max(uv.y, 0.5f + SKY_LUT_SEAM_PADDING);
    return gSkyViewLut.SampleLevel(gLinearClamp, uv, 0.0f).rgb;
}

float4 main(PSInput input) : SV_TARGET
{
    const float3 dir = reconstructWorldDirection(input.uv);
    float3 color = sampleSkyView(dir);

    const float3 mirroredDir = normalize(float3(dir.x, max(abs(dir.y), 0.06f), dir.z));
    const float3 mirroredSky = sampleSkyUpperHemisphere(mirroredDir);
    const float3 horizonProbeDir = normalize(float3(dir.x, 0.12f, dir.z));
    const float3 horizonSky = sampleSkyUpperHemisphere(horizonProbeDir);
    const float sunHeight = saturate(uSunDirection.y * 0.5f + 0.5f);
    const float3 horizonTint = lerp(float3(0.20f, 0.24f, 0.30f),
                                    float3(0.68f, 0.76f, 0.88f),
                                    sunHeight);
    const float3 horizonLift = max(horizonSky * 0.85f, mirroredSky * 0.65f);
    const float3 groundedHaze = max(lerp(horizonLift, horizonTint, 0.18f),
                                    horizonTint * 0.78f);
    const float lowerDepth = saturate(-dir.y / 0.25f);
    const float3 lowerHemisphere = lerp(horizonSky * 0.98f, groundedHaze, lowerDepth);
    const float horizonLiftBlend = blendRange(dir.y, 0.08f, 0.0f);
    color = lerp(color, max(color, horizonSky * 0.92f), horizonLiftBlend * 0.20f);

    if (dir.y < 0.08f)
    {
        const float groundBlend = blendRange(dir.y, 0.08f, -0.20f);
        color = lerp(color, lowerHemisphere, groundBlend);
    }

    const float sunAmount = saturate(dot(dir, normalize(uSunDirection.xyz)));
    const float sunDisk = smoothstep(0.9992f, 0.99995f, sunAmount);
    color += uSunIlluminance.rgb * sunDisk * 0.015f;
    return float4(color, 1.0f);
}
