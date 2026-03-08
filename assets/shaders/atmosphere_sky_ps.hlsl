#include "atmosphere_common.hlsli"

Texture2D gSkyViewLut : register(t0);
Texture2D gUnused1 : register(t1);
Texture2D gUnused2 : register(t2);

SamplerState gLinearClamp : register(s0);

float2 directionToSkyUv(float3 dir)
{
    const float azimuth = atan2(dir.x, dir.z);
    const float latitude = asin(clamp(dir.y, -1.0f, 1.0f));
    const float u = azimuth / (2.0f * PI) + 0.5f;
    const float v = 0.5f + 0.5f * sign(latitude) * sqrt(abs(latitude) / (0.5f * PI));
    return float2(u, v);
}

float4 main(PSInput input) : SV_TARGET
{
    const float3 dir = reconstructWorldDirection(input.uv);
    const float2 skyUv = directionToSkyUv(dir);
    float3 color = gSkyViewLut.SampleLevel(gLinearClamp, skyUv, 0.0f).rgb;

    if (dir.y < 0.02f)
    {
        const float3 mirroredDir = normalize(float3(dir.x, max(abs(dir.y), 0.04f), dir.z));
        const float3 mirroredSky = gSkyViewLut.SampleLevel(gLinearClamp, directionToSkyUv(mirroredDir), 0.0f).rgb;
        const float sunHeight = saturate(uSunDirection.y * 0.5f + 0.5f);
        const float3 horizonTint = lerp(float3(0.06f, 0.08f, 0.12f),
                                        float3(0.20f, 0.26f, 0.34f),
                                        sunHeight);
        const float3 lowerHemisphere = lerp(horizonTint, mirroredSky * 0.35f + horizonTint * 0.65f, 0.5f);
        const float groundBlend = smoothstep(0.02f, -0.12f, dir.y);
        color = lerp(color, lowerHemisphere, groundBlend);
    }

    const float sunAmount = saturate(dot(dir, normalize(uSunDirection.xyz)));
    const float sunDisk = smoothstep(0.9992f, 0.99995f, sunAmount);
    color += uSunIlluminance.rgb * sunDisk * 0.015f;
    return float4(color, 1.0f);
}
