#include "atmosphere_common.hlsli"

Texture2D gTransmittanceLut : register(t0);
Texture2D gMultiScatteringLut : register(t1);
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

float3 skyUvToDirection(float2 uv)
{
    const float azimuth = (uv.x - 0.5f) * 2.0f * PI;
    const float remap = (uv.y - 0.5f) * 2.0f;
    const float latitude = sign(remap) * remap * remap * (0.5f * PI);
    const float cosLat = cos(latitude);
    return normalize(float3(sin(azimuth) * cosLat, sin(latitude), cos(azimuth) * cosLat));
}

float4 main(PSInput input) : SV_TARGET
{
    const float3 viewDirLocal = skyUvToDirection(input.uv);
    const float3x3 invViewRotation = transpose((float3x3)uView);
    const float3 viewDir = normalize(mul(invViewRotation, viewDirLocal));
    const float3 sky = sampleSkyLuminance(gTransmittanceLut, gMultiScatteringLut, gLinearClamp, uCameraPos.xyz, viewDir);
    return float4(max(sky, 0.0f.xxx), 1.0f);
}
