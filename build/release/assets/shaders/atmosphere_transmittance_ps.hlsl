#include "atmosphere_common.hlsli"

Texture2D gUnused0 : register(t0);
Texture2D gUnused1 : register(t1);
Texture2D gUnused2 : register(t2);

SamplerState gLinearClamp : register(s0);

float4 main(PSInput input) : SV_TARGET
{
    const float altitudeKm = input.uv.y * (uAtmosphereHeights.y - uAtmosphereHeights.x);
    const float height = uAtmosphereHeights.x + altitudeKm;
    const float mu = input.uv.x * 2.0f - 1.0f;
    const float sinTheta = sqrt(saturate(1.0f - mu * mu));
    const float3 origin = float3(0.0f, height, 0.0f);
    const float3 dir = normalize(float3(sinTheta, mu, 0.0f));
    const float maxDistance = raySphereExitDistance(origin, dir, uAtmosphereHeights.y);
    const float3 transmittance = integrateTransmittance(origin, dir, max(maxDistance, 0.0f), 24);
    return float4(transmittance, 1.0f);
}
