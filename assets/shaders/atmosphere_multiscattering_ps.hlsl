#include "atmosphere_common.hlsli"

Texture2D gTransmittanceLut : register(t0);
Texture2D gUnused1 : register(t1);
Texture2D gUnused2 : register(t2);

SamplerState gLinearClamp : register(s0);

float4 main(PSInput input) : SV_TARGET
{
    const float altitudeKm = input.uv.y * (uAtmosphereHeights.y - uAtmosphereHeights.x);
    const float3 origin = float3(0.0f, uAtmosphereHeights.x + altitudeKm, 0.0f);
    const float sunZenithCos = input.uv.x * 2.0f - 1.0f;
    const float sunZenithSin = sqrt(saturate(1.0f - sunZenithCos * sunZenithCos));
    float3 secondOrder = 0.0f.xxx;
    float transferFactor = 0.0f;
    const int directionCount = 64;

    [loop]
    for (int i = 0; i < directionCount; ++i)
    {
        const float phi = (2.0f * PI * (i + 0.5f)) / directionCount;
        const float cosTheta = 1.0f - 2.0f * ((i + 0.5f) / directionCount);
        const float sinTheta = sqrt(saturate(1.0f - cosTheta * cosTheta));
        const float3 dir = normalize(float3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta));
        float3 transmittance = 1.0f.xxx;
        const float3 radiance = sampleSkyScattering(gTransmittanceLut,
                                                    gTransmittanceLut,
                                                    gLinearClamp,
                                                    float3(0.0f, altitudeKm * 1000.0f, 0.0f),
                                                    dir,
                                                    64.0f,
                                                    12,
                                                    0.0f,
                                                    transmittance);
        secondOrder += radiance * (1.0f / (4.0f * PI));
        transferFactor += (1.0f - dot(transmittance, float3(0.333333f, 0.333333f, 0.333333f))) * (1.0f / directionCount);
    }

    transferFactor = saturate(transferFactor * 0.35f);
    const float series = 1.0f / max(1.0f - transferFactor, 0.25f);
    const float3 result = secondOrder * series * 0.25f;
    return float4(max(result, 0.0f.xxx), 1.0f);
}
