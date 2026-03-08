#include "atmosphere_common.hlsli"

Texture2D gTransmittanceLut : register(t0);
Texture2D gMultiScatteringLut : register(t1);
Texture2D gUnused2 : register(t2);

SamplerState gLinearClamp : register(s0);

float4 main(PSInput input) : SV_TARGET
{
    const float sliceCount = max(uSliceParams.y, 1.0f);
    const float sliceT = (sliceCount > 1.0f) ? (uSliceParams.x / (sliceCount - 1.0f)) : 0.0f;
    const float maxDistanceKm = max(uSliceParams.z, 0.001f);
    const float distanceKm = sliceT * sliceT * maxDistanceKm;
    const float3 dir = reconstructWorldDirection(input.uv);

    float3 transmittance = 1.0f.xxx;
    const float3 inscattering =
        sampleSkyScattering(gTransmittanceLut, gMultiScatteringLut, gLinearClamp, uCameraPos.xyz, dir, distanceKm, 16, transmittance);
    const float meanTransmittance = dot(transmittance, float3(0.333333f, 0.333333f, 0.333333f));
    const float blendedTransmittance = lerp(1.0f, saturate(meanTransmittance), saturate(sliceT));
    return float4(max(inscattering * 0.35f, 0.0f.xxx), blendedTransmittance);
}
