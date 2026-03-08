cbuffer AtmosphereConstants : register(b0)
{
    float4x4 uInvViewProj;
    float4x4 uView;
    float4x4 uProj;
    float4 uCameraPos;
    float4 uSunDirection;
    float4 uSunIlluminance;
    float4 uAtmosphereHeights;
    float4 uOzoneAndPhase;
    float4 uRayleighScattering;
    float4 uMieScattering;
    float4 uMieAbsorption;
    float4 uOzoneAbsorption;
    float4 uViewportAndDepth;
    float4 uSliceParams;
};

static const float PI = 3.14159265f;
static const float ATMOSPHERE_BRIGHTNESS = 0.05f;

struct PSInput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

struct MediumSample
{
    float rayleighDensity;
    float mieDensity;
    float ozoneDensity;
    float3 extinction;
};

float3 worldMetersToPlanetKm(float3 worldMeters)
{
    return float3(worldMeters.x * 0.001f,
                  uAtmosphereHeights.x + worldMeters.y * 0.001f,
                  worldMeters.z * 0.001f);
}

float raySphereExitDistance(float3 origin, float3 dir, float radius)
{
    const float b = dot(origin, dir);
    const float c = dot(origin, origin) - radius * radius;
    const float h = b * b - c;
    if (h < 0.0f)
    {
        return -1.0f;
    }

    const float sqrtH = sqrt(h);
    const float t0 = -b - sqrtH;
    const float t1 = -b + sqrtH;
    if (t1 < 0.0f)
    {
        return -1.0f;
    }
    return t0 > 0.0f ? t0 : t1;
}

MediumSample sampleMedium(float3 planetPosKm)
{
    MediumSample sample;
    const float altitudeKm = max(length(planetPosKm) - uAtmosphereHeights.x, 0.0f);

    sample.rayleighDensity = exp(-altitudeKm / max(uAtmosphereHeights.z, 0.001f));
    sample.mieDensity = exp(-altitudeKm / max(uAtmosphereHeights.w, 0.001f));
    sample.ozoneDensity = saturate(1.0f - abs(altitudeKm - uOzoneAndPhase.x) / max(uOzoneAndPhase.y, 0.001f));
    sample.extinction =
        sample.rayleighDensity * uRayleighScattering.rgb +
        sample.mieDensity * (uMieScattering.rgb + uMieAbsorption.rgb) +
        sample.ozoneDensity * uOzoneAbsorption.rgb;
    return sample;
}

float3 integrateTransmittance(float3 originKm, float3 dir, float maxDistanceKm, int steps)
{
    const int sampleCount = max(steps, 1);
    const float stepSize = maxDistanceKm / sampleCount;
    float3 opticalDepth = 0.0f.xxx;

    [loop]
    for (int i = 0; i < sampleCount; ++i)
    {
        const float t = (i + 0.5f) * stepSize;
        const float3 samplePos = originKm + dir * t;
        if (length(samplePos) < uAtmosphereHeights.x)
        {
            return 0.0f.xxx;
        }

        const MediumSample medium = sampleMedium(samplePos);
        opticalDepth += medium.extinction * stepSize;
    }

    return exp(-opticalDepth);
}

float rayleighPhase(float cosTheta)
{
    return 3.0f * (1.0f + cosTheta * cosTheta) / (16.0f * PI);
}

float miePhase(float cosTheta)
{
    const float g = uOzoneAndPhase.z;
    const float g2 = g * g;
    const float denom = pow(abs(1.0f + g2 - 2.0f * g * cosTheta), 1.5f);
    return (3.0f / (8.0f * PI)) * ((1.0f - g2) * (1.0f + cosTheta * cosTheta)) / ((2.0f + g2) * max(denom, 1e-4f));
}

float2 transmittanceUv(float altitudeKm, float mu)
{
    const float altitude01 = saturate(altitudeKm / max(uAtmosphereHeights.y - uAtmosphereHeights.x, 0.001f));
    const float mu01 = saturate(mu * 0.5f + 0.5f);
    return float2(mu01, altitude01);
}

float3 sampleTransmittanceLut(Texture2D transLut, SamplerState linearClamp, float altitudeKm, float mu)
{
    return transLut.SampleLevel(linearClamp, transmittanceUv(altitudeKm, mu), 0.0f).rgb;
}

float2 multiScatteringUv(float altitudeKm, float sunZenithCos)
{
    const float u = 0.5f + 0.5f * sunZenithCos;
    const float v = saturate(altitudeKm / max(uAtmosphereHeights.y - uAtmosphereHeights.x, 0.001f));
    return float2(u, v);
}

float3 sampleMultiScatteringLut(Texture2D multiLut, SamplerState linearClamp, float altitudeKm, float sunZenithCos)
{
    return multiLut.SampleLevel(linearClamp, multiScatteringUv(altitudeKm, sunZenithCos), 0.0f).rgb;
}

float3 reconstructWorldDirection(float2 uv)
{
    const float4 clip = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 1.0f, 1.0f);
    const float4 world = mul(uInvViewProj, clip);
    return normalize(world.xyz / max(world.w, 1e-5f) - uCameraPos.xyz);
}

float3 sampleSkyScattering(Texture2D transLut,
                           Texture2D multiLut,
                           SamplerState linearClamp,
                           float3 originMeters,
                           float3 dir,
                           float maxDistanceKm,
                           int stepCount,
                           out float3 outTransmittance)
{
    const float3 originKm = worldMetersToPlanetKm(originMeters);
    const float topDistance = raySphereExitDistance(originKm, dir, uAtmosphereHeights.y);
    const float groundDistance = raySphereExitDistance(originKm, dir, uAtmosphereHeights.x);
    float marchDistance = min(maxDistanceKm, topDistance > 0.0f ? topDistance : maxDistanceKm);
    if (groundDistance > 0.0f)
    {
        marchDistance = min(marchDistance, groundDistance);
    }
    const int sampleCount = max(stepCount, 1);
    const float stepSize = max(marchDistance, 0.0f) / sampleCount;

    float3 transmittance = 1.0f.xxx;
    float3 inscattering = 0.0f.xxx;

    if (marchDistance <= 0.0f)
    {
        outTransmittance = transmittance;
        return 0.0f.xxx;
    }

    [loop]
    for (int i = 0; i < sampleCount; ++i)
    {
        const float t = (i + 0.5f) * stepSize;
        const float3 samplePosKm = originKm + dir * t;
        if (length(samplePosKm) < uAtmosphereHeights.x)
        {
            break;
        }

        const float altitudeKm = max(length(samplePosKm) - uAtmosphereHeights.x, 0.0f);
        const MediumSample medium = sampleMedium(samplePosKm);
        const float sunCos = dot(normalize(samplePosKm), normalize(uSunDirection.xyz));
        const float3 transToSun = sampleTransmittanceLut(transLut, linearClamp, altitudeKm, sunCos);
        const float3 multiScatter = sampleMultiScatteringLut(multiLut, linearClamp, altitudeKm, sunCos);
        const float cosTheta = dot(dir, normalize(uSunDirection.xyz));
        const float3 singleScatter =
            medium.rayleighDensity * uRayleighScattering.rgb * rayleighPhase(cosTheta) +
            medium.mieDensity * uMieScattering.rgb * miePhase(cosTheta);
        const float3 scatter = singleScatter + multiScatter * (medium.rayleighDensity + medium.mieDensity) * 0.15f;
        inscattering += transmittance * transToSun * scatter * uSunIlluminance.rgb * stepSize;
        transmittance *= exp(-medium.extinction * stepSize);
    }

    outTransmittance = transmittance;
    return inscattering * ATMOSPHERE_BRIGHTNESS;
}

float3 sampleSkyLuminance(Texture2D transLut,
                          Texture2D multiLut,
                          SamplerState linearClamp,
                          float3 originMeters,
                          float3 dir)
{
    float3 transmittance = 1.0f.xxx;
    return sampleSkyScattering(transLut, multiLut, linearClamp, originMeters, dir, 128.0f, 24, transmittance);
}
