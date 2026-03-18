#ifndef BASE_GAME_SKY_COMMON_HLSLI
#define BASE_GAME_SKY_COMMON_HLSLI

static const float3 kBaseGameSkyColorSrgb = float3(120.0f / 255.0f, 167.0f / 255.0f, 255.0f / 255.0f);
static const float3 kBaseGameHorizonColorSrgb = float3(187.0f / 255.0f, 212.0f / 255.0f, 255.0f / 255.0f);

float3 baseGameSkySrgbToLinear(float3 color)
{
    return pow(color, 2.2f);
}

float3 computeBaseGameSkyGradientFromViewY(float viewY)
{
    const float clampedUp = saturate(viewY);
    const float horizonBlend = pow(1.0f - clampedUp, 1.75f);
    const float horizonBand = pow(1.0f - clampedUp, 3.8f);
    const float3 topSky = baseGameSkySrgbToLinear(kBaseGameSkyColorSrgb) * 1.10f;
    const float3 horizonSky = baseGameSkySrgbToLinear(kBaseGameHorizonColorSrgb) * 1.05f;
    float3 skyColor = lerp(topSky, horizonSky, horizonBlend);
    skyColor = lerp(skyColor, horizonSky, horizonBand * 0.30f);
    return skyColor;
}

float3 computeBaseGameSkyGradientFromScreenUv(float uvY)
{
    return computeBaseGameSkyGradientFromViewY(1.0f - saturate(uvY));
}

#endif
