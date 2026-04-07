#include "world_lighting_common.hlsli"

Texture2D gAtlas : register(t1);
Texture2DArray gAerialPerspective : register(t2);
Texture2D gShadowMap : register(t3);
Texture2D gSkyBackground : register(t4);

SamplerState gTerrainSampler : register(s0);
SamplerState gLinearClamp : register(s1);
SamplerComparisonState gShadowSampler : register(s2);

cbuffer WorldConstants : register(b0)
{
    float4x4 uViewProj;
    float4x4 uShadowViewProj;
    float4 uLightDirection;
    float4 uCameraPos;
    float4 uHighlightedBlock;
    float4 uParams0;
    float4 uParams1;
    float4 uSunColor;
    float4 uSkyAmbient;
    float4 uGroundAmbient;
    float4 uSkyTopColor;
    float4 uSkyHorizonColor;
    float4 uShadowParams;
    float4 uTerrainDebug;
    float4 uTranslucencyParams;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float3 worldPos : POSITION0;
    float3 normal : NORMAL0;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
    float topY : TEXCOORD3;
    float bottomY : TEXCOORD4;
    uint faceKind : TEXCOORD5;
};

float4 sampleAerialPerspective(float2 screenUv, float distanceKm, float sliceCount)
{
    const float safeSliceCount = max(sliceCount, 1.0f);
    const float distance01 = saturate(distanceKm / max(uParams1.x, 0.001f));
    const float sliceF = sqrt(distance01) * (safeSliceCount - 1.0f);
    const float slice0 = floor(sliceF);
    const float slice1 = min(slice0 + 1.0f, safeSliceCount - 1.0f);
    const float blend = saturate(sliceF - slice0);
    const float4 ap0 = gAerialPerspective.SampleLevel(gLinearClamp, float3(screenUv, slice0), 0.0f);
    const float4 ap1 = gAerialPerspective.SampleLevel(gLinearClamp, float3(screenUv, slice1), 0.0f);
    return lerp(ap0, ap1, blend);
}

float sampleShadow(float3 worldPos, float3 normal)
{
    if (uShadowParams.w < 0.5f)
    {
        return 1.0f;
    }

    const float3 shadowOffsetPos = worldPos + normal * uShadowParams.y;
    const float4 shadowClip = mul(uShadowViewProj, float4(shadowOffsetPos, 1.0f));
    const float3 shadowNdc = shadowClip.xyz / max(shadowClip.w, 1e-5f);
    const float2 shadowUv = float2(shadowNdc.x * 0.5f + 0.5f, shadowNdc.y * -0.5f + 0.5f);
    const float shadowDepth = shadowNdc.z;

    if (shadowUv.x <= 0.0f || shadowUv.x >= 1.0f ||
        shadowUv.y <= 0.0f || shadowUv.y >= 1.0f ||
        shadowDepth <= 0.0f || shadowDepth >= 1.0f)
    {
        return 1.0f;
    }

    const float texelSize = max(uShadowParams.x, 1e-5f);
    float visibility = 0.0f;

    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            visibility += gShadowMap.SampleCmpLevelZero(gShadowSampler,
                                                        shadowUv + float2(x, y) * texelSize,
                                                        shadowDepth);
        }
    }

    return visibility * (1.0f / 9.0f);
}

float4 main(PSInput input) : SV_TARGET
{
    const float3 normal = normalize(input.normal);
    const float3 lightDir = normalize(uLightDirection.xyz);
    const float3 viewDir = normalize(uCameraPos.xyz - input.worldPos);
    const float2 wrappedTileUv = frac(input.tileCoord);
    const float wavePhase = sin(input.worldPos.x * 0.0125f + input.worldPos.z * 0.0105f + uCameraPos.x * 0.0015f);
    const float2 waterUvOffset = float2(wavePhase, wavePhase * 0.55f) * (input.atlasSize * 0.20f);
    const float2 atlasUv = input.atlasBase + input.atlasSize * wrappedTileUv + waterUvOffset;
    const float2 atlasUvDdx = ddx(input.tileCoord) * input.atlasSize;
    const float2 atlasUvDdy = ddy(input.tileCoord) * input.atlasSize;
    const float4 textureSample = gAtlas.SampleGrad(gTerrainSampler, atlasUv, atlasUvDdx, atlasUvDdy);

    const float faceShade = faceShadeMultiplier(normal);
    const float diff = max(dot(normal, lightDir), 0.0f);
    const float shadow = sampleShadow(input.worldPos, normal);
    const float hemi = saturate(normal.y * 0.5f + 0.5f);
    const float3 ambientTint = lerp(uGroundAmbient.rgb, uSkyAmbient.rgb, hemi);
    const float3 indirect = ambientTint * faceShade * 1.28f;
    const float3 directLight = uSunColor.rgb * (diff * shadow * faceShade * 0.24f);
    const float3 baseBounce = ambientTint * 0.035f;

    const bool underwater = uTranslucencyParams.y > 0.5f;
    const float thickness = max(input.topY - input.bottomY + 1.0f, 1.0f);
    const float thickness01 = saturate(1.0f - exp(-thickness * 0.14f));
    const float fresnelPower = underwater ? 1.8f : 4.0f;
    const float fresnel = pow(1.0f - saturate(dot(normal, viewDir)), fresnelPower);
    const float shimmer = saturate(0.45f + 0.55f * sin(input.worldPos.x * 0.006f + input.worldPos.z * 0.008f));
    const float3 waterTint = lerp(textureSample.rgb, float3(0.18f, 0.42f, 0.66f), 0.58f);
    const float3 absorbedTint = lerp(waterTint, waterTint * float3(0.78f, 0.88f, 0.96f), 0.22f * thickness01);
    const float3 skyReflection =
        computeTerrainFogColor(normalize(float3(viewDir.x, abs(viewDir.y), viewDir.z)),
                               uSkyTopColor.rgb,
                               uSkyHorizonColor.rgb);

    float3 color = absorbedTint * (indirect * 0.64f + baseBounce * 1.35f + directLight * 0.34f);
    color += skyReflection * ((underwater ? 0.04f : 0.10f) + fresnel * (underwater ? 0.10f : 0.26f));
    color += uSunColor.rgb * (0.02f + 0.05f * shimmer) * fresnel * (underwater ? 0.35f : 1.0f);
    color = lerp(color, absorbedTint * 0.82f, thickness01 * 0.28f);

    const float distanceBlocks = distance(input.worldPos, uCameraPos.xyz);
    const float horizontalDistanceBlocks = distance(input.worldPos.xz, uCameraPos.xz);
    const float3 fogViewDir = normalize(input.worldPos - uCameraPos.xyz);
    const float2 screenUv = input.position.xy * uParams0.zw;
    const bool useAnalyticFogBackground = uTerrainDebug.z > 0.5f;
    const float3 fogColor = useAnalyticFogBackground
                                ? computeTerrainFogColor(fogViewDir, uSkyTopColor.rgb, uSkyHorizonColor.rgb)
                                : gSkyBackground.SampleLevel(gLinearClamp, screenUv, 0.0f).rgb;
    if (uParams0.x > 0.5f)
    {
        const float4 aerial = sampleAerialPerspective(screenUv, distanceBlocks * 0.001f, uParams1.y);
        const float transmittance = saturate(max(aerial.a, 0.18f));
        color = color * transmittance + aerial.rgb;

        const FogBlendResult fogBlend = computeLayeredFog(distanceBlocks,
                                                          horizontalDistanceBlocks,
                                                          fogViewDir,
                                                          input.worldPos.y,
                                                          uCameraPos.y,
                                                          uParams1.z,
                                                          uParams1.w,
                                                          0.85f,
                                                          0.55f);
        color = color * fogBlend.transmittance + fogColor * fogBlend.inscatter;
    }
    else if (uParams1.w > uParams1.z)
    {
        const FogBlendResult fogBlend = computeLayeredFog(distanceBlocks,
                                                          horizontalDistanceBlocks,
                                                          fogViewDir,
                                                          input.worldPos.y,
                                                          uCameraPos.y,
                                                          uParams1.z,
                                                          uParams1.w,
                                                          1.20f,
                                                          0.92f);
        color = color * fogBlend.transmittance + fogColor * fogBlend.inscatter;
    }

    float alpha = saturate(0.18f + thickness01 * 0.36f + uTranslucencyParams.x * 0.24f);
    if (input.faceKind == 0u)
    {
        alpha = saturate(alpha + (underwater ? 0.04f : 0.08f) * fresnel);
    }
    else
    {
        alpha = saturate(alpha + thickness01 * 0.08f);
    }
    if (underwater)
    {
        alpha = saturate(alpha * 0.88f + 0.06f);
    }

    return float4(color, alpha);
}
