// mob_ps.hlsl
// Shades static debug mob meshes with optional texture sampling and a pink fallback material.

#include "world_lighting_common.hlsli"

Texture2D gMobTexture : register(t0);
Texture2DArray gAerialPerspective : register(t1);
Texture2D gShadowMap : register(t2);
Texture2D gSkyBackground : register(t3);

SamplerState gTerrainSampler : register(s0);
SamplerState gLinearClamp : register(s1);

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
};

struct PSInput
{
    float4 position : SV_POSITION;
    float3 worldPos : POSITION0;
    float3 normal : NORMAL0;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
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

float4 main(PSInput input) : SV_TARGET
{
    const bool useTexture = uParams0.x > 0.5f;
    const bool useAerialPerspective = uParams0.y > 0.5f;
    const float2 screenUv = input.position.xy * uParams0.zw;

    float4 albedoSample = input.color;
    if (useTexture)
    {
        albedoSample = gMobTexture.Sample(gTerrainSampler, input.uv) * input.color;
        clip(albedoSample.a - 0.05f);
    }

    const float3 normal = normalize(input.normal);
    const float3 lightDir = normalize(uLightDirection.xyz);
    const float3 viewDir = normalize(uCameraPos.xyz - input.worldPos);
    const float faceShade = faceShadeMultiplier(normal);
    const float directSunGate = uTerrainDebug.x;
    const float diff = max(dot(normal, lightDir), 0.0f);
    const float3 halfDir = normalize(lightDir + viewDir);
    const float spec = pow(max(dot(normal, halfDir), 0.0f), 20.0f);
    const float hemi = saturate(normal.y * 0.5f + 0.5f);
    const float3 ambientTint = lerp(uGroundAmbient.rgb, uSkyAmbient.rgb, hemi);
    const float3 diffuse = ambientTint * 0.90f + uSunColor.rgb * (diff * faceShade * directSunGate * 0.22f);
    const float3 specular = uSunColor.rgb * (spec * faceShade * directSunGate * 0.01f);

    float3 color = albedoSample.rgb * diffuse + specular;

    const float distanceBlocks = distance(input.worldPos, uCameraPos.xyz);
    const float horizontalDistanceBlocks = distance(input.worldPos.xz, uCameraPos.xz);
    const float3 fogViewDir = normalize(input.worldPos - uCameraPos.xyz);
    const bool useAnalyticFogBackground = uTerrainDebug.z > 0.5f;
    const float3 fogColor = useAnalyticFogBackground
                                ? computeTerrainFogColor(fogViewDir, uSkyTopColor.rgb, uSkyHorizonColor.rgb)
                                : gSkyBackground.SampleLevel(gLinearClamp, screenUv, 0.0f).rgb;

    if (useAerialPerspective)
    {
        const float4 aerial = sampleAerialPerspective(screenUv, distanceBlocks * 0.001f, uParams1.y);
        const float transmittance = saturate(max(aerial.a, 0.18f));
        color = color * transmittance + aerial.rgb;
    }

    if (uParams1.w > uParams1.z)
    {
        const FogBlendResult fogBlend = computeLayeredFog(distanceBlocks,
                                                          horizontalDistanceBlocks,
                                                          fogViewDir,
                                                          input.worldPos.y,
                                                          uCameraPos.y,
                                                          uParams1.z,
                                                          uParams1.w,
                                                          1.0f,
                                                          0.78f);
        color = color * fogBlend.transmittance + fogColor * fogBlend.inscatter;
    }

    return float4(color, albedoSample.a);
}
