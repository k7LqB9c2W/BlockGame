#include "world_lighting_common.hlsli"

Texture2D gAtlas : register(t0);
Texture2DArray gAerialPerspective : register(t1);
Texture2D gShadowMap : register(t2);
Texture2D gSkyBackground : register(t3);

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
    float2 lightChannels : TEXCOORD3;
    float ao : TEXCOORD4;
    uint materialFlags : TEXCOORD5;
    uint alphaCutout : TEXCOORD7;
    uint blockId : TEXCOORD8;
};

struct PSOutput
{
    float4 accum : SV_TARGET0;
    float reveal : SV_TARGET1;
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

PSOutput main(PSInput input)
{
    if (!isTranslucentBlockId(input.blockId))
    {
        discard;
    }

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

    const float skyLight = saturate(input.lightChannels.x);
    const float blockLight = saturate(input.lightChannels.y);
    const float aoStrength = max(uTerrainDebug.w, 0.01f);
    const float ao = pow(saturate(input.ao), aoStrength);
    const float faceShade = faceShadeMultiplier(normal);
    const float indirectFaceShade = lerp(1.0f, faceShade, 0.55f);
    const float directFaceShade = lerp(1.0f, faceShade, 0.25f);
    const float diff = max(dot(normal, lightDir), 0.0f);
    const float hemi = saturate(normal.y * 0.5f + 0.5f);
    const float3 ambientTint = lerp(uGroundAmbient.rgb, uSkyAmbient.rgb, hemi);
    const float3 skyTint = ambientTint * 2.15f + float3(0.08f, 0.09f, 0.11f);
    const float3 skyIndirect = skyTint * skyLight;
    const float3 blockIndirect = float3(1.12f, 0.95f, 0.70f) * blockLight;
    const float indirectAo = lerp(1.0f, ao, 0.78f);
    const float3 indirect = (skyIndirect + blockIndirect) * indirectFaceShade * indirectAo;
    const float directSunGate = uTerrainDebug.x * saturate(skyLight * 1.08f);
    const float directAo = lerp(1.0f, ao, 0.18f);
    const float3 directLight =
        uSunColor.rgb * (diff * directSunGate * directFaceShade * directAo * 0.20f);
    const float3 baseBounce = ambientTint * 0.024f;

    const float fresnel = pow(1.0f - saturate(dot(normal, viewDir)), 4.0f);
    const float shimmer = saturate(0.45f + 0.55f * sin(input.worldPos.x * 0.006f + input.worldPos.z * 0.008f));
    const float3 waterTint = lerp(textureSample.rgb, float3(0.18f, 0.42f, 0.66f), 0.58f);
    const float3 skyReflection =
        computeTerrainFogColor(normalize(float3(viewDir.x, abs(viewDir.y), viewDir.z)),
                               uSkyTopColor.rgb,
                               uSkyHorizonColor.rgb);
    float3 color = waterTint * (indirect * 0.62f + baseBounce * 1.35f + directLight * 0.32f);
    color += skyReflection * (0.10f + fresnel * 0.28f);
    color += uSunColor.rgb * (0.02f + 0.05f * shimmer) * fresnel;

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

    const float alpha = opacityForBlock(input.blockId, uTranslucencyParams.x);
    PSOutput output;
    output.accum = float4(color * alpha, alpha);
    output.reveal = alpha;
    return output;
}
