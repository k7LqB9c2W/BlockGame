#include "world_lighting_common.hlsli"

Texture2D gAtlas : register(t0);
Texture2DArray gAerialPerspective : register(t1);
Texture2D gShadowMap : register(t2);

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
    float4 uShadowParams;
    float4 uTerrainDebug;
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

    visibility *= (1.0f / 9.0f);
    return lerp(1.0f, visibility, saturate(uShadowParams.z));
}

float computeMipLevel(float2 atlasUvDdx, float2 atlasUvDdy)
{
    uint width = 1;
    uint height = 1;
    uint mipCount = 1;
    gAtlas.GetDimensions(0, width, height, mipCount);
    const float2 texelScale = float2((float)width, (float)height);
    const float2 dx = atlasUvDdx * texelScale;
    const float2 dy = atlasUvDdy * texelScale;
    const float rho = max(dot(dx, dx), dot(dy, dy));
    return max(0.0f, 0.5f * log2(max(rho, 1e-8f)));
}

float3 debugMipColor(float mipLevel)
{
    uint width = 1;
    uint height = 1;
    uint mipCount = 1;
    gAtlas.GetDimensions(0, width, height, mipCount);
    const float mip01 = saturate(mipLevel / max((float)(mipCount - 1), 1.0f));
    return lerp(float3(0.10f, 0.55f, 0.95f), float3(0.95f, 0.30f, 0.10f), mip01);
}

float4 main(PSInput input) : SV_TARGET
{
    const float3 normal = normalize(input.normal);
    const float3 lightDir = normalize(uLightDirection.xyz);
    const float3 viewDir = normalize(uCameraPos.xyz - input.worldPos);
    const float2 wrappedTileUv = frac(input.tileCoord);
    const float2 atlasUv = input.atlasBase + input.atlasSize * wrappedTileUv;
    const float2 atlasUvDdx = ddx(input.tileCoord) * input.atlasSize;
    const float2 atlasUvDdy = ddy(input.tileCoord) * input.atlasSize;
    const float4 textureSample = gAtlas.SampleGrad(gTerrainSampler, atlasUv, atlasUvDdx, atlasUvDdy);
    clip(textureSample.a - 0.5f);

    const float skyLight = saturate(input.lightChannels.x);
    const float blockLight = saturate(input.lightChannels.y);
    const float ao = saturate(input.ao);
    const float mipLevel = computeMipLevel(atlasUvDdx, atlasUvDdy);
    const int debugView = (int)uTerrainDebug.y;

    if (debugView == 1)
    {
        return float4(skyLight.xxx, 1.0f);
    }
    if (debugView == 2)
    {
        return float4(blockLight.xxx, 1.0f);
    }
    if (debugView == 3)
    {
        return float4(debugMipColor(mipLevel), 1.0f);
    }
    if (debugView == 4)
    {
        return float4(ao.xxx, 1.0f);
    }

    const float faceShade = faceShadeMultiplier(normal);
    const float diff = max(dot(normal, lightDir), 0.0f);
    const float3 halfDir = normalize(lightDir + viewDir);
    const float spec = pow(max(dot(normal, halfDir), 0.0f), 16.0f);
    const float shadow = sampleShadow(input.worldPos, normal);
    const float hemi = saturate(normal.y * 0.5f + 0.5f);
    const float3 ambientTint = lerp(uGroundAmbient.rgb, uSkyAmbient.rgb, hemi);
    const float3 skyTint = ambientTint * 2.15f + float3(0.08f, 0.09f, 0.11f);
    const float3 skyIndirect = skyTint * skyLight;
    const float3 blockIndirect = float3(1.12f, 0.95f, 0.70f) * blockLight;
    const float3 indirect = (skyIndirect + blockIndirect) * faceShade * ao;

    const float directSunGate = uTerrainDebug.x * saturate(skyLight * 1.08f);
    const float directAo = lerp(1.0f, ao, 0.35f);
    const float3 directLight =
        uSunColor.rgb * (diff * shadow * directSunGate * faceShade * directAo * 0.38f);
    const float3 specularLight =
        uSunColor.rgb * (spec * shadow * directSunGate * faceShade * directAo * 0.012f);
    const float3 baseBounce = ambientTint * 0.026f;

    float3 color = textureSample.rgb * (indirect + baseBounce + directLight) + specularLight;

    if (uParams0.y > 0.5f)
    {
        const int3 currentBlock = int3(floor(input.worldPos));
        const int3 targetBlock = int3(uHighlightedBlock.xyz);
        if (all(currentBlock == targetBlock))
        {
            color = min(color + float3(0.28f, 0.28f, 0.28f), float3(1.0f, 1.0f, 1.0f));
        }
    }

    const float distanceBlocks = distance(input.worldPos, uCameraPos.xyz);
    const float3 fogColor = computeTerrainFogColor(normalize(input.worldPos - uCameraPos.xyz));
    if (uParams0.x > 0.5f)
    {
        const float2 screenUv = input.position.xy * uParams0.zw;
        const float4 aerial = sampleAerialPerspective(screenUv, distanceBlocks * 0.001f, uParams1.y);
        const float transmittance = saturate(max(aerial.a, 0.18f));
        color = color * transmittance + aerial.rgb;

        const FogBlendResult fogBlend = computeRoundedFog(distanceBlocks,
                                                          viewDir,
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
        const FogBlendResult fogBlend = computeRoundedFog(distanceBlocks,
                                                          viewDir,
                                                          input.worldPos.y,
                                                          uCameraPos.y,
                                                          uParams1.z,
                                                          uParams1.w,
                                                          1.20f,
                                                          0.92f);
        color = color * fogBlend.transmittance + fogColor * fogBlend.inscatter;
    }

    return float4(color, 1.0f);
}
