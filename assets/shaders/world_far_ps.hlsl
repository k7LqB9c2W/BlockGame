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
    float farVoxelScale : TEXCOORD6;
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
    const bool isWater = (input.materialFlags & kMaterialFlagWater) != 0u;
    const float2 wrappedTileUv = frac(input.tileCoord);
    const float wavePhase = sin(input.worldPos.x * 0.0125f + input.worldPos.z * 0.0105f + uCameraPos.x * 0.0015f);
    const float2 waterUvOffset = isWater
                                     ? float2(wavePhase, wavePhase * 0.55f) * (input.atlasSize * 0.20f)
                                     : float2(0.0f, 0.0f);
    const float2 atlasUv = input.atlasBase + input.atlasSize * wrappedTileUv + waterUvOffset;
    const float2 atlasUvDdx = ddx(input.tileCoord) * input.atlasSize;
    const float2 atlasUvDdy = ddy(input.tileCoord) * input.atlasSize;
    const float4 textureSample = gAtlas.SampleGrad(gTerrainSampler, atlasUv, atlasUvDdx, atlasUvDdy);
    const uint grassTintIndex = decodeGrassTintIndex(input.materialFlags);
    float3 albedo = textureSample.rgb;
    if (grassTintIndex != 0u)
    {
        const float3 tint = biomeGrassTint(grassTintIndex);
        const float tintMask = isGrassSideTint(input.materialFlags) ? grassSideTintMask(wrappedTileUv) : 1.0f;
        albedo = lerp(albedo, albedo * tint, tintMask);
    }
    // Far LOD tiles are expected to behave like a sealed opaque terrain shell. Applying the
    // same alpha cutout threshold used by exact voxel geometry turns atlas mip bleed into
    // screen-door holes across the distant terrain, especially on grass and snow edges.
    if (!((input.materialFlags & kMaterialFlagFarLod) != 0u))
    {
        clip(textureSample.a - 0.5f);
    }

    const float skyLight = saturate(input.lightChannels.x);
    const float blockLight = saturate(input.lightChannels.y);
    const float aoStrength = max(uTerrainDebug.w, 0.01f);
    const float ao = pow(saturate(input.ao), aoStrength);
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
    const float indirectFaceShade = lerp(1.0f, faceShade, 0.55f);
    const float directFaceShade = lerp(1.0f, faceShade, 0.25f);
    const float diff = max(dot(normal, lightDir), 0.0f);
    const float hemi = saturate(normal.y * 0.5f + 0.5f);
    const float3 ambientTint = lerp(uGroundAmbient.rgb, uSkyAmbient.rgb, hemi);
    const float3 skyTint = ambientTint * 2.00f + float3(0.08f, 0.09f, 0.11f);
    const float3 skyIndirect = skyTint * skyLight;
    const float3 blockIndirect = float3(1.08f, 0.92f, 0.68f) * blockLight;
    const float indirectAo = lerp(1.0f, ao, 0.78f);
    const float3 indirect = (skyIndirect + blockIndirect) * indirectFaceShade * indirectAo;
    const float directSunGate = uTerrainDebug.x * saturate(skyLight * 1.05f);
    const float directAo = lerp(1.0f, ao, 0.18f);
    const float3 directLight =
        uSunColor.rgb * (diff * directSunGate * directFaceShade * directAo * 0.16f);
    const float3 baseBounce = ambientTint * 0.018f;

    float3 color = albedo * (indirect + baseBounce + directLight);
    if (isWater)
    {
        const float fresnel = pow(1.0f - saturate(dot(normal, viewDir)), 4.0f);
        const float shimmer = saturate(0.45f + 0.55f * sin(input.worldPos.x * 0.006f + input.worldPos.z * 0.008f));
        const float3 waterTint = lerp(textureSample.rgb, float3(0.18f, 0.42f, 0.66f), 0.58f);
        const float3 skyReflection =
            computeTerrainFogColor(normalize(float3(viewDir.x, abs(viewDir.y), viewDir.z)),
                                   uSkyTopColor.rgb,
                                   uSkyHorizonColor.rgb);
        color = waterTint * (indirect * 0.62f + baseBounce * 1.35f + directLight * 0.32f);
        color += skyReflection * (0.10f + fresnel * 0.28f);
        color += uSunColor.rgb * (0.02f + 0.05f * shimmer) * fresnel;
    }
    else if ((input.materialFlags & kMaterialFlagFarLod) != 0u)
    {
        // Optional far-only block edge hint for very coarse voxel scales. Near far levels rely on
        // greedy-merge clamping for blockiness to avoid a dense checker overlay.
        const float voxelScale = max(input.farVoxelScale, 1.0f);
        if (voxelScale >= 8.0f)
        {
            const float2 cell = input.tileCoord / voxelScale;
            const float2 f = frac(cell);
            const float2 distToEdge = min(f, 1.0f - f);
            const float edgeDist = min(distToEdge.x, distToEdge.y);
            const float edgeWidth = max(max(fwidth(cell.x), fwidth(cell.y)) * 1.45f, 1e-4f);
            const float edgeMask = 1.0f - smoothstep(0.0f, edgeWidth, edgeDist);
            color *= (1.0f - edgeMask * 0.12f);
        }
    }

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
        const float transmittance = saturate(max(aerial.a, 0.16f));
        color = color * transmittance + aerial.rgb;

        const FogBlendResult fogBlend = computeLayeredFog(distanceBlocks,
                                                          horizontalDistanceBlocks,
                                                          fogViewDir,
                                                          input.worldPos.y,
                                                          uCameraPos.y,
                                                          uParams1.z,
                                                          uParams1.w,
                                                          1.10f,
                                                          0.68f);
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
                                                          1.35f,
                                                          0.97f);
        color = color * fogBlend.transmittance + fogColor * fogBlend.inscatter;
    }

    return float4(color, 1.0f);
}
