Texture2D gAtlas : register(t0);
Texture2DArray gAerialPerspective : register(t1);
Texture2D gShadowMap : register(t2);

SamplerState gPointWrap : register(s0);
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
};

struct PSInput
{
    float4 position : SV_POSITION;
    float3 worldPos : POSITION0;
    float3 normal : NORMAL0;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
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

float4 main(PSInput input) : SV_TARGET
{
    const float3 normal = normalize(input.normal);
    const float3 lightDir = normalize(uLightDirection.xyz);
    const float3 viewDir = normalize(uCameraPos.xyz - input.worldPos);
    const float diff = max(dot(normal, lightDir), 0.0f);
    const float ambient = 0.45f;
    const float3 halfDir = normalize(lightDir + viewDir);
    const float spec = pow(max(dot(normal, halfDir), 0.0f), 24.0f);

    const float2 tileUv = frac(input.tileCoord);
    const float2 atlasUv = input.atlasBase + input.atlasSize * tileUv;
    const float4 textureSample = gAtlas.Sample(gPointWrap, atlasUv);
    clip(textureSample.a - 0.5f);

    const float shadow = sampleShadow(input.worldPos, normal);
    const float hemi = saturate(normal.y * 0.5f + 0.5f);
    const float3 ambientColor = lerp(uGroundAmbient.rgb, uSkyAmbient.rgb, hemi);
    const float3 directLight = uSunColor.rgb * (diff * shadow);
    const float3 specularLight = uSunColor.rgb * (spec * shadow * 0.12f);

    float3 color = textureSample.rgb * (ambientColor + directLight) + specularLight;

    if (uParams0.y > 0.5f)
    {
        const int3 currentBlock = int3(floor(input.worldPos));
        const int3 targetBlock = int3(uHighlightedBlock.xyz);
        if (all(currentBlock == targetBlock))
        {
            color = min(color + float3(0.3f, 0.3f, 0.3f), 1.0f.xxx);
        }
    }

    const float distanceBlocks = distance(input.worldPos, uCameraPos.xyz);
    if (uParams0.x > 0.5f)
    {
        const float2 screenUv = input.position.xy * uParams0.zw;
        const float4 aerial = sampleAerialPerspective(screenUv, distanceBlocks * 0.001f, uParams1.y);
        const float transmittance = saturate(max(aerial.a, 0.35f));
        color = color * transmittance + aerial.rgb;
    }
    else if (uParams1.w > uParams1.z)
    {
        const float fogFactor = saturate((distanceBlocks - uParams1.z) / (uParams1.w - uParams1.z));
        color = lerp(color, float3(0.55f, 0.78f, 0.95f), fogFactor);
    }

    return float4(color, 1.0f);
}
