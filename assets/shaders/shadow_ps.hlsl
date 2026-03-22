// Terrain shadow-map pixel shader. It alpha-tests the atlas so cutout foliage
// shadows match the visible leaf texture instead of the full voxel cube.
Texture2D gAtlas : register(t0);

SamplerState gTerrainSampler : register(s0);

struct PSInput
{
    float4 position : SV_POSITION;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
};

void main(PSInput input)
{
    const float2 wrappedTileUv = frac(input.tileCoord);
    const float2 atlasUv = input.atlasBase + input.atlasSize * wrappedTileUv;
    const float alpha = gAtlas.SampleLevel(gTerrainSampler, atlasUv, 0.0f).a;
    clip(alpha - 0.5f);
}
