cbuffer AtlasUpdateParams : register(b0)
{
    int gAtlasOriginCellX;
    int gAtlasOriginCellZ;
    int gAtlasSizeX;
    int gAtlasSizeZ;
    int gUpdateOriginCellX;
    int gUpdateOriginCellZ;
    int gUpdateSizeX;
    int gUpdateSizeZ;
    int gBlockScale;
    int gSeaLevel;
};

struct FarLodGpuWorldgenHeader
{
    int seaLevel;
    uint seed;
    uint biomeCount;
    float warpFrequency;
    float warpAmplitude;
    float mainFrequency;
    uint mainOctaves;
    float mainGain;
    float mainLacunarity;
    float mediumFrequency;
    uint mediumOctaves;
    float mediumGain;
    float mediumLacunarity;
    float detailFrequency;
    uint detailOctaves;
    float detailGain;
    float detailLacunarity;
    float mountainFrequency;
    uint mountainOctaves;
    float mountainGain;
    float mountainLacunarity;
};

struct FarLodGpuBiome
{
    uint flags;
    float spawnWeight;
    float minHeight;
    float maxHeight;
    float heightOffset;
    float heightScale;
    float roughness;
    float hills;
    float mountains;
    uint surfaceBlock;
    uint fillerBlock;
    int waterMaxDepth;
    int coastProfile;
    uint padding0;
    uint padding1;
};

struct GpuTerrainAtlasSample
{
    uint hasSolid;
    uint waterEnabled; // Aggregated water presence votes within this cell (0..N).
    int surfaceY;
    int waterBottomY;
    int minSurfaceY;
    int maxSurfaceY;
    uint surfaceBlock;
    uint fillerBlock;
};

StructuredBuffer<FarLodGpuWorldgenHeader> gWorldgenHeader : register(t0);
StructuredBuffer<FarLodGpuBiome> gBiomes : register(t1);
StructuredBuffer<GpuTerrainAtlasSample> gCanonicalSamples : register(t2);
RWStructuredBuffer<GpuTerrainAtlasSample> gAtlasSamples : register(u0);

int positiveModulo(int value, int divisor)
{
    const int result = value % divisor;
    return result < 0 ? result + divisor : result;
}

uint atlasIndex(int2 cellCoord)
{
    const int atlasX = positiveModulo(cellCoord.x - gAtlasOriginCellX, gAtlasSizeX);
    const int atlasZ = positiveModulo(cellCoord.y - gAtlasOriginCellZ, gAtlasSizeZ);
    return (uint)(atlasZ * gAtlasSizeX + atlasX);
}

[numthreads(8, 8, 1)]
void FarLodColumnAtlasUpdateMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= (uint)gUpdateSizeX || dispatchThreadId.y >= (uint)gUpdateSizeZ)
    {
        return;
    }

    const uint updateIndex = dispatchThreadId.y * (uint)gUpdateSizeX + dispatchThreadId.x;
    const int2 cellCoord = int2(gUpdateOriginCellX + (int)dispatchThreadId.x,
                                gUpdateOriginCellZ + (int)dispatchThreadId.y);

    gAtlasSamples[atlasIndex(cellCoord)] = gCanonicalSamples[updateIndex];
}
