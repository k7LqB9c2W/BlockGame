cbuffer ChunkParams : register(b0)
{
    int3 gWorldMin;
    int gBlockScale;
    int gSeaLevel;
};

static const uint kLogicalSize = 16u;
static const uint kVoxelCount = kLogicalSize * kLogicalSize * kLogicalSize;

struct GpuTerrainFootprintSample
{
    int surfaceY;
    int waterBottomY;
    uint surfaceBlock;
    uint fillerBlock;
    uint flags;
};

StructuredBuffer<GpuTerrainFootprintSample> gFootprintSamples : register(t0);
RWStructuredBuffer<uint> gVoxelBuffer : register(u0);

uint voxelIndex(uint3 localCoord)
{
    return (localCoord.y * kLogicalSize + localCoord.z) * kLogicalSize + localCoord.x;
}

uint packVoxel(bool occupied, uint material, uint flags)
{
    uint packed = occupied ? 1u : 0u;
    if ((flags & 0x01u) != 0u) packed |= 0x2u;
    if ((flags & 0x02u) != 0u) packed |= 0x4u;
    if ((flags & 0x04u) != 0u) packed |= 0x8u;
    if ((flags & 0x08u) != 0u) packed |= 0x10u;
    packed |= ((material & 0xffu) << 8u);
    return packed;
}

[numthreads(4, 4, 4)]
void FarLodChunkSynthMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kLogicalSize || dispatchThreadId.y >= kLogicalSize || dispatchThreadId.z >= kLogicalSize)
    {
        return;
    }

    const uint flatVoxelIndex = voxelIndex(dispatchThreadId);
    const int voxelMinY = gWorldMin.y + int(dispatchThreadId.y) * gBlockScale;
    const int voxelMaxY = voxelMinY + gBlockScale - 1;
    const uint sampleBase = flatVoxelIndex * 5u;

    int minSurfaceY = 2147483647;
    uint solidHitCount = 0u;
    uint waterHitCount = 0u;
    int minWaterBottomY = 2147483647;
    GpuTerrainFootprintSample centerSample = (GpuTerrainFootprintSample)0;

    [unroll]
    for (uint sampleIndex = 0u; sampleIndex < 5u; ++sampleIndex)
    {
        const GpuTerrainFootprintSample sample = gFootprintSamples[sampleBase + sampleIndex];
        minSurfaceY = min(minSurfaceY, sample.surfaceY);
        if ((sample.flags & 0x1u) != 0u && sample.surfaceY >= voxelMinY)
        {
            solidHitCount += 1u;
        }
        if ((sample.flags & 0x2u) != 0u && sample.surfaceY < gSeaLevel)
        {
            waterHitCount += 1u;
            minWaterBottomY = min(minWaterBottomY, sample.waterBottomY);
        }
        if (sampleIndex == 4u)
        {
            centerSample = sample;
        }
    }

    uint packed = 0u;
    if ((centerSample.flags & 0x1u) != 0u && solidHitCount >= 3u)
    {
        const uint material = (minSurfaceY > voxelMaxY) ? centerSample.fillerBlock : centerSample.surfaceBlock;
        packed = packVoxel(true, material, 0x08u);
    }
    else if ((centerSample.flags & 0x2u) != 0u &&
             waterHitCount >= 3u &&
             minWaterBottomY <= voxelMaxY &&
             gSeaLevel >= voxelMinY)
    {
        packed = packVoxel(true, 5u, 0x01u);
    }

    gVoxelBuffer[flatVoxelIndex] = packed;
}
