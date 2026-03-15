cbuffer ChunkParams : register(b0)
{
    int3 gWorldMin;
    int gBlockScale;
    int gSeaLevel;
};

struct GpuTerrainColumnDescriptor
{
    uint centerHasSolid;
    uint centerWaterEnabled;
    int centerSurfaceY;
    int centerWaterBottomY;
    uint centerSurfaceBlock;
    uint centerFillerBlock;
    int minSurfaceY;
    int maxSurfaceY;
    uint waterHitCount;
    int minWaterBottomY;
};

StructuredBuffer<GpuTerrainColumnDescriptor> gColumnBuffer : register(t0);
RWStructuredBuffer<uint> gVoxelBuffer : register(u0);

static const uint kLogicalSize = 16u;
static const uint kFlagWater = 0x01u;
static const uint kFlagTerrain = 0x08u;

uint columnIndex(uint localX, uint localZ)
{
    return localZ * kLogicalSize + localX;
}

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

[numthreads(4, 4, 1)]
void FarLodChunkFillMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= kLogicalSize || dispatchThreadId.y >= kLogicalSize)
    {
        return;
    }

    const uint localX = dispatchThreadId.x;
    const uint localZ = dispatchThreadId.y;
    const GpuTerrainColumnDescriptor descriptor = gColumnBuffer[columnIndex(localX, localZ)];

    [loop]
    for (uint localY = 0u; localY < kLogicalSize; ++localY)
    {
        const uint3 localCoord = uint3(localX, localY, localZ);
        const uint flatVoxelIndex = voxelIndex(localCoord);
        const int voxelMinY = gWorldMin.y + int(localY) * gBlockScale;
        const int voxelMaxY = voxelMinY + (gBlockScale - 1);

        uint solidHitCount = 0u;
        if (descriptor.centerHasSolid != 0u)
        {
            if (descriptor.maxSurfaceY >= voxelMinY && descriptor.minSurfaceY >= voxelMinY)
            {
                solidHitCount = 5u;
            }
            else if (descriptor.maxSurfaceY >= voxelMinY)
            {
                solidHitCount = 3u;
            }
        }

        uint packed = packVoxel(false, 0u, 0u);
        if (descriptor.centerHasSolid != 0u && solidHitCount >= 3u)
        {
            const uint material = (descriptor.minSurfaceY > voxelMaxY) ? descriptor.centerFillerBlock : descriptor.centerSurfaceBlock;
            packed = packVoxel(true, material, kFlagTerrain);
        }
        else if (descriptor.centerWaterEnabled != 0u &&
                 descriptor.waterHitCount >= 3u &&
                 descriptor.minWaterBottomY <= voxelMaxY &&
                 gSeaLevel >= voxelMinY)
        {
            packed = packVoxel(true, 5u, kFlagWater);
        }

        gVoxelBuffer[flatVoxelIndex] = packed;
    }
}
