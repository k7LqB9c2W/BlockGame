cbuffer FaceCountParams : register(b0)
{
    uint gNegativeNeighborMask;
};

StructuredBuffer<uint> gVoxelBuffer : register(t0);
StructuredBuffer<uint> gNeighborPosX : register(t1);
StructuredBuffer<uint> gNeighborPosY : register(t2);
StructuredBuffer<uint> gNeighborPosZ : register(t3);
RWStructuredBuffer<uint> gFaceCounts : register(u0);

static const uint kLogicalSize = 16u;
static const uint kNegativeNeighborX = 0x1u;
static const uint kNegativeNeighborY = 0x2u;
static const uint kNegativeNeighborZ = 0x4u;

static const uint kBlockAir = 0u;
static const uint kBlockLeaves = 3u;
static const uint kBlockWater = 5u;
static const uint kBlockSpruceLeaves = 8u;

uint voxelIndex(uint x, uint y, uint z)
{
    return (y * kLogicalSize + z) * kLogicalSize + x;
}

bool isOccupied(uint packedVoxel)
{
    return (packedVoxel & 0x1u) != 0u;
}

uint voxelMaterial(uint packedVoxel)
{
    return (packedVoxel >> 8u) & 0xffu;
}

bool isAlphaCutoutBlock(uint blockId)
{
    return blockId == kBlockLeaves || blockId == kBlockSpruceLeaves;
}

bool isNonOpaqueBlock(uint blockId)
{
    return blockId == kBlockAir || blockId == kBlockWater || isAlphaCutoutBlock(blockId);
}

bool shouldRenderBlockFace(uint owningBlock, uint neighborBlock)
{
    if (owningBlock == kBlockAir)
    {
        return false;
    }

    if (neighborBlock == kBlockAir)
    {
        return true;
    }

    if (isAlphaCutoutBlock(owningBlock))
    {
        if (isAlphaCutoutBlock(neighborBlock))
        {
            return owningBlock != neighborBlock;
        }

        return neighborBlock == kBlockWater;
    }

    if (owningBlock == kBlockWater)
    {
        return neighborBlock == kBlockAir;
    }

    return isNonOpaqueBlock(neighborBlock);
}

uint samplePositiveX(uint x, uint y, uint z)
{
    return (x + 1u < kLogicalSize) ? gVoxelBuffer[voxelIndex(x + 1u, y, z)]
                                   : gNeighborPosX[voxelIndex(0u, y, z)];
}

uint samplePositiveY(uint x, uint y, uint z)
{
    return (y + 1u < kLogicalSize) ? gVoxelBuffer[voxelIndex(x, y + 1u, z)]
                                   : gNeighborPosY[voxelIndex(x, 0u, z)];
}

uint samplePositiveZ(uint x, uint y, uint z)
{
    return (z + 1u < kLogicalSize) ? gVoxelBuffer[voxelIndex(x, y, z + 1u)]
                                   : gNeighborPosZ[voxelIndex(x, y, 0u)];
}

uint countVisibleFaces(uint x, uint y, uint z, uint packedVoxel)
{
    if (!isOccupied(packedVoxel))
    {
        return 0u;
    }

    const uint material = voxelMaterial(packedVoxel);
    uint count = 0u;

    if (shouldRenderBlockFace(material, samplePositiveY(x, y, z)))
    {
        count += 1u;
    }

    if ((y != 0u || (gNegativeNeighborMask & kNegativeNeighborY) == 0u) &&
        shouldRenderBlockFace(material, (y > 0u) ? gVoxelBuffer[voxelIndex(x, y - 1u, z)] : 0u))
    {
        count += 1u;
    }

    if ((z != 0u || (gNegativeNeighborMask & kNegativeNeighborZ) == 0u) &&
        shouldRenderBlockFace(material, (z > 0u) ? gVoxelBuffer[voxelIndex(x, y, z - 1u)] : 0u))
    {
        count += 1u;
    }

    if (shouldRenderBlockFace(material, samplePositiveZ(x, y, z)))
    {
        count += 1u;
    }

    if (shouldRenderBlockFace(material, samplePositiveX(x, y, z)))
    {
        count += 1u;
    }

    if ((x != 0u || (gNegativeNeighborMask & kNegativeNeighborX) == 0u) &&
        shouldRenderBlockFace(material, (x > 0u) ? gVoxelBuffer[voxelIndex(x - 1u, y, z)] : 0u))
    {
        count += 1u;
    }

    return count;
}

[numthreads(64, 1, 1)]
void FarLodChunkFaceCountMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint linearIndex = dispatchThreadId.x;
    if (linearIndex >= kLogicalSize * kLogicalSize * kLogicalSize)
    {
        return;
    }

    const uint x = linearIndex % kLogicalSize;
    const uint y = (linearIndex / kLogicalSize) % kLogicalSize;
    const uint z = linearIndex / (kLogicalSize * kLogicalSize);
    const uint packedVoxel = gVoxelBuffer[linearIndex];
    gFaceCounts[linearIndex] = countVisibleFaces(x, y, z, packedVoxel);
}
