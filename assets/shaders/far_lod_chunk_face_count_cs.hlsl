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
static const uint kSliceCount = 96u;

static const uint kBlockAir = 0u;
static const uint kBlockLeaves = 3u;
static const uint kBlockWater = 5u;
static const uint kBlockSpruceLeaves = 8u;

static const uint kFaceTop = 0u;
static const uint kFaceBottom = 1u;
static const uint kFaceNorth = 2u;
static const uint kFaceSouth = 3u;
static const uint kFaceEast = 4u;
static const uint kFaceWest = 5u;

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

bool faceVisible(uint faceId, uint x, uint y, uint z, uint packedVoxel)
{
    if (!isOccupied(packedVoxel))
    {
        return false;
    }

    const uint material = voxelMaterial(packedVoxel);
    if (faceId == kFaceTop)
    {
        return shouldRenderBlockFace(material, samplePositiveY(x, y, z));
    }
    if (faceId == kFaceBottom)
    {
        if (y == 0u && (gNegativeNeighborMask & kNegativeNeighborY) != 0u)
        {
            return false;
        }
        return shouldRenderBlockFace(material, (y > 0u) ? gVoxelBuffer[voxelIndex(x, y - 1u, z)] : 0u);
    }
    if (faceId == kFaceNorth)
    {
        if (z == 0u && (gNegativeNeighborMask & kNegativeNeighborZ) != 0u)
        {
            return false;
        }
        return shouldRenderBlockFace(material, (z > 0u) ? gVoxelBuffer[voxelIndex(x, y, z - 1u)] : 0u);
    }
    if (faceId == kFaceSouth)
    {
        return shouldRenderBlockFace(material, samplePositiveZ(x, y, z));
    }
    if (faceId == kFaceEast)
    {
        return shouldRenderBlockFace(material, samplePositiveX(x, y, z));
    }

    if (x == 0u && (gNegativeNeighborMask & kNegativeNeighborX) != 0u)
    {
        return false;
    }
    return shouldRenderBlockFace(material, (x > 0u) ? gVoxelBuffer[voxelIndex(x - 1u, y, z)] : 0u);
}

uint faceMergeKey(uint faceId, uint x, uint y, uint z, uint packedVoxel)
{
    if (!faceVisible(faceId, x, y, z, packedVoxel))
    {
        return 0u;
    }

    // Merge only when material + voxel flags match exactly (water/structure/cutout/terrain).
    // (Low bits 0x1f hold occupancy+flags; high bits hold the 8-bit material id.)
    // Exclude occupancy so "0" can be safely used as the sentinel for non-emitting cells.
    return packedVoxel & 0xFF1Eu;
}

void decodeSlice(uint sliceIndex, out uint faceId, out uint slice)
{
    faceId = sliceIndex / kLogicalSize;
    slice = sliceIndex - faceId * kLogicalSize;
}

void voxelCoordsForSliceCell(uint faceId, uint slice, uint u, uint v, out uint x, out uint y, out uint z)
{
    // Each slice is a 16x16 grid. Greedy merge runs in that 2D plane, per face and per slice.
    if (faceId == kFaceTop || faceId == kFaceBottom)
    {
        x = u;
        y = slice;
        z = v;
        return;
    }
    if (faceId == kFaceNorth || faceId == kFaceSouth)
    {
        x = u;
        y = v;
        z = slice;
        return;
    }

    // East/West: u maps to Z, v maps to Y.
    x = slice;
    y = v;
    z = u;
}

uint greedyQuadCount(uint faceId, uint slice)
{
    uint keys[256];
    uint visitedMask[16];
    for (uint initRow = 0u; initRow < 16u; ++initRow)
    {
        visitedMask[initRow] = 0u;
    }

    for (uint vFill = 0u; vFill < 16u; ++vFill)
    {
        for (uint uFill = 0u; uFill < 16u; ++uFill)
        {
            uint x;
            uint y;
            uint z;
            voxelCoordsForSliceCell(faceId, slice, uFill, vFill, x, y, z);
            const uint packedVoxel = gVoxelBuffer[voxelIndex(x, y, z)];
            keys[vFill * 16u + uFill] = faceMergeKey(faceId, x, y, z, packedVoxel);
        }
    }

    uint quadCount = 0u;
    for (uint vScan = 0u; vScan < 16u; ++vScan)
    {
        for (uint uScan = 0u; uScan < 16u; ++uScan)
        {
            const uint bit = 1u << uScan;
            if ((visitedMask[vScan] & bit) != 0u)
            {
                continue;
            }

            const uint key = keys[vScan * 16u + uScan];
            if (key == 0u)
            {
                visitedMask[vScan] |= bit;
                continue;
            }

            uint width = 1u;
            for (uint uNext = uScan + 1u; uNext < 16u; ++uNext)
            {
                const uint testBit = 1u << uNext;
                if ((visitedMask[vScan] & testBit) != 0u)
                {
                    break;
                }
                if (keys[vScan * 16u + uNext] != key)
                {
                    break;
                }
                width += 1u;
            }

            uint height = 1u;
            for (uint vNext = vScan + 1u; vNext < 16u; ++vNext)
            {
                bool rowOk = true;
                for (uint dx = 0u; dx < width; ++dx)
                {
                    const uint uTest = uScan + dx;
                    const uint testBit = 1u << uTest;
                    if ((visitedMask[vNext] & testBit) != 0u ||
                        keys[vNext * 16u + uTest] != key)
                    {
                        rowOk = false;
                        break;
                    }
                }
                if (!rowOk)
                {
                    break;
                }
                height += 1u;
            }

            const uint rowMask = ((1u << width) - 1u) << uScan;
            for (uint dy = 0u; dy < height; ++dy)
            {
                visitedMask[vScan + dy] |= rowMask;
            }

            quadCount += 1u;
        }
    }
    return quadCount;
}

[numthreads(64, 1, 1)]
void FarLodChunkFaceCountMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint linearIndex = dispatchThreadId.x;
    if (linearIndex >= kLogicalSize * kLogicalSize * kLogicalSize)
    {
        return;
    }

    // This buffer is scanned as a fixed 4096-element prefix sum by the existing pipeline.
    // We repurpose the first 96 entries (6 faces * 16 slices) to contain greedy-merged quad counts.
    // The remaining entries must be zeroed for deterministic emission.
    if (linearIndex >= kSliceCount)
    {
        gFaceCounts[linearIndex] = 0u;
        return;
    }

    uint faceId;
    uint slice;
    decodeSlice(linearIndex, faceId, slice);
    gFaceCounts[linearIndex] = greedyQuadCount(faceId, slice);
}
