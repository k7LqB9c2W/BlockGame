cbuffer PageParams : register(b0)
{
    uint gGridCount;
    int gWorldMinX;
    int gWorldMinY;
    int gWorldMinZ;
    int gCellScaleBlocks;
    uint gCellCount;
    uint gStructureVoxelCount;
    uint gPadding0;
};

static const int kMinSentinel = (-2147483647 - 1);
static const uint kMacroSamplesPerColumn = 5u;
static const uint kBlockCount = 16u;
static const uint kFixedGridCount = 16u;

struct MacroTerrainSampleGpu
{
    int solidTopY;
    uint solidBlock;
    int waterTopY;
    int waterBottomY;
    uint hasWater;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct TerrainColumnGpu
{
    int solidTopY;
    int solidBottomY;
    uint solidBlock;
    int waterTopY;
    int waterBottomY;
    uint hasWater;
    uint padding;
};

struct StructureVoxelInputGpu
{
    int worldX;
    int worldY;
    int worldZ;
    uint block;
};

struct PageCellGpu
{
    int solidTopY;
    uint solidBlock;
    int waterTopY;
    uint hasWater;
    int canopyBaseY;
    int canopyTopY;
    uint canopyBlock;
    uint hasCanopy;
    int trunkTopY;
    uint trunkBlock;
    uint hasTrunk;
    uint padding;
};

StructuredBuffer<MacroTerrainSampleGpu> gMacroSamples : register(t0);
StructuredBuffer<StructureVoxelInputGpu> gStructureVoxels : register(t1);
RWStructuredBuffer<TerrainColumnGpu> gColumns : register(u0);
RWStructuredBuffer<PageCellGpu> gCells : register(u1);
RWStructuredBuffer<uint> gFaceMasks : register(u2);
RWStructuredBuffer<int> gSummary : register(u3);

groupshared TerrainColumnGpu gColumnStateA[kFixedGridCount * kFixedGridCount];
groupshared TerrainColumnGpu gColumnStateB[kFixedGridCount * kFixedGridCount];

uint pageColumnIndex(uint x, uint z)
{
    return z * gGridCount + x;
}

uint macroSampleIndex(uint x, uint z, uint sampleIndex)
{
    return pageColumnIndex(x, z) * kMacroSamplesPerColumn + sampleIndex;
}

uint pageCellIndex(uint x, uint y, uint z)
{
    return ((y * gGridCount) + z) * gGridCount + x;
}

int floorDivInt(int numerator, int denominator)
{
    int quotient = numerator / denominator;
    const int remainder = numerator % denominator;
    if (remainder != 0 && ((remainder < 0) != (denominator < 0)))
    {
        quotient -= 1;
    }
    return quotient;
}

int quantizeMacroTopY(int topBlockY)
{
    if (topBlockY == kMinSentinel || gCellScaleBlocks <= 1)
    {
        return topBlockY;
    }

    return floorDivInt(topBlockY, gCellScaleBlocks) * gCellScaleBlocks + (gCellScaleBlocks - 1);
}

int quantizeMacroBaseY(int blockY)
{
    if (blockY == kMinSentinel || gCellScaleBlocks <= 1)
    {
        return blockY;
    }

    return floorDivInt(blockY, gCellScaleBlocks) * gCellScaleBlocks;
}

bool hasSolidMacroSample(MacroTerrainSampleGpu sample)
{
    return sample.solidBlock != 0u && sample.solidTopY != kMinSentinel;
}

bool isSubmergedMacroSample(MacroTerrainSampleGpu sample)
{
    return sample.hasWater != 0u &&
           sample.waterTopY != kMinSentinel &&
           hasSolidMacroSample(sample) &&
           sample.solidTopY < sample.waterTopY;
}

bool hasSolidColumn(TerrainColumnGpu column)
{
    return column.solidBlock != 0u && column.solidTopY != kMinSentinel;
}

int resolvedSolidBottomY(TerrainColumnGpu column)
{
    if (column.solidBottomY != kMinSentinel)
    {
        return column.solidBottomY;
    }
    return quantizeMacroBaseY(column.solidTopY - gCellScaleBlocks + 1);
}

bool hasRenderableSolid(PageCellGpu cell)
{
    return cell.solidBlock != 0u && cell.solidTopY != kMinSentinel;
}

bool hasRenderableWater(PageCellGpu cell)
{
    return cell.hasWater != 0u && cell.waterTopY != kMinSentinel;
}

bool hasRenderableCanopy(PageCellGpu cell)
{
    return cell.hasCanopy != 0u &&
           cell.canopyBlock != 0u &&
           cell.canopyBaseY != kMinSentinel &&
           cell.canopyTopY != kMinSentinel;
}

bool hasRenderableTrunk(PageCellGpu cell)
{
    return cell.hasTrunk != 0u &&
           cell.trunkBlock != 0u &&
           cell.trunkTopY != kMinSentinel;
}

bool isOccupied(PageCellGpu cell)
{
    return hasRenderableTrunk(cell) ||
           hasRenderableCanopy(cell) ||
           hasRenderableSolid(cell) ||
           hasRenderableWater(cell);
}

PageCellGpu emptyCell()
{
    PageCellGpu cell;
    cell.solidTopY = kMinSentinel;
    cell.solidBlock = 0u;
    cell.waterTopY = kMinSentinel;
    cell.hasWater = 0u;
    cell.canopyBaseY = kMinSentinel;
    cell.canopyTopY = kMinSentinel;
    cell.canopyBlock = 0u;
    cell.hasCanopy = 0u;
    cell.trunkTopY = kMinSentinel;
    cell.trunkBlock = 0u;
    cell.hasTrunk = 0u;
    cell.padding = 0u;
    return cell;
}

void sortInt5(inout int values[5], uint count)
{
    [unroll]
    for (uint i = 1u; i < 5u; ++i)
    {
        if (i >= count)
        {
            break;
        }

        const int key = values[i];
        int j = int(i) - 1;
        [loop]
        while (j >= 0 && values[j] > key)
        {
            values[j + 1] = values[j];
            j -= 1;
        }
        values[j + 1] = key;
    }
}

void sortInt9(inout int values[9], uint count)
{
    [unroll]
    for (uint i = 1u; i < 9u; ++i)
    {
        if (i >= count)
        {
            break;
        }

        const int key = values[i];
        int j = int(i) - 1;
        [loop]
        while (j >= 0 && values[j] > key)
        {
            values[j + 1] = values[j];
            j -= 1;
        }
        values[j + 1] = key;
    }
}

uint dominantBlockForMacroCell(MacroTerrainSampleGpu samples[5])
{
    int counts[kBlockCount];
    [unroll]
    for (uint i = 0u; i < kBlockCount; ++i)
    {
        counts[i] = 0;
    }

    uint fallback = 1u;
    [unroll]
    for (uint sampleIndex = 0u; sampleIndex < kMacroSamplesPerColumn; ++sampleIndex)
    {
        const MacroTerrainSampleGpu sample = samples[sampleIndex];
        if (!hasSolidMacroSample(sample))
        {
            continue;
        }

        fallback = sample.solidBlock;
        if (isSubmergedMacroSample(sample))
        {
            continue;
        }

        if (sample.solidBlock < kBlockCount)
        {
            counts[sample.solidBlock] += 1;
        }
    }

    int bestCount = 0;
    uint bestBlock = fallback;
    [unroll]
    for (uint blockIndex = 0u; blockIndex < kBlockCount; ++blockIndex)
    {
        if (counts[blockIndex] > bestCount)
        {
            bestCount = counts[blockIndex];
            bestBlock = blockIndex;
        }
    }
    return bestBlock;
}

TerrainColumnGpu emptyColumn()
{
    TerrainColumnGpu column;
    column.solidTopY = kMinSentinel;
    column.solidBottomY = kMinSentinel;
    column.solidBlock = 0u;
    column.waterTopY = kMinSentinel;
    column.waterBottomY = kMinSentinel;
    column.hasWater = 0u;
    column.padding = 0u;
    return column;
}

bool occupiedAt(int x, int y, int z)
{
    if (x < 0 || y < 0 || z < 0 ||
        x >= int(gGridCount) || y >= int(gGridCount) || z >= int(gGridCount))
    {
        return false;
    }

    return isOccupied(gCells[pageCellIndex(uint(x), uint(y), uint(z))]);
}

[numthreads(16, 16, 1)]
void ColumnSynthesisMain(uint3 groupThreadId : SV_GroupThreadID)
{
    const bool active = groupThreadId.x < gGridCount && groupThreadId.y < gGridCount;
    const uint linearIndex = groupThreadId.y * kFixedGridCount + groupThreadId.x;

    TerrainColumnGpu column = emptyColumn();
    if (active)
    {
        MacroTerrainSampleGpu samples[5];
        int solidTopYs[5];
        int highestSolidTopY = kMinSentinel;
        int lowestSolidTopY = 2147483647;
        uint solidSampleCount = 0u;
        int highestWaterTopY = kMinSentinel;
        int lowestWaterBottomY = 2147483647;
        uint waterSampleCount = 0u;

        [unroll]
        for (uint sampleIndex = 0u; sampleIndex < kMacroSamplesPerColumn; ++sampleIndex)
        {
            const MacroTerrainSampleGpu sample = gMacroSamples[macroSampleIndex(groupThreadId.x, groupThreadId.y, sampleIndex)];
            samples[sampleIndex] = sample;
            if (hasSolidMacroSample(sample))
            {
                highestSolidTopY = max(highestSolidTopY, sample.solidTopY);
                lowestSolidTopY = min(lowestSolidTopY, sample.solidTopY);
                solidTopYs[solidSampleCount++] = sample.solidTopY;
            }
            if (sample.hasWater != 0u && sample.waterTopY != kMinSentinel)
            {
                highestWaterTopY = max(highestWaterTopY, sample.waterTopY);
                lowestWaterBottomY = min(lowestWaterBottomY,
                                         (sample.waterBottomY == kMinSentinel) ? sample.waterTopY : sample.waterBottomY);
                waterSampleCount += 1u;
            }
        }

        const MacroTerrainSampleGpu centerSample = samples[kMacroSamplesPerColumn - 1u];
        int representativeSolidTopY = highestSolidTopY;
        if (solidSampleCount > 0u)
        {
            sortInt5(solidTopYs, solidSampleCount);
            representativeSolidTopY = solidTopYs[solidSampleCount / 2u];
        }

        column.solidTopY = quantizeMacroTopY(representativeSolidTopY);
        if (hasSolidMacroSample(centerSample) &&
            abs(centerSample.solidTopY - representativeSolidTopY) <= gCellScaleBlocks)
        {
            column.solidBlock = centerSample.solidBlock;
        }
        else if (highestSolidTopY != kMinSentinel)
        {
            column.solidBlock = dominantBlockForMacroCell(samples);
        }

        const bool representativeWater =
            (centerSample.hasWater != 0u && centerSample.waterTopY != kMinSentinel) ||
            waterSampleCount >= 3u;
        column.waterTopY = quantizeMacroTopY((centerSample.hasWater != 0u && centerSample.waterTopY != kMinSentinel)
                                                 ? centerSample.waterTopY
                                                 : (representativeWater ? highestWaterTopY : kMinSentinel));
        column.waterBottomY = representativeWater ? quantizeMacroBaseY(lowestWaterBottomY) : kMinSentinel;
        column.hasWater = representativeWater ? 1u : 0u;
        column.solidBottomY = (lowestSolidTopY == 2147483647)
                                  ? kMinSentinel
                                  : quantizeMacroBaseY(lowestSolidTopY - gCellScaleBlocks + 1);
    }

    gColumnStateA[linearIndex] = column;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint iteration = 0u; iteration < 3u; ++iteration)
    {
        TerrainColumnGpu current = gColumnStateA[linearIndex];
        TerrainColumnGpu dst = current;
        if (active)
        {
            int supportingNeighbors = 0;
            int strongestNeighborTopY = kMinSentinel;
            uint strongestNeighborBlock = current.solidBlock;

            for (int neighborZ = max(0, int(groupThreadId.y) - 1);
                 neighborZ <= min(int(gGridCount) - 1, int(groupThreadId.y) + 1);
                 ++neighborZ)
            {
                for (int neighborX = max(0, int(groupThreadId.x) - 1);
                     neighborX <= min(int(gGridCount) - 1, int(groupThreadId.x) + 1);
                     ++neighborX)
                {
                    if (neighborX == int(groupThreadId.x) && neighborZ == int(groupThreadId.y))
                    {
                        continue;
                    }

                    const TerrainColumnGpu neighbor = gColumnStateA[pageColumnIndex(uint(neighborX), uint(neighborZ))];
                    if (!hasSolidColumn(neighbor))
                    {
                        continue;
                    }

                    supportingNeighbors += 1;
                    if (neighbor.solidTopY > strongestNeighborTopY)
                    {
                        strongestNeighborTopY = neighbor.solidTopY;
                        strongestNeighborBlock = neighbor.solidBlock;
                    }
                }
            }

            if (strongestNeighborTopY != kMinSentinel)
            {
                const int supportedTopY = strongestNeighborTopY - gCellScaleBlocks;
                if (!hasSolidColumn(current))
                {
                    if (supportingNeighbors >= 4)
                    {
                        dst.solidTopY = supportedTopY;
                        dst.solidBlock = strongestNeighborBlock;
                    }
                }
                else
                {
                    const int heightGap = supportedTopY - current.solidTopY;
                    if (supportingNeighbors >= 3 &&
                        heightGap > 0 &&
                        heightGap <= gCellScaleBlocks * 3)
                    {
                        dst.solidTopY = supportedTopY;
                        if (dst.solidBlock == 0u)
                        {
                            dst.solidBlock = strongestNeighborBlock;
                        }
                    }
                }
            }
        }

        gColumnStateB[linearIndex] = dst;
        GroupMemoryBarrierWithGroupSync();
        gColumnStateA[linearIndex] = gColumnStateB[linearIndex];
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint iteration = 0u; iteration < 2u; ++iteration)
    {
        TerrainColumnGpu current = gColumnStateA[linearIndex];
        TerrainColumnGpu dst = current;
        if (active && hasSolidColumn(current))
        {
            int neighborTopYs[9];
            uint neighborCount = 0u;

            for (int neighborZ = max(0, int(groupThreadId.y) - 1);
                 neighborZ <= min(int(gGridCount) - 1, int(groupThreadId.y) + 1);
                 ++neighborZ)
            {
                for (int neighborX = max(0, int(groupThreadId.x) - 1);
                     neighborX <= min(int(gGridCount) - 1, int(groupThreadId.x) + 1);
                     ++neighborX)
                {
                    const TerrainColumnGpu neighbor = gColumnStateA[pageColumnIndex(uint(neighborX), uint(neighborZ))];
                    if (!hasSolidColumn(neighbor))
                    {
                        continue;
                    }
                    neighborTopYs[neighborCount++] = neighbor.solidTopY;
                }
            }

            if (neighborCount >= 4u)
            {
                sortInt9(neighborTopYs, neighborCount);
                const int medianTopY = neighborTopYs[neighborCount / 2u];
                const int delta = current.solidTopY - medianTopY;
                if (delta > gCellScaleBlocks * 2)
                {
                    dst.solidTopY = current.solidTopY - gCellScaleBlocks;
                }
                else if (delta < -gCellScaleBlocks * 2)
                {
                    dst.solidTopY = current.solidTopY + gCellScaleBlocks;
                }
            }
        }

        gColumnStateB[linearIndex] = dst;
        GroupMemoryBarrierWithGroupSync();
        gColumnStateA[linearIndex] = gColumnStateB[linearIndex];
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint iteration = 0u; iteration < 2u; ++iteration)
    {
        TerrainColumnGpu current = gColumnStateA[linearIndex];
        TerrainColumnGpu dst = current;
        if (active && hasSolidColumn(current))
        {
            int neighborhoodMinBottomY = resolvedSolidBottomY(current);
            for (int neighborZ = max(0, int(groupThreadId.y) - 2);
                 neighborZ <= min(int(gGridCount) - 1, int(groupThreadId.y) + 2);
                 ++neighborZ)
            {
                for (int neighborX = max(0, int(groupThreadId.x) - 2);
                     neighborX <= min(int(gGridCount) - 1, int(groupThreadId.x) + 2);
                     ++neighborX)
                {
                    const TerrainColumnGpu neighbor = gColumnStateA[pageColumnIndex(uint(neighborX), uint(neighborZ))];
                    if (!hasSolidColumn(neighbor))
                    {
                        continue;
                    }
                    neighborhoodMinBottomY = min(neighborhoodMinBottomY, resolvedSolidBottomY(neighbor));
                }
            }

            dst.solidBottomY = min(resolvedSolidBottomY(current), neighborhoodMinBottomY);
        }

        gColumnStateB[linearIndex] = dst;
        GroupMemoryBarrierWithGroupSync();
        gColumnStateA[linearIndex] = gColumnStateB[linearIndex];
        GroupMemoryBarrierWithGroupSync();
    }

    if (active)
    {
        gColumns[pageColumnIndex(groupThreadId.x, groupThreadId.y)] = gColumnStateA[linearIndex];
    }
}

[numthreads(64, 1, 1)]
void SynthesizeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    if (index >= gCellCount)
    {
        return;
    }

    const uint x = index % gGridCount;
    const uint yz = index / gGridCount;
    const uint z = yz % gGridCount;
    const uint y = yz / gGridCount;
    const uint columnIndex = pageColumnIndex(x, z);
    const TerrainColumnGpu column = gColumns[columnIndex];
    const int cellMinY = gWorldMinY + int(y) * gCellScaleBlocks;
    const int cellTopY = cellMinY + (gCellScaleBlocks - 1);

    PageCellGpu cell = emptyCell();

    if (column.solidBlock != 0u &&
        column.solidTopY >= cellTopY &&
        column.solidBottomY <= cellMinY)
    {
        cell.solidTopY = cellTopY;
        cell.solidBlock = column.solidBlock;
        InterlockedMin(gSummary[0], cellMinY);
        InterlockedMax(gSummary[1], cellTopY + 1);
        InterlockedAdd(gSummary[6], 1);
    }
    // Water stays as a capped far-LOD shell so distant seas read as a continuous surface.
    else if (column.hasWater != 0u &&
             column.waterTopY >= cellMinY &&
             column.waterTopY <= cellTopY)
    {
        cell.waterTopY = cellTopY;
        cell.hasWater = 1u;
        InterlockedMax(gSummary[2], cellTopY + 1);
        InterlockedAdd(gSummary[3], 1);
        InterlockedAdd(gSummary[6], 1);
    }

    gCells[index] = cell;
    gFaceMasks[index] = 0u;
}

[numthreads(64, 1, 1)]
void StructureStampMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    if (index >= gCellCount)
    {
        return;
    }

    const uint x = index % gGridCount;
    const uint yz = index / gGridCount;
    const uint z = yz % gGridCount;
    const uint y = yz / gGridCount;
    const int cellMinX = gWorldMinX + int(x) * gCellScaleBlocks;
    const int cellMinY = gWorldMinY + int(y) * gCellScaleBlocks;
    const int cellMinZ = gWorldMinZ + int(z) * gCellScaleBlocks;
    const int cellMaxX = cellMinX + gCellScaleBlocks - 1;
    const int cellMaxY = cellMinY + gCellScaleBlocks - 1;
    const int cellMaxZ = cellMinZ + gCellScaleBlocks - 1;

    PageCellGpu cell = gCells[index];
    const bool occupiedBefore = isOccupied(cell);
    const bool hadCanopy = cell.hasCanopy != 0u;
    const bool hadTrunk = cell.hasTrunk != 0u;

    [loop]
    for (uint voxelIndex = 0; voxelIndex < gStructureVoxelCount; ++voxelIndex)
    {
        const StructureVoxelInputGpu voxel = gStructureVoxels[voxelIndex];
        if (voxel.worldX < cellMinX || voxel.worldX > cellMaxX ||
            voxel.worldY < cellMinY || voxel.worldY > cellMaxY ||
            voxel.worldZ < cellMinZ || voxel.worldZ > cellMaxZ)
        {
            continue;
        }

        if (voxel.block == 2u || voxel.block == 7u)
        {
            cell.hasTrunk = 1u;
            cell.trunkTopY = max(cell.trunkTopY, cellMaxY);
            cell.trunkBlock = voxel.block;
        }
        else if (voxel.block == 3u || voxel.block == 8u)
        {
            cell.hasCanopy = 1u;
            cell.canopyBaseY = (cell.canopyBaseY == kMinSentinel) ? cellMinY : min(cell.canopyBaseY, cellMinY);
            cell.canopyTopY = max(cell.canopyTopY, cellMaxY);
            cell.canopyBlock = voxel.block;
        }
    }

    if (!hadCanopy && cell.hasCanopy != 0u)
    {
        InterlockedAdd(gSummary[4], 1);
    }
    if (!hadTrunk && cell.hasTrunk != 0u)
    {
        InterlockedAdd(gSummary[5], 1);
    }
    if (!occupiedBefore && isOccupied(cell))
    {
        InterlockedAdd(gSummary[6], 1);
    }
    if (isOccupied(cell))
    {
        InterlockedMin(gSummary[0], cellMinY);
        InterlockedMax(gSummary[1], cellMaxY + 1);
    }

    gCells[index] = cell;
}

[numthreads(64, 1, 1)]
void FaceMaskMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    if (index >= gCellCount)
    {
        return;
    }

    const uint x = index % gGridCount;
    const uint yz = index / gGridCount;
    const uint z = yz % gGridCount;
    const uint y = yz / gGridCount;
    const PageCellGpu cell = gCells[index];
    if (!isOccupied(cell))
    {
        gFaceMasks[index] = 0u;
        return;
    }

    uint faceMask = 0u;
    if (!occupiedAt(int(x), int(y) + 1, int(z))) faceMask |= 1u << 0;
    if (!occupiedAt(int(x), int(y) - 1, int(z))) faceMask |= 1u << 1;
    if (!occupiedAt(int(x), int(y), int(z) - 1)) faceMask |= 1u << 2;
    if (!occupiedAt(int(x), int(y), int(z) + 1)) faceMask |= 1u << 3;
    if (!occupiedAt(int(x) + 1, int(y), int(z))) faceMask |= 1u << 4;
    if (!occupiedAt(int(x) - 1, int(y), int(z))) faceMask |= 1u << 5;

    gFaceMasks[index] = faceMask;
    InterlockedAdd(gSummary[7], countbits(faceMask));
}
