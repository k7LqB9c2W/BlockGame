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

static const int kMinSentinel = -2147483647;

struct TerrainColumnInputGpu
{
    int solidTopY;
    uint solidBlock;
    int waterTopY;
    uint hasWater;
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

StructuredBuffer<TerrainColumnInputGpu> gColumns : register(t0);
StructuredBuffer<StructureVoxelInputGpu> gStructureVoxels : register(t1);
RWStructuredBuffer<PageCellGpu> gCells : register(u0);
RWStructuredBuffer<uint> gFaceMasks : register(u1);
RWStructuredBuffer<int> gSummary : register(u2);

uint pageCellIndex(uint x, uint y, uint z)
{
    return ((y * gGridCount) + z) * gGridCount + x;
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
    const uint columnIndex = z * gGridCount + x;
    const TerrainColumnInputGpu column = gColumns[columnIndex];
    const int cellMinY = gWorldMinY + int(y) * gCellScaleBlocks;
    const int cellTopY = cellMinY + (gCellScaleBlocks - 1);

    PageCellGpu cell = emptyCell();

    if (column.solidBlock != 0u && column.solidTopY >= cellTopY)
    {
        cell.solidTopY = cellTopY;
        cell.solidBlock = column.solidBlock;
        InterlockedMin(gSummary[0], cellMinY);
        InterlockedMax(gSummary[1], cellTopY + 1);
        InterlockedAdd(gSummary[6], 1);
    }
    else if (column.hasWater != 0u && column.waterTopY >= cellTopY)
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

bool occupiedAt(int x, int y, int z)
{
    if (x < 0 || y < 0 || z < 0 ||
        x >= int(gGridCount) || y >= int(gGridCount) || z >= int(gGridCount))
    {
        return false;
    }

    return isOccupied(gCells[pageCellIndex(uint(x), uint(y), uint(z))]);
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
