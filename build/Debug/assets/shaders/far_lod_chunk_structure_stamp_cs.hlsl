cbuffer StructureStampParams : register(b0)
{
    int3 gWorldMin;
    int gBlockScale;
    uint gLodLevel;
    uint gStructureCount;
};

struct GpuStructureInstance
{
    uint type;
    int originX;
    int originY;
    int originZ;
    int boundsMinX;
    int boundsMinY;
    int boundsMinZ;
    int boundsMaxX;
    int boundsMaxY;
    int boundsMaxZ;
    uint trunkHeight;
    uint bareTrunkHeight;
    uint maxLodLevel;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

StructuredBuffer<GpuStructureInstance> gStructures : register(t0);
RWStructuredBuffer<uint> gVoxelBuffer : register(u0);

static const uint kLogicalSize = 16u;
static const uint kStructureTypeDefaultTree = 0u;
static const uint kStructureTypeTaigaSpruce = 1u;

static const uint kBlockAir = 0u;
static const uint kBlockWood = 2u;
static const uint kBlockLeaves = 3u;
static const uint kBlockSpruceLog = 7u;
static const uint kBlockSpruceLeaves = 8u;

static const uint kFlagStructure = 0x02u;
static const uint kFlagCutout = 0x04u;

uint voxelIndex(uint x, uint y, uint z)
{
    return (y * kLogicalSize + z) * kLogicalSize + x;
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

bool isOccupied(uint packedVoxel)
{
    return (packedVoxel & 0x1u) != 0u;
}

bool overlapsAabb(int3 minA, int3 maxA, int3 minB, int3 maxB)
{
    return minA.x <= maxB.x && maxA.x >= minB.x &&
           minA.y <= maxB.y && maxA.y >= minB.y &&
           minA.z <= maxB.z && maxA.z >= minB.z;
}

bool voxelContainsWorldBlock(int3 voxelMin, int3 voxelMax, int worldX, int worldY, int worldZ)
{
    return worldX >= voxelMin.x && worldX <= voxelMax.x &&
           worldY >= voxelMin.y && worldY <= voxelMax.y &&
           worldZ >= voxelMin.z && worldZ <= voxelMax.z;
}

uint structureBlockPriority(uint blockId)
{
    if (blockId == kBlockWood || blockId == kBlockSpruceLog)
    {
        return 2u;
    }
    if (blockId == kBlockLeaves || blockId == kBlockSpruceLeaves)
    {
        return 1u;
    }
    return 0u;
}

uint packStructureVoxel(uint blockId)
{
    uint flags = kFlagStructure;
    if (blockId == kBlockLeaves || blockId == kBlockSpruceLeaves)
    {
        flags |= kFlagCutout;
    }
    return packVoxel(blockId != kBlockAir, blockId, flags);
}

void considerStructureBlock(int3 voxelMin,
                            int3 voxelMax,
                            int worldX,
                            int worldY,
                            int worldZ,
                            uint blockId,
                            bool replaceSolid,
                            uint currentPackedVoxel,
                            inout bool hasCandidate,
                            inout uint candidatePriority,
                            inout uint candidatePackedVoxel)
{
    if (!voxelContainsWorldBlock(voxelMin, voxelMax, worldX, worldY, worldZ))
    {
        return;
    }

    if (!replaceSolid && isOccupied(currentPackedVoxel))
    {
        return;
    }

    const uint priority = structureBlockPriority(blockId);
    if (!hasCandidate || priority > candidatePriority)
    {
        hasCandidate = true;
        candidatePriority = priority;
        candidatePackedVoxel = packStructureVoxel(blockId);
    }
}

int taigaSpruceLeafRadiusForLayer(int layerFromBottom, int totalLayers)
{
    if (totalLayers <= 1)
    {
        return 0;
    }

    const float t = (float)layerFromBottom / (float)max(totalLayers - 1, 1);
    int radius = 1 + (int)round((1.0f - t) * 3.0f);
    if ((layerFromBottom % 3) == 0 && layerFromBottom < (totalLayers * 3) / 4)
    {
        radius = min(radius + 1, 4);
    }
    if (t > 0.88f)
    {
        radius = 1;
    }
    if (t > 0.97f)
    {
        radius = 0;
    }
    return clamp(radius, 0, 4);
}

int distanceToInclusiveRange(int value, int minValue, int maxValue)
{
    if (value < minValue)
    {
        return minValue - value;
    }
    if (value > maxValue)
    {
        return value - maxValue;
    }
    return 0;
}

bool taigaSpruceLeafOccupiesCell(int originX,
                                 int originZ,
                                 int worldX,
                                 int worldZ,
                                 int radius,
                                 int layerFromBottom,
                                 int totalLayers)
{
    if (radius <= 0)
    {
        return false;
    }

    if (worldX >= originX && worldX <= originX + 1 &&
        worldZ >= originZ && worldZ <= originZ + 1)
    {
        return false;
    }

    const int dx = distanceToInclusiveRange(worldX, originX, originX + 1);
    const int dz = distanceToInclusiveRange(worldZ, originZ, originZ + 1);
    const int chebyshev = max(dx, dz);
    if (chebyshev > radius)
    {
        return false;
    }

    int manhattanAllowance = radius + 1;
    if (radius >= 4 && layerFromBottom < totalLayers / 3)
    {
        manhattanAllowance += 1;
    }

    return (dx + dz) <= manhattanAllowance;
}

void stampDefaultTree(const GpuStructureInstance instance,
                      int3 voxelMin,
                      int3 voxelMax,
                      uint currentPackedVoxel,
                      inout bool hasCandidate,
                      inout uint candidatePriority,
                      inout uint candidatePackedVoxel)
{
    [loop]
    for (uint dy = 0u; dy < instance.trunkHeight; ++dy)
    {
        considerStructureBlock(voxelMin,
                               voxelMax,
                               instance.originX,
                               instance.originY + (int)dy,
                               instance.originZ,
                               kBlockWood,
                               true,
                               currentPackedVoxel,
                               hasCandidate,
                               candidatePriority,
                               candidatePackedVoxel);
    }

    const int canopyBaseWorld = instance.originY + (int)instance.trunkHeight - 3;
    const int canopyTopWorld = instance.originY + (int)instance.trunkHeight;
    [loop]
    for (int worldY = canopyBaseWorld; worldY <= canopyTopWorld; ++worldY)
    {
        const int layer = worldY - canopyBaseWorld;
        int radius = 2;
        if (worldY >= canopyTopWorld - 1)
        {
            radius = 1;
        }

        [loop]
        for (int dx = -radius; dx <= radius; ++dx)
        {
            [loop]
            for (int dz = -radius; dz <= radius; ++dz)
            {
                if (abs(dx) == radius && abs(dz) == radius && radius > 1)
                {
                    continue;
                }
                if (dx == 0 && dz == 0 && worldY <= instance.originY + (int)instance.trunkHeight - 1)
                {
                    continue;
                }
                if (layer == 0 && abs(dx) + abs(dz) > 3)
                {
                    continue;
                }

                considerStructureBlock(voxelMin,
                                       voxelMax,
                                       instance.originX + dx,
                                       worldY,
                                       instance.originZ + dz,
                                       kBlockLeaves,
                                       false,
                                       currentPackedVoxel,
                                       hasCandidate,
                                       candidatePriority,
                                       candidatePackedVoxel);
            }
        }
    }
}

void stampTaigaSpruce(const GpuStructureInstance instance,
                      int3 voxelMin,
                      int3 voxelMax,
                      uint currentPackedVoxel,
                      inout bool hasCandidate,
                      inout uint candidatePriority,
                      inout uint candidatePackedVoxel)
{
    [loop]
    for (int trunkX = 0; trunkX < 2; ++trunkX)
    {
        [loop]
        for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
        {
            [loop]
            for (uint dy = 1u; dy <= instance.trunkHeight; ++dy)
            {
                considerStructureBlock(voxelMin,
                                       voxelMax,
                                       instance.originX + trunkX,
                                       instance.originY + (int)dy,
                                       instance.originZ + trunkZ,
                                       kBlockSpruceLog,
                                       true,
                                       currentPackedVoxel,
                                       hasCandidate,
                                       candidatePriority,
                                       candidatePackedVoxel);
            }
        }
    }

    const int canopyBaseWorld = instance.originY + (int)instance.bareTrunkHeight + 1;
    const int canopyTopWorld = instance.originY + (int)instance.trunkHeight;
    const int totalLayers = max(1, canopyTopWorld - canopyBaseWorld + 1);
    [loop]
    for (int worldY = canopyBaseWorld; worldY <= canopyTopWorld; ++worldY)
    {
        const int layerFromBottom = worldY - canopyBaseWorld;
        const int radius = taigaSpruceLeafRadiusForLayer(layerFromBottom, totalLayers);
        if (radius <= 0)
        {
            continue;
        }

        [loop]
        for (int worldX = instance.originX - radius; worldX <= instance.originX + 1 + radius; ++worldX)
        {
            [loop]
            for (int worldZ = instance.originZ - radius; worldZ <= instance.originZ + 1 + radius; ++worldZ)
            {
                if (!taigaSpruceLeafOccupiesCell(instance.originX,
                                                 instance.originZ,
                                                 worldX,
                                                 worldZ,
                                                 radius,
                                                 layerFromBottom,
                                                 totalLayers))
                {
                    continue;
                }

                considerStructureBlock(voxelMin,
                                       voxelMax,
                                       worldX,
                                       worldY,
                                       worldZ,
                                       kBlockSpruceLeaves,
                                       false,
                                       currentPackedVoxel,
                                       hasCandidate,
                                       candidatePriority,
                                       candidatePackedVoxel);
            }
        }
    }

    const int crownWorldY = canopyTopWorld + 1;
    [loop]
    for (int trunkX = 0; trunkX < 2; ++trunkX)
    {
        [loop]
        for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
        {
            considerStructureBlock(voxelMin,
                                   voxelMax,
                                   instance.originX + trunkX,
                                   crownWorldY,
                                   instance.originZ + trunkZ,
                                   kBlockSpruceLeaves,
                                   false,
                                   currentPackedVoxel,
                                   hasCandidate,
                                   candidatePriority,
                                   candidatePackedVoxel);
        }
    }
}

[numthreads(64, 1, 1)]
void FarLodChunkStructureStampMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint linearIndex = dispatchThreadId.x;
    if (linearIndex >= kLogicalSize * kLogicalSize * kLogicalSize)
    {
        return;
    }

    const uint localX = linearIndex % kLogicalSize;
    const uint localY = (linearIndex / kLogicalSize) % kLogicalSize;
    const uint localZ = linearIndex / (kLogicalSize * kLogicalSize);

    const int3 voxelMin = gWorldMin + int3((int)localX, (int)localY, (int)localZ) * gBlockScale;
    const int3 voxelMax = voxelMin + int3(gBlockScale - 1, gBlockScale - 1, gBlockScale - 1);
    const uint currentPackedVoxel = gVoxelBuffer[linearIndex];

    bool hasCandidate = false;
    uint candidatePriority = 0u;
    uint candidatePackedVoxel = currentPackedVoxel;

    [loop]
    for (uint instanceIndex = 0u; instanceIndex < gStructureCount; ++instanceIndex)
    {
        const GpuStructureInstance instance = gStructures[instanceIndex];
        if (gLodLevel > instance.maxLodLevel)
        {
            continue;
        }

        if (!overlapsAabb(voxelMin,
                          voxelMax,
                          int3(instance.boundsMinX, instance.boundsMinY, instance.boundsMinZ),
                          int3(instance.boundsMaxX, instance.boundsMaxY, instance.boundsMaxZ)))
        {
            continue;
        }

        if (instance.type == kStructureTypeTaigaSpruce)
        {
            stampTaigaSpruce(instance,
                             voxelMin,
                             voxelMax,
                             currentPackedVoxel,
                             hasCandidate,
                             candidatePriority,
                             candidatePackedVoxel);
        }
        else
        {
            stampDefaultTree(instance,
                             voxelMin,
                             voxelMax,
                             currentPackedVoxel,
                             hasCandidate,
                             candidatePriority,
                             candidatePackedVoxel);
        }
    }

    if (hasCandidate)
    {
        gVoxelBuffer[linearIndex] = candidatePackedVoxel;
    }
}
