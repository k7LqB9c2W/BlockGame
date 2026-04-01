static const uint kExactChunkSize = 16u;
static const uint kExactChunkColumnCount = kExactChunkSize * kExactChunkSize;
static const uint kExactChunkVoxelCount = kExactChunkSize * kExactChunkSize * kExactChunkSize;
static const uint kExactChunkPlaneCount = 102u;
static const uint kExactChunkMaxDescriptorsPerPlane = kExactChunkSize * kExactChunkSize;
static const uint kExactChunkFaceDescriptorCount = kExactChunkPlaneCount * kExactChunkMaxDescriptorsPerPlane;
static const uint kExactChunkHaloFaceVoxelCount = kExactChunkSize * kExactChunkSize;
static const uint kExactChunkHaloFaceCount = 6u;
static const uint kExactChunkHaloVoxelCount = kExactChunkHaloFaceCount * kExactChunkHaloFaceVoxelCount;
static const uint kWorldgenPageSize = 64u;
static const uint kWorldgenPageColumnCount = kWorldgenPageSize * kWorldgenPageSize;
static const uint kExactNeighborPosXBit = 1u << 0u;
static const uint kExactNeighborNegXBit = 1u << 1u;
static const uint kExactNeighborPosYBit = 1u << 2u;
static const uint kExactNeighborNegYBit = 1u << 3u;
static const uint kExactNeighborPosZBit = 1u << 4u;
static const uint kExactNeighborNegZBit = 1u << 5u;
static const uint kExactHaloFacePosX = 0u;
static const uint kExactHaloFaceNegX = 1u;
static const uint kExactHaloFacePosY = 2u;
static const uint kExactHaloFaceNegY = 3u;
static const uint kExactHaloFacePosZ = 4u;
static const uint kExactHaloFaceNegZ = 5u;
static const uint kMaterialFlagWater = 0x01u;
static const uint kMaterialFlagGrassTintShift = 2u;
static const uint kMaterialFlagGrassSideTint = 0x20u;
static const uint kExactDrawRecordOverflowFlag = 0x80000000u;
static const uint kExactDrawRecordActiveBit = 0x40000000u;
static const uint kExactDrawRecordFaceCountMask = 0x3fffffffu;
static const uint kExactCompletionStatusCompletedBit = 1u << 0u;
static const uint kExactCompletionStatusOverflowBit = 1u << 1u;
static const uint kExactCompletionStatusZeroFacesBit = 1u << 2u;
static const uint kExactCompletionStatusAllocatorExhaustedBit = 1u << 3u;
static const uint kExactChunkAllocationPhasePrepassSubmitted = 1u;
static const uint kExactChunkAllocationPhaseEmitSubmitted = 2u;
static const uint kChunkBufferPageStateFree = 0u;
static const uint kChunkBufferPageStateOpenWritable = 1u;
static const uint kChunkBufferPageStateSealedPendingResident = 2u;
static const uint kChunkBufferPageStateResidentImmutable = 3u;
static const uint kChunkBufferPageStateRetiring = 4u;
static const uint kChunkBufferPageUsageCpuUpload = 0u;
static const uint kChunkBufferPageUsageExactGpu = 1u;
static const uint kInvalidExactPageIndex = 0xffffffffu;
static const uint kInvalidExactRecordIndex = 0xffffffffu;

static const uint kBlockAir = 0u;
static const uint kBlockGrass = 1u;
static const uint kBlockLeaves = 3u;
static const uint kBlockSand = 4u;
static const uint kBlockWater = 5u;
static const uint kBlockSpruceLeaves = 8u;
static const uint kBlockPodzol = 9u;
static const uint kBlockDebugLamp = 10u;
static const uint kBlockDarkOakLeaves = 12u;
static const uint kBlockBirchLeaves = 14u;
static const uint kBlockAcaciaLeaves = 16u;
static const uint kBlockNeighborSolidSentinel = 255u;

static const uint kGrassTintDefault = 1u;
static const uint kGrassTintDarkForest = 2u;
static const uint kGrassTintTaiga = 3u;
static const uint kGrassTintWarm = 4u;

struct GpuExactColumnDescriptor
{
    int surfaceY;
    int highestSolidWorld;
    int waterTopWorld;
    int waterBottomWorld;
    int stripeOffset;
    uint flags;
    uint stripePeriod;
    uint stripeThickness;
    uint grassTintIndex;
    uint surfaceBlock;
    uint fillerBlock;
    uint waterBlock;
    uint stripeBlock;
    uint skyLightFromAbove;
    uint reserved1;
    uint reserved2;
};

struct GpuWorldgenPageColumn
{
    int surfaceY;
    float distanceToCoast;
    float soilCreepStrength;
    float stripeNoiseThreshold;
    uint packedBlocks;
    uint packedFlagsTintWaterDepth;
    uint packedSoilDepths;
    uint packedStripes;
};

struct GpuExactDescriptorBuildParams
{
    int chunkBaseWorldX;
    int chunkBaseWorldZ;
    int chunkMinWorldY;
    int sampleMinPageBaseWorldX;
    int sampleMinPageBaseWorldZ;
    uint pageIndex00;
    uint pageIndex10;
    uint pageIndex01;
    uint pageIndex11;
    uint skyLightOffset;
    uint descriptorOffset;
    uint reserved0;
};

struct GpuExactSparseVoxel
{
    uint packedLocalPos;
    uint block;
    uint flags;
    uint reserved;
};

struct GpuExactPrepassRecord
{
    int chunkWorldMinX;
    int chunkWorldMinY;
    int chunkWorldMinZ;
    uint scratchSliceIndex;
    uint descriptorOffset;
    uint rebuildVoxelInputs;
    uint sparseVoxelCount;
    uint resolvedNeighborMask;
    uint closedNeighborMask;
    uint pendingNeighborMask;
    uint centerVoxelSrvDescriptorIndex;
    uint centerVoxelUavDescriptorIndex;
    uint haloVoxelSrvDescriptorIndex;
    uint haloVoxelUavDescriptorIndex;
    uint lightScratchVoxelSrvDescriptorIndex;
    uint lightScratchVoxelUavDescriptorIndex;
    uint sparseVoxelSrvDescriptorIndex;
    uint neighborPosXVoxelSrvDescriptorIndex;
    uint neighborNegXVoxelSrvDescriptorIndex;
    uint neighborPosYVoxelSrvDescriptorIndex;
    uint neighborNegYVoxelSrvDescriptorIndex;
    uint neighborPosZVoxelSrvDescriptorIndex;
    uint neighborNegZVoxelSrvDescriptorIndex;
    uint reserved0;
};

struct GpuExactFaceDescriptor
{
    uint packedLocal;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct GpuExactOverflowEntry
{
    uint buildIndex;
    uint requiredFaces;
    uint reserved0;
    uint reserved1;
};

struct GpuExactAllocatorState
{
    uint pageCount;
    uint freePageCount;
    uint buildRecordCount;
    uint blockFaceUvDescriptorIndex;
};

struct GpuExactAllocatorPageMetadata
{
    uint pageIndex;
    uint usage;
    uint state;
    uint allocationLockWord;
    uint recordCapacity;
    uint vertexCapacity;
    uint indexCapacity;
    uint reserved0;
    uint vertexCursor;
    uint indexCursor;
    uint recordCursor;
    uint recordActiveCount;
    uint residentChunks;
    uint pendingChunks;
    uint vertexUavDescriptorIndex;
    uint indexUavDescriptorIndex;
    uint drawRecordUavDescriptorIndex;
    uint drawRecordMetadataUavDescriptorIndex;
    uint pendingBatchIdLo;
    uint pendingBatchIdHi;
    uint uploadFenceValueLo;
    uint uploadFenceValueHi;
    uint retireFenceValueLo;
    uint retireFenceValueHi;
};

struct GpuExactAllocatorFreePageEntry
{
    uint pageIndex;
};

struct GpuExactChunkAllocationRecord
{
    int chunkWorldMinX;
    int chunkWorldMinY;
    int chunkWorldMinZ;
    uint phase;
    uint statusFlags;
    uint buildVersion;
    uint generationEpoch;
    uint requiredFaceCount;
    uint pageIndex;
    uint recordIndex;
    uint vertexBase;
    uint indexBase;
    uint reservedFaceCapacity;
    uint centerVoxelSrvDescriptorIndex;
    uint haloSrvDescriptorIndex;
    uint reserved0;
    uint inputVersionLo;
    uint inputVersionHi;
    uint reserved1;
    uint reserved2;
};

struct GpuExactDrawRecordMetadata
{
    int chunkWorldMinX;
    int chunkWorldMinY;
    int chunkWorldMinZ;
    uint pageIndex;
    uint recordIndex;
    uint buildIndex;
    uint vertexBase;
    uint indexBase;
    uint faceCount;
    uint statusFlags;
    uint buildVersion;
    uint generationEpoch;
    uint inputVersionLo;
    uint inputVersionHi;
    uint reserved0;
    uint reserved1;
};

struct GpuExactCompletionEntry
{
    uint buildIndex;
    uint statusFlags;
    uint requiredFaces;
    uint reservedFaceCapacity;
    int chunkWorldMinX;
    int chunkWorldMinY;
    int chunkWorldMinZ;
    uint pageIndex;
    uint recordIndex;
    uint vertexBase;
    uint indexBase;
    uint buildVersion;
    uint generationEpoch;
    uint inputVersionLo;
    uint inputVersionHi;
    uint reserved0;
};

struct GpuBlockFaceUv
{
    float2 base;
    float2 size;
};

struct WorldVertex
{
    float3 position;
    float3 normal;
    float2 tileCoord;
    float2 atlasBase;
    float2 atlasSize;
    uint lightingData;
};

struct GpuCullRecord
{
    float4 boundsMin;
    float4 boundsMax;
    uint indexCount;
    uint firstIndexLocation;
    int baseVertex;
    uint reserved;
};

uint voxelIndex(uint x, uint y, uint z)
{
    return y * (kExactChunkSize * kExactChunkSize) + z * kExactChunkSize + x;
}

uint columnIndex(uint x, uint z)
{
    return z * kExactChunkSize + x;
}

uint haloFaceVoxelIndex(uint faceIndex, uint u, uint v)
{
    return faceIndex * kExactChunkHaloFaceVoxelCount + v * kExactChunkSize + u;
}

uint encodeVoxel(uint blockId, uint skyLight, uint blockLight)
{
    return (blockId & 0xFFu) | ((skyLight & 0x0Fu) << 8u) | ((blockLight & 0x0Fu) << 12u);
}

uint voxelBlock(uint packedVoxel)
{
    return packedVoxel & 0xFFu;
}

uint voxelSkyLight(uint packedVoxel)
{
    return (packedVoxel >> 8u) & 0x0Fu;
}

uint voxelBlockLight(uint packedVoxel)
{
    return (packedVoxel >> 12u) & 0x0Fu;
}

uint repackVoxelLights(uint packedVoxel, uint skyLight, uint blockLight)
{
    return encodeVoxel(voxelBlock(packedVoxel), skyLight, blockLight);
}

uint decodeLocalX(uint packedLocalPos)
{
    return packedLocalPos & 0x1Fu;
}

uint decodeLocalY(uint packedLocalPos)
{
    return (packedLocalPos >> 5u) & 0x1Fu;
}

uint decodeLocalZ(uint packedLocalPos)
{
    return (packedLocalPos >> 10u) & 0x1Fu;
}

uint packFaceLocal(uint x, uint y, uint z, uint faceId)
{
    return x | (y << 5u) | (z << 10u) | (faceId << 15u);
}

uint faceLocalX(uint packed)
{
    return packed & 0x1Fu;
}

uint faceLocalY(uint packed)
{
    return (packed >> 5u) & 0x1Fu;
}

uint faceLocalZ(uint packed)
{
    return (packed >> 10u) & 0x1Fu;
}

uint faceLocalFaceId(uint packed)
{
    return (packed >> 15u) & 0x7u;
}

bool isLeafBlock(uint blockId)
{
    return blockId == kBlockLeaves ||
           blockId == kBlockSpruceLeaves ||
           blockId == kBlockDarkOakLeaves ||
           blockId == kBlockBirchLeaves ||
           blockId == kBlockAcaciaLeaves;
}

bool isAlphaCutoutBlock(uint blockId)
{
    return isLeafBlock(blockId);
}

bool isOpaqueForLighting(uint blockId)
{
    return blockId != kBlockAir &&
           blockId != kBlockWater &&
           !isLeafBlock(blockId);
}

uint skyAttenuationForBlock(uint blockId)
{
    if (blockId == kBlockAir)
    {
        return 0u;
    }
    if (isLeafBlock(blockId))
    {
        return 1u;
    }
    if (blockId == kBlockWater)
    {
        return 2u;
    }
    return 15u;
}

uint propagationLossForBlock(uint blockId)
{
    if (blockId == kBlockAir)
    {
        return 1u;
    }
    if (isLeafBlock(blockId))
    {
        return 1u;
    }
    if (blockId == kBlockWater)
    {
        return 2u;
    }
    return 15u;
}

uint attenuateLight(uint light, uint loss)
{
    return (light > loss) ? (light - loss) : 0u;
}

uint blockEmissionForBlock(uint blockId)
{
    return blockId == kBlockDebugLamp ? 14u : 0u;
}

bool isAoSolid(uint blockId)
{
    return blockId != kBlockAir && blockId != kBlockWater;
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

    return neighborBlock == kBlockWater || isAlphaCutoutBlock(neighborBlock);
}

uint sampleHaloVoxel(StructuredBuffer<uint> haloBuffer, uint seamBit, int x, int y, int z)
{
    uint faceIndex = 0u;
    uint u = 0u;
    uint v = 0u;

    if (seamBit == kExactNeighborPosXBit)
    {
        faceIndex = kExactHaloFacePosX;
        u = uint(z);
        v = uint(y);
    }
    else if (seamBit == kExactNeighborNegXBit)
    {
        faceIndex = kExactHaloFaceNegX;
        u = uint(z);
        v = uint(y);
    }
    else if (seamBit == kExactNeighborPosYBit)
    {
        faceIndex = kExactHaloFacePosY;
        u = uint(x);
        v = uint(z);
    }
    else if (seamBit == kExactNeighborNegYBit)
    {
        faceIndex = kExactHaloFaceNegY;
        u = uint(x);
        v = uint(z);
    }
    else if (seamBit == kExactNeighborPosZBit)
    {
        faceIndex = kExactHaloFacePosZ;
        u = uint(x);
        v = uint(y);
    }
    else if (seamBit == kExactNeighborNegZBit)
    {
        faceIndex = kExactHaloFaceNegZ;
        u = uint(x);
        v = uint(y);
    }
    else
    {
        return encodeVoxel(kBlockAir, 0u, 0u);
    }

    if (u >= kExactChunkSize || v >= kExactChunkSize)
    {
        return encodeVoxel(kBlockAir, 0u, 0u);
    }

    return haloBuffer[haloFaceVoxelIndex(faceIndex, u, v)];
}

float2 projectTileCoord(uint faceId, float3 position)
{
    if (faceId == 0u || faceId == 1u)
    {
        return float2(position.x, position.z);
    }
    if (faceId == 4u || faceId == 5u)
    {
        return float2(position.z, position.y);
    }
    return float2(position.x, position.y);
}

uint packLightingData(uint skyLight, uint blockLight, uint aoLevel, uint materialFlags)
{
    const uint packedLight = ((skyLight & 0x0Fu) << 4u) | (blockLight & 0x0Fu);
    return packedLight | ((aoLevel & 0x03u) << 8u) | ((materialFlags & 0x3Fu) << 10u);
}
