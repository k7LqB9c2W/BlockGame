#include "exact_chunk_common.hlsli"

cbuffer ExactChunkFaceEmitParams : register(b0)
{
    uint gBatchBuildCount;
    uint gReserved0;
    uint gReserved1;
    uint gReserved2;
};

StructuredBuffer<GpuExactEmitConfig> gEmitConfigBuffer : register(t0);
StructuredBuffer<GpuExactChunkAllocationRecord> gBuildRecords : register(t1);
StructuredBuffer<GpuExactPageMetadata> gPageMetadata : register(t2);
StructuredBuffer<GpuExactColumnDescriptor> gColumnScratch : register(t3);
StructuredBuffer<uint> gFaceCountScratch : register(t4);
StructuredBuffer<GpuExactFaceDescriptor> gFaceDescriptorScratch : register(t5);
StructuredBuffer<uint> gFacePrefixScratch : register(t6);
StructuredBuffer<uint> gBatchBuildIndices : register(t7);
RWStructuredBuffer<uint> gOverflowCount : register(u0);
RWStructuredBuffer<GpuExactOverflowEntry> gOverflowEntries : register(u1);
RWStructuredBuffer<GpuExactCompletionEntry> gCompletionEntries : register(u2);

static const uint kLightingFlagAlphaCutoutBit = 1u << 24u;
static const uint kExactIndirectRootBufferAlignment = 256u;
static const uint kExactFaceCountScratchStride =
    (((kExactChunkPlaneCount * 4u) + kExactIndirectRootBufferAlignment - 1u) / kExactIndirectRootBufferAlignment) *
    (kExactIndirectRootBufferAlignment / 4u);
static const uint kExactFacePrefixScratchStride = kExactFaceCountScratchStride;

struct GpuExactDrawRecord
{
    float4 boundsMin;
    float4 boundsMax;
    uint faceCount;
    uint faceOffset;
    uint reserved0;
    uint reserved;
};

uint sampleVoxel(StructuredBuffer<uint> bufferRef, int x, int y, int z)
{
    if (x < 0 || y < 0 || z < 0 ||
        x >= int(kExactChunkSize) || y >= int(kExactChunkSize) || z >= int(kExactChunkSize))
    {
        return encodeVoxel(kBlockAir, 0u, 0u);
    }

    return bufferRef[voxelIndex(uint(x), uint(y), uint(z))];
}

uint sampleVoxelWithNeighbors(StructuredBuffer<uint> centerVoxels,
                              StructuredBuffer<uint> haloVoxels,
                              int x,
                              int y,
                              int z)
{
    if (x >= 0 && y >= 0 && z >= 0 &&
        x < int(kExactChunkSize) && y < int(kExactChunkSize) && z < int(kExactChunkSize))
    {
        return sampleVoxel(centerVoxels, x, y, z);
    }

    uint seamBit = 0u;
    if (x == int(kExactChunkSize) && y >= 0 && y < int(kExactChunkSize) && z >= 0 && z < int(kExactChunkSize))
    {
        seamBit = kExactNeighborPosXBit;
        x = 0;
    }
    else if (x == -1 && y >= 0 && y < int(kExactChunkSize) && z >= 0 && z < int(kExactChunkSize))
    {
        seamBit = kExactNeighborNegXBit;
        x = int(kExactChunkSize) - 1;
    }
    else if (y == int(kExactChunkSize) && x >= 0 && x < int(kExactChunkSize) && z >= 0 && z < int(kExactChunkSize))
    {
        seamBit = kExactNeighborPosYBit;
        y = 0;
    }
    else if (y == -1 && x >= 0 && x < int(kExactChunkSize) && z >= 0 && z < int(kExactChunkSize))
    {
        seamBit = kExactNeighborNegYBit;
        y = int(kExactChunkSize) - 1;
    }
    else if (z == int(kExactChunkSize) && x >= 0 && x < int(kExactChunkSize) && y >= 0 && y < int(kExactChunkSize))
    {
        seamBit = kExactNeighborPosZBit;
        z = 0;
    }
    else if (z == -1 && x >= 0 && x < int(kExactChunkSize) && y >= 0 && y < int(kExactChunkSize))
    {
        seamBit = kExactNeighborNegZBit;
        z = int(kExactChunkSize) - 1;
    }

    if (seamBit == 0u)
    {
        return encodeVoxel(kBlockAir, 0u, 0u);
    }
    return sampleHaloVoxel(haloVoxels, seamBit, x, y, z);
}

uint sampleLightingVoxelWithNeighbors(StructuredBuffer<uint> centerVoxels,
                                      StructuredBuffer<uint> haloVoxels,
                                      int x,
                                      int y,
                                      int z)
{
    if (x >= 0 && y >= 0 && z >= 0 &&
        x < int(kExactChunkSize) && y < int(kExactChunkSize) && z < int(kExactChunkSize))
    {
        return sampleVoxel(centerVoxels, x, y, z);
    }

    const bool posX = x >= int(kExactChunkSize);
    const bool negX = x < 0;
    const bool posY = y >= int(kExactChunkSize);
    const bool negY = y < 0;
    const bool posZ = z >= int(kExactChunkSize);
    const bool negZ = z < 0;

    const int clampedX = min(max(x, 0), int(kExactChunkSize) - 1);
    const int clampedY = min(max(y, 0), int(kExactChunkSize) - 1);
    const int clampedZ = min(max(z, 0), int(kExactChunkSize) - 1);

    uint packedSamples[3];
    uint sampleCount = 0u;

    if (posX)
    {
        packedSamples[sampleCount++] = sampleHaloVoxel(haloVoxels, kExactNeighborPosXBit, 0, clampedY, clampedZ);
    }
    else if (negX)
    {
        packedSamples[sampleCount++] =
            sampleHaloVoxel(haloVoxels, kExactNeighborNegXBit, int(kExactChunkSize) - 1, clampedY, clampedZ);
    }

    if (posY)
    {
        packedSamples[sampleCount++] = sampleHaloVoxel(haloVoxels, kExactNeighborPosYBit, clampedX, 0, clampedZ);
    }
    else if (negY)
    {
        packedSamples[sampleCount++] =
            sampleHaloVoxel(haloVoxels, kExactNeighborNegYBit, clampedX, int(kExactChunkSize) - 1, clampedZ);
    }

    if (posZ)
    {
        packedSamples[sampleCount++] = sampleHaloVoxel(haloVoxels, kExactNeighborPosZBit, clampedX, clampedY, 0);
    }
    else if (negZ)
    {
        packedSamples[sampleCount++] =
            sampleHaloVoxel(haloVoxels, kExactNeighborNegZBit, clampedX, clampedY, int(kExactChunkSize) - 1);
    }

    if (sampleCount == 0u)
    {
        return encodeVoxel(kBlockAir, 0u, 0u);
    }

    if (sampleCount == 1u)
    {
        return packedSamples[0];
    }

    uint maxSky = voxelSkyLight(packedSamples[0]);
    uint maxBlock = voxelBlockLight(packedSamples[0]);
    [unroll]
    for (uint i = 0u; i < 3u; ++i)
    {
        if (i >= sampleCount)
        {
            break;
        }

        maxSky = max(maxSky, voxelSkyLight(packedSamples[i]));
        maxBlock = max(maxBlock, voxelBlockLight(packedSamples[i]));
    }

    uint skySum = 0u;
    uint blockSum = 0u;
    [unroll]
    for (uint i = 0u; i < 3u; ++i)
    {
        if (i >= sampleCount)
        {
            break;
        }

        skySum += voxelSkyLight(packedSamples[i]);
        blockSum += voxelBlockLight(packedSamples[i]);
    }

    return encodeVoxel(kBlockAir,
                       max(maxSky, (skySum + sampleCount / 2u) / sampleCount),
                       max(maxBlock, (blockSum + sampleCount / 2u) / sampleCount));
}

void faceVectors(uint faceId, out int3 outward, out int3 sideU, out int3 sideV, out float3 normal)
{
    if (faceId == 0u)
    {
        outward = int3(0, 1, 0);
        sideU = int3(1, 0, 0);
        sideV = int3(0, 0, 1);
        normal = float3(0.0f, 1.0f, 0.0f);
        return;
    }
    if (faceId == 1u)
    {
        outward = int3(0, -1, 0);
        sideU = int3(1, 0, 0);
        sideV = int3(0, 0, -1);
        normal = float3(0.0f, -1.0f, 0.0f);
        return;
    }
    if (faceId == 2u)
    {
        outward = int3(0, 0, -1);
        sideU = int3(1, 0, 0);
        sideV = int3(0, 1, 0);
        normal = float3(0.0f, 0.0f, -1.0f);
        return;
    }
    if (faceId == 3u)
    {
        outward = int3(0, 0, 1);
        sideU = int3(-1, 0, 0);
        sideV = int3(0, 1, 0);
        normal = float3(0.0f, 0.0f, 1.0f);
        return;
    }
    if (faceId == 4u)
    {
        outward = int3(1, 0, 0);
        sideU = int3(0, 0, 1);
        sideV = int3(0, 1, 0);
        normal = float3(1.0f, 0.0f, 0.0f);
        return;
    }

    outward = int3(-1, 0, 0);
    sideU = int3(0, 0, -1);
    sideV = int3(0, 1, 0);
    normal = float3(-1.0f, 0.0f, 0.0f);
}

uint grassMaterialFlags(uint blockId, uint faceId, GpuExactColumnDescriptor column)
{
    if (blockId != kBlockGrass)
    {
        return 0u;
    }

    const uint tintIndex = (column.grassTintIndex == 0u) ? kGrassTintDefault : column.grassTintIndex;
    if (faceId == 0u)
    {
        return tintIndex << kMaterialFlagGrassTintShift;
    }
    if (faceId == 1u)
    {
        return 0u;
    }
    return (tintIndex << kMaterialFlagGrassTintShift) | kMaterialFlagGrassSideTint;
}

uint materialFlagsForFace(uint blockId, uint faceId, GpuExactColumnDescriptor column)
{
    uint flags = grassMaterialFlags(blockId, faceId, column);
    if (blockId == kBlockWater)
    {
        flags |= kMaterialFlagWater;
    }
    return flags;
}

uint cornerIndexForSigns(int uSign, int vSign)
{
    if (uSign > 0)
    {
        return vSign > 0 ? 2u : 1u;
    }
    return vSign > 0 ? 3u : 0u;
}

int lightingMetricFromPackedVertex(uint packedLighting)
{
    const uint packedLight = packedLighting & 0xFFu;
    const int sky = int((packedLight >> 4u) & 0x0Fu);
    const int block = int(packedLight & 0x0Fu);
    const int ao = int((packedLighting >> 8u) & 0x03u);
    return sky * 24 + block * 18 + (3 - ao) * 20;
}

uint buildCornerLighting(StructuredBuffer<uint> centerVoxels,
                         StructuredBuffer<uint> haloVoxels,
                         int3 owningLocal,
                         uint faceId,
                         int uSign,
                         int vSign)
{
    int3 outward;
    int3 sideU;
    int3 sideV;
    float3 normalUnused;
    faceVectors(faceId, outward, sideU, sideV, normalUnused);

    const int3 fallbackSample = owningLocal + outward;
    const int3 sample0 = fallbackSample;
    const int3 sample1 = fallbackSample + sideU * uSign;
    const int3 sample2 = fallbackSample + sideV * vSign;
    const int3 sample3 = fallbackSample + sideU * uSign + sideV * vSign;

    uint skySum = 0u;
    uint blockSum = 0u;
    uint validSamples = 0u;
    const uint packedSamples[4] = {
        sampleLightingVoxelWithNeighbors(centerVoxels, haloVoxels, sample0.x, sample0.y, sample0.z),
        sampleLightingVoxelWithNeighbors(centerVoxels, haloVoxels, sample1.x, sample1.y, sample1.z),
        sampleLightingVoxelWithNeighbors(centerVoxels, haloVoxels, sample2.x, sample2.y, sample2.z),
        sampleLightingVoxelWithNeighbors(centerVoxels, haloVoxels, sample3.x, sample3.y, sample3.z)
    };

    [unroll]
    for (uint i = 0u; i < 4u; ++i)
    {
        const uint sampleBlock = voxelBlock(packedSamples[i]);
        if (isOpaqueForLighting(sampleBlock))
        {
            continue;
        }

        skySum += voxelSkyLight(packedSamples[i]);
        blockSum += voxelBlockLight(packedSamples[i]);
        validSamples += 1u;
    }

    uint skyLight = 0u;
    uint blockLight = 0u;
    if (validSamples > 0u)
    {
        skyLight = (skySum + validSamples / 2u) / validSamples;
        blockLight = (blockSum + validSamples / 2u) / validSamples;
    }
    else
    {
        const uint fallbackPacked =
            sampleLightingVoxelWithNeighbors(centerVoxels, haloVoxels, fallbackSample.x, fallbackSample.y, fallbackSample.z);
        skyLight = voxelSkyLight(fallbackPacked);
        blockLight = voxelBlockLight(fallbackPacked);
    }

    const uint side1Solid =
        isAoSolid(voxelBlock(sampleLightingVoxelWithNeighbors(centerVoxels, haloVoxels, sample1.x, sample1.y, sample1.z))) ? 1u : 0u;
    const uint side2Solid =
        isAoSolid(voxelBlock(sampleLightingVoxelWithNeighbors(centerVoxels, haloVoxels, sample2.x, sample2.y, sample2.z))) ? 1u : 0u;
    const uint cornerSolid =
        isAoSolid(voxelBlock(sampleLightingVoxelWithNeighbors(centerVoxels, haloVoxels, sample3.x, sample3.y, sample3.z))) ? 1u : 0u;
    const uint aoLevel = (side1Solid != 0u && side2Solid != 0u) ? 3u : (side1Solid + side2Solid + cornerSolid);
    return packLightingData(skyLight, blockLight, aoLevel, 0u);
}

void faceVertices(int3 chunkWorldMin,
                  uint3 localPos,
                  uint faceId,
                  out float3 p0,
                  out float3 p1,
                  out float3 p2,
                  out float3 p3,
                  out float3 normal)
{
    const float3 base = float3(float(chunkWorldMin.x + int(localPos.x)),
                               float(chunkWorldMin.y + int(localPos.y)),
                               float(chunkWorldMin.z + int(localPos.z)));

    if (faceId == 0u)
    {
        p0 = base + float3(0.0f, 1.0f, 0.0f);
        p1 = base + float3(0.0f, 1.0f, 1.0f);
        p2 = base + float3(1.0f, 1.0f, 1.0f);
        p3 = base + float3(1.0f, 1.0f, 0.0f);
        normal = float3(0.0f, 1.0f, 0.0f);
        return;
    }
    if (faceId == 1u)
    {
        p0 = base + float3(0.0f, 0.0f, 0.0f);
        p1 = base + float3(1.0f, 0.0f, 0.0f);
        p2 = base + float3(1.0f, 0.0f, 1.0f);
        p3 = base + float3(0.0f, 0.0f, 1.0f);
        normal = float3(0.0f, -1.0f, 0.0f);
        p1 = base + float3(1.0f, 0.0f, 0.0f);
        p3 = base + float3(0.0f, 0.0f, 1.0f);
        return;
    }
    if (faceId == 2u)
    {
        p0 = base + float3(0.0f, 0.0f, 0.0f);
        p1 = base + float3(0.0f, 1.0f, 0.0f);
        p2 = base + float3(1.0f, 1.0f, 0.0f);
        p3 = base + float3(1.0f, 0.0f, 0.0f);
        normal = float3(0.0f, 0.0f, -1.0f);
        return;
    }
    if (faceId == 3u)
    {
        p0 = base + float3(0.0f, 0.0f, 1.0f);
        p1 = base + float3(1.0f, 0.0f, 1.0f);
        p2 = base + float3(1.0f, 1.0f, 1.0f);
        p3 = base + float3(0.0f, 1.0f, 1.0f);
        normal = float3(0.0f, 0.0f, 1.0f);
        return;
    }
    if (faceId == 4u)
    {
        p0 = base + float3(1.0f, 0.0f, 0.0f);
        p1 = base + float3(1.0f, 1.0f, 0.0f);
        p2 = base + float3(1.0f, 1.0f, 1.0f);
        p3 = base + float3(1.0f, 0.0f, 1.0f);
        normal = float3(1.0f, 0.0f, 0.0f);
        return;
    }

    p0 = base + float3(0.0f, 0.0f, 0.0f);
    p1 = base + float3(0.0f, 0.0f, 1.0f);
    p2 = base + float3(0.0f, 1.0f, 1.0f);
    p3 = base + float3(0.0f, 1.0f, 0.0f);
    normal = float3(-1.0f, 0.0f, 0.0f);
}

[numthreads(64, 1, 1)]
void ExactChunkFaceEmitMain(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint planeIndex = groupId.x;
    const uint buildOrdinal = groupId.y;
    if (planeIndex >= kExactChunkPlaneCount)
    {
        return;
    }
    if (buildOrdinal >= gBatchBuildCount)
    {
        return;
    }

    const GpuExactEmitConfig emitConfig = gEmitConfigBuffer[0];
    const uint buildIndex = gBatchBuildIndices[buildOrdinal];
    const bool validBuildIndex = buildIndex < emitConfig.buildRecordCount;
    GpuExactChunkAllocationRecord build = (GpuExactChunkAllocationRecord)0;
    if (validBuildIndex)
    {
        build = gBuildRecords[buildIndex];
    }
    const bool emitPhaseSubmitted =
        validBuildIndex &&
        build.phase == kExactChunkAllocationPhaseEmitSubmitted;
    const bool zeroFaceBuild =
        emitPhaseSubmitted &&
        build.requiredFaceCount == 0u;
    const bool validPageIndex =
        emitPhaseSubmitted &&
        build.pageIndex != 0xffffffffu &&
        build.pageIndex < emitConfig.pageCount;
    GpuExactPageMetadata page = (GpuExactPageMetadata)0;
    if (validPageIndex)
    {
        page = gPageMetadata[build.pageIndex];
    }
    const bool validEmitRecord =
        emitPhaseSubmitted &&
        build.requiredFaceCount > 0u &&
        validPageIndex &&
        build.recordIndex != 0xffffffffu &&
        build.recordIndex < page.recordCapacity &&
        build.reservedFaceCapacity >= build.requiredFaceCount &&
        page.state == kChunkBufferPageStateOpenWritable &&
        emitConfig.blockFaceUvDescriptorIndex != 0xffffffffu &&
        build.centerVoxelSrvDescriptorIndex != 0xffffffffu &&
        build.haloSrvDescriptorIndex != 0xffffffffu &&
        page.faceDescriptorUavDescriptorIndex != 0xffffffffu &&
        page.drawRecordUavDescriptorIndex != 0xffffffffu &&
        page.drawRecordMetadataUavDescriptorIndex != 0xffffffffu;

    if (!validEmitRecord && !zeroFaceBuild)
    {
        return;
    }

    if (zeroFaceBuild)
    {
        if (planeIndex == 0u && groupThreadId.x == 0u)
        {
            GpuExactCompletionEntry completion;
            completion.buildIndex = buildIndex;
            completion.statusFlags =
                build.statusFlags | kExactCompletionStatusCompletedBit | kExactCompletionStatusZeroFacesBit;
            completion.requiredFaces = 0u;
            completion.reservedFaceCapacity = build.reservedFaceCapacity;
            completion.chunkWorldMinX = build.chunkWorldMinX;
            completion.chunkWorldMinY = build.chunkWorldMinY;
            completion.chunkWorldMinZ = build.chunkWorldMinZ;
            completion.pageIndex = build.pageIndex;
            completion.recordIndex = build.recordIndex;
            completion.faceBase = build.faceBase;
            completion.buildVersion = build.buildVersion;
            completion.generationEpoch = build.generationEpoch;
            completion.inputVersionLo = build.inputVersionLo;
            completion.inputVersionHi = build.inputVersionHi;
            completion.reserved0 = 0u;
            completion.reserved1 = 0u;
            gCompletionEntries[buildIndex] = completion;
        }
        return;
    }

    StructuredBuffer<uint> centerVoxels =
        ResourceDescriptorHeap[NonUniformResourceIndex(build.centerVoxelSrvDescriptorIndex)];
    StructuredBuffer<uint> haloVoxels =
        ResourceDescriptorHeap[NonUniformResourceIndex(build.haloSrvDescriptorIndex)];
    StructuredBuffer<GpuBlockFaceUv> blockFaceUvs =
        ResourceDescriptorHeap[NonUniformResourceIndex(emitConfig.blockFaceUvDescriptorIndex)];
    RWStructuredBuffer<GpuExactFaceDescriptor> faceDescriptors =
        ResourceDescriptorHeap[NonUniformResourceIndex(page.faceDescriptorUavDescriptorIndex)];
    RWStructuredBuffer<GpuExactDrawRecord> drawRecords =
        ResourceDescriptorHeap[NonUniformResourceIndex(page.drawRecordUavDescriptorIndex)];
    RWStructuredBuffer<GpuExactDrawRecordMetadata> drawRecordMetadata =
        ResourceDescriptorHeap[NonUniformResourceIndex(page.drawRecordMetadataUavDescriptorIndex)];

    const int3 chunkWorldMin = int3(build.chunkWorldMinX, build.chunkWorldMinY, build.chunkWorldMinZ);
    const uint totalFaces = build.requiredFaceCount;
    if (planeIndex == 0u && groupThreadId.x == 0u)
    {
        uint statusFlags = build.statusFlags | kExactCompletionStatusCompletedBit;
        if (totalFaces == 0u)
        {
            statusFlags |= kExactCompletionStatusZeroFacesBit;
        }

        GpuExactDrawRecord record;
        record.boundsMin = float4(float(build.chunkWorldMinX), float(build.chunkWorldMinY), float(build.chunkWorldMinZ), 1.0f);
        record.boundsMax = float4(float(build.chunkWorldMinX + int(kExactChunkSize)),
                                  float(build.chunkWorldMinY + int(kExactChunkSize)),
                                  float(build.chunkWorldMinZ + int(kExactChunkSize)),
                                  1.0f);
        record.faceCount = (totalFaces <= build.reservedFaceCapacity) ? totalFaces : 0u;
        record.faceOffset = build.faceBase;
        record.reserved0 = 0u;
        record.reserved = kExactDrawRecordActiveBit |
                          min(totalFaces, kExactDrawRecordFaceCountMask) |
                          ((totalFaces > build.reservedFaceCapacity) ? kExactDrawRecordOverflowFlag : 0u);
        drawRecords[build.recordIndex] = record;

        if (totalFaces > build.reservedFaceCapacity)
        {
            statusFlags |= kExactCompletionStatusOverflowBit;
            uint overflowIndex = 0u;
            InterlockedAdd(gOverflowCount[0], 1u, overflowIndex);
            GpuExactOverflowEntry entry;
            entry.buildIndex = buildOrdinal;
            entry.requiredFaces = totalFaces;
            entry.reserved0 = 0u;
            entry.reserved1 = 0u;
            gOverflowEntries[overflowIndex] = entry;
        }

        GpuExactDrawRecordMetadata metadata;
        metadata.chunkWorldMinX = build.chunkWorldMinX;
        metadata.chunkWorldMinY = build.chunkWorldMinY;
        metadata.chunkWorldMinZ = build.chunkWorldMinZ;
        metadata.pageIndex = build.pageIndex;
        metadata.recordIndex = build.recordIndex;
        metadata.buildIndex = buildIndex;
        metadata.faceBase = build.faceBase;
        metadata.faceCount = totalFaces;
        metadata.statusFlags = statusFlags;
        metadata.buildVersion = build.buildVersion;
        metadata.generationEpoch = build.generationEpoch;
        metadata.inputVersionLo = build.inputVersionLo;
        metadata.inputVersionHi = build.inputVersionHi;
        metadata.reserved0 = 0u;
        metadata.reserved1 = 0u;
        metadata.reserved2 = 0u;
        drawRecordMetadata[build.recordIndex] = metadata;

        GpuExactCompletionEntry completion;
        completion.buildIndex = buildIndex;
        completion.statusFlags = statusFlags;
        completion.requiredFaces = totalFaces;
        completion.reservedFaceCapacity = build.reservedFaceCapacity;
        completion.chunkWorldMinX = build.chunkWorldMinX;
        completion.chunkWorldMinY = build.chunkWorldMinY;
        completion.chunkWorldMinZ = build.chunkWorldMinZ;
        completion.pageIndex = build.pageIndex;
        completion.recordIndex = build.recordIndex;
        completion.faceBase = build.faceBase;
        completion.buildVersion = build.buildVersion;
        completion.generationEpoch = build.generationEpoch;
        completion.inputVersionLo = build.inputVersionLo;
        completion.inputVersionHi = build.inputVersionHi;
        completion.reserved0 = 0u;
        completion.reserved1 = 0u;
        gCompletionEntries[buildIndex] = completion;
    }

    if (totalFaces == 0u || totalFaces > build.reservedFaceCapacity)
    {
        return;
    }

    const uint faceCount = gFaceCountScratch[buildIndex * kExactFaceCountScratchStride + planeIndex];
    const uint faceBase = gFacePrefixScratch[buildIndex * kExactFacePrefixScratchStride + planeIndex];
    const uint descriptorBase =
        buildIndex * kExactChunkFaceDescriptorCount + planeIndex * kExactChunkMaxDescriptorsPerPlane;
    const uint columnBase = buildIndex * kExactChunkColumnCount;

    for (uint localIndex = groupThreadId.x; localIndex < faceCount; localIndex += 64u)
    {
        const GpuExactFaceDescriptor descriptor = gFaceDescriptorScratch[descriptorBase + localIndex];
        const uint localX = faceLocalX(descriptor.packedLocal);
        const uint localY = faceLocalY(descriptor.packedLocal);
        const uint localZ = faceLocalZ(descriptor.packedLocal);
        const uint faceId = faceLocalFaceId(descriptor.packedLocal);
        const uint packedVoxel = sampleVoxel(centerVoxels, int(localX), int(localY), int(localZ));
        const uint blockId = voxelBlock(packedVoxel);
        const GpuExactColumnDescriptor column = gColumnScratch[columnBase + columnIndex(localX, localZ)];
        const uint materialFlags = materialFlagsForFace(blockId, faceId, column);
        const uint alphaCutoutLightingFlag = isAlphaCutoutBlock(blockId) ? kLightingFlagAlphaCutoutBit : 0u;
        const uint faceIndex = faceBase + localIndex;
        const uint blockFaceUvIndex = blockId * 6u + faceId;
        const int cornerUSigns[4] = {-1, 1, 1, -1};
        const int cornerVSigns[4] = {-1, -1, 1, 1};
        uint cornerLighting[4];
        [unroll]
        for (uint i = 0u; i < 4u; ++i)
        {
            cornerLighting[i] = buildCornerLighting(centerVoxels,
                                                    haloVoxels,
                                                    int3(localX, localY, localZ),
                                                    faceId,
                                                    cornerUSigns[i],
                                                    cornerVSigns[i]);
        }
        uint vertexLighting[4];
        [unroll]
        for (uint i = 0u; i < 4u; ++i)
        {
            vertexLighting[i] = cornerLighting[i];
        }
        const int diagonal02 =
            lightingMetricFromPackedVertex(vertexLighting[0]) +
            lightingMetricFromPackedVertex(vertexLighting[2]);
        const int diagonal13 =
            lightingMetricFromPackedVertex(vertexLighting[1]) +
            lightingMetricFromPackedVertex(vertexLighting[3]);
        const bool flipDiagonal = diagonal13 > diagonal02;
        const uint flipBit = flipDiagonal ? (1u << 31u) : 0u;

        GpuExactFaceDescriptor finalDescriptor;
        finalDescriptor.packedLocal = descriptor.packedLocal | flipBit;
        finalDescriptor.blockFaceUvIndex = blockFaceUvIndex;
        finalDescriptor.packedLighting0 =
            ((vertexLighting[0] & ~((0x3Fu) << 10u)) | ((materialFlags & 0x3Fu) << 10u)) |
            alphaCutoutLightingFlag;
        finalDescriptor.packedLighting1 =
            ((vertexLighting[1] & ~((0x3Fu) << 10u)) | ((materialFlags & 0x3Fu) << 10u)) |
            alphaCutoutLightingFlag;
        finalDescriptor.packedLighting2 =
            ((vertexLighting[2] & ~((0x3Fu) << 10u)) | ((materialFlags & 0x3Fu) << 10u)) |
            alphaCutoutLightingFlag;
        finalDescriptor.packedLighting3 =
            ((vertexLighting[3] & ~((0x3Fu) << 10u)) | ((materialFlags & 0x3Fu) << 10u)) |
            alphaCutoutLightingFlag;
        finalDescriptor.reserved0 = 0u;
        finalDescriptor.reserved1 = 0u;
        faceDescriptors[build.faceBase + faceIndex] = finalDescriptor;
    }
}
