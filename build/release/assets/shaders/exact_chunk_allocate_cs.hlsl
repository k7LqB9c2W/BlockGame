#include "exact_chunk_common.hlsli"

cbuffer ExactChunkAllocateParams : register(b0)
{
    uint gBatchBuildCount;
    uint gReserved0;
    uint gReserved1;
    uint gReserved2;
};

StructuredBuffer<uint> gFaceTotalScratch : register(t0);
StructuredBuffer<uint> gBatchBuildIndices : register(t1);
RWStructuredBuffer<GpuExactAllocatorState> gAllocatorStateBuffer : register(u0);
RWStructuredBuffer<GpuExactAllocatorPageMetadata> gPageMetadata : register(u1);
RWStructuredBuffer<GpuExactAllocatorFreePageEntry> gFreePageList : register(u2);
RWStructuredBuffer<GpuExactChunkAllocationRecord> gBuildRecords : register(u3);
RWStructuredBuffer<GpuExactCompletionEntry> gCompletionEntries : register(u4);

static const uint kExactFaceTotalScratchStride = 64u;

GpuExactCompletionEntry makeBaseCompletion(uint buildIndex, GpuExactChunkAllocationRecord build, uint requiredFaces)
{
    GpuExactCompletionEntry completion = (GpuExactCompletionEntry)0;
    completion.buildIndex = buildIndex;
    completion.requiredFaces = requiredFaces;
    completion.reservedFaceCapacity = 0u;
    completion.chunkWorldMinX = build.chunkWorldMinX;
    completion.chunkWorldMinY = build.chunkWorldMinY;
    completion.chunkWorldMinZ = build.chunkWorldMinZ;
    completion.pageIndex = kInvalidExactPageIndex;
    completion.recordIndex = kInvalidExactRecordIndex;
    completion.vertexBase = 0u;
    completion.indexBase = 0u;
    completion.buildVersion = build.buildVersion;
    completion.generationEpoch = build.generationEpoch;
    completion.inputVersionLo = build.inputVersionLo;
    completion.inputVersionHi = build.inputVersionHi;
    completion.reserved0 = 0u;
    return completion;
}

bool tryLockPage(uint pageIndex, out GpuExactAllocatorPageMetadata page)
{
    page = (GpuExactAllocatorPageMetadata)0;
    uint originalLockWord = 0u;
    InterlockedCompareExchange(gPageMetadata[pageIndex].allocationLockWord, 0u, 1u, originalLockWord);
    if (originalLockWord != 0u)
    {
        return false;
    }

    page = gPageMetadata[pageIndex];
    page.pageIndex = pageIndex;
    page.allocationLockWord = 1u;
    return true;
}

void unlockPage(uint pageIndex)
{
    uint originalLockWord = 0u;
    InterlockedExchange(gPageMetadata[pageIndex].allocationLockWord, 0u, originalLockWord);
}

bool pageFits(GpuExactAllocatorPageMetadata page, uint requiredVertexCount, uint requiredIndexCount)
{
    return page.recordCursor + 1u <= page.recordCapacity &&
           page.vertexCursor + requiredVertexCount <= page.vertexCapacity &&
           page.indexCursor + requiredIndexCount <= page.indexCapacity;
}

bool reserveAllocationInPage(uint pageIndex,
                             inout GpuExactAllocatorPageMetadata page,
                             uint requiredFaces,
                             uint requiredVertexCount,
                             uint requiredIndexCount,
                             inout GpuExactChunkAllocationRecord build)
{
    if (page.usage != kChunkBufferPageUsageExactGpu ||
        page.state == kChunkBufferPageStatePendingUploaded ||
        page.state == kChunkBufferPageStateRetiring)
    {
        return false;
    }

    if (!pageFits(page, requiredVertexCount, requiredIndexCount))
    {
        return false;
    }

    if (page.state == kChunkBufferPageStateAvailable || page.state == kChunkBufferPageStateResident)
    {
        page.state = kChunkBufferPageStatePendingOpen;
    }

    build.requiredFaceCount = requiredFaces;
    build.pageIndex = pageIndex;
    build.recordIndex = page.recordCursor;
    build.vertexBase = page.vertexCursor;
    build.indexBase = page.indexCursor;
    build.reservedFaceCapacity = requiredFaces;
    build.phase = kExactChunkAllocationPhaseEmitSubmitted;
    build.statusFlags = 0u;

    page.vertexCursor += requiredVertexCount;
    page.indexCursor += requiredIndexCount;
    page.recordCursor += 1u;
    page.recordActiveCount = max(page.recordActiveCount, page.recordCursor);
    page.pendingChunks += 1u;
    page.pendingBatchIdLo = 0u;
    page.pendingBatchIdHi = 0u;
    page.uploadFenceValueLo = 0u;
    page.uploadFenceValueHi = 0u;
    page.retireFenceValueLo = 0u;
    page.retireFenceValueHi = 0u;
    return true;
}

bool tryReserveFromState(uint desiredState,
                         uint requiredFaces,
                         uint requiredVertexCount,
                         uint requiredIndexCount,
                         inout GpuExactChunkAllocationRecord build)
{
    const uint pageCount = gAllocatorStateBuffer[0].pageCount;
    for (uint pageIndex = 0u; pageIndex < pageCount; ++pageIndex)
    {
        GpuExactAllocatorPageMetadata page;
        if (!tryLockPage(pageIndex, page))
        {
            continue;
        }

        const bool matchesState = page.state == desiredState;
        const bool reserved =
            matchesState && reserveAllocationInPage(pageIndex,
                                                    page,
                                                    requiredFaces,
                                                    requiredVertexCount,
                                                    requiredIndexCount,
                                                    build);
        gPageMetadata[pageIndex] = page;
        unlockPage(pageIndex);
        if (reserved)
        {
            return true;
        }
    }

    return false;
}

bool popFreePage(out uint pageIndex)
{
    pageIndex = kInvalidExactPageIndex;
    for (;;)
    {
        const uint freePageCount = gAllocatorStateBuffer[0].freePageCount;
        if (freePageCount == 0u)
        {
            return false;
        }

        uint originalFreePageCount = 0u;
        InterlockedCompareExchange(gAllocatorStateBuffer[0].freePageCount,
                                   freePageCount,
                                   freePageCount - 1u,
                                   originalFreePageCount);
        if (originalFreePageCount == freePageCount)
        {
            pageIndex = gFreePageList[freePageCount - 1u].pageIndex;
            return pageIndex != kInvalidExactPageIndex;
        }
    }
}

bool tryReserveFromFreePages(uint requiredFaces,
                             uint requiredVertexCount,
                             uint requiredIndexCount,
                             inout GpuExactChunkAllocationRecord build)
{
    uint pageIndex = kInvalidExactPageIndex;
    if (!popFreePage(pageIndex))
    {
        return false;
    }

    GpuExactAllocatorPageMetadata page;
    if (!tryLockPage(pageIndex, page))
    {
        return false;
    }

    const bool reserved =
        page.state == kChunkBufferPageStateAvailable &&
        reserveAllocationInPage(pageIndex,
                                page,
                                requiredFaces,
                                requiredVertexCount,
                                requiredIndexCount,
                                build);
    gPageMetadata[pageIndex] = page;
    unlockPage(pageIndex);
    return reserved;
}

[numthreads(64, 1, 1)]
void ExactChunkAllocateMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint buildOrdinal = dispatchThreadId.x;
    if (buildOrdinal >= gBatchBuildCount)
    {
        return;
    }

    const GpuExactAllocatorState allocatorState = gAllocatorStateBuffer[0];
    const uint buildIndex = gBatchBuildIndices[buildOrdinal];
    if (buildIndex >= allocatorState.buildRecordCount)
    {
        return;
    }

    GpuExactChunkAllocationRecord build = gBuildRecords[buildIndex];
    if (build.phase != kExactChunkAllocationPhasePrepassSubmitted)
    {
        return;
    }

    const uint requiredFaces = gFaceTotalScratch[buildIndex * kExactFaceTotalScratchStride];
    build.requiredFaceCount = requiredFaces;
    build.pageIndex = kInvalidExactPageIndex;
    build.recordIndex = kInvalidExactRecordIndex;
    build.vertexBase = 0u;
    build.indexBase = 0u;
    build.reservedFaceCapacity = 0u;
    build.statusFlags = 0u;

    GpuExactCompletionEntry completion = makeBaseCompletion(buildIndex, build, requiredFaces);
    if (requiredFaces == 0u)
    {
        build.statusFlags = kExactCompletionStatusCompletedBit | kExactCompletionStatusZeroFacesBit;
        completion.statusFlags = build.statusFlags;
        gBuildRecords[buildIndex] = build;
        gCompletionEntries[buildIndex] = completion;
        return;
    }

    const uint requiredVertexCount = requiredFaces * 4u;
    const uint requiredIndexCount = requiredFaces * 6u;
    bool reserved = tryReserveFromState(kChunkBufferPageStatePendingOpen,
                                        requiredFaces,
                                        requiredVertexCount,
                                        requiredIndexCount,
                                        build);
    if (!reserved)
    {
        reserved = tryReserveFromState(kChunkBufferPageStateResident,
                                       requiredFaces,
                                       requiredVertexCount,
                                       requiredIndexCount,
                                       build);
    }
    if (!reserved)
    {
        reserved = tryReserveFromFreePages(requiredFaces,
                                           requiredVertexCount,
                                           requiredIndexCount,
                                           build);
    }

    if (!reserved)
    {
        build.statusFlags = kExactCompletionStatusAllocatorExhaustedBit;
        completion.statusFlags = kExactCompletionStatusAllocatorExhaustedBit;
        gBuildRecords[buildIndex] = build;
        gCompletionEntries[buildIndex] = completion;
        return;
    }

    completion.pageIndex = build.pageIndex;
    completion.recordIndex = build.recordIndex;
    completion.vertexBase = build.vertexBase;
    completion.indexBase = build.indexBase;
    completion.reservedFaceCapacity = build.reservedFaceCapacity;
    gBuildRecords[buildIndex] = build;
    gCompletionEntries[buildIndex] = completion;
}
