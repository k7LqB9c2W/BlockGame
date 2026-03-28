// chunk_manager_gpu_contexts.inl
// Defines the internal upload and far-LOD GPU context helpers used by ChunkManager.

class UploadContext
{
public:
    struct FlushTimings
    {
        double waitMs{0.0};
        double submitMs{0.0};
    };

    ~UploadContext()
    {
        shutdown();
    }

    void initialize(ID3D12Device* device)
    {
        shutdown();
        if (device == nullptr)
        {
            return;
        }

        device_ = device;

        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        throwIfFailedDx(device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue_)),
                        "failed to create upload command queue");
        throwIfFailedDx(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&allocator_)),
                        "failed to create upload command allocator");
        throwIfFailedDx(device_->CreateCommandList(0,
                                                   D3D12_COMMAND_LIST_TYPE_COPY,
                                                   allocator_.Get(),
                                                   nullptr,
                                                   IID_PPV_ARGS(&commandList_)),
                        "failed to create upload command list");
        throwIfFailedDx(commandList_->Close(), "failed to close initial upload command list");
        throwIfFailedDx(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
                        "failed to create upload fence");
        fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (fenceEvent_ == nullptr)
        {
            throw std::runtime_error("failed to create upload fence event");
        }
    }

    void shutdown()
    {
        graphicsFence_ = nullptr;
        graphicsFenceValue_ = 0;
        waitForIdle();
        if (fenceEvent_ != nullptr)
        {
            CloseHandle(fenceEvent_);
            fenceEvent_ = nullptr;
        }
        commandList_.Reset();
        allocator_.Reset();
        queue_.Reset();
        fence_.Reset();
        device_.Reset();
        graphicsFence_ = nullptr;
        graphicsFenceValue_ = 0;
        lastSubmittedFenceValue_ = 0;
        open_ = false;
        hasCommands_ = false;
    }

    void setGraphicsFenceDependency(ID3D12Fence* graphicsFence, UINT64 graphicsFenceValue) noexcept
    {
        graphicsFence_ = graphicsFence;
        graphicsFenceValue_ = graphicsFenceValue;
    }

    [[nodiscard]] bool begin()
    {
        if (device_ == nullptr)
        {
            return false;
        }
        if (open_)
        {
            return true;
        }
        if (fence_ != nullptr && fenceValue_ > 0 && fence_->GetCompletedValue() < fenceValue_)
        {
            return false;
        }

        throwIfFailedDx(allocator_->Reset(), "failed to reset upload command allocator");
        throwIfFailedDx(commandList_->Reset(allocator_.Get(), nullptr), "failed to reset upload command list");
        open_ = true;
        hasCommands_ = false;
        return true;
    }

    void transition(ID3D12Resource* resource,
                    D3D12_RESOURCE_STATES before,
                    D3D12_RESOURCE_STATES after)
    {
        if (!open_ || resource == nullptr || before == after)
        {
            return;
        }

        const D3D12_RESOURCE_BARRIER barrier = transitionBarrier(resource, before, after);
        commandList_->ResourceBarrier(1, &barrier);
        hasCommands_ = true;
    }

    void copyBuffer(ID3D12Resource* destination,
                    std::uint64_t destinationOffset,
                    ID3D12Resource* source,
                    std::uint64_t sourceOffset,
                    std::uint64_t sizeInBytes)
    {
        if (!open_ || destination == nullptr || source == nullptr || sizeInBytes == 0)
        {
            return;
        }

        commandList_->CopyBufferRegion(destination, destinationOffset, source, sourceOffset, sizeInBytes);
        hasCommands_ = true;
    }

    void waitForFence(ID3D12Fence* fence, UINT64 fenceValue)
    {
        if (!open_ || queue_ == nullptr || fence == nullptr || fenceValue == 0)
        {
            return;
        }

        throwIfFailedDx(queue_->Wait(fence, fenceValue), "failed to wait for external fence before upload");
        hasCommands_ = true;
    }

    void flush(FlushTimings* timings = nullptr)
    {
        if (!open_)
        {
            return;
        }

        if (timings != nullptr)
        {
            *timings = FlushTimings{};
        }

        throwIfFailedDx(commandList_->Close(), "failed to close upload command list");
        if (hasCommands_)
        {
            // Chunk buffers are shared with the render path. Do not transition/copy them
            // until the last submitted graphics work that referenced them has completed.
            if (graphicsFence_ != nullptr && graphicsFenceValue_ > 0)
            {
                const SteadyClock::time_point waitStart = SteadyClock::now();
                throwIfFailedDx(queue_->Wait(graphicsFence_, graphicsFenceValue_),
                                "failed to wait for graphics fence before upload");
                if (timings != nullptr)
                {
                    timings->waitMs =
                        std::chrono::duration<double, std::milli>(SteadyClock::now() - waitStart).count();
                }
            }

            const SteadyClock::time_point submitStart = SteadyClock::now();
            ID3D12CommandList* lists[] = {commandList_.Get()};
            queue_->ExecuteCommandLists(static_cast<UINT>(std::size(lists)), lists);

            ++fenceValue_;
            throwIfFailedDx(queue_->Signal(fence_.Get(), fenceValue_), "failed to signal upload fence");
            lastSubmittedFenceValue_ = fenceValue_;
            if (timings != nullptr)
            {
                timings->submitMs =
                    std::chrono::duration<double, std::milli>(SteadyClock::now() - submitStart).count();
            }
        }

        open_ = false;
        hasCommands_ = false;
    }

    [[nodiscard]] bool ready() const noexcept
    {
        return device_ != nullptr && queue_ != nullptr && allocator_ != nullptr && commandList_ != nullptr;
    }

    [[nodiscard]] ID3D12Fence* fence() const noexcept
    {
        return fence_.Get();
    }

    [[nodiscard]] UINT64 lastSubmittedFenceValue() const noexcept
    {
        return lastSubmittedFenceValue_;
    }

    [[nodiscard]] UINT64 completedFenceValue() const noexcept
    {
        return (fence_ != nullptr) ? fence_->GetCompletedValue() : 0;
    }

    void waitForIdle()
    {
        if (fence_ == nullptr || lastSubmittedFenceValue_ == 0)
        {
            return;
        }
        if (fence_->GetCompletedValue() >= lastSubmittedFenceValue_)
        {
            return;
        }
        throwIfFailedDx(fence_->SetEventOnCompletion(lastSubmittedFenceValue_, fenceEvent_),
                        "failed to wait for upload fence");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }

private:
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_{nullptr};
    UINT64 fenceValue_{0};
    UINT64 lastSubmittedFenceValue_{0};
    ID3D12Fence* graphicsFence_{nullptr};
    UINT64 graphicsFenceValue_{0};
    bool open_{false};
    bool hasCommands_{false};
};

class FarLodGpuContext
{
public:
    struct ExactPassTimings
    {
        double synthMs{0.0};
        double stampMs{0.0};
        double lightMs{0.0};
        double faceCountMs{0.0};
        double facePrefixMs{0.0};
        double faceEmitMs{0.0};
        double totalMs{0.0};
    };

    struct ScratchAllocation
    {
        ID3D12Resource* resource{nullptr};
        std::byte* cpuPtr{nullptr};
        D3D12_GPU_VIRTUAL_ADDRESS gpuAddress{0};
        std::uint64_t offset{0};
        std::uint64_t size{0};
    };

    struct ExactOverflowReadback
    {
        ScratchAllocation allocation{};
        std::uint64_t entryOffset{0};
    };

    struct ExactFaceTotalsReadback
    {
        ScratchAllocation allocation{};
        std::uint64_t strideBytes{0};
        std::uint32_t buildCount{0};
    };

    struct ExactOverflowEntry
    {
        std::uint32_t buildIndex{0};
        std::uint32_t requiredFaces{0};
        std::uint32_t reserved0{0};
        std::uint32_t reserved1{0};
    };

    struct FlushResult
    {
        UINT64 fenceValue{0};
    };

    ~FarLodGpuContext()
    {
        shutdown();
    }

    void initialize(ID3D12Device* device)
    {
        shutdown();
        if (device == nullptr)
        {
            return;
        }

        device_ = device;

        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        throwIfFailedDx(device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue_)),
                        "failed to create far lod compute queue");
        throwIfFailedDx(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator_)),
                        "failed to create far lod compute allocator");
        throwIfFailedDx(device_->CreateCommandList(0,
                                                   D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                   allocator_.Get(),
                                                   nullptr,
                                                   IID_PPV_ARGS(&commandList_)),
                        "failed to create far lod compute command list");
        throwIfFailedDx(commandList_->Close(), "failed to close initial far lod compute command list");
        throwIfFailedDx(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
                        "failed to create far lod compute fence");
        setDebugObjectName(queue_.Get(), L"FarLodComputeQueue");
        setDebugObjectName(allocator_.Get(), L"FarLodComputeAllocator");
        setDebugObjectName(commandList_.Get(), L"FarLodComputeCommandList");
        setDebugObjectName(fence_.Get(), L"FarLodComputeFence");
        fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (fenceEvent_ == nullptr)
        {
            throw std::runtime_error("failed to create far lod compute fence event");
        }

        uploadScratch_ = createUploadBuffer(device_.Get(), kUploadScratchSizeBytes, uploadScratchMapped_);
        setDebugObjectName(uploadScratch_.Get(), L"FarLodComputeUploadScratch");
        compileShaders();
        createDescriptorHeap();
        createPipelines();
        createExactResources();
        createExactTimestampResources();
        chunkManagerDebugLog("Far LOD GPU context initialized");
    }

    void shutdown()
    {
        waitForIdle();
        if (readbackScratch_ != nullptr)
        {
            readbackScratch_->Unmap(0, nullptr);
        }
        if (uploadScratch_ != nullptr)
        {
            uploadScratch_->Unmap(0, nullptr);
        }
        if (fenceEvent_ != nullptr)
        {
            CloseHandle(fenceEvent_);
            fenceEvent_ = nullptr;
        }

        atlasSeedCachePipelineState_.Reset();
        atlasSampleCachePipelineState_.Reset();
        atlasUpdatePipelineState_.Reset();
        synthColumnPipelineState_.Reset();
        stampPipelineState_.Reset();
        faceCountPipelineState_.Reset();
        facePrefixGroupPipelineState_.Reset();
        facePrefixScanPipelineState_.Reset();
        facePrefixAddPipelineState_.Reset();
        faceEmitPipelineState_.Reset();
        exactSynthPipelineState_.Reset();
        exactStampPipelineState_.Reset();
        exactHaloCachePipelineState_.Reset();
        exactLightPipelineState_.Reset();
        exactFaceCountPipelineState_.Reset();
        exactFacePrefixPipelineState_.Reset();
        exactFaceEmitPipelineState_.Reset();
        atlasSeedRootSignature_.Reset();
        atlasSampleRootSignature_.Reset();
        atlasFinalizeRootSignature_.Reset();
        synthColumnRootSignature_.Reset();
        stampRootSignature_.Reset();
        faceCountRootSignature_.Reset();
        facePrefixGroupRootSignature_.Reset();
        facePrefixScanRootSignature_.Reset();
        facePrefixAddRootSignature_.Reset();
        faceEmitRootSignature_.Reset();
        exactSynthRootSignature_.Reset();
        exactStampRootSignature_.Reset();
        exactHaloCacheRootSignature_.Reset();
        exactLightRootSignature_.Reset();
        exactFaceCountRootSignature_.Reset();
        exactFacePrefixRootSignature_.Reset();
        exactFaceEmitRootSignature_.Reset();
        descriptorHeap_.Reset();
        atlasSeedCacheShader_.Reset();
        atlasSampleCacheShader_.Reset();
        atlasUpdateShader_.Reset();
        synthColumnShader_.Reset();
        stampShader_.Reset();
        faceCountShader_.Reset();
        facePrefixGroupShader_.Reset();
        facePrefixScanShader_.Reset();
        facePrefixAddShader_.Reset();
        faceEmitShader_.Reset();
        exactSynthShader_.Reset();
        exactStampShader_.Reset();
        exactHaloCacheShader_.Reset();
        exactLightShader_.Reset();
        exactFaceCountShader_.Reset();
        exactFacePrefixShader_.Reset();
        exactFaceEmitShader_.Reset();
        exactFaceCountScratchBuffer_.Reset();
        exactFaceDescriptorScratchBuffer_.Reset();
        exactFacePrefixScratchBuffer_.Reset();
        exactFaceTotalScratchBuffer_.Reset();
        exactOverflowCountScratchBuffer_.Reset();
        exactOverflowEntryScratchBuffer_.Reset();
        exactTimestampReadbackBuffer_.Reset();
        exactTimestampQueryHeap_.Reset();
        readbackScratch_.Reset();
        uploadScratch_.Reset();
        commandList_.Reset();
        allocator_.Reset();
        queue_.Reset();
        fence_.Reset();
        device_.Reset();
        uploadScratchMapped_ = nullptr;
        readbackScratchMapped_ = nullptr;
        open_ = false;
        hasCommands_ = false;
        readbackEnabled_ = false;
        uploadCursor_ = 0;
        readbackCursor_ = 0;
        lastSubmittedFenceValue_ = 0;
        fenceValue_ = 0;
        exactTimestampFrequency_ = 0;
        exactTimestampCursor_ = 0;
        exactTimestampSubmittedCount_ = 0;
        exactTimestampPendingFenceValue_ = 0;
        exactTimingPending_ = false;
        exactTimingCaptureActive_ = false;
        exactLastCompletedTimings_ = {};
    }

    void setReadbackEnabled(bool enabled)
    {
        if (device_ == nullptr)
        {
            readbackEnabled_ = enabled;
            return;
        }
        if (enabled == readbackEnabled_)
        {
            return;
        }
        waitForIdle();
        if (enabled)
        {
            readbackScratch_ = createReadbackBuffer(device_.Get(), kReadbackScratchSizeBytes, readbackScratchMapped_);
            setDebugObjectName(readbackScratch_.Get(), L"FarLodComputeReadbackScratch");
        }
        else
        {
            if (readbackScratch_ != nullptr)
            {
                readbackScratch_->Unmap(0, nullptr);
            }
            readbackScratch_.Reset();
            readbackScratchMapped_ = nullptr;
        }
        readbackEnabled_ = enabled;
        readbackCursor_ = 0;
    }

    [[nodiscard]] bool begin()
    {
        if (device_ == nullptr)
        {
            return false;
        }
        if (open_)
        {
            return true;
        }
        if (fence_ != nullptr && fenceValue_ > 0 && fence_->GetCompletedValue() < fenceValue_)
        {
            return false;
        }

        throwIfFailedDx(allocator_->Reset(), "failed to reset far lod compute allocator");
        throwIfFailedDx(commandList_->Reset(allocator_.Get(), nullptr), "failed to reset far lod compute command list");
        open_ = true;
        hasCommands_ = false;
        uploadCursor_ = 0;
        readbackCursor_ = 0;
        descriptorCursor_ = 0;
        exactTimestampCursor_ = 0;
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD compute begin nextFence=" << (fenceValue_ + 1)
                   << " completedFence=" << ((fence_ != nullptr) ? fence_->GetCompletedValue() : 0);
            if (fence_ != nullptr && fence_->GetCompletedValue() == std::numeric_limits<std::uint64_t>::max())
            {
                const std::string dredMessages = collectDeviceDredMessages(device_.Get());
                if (!dredMessages.empty())
                {
                    stream << dredMessages;
                }
            }
            chunkManagerDebugLog(stream.str());
        }
        return true;
    }

    [[nodiscard]] ScratchAllocation allocateUpload(std::uint64_t sizeInBytes,
                                                   std::uint64_t alignment = 16u)
    {
        ScratchAllocation allocation{};
        if (!open_ || uploadScratch_ == nullptr || sizeInBytes == 0)
        {
            return allocation;
        }

        const std::uint64_t alignedOffset = (uploadCursor_ + alignment - 1u) / alignment * alignment;
        if (alignedOffset + sizeInBytes > kUploadScratchSizeBytes)
        {
            return allocation;
        }

        allocation.resource = uploadScratch_.Get();
        allocation.cpuPtr = uploadScratchMapped_ + alignedOffset;
        allocation.gpuAddress = uploadScratch_->GetGPUVirtualAddress() + alignedOffset;
        allocation.offset = alignedOffset;
        allocation.size = sizeInBytes;
        uploadCursor_ = alignedOffset + sizeInBytes;
        return allocation;
    }

    [[nodiscard]] ScratchAllocation allocateReadback(std::uint64_t sizeInBytes,
                                                     std::uint64_t alignment = 16u)
    {
        ScratchAllocation allocation{};
        if (!open_ || readbackScratch_ == nullptr || sizeInBytes == 0)
        {
            return allocation;
        }

        const std::uint64_t alignedOffset = (readbackCursor_ + alignment - 1u) / alignment * alignment;
        if (alignedOffset + sizeInBytes > kReadbackScratchSizeBytes)
        {
            return allocation;
        }

        allocation.resource = readbackScratch_.Get();
        allocation.cpuPtr = readbackScratchMapped_ + alignedOffset;
        allocation.gpuAddress = readbackScratch_->GetGPUVirtualAddress() + alignedOffset;
        allocation.offset = alignedOffset;
        allocation.size = sizeInBytes;
        readbackCursor_ = alignedOffset + sizeInBytes;
        return allocation;
    }

    void beginExactTimingBatch()
    {
        exactTimestampCursor_ = 0;
        exactTimingCaptureActive_ = !exactTimingPending_;
    }

    void markExactTimingBegin()
    {
        if (!open_ || !exactTimingCaptureActive_ || exactTimestampQueryHeap_ == nullptr ||
            exactTimestampCursor_ >= kExactTimestampQueryCount)
        {
            return;
        }

        commandList_->EndQuery(exactTimestampQueryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, exactTimestampCursor_++);
        hasCommands_ = true;
    }

    void markExactTimingEnd()
    {
        if (!open_ || !exactTimingCaptureActive_ || exactTimestampQueryHeap_ == nullptr ||
            exactTimestampCursor_ >= kExactTimestampQueryCount)
        {
            return;
        }

        commandList_->EndQuery(exactTimestampQueryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, exactTimestampCursor_++);
        hasCommands_ = true;
    }

    void transition(ID3D12Resource* resource,
                    D3D12_RESOURCE_STATES before,
                    D3D12_RESOURCE_STATES after)
    {
        if (!open_ || resource == nullptr || before == after)
        {
            return;
        }

        const D3D12_RESOURCE_BARRIER barrier = transitionBarrier(resource, before, after);
        commandList_->ResourceBarrier(1, &barrier);
        hasCommands_ = true;
    }

    void uavBarrier(ID3D12Resource* resource)
    {
        if (!open_)
        {
            return;
        }

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = resource;
        commandList_->ResourceBarrier(1, &barrier);
        hasCommands_ = true;
    }

    void copyBuffer(ID3D12Resource* destination,
                    std::uint64_t destinationOffset,
                    ID3D12Resource* source,
                    std::uint64_t sourceOffset,
                    std::uint64_t sizeInBytes)
    {
        if (!open_ || destination == nullptr || source == nullptr || sizeInBytes == 0)
        {
            return;
        }
        commandList_->CopyBufferRegion(destination, destinationOffset, source, sourceOffset, sizeInBytes);
        hasCommands_ = true;
    }

    void setAtlasBindings(ID3D12RootSignature* rootSignature,
                          ID3D12PipelineState* pipelineState,
                          const std::array<std::uint32_t, 14>& constants,
                          UINT srvDescriptorIndex,
                          UINT uavDescriptorIndex)
    {
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(pipelineState);
        commandList_->SetComputeRootSignature(rootSignature);
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<UINT64>(srvDescriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, srvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        uavHandle.ptr += static_cast<UINT64>(uavDescriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(2, uavHandle);
    }

    void bindAtlasSeedDescriptors(UINT srvDescriptorIndex,
                                  UINT uavDescriptorIndex,
                                  ID3D12Resource* worldgenHeaderBuffer,
                                  ID3D12Resource* worldgenBiomeBuffer,
                                  std::uint32_t biomeCount,
                                  ID3D12Resource* biomeSelectionBuffer,
                                  std::uint32_t biomeSelectionCount,
                                  ID3D12Resource* oceanSelectionBuffer,
                                  std::uint32_t oceanSelectionCount,
                                  ID3D12Resource* subBiomeBuffer,
                                  std::uint32_t subBiomeCount,
                                  ID3D12Resource* seedHeaderUavBuffer,
                                  std::uint32_t seedHeaderElementCount,
                                  ID3D12Resource* seedDataUavBuffer,
                                  std::uint32_t seedDataElementCount)
    {
        writeStructuredSrvDescriptor(srvDescriptorIndex,
                                     worldgenHeaderBuffer,
                                     0,
                                     1,
                                     static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuWorldgenHeader)));
        writeStructuredSrvDescriptor(srvDescriptorIndex + 1u,
                                     worldgenBiomeBuffer,
                                     0,
                                     biomeCount,
                                     static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuBiome)));
        writeStructuredSrvDescriptor(srvDescriptorIndex + 2u,
                                     biomeSelectionBuffer,
                                     0,
                                     biomeSelectionCount,
                                     kGpuContextBiomeSelectionStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 3u,
                                     oceanSelectionBuffer,
                                     0,
                                     oceanSelectionCount,
                                     kGpuContextBiomeSelectionStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 4u,
                                     subBiomeBuffer,
                                     0,
                                     subBiomeCount,
                                     kGpuContextSubBiomeStrideBytes);
        writeStructuredUavDescriptor(uavDescriptorIndex,
                                     seedHeaderUavBuffer,
                                     0,
                                     seedHeaderElementCount,
                                     kGpuContextChunkSeedCacheHeaderStrideBytes);
        writeStructuredUavDescriptor(uavDescriptorIndex + 1u,
                                     seedDataUavBuffer,
                                     0,
                                     seedDataElementCount,
                                     kGpuContextChunkSeedStrideBytes);
    }

    void bindAtlasSampleDescriptors(UINT srvDescriptorIndex,
                                    UINT uavDescriptorIndex,
                                    ID3D12Resource* worldgenHeaderBuffer,
                                    ID3D12Resource* worldgenBiomeBuffer,
                                    std::uint32_t biomeCount,
                                    ID3D12Resource* biomeSelectionBuffer,
                                    std::uint32_t biomeSelectionCount,
                                    ID3D12Resource* oceanSelectionBuffer,
                                    std::uint32_t oceanSelectionCount,
                                    ID3D12Resource* subBiomeBuffer,
                                    std::uint32_t subBiomeCount,
                                    ID3D12Resource* permutationBuffer,
                                    std::uint32_t permutationCount,
                                    ID3D12Resource* seedHeaderSrvBuffer,
                                    std::uint32_t seedHeaderElementCount,
                                    ID3D12Resource* seedDataSrvBuffer,
                                    std::uint32_t seedDataElementCount,
                                    ID3D12Resource* sampleUavBuffer,
                                    std::uint32_t sampleUavElementCount)
    {
        const auto logDescriptorWrite = [&](const char* kind,
                                            UINT descriptorIndexValue,
                                            const char* label,
                                            ID3D12Resource* resource,
                                            std::uint32_t elementCount,
                                            std::uint32_t strideBytes)
        {
            if (!chunkManagerDebugLoggingEnabled())
            {
                return;
            }

            std::ostringstream stream;
            stream << "Far LOD sample descriptor write"
                   << " kind=" << kind
                   << " descriptor=" << descriptorIndexValue
                   << " label=" << label
                   << " resource=" << hexPtr(resource)
                   << " elements=" << elementCount
                   << " stride=" << strideBytes;
            chunkManagerDebugLog(stream.str());
        };

        logDescriptorWrite("srv",
                           srvDescriptorIndex,
                           "worldgen_header",
                           worldgenHeaderBuffer,
                           1u,
                           static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuWorldgenHeader)));
        writeStructuredSrvDescriptor(srvDescriptorIndex,
                                     worldgenHeaderBuffer,
                                     0,
                                     1,
                                     static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuWorldgenHeader)));
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 1u,
                           "biomes",
                           worldgenBiomeBuffer,
                           biomeCount,
                           static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuBiome)));
        writeStructuredSrvDescriptor(srvDescriptorIndex + 1u,
                                     worldgenBiomeBuffer,
                                     0,
                                     biomeCount,
                                     static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuBiome)));
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 2u,
                           "biome_selections",
                           biomeSelectionBuffer,
                           biomeSelectionCount,
                           kGpuContextBiomeSelectionStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 2u,
                                     biomeSelectionBuffer,
                                     0,
                                     biomeSelectionCount,
                                     kGpuContextBiomeSelectionStrideBytes);
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 3u,
                           "ocean_selections",
                           oceanSelectionBuffer,
                           oceanSelectionCount,
                           kGpuContextBiomeSelectionStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 3u,
                                     oceanSelectionBuffer,
                                     0,
                                     oceanSelectionCount,
                                     kGpuContextBiomeSelectionStrideBytes);
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 4u,
                           "sub_biomes",
                           subBiomeBuffer,
                           subBiomeCount,
                           kGpuContextSubBiomeStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 4u,
                                     subBiomeBuffer,
                                     0,
                                     subBiomeCount,
                                     kGpuContextSubBiomeStrideBytes);
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 5u,
                           "surface_permutation",
                           permutationBuffer,
                           permutationCount,
                           kGpuContextPermutationStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 5u,
                                     permutationBuffer,
                                     0,
                                     permutationCount,
                                     kGpuContextPermutationStrideBytes);
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 6u,
                           "seed_headers",
                           seedHeaderSrvBuffer,
                           seedHeaderElementCount,
                           kGpuContextChunkSeedCacheHeaderStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 6u,
                                     seedHeaderSrvBuffer,
                                     0,
                                     seedHeaderElementCount,
                                     kGpuContextChunkSeedCacheHeaderStrideBytes);
        logDescriptorWrite("srv",
                           srvDescriptorIndex + 7u,
                           "seed_data",
                           seedDataSrvBuffer,
                           seedDataElementCount,
                           kGpuContextChunkSeedStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 7u,
                                     seedDataSrvBuffer,
                                     0,
                                     seedDataElementCount,
                                     kGpuContextChunkSeedStrideBytes);
        logDescriptorWrite("uav",
                           uavDescriptorIndex,
                           "sample_cache_out",
                           sampleUavBuffer,
                           sampleUavElementCount,
                           kGpuContextAtlasSamplePointCacheStrideBytes);
        writeStructuredUavDescriptor(uavDescriptorIndex,
                                     sampleUavBuffer,
                                     0,
                                     sampleUavElementCount,
                                     kGpuContextAtlasSamplePointCacheStrideBytes);
    }

    void bindAtlasFinalizeDescriptors(UINT srvDescriptorIndex,
                                      UINT uavDescriptorIndex,
                                      ID3D12Resource* worldgenHeaderBuffer,
                                      ID3D12Resource* worldgenBiomeBuffer,
                                      std::uint32_t biomeCount,
                                      ID3D12Resource* permutationBuffer,
                                      std::uint32_t permutationCount,
                                      ID3D12Resource* sampleSrvBuffer,
                                      std::uint32_t sampleSrvElementCount,
                                      ID3D12Resource* atlasUavBuffer,
                                      std::uint32_t atlasUavElementCount)
    {
        writeStructuredSrvDescriptor(srvDescriptorIndex,
                                     worldgenHeaderBuffer,
                                     0,
                                     1,
                                     static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuWorldgenHeader)));
        writeStructuredSrvDescriptor(srvDescriptorIndex + 1u,
                                     worldgenBiomeBuffer,
                                     0,
                                     biomeCount,
                                     static_cast<std::uint32_t>(sizeof(terrain::FarLodGpuBiome)));
        writeStructuredSrvDescriptor(srvDescriptorIndex + 2u,
                                     permutationBuffer,
                                     0,
                                     permutationCount,
                                     kGpuContextPermutationStrideBytes);
        writeStructuredSrvDescriptor(srvDescriptorIndex + 3u,
                                     sampleSrvBuffer,
                                     0,
                                     sampleSrvElementCount,
                                     kGpuContextAtlasSamplePointCacheStrideBytes);
        writeStructuredUavDescriptor(uavDescriptorIndex,
                                     atlasUavBuffer,
                                     0,
                                     atlasUavElementCount,
                                     kGpuContextAtlasSampleStrideBytes);
    }

    void dispatchSeedCacheUpdate(const glm::ivec2& updateOriginChunk,
                                 const glm::ivec2& updateSizeChunks,
                                 const glm::ivec2& seedOriginChunk,
                                 const glm::ivec2& seedSizeChunks,
                                 ID3D12Resource* worldgenHeaderBuffer,
                                 ID3D12Resource* worldgenBiomeBuffer,
                                 std::uint32_t biomeCount,
                                 ID3D12Resource* biomeSelectionBuffer,
                                 std::uint32_t biomeSelectionCount,
                                 ID3D12Resource* oceanSelectionBuffer,
                                 std::uint32_t oceanSelectionCount,
                                 ID3D12Resource* subBiomeBuffer,
                                 std::uint32_t subBiomeCount,
                                 ID3D12Resource* seedHeaderBuffer,
                                 std::uint32_t seedHeaderElementCount,
                                 ID3D12Resource* seedDataBuffer,
                                 std::uint32_t seedDataElementCount)
    {
        if (!open_ || worldgenHeaderBuffer == nullptr || worldgenBiomeBuffer == nullptr ||
            biomeSelectionBuffer == nullptr || oceanSelectionBuffer == nullptr ||
            subBiomeBuffer == nullptr || seedHeaderBuffer == nullptr || seedDataBuffer == nullptr ||
            updateSizeChunks.x <= 0 || updateSizeChunks.y <= 0)
        {
            return;
        }
        if (seedHeaderElementCount == 0 || seedDataElementCount == 0)
        {
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD seed dispatch skipped due to empty seed buffers"
                       << " seedHeader=" << seedHeaderElementCount
                       << " seedData=" << seedDataElementCount;
                chunkManagerDebugLog(stream.str());
            }
            return;
        }
        if (seedSizeChunks.x <= 0 || seedSizeChunks.y <= 0)
        {
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD seed dispatch skipped due to invalid seed cache dimensions"
                       << " seedOrigin=[" << seedOriginChunk.x << "," << seedOriginChunk.y << "]"
                       << " seedSize=[" << seedSizeChunks.x << "," << seedSizeChunks.y << "]";
                chunkManagerDebugLog(stream.str());
            }
            return;
        }
        const std::uint64_t expectedSeedHeaderElementCount =
            static_cast<std::uint64_t>(seedSizeChunks.x) * static_cast<std::uint64_t>(seedSizeChunks.y);
        const std::uint64_t expectedSeedDataElementCount =
            expectedSeedHeaderElementCount * static_cast<std::uint64_t>(kGpuContextChunkSeedCountPerCacheEntry);
        if (static_cast<std::uint64_t>(seedHeaderElementCount) < expectedSeedHeaderElementCount ||
            static_cast<std::uint64_t>(seedDataElementCount) < expectedSeedDataElementCount)
        {
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD seed dispatch skipped due to undersized seed buffers"
                       << " expectedHeader=" << expectedSeedHeaderElementCount
                       << " expectedData=" << expectedSeedDataElementCount
                       << " actualHeader=" << seedHeaderElementCount
                       << " actualData=" << seedDataElementCount;
                chunkManagerDebugLog(stream.str());
            }
            return;
        }

        std::array<std::uint32_t, 14> constants{
            0u, 0u, 0u, 0u,
            static_cast<std::uint32_t>(updateOriginChunk.x),
            static_cast<std::uint32_t>(updateOriginChunk.y),
            static_cast<std::uint32_t>(updateSizeChunks.x),
            static_cast<std::uint32_t>(updateSizeChunks.y),
            0u,
            0u,
            static_cast<std::uint32_t>(seedOriginChunk.x),
            static_cast<std::uint32_t>(seedOriginChunk.y),
            static_cast<std::uint32_t>(seedSizeChunks.x),
            static_cast<std::uint32_t>(seedSizeChunks.y)};
        const UINT srvDescriptorIndex = allocateDescriptorRange(5);
        const UINT uavDescriptorIndex = allocateDescriptorRange(2);
        bindAtlasSeedDescriptors(srvDescriptorIndex,
                                 uavDescriptorIndex,
                                 worldgenHeaderBuffer,
                                 worldgenBiomeBuffer,
                                 biomeCount,
                                 biomeSelectionBuffer,
                                 biomeSelectionCount,
                                 oceanSelectionBuffer,
                                 oceanSelectionCount,
                                 subBiomeBuffer,
                                 subBiomeCount,
                                 seedHeaderBuffer,
                                 seedHeaderElementCount,
                                 seedDataBuffer,
                                 seedDataElementCount);
        setAtlasBindings(atlasSeedRootSignature_.Get(),
                         atlasSeedCachePipelineState_.Get(),
                         constants,
                         srvDescriptorIndex,
                         uavDescriptorIndex);
        commandList_->Dispatch(static_cast<UINT>((updateSizeChunks.x + 7) / 8),
                               static_cast<UINT>((updateSizeChunks.y + 7) / 8),
                               1);
        hasCommands_ = true;
    }

    void dispatchAtlasSampleCacheUpdate(const glm::ivec2& atlasOriginCell,
                                        const glm::ivec2& atlasSizeCells,
                                        const glm::ivec2& updateOriginCell,
                                        const glm::ivec2& updateSizeCells,
                                        int blockScale,
                                        const glm::ivec2& seedOriginChunk,
                                        const glm::ivec2& seedSizeChunks,
                                        ID3D12Resource* worldgenHeaderBuffer,
                                        ID3D12Resource* worldgenBiomeBuffer,
                                        std::uint32_t biomeCount,
                                        ID3D12Resource* biomeSelectionBuffer,
                                        std::uint32_t biomeSelectionCount,
                                        ID3D12Resource* oceanSelectionBuffer,
                                        std::uint32_t oceanSelectionCount,
                                        ID3D12Resource* subBiomeBuffer,
                                        std::uint32_t subBiomeCount,
                                        ID3D12Resource* permutationBuffer,
                                        std::uint32_t permutationCount,
                                        ID3D12Resource* seedHeaderBuffer,
                                        std::uint32_t seedHeaderElementCount,
                                        ID3D12Resource* seedDataBuffer,
                                        std::uint32_t seedDataElementCount,
                                        ID3D12Resource* sampleBuffer,
                                        std::uint32_t sampleElementCount)
    {
        static std::uint64_t sampleDispatchSequence = 0u;
        const std::uint64_t dispatchId = ++sampleDispatchSequence;
        const bool logEnabled = chunkManagerDebugLoggingEnabled();
        if (!open_ || worldgenHeaderBuffer == nullptr || worldgenBiomeBuffer == nullptr ||
            biomeSelectionBuffer == nullptr || oceanSelectionBuffer == nullptr || subBiomeBuffer == nullptr ||
            permutationBuffer == nullptr || seedHeaderBuffer == nullptr || seedDataBuffer == nullptr ||
            sampleBuffer == nullptr || updateSizeCells.x <= 0 || updateSizeCells.y <= 0)
        {
            return;
        }
        if (seedSizeChunks.x <= 0 || seedSizeChunks.y <= 0)
        {
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD sample dispatch skipped due to invalid seed cache dimensions"
                       << " seedOrigin=[" << seedOriginChunk.x << "," << seedOriginChunk.y << "]"
                       << " seedSize=[" << seedSizeChunks.x << "," << seedSizeChunks.y << "]";
                chunkManagerDebugLog(stream.str());
            }
            return;
        }
        const std::uint64_t expectedSeedHeaderElementCount =
            static_cast<std::uint64_t>(seedSizeChunks.x) * static_cast<std::uint64_t>(seedSizeChunks.y);
        const std::uint64_t expectedSeedDataElementCount =
            expectedSeedHeaderElementCount * static_cast<std::uint64_t>(kGpuContextChunkSeedCountPerCacheEntry);
        if (static_cast<std::uint64_t>(seedHeaderElementCount) < expectedSeedHeaderElementCount ||
            static_cast<std::uint64_t>(seedDataElementCount) < expectedSeedDataElementCount)
        {
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD sample dispatch skipped due to undersized seed buffers"
                       << " expectedHeader=" << expectedSeedHeaderElementCount
                       << " expectedData=" << expectedSeedDataElementCount
                       << " actualHeader=" << seedHeaderElementCount
                       << " actualData=" << seedDataElementCount;
                chunkManagerDebugLog(stream.str());
            }
            return;
        }

        if (logEnabled)
        {
            std::ostringstream stream;
            stream << "Far LOD sample dispatch begin"
                   << " id=" << dispatchId
                   << " atlasOrigin=[" << atlasOriginCell.x << "," << atlasOriginCell.y << "]"
                   << " atlasSize=[" << atlasSizeCells.x << "," << atlasSizeCells.y << "]"
                   << " updateOrigin=[" << updateOriginCell.x << "," << updateOriginCell.y << "]"
                   << " updateSize=[" << updateSizeCells.x << "," << updateSizeCells.y << "]"
                   << " blockScale=" << blockScale
                   << " seedOrigin=[" << seedOriginChunk.x << "," << seedOriginChunk.y << "]"
                   << " seedSize=[" << seedSizeChunks.x << "," << seedSizeChunks.y << "]"
                   << " biomeCount=" << biomeCount
                   << " biomeSelCount=" << biomeSelectionCount
                   << " oceanSelCount=" << oceanSelectionCount
                   << " subBiomeCount=" << subBiomeCount
                   << " permutationCount=" << permutationCount
                   << " seedHeaderCount=" << seedHeaderElementCount
                   << " seedDataCount=" << seedDataElementCount
                   << " sampleCount=" << sampleElementCount
                   << " descriptorCursor=" << descriptorCursor_
                   << " resources=[header=" << hexPtr(worldgenHeaderBuffer)
                   << ",biomes=" << hexPtr(worldgenBiomeBuffer)
                   << ",biomeSel=" << hexPtr(biomeSelectionBuffer)
                   << ",oceanSel=" << hexPtr(oceanSelectionBuffer)
                   << ",subBiome=" << hexPtr(subBiomeBuffer)
                   << ",perm=" << hexPtr(permutationBuffer)
                   << ",seedHeader=" << hexPtr(seedHeaderBuffer)
                   << ",seedData=" << hexPtr(seedDataBuffer)
                   << ",sample=" << hexPtr(sampleBuffer) << "]";
            chunkManagerDebugLog(stream.str());
        }

        std::array<std::uint32_t, 14> constants{
            static_cast<std::uint32_t>(atlasOriginCell.x),
            static_cast<std::uint32_t>(atlasOriginCell.y),
            static_cast<std::uint32_t>(atlasSizeCells.x),
            static_cast<std::uint32_t>(atlasSizeCells.y),
            static_cast<std::uint32_t>(updateOriginCell.x),
            static_cast<std::uint32_t>(updateOriginCell.y),
            static_cast<std::uint32_t>(updateSizeCells.x),
            static_cast<std::uint32_t>(updateSizeCells.y),
            static_cast<std::uint32_t>(blockScale),
            0u,
            static_cast<std::uint32_t>(seedOriginChunk.x),
            static_cast<std::uint32_t>(seedOriginChunk.y),
            static_cast<std::uint32_t>(seedSizeChunks.x),
            static_cast<std::uint32_t>(seedSizeChunks.y)};
        const UINT srvDescriptorIndex = allocateDescriptorRange(8);
        const UINT uavDescriptorIndex = allocateDescriptorRange(1);
        if (logEnabled)
        {
            std::ostringstream stream;
            stream << "Far LOD sample dispatch descriptors"
                   << " id=" << dispatchId
                   << " srvBase=" << srvDescriptorIndex
                   << " uavBase=" << uavDescriptorIndex
                   << " descriptorCursor=" << descriptorCursor_;
            chunkManagerDebugLog(stream.str());
        }

        if (logEnabled)
        {
            chunkManagerDebugLog("Far LOD sample dispatch bind begin id=" + std::to_string(dispatchId));
        }
        bindAtlasSampleDescriptors(srvDescriptorIndex,
                                   uavDescriptorIndex,
                                   worldgenHeaderBuffer,
                                   worldgenBiomeBuffer,
                                   biomeCount,
                                   biomeSelectionBuffer,
                                   biomeSelectionCount,
                                   oceanSelectionBuffer,
                                   oceanSelectionCount,
                                   subBiomeBuffer,
                                   subBiomeCount,
                                   permutationBuffer,
                                   permutationCount,
                                   seedHeaderBuffer,
                                   seedHeaderElementCount,
                                   seedDataBuffer,
                                   seedDataElementCount,
                                   sampleBuffer,
                                   sampleElementCount);
        if (logEnabled)
        {
            chunkManagerDebugLog("Far LOD sample dispatch bind end id=" + std::to_string(dispatchId));
            chunkManagerDebugLog("Far LOD sample dispatch set_bindings begin id=" + std::to_string(dispatchId));
        }
        setAtlasBindings(atlasSampleRootSignature_.Get(),
                         atlasSampleCachePipelineState_.Get(),
                         constants,
                         srvDescriptorIndex,
                         uavDescriptorIndex);
        if (logEnabled)
        {
            chunkManagerDebugLog("Far LOD sample dispatch set_bindings end id=" + std::to_string(dispatchId));
            chunkManagerDebugLog("Far LOD sample dispatch dispatch begin id=" + std::to_string(dispatchId));
        }
        commandList_->Dispatch(static_cast<UINT>((updateSizeCells.x + 7) / 8),
                               static_cast<UINT>((updateSizeCells.y + 7) / 8),
                               1);
        if (logEnabled)
        {
            chunkManagerDebugLog("Far LOD sample dispatch dispatch end id=" + std::to_string(dispatchId));
        }
        hasCommands_ = true;
        if (logEnabled)
        {
            chunkManagerDebugLog("Far LOD sample dispatch complete id=" + std::to_string(dispatchId));
        }
    }

    void dispatchAtlasUpdate(const glm::ivec2& atlasOriginCell,
                             const glm::ivec2& atlasSizeCells,
                             const glm::ivec2& updateOriginCell,
                             const glm::ivec2& updateSizeCells,
                             int blockScale,
                             int seaLevel,
                             ID3D12Resource* worldgenHeaderBuffer,
                             ID3D12Resource* worldgenBiomeBuffer,
                             std::uint32_t biomeCount,
                             ID3D12Resource* permutationBuffer,
                             std::uint32_t permutationCount,
                             ID3D12Resource* sampleBuffer,
                             std::uint32_t sampleElementCount,
                             ID3D12Resource* atlasBuffer,
                             std::uint32_t atlasElementCount)
    {
        if (!open_ || worldgenHeaderBuffer == nullptr || worldgenBiomeBuffer == nullptr ||
            permutationBuffer == nullptr || sampleBuffer == nullptr ||
            atlasBuffer == nullptr || updateSizeCells.x <= 0 || updateSizeCells.y <= 0)
        {
            return;
        }

        std::array<std::uint32_t, 14> constants{
            static_cast<std::uint32_t>(atlasOriginCell.x),
            static_cast<std::uint32_t>(atlasOriginCell.y),
            static_cast<std::uint32_t>(atlasSizeCells.x),
            static_cast<std::uint32_t>(atlasSizeCells.y),
            static_cast<std::uint32_t>(updateOriginCell.x),
            static_cast<std::uint32_t>(updateOriginCell.y),
            static_cast<std::uint32_t>(updateSizeCells.x),
            static_cast<std::uint32_t>(updateSizeCells.y),
            static_cast<std::uint32_t>(blockScale),
            static_cast<std::uint32_t>(seaLevel),
            0u,
            0u,
            0u,
            0u};
        const UINT srvDescriptorIndex = allocateDescriptorRange(4);
        const UINT uavDescriptorIndex = allocateDescriptorRange(1);
        bindAtlasFinalizeDescriptors(srvDescriptorIndex,
                                     uavDescriptorIndex,
                                     worldgenHeaderBuffer,
                                     worldgenBiomeBuffer,
                                     biomeCount,
                                     permutationBuffer,
                                     permutationCount,
                                     sampleBuffer,
                                     sampleElementCount,
                                     atlasBuffer,
                                     atlasElementCount);
        setAtlasBindings(atlasFinalizeRootSignature_.Get(),
                         atlasUpdatePipelineState_.Get(),
                         constants,
                         srvDescriptorIndex,
                         uavDescriptorIndex);
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD GPU dispatch atlas-finalize originCell=[" << updateOriginCell.x << "," << updateOriginCell.y
                   << "] size=[" << updateSizeCells.x << "," << updateSizeCells.y << "]"
                   << " atlasOrigin=[" << atlasOriginCell.x << "," << atlasOriginCell.y << "]"
                   << " atlasSize=[" << atlasSizeCells.x << "," << atlasSizeCells.y << "]"
                   << " blockScale=" << blockScale
                   << " descriptorBase=" << srvDescriptorIndex;
            chunkManagerDebugLog(stream.str());
        }
        commandList_->Dispatch(static_cast<UINT>((updateSizeCells.x + 7) / 8),
                               static_cast<UINT>((updateSizeCells.y + 7) / 8),
                               1);
        hasCommands_ = true;
    }

    void dispatchSynth(const glm::ivec3& worldMin,
                       int blockScale,
                       int seaLevel,
                       const glm::ivec2& atlasOriginCell,
                       const glm::ivec2& atlasSizeCells,
                       ID3D12Resource* atlasBuffer,
                       std::uint32_t atlasElementCount,
                       ID3D12Resource* columnBuffer)
    {
        if (!open_ || atlasBuffer == nullptr || columnBuffer == nullptr)
        {
            return;
        }

        std::array<std::uint32_t, 9> constants{
            static_cast<std::uint32_t>(worldMin.x),
            static_cast<std::uint32_t>(worldMin.y),
            static_cast<std::uint32_t>(worldMin.z),
            static_cast<std::uint32_t>(blockScale),
            static_cast<std::uint32_t>(seaLevel),
            static_cast<std::uint32_t>(atlasOriginCell.x),
            static_cast<std::uint32_t>(atlasOriginCell.y),
            static_cast<std::uint32_t>(atlasSizeCells.x),
            static_cast<std::uint32_t>(atlasSizeCells.y)};
        const UINT descriptorIndex = allocateDescriptorRange(2);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     atlasBuffer,
                                     0,
                                     atlasElementCount,
                                     kGpuContextAtlasSampleStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 1u,
                                     columnBuffer,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(synthColumnPipelineState_.Get());
        commandList_->SetComputeRootSignature(synthColumnRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, srvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE columnUavHandle = srvHandle;
        columnUavHandle.ptr += descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(2, columnUavHandle);
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD GPU dispatch synth-columns worldMin=[" << worldMin.x << "," << worldMin.y << "," << worldMin.z
                   << "] blockScale=" << blockScale
                   << " atlasSize=[" << atlasSizeCells.x << "," << atlasSizeCells.y << "]"
                   << " descriptorBase=" << descriptorIndex;
            chunkManagerDebugLog(stream.str());
        }
        commandList_->Dispatch(4, 4, 1);
        hasCommands_ = true;
    }

    void dispatchStamp(const glm::ivec3& worldMin,
                       int blockScale,
                       int lodLevel,
                       std::uint32_t structureCount,
                       ID3D12Resource* structureBuffer,
                       ID3D12Resource* voxelBuffer)
    {
        if (!open_ || structureCount == 0 || structureBuffer == nullptr || voxelBuffer == nullptr)
        {
            return;
        }

        std::array<std::uint32_t, 6> constants{
            static_cast<std::uint32_t>(worldMin.x),
            static_cast<std::uint32_t>(worldMin.y),
            static_cast<std::uint32_t>(worldMin.z),
            static_cast<std::uint32_t>(blockScale),
            static_cast<std::uint32_t>(lodLevel),
            structureCount};
        const UINT descriptorIndex = allocateDescriptorRange(2);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     structureBuffer,
                                     0,
                                     structureCount,
                                     kGpuContextStructureInstanceStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 1u,
                                     voxelBuffer,
                                     0,
                                     kGpuContextVoxelCount,
                                     kGpuContextPackedVoxelStrideBytes);
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(stampPipelineState_.Get());
        commandList_->SetComputeRootSignature(stampRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, srvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = srvHandle;
        uavHandle.ptr += descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(2, uavHandle);
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD GPU dispatch structure-stamp structureCount=" << structureCount
                   << " worldMin=[" << worldMin.x << "," << worldMin.y << "," << worldMin.z << "]"
                   << " blockScale=" << blockScale
                   << " lodLevel=" << lodLevel
                   << " descriptorBase=" << descriptorIndex;
            chunkManagerDebugLog(stream.str());
        }
        commandList_->Dispatch((kGpuContextVoxelCount + 63u) / 64u, 1, 1);
        hasCommands_ = true;
    }

    void dispatchFaceCount(int worldMinY,
                           int blockScale,
                           std::uint32_t reservedFlags,
                           std::uint32_t maxMergeExtent,
                           ID3D12Resource* voxelBuffer,
                           ID3D12Resource* neighborPosX,
                           ID3D12Resource* neighborNegX,
                           ID3D12Resource* neighborPosZ,
                           ID3D12Resource* neighborNegZ,
                           ID3D12Resource* faceCountBuffer,
                           ID3D12Resource* faceAnalysisBuffer,
                           ID3D12Resource* faceDescriptorBuffer)
    {
        if (!open_ || voxelBuffer == nullptr || neighborPosX == nullptr ||
            neighborNegX == nullptr || neighborPosZ == nullptr || neighborNegZ == nullptr ||
            faceCountBuffer == nullptr || faceAnalysisBuffer == nullptr || faceDescriptorBuffer == nullptr)
        {
            return;
        }

        std::array<std::uint32_t, 4> constants{
            static_cast<std::uint32_t>(worldMinY),
            static_cast<std::uint32_t>(blockScale),
            reservedFlags,
            maxMergeExtent};
        const UINT descriptorIndex = allocateDescriptorRange(8);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     voxelBuffer,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 1u,
                                     neighborPosX,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 2u,
                                     neighborNegX,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 3u,
                                     neighborPosZ,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 4u,
                                     neighborNegZ,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 5u,
                                     faceCountBuffer,
                                     0,
                                     kGpuContextPlaneCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 6u,
                                     faceAnalysisBuffer,
                                     0,
                                     kGpuContextFaceMetadataEntryCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 7u,
                                     faceDescriptorBuffer,
                                     0,
                                     kGpuContextFaceDescriptorCount,
                                     kGpuContextFaceDescriptorStrideBytes);
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(faceCountPipelineState_.Get());
        commandList_->SetComputeRootSignature(faceCountRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, srvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = srvHandle;
        uavHandle.ptr += descriptorSize_ * 5u;
        commandList_->SetComputeRootDescriptorTable(2, uavHandle);
        commandList_->Dispatch(kGpuContextPlaneDispatchGroupCount, 1, 1);
        hasCommands_ = true;
    }

    void dispatchFacePrefix(ID3D12Resource* faceCountBuffer,
                            ID3D12Resource* facePrefixBuffer,
                            ID3D12Resource* faceGroupSumBuffer)
    {
        if (!open_ || faceCountBuffer == nullptr || facePrefixBuffer == nullptr || faceGroupSumBuffer == nullptr)
        {
            return;
        }

        UINT descriptorIndex = allocateDescriptorRange(3);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     faceCountBuffer,
                                     0,
                                     kGpuContextPlaneCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 1u,
                                     facePrefixBuffer,
                                     0,
                                     kGpuContextPlaneCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 2u,
                                     faceGroupSumBuffer,
                                     0,
                                     kGpuContextFacePrefixGroupCount,
                                     kGpuContextPackedVoxelStrideBytes);
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(facePrefixGroupPipelineState_.Get());
        commandList_->SetComputeRootSignature(facePrefixGroupRootSignature_.Get());
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(0, srvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = srvHandle;
        uavHandle.ptr += descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, uavHandle);
        commandList_->Dispatch(kGpuContextFacePrefixGroupCount, 1, 1);
        uavBarrier(facePrefixBuffer);
        uavBarrier(faceGroupSumBuffer);

        descriptorIndex = allocateDescriptorRange(1);
        writeStructuredUavDescriptor(descriptorIndex,
                                     faceGroupSumBuffer,
                                     0,
                                     kGpuContextFacePrefixGroupCount,
                                     kGpuContextPackedVoxelStrideBytes);
        D3D12_GPU_DESCRIPTOR_HANDLE scanUavHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        scanUavHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetPipelineState(facePrefixScanPipelineState_.Get());
        commandList_->SetComputeRootSignature(facePrefixScanRootSignature_.Get());
        commandList_->SetComputeRootDescriptorTable(0, scanUavHandle);
        commandList_->Dispatch(1, 1, 1);
        uavBarrier(faceGroupSumBuffer);

        descriptorIndex = allocateDescriptorRange(2);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     faceGroupSumBuffer,
                                     0,
                                     kGpuContextFacePrefixGroupCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 1u,
                                     facePrefixBuffer,
                                     0,
                                     kGpuContextPlaneCount,
                                     kGpuContextPackedVoxelStrideBytes);
        D3D12_GPU_DESCRIPTOR_HANDLE addSrvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        addSrvHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetPipelineState(facePrefixAddPipelineState_.Get());
        commandList_->SetComputeRootSignature(facePrefixAddRootSignature_.Get());
        commandList_->SetComputeRootDescriptorTable(0, addSrvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE addUavHandle = addSrvHandle;
        addUavHandle.ptr += descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, addUavHandle);
        commandList_->Dispatch(kGpuContextFacePrefixGroupCount, 1, 1);
        uavBarrier(facePrefixBuffer);
        hasCommands_ = true;
    }

    void dispatchFaceEmit(const glm::ivec3& worldMin,
                          int blockScale,
                          std::uint32_t maxMergeExtent,
                          std::uint32_t vertexBase,
                          std::uint32_t indexBase,
                          std::uint32_t recordIndex,
                          std::uint32_t reservedFaceCapacity,
                          ID3D12Resource* voxelBuffer,
                          ID3D12Resource* faceCountBuffer,
                          ID3D12Resource* faceAnalysisBuffer,
                          ID3D12Resource* faceDescriptorBuffer,
                          ID3D12Resource* facePrefixBuffer,
                          ID3D12Resource* blockUvBuffer,
                          std::uint32_t blockUvCount,
                          std::uint32_t blockUvStrideBytes,
                          ID3D12Resource* vertexBuffer,
                          std::uint32_t vertexBufferCount,
                          ID3D12Resource* indexBuffer,
                          std::uint32_t indexBufferCount,
                          ID3D12Resource* drawRecordBuffer,
                          std::uint32_t drawRecordCount)
    {
        if (!open_ || voxelBuffer == nullptr || faceCountBuffer == nullptr ||
            faceAnalysisBuffer == nullptr || faceDescriptorBuffer == nullptr || facePrefixBuffer == nullptr ||
            blockUvBuffer == nullptr || vertexBuffer == nullptr || indexBuffer == nullptr || drawRecordBuffer == nullptr)
        {
            return;
        }

        std::array<std::uint32_t, 9> constants{
            static_cast<std::uint32_t>(worldMin.x),
            static_cast<std::uint32_t>(worldMin.y),
            static_cast<std::uint32_t>(worldMin.z),
            static_cast<std::uint32_t>(blockScale),
            maxMergeExtent,
            vertexBase,
            indexBase,
            recordIndex,
            reservedFaceCapacity};
        const UINT descriptorIndex = allocateDescriptorRange(9);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     voxelBuffer,
                                     0,
                                     kGpuContextColumnCount,
                                     kGpuContextColumnDescriptorStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 1u,
                                     faceCountBuffer,
                                     0,
                                     kGpuContextPlaneCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 2u,
                                     faceAnalysisBuffer,
                                     0,
                                     kGpuContextFaceMetadataEntryCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 3u,
                                     faceDescriptorBuffer,
                                     0,
                                     kGpuContextFaceDescriptorCount,
                                     kGpuContextFaceDescriptorStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 4u,
                                     facePrefixBuffer,
                                     0,
                                     kGpuContextPlaneCount,
                                     kGpuContextPackedVoxelStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 5u,
                                     blockUvBuffer,
                                     0,
                                     blockUvCount,
                                     blockUvStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 6u,
                                     vertexBuffer,
                                     0,
                                     vertexBufferCount,
                                     static_cast<std::uint32_t>(sizeof(Vertex)));
        writeStructuredUavDescriptor(descriptorIndex + 7u,
                                     indexBuffer,
                                     0,
                                     indexBufferCount,
                                     static_cast<std::uint32_t>(sizeof(std::uint32_t)));
        writeStructuredUavDescriptor(descriptorIndex + 8u,
                                     drawRecordBuffer,
                                     0,
                                     drawRecordCount,
                                     static_cast<std::uint32_t>(sizeof(ChunkRenderBatch::GpuCullRecord)));
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(faceEmitPipelineState_.Get());
        commandList_->SetComputeRootSignature(faceEmitRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<UINT64>(descriptorIndex) * descriptorSize_;
        commandList_->SetComputeRootDescriptorTable(1, srvHandle);
        D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = srvHandle;
        uavHandle.ptr += descriptorSize_ * 6u;
        commandList_->SetComputeRootDescriptorTable(2, uavHandle);
        commandList_->Dispatch(kGpuContextPlaneDispatchGroupCount, 1, 1);
        hasCommands_ = true;
    }

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS exactFaceCountScratchAddress(std::uint32_t buildIndex) const noexcept
    {
        if (exactFaceCountScratchBuffer_ == nullptr || buildIndex >= kMaxExactGpuBuildBatches)
        {
            return 0;
        }

        return exactFaceCountScratchBuffer_->GetGPUVirtualAddress() +
               static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(buildIndex) *
                   static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(kExactFaceCountScratchSliceBytes);
    }

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS exactFaceDescriptorScratchAddress(std::uint32_t buildIndex) const noexcept
    {
        if (exactFaceDescriptorScratchBuffer_ == nullptr || buildIndex >= kMaxExactGpuBuildBatches)
        {
            return 0;
        }

        return exactFaceDescriptorScratchBuffer_->GetGPUVirtualAddress() +
               static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(buildIndex) *
                   static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(kExactFaceDescriptorScratchSliceBytes);
    }

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS exactFacePrefixScratchAddress(std::uint32_t buildIndex) const noexcept
    {
        if (exactFacePrefixScratchBuffer_ == nullptr || buildIndex >= kMaxExactGpuBuildBatches)
        {
            return 0;
        }

        return exactFacePrefixScratchBuffer_->GetGPUVirtualAddress() +
               static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(buildIndex) *
                   static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(kExactFacePrefixScratchSliceBytes);
    }

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS exactFaceTotalScratchAddress(std::uint32_t buildIndex) const noexcept
    {
        if (exactFaceTotalScratchBuffer_ == nullptr || buildIndex >= kMaxExactGpuBuildBatches)
        {
            return 0;
        }

        return exactFaceTotalScratchBuffer_->GetGPUVirtualAddress() +
               static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(buildIndex) *
                   static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(kExactFaceTotalScratchSliceBytes);
    }

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS exactOverflowCountScratchAddress() const noexcept
    {
        return (exactOverflowCountScratchBuffer_ != nullptr) ? exactOverflowCountScratchBuffer_->GetGPUVirtualAddress()
                                                             : 0;
    }

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS exactOverflowEntryScratchAddress() const noexcept
    {
        return (exactOverflowEntryScratchBuffer_ != nullptr) ? exactOverflowEntryScratchBuffer_->GetGPUVirtualAddress()
                                                             : 0;
    }

    bool clearExactOverflowCounter()
    {
        if (!open_ || exactOverflowCountScratchBuffer_ == nullptr)
        {
            return false;
        }

        const ScratchAllocation zeroUpload = allocateUpload(sizeof(std::uint32_t), alignof(std::uint32_t));
        if (zeroUpload.resource == nullptr || zeroUpload.cpuPtr == nullptr)
        {
            return false;
        }

        *reinterpret_cast<std::uint32_t*>(zeroUpload.cpuPtr) = 0u;
        if (exactOverflowCountScratchState_ != D3D12_RESOURCE_STATE_COPY_DEST)
        {
            transition(exactOverflowCountScratchBuffer_.Get(),
                       exactOverflowCountScratchState_,
                       D3D12_RESOURCE_STATE_COPY_DEST);
            exactOverflowCountScratchState_ = D3D12_RESOURCE_STATE_COPY_DEST;
        }

        copyBuffer(exactOverflowCountScratchBuffer_.Get(),
                   0,
                   zeroUpload.resource,
                   zeroUpload.offset,
                   sizeof(std::uint32_t));
        return true;
    }

    [[nodiscard]] ExactOverflowReadback queueExactOverflowReadback(std::uint32_t maxEntryCount)
    {
        ExactOverflowReadback readback{};
        if (!open_ || exactOverflowCountScratchBuffer_ == nullptr || exactOverflowEntryScratchBuffer_ == nullptr)
        {
            return readback;
        }

        const std::uint64_t entryOffset = (sizeof(std::uint32_t) + alignof(ExactOverflowEntry) - 1u) /
                                          alignof(ExactOverflowEntry) * alignof(ExactOverflowEntry);
        const std::uint64_t totalSize =
            entryOffset + static_cast<std::uint64_t>(maxEntryCount) * sizeof(ExactOverflowEntry);
        readback.allocation = allocateReadback(totalSize, alignof(ExactOverflowEntry));
        readback.entryOffset = entryOffset;
        if (readback.allocation.resource == nullptr)
        {
            return readback;
        }

        if (exactOverflowCountScratchState_ != D3D12_RESOURCE_STATE_COPY_SOURCE)
        {
            transition(exactOverflowCountScratchBuffer_.Get(),
                       exactOverflowCountScratchState_,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            exactOverflowCountScratchState_ = D3D12_RESOURCE_STATE_COPY_SOURCE;
        }
        if (exactOverflowEntryScratchState_ != D3D12_RESOURCE_STATE_COPY_SOURCE)
        {
            transition(exactOverflowEntryScratchBuffer_.Get(),
                       exactOverflowEntryScratchState_,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            exactOverflowEntryScratchState_ = D3D12_RESOURCE_STATE_COPY_SOURCE;
        }

        copyBuffer(readback.allocation.resource,
                   readback.allocation.offset,
                   exactOverflowCountScratchBuffer_.Get(),
                   0,
                   sizeof(std::uint32_t));
        if (maxEntryCount > 0u)
        {
            copyBuffer(readback.allocation.resource,
                       readback.allocation.offset + entryOffset,
                       exactOverflowEntryScratchBuffer_.Get(),
                       0,
                       static_cast<std::uint64_t>(maxEntryCount) * sizeof(ExactOverflowEntry));
        }
        return readback;
    }

    [[nodiscard]] ExactFaceTotalsReadback queueExactFaceTotalsReadback(std::uint32_t buildCount)
    {
        ExactFaceTotalsReadback readback{};
        if (!open_ || buildCount == 0u || exactFaceTotalScratchBuffer_ == nullptr)
        {
            return readback;
        }

        const std::uint64_t totalSize =
            static_cast<std::uint64_t>(buildCount) * static_cast<std::uint64_t>(kExactFaceTotalScratchSliceBytes);
        readback.allocation = allocateReadback(totalSize, alignof(std::uint32_t));
        readback.strideBytes = static_cast<std::uint64_t>(kExactFaceTotalScratchSliceBytes);
        readback.buildCount = buildCount;
        if (readback.allocation.resource == nullptr)
        {
            return readback;
        }

        if (exactFaceTotalScratchState_ != D3D12_RESOURCE_STATE_COPY_SOURCE)
        {
            transition(exactFaceTotalScratchBuffer_.Get(),
                       exactFaceTotalScratchState_,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            exactFaceTotalScratchState_ = D3D12_RESOURCE_STATE_COPY_SOURCE;
        }

        copyBuffer(readback.allocation.resource,
                   readback.allocation.offset,
                   exactFaceTotalScratchBuffer_.Get(),
                   0,
                   totalSize);
        return readback;
    }

    void dispatchExactSynth(int chunkMinWorldY,
                            ID3D12Resource* columnBuffer,
                            ID3D12Resource* voxelBuffer)
    {
        if (!open_ ||
            exactSynthPipelineState_ == nullptr ||
            columnBuffer == nullptr ||
            voxelBuffer == nullptr)
        {
            return;
        }

        const std::array<std::uint32_t, 4> constants{
            static_cast<std::uint32_t>(chunkMinWorldY),
            kExactChunkColumnCount,
            kExactChunkVoxelCount,
            0u};
        commandList_->SetPipelineState(exactSynthPipelineState_.Get());
        commandList_->SetComputeRootSignature(exactSynthRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        commandList_->SetComputeRootShaderResourceView(1, columnBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(2, voxelBuffer->GetGPUVirtualAddress());
        commandList_->Dispatch((kExactChunkSize + 7u) / 8u,
                               (kExactChunkSize + 7u) / 8u,
                               1u);
        hasCommands_ = true;
    }

    void dispatchExactStamp(std::uint32_t sparseVoxelCount,
                            ID3D12Resource* sparseVoxelBuffer,
                            ID3D12Resource* voxelBuffer)
    {
        if (!open_ ||
            exactStampPipelineState_ == nullptr ||
            sparseVoxelBuffer == nullptr ||
            voxelBuffer == nullptr ||
            sparseVoxelCount == 0u)
        {
            return;
        }

        const std::array<std::uint32_t, 4> constants{
            sparseVoxelCount,
            kExactChunkVoxelCount,
            0u,
            0u};
        commandList_->SetPipelineState(exactStampPipelineState_.Get());
        commandList_->SetComputeRootSignature(exactStampRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        commandList_->SetComputeRootShaderResourceView(1, sparseVoxelBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(2, voxelBuffer->GetGPUVirtualAddress());
        commandList_->Dispatch((sparseVoxelCount + 63u) / 64u, 1u, 1u);
        hasCommands_ = true;
    }

    void dispatchExactHaloCache(ID3D12Resource* neighborPosXBuffer,
                                ID3D12Resource* neighborNegXBuffer,
                                ID3D12Resource* neighborPosYBuffer,
                                ID3D12Resource* neighborNegYBuffer,
                                ID3D12Resource* neighborPosZBuffer,
                                ID3D12Resource* neighborNegZBuffer,
                                ID3D12Resource* haloBuffer,
                                std::uint32_t resolvedNeighborMask,
                                std::uint32_t closedNeighborMask,
                                std::int32_t chunkMinWorldY)
    {
        if (!open_ ||
            exactHaloCachePipelineState_ == nullptr ||
            neighborPosXBuffer == nullptr ||
            neighborNegXBuffer == nullptr ||
            neighborPosYBuffer == nullptr ||
            neighborNegYBuffer == nullptr ||
            neighborPosZBuffer == nullptr ||
            neighborNegZBuffer == nullptr ||
            haloBuffer == nullptr)
        {
            return;
        }

        const std::array<std::uint32_t, 4> constants{
            resolvedNeighborMask,
            closedNeighborMask,
            static_cast<std::uint32_t>(chunkMinWorldY),
            0u};
        commandList_->SetPipelineState(exactHaloCachePipelineState_.Get());
        commandList_->SetComputeRootSignature(exactHaloCacheRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        commandList_->SetComputeRootShaderResourceView(1, neighborPosXBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(2, neighborNegXBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(3, neighborPosYBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(4, neighborNegYBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(5, neighborPosZBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(6, neighborNegZBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(7, haloBuffer->GetGPUVirtualAddress());
        commandList_->Dispatch((kExactChunkSize + 7u) / 8u,
                               (kExactChunkSize + 7u) / 8u,
                               6u);
        hasCommands_ = true;
    }

    void dispatchExactLight(ID3D12Resource* columnBuffer,
                            ID3D12Resource* centerVoxelBuffer,
                            ID3D12Resource* haloBuffer,
                            ID3D12Resource* destinationVoxelBuffer,
                            std::uint32_t resolvedNeighborMask,
                            std::uint32_t closedNeighborMask,
                            std::uint32_t propagationPassCount)
    {
        if (!open_ ||
            exactLightPipelineState_ == nullptr ||
            columnBuffer == nullptr ||
            centerVoxelBuffer == nullptr ||
            haloBuffer == nullptr ||
            destinationVoxelBuffer == nullptr)
        {
            return;
        }

        const std::array<std::uint32_t, 4> constants{
            kExactChunkVoxelCount,
            resolvedNeighborMask,
            closedNeighborMask,
            propagationPassCount};
        commandList_->SetPipelineState(exactLightPipelineState_.Get());
        commandList_->SetComputeRootSignature(exactLightRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        commandList_->SetComputeRootShaderResourceView(1, columnBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(2, centerVoxelBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(3, haloBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(4, destinationVoxelBuffer->GetGPUVirtualAddress());
        commandList_->Dispatch(1u, 1u, 1u);
        hasCommands_ = true;
    }

    void dispatchExactFaceCount(ID3D12Resource* centerVoxelBuffer,
                                ID3D12Resource* haloBuffer,
                                std::uint32_t resolvedNeighborMask,
                                std::uint32_t closedNeighborMask,
                                std::uint32_t buildIndex)
    {
        if (!open_ ||
            exactFaceCountPipelineState_ == nullptr ||
            centerVoxelBuffer == nullptr ||
            haloBuffer == nullptr ||
            exactFaceCountScratchBuffer_ == nullptr ||
            exactFaceDescriptorScratchBuffer_ == nullptr)
        {
            return;
        }

        if (exactFaceCountScratchState_ != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            transition(exactFaceCountScratchBuffer_.Get(),
                       exactFaceCountScratchState_,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            exactFaceCountScratchState_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (exactFaceDescriptorScratchState_ != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            transition(exactFaceDescriptorScratchBuffer_.Get(),
                       exactFaceDescriptorScratchState_,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            exactFaceDescriptorScratchState_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }

        const std::array<std::uint32_t, 6> constants{
            kExactChunkPlaneCount,
            kExactChunkVoxelCount,
            kExactChunkPlaneCount * kExactChunkSize * kExactChunkSize,
            resolvedNeighborMask,
            closedNeighborMask,
            0u};
        commandList_->SetPipelineState(exactFaceCountPipelineState_.Get());
        commandList_->SetComputeRootSignature(exactFaceCountRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        commandList_->SetComputeRootShaderResourceView(1, centerVoxelBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(2, haloBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(3, exactFaceCountScratchAddress(buildIndex));
        commandList_->SetComputeRootUnorderedAccessView(4, exactFaceDescriptorScratchAddress(buildIndex));
        commandList_->Dispatch(kExactChunkPlaneDispatchGroupCount, 1u, 1u);
        hasCommands_ = true;
    }

    void dispatchExactFacePrefix(std::uint32_t buildIndex)
    {
        if (!open_ ||
            exactFacePrefixPipelineState_ == nullptr ||
            exactFaceCountScratchBuffer_ == nullptr ||
            exactFacePrefixScratchBuffer_ == nullptr ||
            exactFaceTotalScratchBuffer_ == nullptr)
        {
            return;
        }

        if (exactFaceCountScratchState_ != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            transition(exactFaceCountScratchBuffer_.Get(),
                       exactFaceCountScratchState_,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            exactFaceCountScratchState_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (exactFacePrefixScratchState_ != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            transition(exactFacePrefixScratchBuffer_.Get(),
                       exactFacePrefixScratchState_,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            exactFacePrefixScratchState_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (exactFaceTotalScratchState_ != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            transition(exactFaceTotalScratchBuffer_.Get(),
                       exactFaceTotalScratchState_,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            exactFaceTotalScratchState_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }

        const std::array<std::uint32_t, 4> constants{kExactChunkPlaneCount, 0u, 0u, 0u};
        commandList_->SetPipelineState(exactFacePrefixPipelineState_.Get());
        commandList_->SetComputeRootSignature(exactFacePrefixRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        commandList_->SetComputeRootUnorderedAccessView(1, exactFaceCountScratchAddress(buildIndex));
        commandList_->SetComputeRootUnorderedAccessView(2, exactFacePrefixScratchAddress(buildIndex));
        commandList_->SetComputeRootUnorderedAccessView(3, exactFaceTotalScratchAddress(buildIndex));
        commandList_->Dispatch(1u, 1u, 1u);
        hasCommands_ = true;
    }

    void dispatchExactFaceEmit(const glm::ivec3& worldMin,
                               std::uint32_t vertexBase,
                               std::uint32_t indexBase,
                               std::uint32_t recordIndex,
                               std::uint32_t reservedFaceCapacity,
                               std::uint32_t resolvedNeighborMask,
                               std::uint32_t closedNeighborMask,
                               std::uint32_t buildIndex,
                               ID3D12Resource* columnBuffer,
                               ID3D12Resource* centerVoxelBuffer,
                               ID3D12Resource* haloBuffer,
                               ID3D12Resource* blockUvBuffer,
                               ID3D12Resource* vertexBuffer,
                               ID3D12Resource* indexBuffer,
                               ID3D12Resource* drawRecordBuffer)
    {
        if (!open_ ||
            exactFaceEmitPipelineState_ == nullptr ||
            columnBuffer == nullptr ||
            centerVoxelBuffer == nullptr ||
            haloBuffer == nullptr ||
            blockUvBuffer == nullptr ||
            vertexBuffer == nullptr ||
            indexBuffer == nullptr ||
            drawRecordBuffer == nullptr ||
            exactFaceCountScratchBuffer_ == nullptr ||
            exactFaceDescriptorScratchBuffer_ == nullptr ||
            exactFacePrefixScratchBuffer_ == nullptr ||
            exactFaceTotalScratchBuffer_ == nullptr ||
            exactOverflowCountScratchBuffer_ == nullptr ||
            exactOverflowEntryScratchBuffer_ == nullptr)
        {
            return;
        }

        if (exactFaceCountScratchState_ != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        {
            transition(exactFaceCountScratchBuffer_.Get(),
                       exactFaceCountScratchState_,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            exactFaceCountScratchState_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        if (exactFaceDescriptorScratchState_ != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        {
            transition(exactFaceDescriptorScratchBuffer_.Get(),
                       exactFaceDescriptorScratchState_,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            exactFaceDescriptorScratchState_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        if (exactFacePrefixScratchState_ != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        {
            transition(exactFacePrefixScratchBuffer_.Get(),
                       exactFacePrefixScratchState_,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            exactFacePrefixScratchState_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        if (exactFaceTotalScratchState_ != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        {
            transition(exactFaceTotalScratchBuffer_.Get(),
                       exactFaceTotalScratchState_,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            exactFaceTotalScratchState_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        if (exactOverflowCountScratchState_ != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            transition(exactOverflowCountScratchBuffer_.Get(),
                       exactOverflowCountScratchState_,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            exactOverflowCountScratchState_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (exactOverflowEntryScratchState_ != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            transition(exactOverflowEntryScratchBuffer_.Get(),
                       exactOverflowEntryScratchState_,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            exactOverflowEntryScratchState_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }

        const std::array<std::uint32_t, 12> constants{
            static_cast<std::uint32_t>(worldMin.x),
            static_cast<std::uint32_t>(worldMin.y),
            static_cast<std::uint32_t>(worldMin.z),
            kExactChunkPlaneCount,
            vertexBase,
            indexBase,
            recordIndex,
            reservedFaceCapacity,
            resolvedNeighborMask,
            closedNeighborMask,
            buildIndex,
            0u};
        commandList_->SetPipelineState(exactFaceEmitPipelineState_.Get());
        commandList_->SetComputeRootSignature(exactFaceEmitRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        commandList_->SetComputeRootShaderResourceView(1, columnBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(2, centerVoxelBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(3, haloBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(4, exactFaceCountScratchAddress(buildIndex));
        commandList_->SetComputeRootShaderResourceView(5, exactFaceDescriptorScratchAddress(buildIndex));
        commandList_->SetComputeRootShaderResourceView(6, exactFacePrefixScratchAddress(buildIndex));
        commandList_->SetComputeRootShaderResourceView(7, exactFaceTotalScratchAddress(buildIndex));
        commandList_->SetComputeRootShaderResourceView(8, blockUvBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(9, vertexBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(10, indexBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(11, drawRecordBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(12, exactOverflowCountScratchAddress());
        commandList_->SetComputeRootUnorderedAccessView(13, exactOverflowEntryScratchAddress());
        commandList_->Dispatch(kExactChunkPlaneCount, 1u, 1u);
        hasCommands_ = true;
    }

    [[nodiscard]] FlushResult flush()
    {
        FlushResult result{};
        if (!open_)
        {
            return result;
        }

        if (hasCommands_)
        {
            if (exactTimingCaptureActive_ &&
                exactTimestampQueryHeap_ != nullptr &&
                exactTimestampReadbackBuffer_ != nullptr &&
                exactTimestampCursor_ > 1)
            {
                commandList_->ResolveQueryData(exactTimestampQueryHeap_.Get(),
                                               D3D12_QUERY_TYPE_TIMESTAMP,
                                               0,
                                               exactTimestampCursor_,
                                               exactTimestampReadbackBuffer_.Get(),
                                               0);
            }
        }

        throwIfFailedDx(commandList_->Close(), "failed to close far lod compute command list");
        if (hasCommands_)
        {
            ID3D12CommandList* lists[] = {commandList_.Get()};
            queue_->ExecuteCommandLists(static_cast<UINT>(std::size(lists)), lists);
            ++fenceValue_;
            throwIfFailedDx(queue_->Signal(fence_.Get(), fenceValue_), "failed to signal far lod compute fence");
            lastSubmittedFenceValue_ = fenceValue_;
            result.fenceValue = fenceValue_;
            if (exactTimingCaptureActive_ && exactTimestampCursor_ > 1)
            {
                exactTimestampPendingFenceValue_ = fenceValue_;
                exactTimestampSubmittedCount_ = exactTimestampCursor_;
                exactTimingPending_ = true;
            }
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD GPU flush submitted fence=" << fenceValue_;
                chunkManagerDebugLog(stream.str());
            }
        }
        else if (chunkManagerDebugLoggingEnabled())
        {
            chunkManagerDebugLog("Far LOD GPU flush skipped (no commands)");
        }

        open_ = false;
        hasCommands_ = false;
        exactTimingCaptureActive_ = false;
        return result;
    }

    void waitForIdle()
    {
        if (fence_ == nullptr || lastSubmittedFenceValue_ == 0)
        {
            return;
        }
        if (fence_->GetCompletedValue() >= lastSubmittedFenceValue_)
        {
            return;
        }
        throwIfFailedDx(fence_->SetEventOnCompletion(lastSubmittedFenceValue_, fenceEvent_),
                        "failed to wait for far lod compute fence");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }

    void waitForFence(UINT64 fenceValue)
    {
        if (fence_ == nullptr || fenceValue == 0)
        {
            return;
        }
        if (fence_->GetCompletedValue() >= fenceValue)
        {
            return;
        }
        throwIfFailedDx(fence_->SetEventOnCompletion(fenceValue, fenceEvent_),
                        "failed to wait for far lod compute fence value");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }

    void waitForFence(ID3D12Fence* externalFence, UINT64 fenceValue)
    {
        if (!open_ || queue_ == nullptr || externalFence == nullptr || fenceValue == 0)
        {
            return;
        }

        throwIfFailedDx(queue_->Wait(externalFence, fenceValue),
                        "failed to wait for external fence on far lod compute queue");
        hasCommands_ = true;
    }

    [[nodiscard]] bool ready() const noexcept
    {
        return device_ != nullptr && queue_ != nullptr && allocator_ != nullptr && commandList_ != nullptr &&
               descriptorHeap_ != nullptr && descriptorSize_ > 0 &&
               atlasSeedRootSignature_ != nullptr && atlasSampleRootSignature_ != nullptr &&
               atlasFinalizeRootSignature_ != nullptr &&
               atlasSeedCachePipelineState_ != nullptr && atlasSampleCachePipelineState_ != nullptr &&
                atlasUpdatePipelineState_ != nullptr && synthColumnPipelineState_ != nullptr &&
                stampPipelineState_ != nullptr &&
               faceCountPipelineState_ != nullptr &&
               facePrefixGroupPipelineState_ != nullptr &&
               facePrefixScanPipelineState_ != nullptr &&
               facePrefixAddPipelineState_ != nullptr &&
               faceEmitPipelineState_ != nullptr &&
               exactSynthPipelineState_ != nullptr &&
               exactStampPipelineState_ != nullptr &&
               exactLightPipelineState_ != nullptr &&
               exactFaceCountPipelineState_ != nullptr &&
               exactFacePrefixPipelineState_ != nullptr &&
               exactFaceEmitPipelineState_ != nullptr &&
               exactFaceCountScratchBuffer_ != nullptr &&
               exactFaceDescriptorScratchBuffer_ != nullptr &&
               exactFacePrefixScratchBuffer_ != nullptr &&
               exactFaceTotalScratchBuffer_ != nullptr &&
               exactOverflowCountScratchBuffer_ != nullptr &&
               exactOverflowEntryScratchBuffer_ != nullptr;
    }

    [[nodiscard]] UINT64 completedFenceValue() const noexcept
    {
        return (fence_ != nullptr) ? fence_->GetCompletedValue() : 0;
    }

    [[nodiscard]] UINT64 lastSubmittedFenceValue() const noexcept
    {
        return lastSubmittedFenceValue_;
    }

    [[nodiscard]] ID3D12Fence* fence() const noexcept
    {
        return fence_.Get();
    }

    [[nodiscard]] ExactPassTimings latestExactPassTimings()
    {
        if (exactTimingPending_ &&
            exactTimestampReadbackBuffer_ != nullptr &&
            exactTimestampPendingFenceValue_ != 0 &&
            completedFenceValue() >= exactTimestampPendingFenceValue_)
        {
            std::array<std::uint64_t, kExactTimestampQueryCount> timestamps{};
            void* mappedTimestamps = nullptr;
            const D3D12_RANGE timestampRange{
                0, static_cast<SIZE_T>(exactTimestampSubmittedCount_ * sizeof(std::uint64_t))};
            throwIfFailedDx(exactTimestampReadbackBuffer_->Map(0, &timestampRange, &mappedTimestamps),
                            "failed to map exact compute timestamp readback");
            std::memcpy(timestamps.data(),
                        mappedTimestamps,
                        exactTimestampSubmittedCount_ * sizeof(std::uint64_t));
            exactTimestampReadbackBuffer_->Unmap(0, nullptr);

            const auto timestampMs = [this](std::uint64_t begin, std::uint64_t end) -> double
            {
                if (end <= begin || exactTimestampFrequency_ == 0)
                {
                    return 0.0;
                }
                return static_cast<double>(end - begin) * 1000.0 / static_cast<double>(exactTimestampFrequency_);
            };

            ExactPassTimings timings{};
            for (UINT queryBase = 0; queryBase + 11u < exactTimestampSubmittedCount_; queryBase += 12u)
            {
                timings.synthMs += timestampMs(timestamps[queryBase + 0u], timestamps[queryBase + 1u]);
                timings.stampMs += timestampMs(timestamps[queryBase + 2u], timestamps[queryBase + 3u]);
                timings.lightMs += timestampMs(timestamps[queryBase + 4u], timestamps[queryBase + 5u]);
                timings.faceCountMs += timestampMs(timestamps[queryBase + 6u], timestamps[queryBase + 7u]);
                timings.facePrefixMs += timestampMs(timestamps[queryBase + 8u], timestamps[queryBase + 9u]);
                timings.faceEmitMs += timestampMs(timestamps[queryBase + 10u], timestamps[queryBase + 11u]);
            }
            timings.totalMs = timings.synthMs + timings.stampMs + timings.lightMs +
                              timings.faceCountMs + timings.facePrefixMs + timings.faceEmitMs;

            exactLastCompletedTimings_ = timings;
            exactTimingPending_ = false;
        }

        return exactLastCompletedTimings_;
    }

    [[nodiscard]] std::size_t uploadScratchSizeBytes() const noexcept
    {
        return static_cast<std::size_t>(kUploadScratchSizeBytes);
    }

    [[nodiscard]] std::size_t readbackScratchSizeBytes() const noexcept
    {
        return readbackEnabled_ ? static_cast<std::size_t>(kReadbackScratchSizeBytes) : 0u;
    }

    [[nodiscard]] std::size_t exactScratchSizeBytes() const noexcept
    {
        return static_cast<std::size_t>(kMaxExactGpuBuildBatches) *
                   (static_cast<std::size_t>(kExactFaceCountScratchSliceBytes) +
                    static_cast<std::size_t>(kExactFaceDescriptorScratchSliceBytes) +
                    static_cast<std::size_t>(kExactFacePrefixScratchSliceBytes) +
                    static_cast<std::size_t>(kExactFaceTotalScratchSliceBytes)) +
               sizeof(std::uint32_t) +
               static_cast<std::size_t>(kMaxExactGpuBuildBatches) * sizeof(ExactOverflowEntry);
    }

    [[nodiscard]] std::size_t maxExactGpuBuildBatches() const noexcept
    {
        return static_cast<std::size_t>(kMaxExactGpuBuildBatches);
    }

    [[nodiscard]] const std::byte* readbackMappedData() const noexcept
    {
        return readbackScratchMapped_;
    }

private:
    static constexpr std::uint32_t kGpuContextLogicalSize = 16u;
    static constexpr std::uint32_t kGpuContextColumnCount = kGpuContextLogicalSize * kGpuContextLogicalSize;
    static constexpr std::uint32_t kGpuContextVoxelCount = kGpuContextLogicalSize * kGpuContextLogicalSize * kGpuContextLogicalSize;
    static constexpr std::uint32_t kGpuContextFacePrefixGroupSize = 256u;
    static constexpr std::uint32_t kGpuContextTopPlaneCount = 3u;
    static constexpr std::uint32_t kGpuContextSideSlicesPerLayer = 64u;
    static constexpr std::uint32_t kGpuContextPlaneCount =
        kGpuContextTopPlaneCount + 3u * kGpuContextSideSlicesPerLayer;
    static constexpr std::uint32_t kGpuContextPlaneDispatchGroupCount = (kGpuContextPlaneCount + 63u) / 64u;
    static constexpr std::uint32_t kGpuContextFacePrefixGroupCount =
        (kGpuContextPlaneCount + kGpuContextFacePrefixGroupSize - 1u) / kGpuContextFacePrefixGroupSize;
    static constexpr std::uint32_t kGpuContextFaceMetadataEntryCount = 1u;
    static constexpr std::uint32_t kGpuContextMaxTopDescriptorsPerPlane = kGpuContextColumnCount;
    static constexpr std::uint32_t kGpuContextMaxSideDescriptorsPerPlane = kGpuContextLogicalSize;
    static constexpr std::uint32_t kGpuContextFaceDescriptorCount =
        kGpuContextTopPlaneCount * kGpuContextMaxTopDescriptorsPerPlane +
        (kGpuContextPlaneCount - kGpuContextTopPlaneCount) * kGpuContextMaxSideDescriptorsPerPlane;
    static constexpr std::uint32_t kGpuContextColumnDescriptorStrideBytes = 48u;
    static constexpr std::uint32_t kGpuContextAtlasSampleStrideBytes = 48u;
    static constexpr std::uint32_t kGpuContextAtlasSamplePointCacheStrideBytes = 144u;
    static constexpr std::uint32_t kGpuContextBiomeSelectionStrideBytes = 16u;
    static constexpr std::uint32_t kGpuContextTransitionBiomeStrideBytes = 16u;
    static constexpr std::uint32_t kGpuContextSubBiomeStrideBytes = 16u;
    static constexpr std::uint32_t kGpuContextFaceDescriptorStrideBytes = 16u;
    static constexpr std::uint32_t kGpuContextChunkSeedCacheHeaderStrideBytes = 16u;
    static constexpr std::uint32_t kGpuContextChunkSeedCountPerCacheEntry = 64u;
    static constexpr std::uint32_t kGpuContextChunkSeedStrideBytes = 20u;
    static constexpr std::uint32_t kGpuContextPermutationStrideBytes = 4u;
    static constexpr std::uint32_t kGpuContextStructureInstanceStrideBytes = 64u;
    static constexpr std::uint32_t kGpuContextPackedVoxelStrideBytes = 4u;
    static constexpr std::uint32_t kExactChunkSize = 16u;
    static constexpr std::uint32_t kExactChunkColumnCount = kExactChunkSize * kExactChunkSize;
    static constexpr std::uint32_t kExactChunkVoxelCount =
        kExactChunkSize * kExactChunkSize * kExactChunkSize;
    static constexpr std::uint32_t kExactChunkPlaneCount = 102u;
    static constexpr std::uint32_t kExactChunkPlaneDispatchGroupCount =
        (kExactChunkPlaneCount + 63u) / 64u;
    static constexpr std::uint32_t kExactChunkMaxDescriptorsPerPlane =
        kExactChunkSize * kExactChunkSize;
    static constexpr std::uint32_t kExactChunkFaceDescriptorCount =
        kExactChunkPlaneCount * kExactChunkMaxDescriptorsPerPlane;
    static constexpr std::uint32_t kExactChunkColumnDescriptorStrideBytes = 64u;
    static constexpr std::uint32_t kExactChunkSparseVoxelStrideBytes = 16u;
    static constexpr std::uint32_t kExactChunkPackedVoxelStrideBytes = 4u;
    static constexpr std::uint32_t kExactChunkFaceDescriptorStrideBytes = 16u;
    static constexpr std::uint32_t kExactIndirectRootBufferAlignment = 256u;
    static constexpr std::uint32_t kExactFaceCountScratchSliceBytes =
        ((kExactChunkPlaneCount * kExactChunkPackedVoxelStrideBytes + kExactIndirectRootBufferAlignment - 1u) /
         kExactIndirectRootBufferAlignment) *
        kExactIndirectRootBufferAlignment;
    static constexpr std::uint32_t kExactFaceDescriptorScratchSliceBytes =
        ((kExactChunkFaceDescriptorCount * kExactChunkFaceDescriptorStrideBytes + kExactIndirectRootBufferAlignment - 1u) /
         kExactIndirectRootBufferAlignment) *
        kExactIndirectRootBufferAlignment;
    static constexpr std::uint32_t kExactFacePrefixScratchSliceBytes =
        ((kExactChunkPlaneCount * kExactChunkPackedVoxelStrideBytes + kExactIndirectRootBufferAlignment - 1u) /
         kExactIndirectRootBufferAlignment) *
        kExactIndirectRootBufferAlignment;
    static constexpr std::uint32_t kExactFaceTotalScratchSliceBytes =
        ((kExactChunkPackedVoxelStrideBytes + kExactIndirectRootBufferAlignment - 1u) /
         kExactIndirectRootBufferAlignment) *
        kExactIndirectRootBufferAlignment;
    static constexpr std::uint32_t kMaxExactGpuBuildBatches = 64u;
    static constexpr UINT kExactTimestampQueriesPerBuild = 12u;
    static constexpr UINT kExactTimestampQueryCount = kExactTimestampQueriesPerBuild * kMaxExactGpuBuildBatches;
    static constexpr UINT kDescriptorHeapDescriptorCount = 2048u;

    void createExactResources()
    {
        exactFaceCountScratchBuffer_ = createDefaultBuffer(device_.Get(),
                                                           static_cast<std::uint64_t>(kMaxExactGpuBuildBatches) *
                                                               kExactFaceCountScratchSliceBytes,
                                                           D3D12_RESOURCE_STATE_COMMON,
                                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        exactFaceDescriptorScratchBuffer_ = createDefaultBuffer(
            device_.Get(),
            static_cast<std::uint64_t>(kMaxExactGpuBuildBatches) *
                kExactFaceDescriptorScratchSliceBytes,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        exactFacePrefixScratchBuffer_ = createDefaultBuffer(device_.Get(),
                                                            static_cast<std::uint64_t>(kMaxExactGpuBuildBatches) *
                                                                kExactFacePrefixScratchSliceBytes,
                                                            D3D12_RESOURCE_STATE_COMMON,
                                                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        exactFaceTotalScratchBuffer_ = createDefaultBuffer(device_.Get(),
                                                           static_cast<std::uint64_t>(kMaxExactGpuBuildBatches) *
                                                               kExactFaceTotalScratchSliceBytes,
                                                           D3D12_RESOURCE_STATE_COMMON,
                                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        exactOverflowCountScratchBuffer_ = createDefaultBuffer(device_.Get(),
                                                               kExactChunkPackedVoxelStrideBytes,
                                                               D3D12_RESOURCE_STATE_COMMON,
                                                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        exactOverflowEntryScratchBuffer_ = createDefaultBuffer(
            device_.Get(),
            static_cast<std::uint64_t>(kMaxExactGpuBuildBatches) * sizeof(ExactOverflowEntry),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        setDebugObjectName(exactFaceCountScratchBuffer_.Get(), L"ExactChunkFaceCountScratch");
        setDebugObjectName(exactFaceDescriptorScratchBuffer_.Get(), L"ExactChunkFaceDescriptorScratch");
        setDebugObjectName(exactFacePrefixScratchBuffer_.Get(), L"ExactChunkFacePrefixScratch");
        setDebugObjectName(exactFaceTotalScratchBuffer_.Get(), L"ExactChunkFaceTotalScratch");
        setDebugObjectName(exactOverflowCountScratchBuffer_.Get(), L"ExactChunkOverflowCountScratch");
        setDebugObjectName(exactOverflowEntryScratchBuffer_.Get(), L"ExactChunkOverflowEntryScratch");
        exactFaceCountScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        exactFaceDescriptorScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        exactFacePrefixScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        exactFaceTotalScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        exactOverflowCountScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        exactOverflowEntryScratchState_ = D3D12_RESOURCE_STATE_COMMON;
    }

    void createExactTimestampResources()
    {
        D3D12_QUERY_HEAP_DESC queryHeapDesc{};
        queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        queryHeapDesc.Count = kExactTimestampQueryCount;
        throwIfFailedDx(device_->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(&exactTimestampQueryHeap_)),
                        "failed to create exact chunk timestamp query heap");
        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_READBACK;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapProps.CreationNodeMask = 1;
        heapProps.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = static_cast<std::uint64_t>(kExactTimestampQueryCount) * sizeof(std::uint64_t);
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        throwIfFailedDx(device_->CreateCommittedResource(&heapProps,
                                                         D3D12_HEAP_FLAG_NONE,
                                                         &desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST,
                                                         nullptr,
                                                         IID_PPV_ARGS(&exactTimestampReadbackBuffer_)),
                        "failed to create exact chunk timestamp readback buffer");
        throwIfFailedDx(queue_->GetTimestampFrequency(&exactTimestampFrequency_),
                        "failed to query exact chunk compute timestamp frequency");
        setDebugObjectName(exactTimestampQueryHeap_.Get(), L"ExactChunkTimestampQueryHeap");
        setDebugObjectName(exactTimestampReadbackBuffer_.Get(), L"ExactChunkTimestampReadback");
    }

    void createDescriptorHeap()
    {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = kDescriptorHeapDescriptorCount;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        throwIfFailedDx(device_->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&descriptorHeap_)),
                        "failed to create far lod compute descriptor heap");
        setDebugObjectName(descriptorHeap_.Get(), L"FarLodComputeDescriptorHeap");
        descriptorSize_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    [[nodiscard]] UINT allocateDescriptorRange(UINT descriptorCount)
    {
        if (descriptorCursor_ + descriptorCount > kDescriptorHeapDescriptorCount)
        {
            throw std::runtime_error("far lod compute descriptor heap exhausted");
        }
        const UINT baseIndex = descriptorCursor_;
        descriptorCursor_ += descriptorCount;
        return baseIndex;
    }

    void writeStructuredSrvDescriptor(std::uint32_t descriptorIndex,
                                      ID3D12Resource* resource,
                                      std::uint64_t byteOffset,
                                      std::uint32_t elementCount,
                                      std::uint32_t strideBytes)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.Buffer.FirstElement = byteOffset / strideBytes;
        srvDesc.Buffer.NumElements = std::max(elementCount, 1u);
        srvDesc.Buffer.StructureByteStride = strideBytes;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        D3D12_CPU_DESCRIPTOR_HANDLE handle = descriptorHeap_->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<SIZE_T>(descriptorIndex) * descriptorSize_;
        device_->CreateShaderResourceView(resource, &srvDesc, handle);
    }

    void writeStructuredUavDescriptor(std::uint32_t descriptorIndex,
                                      ID3D12Resource* resource,
                                      std::uint64_t byteOffset,
                                      std::uint32_t elementCount,
                                      std::uint32_t strideBytes)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.Buffer.FirstElement = byteOffset / strideBytes;
        uavDesc.Buffer.NumElements = std::max(elementCount, 1u);
        uavDesc.Buffer.StructureByteStride = strideBytes;
        uavDesc.Buffer.CounterOffsetInBytes = 0;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
        D3D12_CPU_DESCRIPTOR_HANDLE handle = descriptorHeap_->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<SIZE_T>(descriptorIndex) * descriptorSize_;
        device_->CreateUnorderedAccessView(resource, nullptr, &uavDesc, handle);
    }

    void compileShaders()
    {
        const std::filesystem::path shaderRoot = std::filesystem::current_path() / "assets" / "shaders";
        atlasSeedCacheShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_column_atlas_update_canonical_cs.hlsl").string(), "FarLodChunkSeedCacheMain", "cs_5_0");
        atlasSampleCacheShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_column_atlas_update_canonical_cs.hlsl").string(), "FarLodColumnSampleCacheMain", "cs_5_0");
        atlasUpdateShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_column_atlas_update_canonical_cs.hlsl").string(), "FarLodColumnAtlasUpdateMain", "cs_5_0");
        synthColumnShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_synth_cs.hlsl").string(), "FarLodChunkSynthMain", "cs_5_0");
        stampShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_structure_stamp_cs.hlsl").string(), "FarLodChunkStructureStampMain", "cs_5_0");
        faceCountShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_face_count_cs.hlsl").string(), "FarLodChunkFaceCountMain", "cs_5_0");
        facePrefixGroupShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_face_prefix_cs.hlsl").string(), "FarLodChunkFacePrefixGroupMain", "cs_5_0");
        facePrefixScanShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_face_prefix_cs.hlsl").string(), "FarLodChunkFacePrefixScanMain", "cs_5_0");
        facePrefixAddShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_face_prefix_cs.hlsl").string(), "FarLodChunkFacePrefixAddMain", "cs_5_0");
        faceEmitShader_ =
            loadShaderBytecodeLocal((shaderRoot / "far_lod_chunk_face_emit_cs.hlsl").string(), "FarLodChunkFaceEmitMain", "cs_5_0");
        exactSynthShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_synth_cs.hlsl").string(), "ExactChunkSynthMain", "cs_5_0");
        exactStampShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_structure_stamp_cs.hlsl").string(), "ExactChunkStructureStampMain", "cs_5_0");
        exactHaloCacheShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_halo_cache_cs.hlsl").string(), "ExactChunkHaloCacheMain", "cs_5_0");
        exactLightShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_light_cs.hlsl").string(), "ExactChunkLightMain", "cs_5_0");
        exactFaceCountShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_face_count_cs.hlsl").string(), "ExactChunkFaceCountMain", "cs_5_0");
        exactFacePrefixShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_face_prefix_cs.hlsl").string(), "ExactChunkFacePrefixMain", "cs_5_0");
        exactFaceEmitShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_face_emit_cs.hlsl").string(), "ExactChunkFaceEmitMain", "cs_5_0");
    }

    void createPipelines()
    {
        auto createRootSignature = [this](const D3D12_ROOT_SIGNATURE_DESC& desc,
                                          Microsoft::WRL::ComPtr<ID3D12RootSignature>& rootSignature,
                                          const char* label)
        {
            Microsoft::WRL::ComPtr<ID3DBlob> serialized;
            Microsoft::WRL::ComPtr<ID3DBlob> rootErrors;
            const std::string serializeMessage = std::string("failed to serialize ") + label;
            const std::string createMessage = std::string("failed to create ") + label;
            throwIfFailedDx(D3D12SerializeRootSignature(&desc,
                                                        D3D_ROOT_SIGNATURE_VERSION_1,
                                                        &serialized,
                                                        &rootErrors),
                            serializeMessage.c_str());
            throwIfFailedDx(device_->CreateRootSignature(0,
                                                         serialized->GetBufferPointer(),
                                                         serialized->GetBufferSize(),
                                                         IID_PPV_ARGS(&rootSignature)),
                            createMessage.c_str());
        };

        D3D12_DESCRIPTOR_RANGE atlasSeedSrvRanges[2]{};
        atlasSeedSrvRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasSeedSrvRanges[0].NumDescriptors = 4;
        atlasSeedSrvRanges[0].BaseShaderRegister = 0;
        atlasSeedSrvRanges[0].OffsetInDescriptorsFromTableStart = 0;
        atlasSeedSrvRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasSeedSrvRanges[1].NumDescriptors = 1;
        atlasSeedSrvRanges[1].BaseShaderRegister = 5;
        atlasSeedSrvRanges[1].OffsetInDescriptorsFromTableStart = 4;
        D3D12_DESCRIPTOR_RANGE atlasSeedUavRange{};
        atlasSeedUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        atlasSeedUavRange.NumDescriptors = 2;
        atlasSeedUavRange.BaseShaderRegister = 0;
        atlasSeedUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> atlasSeedParams{};
        atlasSeedParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        atlasSeedParams[0].Constants.ShaderRegister = 0;
        atlasSeedParams[0].Constants.Num32BitValues = 14;
        atlasSeedParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        atlasSeedParams[1].DescriptorTable.NumDescriptorRanges = static_cast<UINT>(std::size(atlasSeedSrvRanges));
        atlasSeedParams[1].DescriptorTable.pDescriptorRanges = atlasSeedSrvRanges;
        atlasSeedParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        atlasSeedParams[2].DescriptorTable.NumDescriptorRanges = 1;
        atlasSeedParams[2].DescriptorTable.pDescriptorRanges = &atlasSeedUavRange;
        D3D12_ROOT_SIGNATURE_DESC atlasSeedDesc{};
        atlasSeedDesc.NumParameters = static_cast<UINT>(atlasSeedParams.size());
        atlasSeedDesc.pParameters = atlasSeedParams.data();
        createRootSignature(atlasSeedDesc, atlasSeedRootSignature_, "far lod atlas seed root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC atlasSeedPso{};
        atlasSeedPso.pRootSignature = atlasSeedRootSignature_.Get();
        atlasSeedPso.CS = {atlasSeedCacheShader_->GetBufferPointer(), atlasSeedCacheShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&atlasSeedPso, IID_PPV_ARGS(&atlasSeedCachePipelineState_)),
                        "failed to create far lod atlas seed cache pipeline");

        D3D12_DESCRIPTOR_RANGE atlasSampleSrvRanges[2]{};
        atlasSampleSrvRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasSampleSrvRanges[0].NumDescriptors = 4;
        atlasSampleSrvRanges[0].BaseShaderRegister = 0;
        atlasSampleSrvRanges[0].OffsetInDescriptorsFromTableStart = 0;
        atlasSampleSrvRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasSampleSrvRanges[1].NumDescriptors = 4;
        atlasSampleSrvRanges[1].BaseShaderRegister = 5;
        atlasSampleSrvRanges[1].OffsetInDescriptorsFromTableStart = 4;
        D3D12_DESCRIPTOR_RANGE atlasSampleUavRange{};
        atlasSampleUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        atlasSampleUavRange.NumDescriptors = 1;
        atlasSampleUavRange.BaseShaderRegister = 2;
        atlasSampleUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> atlasSampleParams{};
        atlasSampleParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        atlasSampleParams[0].Constants.ShaderRegister = 0;
        atlasSampleParams[0].Constants.Num32BitValues = 14;
        atlasSampleParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        atlasSampleParams[1].DescriptorTable.NumDescriptorRanges = static_cast<UINT>(std::size(atlasSampleSrvRanges));
        atlasSampleParams[1].DescriptorTable.pDescriptorRanges = atlasSampleSrvRanges;
        atlasSampleParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        atlasSampleParams[2].DescriptorTable.NumDescriptorRanges = 1;
        atlasSampleParams[2].DescriptorTable.pDescriptorRanges = &atlasSampleUavRange;
        D3D12_ROOT_SIGNATURE_DESC atlasSampleDesc{};
        atlasSampleDesc.NumParameters = static_cast<UINT>(atlasSampleParams.size());
        atlasSampleDesc.pParameters = atlasSampleParams.data();
        createRootSignature(atlasSampleDesc, atlasSampleRootSignature_, "far lod atlas sample root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC atlasSamplePso{};
        atlasSamplePso.pRootSignature = atlasSampleRootSignature_.Get();
        atlasSamplePso.CS = {atlasSampleCacheShader_->GetBufferPointer(), atlasSampleCacheShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&atlasSamplePso, IID_PPV_ARGS(&atlasSampleCachePipelineState_)),
                        "failed to create far lod atlas sample cache pipeline");

        D3D12_DESCRIPTOR_RANGE atlasFinalizeSrvRanges[3]{};
        atlasFinalizeSrvRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasFinalizeSrvRanges[0].NumDescriptors = 2;
        atlasFinalizeSrvRanges[0].BaseShaderRegister = 0;
        atlasFinalizeSrvRanges[0].OffsetInDescriptorsFromTableStart = 0;
        atlasFinalizeSrvRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasFinalizeSrvRanges[1].NumDescriptors = 1;
        atlasFinalizeSrvRanges[1].BaseShaderRegister = 6;
        atlasFinalizeSrvRanges[1].OffsetInDescriptorsFromTableStart = 2;
        atlasFinalizeSrvRanges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        atlasFinalizeSrvRanges[2].NumDescriptors = 1;
        atlasFinalizeSrvRanges[2].BaseShaderRegister = 9;
        atlasFinalizeSrvRanges[2].OffsetInDescriptorsFromTableStart = 3;
        D3D12_DESCRIPTOR_RANGE atlasFinalizeUavRange{};
        atlasFinalizeUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        atlasFinalizeUavRange.NumDescriptors = 1;
        atlasFinalizeUavRange.BaseShaderRegister = 3;
        atlasFinalizeUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> atlasFinalizeParams{};
        atlasFinalizeParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        atlasFinalizeParams[0].Constants.ShaderRegister = 0;
        atlasFinalizeParams[0].Constants.Num32BitValues = 14;
        atlasFinalizeParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        atlasFinalizeParams[1].DescriptorTable.NumDescriptorRanges = static_cast<UINT>(std::size(atlasFinalizeSrvRanges));
        atlasFinalizeParams[1].DescriptorTable.pDescriptorRanges = atlasFinalizeSrvRanges;
        atlasFinalizeParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        atlasFinalizeParams[2].DescriptorTable.NumDescriptorRanges = 1;
        atlasFinalizeParams[2].DescriptorTable.pDescriptorRanges = &atlasFinalizeUavRange;
        D3D12_ROOT_SIGNATURE_DESC atlasFinalizeDesc{};
        atlasFinalizeDesc.NumParameters = static_cast<UINT>(atlasFinalizeParams.size());
        atlasFinalizeDesc.pParameters = atlasFinalizeParams.data();
        createRootSignature(atlasFinalizeDesc, atlasFinalizeRootSignature_, "far lod atlas finalize root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC atlasPso{};
        atlasPso.pRootSignature = atlasFinalizeRootSignature_.Get();
        atlasPso.CS = {atlasUpdateShader_->GetBufferPointer(), atlasUpdateShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&atlasPso, IID_PPV_ARGS(&atlasUpdatePipelineState_)),
                        "failed to create far lod atlas update pipeline");

        D3D12_DESCRIPTOR_RANGE synthSrvRange{};
        synthSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        synthSrvRange.NumDescriptors = 1;
        synthSrvRange.BaseShaderRegister = 0;
        synthSrvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE synthUavRange{};
        synthUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        synthUavRange.NumDescriptors = 1;
        synthUavRange.BaseShaderRegister = 0;
        synthUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> synthParams{};
        synthParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        synthParams[0].Constants.ShaderRegister = 0;
        synthParams[0].Constants.Num32BitValues = 9;
        synthParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        synthParams[1].DescriptorTable.NumDescriptorRanges = 1;
        synthParams[1].DescriptorTable.pDescriptorRanges = &synthSrvRange;
        synthParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        synthParams[2].DescriptorTable.NumDescriptorRanges = 1;
        synthParams[2].DescriptorTable.pDescriptorRanges = &synthUavRange;
        D3D12_ROOT_SIGNATURE_DESC synthDesc{};
        synthDesc.NumParameters = static_cast<UINT>(synthParams.size());
        synthDesc.pParameters = synthParams.data();
        createRootSignature(synthDesc, synthColumnRootSignature_, "far lod synth column root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC synthColumnPso{};
        synthColumnPso.pRootSignature = synthColumnRootSignature_.Get();
        synthColumnPso.CS = {synthColumnShader_->GetBufferPointer(), synthColumnShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&synthColumnPso, IID_PPV_ARGS(&synthColumnPipelineState_)),
                        "failed to create far lod synth column pipeline");

        D3D12_DESCRIPTOR_RANGE stampSrvRange{};
        stampSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        stampSrvRange.NumDescriptors = 1;
        stampSrvRange.BaseShaderRegister = 0;
        stampSrvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE stampUavRange{};
        stampUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        stampUavRange.NumDescriptors = 1;
        stampUavRange.BaseShaderRegister = 0;
        stampUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> stampParams{};
        stampParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        stampParams[0].Constants.ShaderRegister = 0;
        stampParams[0].Constants.Num32BitValues = 6;
        stampParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        stampParams[1].DescriptorTable.NumDescriptorRanges = 1;
        stampParams[1].DescriptorTable.pDescriptorRanges = &stampSrvRange;
        stampParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        stampParams[2].DescriptorTable.NumDescriptorRanges = 1;
        stampParams[2].DescriptorTable.pDescriptorRanges = &stampUavRange;
        D3D12_ROOT_SIGNATURE_DESC stampDesc{};
        stampDesc.NumParameters = static_cast<UINT>(stampParams.size());
        stampDesc.pParameters = stampParams.data();
        createRootSignature(stampDesc, stampRootSignature_, "far lod stamp root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC stampPso{};
        stampPso.pRootSignature = stampRootSignature_.Get();
        stampPso.CS = {stampShader_->GetBufferPointer(), stampShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&stampPso, IID_PPV_ARGS(&stampPipelineState_)),
                        "failed to create far lod stamp pipeline");

        D3D12_DESCRIPTOR_RANGE faceCountSrvRange{};
        faceCountSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        faceCountSrvRange.NumDescriptors = 5;
        faceCountSrvRange.BaseShaderRegister = 0;
        faceCountSrvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE faceCountUavRange{};
        faceCountUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        faceCountUavRange.NumDescriptors = 3;
        faceCountUavRange.BaseShaderRegister = 0;
        faceCountUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> faceCountParams{};
        faceCountParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        faceCountParams[0].Constants.ShaderRegister = 0;
        faceCountParams[0].Constants.Num32BitValues = 4;
        faceCountParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        faceCountParams[1].DescriptorTable.NumDescriptorRanges = 1;
        faceCountParams[1].DescriptorTable.pDescriptorRanges = &faceCountSrvRange;
        faceCountParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        faceCountParams[2].DescriptorTable.NumDescriptorRanges = 1;
        faceCountParams[2].DescriptorTable.pDescriptorRanges = &faceCountUavRange;
        D3D12_ROOT_SIGNATURE_DESC faceCountDesc{};
        faceCountDesc.NumParameters = static_cast<UINT>(faceCountParams.size());
        faceCountDesc.pParameters = faceCountParams.data();
        createRootSignature(faceCountDesc, faceCountRootSignature_, "far lod face count root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC faceCountPso{};
        faceCountPso.pRootSignature = faceCountRootSignature_.Get();
        faceCountPso.CS = {faceCountShader_->GetBufferPointer(), faceCountShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&faceCountPso, IID_PPV_ARGS(&faceCountPipelineState_)),
                        "failed to create far lod face count pipeline");

        D3D12_DESCRIPTOR_RANGE prefixSrvRange{};
        prefixSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        prefixSrvRange.NumDescriptors = 1;
        prefixSrvRange.BaseShaderRegister = 0;
        prefixSrvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE prefixUavRange{};
        prefixUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        prefixUavRange.NumDescriptors = 2;
        prefixUavRange.BaseShaderRegister = 0;
        prefixUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 2> prefixParams{};
        prefixParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        prefixParams[0].DescriptorTable.NumDescriptorRanges = 1;
        prefixParams[0].DescriptorTable.pDescriptorRanges = &prefixSrvRange;
        prefixParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        prefixParams[1].DescriptorTable.NumDescriptorRanges = 1;
        prefixParams[1].DescriptorTable.pDescriptorRanges = &prefixUavRange;
        D3D12_ROOT_SIGNATURE_DESC prefixDesc{};
        prefixDesc.NumParameters = static_cast<UINT>(prefixParams.size());
        prefixDesc.pParameters = prefixParams.data();
        createRootSignature(prefixDesc, facePrefixGroupRootSignature_, "far lod face prefix group root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC prefixGroupPso{};
        prefixGroupPso.pRootSignature = facePrefixGroupRootSignature_.Get();
        prefixGroupPso.CS = {facePrefixGroupShader_->GetBufferPointer(), facePrefixGroupShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&prefixGroupPso, IID_PPV_ARGS(&facePrefixGroupPipelineState_)),
                        "failed to create far lod face prefix group pipeline");

        D3D12_DESCRIPTOR_RANGE prefixScanUavRange{};
        prefixScanUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        prefixScanUavRange.NumDescriptors = 1;
        prefixScanUavRange.BaseShaderRegister = 0;
        prefixScanUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 1> prefixScanParams{};
        prefixScanParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        prefixScanParams[0].DescriptorTable.NumDescriptorRanges = 1;
        prefixScanParams[0].DescriptorTable.pDescriptorRanges = &prefixScanUavRange;
        D3D12_ROOT_SIGNATURE_DESC prefixScanDesc{};
        prefixScanDesc.NumParameters = static_cast<UINT>(prefixScanParams.size());
        prefixScanDesc.pParameters = prefixScanParams.data();
        createRootSignature(prefixScanDesc, facePrefixScanRootSignature_, "far lod face prefix scan root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC prefixScanPso{};
        prefixScanPso.pRootSignature = facePrefixScanRootSignature_.Get();
        prefixScanPso.CS = {facePrefixScanShader_->GetBufferPointer(), facePrefixScanShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&prefixScanPso, IID_PPV_ARGS(&facePrefixScanPipelineState_)),
                        "failed to create far lod face prefix scan pipeline");

        D3D12_DESCRIPTOR_RANGE prefixAddSrvRange{};
        prefixAddSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        prefixAddSrvRange.NumDescriptors = 1;
        prefixAddSrvRange.BaseShaderRegister = 0;
        prefixAddSrvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE prefixAddUavRange{};
        prefixAddUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        prefixAddUavRange.NumDescriptors = 1;
        prefixAddUavRange.BaseShaderRegister = 0;
        prefixAddUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 2> prefixAddParams{};
        prefixAddParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        prefixAddParams[0].DescriptorTable.NumDescriptorRanges = 1;
        prefixAddParams[0].DescriptorTable.pDescriptorRanges = &prefixAddSrvRange;
        prefixAddParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        prefixAddParams[1].DescriptorTable.NumDescriptorRanges = 1;
        prefixAddParams[1].DescriptorTable.pDescriptorRanges = &prefixAddUavRange;
        D3D12_ROOT_SIGNATURE_DESC prefixAddDesc{};
        prefixAddDesc.NumParameters = static_cast<UINT>(prefixAddParams.size());
        prefixAddDesc.pParameters = prefixAddParams.data();
        createRootSignature(prefixAddDesc, facePrefixAddRootSignature_, "far lod face prefix add root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC prefixAddPso{};
        prefixAddPso.pRootSignature = facePrefixAddRootSignature_.Get();
        prefixAddPso.CS = {facePrefixAddShader_->GetBufferPointer(), facePrefixAddShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&prefixAddPso, IID_PPV_ARGS(&facePrefixAddPipelineState_)),
                        "failed to create far lod face prefix add pipeline");

        D3D12_DESCRIPTOR_RANGE faceEmitSrvRange{};
        faceEmitSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        faceEmitSrvRange.NumDescriptors = 6;
        faceEmitSrvRange.BaseShaderRegister = 0;
        faceEmitSrvRange.OffsetInDescriptorsFromTableStart = 0;
        D3D12_DESCRIPTOR_RANGE faceEmitUavRange{};
        faceEmitUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        faceEmitUavRange.NumDescriptors = 3;
        faceEmitUavRange.BaseShaderRegister = 0;
        faceEmitUavRange.OffsetInDescriptorsFromTableStart = 0;

        std::array<D3D12_ROOT_PARAMETER, 3> faceEmitParams{};
        faceEmitParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        faceEmitParams[0].Constants.ShaderRegister = 0;
        faceEmitParams[0].Constants.Num32BitValues = 9;
        faceEmitParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        faceEmitParams[1].DescriptorTable.NumDescriptorRanges = 1;
        faceEmitParams[1].DescriptorTable.pDescriptorRanges = &faceEmitSrvRange;
        faceEmitParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        faceEmitParams[2].DescriptorTable.NumDescriptorRanges = 1;
        faceEmitParams[2].DescriptorTable.pDescriptorRanges = &faceEmitUavRange;
        D3D12_ROOT_SIGNATURE_DESC faceEmitDesc{};
        faceEmitDesc.NumParameters = static_cast<UINT>(faceEmitParams.size());
        faceEmitDesc.pParameters = faceEmitParams.data();
        createRootSignature(faceEmitDesc, faceEmitRootSignature_, "far lod face emit root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC faceEmitPso{};
        faceEmitPso.pRootSignature = faceEmitRootSignature_.Get();
        faceEmitPso.CS = {faceEmitShader_->GetBufferPointer(), faceEmitShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&faceEmitPso, IID_PPV_ARGS(&faceEmitPipelineState_)),
                        "failed to create far lod face emit pipeline");

        std::array<D3D12_ROOT_PARAMETER, 3> exactSynthParams{};
        exactSynthParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        exactSynthParams[0].Constants.ShaderRegister = 0;
        exactSynthParams[0].Constants.Num32BitValues = 4;
        exactSynthParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        exactSynthParams[1].Descriptor.ShaderRegister = 0;
        exactSynthParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        exactSynthParams[2].Descriptor.ShaderRegister = 0;
        D3D12_ROOT_SIGNATURE_DESC exactSynthDesc{};
        exactSynthDesc.NumParameters = static_cast<UINT>(exactSynthParams.size());
        exactSynthDesc.pParameters = exactSynthParams.data();
        createRootSignature(exactSynthDesc, exactSynthRootSignature_, "exact chunk synth root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactSynthPso{};
        exactSynthPso.pRootSignature = exactSynthRootSignature_.Get();
        exactSynthPso.CS = {exactSynthShader_->GetBufferPointer(), exactSynthShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactSynthPso, IID_PPV_ARGS(&exactSynthPipelineState_)),
                        "failed to create exact chunk synth pipeline");

        std::array<D3D12_ROOT_PARAMETER, 3> exactStampParams{};
        exactStampParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        exactStampParams[0].Constants.ShaderRegister = 0;
        exactStampParams[0].Constants.Num32BitValues = 4;
        exactStampParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        exactStampParams[1].Descriptor.ShaderRegister = 0;
        exactStampParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        exactStampParams[2].Descriptor.ShaderRegister = 0;
        D3D12_ROOT_SIGNATURE_DESC exactStampDesc{};
        exactStampDesc.NumParameters = static_cast<UINT>(exactStampParams.size());
        exactStampDesc.pParameters = exactStampParams.data();
        createRootSignature(exactStampDesc, exactStampRootSignature_, "exact chunk stamp root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactStampPso{};
        exactStampPso.pRootSignature = exactStampRootSignature_.Get();
        exactStampPso.CS = {exactStampShader_->GetBufferPointer(), exactStampShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactStampPso, IID_PPV_ARGS(&exactStampPipelineState_)),
                        "failed to create exact chunk stamp pipeline");

        std::array<D3D12_ROOT_PARAMETER, 8> exactHaloCacheParams{};
        exactHaloCacheParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        exactHaloCacheParams[0].Constants.ShaderRegister = 0;
        exactHaloCacheParams[0].Constants.Num32BitValues = 4;
        for (UINT parameterIndex = 1; parameterIndex <= 6; ++parameterIndex)
        {
            exactHaloCacheParams[parameterIndex].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            exactHaloCacheParams[parameterIndex].Descriptor.ShaderRegister = parameterIndex - 1;
        }
        exactHaloCacheParams[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        exactHaloCacheParams[7].Descriptor.ShaderRegister = 0;
        D3D12_ROOT_SIGNATURE_DESC exactHaloCacheDesc{};
        exactHaloCacheDesc.NumParameters = static_cast<UINT>(exactHaloCacheParams.size());
        exactHaloCacheDesc.pParameters = exactHaloCacheParams.data();
        createRootSignature(exactHaloCacheDesc, exactHaloCacheRootSignature_, "exact chunk halo cache root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactHaloCachePso{};
        exactHaloCachePso.pRootSignature = exactHaloCacheRootSignature_.Get();
        exactHaloCachePso.CS = {exactHaloCacheShader_->GetBufferPointer(), exactHaloCacheShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactHaloCachePso,
                                                            IID_PPV_ARGS(&exactHaloCachePipelineState_)),
                        "failed to create exact chunk halo cache pipeline");

        std::array<D3D12_ROOT_PARAMETER, 5> exactLightParams{};
        exactLightParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        exactLightParams[0].Constants.ShaderRegister = 0;
        exactLightParams[0].Constants.Num32BitValues = 4;
        for (UINT parameterIndex = 1; parameterIndex <= 3; ++parameterIndex)
        {
            exactLightParams[parameterIndex].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            exactLightParams[parameterIndex].Descriptor.ShaderRegister = parameterIndex - 1;
        }
        exactLightParams[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        exactLightParams[4].Descriptor.ShaderRegister = 0;
        D3D12_ROOT_SIGNATURE_DESC exactLightDesc{};
        exactLightDesc.NumParameters = static_cast<UINT>(exactLightParams.size());
        exactLightDesc.pParameters = exactLightParams.data();
        createRootSignature(exactLightDesc, exactLightRootSignature_, "exact chunk light root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactLightPso{};
        exactLightPso.pRootSignature = exactLightRootSignature_.Get();
        exactLightPso.CS = {exactLightShader_->GetBufferPointer(), exactLightShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactLightPso,
                                                            IID_PPV_ARGS(&exactLightPipelineState_)),
                        "failed to create exact chunk light pipeline");

        std::array<D3D12_ROOT_PARAMETER, 5> exactFaceCountParams{};
        exactFaceCountParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        exactFaceCountParams[0].Constants.ShaderRegister = 0;
        exactFaceCountParams[0].Constants.Num32BitValues = 6;
        exactFaceCountParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        exactFaceCountParams[1].Descriptor.ShaderRegister = 0;
        exactFaceCountParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        exactFaceCountParams[2].Descriptor.ShaderRegister = 1;
        exactFaceCountParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        exactFaceCountParams[3].Descriptor.ShaderRegister = 0;
        exactFaceCountParams[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        exactFaceCountParams[4].Descriptor.ShaderRegister = 1;
        D3D12_ROOT_SIGNATURE_DESC exactFaceCountDesc{};
        exactFaceCountDesc.NumParameters = static_cast<UINT>(exactFaceCountParams.size());
        exactFaceCountDesc.pParameters = exactFaceCountParams.data();
        createRootSignature(exactFaceCountDesc, exactFaceCountRootSignature_, "exact chunk face count root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactFaceCountPso{};
        exactFaceCountPso.pRootSignature = exactFaceCountRootSignature_.Get();
        exactFaceCountPso.CS = {exactFaceCountShader_->GetBufferPointer(), exactFaceCountShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactFaceCountPso, IID_PPV_ARGS(&exactFaceCountPipelineState_)),
                        "failed to create exact chunk face count pipeline");

        std::array<D3D12_ROOT_PARAMETER, 4> exactPrefixParams{};
        exactPrefixParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        exactPrefixParams[0].Constants.ShaderRegister = 0;
        exactPrefixParams[0].Constants.Num32BitValues = 4;
        exactPrefixParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        exactPrefixParams[1].Descriptor.ShaderRegister = 0;
        exactPrefixParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        exactPrefixParams[2].Descriptor.ShaderRegister = 1;
        exactPrefixParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        exactPrefixParams[3].Descriptor.ShaderRegister = 2;
        D3D12_ROOT_SIGNATURE_DESC exactPrefixDesc{};
        exactPrefixDesc.NumParameters = static_cast<UINT>(exactPrefixParams.size());
        exactPrefixDesc.pParameters = exactPrefixParams.data();
        createRootSignature(exactPrefixDesc, exactFacePrefixRootSignature_, "exact chunk face prefix root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactPrefixPso{};
        exactPrefixPso.pRootSignature = exactFacePrefixRootSignature_.Get();
        exactPrefixPso.CS = {exactFacePrefixShader_->GetBufferPointer(), exactFacePrefixShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactPrefixPso, IID_PPV_ARGS(&exactFacePrefixPipelineState_)),
                        "failed to create exact chunk face prefix pipeline");

        std::array<D3D12_ROOT_PARAMETER, 14> exactFaceEmitParams{};
        exactFaceEmitParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        exactFaceEmitParams[0].Constants.ShaderRegister = 0;
        exactFaceEmitParams[0].Constants.Num32BitValues = 12;
        for (UINT parameterIndex = 1; parameterIndex <= 8; ++parameterIndex)
        {
            exactFaceEmitParams[parameterIndex].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            exactFaceEmitParams[parameterIndex].Descriptor.ShaderRegister = parameterIndex - 1;
        }
        for (UINT parameterIndex = 9; parameterIndex <= 13; ++parameterIndex)
        {
            exactFaceEmitParams[parameterIndex].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            exactFaceEmitParams[parameterIndex].Descriptor.ShaderRegister = parameterIndex - 9;
        }
        D3D12_ROOT_SIGNATURE_DESC exactFaceEmitDesc{};
        exactFaceEmitDesc.NumParameters = static_cast<UINT>(exactFaceEmitParams.size());
        exactFaceEmitDesc.pParameters = exactFaceEmitParams.data();
        createRootSignature(exactFaceEmitDesc, exactFaceEmitRootSignature_, "exact chunk face emit root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactFaceEmitPso{};
        exactFaceEmitPso.pRootSignature = exactFaceEmitRootSignature_.Get();
        exactFaceEmitPso.CS = {exactFaceEmitShader_->GetBufferPointer(), exactFaceEmitShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactFaceEmitPso, IID_PPV_ARGS(&exactFaceEmitPipelineState_)),
                        "failed to create exact chunk face emit pipeline");

    }

    static constexpr std::uint64_t kUploadScratchSizeBytes = 16ull * 1024ull * 1024ull;
    static constexpr std::uint64_t kReadbackScratchSizeBytes = 4ull * 1024ull * 1024ull;

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_{nullptr};
    UINT64 fenceValue_{0};
    UINT64 lastSubmittedFenceValue_{0};
    Microsoft::WRL::ComPtr<ID3DBlob> atlasSeedCacheShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> atlasSampleCacheShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> synthColumnShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> stampShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> atlasUpdateShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> faceCountShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> facePrefixGroupShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> facePrefixScanShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> facePrefixAddShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> faceEmitShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactSynthShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactStampShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactHaloCacheShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactLightShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactFaceCountShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactFacePrefixShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactFaceEmitShader_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> atlasSeedRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> atlasSampleRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> atlasFinalizeRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> synthColumnRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> stampRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> faceCountRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> facePrefixGroupRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> facePrefixScanRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> facePrefixAddRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> faceEmitRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> exactSynthRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> exactStampRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> exactHaloCacheRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> exactLightRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> exactFaceCountRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> exactFacePrefixRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> exactFaceEmitRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> atlasSeedCachePipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> atlasSampleCachePipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> atlasUpdatePipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> synthColumnPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> stampPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> faceCountPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> facePrefixGroupPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> facePrefixScanPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> facePrefixAddPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> faceEmitPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactSynthPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactStampPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactHaloCachePipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactLightPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactFaceCountPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactFacePrefixPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactFaceEmitPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptorHeap_;
    Microsoft::WRL::ComPtr<ID3D12Resource> uploadScratch_;
    std::byte* uploadScratchMapped_{nullptr};
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackScratch_;
    std::byte* readbackScratchMapped_{nullptr};
    Microsoft::WRL::ComPtr<ID3D12Resource> exactFaceCountScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactFaceDescriptorScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactFacePrefixScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactFaceTotalScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactOverflowCountScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactOverflowEntryScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> exactTimestampQueryHeap_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactTimestampReadbackBuffer_;
    std::uint64_t uploadCursor_{0};
    std::uint64_t readbackCursor_{0};
    UINT descriptorSize_{0};
    UINT descriptorCursor_{0};
    UINT exactTimestampCursor_{0};
    UINT exactTimestampSubmittedCount_{0};
    D3D12_RESOURCE_STATES exactFaceCountScratchState_{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES exactFaceDescriptorScratchState_{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES exactFacePrefixScratchState_{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES exactFaceTotalScratchState_{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES exactOverflowCountScratchState_{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES exactOverflowEntryScratchState_{D3D12_RESOURCE_STATE_COMMON};
    UINT64 exactTimestampFrequency_{0};
    UINT64 exactTimestampPendingFenceValue_{0};
    bool exactTimingPending_{false};
    bool exactTimingCaptureActive_{false};
    ExactPassTimings exactLastCompletedTimings_{};
    bool open_{false};
    bool hasCommands_{false};
    bool readbackEnabled_{false};
};

inline int floorDiv(int value, int divisor) noexcept
{
    int quotient = value / divisor;
    int remainder = value % divisor;
    if ((remainder != 0) && ((remainder < 0) != (divisor < 0)))
    {
        --quotient;
    }
    return quotient;
}

inline int wrapIndex(int value, int modulus) noexcept
{
    int result = value % modulus;
    if (result < 0)
    {
        result += modulus;
    }
    return result;
}

inline glm::ivec3 localBlockCoords(const glm::ivec3& worldPos, const glm::ivec3& chunkCoord) noexcept
{
    return {
        worldPos.x - chunkCoord.x * kChunkSizeX,
        worldPos.y - chunkCoord.y * kChunkSizeY,
        worldPos.z - chunkCoord.z * kChunkSizeZ
    };
}

enum class BlockFace : std::uint8_t
{
    Top = 0,
    Bottom,
    North,
    South,
    East,
    West,
    Count
};

constexpr std::size_t toIndex(BlockFace face) noexcept
{
    return static_cast<std::size_t>(face);
}

constexpr std::size_t kBlockFaceCount = toIndex(BlockFace::Count);
constexpr std::uint8_t kMaxLightLevel = 15;

inline glm::ivec3 faceOffset(BlockFace face) noexcept
{
    switch (face)
    {
    case BlockFace::Top:
        return glm::ivec3(0, 1, 0);
    case BlockFace::Bottom:
        return glm::ivec3(0, -1, 0);
    case BlockFace::North:
        return glm::ivec3(0, 0, -1);
    case BlockFace::South:
        return glm::ivec3(0, 0, 1);
    case BlockFace::East:
        return glm::ivec3(1, 0, 0);
    case BlockFace::West:
    default:
        return glm::ivec3(-1, 0, 0);
    }
}

inline BlockFace oppositeFace(BlockFace face) noexcept
{
    switch (face)
    {
    case BlockFace::Top:
        return BlockFace::Bottom;
    case BlockFace::Bottom:
        return BlockFace::Top;
    case BlockFace::North:
        return BlockFace::South;
    case BlockFace::South:
        return BlockFace::North;
    case BlockFace::East:
        return BlockFace::West;
    case BlockFace::West:
    default:
        return BlockFace::East;
    }
}

inline std::size_t blockIndex(int x, int y, int z) noexcept
{
    return static_cast<std::size_t>(y) * (kChunkSizeX * kChunkSizeZ) + static_cast<std::size_t>(z) * kChunkSizeX + static_cast<std::size_t>(x);
}

inline std::size_t columnIndex(int x, int z) noexcept
{
    return static_cast<std::size_t>(z) * kChunkSizeX + static_cast<std::size_t>(x);
}

inline int ceilToIntPositive(float value)
{
    const int truncated = static_cast<int>(value);
    return (static_cast<float>(truncated) < value) ? truncated + 1 : truncated;
}

using terrain::BiomeDefinition;
using terrain::ColumnBuildResult;
using terrain::ColumnSample;

// To introduce a new biome:
// 1. Create a new TOML file under assets/biomes describing the biome parameters.
// 2. Provide textures for any new blocks in setBlockTextureAtlasConfig.

inline float hashToUnitFloat(int x, int y, int z) noexcept
{
    constexpr std::uint64_t kMulX = 374761393ull;
    constexpr std::uint64_t kMulY = 668265263ull;
    constexpr std::uint64_t kMulZ = 2147483647ull;
    constexpr std::uint64_t kMixMul = 1274126177ull;
    constexpr std::uint64_t kMask24 = 0xFFFFFFull;

    const auto widen = [](int value) noexcept -> std::uint64_t {
        return static_cast<std::uint64_t>(static_cast<std::uint32_t>(value));
    };

    std::uint64_t h = widen(x) * kMulX + widen(y) * kMulY + widen(z) * kMulZ;
    h = (h ^ (h >> 13)) * kMixMul;
    h ^= (h >> 16);
    return static_cast<float>(h & kMask24) / static_cast<float>(kMask24);
}



