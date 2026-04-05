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
        // When the D3D12 debug layer is enabled, the InfoQueue contains the actionable message
        // explaining why a command list Close() fails. Capturing it here avoids requiring an
        // attached debugger or external debug viewers.
        device_->QueryInterface(IID_PPV_ARGS(&infoQueue_));

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
        infoQueue_.Reset();
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

        const HRESULT closeHr = commandList_->Close();
        if (FAILED(closeHr))
        {
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream failure;
                failure << "failed to close upload command list"
                        << " (hr=" << hexHr(closeHr) << ")";
                const std::string debugMessages = collectInfoQueueMessages();
                if (!debugMessages.empty())
                {
                    failure << "; D3D12 debug messages:" << debugMessages;
                }
                const std::string dredMessages = collectDeviceDredMessages(device_.Get());
                if (!dredMessages.empty())
                {
                    failure << "; DRED:" << dredMessages;
                }
                chunkManagerDebugLog(failure.str());
            }

            throw std::runtime_error("failed to close upload command list");
        }
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
    [[nodiscard]] std::string collectInfoQueueMessages() const
    {
        if (infoQueue_ == nullptr)
        {
            return {};
        }

        const UINT64 messageCount = infoQueue_->GetNumStoredMessagesAllowedByRetrievalFilter();
        if (messageCount == 0)
        {
            return {};
        }

        std::ostringstream oss;
        const UINT64 firstMessage = messageCount > 12 ? messageCount - 12 : 0;
        for (UINT64 i = firstMessage; i < messageCount; ++i)
        {
            SIZE_T messageSize = 0;
            if (FAILED(infoQueue_->GetMessage(i, nullptr, &messageSize)) || messageSize == 0)
            {
                continue;
            }

            std::vector<std::byte> storage(messageSize);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            if (FAILED(infoQueue_->GetMessage(i, message, &messageSize)))
            {
                continue;
            }

            oss << "\n  [" << i << "] " << (message->pDescription != nullptr ? message->pDescription : "<null>");
        }

        return oss.str();
    }

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    Microsoft::WRL::ComPtr<ID3D12InfoQueue> infoQueue_;
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
        double allocateMs{0.0};
        double faceEmitMs{0.0};
        double totalMs{0.0};
    };

    enum class ExactTimingPass : std::uint8_t
    {
        Synth = 0,
        Stamp,
        Light,
        FaceCount,
        FacePrefix,
        Allocate,
        FaceEmit,
        Count,
    };

    struct ScratchAllocation
    {
        ID3D12Resource* resource{nullptr};
        std::byte* cpuPtr{nullptr};
        D3D12_GPU_VIRTUAL_ADDRESS gpuAddress{0};
        std::uint64_t offset{0};
        std::uint64_t size{0};
    };

    struct ExactCompletionReadback
    {
        ScratchAllocation allocation{};
        std::uint64_t strideBytes{0};
        std::uint32_t buildCount{0};
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

    struct ExactEmitConfig
    {
        std::uint32_t pageCount{0};
        std::uint32_t buildRecordCount{0};
        std::uint32_t blockFaceUvDescriptorIndex{std::numeric_limits<std::uint32_t>::max()};
        std::uint32_t reserved0{0};
    };

    struct ExactPageGpuMetadata
    {
        std::uint32_t pageIndex{0};
        std::uint32_t state{0};
        std::uint32_t recordCapacity{0};
        std::uint32_t reserved0{0};
        std::uint32_t faceDescriptorUavDescriptorIndex{std::numeric_limits<std::uint32_t>::max()};
        std::uint32_t drawRecordUavDescriptorIndex{std::numeric_limits<std::uint32_t>::max()};
        std::uint32_t drawRecordMetadataUavDescriptorIndex{std::numeric_limits<std::uint32_t>::max()};
        std::uint32_t reserved1{0};
    };

    struct ExactChunkAllocationRecord
    {
        std::int32_t chunkWorldMinX{0};
        std::int32_t chunkWorldMinY{0};
        std::int32_t chunkWorldMinZ{0};
        std::uint32_t phase{0};
        std::uint32_t statusFlags{0};
        std::uint32_t buildVersion{0};
        std::uint32_t generationEpoch{0};
        std::uint32_t requiredFaceCount{0};
        std::uint32_t pageIndex{0};
        std::uint32_t recordIndex{0};
        std::uint32_t faceBase{0};
        std::uint32_t reservedFaceCapacity{0};
        std::uint32_t centerVoxelSrvDescriptorIndex{std::numeric_limits<std::uint32_t>::max()};
        std::uint32_t haloSrvDescriptorIndex{std::numeric_limits<std::uint32_t>::max()};
        std::uint32_t reserved0{0};
        std::uint32_t reserved1{0};
        std::uint32_t inputVersionLo{0};
        std::uint32_t inputVersionHi{0};
        std::uint32_t reserved2{0};
        std::uint32_t reserved3{0};
    };

    struct ExactDrawRecordMetadata
    {
        std::int32_t chunkWorldMinX{0};
        std::int32_t chunkWorldMinY{0};
        std::int32_t chunkWorldMinZ{0};
        std::uint32_t pageIndex{0};
        std::uint32_t recordIndex{0};
        std::uint32_t buildIndex{0};
        std::uint32_t faceBase{0};
        std::uint32_t faceCount{0};
        std::uint32_t statusFlags{0};
        std::uint32_t buildVersion{0};
        std::uint32_t generationEpoch{0};
        std::uint32_t inputVersionLo{0};
        std::uint32_t inputVersionHi{0};
        std::uint32_t reserved0{0};
        std::uint32_t reserved1{0};
        std::uint32_t reserved2{0};
    };

    struct ExactCompletionEntry
    {
        std::uint32_t buildIndex{0};
        std::uint32_t statusFlags{0};
        std::uint32_t requiredFaces{0};
        std::uint32_t reservedFaceCapacity{0};
        std::int32_t chunkWorldMinX{0};
        std::int32_t chunkWorldMinY{0};
        std::int32_t chunkWorldMinZ{0};
        std::uint32_t pageIndex{0};
        std::uint32_t recordIndex{0};
        std::uint32_t faceBase{0};
        std::uint32_t buildVersion{0};
        std::uint32_t generationEpoch{0};
        std::uint32_t inputVersionLo{0};
        std::uint32_t inputVersionHi{0};
        std::uint32_t reserved0{0};
        std::uint32_t reserved1{0};
    };

    static constexpr std::uint32_t kInvalidDescriptorIndex = std::numeric_limits<std::uint32_t>::max();

    static_assert(sizeof(ExactEmitConfig) == 16u);
    static_assert(sizeof(ExactPageGpuMetadata) == 32u);
    static_assert(sizeof(ExactChunkAllocationRecord) == 80u);
    static_assert(sizeof(ExactDrawRecordMetadata) == 64u);
    static_assert(sizeof(ExactCompletionEntry) == 64u);

    struct FlushResult
    {
        UINT64 fenceValue{0};
        std::uint32_t submissionSlotIndex{std::numeric_limits<std::uint32_t>::max()};
    };

    struct SubmissionSlot
    {
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator{};
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList{};
        UINT64 fenceValue{0};
        UINT exactTimestampSubmittedCount{0};
        UINT exactTimingPassCount{0};
        std::array<ExactTimingPass, static_cast<std::size_t>(ExactTimingPass::Count)> exactTimingPasses{};
        bool exactTimingPending{false};
        bool reserved{false};
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
        throwIfFailedDx(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
                        "failed to create far lod compute fence");
        setDebugObjectName(queue_.Get(), L"FarLodComputeQueue");
        setDebugObjectName(fence_.Get(), L"FarLodComputeFence");
        fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (fenceEvent_ == nullptr)
        {
            throw std::runtime_error("failed to create far lod compute fence event");
        }

        for (std::uint32_t slotIndex = 0; slotIndex < kMaxInFlightSubmissionSlots; ++slotIndex)
        {
            SubmissionSlot& slot = submissionSlots_[slotIndex];
            slot = {};
            throwIfFailedDx(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                            IID_PPV_ARGS(&slot.allocator)),
                            "failed to create far lod compute allocator");
            throwIfFailedDx(device_->CreateCommandList(0,
                                                       D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                       slot.allocator.Get(),
                                                       nullptr,
                                                       IID_PPV_ARGS(&slot.commandList)),
                            "failed to create far lod compute command list");
            throwIfFailedDx(slot.commandList->Close(), "failed to close initial far lod compute command list");

            std::wostringstream allocatorName;
            allocatorName << L"FarLodComputeAllocator[" << slotIndex << L"]";
            setDebugObjectName(slot.allocator.Get(), allocatorName.str().c_str());
            std::wostringstream commandListName;
            commandListName << L"FarLodComputeCommandList[" << slotIndex << L"]";
            setDebugObjectName(slot.commandList.Get(), commandListName.str().c_str());
        }
        allocator_ = submissionSlots_[0].allocator;
        commandList_ = submissionSlots_[0].commandList;

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
        exactDescriptorGenPipelineState_.Reset();
        exactSynthPipelineState_.Reset();
        exactStampPipelineState_.Reset();
        exactHaloCachePipelineState_.Reset();
        exactLightPipelineState_.Reset();
        exactSeamExportPipelineState_.Reset();
        exactFaceCountPipelineState_.Reset();
        exactFacePrefixPipelineState_.Reset();
        exactFaceEmitPipelineState_.Reset();
        exactDrawRecordClearPipelineState_.Reset();
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
        exactDescriptorGenRootSignature_.Reset();
        exactPrepassRootSignature_.Reset();
        exactFaceEmitRootSignature_.Reset();
        exactDrawRecordClearRootSignature_.Reset();
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
        exactSeamExportShader_.Reset();
        exactFaceCountShader_.Reset();
        exactFacePrefixShader_.Reset();
        exactFaceEmitShader_.Reset();
        exactDrawRecordClearShader_.Reset();
        exactFaceCountScratchBuffer_.Reset();
        exactFaceDescriptorScratchBuffer_.Reset();
        exactFacePrefixScratchBuffer_.Reset();
        exactFaceTotalScratchBuffer_.Reset();
        exactDescriptorScratchBuffer_.Reset();
        exactOverflowCountScratchBuffer_.Reset();
        exactOverflowEntryScratchBuffer_.Reset();
        exactCompletionScratchBuffer_.Reset();
        exactTimestampReadbackBuffer_.Reset();
        exactTimestampQueryHeap_.Reset();
        readbackScratch_.Reset();
        uploadScratch_.Reset();
        for (SubmissionSlot& slot : submissionSlots_)
        {
            slot.commandList.Reset();
            slot.allocator.Reset();
            slot.fenceValue = 0;
            slot.exactTimestampSubmittedCount = 0;
            slot.exactTimingPassCount = 0;
            slot.exactTimingPending = false;
            slot.reserved = false;
        }
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
        persistentDescriptorCursor_ = 0;
        lastSubmittedFenceValue_ = 0;
        fenceValue_ = 0;
        exactTimestampFrequency_ = 0;
        exactTimestampCursor_ = 0;
        exactTimingPassCount_ = 0;
        exactTimingCaptureActive_ = false;
        exactLastCompletedTimings_ = {};
        exactCompletionScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        activeSubmissionSlotIndex_ = std::numeric_limits<std::uint32_t>::max();
        submissionSlotCursor_ = 0;
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
        const UINT64 completedFence = completedFenceValue();
        const std::uint32_t slotIndex = acquireSubmissionSlot(completedFence);
        if (slotIndex == std::numeric_limits<std::uint32_t>::max())
        {
            return false;
        }

        SubmissionSlot& slot = submissionSlots_[slotIndex];
        allocator_ = slot.allocator;
        commandList_ = slot.commandList;
        activeSubmissionSlotIndex_ = slotIndex;
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
            stream << "Far LOD compute begin slot=" << slotIndex
                   << " nextFence=" << (fenceValue_ + 1)
                   << " completedFence=" << completedFence;
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
        if (alignedOffset + sizeInBytes > kUploadScratchBytesPerSubmission)
        {
            return allocation;
        }

        const std::uint64_t baseOffset =
            static_cast<std::uint64_t>(activeSubmissionSlotIndex_) * kUploadScratchBytesPerSubmission;
        const std::uint64_t absoluteOffset = baseOffset + alignedOffset;
        allocation.resource = uploadScratch_.Get();
        allocation.cpuPtr = uploadScratchMapped_ + absoluteOffset;
        allocation.gpuAddress = uploadScratch_->GetGPUVirtualAddress() + absoluteOffset;
        allocation.offset = absoluteOffset;
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
        if (alignedOffset + sizeInBytes > kReadbackScratchBytesPerSubmission)
        {
            return allocation;
        }

        const std::uint64_t baseOffset =
            static_cast<std::uint64_t>(activeSubmissionSlotIndex_) * kReadbackScratchBytesPerSubmission;
        const std::uint64_t absoluteOffset = baseOffset + alignedOffset;
        allocation.resource = readbackScratch_.Get();
        allocation.cpuPtr = readbackScratchMapped_ + absoluteOffset;
        allocation.gpuAddress = readbackScratch_->GetGPUVirtualAddress() + absoluteOffset;
        allocation.offset = absoluteOffset;
        allocation.size = sizeInBytes;
        readbackCursor_ = alignedOffset + sizeInBytes;
        return allocation;
    }

    void beginExactTimingBatch()
    {
        exactTimestampCursor_ = 0;
        exactTimingPassCount_ = 0;
        exactTimingCaptureActive_ = false;
        if (!open_ || activeSubmissionSlotIndex_ >= kMaxInFlightSubmissionSlots)
        {
            return;
        }

        const SubmissionSlot& slot = submissionSlots_[activeSubmissionSlotIndex_];
        exactTimingCaptureActive_ = !slot.exactTimingPending;
    }

    void markExactTimingBegin(ExactTimingPass pass)
    {
        if (!open_ || !exactTimingCaptureActive_ || exactTimestampQueryHeap_ == nullptr ||
            exactTimestampCursor_ >= kExactTimestampQueriesPerSubmission ||
            exactTimingPassCount_ >= exactTimingPassesCurrent_.size() ||
            activeSubmissionSlotIndex_ >= kMaxInFlightSubmissionSlots)
        {
            return;
        }

        exactTimingPassesCurrent_[exactTimingPassCount_] = pass;
        const UINT queryIndex =
            activeSubmissionSlotIndex_ * kExactTimestampQueriesPerSubmission + exactTimestampCursor_++;
        commandList_->EndQuery(exactTimestampQueryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, queryIndex);
        hasCommands_ = true;
    }

    void markExactTimingEnd()
    {
        if (!open_ || !exactTimingCaptureActive_ || exactTimestampQueryHeap_ == nullptr ||
            exactTimestampCursor_ >= kExactTimestampQueriesPerSubmission ||
            exactTimingPassCount_ >= exactTimingPassesCurrent_.size() ||
            activeSubmissionSlotIndex_ >= kMaxInFlightSubmissionSlots)
        {
            return;
        }

        const UINT queryIndex =
            activeSubmissionSlotIndex_ * kExactTimestampQueriesPerSubmission + exactTimestampCursor_++;
        commandList_->EndQuery(exactTimestampQueryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, queryIndex);
        ++exactTimingPassCount_;
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

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS exactOverflowCountScratchAddress(std::uint32_t submissionSlotIndex) const noexcept
    {
        if (exactOverflowCountScratchBuffer_ == nullptr || submissionSlotIndex >= kMaxInFlightSubmissionSlots)
        {
            return 0;
        }

        return exactOverflowCountScratchBuffer_->GetGPUVirtualAddress() +
               static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(submissionSlotIndex) *
                   static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(sizeof(std::uint32_t));
    }

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS exactOverflowEntryScratchAddress(std::uint32_t submissionSlotIndex) const noexcept
    {
        if (exactOverflowEntryScratchBuffer_ == nullptr || submissionSlotIndex >= kMaxInFlightSubmissionSlots)
        {
            return 0;
        }

        return exactOverflowEntryScratchBuffer_->GetGPUVirtualAddress() +
               static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(submissionSlotIndex) *
                   static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(kMaxExactGpuBuildBatches) *
                   static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(sizeof(ExactOverflowEntry));
    }

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS exactCompletionScratchAddress(std::uint32_t buildIndex) const noexcept
    {
        if (exactCompletionScratchBuffer_ == nullptr || buildIndex >= kMaxExactGpuBuildBatches)
        {
            return 0;
        }

        return exactCompletionScratchBuffer_->GetGPUVirtualAddress() +
               static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(buildIndex) *
                   static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(sizeof(ExactCompletionEntry));
    }

    bool clearExactOverflowCounter(const ScratchAllocation& zeroUpload)
    {
        if (!open_ ||
            exactOverflowCountScratchBuffer_ == nullptr ||
            zeroUpload.resource == nullptr ||
            zeroUpload.cpuPtr == nullptr ||
            zeroUpload.size < sizeof(std::uint32_t))
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
                   static_cast<std::uint64_t>(activeSubmissionSlotIndex_) * sizeof(std::uint32_t),
                   zeroUpload.resource,
                   zeroUpload.offset,
                   sizeof(std::uint32_t));
        return true;
    }

    [[nodiscard]] ExactCompletionReadback queueExactCompletionReadback(std::span<const std::uint32_t> buildIndices)
    {
        ExactCompletionReadback readback{};
        if (!open_ || buildIndices.empty() || exactCompletionScratchBuffer_ == nullptr)
        {
            return readback;
        }

        const std::uint64_t totalSize =
            static_cast<std::uint64_t>(buildIndices.size()) * sizeof(ExactCompletionEntry);
        readback.allocation = allocateReadback(totalSize, alignof(ExactCompletionEntry));
        readback.strideBytes = sizeof(ExactCompletionEntry);
        readback.buildCount = static_cast<std::uint32_t>(buildIndices.size());
        if (readback.allocation.resource == nullptr)
        {
            return readback;
        }

        if (exactCompletionScratchState_ != D3D12_RESOURCE_STATE_COPY_SOURCE)
        {
            transition(exactCompletionScratchBuffer_.Get(),
                       exactCompletionScratchState_,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            exactCompletionScratchState_ = D3D12_RESOURCE_STATE_COPY_SOURCE;
        }

        for (std::size_t buildOrdinal = 0; buildOrdinal < buildIndices.size(); ++buildOrdinal)
        {
            copyBuffer(readback.allocation.resource,
                       readback.allocation.offset + buildOrdinal * readback.strideBytes,
                       exactCompletionScratchBuffer_.Get(),
                       static_cast<std::uint64_t>(buildIndices[buildOrdinal]) * sizeof(ExactCompletionEntry),
                       sizeof(ExactCompletionEntry));
        }
        return readback;
    }

    [[nodiscard]] ExactFaceTotalsReadback queueExactFaceTotalsReadback(std::span<const std::uint32_t> buildIndices)
    {
        ExactFaceTotalsReadback readback{};
        if (!open_ || buildIndices.empty() || exactFaceTotalScratchBuffer_ == nullptr)
        {
            return readback;
        }

        const std::uint64_t totalSize =
            static_cast<std::uint64_t>(buildIndices.size()) * sizeof(std::uint32_t);
        readback.allocation = allocateReadback(totalSize, alignof(std::uint32_t));
        readback.strideBytes = sizeof(std::uint32_t);
        readback.buildCount = static_cast<std::uint32_t>(buildIndices.size());
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

        for (std::size_t buildOrdinal = 0; buildOrdinal < buildIndices.size(); ++buildOrdinal)
        {
            copyBuffer(readback.allocation.resource,
                       readback.allocation.offset + buildOrdinal * readback.strideBytes,
                       exactFaceTotalScratchBuffer_.Get(),
                       static_cast<std::uint64_t>(buildIndices[buildOrdinal]) *
                           static_cast<std::uint64_t>(kExactFaceTotalScratchSliceBytes),
                       sizeof(std::uint32_t));
        }
        return readback;
    }

    [[nodiscard]] ID3D12Resource* exactDescriptorScratchBuffer() const noexcept
    {
        return exactDescriptorScratchBuffer_.Get();
    }

    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS exactDescriptorScratchAddress(std::uint32_t buildIndex) const noexcept
    {
        if (exactDescriptorScratchBuffer_ == nullptr || buildIndex >= kMaxExactGpuBuildBatches)
        {
            return 0;
        }

        return exactDescriptorScratchBuffer_->GetGPUVirtualAddress() +
               static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(buildIndex) *
                   static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(kExactDescriptorScratchSliceBytes);
    }

    [[nodiscard]] std::uint64_t exactDescriptorScratchOffset(std::uint32_t buildIndex) const noexcept
    {
        return static_cast<std::uint64_t>(buildIndex) * kExactDescriptorScratchSliceBytes;
    }

    void transitionExactDescriptorScratch(D3D12_RESOURCE_STATES targetState)
    {
        if (!open_ || exactDescriptorScratchBuffer_ == nullptr || exactDescriptorScratchState_ == targetState)
        {
            return;
        }

        transition(exactDescriptorScratchBuffer_.Get(), exactDescriptorScratchState_, targetState);
        exactDescriptorScratchState_ = targetState;
    }

    void dispatchExactDescriptorGen(int seaLevel,
                                    std::uint32_t buildCount,
                                    D3D12_GPU_VIRTUAL_ADDRESS pageColumnsAddress,
                                    D3D12_GPU_VIRTUAL_ADDRESS buildParamsAddress,
                                    D3D12_GPU_VIRTUAL_ADDRESS skyLightAddress)
    {
        if (!open_ ||
            exactDescriptorGenPipelineState_ == nullptr ||
            exactDescriptorScratchBuffer_ == nullptr ||
            buildCount == 0u ||
            pageColumnsAddress == 0u ||
            buildParamsAddress == 0u ||
            skyLightAddress == 0u)
        {
            return;
        }

        transitionExactDescriptorScratch(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const std::array<std::uint32_t, 4> constants{
            static_cast<std::uint32_t>(seaLevel),
            buildCount,
            0u,
            0u};
        commandList_->SetPipelineState(exactDescriptorGenPipelineState_.Get());
        commandList_->SetComputeRootSignature(exactDescriptorGenRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        commandList_->SetComputeRootShaderResourceView(1, pageColumnsAddress);
        commandList_->SetComputeRootShaderResourceView(2, buildParamsAddress);
        commandList_->SetComputeRootShaderResourceView(3, skyLightAddress);
        commandList_->SetComputeRootUnorderedAccessView(4, exactDescriptorScratchBuffer_->GetGPUVirtualAddress());
        commandList_->Dispatch((kExactChunkSize + 7u) / 8u,
                               (kExactChunkSize + 7u) / 8u,
                               buildCount);
        hasCommands_ = true;
    }

    [[nodiscard]] bool prepareExactPrepassDispatch(ID3D12PipelineState* pipelineState,
                                                   D3D12_GPU_VIRTUAL_ADDRESS prepassRecordAddress,
                                                   const std::array<std::uint32_t, 4>& constants)
    {
        if (!open_ ||
            pipelineState == nullptr ||
            exactPrepassRootSignature_ == nullptr ||
            descriptorHeap_ == nullptr ||
            exactDescriptorScratchBuffer_ == nullptr ||
            exactFaceCountScratchBuffer_ == nullptr ||
            exactFaceDescriptorScratchBuffer_ == nullptr ||
            exactFacePrefixScratchBuffer_ == nullptr ||
            exactFaceTotalScratchBuffer_ == nullptr ||
            prepassRecordAddress == 0u)
        {
            return false;
        }

        transitionExactDescriptorScratch(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
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

        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(pipelineState);
        commandList_->SetComputeRootSignature(exactPrepassRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        commandList_->SetComputeRootShaderResourceView(1, prepassRecordAddress);
        commandList_->SetComputeRootShaderResourceView(2, exactDescriptorScratchBuffer_->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(3, exactFaceCountScratchBuffer_->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(4, exactFaceDescriptorScratchBuffer_->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(5, exactFacePrefixScratchBuffer_->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(6, exactFaceTotalScratchBuffer_->GetGPUVirtualAddress());
        return true;
    }

    void dispatchExactSynthBatch(std::uint32_t batchBuildCount,
                                 D3D12_GPU_VIRTUAL_ADDRESS prepassRecordAddress)
    {
        if (batchBuildCount == 0u)
        {
            return;
        }

        const std::array<std::uint32_t, 4> constants{
            batchBuildCount,
            0u,
            0u,
            0u};
        if (!prepareExactPrepassDispatch(exactSynthPipelineState_.Get(), prepassRecordAddress, constants))
        {
            return;
        }

        commandList_->Dispatch((kExactChunkSize + 7u) / 8u,
                               (kExactChunkSize + 7u) / 8u,
                               batchBuildCount);
        hasCommands_ = true;
    }

    void dispatchExactStampBatch(std::uint32_t batchBuildCount,
                                 std::uint32_t maxSparseVoxelGroups,
                                 D3D12_GPU_VIRTUAL_ADDRESS prepassRecordAddress)
    {
        if (batchBuildCount == 0u)
        {
            return;
        }

        const std::array<std::uint32_t, 4> constants{
            batchBuildCount,
            maxSparseVoxelGroups,
            0u,
            0u};
        if (!prepareExactPrepassDispatch(exactStampPipelineState_.Get(), prepassRecordAddress, constants))
        {
            return;
        }

        if (maxSparseVoxelGroups > 0u)
        {
            commandList_->Dispatch(maxSparseVoxelGroups, batchBuildCount, 1u);
            hasCommands_ = true;
        }
    }

    void dispatchExactHaloCacheBatch(std::uint32_t batchBuildCount,
                                     D3D12_GPU_VIRTUAL_ADDRESS prepassRecordAddress)
    {
        if (batchBuildCount == 0u)
        {
            return;
        }

        const std::array<std::uint32_t, 4> constants{
            batchBuildCount,
            0u,
            0u,
            0u};
        if (!prepareExactPrepassDispatch(exactHaloCachePipelineState_.Get(), prepassRecordAddress, constants))
        {
            return;
        }

        commandList_->Dispatch((kExactChunkSize + 7u) / 8u,
                               (kExactChunkSize + 7u) / 8u,
                               kExactChunkHaloFaceCount * batchBuildCount);
        hasCommands_ = true;
    }

    void dispatchExactLightBatch(std::uint32_t batchBuildCount,
                                 D3D12_GPU_VIRTUAL_ADDRESS prepassRecordAddress,
                                 std::uint32_t propagationPassCount)
    {
        if (batchBuildCount == 0u)
        {
            return;
        }

        const std::array<std::uint32_t, 4> constants{
            batchBuildCount,
            propagationPassCount,
            0u,
            0u};
        if (!prepareExactPrepassDispatch(exactLightPipelineState_.Get(), prepassRecordAddress, constants))
        {
            return;
        }

        commandList_->Dispatch(1u, 1u, batchBuildCount);
        hasCommands_ = true;
    }

    void dispatchExactSeamExportBatch(std::uint32_t batchBuildCount,
                                      D3D12_GPU_VIRTUAL_ADDRESS prepassRecordAddress)
    {
        if (batchBuildCount == 0u)
        {
            return;
        }

        const std::array<std::uint32_t, 4> constants{
            batchBuildCount,
            0u,
            0u,
            0u};
        if (!prepareExactPrepassDispatch(exactSeamExportPipelineState_.Get(), prepassRecordAddress, constants))
        {
            return;
        }

        commandList_->Dispatch((kExactChunkSize + 7u) / 8u,
                               (kExactChunkSize + 7u) / 8u,
                               kExactChunkHaloFaceCount * batchBuildCount);
        hasCommands_ = true;
    }

    void dispatchExactFaceCountBatch(std::uint32_t batchBuildCount,
                                     D3D12_GPU_VIRTUAL_ADDRESS prepassRecordAddress)
    {
        if (batchBuildCount == 0u)
        {
            return;
        }

        const std::array<std::uint32_t, 4> constants{
            batchBuildCount,
            kExactChunkPlaneCount,
            kExactChunkPlaneCount * kExactChunkSize * kExactChunkSize,
            0u};
        if (!prepareExactPrepassDispatch(exactFaceCountPipelineState_.Get(), prepassRecordAddress, constants))
        {
            return;
        }

        commandList_->Dispatch(kExactChunkPlaneDispatchGroupCount, batchBuildCount, 1u);
        hasCommands_ = true;
    }

    void dispatchExactFacePrefixBatch(std::uint32_t batchBuildCount,
                                      D3D12_GPU_VIRTUAL_ADDRESS prepassRecordAddress)
    {
        if (batchBuildCount == 0u)
        {
            return;
        }

        const std::array<std::uint32_t, 4> constants{
            batchBuildCount,
            kExactChunkPlaneCount,
            0u,
            0u};
        if (!prepareExactPrepassDispatch(exactFacePrefixPipelineState_.Get(), prepassRecordAddress, constants))
        {
            return;
        }

        commandList_->Dispatch(1u, batchBuildCount, 1u);
        hasCommands_ = true;
    }

    void dispatchExactFaceEmit(std::uint32_t batchBuildCount,
                               D3D12_GPU_VIRTUAL_ADDRESS batchBuildIndicesAddress,
                               ID3D12Resource* emitConfigBuffer,
                               ID3D12Resource* buildRecordBuffer,
                               ID3D12Resource* pageMetadataBuffer)
    {
        if (!open_ ||
            exactFaceEmitPipelineState_ == nullptr ||
            batchBuildCount == 0u ||
            batchBuildIndicesAddress == 0u ||
            emitConfigBuffer == nullptr ||
            buildRecordBuffer == nullptr ||
            pageMetadataBuffer == nullptr ||
            descriptorHeap_ == nullptr ||
            exactFaceCountScratchBuffer_ == nullptr ||
            exactFaceDescriptorScratchBuffer_ == nullptr ||
            exactFacePrefixScratchBuffer_ == nullptr ||
            exactDescriptorScratchBuffer_ == nullptr ||
            exactOverflowCountScratchBuffer_ == nullptr ||
            exactOverflowEntryScratchBuffer_ == nullptr ||
            exactCompletionScratchBuffer_ == nullptr)
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
        if (exactCompletionScratchState_ != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            transition(exactCompletionScratchBuffer_.Get(),
                       exactCompletionScratchState_,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            exactCompletionScratchState_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (exactDescriptorScratchState_ != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        {
            transition(exactDescriptorScratchBuffer_.Get(),
                       exactDescriptorScratchState_,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            exactDescriptorScratchState_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }

        const std::array<std::uint32_t, 1> constants{batchBuildCount};
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
        commandList_->SetPipelineState(exactFaceEmitPipelineState_.Get());
        commandList_->SetComputeRootSignature(exactFaceEmitRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        commandList_->SetComputeRootShaderResourceView(1, emitConfigBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(2, buildRecordBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(3, pageMetadataBuffer->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(4, exactDescriptorScratchBuffer_->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(5, exactFaceCountScratchBuffer_->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(6, exactFaceDescriptorScratchBuffer_->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(7, exactFacePrefixScratchBuffer_->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(8, batchBuildIndicesAddress);
        commandList_->SetComputeRootUnorderedAccessView(9, exactOverflowCountScratchAddress(activeSubmissionSlotIndex_));
        commandList_->SetComputeRootUnorderedAccessView(10, exactOverflowEntryScratchAddress(activeSubmissionSlotIndex_));
        commandList_->SetComputeRootUnorderedAccessView(11, exactCompletionScratchBuffer_->GetGPUVirtualAddress());
        commandList_->Dispatch(kExactChunkPlaneCount, batchBuildCount, 1u);
        hasCommands_ = true;
    }

    [[nodiscard]] bool clearExactDrawRecordReservedFlags(ID3D12Resource* drawRecordBuffer,
                                                        std::span<const std::uint32_t> recordIndices)
    {
        if (!open_ ||
            exactDrawRecordClearPipelineState_ == nullptr ||
            exactDrawRecordClearRootSignature_ == nullptr ||
            drawRecordBuffer == nullptr ||
            recordIndices.empty())
        {
            return false;
        }

        ScratchAllocation indicesUpload =
            allocateUpload(static_cast<std::uint64_t>(recordIndices.size()) * sizeof(std::uint32_t),
                           alignof(std::uint32_t));
        if (indicesUpload.resource == nullptr || indicesUpload.cpuPtr == nullptr || indicesUpload.gpuAddress == 0)
        {
            return false;
        }

        std::memcpy(indicesUpload.cpuPtr,
                    recordIndices.data(),
                    recordIndices.size() * sizeof(std::uint32_t));

        const std::array<std::uint32_t, 4> constants{
            static_cast<std::uint32_t>(recordIndices.size()),
            0u,
            0u,
            0u};
        const UINT dispatchGroups = static_cast<UINT>((recordIndices.size() + 63u) / 64u);

        commandList_->SetPipelineState(exactDrawRecordClearPipelineState_.Get());
        commandList_->SetComputeRootSignature(exactDrawRecordClearRootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, static_cast<UINT>(constants.size()), constants.data(), 0);
        commandList_->SetComputeRootShaderResourceView(1, indicesUpload.gpuAddress);
        commandList_->SetComputeRootUnorderedAccessView(2, drawRecordBuffer->GetGPUVirtualAddress());
        commandList_->Dispatch(dispatchGroups, 1u, 1u);

        uavBarrier(drawRecordBuffer);
        hasCommands_ = true;
        return true;
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
                exactTimestampCursor_ > 1 &&
                activeSubmissionSlotIndex_ < kMaxInFlightSubmissionSlots)
            {
                const UINT queryOffset = activeSubmissionSlotIndex_ * kExactTimestampQueriesPerSubmission;
                const std::uint64_t readbackOffsetBytes =
                    static_cast<std::uint64_t>(queryOffset) * sizeof(std::uint64_t);
                commandList_->ResolveQueryData(exactTimestampQueryHeap_.Get(),
                                               D3D12_QUERY_TYPE_TIMESTAMP,
                                               queryOffset,
                                               exactTimestampCursor_,
                                               exactTimestampReadbackBuffer_.Get(),
                                               readbackOffsetBytes);
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
            result.submissionSlotIndex = activeSubmissionSlotIndex_;
            SubmissionSlot& slot = submissionSlots_[activeSubmissionSlotIndex_];
            slot.fenceValue = fenceValue_;
            if (exactTimingCaptureActive_ && exactTimestampCursor_ > 1)
            {
                slot.exactTimestampSubmittedCount = exactTimestampCursor_;
                slot.exactTimingPassCount = exactTimingPassCount_;
                for (UINT passIndex = 0; passIndex < exactTimingPassCount_; ++passIndex)
                {
                    slot.exactTimingPasses[passIndex] = exactTimingPassesCurrent_[passIndex];
                }
                slot.exactTimingPending = true;
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

        if (!hasCommands_ &&
            activeSubmissionSlotIndex_ != std::numeric_limits<std::uint32_t>::max())
        {
            releaseSubmissionSlot(activeSubmissionSlotIndex_);
        }
        open_ = false;
        hasCommands_ = false;
        exactTimingCaptureActive_ = false;
        activeSubmissionSlotIndex_ = std::numeric_limits<std::uint32_t>::max();
        return result;
    }

    [[nodiscard]] std::uint32_t currentSubmissionSlotIndex() const noexcept
    {
        return activeSubmissionSlotIndex_;
    }

    void releaseSubmissionSlot(std::uint32_t slotIndex)
    {
        if (slotIndex >= kMaxInFlightSubmissionSlots)
        {
            return;
        }

        SubmissionSlot& slot = submissionSlots_[slotIndex];
        slot.reserved = false;
        if (slot.fenceValue != 0 &&
            completedFenceValue() >= slot.fenceValue &&
            !slot.exactTimingPending)
        {
            slot.fenceValue = 0;
        }
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
               exactDescriptorGenRootSignature_ != nullptr &&
               exactPrepassRootSignature_ != nullptr &&
               exactFaceEmitRootSignature_ != nullptr &&
               exactDrawRecordClearRootSignature_ != nullptr &&
               atlasSeedCachePipelineState_ != nullptr && atlasSampleCachePipelineState_ != nullptr &&
                atlasUpdatePipelineState_ != nullptr && synthColumnPipelineState_ != nullptr &&
                stampPipelineState_ != nullptr &&
               faceCountPipelineState_ != nullptr &&
               facePrefixGroupPipelineState_ != nullptr &&
               facePrefixScanPipelineState_ != nullptr &&
               facePrefixAddPipelineState_ != nullptr &&
               faceEmitPipelineState_ != nullptr &&
               exactDescriptorGenPipelineState_ != nullptr &&
               exactSynthPipelineState_ != nullptr &&
               exactStampPipelineState_ != nullptr &&
               exactHaloCachePipelineState_ != nullptr &&
               exactLightPipelineState_ != nullptr &&
               exactSeamExportPipelineState_ != nullptr &&
               exactFaceCountPipelineState_ != nullptr &&
               exactFacePrefixPipelineState_ != nullptr &&
               exactFaceEmitPipelineState_ != nullptr &&
               exactDrawRecordClearPipelineState_ != nullptr &&
               exactDescriptorScratchBuffer_ != nullptr &&
               exactFaceCountScratchBuffer_ != nullptr &&
               exactFaceDescriptorScratchBuffer_ != nullptr &&
               exactFacePrefixScratchBuffer_ != nullptr &&
               exactFaceTotalScratchBuffer_ != nullptr &&
               exactOverflowCountScratchBuffer_ != nullptr &&
               exactOverflowEntryScratchBuffer_ != nullptr &&
               exactCompletionScratchBuffer_ != nullptr;
    }

    [[nodiscard]] UINT64 completedFenceValue() const noexcept
    {
        return (fence_ != nullptr) ? fence_->GetCompletedValue() : 0;
    }

    [[nodiscard]] UINT64 lastSubmittedFenceValue() const noexcept
    {
        return lastSubmittedFenceValue_;
    }

    [[nodiscard]] std::uint32_t inFlightSubmissionCount() const noexcept
    {
        const UINT64 completedValue = completedFenceValue();
        std::uint32_t count = 0u;
        for (const SubmissionSlot& slot : submissionSlots_)
        {
            if (slot.fenceValue != 0 && completedValue < slot.fenceValue)
            {
                ++count;
            }
        }
        return count;
    }

    [[nodiscard]] ID3D12Fence* fence() const noexcept
    {
        return fence_.Get();
    }

    [[nodiscard]] ExactPassTimings latestExactPassTimings()
    {
        bool foundCompletedTimings = false;
        ExactPassTimings completedTimings{};
        if (exactTimestampReadbackBuffer_ != nullptr)
        {
            const auto timestampMs = [this](std::uint64_t begin, std::uint64_t end) -> double
            {
                if (end <= begin || exactTimestampFrequency_ == 0)
                {
                    return 0.0;
                }
                return static_cast<double>(end - begin) * 1000.0 / static_cast<double>(exactTimestampFrequency_);
            };
            const UINT64 completedValue = completedFenceValue();
            for (std::uint32_t slotIndex = 0; slotIndex < kMaxInFlightSubmissionSlots; ++slotIndex)
            {
                SubmissionSlot& slot = submissionSlots_[slotIndex];
                if (!slot.exactTimingPending ||
                    slot.exactTimestampSubmittedCount < 2u ||
                    slot.fenceValue == 0 ||
                    completedValue < slot.fenceValue)
                {
                    continue;
                }

                std::array<std::uint64_t, kExactTimestampQueriesPerSubmission> timestamps{};
                void* mappedTimestamps = nullptr;
                const std::uint64_t slotOffsetBytes =
                    static_cast<std::uint64_t>(slotIndex) *
                    static_cast<std::uint64_t>(kExactTimestampQueriesPerSubmission) *
                    sizeof(std::uint64_t);
                const D3D12_RANGE timestampRange{
                    static_cast<SIZE_T>(slotOffsetBytes),
                    static_cast<SIZE_T>(slotOffsetBytes +
                                        static_cast<std::uint64_t>(slot.exactTimestampSubmittedCount) * sizeof(std::uint64_t))};
                throwIfFailedDx(exactTimestampReadbackBuffer_->Map(0, &timestampRange, &mappedTimestamps),
                                "failed to map exact compute timestamp readback");
                std::memcpy(timestamps.data(),
                            static_cast<const std::byte*>(mappedTimestamps) + slotOffsetBytes,
                            static_cast<std::size_t>(slot.exactTimestampSubmittedCount) * sizeof(std::uint64_t));
                exactTimestampReadbackBuffer_->Unmap(0, nullptr);

                for (UINT passIndex = 0; passIndex < slot.exactTimingPassCount; ++passIndex)
                {
                    const UINT queryBase = passIndex * kExactTimestampQueriesPerPass;
                    if (queryBase + 1u >= slot.exactTimestampSubmittedCount)
                    {
                        break;
                    }

                    const double passMs = timestampMs(timestamps[queryBase], timestamps[queryBase + 1u]);
                    switch (slot.exactTimingPasses[passIndex])
                    {
                    case ExactTimingPass::Synth:
                        completedTimings.synthMs += passMs;
                        break;
                    case ExactTimingPass::Stamp:
                        completedTimings.stampMs += passMs;
                        break;
                    case ExactTimingPass::Light:
                        completedTimings.lightMs += passMs;
                        break;
                    case ExactTimingPass::FaceCount:
                        completedTimings.faceCountMs += passMs;
                        break;
                    case ExactTimingPass::FacePrefix:
                        completedTimings.facePrefixMs += passMs;
                        break;
                    case ExactTimingPass::Allocate:
                        completedTimings.allocateMs += passMs;
                        break;
                    case ExactTimingPass::FaceEmit:
                        completedTimings.faceEmitMs += passMs;
                        break;
                    case ExactTimingPass::Count:
                    default:
                        break;
                    }
                }

                slot.exactTimingPending = false;
                slot.exactTimestampSubmittedCount = 0;
                slot.exactTimingPassCount = 0;
                if (!slot.reserved && completedValue >= slot.fenceValue)
                {
                    slot.fenceValue = 0;
                }
                foundCompletedTimings = true;
            }
            completedTimings.totalMs =
                completedTimings.synthMs +
                completedTimings.stampMs +
                completedTimings.lightMs +
                completedTimings.faceCountMs +
                completedTimings.facePrefixMs +
                completedTimings.allocateMs +
                completedTimings.faceEmitMs;
        }

        if (foundCompletedTimings)
        {
            exactLastCompletedTimings_ = completedTimings;
        }

        return exactLastCompletedTimings_;
    }

    [[nodiscard]] std::size_t uploadScratchSizeBytes() const noexcept
    {
        return static_cast<std::size_t>(kUploadScratchSizeBytes);
    }

    [[nodiscard]] std::size_t uploadScratchBytesPerSubmission() const noexcept
    {
        return static_cast<std::size_t>(kUploadScratchBytesPerSubmission);
    }

    [[nodiscard]] std::size_t readbackScratchSizeBytes() const noexcept
    {
        return readbackEnabled_ ? static_cast<std::size_t>(kReadbackScratchSizeBytes) : 0u;
    }

    [[nodiscard]] std::size_t readbackScratchBytesPerSubmission() const noexcept
    {
        return readbackEnabled_ ? static_cast<std::size_t>(kReadbackScratchBytesPerSubmission) : 0u;
    }

    [[nodiscard]] std::size_t exactScratchSizeBytes() const noexcept
    {
        const std::size_t descriptorScratchBytes =
            static_cast<std::size_t>(kMaxExactGpuBuildBatches) * kExactDescriptorScratchSliceBytes;
        return static_cast<std::size_t>(kMaxExactGpuBuildBatches) *
                   (static_cast<std::size_t>(kExactFaceCountScratchSliceBytes) +
                    static_cast<std::size_t>(kExactFaceDescriptorScratchSliceBytes) +
                    static_cast<std::size_t>(kExactFacePrefixScratchSliceBytes) +
               static_cast<std::size_t>(kExactFaceTotalScratchSliceBytes)) +
               descriptorScratchBytes +
               static_cast<std::size_t>(kMaxExactGpuBuildBatches) * sizeof(ExactCompletionEntry) +
               static_cast<std::size_t>(kMaxInFlightSubmissionSlots) * sizeof(std::uint32_t) +
               static_cast<std::size_t>(kMaxInFlightSubmissionSlots) *
                   static_cast<std::size_t>(kMaxExactGpuBuildBatches) * sizeof(ExactOverflowEntry);
    }

    [[nodiscard]] std::size_t maxExactGpuBuildBatches() const noexcept
    {
        return static_cast<std::size_t>(kMaxExactGpuBuildBatches);
    }

    [[nodiscard]] std::size_t maxInFlightSubmissionSlots() const noexcept
    {
        return static_cast<std::size_t>(kMaxInFlightSubmissionSlots);
    }

    [[nodiscard]] std::size_t exactCompletionReadbackBytes(std::size_t buildCount) const noexcept
    {
        return buildCount * sizeof(ExactCompletionEntry);
    }

    [[nodiscard]] std::size_t exactFaceTotalsReadbackBytes(std::size_t buildCount) const noexcept
    {
        return buildCount * sizeof(std::uint32_t);
    }

    [[nodiscard]] const std::byte* readbackMappedData() const noexcept
    {
        return readbackScratchMapped_;
    }

    [[nodiscard]] UINT allocatePersistentDescriptorRange(UINT descriptorCount)
    {
        if (persistentDescriptorCursor_ + descriptorCount > kDescriptorHeapPersistentDescriptorCount)
        {
            throw std::runtime_error("far lod compute persistent descriptor heap exhausted");
        }

        const UINT baseIndex = persistentDescriptorCursor_;
        persistentDescriptorCursor_ += descriptorCount;
        return baseIndex;
    }

    void writePersistentStructuredSrvDescriptor(std::uint32_t descriptorIndex,
                                                ID3D12Resource* resource,
                                                std::uint64_t byteOffset,
                                                std::uint32_t elementCount,
                                                std::uint32_t strideBytes)
    {
        writeStructuredSrvDescriptor(descriptorIndex, resource, byteOffset, elementCount, strideBytes);
    }

    void writePersistentStructuredUavDescriptor(std::uint32_t descriptorIndex,
                                                ID3D12Resource* resource,
                                                std::uint64_t byteOffset,
                                                std::uint32_t elementCount,
                                                std::uint32_t strideBytes)
    {
        writeStructuredUavDescriptor(descriptorIndex, resource, byteOffset, elementCount, strideBytes);
    }

    struct ExactBuildInputDescriptorIndices
    {
        std::uint32_t centerVoxelSrvDescriptorIndex{kInvalidDescriptorIndex};
        std::uint32_t haloSrvDescriptorIndex{kInvalidDescriptorIndex};
    };

    struct ExactPrepassDescriptorIndices
    {
        std::uint32_t centerVoxelSrvDescriptorIndex{kInvalidDescriptorIndex};
        std::uint32_t centerVoxelUavDescriptorIndex{kInvalidDescriptorIndex};
        std::uint32_t haloVoxelSrvDescriptorIndex{kInvalidDescriptorIndex};
        std::uint32_t haloVoxelUavDescriptorIndex{kInvalidDescriptorIndex};
        std::uint32_t lightScratchVoxelSrvDescriptorIndex{kInvalidDescriptorIndex};
        std::uint32_t lightScratchVoxelUavDescriptorIndex{kInvalidDescriptorIndex};
        std::uint32_t sparseVoxelSrvDescriptorIndex{kInvalidDescriptorIndex};
        std::uint32_t seamVoxelUavDescriptorIndex{kInvalidDescriptorIndex};
        std::array<std::uint32_t, 6> neighborSeamSrvDescriptorIndices{
            kInvalidDescriptorIndex,
            kInvalidDescriptorIndex,
            kInvalidDescriptorIndex,
            kInvalidDescriptorIndex,
            kInvalidDescriptorIndex,
            kInvalidDescriptorIndex};

        [[nodiscard]] bool valid() const noexcept
        {
            return centerVoxelSrvDescriptorIndex != kInvalidDescriptorIndex &&
                   centerVoxelUavDescriptorIndex != kInvalidDescriptorIndex &&
                   haloVoxelSrvDescriptorIndex != kInvalidDescriptorIndex &&
                   haloVoxelUavDescriptorIndex != kInvalidDescriptorIndex &&
                   lightScratchVoxelSrvDescriptorIndex != kInvalidDescriptorIndex &&
                   lightScratchVoxelUavDescriptorIndex != kInvalidDescriptorIndex &&
                   seamVoxelUavDescriptorIndex != kInvalidDescriptorIndex;
        }
    };

    [[nodiscard]] ExactBuildInputDescriptorIndices writeExactBuildInputDescriptors(
        ID3D12Resource* centerVoxelBuffer,
        ID3D12Resource* haloBuffer)
    {
        ExactBuildInputDescriptorIndices indices{};
        if (centerVoxelBuffer == nullptr || haloBuffer == nullptr)
        {
            return indices;
        }

        const UINT descriptorIndex = allocateDescriptorRange(kExactEmitInputDescriptorCountPerBuild);
        writeStructuredSrvDescriptor(descriptorIndex,
                                     centerVoxelBuffer,
                                     0,
                                     kExactChunkVoxelCount,
                                     kExactChunkPackedVoxelStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 1u,
                                     haloBuffer,
                                     0,
                                     kExactChunkHaloVoxelCount,
                                     kExactChunkPackedVoxelStrideBytes);
        indices.centerVoxelSrvDescriptorIndex = descriptorIndex;
        indices.haloSrvDescriptorIndex = descriptorIndex + 1u;
        return indices;
    }

    [[nodiscard]] ExactPrepassDescriptorIndices writeExactPrepassInputDescriptors(
        ID3D12Resource* centerVoxelBuffer,
        ID3D12Resource* haloBuffer,
        ID3D12Resource* lightScratchBuffer,
        ID3D12Resource* seamBuffer,
        ID3D12Resource* sparseVoxelBuffer,
        std::uint32_t sparseVoxelCount,
        const std::array<ID3D12Resource*, 6>& neighborSeamBuffers)
    {
        ExactPrepassDescriptorIndices indices{};
        if (centerVoxelBuffer == nullptr ||
            haloBuffer == nullptr ||
            lightScratchBuffer == nullptr ||
            seamBuffer == nullptr)
        {
            return indices;
        }
        if (sparseVoxelCount > 0u && sparseVoxelBuffer == nullptr)
        {
            return indices;
        }

        const UINT descriptorIndex = allocateDescriptorRange(kExactPrepassDescriptorCountPerBuild);
        writeStructuredSrvDescriptor(descriptorIndex + 0u,
                                     centerVoxelBuffer,
                                     0,
                                     kExactChunkVoxelCount,
                                     kExactChunkPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 1u,
                                     centerVoxelBuffer,
                                     0,
                                     kExactChunkVoxelCount,
                                     kExactChunkPackedVoxelStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 2u,
                                     haloBuffer,
                                     0,
                                     kExactChunkHaloVoxelCount,
                                     kExactChunkPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 3u,
                                     haloBuffer,
                                     0,
                                     kExactChunkHaloVoxelCount,
                                     kExactChunkPackedVoxelStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 4u,
                                     lightScratchBuffer,
                                     0,
                                     kExactChunkVoxelCount,
                                     kExactChunkPackedVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 5u,
                                     lightScratchBuffer,
                                     0,
                                     kExactChunkVoxelCount,
                                     kExactChunkPackedVoxelStrideBytes);
        writeStructuredSrvDescriptor(descriptorIndex + 6u,
                                     sparseVoxelBuffer,
                                     0,
                                     std::max(sparseVoxelCount, 1u),
                                     kExactChunkSparseVoxelStrideBytes);
        writeStructuredUavDescriptor(descriptorIndex + 7u,
                                     seamBuffer,
                                     0,
                                     kExactChunkHaloVoxelCount,
                                     kExactChunkPackedVoxelStrideBytes);
        for (std::uint32_t neighborIndex = 0; neighborIndex < static_cast<std::uint32_t>(neighborSeamBuffers.size());
             ++neighborIndex)
        {
            writeStructuredSrvDescriptor(descriptorIndex + 8u + neighborIndex,
                                         neighborSeamBuffers[neighborIndex],
                                         0,
                                         kExactChunkHaloVoxelCount,
                                         kExactChunkPackedVoxelStrideBytes);
        }

        indices.centerVoxelSrvDescriptorIndex = descriptorIndex + 0u;
        indices.centerVoxelUavDescriptorIndex = descriptorIndex + 1u;
        indices.haloVoxelSrvDescriptorIndex = descriptorIndex + 2u;
        indices.haloVoxelUavDescriptorIndex = descriptorIndex + 3u;
        indices.lightScratchVoxelSrvDescriptorIndex = descriptorIndex + 4u;
        indices.lightScratchVoxelUavDescriptorIndex = descriptorIndex + 5u;
        indices.sparseVoxelSrvDescriptorIndex = descriptorIndex + 6u;
        indices.seamVoxelUavDescriptorIndex = descriptorIndex + 7u;
        for (std::uint32_t neighborIndex = 0;
             neighborIndex < static_cast<std::uint32_t>(indices.neighborSeamSrvDescriptorIndices.size());
             ++neighborIndex)
        {
            indices.neighborSeamSrvDescriptorIndices[neighborIndex] = descriptorIndex + 8u + neighborIndex;
        }
        return indices;
    }

    [[nodiscard]] std::uint32_t descriptorCapacityPerSubmission() const noexcept
    {
        return kDescriptorHeapDescriptorsPerSubmission;
    }

    [[nodiscard]] std::uint32_t exactPrepassDescriptorCountPerBuild() const noexcept
    {
        return kExactPrepassDescriptorCountPerBuild;
    }

    [[nodiscard]] std::uint32_t exactEmitInputDescriptorCountPerBuild() const noexcept
    {
        return kExactEmitInputDescriptorCountPerBuild;
    }

private:
    [[nodiscard]] std::uint32_t acquireSubmissionSlot(UINT64 completedFenceValue) noexcept
    {
        for (std::uint32_t attempt = 0; attempt < kMaxInFlightSubmissionSlots; ++attempt)
        {
            const std::uint32_t slotIndex = (submissionSlotCursor_ + attempt) % kMaxInFlightSubmissionSlots;
            SubmissionSlot& slot = submissionSlots_[slotIndex];
            // Never reuse allocators/command lists still in-flight on the GPU.
            if (slot.fenceValue != 0 && completedFenceValue < slot.fenceValue)
            {
                continue;
            }
            if (slot.reserved)
            {
                if (slot.fenceValue != 0 && completedFenceValue >= slot.fenceValue)
                {
                    slot.fenceValue = 0;
                }
                continue;
            }
            if (slot.exactTimingPending)
            {
                continue;
            }

            slot.reserved = true;
            submissionSlotCursor_ = (slotIndex + 1u) % kMaxInFlightSubmissionSlots;
            return slotIndex;
        }

        return std::numeric_limits<std::uint32_t>::max();
    }

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
    static constexpr std::uint32_t kExactChunkHaloFaceCount = 6u;
    static constexpr std::uint32_t kExactChunkHaloFaceVoxelCount = kExactChunkSize * kExactChunkSize;
    static constexpr std::uint32_t kExactChunkHaloVoxelCount =
        kExactChunkHaloFaceCount * kExactChunkHaloFaceVoxelCount;
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
    static constexpr std::uint32_t kExactChunkFaceDescriptorStrideBytes = 32u;
    static constexpr std::uint32_t kWorldgenPageSize = 64u;
    static constexpr std::uint32_t kWorldgenPageColumnCount = kWorldgenPageSize * kWorldgenPageSize;
    static constexpr std::uint32_t kExactWorldgenPageColumnStrideBytes = 32u;
    static constexpr std::uint32_t kExactDescriptorBuildParamsStrideBytes = 48u;
    static constexpr std::uint32_t kExactDescriptorScratchSliceBytes =
        kExactChunkColumnCount * kExactChunkColumnDescriptorStrideBytes;
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
    static constexpr std::uint32_t kMaxExactGpuBuildBatches = 1024u;
    static constexpr std::uint32_t kMaxInFlightSubmissionSlots = 4u;
    static constexpr UINT kExactTimestampPassCount = static_cast<UINT>(ExactTimingPass::Count);
    static constexpr UINT kExactTimestampQueriesPerPass = 2u;
    static constexpr UINT kExactTimestampQueriesPerSubmission =
        kExactTimestampPassCount * kExactTimestampQueriesPerPass;
    static constexpr UINT kExactTimestampQueryCount =
        kExactTimestampQueriesPerSubmission * kMaxInFlightSubmissionSlots;
    // Exact GPU pages consume three persistent UAV descriptors each and pages are not
    // reopened once they become resident. Full exact-bubble fills can therefore require
    // substantially more persistent descriptors than the old 4096-entry budget allowed.
    static constexpr UINT kDescriptorHeapPersistentDescriptorCount = 16384u;
    static constexpr UINT kDescriptorHeapSubmissionDescriptorCount = 8192u;
    static constexpr UINT kDescriptorHeapDescriptorCount =
        kDescriptorHeapPersistentDescriptorCount + kDescriptorHeapSubmissionDescriptorCount;
    static constexpr std::uint64_t kUploadScratchSizeBytes = 16ull * 1024ull * 1024ull;
    static constexpr std::uint64_t kReadbackScratchSizeBytes = 4ull * 1024ull * 1024ull;
    static constexpr std::uint64_t kUploadScratchBytesPerSubmission =
        kUploadScratchSizeBytes / kMaxInFlightSubmissionSlots;
    static constexpr std::uint64_t kReadbackScratchBytesPerSubmission =
        kReadbackScratchSizeBytes / kMaxInFlightSubmissionSlots;
    static constexpr UINT kDescriptorHeapDescriptorsPerSubmission =
        kDescriptorHeapSubmissionDescriptorCount / kMaxInFlightSubmissionSlots;
    static constexpr UINT kExactPrepassDescriptorCountPerBuild = 14u;
    static constexpr UINT kExactEmitInputDescriptorCountPerBuild = 2u;

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
        exactDescriptorScratchBuffer_ = createDefaultBuffer(
            device_.Get(),
            static_cast<std::uint64_t>(kMaxExactGpuBuildBatches) * kExactDescriptorScratchSliceBytes,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        exactOverflowCountScratchBuffer_ = createDefaultBuffer(device_.Get(),
                                                               static_cast<std::uint64_t>(kMaxInFlightSubmissionSlots) *
                                                                   kExactChunkPackedVoxelStrideBytes,
                                                               D3D12_RESOURCE_STATE_COMMON,
                                                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        exactOverflowEntryScratchBuffer_ = createDefaultBuffer(
            device_.Get(),
            static_cast<std::uint64_t>(kMaxInFlightSubmissionSlots) *
                static_cast<std::uint64_t>(kMaxExactGpuBuildBatches) * sizeof(ExactOverflowEntry),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        exactCompletionScratchBuffer_ = createDefaultBuffer(
            device_.Get(),
            static_cast<std::uint64_t>(kMaxExactGpuBuildBatches) * sizeof(ExactCompletionEntry),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        setDebugObjectName(exactFaceCountScratchBuffer_.Get(), L"ExactChunkFaceCountScratch");
        setDebugObjectName(exactFaceDescriptorScratchBuffer_.Get(), L"ExactChunkFaceDescriptorScratch");
        setDebugObjectName(exactFacePrefixScratchBuffer_.Get(), L"ExactChunkFacePrefixScratch");
        setDebugObjectName(exactFaceTotalScratchBuffer_.Get(), L"ExactChunkFaceTotalScratch");
        setDebugObjectName(exactDescriptorScratchBuffer_.Get(), L"ExactChunkDescriptorScratch");
        setDebugObjectName(exactOverflowCountScratchBuffer_.Get(), L"ExactChunkOverflowCountScratch");
        setDebugObjectName(exactOverflowEntryScratchBuffer_.Get(), L"ExactChunkOverflowEntryScratch");
        setDebugObjectName(exactCompletionScratchBuffer_.Get(), L"ExactChunkCompletionScratch");
        exactFaceCountScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        exactFaceDescriptorScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        exactFacePrefixScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        exactFaceTotalScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        exactDescriptorScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        exactOverflowCountScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        exactOverflowEntryScratchState_ = D3D12_RESOURCE_STATE_COMMON;
        exactCompletionScratchState_ = D3D12_RESOURCE_STATE_COMMON;
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
        if (descriptorCursor_ + descriptorCount > kDescriptorHeapDescriptorsPerSubmission)
        {
            throw std::runtime_error("far lod compute descriptor heap exhausted");
        }
        const UINT baseIndex = kDescriptorHeapPersistentDescriptorCount +
                               activeSubmissionSlotIndex_ * kDescriptorHeapDescriptorsPerSubmission +
                               descriptorCursor_;
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
        exactDescriptorGenShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_descriptor_gen_cs.hlsl").string(), "ExactChunkDescriptorGenMain", "cs_6_6");
        exactSynthShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_synth_cs.hlsl").string(), "ExactChunkSynthMain", "cs_6_6");
        exactStampShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_structure_stamp_cs.hlsl").string(), "ExactChunkStructureStampMain", "cs_6_6");
        exactHaloCacheShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_halo_cache_cs.hlsl").string(), "ExactChunkHaloCacheMain", "cs_6_6");
        exactLightShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_light_cs.hlsl").string(), "ExactChunkLightMain", "cs_6_6");
        exactSeamExportShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_seam_export_cs.hlsl").string(), "ExactChunkSeamExportMain", "cs_6_6");
        exactFaceCountShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_face_count_cs.hlsl").string(), "ExactChunkFaceCountMain", "cs_6_6");
        exactFacePrefixShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_face_prefix_cs.hlsl").string(), "ExactChunkFacePrefixMain", "cs_6_6");
        exactFaceEmitShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_face_emit_cs.hlsl").string(), "ExactChunkFaceEmitMain", "cs_6_6");
        exactDrawRecordClearShader_ =
            loadShaderBytecodeLocal((shaderRoot / "exact_chunk_draw_record_clear_cs.hlsl").string(), "ExactChunkDrawRecordClearMain", "cs_6_6");
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

        std::array<D3D12_ROOT_PARAMETER, 5> exactDescriptorGenParams{};
        exactDescriptorGenParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        exactDescriptorGenParams[0].Constants.ShaderRegister = 0;
        exactDescriptorGenParams[0].Constants.Num32BitValues = 4;
        for (UINT parameterIndex = 1; parameterIndex <= 3; ++parameterIndex)
        {
            exactDescriptorGenParams[parameterIndex].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            exactDescriptorGenParams[parameterIndex].Descriptor.ShaderRegister = parameterIndex - 1;
        }
        exactDescriptorGenParams[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        exactDescriptorGenParams[4].Descriptor.ShaderRegister = 0;
        D3D12_ROOT_SIGNATURE_DESC exactDescriptorGenDesc{};
        exactDescriptorGenDesc.NumParameters = static_cast<UINT>(exactDescriptorGenParams.size());
        exactDescriptorGenDesc.pParameters = exactDescriptorGenParams.data();
        createRootSignature(exactDescriptorGenDesc,
                            exactDescriptorGenRootSignature_,
                            "exact chunk descriptor gen root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactDescriptorGenPso{};
        exactDescriptorGenPso.pRootSignature = exactDescriptorGenRootSignature_.Get();
        exactDescriptorGenPso.CS = {exactDescriptorGenShader_->GetBufferPointer(),
                                    exactDescriptorGenShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactDescriptorGenPso,
                                                            IID_PPV_ARGS(&exactDescriptorGenPipelineState_)),
                        "failed to create exact chunk descriptor gen pipeline");

        std::array<D3D12_ROOT_PARAMETER, 7> exactPrepassParams{};
        exactPrepassParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        exactPrepassParams[0].Constants.ShaderRegister = 0;
        exactPrepassParams[0].Constants.Num32BitValues = 4;
        exactPrepassParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        exactPrepassParams[1].Descriptor.ShaderRegister = 0;
        exactPrepassParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        exactPrepassParams[2].Descriptor.ShaderRegister = 1;
        for (UINT parameterIndex = 3; parameterIndex <= 6; ++parameterIndex)
        {
            exactPrepassParams[parameterIndex].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            exactPrepassParams[parameterIndex].Descriptor.ShaderRegister = parameterIndex - 3;
        }
        D3D12_ROOT_SIGNATURE_DESC exactPrepassDesc{};
        exactPrepassDesc.NumParameters = static_cast<UINT>(exactPrepassParams.size());
        exactPrepassDesc.pParameters = exactPrepassParams.data();
        exactPrepassDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
        createRootSignature(exactPrepassDesc, exactPrepassRootSignature_, "exact chunk prepass root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactSynthPso{};
        exactSynthPso.pRootSignature = exactPrepassRootSignature_.Get();
        exactSynthPso.CS = {exactSynthShader_->GetBufferPointer(), exactSynthShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactSynthPso, IID_PPV_ARGS(&exactSynthPipelineState_)),
                        "failed to create exact chunk synth pipeline");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactStampPso{};
        exactStampPso.pRootSignature = exactPrepassRootSignature_.Get();
        exactStampPso.CS = {exactStampShader_->GetBufferPointer(), exactStampShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactStampPso, IID_PPV_ARGS(&exactStampPipelineState_)),
                        "failed to create exact chunk stamp pipeline");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactHaloCachePso{};
        exactHaloCachePso.pRootSignature = exactPrepassRootSignature_.Get();
        exactHaloCachePso.CS = {exactHaloCacheShader_->GetBufferPointer(), exactHaloCacheShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactHaloCachePso,
                                                            IID_PPV_ARGS(&exactHaloCachePipelineState_)),
                        "failed to create exact chunk halo cache pipeline");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactLightPso{};
        exactLightPso.pRootSignature = exactPrepassRootSignature_.Get();
        exactLightPso.CS = {exactLightShader_->GetBufferPointer(), exactLightShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactLightPso,
                                                            IID_PPV_ARGS(&exactLightPipelineState_)),
                        "failed to create exact chunk light pipeline");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactSeamExportPso{};
        exactSeamExportPso.pRootSignature = exactPrepassRootSignature_.Get();
        exactSeamExportPso.CS = {exactSeamExportShader_->GetBufferPointer(), exactSeamExportShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactSeamExportPso,
                                                            IID_PPV_ARGS(&exactSeamExportPipelineState_)),
                        "failed to create exact chunk seam export pipeline");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactFaceCountPso{};
        exactFaceCountPso.pRootSignature = exactPrepassRootSignature_.Get();
        exactFaceCountPso.CS = {exactFaceCountShader_->GetBufferPointer(), exactFaceCountShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactFaceCountPso,
                                                            IID_PPV_ARGS(&exactFaceCountPipelineState_)),
                        "failed to create exact chunk face count pipeline");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactPrefixPso{};
        exactPrefixPso.pRootSignature = exactPrepassRootSignature_.Get();
        exactPrefixPso.CS = {exactFacePrefixShader_->GetBufferPointer(), exactFacePrefixShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactPrefixPso,
                                                            IID_PPV_ARGS(&exactFacePrefixPipelineState_)),
                        "failed to create exact chunk face prefix pipeline");

        std::array<D3D12_ROOT_PARAMETER, 12> exactFaceEmitParams{};
        exactFaceEmitParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        exactFaceEmitParams[0].Constants.ShaderRegister = 0;
        exactFaceEmitParams[0].Constants.Num32BitValues = 1;
        for (UINT parameterIndex = 1; parameterIndex <= 8; ++parameterIndex)
        {
            exactFaceEmitParams[parameterIndex].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            exactFaceEmitParams[parameterIndex].Descriptor.ShaderRegister = parameterIndex - 1;
        }
        for (UINT parameterIndex = 9; parameterIndex <= 11; ++parameterIndex)
        {
            exactFaceEmitParams[parameterIndex].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            exactFaceEmitParams[parameterIndex].Descriptor.ShaderRegister = parameterIndex - 9;
        }
        D3D12_ROOT_SIGNATURE_DESC exactFaceEmitDesc{};
        exactFaceEmitDesc.NumParameters = static_cast<UINT>(exactFaceEmitParams.size());
        exactFaceEmitDesc.pParameters = exactFaceEmitParams.data();
        exactFaceEmitDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
        createRootSignature(exactFaceEmitDesc, exactFaceEmitRootSignature_, "exact chunk face emit root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactFaceEmitPso{};
        exactFaceEmitPso.pRootSignature = exactFaceEmitRootSignature_.Get();
        exactFaceEmitPso.CS = {exactFaceEmitShader_->GetBufferPointer(), exactFaceEmitShader_->GetBufferSize()};
        throwIfFailedDx(device_->CreateComputePipelineState(&exactFaceEmitPso, IID_PPV_ARGS(&exactFaceEmitPipelineState_)),
                        "failed to create exact chunk face emit pipeline");

        std::array<D3D12_ROOT_PARAMETER, 3> exactDrawRecordClearParams{};
        exactDrawRecordClearParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        exactDrawRecordClearParams[0].Constants.ShaderRegister = 0;
        exactDrawRecordClearParams[0].Constants.Num32BitValues = 4;
        exactDrawRecordClearParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        exactDrawRecordClearParams[1].Descriptor.ShaderRegister = 0;
        exactDrawRecordClearParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        exactDrawRecordClearParams[2].Descriptor.ShaderRegister = 0;
        D3D12_ROOT_SIGNATURE_DESC exactDrawRecordClearDesc{};
        exactDrawRecordClearDesc.NumParameters = static_cast<UINT>(exactDrawRecordClearParams.size());
        exactDrawRecordClearDesc.pParameters = exactDrawRecordClearParams.data();
        createRootSignature(exactDrawRecordClearDesc,
                            exactDrawRecordClearRootSignature_,
                            "exact chunk draw record clear root signature");

        D3D12_COMPUTE_PIPELINE_STATE_DESC exactDrawRecordClearPso{};
        exactDrawRecordClearPso.pRootSignature = exactDrawRecordClearRootSignature_.Get();
        exactDrawRecordClearPso.CS = {exactDrawRecordClearShader_->GetBufferPointer(),
                                      exactDrawRecordClearShader_->GetBufferSize()};
        throwIfFailedDx(
            device_->CreateComputePipelineState(&exactDrawRecordClearPso,
                                                IID_PPV_ARGS(&exactDrawRecordClearPipelineState_)),
            "failed to create exact chunk draw record clear pipeline");

    }
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    std::array<SubmissionSlot, kMaxInFlightSubmissionSlots> submissionSlots_{};
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
    Microsoft::WRL::ComPtr<ID3DBlob> exactDescriptorGenShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactSynthShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactStampShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactHaloCacheShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactLightShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactSeamExportShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactFaceCountShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactFacePrefixShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactFaceEmitShader_;
    Microsoft::WRL::ComPtr<ID3DBlob> exactDrawRecordClearShader_;
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
    Microsoft::WRL::ComPtr<ID3D12RootSignature> exactDescriptorGenRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> exactPrepassRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> exactFaceEmitRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> exactDrawRecordClearRootSignature_;
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
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactDescriptorGenPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactSynthPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactStampPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactHaloCachePipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactLightPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactSeamExportPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactFaceCountPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactFacePrefixPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactFaceEmitPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> exactDrawRecordClearPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptorHeap_;
    Microsoft::WRL::ComPtr<ID3D12Resource> uploadScratch_;
    std::byte* uploadScratchMapped_{nullptr};
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackScratch_;
    std::byte* readbackScratchMapped_{nullptr};
    Microsoft::WRL::ComPtr<ID3D12Resource> exactFaceCountScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactFaceDescriptorScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactFacePrefixScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactFaceTotalScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactDescriptorScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactOverflowCountScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactOverflowEntryScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactCompletionScratchBuffer_;
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> exactTimestampQueryHeap_;
    Microsoft::WRL::ComPtr<ID3D12Resource> exactTimestampReadbackBuffer_;
    std::uint64_t uploadCursor_{0};
    std::uint64_t readbackCursor_{0};
    UINT descriptorSize_{0};
    UINT descriptorCursor_{0};
    UINT persistentDescriptorCursor_{0};
    std::uint32_t activeSubmissionSlotIndex_{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t submissionSlotCursor_{0};
    UINT exactTimestampCursor_{0};
    UINT exactTimingPassCount_{0};
    std::array<ExactTimingPass, static_cast<std::size_t>(ExactTimingPass::Count)> exactTimingPassesCurrent_{};
    D3D12_RESOURCE_STATES exactFaceCountScratchState_{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES exactFaceDescriptorScratchState_{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES exactFacePrefixScratchState_{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES exactFaceTotalScratchState_{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES exactDescriptorScratchState_{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES exactOverflowCountScratchState_{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES exactOverflowEntryScratchState_{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES exactCompletionScratchState_{D3D12_RESOURCE_STATE_COMMON};
    UINT64 exactTimestampFrequency_{0};
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



