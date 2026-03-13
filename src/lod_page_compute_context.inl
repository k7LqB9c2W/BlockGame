class PageComputeContext
{
public:
    struct Summary
    {
        int minSolidY{0};
        int maxSolidY{1};
        int maxWaterY{std::numeric_limits<int>::min()};
        int waterCells{0};
        int canopyCells{0};
        int trunkCells{0};
        int occupiedCells{0};
        int visibleFaceCount{0};
    };

    struct PassTimings
    {
        double synthesisMs{0.0};
        double stampMs{0.0};
        double faceBuildMs{0.0};
    };

    ~PageComputeContext()
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
                        "failed to create page compute queue");
        throwIfFailedDx(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator_)),
                        "failed to create page compute allocator");
        throwIfFailedDx(device_->CreateCommandList(0,
                                                   D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                   allocator_.Get(),
                                                   nullptr,
                                                   IID_PPV_ARGS(&commandList_)),
                        "failed to create page compute command list");
        throwIfFailedDx(commandList_->Close(), "failed to close initial page compute command list");
        throwIfFailedDx(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
                        "failed to create page compute fence");
        fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (fenceEvent_ == nullptr)
        {
            throw std::runtime_error("failed to create page compute fence event");
        }

        const std::filesystem::path shaderPath = std::filesystem::path("assets") / "shaders" / "lod_page_compute.hlsl";
        auto compileShader = [&](const char* entryPoint) -> Microsoft::WRL::ComPtr<ID3DBlob>
        {
            UINT compileFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
            compileFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
            compileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif

            Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
            Microsoft::WRL::ComPtr<ID3DBlob> errors;
            const std::wstring widePath = shaderPath.wstring();
            const HRESULT hr = D3DCompileFromFile(widePath.c_str(),
                                                  nullptr,
                                                  D3D_COMPILE_STANDARD_FILE_INCLUDE,
                                                  entryPoint,
                                                  "cs_5_0",
                                                  compileFlags,
                                                  0,
                                                  &bytecode,
                                                  &errors);
            if (FAILED(hr))
            {
                std::string message = "failed to compile lod page compute shader";
                if (errors != nullptr && errors->GetBufferPointer() != nullptr)
                {
                    message += ": ";
                    message.append(static_cast<const char*>(errors->GetBufferPointer()),
                                   errors->GetBufferSize());
                }
                throwIfFailedDx(hr, message.c_str());
            }
            return bytecode;
        };

        const Microsoft::WRL::ComPtr<ID3DBlob> synthShader = compileShader("SynthesizeMain");
        const Microsoft::WRL::ComPtr<ID3DBlob> stampShader = compileShader("StructureStampMain");
        const Microsoft::WRL::ComPtr<ID3DBlob> faceShader = compileShader("FaceMaskMain");

        std::array<D3D12_ROOT_PARAMETER, 6> rootParams{};
        rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParams[0].Constants.Num32BitValues = 8;
        rootParams[0].Constants.RegisterSpace = 0;
        rootParams[0].Constants.ShaderRegister = 0;
        rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rootParams[1].Descriptor.RegisterSpace = 0;
        rootParams[1].Descriptor.ShaderRegister = 0;
        rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rootParams[2].Descriptor.RegisterSpace = 0;
        rootParams[2].Descriptor.ShaderRegister = 1;
        rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rootParams[3].Descriptor.RegisterSpace = 0;
        rootParams[3].Descriptor.ShaderRegister = 0;
        rootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rootParams[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rootParams[4].Descriptor.RegisterSpace = 0;
        rootParams[4].Descriptor.ShaderRegister = 1;
        rootParams[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rootParams[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rootParams[5].Descriptor.RegisterSpace = 0;
        rootParams[5].Descriptor.ShaderRegister = 2;
        rootParams[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rootSigDesc{};
        rootSigDesc.NumParameters = static_cast<UINT>(rootParams.size());
        rootSigDesc.pParameters = rootParams.data();
        rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        Microsoft::WRL::ComPtr<ID3DBlob> rootSigBlob;
        Microsoft::WRL::ComPtr<ID3DBlob> rootSigErrorBlob;
        throwIfFailedDx(D3D12SerializeRootSignature(&rootSigDesc,
                                                    D3D_ROOT_SIGNATURE_VERSION_1,
                                                    &rootSigBlob,
                                                    &rootSigErrorBlob),
                        "failed to serialize page compute root signature");
        throwIfFailedDx(device_->CreateRootSignature(0,
                                                     rootSigBlob->GetBufferPointer(),
                                                     rootSigBlob->GetBufferSize(),
                                                     IID_PPV_ARGS(&rootSignature_)),
                        "failed to create page compute root signature");

        auto createPipeline = [this](ID3DBlob* shaderBlob, Microsoft::WRL::ComPtr<ID3D12PipelineState>& pipeline)
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
            psoDesc.pRootSignature = rootSignature_.Get();
            psoDesc.CS = {shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize()};
            throwIfFailedDx(device_->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipeline)),
                            "failed to create page compute pipeline");
        };
        createPipeline(synthShader.Get(), synthesizePipelineState_);
        createPipeline(stampShader.Get(), structureStampPipelineState_);
        createPipeline(faceShader.Get(), faceMaskPipelineState_);

        D3D12_QUERY_HEAP_DESC queryHeapDesc{};
        queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        queryHeapDesc.Count = kTimestampQueryCount;
        throwIfFailedDx(device_->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(&timestampQueryHeap_)),
                        "failed to create page compute timestamp query heap");
        timestampReadbackBuffer_ = createBuffer(D3D12_HEAP_TYPE_READBACK,
                                                static_cast<std::uint64_t>(kTimestampQueryCount) * sizeof(std::uint64_t),
                                                D3D12_RESOURCE_STATE_COPY_DEST,
                                                D3D12_RESOURCE_FLAG_NONE);
        throwIfFailedDx(queue_->GetTimestampFrequency(&timestampFrequency_),
                        "failed to query page compute timestamp frequency");
    }

    void shutdown()
    {
        waitForIdle();
        if (fenceEvent_ != nullptr)
        {
            CloseHandle(fenceEvent_);
            fenceEvent_ = nullptr;
        }
        timestampReadbackBuffer_.Reset();
        timestampQueryHeap_.Reset();
        faceMaskPipelineState_.Reset();
        structureStampPipelineState_.Reset();
        synthesizePipelineState_.Reset();
        rootSignature_.Reset();
        commandList_.Reset();
        allocator_.Reset();
        queue_.Reset();
        fence_.Reset();
        device_.Reset();
        fenceValue_ = 0;
        lastSubmittedFenceValue_ = 0;
        timestampFrequency_ = 0;
    }

    [[nodiscard]] bool ready() const noexcept
    {
        return device_ != nullptr &&
               queue_ != nullptr &&
               allocator_ != nullptr &&
               commandList_ != nullptr &&
               rootSignature_ != nullptr &&
               synthesizePipelineState_ != nullptr &&
               structureStampPipelineState_ != nullptr &&
               faceMaskPipelineState_ != nullptr;
    }

    template <typename PageType>
    bool processPage(PageType& page, Summary& outSummary, PassTimings& outTimings)
    {
        outSummary = {};
        outTimings = {};
        if (!ready() || page.cells.empty() || page.terrainColumns.empty())
        {
            return false;
        }

        struct GpuTerrainColumn
        {
            std::int32_t solidTopY;
            std::uint32_t solidBlock;
            std::int32_t waterTopY;
            std::uint32_t hasWater;
        };

        struct GpuStructureVoxel
        {
            std::int32_t worldX;
            std::int32_t worldY;
            std::int32_t worldZ;
            std::uint32_t block;
        };

        struct GpuPageCell
        {
            std::int32_t solidTopY;
            std::uint32_t solidBlock;
            std::int32_t waterTopY;
            std::uint32_t hasWater;
            std::int32_t canopyBaseY;
            std::int32_t canopyTopY;
            std::uint32_t canopyBlock;
            std::uint32_t hasCanopy;
            std::int32_t trunkTopY;
            std::uint32_t trunkBlock;
            std::uint32_t hasTrunk;
            std::uint32_t padding;
        };

        struct RootParams
        {
            UINT gridCount;
            INT worldMinX;
            INT worldMinY;
            INT worldMinZ;
            INT cellScaleBlocks;
            UINT cellCount;
            UINT structureVoxelCount;
            UINT padding;
        };

        constexpr int kSummaryValueCount = 8;
        const std::size_t cellCount = page.cells.size();
        std::vector<GpuTerrainColumn> gpuColumns(page.terrainColumns.size());
        for (std::size_t i = 0; i < page.terrainColumns.size(); ++i)
        {
            const auto& src = page.terrainColumns[i];
            gpuColumns[i] = GpuTerrainColumn{
                src.solidTopY,
                static_cast<std::uint32_t>(src.solidBlock),
                src.waterTopY,
                src.hasWater ? 1u : 0u};
        }

        std::vector<GpuStructureVoxel> gpuStructureVoxels;
        gpuStructureVoxels.reserve(page.structureVoxels.size());
        for (const auto& src : page.structureVoxels)
        {
            gpuStructureVoxels.push_back(GpuStructureVoxel{
                src.worldX,
                src.worldY,
                src.worldZ,
                static_cast<std::uint32_t>(src.block)});
        }

        std::vector<GpuPageCell> gpuCells(cellCount);
        std::vector<std::uint32_t> gpuFaceMasks(cellCount, 0u);
        std::array<std::int32_t, kSummaryValueCount> initialSummary{
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::min(),
            std::numeric_limits<int>::min(),
            0,
            0,
            0,
            0,
            0};

        const std::uint64_t inputBytes = static_cast<std::uint64_t>(gpuColumns.size() * sizeof(GpuTerrainColumn));
        const std::uint64_t structureBytes = static_cast<std::uint64_t>(
            std::max<std::size_t>(gpuStructureVoxels.size(), 1u) * sizeof(GpuStructureVoxel));
        const std::uint64_t cellBytes = static_cast<std::uint64_t>(gpuCells.size() * sizeof(GpuPageCell));
        const std::uint64_t faceBytes = static_cast<std::uint64_t>(gpuFaceMasks.size() * sizeof(std::uint32_t));
        const std::uint64_t summaryBytes = static_cast<std::uint64_t>(initialSummary.size() * sizeof(std::int32_t));

        Microsoft::WRL::ComPtr<ID3D12Resource> inputDefault =
            createBuffer(D3D12_HEAP_TYPE_DEFAULT, inputBytes, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
        Microsoft::WRL::ComPtr<ID3D12Resource> inputUpload =
            createBuffer(D3D12_HEAP_TYPE_UPLOAD, inputBytes, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
        Microsoft::WRL::ComPtr<ID3D12Resource> structureDefault =
            createBuffer(D3D12_HEAP_TYPE_DEFAULT, structureBytes, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
        Microsoft::WRL::ComPtr<ID3D12Resource> structureUpload =
            createBuffer(D3D12_HEAP_TYPE_UPLOAD, structureBytes, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
        Microsoft::WRL::ComPtr<ID3D12Resource> cellDefault =
            createBuffer(D3D12_HEAP_TYPE_DEFAULT, cellBytes, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Microsoft::WRL::ComPtr<ID3D12Resource> cellReadback =
            createBuffer(D3D12_HEAP_TYPE_READBACK, cellBytes, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
        Microsoft::WRL::ComPtr<ID3D12Resource> faceDefault =
            createBuffer(D3D12_HEAP_TYPE_DEFAULT, faceBytes, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Microsoft::WRL::ComPtr<ID3D12Resource> faceReadback =
            createBuffer(D3D12_HEAP_TYPE_READBACK, faceBytes, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
        Microsoft::WRL::ComPtr<ID3D12Resource> summaryDefault =
            createBuffer(D3D12_HEAP_TYPE_DEFAULT, summaryBytes, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Microsoft::WRL::ComPtr<ID3D12Resource> summaryUpload =
            createBuffer(D3D12_HEAP_TYPE_UPLOAD, summaryBytes, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
        Microsoft::WRL::ComPtr<ID3D12Resource> summaryReadback =
            createBuffer(D3D12_HEAP_TYPE_READBACK, summaryBytes, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);

        auto uploadBufferData = [](ID3D12Resource* uploadBuffer, const void* data, std::uint64_t byteCount, const char* message)
        {
            void* mapped = nullptr;
            throwIfFailedDx(uploadBuffer->Map(0, nullptr, &mapped), message);
            if (data != nullptr && byteCount > 0)
            {
                std::memcpy(mapped, data, static_cast<std::size_t>(byteCount));
            }
            uploadBuffer->Unmap(0, nullptr);
        };

        uploadBufferData(inputUpload.Get(), gpuColumns.data(), inputBytes, "failed to map terrain upload buffer");
        if (!gpuStructureVoxels.empty())
        {
            uploadBufferData(structureUpload.Get(),
                             gpuStructureVoxels.data(),
                             static_cast<std::uint64_t>(gpuStructureVoxels.size() * sizeof(GpuStructureVoxel)),
                             "failed to map structure upload buffer");
        }
        else
        {
            GpuStructureVoxel emptyVoxel{};
            uploadBufferData(structureUpload.Get(), &emptyVoxel, sizeof(emptyVoxel), "failed to map structure upload buffer");
        }
        uploadBufferData(summaryUpload.Get(), initialSummary.data(), summaryBytes, "failed to map summary upload buffer");

        if (fence_ != nullptr && fenceValue_ > 0 && fence_->GetCompletedValue() < fenceValue_)
        {
            waitForIdle();
        }

        throwIfFailedDx(allocator_->Reset(), "failed to reset page compute allocator");
        throwIfFailedDx(commandList_->Reset(allocator_.Get(), nullptr), "failed to reset page compute command list");

        commandList_->CopyBufferRegion(inputDefault.Get(), 0, inputUpload.Get(), 0, inputBytes);
        commandList_->CopyBufferRegion(structureDefault.Get(), 0, structureUpload.Get(), 0, structureBytes);
        commandList_->CopyBufferRegion(summaryDefault.Get(), 0, summaryUpload.Get(), 0, summaryBytes);

        const D3D12_RESOURCE_BARRIER copyBarriers[] = {
            transitionBarrier(inputDefault.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transitionBarrier(structureDefault.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transitionBarrier(summaryDefault.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
        commandList_->ResourceBarrier(static_cast<UINT>(std::size(copyBarriers)), copyBarriers);

        const RootParams rootParams{
            static_cast<UINT>(page.gridCount),
            static_cast<INT>(page.worldMinX),
            static_cast<INT>(page.worldMinY),
            static_cast<INT>(page.worldMinZ),
            static_cast<INT>(page.cellScaleBlocks),
            static_cast<UINT>(cellCount),
            static_cast<UINT>(gpuStructureVoxels.size()),
            0u};
        const UINT groups = std::max<UINT>(1u, static_cast<UINT>((cellCount + 63u) / 64u));

        commandList_->SetComputeRootSignature(rootSignature_.Get());
        commandList_->SetComputeRoot32BitConstants(0, 8, &rootParams, 0);
        commandList_->SetComputeRootShaderResourceView(1, inputDefault->GetGPUVirtualAddress());
        commandList_->SetComputeRootShaderResourceView(2, structureDefault->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(3, cellDefault->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(4, faceDefault->GetGPUVirtualAddress());
        commandList_->SetComputeRootUnorderedAccessView(5, summaryDefault->GetGPUVirtualAddress());

        commandList_->SetPipelineState(synthesizePipelineState_.Get());
        commandList_->EndQuery(timestampQueryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        commandList_->Dispatch(groups, 1, 1);
        commandList_->EndQuery(timestampQueryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);

        std::array<D3D12_RESOURCE_BARRIER, 3> synthBarriers{};
        synthBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        synthBarriers[0].UAV.pResource = cellDefault.Get();
        synthBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        synthBarriers[1].UAV.pResource = faceDefault.Get();
        synthBarriers[2].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        synthBarriers[2].UAV.pResource = summaryDefault.Get();
        commandList_->ResourceBarrier(static_cast<UINT>(synthBarriers.size()), synthBarriers.data());

        commandList_->SetPipelineState(structureStampPipelineState_.Get());
        commandList_->EndQuery(timestampQueryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
        commandList_->Dispatch(groups, 1, 1);
        commandList_->EndQuery(timestampQueryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);

        std::array<D3D12_RESOURCE_BARRIER, 2> stampBarriers{};
        stampBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        stampBarriers[0].UAV.pResource = cellDefault.Get();
        stampBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        stampBarriers[1].UAV.pResource = summaryDefault.Get();
        commandList_->ResourceBarrier(static_cast<UINT>(stampBarriers.size()), stampBarriers.data());

        commandList_->SetPipelineState(faceMaskPipelineState_.Get());
        commandList_->EndQuery(timestampQueryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 4);
        commandList_->Dispatch(groups, 1, 1);
        commandList_->EndQuery(timestampQueryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 5);

        std::array<D3D12_RESOURCE_BARRIER, 2> faceBarriers{};
        faceBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        faceBarriers[0].UAV.pResource = faceDefault.Get();
        faceBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        faceBarriers[1].UAV.pResource = summaryDefault.Get();
        commandList_->ResourceBarrier(static_cast<UINT>(faceBarriers.size()), faceBarriers.data());

        const D3D12_RESOURCE_BARRIER toCopyBarriers[] = {
            transitionBarrier(cellDefault.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transitionBarrier(faceDefault.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transitionBarrier(summaryDefault.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)};
        commandList_->ResourceBarrier(static_cast<UINT>(std::size(toCopyBarriers)), toCopyBarriers);
        commandList_->CopyBufferRegion(cellReadback.Get(), 0, cellDefault.Get(), 0, cellBytes);
        commandList_->CopyBufferRegion(faceReadback.Get(), 0, faceDefault.Get(), 0, faceBytes);
        commandList_->CopyBufferRegion(summaryReadback.Get(), 0, summaryDefault.Get(), 0, summaryBytes);
        commandList_->ResolveQueryData(timestampQueryHeap_.Get(),
                                       D3D12_QUERY_TYPE_TIMESTAMP,
                                       0,
                                       kTimestampQueryCount,
                                       timestampReadbackBuffer_.Get(),
                                       0);

        throwIfFailedDx(commandList_->Close(), "failed to close page compute command list");
        ID3D12CommandList* commandLists[] = {commandList_.Get()};
        queue_->ExecuteCommandLists(static_cast<UINT>(std::size(commandLists)), commandLists);
        ++fenceValue_;
        throwIfFailedDx(queue_->Signal(fence_.Get(), fenceValue_), "failed to signal page compute fence");
        lastSubmittedFenceValue_ = fenceValue_;
        waitForIdle();

        std::array<std::int32_t, kSummaryValueCount> summaryValues{};
        void* mappedSummary = nullptr;
        D3D12_RANGE summaryRange{0, static_cast<SIZE_T>(summaryBytes)};
        throwIfFailedDx(summaryReadback->Map(0, &summaryRange, &mappedSummary), "failed to map page compute summary readback");
        std::memcpy(summaryValues.data(), mappedSummary, static_cast<std::size_t>(summaryBytes));
        summaryReadback->Unmap(0, nullptr);

        void* mappedCells = nullptr;
        D3D12_RANGE cellRange{0, static_cast<SIZE_T>(cellBytes)};
        throwIfFailedDx(cellReadback->Map(0, &cellRange, &mappedCells), "failed to map page compute cell readback");
        std::memcpy(gpuCells.data(), mappedCells, static_cast<std::size_t>(cellBytes));
        cellReadback->Unmap(0, nullptr);

        void* mappedFaces = nullptr;
        D3D12_RANGE faceRange{0, static_cast<SIZE_T>(faceBytes)};
        throwIfFailedDx(faceReadback->Map(0, &faceRange, &mappedFaces), "failed to map page compute face readback");
        std::memcpy(gpuFaceMasks.data(), mappedFaces, static_cast<std::size_t>(faceBytes));
        faceReadback->Unmap(0, nullptr);

        std::array<std::uint64_t, kTimestampQueryCount> timestamps{};
        if (timestampReadbackBuffer_ != nullptr)
        {
            void* mappedTimestamps = nullptr;
            D3D12_RANGE timestampRange{0, static_cast<SIZE_T>(timestamps.size() * sizeof(std::uint64_t))};
            throwIfFailedDx(timestampReadbackBuffer_->Map(0, &timestampRange, &mappedTimestamps),
                            "failed to map page compute timestamp readback");
            std::memcpy(timestamps.data(), mappedTimestamps, timestamps.size() * sizeof(std::uint64_t));
            timestampReadbackBuffer_->Unmap(0, nullptr);
        }

        auto cellOccupied = [](const GpuPageCell& cell) noexcept
        {
            const bool hasSolid = cell.solidBlock != 0u && cell.solidTopY != std::numeric_limits<int>::min();
            const bool hasWater = cell.hasWater != 0u && cell.waterTopY != std::numeric_limits<int>::min();
            const bool hasCanopy = cell.hasCanopy != 0u &&
                                   cell.canopyBlock != 0u &&
                                   cell.canopyBaseY != std::numeric_limits<int>::min() &&
                                   cell.canopyTopY != std::numeric_limits<int>::min();
            const bool hasTrunk = cell.hasTrunk != 0u &&
                                  cell.trunkBlock != 0u &&
                                  cell.trunkTopY != std::numeric_limits<int>::min();
            return hasSolid || hasWater || hasCanopy || hasTrunk;
        };

        page.minY = std::numeric_limits<int>::max();
        page.maxY = std::numeric_limits<int>::min();
        page.faceMasks.assign(cellCount, 0);
        page.faceMasksReady = true;
        for (std::size_t i = 0; i < cellCount; ++i)
        {
            auto& dst = page.cells[i];
            const GpuPageCell& src = gpuCells[i];
            dst.solidTopY = src.solidTopY;
            dst.solidBlock = static_cast<BlockId>(src.solidBlock);
            dst.waterTopY = src.waterTopY;
            dst.hasWater = src.hasWater != 0;
            dst.canopyBaseY = src.canopyBaseY;
            dst.canopyTopY = src.canopyTopY;
            dst.canopyBlock = static_cast<BlockId>(src.canopyBlock);
            dst.hasCanopy = src.hasCanopy != 0;
            dst.trunkTopY = src.trunkTopY;
            dst.trunkBlock = static_cast<BlockId>(src.trunkBlock);
            dst.hasTrunk = src.hasTrunk != 0;
            page.faceMasks[i] = static_cast<std::uint8_t>(gpuFaceMasks[i] & 0x3Fu);

            if (cellOccupied(src))
            {
                const int localY = static_cast<int>(i / static_cast<std::size_t>(page.gridCount * page.gridCount));
                const int cellMinY = page.worldMinY + localY * page.cellScaleBlocks;
                const int cellMaxY = cellMinY + page.cellScaleBlocks;
                page.minY = std::min(page.minY, cellMinY);
                page.maxY = std::max(page.maxY, cellMaxY);
            }
        }

        outSummary.minSolidY = (summaryValues[0] == std::numeric_limits<int>::max()) ? 0 : summaryValues[0];
        outSummary.maxSolidY = (summaryValues[1] == std::numeric_limits<int>::min()) ? 1 : summaryValues[1];
        outSummary.maxWaterY = summaryValues[2];
        outSummary.waterCells = summaryValues[3];
        outSummary.canopyCells = summaryValues[4];
        outSummary.trunkCells = summaryValues[5];
        outSummary.occupiedCells = summaryValues[6];
        outSummary.visibleFaceCount = summaryValues[7];

        if (page.minY == std::numeric_limits<int>::max() || page.maxY == std::numeric_limits<int>::min())
        {
            page.minY = page.worldMinY;
            page.maxY = page.worldMinY + 1;
        }

        if (timestampFrequency_ > 0)
        {
            const auto timestampMs = [this](std::uint64_t begin, std::uint64_t end) -> double
            {
                if (end <= begin || timestampFrequency_ == 0)
                {
                    return 0.0;
                }
                return static_cast<double>(end - begin) * 1000.0 / static_cast<double>(timestampFrequency_);
            };

            outTimings.synthesisMs = timestampMs(timestamps[0], timestamps[1]);
            outTimings.stampMs = timestampMs(timestamps[2], timestamps[3]);
            outTimings.faceBuildMs = timestampMs(timestamps[4], timestamps[5]);
        }

        return true;
    }

private:
    static constexpr UINT kTimestampQueryCount = 6u;

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
                        "failed to wait for page compute fence");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }

    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12Resource> createBuffer(D3D12_HEAP_TYPE heapType,
                                                                      std::uint64_t sizeInBytes,
                                                                      D3D12_RESOURCE_STATES initialState,
                                                                      D3D12_RESOURCE_FLAGS flags)
    {
        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = heapType;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapProps.CreationNodeMask = 1;
        heapProps.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = std::max<std::uint64_t>(sizeInBytes, 4u);
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = flags;

        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        throwIfFailedDx(device_->CreateCommittedResource(&heapProps,
                                                         D3D12_HEAP_FLAG_NONE,
                                                         &desc,
                                                         initialState,
                                                         nullptr,
                                                         IID_PPV_ARGS(&resource)),
                        "failed to create page compute buffer");
        return resource;
    }

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> synthesizePipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> structureStampPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> faceMaskPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> timestampQueryHeap_;
    Microsoft::WRL::ComPtr<ID3D12Resource> timestampReadbackBuffer_;
    HANDLE fenceEvent_{nullptr};
    UINT64 fenceValue_{0};
    UINT64 lastSubmittedFenceValue_{0};
    UINT64 timestampFrequency_{0};
};
