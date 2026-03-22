// chunk_manager_far_terrain.inl
// Defines the internal far-LOD terrain streaming and rendering subsystem used by ChunkManager.

class FarTerrainManager
{
public:
    static constexpr int kLogicalSize = 16;
    static constexpr std::size_t kVoxelCount =
        static_cast<std::size_t>(kLogicalSize) *
        static_cast<std::size_t>(kLogicalSize) *
        static_cast<std::size_t>(kLogicalSize);
    static constexpr std::size_t kMaxFarFacesPerTile = 8192u;
    static constexpr std::size_t kMaxFarVerticesPerTile = kMaxFarFacesPerTile * 4u;
    static constexpr std::size_t kMaxFarIndicesPerTile = kMaxFarFacesPerTile * 6u;
    static constexpr std::uint32_t kInvalidFarDrawRecordIndex = std::numeric_limits<std::uint32_t>::max();
    static constexpr std::uint32_t kTopPlaneCount = 3u;
    static constexpr std::uint32_t kSideSlicesPerLayer = 64u;
    static constexpr std::uint32_t kPlaneCount = kTopPlaneCount + 3u * kSideSlicesPerLayer;
    static constexpr std::uint32_t kFacePrefixGroupSize = 256u;
    static constexpr std::uint32_t kFacePrefixGroupCount =
        static_cast<std::uint32_t>((kPlaneCount + kFacePrefixGroupSize - 1u) / kFacePrefixGroupSize);
    static constexpr std::uint32_t kFaceMetadataEntryCount = 1u;
    static constexpr std::uint32_t kMaxTopDescriptorsPerPlane =
        static_cast<std::uint32_t>(kLogicalSize * kLogicalSize);
    static constexpr std::uint32_t kMaxSideDescriptorsPerPlane = static_cast<std::uint32_t>(kLogicalSize);
    static constexpr std::uint32_t kFaceDescriptorCount =
        kTopPlaneCount * kMaxTopDescriptorsPerPlane +
        (kPlaneCount - kTopPlaneCount) * kMaxSideDescriptorsPerPlane;
    static constexpr std::uint32_t kGpuDrawRecordOverflowFlag = 0x80000000u;
    static constexpr std::uint32_t kGpuDrawRecordFaceCountMask = 0x7fffffffu;

    struct FarLodChunkKey
    {
        int level{0};
        glm::ivec3 coord{0};

        bool operator==(const FarLodChunkKey& other) const noexcept
        {
            return level == other.level && coord == other.coord;
        }
    };

    struct FarLodLevelConfig
    {
        int level{0};
        int blockScale{1};
        int innerRadiusChunks{0};
        int outerRadiusChunks{0};

        [[nodiscard]] int chunkSpanBlocks() const noexcept
        {
            return kLogicalSize * blockScale;
        }
    };

    struct FarLodVoxel
    {
        std::uint8_t occupied{0};
        BlockId material{BlockId::Air};
        std::uint8_t flags{0};
    };

    using PackedFarLodVoxelGpu = std::uint32_t;

    struct GpuBlockFaceUv
    {
        glm::vec2 base{0.0f};
        glm::vec2 size{0.0f};
    };
    static_assert(sizeof(GpuBlockFaceUv) == 16u);

    struct GpuStructureInstance
    {
        std::uint32_t type{0};
        std::int32_t originX{0};
        std::int32_t originY{0};
        std::int32_t originZ{0};
        std::int32_t boundsMinX{0};
        std::int32_t boundsMinY{0};
        std::int32_t boundsMinZ{0};
        std::int32_t boundsMaxX{0};
        std::int32_t boundsMaxY{0};
        std::int32_t boundsMaxZ{0};
        std::uint32_t trunkHeight{0};
        std::uint32_t bareTrunkHeight{0};
        std::uint32_t maxLodLevel{0};
        std::uint32_t reserved0{0};
        std::uint32_t reserved1{0};
        std::uint32_t reserved2{0};
    };
    static_assert(sizeof(GpuStructureInstance) == 64u);

    struct GpuStructureRegionState
    {
        StructureRegionKey key{};
        std::uint32_t instanceCount{0};
        Microsoft::WRL::ComPtr<ID3D12Resource> instanceBuffer;
        D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};

        [[nodiscard]] bool valid() const noexcept
        {
            return instanceBuffer != nullptr && instanceCount > 0;
        }
    };

    struct FarLodChunkCpu
    {
        static constexpr int logicalSize = kLogicalSize;

        FarLodChunkKey key{};
        glm::ivec3 worldMin{0};
        int blockScale{1};
        std::array<FarLodVoxel, kVoxelCount> voxels{};
        glm::vec3 boundsMin{0.0f};
        glm::vec3 boundsMax{0.0f};
        int minOccupiedLocalY{logicalSize};
        int maxOccupiedLocalY{-1};
        bool terrainReady{false};
        bool structuresReady{false};
        bool meshReady{false};
    };

    struct FarLodChunkGpuState
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> columnBuffer;
        D3D12_RESOURCE_STATES columnState{D3D12_RESOURCE_STATE_COMMON};
        Microsoft::WRL::ComPtr<ID3D12Resource> voxelBuffer;
        D3D12_RESOURCE_STATES voxelState{D3D12_RESOURCE_STATE_COMMON};
        Microsoft::WRL::ComPtr<ID3D12Resource> faceCountBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> faceAnalysisBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> faceDescriptorBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> facePrefixBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> faceGroupSumBuffer;
        D3D12_RESOURCE_STATES faceCountState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES faceAnalysisState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES faceDescriptorState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES facePrefixState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES faceGroupSumState{D3D12_RESOURCE_STATE_COMMON};
        UINT64 voxelFenceValue{0};
        UINT64 readbackFenceValue{0};
        bool voxelReady{false};
        std::uint32_t parityMismatchCount{0};
        bool parityValidated{false};
        std::uint32_t pageIndex{kInvalidChunkBufferPage};
        std::size_t vertexOffset{0};
        std::size_t indexOffset{0};
        std::size_t vertexCount{0};
        std::uint32_t indexCount{0};
        std::size_t reservedVertexCount{0};
        std::size_t reservedIndexCount{0};
        std::uint32_t faceCapacity{0};
        std::uint32_t recordIndex{kInvalidFarDrawRecordIndex};
        bool resident{false};
    };

    using ColumnSampleFn =
        std::function<ColumnSample(int worldX, int worldZ, int slabMinWorldY, int slabMaxWorldY)>;

    FarTerrainManager()
        : levels_{
              FarLodLevelConfig{1, 2, kDefaultNearRenderDistance, 32},
              FarLodLevelConfig{2, 4, 32, 80},
              FarLodLevelConfig{3, 8, 128, 192},
              FarLodLevelConfig{4, 16, 192, 320},
              FarLodLevelConfig{5, 32, 320, kMaxTotalRenderDistanceChunks}}
    {
        const char* parityEnv = std::getenv("BLOCKGAME_ENABLE_LOD_GPU_PARITY");
        gpuParityEnabled_ =
            (parityEnv != nullptr && std::string_view(parityEnv) != "0" && std::string_view(parityEnv) != "false");
    }

    ~FarTerrainManager()
    {
        shutdown();
    }

    void shutdown()
    {
        setRenderSynchronization(nullptr, 0);
        stopWorkers();
        clear();
        uploadContext_.shutdown();
        gpuContext_.shutdown();
        worldgenHeaderBuffer_.Reset();
        worldgenBiomeBuffer_.Reset();
        blockUvBuffer_.Reset();
        blockUvCount_ = 0;
        emptyVoxelBuffer_.Reset();
        levelAtlases_.clear();
        {
            std::lock_guard<std::mutex> lock(structureRegionMutex_);
            destroyGpuStructureRegions();
        }
        if (parityReadbackScratch_ != nullptr)
        {
            parityReadbackScratch_->Unmap(0, nullptr);
        }
        parityReadbackScratch_.Reset();
        parityReadbackMapped_ = nullptr;
    }

    void setEnabled(bool enabled)
    {
        if (enabled_ == enabled)
        {
            return;
        }

        enabled_ = enabled;
        if (!enabled_)
        {
            clear();
        }
    }

    [[nodiscard]] bool enabled() const noexcept
    {
        return enabled_;
    }

    void setWorkerCount(std::size_t count)
    {
        const std::size_t clamped = std::max<std::size_t>(count, 1);
        if (workerCount_ == clamped)
        {
            return;
        }

        stopWorkers();
        workerCount_ = clamped;
    }

    void setDistanceBlocks(int blocks)
    {
        farDistanceTargetBlocks_ = blocks <= 0 ? 0 : std::max(blocks, 256);
        if (farDistanceTargetBlocks_ <= 0)
        {
            farDistanceBlocks_ = 0;
        }
        else if (farDistanceBlocks_ > farDistanceTargetBlocks_)
        {
            farDistanceBlocks_ = farDistanceTargetBlocks_;
        }
    }

    [[nodiscard]] int distanceBlocks() const noexcept
    {
        return farDistanceTargetBlocks_;
    }

    void setFogStartBlocks(int blocks) noexcept
    {
        fogStartBlocks_ = std::max(blocks, 0);
    }

    [[nodiscard]] int fogStartBlocks() const noexcept
    {
        return fogStartBlocks_;
    }

    void setDevice(ID3D12Device* device)
    {
        device_ = device;
        uploadContext_.initialize(device_.Get());
        gpuContext_.initialize(device_.Get());
        gpuContext_.setReadbackEnabled(false);
        levelAtlases_.clear();
        uploadWorldgenTables();
        emptyVoxelBuffer_.Reset();
        {
            const std::uint64_t bufferBytes =
                static_cast<std::uint64_t>(kLogicalSize * kLogicalSize * sizeof(GpuTerrainColumnDescriptor));
            emptyVoxelBuffer_ = createDefaultBuffer(device_.Get(), bufferBytes, D3D12_RESOURCE_STATE_COMMON);
            if (emptyVoxelBuffer_ != nullptr)
            {
                std::byte* uploadMapped = nullptr;
                Microsoft::WRL::ComPtr<ID3D12Resource> upload = createUploadBuffer(device_.Get(), bufferBytes, uploadMapped);
                if (upload != nullptr && uploadMapped != nullptr)
                {
                    std::memset(uploadMapped, 0, static_cast<std::size_t>(bufferBytes));
                    if (uploadContext_.ready() && uploadContext_.begin())
                    {
                        uploadContext_.transition(emptyVoxelBuffer_.Get(),
                                                  D3D12_RESOURCE_STATE_COMMON,
                                                  D3D12_RESOURCE_STATE_COPY_DEST);
                        uploadContext_.copyBuffer(emptyVoxelBuffer_.Get(), 0, upload.Get(), 0, bufferBytes);
                        uploadContext_.transition(emptyVoxelBuffer_.Get(),
                                                  D3D12_RESOURCE_STATE_COPY_DEST,
                                                  D3D12_RESOURCE_STATE_COMMON);
                        uploadContext_.flush(nullptr);
                        uploadContext_.waitForIdle();
                    }
                    upload->Unmap(0, nullptr);
                }
            }
        }
        if (parityReadbackScratch_ != nullptr)
        {
            parityReadbackScratch_->Unmap(0, nullptr);
        }
        parityReadbackScratch_.Reset();
        parityReadbackMapped_ = nullptr;
        if (gpuParityEnabled_)
        {
            parityReadbackScratch_ = createReadbackBuffer(device_.Get(),
                                                          static_cast<std::uint64_t>(kVoxelCount * sizeof(PackedFarLodVoxelGpu)),
                                                          parityReadbackMapped_);
            setDebugObjectName(parityReadbackScratch_.Get(), L"FarLodParityReadbackScratch");
        }
        clear();
    }

    void setRenderSynchronization(ID3D12Fence* graphicsFence, UINT64 graphicsFenceValue)
    {
        uploadContext_.setGraphicsFenceDependency(graphicsFence, graphicsFenceValue);
    }

    [[nodiscard]] ID3D12Fence* uploadFence() const noexcept
    {
        return uploadContext_.fence();
    }

    [[nodiscard]] UINT64 lastSubmittedUploadFenceValue() const noexcept
    {
        return uploadContext_.lastSubmittedFenceValue();
    }

    void setBenchmarkMetrics(ChunkBenchmarkMetrics* metrics) noexcept
    {
        benchmarkMetrics_ = metrics;
    }

    void setSeaLevel(int seaLevel) noexcept
    {
        seaLevel_ = seaLevel;
    }

    void setWorldgenTables(const terrain::FarLodWorldgenTables& tables)
    {
        worldgenTables_ = tables;
        uploadWorldgenTables();
        for (auto& [levelId, atlas] : levelAtlases_)
        {
            (void)levelId;
            atlas.initialized = false;
        }
    }

    void setStructureFieldSources(StructureSampleColumnFn sampleColumnFn,
                                  StructureSurfaceBlockFn surfaceBlockFn,
                                  StructureDensityFn densityFn)
    {
        std::lock_guard<std::mutex> lock(structureRegionMutex_);
        structureSampleColumnFn_ = std::move(sampleColumnFn);
        structureSurfaceBlockFn_ = std::move(surfaceBlockFn);
        structureDensityFn_ = std::move(densityFn);
        destroyGpuStructureRegions();
    }

    void setBlockUvTable(const std::vector<GpuBlockFaceUv>& table)
    {
        blockUvBuffer_.Reset();
        blockUvCount_ = 0;
        if (device_ == nullptr || table.empty())
        {
            return;
        }

        const std::uint64_t bufferBytes = static_cast<std::uint64_t>(table.size() * sizeof(GpuBlockFaceUv));
        blockUvBuffer_ = createDefaultBuffer(device_.Get(), bufferBytes, D3D12_RESOURCE_STATE_COMMON);
        if (blockUvBuffer_ == nullptr)
        {
            return;
        }

        std::byte* uploadMapped = nullptr;
        Microsoft::WRL::ComPtr<ID3D12Resource> upload = createUploadBuffer(device_.Get(), bufferBytes, uploadMapped);
        if (upload == nullptr || uploadMapped == nullptr)
        {
            return;
        }
        std::memcpy(uploadMapped, table.data(), static_cast<std::size_t>(bufferBytes));

        if (uploadContext_.ready() && uploadContext_.begin())
        {
            uploadContext_.transition(blockUvBuffer_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            uploadContext_.copyBuffer(blockUvBuffer_.Get(), 0, upload.Get(), 0, bufferBytes);
            uploadContext_.transition(blockUvBuffer_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            uploadContext_.flush(nullptr);
            uploadContext_.waitForIdle();
        }

        upload->Unmap(0, nullptr);
        blockUvCount_ = static_cast<std::uint32_t>(table.size());
    }

    void setVisibility(const Frustum& frustum, const glm::vec3& cameraWorldPos) const
    {
        std::lock_guard<std::mutex> lock(configMutex_);
        lastVisibilityFrustum_ = frustum;
        lastVisibilityCameraPos_ = cameraWorldPos;
        hasVisibilityFrustum_ = true;
    }

    void setBacklogPressure(int exactMissingChunks, std::size_t exactPendingUploads) noexcept
    {
        exactMissingChunks_ = std::max(exactMissingChunks, 0);
        exactPendingUploads_ = exactPendingUploads;
    }

    void update(const glm::ivec3& cameraChunk,
                const glm::vec3& cameraForward,
                int nearRadiusChunks,
                int realDistanceBlocks,
                double uploadBudgetMs,
                const ColumnSampleFn& columnSampleFn)
    {
        builtTilesLastUpdate_ = 0;
        if (!enabled_ || realDistanceBlocks <= 0)
        {
            clear();
            return;
        }
        if (workerThreads_.empty())
        {
            startWorkers();
        }

        commitPendingMeshUploads();
        ++updateStamp_;
        cameraChunk_ = cameraChunk;
        if (glm::dot(cameraForward, cameraForward) > kEpsilon)
        {
            cameraForward_ = glm::normalize(cameraForward);
        }

        {
            std::lock_guard<std::mutex> lock(configMutex_);
            columnSampleFn_ = columnSampleFn;
        }

        FarLodWorkBudget workBudget = computeWorkBudget();
        const int requestedDistanceBlocks = realDistanceBlocks <= 0 ? 0 : std::max(realDistanceBlocks, 256);
        if (farDistanceTargetBlocks_ <= 0)
        {
            farDistanceTargetBlocks_ = requestedDistanceBlocks;
        }

        const int minimumActiveRadiusChunks = nearRadiusChunks + 1;
        const int targetRadiusChunks =
            farDistanceTargetBlocks_ > 0
                ? std::max(minimumActiveRadiusChunks,
                           ceilToIntPositive(static_cast<float>(farDistanceTargetBlocks_) /
                                             static_cast<float>(kChunkSizeX)))
                : 0;
        int currentRadiusChunks =
            farDistanceBlocks_ > 0
                ? std::max(minimumActiveRadiusChunks,
                           ceilToIntPositive(static_cast<float>(farDistanceBlocks_) /
                                             static_cast<float>(kChunkSizeX)))
                : minimumActiveRadiusChunks;

        if (targetRadiusChunks <= 0)
        {
            farDistanceBlocks_ = 0;
        }
        else if (currentRadiusChunks > targetRadiusChunks)
        {
            currentRadiusChunks = targetRadiusChunks;
            farDistanceBlocks_ = chunksToBlocks(currentRadiusChunks);
        }
        else if (currentRadiusChunks < targetRadiusChunks)
        {
            currentRadiusChunks = std::min(targetRadiusChunks,
                                           currentRadiusChunks + std::max(workBudget.distanceRampStepChunks, 0));
            farDistanceBlocks_ = chunksToBlocks(currentRadiusChunks);
        }
        else
        {
            farDistanceBlocks_ = chunksToBlocks(currentRadiusChunks);
        }

        const int realRadiusChunks = std::max(minimumActiveRadiusChunks, currentRadiusChunks);
        refreshLevels(nearRadiusChunks, realRadiusChunks);
        std::size_t remainingNewChunkActivations = workBudget.newChunkActivations;
        std::size_t remainingNewFallbackActivations = workBudget.newFallbackActivations;

        for (const FarLodLevelConfig& level : levels_)
        {
            if (level.outerRadiusChunks > level.innerRadiusChunks)
            {
                int& activationOuterRadiusChunks = levelActivationOuterRadiusChunks_[level.level];
                activationOuterRadiusChunks = std::max(activationOuterRadiusChunks, level.innerRadiusChunks);
                activationOuterRadiusChunks =
                    std::min(level.outerRadiusChunks, activationOuterRadiusChunks + workBudget.activationStepChunks);
            }
        }

        consumeReadyTouchLevelPlan();
        for (const FarLodLevelConfig& level : levels_)
        {
            if (level.outerRadiusChunks > level.innerRadiusChunks)
            {
                applyTouchLevelCache(level, remainingNewChunkActivations, remainingNewFallbackActivations);
            }
        }
        requestTouchLevelPlan(cameraChunk_);

        static constexpr std::uint64_t kFarTileUntouchedGraceUpdates = 12u;
        static constexpr std::uint64_t kFarFallbackUntouchedGraceUpdates = 2u;
        std::vector<FarLodChunkKey> staleKeys;
        staleKeys.reserve(chunks_.size());
        for (auto& [key, chunk] : chunks_)
        {
            if (chunk.lastTouchedStamp == updateStamp_)
            {
                continue;
            }

            if (chunk.fallbackOnly)
            {
                // Fallback parents are temporary: stop drawing them as soon as they are no longer needed,
                // then retire their residency quickly so they don't hang around as redundant coverage.
                chunk.active = false;
                if (chunk.gpu.resident && chunk.gpu.indexCount > 0)
                {
                    releaseChunkRenderAllocation(chunk);
                }
            }

            const std::uint64_t untouchedUpdates = updateStamp_ - chunk.lastTouchedStamp;
            const std::uint64_t graceUpdates = chunk.fallbackOnly ? kFarFallbackUntouchedGraceUpdates : kFarTileUntouchedGraceUpdates;
            if (untouchedUpdates > graceUpdates)
            {
                staleKeys.push_back(key);
            }
        }

        if (staleKeys.size() > workBudget.staleReleaseCount)
        {
            staleKeys.resize(workBudget.staleReleaseCount);
        }

        for (const FarLodChunkKey& key : staleKeys)
        {
            auto it = chunks_.find(key);
            if (it == chunks_.end())
            {
                continue;
            }

            releaseChunkGpu(it->second);
            chunks_.erase(it);
        }

        if (lodVisibilityDebugLoggingEnabled())
        {
            std::size_t touchedCount = 0;
            std::size_t activeCount = 0;
            std::size_t residentCount = 0;
            std::size_t dirtyCount = 0;
            for (const auto& [key, chunk] : chunks_)
            {
                (void)key;
                if (chunk.lastTouchedStamp == updateStamp_)
                {
                    ++touchedCount;
                }
                if (chunk.active)
                {
                    ++activeCount;
                }
                if (chunk.gpu.resident && chunk.gpu.indexCount > 0)
                {
                    ++residentCount;
                }
                if (chunk.dirty)
                {
                    ++dirtyCount;
                }
            }

            std::ostringstream stream;
            stream << "lodvis update stamp=" << updateStamp_
                   << " chunks=" << chunks_.size()
                   << " touched=" << touchedCount
                   << " active=" << activeCount
                   << " resident=" << residentCount
                   << " dirty=" << dirtyCount
                   << " stale_erased=" << staleKeys.size();
            lodVisibilityDebugLog(stream.str());
        }

        scheduleDirtyBuilds();
        submitGpuSynthesisRequests(uploadBudgetMs);
        collectCompletedBuilds(uploadBudgetMs);
        pollGpuParityResults();
    }

    [[nodiscard]] std::vector<ChunkRenderBatch> buildRenderBatches(const Frustum& frustum) const
    {
        (void)frustum;
        std::lock_guard<std::mutex> lock(configMutex_);
        std::vector<ChunkRenderBatch> batches(bufferPages_.size());
        lastRenderedFaceCount_ = 0;
        lastRenderedVertexCount_ = 0;
        std::size_t emittedPages = 0;
        bool loggedFirstRecord = false;
        for (std::size_t pageIndex = 0; pageIndex < bufferPages_.size(); ++pageIndex)
        {
            const BufferPage& page = bufferPages_[pageIndex];
            ChunkRenderBatch& batch = batches[pageIndex];
            batch.vertexBufferView = page.vertexView;
            batch.indexBufferView = page.indexView;
            batch.gpuCullRecordBuffer = page.drawRecordBuffer.Get();
            batch.gpuCullRecordCount = static_cast<std::uint32_t>(page.recordActiveCount);
            batch.supportsGpuCull = (batch.gpuCullRecordBuffer != nullptr && batch.gpuCullRecordCount > 0);
            batch.debugPageIndex = static_cast<std::uint32_t>(pageIndex);
            if (batch.supportsGpuCull)
            {
                ++emittedPages;
                if (lodVisibilityDebugLoggingEnabled() && !loggedFirstRecord)
                {
                    for (const auto& [key, chunk] : chunks_)
                    {
                        (void)key;
                        if (!chunk.active ||
                            !chunk.gpu.resident ||
                            chunk.gpu.pageIndex != pageIndex ||
                            chunk.gpu.indexCount == 0)
                        {
                            continue;
                        }

                        std::ostringstream recordStream;
                        recordStream << "lodvis first_record page=" << pageIndex
                                     << " level=" << chunk.key.level
                                     << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                                     << " boundsMin=[" << chunk.residentBoundsMin.x << "," << chunk.residentBoundsMin.y << "," << chunk.residentBoundsMin.z << "]"
                                     << " boundsMax=[" << chunk.residentBoundsMax.x << "," << chunk.residentBoundsMax.y << "," << chunk.residentBoundsMax.z << "]"
                                     << " indexCount=" << chunk.gpu.indexCount
                                     << " firstIndex=" << chunk.gpu.indexOffset
                                     << " baseVertex=" << chunk.gpu.vertexOffset
                                     << " recordIndex=" << chunk.gpu.recordIndex
                                     << " pageRecordCount=" << page.recordActiveCount;
                        lodVisibilityDebugLog(recordStream.str());
                        loggedFirstRecord = true;
                        break;
                    }
                }
            }
        }

        auto emptyIt = std::remove_if(batches.begin(),
                                      batches.end(),
                                      [](const ChunkRenderBatch& batch)
                                      {
                                          return batch.gpuCullRecordCount == 0;
                                      });
        batches.erase(emptyIt, batches.end());
        if (lodVisibilityDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "lodvis batches emitted_pages=" << emittedPages
                   << " batch_count=" << batches.size()
                   << " skipped_inactive=0"
                   << " skipped_nonresident=0"
                   << " skipped_noindices=0"
                   << " skipped_invalidpage=0"
                   << " skipped_frustum=0";
            lodVisibilityDebugLog(stream.str());
        }
        return batches;
    }

    void invalidateWorldBlock(const glm::ivec3& worldPos)
    {
        std::lock_guard<std::mutex> lock(configMutex_);
        for (auto& [levelId, atlas] : levelAtlases_)
        {
            (void)levelId;
            const int scale = std::max(atlas.blockScale, 1);
            const glm::ivec2 cellCoord(floorDiv(worldPos.x, scale), floorDiv(worldPos.z, scale));
            appendClippedAtlasUpdateRect(atlas.pendingDirtyRects,
                                         AtlasUpdateRect{cellCoord, glm::ivec2(1, 1)},
                                         atlas.originCell,
                                         atlas.atlasSizeCells);
        }

        for (auto& [key, chunk] : chunks_)
        {
            (void)key;
            const int span = chunk.level.chunkSpanBlocks();
            const glm::ivec3 minWorld = chunk.cpu.worldMin;
            const glm::ivec3 maxWorld = minWorld + glm::ivec3(span - 1);
            if (worldPos.x < minWorld.x - span || worldPos.x > maxWorld.x + span ||
                worldPos.y < minWorld.y - span || worldPos.y > maxWorld.y + span ||
                worldPos.z < minWorld.z - span || worldPos.z > maxWorld.z + span)
            {
                continue;
            }

            markDirty(chunk);
        }
    }

    void clear()
    {
        buildEpoch_.fetch_add(1, std::memory_order_acq_rel);
        {
            std::lock_guard<std::mutex> lock(buildQueueMutex_);
            buildQueue_.clear();
            queuedKeys_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(completedMutex_);
            completedBuilds_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            gpuSynthesisRequests_.clear();
            pendingGpuParityReadbacks_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(dirtyBuildPlanMutex_);
            hasPendingDirtyBuildPlanRequest_ = false;
            hasReadyDirtyBuildPlanResult_ = false;
            pendingDirtyBuildPlanRequest_ = {};
            readyDirtyBuildPlanResult_ = {};
        }
        {
            std::lock_guard<std::mutex> lock(touchLevelPlanMutex_);
            hasPendingTouchLevelPlanRequest_ = false;
            hasReadyTouchLevelPlanResult_ = false;
            touchLevelPlannerBusy_ = false;
            pendingTouchLevelPlanRequest_ = {};
            readyTouchLevelPlanResult_ = {};
            touchLevelCaches_.clear();
            hasLastTouchLevelPlanRequest_ = false;
            lastTouchLevelPlanLevels_.clear();
        }
        std::lock_guard<std::mutex> lock(configMutex_);
        for (auto& [key, chunk] : chunks_)
        {
            (void)key;
            releaseChunkGpu(chunk);
            chunk.inFlight = false;
        }
        chunks_.clear();
        destroyBufferPages();
        builtTilesLastUpdate_ = 0;
        skippedTilesLastUpdate_ = 0;
        lastAverageBuildMs_ = 0.0;
        lastAverageGpuSynthesisMs_ = 0.0;
        lastAverageGpuStampMs_ = 0.0;
        lastAverageGpuFaceBuildMs_ = 0.0;
        lastAverageCpuTerrainSynthesisMs_ = 0.0;
        lastAverageCpuStructureStampMs_ = 0.0;
        lastAverageCpuMeshMs_ = 0.0;
        lastAverageUploadWaitMs_ = 0.0;
        lastAverageUploadCopyMs_ = 0.0;
        lastCollectMs_ = 0.0;
        lastUploadMs_ = 0.0;
        lastBuiltFaceCount_ = 0;
        lastBuiltVertexCount_ = 0;
        lastBuiltIndexCount_ = 0;
        lastRenderedFaceCount_ = 0;
        lastRenderedVertexCount_ = 0;
        rollingGpuParityMismatchCount_ = 0;
        levelActivationOuterRadiusChunks_.clear();
        {
            std::lock_guard<std::mutex> structureLock(structureRegionMutex_);
            destroyGpuStructureRegions();
        }
    }

    [[nodiscard]] int activeTileCount() const noexcept
    {
        std::lock_guard<std::mutex> lock(configMutex_);
        return static_cast<int>(chunks_.size());
    }

    [[nodiscard]] int dirtyTileCount() const noexcept
    {
        int dirty = 0;
        std::lock_guard<std::mutex> lock(configMutex_);
        for (const auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (chunk.active && chunk.dirty)
            {
                ++dirty;
            }
        }
        return dirty;
    }

    [[nodiscard]] int builtTilesLastUpdate() const noexcept
    {
        return builtTilesLastUpdate_;
    }

    [[nodiscard]] int skippedTilesLastUpdate() const noexcept
    {
        return skippedTilesLastUpdate_;
    }

    [[nodiscard]] double averageCpuTerrainSynthesisMs() const noexcept
    {
        return lastAverageCpuTerrainSynthesisMs_;
    }

    [[nodiscard]] double averageCpuStructureStampMs() const noexcept
    {
        return lastAverageCpuStructureStampMs_;
    }

    [[nodiscard]] double averageCpuMeshMs() const noexcept
    {
        return lastAverageCpuMeshMs_;
    }

    [[nodiscard]] double averageUploadWaitMs() const noexcept
    {
        return lastAverageUploadWaitMs_;
    }

    [[nodiscard]] double averageUploadCopyMs() const noexcept
    {
        return lastAverageUploadCopyMs_;
    }

    [[nodiscard]] std::size_t renderedFaceCount() const noexcept
    {
        return lastRenderedFaceCount_;
    }

    [[nodiscard]] std::size_t renderedVertexCount() const noexcept
    {
        return lastRenderedVertexCount_;
    }

    [[nodiscard]] double lastCollectMs() const noexcept
    {
        return lastCollectMs_;
    }

    [[nodiscard]] double lastUploadMs() const noexcept
    {
        return lastUploadMs_;
    }

private:
    void refreshLevels(int nearRadiusChunks, int totalRadiusChunks)
    {
        struct LevelTemplate
        {
            int level;
            int blockScale;
            int outerRadiusChunks;
        };

        static constexpr std::array<LevelTemplate, 8> kLevelTemplates{{
            {1, 2, 32},
            {2, 4, 80},
            {3, 8, 128},
            {4, 16, 192},
            {5, 32, 288},
            {6, 64, 384},
            {7, 128, 480},
            {8, 256, kMaxTotalRenderDistanceChunks},
        }};

        levels_.clear();
        if (totalRadiusChunks <= nearRadiusChunks)
        {
            return;
        }

        constexpr int kExactOverlapChunks = 4;
        int previousOuterRadiusChunks = nearRadiusChunks;
        for (const LevelTemplate& levelTemplate : kLevelTemplates)
        {
            const int outerRadiusChunks = std::min(totalRadiusChunks, levelTemplate.outerRadiusChunks);
            const int innerRadiusChunks = levels_.empty()
                                              ? std::max(0, nearRadiusChunks - kExactOverlapChunks)
                                              : std::max(0, previousOuterRadiusChunks - levelTemplate.blockScale);
            if (outerRadiusChunks <= innerRadiusChunks)
            {
                continue;
            }

            levels_.push_back(FarLodLevelConfig{
                levelTemplate.level,
                levelTemplate.blockScale,
                innerRadiusChunks,
                outerRadiusChunks});
            previousOuterRadiusChunks = outerRadiusChunks;
            if (outerRadiusChunks == totalRadiusChunks)
            {
                break;
            }
        }

        std::unordered_set<int> activeLevelIds;
        activeLevelIds.reserve(levels_.size());
        for (const FarLodLevelConfig& level : levels_)
        {
            activeLevelIds.insert(level.level);
            int& progress = levelActivationOuterRadiusChunks_[level.level];
            if (progress < level.innerRadiusChunks)
            {
                progress = level.innerRadiusChunks;
            }
            progress = std::min(progress, level.outerRadiusChunks);
        }

        for (auto it = levelActivationOuterRadiusChunks_.begin(); it != levelActivationOuterRadiusChunks_.end();)
        {
            if (!activeLevelIds.contains(it->first))
            {
                it = levelActivationOuterRadiusChunks_.erase(it);
            }
            else
            {
                ++it;
            }
        }

        for (auto it = touchLevelCaches_.begin(); it != touchLevelCaches_.end();)
        {
            if (!activeLevelIds.contains(it->first))
            {
                it = touchLevelCaches_.erase(it);
            }
            else
            {
                ++it;
            }
        }
        if (!activeLevelIds.size())
        {
            hasLastTouchLevelPlanRequest_ = false;
            lastTouchLevelPlanLevels_.clear();
        }
    }

    struct FarLodChunkKeyHasher
    {
        std::size_t operator()(const FarLodChunkKey& key) const noexcept
        {
            std::size_t hash = static_cast<std::size_t>(key.level) * 73856093u;
            hash ^= static_cast<std::size_t>(key.coord.x) * 19349663u;
            hash ^= static_cast<std::size_t>(key.coord.y) * 83492791u;
            hash ^= static_cast<std::size_t>(key.coord.z) * 2654435761u;
            return hash;
        }
    };

    struct BufferPage
    {
        struct Range
        {
            std::size_t offset{0};
            std::size_t size{0};
        };

        Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> indexBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> vertexUploadBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> indexUploadBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> drawRecordBuffer;
        D3D12_VERTEX_BUFFER_VIEW vertexView{};
        D3D12_INDEX_BUFFER_VIEW indexView{};
        std::byte* mappedVertexData{nullptr};
        std::byte* mappedIndexData{nullptr};
        D3D12_RESOURCE_STATES vertexState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES indexState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES drawRecordState{D3D12_RESOURCE_STATE_COMMON};
        std::size_t vertexCapacity{0};
        std::size_t indexCapacity{0};
        std::size_t recordCapacity{0};
        std::size_t vertexCursor{0};
        std::size_t indexCursor{0};
        std::size_t recordCursor{0};
        std::size_t recordActiveCount{0};
        std::vector<Range> freeVertices;
        std::vector<Range> freeIndices;
        std::vector<Range> freeRecords;
    };

    struct Allocation
    {
        std::uint32_t pageIndex{kInvalidChunkBufferPage};
        std::size_t vertexOffset{0};
        std::size_t indexOffset{0};
        std::uint32_t recordIndex{kInvalidFarDrawRecordIndex};
    };

    struct FarLodChunkRecord
    {
        struct PendingRenderMesh
        {
            std::uint32_t pageIndex{kInvalidChunkBufferPage};
            std::size_t vertexOffset{0};
            std::size_t indexOffset{0};
            std::size_t vertexCount{0};
            std::uint32_t indexCount{0};
            std::uint32_t faceCapacity{0};
            std::uint32_t recordIndex{kInvalidFarDrawRecordIndex};
            glm::vec3 boundsMin{0.0f};
            glm::vec3 boundsMax{1.0f};
            UINT64 uploadFenceValue{0};
            UINT64 gpuFenceValue{0};
            std::uint32_t buildVersion{0};
            std::uint64_t epoch{0};
            bool gpuGenerated{false};
            Microsoft::WRL::ComPtr<ID3D12Resource> drawRecordReadbackBuffer;
            ChunkRenderBatch::GpuCullRecord* mappedDrawRecord{nullptr};

            [[nodiscard]] bool valid() const noexcept
            {
                return pageIndex != kInvalidChunkBufferPage;
            }
        };

        FarLodChunkKey key{};
        FarLodLevelConfig level{};
        FarLodChunkCpu cpu{};
        FarLodChunkGpuState gpu{};
        glm::vec3 residentBoundsMin{0.0f};
        glm::vec3 residentBoundsMax{1.0f};
        PendingRenderMesh pendingMesh{};
        bool fallbackOnly{false};
        std::uint64_t seamSignature{0};
        std::uint64_t lastBuiltAtlasDependencyRevision{0};
        std::uint64_t lastTouchedStamp{0};
        std::uint32_t buildVersion{1};
        std::uint32_t requestedFaceCapacityHint{0};
        bool active{false};
        bool dirty{true};
        bool initialized{false};
        bool inFlight{false};
    };

    struct ChunkMesh
    {
        std::vector<Vertex> vertices;
        std::vector<std::uint32_t> indices;
        glm::vec3 boundsMin{0.0f};
        glm::vec3 boundsMax{1.0f};
    };

    struct BuildResult
    {
        FarLodChunkKey key{};
        std::uint32_t buildVersion{0};
        std::uint64_t epoch{0};
        FarLodChunkCpu cpu{};
        ChunkMesh mesh{};
        double buildMs{0.0};
        double cpuTerrainSynthesisMs{0.0};
        double cpuStructureStampMs{0.0};
        double cpuMeshMs{0.0};
        double uploadWaitMs{0.0};
        double uploadCopyMs{0.0};
        std::size_t faceCount{0};
        std::size_t vertexCount{0};
        std::size_t indexCount{0};
        bool skippedByRelevance{false};
    };

    struct GpuTerrainColumnDescriptor
    {
        std::uint32_t flags{0};
        std::int32_t terrainTopY{0};
        std::int32_t terrainBaseY{0};
        std::int32_t waterTopY{0};
        std::int32_t waterBottomY{0};
        std::int32_t canopyTopY{0};
        std::int32_t canopyBottomY{0};
        std::uint32_t terrainTopBlock{0};
        std::uint32_t terrainSideBlock{0};
        std::uint32_t waterBlock{0};
        std::uint32_t canopyBlock{0};
        std::uint32_t reserved{0};
    };
    static_assert(sizeof(GpuTerrainColumnDescriptor) == 48u);

    struct GpuTerrainAtlasSample
    {
        std::uint32_t hasSolid{0};
        std::uint32_t waterEnabled{0}; // Aggregated water presence votes within this cell (0..N).
        std::int32_t surfaceY{0};
        std::int32_t waterBottomY{0};
        std::int32_t minSurfaceY{0};
        std::int32_t maxSurfaceY{0};
        std::uint32_t surfaceBlock{0};
        std::uint32_t fillerBlock{0};
        std::int32_t canopyBottomY{0};
        std::int32_t canopyTopY{0};
        std::uint32_t canopyBlock{0};
        std::uint32_t canopyStrength{0};
    };
    static_assert(sizeof(GpuTerrainAtlasSample) == 48u);

    struct GpuAtlasSamplePoint
    {
        std::uint32_t biomeIndex{0};
        std::uint32_t biomeFlags{0};
        std::int32_t surfaceY{0};
        float distanceToShore{0.0f};
    };
    static_assert(sizeof(GpuAtlasSamplePoint) == 16u);

    struct GpuAtlasSampleCacheEntry
    {
        std::array<GpuAtlasSamplePoint, 9> points{};
    };
    static_assert(sizeof(GpuAtlasSampleCacheEntry) == 144u);

    struct GpuChunkSeedCacheSeed
    {
        std::uint32_t biomeIndex{0};
        std::int32_t positionX{0};
        std::int32_t positionZ{0};
        float radius{0.0f};
        float baseHeight{0.0f};
    };
    static_assert(sizeof(GpuChunkSeedCacheSeed) == 20u);

    struct GpuChunkSeedCacheHeader
    {
        std::uint32_t seedCount{0};
        std::uint32_t baseSeedIndex{0};
        std::uint32_t reserved0{0};
        std::uint32_t reserved1{0};
    };
    static_assert(sizeof(GpuChunkSeedCacheHeader) == 16u);

    struct GpuSynthesisRequest
    {
        FarLodChunkKey key{};
        std::uint32_t buildVersion{0};
        std::uint64_t epoch{0};
        glm::ivec3 worldMin{0};
        int blockScale{1};
        int lodLevel{0};
        std::vector<StructureRegionKey> structureRegionKeys;
        std::shared_ptr<std::array<PackedFarLodVoxelGpu, kVoxelCount>> cpuPackedParityVoxels;
        std::shared_ptr<std::array<GpuTerrainColumnDescriptor, kLogicalSize * kLogicalSize>> cpuColumnParityDescriptors;
        bool parityRequested{false};
    };

    struct PendingGpuParityReadback
    {
        FarLodChunkKey key{};
        std::uint32_t buildVersion{0};
        std::uint64_t epoch{0};
        UINT64 computeFenceValue{0};
        UINT64 copyFenceValue{0};
        std::shared_ptr<std::array<PackedFarLodVoxelGpu, kVoxelCount>> cpuPackedParityVoxels;
        std::shared_ptr<std::array<GpuTerrainColumnDescriptor, kLogicalSize * kLogicalSize>> cpuColumnParityDescriptors;
        bool parityRequested{false};
        bool copySubmitted{false};
    };

    struct BuildJob
    {
        FarLodChunkKey key{};
        FarLodLevelConfig level{};
        std::uint32_t buildVersion{0};
        std::uint64_t epoch{0};
        int ringDistanceChunks{0};
        bool hadResidentMesh{false};
        bool fallbackOnly{false};
    };

    struct DirtyBuildPlanRequest
    {
        std::uint64_t sequence{0};
        std::uint64_t epoch{0};
        glm::ivec3 cameraChunk{0};
        glm::vec3 cameraForward{0.0f, 0.0f, -1.0f};
        Frustum visibilityFrustum{};
        glm::vec3 visibilityCameraPos{0.0f};
        bool hasVisibilityFrustum{false};
        int exactMissingChunks{0};
        std::size_t exactPendingUploads{0};
        std::size_t workerBudget{1};
    };

    struct DirtyBuildPlanResult
    {
        std::uint64_t sequence{0};
        std::uint64_t epoch{0};
        std::vector<BuildJob> jobs;
    };

    struct HorizontalRingSpan
    {
        int chunkZ{0};
        int minChunkX{0};
        int maxChunkX{0};
    };

    struct TouchLevelPlanLevelRequest
    {
        FarLodLevelConfig level{};
        int activeOuterRadiusChunks{0};
    };

    struct TouchLevelPlanLevelResult
    {
        FarLodLevelConfig level{};
        int activeOuterRadiusChunks{0};
        std::vector<FarLodChunkKey> activeKeys;
    };

    struct TouchLevelPlanRequest
    {
        std::uint64_t sequence{0};
        std::uint64_t epoch{0};
        glm::ivec3 cameraChunk{0};
        std::size_t workerBudget{1};
        std::vector<TouchLevelPlanLevelRequest> levels;
    };

    struct TouchLevelPlanResult
    {
        std::uint64_t sequence{0};
        std::uint64_t epoch{0};
        std::vector<TouchLevelPlanLevelResult> levels;
    };

    struct TouchLevelCacheState
    {
        FarLodLevelConfig level{};
        int activeOuterRadiusChunks{0};
        glm::ivec3 cameraChunk{0};
        std::vector<FarLodChunkKey> activeKeys;
    };

    struct AtlasUpdateRect
    {
        glm::ivec2 originCell{0};
        glm::ivec2 sizeCells{0};
    };

    struct FarLodWorkBudget
    {
        int activationStepChunks{1};
        int distanceRampStepChunks{0};
        std::size_t newChunkActivations{0};
        std::size_t newFallbackActivations{0};
        std::size_t staleReleaseCount{0};
        std::size_t atlasUpdateCells{0};
        std::size_t gpuDispatchBudgetUnits{0};
        std::size_t maxGpuSubmissions{0};
    };

    struct FarLodLevelAtlasState
    {
        int level{0};
        int blockScale{1};
        glm::ivec2 atlasSizeCells{0};
        glm::ivec2 originCell{0};
        Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> sampleBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> seedHeaderBuffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> seedDataBuffer;
        D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES sampleState{D3D12_RESOURCE_STATE_COMMON};
        glm::ivec2 seedOriginChunk{0};
        glm::ivec2 seedSizeChunks{0};
        D3D12_RESOURCE_STATES seedHeaderState{D3D12_RESOURCE_STATE_COMMON};
        D3D12_RESOURCE_STATES seedDataState{D3D12_RESOURCE_STATE_COMMON};
        bool initialized{false};
        bool seedInitialized{false};
        std::vector<AtlasUpdateRect> pendingDirtyRects;
        std::vector<AtlasUpdateRect> pendingSeedDirtyRects;
        std::unordered_map<std::uint64_t, std::uint64_t> cellRevisions;

        [[nodiscard]] std::uint32_t elementCount() const noexcept
        {
            return atlasSizeCells.x > 0 && atlasSizeCells.y > 0
                       ? static_cast<std::uint32_t>(atlasSizeCells.x * atlasSizeCells.y)
                       : 0u;
        }

        [[nodiscard]] std::uint32_t seedHeaderElementCount() const noexcept
        {
            return seedSizeChunks.x > 0 && seedSizeChunks.y > 0
                       ? static_cast<std::uint32_t>(seedSizeChunks.x * seedSizeChunks.y)
                       : 0u;
        }

        [[nodiscard]] std::uint32_t seedDataElementCount() const noexcept
        {
            return seedHeaderElementCount() * kFarLodChunkSeedCountPerCacheEntry;
        }
    };

    struct VoxelFootprintClassification
    {
        FarLodVoxel voxel{};
        int minSampledSurfaceY{std::numeric_limits<int>::max()};
        int maxSampledSurfaceY{std::numeric_limits<int>::min()};
    };

    struct GreedyMaskCell
    {
        bool visible{false};
        BlockId material{BlockId::Air};
        std::uint8_t flags{0};

        [[nodiscard]] bool mergeEquals(const GreedyMaskCell& other) const noexcept
        {
            return visible == other.visible && material == other.material && flags == other.flags;
        }
    };

    static glm::vec2 projectTileCoord(const glm::vec3& position, const glm::vec3& normal) noexcept
    {
        const glm::vec3 absNormal = glm::abs(normal);
        if (absNormal.y >= absNormal.x && absNormal.y >= absNormal.z)
        {
            return glm::vec2(position.x, position.z);
        }
        if (absNormal.x >= absNormal.z)
        {
            return glm::vec2(position.z, position.y);
        }
        return glm::vec2(position.x, position.y);
    }

    static void appendQuad(std::vector<Vertex>& vertices,
                           std::vector<std::uint32_t>& indices,
                           const glm::vec3& p0,
                           const glm::vec3& p1,
                           const glm::vec3& p2,
                           const glm::vec3& p3,
                           const glm::vec3& normal,
                           const std::pair<glm::vec2, glm::vec2>& uv,
                           std::uint8_t flags = 0)
    {
        const std::uint32_t baseIndex = static_cast<std::uint32_t>(vertices.size());
        const std::uint32_t lightingData = packVertexLighting(packLightLevels(kMaxLightLevel, 0), 0, flags);
        vertices.push_back(Vertex{p0, normal, projectTileCoord(p0, normal), uv.first, uv.second, lightingData});
        vertices.push_back(Vertex{p1, normal, projectTileCoord(p1, normal), uv.first, uv.second, lightingData});
        vertices.push_back(Vertex{p2, normal, projectTileCoord(p2, normal), uv.first, uv.second, lightingData});
        vertices.push_back(Vertex{p3, normal, projectTileCoord(p3, normal), uv.first, uv.second, lightingData});

        indices.push_back(baseIndex + 0);
        indices.push_back(baseIndex + 1);
        indices.push_back(baseIndex + 2);
        indices.push_back(baseIndex + 0);
        indices.push_back(baseIndex + 2);
        indices.push_back(baseIndex + 3);
    }

    static std::size_t nextPowerOfTwo(std::size_t value) noexcept
    {
        if (value <= 1)
        {
            return 1;
        }

        --value;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
#if SIZE_MAX > 0xffffffffu
        value |= value >> 32;
#endif
        return value + 1;
    }

    [[nodiscard]] static std::uint32_t roundReservedFaceCapacity(std::uint32_t faceCount) noexcept
    {
        constexpr std::uint32_t kMinFaceCapacity = 96u;
        faceCount = std::clamp(faceCount, kMinFaceCapacity, kFaceDescriptorCount);
        std::uint32_t rounded = 1u;
        while (rounded < faceCount && rounded < kFaceDescriptorCount)
        {
            rounded <<= 1u;
        }
        return std::min<std::uint32_t>(rounded, kFaceDescriptorCount);
    }

    [[nodiscard]] static std::uint32_t initialReservedFaceCapacity(const FarLodLevelConfig& level) noexcept
    {
        if (level.blockScale <= 4)
        {
            return 1024u;
        }
        if (level.blockScale <= 8)
        {
            return 768u;
        }
        if (level.blockScale <= 16)
        {
            return 512u;
        }
        if (level.blockScale <= 32)
        {
            return 320u;
        }
        if (level.blockScale <= 64)
        {
            return 192u;
        }
        return 128u;
    }

    [[nodiscard]] static std::uint32_t growReservedFaceCapacity(std::uint32_t requiredFaces) noexcept
    {
        const std::uint32_t paddedFaces =
            requiredFaces + std::max<std::uint32_t>(requiredFaces / 2u, 32u);
        return roundReservedFaceCapacity(paddedFaces);
    }

    [[nodiscard]] static std::size_t voxelIndex(int localX, int localY, int localZ) noexcept
    {
        return (static_cast<std::size_t>(localY) * kLogicalSize + static_cast<std::size_t>(localZ)) *
                   kLogicalSize +
               static_cast<std::size_t>(localX);
    }

    [[nodiscard]] static std::uint64_t computeSeamSignature(const FarLodChunkCpu& cpu) noexcept
    {
        constexpr std::uint64_t kOffsetBasis = 1469598103934665603ull;
        constexpr std::uint64_t kPrime = 1099511628211ull;

        auto hashVoxel = [](std::uint64_t hash, const FarLodVoxel& voxel, std::uint8_t faceId) noexcept
        {
            const std::uint32_t packed =
                static_cast<std::uint32_t>(voxel.occupied) |
                (static_cast<std::uint32_t>(voxel.flags) << 8u) |
                (static_cast<std::uint32_t>(toIndex(voxel.material)) << 16u) |
                (static_cast<std::uint32_t>(faceId) << 24u);
            hash ^= static_cast<std::uint64_t>(packed);
            hash *= kPrime;
            return hash;
        };

        std::uint64_t hash = kOffsetBasis;
        for (int localY = 0; localY < kLogicalSize; ++localY)
        {
            for (int localZ = 0; localZ < kLogicalSize; ++localZ)
            {
                hash = hashVoxel(hash, cpu.voxels[voxelIndex(0, localY, localZ)], 0u);
                hash = hashVoxel(hash, cpu.voxels[voxelIndex(kLogicalSize - 1, localY, localZ)], 1u);
            }
        }
        for (int localY = 0; localY < kLogicalSize; ++localY)
        {
            for (int localX = 0; localX < kLogicalSize; ++localX)
            {
                hash = hashVoxel(hash, cpu.voxels[voxelIndex(localX, localY, 0)], 2u);
                hash = hashVoxel(hash, cpu.voxels[voxelIndex(localX, localY, kLogicalSize - 1)], 3u);
            }
        }
        for (int localZ = 0; localZ < kLogicalSize; ++localZ)
        {
            for (int localX = 0; localX < kLogicalSize; ++localX)
            {
                hash = hashVoxel(hash, cpu.voxels[voxelIndex(localX, 0, localZ)], 4u);
                hash = hashVoxel(hash, cpu.voxels[voxelIndex(localX, kLogicalSize - 1, localZ)], 5u);
            }
        }
        return hash;
    }

    [[nodiscard]] static std::uint64_t packAtlasCellKey(int cellX, int cellZ) noexcept
    {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cellX)) << 32u) |
               static_cast<std::uint32_t>(cellZ);
    }

    [[nodiscard]] static std::uint64_t staggeredChunkUpdateHash(const FarLodChunkKey& key) noexcept
    {
        std::uint64_t hash = 1469598103934665603ull;
        const auto mix = [&hash](std::uint64_t value) noexcept
        {
            hash ^= value;
            hash *= 1099511628211ull;
        };
        mix(static_cast<std::uint32_t>(key.level));
        mix(static_cast<std::uint32_t>(key.coord.x));
        mix(static_cast<std::uint32_t>(key.coord.y));
        mix(static_cast<std::uint32_t>(key.coord.z));
        return hash;
    }

    [[nodiscard]] static int atlasOriginSnapStrideChunks(const FarLodLevelConfig& level) noexcept
    {
        return level.level >= 3 ? 2 : 1;
    }

    [[nodiscard]] static std::uint64_t residentRefreshCadenceUpdates(const FarLodLevelConfig& level) noexcept
    {
        if (level.level >= 5)
        {
            return 4u;
        }
        if (level.level >= 4)
        {
            return 3u;
        }
        if (level.level >= 3)
        {
            return 2u;
        }
        return 1u;
    }

    [[nodiscard]] static PackedFarLodVoxelGpu packGpuVoxel(const FarLodVoxel& voxel) noexcept
    {
        PackedFarLodVoxelGpu packed = 0u;
        if (voxel.occupied)
        {
            packed |= 0x1u;
        }
        if ((voxel.flags & 0x01u) != 0u)
        {
            packed |= 0x2u;
        }
        if ((voxel.flags & 0x02u) != 0u)
        {
            packed |= 0x4u;
        }
        if ((voxel.flags & 0x04u) != 0u)
        {
            packed |= 0x8u;
        }
        if ((voxel.flags & 0x08u) != 0u)
        {
            packed |= 0x10u;
        }
        packed |= (static_cast<PackedFarLodVoxelGpu>(toIndex(voxel.material) & 0xffu) << 8u);
        return packed;
    }

    [[nodiscard]] static FarLodVoxel unpackGpuVoxel(PackedFarLodVoxelGpu packed) noexcept
    {
        FarLodVoxel voxel{};
        voxel.occupied = (packed & 0x1u) ? 1u : 0u;
        voxel.material = static_cast<BlockId>((packed >> 8u) & 0xffu);
        if ((packed & 0x2u) != 0u)
        {
            voxel.flags |= 0x01u;
        }
        if ((packed & 0x4u) != 0u)
        {
            voxel.flags |= 0x02u;
        }
        if ((packed & 0x8u) != 0u)
        {
            voxel.flags |= 0x04u;
        }
        if ((packed & 0x10u) != 0u)
        {
            voxel.flags |= 0x08u;
        }
        return voxel;
    }

    [[nodiscard]] static bool intersectsRange(int minA, int maxA, int minB, int maxB) noexcept
    {
        return minA <= maxB && minB <= maxA;
    }

    [[nodiscard]] static BlockFace blockFaceForNormal(const glm::ivec3& normal) noexcept
    {
        if (normal.x > 0) return BlockFace::East;
        if (normal.x < 0) return BlockFace::West;
        if (normal.y > 0) return BlockFace::Top;
        if (normal.y < 0) return BlockFace::Bottom;
        if (normal.z > 0) return BlockFace::South;
        return BlockFace::North;
    }

    [[nodiscard]] static int structureMaterialPriority(BlockId block) noexcept
    {
        if (block == BlockId::SpruceLog || block == BlockId::Wood ||
            block == BlockId::DarkOakLog || block == BlockId::BirchLog ||
            block == BlockId::AcaciaLog)
        {
            return 4;
        }
        if (block == BlockId::SpruceLeaves || block == BlockId::Leaves ||
            block == BlockId::DarkOakLeaves || block == BlockId::BirchLeaves ||
            block == BlockId::AcaciaLeaves)
        {
            return 3;
        }
        if (block == BlockId::Water)
        {
            return 1;
        }
        if (block != BlockId::Air)
        {
            return 2;
        }
        return 0;
    }

    [[nodiscard]] static int chunkMinHorizontalRingDistanceChunks(const FarLodLevelConfig& level,
                                                                  const glm::ivec3& cameraChunk,
                                                                  const glm::ivec3& chunkCoord) noexcept
    {
        const glm::ivec2 minCoord(chunkCoord.x * level.blockScale, chunkCoord.z * level.blockScale);
        const glm::ivec2 maxCoord = minCoord + glm::ivec2(level.blockScale - 1);
        const auto minAxisDistance = [](int center, int minValue, int maxValue) noexcept
        {
            if (center < minValue) return minValue - center;
            if (center > maxValue) return center - maxValue;
            return 0;
        };

        const int minDistance = std::max(minAxisDistance(cameraChunk.x, minCoord.x, maxCoord.x),
                                         minAxisDistance(cameraChunk.z, minCoord.y, maxCoord.y));
        return std::max(0, minDistance - level.innerRadiusChunks);
    }

    [[nodiscard]] static int ceilDivPositive(int value, int divisor) noexcept
    {
        return -floorDiv(-value, divisor);
    }

    [[nodiscard]] static std::vector<HorizontalRingSpan> buildHorizontalRingSpans(const FarLodLevelConfig& level,
                                                                                   const glm::ivec3& cameraChunk,
                                                                                   int outerRadiusChunks)
    {
        std::vector<HorizontalRingSpan> spans;
        if (outerRadiusChunks <= level.innerRadiusChunks || level.blockScale <= 0)
        {
            return spans;
        }

        const int blockScale = level.blockScale;
        const int outerMinChunkX = floorDiv(cameraChunk.x - outerRadiusChunks, blockScale);
        const int outerMaxChunkX = floorDiv(cameraChunk.x + outerRadiusChunks, blockScale);
        const int outerMinChunkZ = floorDiv(cameraChunk.z - outerRadiusChunks, blockScale);
        const int outerMaxChunkZ = floorDiv(cameraChunk.z + outerRadiusChunks, blockScale);

        spans.reserve(static_cast<std::size_t>(std::max(outerMaxChunkZ - outerMinChunkZ + 1, 0) * 2));

        const auto minAxisDistance = [](int center, int minValue, int maxValue) noexcept
        {
            if (center < minValue) return minValue - center;
            if (center > maxValue) return center - maxValue;
            return 0;
        };
        const auto maxAxisDistance = [](int center, int minValue, int maxValue) noexcept
        {
            return std::max(std::abs(center - minValue), std::abs(center - maxValue));
        };

        for (int chunkZ = outerMinChunkZ; chunkZ <= outerMaxChunkZ; ++chunkZ)
        {
            const int zMin = chunkZ * blockScale;
            const int zMax = zMin + blockScale - 1;
            const int zMinDistance = minAxisDistance(cameraChunk.z, zMin, zMax);
            if (zMinDistance > outerRadiusChunks)
            {
                continue;
            }

            const int zMaxDistance = maxAxisDistance(cameraChunk.z, zMin, zMax);
            if (zMaxDistance > level.innerRadiusChunks)
            {
                spans.push_back(HorizontalRingSpan{chunkZ, outerMinChunkX, outerMaxChunkX});
                continue;
            }

            const int fullyInsideMinChunkX =
                ceilDivPositive(cameraChunk.x - level.innerRadiusChunks, blockScale);
            const int fullyInsideMaxChunkX =
                floorDiv(cameraChunk.x + level.innerRadiusChunks - (blockScale - 1), blockScale);
            if (fullyInsideMinChunkX > fullyInsideMaxChunkX)
            {
                spans.push_back(HorizontalRingSpan{chunkZ, outerMinChunkX, outerMaxChunkX});
                continue;
            }

            if (outerMinChunkX < fullyInsideMinChunkX)
            {
                spans.push_back(HorizontalRingSpan{
                    chunkZ,
                    outerMinChunkX,
                    std::min(outerMaxChunkX, fullyInsideMinChunkX - 1)});
            }
            if (fullyInsideMaxChunkX < outerMaxChunkX)
            {
                spans.push_back(HorizontalRingSpan{
                    chunkZ,
                    std::max(outerMinChunkX, fullyInsideMaxChunkX + 1),
                    outerMaxChunkX});
            }
        }

        return spans;
    }

    [[nodiscard]] std::vector<FarLodChunkKey> buildTouchLevelActiveKeys(const TouchLevelPlanLevelRequest& levelRequest,
                                                                        const glm::ivec3& cameraChunk,
                                                                        std::size_t workerBudget) const
    {
        const std::vector<HorizontalRingSpan> spans =
            buildHorizontalRingSpans(levelRequest.level, cameraChunk, levelRequest.activeOuterRadiusChunks);
        if (spans.empty())
        {
            return {};
        }

        std::size_t totalKeyCount = 0;
        for (const HorizontalRingSpan& span : spans)
        {
            totalKeyCount += static_cast<std::size_t>(std::max(span.maxChunkX - span.minChunkX + 1, 0));
        }

        const std::size_t threadCount =
            std::min<std::size_t>(std::max<std::size_t>(workerBudget, 1), spans.size());
        std::vector<std::vector<FarLodChunkKey>> threadKeys(threadCount);
        std::vector<std::thread> workers;
        workers.reserve(threadCount > 0 ? threadCount - 1 : 0);

        auto buildRange = [&](std::size_t threadIndex, std::size_t beginSpan, std::size_t endSpan)
        {
            std::size_t localCount = 0;
            for (std::size_t spanIndex = beginSpan; spanIndex < endSpan; ++spanIndex)
            {
                localCount += static_cast<std::size_t>(
                    std::max(spans[spanIndex].maxChunkX - spans[spanIndex].minChunkX + 1, 0));
            }

            std::vector<FarLodChunkKey>& out = threadKeys[threadIndex];
            out.reserve(localCount);
            for (std::size_t spanIndex = beginSpan; spanIndex < endSpan; ++spanIndex)
            {
                const HorizontalRingSpan& span = spans[spanIndex];
                for (int chunkX = span.minChunkX; chunkX <= span.maxChunkX; ++chunkX)
                {
                    out.push_back(FarLodChunkKey{
                        levelRequest.level.level,
                        glm::ivec3(chunkX, 0, span.chunkZ)});
                }
            }
        };

        const std::size_t spansPerThread = (spans.size() + threadCount - 1) / threadCount;
        for (std::size_t threadIndex = 1; threadIndex < threadCount; ++threadIndex)
        {
            const std::size_t beginSpan = threadIndex * spansPerThread;
            const std::size_t endSpan = std::min(spans.size(), beginSpan + spansPerThread);
            workers.emplace_back(buildRange, threadIndex, beginSpan, endSpan);
        }

        buildRange(0u, 0u, std::min(spans.size(), spansPerThread));
        for (std::thread& worker : workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        std::vector<FarLodChunkKey> keys;
        keys.reserve(totalKeyCount);
        for (std::vector<FarLodChunkKey>& local : threadKeys)
        {
            keys.insert(keys.end(),
                        std::make_move_iterator(local.begin()),
                        std::make_move_iterator(local.end()));
        }
        return keys;
    }

    [[nodiscard]] TouchLevelPlanResult buildTouchLevelPlan(const TouchLevelPlanRequest& request) const
    {
        TouchLevelPlanResult result{};
        result.sequence = request.sequence;
        result.epoch = request.epoch;
        result.levels.reserve(request.levels.size());
        for (const TouchLevelPlanLevelRequest& levelRequest : request.levels)
        {
            TouchLevelPlanLevelResult levelResult{};
            levelResult.level = levelRequest.level;
            levelResult.activeOuterRadiusChunks = levelRequest.activeOuterRadiusChunks;
            levelResult.activeKeys = buildTouchLevelActiveKeys(levelRequest, request.cameraChunk, request.workerBudget);
            result.levels.push_back(std::move(levelResult));
        }
        return result;
    }

    [[nodiscard]] const FarLodLevelConfig* nextCoarserLevel(int levelId) const noexcept
    {
        for (std::size_t i = 0; i < levels_.size(); ++i)
        {
            if (levels_[i].level == levelId)
            {
                return (i + 1 < levels_.size()) ? &levels_[i + 1] : nullptr;
            }
        }
        return nullptr;
    }

    void updateChunkConservativeBounds(FarLodChunkRecord& chunk,
                                       const FarLodLevelConfig& level,
                                       int verticalBandMinWorldY,
                                       int verticalBandMaxWorldY) const
    {
        constexpr int kFarStructureHeadroomBlocks = 48;
        const int conservativeMinWorldY =
            std::min(verticalBandMinWorldY - level.blockScale,
                     seaLevel_ - (level.chunkSpanBlocks() + level.blockScale));
        const int conservativeMaxWorldY =
            std::max(verticalBandMaxWorldY + level.blockScale + kFarStructureHeadroomBlocks,
                     seaLevel_ + level.blockScale);
        chunk.cpu.boundsMin = glm::vec3(chunk.cpu.worldMin.x,
                                        static_cast<float>(conservativeMinWorldY),
                                        chunk.cpu.worldMin.z);
        chunk.cpu.boundsMax = glm::vec3(chunk.cpu.worldMin.x + level.chunkSpanBlocks(),
                                        static_cast<float>(conservativeMaxWorldY + 1),
                                        chunk.cpu.worldMin.z + level.chunkSpanBlocks());
    }

    void requestTouchLevelPlan(const glm::ivec3& cameraChunk)
    {
        std::vector<std::pair<int, int>> levelState;
        levelState.reserve(levels_.size());
        TouchLevelPlanRequest request{};
        request.sequence = touchLevelPlanSequence_.fetch_add(1, std::memory_order_acq_rel) + 1u;
        request.epoch = buildEpoch_.load(std::memory_order_acquire);
        request.cameraChunk = cameraChunk;
        request.workerBudget = std::max<std::size_t>(workerCount_, 1);
        request.levels.reserve(levels_.size());

        for (const FarLodLevelConfig& level : levels_)
        {
            if (level.outerRadiusChunks <= level.innerRadiusChunks)
            {
                continue;
            }
            const auto it = levelActivationOuterRadiusChunks_.find(level.level);
            const int activeOuterRadiusChunks =
                (it != levelActivationOuterRadiusChunks_.end()) ? it->second : level.innerRadiusChunks;
            levelState.emplace_back(level.level, activeOuterRadiusChunks);
            request.levels.push_back(TouchLevelPlanLevelRequest{level, activeOuterRadiusChunks});
        }

        if (hasLastTouchLevelPlanRequest_ &&
            lastTouchLevelPlanCameraChunk_.x == cameraChunk.x &&
            lastTouchLevelPlanCameraChunk_.z == cameraChunk.z &&
            lastTouchLevelPlanLevels_ == levelState)
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(touchLevelPlanMutex_);
            pendingTouchLevelPlanRequest_ = std::move(request);
            hasPendingTouchLevelPlanRequest_ = true;
        }
        touchLevelPlanCv_.notify_one();
        lastTouchLevelPlanCameraChunk_ = cameraChunk;
        lastTouchLevelPlanLevels_ = std::move(levelState);
        hasLastTouchLevelPlanRequest_ = true;
    }

    void consumeReadyTouchLevelPlan()
    {
        TouchLevelPlanResult result{};
        {
            std::lock_guard<std::mutex> lock(touchLevelPlanMutex_);
            if (!hasReadyTouchLevelPlanResult_)
            {
                return;
            }
            result = std::move(readyTouchLevelPlanResult_);
            readyTouchLevelPlanResult_ = {};
            hasReadyTouchLevelPlanResult_ = false;
        }

        if (result.epoch != buildEpoch_.load(std::memory_order_acquire))
        {
            return;
        }

        for (TouchLevelPlanLevelResult& levelResult : result.levels)
        {
            TouchLevelCacheState& cache = touchLevelCaches_[levelResult.level.level];
            cache.level = levelResult.level;
            cache.activeOuterRadiusChunks = levelResult.activeOuterRadiusChunks;
            cache.cameraChunk = lastTouchLevelPlanCameraChunk_;
            cache.activeKeys = std::move(levelResult.activeKeys);
        }
    }

    void applyTouchLevelCache(const FarLodLevelConfig& level,
                              std::size_t& remainingNewChunkActivations,
                              std::size_t& remainingNewFallbackActivations)
    {
        const auto cacheIt = touchLevelCaches_.find(level.level);
        if (cacheIt == touchLevelCaches_.end())
        {
            return;
        }

        constexpr int kFarVerticalHeadroomChunks = 6;
        const int verticalRadiusChunks =
            std::max(gActiveVerticalRadius.load(std::memory_order_relaxed) + kFarVerticalHeadroomChunks, 8);
        const int chunkMinY = floorDiv(cameraChunk_.y - verticalRadiusChunks, level.blockScale);
        const int chunkMaxY = floorDiv(cameraChunk_.y + verticalRadiusChunks, level.blockScale);
        const int span = level.chunkSpanBlocks();
        const int verticalBandMinWorldY = chunkMinY * span;
        const int verticalBandMaxWorldY = ((chunkMaxY + 1) * span) - 1;

        std::vector<FarLodChunkKey> fallbackParents;
        fallbackParents.reserve(cacheIt->second.activeKeys.size() / 2 + 1);
        std::unordered_set<FarLodChunkKey, FarLodChunkKeyHasher> uniqueFallbackParents;

        for (const FarLodChunkKey& key : cacheIt->second.activeKeys)
        {
            auto chunkIt = chunks_.find(key);
            const bool chunkMissing = (chunkIt == chunks_.end());
            if (chunkMissing && remainingNewChunkActivations == 0)
            {
                continue;
            }

            FarLodChunkRecord& chunk = chunkMissing ? chunks_[key] : chunkIt->second;
            if (chunkMissing)
            {
                --remainingNewChunkActivations;
            }

            chunk.key = key;
            chunk.level = level;
            chunk.lastTouchedStamp = updateStamp_;
            chunk.active = true;
            chunk.fallbackOnly = false;
            if (!chunk.initialized)
            {
                initializeChunk(chunk, level);
                chunk.dirty = true;
            }
            updateChunkConservativeBounds(chunk, level, verticalBandMinWorldY, verticalBandMaxWorldY);
            if (!(chunk.gpu.resident && chunk.gpu.indexCount > 0) && !chunk.dirty && !chunk.inFlight)
            {
                markDirty(chunk);
            }

            if (chunk.gpu.resident && chunk.gpu.indexCount > 0)
            {
                continue;
            }

            const FarLodLevelConfig* parentLevel = nextCoarserLevel(level.level);
            if (parentLevel == nullptr)
            {
                continue;
            }

            const glm::ivec3 baseMin = key.coord * level.blockScale;
            const glm::ivec3 parentCoord{
                floorDiv(baseMin.x, parentLevel->blockScale),
                floorDiv(baseMin.y, parentLevel->blockScale),
                floorDiv(baseMin.z, parentLevel->blockScale)};
            FarLodChunkKey parentKey{parentLevel->level, parentCoord};
            if (uniqueFallbackParents.insert(parentKey).second)
            {
                fallbackParents.push_back(parentKey);
            }
        }

        for (const FarLodChunkKey& parentKey : fallbackParents)
        {
            auto parentIt = chunks_.find(parentKey);
            const bool parentMissing = (parentIt == chunks_.end());
            if (parentMissing && remainingNewFallbackActivations == 0)
            {
                continue;
            }

            const FarLodLevelConfig* parentLevel = nextCoarserLevel(level.level);
            if (parentLevel == nullptr)
            {
                continue;
            }

            FarLodChunkRecord& parent = parentMissing ? chunks_[parentKey] : parentIt->second;
            if (parentMissing)
            {
                --remainingNewFallbackActivations;
            }

            const bool alreadyTouchedThisUpdate = (parent.lastTouchedStamp == updateStamp_);
            parent.key = parentKey;
            parent.level = *parentLevel;
            parent.lastTouchedStamp = updateStamp_;
            parent.active = true;
            if (!(alreadyTouchedThisUpdate && !parent.fallbackOnly))
            {
                parent.fallbackOnly = true;
            }
            if (!parent.initialized)
            {
                initializeChunk(parent, *parentLevel);
                parent.dirty = true;
            }
            else if (!(parent.gpu.resident && parent.gpu.indexCount > 0) && !parent.dirty)
            {
                markDirty(parent);
            }
        }
    }

    void initializeChunk(FarLodChunkRecord& chunk, const FarLodLevelConfig& level)
    {
        chunk.level = level;
        chunk.cpu.key = chunk.key;
        chunk.cpu.blockScale = level.blockScale;
        chunk.cpu.worldMin = chunk.key.coord * level.chunkSpanBlocks();
        chunk.cpu.boundsMin = glm::vec3(chunk.cpu.worldMin);
        chunk.cpu.boundsMax = glm::vec3(chunk.cpu.worldMin + glm::ivec3(level.chunkSpanBlocks()));
        chunk.residentBoundsMin = chunk.cpu.boundsMin;
        chunk.residentBoundsMax = chunk.cpu.boundsMax;
        chunk.pendingMesh = {};
        chunk.fallbackOnly = false;
        chunk.lastBuiltAtlasDependencyRevision = 0;
        chunk.gpu.columnBuffer = createDefaultBuffer(device_.Get(),
                                                     static_cast<std::uint64_t>(kLogicalSize * kLogicalSize *
                                                                                sizeof(GpuTerrainColumnDescriptor)),
                                                     D3D12_RESOURCE_STATE_COMMON,
                                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        chunk.gpu.faceCountBuffer = createDefaultBuffer(device_.Get(),
                                                        static_cast<std::uint64_t>(kPlaneCount * sizeof(std::uint32_t)),
                                                        D3D12_RESOURCE_STATE_COMMON,
                                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        chunk.gpu.faceAnalysisBuffer = createDefaultBuffer(device_.Get(),
                                                           static_cast<std::uint64_t>(kFaceMetadataEntryCount *
                                                                                      sizeof(std::uint32_t)),
                                                           D3D12_RESOURCE_STATE_COMMON,
                                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        chunk.gpu.faceDescriptorBuffer = createDefaultBuffer(device_.Get(),
                                                             static_cast<std::uint64_t>(kFaceDescriptorCount * sizeof(std::uint32_t) * 4u),
                                                             D3D12_RESOURCE_STATE_COMMON,
                                                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        chunk.gpu.facePrefixBuffer = createDefaultBuffer(device_.Get(),
                                                         static_cast<std::uint64_t>(kPlaneCount * sizeof(std::uint32_t)),
                                                         D3D12_RESOURCE_STATE_COMMON,
                                                         D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        chunk.gpu.faceGroupSumBuffer = createDefaultBuffer(device_.Get(),
                                                           static_cast<std::uint64_t>(kFacePrefixGroupCount * sizeof(std::uint32_t)),
                                                           D3D12_RESOURCE_STATE_COMMON,
                                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (chunk.gpu.columnBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodColumns_L" << chunk.key.level
                 << L"_" << chunk.key.coord.x
                 << L"_" << chunk.key.coord.y
                 << L"_" << chunk.key.coord.z;
            setDebugObjectName(chunk.gpu.columnBuffer.Get(), name.str());
        }
        if (chunk.gpu.faceCountBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodFaceCount_L" << chunk.key.level
                 << L"_" << chunk.key.coord.x
                 << L"_" << chunk.key.coord.y
                 << L"_" << chunk.key.coord.z;
            setDebugObjectName(chunk.gpu.faceCountBuffer.Get(), name.str());
        }
        if (chunk.gpu.faceAnalysisBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodFaceAnalysis_L" << chunk.key.level
                 << L"_" << chunk.key.coord.x
                 << L"_" << chunk.key.coord.y
                 << L"_" << chunk.key.coord.z;
            setDebugObjectName(chunk.gpu.faceAnalysisBuffer.Get(), name.str());
        }
        if (chunk.gpu.faceDescriptorBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodFaceDescriptor_L" << chunk.key.level
                 << L"_" << chunk.key.coord.x
                 << L"_" << chunk.key.coord.y
                 << L"_" << chunk.key.coord.z;
            setDebugObjectName(chunk.gpu.faceDescriptorBuffer.Get(), name.str());
        }
        if (chunk.gpu.facePrefixBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodFacePrefix_L" << chunk.key.level
                 << L"_" << chunk.key.coord.x
                 << L"_" << chunk.key.coord.y
                 << L"_" << chunk.key.coord.z;
            setDebugObjectName(chunk.gpu.facePrefixBuffer.Get(), name.str());
        }
        if (chunk.gpu.faceGroupSumBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodFaceGroupSum_L" << chunk.key.level
                 << L"_" << chunk.key.coord.x
                 << L"_" << chunk.key.coord.y
                 << L"_" << chunk.key.coord.z;
            setDebugObjectName(chunk.gpu.faceGroupSumBuffer.Get(), name.str());
        }
        chunk.gpu.columnState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceCountState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceAnalysisState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceDescriptorState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.facePrefixState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceGroupSumState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.voxelFenceValue = 0;
        chunk.gpu.readbackFenceValue = 0;
        chunk.gpu.voxelReady = false;
        chunk.gpu.parityMismatchCount = 0;
        chunk.gpu.parityValidated = false;
        chunk.gpu.reservedVertexCount = 0;
        chunk.gpu.reservedIndexCount = 0;
        chunk.gpu.faceCapacity = 0;
        chunk.gpu.recordIndex = kInvalidFarDrawRecordIndex;
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD chunk initialized level=" << chunk.key.level
                   << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                   << " blockScale=" << level.blockScale
                   << " span=" << level.chunkSpanBlocks();
            chunkManagerDebugLog(stream.str());
        }
        chunk.initialized = true;
    }

    void markDirty(FarLodChunkRecord& chunk)
    {
        chunk.dirty = true;
        ++chunk.buildVersion;
        if (chunk.buildVersion == 0)
        {
            chunk.buildVersion = 1;
        }
    }

    [[nodiscard]] static std::uint32_t committedFaceCount(const FarLodChunkGpuState& gpu) noexcept
    {
        return (gpu.indexCount > 0u) ? (gpu.indexCount / 6u) : 0u;
    }

    [[nodiscard]] static std::uint32_t requestedFaceCapacity(const FarLodChunkRecord& chunk) noexcept
    {
        std::uint32_t requested = initialReservedFaceCapacity(chunk.level);
        requested = std::max(requested, chunk.gpu.faceCapacity);
        requested = std::max(requested, chunk.requestedFaceCapacityHint);
        const std::uint32_t actualFaces = committedFaceCount(chunk.gpu);
        if (actualFaces > 0u)
        {
            requested = std::max(requested, growReservedFaceCapacity(actualFaces));
        }
        return std::min<std::uint32_t>(roundReservedFaceCapacity(requested), kFaceDescriptorCount);
    }

    void markSameLevelNeighborSeamsDirty(const FarLodChunkKey& key)
    {
        static constexpr std::array<glm::ivec3, 6> kNeighborOffsets{{
            glm::ivec3( 1,  0,  0),
            glm::ivec3(-1,  0,  0),
            glm::ivec3( 0,  1,  0),
            glm::ivec3( 0, -1,  0),
            glm::ivec3( 0,  0,  1),
            glm::ivec3( 0,  0, -1),
        }};

        for (const glm::ivec3& offset : kNeighborOffsets)
        {
            const FarLodChunkKey neighborKey{key.level, key.coord + offset};
            const auto it = chunks_.find(neighborKey);
            if (it == chunks_.end())
            {
                continue;
            }

            FarLodChunkRecord& neighbor = it->second;
            if (!neighbor.initialized || !neighbor.active)
            {
                continue;
            }

            if (!neighbor.dirty)
            {
                markDirty(neighbor);
            }
        }
    }

    BufferPage createBufferPage(std::size_t vertexCount, std::size_t indexCount)
    {
        static constexpr std::size_t kDefaultVertexCapacity = 131072;
        static constexpr std::size_t kDefaultIndexCapacity = 196608;

        BufferPage page;
        page.vertexCapacity = std::max(nextPowerOfTwo(vertexCount), kDefaultVertexCapacity);
        page.indexCapacity = std::max(nextPowerOfTwo(indexCount), kDefaultIndexCapacity);
        page.recordCapacity = std::max<std::size_t>(
            1,
            std::min(page.vertexCapacity / 4u, page.indexCapacity / 6u));
        page.vertexBuffer = createDefaultBuffer(device_.Get(),
                                                static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex)),
                                                D3D12_RESOURCE_STATE_COMMON,
                                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        page.indexBuffer = createDefaultBuffer(device_.Get(),
                                               static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t)),
                                               D3D12_RESOURCE_STATE_COMMON,
                                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        page.vertexUploadBuffer = createUploadBuffer(device_.Get(),
                                                     static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex)),
                                                     page.mappedVertexData);
        page.indexUploadBuffer = createUploadBuffer(device_.Get(),
                                                    static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t)),
                                                    page.mappedIndexData);
        page.drawRecordBuffer = createDefaultBuffer(device_.Get(),
                                                    static_cast<std::uint64_t>(page.recordCapacity * sizeof(ChunkRenderBatch::GpuCullRecord)),
                                                    D3D12_RESOURCE_STATE_COMMON,
                                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        page.vertexView.BufferLocation = page.vertexBuffer ? page.vertexBuffer->GetGPUVirtualAddress() : 0;
        page.vertexView.StrideInBytes = sizeof(Vertex);
        page.vertexView.SizeInBytes = static_cast<UINT>(page.vertexCapacity * sizeof(Vertex));
        page.indexView.BufferLocation = page.indexBuffer ? page.indexBuffer->GetGPUVirtualAddress() : 0;
        page.indexView.SizeInBytes = static_cast<UINT>(page.indexCapacity * sizeof(std::uint32_t));
        page.indexView.Format = DXGI_FORMAT_R32_UINT;
        return page;
    }

    static bool tryAllocateRange(std::vector<BufferPage::Range>& ranges,
                                 std::size_t& cursor,
                                 std::size_t capacity,
                                 std::size_t count,
                                 std::size_t& outOffset)
    {
        if (count == 0)
        {
            outOffset = cursor;
            return true;
        }

        for (auto it = ranges.begin(); it != ranges.end(); ++it)
        {
            if (it->size >= count)
            {
                outOffset = it->offset;
                it->offset += count;
                it->size -= count;
                if (it->size == 0)
                {
                    ranges.erase(it);
                }
                return true;
            }
        }

        if (cursor + count <= capacity)
        {
            outOffset = cursor;
            cursor += count;
            return true;
        }

        return false;
    }

    static void mergeRange(std::vector<BufferPage::Range>& ranges, std::size_t offset, std::size_t size)
    {
        if (size == 0)
        {
            return;
        }

        BufferPage::Range range{offset, size};
        auto it = std::lower_bound(ranges.begin(), ranges.end(), range.offset,
                                   [](const BufferPage::Range& lhs, std::size_t value)
                                   {
                                       return lhs.offset < value;
                                   });
        it = ranges.insert(it, range);

        if (it != ranges.begin())
        {
            auto prev = std::prev(it);
            if (prev->offset + prev->size == it->offset)
            {
                prev->size += it->size;
                it = ranges.erase(it);
                it = prev;
            }
        }

        auto next = std::next(it);
        if (next != ranges.end() && it->offset + it->size == next->offset)
        {
            it->size += next->size;
            ranges.erase(next);
        }
    }

    Allocation acquireAllocation(std::size_t vertexCount, std::size_t indexCount)
    {
        Allocation allocation{};
        if (vertexCount == 0 || indexCount == 0)
        {
            return allocation;
        }

        for (std::uint32_t pageIndex = 0; pageIndex < bufferPages_.size(); ++pageIndex)
        {
            BufferPage& page = bufferPages_[pageIndex];
            std::size_t vertexOffset = 0;
            if (!tryAllocateRange(page.freeVertices, page.vertexCursor, page.vertexCapacity, vertexCount, vertexOffset))
            {
                continue;
            }

            std::size_t indexOffset = 0;
            if (!tryAllocateRange(page.freeIndices, page.indexCursor, page.indexCapacity, indexCount, indexOffset))
            {
                mergeRange(page.freeVertices, vertexOffset, vertexCount);
                continue;
            }

            std::size_t recordOffset = 0;
            if (!tryAllocateRange(page.freeRecords, page.recordCursor, page.recordCapacity, 1, recordOffset))
            {
                mergeRange(page.freeVertices, vertexOffset, vertexCount);
                mergeRange(page.freeIndices, indexOffset, indexCount);
                continue;
            }

            allocation.pageIndex = pageIndex;
            allocation.vertexOffset = vertexOffset;
            allocation.indexOffset = indexOffset;
            allocation.recordIndex = static_cast<std::uint32_t>(recordOffset);
            return allocation;
        }

        BufferPage newPage = createBufferPage(vertexCount, indexCount);
        bufferPages_.push_back(std::move(newPage));
        BufferPage& page = bufferPages_.back();
        allocation.pageIndex = static_cast<std::uint32_t>(bufferPages_.size() - 1);
        tryAllocateRange(page.freeVertices, page.vertexCursor, page.vertexCapacity, vertexCount, allocation.vertexOffset);
        tryAllocateRange(page.freeIndices, page.indexCursor, page.indexCapacity, indexCount, allocation.indexOffset);
        std::size_t recordOffset = 0;
        tryAllocateRange(page.freeRecords, page.recordCursor, page.recordCapacity, 1, recordOffset);
        allocation.recordIndex = static_cast<std::uint32_t>(recordOffset);
        return allocation;
    }

    void releaseAllocationRange(std::uint32_t pageIndex,
                                std::size_t vertexOffset,
                                std::size_t vertexCount,
                                std::size_t indexOffset,
                                std::uint32_t indexCount,
                                std::uint32_t recordIndex) noexcept
    {
        if (pageIndex == kInvalidChunkBufferPage)
        {
            return;
        }
        if (pageIndex >= bufferPages_.size())
        {
            return;
        }
        BufferPage& page = bufferPages_[pageIndex];
        mergeRange(page.freeVertices, vertexOffset, vertexCount);
        mergeRange(page.freeIndices, indexOffset, static_cast<std::size_t>(indexCount));
        if (recordIndex != kInvalidFarDrawRecordIndex)
        {
            mergeRange(page.freeRecords, recordIndex, 1);
        }
    }

    void clearDrawRecord(BufferPage& page, std::uint32_t recordIndex)
    {
        if (recordIndex == kInvalidFarDrawRecordIndex || page.drawRecordBuffer == nullptr || device_ == nullptr)
        {
            return;
        }
        if (!uploadContext_.ready())
        {
            return;
        }

        std::byte* mapped = nullptr;
        Microsoft::WRL::ComPtr<ID3D12Resource> upload = createUploadBuffer(
            device_.Get(),
            static_cast<std::uint64_t>(sizeof(ChunkRenderBatch::GpuCullRecord)),
            mapped);
        if (upload == nullptr || mapped == nullptr)
        {
            return;
        }
        std::memset(mapped, 0, sizeof(ChunkRenderBatch::GpuCullRecord));

        if (!uploadContext_.begin())
        {
            upload->Unmap(0, nullptr);
            return;
        }

        uploadContext_.copyBuffer(page.drawRecordBuffer.Get(),
                                  static_cast<std::uint64_t>(recordIndex * sizeof(ChunkRenderBatch::GpuCullRecord)),
                                  upload.Get(),
                                  0,
                                  static_cast<std::uint64_t>(sizeof(ChunkRenderBatch::GpuCullRecord)));
        uploadContext_.flush(nullptr);
        uploadContext_.waitForIdle();
        upload->Unmap(0, nullptr);
    }

    static void releasePendingDrawRecordReadback(FarLodChunkRecord::PendingRenderMesh& pendingMesh) noexcept
    {
        if (pendingMesh.drawRecordReadbackBuffer != nullptr && pendingMesh.mappedDrawRecord != nullptr)
        {
            pendingMesh.drawRecordReadbackBuffer->Unmap(0, nullptr);
        }
        pendingMesh.drawRecordReadbackBuffer.Reset();
        pendingMesh.mappedDrawRecord = nullptr;
    }

    void releasePendingMeshAllocation(FarLodChunkRecord& chunk)
    {
        if (!chunk.pendingMesh.valid())
        {
            return;
        }

        if (chunk.pendingMesh.uploadFenceValue != 0)
        {
            const UINT64 completedUploadFenceValue = uploadContext_.completedFenceValue();
            if (completedUploadFenceValue < chunk.pendingMesh.uploadFenceValue)
            {
                uploadContext_.waitForIdle();
            }
        }
        if (chunk.pendingMesh.gpuGenerated && chunk.pendingMesh.gpuFenceValue != 0)
        {
            const UINT64 completedGpuFenceValue = gpuContext_.completedFenceValue();
            if (completedGpuFenceValue < chunk.pendingMesh.gpuFenceValue)
            {
                gpuContext_.waitForFence(chunk.pendingMesh.gpuFenceValue);
            }
        }

        if (chunk.pendingMesh.pageIndex < bufferPages_.size() && chunk.pendingMesh.recordIndex != kInvalidFarDrawRecordIndex)
        {
            BufferPage& page = bufferPages_[chunk.pendingMesh.pageIndex];
            clearDrawRecord(page, chunk.pendingMesh.recordIndex);
        }

        releaseAllocationRange(chunk.pendingMesh.pageIndex,
                               chunk.pendingMesh.vertexOffset,
                               chunk.pendingMesh.vertexCount,
                               chunk.pendingMesh.indexOffset,
                               chunk.pendingMesh.indexCount,
                               chunk.pendingMesh.recordIndex);
        releasePendingDrawRecordReadback(chunk.pendingMesh);
        chunk.pendingMesh = {};
    }

    void releaseChunkRenderAllocation(FarLodChunkRecord& chunk)
    {
        if (chunk.gpu.pageIndex == kInvalidChunkBufferPage)
        {
            chunk.gpu.vertexOffset = 0;
            chunk.gpu.indexOffset = 0;
            chunk.gpu.vertexCount = 0;
            chunk.gpu.indexCount = 0;
            chunk.gpu.reservedVertexCount = 0;
            chunk.gpu.reservedIndexCount = 0;
            chunk.gpu.resident = false;
            return;
        }

        const UINT64 completedUploadFenceValue = uploadContext_.completedFenceValue();
        const UINT64 submittedUploadFenceValue = uploadContext_.lastSubmittedFenceValue();
        if (submittedUploadFenceValue != 0 && completedUploadFenceValue < submittedUploadFenceValue)
        {
            uploadContext_.waitForIdle();
        }

        if (chunk.gpu.pageIndex < bufferPages_.size())
        {
            BufferPage& page = bufferPages_[chunk.gpu.pageIndex];
            mergeRange(page.freeVertices, chunk.gpu.vertexOffset, chunk.gpu.reservedVertexCount);
            mergeRange(page.freeIndices, chunk.gpu.indexOffset, chunk.gpu.reservedIndexCount);
            if (chunk.gpu.recordIndex != kInvalidFarDrawRecordIndex)
            {
                clearDrawRecord(page, chunk.gpu.recordIndex);
                mergeRange(page.freeRecords, chunk.gpu.recordIndex, 1);
            }
        }

        chunk.gpu.pageIndex = kInvalidChunkBufferPage;
        chunk.gpu.vertexOffset = 0;
        chunk.gpu.indexOffset = 0;
        chunk.gpu.vertexCount = 0;
        chunk.gpu.indexCount = 0;
        chunk.gpu.reservedVertexCount = 0;
        chunk.gpu.reservedIndexCount = 0;
        chunk.gpu.faceCapacity = 0;
        chunk.gpu.recordIndex = kInvalidFarDrawRecordIndex;
        chunk.gpu.resident = false;
    }

    void releaseChunkGpu(FarLodChunkRecord& chunk)
    {
        releasePendingMeshAllocation(chunk);
        if ((chunk.gpu.voxelBuffer != nullptr || chunk.gpu.columnBuffer != nullptr) && chunk.gpu.voxelFenceValue != 0)
        {
            const UINT64 completedFenceValue = gpuContext_.completedFenceValue();
            if (completedFenceValue < chunk.gpu.voxelFenceValue)
            {
                gpuContext_.waitForFence(chunk.gpu.voxelFenceValue);
            }
        }
        if (chunk.gpu.readbackFenceValue != 0)
        {
            const UINT64 completedUploadFenceValue = uploadContext_.completedFenceValue();
            if (completedUploadFenceValue < chunk.gpu.readbackFenceValue)
            {
                uploadContext_.waitForIdle();
            }
        }
        chunk.gpu.columnBuffer.Reset();
        chunk.gpu.columnState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.voxelBuffer.Reset();
        chunk.gpu.voxelState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceCountBuffer.Reset();
        chunk.gpu.faceAnalysisBuffer.Reset();
        chunk.gpu.faceDescriptorBuffer.Reset();
        chunk.gpu.facePrefixBuffer.Reset();
        chunk.gpu.faceGroupSumBuffer.Reset();
        chunk.gpu.faceCountState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceAnalysisState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceDescriptorState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.facePrefixState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.faceGroupSumState = D3D12_RESOURCE_STATE_COMMON;
        chunk.gpu.voxelFenceValue = 0;
        chunk.gpu.readbackFenceValue = 0;
        chunk.gpu.voxelReady = false;
        chunk.gpu.parityMismatchCount = 0;
        chunk.gpu.parityValidated = false;
        releaseChunkRenderAllocation(chunk);
        chunk.requestedFaceCapacityHint = 0;
    }

    [[nodiscard]] glm::ivec2 computeAtlasSizeCells(const FarLodLevelConfig& level) const noexcept
    {
        const auto it = levelActivationOuterRadiusChunks_.find(level.level);
        const int activeOuterRadiusChunks =
            (it != levelActivationOuterRadiusChunks_.end()) ? it->second : level.innerRadiusChunks;
        const int minChunk = floorDiv(-activeOuterRadiusChunks, level.blockScale);
        const int maxChunk = floorDiv(activeOuterRadiusChunks, level.blockScale);
        const int chunkCount = (maxChunk - minChunk + 1) + 2;
        const int cells = std::max(chunkCount * kLogicalSize, kLogicalSize * 4);
        return glm::ivec2(cells, cells);
    }

    [[nodiscard]] glm::ivec2 computeDesiredAtlasOriginCell(const FarLodLevelConfig& level) const noexcept
    {
        const auto it = levelActivationOuterRadiusChunks_.find(level.level);
        const int activeOuterRadiusChunks =
            (it != levelActivationOuterRadiusChunks_.end()) ? it->second : level.innerRadiusChunks;
        int chunkMinX = floorDiv(cameraChunk_.x - activeOuterRadiusChunks, level.blockScale) - 1;
        int chunkMinZ = floorDiv(cameraChunk_.z - activeOuterRadiusChunks, level.blockScale) - 1;
        const int snapStrideChunks = atlasOriginSnapStrideChunks(level);
        if (snapStrideChunks > 1)
        {
            chunkMinX = floorDiv(chunkMinX, snapStrideChunks) * snapStrideChunks;
            chunkMinZ = floorDiv(chunkMinZ, snapStrideChunks) * snapStrideChunks;
        }
        return glm::ivec2(chunkMinX * kLogicalSize, chunkMinZ * kLogicalSize);
    }

    [[nodiscard]] glm::ivec2 computeSeedCacheSizeChunks(const FarLodLevelConfig& level) const noexcept
    {
        const terrain::FarLodGpuWorldgenHeader& header = worldgenTables_.header;
        if (header.chunkSpan <= 0)
        {
            return glm::ivec2(0, 0);
        }

        const glm::ivec2 atlasSizeCells = computeAtlasSizeCells(level);
        const int coveredChunksX = ((atlasSizeCells.x * level.blockScale) + header.chunkSpan - 1) / header.chunkSpan + 1;
        const int coveredChunksZ = ((atlasSizeCells.y * level.blockScale) + header.chunkSpan - 1) / header.chunkSpan + 1;
        return glm::ivec2(coveredChunksX + header.neighborRadius * 2,
                          coveredChunksZ + header.neighborRadius * 2);
    }

    [[nodiscard]] glm::ivec2 computeDesiredSeedOriginChunk(const FarLodLevelConfig& level,
                                                           const glm::ivec2& atlasOriginCell) const noexcept
    {
        const terrain::FarLodGpuWorldgenHeader& header = worldgenTables_.header;
        if (header.chunkSpan <= 0)
        {
            return glm::ivec2(0, 0);
        }

        const int minWorldX = atlasOriginCell.x * level.blockScale;
        const int minWorldZ = atlasOriginCell.y * level.blockScale;
        return glm::ivec2(floorDiv(minWorldX, header.chunkSpan) - header.neighborRadius,
                          floorDiv(minWorldZ, header.chunkSpan) - header.neighborRadius);
    }

    [[nodiscard]] FarLodWorkBudget computeWorkBudget() const
    {
        std::size_t buildBacklog = 0;
        {
            std::lock_guard<std::mutex> lock(buildQueueMutex_);
            buildBacklog = buildQueue_.size() + queuedKeys_.size();
        }

        std::size_t gpuBacklog = 0;
        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            gpuBacklog = gpuSynthesisRequests_.size() +
                         pendingGpuParityReadbacks_.size();
        }

        const std::size_t totalBacklog = buildBacklog + gpuBacklog;
        const int exactMissing = exactMissingChunks_.load(std::memory_order_relaxed);
        const std::size_t exactPendingUploads = exactPendingUploads_.load(std::memory_order_relaxed);
        const std::size_t workerBudget = std::max<std::size_t>(workerCount_, 1);

        FarLodWorkBudget budget{};
        if (totalBacklog > 512 || exactMissing > 48 || exactPendingUploads > 48)
        {
            budget.activationStepChunks = 1;
            budget.distanceRampStepChunks = 0;
            budget.newChunkActivations = std::max<std::size_t>(workerBudget * 2, 4);
            budget.newFallbackActivations = 2;
            budget.staleReleaseCount = 4;
            budget.atlasUpdateCells = 8u * 1024u;
            budget.gpuDispatchBudgetUnits = 8;
            budget.maxGpuSubmissions = 2;
        }
        else if (totalBacklog > 256 || exactMissing > 24 || exactPendingUploads > 24)
        {
            budget.activationStepChunks = 2;
            budget.distanceRampStepChunks = 1;
            budget.newChunkActivations = std::max<std::size_t>(workerBudget * 4, 8);
            budget.newFallbackActivations = 4;
            budget.staleReleaseCount = 8;
            budget.atlasUpdateCells = 16u * 1024u;
            budget.gpuDispatchBudgetUnits = 12;
            budget.maxGpuSubmissions = 3;
        }
        else if (totalBacklog > 96 || exactMissing > 8 || exactPendingUploads > 8)
        {
            budget.activationStepChunks = 4;
            budget.distanceRampStepChunks = 2;
            budget.newChunkActivations = std::max<std::size_t>(workerBudget * 8, 16);
            budget.newFallbackActivations = 6;
            budget.staleReleaseCount = 16;
            budget.atlasUpdateCells = 48u * 1024u;
            budget.gpuDispatchBudgetUnits = 24;
            budget.maxGpuSubmissions = 5;
        }
        else
        {
            budget.activationStepChunks = 6;
            budget.distanceRampStepChunks = 4;
            budget.newChunkActivations = std::max<std::size_t>(workerBudget * 12, 32);
            budget.newFallbackActivations = 8;
            budget.staleReleaseCount = 24;
            budget.atlasUpdateCells = 160u * 1024u;
            budget.gpuDispatchBudgetUnits = 36;
            budget.maxGpuSubmissions = 8;
        }
        return budget;
    }

    void ensureAtlasState(const FarLodLevelConfig& level)
    {
        if (device_ == nullptr)
        {
            return;
        }

        FarLodLevelAtlasState& atlas = levelAtlases_[level.level];
        const glm::ivec2 requiredSize = computeAtlasSizeCells(level);
        const glm::ivec2 requiredSeedSize = computeSeedCacheSizeChunks(level);
        if (atlas.buffer != nullptr &&
            atlas.sampleBuffer != nullptr &&
            atlas.seedHeaderBuffer != nullptr &&
            atlas.seedDataBuffer != nullptr &&
            atlas.blockScale == level.blockScale &&
            atlas.atlasSizeCells == requiredSize &&
            atlas.seedSizeChunks == requiredSeedSize)
        {
            return;
        }

        atlas = FarLodLevelAtlasState{};
        atlas.level = level.level;
        atlas.blockScale = level.blockScale;
        atlas.atlasSizeCells = requiredSize;
        atlas.seedSizeChunks = requiredSeedSize;
        const std::uint64_t bufferBytes =
            static_cast<std::uint64_t>(requiredSize.x) * static_cast<std::uint64_t>(requiredSize.y) *
            sizeof(GpuTerrainAtlasSample);
        const std::uint64_t sampleBufferBytes =
            static_cast<std::uint64_t>(requiredSize.x) * static_cast<std::uint64_t>(requiredSize.y) *
            sizeof(GpuAtlasSampleCacheEntry);
        const std::uint64_t seedHeaderBufferBytes =
            static_cast<std::uint64_t>(requiredSeedSize.x) * static_cast<std::uint64_t>(requiredSeedSize.y) *
            sizeof(GpuChunkSeedCacheHeader);
        const std::uint64_t seedDataBufferBytes =
            static_cast<std::uint64_t>(requiredSeedSize.x) * static_cast<std::uint64_t>(requiredSeedSize.y) *
            kFarLodChunkSeedCountPerCacheEntry * sizeof(GpuChunkSeedCacheSeed);
        atlas.buffer = createDefaultBuffer(device_.Get(),
                                           bufferBytes,
                                           D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        atlas.sampleBuffer = createDefaultBuffer(device_.Get(),
                                                 sampleBufferBytes,
                                                 D3D12_RESOURCE_STATE_COMMON,
                                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        atlas.seedHeaderBuffer = createDefaultBuffer(device_.Get(),
                                                     seedHeaderBufferBytes,
                                                     D3D12_RESOURCE_STATE_COMMON,
                                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        atlas.seedDataBuffer = createDefaultBuffer(device_.Get(),
                                                   seedDataBufferBytes,
                                                   D3D12_RESOURCE_STATE_COMMON,
                                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (atlas.buffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodAtlas_L" << level.level;
            setDebugObjectName(atlas.buffer.Get(), name.str());
        }
        if (atlas.sampleBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodAtlasSamples_L" << level.level;
            setDebugObjectName(atlas.sampleBuffer.Get(), name.str());
        }
        if (atlas.seedHeaderBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodChunkSeedHeaders_L" << level.level;
            setDebugObjectName(atlas.seedHeaderBuffer.Get(), name.str());
        }
        if (atlas.seedDataBuffer != nullptr)
        {
            std::wostringstream name;
            name << L"FarLodChunkSeedData_L" << level.level;
            setDebugObjectName(atlas.seedDataBuffer.Get(), name.str());
        }
        atlas.state = D3D12_RESOURCE_STATE_COMMON;
        atlas.sampleState = D3D12_RESOURCE_STATE_COMMON;
        atlas.seedHeaderState = D3D12_RESOURCE_STATE_COMMON;
        atlas.seedDataState = D3D12_RESOURCE_STATE_COMMON;
        atlas.originCell = computeDesiredAtlasOriginCell(level);
        atlas.seedOriginChunk = computeDesiredSeedOriginChunk(level, atlas.originCell);
        atlas.initialized = false;
        atlas.seedInitialized = false;
        atlas.pendingDirtyRects.clear();
        atlas.pendingSeedDirtyRects.clear();
    }

    [[nodiscard]] std::vector<AtlasUpdateRect> prepareAtlasUpdateRects(FarLodLevelAtlasState& atlas,
                                                                       const FarLodLevelConfig& level)
    {
        std::vector<AtlasUpdateRect> rects;
        const glm::ivec2 desiredOriginCell = computeDesiredAtlasOriginCell(level);
        if (!atlas.initialized)
        {
            atlas.originCell = desiredOriginCell;
            atlas.initialized = true;
            rects.push_back(AtlasUpdateRect{atlas.originCell, atlas.atlasSizeCells});
            return rects;
        }

        const glm::ivec2 delta = desiredOriginCell - atlas.originCell;
        if (delta.x == 0 && delta.y == 0)
        {
            return rects;
        }

        if (std::abs(delta.x) >= atlas.atlasSizeCells.x || std::abs(delta.y) >= atlas.atlasSizeCells.y)
        {
            atlas.originCell = desiredOriginCell;
            rects.push_back(AtlasUpdateRect{atlas.originCell, atlas.atlasSizeCells});
            return rects;
        }

        atlas.originCell = desiredOriginCell;
        const int absDeltaX = std::abs(delta.x);
        if (delta.x > 0)
        {
            rects.push_back(AtlasUpdateRect{
                {atlas.originCell.x + atlas.atlasSizeCells.x - delta.x, atlas.originCell.y},
                {delta.x, atlas.atlasSizeCells.y}});
        }
        else if (delta.x < 0)
        {
            rects.push_back(AtlasUpdateRect{
                {atlas.originCell.x, atlas.originCell.y},
                {-delta.x, atlas.atlasSizeCells.y}});
        }

        if (delta.y > 0)
        {
            const int originX = atlas.originCell.x + (delta.x < 0 ? absDeltaX : 0);
            const int width = atlas.atlasSizeCells.x - absDeltaX;
            if (width > 0)
            {
            rects.push_back(AtlasUpdateRect{
                    {originX, atlas.originCell.y + atlas.atlasSizeCells.y - delta.y},
                    {width, delta.y}});
            }
        }
        else if (delta.y < 0)
        {
            const int originX = atlas.originCell.x + (delta.x < 0 ? absDeltaX : 0);
            const int width = atlas.atlasSizeCells.x - absDeltaX;
            if (width > 0)
            {
            rects.push_back(AtlasUpdateRect{
                    {originX, atlas.originCell.y},
                    {width, -delta.y}});
            }
        }
        return rects;
    }

    [[nodiscard]] std::vector<AtlasUpdateRect> prepareSeedCacheUpdateRects(FarLodLevelAtlasState& atlas,
                                                                           const FarLodLevelConfig& level)
    {
        std::vector<AtlasUpdateRect> rects;
        const glm::ivec2 desiredOriginChunk = computeDesiredSeedOriginChunk(level, atlas.originCell);
        if (!atlas.seedInitialized)
        {
            atlas.seedOriginChunk = desiredOriginChunk;
            atlas.seedInitialized = true;
            rects.push_back(AtlasUpdateRect{atlas.seedOriginChunk, atlas.seedSizeChunks});
            return rects;
        }

        const glm::ivec2 delta = desiredOriginChunk - atlas.seedOriginChunk;
        if (delta.x == 0 && delta.y == 0)
        {
            return rects;
        }

        if (std::abs(delta.x) >= atlas.seedSizeChunks.x || std::abs(delta.y) >= atlas.seedSizeChunks.y)
        {
            atlas.seedOriginChunk = desiredOriginChunk;
            rects.push_back(AtlasUpdateRect{atlas.seedOriginChunk, atlas.seedSizeChunks});
            return rects;
        }

        atlas.seedOriginChunk = desiredOriginChunk;
        const int absDeltaX = std::abs(delta.x);
        if (delta.x > 0)
        {
            rects.push_back(AtlasUpdateRect{
                {atlas.seedOriginChunk.x + atlas.seedSizeChunks.x - delta.x, atlas.seedOriginChunk.y},
                {delta.x, atlas.seedSizeChunks.y}});
        }
        else if (delta.x < 0)
        {
            rects.push_back(AtlasUpdateRect{
                {atlas.seedOriginChunk.x, atlas.seedOriginChunk.y},
                {-delta.x, atlas.seedSizeChunks.y}});
        }

        if (delta.y > 0)
        {
            const int originX = atlas.seedOriginChunk.x + (delta.x < 0 ? absDeltaX : 0);
            const int width = atlas.seedSizeChunks.x - absDeltaX;
            if (width > 0)
            {
                rects.push_back(AtlasUpdateRect{
                    {originX, atlas.seedOriginChunk.y + atlas.seedSizeChunks.y - delta.y},
                    {width, delta.y}});
            }
        }
        else if (delta.y < 0)
        {
            const int originX = atlas.seedOriginChunk.x + (delta.x < 0 ? absDeltaX : 0);
            const int width = atlas.seedSizeChunks.x - absDeltaX;
            if (width > 0)
            {
                rects.push_back(AtlasUpdateRect{
                    {originX, atlas.seedOriginChunk.y},
                    {width, -delta.y}});
            }
        }
        return rects;
    }

    [[nodiscard]] static bool atlasRectValid(const AtlasUpdateRect& rect) noexcept
    {
        return rect.sizeCells.x > 0 && rect.sizeCells.y > 0;
    }

    [[nodiscard]] static std::optional<AtlasUpdateRect> clipAtlasUpdateRect(const AtlasUpdateRect& rect,
                                                                            const glm::ivec2& originCell,
                                                                            const glm::ivec2& atlasSizeCells) noexcept
    {
        const int minX = std::max(rect.originCell.x, originCell.x);
        const int minZ = std::max(rect.originCell.y, originCell.y);
        const int maxX = std::min(rect.originCell.x + rect.sizeCells.x, originCell.x + atlasSizeCells.x);
        const int maxZ = std::min(rect.originCell.y + rect.sizeCells.y, originCell.y + atlasSizeCells.y);
        if (maxX <= minX || maxZ <= minZ)
        {
            return std::nullopt;
        }
        return AtlasUpdateRect{{minX, minZ}, {maxX - minX, maxZ - minZ}};
    }

    [[nodiscard]] static std::optional<AtlasUpdateRect> intersectAtlasUpdateRect(const AtlasUpdateRect& a,
                                                                                 const AtlasUpdateRect& b) noexcept
    {
        const int minX = std::max(a.originCell.x, b.originCell.x);
        const int minZ = std::max(a.originCell.y, b.originCell.y);
        const int maxX = std::min(a.originCell.x + a.sizeCells.x, b.originCell.x + b.sizeCells.x);
        const int maxZ = std::min(a.originCell.y + a.sizeCells.y, b.originCell.y + b.sizeCells.y);
        if (maxX <= minX || maxZ <= minZ)
        {
            return std::nullopt;
        }
        return AtlasUpdateRect{{minX, minZ}, {maxX - minX, maxZ - minZ}};
    }

    static void subtractAtlasUpdateRect(const AtlasUpdateRect& rect,
                                        const AtlasUpdateRect& cut,
                                        std::vector<AtlasUpdateRect>& leftovers)
    {
        const std::optional<AtlasUpdateRect> intersection = intersectAtlasUpdateRect(rect, cut);
        if (!intersection.has_value())
        {
            leftovers.push_back(rect);
            return;
        }

        const AtlasUpdateRect& overlap = *intersection;
        const int rectMinX = rect.originCell.x;
        const int rectMinZ = rect.originCell.y;
        const int rectMaxX = rect.originCell.x + rect.sizeCells.x;
        const int rectMaxZ = rect.originCell.y + rect.sizeCells.y;
        const int cutMinX = overlap.originCell.x;
        const int cutMinZ = overlap.originCell.y;
        const int cutMaxX = overlap.originCell.x + overlap.sizeCells.x;
        const int cutMaxZ = overlap.originCell.y + overlap.sizeCells.y;

        if (cutMinZ > rectMinZ)
        {
            leftovers.push_back(AtlasUpdateRect{{rectMinX, rectMinZ}, {rect.sizeCells.x, cutMinZ - rectMinZ}});
        }
        if (cutMaxZ < rectMaxZ)
        {
            leftovers.push_back(AtlasUpdateRect{{rectMinX, cutMaxZ}, {rect.sizeCells.x, rectMaxZ - cutMaxZ}});
        }
        if (cutMinX > rectMinX)
        {
            leftovers.push_back(AtlasUpdateRect{{rectMinX, cutMinZ}, {cutMinX - rectMinX, overlap.sizeCells.y}});
        }
        if (cutMaxX < rectMaxX)
        {
            leftovers.push_back(AtlasUpdateRect{{cutMaxX, cutMinZ}, {rectMaxX - cutMaxX, overlap.sizeCells.y}});
        }
    }

    static void mergeAtlasUpdateRect(std::vector<AtlasUpdateRect>& rects, const AtlasUpdateRect& rect)
    {
        if (!atlasRectValid(rect))
        {
            return;
        }

        AtlasUpdateRect merged = rect;
        bool mergedAny = true;
        while (mergedAny)
        {
            mergedAny = false;
            for (auto it = rects.begin(); it != rects.end();)
            {
                const AtlasUpdateRect& existing = *it;
                const bool overlapOrTouch =
                    merged.originCell.x <= existing.originCell.x + existing.sizeCells.x &&
                    merged.originCell.x + merged.sizeCells.x >= existing.originCell.x &&
                    merged.originCell.y <= existing.originCell.y + existing.sizeCells.y &&
                    merged.originCell.y + merged.sizeCells.y >= existing.originCell.y;
                if (!overlapOrTouch)
                {
                    ++it;
                    continue;
                }

                const int minX = std::min(merged.originCell.x, existing.originCell.x);
                const int minZ = std::min(merged.originCell.y, existing.originCell.y);
                const int maxX = std::max(merged.originCell.x + merged.sizeCells.x,
                                          existing.originCell.x + existing.sizeCells.x);
                const int maxZ = std::max(merged.originCell.y + merged.sizeCells.y,
                                          existing.originCell.y + existing.sizeCells.y);
                merged.originCell = glm::ivec2(minX, minZ);
                merged.sizeCells = glm::ivec2(maxX - minX, maxZ - minZ);
                it = rects.erase(it);
                mergedAny = true;
            }
        }

        rects.push_back(merged);
    }

    static void appendClippedAtlasUpdateRect(std::vector<AtlasUpdateRect>& rects,
                                             const AtlasUpdateRect& rect,
                                             const glm::ivec2& originCell,
                                             const glm::ivec2& atlasSizeCells)
    {
        const std::optional<AtlasUpdateRect> clipped = clipAtlasUpdateRect(rect, originCell, atlasSizeCells);
        if (!clipped.has_value())
        {
            return;
        }
        mergeAtlasUpdateRect(rects, *clipped);
    }

    void markAtlasCellsUpdated(FarLodLevelAtlasState& atlas, const AtlasUpdateRect& rect)
    {
        if (!atlasRectValid(rect))
        {
            return;
        }

        const std::uint64_t revision = ++atlasRevisionCounter_;
        const int maxCellZ = rect.originCell.y + rect.sizeCells.y;
        const int maxCellX = rect.originCell.x + rect.sizeCells.x;
        for (int cellZ = rect.originCell.y; cellZ < maxCellZ; ++cellZ)
        {
            for (int cellX = rect.originCell.x; cellX < maxCellX; ++cellX)
            {
                atlas.cellRevisions[packAtlasCellKey(cellX, cellZ)] = revision;
            }
        }
    }

    [[nodiscard]] std::uint64_t currentAtlasDependencyRevision(const FarLodChunkRecord& chunk) const
    {
        const auto atlasIt = levelAtlases_.find(chunk.level.level);
        if (atlasIt == levelAtlases_.end())
        {
            return 0;
        }

        const FarLodLevelAtlasState& atlas = atlasIt->second;
        const int cellOriginX = (chunk.key.coord.x - 1) * kLogicalSize;
        const int cellOriginZ = (chunk.key.coord.z - 1) * kLogicalSize;
        const int cellMaxX = cellOriginX + (kLogicalSize * 3);
        const int cellMaxZ = cellOriginZ + (kLogicalSize * 3);
        std::uint64_t revision = 0;
        for (int cellZ = cellOriginZ; cellZ < cellMaxZ; ++cellZ)
        {
            for (int cellX = cellOriginX; cellX < cellMaxX; ++cellX)
            {
                const auto revisionIt = atlas.cellRevisions.find(packAtlasCellKey(cellX, cellZ));
                if (revisionIt != atlas.cellRevisions.end())
                {
                    revision = std::max(revision, revisionIt->second);
                }
            }
        }
        return revision;
    }

    [[nodiscard]] AtlasUpdateRect computeSeedRectForAtlasUpdate(const FarLodLevelConfig& level,
                                                                const AtlasUpdateRect& rect) const
    {
        const terrain::FarLodGpuWorldgenHeader& header = worldgenTables_.header;
        const int blockScale = std::max(level.blockScale, 1);
        const int worldMinX = rect.originCell.x * blockScale;
        const int worldMinZ = rect.originCell.y * blockScale;
        const int worldMaxX = (rect.originCell.x + rect.sizeCells.x - 1) * blockScale + (blockScale - 1);
        const int worldMaxZ = (rect.originCell.y + rect.sizeCells.y - 1) * blockScale + (blockScale - 1);
        const int chunkMinX = floorDiv(worldMinX, header.chunkSpan) - header.neighborRadius;
        const int chunkMinZ = floorDiv(worldMinZ, header.chunkSpan) - header.neighborRadius;
        const int chunkMaxX = floorDiv(worldMaxX, header.chunkSpan) + header.neighborRadius;
        const int chunkMaxZ = floorDiv(worldMaxZ, header.chunkSpan) + header.neighborRadius;
        return AtlasUpdateRect{
            {chunkMinX, chunkMinZ},
            {chunkMaxX - chunkMinX + 1, chunkMaxZ - chunkMinZ + 1}};
    }

    void appendAtlasUpdates(const std::deque<GpuSynthesisRequest>& requests, std::size_t atlasCellBudget)
    {
        atlasCellBudget = std::min(atlasCellBudget, kMaxFarLodAtlasUpdateCellsPerSubmission);

        std::unordered_set<int> requestedLevels;
        requestedLevels.reserve(requests.size());
        for (const GpuSynthesisRequest& request : requests)
        {
            requestedLevels.insert(request.key.level);
        }

        for (const FarLodLevelConfig& level : levels_)
        {
            if (level.outerRadiusChunks <= level.innerRadiusChunks)
            {
                continue;
            }
            if (!requestedLevels.contains(level.level))
            {
                continue;
            }

            ensureAtlasState(level);
            auto atlasIt = levelAtlases_.find(level.level);
            if (atlasIt == levelAtlases_.end())
            {
                continue;
            }

            FarLodLevelAtlasState& atlas = atlasIt->second;
            if (atlas.buffer == nullptr || atlas.sampleBuffer == nullptr ||
                atlas.seedHeaderBuffer == nullptr || atlas.seedDataBuffer == nullptr)
            {
                continue;
            }

            std::vector<AtlasUpdateRect> rects;
            for (const AtlasUpdateRect& rect : prepareAtlasUpdateRects(atlas, level))
            {
                appendClippedAtlasUpdateRect(rects, rect, atlas.originCell, atlas.atlasSizeCells);
            }
            for (const AtlasUpdateRect& rect : atlas.pendingDirtyRects)
            {
                appendClippedAtlasUpdateRect(rects, rect, atlas.originCell, atlas.atlasSizeCells);
            }
            atlas.pendingDirtyRects.clear();

            std::vector<AtlasUpdateRect> seedRects;
            for (const AtlasUpdateRect& rect : prepareSeedCacheUpdateRects(atlas, level))
            {
                appendClippedAtlasUpdateRect(seedRects, rect, atlas.seedOriginChunk, atlas.seedSizeChunks);
            }
            for (const AtlasUpdateRect& rect : atlas.pendingSeedDirtyRects)
            {
                appendClippedAtlasUpdateRect(seedRects, rect, atlas.seedOriginChunk, atlas.seedSizeChunks);
            }
            atlas.pendingSeedDirtyRects.clear();

            if (atlasCellBudget == 0 || rects.empty())
            {
                for (const AtlasUpdateRect& rect : rects)
                {
                    mergeAtlasUpdateRect(atlas.pendingDirtyRects, rect);
                }
                for (const AtlasUpdateRect& rect : seedRects)
                {
                    mergeAtlasUpdateRect(atlas.pendingSeedDirtyRects, rect);
                }
                continue;
            }

            std::vector<AtlasUpdateRect> uploadRects;
            for (const AtlasUpdateRect& rect : rects)
            {
                if (atlasCellBudget == 0)
                {
                    mergeAtlasUpdateRect(atlas.pendingDirtyRects, rect);
                    continue;
                }

                const std::size_t rectCellCount =
                    static_cast<std::size_t>(rect.sizeCells.x) * static_cast<std::size_t>(rect.sizeCells.y);
                int uploadHeight = rect.sizeCells.y;
                if (rectCellCount > atlasCellBudget)
                {
                    const std::size_t maxRows = std::max<std::size_t>(1, atlasCellBudget / static_cast<std::size_t>(rect.sizeCells.x));
                    uploadHeight = static_cast<int>(std::min<std::size_t>(maxRows, static_cast<std::size_t>(rect.sizeCells.y)));
                }

                const AtlasUpdateRect uploadRect{rect.originCell, glm::ivec2(rect.sizeCells.x, uploadHeight)};
                const int remainingHeight = rect.sizeCells.y - uploadHeight;
                if (remainingHeight > 0)
                {
                    mergeAtlasUpdateRect(atlas.pendingDirtyRects,
                                         AtlasUpdateRect{
                                             glm::ivec2(rect.originCell.x, rect.originCell.y + uploadHeight),
                                             glm::ivec2(rect.sizeCells.x, remainingHeight)});
                }

                uploadRects.push_back(uploadRect);
                const std::size_t uploadedCells =
                    static_cast<std::size_t>(uploadRect.sizeCells.x) * static_cast<std::size_t>(uploadRect.sizeCells.y);
                atlasCellBudget = (uploadedCells >= atlasCellBudget) ? 0 : (atlasCellBudget - uploadedCells);
            }

            if (uploadRects.empty())
            {
                for (const AtlasUpdateRect& rect : seedRects)
                {
                    mergeAtlasUpdateRect(atlas.pendingSeedDirtyRects, rect);
                }
                continue;
            }

            std::vector<AtlasUpdateRect> requiredSeedRects;
            for (const AtlasUpdateRect& uploadRect : uploadRects)
            {
                appendClippedAtlasUpdateRect(requiredSeedRects,
                                            computeSeedRectForAtlasUpdate(level, uploadRect),
                                            atlas.seedOriginChunk,
                                            atlas.seedSizeChunks);
            }

            std::vector<AtlasUpdateRect> seedUploadRects;
            for (const AtlasUpdateRect& seedRect : seedRects)
            {
                std::vector<AtlasUpdateRect> remainingParts{seedRect};
                for (const AtlasUpdateRect& requiredRect : requiredSeedRects)
                {
                    std::vector<AtlasUpdateRect> nextParts;
                    for (const AtlasUpdateRect& part : remainingParts)
                    {
                        const std::optional<AtlasUpdateRect> overlap = intersectAtlasUpdateRect(part, requiredRect);
                        if (!overlap.has_value())
                        {
                            nextParts.push_back(part);
                            continue;
                        }

                        mergeAtlasUpdateRect(seedUploadRects, *overlap);
                        subtractAtlasUpdateRect(part, *overlap, nextParts);
                    }
                    remainingParts = std::move(nextParts);
                    if (remainingParts.empty())
                    {
                        break;
                    }
                }

                for (const AtlasUpdateRect& remainingRect : remainingParts)
                {
                    mergeAtlasUpdateRect(atlas.pendingSeedDirtyRects, remainingRect);
                }
            }

            if (!seedUploadRects.empty())
            {
                if (chunkManagerDebugLoggingEnabled())
                {
                    std::ostringstream stream;
                    stream << "Far LOD seed dispatch: biome=" << worldgenTables_.header.biomeCount
                           << " biomeSel=" << worldgenTables_.header.biomeSelectionCount
                           << " oceanSel=" << worldgenTables_.header.oceanSelectionCount
                           << " subBiome=" << worldgenTables_.header.subBiomeCount
                           << " seedHeader=" << atlas.seedHeaderElementCount()
                           << " seedData=" << atlas.seedDataElementCount()
                           << " rects=" << seedUploadRects.size();
                    chunkManagerDebugLog(stream.str());
                }

                gpuContext_.transition(atlas.seedHeaderBuffer.Get(), atlas.seedHeaderState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                atlas.seedHeaderState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                gpuContext_.transition(atlas.seedDataBuffer.Get(), atlas.seedDataState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                atlas.seedDataState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                for (const AtlasUpdateRect& seedRect : seedUploadRects)
                {
                    gpuContext_.dispatchSeedCacheUpdate(seedRect.originCell,
                                                        seedRect.sizeCells,
                                                        atlas.seedOriginChunk,
                                                        atlas.seedSizeChunks,
                                                        worldgenHeaderBuffer_.Get(),
                                                        worldgenBiomeBuffer_.Get(),
                                                        worldgenTables_.header.biomeCount,
                                                        worldgenBiomeSelectionBuffer_.Get(),
                                                        worldgenTables_.header.biomeSelectionCount,
                                                        worldgenOceanSelectionBuffer_.Get(),
                                                        worldgenTables_.header.oceanSelectionCount,
                                                        worldgenSubBiomeBuffer_.Get(),
                                                        worldgenTables_.header.subBiomeCount,
                                                        atlas.seedHeaderBuffer.Get(),
                                                        atlas.seedHeaderElementCount(),
                                                        atlas.seedDataBuffer.Get(),
                                                        atlas.seedDataElementCount());
                }
                gpuContext_.uavBarrier(atlas.seedHeaderBuffer.Get());
                gpuContext_.uavBarrier(atlas.seedDataBuffer.Get());
                gpuContext_.transition(atlas.seedHeaderBuffer.Get(),
                                       atlas.seedHeaderState,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                atlas.seedHeaderState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                gpuContext_.transition(atlas.seedDataBuffer.Get(),
                                       atlas.seedDataState,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                atlas.seedDataState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            }

            gpuContext_.transition(atlas.sampleBuffer.Get(), atlas.sampleState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            atlas.sampleState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            for (const AtlasUpdateRect& uploadRect : uploadRects)
            {
                gpuContext_.dispatchAtlasSampleCacheUpdate(atlas.originCell,
                                                           atlas.atlasSizeCells,
                                                           uploadRect.originCell,
                                                           uploadRect.sizeCells,
                                                           level.blockScale,
                                                           atlas.seedOriginChunk,
                                                           atlas.seedSizeChunks,
                                                           worldgenHeaderBuffer_.Get(),
                                                           worldgenBiomeBuffer_.Get(),
                                                           worldgenTables_.header.biomeCount,
                                                           worldgenBiomeSelectionBuffer_.Get(),
                                                           worldgenTables_.header.biomeSelectionCount,
                                                           worldgenOceanSelectionBuffer_.Get(),
                                                           worldgenTables_.header.oceanSelectionCount,
                                                           worldgenSubBiomeBuffer_.Get(),
                                                           worldgenTables_.header.subBiomeCount,
                                                           worldgenPermutationBuffer_.Get(),
                                                           static_cast<std::uint32_t>(worldgenTables_.surfacePermutation.size()),
                                                           atlas.seedHeaderBuffer.Get(),
                                                           atlas.seedHeaderElementCount(),
                                                           atlas.seedDataBuffer.Get(),
                                                           atlas.seedDataElementCount(),
                                                           atlas.sampleBuffer.Get(),
                                                           atlas.elementCount());
            }
            gpuContext_.uavBarrier(atlas.sampleBuffer.Get());
            gpuContext_.transition(atlas.sampleBuffer.Get(),
                                   atlas.sampleState,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            atlas.sampleState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

            gpuContext_.transition(atlas.buffer.Get(), atlas.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            atlas.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            for (const AtlasUpdateRect& uploadRect : uploadRects)
            {
                gpuContext_.dispatchAtlasUpdate(atlas.originCell,
                                                atlas.atlasSizeCells,
                                                uploadRect.originCell,
                                                uploadRect.sizeCells,
                                                level.blockScale,
                                                worldgenTables_.header.seaLevel,
                                                worldgenHeaderBuffer_.Get(),
                                                worldgenBiomeBuffer_.Get(),
                                                worldgenTables_.header.biomeCount,
                                                worldgenPermutationBuffer_.Get(),
                                                static_cast<std::uint32_t>(worldgenTables_.surfacePermutation.size()),
                                                atlas.sampleBuffer.Get(),
                                                atlas.elementCount(),
                                                atlas.buffer.Get(),
                                                atlas.elementCount());
                markAtlasCellsUpdated(atlas, uploadRect);
            }
            gpuContext_.uavBarrier(atlas.buffer.Get());
            gpuContext_.transition(atlas.buffer.Get(),
                                   atlas.state,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            atlas.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
    }

    [[nodiscard]] static GpuStructureInstance packGpuStructureInstance(const StructureInstance& instance) noexcept
    {
        GpuStructureInstance gpuInstance{};
        gpuInstance.type = static_cast<std::uint32_t>(instance.type);
        gpuInstance.originX = instance.origin.x;
        gpuInstance.originY = instance.origin.y;
        gpuInstance.originZ = instance.origin.z;
        gpuInstance.boundsMinX = instance.bounds.min.x;
        gpuInstance.boundsMinY = instance.bounds.min.y;
        gpuInstance.boundsMinZ = instance.bounds.min.z;
        gpuInstance.boundsMaxX = instance.bounds.max.x;
        gpuInstance.boundsMaxY = instance.bounds.max.y;
        gpuInstance.boundsMaxZ = instance.bounds.max.z;
        gpuInstance.trunkHeight = static_cast<std::uint32_t>(std::max(instance.trunkHeight, 0));
        gpuInstance.bareTrunkHeight = static_cast<std::uint32_t>(std::max(instance.bareTrunkHeight, 0));
        gpuInstance.maxLodLevel = static_cast<std::uint32_t>(std::max(instance.maxLodLevel, 0));
        return gpuInstance;
    }

    [[nodiscard]] static std::vector<StructureRegionKey> computeStructureRegionKeysForTile(const glm::ivec3& worldMin,
                                                                                            int chunkSpanBlocks)
    {
        std::vector<StructureRegionKey> keys;
        if (chunkSpanBlocks <= 0)
        {
            return keys;
        }

        const int minRegionX = floorDiv(worldMin.x - kMaxStructureHorizontalRadius, kStructureRegionSize);
        const int maxRegionX = floorDiv(worldMin.x + chunkSpanBlocks - 1 + kMaxStructureHorizontalRadius, kStructureRegionSize);
        const int minRegionZ = floorDiv(worldMin.z - kMaxStructureHorizontalRadius, kStructureRegionSize);
        const int maxRegionZ = floorDiv(worldMin.z + chunkSpanBlocks - 1 + kMaxStructureHorizontalRadius, kStructureRegionSize);
        keys.reserve(static_cast<std::size_t>(std::max(maxRegionX - minRegionX + 1, 0) *
                                              std::max(maxRegionZ - minRegionZ + 1, 0)));
        for (int regionZ = minRegionZ; regionZ <= maxRegionZ; ++regionZ)
        {
            for (int regionX = minRegionX; regionX <= maxRegionX; ++regionX)
            {
                keys.push_back(StructureRegionKey{regionX, regionZ});
            }
        }
        return keys;
    }

    void destroyGpuStructureRegions()
    {
        for (auto& [key, region] : gpuStructureRegions_)
        {
            (void)key;
            region.instanceBuffer.Reset();
        }
        gpuStructureRegions_.clear();
    }

    void ensureGpuStructureRegions(const std::deque<GpuSynthesisRequest>& requests)
    {
        if (device_ == nullptr)
        {
            return;
        }

        StructureSampleColumnFn sampleColumnFn;
        StructureSurfaceBlockFn surfaceBlockFn;
        StructureDensityFn densityFn;
        {
            std::lock_guard<std::mutex> lock(structureRegionMutex_);
            sampleColumnFn = structureSampleColumnFn_;
            surfaceBlockFn = structureSurfaceBlockFn_;
            densityFn = structureDensityFn_;
        }
        if (!sampleColumnFn || !surfaceBlockFn || !densityFn)
        {
            return;
        }

        std::vector<StructureRegionKey> missingKeys;
        {
            std::lock_guard<std::mutex> lock(structureRegionMutex_);
            std::unordered_set<StructureRegionKey, StructureRegionKeyHasher> requestedKeys;
            for (const GpuSynthesisRequest& request : requests)
            {
                for (const StructureRegionKey& key : request.structureRegionKeys)
                {
                    if (!requestedKeys.insert(key).second)
                    {
                        continue;
                    }
                    if (!gpuStructureRegions_.contains(key))
                    {
                        missingKeys.push_back(key);
                    }
                }
            }
        }

        if (missingKeys.empty())
        {
            return;
        }

        struct PendingRegionUpload
        {
            StructureRegionKey key{};
            std::uint32_t instanceCount{0};
            Microsoft::WRL::ComPtr<ID3D12Resource> defaultBuffer;
            Microsoft::WRL::ComPtr<ID3D12Resource> uploadBuffer;
            D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
            bool uploadReady{false};
        };

        std::vector<PendingRegionUpload> uploads;
        uploads.reserve(missingKeys.size());
        for (const StructureRegionKey& key : missingKeys)
        {
            const StructureRegion region = buildStructureRegionData(key, sampleColumnFn, surfaceBlockFn, densityFn);
            const std::vector<StructureInstance>& instances = region.instances;
            if (instances.empty())
            {
                PendingRegionUpload upload{};
                upload.key = key;
                uploads.push_back(std::move(upload));
                continue;
            }

            std::vector<GpuStructureInstance> gpuInstances;
            gpuInstances.reserve(instances.size());
            for (const StructureInstance& instance : instances)
            {
                gpuInstances.push_back(packGpuStructureInstance(instance));
            }

            PendingRegionUpload upload{};
            upload.key = key;
            upload.instanceCount = static_cast<std::uint32_t>(gpuInstances.size());
            const std::uint64_t bufferBytes =
                static_cast<std::uint64_t>(gpuInstances.size() * sizeof(GpuStructureInstance));
            upload.defaultBuffer = createDefaultBuffer(device_.Get(), bufferBytes, D3D12_RESOURCE_STATE_COMMON);
            if (upload.defaultBuffer == nullptr)
            {
                continue;
            }

            std::byte* uploadMapped = nullptr;
            upload.uploadBuffer = createUploadBuffer(device_.Get(), bufferBytes, uploadMapped);
            if (upload.uploadBuffer == nullptr || uploadMapped == nullptr)
            {
                continue;
            }

            std::memcpy(uploadMapped, gpuInstances.data(), static_cast<std::size_t>(bufferBytes));
            upload.uploadBuffer->Unmap(0, nullptr);
            uploadMapped = nullptr;
            upload.state = D3D12_RESOURCE_STATE_COMMON;
            upload.uploadReady = true;

            std::wostringstream name;
            name << L"FarLodStructureRegion_" << key.regionX << L"_" << key.regionZ;
            setDebugObjectName(upload.defaultBuffer.Get(), name.str());
            uploads.push_back(std::move(upload));
        }

        bool uploadedAnyBuffers = false;
        bool uploadedBuffersCommitted = false;
        if (uploadContext_.ready() && uploadContext_.begin())
        {
            for (PendingRegionUpload& upload : uploads)
            {
                if (!upload.uploadReady || upload.defaultBuffer == nullptr || upload.uploadBuffer == nullptr)
                {
                    continue;
                }
                uploadedAnyBuffers = true;
                uploadContext_.transition(upload.defaultBuffer.Get(),
                                          D3D12_RESOURCE_STATE_COMMON,
                                          D3D12_RESOURCE_STATE_COPY_DEST);
                uploadContext_.copyBuffer(upload.defaultBuffer.Get(), 0, upload.uploadBuffer.Get(), 0,
                                          static_cast<std::uint64_t>(upload.instanceCount) * sizeof(GpuStructureInstance));
                uploadContext_.transition(upload.defaultBuffer.Get(),
                                          D3D12_RESOURCE_STATE_COPY_DEST,
                                          D3D12_RESOURCE_STATE_COMMON);
            }
            if (uploadedAnyBuffers)
            {
                uploadContext_.flush(nullptr);
                uploadContext_.waitForIdle();
            }
            uploadedBuffersCommitted = true;
        }

        std::lock_guard<std::mutex> lock(structureRegionMutex_);
        for (PendingRegionUpload& upload : uploads)
        {
            if (gpuStructureRegions_.contains(upload.key))
            {
                continue;
            }
            if (upload.instanceCount > 0 && !uploadedBuffersCommitted)
            {
                continue;
            }

            GpuStructureRegionState region{};
            region.key = upload.key;
            region.instanceCount = upload.instanceCount;
            if (upload.uploadReady)
            {
                region.instanceBuffer = std::move(upload.defaultBuffer);
                region.state = upload.state;
            }
            gpuStructureRegions_.emplace(upload.key, std::move(region));
        }
    }

    void destroyBufferPages()
    {
        for (BufferPage& page : bufferPages_)
        {
            page.vertexBuffer.Reset();
            page.indexBuffer.Reset();
            page.vertexUploadBuffer.Reset();
            page.indexUploadBuffer.Reset();
            page.drawRecordBuffer.Reset();
            page.mappedVertexData = nullptr;
            page.mappedIndexData = nullptr;
        }
        bufferPages_.clear();
    }

public:
    [[nodiscard]] int readyTileCount() const noexcept
    {
        int ready = 0;
        std::lock_guard<std::mutex> lock(configMutex_);
        for (const auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (chunk.active && chunk.gpu.resident)
            {
                ++ready;
            }
        }
        return ready;
    }

    [[nodiscard]] int queuedTileCount() const noexcept
    {
        int queued = 0;
        std::lock_guard<std::mutex> lock(configMutex_);
        for (const auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (chunk.active && chunk.inFlight)
            {
                ++queued;
            }
        }
        return queued;
    }

    [[nodiscard]] int pendingUploadTileCount() const noexcept
    {
        int pending = 0;
        std::lock_guard<std::mutex> lock(configMutex_);
        for (const auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (chunk.active && chunk.pendingMesh.valid())
            {
                ++pending;
            }
        }
        return pending;
    }

    [[nodiscard]] int buildQueueDepth() const noexcept
    {
        std::lock_guard<std::mutex> lock(buildQueueMutex_);
        return static_cast<int>(buildQueue_.size());
    }

    [[nodiscard]] double averageBuildMs() const noexcept
    {
        return lastAverageBuildMs_;
    }

    [[nodiscard]] double averageGpuSynthesisMs() const noexcept
    {
        return lastAverageGpuSynthesisMs_;
    }

    [[nodiscard]] double averageGpuStampMs() const noexcept
    {
        return lastAverageGpuStampMs_;
    }

    [[nodiscard]] double averageGpuFaceBuildMs() const noexcept
    {
        return lastAverageGpuFaceBuildMs_;
    }

    [[nodiscard]] LodDiagnosticsSnapshot diagnosticsSnapshot(const glm::vec3& cameraPos) const
    {
        struct OrderedRegion
        {
            const FarLodChunkRecord* chunk{nullptr};
            float distanceSq{0.0f};
        };

        constexpr std::size_t kMaxSnapshotTiles = 8;
        LodDiagnosticsSnapshot snapshot{};

        std::lock_guard<std::mutex> lock(configMutex_);
        std::vector<OrderedRegion> orderedRegions;
        orderedRegions.reserve(chunks_.size());
        for (const auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (!chunk.active)
            {
                continue;
            }

            ++snapshot.activeTiles;
            if (chunk.gpu.resident)
            {
                ++snapshot.readyTiles;
            }
            if (chunk.dirty)
            {
                ++snapshot.dirtyTiles;
            }
            if (chunk.inFlight)
            {
                ++snapshot.inFlightTiles;
            }

            const glm::vec3 center = (chunkDrawBoundsMin(chunk) + chunkDrawBoundsMax(chunk)) * 0.5f;
            const glm::vec3 delta = center - cameraPos;
            orderedRegions.push_back(OrderedRegion{&chunk, glm::dot(delta, delta)});
        }

        snapshot.averageBuildMs = lastAverageBuildMs_;
        snapshot.averageGpuSynthesisMs = lastAverageGpuSynthesisMs_;
        snapshot.averageGpuStampMs = lastAverageGpuStampMs_;
        snapshot.averageGpuFaceBuildMs = lastAverageGpuFaceBuildMs_;

        std::sort(orderedRegions.begin(),
                  orderedRegions.end(),
                  [](const OrderedRegion& lhs, const OrderedRegion& rhs)
                  {
                      return lhs.distanceSq < rhs.distanceSq;
                  });

        snapshot.tiles.reserve(std::min<std::size_t>(orderedRegions.size(), kMaxSnapshotTiles));
        for (std::size_t tileIndex = 0; tileIndex < orderedRegions.size() && tileIndex < kMaxSnapshotTiles; ++tileIndex)
        {
            const FarLodChunkRecord& chunk = *orderedRegions[tileIndex].chunk;
            LodDiagnosticsTileSnapshot tileSnapshot{};
            tileSnapshot.level = chunk.key.level;
            tileSnapshot.tileCoord = glm::ivec2(chunk.key.coord.x, chunk.key.coord.z);
            tileSnapshot.distanceSq = orderedRegions[tileIndex].distanceSq;
            tileSnapshot.active = chunk.active;
            tileSnapshot.dirty = chunk.dirty;
            tileSnapshot.inFlight = chunk.inFlight;
            tileSnapshot.indexCount = chunk.gpu.indexCount;
            tileSnapshot.blockScaleBlocks = chunk.level.blockScale;
            tileSnapshot.chunkSpanBlocks = chunk.level.chunkSpanBlocks();
            tileSnapshot.worldMin = chunk.cpu.worldMin;
            tileSnapshot.worldMax = chunk.cpu.worldMin + glm::ivec3(chunk.level.chunkSpanBlocks());

            snapshot.tiles.push_back(tileSnapshot);
        }

        return snapshot;
    }

    void writeDebugSnapshot(const std::filesystem::path& outputPath, const glm::vec3& cameraPos) const
    {
        struct OrderedRegion
        {
            const FarLodChunkRecord* chunk{nullptr};
            float distanceSq{0.0f};
        };

        constexpr std::size_t kMaxSnapshotTiles = 12;
        auto writeBool = [](std::ostream& out, bool value)
        {
            out << (value ? "true" : "false");
        };
        auto writeVec3 = [](std::ostream& out, const glm::vec3& value)
        {
            out << '[' << value.x << ',' << value.y << ',' << value.z << ']';
        };
        std::lock_guard<std::mutex> lock(configMutex_);
        std::vector<OrderedRegion> orderedRegions;
        orderedRegions.reserve(chunks_.size());
        int activeTiles = 0;
        int readyTiles = 0;
        int dirtyTiles = 0;
        int inFlightTiles = 0;
        for (const auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (!chunk.active)
            {
                continue;
            }

            ++activeTiles;
            if (chunk.gpu.resident)
            {
                ++readyTiles;
            }
            if (chunk.dirty)
            {
                ++dirtyTiles;
            }
            if (chunk.inFlight)
            {
                ++inFlightTiles;
            }

            const glm::vec3 center = (chunkDrawBoundsMin(chunk) + chunkDrawBoundsMax(chunk)) * 0.5f;
            const glm::vec3 delta = center - cameraPos;
            orderedRegions.push_back(OrderedRegion{
                &chunk,
                glm::dot(delta, delta)});
        }

        std::sort(orderedRegions.begin(),
                  orderedRegions.end(),
                  [](const OrderedRegion& lhs, const OrderedRegion& rhs)
                  {
                      return lhs.distanceSq < rhs.distanceSq;
                  });

        std::error_code ec;
        const std::filesystem::path parentPath = outputPath.parent_path();
        if (!parentPath.empty())
        {
            std::filesystem::create_directories(parentPath, ec);
        }

        std::ofstream out(outputPath, std::ios::trunc);
        if (!out)
        {
            std::cerr << "Failed to open LOD debug snapshot: " << outputPath << std::endl;
            return;
        }

        out << "{\n";
        out << "  \"camera\":";
        writeVec3(out, cameraPos);
        out << ",\n";
        out << "  \"active_tiles\":" << activeTiles
            << ",\n  \"ready_tiles\":" << readyTiles
            << ",\n  \"dirty_tiles\":" << dirtyTiles
            << ",\n  \"in_flight_tiles\":" << inFlightTiles
            << ",\n  \"built_tiles_last_update\":" << builtTilesLastUpdate_
            << ",\n  \"skipped_tiles_last_update\":" << skippedTilesLastUpdate_
            << ",\n  \"average_cpu_terrain_synthesis_ms\":" << lastAverageCpuTerrainSynthesisMs_
            << ",\n  \"average_cpu_structure_stamp_ms\":" << lastAverageCpuStructureStampMs_
            << ",\n  \"average_cpu_mesh_ms\":" << lastAverageCpuMeshMs_
            << ",\n  \"average_upload_wait_ms\":" << lastAverageUploadWaitMs_
            << ",\n  \"average_upload_copy_ms\":" << lastAverageUploadCopyMs_
            << ",\n  \"rendered_face_count\":" << lastRenderedFaceCount_
            << ",\n  \"rendered_vertex_count\":" << lastRenderedVertexCount_
            << ",\n  \"tiles\":[\n";

        const std::size_t tileCount = std::min<std::size_t>(orderedRegions.size(), kMaxSnapshotTiles);
        for (std::size_t tileIndex = 0; tileIndex < tileCount; ++tileIndex)
        {
            const FarLodChunkRecord& chunk = *orderedRegions[tileIndex].chunk;
            if (tileIndex > 0)
            {
                out << ",\n";
            }

            out << "    {\n";
            out << "      \"level\":" << chunk.key.level
                << ",\n      \"chunk_x\":" << chunk.key.coord.x
                << ",\n      \"chunk_y\":" << chunk.key.coord.y
                << ",\n      \"chunk_z\":" << chunk.key.coord.z
                << ",\n      \"distance_sq\":" << orderedRegions[tileIndex].distanceSq
                << ",\n      \"active\":";
            writeBool(out, chunk.active);
            out << ",\n      \"dirty\":";
            writeBool(out, chunk.dirty);
            out << ",\n      \"in_flight\":";
            writeBool(out, chunk.inFlight);
            out << ",\n      \"vertex_count\":" << chunk.gpu.vertexCount
                << ",\n      \"index_count\":" << chunk.gpu.indexCount
                << ",\n      \"block_scale\":" << chunk.level.blockScale
                << ",\n      \"chunk_span_blocks\":" << chunk.level.chunkSpanBlocks()
                << ",\n      \"bounds_min\":";
            writeVec3(out, chunkDrawBoundsMin(chunk));
            out << ",\n      \"bounds_max\":";
            writeVec3(out, chunkDrawBoundsMax(chunk));

            out << "\n    }";
        }

        out << "\n  ]\n}\n";
    }

private:
    void startWorkers()
    {
        stopWorkers_.store(false, std::memory_order_release);
        const std::size_t desired = std::max<std::size_t>(workerCount_, 1);
        workerThreads_.reserve(desired);
        for (std::size_t i = 0; i < desired; ++i)
        {
            workerThreads_.emplace_back(&FarTerrainManager::workerThreadLoop, this);
        }
        if (!dirtyBuildPlannerThread_.joinable())
        {
            dirtyBuildPlannerThread_ = std::thread(&FarTerrainManager::dirtyBuildPlannerLoop, this);
        }
        if (!touchLevelPlannerThread_.joinable())
        {
            touchLevelPlannerThread_ = std::thread(&FarTerrainManager::touchLevelPlannerLoop, this);
        }
    }

    void stopWorkers()
    {
        stopWorkers_.store(true, std::memory_order_release);
        buildQueueCv_.notify_all();
        for (std::thread& worker : workerThreads_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        workerThreads_.clear();
        {
            std::lock_guard<std::mutex> lock(dirtyBuildPlanMutex_);
            hasPendingDirtyBuildPlanRequest_ = false;
            hasReadyDirtyBuildPlanResult_ = false;
            dirtyBuildPlannerBusy_ = false;
            pendingDirtyBuildPlanRequest_ = {};
            readyDirtyBuildPlanResult_ = {};
        }
        dirtyBuildPlanCv_.notify_all();
        if (dirtyBuildPlannerThread_.joinable())
        {
            dirtyBuildPlannerThread_.join();
        }
        {
            std::lock_guard<std::mutex> lock(touchLevelPlanMutex_);
            hasPendingTouchLevelPlanRequest_ = false;
            hasReadyTouchLevelPlanResult_ = false;
            touchLevelPlannerBusy_ = false;
            pendingTouchLevelPlanRequest_ = {};
            readyTouchLevelPlanResult_ = {};
            touchLevelCaches_.clear();
            hasLastTouchLevelPlanRequest_ = false;
            lastTouchLevelPlanLevels_.clear();
        }
        touchLevelPlanCv_.notify_all();
        if (touchLevelPlannerThread_.joinable())
        {
            touchLevelPlannerThread_.join();
        }
        {
            std::lock_guard<std::mutex> lock(buildQueueMutex_);
            buildQueue_.clear();
            queuedKeys_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(completedMutex_);
            completedBuilds_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            gpuSynthesisRequests_.clear();
            pendingGpuParityReadbacks_.clear();
        }
    }

    void dirtyBuildPlannerLoop()
    {
        for (;;)
        {
            DirtyBuildPlanRequest request{};
            {
                std::unique_lock<std::mutex> lock(dirtyBuildPlanMutex_);
                dirtyBuildPlanCv_.wait(lock,
                                       [this]
                                       {
                                           return stopWorkers_.load(std::memory_order_acquire) ||
                                                  hasPendingDirtyBuildPlanRequest_;
                                       });
                if (stopWorkers_.load(std::memory_order_acquire) && !hasPendingDirtyBuildPlanRequest_)
                {
                    return;
                }

                request = std::move(pendingDirtyBuildPlanRequest_);
                pendingDirtyBuildPlanRequest_ = {};
                hasPendingDirtyBuildPlanRequest_ = false;
                dirtyBuildPlannerBusy_ = true;
            }

            DirtyBuildPlanResult result = buildDirtyPlanFromSnapshot(request);

            {
                std::lock_guard<std::mutex> lock(dirtyBuildPlanMutex_);
                readyDirtyBuildPlanResult_ = std::move(result);
                hasReadyDirtyBuildPlanResult_ = true;
                dirtyBuildPlannerBusy_ = false;
            }
        }
    }

    void touchLevelPlannerLoop()
    {
        for (;;)
        {
            TouchLevelPlanRequest request{};
            {
                std::unique_lock<std::mutex> lock(touchLevelPlanMutex_);
                touchLevelPlanCv_.wait(lock,
                                       [this]
                                       {
                                           return stopWorkers_.load(std::memory_order_acquire) ||
                                                  hasPendingTouchLevelPlanRequest_;
                                       });
                if (stopWorkers_.load(std::memory_order_acquire) && !hasPendingTouchLevelPlanRequest_)
                {
                    return;
                }

                request = std::move(pendingTouchLevelPlanRequest_);
                pendingTouchLevelPlanRequest_ = {};
                hasPendingTouchLevelPlanRequest_ = false;
                touchLevelPlannerBusy_ = true;
            }

            TouchLevelPlanResult result = buildTouchLevelPlan(request);

            {
                std::lock_guard<std::mutex> lock(touchLevelPlanMutex_);
                readyTouchLevelPlanResult_ = std::move(result);
                hasReadyTouchLevelPlanResult_ = true;
                touchLevelPlannerBusy_ = false;
            }
        }
    }

    void workerThreadLoop()
    {
        for (;;)
        {
            BuildJob job{};
            {
                std::unique_lock<std::mutex> lock(buildQueueMutex_);
                buildQueueCv_.wait(lock,
                                   [this]
                                   {
                                       return stopWorkers_.load(std::memory_order_acquire) || !buildQueue_.empty();
                                   });
                if (stopWorkers_.load(std::memory_order_acquire) && buildQueue_.empty())
                {
                    return;
                }

                job = buildQueue_.front();
                buildQueue_.pop_front();
                queuedKeys_.erase(job.key);
            }

            ColumnSampleFn columnSampleFn;
            {
                std::lock_guard<std::mutex> lock(configMutex_);
                columnSampleFn = columnSampleFn_;
            }
            if (!columnSampleFn)
            {
                continue;
            }

            const SteadyClock::time_point buildStart = SteadyClock::now();
            BuildResult result{};
            result.key = job.key;
            result.buildVersion = job.buildVersion;
            result.epoch = job.epoch;
            {
                const SteadyClock::time_point terrainStart = SteadyClock::now();
                result.cpu.key = job.key;
                result.cpu.blockScale = job.level.blockScale;
                result.cpu.worldMin = job.key.coord * job.level.chunkSpanBlocks();
                result.cpu.boundsMin = glm::vec3(result.cpu.worldMin);
                result.cpu.boundsMax = glm::vec3(result.cpu.worldMin + glm::ivec3(job.level.chunkSpanBlocks()));
                result.cpuTerrainSynthesisMs =
                    std::chrono::duration<double, std::milli>(SteadyClock::now() - terrainStart).count();

                GpuSynthesisRequest gpuRequest{};
                gpuRequest.key = job.key;
                gpuRequest.buildVersion = job.buildVersion;
                gpuRequest.epoch = job.epoch;
                gpuRequest.worldMin = result.cpu.worldMin;
                gpuRequest.blockScale = result.cpu.blockScale;
                gpuRequest.lodLevel = job.level.level;
                gpuRequest.structureRegionKeys =
                    computeStructureRegionKeysForTile(result.cpu.worldMin, job.level.chunkSpanBlocks());
                gpuRequest.parityRequested = gpuParityEnabled_;
                if (gpuRequest.parityRequested)
                {
                    gpuRequest.cpuPackedParityVoxels =
                        std::make_shared<std::array<PackedFarLodVoxelGpu, kVoxelCount>>();
                    for (std::size_t voxelIdx = 0; voxelIdx < kVoxelCount; ++voxelIdx)
                    {
                        (*gpuRequest.cpuPackedParityVoxels)[voxelIdx] = packGpuVoxel(result.cpu.voxels[voxelIdx]);
                    }
                }
                {
                    std::lock_guard<std::mutex> gpuLock(gpuRequestMutex_);
                    gpuSynthesisRequests_.push_back(std::move(gpuRequest));
                }

            }
            result.buildMs =
                std::chrono::duration<double, std::milli>(SteadyClock::now() - buildStart).count();

            std::lock_guard<std::mutex> lock(completedMutex_);
            completedBuilds_.push_back(std::move(result));
        }
    }

    [[nodiscard]] static glm::vec3 chunkDrawBoundsMin(const FarLodChunkRecord& chunk) noexcept
    {
        return (chunk.gpu.resident && chunk.gpu.indexCount > 0) ? chunk.residentBoundsMin : chunk.cpu.boundsMin;
    }

    [[nodiscard]] static glm::vec3 chunkDrawBoundsMax(const FarLodChunkRecord& chunk) noexcept
    {
        return (chunk.gpu.resident && chunk.gpu.indexCount > 0) ? chunk.residentBoundsMax : chunk.cpu.boundsMax;
    }

    [[nodiscard]] static bool dirtyBuildJobPriorityLess(const BuildJob& lhs, const BuildJob& rhs) noexcept
    {
        const int lhsTier = !lhs.hadResidentMesh ? (lhs.fallbackOnly ? 1 : 0) : 2;
        const int rhsTier = !rhs.hadResidentMesh ? (rhs.fallbackOnly ? 1 : 0) : 2;
        if (lhsTier != rhsTier)
        {
            return lhsTier < rhsTier;
        }
        if (lhs.hadResidentMesh != rhs.hadResidentMesh)
        {
            return rhs.hadResidentMesh;
        }
        if (!lhs.hadResidentMesh && !rhs.hadResidentMesh)
        {
            if (lhs.level.level != rhs.level.level)
            {
                return lhs.level.level > rhs.level.level;
            }
        }
        else if (lhs.level.level != rhs.level.level)
        {
            return lhs.level.level < rhs.level.level;
        }
        return lhs.ringDistanceChunks < rhs.ringDistanceChunks;
    }

    static void trimDirtyBuildJobs(std::vector<BuildJob>& jobs,
                                   std::size_t workerBudget,
                                   int exactMissingChunks,
                                   std::size_t exactPendingUploads)
    {
        static constexpr std::size_t kFallbackReservation = 2;
        const std::size_t exactBacklogPenalty =
            (exactMissingChunks > 32 || exactPendingUploads > 24)
                ? 8
                : ((exactMissingChunks > 8 || exactPendingUploads > 8) ? 16 : 32);
        const std::size_t maxQueuedPerUpdate = std::max<std::size_t>(workerBudget * 12, exactBacklogPenalty);
        if (jobs.size() <= maxQueuedPerUpdate)
        {
            return;
        }

        std::vector<BuildJob> trimmed;
        trimmed.reserve(maxQueuedPerUpdate);
        std::size_t fallbackAdded = 0;
        const std::size_t desiredFallback = std::min(kFallbackReservation, maxQueuedPerUpdate);
        for (const BuildJob& job : jobs)
        {
            if (trimmed.size() >= maxQueuedPerUpdate)
            {
                break;
            }
            if (job.fallbackOnly)
            {
                continue;
            }
            const std::size_t remaining = maxQueuedPerUpdate - trimmed.size();
            if (remaining <= (desiredFallback - fallbackAdded))
            {
                break;
            }
            trimmed.push_back(job);
        }
        if (trimmed.size() < maxQueuedPerUpdate && fallbackAdded < desiredFallback)
        {
            for (const BuildJob& job : jobs)
            {
                if (!job.fallbackOnly)
                {
                    continue;
                }
                if (trimmed.size() >= maxQueuedPerUpdate)
                {
                    break;
                }
                trimmed.push_back(job);
                ++fallbackAdded;
                if (fallbackAdded >= desiredFallback)
                {
                    break;
                }
            }
        }
        jobs.swap(trimmed);
    }

    [[nodiscard]] DirtyBuildPlanResult buildDirtyPlanFromSnapshot(const DirtyBuildPlanRequest& request)
    {
        DirtyBuildPlanResult result{};
        result.sequence = request.sequence;
        result.epoch = request.epoch;

        std::unordered_set<FarLodChunkKey, FarLodChunkKeyHasher> queuedKeysSnapshot;
        {
            std::lock_guard<std::mutex> lock(buildQueueMutex_);
            queuedKeysSnapshot = queuedKeys_;
        }

        constexpr float kForwardDotThreshold = -0.35f;
        const glm::vec2 forwardXZ = normalizePriorityForwardXZ(request.cameraForward);

        std::lock_guard<std::mutex> lock(configMutex_);
        for (const auto& [key, chunk] : chunks_)
        {
            if (!chunk.active || !chunk.dirty || chunk.inFlight || queuedKeysSnapshot.contains(key))
            {
                continue;
            }

            const bool keepResidentStable = chunk.gpu.resident && chunk.gpu.indexCount > 0;
            const int ringDistanceChunks =
                chunkMinHorizontalRingDistanceChunks(chunk.level, request.cameraChunk, key.coord);
            const int nearBuildGraceChunks = std::max(2, chunk.level.blockScale * 2);
            const bool keepNearbyStable = ringDistanceChunks <= nearBuildGraceChunks;

            if (keepResidentStable)
            {
                const std::uint64_t cadence = residentRefreshCadenceUpdates(chunk.level);
                if (cadence > 1u && ringDistanceChunks > nearBuildGraceChunks)
                {
                    const std::uint64_t stagger = staggeredChunkUpdateHash(key) % cadence;
                    if ((updateStamp_ + stagger) % cadence != 0u)
                    {
                        continue;
                    }
                }
            }

            if (!keepResidentStable && !keepNearbyStable)
            {
                const glm::vec3 boundsMin = chunkDrawBoundsMin(chunk);
                const glm::vec3 boundsMax = chunkDrawBoundsMax(chunk);
                const glm::vec3 center = (boundsMin + boundsMax) * 0.5f;
                const glm::vec2 toChunk(center.x - static_cast<float>(request.cameraChunk.x * kChunkSizeX),
                                        center.z - static_cast<float>(request.cameraChunk.z * kChunkSizeZ));
                if (glm::dot(toChunk, toChunk) > kEpsilon)
                {
                    const float facing = glm::dot(glm::normalize(toChunk), forwardXZ);
                    if (facing < kForwardDotThreshold)
                    {
                        continue;
                    }
                }
                if (request.hasVisibilityFrustum && !request.visibilityFrustum.intersectsAABB(boundsMin, boundsMax))
                {
                    const glm::vec3 toChunk3 = center - request.visibilityCameraPos;
                    if (glm::dot(toChunk3, toChunk3) >
                        static_cast<float>(chunk.level.chunkSpanBlocks() * chunk.level.chunkSpanBlocks() * 4))
                    {
                        continue;
                    }
                }
            }

            result.jobs.push_back(BuildJob{
                key,
                chunk.level,
                chunk.buildVersion,
                request.epoch,
                ringDistanceChunks,
                keepResidentStable,
                chunk.fallbackOnly});
        }

        std::sort(result.jobs.begin(), result.jobs.end(), dirtyBuildJobPriorityLess);
        trimDirtyBuildJobs(result.jobs,
                           request.workerBudget,
                           request.exactMissingChunks,
                           request.exactPendingUploads);
        return result;
    }

    void requestDirtyBuildPlan()
    {
        DirtyBuildPlanRequest request{};
        request.sequence = dirtyBuildPlanSequence_.fetch_add(1, std::memory_order_acq_rel) + 1u;
        request.epoch = buildEpoch_.load(std::memory_order_acquire);
        request.cameraChunk = cameraChunk_;
        request.cameraForward = cameraForward_;
        request.exactMissingChunks = exactMissingChunks_.load(std::memory_order_relaxed);
        request.exactPendingUploads = exactPendingUploads_.load(std::memory_order_relaxed);
        request.workerBudget = std::max<std::size_t>(workerCount_, 1);
        {
            std::lock_guard<std::mutex> lock(configMutex_);
            request.visibilityFrustum = lastVisibilityFrustum_;
            request.visibilityCameraPos = lastVisibilityCameraPos_;
            request.hasVisibilityFrustum = hasVisibilityFrustum_;
        }

        std::lock_guard<std::mutex> lock(dirtyBuildPlanMutex_);
        if (dirtyBuildPlannerBusy_ || hasPendingDirtyBuildPlanRequest_)
        {
            return;
        }
        pendingDirtyBuildPlanRequest_ = std::move(request);
        hasPendingDirtyBuildPlanRequest_ = true;
        dirtyBuildPlanCv_.notify_one();
    }

    void consumeReadyDirtyBuildPlan()
    {
        DirtyBuildPlanResult result{};
        {
            std::lock_guard<std::mutex> lock(dirtyBuildPlanMutex_);
            if (!hasReadyDirtyBuildPlanResult_)
            {
                return;
            }
            result = std::move(readyDirtyBuildPlanResult_);
            readyDirtyBuildPlanResult_ = {};
            hasReadyDirtyBuildPlanResult_ = false;
        }

        if (result.jobs.empty())
        {
            return;
        }

        const std::uint64_t currentEpoch = buildEpoch_.load(std::memory_order_acquire);
        std::vector<BuildJob> readyJobs;
        readyJobs.reserve(result.jobs.size());
        {
            std::lock_guard<std::mutex> lock(configMutex_);
            if (result.epoch != currentEpoch)
            {
                return;
            }

            for (const BuildJob& job : result.jobs)
            {
                auto it = chunks_.find(job.key);
                if (it == chunks_.end())
                {
                    continue;
                }

                FarLodChunkRecord& chunk = it->second;
                if (!chunk.active || !chunk.dirty || chunk.inFlight || chunk.buildVersion != job.buildVersion)
                {
                    continue;
                }

                chunk.inFlight = true;
                readyJobs.push_back(job);
            }
        }

        if (readyJobs.empty())
        {
            return;
        }

        std::vector<FarLodChunkKey> duplicateKeys;
        duplicateKeys.reserve(readyJobs.size());
        {
            std::lock_guard<std::mutex> lock(buildQueueMutex_);
            for (const BuildJob& job : readyJobs)
            {
                if (!queuedKeys_.insert(job.key).second)
                {
                    duplicateKeys.push_back(job.key);
                    continue;
                }
                buildQueue_.push_back(job);
            }
        }

        if (!duplicateKeys.empty())
        {
            std::lock_guard<std::mutex> lock(configMutex_);
            for (const FarLodChunkKey& key : duplicateKeys)
            {
                auto it = chunks_.find(key);
                if (it != chunks_.end())
                {
                    it->second.inFlight = false;
                }
            }
        }

        if (readyJobs.size() > duplicateKeys.size())
        {
            buildQueueCv_.notify_all();
        }
    }

    void scheduleDirtyBuilds()
    {
        consumeReadyDirtyBuildPlan();
        requestDirtyBuildPlan();
    }

    void submitGpuSynthesisRequests(double budgetMs)
    {
        std::deque<GpuSynthesisRequest> requests;
        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            requests.swap(gpuSynthesisRequests_);
        }

        lastAverageGpuSynthesisMs_ = 0.0;
        lastAverageGpuStampMs_ = 0.0;
        lastAverageGpuFaceBuildMs_ = 0.0;
        if (requests.empty() || !gpuContext_.ready())
        {
            if (!requests.empty())
            {
                std::lock_guard<std::mutex> lock(gpuRequestMutex_);
                for (GpuSynthesisRequest& request : requests)
                {
                    gpuSynthesisRequests_.push_back(std::move(request));
                }
            }
            return;
        }

        gpuContext_.setReadbackEnabled(false);
        if (!gpuContext_.begin())
        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            for (GpuSynthesisRequest& request : requests)
            {
                gpuSynthesisRequests_.push_back(std::move(request));
            }
            return;
        }

        const FarLodWorkBudget workBudget = computeWorkBudget();
        appendAtlasUpdates(requests, workBudget.atlasUpdateCells);

        std::vector<PendingGpuParityReadback> pendingReadbacks;
        std::vector<FarLodChunkKey> submittedKeys;
        std::vector<FarLodChunkKey> stagedGpuMeshes;
        const double budgetLimit = budgetMs <= 0.0 ? std::numeric_limits<double>::max() : budgetMs;
        const SteadyClock::time_point submitStart = SteadyClock::now();
        double totalGpuSynthesisMs = 0.0;
        double totalGpuStampMs = 0.0;
        double totalGpuFaceBuildMs = 0.0;
        std::size_t submittedCount = 0;
        std::size_t synthSubmittedCount = 0;
        std::size_t faceBuildSamples = 0;
        std::size_t remainingGpuBudgetUnits = std::max<std::size_t>(workBudget.gpuDispatchBudgetUnits, 1);
        const std::size_t maxGpuSubmissions = std::max<std::size_t>(workBudget.maxGpuSubmissions, 1);
        auto computeMaxMergeExtent = [](int blockScale) noexcept
        {
            return (blockScale <= 2)   ? 2u
                 : (blockScale <= 4)   ? 4u
                 : (blockScale <= 8)   ? 8u
                                       : 16u;
        };

        while (!requests.empty())
        {
            if (submittedCount >= maxGpuSubmissions)
            {
                break;
            }
            if (std::chrono::duration<double, std::milli>(SteadyClock::now() - submitStart).count() >= budgetLimit)
            {
                break;
            }

            GpuSynthesisRequest request = std::move(requests.front());
            requests.pop_front();

            std::lock_guard<std::mutex> lock(configMutex_);
            auto it = chunks_.find(request.key);
            if (it == chunks_.end())
            {
                continue;
            }

            FarLodChunkRecord& chunk = it->second;
            if (!chunk.active || request.epoch != buildEpoch_.load(std::memory_order_acquire) ||
                request.buildVersion != chunk.buildVersion ||
                chunk.gpu.columnBuffer == nullptr)
            {
                continue;
            }

            auto atlasIt = levelAtlases_.find(request.key.level);
            if (atlasIt == levelAtlases_.end() || atlasIt->second.buffer == nullptr)
            {
                continue;
            }
            FarLodLevelAtlasState& atlas = atlasIt->second;

            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD GPU submit chunk level=" << request.key.level
                       << " coord=[" << request.key.coord.x << "," << request.key.coord.y << "," << request.key.coord.z << "]"
                       << " buildVersion=" << request.buildVersion
                       << " epoch=" << request.epoch
                       << " blockScale=" << request.blockScale
                       << " lodLevel=" << request.lodLevel
                       << " parity=" << (request.parityRequested ? "on" : "off")
                       << " columnStateBefore=" << resourceStateName(chunk.gpu.columnState);
                chunkManagerDebugLog(stream.str());
            }

            const SteadyClock::time_point synthStart = SteadyClock::now();
            gpuContext_.transition(chunk.gpu.columnBuffer.Get(),
                                   chunk.gpu.columnState,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            chunk.gpu.columnState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            gpuContext_.dispatchSynth(request.worldMin,
                                      request.blockScale,
                                      worldgenTables_.header.seaLevel,
                                      atlas.originCell,
                                      atlas.atlasSizeCells,
                                      atlas.buffer.Get(),
                                      atlas.elementCount(),
                                      chunk.gpu.columnBuffer.Get());
            gpuContext_.uavBarrier(chunk.gpu.columnBuffer.Get());
            chunk.gpu.columnState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD GPU synth dispatched chunk level=" << request.key.level
                       << " coord=[" << request.key.coord.x << "," << request.key.coord.y << "," << request.key.coord.z << "]";
                chunkManagerDebugLog(stream.str());
            }
            totalGpuSynthesisMs +=
                std::chrono::duration<double, std::milli>(SteadyClock::now() - synthStart).count();

            const bool canBuildGpuMesh =
                blockUvBuffer_ != nullptr &&
                blockUvCount_ > 0 &&
                emptyVoxelBuffer_ != nullptr &&
                chunk.gpu.faceCountBuffer != nullptr &&
                chunk.gpu.faceAnalysisBuffer != nullptr &&
                chunk.gpu.faceDescriptorBuffer != nullptr &&
                chunk.gpu.facePrefixBuffer != nullptr &&
                chunk.gpu.faceGroupSumBuffer != nullptr;
            const std::size_t synthCostUnits = canBuildGpuMesh ? 8u : 2u;
            if (remainingGpuBudgetUnits < synthCostUnits)
            {
                requests.push_front(std::move(request));
                break;
            }
            if (canBuildGpuMesh)
            {
                ID3D12Resource* neighborPosX = emptyVoxelBuffer_.Get();
                ID3D12Resource* neighborNegX = emptyVoxelBuffer_.Get();
                ID3D12Resource* neighborPosZ = emptyVoxelBuffer_.Get();
                ID3D12Resource* neighborNegZ = emptyVoxelBuffer_.Get();
                auto bindNeighbor = [&](const glm::ivec3& offset,
                                        ID3D12Resource*& outBuffer)
                {
                    const FarLodChunkKey neighborKey{request.key.level, request.key.coord + offset};
                    auto neighborIt = chunks_.find(neighborKey);
                    if (neighborIt == chunks_.end())
                    {
                        return;
                    }
                    FarLodChunkRecord& neighbor = neighborIt->second;
                    if (!neighbor.gpu.voxelReady || neighbor.gpu.columnBuffer == nullptr)
                    {
                        return;
                    }
                    if (gpuContext_.completedFenceValue() < neighbor.gpu.voxelFenceValue)
                    {
                        return;
                    }
                    outBuffer = neighbor.gpu.columnBuffer.Get();
                    if (neighbor.gpu.columnState != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
                    {
                        gpuContext_.transition(neighbor.gpu.columnBuffer.Get(),
                                               neighbor.gpu.columnState,
                                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                        neighbor.gpu.columnState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    }
                };

                bindNeighbor(glm::ivec3(1, 0, 0), neighborPosX);
                bindNeighbor(glm::ivec3(-1, 0, 0), neighborNegX);
                bindNeighbor(glm::ivec3(0, 0, 1), neighborPosZ);
                bindNeighbor(glm::ivec3(0, 0, -1), neighborNegZ);

                const std::uint32_t faceCapacity = requestedFaceCapacity(chunk);
                const std::size_t reservedVertexCount = static_cast<std::size_t>(faceCapacity) * 4u;
                const std::uint32_t reservedIndexCount = faceCapacity * 6u;
                Allocation allocation = acquireAllocation(reservedVertexCount, reservedIndexCount);
                if (allocation.pageIndex == kInvalidChunkBufferPage || allocation.pageIndex >= bufferPages_.size())
                {
                    requests.push_front(std::move(request));
                    break;
                }
                if (!stageGpuMeshForCommit(chunk,
                                           allocation,
                                           faceCapacity,
                                           reservedVertexCount,
                                           reservedIndexCount,
                                           request.buildVersion,
                                           request.epoch))
                {
                    releaseAllocationRange(allocation.pageIndex,
                                           allocation.vertexOffset,
                                           reservedVertexCount,
                                           allocation.indexOffset,
                                           reservedIndexCount,
                                           allocation.recordIndex);
                    requests.push_front(std::move(request));
                    break;
                }

                if (chunk.gpu.columnState != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
                {
                    gpuContext_.transition(chunk.gpu.columnBuffer.Get(),
                                           chunk.gpu.columnState,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    chunk.gpu.columnState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                }
                gpuContext_.transition(chunk.gpu.faceCountBuffer.Get(),
                                       chunk.gpu.faceCountState,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chunk.gpu.faceCountState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                gpuContext_.transition(chunk.gpu.faceAnalysisBuffer.Get(),
                                       chunk.gpu.faceAnalysisState,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chunk.gpu.faceAnalysisState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                gpuContext_.transition(chunk.gpu.faceDescriptorBuffer.Get(),
                                       chunk.gpu.faceDescriptorState,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chunk.gpu.faceDescriptorState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                gpuContext_.transition(chunk.gpu.facePrefixBuffer.Get(),
                                       chunk.gpu.facePrefixState,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chunk.gpu.facePrefixState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                gpuContext_.transition(chunk.gpu.faceGroupSumBuffer.Get(),
                                       chunk.gpu.faceGroupSumState,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                chunk.gpu.faceGroupSumState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

                BufferPage& page = bufferPages_[allocation.pageIndex];
                if (page.vertexBuffer != nullptr)
                {
                    gpuContext_.transition(page.vertexBuffer.Get(),
                                           page.vertexState,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    page.vertexState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                }
                if (page.indexBuffer != nullptr)
                {
                    gpuContext_.transition(page.indexBuffer.Get(),
                                           page.indexState,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    page.indexState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                }
                if (page.drawRecordBuffer != nullptr)
                {
                    gpuContext_.transition(page.drawRecordBuffer.Get(),
                                           page.drawRecordState,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    page.drawRecordState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                }

                const SteadyClock::time_point faceBuildStart = SteadyClock::now();
                const std::uint32_t maxMergeExtent = computeMaxMergeExtent(request.blockScale);
                gpuContext_.dispatchFaceCount(static_cast<int>(std::floor(chunk.cpu.boundsMin.y)),
                                              request.blockScale,
                                              0u,
                                              maxMergeExtent,
                                              chunk.gpu.columnBuffer.Get(),
                                              neighborPosX,
                                              neighborNegX,
                                              neighborPosZ,
                                              neighborNegZ,
                                              chunk.gpu.faceCountBuffer.Get(),
                                              chunk.gpu.faceAnalysisBuffer.Get(),
                                              chunk.gpu.faceDescriptorBuffer.Get());
                gpuContext_.uavBarrier(chunk.gpu.faceCountBuffer.Get());
                gpuContext_.uavBarrier(chunk.gpu.faceAnalysisBuffer.Get());
                gpuContext_.uavBarrier(chunk.gpu.faceDescriptorBuffer.Get());
                gpuContext_.dispatchFacePrefix(chunk.gpu.faceCountBuffer.Get(),
                                               chunk.gpu.facePrefixBuffer.Get(),
                                               chunk.gpu.faceGroupSumBuffer.Get());
                gpuContext_.uavBarrier(chunk.gpu.facePrefixBuffer.Get());
                gpuContext_.dispatchFaceEmit(glm::ivec3(request.worldMin.x,
                                                        static_cast<int>(std::floor(chunk.cpu.boundsMin.y)),
                                                        request.worldMin.z),
                                             request.blockScale,
                                             maxMergeExtent,
                                             static_cast<std::uint32_t>(allocation.vertexOffset),
                                             static_cast<std::uint32_t>(allocation.indexOffset),
                                             allocation.recordIndex,
                                             faceCapacity,
                                             chunk.gpu.columnBuffer.Get(),
                                             chunk.gpu.faceCountBuffer.Get(),
                                             chunk.gpu.faceAnalysisBuffer.Get(),
                                             chunk.gpu.faceDescriptorBuffer.Get(),
                                             chunk.gpu.facePrefixBuffer.Get(),
                                             blockUvBuffer_.Get(),
                                             blockUvCount_,
                                             static_cast<std::uint32_t>(sizeof(GpuBlockFaceUv)),
                                             page.vertexBuffer.Get(),
                                             static_cast<std::uint32_t>(page.vertexCapacity),
                                             page.indexBuffer.Get(),
                                             static_cast<std::uint32_t>(page.indexCapacity),
                                             page.drawRecordBuffer.Get(),
                                             static_cast<std::uint32_t>(page.recordCapacity));
                if (page.vertexBuffer != nullptr)
                {
                    gpuContext_.uavBarrier(page.vertexBuffer.Get());
                }
                if (page.indexBuffer != nullptr)
                {
                    gpuContext_.uavBarrier(page.indexBuffer.Get());
                }
                if (page.drawRecordBuffer != nullptr)
                {
                    gpuContext_.uavBarrier(page.drawRecordBuffer.Get());
                    gpuContext_.transition(page.drawRecordBuffer.Get(),
                                           page.drawRecordState,
                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
                    page.drawRecordState = D3D12_RESOURCE_STATE_COPY_SOURCE;
                    gpuContext_.copyBuffer(chunk.pendingMesh.drawRecordReadbackBuffer.Get(),
                                           0,
                                           page.drawRecordBuffer.Get(),
                                           static_cast<std::uint64_t>(allocation.recordIndex) *
                                               sizeof(ChunkRenderBatch::GpuCullRecord),
                                           static_cast<std::uint64_t>(sizeof(ChunkRenderBatch::GpuCullRecord)));
                }
                gpuContext_.transition(chunk.gpu.faceCountBuffer.Get(),
                                       chunk.gpu.faceCountState,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                chunk.gpu.faceCountState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                gpuContext_.transition(chunk.gpu.faceAnalysisBuffer.Get(),
                                       chunk.gpu.faceAnalysisState,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                chunk.gpu.faceAnalysisState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                gpuContext_.transition(chunk.gpu.faceDescriptorBuffer.Get(),
                                       chunk.gpu.faceDescriptorState,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                chunk.gpu.faceDescriptorState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                gpuContext_.transition(chunk.gpu.facePrefixBuffer.Get(),
                                       chunk.gpu.facePrefixState,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                chunk.gpu.facePrefixState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                gpuContext_.transition(chunk.gpu.faceGroupSumBuffer.Get(),
                                       chunk.gpu.faceGroupSumState,
                                       D3D12_RESOURCE_STATE_COMMON);
                chunk.gpu.faceGroupSumState = D3D12_RESOURCE_STATE_COMMON;
                if (page.vertexBuffer != nullptr)
                {
                    gpuContext_.transition(page.vertexBuffer.Get(),
                                           page.vertexState,
                                           D3D12_RESOURCE_STATE_COMMON);
                    page.vertexState = D3D12_RESOURCE_STATE_COMMON;
                }
                if (page.indexBuffer != nullptr)
                {
                    gpuContext_.transition(page.indexBuffer.Get(),
                                           page.indexState,
                                           D3D12_RESOURCE_STATE_COMMON);
                    page.indexState = D3D12_RESOURCE_STATE_COMMON;
                }
                if (page.drawRecordBuffer != nullptr)
                {
                    gpuContext_.transition(page.drawRecordBuffer.Get(),
                                           page.drawRecordState,
                                           D3D12_RESOURCE_STATE_COMMON);
                    page.drawRecordState = D3D12_RESOURCE_STATE_COMMON;
                }

                totalGpuFaceBuildMs +=
                    std::chrono::duration<double, std::milli>(SteadyClock::now() - faceBuildStart).count();
                ++faceBuildSamples;
                stagedGpuMeshes.push_back(request.key);
            }

            if (request.parityRequested)
            {
                gpuContext_.transition(chunk.gpu.columnBuffer.Get(),
                                       chunk.gpu.columnState,
                                       D3D12_RESOURCE_STATE_COMMON);
                chunk.gpu.columnState = D3D12_RESOURCE_STATE_COMMON;
                PendingGpuParityReadback pending{};
                pending.key = request.key;
                pending.buildVersion = request.buildVersion;
                pending.epoch = request.epoch;
                pending.cpuPackedParityVoxels = std::move(request.cpuPackedParityVoxels);
                pending.cpuColumnParityDescriptors = std::move(request.cpuColumnParityDescriptors);
                pending.parityRequested = true;
                pending.copySubmitted = false;
                pendingReadbacks.push_back(std::move(pending));
            }
            else
            {
                gpuContext_.transition(chunk.gpu.columnBuffer.Get(),
                                       chunk.gpu.columnState,
                                       D3D12_RESOURCE_STATE_COMMON);
                chunk.gpu.columnState = D3D12_RESOURCE_STATE_COMMON;
            }

            submittedKeys.push_back(request.key);
            remainingGpuBudgetUnits -= synthCostUnits;
            ++synthSubmittedCount;
            ++submittedCount;
        }

        const FarLodGpuContext::FlushResult flushResult = gpuContext_.flush();
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD GPU submit batch flushed submittedCount=" << submittedCount
                   << " stagedGpuMeshes=" << stagedGpuMeshes.size()
                   << " pendingParity=" << pendingReadbacks.size()
                   << " fence=" << flushResult.fenceValue;
            chunkManagerDebugLog(stream.str());
        }
        for (PendingGpuParityReadback& pending : pendingReadbacks)
        {
            pending.computeFenceValue = flushResult.fenceValue;
        }

        {
            std::lock_guard<std::mutex> lock(configMutex_);
            for (const PendingGpuParityReadback& pending : pendingReadbacks)
            {
                auto it = chunks_.find(pending.key);
                if (it == chunks_.end())
                {
                    continue;
                }
                it->second.gpu.voxelFenceValue = flushResult.fenceValue;
                it->second.gpu.voxelReady = (flushResult.fenceValue == 0);
                it->second.gpu.parityValidated = false;
            }
            if (!gpuParityEnabled_)
            {
                for (const FarLodChunkKey& key : submittedKeys)
                {
                    auto it = chunks_.find(key);
                    if (it == chunks_.end())
                    {
                        continue;
                    }
                    it->second.gpu.voxelFenceValue = flushResult.fenceValue;
                    it->second.gpu.voxelReady = true;
                    it->second.gpu.parityValidated = false;
                }
            }
            for (const FarLodChunkKey& key : stagedGpuMeshes)
            {
                auto it = chunks_.find(key);
                if (it == chunks_.end())
                {
                    continue;
                }
                FarLodChunkRecord& chunk = it->second;
                if (chunk.pendingMesh.valid() && chunk.pendingMesh.gpuGenerated &&
                    chunk.pendingMesh.buildVersion == chunk.buildVersion &&
                    chunk.pendingMesh.epoch == buildEpoch_.load(std::memory_order_acquire))
                {
                    chunk.pendingMesh.gpuFenceValue = flushResult.fenceValue;
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            for (GpuSynthesisRequest& request : requests)
            {
                gpuSynthesisRequests_.push_back(std::move(request));
            }
            for (PendingGpuParityReadback& pending : pendingReadbacks)
            {
                pendingGpuParityReadbacks_.push_back(std::move(pending));
            }
        }

        if (submittedCount > 0)
        {
            if (synthSubmittedCount > 0)
            {
                lastAverageGpuSynthesisMs_ = totalGpuSynthesisMs / static_cast<double>(synthSubmittedCount);
                lastAverageGpuStampMs_ = totalGpuStampMs / static_cast<double>(synthSubmittedCount);
            }
            else
            {
                lastAverageGpuSynthesisMs_ = 0.0;
                lastAverageGpuStampMs_ = 0.0;
            }
            if (faceBuildSamples > 0)
            {
                lastAverageGpuFaceBuildMs_ = totalGpuFaceBuildMs / static_cast<double>(faceBuildSamples);
                if (benchmarkMetrics_ != nullptr)
                {
                    benchmarkMetrics_->lodGpuFaceBuildStage.recordMicros(
                        static_cast<std::uint64_t>(std::max(lastAverageGpuFaceBuildMs_, 0.0) * 1000.0));
                }
            }
            if (benchmarkMetrics_ != nullptr)
            {
                benchmarkMetrics_->lodGpuSynthesisStage.recordMicros(
                    static_cast<std::uint64_t>(std::max(lastAverageGpuSynthesisMs_, 0.0) * 1000.0));
                benchmarkMetrics_->lodGpuStampStage.recordMicros(
                    static_cast<std::uint64_t>(std::max(lastAverageGpuStampMs_, 0.0) * 1000.0));
            }
        }
    }

    void collectCompletedBuilds(double uploadBudgetMs)
    {
        const SteadyClock::time_point collectStart = SteadyClock::now();
        std::deque<BuildResult> completed;
        {
            std::lock_guard<std::mutex> lock(completedMutex_);
            completed.swap(completedBuilds_);
        }

        builtTilesLastUpdate_ = 0;
        skippedTilesLastUpdate_ = 0;
        lastBuiltFaceCount_ = 0;
        lastBuiltVertexCount_ = 0;
        lastBuiltIndexCount_ = 0;
        lastAverageBuildMs_ = 0.0;
        lastAverageCpuTerrainSynthesisMs_ = 0.0;
        lastAverageCpuStructureStampMs_ = 0.0;
        lastAverageCpuMeshMs_ = 0.0;
        lastAverageUploadWaitMs_ = 0.0;
        lastAverageUploadCopyMs_ = 0.0;
        if (completed.empty())
        {
            lastCollectMs_ =
                std::chrono::duration<double, std::milli>(SteadyClock::now() - collectStart).count();
            lastUploadMs_ = 0.0;
            return;
        }

        const double uploadBudgetLimit = uploadBudgetMs <= 0.0 ? std::numeric_limits<double>::max() : uploadBudgetMs;
        const SteadyClock::time_point uploadStart = SteadyClock::now();
        double totalBuildMs = 0.0;
        double totalCpuTerrainSynthesisMs = 0.0;
        double totalCpuStructureStampMs = 0.0;
        double totalCpuMeshMs = 0.0;
        double totalUploadCopyMs = 0.0;
        std::size_t applied = 0;
        std::deque<BuildResult> stillPendingCompleted;

        while (!completed.empty())
        {
            if (std::chrono::duration<double, std::milli>(SteadyClock::now() - uploadStart).count() >= uploadBudgetLimit)
            {
                break;
            }

            BuildResult result = std::move(completed.front());
            completed.pop_front();

            std::lock_guard<std::mutex> lock(configMutex_);
            auto it = chunks_.find(result.key);
            if (it == chunks_.end())
            {
                continue;
            }

            FarLodChunkRecord& chunk = it->second;
            chunk.inFlight = false;
            if (result.epoch != buildEpoch_.load(std::memory_order_acquire) || result.buildVersion != chunk.buildVersion)
            {
                continue;
            }

            if (result.skippedByRelevance)
            {
                if (lodVisibilityDebugLoggingEnabled())
                {
                    std::ostringstream stream;
                    stream << "lodvis skipped_by_relevance_preserve_resident level=" << chunk.key.level
                           << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                           << " resident=" << (chunk.gpu.resident ? "y" : "n")
                           << " indexCount=" << chunk.gpu.indexCount;
                    lodVisibilityDebugLog(stream.str());
                }
                chunk.dirty = true;
                ++skippedTilesLastUpdate_;
                continue;
            }

            const std::uint64_t currentDependencyRevision = currentAtlasDependencyRevision(chunk);
            const bool staged =
                chunk.pendingMesh.valid() &&
                chunk.pendingMesh.gpuGenerated &&
                chunk.pendingMesh.epoch == result.epoch &&
                chunk.pendingMesh.buildVersion == result.buildVersion;
            const bool canPreserveResidentMesh =
                chunk.gpu.resident &&
                chunk.gpu.indexCount > 0 &&
                chunk.lastBuiltAtlasDependencyRevision != 0 &&
                currentDependencyRevision == chunk.lastBuiltAtlasDependencyRevision;
            if (!staged)
            {
                if (canPreserveResidentMesh)
                {
                    chunk.dirty = false;
                    ++skippedTilesLastUpdate_;
                    if (lodVisibilityDebugLoggingEnabled())
                    {
                        std::ostringstream stream;
                        stream << "lodvis apply_skip_unchanged_gpu_mesh level=" << chunk.key.level
                               << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                               << " resident=y"
                               << " indexCount=" << chunk.gpu.indexCount
                               << " dependencyRevision=" << currentDependencyRevision;
                        lodVisibilityDebugLog(stream.str());
                    }
                    continue;
                }
                // Keep the old resident allocation alive; this chunk remains dirty until a replacement commits.
                chunk.dirty = true;
                ++skippedTilesLastUpdate_;
                if (lodVisibilityDebugLoggingEnabled())
                {
                    std::ostringstream stream;
                    stream << "lodvis apply_deferred_no_gpu_mesh level=" << chunk.key.level
                           << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                           << " resident=" << (chunk.gpu.resident ? "y" : "n")
                           << " indexCount=" << chunk.gpu.indexCount;
                    lodVisibilityDebugLog(stream.str());
                }
                continue;
            }

            chunk.dirty = false;
            chunk.lastBuiltAtlasDependencyRevision = currentDependencyRevision;
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis apply_gpu_mesh_staged level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " resident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount
                       << " pendingReservedIndexCount=" << chunk.pendingMesh.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            ++builtTilesLastUpdate_;
            ++applied;
            totalBuildMs += result.buildMs;
            totalCpuTerrainSynthesisMs += result.cpuTerrainSynthesisMs;
            totalCpuStructureStampMs += result.cpuStructureStampMs;
            totalCpuMeshMs += result.cpuMeshMs;
            totalUploadCopyMs += result.uploadCopyMs;
        }

        if (applied > 0)
        {
            lastAverageBuildMs_ = totalBuildMs / static_cast<double>(applied);
            lastAverageCpuTerrainSynthesisMs_ = totalCpuTerrainSynthesisMs / static_cast<double>(applied);
            lastAverageCpuStructureStampMs_ = totalCpuStructureStampMs / static_cast<double>(applied);
            lastAverageCpuMeshMs_ = totalCpuMeshMs / static_cast<double>(applied);
            lastAverageUploadCopyMs_ = totalUploadCopyMs / static_cast<double>(applied);
            if (benchmarkMetrics_ != nullptr)
            {
                benchmarkMetrics_->farBuildStage.recordMicros(
                    static_cast<std::uint64_t>(std::max(lastAverageBuildMs_, 0.0) * 1000.0));
            }
        }

        {
            std::lock_guard<std::mutex> lock(completedMutex_);
            for (BuildResult& result : completed)
            {
                completedBuilds_.push_back(std::move(result));
            }
            for (BuildResult& result : stillPendingCompleted)
            {
                completedBuilds_.push_back(std::move(result));
            }
        }
        lastCollectMs_ = std::chrono::duration<double, std::milli>(SteadyClock::now() - collectStart).count();
        lastUploadMs_ = std::chrono::duration<double, std::milli>(SteadyClock::now() - uploadStart).count();
    }

    void pollGpuParityResults()
    {
        std::deque<PendingGpuParityReadback> pending;
        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            pending.swap(pendingGpuParityReadbacks_);
        }

        if (pending.empty())
        {
            return;
        }

        std::deque<PendingGpuParityReadback> stillPending;
        for (PendingGpuParityReadback& parity : pending)
        {
            if (!parity.copySubmitted)
            {
                bool copyAlreadyPending = false;
                for (const PendingGpuParityReadback& queuedParity : stillPending)
                {
                    if (queuedParity.copySubmitted)
                    {
                        copyAlreadyPending = true;
                        break;
                    }
                }
                if (copyAlreadyPending)
                {
                    stillPending.push_back(std::move(parity));
                    continue;
                }

                if (parity.computeFenceValue == 0)
                {
                    stillPending.push_back(std::move(parity));
                    continue;
                }

                if (parityReadbackScratch_ == nullptr || parityReadbackMapped_ == nullptr || !uploadContext_.ready())
                {
                    if (chunkManagerDebugLoggingEnabled())
                    {
                        std::ostringstream stream;
                        stream << "Far LOD GPU parity waiting for readback resources chunk level=" << parity.key.level
                               << " coord=[" << parity.key.coord.x << "," << parity.key.coord.y << "," << parity.key.coord.z << "]";
                        chunkManagerDebugLog(stream.str());
                    }
                    stillPending.push_back(std::move(parity));
                    continue;
                }

                if (!uploadContext_.begin())
                {
                    if (chunkManagerDebugLoggingEnabled())
                    {
                        std::ostringstream stream;
                        stream << "Far LOD GPU parity waiting for upload context chunk level=" << parity.key.level
                               << " coord=[" << parity.key.coord.x << "," << parity.key.coord.y << "," << parity.key.coord.z << "]";
                        chunkManagerDebugLog(stream.str());
                    }
                    stillPending.push_back(std::move(parity));
                    continue;
                }

                bool submittedCopy = false;
                {
                    std::lock_guard<std::mutex> configLock(configMutex_);
                    auto it = chunks_.find(parity.key);
                    if (it == chunks_.end() ||
                        parity.epoch != buildEpoch_.load(std::memory_order_acquire) ||
                        parity.buildVersion != it->second.buildVersion ||
                        it->second.gpu.columnBuffer == nullptr)
                    {
                        // No longer relevant; drop it.
                    }
                    else
                    {
                        FarLodChunkRecord& chunk = it->second;
                        if (chunkManagerDebugLoggingEnabled())
                        {
                            std::ostringstream stream;
                            stream << "Far LOD GPU atlas parity copy submit chunk level=" << parity.key.level
                                   << " coord=[" << parity.key.coord.x << "," << parity.key.coord.y << "," << parity.key.coord.z << "]"
                                   << " computeFence=" << parity.computeFenceValue
                                   << " columnState=" << resourceStateName(chunk.gpu.columnState);
                            chunkManagerDebugLog(stream.str());
                        }
                        uploadContext_.waitForFence(gpuContext_.fence(), parity.computeFenceValue);
                        uploadContext_.copyBuffer(parityReadbackScratch_.Get(),
                                                  0,
                                                  chunk.gpu.columnBuffer.Get(),
                                                  0,
                                                  static_cast<std::uint64_t>(kLogicalSize * kLogicalSize * sizeof(GpuTerrainColumnDescriptor)));
                        submittedCopy = true;
                    }
                }

                if (!submittedCopy)
                {
                    uploadContext_.flush(nullptr);
                    continue;
                }

                uploadContext_.flush(nullptr);
                parity.copyFenceValue = uploadContext_.lastSubmittedFenceValue();
                parity.copySubmitted = true;
                if (chunkManagerDebugLoggingEnabled())
                {
                    std::ostringstream stream;
                    stream << "Far LOD GPU atlas parity copy queued chunk level=" << parity.key.level
                           << " coord=[" << parity.key.coord.x << "," << parity.key.coord.y << "," << parity.key.coord.z << "]"
                           << " copyFence=" << parity.copyFenceValue;
                    chunkManagerDebugLog(stream.str());
                }
                stillPending.push_back(std::move(parity));
                continue;
            }

            if (parity.copyFenceValue == 0 || parity.copyFenceValue > uploadContext_.completedFenceValue())
            {
                stillPending.push_back(std::move(parity));
                continue;
            }

            const auto* gpuReadback = reinterpret_cast<const GpuTerrainColumnDescriptor*>(parityReadbackMapped_);
            std::uint32_t mismatchCount = 0;
            std::size_t firstMismatchIndex = kLogicalSize * kLogicalSize;
            auto descriptorsMatchForParity = [](const GpuTerrainColumnDescriptor& cpu,
                                                const GpuTerrainColumnDescriptor& gpu) noexcept
            {
                constexpr std::uint32_t kParityRelevantFlags = 0x0Bu;
                return (cpu.flags & kParityRelevantFlags) == (gpu.flags & kParityRelevantFlags) &&
                       cpu.terrainTopY == gpu.terrainTopY &&
                       cpu.terrainBaseY == gpu.terrainBaseY &&
                       cpu.waterTopY == gpu.waterTopY &&
                       cpu.waterBottomY == gpu.waterBottomY &&
                       cpu.terrainTopBlock == gpu.terrainTopBlock &&
                       cpu.terrainSideBlock == gpu.terrainSideBlock &&
                       cpu.waterBlock == gpu.waterBlock;
            };

            GpuTerrainColumnDescriptor cpuValue{};
            GpuTerrainColumnDescriptor gpuValue{};
            if (parity.cpuColumnParityDescriptors)
            {
                for (std::size_t columnIdx = 0; columnIdx < static_cast<std::size_t>(kLogicalSize * kLogicalSize); ++columnIdx)
                {
                    if (descriptorsMatchForParity((*parity.cpuColumnParityDescriptors)[columnIdx],
                                                  gpuReadback[columnIdx]))
                    {
                        continue;
                    }

                    if (firstMismatchIndex == static_cast<std::size_t>(kLogicalSize * kLogicalSize))
                    {
                        firstMismatchIndex = columnIdx;
                        cpuValue = (*parity.cpuColumnParityDescriptors)[columnIdx];
                        gpuValue = gpuReadback[columnIdx];
                    }
                    ++mismatchCount;
                }
            }

            std::lock_guard<std::mutex> configLock(configMutex_);
            auto it = chunks_.find(parity.key);
            if (it == chunks_.end())
            {
                continue;
            }

            FarLodChunkRecord& chunk = it->second;
            if (parity.epoch != buildEpoch_.load(std::memory_order_acquire) || parity.buildVersion != chunk.buildVersion)
            {
                continue;
            }

            chunk.gpu.voxelReady = true;
            chunk.gpu.parityValidated = parity.parityRequested;
            chunk.gpu.parityMismatchCount = mismatchCount;
            rollingGpuParityMismatchCount_ += mismatchCount;
            if (chunkManagerDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "Far LOD GPU atlas parity complete chunk level=" << parity.key.level
                       << " coord=[" << parity.key.coord.x << "," << parity.key.coord.y << "," << parity.key.coord.z << "]"
                       << " mismatches=" << mismatchCount
                       << " copyFence=" << parity.copyFenceValue;
                chunkManagerDebugLog(stream.str());
            }
            if (mismatchCount > 0 && firstMismatchIndex < static_cast<std::size_t>(kLogicalSize * kLogicalSize))
            {
                std::cerr << "Far LOD GPU atlas parity mismatch chunk level=" << parity.key.level
                          << " coord=[" << parity.key.coord.x << "," << parity.key.coord.y << "," << parity.key.coord.z
                          << "] firstColumn=" << firstMismatchIndex
                          << " cpu={flags=0x" << std::hex << cpuValue.flags << std::dec
                          << ", terrainTopY=" << cpuValue.terrainTopY
                          << ", terrainBaseY=" << cpuValue.terrainBaseY
                          << ", waterTopY=" << cpuValue.waterTopY
                          << ", waterBottomY=" << cpuValue.waterBottomY
                          << ", canopyTopY=" << cpuValue.canopyTopY
                          << ", canopyBottomY=" << cpuValue.canopyBottomY
                          << ", terrainTopBlock=" << cpuValue.terrainTopBlock
                          << ", terrainSideBlock=" << cpuValue.terrainSideBlock
                          << ", waterBlock=" << cpuValue.waterBlock
                          << ", canopyBlock=" << cpuValue.canopyBlock
                          << "}"
                          << " gpu={flags=0x" << std::hex << gpuValue.flags << std::dec
                          << ", terrainTopY=" << gpuValue.terrainTopY
                          << ", terrainBaseY=" << gpuValue.terrainBaseY
                          << ", waterTopY=" << gpuValue.waterTopY
                          << ", waterBottomY=" << gpuValue.waterBottomY
                          << ", canopyTopY=" << gpuValue.canopyTopY
                          << ", canopyBottomY=" << gpuValue.canopyBottomY
                          << ", terrainTopBlock=" << gpuValue.terrainTopBlock
                          << ", terrainSideBlock=" << gpuValue.terrainSideBlock
                          << ", waterBlock=" << gpuValue.waterBlock
                          << ", canopyBlock=" << gpuValue.canopyBlock
                          << "}" << std::endl;
            }
        }

        {
            std::lock_guard<std::mutex> lock(gpuRequestMutex_);
            for (PendingGpuParityReadback& parity : stillPending)
            {
                pendingGpuParityReadbacks_.push_back(std::move(parity));
            }
        }
    }

    [[nodiscard]] ChunkMesh buildChunkMesh(const FarLodChunkCpu& cpu,
                                           int lodLevel) const;
    void uploadWorldgenTables()
    {
        worldgenHeaderBuffer_.Reset();
        worldgenBiomeBuffer_.Reset();
        worldgenBiomeSelectionBuffer_.Reset();
        worldgenOceanSelectionBuffer_.Reset();
        worldgenTransitionBuffer_.Reset();
        worldgenSubBiomeBuffer_.Reset();
        worldgenPermutationBuffer_.Reset();
        if (device_ == nullptr)
        {
            return;
        }

        struct PendingUpload
        {
            ID3D12Resource* destination{nullptr};
            Microsoft::WRL::ComPtr<ID3D12Resource> upload;
            std::uint64_t bytes{0};
        };

        std::vector<PendingUpload> pending;
        pending.reserve(6);

        auto stageRawBuffer = [&](Microsoft::WRL::ComPtr<ID3D12Resource>& destination,
                                  const wchar_t* debugName,
                                  const void* data,
                                  std::uint64_t bytes,
                                  std::uint64_t minimumBytes) {
            const std::uint64_t uploadBytes = std::max(bytes, minimumBytes);
            destination = createDefaultBuffer(device_.Get(), uploadBytes, D3D12_RESOURCE_STATE_COMMON);
            if (destination == nullptr)
            {
                throw std::runtime_error("failed to allocate far lod worldgen buffer");
            }
            setDebugObjectName(destination.Get(), debugName);

            std::byte* uploadMapped = nullptr;
            Microsoft::WRL::ComPtr<ID3D12Resource> upload =
                createUploadBuffer(device_.Get(), uploadBytes, uploadMapped);
            if (upload == nullptr || uploadMapped == nullptr)
            {
                throw std::runtime_error("failed to allocate far lod worldgen upload buffer");
            }

            std::memset(uploadMapped, 0, static_cast<std::size_t>(uploadBytes));
            if (data != nullptr && bytes > 0u)
            {
                std::memcpy(uploadMapped, data, static_cast<std::size_t>(bytes));
            }
            pending.push_back(PendingUpload{destination.Get(), upload, uploadBytes});
        };

        stageRawBuffer(worldgenHeaderBuffer_,
                       L"FarLodWorldgenHeader",
                       &worldgenTables_.header,
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuWorldgenHeader)),
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuWorldgenHeader)));
        stageRawBuffer(worldgenBiomeBuffer_,
                       L"FarLodWorldgenBiomes",
                       worldgenTables_.biomes.empty() ? nullptr : worldgenTables_.biomes.data(),
                       static_cast<std::uint64_t>(worldgenTables_.biomes.size() * sizeof(terrain::FarLodGpuBiome)),
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuBiome)));
        stageRawBuffer(worldgenBiomeSelectionBuffer_,
                       L"FarLodWorldgenBiomeSelections",
                       worldgenTables_.biomeSelections.empty() ? nullptr : worldgenTables_.biomeSelections.data(),
                       static_cast<std::uint64_t>(worldgenTables_.biomeSelections.size() *
                                                  sizeof(terrain::FarLodGpuBiomeSelection)),
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuBiomeSelection)));
        stageRawBuffer(worldgenOceanSelectionBuffer_,
                       L"FarLodWorldgenOceanSelections",
                       worldgenTables_.oceanSelections.empty() ? nullptr : worldgenTables_.oceanSelections.data(),
                       static_cast<std::uint64_t>(worldgenTables_.oceanSelections.size() *
                                                  sizeof(terrain::FarLodGpuBiomeSelection)),
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuBiomeSelection)));
        stageRawBuffer(worldgenTransitionBuffer_,
                       L"FarLodWorldgenTransitions",
                       worldgenTables_.transitionBiomes.empty() ? nullptr : worldgenTables_.transitionBiomes.data(),
                       static_cast<std::uint64_t>(worldgenTables_.transitionBiomes.size() *
                                                  sizeof(terrain::FarLodGpuTransitionBiome)),
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuTransitionBiome)));
        stageRawBuffer(worldgenSubBiomeBuffer_,
                       L"FarLodWorldgenSubBiomes",
                       worldgenTables_.subBiomes.empty() ? nullptr : worldgenTables_.subBiomes.data(),
                       static_cast<std::uint64_t>(worldgenTables_.subBiomes.size() *
                                                  sizeof(terrain::FarLodGpuSubBiome)),
                       static_cast<std::uint64_t>(sizeof(terrain::FarLodGpuSubBiome)));
        stageRawBuffer(worldgenPermutationBuffer_,
                       L"FarLodWorldgenSurfacePermutation",
                       worldgenTables_.surfacePermutation.empty() ? nullptr : worldgenTables_.surfacePermutation.data(),
                       static_cast<std::uint64_t>(worldgenTables_.surfacePermutation.size() * sizeof(std::uint32_t)),
                       static_cast<std::uint64_t>(sizeof(std::uint32_t)));

        if (!uploadContext_.begin())
        {
            throw std::runtime_error("failed to begin upload for far lod worldgen tables");
        }
        for (const PendingUpload& upload : pending)
        {
            uploadContext_.transition(upload.destination, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            uploadContext_.copyBuffer(upload.destination, 0, upload.upload.Get(), 0, upload.bytes);
            uploadContext_.transition(upload.destination, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        }
        uploadContext_.flush(nullptr);
        uploadContext_.waitForIdle();

        for (PendingUpload& upload : pending)
        {
            if (upload.upload != nullptr)
            {
                upload.upload->Unmap(0, nullptr);
            }
        }
        if (chunkManagerDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "Far LOD worldgen tables uploaded biomeCount=" << worldgenTables_.biomes.size()
                   << " biomeSelectionCount=" << worldgenTables_.biomeSelections.size()
                   << " oceanSelectionCount=" << worldgenTables_.oceanSelections.size()
                   << " transitionCount=" << worldgenTables_.transitionBiomes.size()
                   << " subBiomeCount=" << worldgenTables_.subBiomes.size()
                   << " seaLevel=" << worldgenTables_.header.seaLevel
                   << " seed=" << worldgenTables_.header.seed;
            chunkManagerDebugLog(stream.str());
        }
    }

    static constexpr std::uint8_t kFarLodVoxelWater = 0x01u;
    static constexpr std::uint8_t kFarLodVoxelStructure = 0x02u;
    static constexpr std::uint8_t kFarLodVoxelCutout = 0x04u;
    static constexpr std::uint8_t kFarLodVoxelTerrain = 0x08u;
    static constexpr std::uint32_t kFarColumnFlagTerrain = 0x01u;
    static constexpr std::uint32_t kFarColumnFlagWater = 0x02u;
    static constexpr std::uint32_t kFarColumnFlagCanopy = 0x04u;
    static constexpr std::uint32_t kFarColumnFlagSteep = 0x08u;

    [[nodiscard]] bool stageBuiltChunkForCommit(FarLodChunkRecord& chunk,
                                                const ChunkMesh& mesh,
                                                std::uint32_t buildVersion,
                                                std::uint64_t epoch,
                                                double& outCopyMs)
    {
        outCopyMs = 0.0;
        if (chunk.pendingMesh.valid())
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_pending_replacement level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }
        if (mesh.vertices.empty() || mesh.indices.empty())
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_empty_mesh level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        if (!uploadContext_.ready())
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_upload_not_ready level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        Allocation allocation = acquireAllocation(mesh.vertices.size(), mesh.indices.size());
        if (allocation.pageIndex == kInvalidChunkBufferPage || allocation.pageIndex >= bufferPages_.size())
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_allocation_failed level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount
                       << " requestedVertices=" << mesh.vertices.size()
                       << " requestedIndices=" << mesh.indices.size();
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        BufferPage& page = bufferPages_[allocation.pageIndex];
        if (page.mappedVertexData == nullptr || page.mappedIndexData == nullptr ||
            page.vertexUploadBuffer == nullptr || page.indexUploadBuffer == nullptr ||
            page.vertexBuffer == nullptr || page.indexBuffer == nullptr)
        {
            releaseAllocationRange(allocation.pageIndex,
                                   allocation.vertexOffset,
                                   mesh.vertices.size(),
                                   allocation.indexOffset,
                                   static_cast<std::uint32_t>(mesh.indices.size()),
                                   allocation.recordIndex);
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_upload_buffers_missing level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        const SteadyClock::time_point copyStart = SteadyClock::now();
        if (!mesh.vertices.empty())
        {
            std::memcpy(page.mappedVertexData + allocation.vertexOffset * sizeof(Vertex),
                        mesh.vertices.data(),
                        mesh.vertices.size() * sizeof(Vertex));
            uploadContext_.copyBuffer(page.vertexBuffer.Get(),
                                      static_cast<std::uint64_t>(allocation.vertexOffset * sizeof(Vertex)),
                                      page.vertexUploadBuffer.Get(),
                                      static_cast<std::uint64_t>(allocation.vertexOffset * sizeof(Vertex)),
                                      static_cast<std::uint64_t>(mesh.vertices.size() * sizeof(Vertex)));
        }
        if (!mesh.indices.empty())
        {
            std::memcpy(page.mappedIndexData + allocation.indexOffset * sizeof(std::uint32_t),
                        mesh.indices.data(),
                        mesh.indices.size() * sizeof(std::uint32_t));
            uploadContext_.copyBuffer(page.indexBuffer.Get(),
                                      static_cast<std::uint64_t>(allocation.indexOffset * sizeof(std::uint32_t)),
                                      page.indexUploadBuffer.Get(),
                                      static_cast<std::uint64_t>(allocation.indexOffset * sizeof(std::uint32_t)),
                                      static_cast<std::uint64_t>(mesh.indices.size() * sizeof(std::uint32_t)));
        }
        outCopyMs = std::chrono::duration<double, std::milli>(SteadyClock::now() - copyStart).count();

        chunk.pendingMesh.pageIndex = allocation.pageIndex;
        chunk.pendingMesh.vertexOffset = allocation.vertexOffset;
        chunk.pendingMesh.indexOffset = allocation.indexOffset;
        chunk.pendingMesh.vertexCount = mesh.vertices.size();
        chunk.pendingMesh.indexCount = static_cast<std::uint32_t>(mesh.indices.size());
        chunk.pendingMesh.boundsMin = mesh.boundsMin;
        chunk.pendingMesh.boundsMax = mesh.boundsMax;
        chunk.pendingMesh.uploadFenceValue = 0;
        chunk.pendingMesh.buildVersion = buildVersion;
        chunk.pendingMesh.epoch = epoch;

        if (lodVisibilityDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "lodvis staged_replacement level=" << chunk.key.level
                   << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                   << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                   << " oldIndexCount=" << chunk.gpu.indexCount
                   << " reservedIndexCount=" << chunk.pendingMesh.indexCount;
            lodVisibilityDebugLog(stream.str());
        }
        return true;
    }

    [[nodiscard]] bool stageGpuMeshForCommit(FarLodChunkRecord& chunk,
                                             const Allocation& allocation,
                                             std::uint32_t faceCapacity,
                                             std::size_t vertexCount,
                                             std::uint32_t indexCount,
                                             std::uint32_t buildVersion,
                                             std::uint64_t epoch)
    {
        if (chunk.pendingMesh.valid())
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_pending_gpu_replacement level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        if (allocation.pageIndex == kInvalidChunkBufferPage || allocation.pageIndex >= bufferPages_.size())
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_gpu_allocation_failed level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        std::byte* mappedDrawRecordBytes = nullptr;
        Microsoft::WRL::ComPtr<ID3D12Resource> readbackBuffer =
            createReadbackBuffer(device_.Get(),
                                 static_cast<std::uint64_t>(sizeof(ChunkRenderBatch::GpuCullRecord)),
                                 mappedDrawRecordBytes);
        ChunkRenderBatch::GpuCullRecord* mappedDrawRecord =
            reinterpret_cast<ChunkRenderBatch::GpuCullRecord*>(mappedDrawRecordBytes);
        if (readbackBuffer == nullptr || mappedDrawRecord == nullptr)
        {
            if (lodVisibilityDebugLoggingEnabled())
            {
                std::ostringstream stream;
                stream << "lodvis preserve_resident_gpu_readback_failed level=" << chunk.key.level
                       << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                       << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                       << " oldIndexCount=" << chunk.gpu.indexCount;
                lodVisibilityDebugLog(stream.str());
            }
            return false;
        }

        chunk.pendingMesh.pageIndex = allocation.pageIndex;
        chunk.pendingMesh.vertexOffset = allocation.vertexOffset;
        chunk.pendingMesh.indexOffset = allocation.indexOffset;
        chunk.pendingMesh.vertexCount = vertexCount;
        chunk.pendingMesh.indexCount = indexCount;
        chunk.pendingMesh.faceCapacity = faceCapacity;
        chunk.pendingMesh.recordIndex = allocation.recordIndex;
        chunk.pendingMesh.boundsMin = chunk.cpu.boundsMin;
        chunk.pendingMesh.boundsMax = chunk.cpu.boundsMax;
        chunk.pendingMesh.uploadFenceValue = 0;
        chunk.pendingMesh.gpuFenceValue = 0;
        chunk.pendingMesh.gpuGenerated = true;
        chunk.pendingMesh.buildVersion = buildVersion;
        chunk.pendingMesh.epoch = epoch;
        chunk.pendingMesh.drawRecordReadbackBuffer = std::move(readbackBuffer);
        chunk.pendingMesh.mappedDrawRecord = mappedDrawRecord;

        if (lodVisibilityDebugLoggingEnabled())
        {
            std::ostringstream stream;
            stream << "lodvis staged_gpu_replacement level=" << chunk.key.level
                   << " coord=[" << chunk.key.coord.x << "," << chunk.key.coord.y << "," << chunk.key.coord.z << "]"
                   << " hadResident=" << (chunk.gpu.resident ? "y" : "n")
                   << " recordIndex=" << chunk.pendingMesh.recordIndex
                   << " reservedFaces=" << chunk.pendingMesh.faceCapacity;
            lodVisibilityDebugLog(stream.str());
        }
        return true;
    }

    void commitPendingMeshUploads()
    {
        const UINT64 completedUploadFenceValue = uploadContext_.completedFenceValue();
        const UINT64 completedGpuFenceValue = gpuContext_.completedFenceValue();
        const std::uint64_t epoch = buildEpoch_.load(std::memory_order_acquire);

        std::lock_guard<std::mutex> lock(configMutex_);
        for (auto& [key, chunk] : chunks_)
        {
            (void)key;
            if (!chunk.pendingMesh.valid())
            {
                continue;
            }
            if (chunk.pendingMesh.gpuGenerated)
            {
                if (chunk.pendingMesh.gpuFenceValue != 0 && completedGpuFenceValue < chunk.pendingMesh.gpuFenceValue)
                {
                    continue;
                }
            }
            else
            {
                if (chunk.pendingMesh.uploadFenceValue == 0 || completedUploadFenceValue < chunk.pendingMesh.uploadFenceValue)
                {
                    continue;
                }
            }

            if (chunk.pendingMesh.epoch != epoch || chunk.pendingMesh.buildVersion != chunk.buildVersion)
            {
                releasePendingMeshAllocation(chunk);
                continue;
            }

            if (chunk.pendingMesh.gpuGenerated && chunk.pendingMesh.mappedDrawRecord == nullptr)
            {
                releasePendingMeshAllocation(chunk);
                if (!chunk.dirty)
                {
                    markDirty(chunk);
                }
                continue;
            }

            ChunkRenderBatch::GpuCullRecord pendingRecord{};
            std::uint32_t actualFaceCount = 0u;
            bool overflowed = false;
            if (chunk.pendingMesh.gpuGenerated)
            {
                pendingRecord = *chunk.pendingMesh.mappedDrawRecord;
                actualFaceCount = pendingRecord.reserved & kGpuDrawRecordFaceCountMask;
                overflowed = (pendingRecord.reserved & kGpuDrawRecordOverflowFlag) != 0u;
                if (overflowed)
                {
                    chunk.requestedFaceCapacityHint =
                        std::max(chunk.requestedFaceCapacityHint, growReservedFaceCapacity(actualFaceCount));
                    releasePendingMeshAllocation(chunk);
                    if (!chunk.dirty)
                    {
                        markDirty(chunk);
                    }
                    continue;
                }
            }
            else
            {
                pendingRecord.indexCount = chunk.pendingMesh.indexCount;
                actualFaceCount = chunk.pendingMesh.indexCount / 6u;
                pendingRecord.boundsMin = glm::vec4(chunk.pendingMesh.boundsMin, 1.0f);
                pendingRecord.boundsMax = glm::vec4(chunk.pendingMesh.boundsMax, 1.0f);
            }

            const std::uint32_t oldPageIndex = chunk.gpu.pageIndex;
            const std::size_t oldVertexOffset = chunk.gpu.vertexOffset;
            const std::size_t oldIndexOffset = chunk.gpu.indexOffset;
            const std::size_t oldReservedVertexCount = chunk.gpu.reservedVertexCount;
            const std::size_t oldReservedIndexCount = chunk.gpu.reservedIndexCount;
            const std::uint32_t oldRecordIndex = chunk.gpu.recordIndex;
            const std::uint32_t pendingPageIndex = chunk.pendingMesh.pageIndex;
            const std::size_t pendingVertexOffset = chunk.pendingMesh.vertexOffset;
            const std::size_t pendingIndexOffset = chunk.pendingMesh.indexOffset;
            const std::size_t pendingReservedVertexCount = chunk.pendingMesh.vertexCount;
            const std::size_t pendingReservedIndexCount = chunk.pendingMesh.indexCount;
            const std::uint32_t pendingRecordIndex = chunk.pendingMesh.recordIndex;
            const std::uint32_t pendingFaceCapacity = chunk.pendingMesh.faceCapacity;

            const bool hasResidentGeometry = !chunk.pendingMesh.gpuGenerated || pendingRecord.indexCount > 0u;
            if (hasResidentGeometry)
            {
                chunk.gpu.pageIndex = pendingPageIndex;
                chunk.gpu.vertexOffset = pendingVertexOffset;
                chunk.gpu.indexOffset = pendingIndexOffset;
                chunk.gpu.vertexCount = static_cast<std::size_t>(actualFaceCount) * 4u;
                chunk.gpu.indexCount = pendingRecord.indexCount;
                chunk.gpu.reservedVertexCount = pendingReservedVertexCount;
                chunk.gpu.reservedIndexCount = pendingReservedIndexCount;
                chunk.gpu.faceCapacity = pendingFaceCapacity;
                chunk.gpu.recordIndex = pendingRecordIndex;
                chunk.gpu.resident = (chunk.gpu.indexCount > 0u);
                chunk.residentBoundsMin = glm::vec3(pendingRecord.boundsMin);
                chunk.residentBoundsMax = glm::vec3(pendingRecord.boundsMax);
                if (chunk.gpu.pageIndex != kInvalidChunkBufferPage && chunk.gpu.pageIndex < bufferPages_.size())
                {
                    BufferPage& page = bufferPages_[chunk.gpu.pageIndex];
                    page.recordActiveCount = std::max(page.recordActiveCount,
                                                      static_cast<std::size_t>(chunk.gpu.recordIndex + 1u));
                }
            }
            else
            {
                releasePendingMeshAllocation(chunk);
                chunk.gpu.pageIndex = kInvalidChunkBufferPage;
                chunk.gpu.vertexOffset = 0;
                chunk.gpu.indexOffset = 0;
                chunk.gpu.vertexCount = 0;
                chunk.gpu.indexCount = 0;
                chunk.gpu.reservedVertexCount = 0;
                chunk.gpu.reservedIndexCount = 0;
                chunk.gpu.faceCapacity = pendingFaceCapacity;
                chunk.gpu.recordIndex = kInvalidFarDrawRecordIndex;
                chunk.gpu.resident = false;
            }
            releasePendingDrawRecordReadback(chunk.pendingMesh);
            chunk.pendingMesh = {};

            if (oldPageIndex != kInvalidChunkBufferPage && oldPageIndex < bufferPages_.size())
            {
                BufferPage& oldPage = bufferPages_[oldPageIndex];
                if (oldRecordIndex != kInvalidFarDrawRecordIndex)
                {
                    clearDrawRecord(oldPage, oldRecordIndex);
                }
                releaseAllocationRange(oldPageIndex,
                                       oldVertexOffset,
                                       oldReservedVertexCount,
                                       oldIndexOffset,
                                       static_cast<std::uint32_t>(oldReservedIndexCount),
                                       oldRecordIndex);
            }
            chunk.requestedFaceCapacityHint = 0;
            chunk.cpu.meshReady = chunk.gpu.resident;
        }
    }

    bool enabled_{true};
    int farDistanceBlocks_{chunksToBlocks(kDefaultTotalRenderDistanceChunks)};
    int farDistanceTargetBlocks_{chunksToBlocks(kDefaultTotalRenderDistanceChunks)};
    int fogStartBlocks_{kDefaultFarFogStartBlocks};
    int seaLevel_{20};
    glm::ivec3 cameraChunk_{0};
    glm::vec3 cameraForward_{0.0f, 0.0f, -1.0f};
    mutable Frustum lastVisibilityFrustum_{};
    mutable glm::vec3 lastVisibilityCameraPos_{0.0f};
    mutable bool hasVisibilityFrustum_{false};
    std::uint64_t updateStamp_{0};
    int builtTilesLastUpdate_{0};
    int skippedTilesLastUpdate_{0};
    double lastAverageBuildMs_{0.0};
    double lastAverageGpuSynthesisMs_{0.0};
    double lastAverageGpuStampMs_{0.0};
    double lastAverageGpuFaceBuildMs_{0.0};
    double lastAverageCpuTerrainSynthesisMs_{0.0};
    double lastAverageCpuStructureStampMs_{0.0};
    double lastAverageCpuMeshMs_{0.0};
    double lastAverageUploadWaitMs_{0.0};
    double lastAverageUploadCopyMs_{0.0};
    double lastCollectMs_{0.0};
    double lastUploadMs_{0.0};
    std::size_t lastBuiltFaceCount_{0};
    std::size_t lastBuiltVertexCount_{0};
    std::size_t lastBuiltIndexCount_{0};
    mutable std::size_t lastRenderedFaceCount_{0};
    mutable std::size_t lastRenderedVertexCount_{0};
    std::vector<FarLodLevelConfig> levels_;
    std::vector<BufferPage> bufferPages_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    UploadContext uploadContext_{};
    FarLodGpuContext gpuContext_{};
    Microsoft::WRL::ComPtr<ID3D12Resource> parityReadbackScratch_;
    std::byte* parityReadbackMapped_{nullptr};
    terrain::FarLodWorldgenTables worldgenTables_{};
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenHeaderBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenBiomeBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenBiomeSelectionBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenOceanSelectionBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenTransitionBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenSubBiomeBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> worldgenPermutationBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> blockUvBuffer_;
    std::uint32_t blockUvCount_{0};
    Microsoft::WRL::ComPtr<ID3D12Resource> emptyVoxelBuffer_;
    std::unordered_map<int, FarLodLevelAtlasState> levelAtlases_;
    std::uint64_t atlasRevisionCounter_{1};
    mutable std::mutex structureRegionMutex_;
    StructureSampleColumnFn structureSampleColumnFn_{};
    StructureSurfaceBlockFn structureSurfaceBlockFn_{};
    StructureDensityFn structureDensityFn_{};
    std::unordered_map<StructureRegionKey, GpuStructureRegionState, StructureRegionKeyHasher> gpuStructureRegions_;
    std::unordered_map<FarLodChunkKey, FarLodChunkRecord, FarLodChunkKeyHasher> chunks_;
    mutable std::mutex configMutex_;
    ColumnSampleFn columnSampleFn_{};
    mutable std::mutex buildQueueMutex_;
    std::condition_variable buildQueueCv_;
    std::deque<BuildJob> buildQueue_;
    std::unordered_set<FarLodChunkKey, FarLodChunkKeyHasher> queuedKeys_;
    mutable std::mutex completedMutex_;
    std::deque<BuildResult> completedBuilds_;
    std::deque<GpuSynthesisRequest> gpuSynthesisRequests_;
    std::deque<PendingGpuParityReadback> pendingGpuParityReadbacks_;
    std::vector<std::thread> workerThreads_;
    std::thread dirtyBuildPlannerThread_;
    std::thread touchLevelPlannerThread_;
    std::atomic<bool> stopWorkers_{false};
    std::atomic<std::uint64_t> buildEpoch_{1};
    std::atomic<std::uint64_t> dirtyBuildPlanSequence_{0};
    std::atomic<std::uint64_t> touchLevelPlanSequence_{0};
    std::size_t workerCount_{1};
    std::atomic<int> exactMissingChunks_{0};
    std::atomic<std::size_t> exactPendingUploads_{0};
    ChunkBenchmarkMetrics* benchmarkMetrics_{nullptr};
    std::unordered_map<int, int> levelActivationOuterRadiusChunks_;
    mutable std::mutex gpuRequestMutex_;
    mutable std::mutex dirtyBuildPlanMutex_;
    std::condition_variable dirtyBuildPlanCv_;
    DirtyBuildPlanRequest pendingDirtyBuildPlanRequest_{};
    DirtyBuildPlanResult readyDirtyBuildPlanResult_{};
    bool hasPendingDirtyBuildPlanRequest_{false};
    bool hasReadyDirtyBuildPlanResult_{false};
    bool dirtyBuildPlannerBusy_{false};
    mutable std::mutex touchLevelPlanMutex_;
    std::condition_variable touchLevelPlanCv_;
    TouchLevelPlanRequest pendingTouchLevelPlanRequest_{};
    TouchLevelPlanResult readyTouchLevelPlanResult_{};
    bool hasPendingTouchLevelPlanRequest_{false};
    bool hasReadyTouchLevelPlanResult_{false};
    bool touchLevelPlannerBusy_{false};
    bool hasLastTouchLevelPlanRequest_{false};
    glm::ivec3 lastTouchLevelPlanCameraChunk_{0};
    std::vector<std::pair<int, int>> lastTouchLevelPlanLevels_;
    std::unordered_map<int, TouchLevelCacheState> touchLevelCaches_;
    bool gpuParityEnabled_{false};
    std::uint32_t rollingGpuParityMismatchCount_{0};
};

FarTerrainManager::ChunkMesh FarTerrainManager::buildChunkMesh(const FarLodChunkCpu& cpu,
                                                              int lodLevel) const
{
    (void)lodLevel;
    ChunkMesh mesh{};
    mesh.boundsMin = cpu.boundsMin;
    mesh.boundsMax = cpu.boundsMax;

    auto isInsideChunk = [](const glm::ivec3& localCoord) noexcept
    {
        return localCoord.x >= 0 && localCoord.x < kLogicalSize &&
               localCoord.y >= 0 && localCoord.y < kLogicalSize &&
               localCoord.z >= 0 && localCoord.z < kLogicalSize;
    };

    struct LocalCoordLess
    {
        bool operator()(const glm::ivec3& lhs, const glm::ivec3& rhs) const noexcept
        {
            if (lhs.x != rhs.x) return lhs.x < rhs.x;
            if (lhs.y != rhs.y) return lhs.y < rhs.y;
            return lhs.z < rhs.z;
        }
    };

    std::map<glm::ivec3, FarLodVoxel, LocalCoordLess> neighborVoxelCache;
    auto sampleVoxel = [&](const glm::ivec3& localCoord) -> FarLodVoxel
    {
        if (isInsideChunk(localCoord))
        {
            return cpu.voxels[voxelIndex(localCoord.x, localCoord.y, localCoord.z)];
        }

        const auto cached = neighborVoxelCache.find(localCoord);
        if (cached != neighborVoxelCache.end())
        {
            return cached->second;
        }

        const glm::ivec3 chunkOffset{
            floorDiv(localCoord.x, kLogicalSize),
            floorDiv(localCoord.y, kLogicalSize),
            floorDiv(localCoord.z, kLogicalSize)
        };

        const glm::ivec3 wrappedLocal{
            wrapIndex(localCoord.x, kLogicalSize),
            wrapIndex(localCoord.y, kLogicalSize),
            wrapIndex(localCoord.z, kLogicalSize)
        };

        FarLodVoxel sampledVoxel{};

        const FarLodChunkKey neighborKey{
            cpu.key.level,
            cpu.key.coord + chunkOffset
        };

        const auto neighborIt = chunks_.find(neighborKey);
        if (neighborIt != chunks_.end())
        {
            const FarLodChunkRecord& neighborChunk = neighborIt->second;
            if (neighborChunk.cpu.blockScale == cpu.blockScale)
            {
                sampledVoxel = neighborChunk.cpu.voxels[
                    voxelIndex(wrappedLocal.x, wrappedLocal.y, wrappedLocal.z)];
            }
        }

        neighborVoxelCache.emplace(localCoord, sampledVoxel);
        return sampledVoxel;
    };

    auto emitGreedyQuad = [&](int axis, bool positiveFace, int slice, int bStart, int cStart, int bSize, int cSize, const GreedyMaskCell& cell)
    {
        glm::ivec3 normal{0};
        normal[axis] = positiveFace ? 1 : -1;

        glm::vec3 base = glm::vec3(cpu.worldMin);
        base[axis] += static_cast<float>(slice * cpu.blockScale);
        base[(axis + 1) % 3] += static_cast<float>(bStart * cpu.blockScale);
        base[(axis + 2) % 3] += static_cast<float>(cStart * cpu.blockScale);

        glm::vec3 du(0.0f);
        du[(axis + 1) % 3] = static_cast<float>(bSize * cpu.blockScale);
        glm::vec3 dv(0.0f);
        dv[(axis + 2) % 3] = static_cast<float>(cSize * cpu.blockScale);

        glm::vec3 p0 = base;
        glm::vec3 p1 = base + du;
        glm::vec3 p2 = base + du + dv;
        glm::vec3 p3 = base + dv;
        if (!positiveFace)
        {
            std::swap(p1, p3);
        }

        const std::pair<glm::vec2, glm::vec2> uv{glm::vec2(0.0f), glm::vec2(1.0f)};
        const std::uint8_t flags =
            static_cast<std::uint8_t>(kVertexFlagFarLod | ((cell.flags & kFarLodVoxelWater) ? kVertexFlagWater : 0));
        appendQuad(mesh.vertices, mesh.indices, p0, p1, p2, p3, glm::vec3(normal), uv, flags);
    };

    for (int axis = 0; axis < 3; ++axis)
    {
        const int sizeA = kLogicalSize;
        const int sizeB = kLogicalSize;
        const int sizeC = kLogicalSize;
        std::vector<GreedyMaskCell> mask(static_cast<std::size_t>(sizeB * sizeC));
        auto maskIndex = [sizeC](int bIndex, int cIndex) noexcept
        {
            return static_cast<std::size_t>(bIndex * sizeC + cIndex);
        };

        for (int dirIndex = 0; dirIndex < 2; ++dirIndex)
        {
            const bool positiveFace = (dirIndex == 0);
            for (int slice = 0; slice <= sizeA; ++slice)
            {
                std::fill(mask.begin(), mask.end(), GreedyMaskCell{});

                for (int bi = 0; bi < sizeB; ++bi)
                {
                    for (int ci = 0; ci < sizeC; ++ci)
                    {
                        glm::ivec3 positiveLocal{0};
                        glm::ivec3 negativeLocal{0};
                        positiveLocal[axis] = slice;
                        negativeLocal[axis] = slice - 1;
                        positiveLocal[(axis + 1) % 3] = bi;
                        positiveLocal[(axis + 2) % 3] = ci;
                        negativeLocal[(axis + 1) % 3] = bi;
                        negativeLocal[(axis + 2) % 3] = ci;

                        const glm::ivec3 owningLocal = positiveFace ? negativeLocal : positiveLocal;
                        if (!isInsideChunk(owningLocal))
                        {
                            continue;
                        }

                        const FarLodVoxel owningVoxel = sampleVoxel(owningLocal);
                        if (!owningVoxel.occupied || owningVoxel.material == BlockId::Air)
                        {
                            continue;
                        }

                        const FarLodVoxel neighborVoxel = sampleVoxel(positiveFace ? positiveLocal : negativeLocal);
                        if (!shouldRenderBlockFace(owningVoxel.material, neighborVoxel.material))
                        {
                            continue;
                        }

                        GreedyMaskCell& cell = mask[maskIndex(bi, ci)];
                        cell.visible = true;
                        cell.material = owningVoxel.material;
                        cell.flags = static_cast<std::uint8_t>(owningVoxel.flags & (kFarLodVoxelWater |
                                                                                   kFarLodVoxelStructure |
                                                                                   kFarLodVoxelCutout));
                    }
                }

                for (int bi = 0; bi < sizeB; ++bi)
                {
                    int ci = 0;
                    while (ci < sizeC)
                    {
                        GreedyMaskCell& startCell = mask[maskIndex(bi, ci)];
                        if (!startCell.visible)
                        {
                            ++ci;
                            continue;
                        }

                        int runC = 1;
                        while (ci + runC < sizeC)
                        {
                            const GreedyMaskCell& nextCell = mask[maskIndex(bi, ci + runC)];
                            if (!nextCell.visible || !nextCell.mergeEquals(startCell))
                            {
                                break;
                            }
                            ++runC;
                        }

                        int runB = 1;
                        while (bi + runB < sizeB)
                        {
                            bool rowMatches = true;
                            for (int offset = 0; offset < runC; ++offset)
                            {
                                const GreedyMaskCell& rowCell = mask[maskIndex(bi + runB, ci + offset)];
                                if (!rowCell.visible || !rowCell.mergeEquals(startCell))
                                {
                                    rowMatches = false;
                                    break;
                                }
                            }
                            if (!rowMatches)
                            {
                                break;
                            }
                            ++runB;
                        }

                        emitGreedyQuad(axis, positiveFace, slice, bi, ci, runB, runC, startCell);

                        for (int bOffset = 0; bOffset < runB; ++bOffset)
                        {
                            for (int cOffset = 0; cOffset < runC; ++cOffset)
                            {
                                mask[maskIndex(bi + bOffset, ci + cOffset)].visible = false;
                            }
                        }

                        ci += runC;
                    }
                }
            }
        }
    }

    return mesh;
}




