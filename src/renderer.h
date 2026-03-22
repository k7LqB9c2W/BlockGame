#pragma once

#include "chunk_manager.h"

#include <glm/glm.hpp>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

struct GLFWwindow;
struct ImGui_ImplDX12_InitInfo;

struct LoadedTexture
{
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    glm::ivec2 size{0};
    glm::ivec2 sourceSize{0};
    glm::ivec2 tileCounts{0};
    UINT mipLevels{1};
    UINT srvIndex{(std::numeric_limits<UINT>::max)()};
    int tileSizePixels{0};
    int tileStridePixels{0};
    int tilePaddingPixels{0};
    D3D12_CPU_DESCRIPTOR_HANDLE srvCpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE srvGpu{};

    [[nodiscard]] bool valid() const noexcept
    {
        return resource != nullptr;
    }
};

enum class TerrainDebugView : int
{
    None = 0,
    SkyLight = 1,
    BlockLight = 2,
    MipLevel = 3,
    AmbientOcclusion = 4
};

// The base-game terrain look is the canonical lighting target for BlockGame.
// Keep the physically-inspired atmosphere path available as an optional enhancement,
// but do not make it the default when future render work lands.
inline constexpr bool kDefaultEnhancedAtmosphereEnabled = false;

struct AtmosphereSettings
{
    float groundRadiusKm{6360.0f};
    float atmosphereRadiusKm{6460.0f};
    float rayleighScaleHeightKm{8.0f};
    float mieScaleHeightKm{1.2f};
    float ozoneCenterHeightKm{25.0f};
    float ozoneHalfWidthKm{15.0f};
    float mieAnisotropy{0.76f};
    float aerialPerspectiveDistanceKm{12.0f};
    glm::vec3 rayleighScattering{5.802e-6f, 13.558e-6f, 33.1e-6f};
    glm::vec3 rayleighAbsorption{0.0f};
    glm::vec3 mieScattering{3.996e-6f};
    glm::vec3 mieAbsorption{4.40e-6f};
    glm::vec3 ozoneAbsorption{0.650e-6f, 1.881e-6f, 0.085e-6f};
};

struct TonemapSettings
{
    float exposure{0.62f};
    float whitePoint{9.0f};
};

struct RenderDebugSettings
{
    bool worldPassEnabled{true};
    bool skyPassEnabled{true};
    bool aerialPerspectiveEnabled{true};
    bool fogFallbackEnabled{true};
    bool shadowsEnabled{true};
    bool directSunEnabled{true};
    float aoIntensity{1.0f};
    TerrainDebugView terrainDebugView{TerrainDebugView::None};
};

struct EnvironmentState
{
    glm::vec3 sunDirection{-0.35f, 0.9f, -0.2f};
    glm::vec3 sunIlluminance{4.5f, 4.8f, 5.5f};
    float timeOfDay{12.0f};
    // Non-default visual mode. Base-game readability tuning should assume this starts off.
    bool atmosphereEnabled{kDefaultEnhancedAtmosphereEnabled};
    float fogStartBlocks{1400.0f};
    float farDistanceBlocks{4800.0f};
    glm::vec3 baseSkyTopColorSrgb{120.0f / 255.0f, 167.0f / 255.0f, 255.0f / 255.0f};
    glm::vec3 baseSkyHorizonColorSrgb{187.0f / 255.0f, 212.0f / 255.0f, 255.0f / 255.0f};
    AtmosphereSettings atmosphere{};
    TonemapSettings tonemap{};
    RenderDebugSettings debug{};
};

struct RendererProfilingSnapshot
{
    double atmosphereLutMs{0.0};
    double skyDrawMs{0.0};
    double shadowDrawMs{0.0};
    double worldDrawMs{0.0};
    double lodGpuCullMs{0.0};
    double lodIndirectBuildMs{0.0};
    double toneMapMs{0.0};
    double presentMs{0.0};
    double endFrameMs{0.0};
};

class Renderer
{
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    void initialize(GLFWwindow* window, int width, int height);
    void shutdown();
    void waitForGpu();
    void resize(int width, int height);
    void setUploadSynchronization(ID3D12Fence* primaryUploadFence,
                                  UINT64 primaryUploadFenceValue,
                                  ID3D12Fence* secondaryUploadFence = nullptr,
                                  UINT64 secondaryUploadFenceValue = 0) noexcept;

    [[nodiscard]] ID3D12Device* device() const noexcept;
    [[nodiscard]] ID3D12Fence* frameFence() const noexcept;
    [[nodiscard]] UINT64 lastSubmittedFrameFenceValue() const noexcept;
    [[nodiscard]] LoadedTexture loadTexture(const char* path);

    void beginFrame(const glm::vec4& clearColor);
    void beginImGuiFrame();
    void renderWorld(const WorldRenderData& renderData,
                     const glm::mat4& view,
                     const glm::mat4& proj,
                     const glm::vec3& cameraPos,
                     const LoadedTexture& atlasTexture,
                     const EnvironmentState& environment);
    void requestScreenshot(const std::filesystem::path& path);
    void endFrame();

    [[nodiscard]] int width() const noexcept;
    [[nodiscard]] int height() const noexcept;
    [[nodiscard]] RendererProfilingSnapshot profilingSnapshot() const noexcept;

private:
    struct WorldConstants
    {
        glm::mat4 viewProj{1.0f};
        glm::mat4 shadowViewProj{1.0f};
        glm::vec4 lightDirection{0.0f, 1.0f, 0.0f, 0.0f};
        glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 0.0f};
        glm::vec4 highlightedBlock{0.0f, 0.0f, 0.0f, 0.0f};
        glm::vec4 params0{0.0f, 0.0f, 0.0f, 0.0f};
        glm::vec4 params1{0.0f, 0.0f, 0.0f, 0.0f};
        glm::vec4 sunColor{1.0f, 1.0f, 1.0f, 0.0f};
        glm::vec4 skyAmbient{0.1f, 0.12f, 0.16f, 0.0f};
        glm::vec4 groundAmbient{0.05f, 0.045f, 0.04f, 0.0f};
        glm::vec4 skyTopColor{0.0f, 0.0f, 0.0f, 0.0f};
        glm::vec4 skyHorizonColor{0.0f, 0.0f, 0.0f, 0.0f};
        glm::vec4 shadowParams{0.0f, 0.0f, 0.0f, 0.0f};
        glm::vec4 terrainDebug{0.0f, 0.0f, 0.0f, 0.0f};
    };

    struct BaseSkyConstants
    {
        glm::vec4 topSkyColor{0.0f};
        glm::vec4 horizonSkyColor{0.0f};
        glm::mat4 invViewProj{1.0f};
        glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 0.0f};
    };

    struct CloudConstants
    {
        glm::mat4 viewProj{1.0f};
        // Decorative cloud layer only. This is intentionally render-only and never part of gameplay collision.
        glm::vec4 cameraPosTime{0.0f};
        glm::vec4 layerParams{0.0f};
        glm::vec4 shapeParams{0.0f};
        glm::vec4 topColor{0.0f};
        glm::vec4 bottomColor{0.0f};
    };

    struct ToneMapConstants
    {
        glm::vec4 exposureWhitePoint{1.0f, 8.0f, 0.0f, 0.0f};
    };

    struct AtmosphereConstants
    {
        glm::mat4 invViewProj{1.0f};
        glm::mat4 view{1.0f};
        glm::mat4 proj{1.0f};
        glm::vec4 cameraPosKm{0.0f};
        glm::vec4 sunDirection{0.0f, -1.0f, 0.0f, 0.0f};
        glm::vec4 sunIlluminance{18.0f, 17.0f, 15.0f, 0.0f};
        glm::vec4 atmosphereHeights{6360.0f, 6460.0f, 8.0f, 1.2f};
        glm::vec4 ozoneAndPhase{25.0f, 15.0f, 0.8f, 32.0f};
        glm::vec4 rayleighScattering{5.802e-6f, 13.558e-6f, 33.1e-6f, 0.0f};
        glm::vec4 mieScattering{3.996e-6f, 3.996e-6f, 3.996e-6f, 0.0f};
        glm::vec4 mieAbsorption{4.40e-6f, 4.40e-6f, 4.40e-6f, 0.0f};
        glm::vec4 ozoneAbsorption{0.650e-6f, 1.881e-6f, 0.085e-6f, 0.0f};
        glm::vec4 viewportAndDepth{1.0f, 1.0f, 1.0f, 32.0f};
        glm::vec4 sliceParams{0.0f, 0.0f, 0.0f, 0.0f};
    };

    struct FrameResource
    {
        static constexpr std::uint32_t kFarCullVisibleCountReadbackMaxEntries = 512u;

        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        Microsoft::WRL::ComPtr<ID3D12Resource> constantBuffer;
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> transientResources;
        Microsoft::WRL::ComPtr<ID3D12Resource> farCullRecordsDefault;
        Microsoft::WRL::ComPtr<ID3D12Resource> farCullRecordsUpload;
        Microsoft::WRL::ComPtr<ID3D12Resource> farCullVisibleIndices;
        Microsoft::WRL::ComPtr<ID3D12Resource> farCullVisibleCount;
        Microsoft::WRL::ComPtr<ID3D12Resource> farCullCountUpload;
        Microsoft::WRL::ComPtr<ID3D12Resource> farCullIndirectArgs;
        Microsoft::WRL::ComPtr<ID3D12Resource> farCullVisibleCountReadback;
        std::byte* mappedConstants{nullptr};
        std::byte* farCullRecordsUploadMapped{nullptr};
        std::byte* farCullCountUploadMapped{nullptr};
        std::byte* farCullVisibleCountReadbackMapped{nullptr};
        std::uint64_t farCullRecordCapacityBytes{0};
        std::uint64_t farCullVisibleIndexCapacityBytes{0};
        std::uint64_t farCullIndirectCapacityBytes{0};
        std::uint32_t farCullVisibleCountReadbackEntryCount{0};
        std::array<std::uint32_t, kFarCullVisibleCountReadbackMaxEntries> farCullVisibleCountReadbackPageIndices{};
        std::array<std::uint32_t, kFarCullVisibleCountReadbackMaxEntries> farCullVisibleCountReadbackRecordCounts{};
        D3D12_RESOURCE_STATES farCullRecordsState{D3D12_RESOURCE_STATE_COPY_DEST};
        D3D12_RESOURCE_STATES farCullVisibleIndicesState{D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
        D3D12_RESOURCE_STATES farCullVisibleCountState{D3D12_RESOURCE_STATE_COPY_DEST};
        D3D12_RESOURCE_STATES farCullIndirectArgsState{D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
        UINT64 fenceValue{0};
    };

    struct UploadSyncPoint
    {
        ID3D12Fence* fence{nullptr};
        UINT64 value{0};
        UINT64 consumedValue{0};
    };

    struct ShadowConstants
    {
        glm::mat4 lightViewProj{1.0f};
    };

    static constexpr UINT kBackBufferCount = 2;
    static constexpr UINT kSrvHeapCapacity = 128;

    static void imguiSrvAlloc(ImGui_ImplDX12_InitInfo* info,
                              D3D12_CPU_DESCRIPTOR_HANDLE* outCpuHandle,
                              D3D12_GPU_DESCRIPTOR_HANDLE* outGpuHandle);
    static void imguiSrvFree(ImGui_ImplDX12_InitInfo* info,
                             D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle,
                             D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle);

    void createFactory();
    void createDevice();
    void createCommandObjects();
    void createSwapChain(GLFWwindow* window);
    void createDescriptorHeaps();
    void createRenderTargets();
    void createDepthBuffer();
    void createDepthPyramid();
    void createShadowResources();
    void createPipelines();
    void createImGui(GLFWwindow* window);
    void destroyRenderTargets();
    void destroyDepthBuffer();
    void destroyDepthPyramid();
    void destroyShadowResources();
    void updateViewport(int width, int height);
    void ensureFrameStarted() const;
    void createFrameResources();
    void destroyFrameResources();
    void createSkyBackground();
    void destroySkyBackground();
    void createSceneColor();
    void destroySceneColor();
    void ensureScreenshotReadbackBuffer();
    void ensureFarCullBuffers(FrameResource& frame,
                              std::uint64_t recordBytes,
                              std::uint64_t visibleIndexBytes,
                              std::uint64_t indirectBytes);
    void writePendingScreenshot(const std::filesystem::path& path);
    void buildDepthPyramid();
    void renderFarBatchGpuCull(const ChunkRenderBatch& batch,
                               const glm::mat4& viewProj,
                               D3D12_GPU_VIRTUAL_ADDRESS farConstantsGpuAddress,
                               D3D12_GPU_DESCRIPTOR_HANDLE atlasSrv,
                               D3D12_GPU_DESCRIPTOR_HANDLE aerialPerspectiveSrv,
                               D3D12_GPU_DESCRIPTOR_HANDLE shadowSrv,
                               D3D12_GPU_DESCRIPTOR_HANDLE skyBackgroundSrv);
    void renderShadowMap(const WorldRenderData& renderData,
                         const LoadedTexture& atlasTexture,
                         const glm::mat4& view,
                         const glm::vec3& cameraPos,
                         const EnvironmentState& environment,
                         WorldConstants& nearConstants);
    [[nodiscard]] std::string collectDebugMessages() const;

    [[nodiscard]] std::string shaderPath(const char* relativePath) const;
    [[nodiscard]] std::uint64_t allocateFrameConstantBytes(std::size_t size, void** cpuPtrOut);

    [[nodiscard]] UINT allocateSrvDescriptor();
    void freeSrvDescriptor(UINT index);
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle(UINT index) const noexcept;
    [[nodiscard]] D3D12_GPU_DESCRIPTOR_HANDLE srvGpuHandle(UINT index) const noexcept;

    GLFWwindow* window_{nullptr};
    int width_{0};
    int height_{0};
    int sceneColorSrvIndex_{-1};
    bool frameStarted_{false};
    bool imguiFrameStarted_{false};
    bool debugLayerEnabled_{false};
    bool initialized_{false};
    bool sceneColorClearLogged_{false};
    std::uint64_t directCommandListSequence_{0};

    Microsoft::WRL::ComPtr<IDXGIFactory6> factory_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> uploadCommandAllocator_;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    Microsoft::WRL::ComPtr<ID3D12InfoQueue> infoQueue_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvHeap_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dsvHeap_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvHeap_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> shadowRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> worldRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> fullscreenRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> depthPyramidRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> lodCullRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> lodIndirectRootSignature_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> shadowPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> nearPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> farPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mobPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> blockOutlinePipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> depthPyramidPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> lodCullPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> lodIndirectPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> drawIndexedCommandSignature_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> baseSkyPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> backgroundCloudPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> cloudPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> toneMapPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12Resource> renderTargets_[kBackBufferCount];
    std::array<D3D12_RESOURCE_STATES, kBackBufferCount> backBufferStates_{
        D3D12_RESOURCE_STATE_PRESENT,
        D3D12_RESOURCE_STATE_PRESENT};
    Microsoft::WRL::ComPtr<ID3D12Resource> depthBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> depthPyramid_;
    Microsoft::WRL::ComPtr<ID3D12Resource> shadowMap_;
    Microsoft::WRL::ComPtr<ID3D12Resource> skyBackground_;
    Microsoft::WRL::ComPtr<ID3D12Resource> sceneColor_;
    Microsoft::WRL::ComPtr<ID3D12Resource> screenshotReadbackBuffer_;
    D3D12_RESOURCE_STATES shadowMapState_{D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
    D3D12_RESOURCE_STATES depthPyramidState_{D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    D3D12_RESOURCE_STATES skyBackgroundState_{D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
    D3D12_RESOURCE_STATES sceneColorState_{D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT screenshotReadbackLayout_{};

    D3D12_CPU_DESCRIPTOR_HANDLE depthDsv_{};
    D3D12_CPU_DESCRIPTOR_HANDLE depthSrvCpu_{};
    D3D12_GPU_DESCRIPTOR_HANDLE depthSrvGpu_{};
    D3D12_CPU_DESCRIPTOR_HANDLE depthPyramidSrvCpu_{};
    D3D12_GPU_DESCRIPTOR_HANDLE depthPyramidSrvGpu_{};
    D3D12_CPU_DESCRIPTOR_HANDLE shadowMapDsv_{};
    D3D12_CPU_DESCRIPTOR_HANDLE shadowMapSrvCpu_{};
    D3D12_GPU_DESCRIPTOR_HANDLE shadowMapSrvGpu_{};
    D3D12_CPU_DESCRIPTOR_HANDLE skyBackgroundRtv_{};
    D3D12_CPU_DESCRIPTOR_HANDLE skyBackgroundSrvCpu_{};
    D3D12_GPU_DESCRIPTOR_HANDLE skyBackgroundSrvGpu_{};
    D3D12_CPU_DESCRIPTOR_HANDLE sceneColorRtv_{};
    D3D12_CPU_DESCRIPTOR_HANDLE sceneColorSrvCpu_{};
    D3D12_GPU_DESCRIPTOR_HANDLE sceneColorSrvGpu_{};
    std::array<FrameResource, kBackBufferCount> frameResources_{};
    std::size_t currentFrameConstantOffset_{0};
    HANDLE fenceEvent_{nullptr};
    UINT64 fenceValue_{0};
    std::array<UploadSyncPoint, 2> uploadSyncPoints_{};
    UINT currentBackBufferIndex_{0};
    UINT rtvDescriptorSize_{0};
    UINT dsvDescriptorSize_{0};
    UINT srvDescriptorSize_{0};
    int depthSrvIndex_{-1};
    int depthPyramidSrvIndex_{-1};
    int shadowMapSrvIndex_{-1};
    int skyBackgroundSrvIndex_{-1};
    std::vector<UINT> depthPyramidUavIndices_{};
    std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> depthPyramidUavCpuHandles_{};
    std::vector<D3D12_GPU_DESCRIPTOR_HANDLE> depthPyramidUavGpuHandles_{};
    UINT depthPyramidMipCount_{0};
    UINT64 screenshotReadbackBufferSize_{0};
    std::vector<bool> srvSlotsInUse_{};
    D3D12_VIEWPORT viewport_{};
    D3D12_RECT scissorRect_{};
    RendererProfilingSnapshot profilingSnapshot_{};
    std::filesystem::path pendingScreenshotPath_{};
    bool screenshotRequested_{false};

    struct AtmosphereRenderer;
    std::unique_ptr<AtmosphereRenderer> atmosphere_;
};
