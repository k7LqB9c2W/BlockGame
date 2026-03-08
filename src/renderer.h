#pragma once

#include "chunk_manager.h"

#include <glm/glm.hpp>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <vector>

struct GLFWwindow;
struct ImGui_ImplDX12_InitInfo;

struct LoadedTexture
{
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    glm::ivec2 size{0};
    D3D12_CPU_DESCRIPTOR_HANDLE srvCpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE srvGpu{};

    [[nodiscard]] bool valid() const noexcept
    {
        return resource != nullptr;
    }
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

    [[nodiscard]] ID3D12Device* device() const noexcept;
    [[nodiscard]] LoadedTexture loadTexture(const char* path);

    void beginFrame(const glm::vec4& clearColor);
    void beginImGuiFrame();
    void renderWorld(const WorldRenderData& renderData,
                     const glm::mat4& viewProj,
                     const glm::vec3& cameraPos,
                     const LoadedTexture& atlasTexture);
    void endFrame();

    [[nodiscard]] int width() const noexcept;
    [[nodiscard]] int height() const noexcept;

private:
    struct SceneConstants
    {
        glm::mat4 viewProj{1.0f};
        glm::vec4 lightDirection{0.0f, 1.0f, 0.0f, 0.0f};
        glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 0.0f};
        glm::vec4 highlightedBlock{0.0f, 0.0f, 0.0f, 0.0f};
        glm::vec4 fogColor{0.55f, 0.78f, 0.95f, 1.0f};
        glm::vec4 params{0.0f, 0.0f, 0.0f, 0.0f};
    };

    static constexpr UINT kBackBufferCount = 2;
    static constexpr UINT kSrvHeapCapacity = 64;

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
    void createConstantBuffer();
    void createPipelines();
    void createImGui(GLFWwindow* window);
    void destroyRenderTargets();
    void destroyDepthBuffer();
    void updateViewport(int width, int height);
    void ensureFrameStarted() const;

    [[nodiscard]] UINT allocateSrvDescriptor();
    void freeSrvDescriptor(UINT index);
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle(UINT index) const noexcept;
    [[nodiscard]] D3D12_GPU_DESCRIPTOR_HANDLE srvGpuHandle(UINT index) const noexcept;

    GLFWwindow* window_{nullptr};
    int width_{0};
    int height_{0};
    bool frameStarted_{false};
    bool imguiFrameStarted_{false};
    bool initialized_{false};

    Microsoft::WRL::ComPtr<IDXGIFactory6> factory_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> swapChain_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvHeap_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> dsvHeap_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srvHeap_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSignature_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> nearPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> farPipelineState_;
    Microsoft::WRL::ComPtr<ID3D12Resource> renderTargets_[kBackBufferCount];
    Microsoft::WRL::ComPtr<ID3D12Resource> depthBuffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> sceneConstantBuffer_;

    SceneConstants* mappedSceneConstants_{nullptr};
    HANDLE fenceEvent_{nullptr};
    UINT64 fenceValue_{0};
    UINT currentBackBufferIndex_{0};
    UINT rtvDescriptorSize_{0};
    UINT dsvDescriptorSize_{0};
    UINT srvDescriptorSize_{0};
    std::vector<bool> srvSlotsInUse_{};
    D3D12_VIEWPORT viewport_{};
    D3D12_RECT scissorRect_{};
};
