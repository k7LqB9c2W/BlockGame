#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <d3dcompiler.h>
#include <imgui.h>
#include <imgui_impl_dx12.h>
#include <imgui_impl_glfw.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <glm/gtc/type_ptr.hpp>

#include "renderer.h"

namespace
{
constexpr DXGI_FORMAT kBackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr DXGI_FORMAT kDepthBufferFormat = DXGI_FORMAT_D32_FLOAT;

[[noreturn]] void throwRenderError(const std::string& message)
{
    throw std::runtime_error("Renderer: " + message);
}

void throwIfFailed(HRESULT hr, const std::string& message)
{
    if (FAILED(hr))
    {
        throwRenderError(message);
    }
}

[[nodiscard]] D3D12_RESOURCE_DESC bufferDesc(std::uint64_t sizeInBytes) noexcept
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = sizeInBytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    return desc;
}

[[nodiscard]] D3D12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE type) noexcept
{
    D3D12_HEAP_PROPERTIES props{};
    props.Type = type;
    props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask = 1;
    props.VisibleNodeMask = 1;
    return props;
}

[[nodiscard]] D3D12_RESOURCE_BARRIER transitionBarrier(ID3D12Resource* resource,
                                                       D3D12_RESOURCE_STATES before,
                                                       D3D12_RESOURCE_STATES after) noexcept
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return barrier;
}

Microsoft::WRL::ComPtr<ID3DBlob> compileShader(const char* source,
                                               const char* entryPoint,
                                               const char* target)
{
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifndef NDEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
    Microsoft::WRL::ComPtr<ID3DBlob> errors;
    const HRESULT hr = D3DCompile(source,
                                  std::strlen(source),
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  entryPoint,
                                  target,
                                  flags,
                                  0,
                                  &bytecode,
                                  &errors);
    if (FAILED(hr))
    {
        std::string message = "shader compilation failed";
        if (errors)
        {
            message += ": ";
            message.append(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
        }
        throwRenderError(message);
    }

    return bytecode;
}

Microsoft::WRL::ComPtr<IDXGIAdapter1> chooseHardwareAdapter(IDXGIFactory6* factory)
{
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    for (UINT index = 0;
         factory->EnumAdapterByGpuPreference(index,
                                             DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                             IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
         ++index)
    {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0)
        {
            continue;
        }

        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr)))
        {
            return adapter;
        }
    }

    adapter.Reset();
    return adapter;
}

const char* kWorldVertexShader = R"(
cbuffer SceneConstants : register(b0)
{
    float4x4 uViewProj;
    float4 uLightDirection;
    float4 uCameraPos;
    float4 uHighlightedBlock;
    float4 uFogColor;
    float4 uParams;
};

struct VSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
};

struct VSOutput
{
    float4 position : SV_POSITION;
    float3 worldPos : POSITION0;
    float3 normal : NORMAL0;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
};

VSOutput main(VSInput input)
{
    VSOutput output;
    output.worldPos = input.position;
    output.normal = input.normal;
    output.tileCoord = input.tileCoord;
    output.atlasBase = input.atlasBase;
    output.atlasSize = input.atlasSize;
    output.position = mul(uViewProj, float4(input.position, 1.0f));
    return output;
}
)";

const char* kNearPixelShader = R"(
Texture2D gAtlas : register(t0);
SamplerState gSampler : register(s0);

cbuffer SceneConstants : register(b0)
{
    float4x4 uViewProj;
    float4 uLightDirection;
    float4 uCameraPos;
    float4 uHighlightedBlock;
    float4 uFogColor;
    float4 uParams;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float3 worldPos : POSITION0;
    float3 normal : NORMAL0;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
};

float4 main(PSInput input) : SV_TARGET
{
    float3 normal = normalize(input.normal);
    float3 lightDir = normalize(-uLightDirection.xyz);
    float3 viewDir = normalize(uCameraPos.xyz - input.worldPos);
    float diff = max(dot(normal, lightDir), 0.0f);
    float ambient = 0.35f;
    float3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0f), 32.0f);

    float2 tileUv = frac(input.tileCoord);
    float2 atlasUv = input.atlasBase + input.atlasSize * tileUv;
    float4 textureSample = gAtlas.Sample(gSampler, atlasUv);
    clip(textureSample.a - 0.5f);
    float3 textureColor = textureSample.rgb;
    float3 color = textureColor * (ambient + diff) + float3(0.1f, 0.1f, 0.1f) * spec;

    if (uParams.z > 0.5f)
    {
        int3 currentBlock = int3(floor(input.worldPos));
        int3 targetBlock = int3(uHighlightedBlock.xyz);
        if (all(currentBlock == targetBlock))
        {
            color = min(color + float3(0.3f, 0.3f, 0.3f), float3(1.0f, 1.0f, 1.0f));
        }
    }

    return float4(color, 1.0f);
}
)";

const char* kFarPixelShader = R"(
Texture2D gAtlas : register(t0);
SamplerState gSampler : register(s0);

cbuffer SceneConstants : register(b0)
{
    float4x4 uViewProj;
    float4 uLightDirection;
    float4 uCameraPos;
    float4 uHighlightedBlock;
    float4 uFogColor;
    float4 uParams;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float3 worldPos : POSITION0;
    float3 normal : NORMAL0;
    float2 tileCoord : TEXCOORD0;
    float2 atlasBase : TEXCOORD1;
    float2 atlasSize : TEXCOORD2;
};

float4 main(PSInput input) : SV_TARGET
{
    float3 normal = normalize(input.normal);
    float3 lightDir = normalize(-uLightDirection.xyz);
    float diff = max(dot(normal, lightDir), 0.0f);
    float ambient = 0.45f;

    float2 tileUv = frac(input.tileCoord);
    float2 atlasUv = input.atlasBase + input.atlasSize * tileUv;
    float4 textureSample = gAtlas.Sample(gSampler, atlasUv);
    clip(textureSample.a - 0.5f);
    float3 textureColor = textureSample.rgb;
    float3 litColor = textureColor * (ambient + diff * 0.55f);

    float horizontalDistance = distance(input.worldPos, uCameraPos.xyz);
    float fogFactor = 0.0f;
    if (uParams.y > uParams.x)
    {
        fogFactor = saturate((horizontalDistance - uParams.x) / (uParams.y - uParams.x));
    }

    float3 color = lerp(litColor, uFogColor.rgb, fogFactor);
    return float4(color, 1.0f);
}
)";
} // namespace

Renderer::Renderer()
    : srvSlotsInUse_(kSrvHeapCapacity, false)
{
}

Renderer::~Renderer()
{
    shutdown();
}

void Renderer::initialize(GLFWwindow* window, int width, int height)
{
    if (initialized_)
    {
        return;
    }

    window_ = window;
    width_ = std::max(width, 1);
    height_ = std::max(height, 1);

    createFactory();
    createDevice();
    createCommandObjects();
    createDescriptorHeaps();
    createSwapChain(window);
    createRenderTargets();
    createDepthBuffer();
    createConstantBuffer();
    createPipelines();
    createImGui(window);
    updateViewport(width_, height_);
    initialized_ = true;
}

void Renderer::shutdown()
{
    if (!initialized_)
    {
        return;
    }

    waitForGpu();

    ImGui_ImplDX12_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    if (ImGui::GetCurrentContext())
    {
        ImGui::DestroyContext();
    }

    mappedSceneConstants_ = nullptr;
    sceneConstantBuffer_.Reset();
    nearPipelineState_.Reset();
    farPipelineState_.Reset();
    rootSignature_.Reset();
    destroyDepthBuffer();
    destroyRenderTargets();
    srvHeap_.Reset();
    dsvHeap_.Reset();
    rtvHeap_.Reset();
    swapChain_.Reset();
    commandList_.Reset();
    commandAllocator_.Reset();
    commandQueue_.Reset();
    fence_.Reset();
    device_.Reset();
    factory_.Reset();

    if (fenceEvent_ != nullptr)
    {
        CloseHandle(fenceEvent_);
        fenceEvent_ = nullptr;
    }

    frameStarted_ = false;
    imguiFrameStarted_ = false;
    initialized_ = false;
}

ID3D12Device* Renderer::device() const noexcept
{
    return device_.Get();
}

int Renderer::width() const noexcept
{
    return width_;
}

int Renderer::height() const noexcept
{
    return height_;
}

void Renderer::createFactory()
{
    UINT flags = 0;
#ifndef NDEBUG
    Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
    {
        debugController->EnableDebugLayer();
        flags |= DXGI_CREATE_FACTORY_DEBUG;
    }
#endif
    throwIfFailed(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory_)), "failed to create DXGI factory");
}

void Renderer::createDevice()
{
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter = chooseHardwareAdapter(factory_.Get());
    if (!adapter)
    {
        throwIfFailed(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)),
                      "failed to create D3D12 device");
        return;
    }

    throwIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)),
                  "failed to create D3D12 device");
}

void Renderer::createCommandObjects()
{
    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    throwIfFailed(device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue_)),
                  "failed to create command queue");

    throwIfFailed(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator_)),
                  "failed to create command allocator");
    throwIfFailed(device_->CreateCommandList(0,
                                             D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             commandAllocator_.Get(),
                                             nullptr,
                                             IID_PPV_ARGS(&commandList_)),
                  "failed to create command list");
    throwIfFailed(commandList_->Close(), "failed to close initial command list");

    throwIfFailed(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
                  "failed to create fence");
    fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (fenceEvent_ == nullptr)
    {
        throwRenderError("failed to create fence event");
    }
}

void Renderer::createSwapChain(GLFWwindow* window)
{
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc{};
    swapChainDesc.BufferCount = kBackBufferCount;
    swapChainDesc.Width = static_cast<UINT>(width_);
    swapChainDesc.Height = static_cast<UINT>(height_);
    swapChainDesc.Format = kBackBufferFormat;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;

    Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain;
    HWND hwnd = glfwGetWin32Window(window);
    throwIfFailed(factory_->CreateSwapChainForHwnd(commandQueue_.Get(),
                                                   hwnd,
                                                   &swapChainDesc,
                                                   nullptr,
                                                   nullptr,
                                                   &swapChain),
                  "failed to create swap chain");
    throwIfFailed(factory_->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER),
                  "failed to associate window with DXGI");
    throwIfFailed(swapChain.As(&swapChain_), "failed to query IDXGISwapChain3");
    currentBackBufferIndex_ = swapChain_->GetCurrentBackBufferIndex();
}

void Renderer::createDescriptorHeaps()
{
    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc{};
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDesc.NumDescriptors = kBackBufferCount;
    throwIfFailed(device_->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&rtvHeap_)),
                  "failed to create RTV heap");

    D3D12_DESCRIPTOR_HEAP_DESC dsvDesc{};
    dsvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDesc.NumDescriptors = 1;
    throwIfFailed(device_->CreateDescriptorHeap(&dsvDesc, IID_PPV_ARGS(&dsvHeap_)),
                  "failed to create DSV heap");

    D3D12_DESCRIPTOR_HEAP_DESC srvDesc{};
    srvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvDesc.NumDescriptors = kSrvHeapCapacity;
    srvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    throwIfFailed(device_->CreateDescriptorHeap(&srvDesc, IID_PPV_ARGS(&srvHeap_)),
                  "failed to create SRV heap");

    rtvDescriptorSize_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    dsvDescriptorSize_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    srvDescriptorSize_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

void Renderer::createRenderTargets()
{
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < kBackBufferCount; ++i)
    {
        throwIfFailed(swapChain_->GetBuffer(i, IID_PPV_ARGS(&renderTargets_[i])),
                      "failed to get back buffer");
        device_->CreateRenderTargetView(renderTargets_[i].Get(), nullptr, rtvHandle);
        rtvHandle.ptr += static_cast<SIZE_T>(rtvDescriptorSize_);
    }
}

void Renderer::destroyRenderTargets()
{
    for (auto& target : renderTargets_)
    {
        target.Reset();
    }
}

void Renderer::createDepthBuffer()
{
    D3D12_CLEAR_VALUE clearValue{};
    clearValue.Format = kDepthBufferFormat;
    clearValue.DepthStencil.Depth = 1.0f;
    clearValue.DepthStencil.Stencil = 0;

    D3D12_RESOURCE_DESC depthDesc{};
    depthDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthDesc.Width = static_cast<UINT>(width_);
    depthDesc.Height = static_cast<UINT>(height_);
    depthDesc.DepthOrArraySize = 1;
    depthDesc.MipLevels = 1;
    depthDesc.Format = kDepthBufferFormat;
    depthDesc.SampleDesc.Count = 1;
    depthDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    depthDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    const D3D12_HEAP_PROPERTIES defaultHeap = heapProps(D3D12_HEAP_TYPE_DEFAULT);
    throwIfFailed(device_->CreateCommittedResource(&defaultHeap,
                                                   D3D12_HEAP_FLAG_NONE,
                                                   &depthDesc,
                                                   D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                   &clearValue,
                                                   IID_PPV_ARGS(&depthBuffer_)),
                  "failed to create depth buffer");

    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
    dsvDesc.Format = kDepthBufferFormat;
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device_->CreateDepthStencilView(depthBuffer_.Get(), &dsvDesc, dsvHeap_->GetCPUDescriptorHandleForHeapStart());
}

void Renderer::destroyDepthBuffer()
{
    depthBuffer_.Reset();
}

void Renderer::createConstantBuffer()
{
    const std::uint64_t alignedSize = (sizeof(SceneConstants) + 255ull) & ~255ull;
    const std::uint64_t bufferSize = alignedSize * 2ull;
    const D3D12_HEAP_PROPERTIES uploadHeap = heapProps(D3D12_HEAP_TYPE_UPLOAD);
    const D3D12_RESOURCE_DESC cbDesc = bufferDesc(bufferSize);
    throwIfFailed(device_->CreateCommittedResource(&uploadHeap,
                                                   D3D12_HEAP_FLAG_NONE,
                                                   &cbDesc,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ,
                                                   nullptr,
                                                   IID_PPV_ARGS(&sceneConstantBuffer_)),
                  "failed to create scene constant buffer");
    throwIfFailed(sceneConstantBuffer_->Map(0, nullptr, reinterpret_cast<void**>(&mappedSceneConstants_)),
                  "failed to map scene constant buffer");
}

void Renderer::createPipelines()
{
    Microsoft::WRL::ComPtr<ID3DBlob> vertexShader = compileShader(kWorldVertexShader, "main", "vs_5_0");
    Microsoft::WRL::ComPtr<ID3DBlob> nearPixelShader = compileShader(kNearPixelShader, "main", "ps_5_0");
    Microsoft::WRL::ComPtr<ID3DBlob> farPixelShader = compileShader(kFarPixelShader, "main", "ps_5_0");

    D3D12_DESCRIPTOR_RANGE srvRange{};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 1;
    srvRange.BaseShaderRegister = 0;
    srvRange.OffsetInDescriptorsFromTableStart = 0;

    std::array<D3D12_ROOT_PARAMETER, 2> rootParameters{};
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParameters[0].Descriptor.ShaderRegister = 0;
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[1].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[1].DescriptorTable.pDescriptorRanges = &srvRange;
    rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderRegister = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc{};
    rootSignatureDesc.NumParameters = static_cast<UINT>(rootParameters.size());
    rootSignatureDesc.pParameters = rootParameters.data();
    rootSignatureDesc.NumStaticSamplers = 1;
    rootSignatureDesc.pStaticSamplers = &sampler;
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    Microsoft::WRL::ComPtr<ID3DBlob> serializedRootSignature;
    Microsoft::WRL::ComPtr<ID3DBlob> errorBlob;
    throwIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc,
                                              D3D_ROOT_SIGNATURE_VERSION_1,
                                              &serializedRootSignature,
                                              &errorBlob),
                  "failed to serialize root signature");
    throwIfFailed(device_->CreateRootSignature(0,
                                               serializedRootSignature->GetBufferPointer(),
                                               serializedRootSignature->GetBufferSize(),
                                               IID_PPV_ARGS(&rootSignature_)),
                  "failed to create root signature");

    constexpr std::array<D3D12_INPUT_ELEMENT_DESC, 5> inputLayout = {{
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, static_cast<UINT>(offsetof(WorldVertex, position)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, static_cast<UINT>(offsetof(WorldVertex, normal)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, static_cast<UINT>(offsetof(WorldVertex, tileCoord)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, static_cast<UINT>(offsetof(WorldVertex, atlasBase)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 2, DXGI_FORMAT_R32G32_FLOAT, 0, static_cast<UINT>(offsetof(WorldVertex, atlasSize)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    }};

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.InputLayout = {inputLayout.data(), static_cast<UINT>(inputLayout.size())};
    psoDesc.pRootSignature = rootSignature_.Get();
    psoDesc.VS = {vertexShader->GetBufferPointer(), vertexShader->GetBufferSize()};
    psoDesc.PS = {nearPixelShader->GetBufferPointer(), nearPixelShader->GetBufferSize()};
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    psoDesc.RasterizerState.FrontCounterClockwise = TRUE;
    psoDesc.RasterizerState.DepthClipEnable = TRUE;
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = kBackBufferFormat;
    psoDesc.DSVFormat = kDepthBufferFormat;
    psoDesc.SampleDesc.Count = 1;
    psoDesc.DepthStencilState.DepthEnable = TRUE;
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;

    throwIfFailed(device_->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&nearPipelineState_)),
                  "failed to create near-world pipeline");

    psoDesc.PS = {farPixelShader->GetBufferPointer(), farPixelShader->GetBufferSize()};
    throwIfFailed(device_->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&farPipelineState_)),
                  "failed to create far-world pipeline");
}

void Renderer::createImGui(GLFWwindow* window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    if (!ImGui_ImplGlfw_InitForOther(window, false))
    {
        throwRenderError("failed to initialize ImGui GLFW backend");
    }

    ImGui_ImplDX12_InitInfo initInfo{};
    initInfo.Device = device_.Get();
    initInfo.CommandQueue = commandQueue_.Get();
    initInfo.NumFramesInFlight = static_cast<int>(kBackBufferCount);
    initInfo.RTVFormat = kBackBufferFormat;
    initInfo.DSVFormat = kDepthBufferFormat;
    initInfo.UserData = this;
    initInfo.SrvDescriptorHeap = srvHeap_.Get();
    initInfo.SrvDescriptorAllocFn = &Renderer::imguiSrvAlloc;
    initInfo.SrvDescriptorFreeFn = &Renderer::imguiSrvFree;
    if (!ImGui_ImplDX12_Init(&initInfo))
    {
        throwRenderError("failed to initialize ImGui D3D12 backend");
    }
}

void Renderer::updateViewport(int width, int height)
{
    viewport_.TopLeftX = 0.0f;
    viewport_.TopLeftY = 0.0f;
    viewport_.Width = static_cast<float>(width);
    viewport_.Height = static_cast<float>(height);
    viewport_.MinDepth = 0.0f;
    viewport_.MaxDepth = 1.0f;

    scissorRect_.left = 0;
    scissorRect_.top = 0;
    scissorRect_.right = width;
    scissorRect_.bottom = height;
}

void Renderer::waitForGpu()
{
    if (!initialized_)
    {
        return;
    }

    ++fenceValue_;
    throwIfFailed(commandQueue_->Signal(fence_.Get(), fenceValue_), "failed to signal fence");
    if (fence_->GetCompletedValue() < fenceValue_)
    {
        throwIfFailed(fence_->SetEventOnCompletion(fenceValue_, fenceEvent_), "failed to set fence event");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }
}

void Renderer::resize(int width, int height)
{
    width = std::max(width, 1);
    height = std::max(height, 1);
    if (!initialized_ || (width == width_ && height == height_))
    {
        width_ = width;
        height_ = height;
        return;
    }

    waitForGpu();
    destroyDepthBuffer();
    destroyRenderTargets();

    DXGI_SWAP_CHAIN_DESC swapChainDesc{};
    throwIfFailed(swapChain_->GetDesc(&swapChainDesc), "failed to query swap chain description");
    throwIfFailed(swapChain_->ResizeBuffers(kBackBufferCount,
                                            static_cast<UINT>(width),
                                            static_cast<UINT>(height),
                                            swapChainDesc.BufferDesc.Format,
                                            swapChainDesc.Flags),
                  "failed to resize swap chain buffers");

    width_ = width;
    height_ = height;
    currentBackBufferIndex_ = swapChain_->GetCurrentBackBufferIndex();
    createRenderTargets();
    createDepthBuffer();
    updateViewport(width_, height_);
}

UINT Renderer::allocateSrvDescriptor()
{
    auto it = std::find(srvSlotsInUse_.begin(), srvSlotsInUse_.end(), false);
    if (it == srvSlotsInUse_.end())
    {
        throwRenderError("SRV heap exhausted");
    }

    const UINT index = static_cast<UINT>(std::distance(srvSlotsInUse_.begin(), it));
    srvSlotsInUse_[index] = true;
    return index;
}

void Renderer::freeSrvDescriptor(UINT index)
{
    if (index < srvSlotsInUse_.size())
    {
        srvSlotsInUse_[index] = false;
    }
}

D3D12_CPU_DESCRIPTOR_HANDLE Renderer::srvCpuHandle(UINT index) const noexcept
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = srvHeap_->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) * static_cast<SIZE_T>(srvDescriptorSize_);
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE Renderer::srvGpuHandle(UINT index) const noexcept
{
    D3D12_GPU_DESCRIPTOR_HANDLE handle = srvHeap_->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<UINT64>(index) * static_cast<UINT64>(srvDescriptorSize_);
    return handle;
}

void Renderer::imguiSrvAlloc(ImGui_ImplDX12_InitInfo* info,
                             D3D12_CPU_DESCRIPTOR_HANDLE* outCpuHandle,
                             D3D12_GPU_DESCRIPTOR_HANDLE* outGpuHandle)
{
    auto* renderer = static_cast<Renderer*>(info->UserData);
    const UINT index = renderer->allocateSrvDescriptor();
    *outCpuHandle = renderer->srvCpuHandle(index);
    *outGpuHandle = renderer->srvGpuHandle(index);
}

void Renderer::imguiSrvFree(ImGui_ImplDX12_InitInfo* info,
                            D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle,
                            D3D12_GPU_DESCRIPTOR_HANDLE)
{
    auto* renderer = static_cast<Renderer*>(info->UserData);
    const SIZE_T base = renderer->srvHeap_->GetCPUDescriptorHandleForHeapStart().ptr;
    if (cpuHandle.ptr < base)
    {
        return;
    }

    const SIZE_T index = (cpuHandle.ptr - base) / renderer->srvDescriptorSize_;
    renderer->freeSrvDescriptor(static_cast<UINT>(index));
}

LoadedTexture Renderer::loadTexture(const char* path)
{
    LoadedTexture texture{};

    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_set_flip_vertically_on_load(false);
    unsigned char* pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels)
    {
        throwRenderError(std::string("failed to load texture: ") + path);
    }

    texture.size = glm::ivec2(width, height);
    const UINT descriptorIndex = allocateSrvDescriptor();
    texture.srvCpu = srvCpuHandle(descriptorIndex);
    texture.srvGpu = srvGpuHandle(descriptorIndex);

    D3D12_RESOURCE_DESC textureDesc{};
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Width = static_cast<UINT>(width);
    textureDesc.Height = static_cast<UINT>(height);
    textureDesc.DepthOrArraySize = 1;
    textureDesc.MipLevels = 1;
    textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    const D3D12_HEAP_PROPERTIES defaultHeap = heapProps(D3D12_HEAP_TYPE_DEFAULT);
    throwIfFailed(device_->CreateCommittedResource(&defaultHeap,
                                                   D3D12_HEAP_FLAG_NONE,
                                                   &textureDesc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr,
                                                   IID_PPV_ARGS(&texture.resource)),
                  "failed to create texture resource");

    UINT64 uploadSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT numRows = 0;
    UINT64 rowSizeInBytes = 0;
    device_->GetCopyableFootprints(&textureDesc, 0, 1, 0, &layout, &numRows, &rowSizeInBytes, &uploadSize);

    Microsoft::WRL::ComPtr<ID3D12Resource> uploadBuffer;
    const D3D12_HEAP_PROPERTIES uploadHeap = heapProps(D3D12_HEAP_TYPE_UPLOAD);
    const D3D12_RESOURCE_DESC uploadDesc = bufferDesc(uploadSize);
    throwIfFailed(device_->CreateCommittedResource(&uploadHeap,
                                                   D3D12_HEAP_FLAG_NONE,
                                                   &uploadDesc,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ,
                                                   nullptr,
                                                   IID_PPV_ARGS(&uploadBuffer)),
                  "failed to create texture upload buffer");

    unsigned char* mapped = nullptr;
    throwIfFailed(uploadBuffer->Map(0, nullptr, reinterpret_cast<void**>(&mapped)),
                  "failed to map texture upload buffer");
    for (UINT row = 0; row < numRows; ++row)
    {
        const std::size_t srcOffset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width) * 4u;
        const std::size_t dstOffset = static_cast<std::size_t>(layout.Offset) +
                                      static_cast<std::size_t>(row) * static_cast<std::size_t>(layout.Footprint.RowPitch);
        std::memcpy(mapped + dstOffset, pixels + srcOffset, static_cast<std::size_t>(width) * 4u);
    }
    uploadBuffer->Unmap(0, nullptr);

    throwIfFailed(commandAllocator_->Reset(), "failed to reset command allocator for texture upload");
    throwIfFailed(commandList_->Reset(commandAllocator_.Get(), nullptr),
                  "failed to reset command list for texture upload");

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = texture.resource.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = uploadBuffer.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = layout;

    commandList_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    const D3D12_RESOURCE_BARRIER barrier =
        transitionBarrier(texture.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    commandList_->ResourceBarrier(1, &barrier);

    throwIfFailed(commandList_->Close(), "failed to close texture upload command list");
    ID3D12CommandList* commandLists[] = {commandList_.Get()};
    commandQueue_->ExecuteCommandLists(static_cast<UINT>(std::size(commandLists)), commandLists);
    waitForGpu();

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    device_->CreateShaderResourceView(texture.resource.Get(), &srvDesc, texture.srvCpu);

    stbi_image_free(pixels);
    return texture;
}

void Renderer::ensureFrameStarted() const
{
    if (!frameStarted_)
    {
        throwRenderError("frame commands requested before beginFrame()");
    }
}

void Renderer::beginFrame(const glm::vec4& clearColor)
{
    if (frameStarted_)
    {
        throwRenderError("beginFrame() called while a frame is already open");
    }

    currentBackBufferIndex_ = swapChain_->GetCurrentBackBufferIndex();
    throwIfFailed(commandAllocator_->Reset(), "failed to reset command allocator");
    throwIfFailed(commandList_->Reset(commandAllocator_.Get(), nullptr), "failed to reset command list");

    const D3D12_RESOURCE_BARRIER barrier =
        transitionBarrier(renderTargets_[currentBackBufferIndex_].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    commandList_->ResourceBarrier(1, &barrier);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
    rtvHandle.ptr += static_cast<SIZE_T>(currentBackBufferIndex_) * static_cast<SIZE_T>(rtvDescriptorSize_);
    const D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvHeap_->GetCPUDescriptorHandleForHeapStart();

    commandList_->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
    commandList_->RSSetViewports(1, &viewport_);
    commandList_->RSSetScissorRects(1, &scissorRect_);
    commandList_->ClearRenderTargetView(rtvHandle, &clearColor.x, 0, nullptr);
    commandList_->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    ID3D12DescriptorHeap* descriptorHeaps[] = {srvHeap_.Get()};
    commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(descriptorHeaps)), descriptorHeaps);

    frameStarted_ = true;
    imguiFrameStarted_ = false;
}

void Renderer::beginImGuiFrame()
{
    ensureFrameStarted();
    if (imguiFrameStarted_)
    {
        return;
    }

    ImGui_ImplDX12_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    imguiFrameStarted_ = true;
}

void Renderer::renderWorld(const WorldRenderData& renderData,
                           const glm::mat4& viewProj,
                           const glm::vec3& cameraPos,
                           const LoadedTexture& atlasTexture)
{
    ensureFrameStarted();
    if (!atlasTexture.valid())
    {
        return;
    }

    commandList_->SetGraphicsRootSignature(rootSignature_.Get());
    commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    commandList_->SetGraphicsRootDescriptorTable(1, atlasTexture.srvGpu);
    const std::uint64_t alignedSize = (sizeof(SceneConstants) + 255ull) & ~255ull;
    const D3D12_GPU_VIRTUAL_ADDRESS constantBufferBase = sceneConstantBuffer_->GetGPUVirtualAddress();

    SceneConstants& nearConstants = mappedSceneConstants_[0];
    nearConstants.viewProj = viewProj;
    nearConstants.lightDirection = glm::vec4(renderData.lightDirection, 0.0f);
    nearConstants.cameraPos = glm::vec4(cameraPos, 0.0f);
    nearConstants.highlightedBlock = glm::vec4(glm::vec3(renderData.highlightedBlock), 0.0f);
    nearConstants.fogColor = glm::vec4(renderData.fogColor, 1.0f);
    nearConstants.params = glm::vec4(renderData.fogStart,
                                     renderData.fogEnd,
                                     renderData.hasHighlight ? 1.0f : 0.0f,
                                     0.0f);

    commandList_->SetPipelineState(nearPipelineState_.Get());
    commandList_->SetGraphicsRootConstantBufferView(0, constantBufferBase);
    for (const ChunkRenderBatch& batch : renderData.nearBatches)
    {
        if (batch.indexCounts.empty())
        {
            continue;
        }

        commandList_->IASetVertexBuffers(0, 1, &batch.vertexBufferView);
        commandList_->IASetIndexBuffer(&batch.indexBufferView);
        for (std::size_t i = 0; i < batch.indexCounts.size(); ++i)
        {
            commandList_->DrawIndexedInstanced(batch.indexCounts[i], 1, batch.firstIndexLocations[i], batch.baseVertices[i], 0);
        }
    }

    SceneConstants& farConstants = mappedSceneConstants_[1];
    farConstants = nearConstants;
    farConstants.params = glm::vec4(renderData.fogStart, renderData.fogEnd, 0.0f, 0.0f);
    commandList_->SetPipelineState(farPipelineState_.Get());
    commandList_->SetGraphicsRootConstantBufferView(0, constantBufferBase + alignedSize);
    for (const ChunkRenderBatch& batch : renderData.farBatches)
    {
        if (batch.indexCounts.empty())
        {
            continue;
        }

        commandList_->IASetVertexBuffers(0, 1, &batch.vertexBufferView);
        commandList_->IASetIndexBuffer(&batch.indexBufferView);
        for (std::size_t i = 0; i < batch.indexCounts.size(); ++i)
        {
            commandList_->DrawIndexedInstanced(batch.indexCounts[i], 1, batch.firstIndexLocations[i], batch.baseVertices[i], 0);
        }
    }
}

void Renderer::endFrame()
{
    ensureFrameStarted();

    if (imguiFrameStarted_)
    {
        ImGui::Render();
        commandList_->SetDescriptorHeaps(1, srvHeap_.GetAddressOf());
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), commandList_.Get());
    }

    const D3D12_RESOURCE_BARRIER barrier =
        transitionBarrier(renderTargets_[currentBackBufferIndex_].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    commandList_->ResourceBarrier(1, &barrier);
    throwIfFailed(commandList_->Close(), "failed to close command list");

    ID3D12CommandList* commandLists[] = {commandList_.Get()};
    commandQueue_->ExecuteCommandLists(static_cast<UINT>(std::size(commandLists)), commandLists);
    throwIfFailed(swapChain_->Present(1, 0), "failed to present swap chain");

    ++fenceValue_;
    throwIfFailed(commandQueue_->Signal(fence_.Get(), fenceValue_), "failed to signal frame fence");

    frameStarted_ = false;
    imguiFrameStarted_ = false;
}
