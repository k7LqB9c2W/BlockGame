#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "renderer.h"

namespace
{
constexpr DXGI_FORMAT kBackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr DXGI_FORMAT kDepthBufferResourceFormat = DXGI_FORMAT_R32_TYPELESS;
constexpr DXGI_FORMAT kDepthBufferDsvFormat = DXGI_FORMAT_D32_FLOAT;
constexpr DXGI_FORMAT kDepthBufferSrvFormat = DXGI_FORMAT_R32_FLOAT;
constexpr DXGI_FORMAT kShadowMapFormat = DXGI_FORMAT_R32_TYPELESS;
constexpr DXGI_FORMAT kShadowMapDsvFormat = DXGI_FORMAT_D32_FLOAT;
constexpr DXGI_FORMAT kShadowMapSrvFormat = DXGI_FORMAT_R32_FLOAT;
constexpr DXGI_FORMAT kSceneColorFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
constexpr DXGI_FORMAT kAtmosphereFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;

constexpr UINT kFrameConstantBufferSize = 64u * 1024u;
constexpr UINT kShadowMapResolution = 2048u;
constexpr UINT kRtvIndexSceneColor = 2;
constexpr UINT kRtvIndexTransmittance = 3;
constexpr UINT kRtvIndexMultiScattering = 4;
constexpr UINT kRtvIndexSkyView = 5;
constexpr UINT kRtvIndexAerialPerspectiveBase = 6;
constexpr UINT kAerialPerspectiveSliceCount = 32;
constexpr UINT kRtvHeapCapacity = kRtvIndexAerialPerspectiveBase + kAerialPerspectiveSliceCount;
constexpr int kAtlasMinimumPaddingPixels = 2;

[[noreturn]] void throwRenderError(const std::string& message)
{
    throw std::runtime_error("Renderer: " + message);
}

void renderDebugLog(const std::string& message)
{
    if (std::getenv("BLOCKGAME_RENDER_DEBUG_LOG") == nullptr)
    {
        return;
    }
    std::cout << message << std::endl;
}

[[nodiscard]] bool envFlagEnabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr)
    {
        return false;
    }

    std::string normalized(value);
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return normalized != "0" &&
           normalized != "false" &&
           normalized != "off" &&
           normalized != "no";
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
    desc.Width = sizeInBytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

[[nodiscard]] D3D12_RESOURCE_DESC texture2DDesc(DXGI_FORMAT format,
                                                UINT width,
                                                UINT height,
                                                UINT16 mipLevels = 1,
                                                D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) noexcept
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = mipLevels;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;
    return desc;
}

[[nodiscard]] D3D12_RESOURCE_DESC texture2DArrayDesc(DXGI_FORMAT format,
                                                     UINT width,
                                                     UINT height,
                                                     UINT arraySize,
                                                     UINT16 mipLevels = 1,
                                                     D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) noexcept
{
    D3D12_RESOURCE_DESC desc = texture2DDesc(format, width, height, mipLevels, flags);
    desc.DepthOrArraySize = static_cast<UINT16>(arraySize);
    return desc;
}

[[nodiscard]] UINT16 computeMipLevelCount(int width, int height, UINT maxLevels) noexcept
{
    UINT16 mipLevels = 1;
    int mipWidth = std::max(width, 1);
    int mipHeight = std::max(height, 1);

    while (mipLevels < maxLevels && (mipWidth > 1 || mipHeight > 1))
    {
        mipWidth = std::max(1, mipWidth / 2);
        mipHeight = std::max(1, mipHeight / 2);
        ++mipLevels;
    }

    return mipLevels;
}

[[nodiscard]] bool shouldBuildRuntimeAtlas(const char* path, int width, int height) noexcept
{
    if (path == nullptr)
    {
        return false;
    }

    const std::string fileName = std::filesystem::path(path).filename().string();
    return fileName.find("atlas") != std::string::npos &&
           width >= kAtlasTileSizePixels &&
           height >= kAtlasTileSizePixels &&
           (width % kAtlasTileSizePixels) == 0 &&
           (height % kAtlasTileSizePixels) == 0;
}

[[nodiscard]] int nextPowerOfTwo(int value) noexcept
{
    int power = 1;
    while (power < value)
    {
        power <<= 1;
    }
    return power;
}

struct RuntimeAtlasInfo
{
    bool enabled{false};
    int tileSizePixels{0};
    int tileStridePixels{0};
    int tilePaddingPixels{0};
    glm::ivec2 tileCounts{0};
};

[[nodiscard]] RuntimeAtlasInfo makeRuntimeAtlasInfo(int sourceWidth, int sourceHeight)
{
    RuntimeAtlasInfo info{};
    info.enabled = true;
    info.tileSizePixels = kAtlasTileSizePixels;
    info.tileCounts = glm::ivec2(sourceWidth / kAtlasTileSizePixels,
                                 sourceHeight / kAtlasTileSizePixels);
    info.tileStridePixels = nextPowerOfTwo(kAtlasTileSizePixels + kAtlasMinimumPaddingPixels * 2);
    info.tilePaddingPixels = (info.tileStridePixels - kAtlasTileSizePixels) / 2;
    return info;
}

struct FrustumPlane
{
    glm::vec4 equation{0.0f};
};

[[nodiscard]] std::array<FrustumPlane, 6> extractFrustumPlanes(const glm::mat4& viewProj)
{
    const glm::mat4 rows = glm::transpose(viewProj);
    std::array<glm::vec4, 6> planes = {
        rows[3] + rows[0],
        rows[3] - rows[0],
        rows[3] + rows[1],
        rows[3] - rows[1],
        rows[3] + rows[2],
        rows[3] - rows[2]};

    std::array<FrustumPlane, 6> normalized{};
    for (std::size_t i = 0; i < planes.size(); ++i)
    {
        const glm::vec3 normal(planes[i]);
        const float length = glm::length(normal);
        normalized[i].equation = (length > 1e-5f) ? (planes[i] / length) : planes[i];
    }
    return normalized;
}

[[nodiscard]] std::vector<std::uint8_t> buildRuntimeAtlasPixels(const std::uint8_t* sourcePixels,
                                                                int sourceWidth,
                                                                int sourceHeight,
                                                                const RuntimeAtlasInfo& atlasInfo)
{
    const int runtimeWidth = atlasInfo.tileCounts.x * atlasInfo.tileStridePixels;
    const int runtimeHeight = atlasInfo.tileCounts.y * atlasInfo.tileStridePixels;
    std::vector<std::uint8_t> paddedPixels(static_cast<std::size_t>(runtimeWidth) *
                                               static_cast<std::size_t>(runtimeHeight) * 4u,
                                           0);

    auto copyPixel = [&](int dstX, int dstY, int srcX, int srcY)
    {
        srcX = std::clamp(srcX, 0, sourceWidth - 1);
        srcY = std::clamp(srcY, 0, sourceHeight - 1);
        const std::size_t srcIndex =
            (static_cast<std::size_t>(srcY) * static_cast<std::size_t>(sourceWidth) + static_cast<std::size_t>(srcX)) * 4u;
        const std::size_t dstIndex =
            (static_cast<std::size_t>(dstY) * static_cast<std::size_t>(runtimeWidth) + static_cast<std::size_t>(dstX)) * 4u;
        for (int channel = 0; channel < 4; ++channel)
        {
            paddedPixels[dstIndex + static_cast<std::size_t>(channel)] =
                sourcePixels[srcIndex + static_cast<std::size_t>(channel)];
        }
    };

    for (int tileY = 0; tileY < atlasInfo.tileCounts.y; ++tileY)
    {
        for (int tileX = 0; tileX < atlasInfo.tileCounts.x; ++tileX)
        {
            const int srcOriginX = tileX * atlasInfo.tileSizePixels;
            const int srcOriginY = tileY * atlasInfo.tileSizePixels;
            const int srcTileMaxX = srcOriginX + atlasInfo.tileSizePixels - 1;
            const int srcTileMaxY = srcOriginY + atlasInfo.tileSizePixels - 1;
            const int dstOriginX = tileX * atlasInfo.tileStridePixels;
            const int dstOriginY = tileY * atlasInfo.tileStridePixels;

            for (int y = 0; y < atlasInfo.tileStridePixels; ++y)
            {
                const int srcY = std::clamp(srcOriginY + (y - atlasInfo.tilePaddingPixels),
                                            srcOriginY,
                                            srcTileMaxY);
                for (int x = 0; x < atlasInfo.tileStridePixels; ++x)
                {
                    const int srcX = std::clamp(srcOriginX + (x - atlasInfo.tilePaddingPixels),
                                                srcOriginX,
                                                srcTileMaxX);
                    copyPixel(dstOriginX + x, dstOriginY + y, srcX, srcY);
                }
            }
        }
    }

    return paddedPixels;
}

[[nodiscard]] std::vector<std::vector<std::uint8_t>> buildTextureMipChain(const std::uint8_t* basePixels,
                                                                          int width,
                                                                          int height,
                                                                          UINT mipLevels,
                                                                          glm::ivec2 tileCounts = glm::ivec2(1))
{
    auto srgbChannelToLinear = [](std::uint8_t value) noexcept -> float
    {
        const float srgb = static_cast<float>(value) / 255.0f;
        if (srgb <= 0.04045f)
        {
            return srgb / 12.92f;
        }
        return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
    };

    auto linearChannelToSrgb = [](float value) noexcept -> std::uint8_t
    {
        const float clamped = std::clamp(value, 0.0f, 1.0f);
        const float srgb = (clamped <= 0.0031308f)
                               ? (clamped * 12.92f)
                               : (1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f);
        return static_cast<std::uint8_t>(std::lround(srgb * 255.0f));
    };

    std::vector<std::vector<std::uint8_t>> mipChain;
    mipChain.reserve(mipLevels);
    mipChain.emplace_back(basePixels,
                          basePixels + static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);

    const bool useTileAwareDownsample = tileCounts.x > 0 && tileCounts.y > 0 &&
                                        (width % tileCounts.x) == 0 &&
                                        (height % tileCounts.y) == 0;
    const int tilesX = useTileAwareDownsample ? tileCounts.x : 1;
    const int tilesY = useTileAwareDownsample ? tileCounts.y : 1;

    int srcWidth = width;
    int srcHeight = height;

    for (UINT mipIndex = 1; mipIndex < mipLevels; ++mipIndex)
    {
        const int dstWidth = std::max(1, srcWidth / 2);
        const int dstHeight = std::max(1, srcHeight / 2);
        std::vector<std::uint8_t> dstPixels(static_cast<std::size_t>(dstWidth) * static_cast<std::size_t>(dstHeight) * 4u, 0);
        const std::vector<std::uint8_t>& srcPixels = mipChain.back();

        const int srcTileWidth = useTileAwareDownsample ? std::max(1, srcWidth / tilesX) : srcWidth;
        const int srcTileHeight = useTileAwareDownsample ? std::max(1, srcHeight / tilesY) : srcHeight;
        const int dstTileWidth = useTileAwareDownsample ? std::max(1, dstWidth / tilesX) : dstWidth;
        const int dstTileHeight = useTileAwareDownsample ? std::max(1, dstHeight / tilesY) : dstHeight;

        for (int tileY = 0; tileY < tilesY; ++tileY)
        {
            for (int tileX = 0; tileX < tilesX; ++tileX)
            {
                const int srcTileOriginX = tileX * srcTileWidth;
                const int srcTileOriginY = tileY * srcTileHeight;
                const int dstTileOriginX = tileX * dstTileWidth;
                const int dstTileOriginY = tileY * dstTileHeight;

                for (int y = 0; y < dstTileHeight; ++y)
                {
                    for (int x = 0; x < dstTileWidth; ++x)
                    {
                        const int srcX0 = std::min(srcTileOriginX + x * 2, srcTileOriginX + srcTileWidth - 1);
                        const int srcY0 = std::min(srcTileOriginY + y * 2, srcTileOriginY + srcTileHeight - 1);
                        const int srcX1 = std::min(srcX0 + 1, srcTileOriginX + srcTileWidth - 1);
                        const int srcY1 = std::min(srcY0 + 1, srcTileOriginY + srcTileHeight - 1);
                        const int dstX = dstTileOriginX + x;
                        const int dstY = dstTileOriginY + y;
                        const std::size_t dstIndex =
                            (static_cast<std::size_t>(dstY) * static_cast<std::size_t>(dstWidth) + static_cast<std::size_t>(dstX)) * 4u;

                        for (int channel = 0; channel < 4; ++channel)
                        {
                            const auto sample = [&](int sampleX, int sampleY) -> std::uint32_t {
                                const std::size_t srcIndex =
                                    (static_cast<std::size_t>(sampleY) * static_cast<std::size_t>(srcWidth) +
                                     static_cast<std::size_t>(sampleX)) * 4u +
                                    static_cast<std::size_t>(channel);
                                return srcPixels[srcIndex];
                            };

                            if (channel == 3)
                            {
                                const std::uint32_t sum =
                                    sample(srcX0, srcY0) +
                                    sample(srcX1, srcY0) +
                                    sample(srcX0, srcY1) +
                                    sample(srcX1, srcY1);
                                dstPixels[dstIndex + static_cast<std::size_t>(channel)] =
                                    static_cast<std::uint8_t>((sum + 2u) / 4u);
                            }
                            else
                            {
                                const float average =
                                    (srgbChannelToLinear(static_cast<std::uint8_t>(sample(srcX0, srcY0))) +
                                     srgbChannelToLinear(static_cast<std::uint8_t>(sample(srcX1, srcY0))) +
                                     srgbChannelToLinear(static_cast<std::uint8_t>(sample(srcX0, srcY1))) +
                                     srgbChannelToLinear(static_cast<std::uint8_t>(sample(srcX1, srcY1)))) * 0.25f;
                                dstPixels[dstIndex + static_cast<std::size_t>(channel)] =
                                    linearChannelToSrgb(average);
                            }
                        }
                    }
                }
            }
        }

        mipChain.push_back(std::move(dstPixels));
        srcWidth = dstWidth;
        srcHeight = dstHeight;
    }

    return mipChain;
}

[[nodiscard]] UINT16 computeRuntimeAtlasMipLevelCount(int tileStridePixels) noexcept
{
    UINT16 mipLevels = 1;
    int mipSize = std::max(tileStridePixels, 1);
    while (mipSize > 1)
    {
        mipSize = std::max(1, mipSize / 2);
        ++mipLevels;
    }
    return mipLevels;
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
                                                       D3D12_RESOURCE_STATES after,
                                                       UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) noexcept
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = subresource;
    return barrier;
}

Microsoft::WRL::ComPtr<ID3DBlob> compileShaderFromFile(const std::string& path,
                                                       const char* entryPoint,
                                                       const char* target)
{
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifndef NDEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
    Microsoft::WRL::ComPtr<ID3DBlob> errors;
    const std::wstring widePath = std::filesystem::path(path).wstring();
    const HRESULT hr = D3DCompileFromFile(widePath.c_str(),
                                          nullptr,
                                          D3D_COMPILE_STANDARD_FILE_INCLUDE,
                                          entryPoint,
                                          target,
                                          flags,
                                          0,
                                          &bytecode,
                                          &errors);
    if (FAILED(hr))
    {
        std::string message = "shader compilation failed for " + path;
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

[[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE rtvHandleAt(ID3D12DescriptorHeap* heap,
                                                      UINT descriptorSize,
                                                      UINT index) noexcept
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(descriptorSize) * static_cast<SIZE_T>(index);
    return handle;
}
} // namespace

struct Renderer::AtmosphereRenderer
{
    struct LutTexture
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        UINT srvIndex{(std::numeric_limits<UINT>::max)()};
        D3D12_CPU_DESCRIPTOR_HANDLE srvCpu{};
        D3D12_GPU_DESCRIPTOR_HANDLE srvGpu{};
        D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
        D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
    };

    struct AerialPerspectiveTexture
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        UINT srvIndex{(std::numeric_limits<UINT>::max)()};
        D3D12_CPU_DESCRIPTOR_HANDLE srvCpu{};
        D3D12_GPU_DESCRIPTOR_HANDLE srvGpu{};
        std::array<D3D12_CPU_DESCRIPTOR_HANDLE, kAerialPerspectiveSliceCount> rtvs{};
        D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
    };

    void initialize(Renderer& renderer);
    void shutdown(Renderer& renderer);
    void resize(Renderer& renderer);
    [[nodiscard]] D3D12_GPU_DESCRIPTOR_HANDLE aerialPerspectiveSrv() const noexcept;
    void update(Renderer& renderer,
                const EnvironmentState& environment,
                const glm::mat4& view,
                const glm::mat4& proj,
                const glm::vec3& cameraPos);
    void renderSky(Renderer& renderer,
                   const EnvironmentState& environment,
                   const glm::mat4& view,
                   const glm::mat4& proj,
                   const glm::vec3& cameraPos);

private:
    void ensureResources(Renderer& renderer);
    void createPipelines(Renderer& renderer);
    void createResources(Renderer& renderer);
    void destroyResources(Renderer& renderer);
    void transition(Renderer& renderer,
                    ID3D12Resource* resource,
                    D3D12_RESOURCE_STATES& currentState,
                    D3D12_RESOURCE_STATES nextState);
    std::uint64_t uploadConstants(Renderer& renderer,
                                  const EnvironmentState& environment,
                                  const glm::mat4& view,
                                  const glm::mat4& proj,
                                  const glm::vec3& cameraPos,
                                  float sliceIndex);
    void renderLut(Renderer& renderer,
                   ID3D12PipelineState* pipelineState,
                   ID3D12Resource* targetResource,
                   D3D12_RESOURCE_STATES& targetState,
                   D3D12_CPU_DESCRIPTOR_HANDLE rtv,
                   const EnvironmentState& environment,
                   const glm::mat4& view,
                   const glm::mat4& proj,
                   const glm::vec3& cameraPos,
                   D3D12_GPU_DESCRIPTOR_HANDLE texture0,
                   D3D12_GPU_DESCRIPTOR_HANDLE texture1,
                   D3D12_GPU_DESCRIPTOR_HANDLE texture2,
                   float sliceIndex,
                   UINT width,
                   UINT height);
    void renderTransmittance(Renderer& renderer,
                             const EnvironmentState& environment,
                             const glm::mat4& view,
                             const glm::mat4& proj,
                             const glm::vec3& cameraPos);
    void renderMultiScattering(Renderer& renderer,
                               const EnvironmentState& environment,
                               const glm::mat4& view,
                               const glm::mat4& proj,
                               const glm::vec3& cameraPos);
    void renderSkyView(Renderer& renderer,
                       const EnvironmentState& environment,
                       const glm::mat4& view,
                       const glm::mat4& proj,
                       const glm::vec3& cameraPos);
    void renderAerialPerspective(Renderer& renderer,
                                 const EnvironmentState& environment,
                                 const glm::mat4& view,
                                 const glm::mat4& proj,
                                 const glm::vec3& cameraPos);

    LutTexture transmittance{};
    LutTexture multiScattering{};
    LutTexture skyView{};
    AerialPerspectiveTexture aerialPerspective{};
    Microsoft::WRL::ComPtr<ID3D12PipelineState> transmittancePso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> multiScatteringPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> skyViewPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> skyPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> aerialPerspectivePso;
    bool initializedStaticLuts{false};
    glm::vec3 lastSunDirection{0.0f, 1.0f, 0.0f};
    float lastGroundRadius{0.0f};
    float lastAtmosphereRadius{0.0f};
    float lastMieAnisotropy{0.0f};
};

Renderer::Renderer()
    : srvSlotsInUse_(kSrvHeapCapacity, false)
{
}

Renderer::~Renderer()
{
    shutdown();
}

void Renderer::AtmosphereRenderer::initialize(Renderer& renderer)
{
    createPipelines(renderer);
    createResources(renderer);
}

void Renderer::AtmosphereRenderer::shutdown(Renderer& renderer)
{
    destroyResources(renderer);
    transmittancePso.Reset();
    multiScatteringPso.Reset();
    skyViewPso.Reset();
    skyPso.Reset();
    aerialPerspectivePso.Reset();
    initializedStaticLuts = false;
}

void Renderer::AtmosphereRenderer::resize(Renderer& renderer)
{
    destroyResources(renderer);
    createResources(renderer);
    initializedStaticLuts = false;
}

D3D12_GPU_DESCRIPTOR_HANDLE Renderer::AtmosphereRenderer::aerialPerspectiveSrv() const noexcept
{
    return aerialPerspective.srvGpu;
}

void Renderer::AtmosphereRenderer::update(Renderer& renderer,
                                          const EnvironmentState& environment,
                                          const glm::mat4& view,
                                          const glm::mat4& proj,
                                          const glm::vec3& cameraPos)
{
    if (!environment.atmosphereEnabled)
    {
        return;
    }

    ensureResources(renderer);
    const bool staticLutsDirty =
        !initializedStaticLuts ||
        glm::distance(lastSunDirection, glm::normalize(environment.sunDirection)) > 1e-4f ||
        std::abs(lastGroundRadius - environment.atmosphere.groundRadiusKm) > 1e-3f ||
        std::abs(lastAtmosphereRadius - environment.atmosphere.atmosphereRadiusKm) > 1e-3f ||
        std::abs(lastMieAnisotropy - environment.atmosphere.mieAnisotropy) > 1e-4f;

    const auto start = std::chrono::steady_clock::now();
    if (staticLutsDirty)
    {
        renderTransmittance(renderer, environment, view, proj, cameraPos);
        renderMultiScattering(renderer, environment, view, proj, cameraPos);
        initializedStaticLuts = true;
        lastSunDirection = glm::normalize(environment.sunDirection);
        lastGroundRadius = environment.atmosphere.groundRadiusKm;
        lastAtmosphereRadius = environment.atmosphere.atmosphereRadiusKm;
        lastMieAnisotropy = environment.atmosphere.mieAnisotropy;
    }
    renderSkyView(renderer, environment, view, proj, cameraPos);
    renderAerialPerspective(renderer, environment, view, proj, cameraPos);
    renderer.profilingSnapshot_.atmosphereLutMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void Renderer::AtmosphereRenderer::renderSky(Renderer& renderer,
                                             const EnvironmentState& environment,
                                             const glm::mat4& view,
                                             const glm::mat4& proj,
                                             const glm::vec3& cameraPos)
{
    if (!environment.atmosphereEnabled)
    {
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    const D3D12_CPU_DESCRIPTOR_HANDLE depthHandle = renderer.depthDsv_;
    renderer.commandList_->OMSetRenderTargets(1, &renderer.sceneColorRtv_, FALSE, &depthHandle);
    renderer.commandList_->RSSetViewports(1, &renderer.viewport_);
    renderer.commandList_->RSSetScissorRects(1, &renderer.scissorRect_);
    renderer.commandList_->SetGraphicsRootSignature(renderer.fullscreenRootSignature_.Get());
    renderer.commandList_->SetPipelineState(skyPso.Get());
    const std::uint64_t cbAddress = uploadConstants(renderer, environment, view, proj, cameraPos, -1.0f);
    renderer.commandList_->SetGraphicsRootConstantBufferView(0, cbAddress);
    renderer.commandList_->SetGraphicsRootDescriptorTable(1, skyView.srvGpu);
    renderer.commandList_->SetGraphicsRootDescriptorTable(2, transmittance.srvGpu);
    renderer.commandList_->SetGraphicsRootDescriptorTable(3, multiScattering.srvGpu);
    renderer.commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    renderer.commandList_->DrawInstanced(3, 1, 0, 0);
    renderer.profilingSnapshot_.skyDrawMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
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
    createDepthPyramid();
    createShadowResources();
    createFrameResources();
    createSceneColor();
    createPipelines();
    atmosphere_ = std::make_unique<AtmosphereRenderer>();
    atmosphere_->initialize(*this);
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

    if (atmosphere_)
    {
        atmosphere_->shutdown(*this);
        atmosphere_.reset();
    }

      destroySceneColor();
      destroyFrameResources();
      toneMapPipelineState_.Reset();
      cloudPipelineState_.Reset();
      baseSkyPipelineState_.Reset();
    shadowPipelineState_.Reset();
    nearPipelineState_.Reset();
    farPipelineState_.Reset();
    lodIndirectPipelineState_.Reset();
    lodCullPipelineState_.Reset();
    depthPyramidPipelineState_.Reset();
    drawIndexedCommandSignature_.Reset();
    lodIndirectRootSignature_.Reset();
    lodCullRootSignature_.Reset();
    depthPyramidRootSignature_.Reset();
    fullscreenRootSignature_.Reset();
    shadowRootSignature_.Reset();
    worldRootSignature_.Reset();
    destroyShadowResources();
    destroyDepthPyramid();
    destroyDepthBuffer();
    destroyRenderTargets();
    srvHeap_.Reset();
    dsvHeap_.Reset();
    rtvHeap_.Reset();
    swapChain_.Reset();
    commandList_.Reset();
    uploadCommandAllocator_.Reset();
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

ID3D12Fence* Renderer::frameFence() const noexcept
{
    return fence_.Get();
}

UINT64 Renderer::lastSubmittedFrameFenceValue() const noexcept
{
    return fenceValue_;
}

void Renderer::setUploadSynchronization(ID3D12Fence* uploadFence, UINT64 uploadFenceValue) noexcept
{
    pendingUploadFence_ = uploadFence;
    pendingUploadFenceValue_ = uploadFenceValue;
}

int Renderer::width() const noexcept
{
    return width_;
}

int Renderer::height() const noexcept
{
    return height_;
}

RendererProfilingSnapshot Renderer::profilingSnapshot() const noexcept
{
    return profilingSnapshot_;
}

void Renderer::createFactory()
{
    UINT flags = 0;
    const bool enableDebugLayer =
#ifndef NDEBUG
        true;
#else
        envFlagEnabled("BLOCKGAME_ENABLE_D3D12_DEBUG_LAYER");
#endif

    if (enableDebugLayer)
    {
        Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
        {
            debugController->EnableDebugLayer();
            if (envFlagEnabled("BLOCKGAME_ENABLE_D3D12_GPU_VALIDATION"))
            {
                Microsoft::WRL::ComPtr<ID3D12Debug1> debugController1;
                if (SUCCEEDED(debugController.As(&debugController1)))
                {
                    debugController1->SetEnableGPUBasedValidation(TRUE);
                }
            }
            flags |= DXGI_CREATE_FACTORY_DEBUG;
            debugLayerEnabled_ = true;
        }
    }
    throwIfFailed(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory_)), "failed to create DXGI factory");
}

void Renderer::createDevice()
{
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter = chooseHardwareAdapter(factory_.Get());
    if (!adapter)
    {
        throwIfFailed(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)),
                      "failed to create D3D12 device");
    }
    else
    {
        throwIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)),
                      "failed to create D3D12 device");
    }

    if (debugLayerEnabled_)
    {
        device_.As(&infoQueue_);
        if (infoQueue_ != nullptr)
        {
            if (envFlagEnabled("BLOCKGAME_BREAK_ON_D3D12_ERROR"))
            {
                infoQueue_->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
                infoQueue_->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
            }
            infoQueue_->ClearStoredMessages();
        }
    }
}

void Renderer::createCommandObjects()
{
    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    throwIfFailed(device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue_)),
                  "failed to create command queue");

    throwIfFailed(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&uploadCommandAllocator_)),
                  "failed to create upload command allocator");
    throwIfFailed(device_->CreateCommandList(0,
                                             D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             uploadCommandAllocator_.Get(),
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
    const HWND hwnd = glfwGetWin32Window(window);
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
    rtvDesc.NumDescriptors = kRtvHeapCapacity;
    throwIfFailed(device_->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&rtvHeap_)),
                  "failed to create RTV heap");

    D3D12_DESCRIPTOR_HEAP_DESC dsvDesc{};
    dsvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDesc.NumDescriptors = 2;
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

    depthDsv_ = dsvHeap_->GetCPUDescriptorHandleForHeapStart();
    shadowMapDsv_ = depthDsv_;
    shadowMapDsv_.ptr += static_cast<SIZE_T>(dsvDescriptorSize_);
}

void Renderer::createRenderTargets()
{
    for (UINT i = 0; i < kBackBufferCount; ++i)
    {
        throwIfFailed(swapChain_->GetBuffer(i, IID_PPV_ARGS(&renderTargets_[i])),
                      "failed to get back buffer");
        device_->CreateRenderTargetView(renderTargets_[i].Get(), nullptr, rtvHandleAt(rtvHeap_.Get(), rtvDescriptorSize_, i));
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
    destroyDepthBuffer();

    D3D12_CLEAR_VALUE clearValue{};
    clearValue.Format = kDepthBufferDsvFormat;
    clearValue.DepthStencil.Depth = 1.0f;

    const D3D12_RESOURCE_DESC depthDesc =
        texture2DDesc(kDepthBufferResourceFormat,
                      static_cast<UINT>(width_),
                      static_cast<UINT>(height_),
                      1,
                      D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    const D3D12_HEAP_PROPERTIES defaultHeap = heapProps(D3D12_HEAP_TYPE_DEFAULT);
    throwIfFailed(device_->CreateCommittedResource(&defaultHeap,
                                                   D3D12_HEAP_FLAG_NONE,
                                                   &depthDesc,
                                                   D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                   &clearValue,
                                                   IID_PPV_ARGS(&depthBuffer_)),
                  "failed to create depth buffer");

    if (depthSrvIndex_ < 0)
    {
        depthSrvIndex_ = static_cast<int>(allocateSrvDescriptor());
    }
    depthSrvCpu_ = srvCpuHandle(static_cast<UINT>(depthSrvIndex_));
    depthSrvGpu_ = srvGpuHandle(static_cast<UINT>(depthSrvIndex_));

    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
    dsvDesc.Format = kDepthBufferDsvFormat;
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device_->CreateDepthStencilView(depthBuffer_.Get(), &dsvDesc, depthDsv_);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = kDepthBufferSrvFormat;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    device_->CreateShaderResourceView(depthBuffer_.Get(), &srvDesc, depthSrvCpu_);
}

void Renderer::destroyDepthBuffer()
{
    if (depthSrvIndex_ >= 0)
    {
        freeSrvDescriptor(static_cast<UINT>(depthSrvIndex_));
        depthSrvIndex_ = -1;
    }
    depthSrvCpu_ = {};
    depthSrvGpu_ = {};
    depthBuffer_.Reset();
}

void Renderer::createDepthPyramid()
{
    destroyDepthPyramid();
    if (!device_ || width_ <= 0 || height_ <= 0)
    {
        return;
    }

    depthPyramidMipCount_ = computeMipLevelCount(width_, height_, 16);
    const D3D12_HEAP_PROPERTIES defaultHeap = heapProps(D3D12_HEAP_TYPE_DEFAULT);
    const D3D12_RESOURCE_DESC desc =
        texture2DDesc(DXGI_FORMAT_R32_FLOAT,
                      static_cast<UINT>(width_),
                      static_cast<UINT>(height_),
                      static_cast<UINT16>(depthPyramidMipCount_),
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    throwIfFailed(device_->CreateCommittedResource(&defaultHeap,
                                                   D3D12_HEAP_FLAG_NONE,
                                                   &desc,
                                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                   nullptr,
                                                   IID_PPV_ARGS(&depthPyramid_)),
                  "failed to create depth pyramid");

    depthPyramidSrvIndex_ = static_cast<int>(allocateSrvDescriptor());
    depthPyramidSrvCpu_ = srvCpuHandle(static_cast<UINT>(depthPyramidSrvIndex_));
    depthPyramidSrvGpu_ = srvGpuHandle(static_cast<UINT>(depthPyramidSrvIndex_));

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = depthPyramidMipCount_;
    device_->CreateShaderResourceView(depthPyramid_.Get(), &srvDesc, depthPyramidSrvCpu_);

    depthPyramidUavIndices_.reserve(depthPyramidMipCount_);
    depthPyramidUavCpuHandles_.reserve(depthPyramidMipCount_);
    depthPyramidUavGpuHandles_.reserve(depthPyramidMipCount_);
    for (UINT mipIndex = 0; mipIndex < depthPyramidMipCount_; ++mipIndex)
    {
        const UINT descriptorIndex = allocateSrvDescriptor();
        depthPyramidUavIndices_.push_back(descriptorIndex);
        depthPyramidUavCpuHandles_.push_back(srvCpuHandle(descriptorIndex));
        depthPyramidUavGpuHandles_.push_back(srvGpuHandle(descriptorIndex));

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
        uavDesc.Format = DXGI_FORMAT_R32_FLOAT;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uavDesc.Texture2D.MipSlice = mipIndex;
        device_->CreateUnorderedAccessView(depthPyramid_.Get(), nullptr, &uavDesc, depthPyramidUavCpuHandles_.back());
    }

    depthPyramidState_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
}

void Renderer::destroyDepthPyramid()
{
    if (depthPyramidSrvIndex_ >= 0)
    {
        freeSrvDescriptor(static_cast<UINT>(depthPyramidSrvIndex_));
        depthPyramidSrvIndex_ = -1;
    }
    for (const UINT descriptorIndex : depthPyramidUavIndices_)
    {
        freeSrvDescriptor(descriptorIndex);
    }
    depthPyramidSrvCpu_ = {};
    depthPyramidSrvGpu_ = {};
    depthPyramidUavIndices_.clear();
    depthPyramidUavCpuHandles_.clear();
    depthPyramidUavGpuHandles_.clear();
    depthPyramidMipCount_ = 0;
    depthPyramid_.Reset();
    depthPyramidState_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
}

void Renderer::createShadowResources()
{
    destroyShadowResources();

    const D3D12_HEAP_PROPERTIES defaultHeap = heapProps(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_CLEAR_VALUE clearValue{};
    clearValue.Format = kShadowMapDsvFormat;
    clearValue.DepthStencil.Depth = 1.0f;

    const D3D12_RESOURCE_DESC shadowDesc =
        texture2DDesc(kShadowMapFormat,
                      kShadowMapResolution,
                      kShadowMapResolution,
                      1,
                      D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    throwIfFailed(device_->CreateCommittedResource(&defaultHeap,
                                                   D3D12_HEAP_FLAG_NONE,
                                                   &shadowDesc,
                                                   D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                                                   &clearValue,
                                                   IID_PPV_ARGS(&shadowMap_)),
                  "failed to create shadow map");

    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
    dsvDesc.Format = kShadowMapDsvFormat;
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device_->CreateDepthStencilView(shadowMap_.Get(), &dsvDesc, shadowMapDsv_);

    shadowMapSrvIndex_ = static_cast<int>(allocateSrvDescriptor());
    shadowMapSrvCpu_ = srvCpuHandle(static_cast<UINT>(shadowMapSrvIndex_));
    shadowMapSrvGpu_ = srvGpuHandle(static_cast<UINT>(shadowMapSrvIndex_));

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = kShadowMapSrvFormat;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    device_->CreateShaderResourceView(shadowMap_.Get(), &srvDesc, shadowMapSrvCpu_);

    shadowMapState_ = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
}

void Renderer::destroyShadowResources()
{
    if (shadowMapSrvIndex_ >= 0)
    {
        freeSrvDescriptor(static_cast<UINT>(shadowMapSrvIndex_));
        shadowMapSrvIndex_ = -1;
    }

    shadowMapSrvCpu_ = {};
    shadowMapSrvGpu_ = {};
    shadowMap_.Reset();
    shadowMapState_ = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
}

void Renderer::createFrameResources()
{
    const D3D12_HEAP_PROPERTIES uploadHeap = heapProps(D3D12_HEAP_TYPE_UPLOAD);
    const D3D12_RESOURCE_DESC bufferDescription = bufferDesc(kFrameConstantBufferSize);
    for (auto& frame : frameResources_)
    {
        throwIfFailed(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&frame.allocator)),
                      "failed to create frame command allocator");
        throwIfFailed(device_->CreateCommittedResource(&uploadHeap,
                                                       D3D12_HEAP_FLAG_NONE,
                                                       &bufferDescription,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ,
                                                       nullptr,
                                                       IID_PPV_ARGS(&frame.constantBuffer)),
                      "failed to create frame constant buffer");
        throwIfFailed(frame.constantBuffer->Map(0, nullptr, reinterpret_cast<void**>(&frame.mappedConstants)),
                      "failed to map frame constant buffer");
        frame.transientResources.clear();
        frame.fenceValue = 0;
    }
}

void Renderer::destroyFrameResources()
{
    for (auto& frame : frameResources_)
    {
        frame.transientResources.clear();
        frame.mappedConstants = nullptr;
        frame.constantBuffer.Reset();
        frame.allocator.Reset();
        frame.fenceValue = 0;
    }
}

void Renderer::createSceneColor()
{
    destroySceneColor();

    const D3D12_HEAP_PROPERTIES defaultHeap = heapProps(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_CLEAR_VALUE clearValue{};
    clearValue.Format = kSceneColorFormat;
    const D3D12_RESOURCE_DESC sceneDesc =
        texture2DDesc(kSceneColorFormat,
                      static_cast<UINT>(width_),
                      static_cast<UINT>(height_),
                      1,
                      D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    throwIfFailed(device_->CreateCommittedResource(&defaultHeap,
                                                   D3D12_HEAP_FLAG_NONE,
                                                   &sceneDesc,
                                                   D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                                                   &clearValue,
                                                   IID_PPV_ARGS(&sceneColor_)),
                  "failed to create HDR scene color");

    sceneColorRtv_ = rtvHandleAt(rtvHeap_.Get(), rtvDescriptorSize_, kRtvIndexSceneColor);
    device_->CreateRenderTargetView(sceneColor_.Get(), nullptr, sceneColorRtv_);

    const UINT descriptorIndex = allocateSrvDescriptor();
    sceneColorSrvIndex_ = static_cast<int>(descriptorIndex);
    sceneColorSrvCpu_ = srvCpuHandle(descriptorIndex);
    sceneColorSrvGpu_ = srvGpuHandle(descriptorIndex);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = kSceneColorFormat;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    device_->CreateShaderResourceView(sceneColor_.Get(), &srvDesc, sceneColorSrvCpu_);
    sceneColorState_ = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
}

void Renderer::destroySceneColor()
{
    if (sceneColorSrvIndex_ >= 0)
    {
        freeSrvDescriptor(static_cast<UINT>(sceneColorSrvIndex_));
        sceneColorSrvIndex_ = -1;
    }
    sceneColorSrvCpu_ = {};
    sceneColorSrvGpu_ = {};
    sceneColorRtv_ = {};
    sceneColor_.Reset();
    sceneColorState_ = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
}

void Renderer::createPipelines()
{
    D3D12_ROOT_PARAMETER shadowRootParam{};
    shadowRootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    shadowRootParam.Descriptor.ShaderRegister = 0;
    shadowRootParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

    D3D12_ROOT_SIGNATURE_DESC shadowRootDesc{};
    shadowRootDesc.NumParameters = 1;
    shadowRootDesc.pParameters = &shadowRootParam;
    shadowRootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    Microsoft::WRL::ComPtr<ID3DBlob> serializedRoot;
    Microsoft::WRL::ComPtr<ID3DBlob> errors;
    throwIfFailed(D3D12SerializeRootSignature(&shadowRootDesc,
                                              D3D_ROOT_SIGNATURE_VERSION_1,
                                              &serializedRoot,
                                              &errors),
                  "failed to serialize shadow root signature");
    throwIfFailed(device_->CreateRootSignature(0,
                                               serializedRoot->GetBufferPointer(),
                                               serializedRoot->GetBufferSize(),
                                               IID_PPV_ARGS(&shadowRootSignature_)),
                  "failed to create shadow root signature");

    D3D12_DESCRIPTOR_RANGE worldAtlasRange{};
    worldAtlasRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    worldAtlasRange.NumDescriptors = 1;
    worldAtlasRange.BaseShaderRegister = 0;

    D3D12_DESCRIPTOR_RANGE worldAerialRange{};
    worldAerialRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    worldAerialRange.NumDescriptors = 1;
    worldAerialRange.BaseShaderRegister = 1;

    D3D12_DESCRIPTOR_RANGE worldShadowRange{};
    worldShadowRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    worldShadowRange.NumDescriptors = 1;
    worldShadowRange.BaseShaderRegister = 2;

    std::array<D3D12_ROOT_PARAMETER, 4> worldRootParams{};
    worldRootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    worldRootParams[0].Descriptor.ShaderRegister = 0;
    worldRootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    worldRootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    worldRootParams[1].DescriptorTable.NumDescriptorRanges = 1;
    worldRootParams[1].DescriptorTable.pDescriptorRanges = &worldAtlasRange;
    worldRootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    worldRootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    worldRootParams[2].DescriptorTable.NumDescriptorRanges = 1;
    worldRootParams[2].DescriptorTable.pDescriptorRanges = &worldAerialRange;
    worldRootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    worldRootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    worldRootParams[3].DescriptorTable.NumDescriptorRanges = 1;
    worldRootParams[3].DescriptorTable.pDescriptorRanges = &worldShadowRange;
    worldRootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    std::array<D3D12_STATIC_SAMPLER_DESC, 3> worldSamplers{};
    worldSamplers[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    worldSamplers[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    worldSamplers[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    worldSamplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    worldSamplers[0].ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    worldSamplers[0].MaxLOD = D3D12_FLOAT32_MAX;
    worldSamplers[0].ShaderRegister = 0;
    worldSamplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    worldSamplers[1] = worldSamplers[0];
    worldSamplers[1].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    worldSamplers[1].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    worldSamplers[1].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    worldSamplers[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    worldSamplers[1].ShaderRegister = 1;
    worldSamplers[2] = worldSamplers[1];
    worldSamplers[2].Filter = D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    worldSamplers[2].AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    worldSamplers[2].AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    worldSamplers[2].AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    worldSamplers[2].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
    worldSamplers[2].ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    worldSamplers[2].ShaderRegister = 2;

    D3D12_ROOT_SIGNATURE_DESC worldRootDesc{};
    worldRootDesc.NumParameters = static_cast<UINT>(worldRootParams.size());
    worldRootDesc.pParameters = worldRootParams.data();
    worldRootDesc.NumStaticSamplers = static_cast<UINT>(worldSamplers.size());
    worldRootDesc.pStaticSamplers = worldSamplers.data();
    worldRootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    serializedRoot.Reset();
    errors.Reset();
    throwIfFailed(D3D12SerializeRootSignature(&worldRootDesc,
                                              D3D_ROOT_SIGNATURE_VERSION_1,
                                              &serializedRoot,
                                              &errors),
                  "failed to serialize world root signature");
    throwIfFailed(device_->CreateRootSignature(0,
                                               serializedRoot->GetBufferPointer(),
                                               serializedRoot->GetBufferSize(),
                                               IID_PPV_ARGS(&worldRootSignature_)),
                  "failed to create world root signature");

    D3D12_DESCRIPTOR_RANGE fullscreenRange0{};
    fullscreenRange0.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    fullscreenRange0.NumDescriptors = 1;
    fullscreenRange0.BaseShaderRegister = 0;
    D3D12_DESCRIPTOR_RANGE fullscreenRange1 = fullscreenRange0;
    fullscreenRange1.BaseShaderRegister = 1;
    D3D12_DESCRIPTOR_RANGE fullscreenRange2 = fullscreenRange0;
    fullscreenRange2.BaseShaderRegister = 2;

    std::array<D3D12_ROOT_PARAMETER, 4> fullscreenRootParams{};
    fullscreenRootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    fullscreenRootParams[0].Descriptor.ShaderRegister = 0;
    fullscreenRootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    fullscreenRootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    fullscreenRootParams[1].DescriptorTable.NumDescriptorRanges = 1;
    fullscreenRootParams[1].DescriptorTable.pDescriptorRanges = &fullscreenRange0;
    fullscreenRootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    fullscreenRootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    fullscreenRootParams[2].DescriptorTable.NumDescriptorRanges = 1;
    fullscreenRootParams[2].DescriptorTable.pDescriptorRanges = &fullscreenRange1;
    fullscreenRootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    fullscreenRootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    fullscreenRootParams[3].DescriptorTable.NumDescriptorRanges = 1;
    fullscreenRootParams[3].DescriptorTable.pDescriptorRanges = &fullscreenRange2;
    fullscreenRootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC fullscreenSampler{};
    fullscreenSampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    fullscreenSampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    fullscreenSampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    fullscreenSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    fullscreenSampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    fullscreenSampler.MaxLOD = D3D12_FLOAT32_MAX;
    fullscreenSampler.ShaderRegister = 0;
    fullscreenSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC fullscreenRootDesc{};
    fullscreenRootDesc.NumParameters = static_cast<UINT>(fullscreenRootParams.size());
    fullscreenRootDesc.pParameters = fullscreenRootParams.data();
    fullscreenRootDesc.NumStaticSamplers = 1;
    fullscreenRootDesc.pStaticSamplers = &fullscreenSampler;
    fullscreenRootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    serializedRoot.Reset();
    errors.Reset();
    throwIfFailed(D3D12SerializeRootSignature(&fullscreenRootDesc,
                                              D3D_ROOT_SIGNATURE_VERSION_1,
                                              &serializedRoot,
                                              &errors),
                  "failed to serialize fullscreen root signature");
    throwIfFailed(device_->CreateRootSignature(0,
                                               serializedRoot->GetBufferPointer(),
                                               serializedRoot->GetBufferSize(),
                                               IID_PPV_ARGS(&fullscreenRootSignature_)),
                  "failed to create fullscreen root signature");

    Microsoft::WRL::ComPtr<ID3DBlob> worldVs =
        compileShaderFromFile(shaderPath("world_vs.hlsl"), "main", "vs_5_0");
    Microsoft::WRL::ComPtr<ID3DBlob> shadowVs =
        compileShaderFromFile(shaderPath("shadow_vs.hlsl"), "main", "vs_5_0");
    Microsoft::WRL::ComPtr<ID3DBlob> nearPs =
        compileShaderFromFile(shaderPath("world_near_ps.hlsl"), "main", "ps_5_0");
      Microsoft::WRL::ComPtr<ID3DBlob> farPs =
          compileShaderFromFile(shaderPath("world_far_ps.hlsl"), "main", "ps_5_0");
      Microsoft::WRL::ComPtr<ID3DBlob> depthPyramidCs =
          compileShaderFromFile(shaderPath("depth_pyramid.hlsl"), "DepthPyramidMain", "cs_5_0");
      Microsoft::WRL::ComPtr<ID3DBlob> lodCullCs =
          compileShaderFromFile(shaderPath("lod_gpu_cull.hlsl"), "LodCullMain", "cs_5_0");
      Microsoft::WRL::ComPtr<ID3DBlob> lodIndirectCs =
          compileShaderFromFile(shaderPath("lod_gpu_cull.hlsl"), "LodIndirectBuildMain", "cs_5_0");
      Microsoft::WRL::ComPtr<ID3DBlob> fullscreenVs =
          compileShaderFromFile(shaderPath("fullscreen_vs.hlsl"), "main", "vs_5_0");
      Microsoft::WRL::ComPtr<ID3DBlob> baseSkyPs =
          compileShaderFromFile(shaderPath("base_sky_ps.hlsl"), "main", "ps_5_0");
      Microsoft::WRL::ComPtr<ID3DBlob> cloudsVs =
          compileShaderFromFile(shaderPath("clouds_vs.hlsl"), "main", "vs_5_0");
      Microsoft::WRL::ComPtr<ID3DBlob> cloudsPs =
          compileShaderFromFile(shaderPath("clouds_ps.hlsl"), "main", "ps_5_0");
      Microsoft::WRL::ComPtr<ID3DBlob> tonePs =
          compileShaderFromFile(shaderPath("tone_map_ps.hlsl"), "main", "ps_5_0");

    constexpr std::array<D3D12_INPUT_ELEMENT_DESC, 6> inputLayout = {{
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, static_cast<UINT>(offsetof(WorldVertex, position)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, static_cast<UINT>(offsetof(WorldVertex, normal)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, static_cast<UINT>(offsetof(WorldVertex, tileCoord)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, static_cast<UINT>(offsetof(WorldVertex, atlasBase)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 2, DXGI_FORMAT_R32G32_FLOAT, 0, static_cast<UINT>(offsetof(WorldVertex, atlasSize)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32_UINT, 0, static_cast<UINT>(offsetof(WorldVertex, lightingData)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    }};

    constexpr std::array<D3D12_INPUT_ELEMENT_DESC, 1> shadowInputLayout = {{
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, static_cast<UINT>(offsetof(WorldVertex, position)), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    }};

    D3D12_GRAPHICS_PIPELINE_STATE_DESC shadowPso{};
    shadowPso.InputLayout = {shadowInputLayout.data(), static_cast<UINT>(shadowInputLayout.size())};
    shadowPso.pRootSignature = shadowRootSignature_.Get();
    shadowPso.VS = {shadowVs->GetBufferPointer(), shadowVs->GetBufferSize()};
    shadowPso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    shadowPso.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    shadowPso.RasterizerState.FrontCounterClockwise = TRUE;
    shadowPso.RasterizerState.DepthClipEnable = TRUE;
    shadowPso.RasterizerState.DepthBias = 1000;
    shadowPso.RasterizerState.SlopeScaledDepthBias = 2.0f;
    shadowPso.BlendState.RenderTarget[0].RenderTargetWriteMask = 0;
    shadowPso.SampleMask = UINT_MAX;
    shadowPso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    shadowPso.NumRenderTargets = 0;
    shadowPso.DSVFormat = kShadowMapDsvFormat;
    shadowPso.SampleDesc.Count = 1;
    shadowPso.DepthStencilState.DepthEnable = TRUE;
    shadowPso.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    shadowPso.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    throwIfFailed(device_->CreateGraphicsPipelineState(&shadowPso, IID_PPV_ARGS(&shadowPipelineState_)),
                  "failed to create shadow pipeline");

    D3D12_GRAPHICS_PIPELINE_STATE_DESC worldPso{};
    worldPso.InputLayout = {inputLayout.data(), static_cast<UINT>(inputLayout.size())};
    worldPso.pRootSignature = worldRootSignature_.Get();
    worldPso.VS = {worldVs->GetBufferPointer(), worldVs->GetBufferSize()};
    worldPso.PS = {nearPs->GetBufferPointer(), nearPs->GetBufferSize()};
    worldPso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    worldPso.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    worldPso.RasterizerState.FrontCounterClockwise = TRUE;
    worldPso.RasterizerState.DepthClipEnable = TRUE;
    worldPso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    worldPso.SampleMask = UINT_MAX;
    worldPso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    worldPso.NumRenderTargets = 1;
    worldPso.RTVFormats[0] = kSceneColorFormat;
    worldPso.DSVFormat = kDepthBufferDsvFormat;
    worldPso.SampleDesc.Count = 1;
    worldPso.DepthStencilState.DepthEnable = TRUE;
    worldPso.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    worldPso.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    throwIfFailed(device_->CreateGraphicsPipelineState(&worldPso, IID_PPV_ARGS(&nearPipelineState_)),
                  "failed to create near pipeline");

    worldPso.PS = {farPs->GetBufferPointer(), farPs->GetBufferSize()};
      throwIfFailed(device_->CreateGraphicsPipelineState(&worldPso, IID_PPV_ARGS(&farPipelineState_)),
                    "failed to create far pipeline");

      D3D12_INDIRECT_ARGUMENT_DESC drawIndexedArgument{};
      drawIndexedArgument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
      D3D12_COMMAND_SIGNATURE_DESC drawIndexedSignature{};
      drawIndexedSignature.ByteStride = sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
      drawIndexedSignature.NumArgumentDescs = 1;
      drawIndexedSignature.pArgumentDescs = &drawIndexedArgument;
      throwIfFailed(device_->CreateCommandSignature(&drawIndexedSignature,
                                                    nullptr,
                                                    IID_PPV_ARGS(&drawIndexedCommandSignature_)),
                    "failed to create draw indexed command signature");

      auto createComputeRootSignature = [this](const D3D12_ROOT_SIGNATURE_DESC& desc,
                                               Microsoft::WRL::ComPtr<ID3D12RootSignature>& rootSignature,
                                               const char* label)
      {
          const std::string serializeMessage = std::string("failed to serialize ") + label;
          const std::string createMessage = std::string("failed to create ") + label;
          Microsoft::WRL::ComPtr<ID3DBlob> serialized;
          Microsoft::WRL::ComPtr<ID3DBlob> rootErrors;
          throwIfFailed(D3D12SerializeRootSignature(&desc,
                                                    D3D_ROOT_SIGNATURE_VERSION_1,
                                                    &serialized,
                                                    &rootErrors),
                        serializeMessage);
          throwIfFailed(device_->CreateRootSignature(0,
                                                     serialized->GetBufferPointer(),
                                                     serialized->GetBufferSize(),
                                                     IID_PPV_ARGS(&rootSignature)),
                        createMessage);
      };

      D3D12_DESCRIPTOR_RANGE depthPyramidSrvRange{};
      depthPyramidSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
      depthPyramidSrvRange.NumDescriptors = 1;
      depthPyramidSrvRange.BaseShaderRegister = 0;
      std::array<D3D12_ROOT_PARAMETER, 3> depthPyramidRootParams{};
      depthPyramidRootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      depthPyramidRootParams[0].Constants.ShaderRegister = 0;
      depthPyramidRootParams[0].Constants.Num32BitValues = 5;
      depthPyramidRootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      depthPyramidRootParams[1].DescriptorTable.NumDescriptorRanges = 1;
      depthPyramidRootParams[1].DescriptorTable.pDescriptorRanges = &depthPyramidSrvRange;
      depthPyramidRootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      D3D12_DESCRIPTOR_RANGE depthPyramidUavRange{};
      depthPyramidUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
      depthPyramidUavRange.NumDescriptors = 1;
      depthPyramidUavRange.BaseShaderRegister = 0;
      depthPyramidRootParams[2].DescriptorTable.NumDescriptorRanges = 1;
      depthPyramidRootParams[2].DescriptorTable.pDescriptorRanges = &depthPyramidUavRange;
      D3D12_ROOT_SIGNATURE_DESC depthPyramidRootDesc{};
      depthPyramidRootDesc.NumParameters = static_cast<UINT>(depthPyramidRootParams.size());
      depthPyramidRootDesc.pParameters = depthPyramidRootParams.data();
      createComputeRootSignature(depthPyramidRootDesc, depthPyramidRootSignature_, "depth pyramid root signature");

      D3D12_COMPUTE_PIPELINE_STATE_DESC depthPyramidPso{};
      depthPyramidPso.pRootSignature = depthPyramidRootSignature_.Get();
      depthPyramidPso.CS = {depthPyramidCs->GetBufferPointer(), depthPyramidCs->GetBufferSize()};
      throwIfFailed(device_->CreateComputePipelineState(&depthPyramidPso, IID_PPV_ARGS(&depthPyramidPipelineState_)),
                    "failed to create depth pyramid pipeline");

      std::array<D3D12_ROOT_PARAMETER, 5> lodCullRootParams{};
      lodCullRootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      lodCullRootParams[0].Constants.ShaderRegister = 0;
      lodCullRootParams[0].Constants.Num32BitValues = 44;
      lodCullRootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      lodCullRootParams[1].DescriptorTable.NumDescriptorRanges = 1;
      lodCullRootParams[1].DescriptorTable.pDescriptorRanges = &depthPyramidSrvRange;
      lodCullRootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
      lodCullRootParams[2].Descriptor.ShaderRegister = 1;
      lodCullRootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
      lodCullRootParams[3].Descriptor.ShaderRegister = 0;
      lodCullRootParams[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
      lodCullRootParams[4].Descriptor.ShaderRegister = 1;
      D3D12_ROOT_SIGNATURE_DESC lodCullRootDesc{};
      lodCullRootDesc.NumParameters = static_cast<UINT>(lodCullRootParams.size());
      lodCullRootDesc.pParameters = lodCullRootParams.data();
      createComputeRootSignature(lodCullRootDesc, lodCullRootSignature_, "lod cull root signature");

      D3D12_COMPUTE_PIPELINE_STATE_DESC lodCullPso{};
      lodCullPso.pRootSignature = lodCullRootSignature_.Get();
      lodCullPso.CS = {lodCullCs->GetBufferPointer(), lodCullCs->GetBufferSize()};
      throwIfFailed(device_->CreateComputePipelineState(&lodCullPso, IID_PPV_ARGS(&lodCullPipelineState_)),
                    "failed to create lod cull pipeline");

      std::array<D3D12_ROOT_PARAMETER, 4> lodIndirectRootParams{};
      lodIndirectRootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
      lodIndirectRootParams[0].Descriptor.ShaderRegister = 1;
      lodIndirectRootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
      lodIndirectRootParams[1].Descriptor.ShaderRegister = 2;
      lodIndirectRootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
      lodIndirectRootParams[2].Descriptor.ShaderRegister = 2;
      lodIndirectRootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
      lodIndirectRootParams[3].Descriptor.ShaderRegister = 1;
      D3D12_ROOT_SIGNATURE_DESC lodIndirectRootDesc{};
      lodIndirectRootDesc.NumParameters = static_cast<UINT>(lodIndirectRootParams.size());
      lodIndirectRootDesc.pParameters = lodIndirectRootParams.data();
      createComputeRootSignature(lodIndirectRootDesc, lodIndirectRootSignature_, "lod indirect root signature");

      D3D12_COMPUTE_PIPELINE_STATE_DESC lodIndirectPso{};
      lodIndirectPso.pRootSignature = lodIndirectRootSignature_.Get();
      lodIndirectPso.CS = {lodIndirectCs->GetBufferPointer(), lodIndirectCs->GetBufferSize()};
      throwIfFailed(device_->CreateComputePipelineState(&lodIndirectPso, IID_PPV_ARGS(&lodIndirectPipelineState_)),
                    "failed to create lod indirect pipeline");

      D3D12_GRAPHICS_PIPELINE_STATE_DESC baseSkyPso{};
      baseSkyPso.pRootSignature = fullscreenRootSignature_.Get();
      baseSkyPso.VS = {fullscreenVs->GetBufferPointer(), fullscreenVs->GetBufferSize()};
      baseSkyPso.PS = {baseSkyPs->GetBufferPointer(), baseSkyPs->GetBufferSize()};
      baseSkyPso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
      baseSkyPso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
      baseSkyPso.RasterizerState.DepthClipEnable = TRUE;
      baseSkyPso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
      baseSkyPso.SampleMask = UINT_MAX;
      baseSkyPso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
      baseSkyPso.NumRenderTargets = 1;
      baseSkyPso.RTVFormats[0] = kSceneColorFormat;
      baseSkyPso.SampleDesc.Count = 1;
      baseSkyPso.DepthStencilState.DepthEnable = FALSE;
      baseSkyPso.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
      baseSkyPso.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
      throwIfFailed(device_->CreateGraphicsPipelineState(&baseSkyPso, IID_PPV_ARGS(&baseSkyPipelineState_)),
                    "failed to create base sky pipeline");

      D3D12_GRAPHICS_PIPELINE_STATE_DESC cloudPso{};
      cloudPso.pRootSignature = fullscreenRootSignature_.Get();
      cloudPso.VS = {cloudsVs->GetBufferPointer(), cloudsVs->GetBufferSize()};
      cloudPso.PS = {cloudsPs->GetBufferPointer(), cloudsPs->GetBufferSize()};
      cloudPso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
      cloudPso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
      cloudPso.RasterizerState.DepthClipEnable = TRUE;
      cloudPso.BlendState.AlphaToCoverageEnable = FALSE;
      cloudPso.BlendState.IndependentBlendEnable = FALSE;
      cloudPso.BlendState.RenderTarget[0].BlendEnable = TRUE;
      cloudPso.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
      cloudPso.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
      cloudPso.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
      cloudPso.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
      cloudPso.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
      cloudPso.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
      cloudPso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
      cloudPso.SampleMask = UINT_MAX;
      cloudPso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
      cloudPso.NumRenderTargets = 1;
      cloudPso.RTVFormats[0] = kSceneColorFormat;
      cloudPso.DSVFormat = kDepthBufferDsvFormat;
      cloudPso.SampleDesc.Count = 1;
      cloudPso.DepthStencilState.DepthEnable = TRUE;
      cloudPso.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
      cloudPso.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
      throwIfFailed(device_->CreateGraphicsPipelineState(&cloudPso, IID_PPV_ARGS(&cloudPipelineState_)),
                    "failed to create cloud pipeline");

      D3D12_GRAPHICS_PIPELINE_STATE_DESC tonePso{};
      tonePso.pRootSignature = fullscreenRootSignature_.Get();
      tonePso.VS = {fullscreenVs->GetBufferPointer(), fullscreenVs->GetBufferSize()};
    tonePso.PS = {tonePs->GetBufferPointer(), tonePs->GetBufferSize()};
    tonePso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    tonePso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    tonePso.RasterizerState.DepthClipEnable = TRUE;
    tonePso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    tonePso.SampleMask = UINT_MAX;
    tonePso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    tonePso.NumRenderTargets = 1;
    tonePso.RTVFormats[0] = kBackBufferFormat;
    tonePso.SampleDesc.Count = 1;
    tonePso.DepthStencilState.DepthEnable = FALSE;
    tonePso.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    tonePso.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    throwIfFailed(device_->CreateGraphicsPipelineState(&tonePso, IID_PPV_ARGS(&toneMapPipelineState_)),
                  "failed to create tone map pipeline");
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
    initInfo.DSVFormat = kDepthBufferDsvFormat;
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

    for (auto& frame : frameResources_)
    {
        frame.fenceValue = fenceValue_;
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
    destroySceneColor();
    destroyDepthPyramid();
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
    createDepthPyramid();
    createSceneColor();
    if (atmosphere_)
    {
        atmosphere_->resize(*this);
    }
    updateViewport(width_, height_);
}

UINT Renderer::allocateSrvDescriptor()
{
    for (UINT i = 0; i < srvSlotsInUse_.size(); ++i)
    {
        if (!srvSlotsInUse_[i])
        {
            srvSlotsInUse_[i] = true;
            return i;
        }
    }
    throwRenderError("SRV heap exhausted");
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

    texture.sourceSize = glm::ivec2(width, height);
    texture.tileCounts = glm::ivec2(1, 1);

    RuntimeAtlasInfo atlasInfo{};
    std::vector<std::uint8_t> runtimeAtlasPixels;
    const std::uint8_t* uploadPixels = pixels;
    int uploadWidth = width;
    int uploadHeight = height;
    if (shouldBuildRuntimeAtlas(path, width, height))
    {
        atlasInfo = makeRuntimeAtlasInfo(width, height);
        runtimeAtlasPixels = buildRuntimeAtlasPixels(pixels, width, height, atlasInfo);
        uploadPixels = runtimeAtlasPixels.data();
        uploadWidth = atlasInfo.tileCounts.x * atlasInfo.tileStridePixels;
        uploadHeight = atlasInfo.tileCounts.y * atlasInfo.tileStridePixels;
        texture.tileCounts = atlasInfo.tileCounts;
        texture.tileSizePixels = atlasInfo.tileSizePixels;
        texture.tileStridePixels = atlasInfo.tileStridePixels;
        texture.tilePaddingPixels = atlasInfo.tilePaddingPixels;
    }

    texture.size = glm::ivec2(uploadWidth, uploadHeight);
    texture.mipLevels = atlasInfo.enabled
                            ? computeRuntimeAtlasMipLevelCount(atlasInfo.tileStridePixels)
                            : computeMipLevelCount(uploadWidth,
                                                   uploadHeight,
                                                   (std::numeric_limits<UINT>::max)());
    texture.srvIndex = allocateSrvDescriptor();
    texture.srvCpu = srvCpuHandle(texture.srvIndex);
    texture.srvGpu = srvGpuHandle(texture.srvIndex);
    const std::vector<std::vector<std::uint8_t>> mipChain =
        buildTextureMipChain(uploadPixels,
                            uploadWidth,
                            uploadHeight,
                            texture.mipLevels,
                            texture.tileCounts);

    const D3D12_RESOURCE_DESC textureDesc = texture2DDesc(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                                                          static_cast<UINT>(uploadWidth),
                                                          static_cast<UINT>(uploadHeight),
                                                          static_cast<UINT16>(texture.mipLevels));
    const D3D12_HEAP_PROPERTIES defaultHeap = heapProps(D3D12_HEAP_TYPE_DEFAULT);
    throwIfFailed(device_->CreateCommittedResource(&defaultHeap,
                                                   D3D12_HEAP_FLAG_NONE,
                                                   &textureDesc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr,
                                                   IID_PPV_ARGS(&texture.resource)),
                  "failed to create texture resource");

    UINT64 uploadSize = 0;
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(texture.mipLevels);
    std::vector<UINT> numRows(texture.mipLevels, 0);
    device_->GetCopyableFootprints(&textureDesc,
                                   0,
                                   texture.mipLevels,
                                   0,
                                   layouts.data(),
                                   numRows.data(),
                                   nullptr,
                                   &uploadSize);

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
    int mipWidth = uploadWidth;
    int mipHeight = uploadHeight;
    for (UINT mipIndex = 0; mipIndex < texture.mipLevels; ++mipIndex)
    {
        const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& layout = layouts[mipIndex];
        const std::vector<std::uint8_t>& mipPixels = mipChain[mipIndex];
        const std::size_t srcRowPitch = static_cast<std::size_t>(mipWidth) * 4u;
        const UINT rowsToCopy = std::min(numRows[mipIndex], static_cast<UINT>(mipHeight));

        for (UINT row = 0; row < rowsToCopy; ++row)
        {
            const std::size_t srcOffset = static_cast<std::size_t>(row) * srcRowPitch;
            const std::size_t dstOffset = static_cast<std::size_t>(layout.Offset) +
                                          static_cast<std::size_t>(row) * static_cast<std::size_t>(layout.Footprint.RowPitch);
            std::memcpy(mapped + dstOffset, mipPixels.data() + srcOffset, srcRowPitch);
        }

        mipWidth = std::max(1, mipWidth / 2);
        mipHeight = std::max(1, mipHeight / 2);
    }
    uploadBuffer->Unmap(0, nullptr);

    throwIfFailed(uploadCommandAllocator_->Reset(), "failed to reset upload allocator");
    throwIfFailed(commandList_->Reset(uploadCommandAllocator_.Get(), nullptr),
                  "failed to reset upload command list");

    for (UINT mipIndex = 0; mipIndex < texture.mipLevels; ++mipIndex)
    {
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = texture.resource.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = mipIndex;

        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = uploadBuffer.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = layouts[mipIndex];

        commandList_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    const D3D12_RESOURCE_BARRIER barrier =
        transitionBarrier(texture.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    commandList_->ResourceBarrier(1, &barrier);
    throwIfFailed(commandList_->Close(), "failed to close upload command list");
    ID3D12CommandList* commandLists[] = {commandList_.Get()};
    commandQueue_->ExecuteCommandLists(static_cast<UINT>(std::size(commandLists)), commandLists);
    waitForGpu();

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = texture.mipLevels;
    device_->CreateShaderResourceView(texture.resource.Get(), &srvDesc, texture.srvCpu);

    stbi_image_free(pixels);
    return texture;
}

void Renderer::requestScreenshot(const std::filesystem::path& path)
{
    if (path.empty())
    {
        throwRenderError("screenshot path must not be empty");
    }

    if (path.has_parent_path())
    {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec)
        {
            throwRenderError("failed to create screenshot directory");
        }
    }

    pendingScreenshotPath_ = path;
    screenshotRequested_ = true;
}

void Renderer::ensureScreenshotReadbackBuffer()
{
    ID3D12Resource* backBuffer = renderTargets_[currentBackBufferIndex_].Get();
    if (backBuffer == nullptr)
    {
        throwRenderError("backbuffer unavailable for screenshot capture");
    }

    const D3D12_RESOURCE_DESC backBufferDesc = backBuffer->GetDesc();
    UINT64 requiredSize = 0;
    UINT numRows = 0;
    UINT64 rowSizeInBytes = 0;
    device_->GetCopyableFootprints(&backBufferDesc,
                                   0,
                                   1,
                                   0,
                                   &screenshotReadbackLayout_,
                                   &numRows,
                                   &rowSizeInBytes,
                                   &requiredSize);

    if (!screenshotReadbackBuffer_ || screenshotReadbackBufferSize_ < requiredSize)
    {
        screenshotReadbackBuffer_.Reset();
        const D3D12_HEAP_PROPERTIES readbackHeap = heapProps(D3D12_HEAP_TYPE_READBACK);
        const D3D12_RESOURCE_DESC readbackDesc = bufferDesc(requiredSize);
        throwIfFailed(device_->CreateCommittedResource(&readbackHeap,
                                                       D3D12_HEAP_FLAG_NONE,
                                                       &readbackDesc,
                                                       D3D12_RESOURCE_STATE_COPY_DEST,
                                                       nullptr,
                                                       IID_PPV_ARGS(&screenshotReadbackBuffer_)),
                      "failed to create screenshot readback buffer");
        screenshotReadbackBufferSize_ = requiredSize;
    }
}

void Renderer::writePendingScreenshot(const std::filesystem::path& path)
{
    if (!screenshotReadbackBuffer_)
    {
        throwRenderError("screenshot readback buffer unavailable");
    }

#pragma pack(push, 1)
    struct BmpFileHeader
    {
        std::uint16_t type{0x4D42};
        std::uint32_t size{0};
        std::uint16_t reserved1{0};
        std::uint16_t reserved2{0};
        std::uint32_t offset{0};
    };

    struct BmpInfoHeader
    {
        std::uint32_t size{40};
        std::int32_t width{0};
        std::int32_t height{0};
        std::uint16_t planes{1};
        std::uint16_t bitCount{32};
        std::uint32_t compression{0};
        std::uint32_t sizeImage{0};
        std::int32_t xPixelsPerMeter{2835};
        std::int32_t yPixelsPerMeter{2835};
        std::uint32_t colorsUsed{0};
        std::uint32_t colorsImportant{0};
    };
#pragma pack(pop)

    const UINT width = static_cast<UINT>(std::max(width_, 1));
    const UINT height = static_cast<UINT>(std::max(height_, 1));
    const std::size_t dstRowSize = static_cast<std::size_t>(width) * 4u;
    const std::size_t pixelBytes = dstRowSize * static_cast<std::size_t>(height);

    BmpFileHeader fileHeader;
    fileHeader.offset = static_cast<std::uint32_t>(sizeof(BmpFileHeader) + sizeof(BmpInfoHeader));
    fileHeader.size = fileHeader.offset + static_cast<std::uint32_t>(pixelBytes);

    BmpInfoHeader infoHeader;
    infoHeader.width = static_cast<std::int32_t>(width);
    infoHeader.height = static_cast<std::int32_t>(height);
    infoHeader.sizeImage = static_cast<std::uint32_t>(pixelBytes);

    D3D12_RANGE readRange{};
    readRange.Begin = 0;
    readRange.End = static_cast<SIZE_T>(screenshotReadbackLayout_.Offset +
                                        screenshotReadbackLayout_.Footprint.RowPitch * height);
    std::byte* mapped = nullptr;
    throwIfFailed(screenshotReadbackBuffer_->Map(0, &readRange, reinterpret_cast<void**>(&mapped)),
                  "failed to map screenshot readback buffer");

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        screenshotReadbackBuffer_->Unmap(0, nullptr);
        throwRenderError("failed to open screenshot output file");
    }

    out.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    out.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    std::vector<std::byte> row(dstRowSize);
    for (int y = static_cast<int>(height) - 1; y >= 0; --y)
    {
        const std::byte* srcRow = mapped +
                                  screenshotReadbackLayout_.Offset +
                                  static_cast<std::size_t>(y) * screenshotReadbackLayout_.Footprint.RowPitch;
        for (UINT x = 0; x < width; ++x)
        {
            const std::size_t srcIndex = static_cast<std::size_t>(x) * 4u;
            const std::size_t dstIndex = static_cast<std::size_t>(x) * 4u;
            row[dstIndex + 0] = srcRow[srcIndex + 2];
            row[dstIndex + 1] = srcRow[srcIndex + 1];
            row[dstIndex + 2] = srcRow[srcIndex + 0];
            row[dstIndex + 3] = srcRow[srcIndex + 3];
        }
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
    }

    screenshotReadbackBuffer_->Unmap(0, nullptr);

    if (!out)
    {
        throwRenderError("failed while writing screenshot output");
    }
}

void Renderer::buildDepthPyramid()
{
    if (!depthBuffer_ ||
        !depthPyramid_ ||
        depthPyramidMipCount_ == 0 ||
        !depthPyramidRootSignature_ ||
        !depthPyramidPipelineState_)
    {
        return;
    }

    renderDebugLog("buildDepthPyramid: begin");

    const D3D12_RESOURCE_BARRIER depthBeginBarrier =
        transitionBarrier(depthBuffer_.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    commandList_->ResourceBarrier(1, &depthBeginBarrier);

    commandList_->SetComputeRootSignature(depthPyramidRootSignature_.Get());
    commandList_->SetPipelineState(depthPyramidPipelineState_.Get());

    struct BuildParams
    {
        UINT srcMip;
        UINT srcWidth;
        UINT srcHeight;
        UINT dstWidth;
        UINT dstHeight;
    };

    for (UINT mipIndex = 0; mipIndex < depthPyramidMipCount_; ++mipIndex)
    {
        const D3D12_RESOURCE_BARRIER mipBeginBarrier =
            transitionBarrier(depthPyramid_.Get(),
                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              mipIndex);
        commandList_->ResourceBarrier(1, &mipBeginBarrier);

        const UINT srcWidth = std::max(1u, static_cast<UINT>(width_) >> ((mipIndex == 0) ? 0u : (mipIndex - 1u)));
        const UINT srcHeight = std::max(1u, static_cast<UINT>(height_) >> ((mipIndex == 0) ? 0u : (mipIndex - 1u)));
        const UINT dstWidth = std::max(1u, static_cast<UINT>(width_) >> mipIndex);
        const UINT dstHeight = std::max(1u, static_cast<UINT>(height_) >> mipIndex);
        const BuildParams params{
            (mipIndex == 0) ? 0u : (mipIndex - 1u),
            srcWidth,
            srcHeight,
            dstWidth,
            dstHeight};

        commandList_->SetComputeRoot32BitConstants(0, 5, &params, 0);
        commandList_->SetComputeRootDescriptorTable(1, (mipIndex == 0) ? depthSrvGpu_ : depthPyramidSrvGpu_);
        commandList_->SetComputeRootDescriptorTable(2, depthPyramidUavGpuHandles_[mipIndex]);
        commandList_->Dispatch((dstWidth + 7u) / 8u, (dstHeight + 7u) / 8u, 1u);

        std::array<D3D12_RESOURCE_BARRIER, 2> mipEndBarriers{};
        mipEndBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        mipEndBarriers[0].UAV.pResource = depthPyramid_.Get();
        mipEndBarriers[1] = transitionBarrier(depthPyramid_.Get(),
                                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                              mipIndex);
        commandList_->ResourceBarrier(static_cast<UINT>(mipEndBarriers.size()), mipEndBarriers.data());
    }

    const D3D12_RESOURCE_BARRIER depthEndBarrier =
        transitionBarrier(depthBuffer_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    commandList_->ResourceBarrier(1, &depthEndBarrier);
    depthPyramidState_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    renderDebugLog("buildDepthPyramid: end");
}

void Renderer::renderFarBatchGpuCull(const ChunkRenderBatch& batch, const glm::mat4& viewProj)
{
    if (batch.gpuCullRecords.empty() ||
        !depthPyramid_ ||
        !lodCullRootSignature_ ||
        !lodCullPipelineState_ ||
        !lodIndirectRootSignature_ ||
        !lodIndirectPipelineState_)
    {
        return;
    }

    FrameResource& frame = frameResources_[currentBackBufferIndex_];
    renderDebugLog("renderFarBatchGpuCull: begin");
    const std::uint64_t recordCount = static_cast<std::uint64_t>(batch.gpuCullRecords.size());
    const std::uint64_t recordBytes = recordCount * sizeof(ChunkRenderBatch::GpuCullRecord);
    const std::uint64_t visibleIndexBytes = std::max<std::uint64_t>(recordCount * sizeof(std::uint32_t), 4u);
    const std::uint64_t countBytes = sizeof(std::uint32_t);
    const std::uint64_t indirectBytes = std::max<std::uint64_t>(recordCount * sizeof(D3D12_DRAW_INDEXED_ARGUMENTS), 4u);

    auto createBuffer = [this](D3D12_HEAP_TYPE heapType,
                               std::uint64_t sizeInBytes,
                               D3D12_RESOURCE_STATES initialState,
                               D3D12_RESOURCE_FLAGS flags) -> Microsoft::WRL::ComPtr<ID3D12Resource>
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        const D3D12_HEAP_PROPERTIES heap = heapProps(heapType);
        D3D12_RESOURCE_DESC desc = bufferDesc(std::max<std::uint64_t>(sizeInBytes, 4u));
        desc.Flags = flags;
        throwIfFailed(device_->CreateCommittedResource(&heap,
                                                       D3D12_HEAP_FLAG_NONE,
                                                       &desc,
                                                       initialState,
                                                       nullptr,
                                                       IID_PPV_ARGS(&resource)),
                      "failed to create transient far cull buffer");
        return resource;
    };

    Microsoft::WRL::ComPtr<ID3D12Resource> recordsDefault =
        createBuffer(D3D12_HEAP_TYPE_DEFAULT, recordBytes, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
    Microsoft::WRL::ComPtr<ID3D12Resource> recordsUpload =
        createBuffer(D3D12_HEAP_TYPE_UPLOAD, recordBytes, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    Microsoft::WRL::ComPtr<ID3D12Resource> visibleIndices =
        createBuffer(D3D12_HEAP_TYPE_DEFAULT,
                     visibleIndexBytes,
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    Microsoft::WRL::ComPtr<ID3D12Resource> visibleCount =
        createBuffer(D3D12_HEAP_TYPE_DEFAULT,
                     countBytes,
                     D3D12_RESOURCE_STATE_COPY_DEST,
                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    Microsoft::WRL::ComPtr<ID3D12Resource> countUpload =
        createBuffer(D3D12_HEAP_TYPE_UPLOAD, countBytes, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    Microsoft::WRL::ComPtr<ID3D12Resource> indirectArgs =
        createBuffer(D3D12_HEAP_TYPE_DEFAULT,
                     indirectBytes,
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    frame.transientResources.push_back(recordsDefault);
    frame.transientResources.push_back(recordsUpload);
    frame.transientResources.push_back(visibleIndices);
    frame.transientResources.push_back(visibleCount);
    frame.transientResources.push_back(countUpload);
    frame.transientResources.push_back(indirectArgs);

    void* mappedRecords = nullptr;
    throwIfFailed(recordsUpload->Map(0, nullptr, &mappedRecords), "failed to map far cull upload buffer");
    std::memcpy(mappedRecords, batch.gpuCullRecords.data(), static_cast<std::size_t>(recordBytes));
    recordsUpload->Unmap(0, nullptr);

    void* mappedCount = nullptr;
    throwIfFailed(countUpload->Map(0, nullptr, &mappedCount), "failed to map far cull count upload");
    const std::uint32_t zeroValue = 0;
    std::memcpy(mappedCount, &zeroValue, sizeof(zeroValue));
    countUpload->Unmap(0, nullptr);

    commandList_->CopyBufferRegion(recordsDefault.Get(), 0, recordsUpload.Get(), 0, recordBytes);
    commandList_->CopyBufferRegion(visibleCount.Get(), 0, countUpload.Get(), 0, countBytes);

    const D3D12_RESOURCE_BARRIER setupBarriers[] = {
        transitionBarrier(recordsDefault.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        transitionBarrier(visibleCount.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
    commandList_->ResourceBarrier(static_cast<UINT>(std::size(setupBarriers)), setupBarriers);

    struct CullRootConstants
    {
        glm::mat4 viewProj{1.0f};
        glm::vec4 frustumPlanes[6]{};
        UINT recordCount{0};
        UINT depthWidth{0};
        UINT depthHeight{0};
        UINT depthMipCount{0};
    } constants{};
    constants.viewProj = viewProj;
    const std::array<FrustumPlane, 6> frustumPlanes = extractFrustumPlanes(viewProj);
    for (std::size_t i = 0; i < frustumPlanes.size(); ++i)
    {
        constants.frustumPlanes[i] = frustumPlanes[i].equation;
    }
    constants.recordCount = static_cast<UINT>(recordCount);
    constants.depthWidth = static_cast<UINT>(std::max(width_, 1));
    constants.depthHeight = static_cast<UINT>(std::max(height_, 1));
    constants.depthMipCount = depthPyramidMipCount_;

    const UINT dispatchGroups = static_cast<UINT>((recordCount + 63u) / 64u);
    const auto cullStart = std::chrono::steady_clock::now();
    renderDebugLog("renderFarBatchGpuCull: dispatch cull");
    commandList_->SetComputeRootSignature(lodCullRootSignature_.Get());
    commandList_->SetPipelineState(lodCullPipelineState_.Get());
    commandList_->SetComputeRoot32BitConstants(0, 44, &constants, 0);
    commandList_->SetComputeRootDescriptorTable(1, depthPyramidSrvGpu_);
    commandList_->SetComputeRootShaderResourceView(2, recordsDefault->GetGPUVirtualAddress());
    commandList_->SetComputeRootUnorderedAccessView(3, visibleIndices->GetGPUVirtualAddress());
    commandList_->SetComputeRootUnorderedAccessView(4, visibleCount->GetGPUVirtualAddress());
    commandList_->Dispatch(dispatchGroups, 1, 1);
    profilingSnapshot_.lodGpuCullMs +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cullStart).count();
    renderDebugLog("renderFarBatchGpuCull: cull complete");

    std::array<D3D12_RESOURCE_BARRIER, 3> cullBarriers{};
    cullBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    cullBarriers[0].UAV.pResource = visibleIndices.Get();
    cullBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    cullBarriers[1].UAV.pResource = visibleCount.Get();
    cullBarriers[2] = transitionBarrier(visibleIndices.Get(),
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    commandList_->ResourceBarrier(static_cast<UINT>(cullBarriers.size()), cullBarriers.data());

    const auto indirectStart = std::chrono::steady_clock::now();
    renderDebugLog("renderFarBatchGpuCull: dispatch indirect build");
    commandList_->SetComputeRootSignature(lodIndirectRootSignature_.Get());
    commandList_->SetPipelineState(lodIndirectPipelineState_.Get());
    commandList_->SetComputeRootShaderResourceView(0, recordsDefault->GetGPUVirtualAddress());
    commandList_->SetComputeRootShaderResourceView(1, visibleIndices->GetGPUVirtualAddress());
    commandList_->SetComputeRootUnorderedAccessView(2, indirectArgs->GetGPUVirtualAddress());
    commandList_->SetComputeRootUnorderedAccessView(3, visibleCount->GetGPUVirtualAddress());
    commandList_->Dispatch(dispatchGroups, 1, 1);
    profilingSnapshot_.lodIndirectBuildMs +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - indirectStart).count();
    renderDebugLog("renderFarBatchGpuCull: indirect build complete");

    std::array<D3D12_RESOURCE_BARRIER, 4> indirectBarriers{};
    indirectBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    indirectBarriers[0].UAV.pResource = indirectArgs.Get();
    indirectBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    indirectBarriers[1].UAV.pResource = visibleCount.Get();
    indirectBarriers[2] = transitionBarrier(indirectArgs.Get(),
                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    indirectBarriers[3] = transitionBarrier(visibleCount.Get(),
                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    commandList_->ResourceBarrier(static_cast<UINT>(indirectBarriers.size()), indirectBarriers.data());

    commandList_->IASetVertexBuffers(0, 1, &batch.vertexBufferView);
    commandList_->IASetIndexBuffer(&batch.indexBufferView);
    renderDebugLog("renderFarBatchGpuCull: execute indirect");
    commandList_->ExecuteIndirect(drawIndexedCommandSignature_.Get(),
                                  static_cast<UINT>(recordCount),
                                  indirectArgs.Get(),
                                  0,
                                  visibleCount.Get(),
                                  0);
    renderDebugLog("renderFarBatchGpuCull: end");
}

std::string Renderer::collectDebugMessages() const
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

        oss << "\n  [" << i << "] " << message->pDescription;
    }

    infoQueue_->ClearStoredMessages();
    return oss.str();
}

void Renderer::ensureFrameStarted() const
{
    if (!frameStarted_)
    {
        throwRenderError("frame commands requested before beginFrame()");
    }
}

std::string Renderer::shaderPath(const char* relativePath) const
{
    return (std::filesystem::path("assets") / "shaders" / relativePath).string();
}

std::uint64_t Renderer::allocateFrameConstantBytes(std::size_t size, void** cpuPtrOut)
{
    ensureFrameStarted();
    const std::size_t alignedSize = (size + 255ull) & ~255ull;
    if (currentFrameConstantOffset_ + alignedSize > kFrameConstantBufferSize)
    {
        throwRenderError("frame constant buffer exhausted");
    }

    FrameResource& frame = frameResources_[currentBackBufferIndex_];
    if (cpuPtrOut != nullptr)
    {
        *cpuPtrOut = frame.mappedConstants + currentFrameConstantOffset_;
    }
    const std::uint64_t gpuAddress = frame.constantBuffer->GetGPUVirtualAddress() + currentFrameConstantOffset_;
    currentFrameConstantOffset_ += alignedSize;
    return gpuAddress;
}

void Renderer::beginFrame(const glm::vec4& clearColor)
{
    if (frameStarted_)
    {
        throwRenderError("beginFrame() called while a frame is already open");
    }

    currentBackBufferIndex_ = swapChain_->GetCurrentBackBufferIndex();
    FrameResource& frame = frameResources_[currentBackBufferIndex_];
    if (frame.fenceValue != 0 && fence_->GetCompletedValue() < frame.fenceValue)
    {
        throwIfFailed(fence_->SetEventOnCompletion(frame.fenceValue, fenceEvent_), "failed to wait for frame fence");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }
    frame.transientResources.clear();

    throwIfFailed(frame.allocator->Reset(), "failed to reset frame allocator");
    throwIfFailed(commandList_->Reset(frame.allocator.Get(), nullptr), "failed to reset command list");
    if (infoQueue_ != nullptr)
    {
        infoQueue_->ClearStoredMessages();
    }

    if (pendingUploadFence_ != nullptr &&
        pendingUploadFenceValue_ > consumedUploadFenceValue_ &&
        pendingUploadFence_->GetCompletedValue() < pendingUploadFenceValue_)
    {
        throwIfFailed(commandQueue_->Wait(pendingUploadFence_, pendingUploadFenceValue_),
                      "failed to wait for chunk upload fence");
    }
    consumedUploadFenceValue_ = std::max(consumedUploadFenceValue_, pendingUploadFenceValue_);

    const D3D12_RESOURCE_BARRIER backbufferBarrier =
        transitionBarrier(renderTargets_[currentBackBufferIndex_].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    commandList_->ResourceBarrier(1, &backbufferBarrier);

    if (sceneColor_ != nullptr && sceneColorState_ != D3D12_RESOURCE_STATE_RENDER_TARGET)
    {
        const D3D12_RESOURCE_BARRIER sceneBarrier =
            transitionBarrier(sceneColor_.Get(), sceneColorState_, D3D12_RESOURCE_STATE_RENDER_TARGET);
        commandList_->ResourceBarrier(1, &sceneBarrier);
        sceneColorState_ = D3D12_RESOURCE_STATE_RENDER_TARGET;
    }

    const D3D12_CPU_DESCRIPTOR_HANDLE backbufferRtv =
        rtvHandleAt(rtvHeap_.Get(), rtvDescriptorSize_, currentBackBufferIndex_);
    const D3D12_CPU_DESCRIPTOR_HANDLE depthHandle = depthDsv_;
    commandList_->OMSetRenderTargets(1, &backbufferRtv, FALSE, &depthHandle);
    commandList_->RSSetViewports(1, &viewport_);
    commandList_->RSSetScissorRects(1, &scissorRect_);
    commandList_->ClearRenderTargetView(backbufferRtv, &clearColor.x, 0, nullptr);
    if (sceneColor_ != nullptr)
    {
        commandList_->ClearRenderTargetView(sceneColorRtv_, &clearColor.x, 0, nullptr);
    }
    commandList_->ClearDepthStencilView(depthHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    ID3D12DescriptorHeap* heaps[] = {srvHeap_.Get()};
    commandList_->SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);

    currentFrameConstantOffset_ = 0;
    profilingSnapshot_ = {};
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
                           const glm::mat4& view,
                           const glm::mat4& proj,
                           const glm::vec3& cameraPos,
                           const LoadedTexture& atlasTexture,
                           const EnvironmentState& environment)
{
    ensureFrameStarted();
    if (!atlasTexture.valid())
    {
        return;
    }

    void* worldCpu = nullptr;
    const std::uint64_t worldCb = allocateFrameConstantBytes(sizeof(WorldConstants), &worldCpu);
    auto* nearConstants = static_cast<WorldConstants*>(worldCpu);
    const glm::mat4 viewProj = proj * view;
    const glm::vec3 lightDir = glm::normalize(environment.sunDirection);
    const float daylight = std::clamp(lightDir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    const glm::vec3 baseSkyTopColor = glm::vec3(0x78 / 255.0f, 0xA7 / 255.0f, 1.0f);
    const glm::vec3 baseSkyHorizonColor = glm::vec3(0xBB / 255.0f, 0xD4 / 255.0f, 1.0f);
    const glm::vec3 cloudTopColor = glm::vec3(0.96f, 0.97f, 1.0f);
    const glm::vec3 cloudBottomColor = glm::vec3(0.82f, 0.87f, 0.96f);
    // The enhanced atmosphere path is intentionally optional. The default visual target
    // for terrain work is the base-game look with this path disabled at startup.
    const bool skyPassEnabled = environment.atmosphereEnabled && environment.debug.skyPassEnabled;
    const bool aerialPerspectiveEnabled =
        environment.atmosphereEnabled && environment.debug.aerialPerspectiveEnabled;
    const float fogStartBlocks =
        environment.debug.fogFallbackEnabled ? environment.fogStartBlocks : environment.farDistanceBlocks;
    const glm::vec3 sunColor = environment.sunIlluminance * 0.18f;
    const glm::vec3 skyAmbient = glm::mix(baseSkyTopColor * 0.16f,
                                          baseSkyTopColor * 1.05f,
                                          std::pow(daylight, 0.55f));
    const glm::vec3 groundAmbient = glm::mix(glm::vec3(0.025f, 0.022f, 0.020f),
                                             glm::vec3(0.08f, 0.075f, 0.065f),
                                             std::pow(daylight, 0.8f));

    nearConstants->viewProj = viewProj;
    nearConstants->shadowViewProj = glm::mat4(1.0f);
    nearConstants->lightDirection = glm::vec4(lightDir, 0.0f);
    nearConstants->cameraPos = glm::vec4(cameraPos, 0.0f);
    nearConstants->highlightedBlock = glm::vec4(glm::vec3(renderData.highlightedBlock), 0.0f);
    nearConstants->params0 = glm::vec4(aerialPerspectiveEnabled ? 1.0f : 0.0f,
                                       renderData.hasHighlight ? 1.0f : 0.0f,
                                       1.0f / static_cast<float>(std::max(width_, 1)),
                                       1.0f / static_cast<float>(std::max(height_, 1)));
    nearConstants->params1 = glm::vec4(environment.atmosphere.aerialPerspectiveDistanceKm,
                                       static_cast<float>(kAerialPerspectiveSliceCount),
                                       fogStartBlocks,
                                       environment.farDistanceBlocks);
    nearConstants->sunColor = glm::vec4(sunColor, 0.0f);
    nearConstants->skyAmbient = glm::vec4(skyAmbient, 0.0f);
    nearConstants->groundAmbient = glm::vec4(groundAmbient, 0.0f);
    nearConstants->shadowParams = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
    nearConstants->terrainDebug = glm::vec4(environment.debug.directSunEnabled ? 1.0f : 0.0f,
                                            static_cast<float>(static_cast<int>(environment.debug.terrainDebugView)),
                                            0.0f,
                                            0.0f);

    if (environment.debug.shadowsEnabled)
    {
        renderShadowMap(renderData, view, cameraPos, environment, *nearConstants);
    }

    if ((skyPassEnabled || aerialPerspectiveEnabled) && atmosphere_)
    {
        atmosphere_->update(*this, environment, view, proj, cameraPos);
    }

    if (skyPassEnabled && atmosphere_)
    {
        atmosphere_->renderSky(*this, environment, view, proj, cameraPos);
    }
    else
    {
        void* skyCpu = nullptr;
        const std::uint64_t skyCb = allocateFrameConstantBytes(sizeof(BaseSkyConstants), &skyCpu);
        auto* skyConstants = static_cast<BaseSkyConstants*>(skyCpu);
        skyConstants->topSkyColor = glm::vec4(baseSkyTopColor, 0.0f);
        skyConstants->horizonSkyColor = glm::vec4(baseSkyHorizonColor, 0.0f);
        skyConstants->params = glm::vec4(daylight, 0.0f, 0.0f, 0.0f);
        skyConstants->sunColor = glm::vec4(sunColor, 0.0f);

        const D3D12_CPU_DESCRIPTOR_HANDLE depthHandle = depthDsv_;
        commandList_->OMSetRenderTargets(1, &sceneColorRtv_, FALSE, &depthHandle);
        commandList_->RSSetViewports(1, &viewport_);
        commandList_->RSSetScissorRects(1, &scissorRect_);
        commandList_->SetGraphicsRootSignature(fullscreenRootSignature_.Get());
        commandList_->SetPipelineState(baseSkyPipelineState_.Get());
        commandList_->SetGraphicsRootConstantBufferView(0, skyCb);
        commandList_->SetGraphicsRootDescriptorTable(1, sceneColorSrvGpu_);
        commandList_->SetGraphicsRootDescriptorTable(2, sceneColorSrvGpu_);
        commandList_->SetGraphicsRootDescriptorTable(3, sceneColorSrvGpu_);
        commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList_->DrawInstanced(3, 1, 0, 0);
    }

    const auto worldStart = std::chrono::steady_clock::now();
    const D3D12_CPU_DESCRIPTOR_HANDLE depthHandle = depthDsv_;
    commandList_->OMSetRenderTargets(1, &sceneColorRtv_, FALSE, &depthHandle);
    commandList_->RSSetViewports(1, &viewport_);
    commandList_->RSSetScissorRects(1, &scissorRect_);
    commandList_->SetGraphicsRootSignature(worldRootSignature_.Get());
    commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    commandList_->SetGraphicsRootDescriptorTable(1, atlasTexture.srvGpu);
    commandList_->SetGraphicsRootDescriptorTable(2, atmosphere_ ? atmosphere_->aerialPerspectiveSrv() : sceneColorSrvGpu_);
    commandList_->SetGraphicsRootDescriptorTable(3, shadowMapSrvGpu_);

    if (environment.debug.worldPassEnabled)
    {
        commandList_->SetPipelineState(nearPipelineState_.Get());
        commandList_->SetGraphicsRootConstantBufferView(0, worldCb);
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

        const bool gpuFarCullDisabled = std::getenv("BLOCKGAME_DISABLE_LOD_GPU_CULL") != nullptr;
        bool shouldUseGpuFarCull = false;
        for (const ChunkRenderBatch& batch : renderData.farBatches)
        {
            if (!gpuFarCullDisabled && batch.supportsGpuCull && !batch.gpuCullRecords.empty())
            {
                shouldUseGpuFarCull = true;
                break;
            }
        }
        if (shouldUseGpuFarCull)
        {
            buildDepthPyramid();
        }

        void* farCpu = nullptr;
        const std::uint64_t farCb = allocateFrameConstantBytes(sizeof(WorldConstants), &farCpu);
        auto* farConstants = static_cast<WorldConstants*>(farCpu);
        *farConstants = *nearConstants;
        farConstants->shadowParams.w = 0.0f;
        commandList_->SetPipelineState(farPipelineState_.Get());
        commandList_->SetGraphicsRootConstantBufferView(0, farCb);
        for (const ChunkRenderBatch& batch : renderData.farBatches)
        {
            if (!gpuFarCullDisabled && batch.supportsGpuCull && !batch.gpuCullRecords.empty())
            {
                renderFarBatchGpuCull(batch, viewProj);
                continue;
            }

            if (batch.indexCounts.empty())
            {
                continue;
            }

            commandList_->IASetVertexBuffers(0, 1, &batch.vertexBufferView);
            commandList_->IASetIndexBuffer(&batch.indexBufferView);
            void* indirectCpu = nullptr;
            const std::size_t commandCount = batch.indexCounts.size();
            const std::size_t commandBytes = commandCount * sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
            const std::uint64_t indirectGpuAddress = allocateFrameConstantBytes(commandBytes, &indirectCpu);
            auto* indirectArgs = static_cast<D3D12_DRAW_INDEXED_ARGUMENTS*>(indirectCpu);
            for (std::size_t i = 0; i < commandCount; ++i)
            {
                indirectArgs[i].IndexCountPerInstance = batch.indexCounts[i];
                indirectArgs[i].InstanceCount = 1;
                indirectArgs[i].StartIndexLocation = batch.firstIndexLocations[i];
                indirectArgs[i].BaseVertexLocation = batch.baseVertices[i];
                indirectArgs[i].StartInstanceLocation = 0;
            }

            FrameResource& frame = frameResources_[currentBackBufferIndex_];
            const std::uint64_t indirectBufferOffset =
                indirectGpuAddress - frame.constantBuffer->GetGPUVirtualAddress();
            commandList_->ExecuteIndirect(drawIndexedCommandSignature_.Get(),
                                          static_cast<UINT>(commandCount),
                                          frame.constantBuffer.Get(),
                                          indirectBufferOffset,
                                          nullptr,
                                          0);
        }
    }
    profilingSnapshot_.worldDrawMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - worldStart).count();

    if (environment.debug.skyPassEnabled)
    {
        void* cloudCpu = nullptr;
        const std::uint64_t cloudCb = allocateFrameConstantBytes(sizeof(CloudConstants), &cloudCpu);
        auto* cloudConstants = static_cast<CloudConstants*>(cloudCpu);
        cloudConstants->viewProj = viewProj;
        cloudConstants->cameraPosTime = glm::vec4(cameraPos, static_cast<float>(glfwGetTime()));
        cloudConstants->layerParams = glm::vec4(300.0f, 4.0f, 56.0f, 8.0f);
        cloudConstants->shapeParams = glm::vec4(0.57f, 1.55f, 0.0f, 0.0f);
        cloudConstants->topColor = glm::vec4(cloudTopColor, 0.82f);
        cloudConstants->bottomColor = glm::vec4(cloudBottomColor, 0.72f);

        commandList_->OMSetRenderTargets(1, &sceneColorRtv_, FALSE, &depthHandle);
        commandList_->RSSetViewports(1, &viewport_);
        commandList_->RSSetScissorRects(1, &scissorRect_);
        commandList_->SetGraphicsRootSignature(fullscreenRootSignature_.Get());
        commandList_->SetPipelineState(cloudPipelineState_.Get());
        commandList_->SetGraphicsRootConstantBufferView(0, cloudCb);
        commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        constexpr UINT kCloudPrismCount = 17u * 17u * 3u;
        commandList_->DrawInstanced(36, kCloudPrismCount, 0, 0);
    }

    if (sceneColorState_ != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
    {
        const D3D12_RESOURCE_BARRIER barrier =
            transitionBarrier(sceneColor_.Get(), sceneColorState_, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        commandList_->ResourceBarrier(1, &barrier);
        sceneColorState_ = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    }

    const auto toneMapStart = std::chrono::steady_clock::now();
    const D3D12_CPU_DESCRIPTOR_HANDLE backbufferRtv =
        rtvHandleAt(rtvHeap_.Get(), rtvDescriptorSize_, currentBackBufferIndex_);
    commandList_->OMSetRenderTargets(1, &backbufferRtv, FALSE, nullptr);
    commandList_->RSSetViewports(1, &viewport_);
    commandList_->RSSetScissorRects(1, &scissorRect_);
    commandList_->SetGraphicsRootSignature(fullscreenRootSignature_.Get());
    commandList_->SetPipelineState(toneMapPipelineState_.Get());

    void* toneCpu = nullptr;
    const std::uint64_t toneCb = allocateFrameConstantBytes(sizeof(ToneMapConstants), &toneCpu);
    auto* toneConstants = static_cast<ToneMapConstants*>(toneCpu);
    toneConstants->exposureWhitePoint = glm::vec4(environment.tonemap.exposure, environment.tonemap.whitePoint, 0.0f, 0.0f);
    commandList_->SetGraphicsRootConstantBufferView(0, toneCb);
    commandList_->SetGraphicsRootDescriptorTable(1, sceneColorSrvGpu_);
    commandList_->SetGraphicsRootDescriptorTable(2, atmosphere_ ? atmosphere_->aerialPerspectiveSrv() : sceneColorSrvGpu_);
    commandList_->SetGraphicsRootDescriptorTable(3, atmosphere_ ? atmosphere_->aerialPerspectiveSrv() : sceneColorSrvGpu_);
    commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    commandList_->DrawInstanced(3, 1, 0, 0);
    profilingSnapshot_.toneMapMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - toneMapStart).count();
}

void Renderer::renderShadowMap(const WorldRenderData& renderData,
                               const glm::mat4& view,
                               const glm::vec3& cameraPos,
                               const EnvironmentState& environment,
                               WorldConstants& nearConstants)
{
    nearConstants.shadowViewProj = glm::mat4(1.0f);
    nearConstants.shadowParams = glm::vec4(1.0f / static_cast<float>(kShadowMapResolution),
                                           0.35f,
                                           1.0f,
                                           0.0f);

    if (!shadowMap_ || renderData.nearBatches.empty())
    {
        return;
    }

    const glm::vec3 lightDir = glm::normalize(environment.sunDirection);
    if (lightDir.y <= 0.02f)
    {
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    if (shadowMapState_ != D3D12_RESOURCE_STATE_DEPTH_WRITE)
    {
        const D3D12_RESOURCE_BARRIER barrier =
            transitionBarrier(shadowMap_.Get(), shadowMapState_, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        commandList_->ResourceBarrier(1, &barrier);
        shadowMapState_ = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    }

    D3D12_VIEWPORT shadowViewport{};
    shadowViewport.Width = static_cast<float>(kShadowMapResolution);
    shadowViewport.Height = static_cast<float>(kShadowMapResolution);
    shadowViewport.MaxDepth = 1.0f;
    D3D12_RECT shadowScissor{0, 0, static_cast<LONG>(kShadowMapResolution), static_cast<LONG>(kShadowMapResolution)};

    commandList_->OMSetRenderTargets(0, nullptr, FALSE, &shadowMapDsv_);
    commandList_->RSSetViewports(1, &shadowViewport);
    commandList_->RSSetScissorRects(1, &shadowScissor);
    commandList_->ClearDepthStencilView(shadowMapDsv_, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    commandList_->SetGraphicsRootSignature(shadowRootSignature_.Get());
    commandList_->SetPipelineState(shadowPipelineState_.Get());
    commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    const glm::mat4 invView = glm::inverse(view);
    glm::vec3 cameraForward = -glm::normalize(glm::vec3(invView[2]));
    if (glm::dot(cameraForward, cameraForward) <= 1e-5f)
    {
        cameraForward = glm::vec3(0.0f, 0.0f, -1.0f);
    }

    glm::vec3 forwardFlat{cameraForward.x, 0.0f, cameraForward.z};
    if (glm::dot(forwardFlat, forwardFlat) <= 1e-5f)
    {
        forwardFlat = glm::vec3(0.0f, 0.0f, -1.0f);
    }
    else
    {
        forwardFlat = glm::normalize(forwardFlat);
    }

    constexpr float kShadowExtent = 220.0f;
    constexpr float kShadowDepth = 640.0f;
    constexpr float kShadowDistance = 320.0f;

    glm::vec3 shadowCenter = cameraPos + forwardFlat * 48.0f;
    shadowCenter.y += 24.0f;

    glm::vec3 lightUp = (std::abs(lightDir.y) > 0.95f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 lightPos = shadowCenter + lightDir * kShadowDistance;
    glm::mat4 lightView = glm::lookAtRH(lightPos, shadowCenter, lightUp);

    const float texelWorldSize = (2.0f * kShadowExtent) / static_cast<float>(kShadowMapResolution);
    glm::vec4 centerLightSpace = lightView * glm::vec4(shadowCenter, 1.0f);
    centerLightSpace.x = std::floor(centerLightSpace.x / texelWorldSize) * texelWorldSize;
    centerLightSpace.y = std::floor(centerLightSpace.y / texelWorldSize) * texelWorldSize;
    const glm::mat4 invLightView = glm::inverse(lightView);
    shadowCenter = glm::vec3(invLightView * glm::vec4(centerLightSpace.x, centerLightSpace.y, centerLightSpace.z, 1.0f));
    lightPos = shadowCenter + lightDir * kShadowDistance;
    lightView = glm::lookAtRH(lightPos, shadowCenter, lightUp);

    const glm::mat4 lightProj = glm::orthoRH_ZO(-kShadowExtent,
                                                kShadowExtent,
                                                -kShadowExtent,
                                                kShadowExtent,
                                                1.0f,
                                                kShadowDepth);
    const glm::mat4 lightViewProj = lightProj * lightView;

    void* shadowCpu = nullptr;
    const std::uint64_t shadowCbAddress = allocateFrameConstantBytes(sizeof(ShadowConstants), &shadowCpu);
    auto* shadowConstants = static_cast<ShadowConstants*>(shadowCpu);
    shadowConstants->lightViewProj = lightViewProj;
    commandList_->SetGraphicsRootConstantBufferView(0, shadowCbAddress);

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

    if (shadowMapState_ != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
    {
        const D3D12_RESOURCE_BARRIER barrier =
            transitionBarrier(shadowMap_.Get(), shadowMapState_, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        commandList_->ResourceBarrier(1, &barrier);
        shadowMapState_ = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    }

    nearConstants.shadowViewProj = lightViewProj;
    nearConstants.shadowParams = glm::vec4(1.0f / static_cast<float>(kShadowMapResolution),
                                           0.45f,
                                           1.0f,
                                           1.0f);
    profilingSnapshot_.shadowDrawMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void Renderer::endFrame()
{
    ensureFrameStarted();
    const auto endFrameStart = std::chrono::steady_clock::now();

    if (sceneColor_ != nullptr && sceneColorState_ == D3D12_RESOURCE_STATE_RENDER_TARGET)
    {
        const D3D12_RESOURCE_BARRIER barrier =
            transitionBarrier(sceneColor_.Get(), sceneColorState_, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        commandList_->ResourceBarrier(1, &barrier);
        sceneColorState_ = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    }

    if (imguiFrameStarted_)
    {
        ImGui::Render();
        commandList_->SetDescriptorHeaps(1, srvHeap_.GetAddressOf());
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), commandList_.Get());
    }

    const bool captureThisFrame = screenshotRequested_ && !pendingScreenshotPath_.empty();
    const std::filesystem::path screenshotPath = pendingScreenshotPath_;
    if (captureThisFrame)
    {
        ensureScreenshotReadbackBuffer();

        const D3D12_RESOURCE_BARRIER copyBarrier =
            transitionBarrier(renderTargets_[currentBackBufferIndex_].Get(),
                              D3D12_RESOURCE_STATE_RENDER_TARGET,
                              D3D12_RESOURCE_STATE_COPY_SOURCE);
        commandList_->ResourceBarrier(1, &copyBarrier);

        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = renderTargets_[currentBackBufferIndex_].Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = screenshotReadbackBuffer_.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = screenshotReadbackLayout_;

        commandList_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

        const D3D12_RESOURCE_BARRIER presentBarrier =
            transitionBarrier(renderTargets_[currentBackBufferIndex_].Get(),
                              D3D12_RESOURCE_STATE_COPY_SOURCE,
                              D3D12_RESOURCE_STATE_PRESENT);
        commandList_->ResourceBarrier(1, &presentBarrier);
    }
    else
    {
        const D3D12_RESOURCE_BARRIER barrier =
            transitionBarrier(renderTargets_[currentBackBufferIndex_].Get(),
                              D3D12_RESOURCE_STATE_RENDER_TARGET,
                              D3D12_RESOURCE_STATE_PRESENT);
        commandList_->ResourceBarrier(1, &barrier);
    }
    const HRESULT closeHr = commandList_->Close();
    if (FAILED(closeHr))
    {
        std::ostringstream message;
        message << "failed to close command list";
        const std::string debugMessages = collectDebugMessages();
        if (!debugMessages.empty())
        {
            message << "; D3D12 debug messages:" << debugMessages;
        }
        throwRenderError(message.str());
    }

    ID3D12CommandList* lists[] = {commandList_.Get()};
    commandQueue_->ExecuteCommandLists(static_cast<UINT>(std::size(lists)), lists);
    const auto presentStart = std::chrono::steady_clock::now();
    throwIfFailed(swapChain_->Present(1, 0), "failed to present swap chain");
    profilingSnapshot_.presentMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - presentStart).count();

    ++fenceValue_;
    throwIfFailed(commandQueue_->Signal(fence_.Get(), fenceValue_), "failed to signal frame fence");
    frameResources_[currentBackBufferIndex_].fenceValue = fenceValue_;

    if (captureThisFrame)
    {
        if (fence_->GetCompletedValue() < fenceValue_)
        {
            throwIfFailed(fence_->SetEventOnCompletion(fenceValue_, fenceEvent_),
                          "failed to wait for screenshot fence");
            WaitForSingleObject(fenceEvent_, INFINITE);
        }
        writePendingScreenshot(screenshotPath);
        screenshotRequested_ = false;
        pendingScreenshotPath_.clear();
    }

    frameStarted_ = false;
    imguiFrameStarted_ = false;
    profilingSnapshot_.endFrameMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - endFrameStart).count();
}

void Renderer::AtmosphereRenderer::ensureResources(Renderer& renderer)
{
    if (!transmittance.resource || !multiScattering.resource || !skyView.resource || !aerialPerspective.resource)
    {
        createResources(renderer);
    }
}

void Renderer::AtmosphereRenderer::createPipelines(Renderer& renderer)
{
    Microsoft::WRL::ComPtr<ID3DBlob> fullscreenVs =
        compileShaderFromFile(renderer.shaderPath("fullscreen_vs.hlsl"), "main", "vs_5_0");
    Microsoft::WRL::ComPtr<ID3DBlob> transPs =
        compileShaderFromFile(renderer.shaderPath("atmosphere_transmittance_ps.hlsl"), "main", "ps_5_0");
    Microsoft::WRL::ComPtr<ID3DBlob> multiPs =
        compileShaderFromFile(renderer.shaderPath("atmosphere_multiscattering_ps.hlsl"), "main", "ps_5_0");
    Microsoft::WRL::ComPtr<ID3DBlob> skyViewPs =
        compileShaderFromFile(renderer.shaderPath("atmosphere_skyview_ps.hlsl"), "main", "ps_5_0");
    Microsoft::WRL::ComPtr<ID3DBlob> skyPs =
        compileShaderFromFile(renderer.shaderPath("atmosphere_sky_ps.hlsl"), "main", "ps_5_0");
    Microsoft::WRL::ComPtr<ID3DBlob> aerialPs =
        compileShaderFromFile(renderer.shaderPath("atmosphere_aerial_perspective_ps.hlsl"), "main", "ps_5_0");

    auto makePso = [&](ID3DBlob* ps, DXGI_FORMAT rtvFormat) -> Microsoft::WRL::ComPtr<ID3D12PipelineState>
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = renderer.fullscreenRootSignature_.Get();
        desc.VS = {fullscreenVs->GetBufferPointer(), fullscreenVs->GetBufferSize()};
        desc.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        desc.RasterizerState.DepthClipEnable = TRUE;
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        desc.SampleMask = UINT_MAX;
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = rtvFormat;
        desc.SampleDesc.Count = 1;
        desc.DepthStencilState.DepthEnable = FALSE;
        desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;

        Microsoft::WRL::ComPtr<ID3D12PipelineState> state;
        throwIfFailed(renderer.device_->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&state)),
                      "failed to create atmosphere pipeline");
        return state;
    };

    transmittancePso = makePso(transPs.Get(), kAtmosphereFormat);
    multiScatteringPso = makePso(multiPs.Get(), kAtmosphereFormat);
    skyViewPso = makePso(skyViewPs.Get(), kAtmosphereFormat);
    skyPso = makePso(skyPs.Get(), kSceneColorFormat);
    aerialPerspectivePso = makePso(aerialPs.Get(), kAtmosphereFormat);
}

void Renderer::AtmosphereRenderer::createResources(Renderer& renderer)
{
    const D3D12_HEAP_PROPERTIES defaultHeap = heapProps(D3D12_HEAP_TYPE_DEFAULT);
    auto createLutResource = [&](LutTexture& texture,
                                 UINT width,
                                 UINT height,
                                 UINT rtvIndex,
                                 D3D12_SRV_DIMENSION srvDimension)
    {
        D3D12_CLEAR_VALUE clearValue{};
        clearValue.Format = kAtmosphereFormat;
        const D3D12_RESOURCE_DESC desc =
            texture2DDesc(kAtmosphereFormat, width, height, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        throwIfFailed(renderer.device_->CreateCommittedResource(&defaultHeap,
                                                                D3D12_HEAP_FLAG_NONE,
                                                                &desc,
                                                                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                                                                &clearValue,
                                                                IID_PPV_ARGS(&texture.resource)),
                      "failed to create atmosphere LUT texture");

        texture.rtv = rtvHandleAt(renderer.rtvHeap_.Get(), renderer.rtvDescriptorSize_, rtvIndex);
        renderer.device_->CreateRenderTargetView(texture.resource.Get(), nullptr, texture.rtv);

        texture.srvIndex = renderer.allocateSrvDescriptor();
        texture.srvCpu = renderer.srvCpuHandle(texture.srvIndex);
        texture.srvGpu = renderer.srvGpuHandle(texture.srvIndex);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = kAtmosphereFormat;
        srvDesc.ViewDimension = srvDimension;
        srvDesc.Texture2D.MipLevels = 1;
        renderer.device_->CreateShaderResourceView(texture.resource.Get(), &srvDesc, texture.srvCpu);
        texture.state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    };

    createLutResource(transmittance, 256, 64, kRtvIndexTransmittance, D3D12_SRV_DIMENSION_TEXTURE2D);
    createLutResource(multiScattering, 32, 32, kRtvIndexMultiScattering, D3D12_SRV_DIMENSION_TEXTURE2D);
    createLutResource(skyView, 256, 128, kRtvIndexSkyView, D3D12_SRV_DIMENSION_TEXTURE2D);

    D3D12_CLEAR_VALUE clearValue{};
    clearValue.Format = kAtmosphereFormat;
    const D3D12_RESOURCE_DESC aerialDesc =
        texture2DArrayDesc(kAtmosphereFormat,
                           32,
                           32,
                           kAerialPerspectiveSliceCount,
                           1,
                           D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    throwIfFailed(renderer.device_->CreateCommittedResource(&defaultHeap,
                                                            D3D12_HEAP_FLAG_NONE,
                                                            &aerialDesc,
                                                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                                                            &clearValue,
                                                            IID_PPV_ARGS(&aerialPerspective.resource)),
                  "failed to create aerial perspective texture");

    aerialPerspective.srvIndex = renderer.allocateSrvDescriptor();
    aerialPerspective.srvCpu = renderer.srvCpuHandle(aerialPerspective.srvIndex);
    aerialPerspective.srvGpu = renderer.srvGpuHandle(aerialPerspective.srvIndex);

    D3D12_SHADER_RESOURCE_VIEW_DESC aerialSrvDesc{};
    aerialSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    aerialSrvDesc.Format = kAtmosphereFormat;
    aerialSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    aerialSrvDesc.Texture2DArray.MipLevels = 1;
    aerialSrvDesc.Texture2DArray.ArraySize = kAerialPerspectiveSliceCount;
    renderer.device_->CreateShaderResourceView(aerialPerspective.resource.Get(), &aerialSrvDesc, aerialPerspective.srvCpu);

    for (UINT slice = 0; slice < kAerialPerspectiveSliceCount; ++slice)
    {
        aerialPerspective.rtvs[slice] =
            rtvHandleAt(renderer.rtvHeap_.Get(), renderer.rtvDescriptorSize_, kRtvIndexAerialPerspectiveBase + slice);
        D3D12_RENDER_TARGET_VIEW_DESC rtvDesc{};
        rtvDesc.Format = kAtmosphereFormat;
        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
        rtvDesc.Texture2DArray.FirstArraySlice = slice;
        rtvDesc.Texture2DArray.ArraySize = 1;
        renderer.device_->CreateRenderTargetView(aerialPerspective.resource.Get(), &rtvDesc, aerialPerspective.rtvs[slice]);
    }
    aerialPerspective.state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
}

void Renderer::AtmosphereRenderer::destroyResources(Renderer& renderer)
{
    auto destroyLut = [&](LutTexture& texture)
    {
        if (texture.srvIndex != (std::numeric_limits<UINT>::max)())
        {
            renderer.freeSrvDescriptor(texture.srvIndex);
            texture.srvIndex = (std::numeric_limits<UINT>::max)();
        }
        texture.resource.Reset();
        texture.srvCpu = {};
        texture.srvGpu = {};
        texture.rtv = {};
        texture.state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    };

    destroyLut(transmittance);
    destroyLut(multiScattering);
    destroyLut(skyView);
    if (aerialPerspective.srvIndex != (std::numeric_limits<UINT>::max)())
    {
        renderer.freeSrvDescriptor(aerialPerspective.srvIndex);
        aerialPerspective.srvIndex = (std::numeric_limits<UINT>::max)();
    }
    aerialPerspective.resource.Reset();
    aerialPerspective.srvCpu = {};
    aerialPerspective.srvGpu = {};
    for (auto& rtv : aerialPerspective.rtvs)
    {
        rtv = {};
    }
    aerialPerspective.state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
}

void Renderer::AtmosphereRenderer::transition(Renderer& renderer,
                                              ID3D12Resource* resource,
                                              D3D12_RESOURCE_STATES& currentState,
                                              D3D12_RESOURCE_STATES nextState)
{
    if (resource == nullptr || currentState == nextState)
    {
        return;
    }

    const D3D12_RESOURCE_BARRIER barrier = transitionBarrier(resource, currentState, nextState);
    renderer.commandList_->ResourceBarrier(1, &barrier);
    currentState = nextState;
}

std::uint64_t Renderer::AtmosphereRenderer::uploadConstants(Renderer& renderer,
                                                            const EnvironmentState& environment,
                                                            const glm::mat4& view,
                                                            const glm::mat4& proj,
                                                            const glm::vec3& cameraPos,
                                                            float sliceIndex)
{
    void* cpuMemory = nullptr;
    const std::uint64_t gpuAddress = renderer.allocateFrameConstantBytes(sizeof(AtmosphereConstants), &cpuMemory);
    auto* constants = static_cast<AtmosphereConstants*>(cpuMemory);
    if (constants == nullptr)
    {
        return 0;
    }

    constants->invViewProj = glm::inverse(proj * view);
    constants->view = view;
    constants->proj = proj;
    constants->cameraPosKm = glm::vec4(cameraPos, 1.0f);
    constants->sunDirection = glm::vec4(glm::normalize(environment.sunDirection), 0.0f);
    constants->sunIlluminance = glm::vec4(environment.sunIlluminance, 0.0f);
    constants->atmosphereHeights = glm::vec4(environment.atmosphere.groundRadiusKm,
                                             environment.atmosphere.atmosphereRadiusKm,
                                             environment.atmosphere.rayleighScaleHeightKm,
                                             environment.atmosphere.mieScaleHeightKm);
    constants->ozoneAndPhase = glm::vec4(environment.atmosphere.ozoneCenterHeightKm,
                                         environment.atmosphere.ozoneHalfWidthKm,
                                         environment.atmosphere.mieAnisotropy,
                                         environment.atmosphere.aerialPerspectiveDistanceKm);
    constants->rayleighScattering = glm::vec4(environment.atmosphere.rayleighScattering, 0.0f);
    constants->mieScattering = glm::vec4(environment.atmosphere.mieScattering, 0.0f);
    constants->mieAbsorption = glm::vec4(environment.atmosphere.mieAbsorption, 0.0f);
    constants->ozoneAbsorption = glm::vec4(environment.atmosphere.ozoneAbsorption, 0.0f);
    constants->viewportAndDepth =
        glm::vec4(static_cast<float>(renderer.width_), static_cast<float>(renderer.height_), kNearPlane, static_cast<float>(kAerialPerspectiveSliceCount));
    constants->sliceParams =
        glm::vec4(sliceIndex,
                  static_cast<float>(kAerialPerspectiveSliceCount),
                  environment.atmosphere.aerialPerspectiveDistanceKm,
                  environment.timeOfDay);
    return gpuAddress;
}

void Renderer::AtmosphereRenderer::renderLut(Renderer& renderer,
                                             ID3D12PipelineState* pipelineState,
                                             ID3D12Resource* targetResource,
                                             D3D12_RESOURCE_STATES& targetState,
                                             D3D12_CPU_DESCRIPTOR_HANDLE rtv,
                                             const EnvironmentState& environment,
                                             const glm::mat4& view,
                                             const glm::mat4& proj,
                                             const glm::vec3& cameraPos,
                                             D3D12_GPU_DESCRIPTOR_HANDLE texture0,
                                             D3D12_GPU_DESCRIPTOR_HANDLE texture1,
                                             D3D12_GPU_DESCRIPTOR_HANDLE texture2,
                                             float sliceIndex,
                                             UINT width,
                                             UINT height)
{
    transition(renderer, targetResource, targetState, D3D12_RESOURCE_STATE_RENDER_TARGET);
    D3D12_VIEWPORT viewport{};
    viewport.Width = static_cast<float>(width);
    viewport.Height = static_cast<float>(height);
    viewport.MaxDepth = 1.0f;
    D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    renderer.commandList_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    renderer.commandList_->RSSetViewports(1, &viewport);
    renderer.commandList_->RSSetScissorRects(1, &scissor);
    const std::uint64_t cbAddress = uploadConstants(renderer, environment, view, proj, cameraPos, sliceIndex);
    renderer.commandList_->SetGraphicsRootSignature(renderer.fullscreenRootSignature_.Get());
    renderer.commandList_->SetPipelineState(pipelineState);
    renderer.commandList_->SetGraphicsRootConstantBufferView(0, cbAddress);
    renderer.commandList_->SetGraphicsRootDescriptorTable(1, texture0);
    renderer.commandList_->SetGraphicsRootDescriptorTable(2, texture1);
    renderer.commandList_->SetGraphicsRootDescriptorTable(3, texture2);
    renderer.commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    renderer.commandList_->DrawInstanced(3, 1, 0, 0);
    transition(renderer, targetResource, targetState, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}

void Renderer::AtmosphereRenderer::renderTransmittance(Renderer& renderer,
                                                       const EnvironmentState& environment,
                                                       const glm::mat4& view,
                                                       const glm::mat4& proj,
                                                       const glm::vec3& cameraPos)
{
    renderLut(renderer,
              transmittancePso.Get(),
              transmittance.resource.Get(),
              transmittance.state,
              transmittance.rtv,
              environment,
              view,
              proj,
              cameraPos,
              transmittance.srvGpu,
              multiScattering.srvGpu,
              skyView.srvGpu,
              -1.0f,
              256,
              64);
}

void Renderer::AtmosphereRenderer::renderMultiScattering(Renderer& renderer,
                                                         const EnvironmentState& environment,
                                                         const glm::mat4& view,
                                                         const glm::mat4& proj,
                                                         const glm::vec3& cameraPos)
{
    renderLut(renderer,
              multiScatteringPso.Get(),
              multiScattering.resource.Get(),
              multiScattering.state,
              multiScattering.rtv,
              environment,
              view,
              proj,
              cameraPos,
              transmittance.srvGpu,
              transmittance.srvGpu,
              skyView.srvGpu,
              -1.0f,
              32,
              32);
}

void Renderer::AtmosphereRenderer::renderSkyView(Renderer& renderer,
                                                 const EnvironmentState& environment,
                                                 const glm::mat4& view,
                                                 const glm::mat4& proj,
                                                 const glm::vec3& cameraPos)
{
    renderLut(renderer,
              skyViewPso.Get(),
              skyView.resource.Get(),
              skyView.state,
              skyView.rtv,
              environment,
              view,
              proj,
              cameraPos,
              transmittance.srvGpu,
              multiScattering.srvGpu,
              skyView.srvGpu,
              -1.0f,
              256,
              128);
}

void Renderer::AtmosphereRenderer::renderAerialPerspective(Renderer& renderer,
                                                           const EnvironmentState& environment,
                                                           const glm::mat4& view,
                                                           const glm::mat4& proj,
                                                           const glm::vec3& cameraPos)
{
    transition(renderer,
               aerialPerspective.resource.Get(),
               aerialPerspective.state,
               D3D12_RESOURCE_STATE_RENDER_TARGET);
    D3D12_VIEWPORT viewport{};
    viewport.Width = 32.0f;
    viewport.Height = 32.0f;
    viewport.MaxDepth = 1.0f;
    D3D12_RECT scissor{0, 0, 32, 32};
    renderer.commandList_->RSSetViewports(1, &viewport);
    renderer.commandList_->RSSetScissorRects(1, &scissor);
    renderer.commandList_->SetGraphicsRootSignature(renderer.fullscreenRootSignature_.Get());
    renderer.commandList_->SetPipelineState(aerialPerspectivePso.Get());
    renderer.commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    for (UINT slice = 0; slice < kAerialPerspectiveSliceCount; ++slice)
    {
        renderer.commandList_->OMSetRenderTargets(1, &aerialPerspective.rtvs[slice], FALSE, nullptr);
        const std::uint64_t cbAddress = uploadConstants(renderer, environment, view, proj, cameraPos, static_cast<float>(slice));
        renderer.commandList_->SetGraphicsRootConstantBufferView(0, cbAddress);
        renderer.commandList_->SetGraphicsRootDescriptorTable(1, transmittance.srvGpu);
        renderer.commandList_->SetGraphicsRootDescriptorTable(2, multiScattering.srvGpu);
        renderer.commandList_->SetGraphicsRootDescriptorTable(3, skyView.srvGpu);
        renderer.commandList_->DrawInstanced(3, 1, 0, 0);
    }
    transition(renderer,
               aerialPerspective.resource.Get(),
               aerialPerspective.state,
               D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}
