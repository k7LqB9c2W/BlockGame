// chunk_manager.cpp
// Implements the chunk streaming, terrain generation, and GPU upload subsystem.

#include "chunk_manager.h"

#include "terrain/biome_database.h"
#include "terrain/climate_map.h"
#include "terrain/surface_map.h"
#include "terrain/terrain_generator.h"
#include "terrain/worldgen_profile.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/noise.hpp>
#include <wrl/client.h>

namespace
{
std::atomic<int> gActiveVerticalRadius{kVerticalStreamingConfig.minRadiusChunks};
}

float computeFarPlaneForViewDistance(int viewDistance) noexcept
{
    const int verticalRadius = std::max(gActiveVerticalRadius.load(std::memory_order_relaxed),
                                        kVerticalStreamingConfig.minRadiusChunks);
    const double horizontalSpan = static_cast<double>(viewDistance + 1)
                                  * static_cast<double>(std::max(kChunkSizeX, kChunkSizeZ));
    const double verticalSpan = static_cast<double>(verticalRadius + 1) * static_cast<double>(kChunkSizeY);
    const double diagonal = std::hypot(horizontalSpan, verticalSpan);
    const double farPlane = std::max(diagonal + static_cast<double>(kFarPlanePadding),
                                     static_cast<double>(kDefaultFarPlane));
    return static_cast<float>(farPlane);
}

float computeFarPlaneForDistanceBlocks(int farDistanceBlocks) noexcept
{
    const int clampedBlocks = std::max(farDistanceBlocks, 1);
    const int verticalRadius = std::max(gActiveVerticalRadius.load(std::memory_order_relaxed),
                                        kVerticalStreamingConfig.minRadiusChunks);
    const double horizontalSpan = static_cast<double>(clampedBlocks);
    const double verticalSpan = static_cast<double>(verticalRadius + 1) * static_cast<double>(kChunkSizeY);
    const double diagonal = std::hypot(horizontalSpan, verticalSpan);
    const double farPlane = std::max(diagonal + static_cast<double>(kFarPlanePadding),
                                     static_cast<double>(kDefaultFarPlane));
    return static_cast<float>(farPlane);
}

float kFarPlane = computeFarPlaneForDistanceBlocks(kDefaultFarRenderDistanceBlocks);

Frustum Frustum::fromMatrix(const glm::mat4& matrix)
{
    Frustum frustum;
    const glm::vec4 row0(matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]);
    const glm::vec4 row1(matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]);
    const glm::vec4 row2(matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]);
    const glm::vec4 row3(matrix[0][3], matrix[1][3], matrix[2][3], matrix[3][3]);

    frustum.planes[0] = row3 + row0; // Left
    frustum.planes[1] = row3 - row0; // Right
    frustum.planes[2] = row3 + row1; // Bottom
    frustum.planes[3] = row3 - row1; // Top
    frustum.planes[4] = row3 + row2; // Near
    frustum.planes[5] = row3 - row2; // Far

    for (auto& plane : frustum.planes)
    {
        const float length = std::sqrt(plane.x * plane.x + plane.y * plane.y + plane.z * plane.z);
        if (length > 0.0f)
        {
            plane /= length;
        }
    }

    return frustum;
}

bool Frustum::intersectsAABB(const glm::vec3& minCorner, const glm::vec3& maxCorner) const noexcept
{
    for (const auto& plane : planes)
    {
        glm::vec3 positiveVertex = minCorner;
        if (plane.x >= 0.0f) positiveVertex.x = maxCorner.x;
        if (plane.y >= 0.0f) positiveVertex.y = maxCorner.y;
        if (plane.z >= 0.0f) positiveVertex.z = maxCorner.z;

        if (glm::dot(glm::vec3(plane), positiveVertex) + plane.w < 0.0f)
        {
            return false;
        }
    }
    return true;
}

namespace
{
using Vertex = WorldVertex;

void throwIfFailedDx(HRESULT hr, const char* message)
{
    if (FAILED(hr))
    {
        throw std::runtime_error(message);
    }
}

Microsoft::WRL::ComPtr<ID3D12Resource> createUploadBuffer(ID3D12Device* device,
                                                          std::uint64_t sizeInBytes,
                                                          std::byte*& mappedData)
{
    mappedData = nullptr;
    if (device == nullptr || sizeInBytes == 0)
    {
        return {};
    }

    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = sizeInBytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    throwIfFailedDx(device->CreateCommittedResource(&heapProps,
                                                    D3D12_HEAP_FLAG_NONE,
                                                    &desc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ,
                                                    nullptr,
                                                    IID_PPV_ARGS(&resource)),
                    "failed to create upload buffer");

    void* mapped = nullptr;
    throwIfFailedDx(resource->Map(0, nullptr, &mapped), "failed to map upload buffer");
    mappedData = static_cast<std::byte*>(mapped);
    return resource;
}

Microsoft::WRL::ComPtr<ID3D12Resource> createDefaultBuffer(ID3D12Device* device,
                                                           std::uint64_t sizeInBytes,
                                                           D3D12_RESOURCE_STATES initialState)
{
    if (device == nullptr || sizeInBytes == 0)
    {
        return {};
    }

    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = sizeInBytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    throwIfFailedDx(device->CreateCommittedResource(&heapProps,
                                                    D3D12_HEAP_FLAG_NONE,
                                                    &desc,
                                                    initialState,
                                                    nullptr,
                                                    IID_PPV_ARGS(&resource)),
                    "failed to create default buffer");
    return resource;
}

D3D12_RESOURCE_BARRIER transitionBarrier(ID3D12Resource* resource,
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

class UploadContext
{
public:
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
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        throwIfFailedDx(device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue_)),
                        "failed to create upload command queue");
        throwIfFailedDx(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_)),
                        "failed to create upload command allocator");
        throwIfFailedDx(device_->CreateCommandList(0,
                                                   D3D12_COMMAND_LIST_TYPE_DIRECT,
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
        open_ = false;
        hasCommands_ = false;
    }

    void begin()
    {
        if (device_ == nullptr)
        {
            return;
        }
        if (open_)
        {
            return;
        }

        throwIfFailedDx(allocator_->Reset(), "failed to reset upload command allocator");
        throwIfFailedDx(commandList_->Reset(allocator_.Get(), nullptr), "failed to reset upload command list");
        open_ = true;
        hasCommands_ = false;
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

    void flush()
    {
        if (!open_)
        {
            return;
        }

        throwIfFailedDx(commandList_->Close(), "failed to close upload command list");
        if (hasCommands_)
        {
            ID3D12CommandList* lists[] = {commandList_.Get()};
            queue_->ExecuteCommandLists(static_cast<UINT>(std::size(lists)), lists);

            ++fenceValue_;
            throwIfFailedDx(queue_->Signal(fence_.Get(), fenceValue_), "failed to signal upload fence");
            if (fence_->GetCompletedValue() < fenceValue_)
            {
                throwIfFailedDx(fence_->SetEventOnCompletion(fenceValue_, fenceEvent_),
                                "failed to wait for upload fence");
                WaitForSingleObject(fenceEvent_, INFINITE);
            }
        }

        open_ = false;
        hasCommands_ = false;
    }

    [[nodiscard]] bool ready() const noexcept
    {
        return device_ != nullptr && queue_ != nullptr && allocator_ != nullptr && commandList_ != nullptr;
    }

private:
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_{nullptr};
    UINT64 fenceValue_{0};
    bool open_{false};
    bool hasCommands_{false};
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

struct BlockLightingProperties
{
    bool opaque{true};
    std::uint8_t skyAttenuation{kMaxLightLevel};
    std::uint8_t blockEmission{0};
    bool aoSolid{true};
};

constexpr std::array<BlockLightingProperties, toIndex(BlockId::Count)> kBlockLightingTable{{
    {false, 0, 0, false},              // Air
    {true, kMaxLightLevel, 0, true},   // Grass
    {true, kMaxLightLevel, 0, true},   // Wood
    {false, 1, 0, true},               // Leaves
    {true, kMaxLightLevel, 0, true},   // Sand
    {false, 2, 0, false},              // Water
    {true, kMaxLightLevel, 0, true},   // Stone
    {true, kMaxLightLevel, 0, true},   // SpruceLog
    {false, 1, 0, true},               // SpruceLeaves
    {true, kMaxLightLevel, 0, true},   // Podzol
    {true, kMaxLightLevel, 14, true},  // DebugLamp
}};

inline const BlockLightingProperties& blockLightingProperties(BlockId block) noexcept
{
    return kBlockLightingTable[toIndex(block)];
}

inline bool isOpaqueForLighting(BlockId block) noexcept
{
    return blockLightingProperties(block).opaque;
}

inline std::uint8_t packLightLevels(std::uint8_t sky, std::uint8_t block) noexcept
{
    return static_cast<std::uint8_t>(((std::min<std::uint8_t>)(sky, kMaxLightLevel) << 4) |
                                     (std::min<std::uint8_t>)(block, kMaxLightLevel));
}

inline std::uint8_t skyLightFromPacked(std::uint8_t packed) noexcept
{
    return static_cast<std::uint8_t>((packed >> 4) & 0x0F);
}

inline std::uint8_t blockLightFromPacked(std::uint8_t packed) noexcept
{
    return static_cast<std::uint8_t>(packed & 0x0F);
}

inline void setSkyLight(std::uint8_t& packed, std::uint8_t sky) noexcept
{
    packed = packLightLevels(sky, blockLightFromPacked(packed));
}

inline void setBlockLight(std::uint8_t& packed, std::uint8_t block) noexcept
{
    packed = packLightLevels(skyLightFromPacked(packed), block);
}

inline std::uint8_t propagationLossFor(BlockId block) noexcept
{
    return static_cast<std::uint8_t>(1 + blockLightingProperties(block).skyAttenuation);
}

inline bool isAoSolid(BlockId block) noexcept
{
    return blockLightingProperties(block).aoSolid;
}

inline std::uint32_t packVertexLighting(std::uint8_t packedLight,
                                        std::uint8_t aoLevel = 0,
                                        std::uint8_t flags = 0) noexcept
{
    return static_cast<std::uint32_t>(packedLight) |
           (static_cast<std::uint32_t>(aoLevel & 0x03u) << 8) |
           (static_cast<std::uint32_t>(flags) << 10);
}

inline std::uint8_t aoLevelFromPackedVertexLighting(std::uint32_t packed) noexcept
{
    return static_cast<std::uint8_t>((packed >> 8) & 0x03u);
}

inline int lightingMetricFromPackedVertex(std::uint32_t packed) noexcept
{
    const std::uint8_t packedLight = static_cast<std::uint8_t>(packed & 0xFFu);
    const int sky = static_cast<int>(skyLightFromPacked(packedLight));
    const int block = static_cast<int>(blockLightFromPacked(packedLight));
    const int ao = static_cast<int>(aoLevelFromPackedVertexLighting(packed));
    return sky * 24 + block * 18 + (3 - ao) * 20;
}

inline bool isAlphaCutoutBlock(BlockId block) noexcept
{
    return block == BlockId::Leaves || block == BlockId::SpruceLeaves;
}

inline bool isNonOpaqueBlock(BlockId block) noexcept
{
    return block == BlockId::Air || block == BlockId::Water || isAlphaCutoutBlock(block);
}

inline bool shouldRenderBlockFace(BlockId owningBlock, BlockId neighborBlock) noexcept
{
    if (owningBlock == BlockId::Air)
    {
        return false;
    }

    if (neighborBlock == BlockId::Air)
    {
        return true;
    }

    if (isAlphaCutoutBlock(owningBlock))
    {
        if (isAlphaCutoutBlock(neighborBlock))
        {
            return owningBlock != neighborBlock;
        }

        return neighborBlock == BlockId::Water;
    }

    if (owningBlock == BlockId::Water)
    {
        return neighborBlock == BlockId::Air;
    }

    return isNonOpaqueBlock(neighborBlock);
}

constexpr int kTaigaSpruceCellSize = 14;
constexpr int kTaigaSpruceMinTrunkHeight = 25;
constexpr int kTaigaSpruceMaxTrunkHeight = 31;
constexpr int kTaigaSpruceMinBareTrunkHeight = 5;
constexpr int kTaigaSpruceMaxBareTrunkHeight = 9;
constexpr int kTaigaSpruceMaxLeafRadius = 4;

inline glm::ivec2 taigaSpruceOriginForCell(int cellX, int cellZ) noexcept
{
    const int offsetX = 3 + static_cast<int>(hashToUnitFloat(cellX, 911, cellZ) * 4.0f);
    const int offsetZ = 3 + static_cast<int>(hashToUnitFloat(cellX, 977, cellZ) * 4.0f);
    return glm::ivec2(cellX * kTaigaSpruceCellSize + offsetX,
                      cellZ * kTaigaSpruceCellSize + offsetZ);
}

inline bool isTaigaSpruceOrigin(int worldX, int worldZ) noexcept
{
    const int cellX = floorDiv(worldX, kTaigaSpruceCellSize);
    const int cellZ = floorDiv(worldZ, kTaigaSpruceCellSize);
    return taigaSpruceOriginForCell(cellX, cellZ) == glm::ivec2(worldX, worldZ);
}

inline float taigaSpruceOccupancyChance(const BiomeDefinition& biome) noexcept
{
    return std::clamp(0.40f + std::max(biome.treeDensityMultiplier, 0.0f) * 0.20f, 0.45f, 0.90f);
}

inline bool shouldSpawnTaigaSpruce(const BiomeDefinition& biome, int worldX, int groundWorldY, int worldZ) noexcept
{
    if (!terrain::isTaigaBiome(biome) || !isTaigaSpruceOrigin(worldX, worldZ))
    {
        return false;
    }

    const int cellX = floorDiv(worldX, kTaigaSpruceCellSize);
    const int cellZ = floorDiv(worldZ, kTaigaSpruceCellSize);
    const float occupancyRoll = hashToUnitFloat(cellX, groundWorldY + 151, cellZ);
    return occupancyRoll <= taigaSpruceOccupancyChance(biome);
}

inline int taigaSpruceTrunkHeight(int worldX, int groundWorldY, int worldZ) noexcept
{
    int height = kTaigaSpruceMinTrunkHeight +
                 static_cast<int>(hashToUnitFloat(worldX, groundWorldY + 37, worldZ) *
                                  static_cast<float>(kTaigaSpruceMaxTrunkHeight - kTaigaSpruceMinTrunkHeight + 1));
    return std::clamp(height, kTaigaSpruceMinTrunkHeight, kTaigaSpruceMaxTrunkHeight);
}

inline int taigaSpruceBareTrunkHeight(int worldX, int groundWorldY, int worldZ) noexcept
{
    int height = kTaigaSpruceMinBareTrunkHeight +
                 static_cast<int>(hashToUnitFloat(worldX, groundWorldY + 83, worldZ) *
                                  static_cast<float>(kTaigaSpruceMaxBareTrunkHeight - kTaigaSpruceMinBareTrunkHeight + 1));
    return std::clamp(height, kTaigaSpruceMinBareTrunkHeight, kTaigaSpruceMaxBareTrunkHeight);
}

inline int taigaSpruceLeafRadiusForLayer(int layerFromBottom, int totalLayers) noexcept
{
    if (totalLayers <= 1)
    {
        return 0;
    }

    const float t = static_cast<float>(layerFromBottom) / static_cast<float>(std::max(totalLayers - 1, 1));
    int radius = 1 + static_cast<int>(std::round((1.0f - t) * 3.0f));

    if (layerFromBottom % 3 == 0 && layerFromBottom < (totalLayers * 3) / 4)
    {
        radius = std::min(radius + 1, kTaigaSpruceMaxLeafRadius);
    }

    if (t > 0.88f)
    {
        radius = 1;
    }
    if (t > 0.97f)
    {
        radius = 0;
    }

    return std::clamp(radius, 0, kTaigaSpruceMaxLeafRadius);
}

inline int distanceToInclusiveRange(int value, int minValue, int maxValue) noexcept
{
    if (value < minValue)
    {
        return minValue - value;
    }
    if (value > maxValue)
    {
        return value - maxValue;
    }
    return 0;
}

inline bool taigaSpruceLeafOccupiesCell(int originX,
                                        int originZ,
                                        int worldX,
                                        int worldZ,
                                        int radius,
                                        int layerFromBottom,
                                        int totalLayers) noexcept
{
    if (radius <= 0)
    {
        return false;
    }

    if (worldX >= originX && worldX <= originX + 1 &&
        worldZ >= originZ && worldZ <= originZ + 1)
    {
        return false;
    }

    const int dx = distanceToInclusiveRange(worldX, originX, originX + 1);
    const int dz = distanceToInclusiveRange(worldZ, originZ, originZ + 1);
    const int chebyshev = std::max(dx, dz);
    if (chebyshev > radius)
    {
        return false;
    }

    int manhattanAllowance = radius + 1;
    if (radius >= 4 && layerFromBottom < totalLayers / 3)
    {
        ++manhattanAllowance;
    }

    return (dx + dz) <= manhattanAllowance;
}

struct MeshData
{
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;

    MeshData()
    {
        vertices.reserve(4096);
        indices.reserve(6144);
    }

    void clear()
    {
        vertices.clear();
        indices.clear();
    }

    bool empty() const
    {
        return vertices.empty() || indices.empty();
    }
};

enum class ChunkState : std::uint8_t
{
    Empty = 0,
    Generating,
    Meshing,
    Ready,
    Uploaded,
    Remeshing
};

enum class JobType : std::uint8_t
{
    Generate = 0,
    Mesh = 1
};

struct FarChunk
{
    static constexpr int kColumnStep = 4;
    static constexpr int kColumnsX = kChunkSizeX / kColumnStep;
    static constexpr int kColumnsZ = kChunkSizeZ / kColumnStep;

    struct SurfaceCell
    {
        int worldY{std::numeric_limits<int>::min()};
        BlockId block{BlockId::Air};
    };

    glm::vec3 origin{0.0f};
    glm::ivec3 size{kChunkSizeX, kChunkSizeY, kChunkSizeZ};
    int lodStep{kColumnStep};
    int thickness{1};
    std::array<SurfaceCell, kColumnsX * kColumnsZ> strata{};

    static constexpr std::size_t index(int x, int z) noexcept
    {
        return static_cast<std::size_t>(z) * static_cast<std::size_t>(kColumnsX) +
               static_cast<std::size_t>(x);
    }
};

constexpr std::uint32_t kInvalidChunkBufferPage = std::numeric_limits<std::uint32_t>::max();

struct Chunk
{
    explicit Chunk(const glm::ivec3& c)
        : coord(c),
          minWorldY(c.y * kChunkSizeY),
          maxWorldY(minWorldY + kChunkSizeY - 1),
          blocks(kChunkBlockCount, BlockId::Air),
          lightLevels(kChunkBlockCount, packLightLevels(kMaxLightLevel, 0)),
          state(ChunkState::Empty)
    {
    }

    void reset(const glm::ivec3& c)
    {
        coord = c;
        minWorldY = c.y * kChunkSizeY;
        maxWorldY = minWorldY + kChunkSizeY - 1;
        if (blocks.size() != static_cast<std::size_t>(kChunkBlockCount))
        {
            blocks.assign(kChunkBlockCount, BlockId::Air);
        }
        else
        {
            std::fill(blocks.begin(), blocks.end(), BlockId::Air);
        }
        if (lightLevels.size() != static_cast<std::size_t>(kChunkBlockCount))
        {
            lightLevels.assign(kChunkBlockCount, packLightLevels(kMaxLightLevel, 0));
        }
        else
        {
            std::fill(lightLevels.begin(), lightLevels.end(), packLightLevels(kMaxLightLevel, 0));
        }
        state.store(ChunkState::Empty, std::memory_order_relaxed);
        meshData.clear();
        meshReady.store(false, std::memory_order_relaxed);
        hasBlocks.store(false, std::memory_order_relaxed);
        queuedForUpload.store(false, std::memory_order_relaxed);
        indexCount.store(0, std::memory_order_relaxed);
        vertexCount.store(0, std::memory_order_relaxed);
        bufferPageIndex.store(kInvalidChunkBufferPage, std::memory_order_relaxed);
        vertexOffset.store(0, std::memory_order_relaxed);
        indexOffset.store(0, std::memory_order_relaxed);
        inFlight.store(0, std::memory_order_relaxed);
        surfaceOnly = false;
        lodData.reset();
        lightBoundaryDirtyMask = 0;
        pendingMeshRefresh.store(false, std::memory_order_relaxed);
    }


    glm::ivec3 coord;
    int minWorldY{0};
    int maxWorldY{0};
    std::vector<BlockId> blocks;
    std::vector<std::uint8_t> lightLevels;
    std::atomic<ChunkState> state;

    std::atomic<std::uint32_t> indexCount{0};
    std::atomic<std::size_t> vertexCount{0};
    std::atomic<std::uint32_t> bufferPageIndex{kInvalidChunkBufferPage};
    std::atomic<std::size_t> vertexOffset{0};
    std::atomic<std::size_t> indexOffset{0};
    std::atomic<bool> queuedForUpload{false};

    mutable std::mutex meshMutex;
    MeshData meshData;
    std::atomic<bool> meshReady{false};
    std::atomic<bool> hasBlocks{false};
    std::atomic<int> inFlight{0};
    bool surfaceOnly{false};
    std::unique_ptr<FarChunk> lodData;
    std::uint8_t lightBoundaryDirtyMask{0};
    std::atomic<bool> pendingMeshRefresh{false};
};

struct ProfilingCounters
{
    std::atomic<long long> generationMicros{0};
    std::atomic<long long> meshingMicros{0};
    std::atomic<std::size_t> uploadedBytes{0};
    std::atomic<int> generatedChunks{0};
    std::atomic<int> meshedChunks{0};
    std::atomic<int> uploadedChunks{0};
    std::atomic<int> throttledUploads{0};
    std::atomic<int> deferredUploads{0};
    std::atomic<int> evictedChunks{0};
};

struct ChunkHasher
{
    std::size_t operator()(const glm::ivec3& v) const noexcept
    {
        std::size_t hash = static_cast<std::size_t>(v.x) * 73856093u;
        hash ^= static_cast<std::size_t>(v.y) * 19349663u;
        hash ^= static_cast<std::size_t>(v.z) * 83492791u;
        return hash;
    }
};

struct ColumnHasher
{
    std::size_t operator()(const glm::ivec2& v) const noexcept
    {
        std::size_t hash = static_cast<std::size_t>(v.x) * 73856093u;
        hash ^= static_cast<std::size_t>(v.y) * 19349663u;

        return hash;
    }
};

struct PendingStructureEdit
{
    glm::ivec3 chunkCoord{0};
    glm::ivec3 worldPos{0};
    BlockId block{BlockId::Air};
    bool replaceSolid{false};
};

struct Job
{
    JobType type;
    glm::ivec3 chunkCoord;
    std::shared_ptr<Chunk> chunk;

    Job(JobType t, const glm::ivec3& coord, std::shared_ptr<Chunk> c)
        : type(t), chunkCoord(coord), chunk(std::move(c)) {}
};

class JobQueue
{
public:
    void push(const Job& job);
    bool tryPop(Job& job);
    Job waitAndPop();
    void stop();
    bool empty() const;
    void updatePriorityOrigin(const glm::ivec3& origin);

private:
    struct PrioritizedJob
    {
        Job job;
        int distance{0};
        int priorityBias{0};
        std::uint64_t sequence{0};
    };

    struct JobComparer
    {
        bool operator()(const PrioritizedJob& lhs, const PrioritizedJob& rhs) const;
    };

    PrioritizedJob wrap(const Job& job);
    static int manhattanDistance(const glm::ivec3& a, const glm::ivec3& b) noexcept;
    void rebuildLocked();

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> shouldStop_{false};
    glm::ivec3 priorityOrigin_{0, 0, 0};
    std::priority_queue<PrioritizedJob, std::vector<PrioritizedJob>, JobComparer> priorityQueue_;
    std::uint64_t nextSequence_{0};
};

class ColumnManager
{
public:
    static constexpr int kNoHeight = std::numeric_limits<int>::min();

    void updateChunk(const Chunk& chunk);
    void updateColumn(const Chunk& chunk, int localX, int localZ);
    void removeChunk(const Chunk& chunk);
    void clear();

    int highestSolidBlock(int worldX, int worldZ) const noexcept;

private:
    struct ColumnData
    {
        std::unordered_map<int, int> slabHeights;
        int highestWorldY{kNoHeight};
    };

    static glm::ivec2 columnKey(const glm::ivec3& chunkCoord, int localX, int localZ) noexcept;
    static int scanColumnHighestWorld(const Chunk& chunk, int localX, int localZ) noexcept;
    static int computeHighest(const ColumnData& data) noexcept;
    void applyHeightLocked(const glm::ivec2& key, int chunkY, int highestWorldY);

    mutable std::mutex mutex_;
    std::unordered_map<glm::ivec2, ColumnData, ColumnHasher> columns_;
};

struct FarTerrainSurfaceSample
{
    int solidTopY{std::numeric_limits<int>::min()};
    BlockId solidBlock{BlockId::Air};
    int waterTopY{std::numeric_limits<int>::min()};
    bool hasVisibleWater{false};
};

class FarTerrainManager
{
public:
    struct LevelConfig
    {
        int id{0};
        int innerRadiusChunks{0};
        int outerRadiusChunks{0};
        int tileSizeChunks{8};
        int sampleStepBlocks{8};
        int lodLevel{3};
        int skirtDepthBlocks{32};
    };

    using SampleFn = std::function<FarTerrainSurfaceSample(int worldX, int worldZ, int lodLevel)>;
    using UvLookupFn = std::function<std::pair<glm::vec2, glm::vec2>(BlockId block, BlockFace face)>;

    FarTerrainManager()
        : levels_{
              LevelConfig{1, kDefaultNearRenderDistance, 64, 8, 8, 3, 32},
              LevelConfig{2, 64, 300, 32, 32, 5, 96}}
    {
    }

    ~FarTerrainManager()
    {
        stopWorkers();
        clear();
        uploadContext_.shutdown();
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
        if (workerCount_ == clamped && !workerThreads_.empty())
        {
            return;
        }

        stopWorkers();
        workerCount_ = clamped;
        startWorkers();
    }

    void setDistanceBlocks(int blocks)
    {
        farDistanceBlocks_ = std::max(blocks, 256);
    }

    [[nodiscard]] int distanceBlocks() const noexcept
    {
        return farDistanceBlocks_;
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
        clear();
    }

    void update(const glm::ivec3& cameraChunk,
                const glm::vec3& cameraForward,
                int nearRadiusChunks,
                int realDistanceBlocks,
                double uploadBudgetMs,
                const SampleFn& sampleFn,
                const UvLookupFn& uvLookup)
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

        farDistanceBlocks_ = std::max(realDistanceBlocks, 256);
        ++updateStamp_;
        cameraChunk_ = cameraChunk;
        if (glm::dot(cameraForward, cameraForward) > kEpsilon)
        {
            cameraForward_ = glm::normalize(cameraForward);
        }

        {
            std::lock_guard<std::mutex> lock(configMutex_);
            sampleFn_ = sampleFn;
            uvLookupFn_ = uvLookup;
        }

        const int realRadiusChunks = std::max(nearRadiusChunks + 1,
                                              ceilToIntPositive(
                                                  static_cast<float>(farDistanceBlocks_) / static_cast<float>(kChunkSizeX)));

        levels_[0].innerRadiusChunks = nearRadiusChunks;
        levels_[0].outerRadiusChunks = std::min(realRadiusChunks, 64);
        levels_[1].innerRadiusChunks = 64;
        levels_[1].outerRadiusChunks = realRadiusChunks;

        auto touchLevel = [&](const LevelConfig& level)
        {
            const int tileMinX = floorDiv(cameraChunk.x - level.outerRadiusChunks, level.tileSizeChunks);
            const int tileMaxX = floorDiv(cameraChunk.x + level.outerRadiusChunks, level.tileSizeChunks);
            const int tileMinZ = floorDiv(cameraChunk.z - level.outerRadiusChunks, level.tileSizeChunks);
            const int tileMaxZ = floorDiv(cameraChunk.z + level.outerRadiusChunks, level.tileSizeChunks);

            for (int tileX = tileMinX; tileX <= tileMaxX; ++tileX)
            {
                for (int tileZ = tileMinZ; tileZ <= tileMaxZ; ++tileZ)
                {
                    if (!tileIntersectsRing(level, cameraChunk_, tileX, tileZ))
                    {
                        continue;
                    }

                    FarTileKey key{level.id, tileX, tileZ};
                    FarTile& tile = tiles_[key];
                    tile.key = key;
                    tile.level = level;
                    tile.lastTouchedStamp = updateStamp_;
                    tile.active = true;
                    if (!tile.initialized)
                    {
                        initializeTile(tile, level);
                        markDirty(tile);
                    }
                }
            }
        };

        if (levels_[0].outerRadiusChunks > levels_[0].innerRadiusChunks)
        {
            touchLevel(levels_[0]);
        }
        if (levels_[1].outerRadiusChunks > levels_[1].innerRadiusChunks)
        {
            touchLevel(levels_[1]);
        }

        std::vector<FarTileKey> staleKeys;
        staleKeys.reserve(tiles_.size());
        for (const auto& [key, tile] : tiles_)
        {
            if (tile.lastTouchedStamp != updateStamp_)
            {
                staleKeys.push_back(key);
            }
        }

        for (const FarTileKey& key : staleKeys)
        {
            auto it = tiles_.find(key);
            if (it == tiles_.end())
            {
                continue;
            }

            releaseTileGpu(it->second);
            tiles_.erase(it);
        }

        collectCompletedBuilds(uploadBudgetMs);
        scheduleDirtyBuilds();
    }

    [[nodiscard]] std::vector<ChunkRenderBatch> buildRenderBatches(const Frustum& frustum) const
    {
        std::vector<ChunkRenderBatch> batches;
        batches.resize(bufferPages_.size());
        for (std::size_t i = 0; i < bufferPages_.size(); ++i)
        {
            batches[i].vertexBufferView = bufferPages_[i].vertexView;
            batches[i].indexBufferView = bufferPages_[i].indexView;
        }

        for (const auto& [key, tile] : tiles_)
        {
            (void)key;
            if (!tile.active || tile.indexCount == 0)
            {
                continue;
            }
            if (!frustum.intersectsAABB(tile.boundsMin, tile.boundsMax))
            {
                continue;
            }
            if (tile.pageIndex == kInvalidChunkBufferPage || tile.pageIndex >= batches.size())
            {
                continue;
            }

            ChunkRenderBatch& batch = batches[tile.pageIndex];
            batch.indexCounts.push_back(tile.indexCount);
            batch.firstIndexLocations.push_back(static_cast<std::uint32_t>(tile.indexOffset));
            batch.baseVertices.push_back(static_cast<std::int32_t>(tile.vertexOffset));
        }

        auto emptyIt = std::remove_if(batches.begin(),
                                      batches.end(),
                                      [](const ChunkRenderBatch& batch)
                                      {
                                          return batch.indexCounts.empty();
                                      });
        batches.erase(emptyIt, batches.end());
        return batches;
    }

    void invalidateWorldBlock(const glm::ivec3& worldPos)
    {
        for (auto& [key, tile] : tiles_)
        {
            (void)key;
            const int tileSpanBlocks = tile.level.tileSizeChunks * kChunkSizeX;
            const int minX = tile.key.tileX * tileSpanBlocks;
            const int minZ = tile.key.tileZ * tileSpanBlocks;
            const int maxX = minX + tileSpanBlocks;
            const int maxZ = minZ + tileSpanBlocks;
            if (worldPos.x < minX - tileSpanBlocks || worldPos.x > maxX + tileSpanBlocks ||
                worldPos.z < minZ - tileSpanBlocks || worldPos.z > maxZ + tileSpanBlocks)
            {
                continue;
            }

            markDirty(tile);
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
        for (auto& [key, tile] : tiles_)
        {
            (void)key;
            releaseTileGpu(tile);
            tile.inFlight = false;
        }
        tiles_.clear();
        destroyBufferPages();
        builtTilesLastUpdate_ = 0;
        lastAverageBuildMs_ = 0.0;
        lastCollectMs_ = 0.0;
        lastUploadMs_ = 0.0;
    }

    [[nodiscard]] int activeTileCount() const noexcept
    {
        return static_cast<int>(tiles_.size());
    }

    [[nodiscard]] int dirtyTileCount() const noexcept
    {
        int dirty = 0;
        for (const auto& [key, tile] : tiles_)
        {
            (void)key;
            if (tile.active && tile.dirty)
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

    [[nodiscard]] double lastCollectMs() const noexcept
    {
        return lastCollectMs_;
    }

    [[nodiscard]] double lastUploadMs() const noexcept
    {
        return lastUploadMs_;
    }

private:
    struct FarTileKey
    {
        int level{0};
        int tileX{0};
        int tileZ{0};

        bool operator==(const FarTileKey& other) const noexcept
        {
            return level == other.level && tileX == other.tileX && tileZ == other.tileZ;
        }
    };

    struct FarTileKeyHasher
    {
        std::size_t operator()(const FarTileKey& key) const noexcept
        {
            std::size_t hash = static_cast<std::size_t>(key.level) * 73856093u;
            hash ^= static_cast<std::size_t>(key.tileX) * 19349663u;
            hash ^= static_cast<std::size_t>(key.tileZ) * 83492791u;
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
        D3D12_VERTEX_BUFFER_VIEW vertexView{};
        D3D12_INDEX_BUFFER_VIEW indexView{};
        std::byte* mappedVertexData{nullptr};
        std::byte* mappedIndexData{nullptr};
        D3D12_RESOURCE_STATES vertexState{D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER};
        D3D12_RESOURCE_STATES indexState{D3D12_RESOURCE_STATE_INDEX_BUFFER};
        std::size_t vertexCapacity{0};
        std::size_t indexCapacity{0};
        std::size_t vertexCursor{0};
        std::size_t indexCursor{0};
        std::vector<Range> freeVertices;
        std::vector<Range> freeIndices;
    };

    struct Allocation
    {
        std::uint32_t pageIndex{kInvalidChunkBufferPage};
        std::size_t vertexOffset{0};
        std::size_t indexOffset{0};
    };

    struct FarTile
    {
        FarTileKey key{};
        LevelConfig level{};
        glm::vec3 boundsMin{0.0f};
        glm::vec3 boundsMax{0.0f};
        std::uint64_t lastTouchedStamp{0};
        std::uint32_t pageIndex{kInvalidChunkBufferPage};
        std::size_t vertexOffset{0};
        std::size_t indexOffset{0};
        std::size_t vertexCount{0};
        std::uint32_t indexCount{0};
        std::uint32_t buildVersion{1};
        bool active{false};
        bool dirty{true};
        bool initialized{false};
        bool inFlight{false};
    };

    struct TileMesh
    {
        std::vector<Vertex> vertices;
        std::vector<std::uint32_t> indices;
        glm::vec3 boundsMin{0.0f};
        glm::vec3 boundsMax{1.0f};
    };

    struct BuildResult
    {
        FarTileKey key{};
        std::uint32_t buildVersion{0};
        std::uint64_t epoch{0};
        TileMesh mesh{};
        double buildMs{0.0};
    };

    struct BuildJob
    {
        FarTileKey key{};
        LevelConfig level{};
        std::uint32_t buildVersion{0};
        std::uint64_t epoch{0};
    };

    struct CandidateBuild
    {
        FarTileKey key{};
        float distanceSq{0.0f};
        int forwardBucket{0};
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
                           const std::pair<glm::vec2, glm::vec2>& uv)
    {
        const std::uint32_t baseIndex = static_cast<std::uint32_t>(vertices.size());
        const std::uint32_t lightingData = packVertexLighting(packLightLevels(kMaxLightLevel, 0));
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

    [[nodiscard]] static bool tileIntersectsRing(const LevelConfig& level,
                                                 const glm::ivec3& cameraChunk,
                                                 int tileX,
                                                 int tileZ) noexcept
    {
        const int minX = tileX * level.tileSizeChunks;
        const int maxX = minX + level.tileSizeChunks - 1;
        const int minZ = tileZ * level.tileSizeChunks;
        const int maxZ = minZ + level.tileSizeChunks - 1;

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

        const int minDistance = std::max(minAxisDistance(cameraChunk.x, minX, maxX),
                                         minAxisDistance(cameraChunk.z, minZ, maxZ));
        const int maxDistance = std::max(maxAxisDistance(cameraChunk.x, minX, maxX),
                                         maxAxisDistance(cameraChunk.z, minZ, maxZ));
        return maxDistance > level.innerRadiusChunks && minDistance <= level.outerRadiusChunks;
    }

    void initializeTile(FarTile& tile, const LevelConfig& level)
    {
        const int tileSpanBlocks = level.tileSizeChunks * kChunkSizeX;
        const float minX = static_cast<float>(tile.key.tileX * tileSpanBlocks);
        const float minZ = static_cast<float>(tile.key.tileZ * tileSpanBlocks);
        const float maxX = static_cast<float>((tile.key.tileX + 1) * tileSpanBlocks);
        const float maxZ = static_cast<float>((tile.key.tileZ + 1) * tileSpanBlocks);
        tile.level = level;
        tile.boundsMin = glm::vec3(minX, 0.0f, minZ);
        tile.boundsMax = glm::vec3(maxX, 1.0f, maxZ);
        tile.initialized = true;
    }

    void markDirty(FarTile& tile)
    {
        tile.dirty = true;
        ++tile.buildVersion;
        if (tile.buildVersion == 0)
        {
            tile.buildVersion = 1;
        }
    }

    BufferPage createBufferPage(std::size_t vertexCount, std::size_t indexCount)
    {
        static constexpr std::size_t kDefaultVertexCapacity = 131072;
        static constexpr std::size_t kDefaultIndexCapacity = 196608;

        BufferPage page;
        page.vertexCapacity = std::max(nextPowerOfTwo(vertexCount), kDefaultVertexCapacity);
        page.indexCapacity = std::max(nextPowerOfTwo(indexCount), kDefaultIndexCapacity);
        page.vertexBuffer = createDefaultBuffer(device_.Get(),
                                                static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex)),
                                                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        page.indexBuffer = createDefaultBuffer(device_.Get(),
                                               static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t)),
                                               D3D12_RESOURCE_STATE_INDEX_BUFFER);
        page.vertexUploadBuffer = createUploadBuffer(device_.Get(),
                                                     static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex)),
                                                     page.mappedVertexData);
        page.indexUploadBuffer = createUploadBuffer(device_.Get(),
                                                    static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t)),
                                                    page.mappedIndexData);
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

            allocation.pageIndex = pageIndex;
            allocation.vertexOffset = vertexOffset;
            allocation.indexOffset = indexOffset;
            return allocation;
        }

        BufferPage newPage = createBufferPage(vertexCount, indexCount);
        bufferPages_.push_back(std::move(newPage));
        BufferPage& page = bufferPages_.back();
        allocation.pageIndex = static_cast<std::uint32_t>(bufferPages_.size() - 1);
        tryAllocateRange(page.freeVertices, page.vertexCursor, page.vertexCapacity, vertexCount, allocation.vertexOffset);
        tryAllocateRange(page.freeIndices, page.indexCursor, page.indexCapacity, indexCount, allocation.indexOffset);
        return allocation;
    }

    void releaseTileGpu(FarTile& tile)
    {
        if (tile.pageIndex == kInvalidChunkBufferPage)
        {
            tile.vertexOffset = 0;
            tile.indexOffset = 0;
            tile.vertexCount = 0;
            tile.indexCount = 0;
            return;
        }

        if (tile.pageIndex < bufferPages_.size())
        {
            BufferPage& page = bufferPages_[tile.pageIndex];
            mergeRange(page.freeVertices, tile.vertexOffset, tile.vertexCount);
            mergeRange(page.freeIndices, tile.indexOffset, static_cast<std::size_t>(tile.indexCount));
        }

        tile.pageIndex = kInvalidChunkBufferPage;
        tile.vertexOffset = 0;
        tile.indexOffset = 0;
        tile.vertexCount = 0;
        tile.indexCount = 0;
    }

    void destroyBufferPages()
    {
        for (BufferPage& page : bufferPages_)
        {
            page.vertexBuffer.Reset();
            page.indexBuffer.Reset();
            page.vertexUploadBuffer.Reset();
            page.indexUploadBuffer.Reset();
            page.mappedVertexData = nullptr;
            page.mappedIndexData = nullptr;
        }
        bufferPages_.clear();
    }

    [[nodiscard]] static bool hasSolidSurface(const FarTerrainSurfaceSample& sample) noexcept
    {
        return sample.solidTopY != std::numeric_limits<int>::min() && sample.solidBlock != BlockId::Air;
    }

    [[nodiscard]] static bool isSubmergedSurface(const FarTerrainSurfaceSample& sample) noexcept
    {
        return sample.hasVisibleWater &&
               sample.waterTopY != std::numeric_limits<int>::min() &&
               hasSolidSurface(sample) &&
               sample.solidTopY < sample.waterTopY;
    }

    [[nodiscard]] static float sampleTopY(const FarTerrainSurfaceSample& sample, float fallbackY) noexcept
    {
        if (hasSolidSurface(sample))
        {
            return static_cast<float>(sample.solidTopY + 1);
        }
        return fallbackY;
    }

    [[nodiscard]] static glm::vec3 computeTerrainNormal(float h00,
                                                        float h10,
                                                        float h11,
                                                        float h01,
                                                        float step) noexcept
    {
        const float safeStep = std::max(step, 1.0f);
        const float dHdX = ((h10 - h00) + (h11 - h01)) * 0.5f / safeStep;
        const float dHdZ = ((h01 - h00) + (h11 - h10)) * 0.5f / safeStep;
        glm::vec3 normal(-dHdX, 1.0f, -dHdZ);
        if (glm::dot(normal, normal) <= kEpsilon)
        {
            return glm::vec3(0.0f, 1.0f, 0.0f);
        }
        return glm::normalize(normal);
    }

    [[nodiscard]] static BlockId dominantBlockForCell(const std::array<FarTerrainSurfaceSample, 4>& corners) noexcept
    {
        std::array<int, toIndex(BlockId::Count)> counts{};
        BlockId fallback = BlockId::Grass;
        for (const FarTerrainSurfaceSample& sample : corners)
        {
            if (!hasSolidSurface(sample))
            {
                continue;
            }

            fallback = sample.solidBlock;
            if (isSubmergedSurface(sample))
            {
                continue;
            }

            ++counts[toIndex(sample.solidBlock)];
        }

        int bestCount = 0;
        BlockId bestBlock = fallback;
        for (std::size_t i = 0; i < counts.size(); ++i)
        {
            if (counts[i] > bestCount)
            {
                bestCount = counts[i];
                bestBlock = static_cast<BlockId>(i);
            }
        }
        return bestBlock;
    }

public:
    [[nodiscard]] int readyTileCount() const noexcept
    {
        int ready = 0;
        for (const auto& [key, tile] : tiles_)
        {
            (void)key;
            if (tile.active && tile.indexCount > 0)
            {
                ++ready;
            }
        }
        return ready;
    }

    [[nodiscard]] int queuedTileCount() const noexcept
    {
        int queued = 0;
        for (const auto& [key, tile] : tiles_)
        {
            (void)key;
            if (tile.active && tile.inFlight)
            {
                ++queued;
            }
        }
        return queued;
    }

    [[nodiscard]] int pendingUploadTileCount() const noexcept
    {
        std::lock_guard<std::mutex> lock(completedMutex_);
        return static_cast<int>(completedBuilds_.size());
    }

    [[nodiscard]] double averageBuildMs() const noexcept
    {
        return lastAverageBuildMs_;
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
            std::lock_guard<std::mutex> lock(buildQueueMutex_);
            buildQueue_.clear();
            queuedKeys_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(completedMutex_);
            completedBuilds_.clear();
        }
    }

    void workerThreadLoop()
    {
        while (true)
        {
            BuildJob job{};
            {
                std::unique_lock<std::mutex> lock(buildQueueMutex_);
                buildQueueCv_.wait(lock,
                                   [this]()
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

            SampleFn sampleFn;
            UvLookupFn uvLookup;
            {
                std::lock_guard<std::mutex> lock(configMutex_);
                sampleFn = sampleFn_;
                uvLookup = uvLookupFn_;
            }

            if (!sampleFn || !uvLookup)
            {
                continue;
            }

            BuildResult result = buildTileMesh(job.key, job.level, sampleFn, uvLookup);
            result.buildVersion = job.buildVersion;
            result.epoch = job.epoch;

            if (job.epoch != buildEpoch_.load(std::memory_order_acquire))
            {
                continue;
            }

            std::lock_guard<std::mutex> completedLock(completedMutex_);
            completedBuilds_.push_back(std::move(result));
        }
    }

    void scheduleDirtyBuilds()
    {
        const std::size_t maxQueued = std::max<std::size_t>(4, std::max<std::size_t>(workerCount_, 1) * 3);
        std::size_t inFlightCount = 0;
        std::vector<CandidateBuild> candidates;
        candidates.reserve(tiles_.size());

        const glm::vec2 cameraCenter(static_cast<float>(cameraChunk_.x * kChunkSizeX),
                                     static_cast<float>(cameraChunk_.z * kChunkSizeZ));
        glm::vec2 forwardXZ(cameraForward_.x, cameraForward_.z);
        if (glm::dot(forwardXZ, forwardXZ) <= kEpsilon)
        {
            forwardXZ = glm::vec2(0.0f, -1.0f);
        }
        else
        {
            forwardXZ = glm::normalize(forwardXZ);
        }

        for (const auto& [key, tile] : tiles_)
        {
            if (!tile.active)
            {
                continue;
            }
            if (tile.inFlight)
            {
                ++inFlightCount;
                continue;
            }
            if (!tile.dirty)
            {
                continue;
            }

            const int tileSpanBlocks = tile.level.tileSizeChunks * kChunkSizeX;
            const glm::vec2 tileCenter(static_cast<float>(key.tileX * tileSpanBlocks + tileSpanBlocks / 2),
                                       static_cast<float>(key.tileZ * tileSpanBlocks + tileSpanBlocks / 2));
            const glm::vec2 delta = tileCenter - cameraCenter;
            const float distanceSq = glm::dot(delta, delta);
            float facingDot = 1.0f;
            if (distanceSq > kEpsilon)
            {
                facingDot = glm::dot(glm::normalize(delta), forwardXZ);
            }

            int forwardBucket = 2;
            if (facingDot >= 0.5f)
            {
                forwardBucket = 0;
            }
            else if (facingDot >= -0.2f)
            {
                forwardBucket = 1;
            }

            candidates.push_back(CandidateBuild{key, distanceSq, forwardBucket});
        }

        if (inFlightCount >= maxQueued || candidates.empty())
        {
            return;
        }

        std::sort(candidates.begin(),
                  candidates.end(),
                  [](const CandidateBuild& lhs, const CandidateBuild& rhs)
                  {
                      if (lhs.forwardBucket != rhs.forwardBucket)
                      {
                          return lhs.forwardBucket < rhs.forwardBucket;
                      }
                      return lhs.distanceSq < rhs.distanceSq;
                  });

        std::size_t available = maxQueued - inFlightCount;
        for (const CandidateBuild& candidate : candidates)
        {
            if (available == 0)
            {
                break;
            }

            auto tileIt = tiles_.find(candidate.key);
            if (tileIt == tiles_.end() || !tileIt->second.active || !tileIt->second.dirty || tileIt->second.inFlight)
            {
                continue;
            }

            bool queued = false;
            {
                std::lock_guard<std::mutex> lock(buildQueueMutex_);
                if (queuedKeys_.insert(candidate.key).second)
                {
                    buildQueue_.push_back(BuildJob{
                        candidate.key,
                        tileIt->second.level,
                        tileIt->second.buildVersion,
                        buildEpoch_.load(std::memory_order_acquire)});
                    queued = true;
                }
            }

            if (!queued)
            {
                continue;
            }

            tileIt->second.inFlight = true;
            --available;
            buildQueueCv_.notify_one();
        }
    }

    void collectCompletedBuilds(double uploadBudgetMs)
    {
        const auto collectStart = std::chrono::steady_clock::now();
        const auto uploadStart = std::chrono::steady_clock::now();
        if (uploadContext_.ready())
        {
            uploadContext_.begin();
        }
        while (true)
        {
            BuildResult result{};
            {
                std::lock_guard<std::mutex> lock(completedMutex_);
                if (completedBuilds_.empty())
                {
                    break;
                }

                result = std::move(completedBuilds_.front());
                completedBuilds_.pop_front();
            }

            if (result.epoch != buildEpoch_.load(std::memory_order_acquire))
            {
                continue;
            }

            auto tileIt = tiles_.find(result.key);
            if (tileIt == tiles_.end())
            {
                continue;
            }

            FarTile& tile = tileIt->second;
            tile.inFlight = false;
            if (!tile.active || tile.buildVersion != result.buildVersion)
            {
                continue;
            }

            uploadBuiltTile(tile, result.mesh);
            tile.dirty = false;
            ++builtTilesLastUpdate_;
            if (builtTilesLastUpdate_ == 1)
            {
                lastAverageBuildMs_ = result.buildMs;
            }
            else
            {
                lastAverageBuildMs_ = (lastAverageBuildMs_ * 0.65) + (result.buildMs * 0.35);
            }

            const double elapsedMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - uploadStart).count();
            if (elapsedMs >= uploadBudgetMs && builtTilesLastUpdate_ > 0)
            {
                break;
            }
        }
        uploadContext_.flush();
        lastCollectMs_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - collectStart).count();
        lastUploadMs_ =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - uploadStart).count();
    }

    void uploadBuiltTile(FarTile& tile, const TileMesh& mesh)
    {
        releaseTileGpu(tile);
        tile.boundsMin = mesh.boundsMin;
        tile.boundsMax = mesh.boundsMax;

        if (mesh.vertices.empty() || mesh.indices.empty())
        {
            return;
        }

        Allocation allocation = acquireAllocation(mesh.vertices.size(), mesh.indices.size());
        if (allocation.pageIndex == kInvalidChunkBufferPage || allocation.pageIndex >= bufferPages_.size())
        {
            return;
        }

        tile.pageIndex = allocation.pageIndex;
        tile.vertexOffset = allocation.vertexOffset;
        tile.indexOffset = allocation.indexOffset;
        tile.vertexCount = mesh.vertices.size();
        tile.indexCount = static_cast<std::uint32_t>(mesh.indices.size());

        BufferPage& page = bufferPages_[allocation.pageIndex];
        if (page.mappedVertexData != nullptr && !mesh.vertices.empty())
        {
            std::memcpy(page.mappedVertexData + tile.vertexOffset * sizeof(Vertex),
                        mesh.vertices.data(),
                        mesh.vertices.size() * sizeof(Vertex));
            if (uploadContext_.ready() && page.vertexUploadBuffer != nullptr && page.vertexBuffer != nullptr)
            {
                uploadContext_.transition(page.vertexBuffer.Get(),
                                          page.vertexState,
                                          D3D12_RESOURCE_STATE_COPY_DEST);
                page.vertexState = D3D12_RESOURCE_STATE_COPY_DEST;
                uploadContext_.copyBuffer(page.vertexBuffer.Get(),
                                          static_cast<std::uint64_t>(tile.vertexOffset * sizeof(Vertex)),
                                          page.vertexUploadBuffer.Get(),
                                          static_cast<std::uint64_t>(tile.vertexOffset * sizeof(Vertex)),
                                          static_cast<std::uint64_t>(mesh.vertices.size() * sizeof(Vertex)));
                uploadContext_.transition(page.vertexBuffer.Get(),
                                          page.vertexState,
                                          D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
                page.vertexState = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
            }
        }
        if (page.mappedIndexData != nullptr && !mesh.indices.empty())
        {
            std::memcpy(page.mappedIndexData + tile.indexOffset * sizeof(std::uint32_t),
                        mesh.indices.data(),
                        mesh.indices.size() * sizeof(std::uint32_t));
            if (uploadContext_.ready() && page.indexUploadBuffer != nullptr && page.indexBuffer != nullptr)
            {
                uploadContext_.transition(page.indexBuffer.Get(),
                                          page.indexState,
                                          D3D12_RESOURCE_STATE_COPY_DEST);
                page.indexState = D3D12_RESOURCE_STATE_COPY_DEST;
                uploadContext_.copyBuffer(page.indexBuffer.Get(),
                                          static_cast<std::uint64_t>(tile.indexOffset * sizeof(std::uint32_t)),
                                          page.indexUploadBuffer.Get(),
                                          static_cast<std::uint64_t>(tile.indexOffset * sizeof(std::uint32_t)),
                                          static_cast<std::uint64_t>(mesh.indices.size() * sizeof(std::uint32_t)));
                uploadContext_.transition(page.indexBuffer.Get(),
                                          page.indexState,
                                          D3D12_RESOURCE_STATE_INDEX_BUFFER);
                page.indexState = D3D12_RESOURCE_STATE_INDEX_BUFFER;
            }
        }
    }

    static BuildResult buildTileMesh(const FarTileKey& key,
                                     const LevelConfig& level,
                                     const SampleFn& sampleFn,
                                     const UvLookupFn& uvLookup)
    {
        BuildResult result{};
        result.key = key;

        const auto start = std::chrono::steady_clock::now();
        const int tileSpanBlocks = level.tileSizeChunks * kChunkSizeX;
        const int gridCount = std::max(1, tileSpanBlocks / std::max(level.sampleStepBlocks, 1));
        const int worldMinX = key.tileX * tileSpanBlocks;
        const int worldMinZ = key.tileZ * tileSpanBlocks;
        const float step = static_cast<float>(level.sampleStepBlocks);

        std::vector<FarTerrainSurfaceSample> vertexSamples(
            static_cast<std::size_t>((gridCount + 1) * (gridCount + 1)));
        auto sampleAt = [&](int x, int z) -> FarTerrainSurfaceSample& {
            return vertexSamples[static_cast<std::size_t>(z * (gridCount + 1) + x)];
        };

        int minY = std::numeric_limits<int>::max();
        int maxY = std::numeric_limits<int>::min();
        for (int z = 0; z <= gridCount; ++z)
        {
            for (int x = 0; x <= gridCount; ++x)
            {
                const int worldX = worldMinX + x * level.sampleStepBlocks;
                const int worldZ = worldMinZ + z * level.sampleStepBlocks;
                FarTerrainSurfaceSample sample = sampleFn(worldX, worldZ, level.lodLevel);
                sampleAt(x, z) = sample;
                if (hasSolidSurface(sample))
                {
                    minY = std::min(minY, sample.solidTopY);
                    maxY = std::max(maxY, sample.solidTopY + 1);
                }
                if (sample.hasVisibleWater)
                {
                    minY = std::min(minY, sample.waterTopY);
                    maxY = std::max(maxY, sample.waterTopY + 1);
                }
            }
        }

        auto& vertices = result.mesh.vertices;
        auto& indices = result.mesh.indices;
        vertices.reserve(static_cast<std::size_t>(gridCount * gridCount * 24));
        indices.reserve(static_cast<std::size_t>(gridCount * gridCount * 36));

        for (int z = 0; z < gridCount; ++z)
        {
            for (int x = 0; x < gridCount; ++x)
            {
                const float minX = static_cast<float>(worldMinX) + static_cast<float>(x) * step;
                const float maxX = minX + step;
                const float minZ = static_cast<float>(worldMinZ) + static_cast<float>(z) * step;
                const float maxZ = minZ + step;
                const FarTerrainSurfaceSample& s00 = sampleAt(x, z);
                const FarTerrainSurfaceSample& s10 = sampleAt(x + 1, z);
                const FarTerrainSurfaceSample& s11 = sampleAt(x + 1, z + 1);
                const FarTerrainSurfaceSample& s01 = sampleAt(x, z + 1);
                const std::array<FarTerrainSurfaceSample, 4> corners{s00, s10, s11, s01};

                bool hasVisibleTerrain = false;
                bool allSubmerged = true;
                float fallbackHeight = 0.0f;
                for (const FarTerrainSurfaceSample& sample : corners)
                {
                    if (hasSolidSurface(sample))
                    {
                        hasVisibleTerrain = true;
                        fallbackHeight = static_cast<float>(sample.solidTopY + 1);
                    }
                    if (!isSubmergedSurface(sample))
                    {
                        allSubmerged = false;
                    }
                }

                if (hasVisibleTerrain && !allSubmerged)
                {
                    const float h00 = sampleTopY(s00, fallbackHeight);
                    const float h10 = sampleTopY(s10, fallbackHeight);
                    const float h11 = sampleTopY(s11, fallbackHeight);
                    const float h01 = sampleTopY(s01, fallbackHeight);
                    const glm::vec3 normal = computeTerrainNormal(h00, h10, h11, h01, step);
                    const BlockId surfaceBlock = dominantBlockForCell(corners);

                    appendQuad(vertices,
                               indices,
                               glm::vec3(minX, h00, minZ),
                               glm::vec3(maxX, h10, minZ),
                               glm::vec3(maxX, h11, maxZ),
                               glm::vec3(minX, h01, maxZ),
                               normal,
                               uvLookup(surfaceBlock, BlockFace::Top));

                    const float skirtDepth = static_cast<float>(level.skirtDepthBlocks);
                    if (x == 0)
                    {
                        appendQuad(vertices,
                                   indices,
                                   glm::vec3(minX, h01 - skirtDepth, maxZ),
                                   glm::vec3(minX, h00 - skirtDepth, minZ),
                                   glm::vec3(minX, h00, minZ),
                                   glm::vec3(minX, h01, maxZ),
                                   glm::vec3(-1.0f, 0.0f, 0.0f),
                                   uvLookup(surfaceBlock, BlockFace::West));
                    }
                    if (x == gridCount - 1)
                    {
                        appendQuad(vertices,
                                   indices,
                                   glm::vec3(maxX, h10 - skirtDepth, minZ),
                                   glm::vec3(maxX, h11 - skirtDepth, maxZ),
                                   glm::vec3(maxX, h11, maxZ),
                                   glm::vec3(maxX, h10, minZ),
                                   glm::vec3(1.0f, 0.0f, 0.0f),
                                   uvLookup(surfaceBlock, BlockFace::East));
                    }
                    if (z == 0)
                    {
                        appendQuad(vertices,
                                   indices,
                                   glm::vec3(maxX, h10 - skirtDepth, minZ),
                                   glm::vec3(minX, h00 - skirtDepth, minZ),
                                   glm::vec3(minX, h00, minZ),
                                   glm::vec3(maxX, h10, minZ),
                                   glm::vec3(0.0f, 0.0f, -1.0f),
                                   uvLookup(surfaceBlock, BlockFace::North));
                    }
                    if (z == gridCount - 1)
                    {
                        appendQuad(vertices,
                                   indices,
                                   glm::vec3(minX, h01 - skirtDepth, maxZ),
                                   glm::vec3(maxX, h11 - skirtDepth, maxZ),
                                   glm::vec3(maxX, h11, maxZ),
                                   glm::vec3(minX, h01, maxZ),
                                   glm::vec3(0.0f, 0.0f, 1.0f),
                                   uvLookup(surfaceBlock, BlockFace::South));
                    }
                }

                bool hasWater = false;
                int waterTopY = std::numeric_limits<int>::min();
                for (const FarTerrainSurfaceSample& sample : corners)
                {
                    if (!sample.hasVisibleWater)
                    {
                        continue;
                    }
                    hasWater = true;
                    waterTopY = std::max(waterTopY, sample.waterTopY);
                }

                if (hasWater && waterTopY != std::numeric_limits<int>::min())
                {
                    const float waterY = static_cast<float>(waterTopY + 1);
                    appendQuad(vertices,
                               indices,
                               glm::vec3(minX, waterY, minZ),
                               glm::vec3(maxX, waterY, minZ),
                               glm::vec3(maxX, waterY, maxZ),
                               glm::vec3(minX, waterY, maxZ),
                               glm::vec3(0.0f, 1.0f, 0.0f),
                               uvLookup(BlockId::Water, BlockFace::Top));
                }
            }
        }

        result.mesh.boundsMin = glm::vec3(static_cast<float>(worldMinX),
                                          static_cast<float>((minY == std::numeric_limits<int>::max()) ? 0 : minY - level.skirtDepthBlocks),
                                          static_cast<float>(worldMinZ));
        result.mesh.boundsMax = glm::vec3(static_cast<float>(worldMinX + tileSpanBlocks),
                                          static_cast<float>((maxY == std::numeric_limits<int>::min()) ? 1 : maxY),
                                          static_cast<float>(worldMinZ + tileSpanBlocks));
        result.buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return result;
    }

    bool enabled_{true};
    int farDistanceBlocks_{kDefaultFarRenderDistanceBlocks};
    int fogStartBlocks_{kDefaultFarFogStartBlocks};
    glm::ivec3 cameraChunk_{0};
    glm::vec3 cameraForward_{0.0f, 0.0f, -1.0f};
    std::uint64_t updateStamp_{0};
    int builtTilesLastUpdate_{0};
    double lastAverageBuildMs_{0.0};
    double lastCollectMs_{0.0};
    double lastUploadMs_{0.0};
    std::vector<LevelConfig> levels_;
    std::vector<BufferPage> bufferPages_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    UploadContext uploadContext_{};
    std::unordered_map<FarTileKey, FarTile, FarTileKeyHasher> tiles_;
    std::mutex configMutex_;
    SampleFn sampleFn_{};
    UvLookupFn uvLookupFn_{};
    std::mutex buildQueueMutex_;
    std::condition_variable buildQueueCv_;
    std::deque<BuildJob> buildQueue_;
    std::unordered_set<FarTileKey, FarTileKeyHasher> queuedKeys_;
    mutable std::mutex completedMutex_;
    std::deque<BuildResult> completedBuilds_;
    std::vector<std::thread> workerThreads_;
    std::atomic<bool> stopWorkers_{false};
    std::atomic<std::uint64_t> buildEpoch_{1};
    std::size_t workerCount_{1};
};


} // namespace

struct ChunkManager::Impl
{
    explicit Impl(unsigned seed);
    ~Impl();

    void initializeRendering(ID3D12Device* device);
    void setBlockTextureAtlasConfig(const BlockTextureAtlasConfig& config);
    void update(const glm::vec3& cameraPos);
    void update(const glm::vec3& cameraPos, const glm::vec3& cameraForward);
    WorldRenderData buildRenderData(const Frustum& frustum) const;

    float surfaceHeight(float worldX, float worldZ) const noexcept;
    ColumnSample sampleColumnAt(const glm::vec3& worldPos,
                                int slabMinWorldY = std::numeric_limits<int>::min(),
                                int slabMaxWorldY = std::numeric_limits<int>::max()) const;
    void clear();

    bool destroyBlock(const glm::ivec3& worldPos);
    bool placeBlock(const glm::ivec3& targetBlockPos, const glm::ivec3& faceNormal, BlockId block);

    RaycastHit raycast(const glm::vec3& origin, const glm::vec3& direction) const;
    void updateHighlight(const glm::vec3& cameraPos, const glm::vec3& cameraDirection);

    void toggleViewDistance();
    int viewDistance() const noexcept;
    int nearRenderDistance() const noexcept;
    int farRenderDistanceBlocks() const noexcept;
    RenderDistanceSettings renderDistanceSettings() const noexcept;
    void setRenderDistance(int distance) noexcept;
    void setNearRenderDistance(int chunks) noexcept;
    void setFarRenderDistanceBlocks(int blocks) noexcept;
    void setFogStartBlocks(int blocks) noexcept;
    void setLodEnabled(bool enabled);
    bool lodEnabled() const noexcept;
    void setFarTerrainEnabled(bool enabled);
    bool farTerrainEnabled() const noexcept;

    BlockId blockAt(const glm::ivec3& worldPos) const noexcept;
    LightSample lightAt(const glm::ivec3& worldPos) const noexcept;
    glm::vec3 findSafeSpawnPosition(float worldX, float worldZ) const;
    void beginSpawnPreload(const glm::vec3& spawnPos);
    bool isSpawnPreloadReady() const noexcept;
    bool playerReleaseReady() const noexcept;
    StreamingPhase streamingPhase() const noexcept;
    void setStartupEnabled(bool enabled) noexcept;
    bool startupEnabled() const noexcept;
    StreamingStatusSnapshot streamingStatusSnapshot() const noexcept;
    ChunkProfilingSnapshot sampleProfilingSnapshot();
    std::string biomeNameAt(const glm::vec3& worldPos) const;

private:
    struct TreeDensityNoise
    {
        TreeDensityNoise() = default;

        explicit TreeDensityNoise(unsigned seed)
        {
            reseed(seed);
        }

        void reseed(unsigned seed)
        {
            seed_ = seed;

            std::mt19937 rng(seed_);
            std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
            for (auto& offset : octaveOffsets_)
            {
                offset = {dist(rng), dist(rng)};
            }
        }

        [[nodiscard]] float fbm(float x,
                                float y,
                                int octaves,
                                float persistence,
                                float lacunarity) const noexcept
        {
            float amplitude = 1.0f;
            float frequency = 1.0f;
            float value = 0.0f;
            float normalization = 0.0f;

            const int octaveCount = std::min<int>(octaves, static_cast<int>(octaveOffsets_.size()));
            for (int i = 0; i < octaveCount; ++i)
            {
                const glm::vec2 sample{x * frequency + octaveOffsets_[i].x,
                                       y * frequency + octaveOffsets_[i].y};
                value += glm::perlin(sample) * amplitude;
                normalization += amplitude;

                amplitude *= persistence;
                frequency *= lacunarity;
            }

            if (normalization > 0.0f)
            {
                value /= normalization;
            }

            return value;
        }

    private:
        unsigned seed_{0};
        std::array<glm::vec2, 16> octaveOffsets_{};
    };

    terrain::WorldgenProfile worldgenProfile_{};
    terrain::BiomeDatabase biomeDatabase_;
    std::unique_ptr<terrain::ClimateMap> climateMap_;
    std::unique_ptr<terrain::SurfaceMap> surfaceMap_;
    std::unique_ptr<terrain::TerrainGenerator> terrainGenerator_;
    int globalSeaLevel_{20};
    TreeDensityNoise noise_{};

    void startWorkerThreads();
    void stopWorkerThreads();
    void workerThreadFunction();
    void enqueueJob(const std::shared_ptr<Chunk>& chunk, JobType type, const glm::ivec3& coord);
    void processJob(const Job& job);
    std::shared_ptr<Chunk> popNextChunkForUpload();
    void queueChunkForUpload(const std::shared_ptr<Chunk>& chunk);
    void requeueChunkForUpload(const std::shared_ptr<Chunk>& chunk, bool toFront);

    struct ChunkBufferPage
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
        D3D12_VERTEX_BUFFER_VIEW vertexView{};
        D3D12_INDEX_BUFFER_VIEW indexView{};
        std::byte* mappedVertexData{nullptr};
        std::byte* mappedIndexData{nullptr};
        D3D12_RESOURCE_STATES vertexState{D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER};
        D3D12_RESOURCE_STATES indexState{D3D12_RESOURCE_STATE_INDEX_BUFFER};
        std::size_t vertexCapacity{0};
        std::size_t indexCapacity{0};
        std::size_t vertexCursor{0};
        std::size_t indexCursor{0};
        std::vector<Range> freeVertices;
        std::vector<Range> freeIndices;
        std::size_t activeChunks{0};
    };

    struct ChunkAllocation
    {
        std::uint32_t pageIndex{kInvalidChunkBufferPage};
        std::size_t vertexOffset{0};
        std::size_t indexOffset{0};
    };

    static std::size_t nextPowerOfTwo(std::size_t value) noexcept;
    ChunkBufferPage createBufferPage(std::size_t vertexCount, std::size_t indexCount);
    ChunkAllocation acquireChunkAllocation(std::size_t vertexCount, std::size_t indexCount);
    void releaseChunkAllocation(Chunk& chunk);
    void recycleChunkGPU(Chunk& chunk);
    void destroyBufferPages();
    int computeVerticalRadius(const glm::ivec3& center, int horizontalRadius, int cameraWorldY);
    int columnRadiusFor(const glm::ivec2& column,
                        const glm::ivec2& cameraColumn,
                        int cameraChunkY,
                        int verticalRadius) const;
    int columnRadiusForHeight(const glm::ivec2& column,
                              const glm::ivec2& cameraColumn,
                              int cameraChunkY,
                              int verticalRadius,
                              int columnHeight) const;
    std::pair<int, int> columnSpanFor(const glm::ivec2& column,
                                      const glm::ivec2& cameraColumn,
                                      int cameraChunkY,
                                      int verticalRadius) const;
    std::pair<int, int> columnSpanForHeight(const glm::ivec2& column,
                                            const glm::ivec2& cameraColumn,
                                            int cameraChunkY,
                                            int verticalRadius,
                                            int columnHeight) const;
    void resetColumnBudgets();
    int baseUploadsPerColumnLimit(int verticalRadius) const noexcept;
    std::size_t estimateUploadQueueSize();
    struct UploadBudgets
    {
        std::size_t byteBudget{kUploadBudgetBytesPerFrame};
        int columnLimit{kVerticalStreamingConfig.uploadBasePerColumn};
        std::size_t queueSize{0};
        double timeBudgetMs{4.0};
    };
    UploadBudgets computeUploadBudgets(int verticalRadius);
    static int computeBacklogSteps(int backlog, int threshold, int stepSize) noexcept;
    int computeGenerationBudget(int horizontalRadius, int verticalRadius, int backlogSteps) const;
    int computeRingExpansionBudget(int backlogChunks) const;
    int computeColumnJobCap(int backlogSteps, int backlogChunks) const;
    int estimateMissingChunks(const glm::ivec3& center, int horizontalRadius, int verticalRadius) const;
    StreamingStatusSnapshot computeStreamingStatusSnapshot() const noexcept;

    struct RingProgress
    {
        bool fullyLoaded{false};
        bool budgetExhausted{false};
    };

    RingProgress ensureVolume(const glm::ivec3& center, int horizontalRadius, int verticalRadius, int& jobBudget);
    void removeDistantChunks(const glm::ivec3& center, int horizontalThreshold, int verticalThreshold);
    bool ensureChunkAsync(const glm::ivec3& coord, bool surfaceOnly);
    void uploadReadyMeshes();
    void uploadChunkMesh(Chunk& chunk);
    void buildChunkMeshAsync(Chunk& chunk);
    static glm::ivec3 worldToChunkCoords(int worldX, int worldY, int worldZ) noexcept;
    std::shared_ptr<Chunk> acquireChunk(const glm::ivec3& coord);

    std::shared_ptr<Chunk> getChunkShared(const glm::ivec3& coord) noexcept;
    std::shared_ptr<const Chunk> getChunkShared(const glm::ivec3& coord) const noexcept;
    Chunk* getChunk(const glm::ivec3& coord) noexcept;
    const Chunk* getChunk(const glm::ivec3& coord) const noexcept;
    void requestChunkRemesh(const std::shared_ptr<Chunk>& chunk);
    void markNeighborsForRemeshingIfNeeded(const glm::ivec3& coord, int localX, int localY, int localZ);
    void relightAroundChunk(const glm::ivec3& centerCoord);
    void queueChunkForLightingRemesh(const std::shared_ptr<Chunk>& chunk);
    std::uint8_t packedLightAtWorld(const glm::ivec3& worldPos) const noexcept;
    void generateChunkBlocks(Chunk& chunk);
    ColumnSample sampleColumn(int worldX,
                              int worldZ,
                              int slabMinWorldY = std::numeric_limits<int>::min(),
                              int slabMaxWorldY = std::numeric_limits<int>::max()) const;
    FarTerrainSurfaceSample sampleFarTerrainSurfaceLod(int worldX, int worldZ, int lodLevel) const;
    int ensureColumnHeightCached(const glm::ivec2& column, int worldX, int worldZ) const;
    bool tryGetPredictedColumnHeight(const glm::ivec2& column, int& outHeight) const;
    int cacheSampledColumnHeight(const glm::ivec2& column, int worldX, int worldZ) const;
    void invalidatePredictedColumn(const glm::ivec2& column) const;
    bool applyPendingStructureEditsLocked(Chunk& chunk);
    void dispatchStructureEdits(const std::vector<PendingStructureEdit>& edits);
    static bool chunkHasSolidBlocks(const Chunk& chunk) noexcept;
    void recycleChunkObject(std::shared_ptr<Chunk> chunk);
    void buildSurfaceOnlyMesh(Chunk& chunk);
    void generateSurfaceOnlyChunk(Chunk& chunk);
    bool shouldUseSurfaceOnly(const glm::ivec3& center, const glm::ivec3& coord) const noexcept;
    std::pair<glm::vec2, glm::vec2> atlasUvFor(BlockId block, BlockFace face) const;

    glm::ivec2 atlasTextureSizePixels_{1, 1};
    int atlasTileSizePixels_{kAtlasTileSizePixels};
    int atlasTileStridePixels_{kAtlasTileSizePixels};
    int atlasTilePaddingPixels_{0};
    struct FaceUV
    {
        glm::vec2 base{0.0f};
        glm::vec2 size{1.0f};
    };

    struct BlockUVSet
    {
        std::array<FaceUV, kBlockFaceCount> faces{};
    };

    std::array<BlockUVSet, toIndex(BlockId::Count)> blockUVTable_{};
    bool blockAtlasConfigured_{false};
    RenderDistanceSettings renderSettings_{};
    FarTerrainManager farTerrainManager_{};

    struct StartupStreamingState
    {
        StreamingPhase phase{StreamingPhase::SpawnResolve};
        double phaseTimeSeconds{0.0};
        double totalTimeSeconds{0.0};
        double healthyTimeSeconds{0.0};
        int exactNearCurrentChunks{0};
        int farCurrentBlocks{0};
        bool preloadStarted{false};
        bool playerReleaseReady{false};
        glm::ivec3 spawnChunk{0};
    };

    std::deque<std::weak_ptr<Chunk>> uploadQueue_;
    std::mutex uploadQueueMutex_;
    std::vector<ChunkBufferPage> bufferPages_;
    mutable std::mutex bufferPageMutex_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    UploadContext uploadContext_{};

    std::unordered_map<glm::ivec3, std::shared_ptr<Chunk>, ChunkHasher> chunks_;
    mutable std::mutex chunksMutex;
    const glm::vec3 lightDirection_{glm::normalize(glm::vec3(0.5f, -1.0f, 0.2f))};
    JobQueue jobQueue_;
    ColumnManager columnManager_;
    mutable std::mutex predictedColumnMutex_;
    mutable std::unordered_map<glm::ivec2, int, ColumnHasher> predictedColumnHeights_;
    std::unordered_map<glm::ivec3, std::vector<PendingStructureEdit>, ChunkHasher> pendingStructureEdits_;
    mutable std::mutex pendingStructureMutex_;

    std::vector<std::thread> workerThreads_;
    std::size_t workerThreadCount_{0};
    std::atomic<bool> shouldStop_;

    glm::ivec3 highlightedBlock_{0};
    bool hasHighlight_{false};

    int viewDistance_;
    int targetViewDistance_;
    std::vector<std::shared_ptr<Chunk>> chunkPool_;
    std::mutex chunkPoolMutex_;
    ProfilingCounters profilingCounters_{};
    std::unordered_map<glm::ivec2, int, ColumnHasher> jobsScheduledThisFrame_{};
    int lastVerticalRadius_{kVerticalStreamingConfig.minRadiusChunks};
    int uploadColumnLimitThisFrame_{kVerticalStreamingConfig.uploadBasePerColumn};
    std::size_t uploadBudgetBytesThisFrame_{kUploadBudgetBytesPerFrame};
    double uploadBudgetMsThisFrame_{4.0};
    double updateMsLastFrame_{0.0};
    std::size_t lastUploadBytesUsed_{0};
    std::size_t pendingUploadsLastFrame_{0};
    int generationColumnCapThisFrame_{kVerticalStreamingConfig.maxGenerationJobsPerColumn};
    int lastGenerationBudget_{kVerticalStreamingConfig.generationBudget.baseJobsPerFrame};
    int lastGenerationJobsIssued_{0};
    int lastRingBudget_{kVerticalStreamingConfig.generationBudget.minRingExpansionsPerFrame};
    int lastRingExpansionsUsed_{0};
    int lastMissingChunks_{0};
    int lastColumnCap_{kVerticalStreamingConfig.maxGenerationJobsPerColumn};
    int lastBacklogSteps_{0};
    bool startupEnabled_{true};
    StartupStreamingState startupState_{};
    glm::vec3 lastCameraForward_{0.0f, 0.0f, -1.0f};
    glm::ivec3 lastCenterChunk_{0};
    std::chrono::steady_clock::time_point lastUpdateTime_{};
    double smoothedFrameMs_{16.0};
    double lastUploadMsUsed_{0.0};
    int farWorkerCount_{1};
    int lastLoggedGenerationBudget_{-1};
    int lastLoggedRingBudget_{-1};
    int lastLoggedColumnCap_{-1};
};

// JobQueue implementations

void JobQueue::push(const Job& job)
{
    std::lock_guard<std::mutex> lock(mutex_);
    priorityQueue_.push(wrap(job));
    condition_.notify_one();
}

bool JobQueue::tryPop(Job& job)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (priorityQueue_.empty())
    {
        return false;
    }
    job = priorityQueue_.top().job;
    priorityQueue_.pop();
    return true;
}

Job JobQueue::waitAndPop()
{
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !priorityQueue_.empty() || shouldStop_.load(std::memory_order_acquire); });

    if (shouldStop_.load(std::memory_order_acquire) && priorityQueue_.empty())
    {
        throw std::runtime_error("Job queue stopped");
    }

    Job job = priorityQueue_.top().job;
    priorityQueue_.pop();
    return job;
}

void JobQueue::stop()
{
    std::lock_guard<std::mutex> lock(mutex_);
    shouldStop_.store(true, std::memory_order_release);
    condition_.notify_all();
}

bool JobQueue::empty() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return priorityQueue_.empty();
}

void JobQueue::updatePriorityOrigin(const glm::ivec3& origin)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (origin == priorityOrigin_)
    {
        return;
    }

    priorityOrigin_ = origin;
    rebuildLocked();
}

bool JobQueue::JobComparer::operator()(const PrioritizedJob& lhs, const PrioritizedJob& rhs) const
{
    if (lhs.distance != rhs.distance)
    {
        return lhs.distance > rhs.distance;
    }
    if (lhs.priorityBias != rhs.priorityBias)
    {
        return lhs.priorityBias > rhs.priorityBias;
    }
    return lhs.sequence > rhs.sequence;
}

JobQueue::PrioritizedJob JobQueue::wrap(const Job& job)
{
    const int distance = manhattanDistance(job.chunkCoord, priorityOrigin_);
    const int bias = (job.type == JobType::Mesh) ? 0 : 1;
    const std::uint64_t sequence = nextSequence_++;
    return PrioritizedJob{job, distance, bias, sequence};
}

int JobQueue::manhattanDistance(const glm::ivec3& a, const glm::ivec3& b) noexcept
{
    return std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z);
}

void JobQueue::rebuildLocked()
{
    if (priorityQueue_.empty())
    {
        return;
    }

    std::vector<PrioritizedJob> jobs;
    jobs.reserve(priorityQueue_.size());
    while (!priorityQueue_.empty())
    {
        jobs.push_back(priorityQueue_.top());
        priorityQueue_.pop();
    }

    for (auto& prioritized : jobs)
    {
        prioritized.distance = manhattanDistance(prioritized.job.chunkCoord, priorityOrigin_);
        priorityQueue_.push(std::move(prioritized));
    }
}

glm::ivec2 ColumnManager::columnKey(const glm::ivec3& chunkCoord, int localX, int localZ) noexcept
{
    return {chunkCoord.x * kChunkSizeX + localX, chunkCoord.z * kChunkSizeZ + localZ};
}

int ColumnManager::scanColumnHighestWorld(const Chunk& chunk, int localX, int localZ) noexcept
{
    for (int y = kChunkSizeY - 1; y >= 0; --y)
    {
        if (isSolid(chunk.blocks[blockIndex(localX, y, localZ)]))
        {
            return chunk.minWorldY + y;
        }
    }
    return kNoHeight;
}

int ColumnManager::computeHighest(const ColumnData& data) noexcept
{
    int highest = kNoHeight;
    for (const auto& entry : data.slabHeights)
    {
        highest = std::max(highest, entry.second);
    }
    return highest;
}

void ColumnManager::applyHeightLocked(const glm::ivec2& key, int chunkY, int highestWorldY)
{
    if (highestWorldY == kNoHeight)
    {
        auto it = columns_.find(key);
        if (it == columns_.end())
        {
            return;
        }

        it->second.slabHeights.erase(chunkY);
        if (it->second.slabHeights.empty())
        {
            columns_.erase(it);
        }
        else
        {
            it->second.highestWorldY = computeHighest(it->second);
        }
        return;
    }

    auto [it, inserted] = columns_.try_emplace(key);
    it->second.slabHeights[chunkY] = highestWorldY;
    it->second.highestWorldY = computeHighest(it->second);
}

void ColumnManager::updateChunk(const Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (int x = 0; x < kChunkSizeX; ++x)
    {
        for (int z = 0; z < kChunkSizeZ; ++z)
        {
            const glm::ivec2 key = columnKey(chunk.coord, x, z);
            const int highestWorld = scanColumnHighestWorld(chunk, x, z);
            applyHeightLocked(key, chunk.coord.y, highestWorld);
        }
    }
}

void ColumnManager::updateColumn(const Chunk& chunk, int localX, int localZ)
{
    const int highestWorld = scanColumnHighestWorld(chunk, localX, localZ);
    std::lock_guard<std::mutex> lock(mutex_);
    applyHeightLocked(columnKey(chunk.coord, localX, localZ), chunk.coord.y, highestWorld);
}

void ColumnManager::removeChunk(const Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (int x = 0; x < kChunkSizeX; ++x)
    {
        for (int z = 0; z < kChunkSizeZ; ++z)
        {
            applyHeightLocked(columnKey(chunk.coord, x, z), chunk.coord.y, kNoHeight);
        }
    }
}

void ColumnManager::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    columns_.clear();
}

int ColumnManager::highestSolidBlock(int worldX, int worldZ) const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    const glm::ivec2 key{worldX, worldZ};
    auto it = columns_.find(key);
    if (it == columns_.end())
    {
        return kNoHeight;
    }
    return it->second.highestWorldY;
}



























// ChunkManager::Impl methods (to be filled)

ChunkManager::Impl::Impl(unsigned seed)
    : worldgenProfile_(terrain::WorldgenProfile::load("assets/worldgen.toml")),
      biomeDatabase_("assets/biomes"),
      globalSeaLevel_(worldgenProfile_.seaLevel),
      noise_(worldgenProfile_.effectiveSeed(seed)),
      shouldStop_(false),
      viewDistance_(renderSettings_.nearChunks),
      targetViewDistance_(renderSettings_.nearChunks)
{
    const unsigned effectiveSeed = worldgenProfile_.effectiveSeed(seed);

    noise_.reseed(effectiveSeed);

    if (biomeDatabase_.biomeCount() == 0)
    {
        throw std::runtime_error("Biome database is empty");
    }

    const auto& climateGeneratorName = worldgenProfile_.climateGenerator;
    std::unique_ptr<terrain::ClimateGenerator> climateGenerator;
    if (climateGeneratorName == "legacy" || climateGeneratorName == "voronoi"
        || climateGeneratorName == "noise_voronoi")
    {
        climateGenerator = std::make_unique<terrain::NoiseVoronoiClimateGenerator>(
            biomeDatabase_, worldgenProfile_, effectiveSeed, kChunkSizeX, kBiomeSizeInChunks);
    }
    else
    {
        throw std::runtime_error("Unsupported climate_generator '" + climateGeneratorName
                                 + "' in assets/worldgen.toml");
    }

    climateMap_ = std::make_unique<terrain::ClimateMap>(std::move(climateGenerator), 256);

    surfaceMap_ = std::make_unique<terrain::SurfaceMap>(
        std::make_unique<terrain::MapGenV1>(biomeDatabase_, *climateMap_, worldgenProfile_, effectiveSeed),
        256);

    terrainGenerator_ = std::make_unique<terrain::TerrainGenerator>(
        *climateMap_,
        *surfaceMap_,
        biomeDatabase_,
        globalSeaLevel_,
        [this](int worldX, int worldZ, int slabMin, int slabMax) {
            return this->sampleColumn(worldX, worldZ, slabMin, slabMax);
        });

    gActiveVerticalRadius.store(kVerticalStreamingConfig.minRadiusChunks, std::memory_order_relaxed);
    farTerrainManager_.setEnabled(renderSettings_.farTerrainEnabled);
    farTerrainManager_.setDistanceBlocks(renderSettings_.farBlocks);
    farTerrainManager_.setFogStartBlocks(renderSettings_.fogStartBlocks);
    unsigned concurrency = std::thread::hardware_concurrency();
    if (concurrency >= 12)
    {
        farWorkerCount_ = 2;
    }
    else
    {
        farWorkerCount_ = 1;
    }
    farTerrainManager_.setWorkerCount(static_cast<std::size_t>(farWorkerCount_));
    kFarPlane = computeFarPlaneForDistanceBlocks(renderSettings_.farBlocks);
    startWorkerThreads();
}

ChunkManager::Impl::~Impl()
{
    stopWorkerThreads();
    clear();
    farTerrainManager_.clear();
    destroyBufferPages();
    uploadContext_.shutdown();
}

void ChunkManager::Impl::initializeRendering(ID3D12Device* device)
{
    device_ = device;
    uploadContext_.initialize(device_.Get());
    farTerrainManager_.setDevice(device_.Get());
    destroyBufferPages();
}

void ChunkManager::Impl::setBlockTextureAtlasConfig(const BlockTextureAtlasConfig& config)
{
    if (config.tileSizePixels <= 0 || config.textureSizePixels.x <= 0 || config.textureSizePixels.y <= 0)
    {
        std::cerr << "Invalid block atlas dimensions provided" << std::endl;
        blockAtlasConfigured_ = false;
        return;
    }

    atlasTextureSizePixels_ = config.textureSizePixels;
    atlasTileSizePixels_ = config.tileSizePixels;
    atlasTileStridePixels_ = (config.tileStridePixels > 0) ? config.tileStridePixels : config.tileSizePixels;
    atlasTilePaddingPixels_ = std::max(config.tilePaddingPixels, 0);
    const glm::vec2 atlasTexelScale(
        1.0f / static_cast<float>(atlasTextureSizePixels_.x),
        1.0f / static_cast<float>(atlasTextureSizePixels_.y));

    for (auto& blockEntry : blockUVTable_)
    {
        for (auto& face : blockEntry.faces)
        {
            face.base = glm::vec2(0.0f);
            face.size = glm::vec2(static_cast<float>(atlasTileSizePixels_) * atlasTexelScale.x,
                                  static_cast<float>(atlasTileSizePixels_) * atlasTexelScale.y);
        }
    }

    auto assignFace = [&](BlockId block, BlockFace face, const glm::ivec2& tile)
    {
        const glm::ivec2 tilePixelOrigin(tile.x * atlasTileStridePixels_ + atlasTilePaddingPixels_,
                                         tile.y * atlasTileStridePixels_ + atlasTilePaddingPixels_);
        const glm::vec2 base = glm::vec2(static_cast<float>(tilePixelOrigin.x) * atlasTexelScale.x,
                                         static_cast<float>(tilePixelOrigin.y) * atlasTexelScale.y);
        auto& faceUV = blockUVTable_[toIndex(block)].faces[toIndex(face)];
        faceUV.base = base;
        faceUV.size = glm::vec2(static_cast<float>(atlasTileSizePixels_) * atlasTexelScale.x,
                                static_cast<float>(atlasTileSizePixels_) * atlasTexelScale.y);
    };

    assignFace(BlockId::Grass, BlockFace::Top, {0, 0});
    assignFace(BlockId::Grass, BlockFace::Bottom, {0, 2});
    for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Grass, face, {0, 1});
    }

    assignFace(BlockId::Wood, BlockFace::Top, {0, 4});
    assignFace(BlockId::Wood, BlockFace::Bottom, {0, 4});
    for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Wood, face, {0, 3});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Leaves, face, {0, 5});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Sand, face, {0, 6});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Water, face, {0, 7});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Stone, face, {0, 8});
    }

    assignFace(BlockId::SpruceLog, BlockFace::Top, {0, 9});
    assignFace(BlockId::SpruceLog, BlockFace::Bottom, {0, 9});
    for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::SpruceLog, face, {0, 10});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::SpruceLeaves, face, {0, 11});
    }

    assignFace(BlockId::Podzol, BlockFace::Top, {0, 13});
    assignFace(BlockId::Podzol, BlockFace::Bottom, {0, 2});
    for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::Podzol, face, {0, 12});
    }

    for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
    {
        assignFace(BlockId::DebugLamp, face, {0, 8});
    }

    blockAtlasConfigured_ = true;
}

std::pair<glm::vec2, glm::vec2> ChunkManager::Impl::atlasUvFor(BlockId block, BlockFace face) const
{
    const FaceUV& uv = blockUVTable_[toIndex(block)].faces[toIndex(face)];
    return {uv.base, uv.size};
}

FarTerrainSurfaceSample ChunkManager::Impl::sampleFarTerrainSurfaceLod(int worldX, int worldZ, int lodLevel) const
{
    FarTerrainSurfaceSample visual{};
    if (!surfaceMap_ || !climateMap_)
    {
        return visual;
    }

    const terrain::SurfaceColumn& surfaceColumn = surfaceMap_->column(worldX, worldZ, lodLevel);
    const terrain::ClimateSample& climateSample = climateMap_->sample(worldX, worldZ);
    if (!surfaceColumn.dominantBiome)
    {
        return visual;
    }

    const BiomeDefinition& biome = *surfaceColumn.dominantBiome;
    ColumnSample resolvedSample{};
    resolvedSample.surfaceY = surfaceColumn.surfaceY;
    resolvedSample.distanceToShore = std::isfinite(climateSample.distanceToCoast)
                                         ? climateSample.distanceToCoast
                                         : std::numeric_limits<float>::infinity();
    const terrain::TerrainColumnBlocks resolvedBlocks =
        terrain::resolveTerrainColumnBlocks(biome, resolvedSample, worldX, worldZ, globalSeaLevel_);

    visual.solidTopY = surfaceColumn.surfaceY;
    visual.solidBlock = resolvedBlocks.surfaceBlock;
    const int cachedTop = columnManager_.highestSolidBlock(worldX, worldZ);
    const int cacheTolerance = std::max(1 << std::clamp(lodLevel, 0, 6), 12);
    if (cachedTop != ColumnManager::kNoHeight &&
        std::abs(cachedTop - surfaceColumn.surfaceY) <= cacheTolerance)
    {
        visual.solidTopY = cachedTop;
    }

    const auto& waterFill = biome.terrainSettings.waterFill;
    if (waterFill.enabled && surfaceColumn.surfaceY < globalSeaLevel_)
    {
        visual.waterTopY = globalSeaLevel_;
        visual.hasVisibleWater = true;
    }

    return visual;
}

void ChunkManager::Impl::update(const glm::vec3& cameraPos)
{
    update(cameraPos, lastCameraForward_);
}

void ChunkManager::Impl::update(const glm::vec3& cameraPos, const glm::vec3& cameraForward)
{
    const auto updateStart = std::chrono::steady_clock::now();
    if (glm::dot(cameraForward, cameraForward) > kEpsilon)
    {
        lastCameraForward_ = glm::normalize(cameraForward);
    }

    const auto now = std::chrono::steady_clock::now();
    double frameSeconds = 1.0 / 60.0;
    if (lastUpdateTime_.time_since_epoch().count() != 0)
    {
        frameSeconds = std::chrono::duration<double>(now - lastUpdateTime_).count();
    }
    lastUpdateTime_ = now;
    frameSeconds = std::clamp(frameSeconds, 1.0 / 240.0, 0.25);
    smoothedFrameMs_ = smoothedFrameMs_ * 0.90 + frameSeconds * 1000.0 * 0.10;

    const int worldX = static_cast<int>(std::floor(cameraPos.x));
    const int worldY = static_cast<int>(std::floor(cameraPos.y));
    const int worldZ = static_cast<int>(std::floor(cameraPos.z));
    const int clampedWorldY = std::max(worldY, 0);
    const glm::ivec3 centerChunk = worldToChunkCoords(worldX, clampedWorldY, worldZ);
    lastCenterChunk_ = centerChunk;

    if (!startupEnabled_ || !startupState_.preloadStarted)
    {
        startupState_.phase = StreamingPhase::SteadyState;
        startupState_.exactNearCurrentChunks = renderSettings_.nearChunks;
        startupState_.farCurrentBlocks = renderSettings_.farBlocks;
        startupState_.playerReleaseReady = true;
    }
    else
    {
        if (startupState_.phase == StreamingPhase::SpawnResolve)
        {
            startupState_.phase = StreamingPhase::ExactPreload;
        }
        startupState_.phaseTimeSeconds += frameSeconds;
        startupState_.totalTimeSeconds += frameSeconds;
    }

    targetViewDistance_ = std::clamp(startupState_.exactNearCurrentChunks, 1, renderSettings_.nearChunks);

    resetColumnBudgets();
    const int verticalRadius = computeVerticalRadius(centerChunk, targetViewDistance_, clampedWorldY);
    lastVerticalRadius_ = verticalRadius;
    gActiveVerticalRadius.store(verticalRadius, std::memory_order_relaxed);
    kFarPlane = computeFarPlaneForDistanceBlocks(renderSettings_.farBlocks);

    UploadBudgets uploadBudgets = computeUploadBudgets(verticalRadius);
    uploadBudgetBytesThisFrame_ = uploadBudgets.byteBudget;
    uploadColumnLimitThisFrame_ = uploadBudgets.columnLimit;
    uploadBudgetMsThisFrame_ = uploadBudgets.timeBudgetMs;
    pendingUploadsLastFrame_ = uploadBudgets.queueSize;

    jobQueue_.updatePriorityOrigin(centerChunk);

    if (viewDistance_ > targetViewDistance_)
    {
        viewDistance_ = targetViewDistance_;
    }

    const int missingChunks = estimateMissingChunks(centerChunk, targetViewDistance_, verticalRadius);
    const int backlogSteps = computeBacklogSteps(missingChunks,
                                                 kVerticalStreamingConfig.generationBudget.backlogStartThreshold,
                                                 kVerticalStreamingConfig.generationBudget.backlogStepSize);
    int columnCap = computeColumnJobCap(backlogSteps, missingChunks);
    if (columnCap <= 0)
    {
        columnCap = std::numeric_limits<int>::max();
    }

    generationColumnCapThisFrame_ = columnCap;

    const int generationBudgetTarget =
        computeGenerationBudget(targetViewDistance_, verticalRadius, backlogSteps);
    const int ringBudget = computeRingExpansionBudget(missingChunks);

    lastGenerationBudget_ = generationBudgetTarget;
    lastRingBudget_ = ringBudget;
    lastMissingChunks_ = missingChunks;
    lastColumnCap_ = generationColumnCapThisFrame_;
    lastBacklogSteps_ = backlogSteps;

    int jobBudget = generationBudgetTarget;

    for (int ring = 0; ring <= viewDistance_ && jobBudget > 0; ++ring)
    {
        RingProgress progress = ensureVolume(centerChunk, ring, verticalRadius, jobBudget);
        if (progress.budgetExhausted)
        {
            break;
        }
    }

    int ringsExpanded = 0;
    while (jobBudget > 0 && viewDistance_ < targetViewDistance_ && ringsExpanded < ringBudget)
    {
        const int nextRing = viewDistance_ + 1;
        RingProgress progress = ensureVolume(centerChunk, nextRing, verticalRadius, jobBudget);

        if (progress.budgetExhausted)
        {
            break;
        }

        if (progress.fullyLoaded)
        {
            ++viewDistance_;
            ++ringsExpanded;
            continue;
        }

        break;
    }

    lastGenerationJobsIssued_ = std::clamp(generationBudgetTarget - jobBudget, 0, generationBudgetTarget);
    lastRingExpansionsUsed_ = ringsExpanded;

    removeDistantChunks(centerChunk,
                        targetViewDistance_ + kVerticalStreamingConfig.horizontalEvictionSlack,
                        verticalRadius);

    uploadReadyMeshes();

    const bool farStreamingActive =
        renderSettings_.farTerrainEnabled &&
        (!startupEnabled_ || !startupState_.preloadStarted ||
         startupState_.phase == StreamingPhase::FarRamp ||
         startupState_.phase == StreamingPhase::SteadyState);
    if (farStreamingActive && startupState_.farCurrentBlocks > 0)
    {
        farTerrainManager_.update(centerChunk,
                                  lastCameraForward_,
                                  targetViewDistance_,
                                  startupState_.farCurrentBlocks,
                                  std::max(0.5, uploadBudgets.timeBudgetMs * 0.75),
                                  [this](int sampleWorldX, int sampleWorldZ, int lodLevel)
                                  {
                                      return this->sampleFarTerrainSurfaceLod(sampleWorldX, sampleWorldZ, lodLevel);
                                  },
                                  [this](BlockId block, BlockFace face)
                                  {
                                      return this->atlasUvFor(block, face);
                                  });
    }
    else
    {
        farTerrainManager_.clear();
    }

    if (startupEnabled_ && startupState_.preloadStarted)
    {
        const bool nearReady = missingChunks == 0;
        const bool uploadReady = pendingUploadsLastFrame_ <= 8;
        const bool exactReady = nearReady && uploadReady;
        const bool farHealthy = smoothedFrameMs_ <= 20.0 &&
                                pendingUploadsLastFrame_ <= 16 &&
                                farTerrainManager_.queuedTileCount() <= std::max(4, farWorkerCount_ * 3) &&
                                farTerrainManager_.pendingUploadTileCount() <= 6;
        const bool farRegressed = smoothedFrameMs_ > 28.0;

        switch (startupState_.phase)
        {
        case StreamingPhase::ExactPreload:
            startupState_.farCurrentBlocks = 0;
            startupState_.playerReleaseReady = exactReady;
            if (exactReady)
            {
                startupState_.phase = StreamingPhase::InteractiveNearOnly;
                startupState_.phaseTimeSeconds = 0.0;
                startupState_.healthyTimeSeconds = 0.0;
                startupState_.exactNearCurrentChunks = std::min(renderSettings_.nearChunks, 6);
                startupState_.playerReleaseReady = true;
            }
            else if (startupState_.phaseTimeSeconds >= 2.0 && startupState_.exactNearCurrentChunks > 4)
            {
                --startupState_.exactNearCurrentChunks;
                startupState_.phaseTimeSeconds = 0.0;
            }
            break;
        case StreamingPhase::InteractiveNearOnly:
            startupState_.playerReleaseReady = true;
            startupState_.farCurrentBlocks = 0;
            if (exactReady)
            {
                startupState_.healthyTimeSeconds += frameSeconds;
            }
            else
            {
                startupState_.healthyTimeSeconds = 0.0;
            }

            if (startupState_.healthyTimeSeconds >= 0.75)
            {
                startupState_.healthyTimeSeconds = 0.0;
                if (startupState_.exactNearCurrentChunks < std::min(renderSettings_.nearChunks, 8))
                {
                    startupState_.exactNearCurrentChunks = std::min(renderSettings_.nearChunks, 8);
                }
                else
                {
                    startupState_.phase = StreamingPhase::FarRamp;
                    startupState_.phaseTimeSeconds = 0.0;
                    startupState_.farCurrentBlocks = std::min(renderSettings_.farBlocks, 768);
                }
            }
            break;
        case StreamingPhase::FarRamp:
            startupState_.playerReleaseReady = true;
            if (startupState_.exactNearCurrentChunks < renderSettings_.nearChunks && exactReady)
            {
                startupState_.healthyTimeSeconds += frameSeconds;
                if (startupState_.healthyTimeSeconds >= 0.75)
                {
                    startupState_.exactNearCurrentChunks = renderSettings_.nearChunks;
                    startupState_.healthyTimeSeconds = 0.0;
                }
            }
            else
            {
                if (farHealthy)
                {
                    startupState_.healthyTimeSeconds += frameSeconds;
                }
                else if (farRegressed)
                {
                    startupState_.healthyTimeSeconds = 0.0;
                    if (startupState_.farCurrentBlocks > 3072)
                    {
                        startupState_.farCurrentBlocks = 3072;
                    }
                    else if (startupState_.farCurrentBlocks > 1536)
                    {
                        startupState_.farCurrentBlocks = 1536;
                    }
                    else if (startupState_.farCurrentBlocks > 768)
                    {
                        startupState_.farCurrentBlocks = 768;
                    }
                }
                else
                {
                    startupState_.healthyTimeSeconds =
                        std::max(0.0, startupState_.healthyTimeSeconds - frameSeconds * 0.5);
                }

                if (startupState_.healthyTimeSeconds >= 1.5)
                {
                    startupState_.healthyTimeSeconds = 0.0;
                    if (startupState_.farCurrentBlocks < std::min(renderSettings_.farBlocks, 1536))
                    {
                        startupState_.farCurrentBlocks = std::min(renderSettings_.farBlocks, 1536);
                    }
                    else if (startupState_.farCurrentBlocks < std::min(renderSettings_.farBlocks, 3072))
                    {
                        startupState_.farCurrentBlocks = std::min(renderSettings_.farBlocks, 3072);
                    }
                    else if (startupState_.farCurrentBlocks < renderSettings_.farBlocks)
                    {
                        startupState_.farCurrentBlocks = renderSettings_.farBlocks;
                    }
                    else
                    {
                        startupState_.phase = StreamingPhase::SteadyState;
                        startupState_.phaseTimeSeconds = 0.0;
                    }
                }
            }
            break;
        case StreamingPhase::SteadyState:
            startupState_.playerReleaseReady = true;
            startupState_.exactNearCurrentChunks = renderSettings_.nearChunks;
            startupState_.farCurrentBlocks = renderSettings_.farBlocks;
            break;
        case StreamingPhase::SpawnResolve:
            startupState_.phase = StreamingPhase::ExactPreload;
            startupState_.phaseTimeSeconds = 0.0;
            startupState_.playerReleaseReady = false;
            startupState_.farCurrentBlocks = 0;
            break;
        }
    }

    updateMsLastFrame_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - updateStart).count();
}

WorldRenderData ChunkManager::Impl::buildRenderData(const Frustum& frustum) const
{
    WorldRenderData renderData;
    renderData.highlightedBlock = highlightedBlock_;
    renderData.hasHighlight = hasHighlight_;

    std::vector<std::pair<glm::ivec3, std::shared_ptr<Chunk>>> snapshot;
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        snapshot.reserve(chunks_.size());
        for (const auto& entry : chunks_)
        {
            snapshot.push_back(entry);
        }
    }

    {
        std::lock_guard<std::mutex> pageLock(bufferPageMutex_);
        const std::size_t pageCount = bufferPages_.size();
        renderData.nearBatches.resize(pageCount);
        for (std::size_t i = 0; i < pageCount; ++i)
        {
            renderData.nearBatches[i].vertexBufferView = bufferPages_[i].vertexView;
            renderData.nearBatches[i].indexBufferView = bufferPages_[i].indexView;
        }
    }

    for (const auto& [coord, chunkPtr] : snapshot)
    {
        if (!chunkPtr)
        {
            continue;
        }

        ChunkState state = chunkPtr->state.load();
        const std::uint32_t indexCount = chunkPtr->indexCount.load(std::memory_order_acquire);
        if ((state != ChunkState::Uploaded && state != ChunkState::Remeshing) || indexCount == 0)
        {
            continue;
        }

        const glm::vec3 minCorner(static_cast<float>(coord.x * kChunkSizeX),
                                  static_cast<float>(chunkPtr->minWorldY),
                                  static_cast<float>(coord.z * kChunkSizeZ));
        const glm::vec3 maxCorner(static_cast<float>((coord.x + 1) * kChunkSizeX),
                                  static_cast<float>(chunkPtr->maxWorldY + 1),
                                  static_cast<float>((coord.z + 1) * kChunkSizeZ));

        if (!frustum.intersectsAABB(minCorner, maxCorner))
        {
            continue;
        }

        const std::uint32_t pageIndex = chunkPtr->bufferPageIndex.load(std::memory_order_acquire);
        if (pageIndex == kInvalidChunkBufferPage || pageIndex >= renderData.nearBatches.size())
        {
            continue;
        }

        const std::size_t vertexOffset = chunkPtr->vertexOffset.load(std::memory_order_acquire);
        const std::size_t indexOffset = chunkPtr->indexOffset.load(std::memory_order_acquire);
        if (vertexOffset > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
            indexOffset > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            continue;
        }

        ChunkRenderBatch& batch = renderData.nearBatches[pageIndex];
        batch.indexCounts.push_back(indexCount);
        batch.firstIndexLocations.push_back(static_cast<std::uint32_t>(indexOffset));
        batch.baseVertices.push_back(static_cast<std::int32_t>(vertexOffset));
    }

    auto emptyIt = std::remove_if(renderData.nearBatches.begin(),
                                  renderData.nearBatches.end(),
                                  [](const ChunkRenderBatch& batch)
                                  {
                                      return batch.indexCounts.empty();
                                  });
    renderData.nearBatches.erase(emptyIt, renderData.nearBatches.end());
    renderData.farBatches = farTerrainManager_.buildRenderBatches(frustum);

    return renderData;
}

float ChunkManager::Impl::surfaceHeight(float worldX, float worldZ) const noexcept
{
    const int wx = static_cast<int>(std::floor(worldX));
    const int wz = static_cast<int>(std::floor(worldZ));
    const int cachedHeight = columnManager_.highestSolidBlock(wx, wz);
    if (cachedHeight != ColumnManager::kNoHeight)

    {
        return static_cast<float>(cachedHeight + 1);
    }

    const ColumnSample sample = sampleColumn(wx, wz);
    return static_cast<float>(sample.surfaceY + 1);

}

ColumnSample ChunkManager::Impl::sampleColumnAt(const glm::vec3& worldPos,
                                                int slabMinWorldY,
                                                int slabMaxWorldY) const
{
    const int worldX = static_cast<int>(std::floor(worldPos.x));
    const int worldZ = static_cast<int>(std::floor(worldPos.z));
    return sampleColumn(worldX, worldZ, slabMinWorldY, slabMaxWorldY);
}

void ChunkManager::Impl::clear()
{
    while (true)
    {
        std::vector<glm::ivec3> coords;
        {
            std::lock_guard<std::mutex> lock(chunksMutex);
            coords.reserve(chunks_.size());
            for (const auto& [coord, chunkPtr] : chunks_)
            {
                coords.push_back(coord);
            }
        }

        if (coords.empty())
        {
            break;
        }

        bool removedAny = false;
        for (const glm::ivec3& coord : coords)
        {
            std::shared_ptr<Chunk> chunk;
            {
                std::lock_guard<std::mutex> lock(chunksMutex);
                auto it = chunks_.find(coord);
                if (it == chunks_.end())
                {
                    continue;
                }

                if (it->second->inFlight.load(std::memory_order_acquire) != 0)
                {
                    continue;
                }

                chunk = it->second;
                chunks_.erase(it);
                removedAny = true;
            }

        if (chunk)
        {
            columnManager_.removeChunk(*chunk);
            invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});
            recycleChunkGPU(*chunk);
            recycleChunkObject(std::move(chunk));

        }
    }

        if (!removedAny)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    {
        std::lock_guard<std::mutex> lock(uploadQueueMutex_);
        uploadQueue_.clear();
    }
    farTerrainManager_.clear();
    columnManager_.clear();
    {
        std::lock_guard<std::mutex> lock(predictedColumnMutex_);
        predictedColumnHeights_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(pendingStructureMutex_);
        pendingStructureEdits_.clear();
    }

    uploadBudgetBytesThisFrame_ = kUploadBudgetBytesPerFrame;
    uploadColumnLimitThisFrame_ = kVerticalStreamingConfig.uploadBasePerColumn;
    lastUploadBytesUsed_ = 0;
    pendingUploadsLastFrame_ = 0;

    if (climateMap_)
    {
        climateMap_->clear();
    }

    if (surfaceMap_)
    {
        surfaceMap_->clear();
    }

}

bool ChunkManager::Impl::destroyBlock(const glm::ivec3& worldPos)
{
    const glm::ivec3 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
    auto chunk = getChunkShared(chunkCoord);
    if (!chunk)
    {
        return false;
    }

    ChunkState currentState = chunk->state.load();
    if (currentState != ChunkState::Uploaded && currentState != ChunkState::Remeshing)
    {
        return false;
    }

    if (worldPos.y < chunk->minWorldY || worldPos.y > chunk->maxWorldY)
    {
        return false;
    }
    const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
    const int localY = worldPos.y - chunk->minWorldY;
    const std::size_t blockIdx = blockIndex(local.x, localY, local.z);


    {
        std::lock_guard<std::mutex> lock(chunk->meshMutex);
        if (!isSolid(chunk->blocks[blockIdx]))
        {
            return false;
        }

        chunk->blocks[blockIdx] = BlockId::Air;
        if (chunk->hasBlocks.load(std::memory_order_relaxed))
        {
            chunk->hasBlocks.store(chunkHasSolidBlocks(*chunk), std::memory_order_relaxed);
        }

        columnManager_.updateColumn(*chunk, local.x, local.z);
        chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
    }

    invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});
    relightAroundChunk(chunkCoord);
    markNeighborsForRemeshingIfNeeded(chunkCoord, local.x, localY, local.z);
    farTerrainManager_.invalidateWorldBlock(worldPos);

    return true;
}

bool ChunkManager::Impl::placeBlock(const glm::ivec3& targetBlockPos, const glm::ivec3& faceNormal, BlockId block)
{
    const glm::ivec3 placePos = targetBlockPos + faceNormal;

    const glm::ivec3 chunkCoord = worldToChunkCoords(placePos.x, placePos.y, placePos.z);
    auto chunk = getChunkShared(chunkCoord);
    if (!chunk)
    {
        return false;
    }

    ChunkState currentState = chunk->state.load();
    if (currentState != ChunkState::Uploaded && currentState != ChunkState::Remeshing)
    {
        return false;
    }

    if (placePos.y < chunk->minWorldY || placePos.y > chunk->maxWorldY)
    {
        return false;
    }
    const glm::ivec3 local = localBlockCoords(placePos, chunkCoord);
    const int localY = placePos.y - chunk->minWorldY;
    const std::size_t blockIdx = blockIndex(local.x, localY, local.z);


    {
        std::lock_guard<std::mutex> lock(chunk->meshMutex);
        if (isSolid(chunk->blocks[blockIdx]))
        {
            return false;
        }

        chunk->blocks[blockIdx] = block;
        chunk->hasBlocks.store(true, std::memory_order_relaxed);

        columnManager_.updateColumn(*chunk, local.x, local.z);
        chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
    }

    invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});
    relightAroundChunk(chunkCoord);
    markNeighborsForRemeshingIfNeeded(chunkCoord, local.x, localY, local.z);
    farTerrainManager_.invalidateWorldBlock(placePos);

    return true;
}

RaycastHit ChunkManager::Impl::raycast(const glm::vec3& origin, const glm::vec3& direction) const
{
    RaycastHit result;

    const float dirLengthSq = glm::dot(direction, direction);
    if (dirLengthSq < kEpsilon * kEpsilon)
    {
        return result;
    }

    const glm::vec3 dir = glm::normalize(direction);
    glm::ivec3 currentBlock{
        static_cast<int>(std::floor(origin.x)),
        static_cast<int>(std::floor(origin.y)),
        static_cast<int>(std::floor(origin.z))
    };

    glm::ivec3 stepVec;
    glm::vec3 tMax;
    glm::vec3 tDelta;

    auto initializeAxis = [](float dirComponent, float originComponent, int blockComponent, int& stepOut, float& tMaxOut, float& tDeltaOut)
    {
        if (dirComponent > 0.0f)
        {
            stepOut = 1;
            const float nextBoundary = static_cast<float>(blockComponent + 1);
            tMaxOut = (nextBoundary - originComponent) / dirComponent;
            tDeltaOut = 1.0f / dirComponent;
        }
        else if (dirComponent < 0.0f)
        {
            stepOut = -1;
            const float nextBoundary = static_cast<float>(blockComponent);
            tMaxOut = (nextBoundary - originComponent) / dirComponent;
            tDeltaOut = -1.0f / dirComponent;
        }
        else
        {
            stepOut = 0;
            tMaxOut = std::numeric_limits<float>::infinity();
            tDeltaOut = std::numeric_limits<float>::infinity();
        }
    };

    initializeAxis(dir.x, origin.x, currentBlock.x, stepVec.x, tMax.x, tDelta.x);
    initializeAxis(dir.y, origin.y, currentBlock.y, stepVec.y, tMax.y, tDelta.y);
    initializeAxis(dir.z, origin.z, currentBlock.z, stepVec.z, tMax.z, tDelta.z);

    glm::ivec3 previousBlock = currentBlock;

    while (true)
    {
        int axis = 0;
        if (tMax.y < tMax.x)
        {
            axis = 1;
        }
        if (tMax.z < tMax[axis])
        {
            axis = 2;
        }

        const float nextT = tMax[axis];
        if (nextT > kMaxRayDistance)
        {
            break;
        }

        previousBlock = currentBlock;
        currentBlock[axis] += stepVec[axis];
        tMax[axis] += tDelta[axis];

        if (isSolid(blockAt(currentBlock)))
        {
            result.hit = true;
            result.blockPos = currentBlock;
            result.distance = nextT;
            result.faceNormal = previousBlock - currentBlock;
            break;
        }
    }

    return result;
}

void ChunkManager::Impl::updateHighlight(const glm::vec3& cameraPos, const glm::vec3& cameraDirection)
{
    RaycastHit hit = raycast(cameraPos, cameraDirection);
    if (hit.hit)
    {
        highlightedBlock_ = hit.blockPos;
        hasHighlight_ = true;
    }
    else
    {
        hasHighlight_ = false;
    }
}

void ChunkManager::Impl::toggleViewDistance()
{
    try
    {
        if (targetViewDistance_ == kDefaultNearRenderDistance)
        {
            std::cout << "Switching to extended near render distance..." << std::endl;
            setNearRenderDistance(kMaxUserRenderDistance);
            const long long width = static_cast<long long>(targetViewDistance_) * 2ll + 1ll;
            const long long totalColumns = width * width;
            std::cout << "Extended near render distance target: " << targetViewDistance_ << " chunks (total: "
                      << totalColumns << " chunks)" << std::endl;
        }
        else
        {
            std::cout << "Switching to default near render distance..." << std::endl;
            setNearRenderDistance(kDefaultNearRenderDistance);
            const long long width = static_cast<long long>(targetViewDistance_) * 2ll + 1ll;
            const long long totalColumns = width * width;
            std::cout << "Default near render distance target: " << targetViewDistance_
                      << " chunks (total: " << totalColumns << " chunks)" << std::endl;
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error toggling view distance: " << ex.what() << std::endl;
        targetViewDistance_ = kDefaultNearRenderDistance;
        viewDistance_ = std::min(viewDistance_, targetViewDistance_);
        renderSettings_.nearChunks = targetViewDistance_;
        kFarPlane = computeFarPlaneForDistanceBlocks(renderSettings_.farBlocks);
    }
}

int ChunkManager::Impl::viewDistance() const noexcept
{
    return targetViewDistance_;
}

int ChunkManager::Impl::nearRenderDistance() const noexcept
{
    return renderSettings_.nearChunks;
}

int ChunkManager::Impl::farRenderDistanceBlocks() const noexcept
{
    return renderSettings_.farBlocks;
}

RenderDistanceSettings ChunkManager::Impl::renderDistanceSettings() const noexcept
{
    return renderSettings_;
}

void ChunkManager::Impl::setRenderDistance(int distance) noexcept
{
    setNearRenderDistance(distance);
}

void ChunkManager::Impl::setNearRenderDistance(int chunks) noexcept
{
    try
    {
        const int clampedDistance = std::clamp(chunks, 1, kMaxUserRenderDistance);
        renderSettings_.nearChunks = clampedDistance;
        if (!startupEnabled_ || !startupState_.preloadStarted || startupState_.phase == StreamingPhase::SteadyState)
        {
            targetViewDistance_ = clampedDistance;
            startupState_.exactNearCurrentChunks = clampedDistance;
        }
        else
        {
            startupState_.exactNearCurrentChunks = std::min(startupState_.exactNearCurrentChunks, clampedDistance);
            targetViewDistance_ = std::min(startupState_.exactNearCurrentChunks, clampedDistance);
        }
        kFarPlane = computeFarPlaneForDistanceBlocks(renderSettings_.farBlocks);
        if (chunks != clampedDistance)
        {
            std::cout << "Near render distance request " << chunks << " clamped to " << clampedDistance << " chunks"
                      << std::endl;
        }

        if (viewDistance_ > targetViewDistance_)
        {
            viewDistance_ = targetViewDistance_;
        }

        const long long width = static_cast<long long>(targetViewDistance_) * 2ll + 1ll;
        const long long totalColumns = width * width;
        std::cout << "Near render distance set to: " << targetViewDistance_ << " chunks (total: "
                  << totalColumns << " chunks)" << std::endl;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error setting near render distance: " << ex.what() << std::endl;
    }
}

void ChunkManager::Impl::setFarRenderDistanceBlocks(int blocks) noexcept
{
    try
    {
        renderSettings_.farBlocks = std::max(blocks, 256);
        if (!startupEnabled_ || !startupState_.preloadStarted || startupState_.phase == StreamingPhase::SteadyState)
        {
            startupState_.farCurrentBlocks = renderSettings_.farBlocks;
        }
        else if (startupState_.phase == StreamingPhase::FarRamp)
        {
            startupState_.farCurrentBlocks = std::min(startupState_.farCurrentBlocks, renderSettings_.farBlocks);
        }
        else
        {
            startupState_.farCurrentBlocks = 0;
        }
        farTerrainManager_.setDistanceBlocks(startupState_.farCurrentBlocks);
        kFarPlane = computeFarPlaneForDistanceBlocks(renderSettings_.farBlocks);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error setting far render distance: " << ex.what() << std::endl;
    }
}

void ChunkManager::Impl::setFogStartBlocks(int blocks) noexcept
{
    renderSettings_.fogStartBlocks = std::max(blocks, 0);
    farTerrainManager_.setFogStartBlocks(renderSettings_.fogStartBlocks);
}

void ChunkManager::Impl::setFarTerrainEnabled(bool enabled)
{
    renderSettings_.farTerrainEnabled = enabled;
    farTerrainManager_.setEnabled(enabled);
    farTerrainManager_.setDistanceBlocks(startupState_.farCurrentBlocks);
    std::cout << "[ChunkManager] Far terrain " << (enabled ? "enabled" : "disabled")
              << " via F3 toggle" << std::endl;
}

bool ChunkManager::Impl::farTerrainEnabled() const noexcept
{
    return renderSettings_.farTerrainEnabled;
}

void ChunkManager::Impl::setLodEnabled(bool enabled)
{
    setFarTerrainEnabled(enabled);
}

bool ChunkManager::Impl::lodEnabled() const noexcept
{
    return farTerrainEnabled();
}

BlockId ChunkManager::Impl::blockAt(const glm::ivec3& worldPos) const noexcept
{
    const glm::ivec3 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
    auto chunk = getChunkShared(chunkCoord);
    if (!chunk)
    {
        return BlockId::Air;
    }

    if (worldPos.y < chunk->minWorldY || worldPos.y > chunk->maxWorldY)
    {
        return BlockId::Air;
    }
    const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
    const int localY = worldPos.y - chunk->minWorldY;
    return chunk->blocks[blockIndex(local.x, localY, local.z)];

}

LightSample ChunkManager::Impl::lightAt(const glm::ivec3& worldPos) const noexcept
{
    const std::uint8_t packed = packedLightAtWorld(worldPos);
    return LightSample{skyLightFromPacked(packed), blockLightFromPacked(packed)};
}

glm::vec3 ChunkManager::Impl::findSafeSpawnPosition(float worldX, float worldZ) const
{
    const int baseX = static_cast<int>(std::floor(worldX));
    const int baseZ = static_cast<int>(std::floor(worldZ));
    int highestSolid = columnManager_.highestSolidBlock(baseX, baseZ);

    auto mergeHeight = [](int current, int candidate)
    {
        if (candidate == ColumnManager::kNoHeight)
        {
            return current;
        }
        if (current == ColumnManager::kNoHeight)
        {
            return candidate;
        }
        return std::max(current, candidate);
    };

    const ColumnSample baseSample = sampleColumn(baseX, baseZ);

    auto predictTreeCanopyTop = [&](int originX, int originZ, const ColumnSample& columnSample, int targetX, int targetZ) -> int
    {
        if (!columnSample.dominantBiome || !columnSample.dominantBiome->generatesTrees)
        {
            return ColumnManager::kNoHeight;
        }

        constexpr float kTreeBiomeWeightThreshold = 0.55f;
        if (columnSample.dominantWeight < kTreeBiomeWeightThreshold)
        {
            return ColumnManager::kNoHeight;
        }

        const int groundWorldY = columnSample.surfaceY;
        if (groundWorldY <= 2)
        {
            return ColumnManager::kNoHeight;
        }

        const BiomeDefinition& biome = *columnSample.dominantBiome;

        if (terrain::isTaigaBiome(biome))
        {
            if (!shouldSpawnTaigaSpruce(biome, originX, groundWorldY, originZ))
            {
                return ColumnManager::kNoHeight;
            }

            int anchorGroundY = std::numeric_limits<int>::min();
            for (int trunkX = 0; trunkX < 2; ++trunkX)
            {
                for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
                {
                    const ColumnSample trunkSample = sampleColumn(originX + trunkX, originZ + trunkZ);
                    if (!trunkSample.dominantBiome || !terrain::isTaigaBiome(*trunkSample.dominantBiome))
                    {
                        return ColumnManager::kNoHeight;
                    }
                    if (trunkSample.dominantWeight < kTreeBiomeWeightThreshold)
                    {
                        return ColumnManager::kNoHeight;
                    }

                    const terrain::TerrainColumnBlocks blocks =
                        terrain::resolveTerrainColumnBlocks(*trunkSample.dominantBiome,
                                                            trunkSample,
                                                            originX + trunkX,
                                                            originZ + trunkZ,
                                                            globalSeaLevel_);
                    if (blocks.surfaceBlock != BlockId::Grass && blocks.surfaceBlock != BlockId::Podzol)
                    {
                        return ColumnManager::kNoHeight;
                    }

                    if (anchorGroundY == std::numeric_limits<int>::min())
                    {
                        anchorGroundY = trunkSample.surfaceY;
                    }
                    else if (trunkSample.surfaceY != anchorGroundY)
                    {
                        return ColumnManager::kNoHeight;
                    }
                }
            }

            for (int dx = -2; dx <= 3; ++dx)
            {
                for (int dz = -2; dz <= 3; ++dz)
                {
                    const ColumnSample neighborSample = sampleColumn(originX + dx, originZ + dz);
                    if (!neighborSample.dominantBiome)
                    {
                        return ColumnManager::kNoHeight;
                    }
                    if (std::abs(neighborSample.surfaceY - anchorGroundY) > 1)
                    {
                        return ColumnManager::kNoHeight;
                    }
                }
            }

            const int trunkHeight = taigaSpruceTrunkHeight(originX, anchorGroundY, originZ);
            const int bareTrunkHeight = taigaSpruceBareTrunkHeight(originX, anchorGroundY, originZ);
            const int canopyBaseWorld = anchorGroundY + bareTrunkHeight + 1;
            const int canopyTopWorld = anchorGroundY + trunkHeight;
            const int totalLayers = std::max(1, canopyTopWorld - canopyBaseWorld + 1);

            int highestCover = ColumnManager::kNoHeight;
            if (targetX >= originX && targetX <= originX + 1 &&
                targetZ >= originZ && targetZ <= originZ + 1)
            {
                highestCover = canopyTopWorld + 1;
            }

            for (int worldY = canopyBaseWorld; worldY <= canopyTopWorld; ++worldY)
            {
                const int layerFromBottom = worldY - canopyBaseWorld;
                const int radius = taigaSpruceLeafRadiusForLayer(layerFromBottom, totalLayers);
                if (taigaSpruceLeafOccupiesCell(originX,
                                               originZ,
                                               targetX,
                                               targetZ,
                                               radius,
                                               layerFromBottom,
                                               totalLayers))
                {
                    highestCover = std::max(highestCover, worldY);
                }
            }

            return highestCover;
        }

        const float density = noise_.fbm(static_cast<float>(originX) * 0.05f,
                                         static_cast<float>(originZ) * 0.05f,
                                         4,
                                         0.55f,
                                         2.0f);
        const float normalizedDensity = std::clamp((density + 1.0f) * 0.5f, 0.0f, 1.0f);
        const float randomValue = hashToUnitFloat(originX, groundWorldY, originZ);
        const float spawnThresholdBase = 0.015f + normalizedDensity * 0.02f;
        const float spawnThreshold =
            std::clamp(spawnThresholdBase * std::max(biome.treeDensityMultiplier, 0.0f), 0.0f, 1.0f);
        if (randomValue > spawnThreshold)
        {
            return ColumnManager::kNoHeight;
        }

        bool terrainSuitable = true;
        for (int dx = -1; dx <= 1 && terrainSuitable; ++dx)
        {
            for (int dz = -1; dz <= 1; ++dz)
            {
                if (dx == 0 && dz == 0)
                {
                    continue;
                }

                const ColumnSample neighborSample = sampleColumn(originX + dx, originZ + dz);
                if (std::abs(neighborSample.surfaceY - groundWorldY) > 1)
                {
                    terrainSuitable = false;
                    break;
                }
            }
        }

        if (!terrainSuitable)
        {
            return ColumnManager::kNoHeight;
        }

        constexpr int kTreeMinHeight = 6;
        constexpr int kTreeMaxHeight = 8;

        int trunkHeight = kTreeMinHeight +
                          static_cast<int>(hashToUnitFloat(originX, groundWorldY + 1, originZ) *
                                           static_cast<float>(kTreeMaxHeight - kTreeMinHeight + 1));
        trunkHeight = std::clamp(trunkHeight, kTreeMinHeight, kTreeMaxHeight);

        int highestCover = ColumnManager::kNoHeight;
        if (targetX == originX && targetZ == originZ)
        {
            highestCover = groundWorldY + trunkHeight;
        }

        const int canopyBaseWorld = groundWorldY + trunkHeight - 3;
        const int canopyTopWorld = groundWorldY + trunkHeight;
        for (int worldY = canopyBaseWorld; worldY <= canopyTopWorld; ++worldY)
        {
            const int layer = worldY - canopyBaseWorld;
            int radius = 2;
            if (worldY >= canopyTopWorld - 1)
            {
                radius = 1;
            }

            for (int dx = -radius; dx <= radius; ++dx)
            {
                for (int dz = -radius; dz <= radius; ++dz)
                {
                    if (std::abs(dx) == radius && std::abs(dz) == radius && radius > 1)
                    {
                        continue;
                    }

                    if (dx == 0 && dz == 0 && worldY <= groundWorldY + trunkHeight - 1)
                    {
                        continue;
                    }

                    if (layer == 0 && std::abs(dx) + std::abs(dz) > 3)
                    {
                        continue;
                    }

                    if (originX + dx == targetX && originZ + dz == targetZ)
                    {
                        highestCover = std::max(highestCover, worldY);
                    }
                }
            }
        }

        return highestCover;
    };

    int predictedHighest = ColumnManager::kNoHeight;
    if (baseSample.dominantBiome)
    {
        predictedHighest = mergeHeight(predictedHighest, baseSample.surfaceY);
    }

    for (int originX = baseX - kTaigaSpruceMaxLeafRadius; originX <= baseX + kTaigaSpruceMaxLeafRadius; ++originX)
    {
        for (int originZ = baseZ - kTaigaSpruceMaxLeafRadius; originZ <= baseZ + kTaigaSpruceMaxLeafRadius; ++originZ)
        {
            const ColumnSample originSample =
                (originX == baseX && originZ == baseZ) ? baseSample : sampleColumn(originX, originZ);
            predictedHighest = mergeHeight(predictedHighest,
                                           predictTreeCanopyTop(originX, originZ, originSample, baseX, baseZ));
        }
    }

    highestSolid = mergeHeight(highestSolid, predictedHighest);
    if (highestSolid == ColumnManager::kNoHeight)
    {
        highestSolid = 0;
    }

    const int clearanceHeight = static_cast<int>(std::ceil(kPlayerHeight)) + 2;
    const int spawnFeetY = std::max(highestSolid + 1, baseSample.surfaceY + 2) + clearanceHeight;
    std::cout << "Predicted spawn at height: " << (spawnFeetY + kCameraEyeHeight)
              << " (feet at: " << spawnFeetY << ")" << std::endl;
    const float fallbackY = static_cast<float>(spawnFeetY) + kCameraEyeHeight;
    return glm::vec3(worldX, fallbackY, worldZ);
}

void ChunkManager::Impl::beginSpawnPreload(const glm::vec3& spawnPos)
{
    startupState_ = StartupStreamingState{};
    startupState_.phase = StreamingPhase::ExactPreload;
    startupState_.preloadStarted = true;
    startupState_.spawnChunk = worldToChunkCoords(static_cast<int>(std::floor(spawnPos.x)),
                                                  std::max(static_cast<int>(std::floor(spawnPos.y)), 0),
                                                  static_cast<int>(std::floor(spawnPos.z)));
    startupState_.exactNearCurrentChunks = std::min(renderSettings_.nearChunks, 6);
    startupState_.farCurrentBlocks = 0;
    targetViewDistance_ = startupState_.exactNearCurrentChunks;
    if (viewDistance_ > targetViewDistance_)
    {
        viewDistance_ = targetViewDistance_;
    }
    farTerrainManager_.setDistanceBlocks(startupState_.farCurrentBlocks);
    farTerrainManager_.clear();
}

bool ChunkManager::Impl::isSpawnPreloadReady() const noexcept
{
    return !startupEnabled_ || !startupState_.preloadStarted || startupState_.phase != StreamingPhase::ExactPreload;
}

bool ChunkManager::Impl::playerReleaseReady() const noexcept
{
    return !startupEnabled_ || !startupState_.preloadStarted || startupState_.playerReleaseReady;
}

StreamingPhase ChunkManager::Impl::streamingPhase() const noexcept
{
    if (!startupEnabled_ || !startupState_.preloadStarted)
    {
        return StreamingPhase::SteadyState;
    }

    return startupState_.phase;
}

void ChunkManager::Impl::setStartupEnabled(bool enabled) noexcept
{
    startupEnabled_ = enabled;
    if (!startupEnabled_)
    {
        startupState_.phase = StreamingPhase::SteadyState;
        startupState_.playerReleaseReady = true;
        startupState_.exactNearCurrentChunks = renderSettings_.nearChunks;
        startupState_.farCurrentBlocks = renderSettings_.farBlocks;
        targetViewDistance_ = renderSettings_.nearChunks;
        farTerrainManager_.setDistanceBlocks(renderSettings_.farBlocks);
    }
}

bool ChunkManager::Impl::startupEnabled() const noexcept
{
    return startupEnabled_;
}

StreamingStatusSnapshot ChunkManager::Impl::computeStreamingStatusSnapshot() const noexcept
{
    StreamingStatusSnapshot snapshot{};
    snapshot.phase = streamingPhase();
    snapshot.playerReleaseReady = playerReleaseReady();
    snapshot.exactPendingUploads = static_cast<int>(
        std::min<std::size_t>(pendingUploadsLastFrame_, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    snapshot.farReadyTiles = farTerrainManager_.readyTileCount();
    snapshot.farQueuedTiles = farTerrainManager_.queuedTileCount();

    const int horizontalRadius = std::clamp(
        (startupEnabled_ && startupState_.preloadStarted && startupState_.exactNearCurrentChunks > 0)
            ? startupState_.exactNearCurrentChunks
            : renderSettings_.nearChunks,
        1,
        renderSettings_.nearChunks);
    const glm::ivec2 cameraColumn{lastCenterChunk_.x, lastCenterChunk_.z};
    const int cameraChunkY = lastCenterChunk_.y;
    const int verticalRadius = std::max(lastVerticalRadius_, kVerticalStreamingConfig.minRadiusChunks);

    int readyChunks = 0;
    int requiredChunks = 0;
    std::lock_guard<std::mutex> lock(chunksMutex);
    for (int dx = -horizontalRadius; dx <= horizontalRadius; ++dx)
    {
        for (int dz = -horizontalRadius; dz <= horizontalRadius; ++dz)
        {
            if (std::max(std::abs(dx), std::abs(dz)) > horizontalRadius)
            {
                continue;
            }

            const int chunkX = lastCenterChunk_.x + dx;
            const int chunkZ = lastCenterChunk_.z + dz;
            const glm::ivec2 column{chunkX, chunkZ};
            const int worldX = chunkX * kChunkSizeX + kChunkSizeX / 2;
            const int worldZ = chunkZ * kChunkSizeZ + kChunkSizeZ / 2;
            const int columnHeight = ensureColumnHeightCached(column, worldX, worldZ);
            const int columnRadius = columnRadiusForHeight(column,
                                                           cameraColumn,
                                                           cameraChunkY,
                                                           verticalRadius,
                                                           columnHeight);
            const int minChunkY = std::max(0, cameraChunkY - columnRadius);
            const int maxChunkY = std::max(minChunkY, cameraChunkY + columnRadius);
            for (int chunkY = minChunkY; chunkY <= maxChunkY; ++chunkY)
            {
                ++requiredChunks;
                const auto it = chunks_.find(glm::ivec3{chunkX, chunkY, chunkZ});
                if (it == chunks_.end() || !it->second)
                {
                    continue;
                }

                const ChunkState state = it->second->state.load(std::memory_order_acquire);
                if (state == ChunkState::Uploaded || state == ChunkState::Ready || state == ChunkState::Remeshing)
                {
                    ++readyChunks;
                }
            }
        }
    }

    snapshot.exactReadyChunks = readyChunks;
    snapshot.exactRequiredChunks = requiredChunks;

    if (snapshot.playerReleaseReady)
    {
        snapshot.blockingReason = "ready";
    }
    else if (snapshot.exactReadyChunks < snapshot.exactRequiredChunks)
    {
        snapshot.blockingReason = "waiting for exact chunks";
    }
    else if (snapshot.exactPendingUploads > 8)
    {
        snapshot.blockingReason = "waiting for mesh uploads";
    }
    else
    {
        snapshot.blockingReason = "stabilizing preload";
    }

    return snapshot;
}

StreamingStatusSnapshot ChunkManager::Impl::streamingStatusSnapshot() const noexcept
{
    return computeStreamingStatusSnapshot();
}

ChunkProfilingSnapshot ChunkManager::Impl::sampleProfilingSnapshot()
{
    ChunkProfilingSnapshot snapshot{};
    const StreamingStatusSnapshot status = computeStreamingStatusSnapshot();
    snapshot.phase = status.phase;

    const int generated = profilingCounters_.generatedChunks.exchange(0, std::memory_order_relaxed);
    const int meshed = profilingCounters_.meshedChunks.exchange(0, std::memory_order_relaxed);
    const int uploaded = profilingCounters_.uploadedChunks.exchange(0, std::memory_order_relaxed);

    snapshot.generatedChunks = generated;
    snapshot.meshedChunks = meshed;
    snapshot.uploadedChunks = uploaded;
    snapshot.uploadedBytes = profilingCounters_.uploadedBytes.exchange(0, std::memory_order_relaxed);
    snapshot.throttledUploads = profilingCounters_.throttledUploads.exchange(0, std::memory_order_relaxed);
    snapshot.deferredUploads = profilingCounters_.deferredUploads.exchange(0, std::memory_order_relaxed);
    snapshot.evictedChunks = profilingCounters_.evictedChunks.exchange(0, std::memory_order_relaxed);
    snapshot.verticalRadius = lastVerticalRadius_;
    snapshot.generationBudget = lastGenerationBudget_;
    snapshot.generationJobsIssued = lastGenerationJobsIssued_;
    snapshot.ringExpansionBudget = lastRingBudget_;
    snapshot.ringExpansionsUsed = lastRingExpansionsUsed_;
    snapshot.missingChunks = lastMissingChunks_;
    snapshot.generationBacklogSteps = lastBacklogSteps_;
    snapshot.generationColumnCap =
        (lastColumnCap_ >= std::numeric_limits<int>::max()) ? -1 : std::max(lastColumnCap_, 0);
    snapshot.workerThreads = static_cast<int>(workerThreadCount_);

    const long long genMicros = profilingCounters_.generationMicros.exchange(0, std::memory_order_relaxed);
    const long long meshMicros = profilingCounters_.meshingMicros.exchange(0, std::memory_order_relaxed);

    if (generated > 0)
    {
        snapshot.averageGenerationMs = static_cast<double>(genMicros) /
                                       (1000.0 * static_cast<double>(generated));
    }
    if (meshed > 0)
    {
        snapshot.averageMeshingMs = static_cast<double>(meshMicros) /
                                    (1000.0 * static_cast<double>(meshed));
    }

    snapshot.uploadBudgetBytes = uploadBudgetBytesThisFrame_;
    snapshot.uploadColumnLimit = uploadColumnLimitThisFrame_;
    snapshot.updateMsLastFrame = updateMsLastFrame_;
    snapshot.uploadMsLastFrame = lastUploadMsUsed_;
    const std::size_t pendingUploads = pendingUploadsLastFrame_;
    snapshot.pendingUploadChunks = static_cast<int>(
        std::min<std::size_t>(pendingUploads, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    snapshot.farBuildMsAverage = farTerrainManager_.averageBuildMs();
    snapshot.farCollectMsLastFrame = farTerrainManager_.lastCollectMs();
    snapshot.farUploadMsLastFrame = farTerrainManager_.lastUploadMs();
    snapshot.farActiveTiles = farTerrainManager_.activeTileCount();
    snapshot.farDirtyTiles = farTerrainManager_.dirtyTileCount();
    snapshot.farShellTilesReady = farTerrainManager_.readyTileCount();
    snapshot.farTilesBuilt = farTerrainManager_.builtTilesLastUpdate();
    snapshot.farTilesQueued = farTerrainManager_.queuedTileCount();
    snapshot.farTilesPendingUpload = farTerrainManager_.pendingUploadTileCount();
    snapshot.exactChunksReady = status.exactReadyChunks;
    snapshot.exactChunksPending = std::max(status.exactRequiredChunks - status.exactReadyChunks, 0);

    return snapshot;
}

std::string ChunkManager::Impl::biomeNameAt(const glm::vec3& worldPos) const
{
    const int worldX = static_cast<int>(std::floor(worldPos.x));
    const int worldZ = static_cast<int>(std::floor(worldPos.z));
    const ColumnSample sample = sampleColumn(worldX, worldZ);
    if (sample.dominantBiome)
    {
        return sample.dominantBiome->name;
    }

    return "Unknown";
}

void ChunkManager::Impl::startWorkerThreads()
{
    shouldStop_.store(false, std::memory_order_release);

    unsigned concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0)
    {
        concurrency = 2;
    }

    unsigned desired = 1u;
    if (concurrency >= 12)
    {
        desired = 6u;
    }
    else if (concurrency >= 8)
    {
        desired = 4u;
    }
    else
    {
        desired = std::max(1u, concurrency > 2 ? concurrency - 2 : 1u);
    }

    if (kVerticalStreamingConfig.maxWorkerThreads > 0)
    {
        desired = std::min(desired, static_cast<unsigned>(kVerticalStreamingConfig.maxWorkerThreads));
    }

    workerThreadCount_ = static_cast<std::size_t>(desired);
    workerThreads_.reserve(workerThreadCount_);

    for (std::size_t i = 0; i < workerThreadCount_; ++i)
    {
        workerThreads_.emplace_back(&ChunkManager::Impl::workerThreadFunction, this);
    }
}

void ChunkManager::Impl::stopWorkerThreads()
{
    shouldStop_.store(true, std::memory_order_release);
    jobQueue_.stop();

    for (auto& thread : workerThreads_)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
    workerThreads_.clear();
    workerThreadCount_ = 0;
}

void ChunkManager::Impl::workerThreadFunction()
{
    while (!shouldStop_.load(std::memory_order_acquire))
    {
        try
        {
            Job job = jobQueue_.waitAndPop();
            processJob(job);
        }
        catch (const std::runtime_error&)
        {
            break;
        }
        catch (const std::exception& ex)
        {
            std::cerr << "Worker thread error: " << ex.what() << std::endl;
        }
    }
}

void ChunkManager::Impl::enqueueJob(const std::shared_ptr<Chunk>& chunk, JobType type, const glm::ivec3& coord)
{
    if (!chunk)
    {
        return;
    }

    chunk->inFlight.fetch_add(1, std::memory_order_relaxed);
    try
    {
        jobQueue_.push(Job(type, coord, chunk));
    }
    catch (...)
    {
        chunk->inFlight.fetch_sub(1, std::memory_order_relaxed);
        throw;
    }
}

void ChunkManager::Impl::processJob(const Job& job)
{
    std::shared_ptr<Chunk> chunk = job.chunk;
    if (!chunk)
    {
        return;
    }

    struct FlightGuard
    {
        Chunk* chunkPtr;
        explicit FlightGuard(Chunk* ptr) : chunkPtr(ptr) {}
        ~FlightGuard()
        {
            if (chunkPtr)
            {
                chunkPtr->inFlight.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    } guard(chunk.get());

    if (job.type == JobType::Generate)
    {
        const auto start = std::chrono::steady_clock::now();
        generateChunkBlocks(*chunk);
        relightAroundChunk(job.chunkCoord);
        const auto end = std::chrono::steady_clock::now();
        const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        profilingCounters_.generationMicros.fetch_add(micros, std::memory_order_relaxed);
        profilingCounters_.generatedChunks.fetch_add(1, std::memory_order_relaxed);

        if (chunk->hasBlocks.load(std::memory_order_acquire))
        {
            chunk->pendingMeshRefresh.store(false, std::memory_order_release);
            chunk->state.store(ChunkState::Meshing, std::memory_order_release);
            enqueueJob(chunk, JobType::Mesh, job.chunkCoord);
        }
        else
        {
            chunk->state.store(ChunkState::Uploaded, std::memory_order_release);
            chunk->meshReady.store(false, std::memory_order_release);
            chunk->indexCount.store(0, std::memory_order_release);
        }
    }
    else if (job.type == JobType::Mesh)
    {
        const auto start = std::chrono::steady_clock::now();
        chunk->pendingMeshRefresh.store(false, std::memory_order_release);
        buildChunkMeshAsync(*chunk);
        const auto end = std::chrono::steady_clock::now();
        const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        profilingCounters_.meshingMicros.fetch_add(micros, std::memory_order_relaxed);
        profilingCounters_.meshedChunks.fetch_add(1, std::memory_order_relaxed);

        const bool meshEmpty = chunk->meshData.empty();
        if (meshEmpty)
        {
            chunk->state.store(ChunkState::Uploaded, std::memory_order_release);
        }
        else
        {
            chunk->state.store(ChunkState::Ready, std::memory_order_release);
        }

        if (chunk->pendingMeshRefresh.exchange(false, std::memory_order_acq_rel))
        {
            chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
            enqueueJob(chunk, JobType::Mesh, job.chunkCoord);
            return;
        }

        if (chunk->meshData.empty())
        {
            recycleChunkGPU(*chunk);
            chunk->meshReady.store(false, std::memory_order_release);
            chunk->indexCount.store(0, std::memory_order_release);
            return;
        }

        if (chunk->state.load(std::memory_order_acquire) == ChunkState::Ready)
        {
            queueChunkForUpload(chunk);
        }
    }
}

std::shared_ptr<Chunk> ChunkManager::Impl::popNextChunkForUpload()
{
    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    while (!uploadQueue_.empty())
    {
        std::shared_ptr<Chunk> chunk = uploadQueue_.front().lock();
        uploadQueue_.pop_front();
        if (!chunk)
        {
            continue;
        }

        chunk->queuedForUpload.store(false, std::memory_order_release);
        return chunk;
    }
    return nullptr;
}

void ChunkManager::Impl::queueChunkForUpload(const std::shared_ptr<Chunk>& chunk)
{
    if (!chunk)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    if (chunk->queuedForUpload.load(std::memory_order_acquire))
    {
        return;
    }

    uploadQueue_.emplace_back(chunk);
    chunk->queuedForUpload.store(true, std::memory_order_release);
}

void ChunkManager::Impl::requeueChunkForUpload(const std::shared_ptr<Chunk>& chunk, bool toFront)
{
    if (!chunk)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    if (chunk->queuedForUpload.load(std::memory_order_acquire))
    {
        return;
    }

    if (toFront)
    {
        uploadQueue_.emplace_front(chunk);
    }
    else
    {
        uploadQueue_.emplace_back(chunk);
    }
    chunk->queuedForUpload.store(true, std::memory_order_release);
}

std::size_t ChunkManager::Impl::nextPowerOfTwo(std::size_t value) noexcept
{
    if (value <= 1)
    {
        return 1;
    }

    value -= 1;
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

ChunkManager::Impl::ChunkBufferPage ChunkManager::Impl::createBufferPage(std::size_t vertexCount, std::size_t indexCount)
{
    static constexpr std::size_t kDefaultVertexCapacity = 262144;
    static constexpr std::size_t kDefaultIndexCapacity = 393216;

    ChunkBufferPage page;
    page.vertexCapacity = std::max(nextPowerOfTwo(vertexCount), kDefaultVertexCapacity);
    page.indexCapacity = std::max(nextPowerOfTwo(indexCount), kDefaultIndexCapacity);
    page.vertexBuffer = createDefaultBuffer(device_.Get(),
                                            static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex)),
                                            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    page.indexBuffer = createDefaultBuffer(device_.Get(),
                                           static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t)),
                                           D3D12_RESOURCE_STATE_INDEX_BUFFER);
    page.vertexUploadBuffer = createUploadBuffer(device_.Get(),
                                                 static_cast<std::uint64_t>(page.vertexCapacity * sizeof(Vertex)),
                                                 page.mappedVertexData);
    page.indexUploadBuffer = createUploadBuffer(device_.Get(),
                                                static_cast<std::uint64_t>(page.indexCapacity * sizeof(std::uint32_t)),
                                                page.mappedIndexData);
    page.vertexView.BufferLocation = page.vertexBuffer ? page.vertexBuffer->GetGPUVirtualAddress() : 0;
    page.vertexView.SizeInBytes = static_cast<UINT>(page.vertexCapacity * sizeof(Vertex));
    page.vertexView.StrideInBytes = sizeof(Vertex);
    page.indexView.BufferLocation = page.indexBuffer ? page.indexBuffer->GetGPUVirtualAddress() : 0;
    page.indexView.SizeInBytes = static_cast<UINT>(page.indexCapacity * sizeof(std::uint32_t));
    page.indexView.Format = DXGI_FORMAT_R32_UINT;

    return page;
}

ChunkManager::Impl::ChunkAllocation ChunkManager::Impl::acquireChunkAllocation(std::size_t vertexCount,
                                                                               std::size_t indexCount)
{
    ChunkAllocation allocation{};
    if (vertexCount == 0 || indexCount == 0)
    {
        return allocation;
    }

    auto tryAllocateRange = [](std::vector<ChunkBufferPage::Range>& ranges,
                               std::size_t& cursor,
                               std::size_t capacity,
                               std::size_t count,
                               std::size_t& outOffset) -> bool
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
    };

    auto mergeRange = [](std::vector<ChunkBufferPage::Range>& ranges,
                         std::size_t offset,
                         std::size_t size)
    {
        if (size == 0)
        {
            return;
        }

        ChunkBufferPage::Range range{offset, size};
        auto it = std::lower_bound(ranges.begin(), ranges.end(), range.offset,
                                   [](const ChunkBufferPage::Range& lhs, std::size_t value)
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
    };

    std::lock_guard<std::mutex> lock(bufferPageMutex_);
    for (std::uint32_t pageIndex = 0; pageIndex < bufferPages_.size(); ++pageIndex)
    {
        ChunkBufferPage& page = bufferPages_[pageIndex];
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

        ++page.activeChunks;
        allocation.pageIndex = pageIndex;
        allocation.vertexOffset = vertexOffset;
        allocation.indexOffset = indexOffset;
        return allocation;
    }

    ChunkBufferPage newPage = createBufferPage(vertexCount, indexCount);
    bufferPages_.push_back(std::move(newPage));
    const std::uint32_t newIndex = static_cast<std::uint32_t>(bufferPages_.size() - 1);
    ChunkBufferPage& page = bufferPages_.back();

    std::size_t vertexOffset = 0;
    std::size_t indexOffset = 0;
    const bool vertexSuccess = tryAllocateRange(page.freeVertices, page.vertexCursor, page.vertexCapacity, vertexCount, vertexOffset);
    const bool indexSuccess = tryAllocateRange(page.freeIndices, page.indexCursor, page.indexCapacity, indexCount, indexOffset);
    (void)vertexSuccess;
    (void)indexSuccess;

    ++page.activeChunks;
    allocation.pageIndex = newIndex;
    allocation.vertexOffset = vertexOffset;
    allocation.indexOffset = indexOffset;
    return allocation;
}

void ChunkManager::Impl::releaseChunkAllocation(Chunk& chunk)
{
    const std::uint32_t pageIndex = chunk.bufferPageIndex.load(std::memory_order_acquire);
    if (pageIndex == kInvalidChunkBufferPage)
    {
        chunk.vertexCount.store(0, std::memory_order_relaxed);
        chunk.indexCount.store(0, std::memory_order_relaxed);
        chunk.vertexOffset.store(0, std::memory_order_relaxed);
        chunk.indexOffset.store(0, std::memory_order_relaxed);
        return;
    }

    const std::size_t vertexCount = chunk.vertexCount.load(std::memory_order_acquire);
    const std::size_t indexCount = static_cast<std::size_t>(chunk.indexCount.load(std::memory_order_acquire));
    const std::size_t vertexOffset = chunk.vertexOffset.load(std::memory_order_acquire);
    const std::size_t indexOffset = chunk.indexOffset.load(std::memory_order_acquire);

    chunk.bufferPageIndex.store(kInvalidChunkBufferPage, std::memory_order_release);
    chunk.vertexCount.store(0, std::memory_order_release);
    chunk.indexCount.store(0, std::memory_order_release);
    chunk.vertexOffset.store(0, std::memory_order_release);
    chunk.indexOffset.store(0, std::memory_order_release);

    auto mergeRange = [](std::vector<ChunkBufferPage::Range>& ranges,
                         std::size_t offset,
                         std::size_t size)
    {
        if (size == 0)
        {
            return;
        }

        ChunkBufferPage::Range range{offset, size};
        auto it = std::lower_bound(ranges.begin(), ranges.end(), range.offset,
                                   [](const ChunkBufferPage::Range& lhs, std::size_t value)
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
    };

    std::lock_guard<std::mutex> lock(bufferPageMutex_);
    if (pageIndex >= bufferPages_.size())
    {
        return;
    }

    ChunkBufferPage& page = bufferPages_[pageIndex];
    mergeRange(page.freeVertices, vertexOffset, vertexCount);
    mergeRange(page.freeIndices, indexOffset, indexCount);
    if (page.activeChunks > 0)
    {
        --page.activeChunks;
    }
}

void ChunkManager::Impl::recycleChunkGPU(Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(chunk.meshMutex);
    releaseChunkAllocation(chunk);
    chunk.meshData.clear();
    chunk.meshReady.store(false, std::memory_order_release);
    chunk.queuedForUpload.store(false, std::memory_order_release);
}

void ChunkManager::Impl::recycleChunkObject(std::shared_ptr<Chunk> chunk)
{
    if (!chunk)
    {
        return;
    }

    {
        std::lock_guard<std::mutex> meshLock(chunk->meshMutex);
        chunk->reset(chunk->coord);
    }

    std::lock_guard<std::mutex> lock(chunkPoolMutex_);
    if (chunkPool_.size() < kChunkPoolSoftCap)
    {
        chunkPool_.push_back(std::move(chunk));
    }
}

void ChunkManager::Impl::destroyBufferPages()
{
    std::lock_guard<std::mutex> lock(bufferPageMutex_);
    for (auto& page : bufferPages_)
    {
        page.vertexBuffer.Reset();
        page.indexBuffer.Reset();
        page.vertexUploadBuffer.Reset();
        page.indexUploadBuffer.Reset();
        page.mappedVertexData = nullptr;
        page.mappedIndexData = nullptr;
    }
    bufferPages_.clear();
}

void ChunkManager::Impl::resetColumnBudgets()
{
    jobsScheduledThisFrame_.clear();
}

int ChunkManager::Impl::baseUploadsPerColumnLimit(int verticalRadius) const noexcept
{
    const int ramp = std::max(0, verticalRadius - kVerticalStreamingConfig.minRadiusChunks);
    const int divisor = std::max(1, kVerticalStreamingConfig.uploadRampDivisor);
    const int bonus = ramp / divisor;
    const int base = kVerticalStreamingConfig.uploadBasePerColumn;
    const int maxLimit = kVerticalStreamingConfig.uploadMaxPerColumn;
    return std::clamp(base + bonus, base, maxLimit);
}

std::size_t ChunkManager::Impl::estimateUploadQueueSize()
{
    std::lock_guard<std::mutex> lock(uploadQueueMutex_);
    std::size_t count = 0;
    for (const auto& entry : uploadQueue_)
    {
        if (!entry.expired())
        {
            ++count;
        }
    }
    return count;
}

ChunkManager::Impl::UploadBudgets ChunkManager::Impl::computeUploadBudgets(int verticalRadius)
{
    UploadBudgets budgets{};
    budgets.columnLimit = baseUploadsPerColumnLimit(verticalRadius);
    budgets.queueSize = estimateUploadQueueSize();
    if (startupEnabled_ && startupState_.preloadStarted)
    {
        if (startupState_.phase == StreamingPhase::ExactPreload)
        {
            budgets.byteBudget = 4ull * 1024ull * 1024ull;
            budgets.columnLimit = std::min(budgets.columnLimit, 4);
            budgets.timeBudgetMs = 2.0;
        }
        else if (startupState_.phase == StreamingPhase::InteractiveNearOnly ||
                 startupState_.phase == StreamingPhase::FarRamp)
        {
            budgets.byteBudget = 8ull * 1024ull * 1024ull;
            budgets.columnLimit = std::min(budgets.columnLimit + 1, 6);
            budgets.timeBudgetMs = 3.0;
        }
        else
        {
            budgets.byteBudget = 32ull * 1024ull * 1024ull;
            budgets.timeBudgetMs = 4.0;
        }
    }
    else
    {
        budgets.byteBudget = 32ull * 1024ull * 1024ull;
        budgets.timeBudgetMs = 4.0;
    }

    return budgets;
}

int ChunkManager::Impl::computeBacklogSteps(int backlog, int threshold, int stepSize) noexcept
{
    if (backlog <= threshold)
    {
        return 0;
    }

    if (stepSize <= 0)
    {
        return 1;
    }

    const long long safeOver = static_cast<long long>(backlog) - static_cast<long long>(threshold);
    const long long safeStep = std::max(stepSize, 1);
    const long long steps = (safeOver + safeStep - 1) / safeStep;
    return static_cast<int>(std::min(steps, static_cast<long long>(std::numeric_limits<int>::max())));
}

int ChunkManager::Impl::computeGenerationBudget(int horizontalRadius, int verticalRadius, int backlogSteps) const
{
    const auto& tuning = kVerticalStreamingConfig.generationBudget;
    const int safeHorizontal = std::max(horizontalRadius, 0);
    const int safeVertical = std::max(verticalRadius, 0);

    double budget = static_cast<double>(tuning.baseJobsPerFrame);
    budget += static_cast<double>(tuning.jobsPerHorizontalRing) * static_cast<double>(safeHorizontal);
    budget += static_cast<double>(tuning.jobsPerVerticalLayer) * static_cast<double>(safeVertical);
    budget += static_cast<double>(tuning.backlogBoostPerStep)
              * static_cast<double>(std::max(backlogSteps, 0));

    long long result = static_cast<long long>(std::ceil(budget));
    if (tuning.maxJobsPerFrame > 0)
    {
        result = std::min(result, static_cast<long long>(tuning.maxJobsPerFrame));
    }

    result = std::max(result, 1ll);
    return static_cast<int>(std::min(result, static_cast<long long>(std::numeric_limits<int>::max())));
}

int ChunkManager::Impl::computeRingExpansionBudget(int backlogChunks) const
{
    const auto& tuning = kVerticalStreamingConfig.generationBudget;
    const int minRings = std::max(0, tuning.minRingExpansionsPerFrame);
    const int maxRings = std::max(minRings, tuning.maxRingExpansionsPerFrame);

    if (maxRings == 0)
    {
        return 0;
    }

    if (tuning.backlogRingStepSize <= 0)
    {
        return maxRings;
    }

    const int steps = computeBacklogSteps(backlogChunks,
                                          tuning.backlogStartThreshold,
                                          tuning.backlogRingStepSize);

    int budget = minRings + steps;
    budget = std::clamp(budget, minRings, maxRings);
    return budget;
}

int ChunkManager::Impl::computeColumnJobCap(int backlogSteps, int backlogChunks) const
{
    int baseCap = kVerticalStreamingConfig.maxGenerationJobsPerColumn;
    if (baseCap <= 0)
    {
        return std::numeric_limits<int>::max();
    }

    if (kVerticalStreamingConfig.backlogColumnCapReleaseThreshold > 0 &&
        backlogChunks >= kVerticalStreamingConfig.backlogColumnCapReleaseThreshold)
    {
        return std::numeric_limits<int>::max();
    }

    const int boostPerStep = kVerticalStreamingConfig.generationBudget.columnCapBoostPerStep;
    if (boostPerStep > 0 && backlogSteps > 0)
    {
        const long long boosted = static_cast<long long>(baseCap) +
                                  static_cast<long long>(backlogSteps) *
                                      static_cast<long long>(boostPerStep);
        baseCap = static_cast<int>(std::min(boosted, static_cast<long long>(std::numeric_limits<int>::max())));
    }

    return std::max(baseCap, 0);
}

int ChunkManager::Impl::estimateMissingChunks(const glm::ivec3& center,
                                              int horizontalRadius,
                                              int verticalRadius) const
{
    const glm::ivec2 cameraColumn{center.x, center.z};
    const int cameraChunkY = center.y;

    int missing = 0;
    std::lock_guard<std::mutex> lock(chunksMutex);
    for (int dx = -horizontalRadius; dx <= horizontalRadius; ++dx)
    {
        for (int dz = -horizontalRadius; dz <= horizontalRadius; ++dz)
        {
            if (std::max(std::abs(dx), std::abs(dz)) > horizontalRadius)
            {
                continue;
            }

            const int chunkX = center.x + dx;
            const int chunkZ = center.z + dz;
            const glm::ivec2 column{chunkX, chunkZ};
            const int worldX = chunkX * kChunkSizeX + kChunkSizeX / 2;
            const int worldZ = chunkZ * kChunkSizeZ + kChunkSizeZ / 2;
            const int columnHeight = ensureColumnHeightCached(column, worldX, worldZ);
            const int columnRadius = columnRadiusForHeight(column,
                                                           cameraColumn,
                                                           cameraChunkY,
                                                           verticalRadius,
                                                           columnHeight);
            const int minChunkY = std::max(0, cameraChunkY - columnRadius);
            const int maxChunkY = std::max(minChunkY, cameraChunkY + columnRadius);
            for (int chunkY = minChunkY; chunkY <= maxChunkY; ++chunkY)
            {
                const glm::ivec3 coord{chunkX, chunkY, chunkZ};
                if (chunks_.find(coord) == chunks_.end())
                {
                    ++missing;
                }
            }
        }
    }

    return missing;
}

int ChunkManager::Impl::computeVerticalRadius(const glm::ivec3& center,
                                              int horizontalRadius,
                                              int cameraWorldY)
{
    int verticalRadius = kVerticalStreamingConfig.minRadiusChunks;

    const glm::ivec2 cameraColumn{center.x, center.z};
    const int cameraChunkY = center.y;
    const int cameraWorldChunk = floorDiv(cameraWorldY, kChunkSizeY);
    verticalRadius = std::max(verticalRadius,
                              std::abs(cameraWorldChunk - cameraChunkY) +
                                  kVerticalStreamingConfig.columnSlackChunks);

    const int sampleRadius = std::max(0,
                                      std::min(horizontalRadius, kVerticalStreamingConfig.sampleRadiusChunks));

    for (int dx = -sampleRadius; dx <= sampleRadius; ++dx)
    {
        for (int dz = -sampleRadius; dz <= sampleRadius; ++dz)
        {
            const glm::ivec2 column{center.x + dx, center.z + dz};
            const int radius = columnRadiusFor(column,
                                               cameraColumn,
                                               cameraChunkY,
                                               verticalRadius);
            verticalRadius = std::max(verticalRadius, radius);
        }
    }

    return std::clamp(verticalRadius,
                      kVerticalStreamingConfig.minRadiusChunks,
                      kVerticalStreamingConfig.maxRadiusChunks);
}

bool ChunkManager::Impl::tryGetPredictedColumnHeight(const glm::ivec2& column, int& outHeight) const
{
    std::lock_guard<std::mutex> lock(predictedColumnMutex_);
    auto it = predictedColumnHeights_.find(column);
    if (it == predictedColumnHeights_.end())
    {
        return false;
    }

    outHeight = it->second;
    return true;
}

int ChunkManager::Impl::cacheSampledColumnHeight(const glm::ivec2& column, int worldX, int worldZ) const
{
    const ColumnSample sample = sampleColumn(worldX, worldZ);
    const int height = sample.surfaceY;
    {
        std::lock_guard<std::mutex> lock(predictedColumnMutex_);
        predictedColumnHeights_[column] = height;
    }
    return height;
}

int ChunkManager::Impl::ensureColumnHeightCached(const glm::ivec2& column,
                                                 int worldX,
                                                 int worldZ) const
{
    int highest = columnManager_.highestSolidBlock(worldX, worldZ);
    if (highest != ColumnManager::kNoHeight)
    {
        return highest;
    }

    int cachedHeight = ColumnManager::kNoHeight;
    if (tryGetPredictedColumnHeight(column, cachedHeight))
    {
        return cachedHeight;
    }

    return cacheSampledColumnHeight(column, worldX, worldZ);
}

void ChunkManager::Impl::invalidatePredictedColumn(const glm::ivec2& column) const
{
    std::lock_guard<std::mutex> lock(predictedColumnMutex_);
    predictedColumnHeights_.erase(column);
}

int ChunkManager::Impl::columnRadiusFor(const glm::ivec2& column,
                                        const glm::ivec2& cameraColumn,
                                        int cameraChunkY,
                                        int verticalRadius) const
{
    const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
    const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
    const int columnHeight = ensureColumnHeightCached(column, worldX, worldZ);
    return columnRadiusForHeight(column, cameraColumn, cameraChunkY, verticalRadius, columnHeight);
}

int ChunkManager::Impl::columnRadiusForHeight(const glm::ivec2& column,
                                              const glm::ivec2& cameraColumn,
                                              int cameraChunkY,
                                              int verticalRadius,
                                              int columnHeight) const
{
    int radius = std::max(verticalRadius, kVerticalStreamingConfig.minRadiusChunks);

    const int falloffStep = kVerticalStreamingConfig.verticalRadiusFalloffStep;
    if (falloffStep > 0)
    {
        const int horizontalDistance = std::max(std::abs(column.x - cameraColumn.x),
                                                std::abs(column.y - cameraColumn.y));
        if (horizontalDistance > 0)
        {
            const int reduction = horizontalDistance / falloffStep;
            if (reduction > 0)
            {
                radius = std::max(kVerticalStreamingConfig.minRadiusChunks, radius - reduction);
            }
        }
    }

    if (columnHeight != ColumnManager::kNoHeight)
    {
        const int highestChunk = floorDiv(columnHeight, kChunkSizeY);
        const int required = std::abs(highestChunk - cameraChunkY) +
                             kVerticalStreamingConfig.columnSlackChunks;
        radius = std::max(radius, required);
    }

    return std::clamp(radius,
                      kVerticalStreamingConfig.minRadiusChunks,
                      kVerticalStreamingConfig.maxRadiusChunks);
}

std::pair<int, int> ChunkManager::Impl::columnSpanFor(const glm::ivec2& column,
                                                       const glm::ivec2& cameraColumn,
                                                       int cameraChunkY,
                                                       int verticalRadius) const
{
    const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
    const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
    const int columnHeight = ensureColumnHeightCached(column, worldX, worldZ);
    return columnSpanForHeight(column, cameraColumn, cameraChunkY, verticalRadius, columnHeight);
}

std::pair<int, int> ChunkManager::Impl::columnSpanForHeight(const glm::ivec2& column,
                                                             const glm::ivec2& cameraColumn,
                                                             int cameraChunkY,
                                                             int verticalRadius,
                                                             int columnHeight) const
{
    const int radius = columnRadiusForHeight(column, cameraColumn, cameraChunkY, verticalRadius, columnHeight);
    const int minChunk = std::max(0, cameraChunkY - radius);
    const int maxChunk = std::max(minChunk, cameraChunkY + radius);
    return {minChunk, maxChunk};
}

ChunkManager::Impl::RingProgress ChunkManager::Impl::ensureVolume(const glm::ivec3& center,
                                                                  int horizontalRadius,
                                                                  int verticalRadius,
                                                                  int& jobBudget)
{
    bool missingFound = false;

    const glm::ivec2 cameraColumn{center.x, center.z};

    struct Candidate
    {
        glm::ivec3 coord;
        float priority{0.0f};
    };

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>((verticalRadius * 2 + 1) *
                                                std::max(1, horizontalRadius * 8)));

    std::unordered_set<glm::ivec2, ColumnHasher> visitedColumns;
    visitedColumns.reserve(static_cast<std::size_t>(std::max(1, horizontalRadius * 8)));
    const int maxJobsPerColumn = generationColumnCapThisFrame_;
    const bool enforceColumnCap = maxJobsPerColumn > 0 &&
                                  maxJobsPerColumn < std::numeric_limits<int>::max();

    auto enqueueColumn = [&](int chunkX, int chunkZ) {
        glm::ivec2 column{chunkX, chunkZ};
        if (!visitedColumns.insert(column).second)
        {
            return;
        }

        const int worldX = column.x * kChunkSizeX + kChunkSizeX / 2;
        const int worldZ = column.y * kChunkSizeZ + kChunkSizeZ / 2;
        const int columnHeight = ensureColumnHeightCached(column, worldX, worldZ);
        const auto [minChunkY, maxChunkY] = columnSpanForHeight(column,
                                                                cameraColumn,
                                                                center.y,
                                                                verticalRadius,
                                                                columnHeight);
        for (int chunkY = minChunkY; chunkY <= maxChunkY; ++chunkY)
        {
            const glm::ivec3 coord{chunkX, chunkY, chunkZ};
            const int dx = coord.x - center.x;
            const int dy = coord.y - center.y;
            const int dz = coord.z - center.z;
            const float horizontal = std::sqrt(static_cast<float>(dx * dx + dz * dz));
            const float priority = horizontal + 0.5f * static_cast<float>(std::abs(dy));
            candidates.push_back(Candidate{coord, priority});
        }
    };

    if (horizontalRadius == 0)
    {
        enqueueColumn(center.x, center.z);
    }
    else
    {
        for (int dx = -horizontalRadius; dx <= horizontalRadius; ++dx)
        {
            enqueueColumn(center.x + dx, center.z - horizontalRadius);
            enqueueColumn(center.x + dx, center.z + horizontalRadius);
        }
        for (int dz = -horizontalRadius + 1; dz <= horizontalRadius - 1; ++dz)
        {
            enqueueColumn(center.x - horizontalRadius, center.z + dz);
            enqueueColumn(center.x + horizontalRadius, center.z + dz);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.priority == rhs.priority)
        {
            if (lhs.coord.y == rhs.coord.y)
            {
                if (lhs.coord.x == rhs.coord.x)
                {
                    return lhs.coord.z < rhs.coord.z;
                }
                return lhs.coord.x < rhs.coord.x;
            }
            return lhs.coord.y < rhs.coord.y;
        }
        return lhs.priority < rhs.priority;
    });

    for (const Candidate& candidate : candidates)
    {
        if (jobBudget <= 0)
        {
            break;
        }

        const glm::ivec2 columnKey{candidate.coord.x, candidate.coord.z};
        int& columnJobs = jobsScheduledThisFrame_[columnKey];

        if (auto existing = getChunkShared(candidate.coord))
        {
            continue;
        }

        missingFound = true;

        if (enforceColumnCap && columnJobs >= maxJobsPerColumn)
        {
            continue;
        }

        if (ensureChunkAsync(candidate.coord, false))
        {
            --jobBudget;
            ++columnJobs;
        }
    }

    return RingProgress{!missingFound, jobBudget <= 0};
}

void ChunkManager::Impl::removeDistantChunks(const glm::ivec3& center,
                                             int horizontalThreshold,
                                             int verticalRadius)
{
    std::vector<glm::ivec3> toRemove;
    const glm::ivec2 cameraColumn{center.x, center.z};
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        toRemove.reserve(chunks_.size());
        for (const auto& [coord, chunkPtr] : chunks_)
        {
            if (coord.y < 0)
            {
                toRemove.push_back(coord);
                continue;
            }

            const int dx = coord.x - center.x;
            const int dz = coord.z - center.z;
            const int horizontalDistance = std::max(std::abs(dx), std::abs(dz));
            if (horizontalDistance > horizontalThreshold)
            {
                toRemove.push_back(coord);
                continue;
            }

            const glm::ivec2 column{coord.x, coord.z};
            const auto [minChunkY, maxChunkY] = columnSpanFor(column,
                                                              cameraColumn,
                                                              center.y,
                                                              verticalRadius);
            const int slack = kVerticalStreamingConfig.columnSlackChunks;
            if (coord.y < (minChunkY - slack) || coord.y > (maxChunkY + slack))
            {
                toRemove.push_back(coord);
            }
        }
    }

    int evictedCount = 0;
    for (const glm::ivec3& coord : toRemove)
    {
        std::shared_ptr<Chunk> chunk;
        {
            std::lock_guard<std::mutex> lock(chunksMutex);
            auto it = chunks_.find(coord);
            if (it == chunks_.end())
            {
                continue;
            }

            if (it->second->inFlight.load(std::memory_order_acquire) != 0)
            {
                continue;
            }

            chunk = it->second;
            chunks_.erase(it);
        }

        if (chunk)
        {
            columnManager_.removeChunk(*chunk);
            invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});
            recycleChunkGPU(*chunk);
            recycleChunkObject(std::move(chunk));
            ++evictedCount;
        }
    }

    if (evictedCount > 0)
    {
        profilingCounters_.evictedChunks.fetch_add(evictedCount, std::memory_order_relaxed);
    }
}

bool ChunkManager::Impl::shouldUseSurfaceOnly(const glm::ivec3& center, const glm::ivec3& coord) const noexcept
{
    (void)center;
    (void)coord;
    return false;
}

bool ChunkManager::Impl::ensureChunkAsync(const glm::ivec3& coord, bool surfaceOnly)
{
    (void)surfaceOnly;

    if (coord.y < 0)
    {
        return false;
    }

    try
    {
        std::shared_ptr<Chunk> chunk;
        {
            std::lock_guard<std::mutex> lock(chunksMutex);
            auto it = chunks_.find(coord);
            if (it != chunks_.end())
            {
                return false;
            }

            chunk = acquireChunk(coord);
            chunk->state.store(ChunkState::Generating, std::memory_order_release);
            chunk->surfaceOnly = false;
            chunk->lodData.reset();
            chunks_.emplace(coord, chunk);
        }

        enqueueJob(chunk, JobType::Generate, coord);
        return true;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error creating chunk at (" << coord.x << ", " << coord.y << ", " << coord.z
                  << "): " << ex.what() << std::endl;
        return false;
    }
}

void ChunkManager::Impl::uploadReadyMeshes()
{
    const std::size_t initialBudget = uploadBudgetBytesThisFrame_;
    std::size_t remainingBudget = initialBudget;
    bool uploadedAnything = false;
    std::unordered_map<glm::ivec2, int, ColumnHasher> uploadsPerColumn;
    std::size_t attempts = 0;
    const int columnUploadLimit = std::max(1, uploadColumnLimitThisFrame_);
    const auto uploadStart = std::chrono::steady_clock::now();
    if (uploadContext_.ready())
    {
        uploadContext_.begin();
    }

    while ((remainingBudget > 0 || !uploadedAnything) && attempts < kUploadQueueScanLimit)
    {
        ++attempts;
        std::shared_ptr<Chunk> chunk = popNextChunkForUpload();
        if (!chunk)
        {
            break;
        }

        if (!chunk->meshReady.load(std::memory_order_acquire) ||
            chunk->state.load(std::memory_order_acquire) != ChunkState::Ready)
        {
            continue;
        }

        const glm::ivec2 columnKey{chunk->coord.x, chunk->coord.z};
        int& columnUploads = uploadsPerColumn[columnKey];
        if (columnUploads >= columnUploadLimit)
        {
            requeueChunkForUpload(chunk, false);
            profilingCounters_.throttledUploads.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        std::size_t vertexBytes = 0;
        std::size_t indexBytes = 0;
        {
            std::lock_guard<std::mutex> meshLock(chunk->meshMutex);
            vertexBytes = chunk->meshData.vertices.size() * sizeof(Vertex);
            indexBytes = chunk->meshData.indices.size() * sizeof(std::uint32_t);
        }
        const std::size_t totalBytes = vertexBytes + indexBytes;

        if (uploadedAnything && totalBytes > remainingBudget && totalBytes > 0)
        {
            requeueChunkForUpload(chunk, true);
            profilingCounters_.deferredUploads.fetch_add(1, std::memory_order_relaxed);
            break;
        }

        uploadChunkMesh(*chunk);
        chunk->state.store(ChunkState::Uploaded, std::memory_order_release);
        chunk->meshReady.store(false, std::memory_order_release);
        uploadedAnything = true;
        ++columnUploads;

        profilingCounters_.uploadedChunks.fetch_add(1, std::memory_order_relaxed);
        profilingCounters_.uploadedBytes.fetch_add(totalBytes, std::memory_order_relaxed);

        if (totalBytes >= remainingBudget)
        {
            remainingBudget = 0;
        }
        else
        {
            remainingBudget -= totalBytes;
        }

        const double elapsedMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - uploadStart).count();
        if (elapsedMs >= uploadBudgetMsThisFrame_ && uploadedAnything)
        {
            break;
        }
    }

    uploadContext_.flush();

    if (initialBudget > remainingBudget)
    {
        lastUploadBytesUsed_ = initialBudget - remainingBudget;
    }
    else
    {
        lastUploadBytesUsed_ = 0;
    }
    lastUploadMsUsed_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - uploadStart).count();

    pendingUploadsLastFrame_ = estimateUploadQueueSize();
}

void ChunkManager::Impl::uploadChunkMesh(Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(chunk.meshMutex);

    if (chunk.meshData.empty())
    {
        releaseChunkAllocation(chunk);
        chunk.meshData.clear();
        return;
    }

    const std::size_t vertexCount = chunk.meshData.vertices.size();
    const std::size_t indexCount = chunk.meshData.indices.size();

    releaseChunkAllocation(chunk);
    ChunkAllocation allocation = acquireChunkAllocation(vertexCount, indexCount);
    if (allocation.pageIndex == kInvalidChunkBufferPage)
    {
        chunk.meshData.clear();
        return;
    }

    chunk.bufferPageIndex.store(allocation.pageIndex, std::memory_order_release);
    chunk.vertexOffset.store(allocation.vertexOffset, std::memory_order_release);
    chunk.indexOffset.store(allocation.indexOffset, std::memory_order_release);
    chunk.vertexCount.store(vertexCount, std::memory_order_release);
    {
        std::lock_guard<std::mutex> pageLock(bufferPageMutex_);
        if (allocation.pageIndex < bufferPages_.size())
        {
            ChunkBufferPage& page = bufferPages_[allocation.pageIndex];
            if (page.mappedVertexData != nullptr && vertexCount > 0)
            {
                const std::size_t chunkVertexOffset = chunk.vertexOffset.load(std::memory_order_acquire);
                std::memcpy(page.mappedVertexData + chunkVertexOffset * sizeof(Vertex),
                            chunk.meshData.vertices.data(),
                            vertexCount * sizeof(Vertex));
                if (uploadContext_.ready() && page.vertexUploadBuffer != nullptr && page.vertexBuffer != nullptr)
                {
                    uploadContext_.transition(page.vertexBuffer.Get(),
                                              page.vertexState,
                                              D3D12_RESOURCE_STATE_COPY_DEST);
                    page.vertexState = D3D12_RESOURCE_STATE_COPY_DEST;
                    uploadContext_.copyBuffer(page.vertexBuffer.Get(),
                                              static_cast<std::uint64_t>(chunkVertexOffset * sizeof(Vertex)),
                                              page.vertexUploadBuffer.Get(),
                                              static_cast<std::uint64_t>(chunkVertexOffset * sizeof(Vertex)),
                                              static_cast<std::uint64_t>(vertexCount * sizeof(Vertex)));
                    uploadContext_.transition(page.vertexBuffer.Get(),
                                              page.vertexState,
                                              D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
                    page.vertexState = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
                }
            }
            if (page.mappedIndexData != nullptr && indexCount > 0)
            {
                const std::size_t chunkIndexOffset = chunk.indexOffset.load(std::memory_order_acquire);
                std::memcpy(page.mappedIndexData + chunkIndexOffset * sizeof(std::uint32_t),
                            chunk.meshData.indices.data(),
                            indexCount * sizeof(std::uint32_t));
                if (uploadContext_.ready() && page.indexUploadBuffer != nullptr && page.indexBuffer != nullptr)
                {
                    uploadContext_.transition(page.indexBuffer.Get(),
                                              page.indexState,
                                              D3D12_RESOURCE_STATE_COPY_DEST);
                    page.indexState = D3D12_RESOURCE_STATE_COPY_DEST;
                    uploadContext_.copyBuffer(page.indexBuffer.Get(),
                                              static_cast<std::uint64_t>(chunkIndexOffset * sizeof(std::uint32_t)),
                                              page.indexUploadBuffer.Get(),
                                              static_cast<std::uint64_t>(chunkIndexOffset * sizeof(std::uint32_t)),
                                              static_cast<std::uint64_t>(indexCount * sizeof(std::uint32_t)));
                    uploadContext_.transition(page.indexBuffer.Get(),
                                              page.indexState,
                                              D3D12_RESOURCE_STATE_INDEX_BUFFER);
                    page.indexState = D3D12_RESOURCE_STATE_INDEX_BUFFER;
                }
            }
        }
    }

    chunk.indexCount.store(static_cast<std::uint32_t>(indexCount), std::memory_order_release);

    chunk.meshData.clear();
}

void ChunkManager::Impl::buildSurfaceOnlyMesh(Chunk& chunk)
{
    if (!chunk.lodData)
    {
        chunk.meshData.clear();
        return;
    }

    FarChunk& lod = *chunk.lodData;
    chunk.meshData.clear();

    const glm::vec3 baseOrigin = lod.origin;
    const glm::vec3 normal{0.0f, 1.0f, 0.0f};
    const float step = static_cast<float>(lod.lodStep);

    for (int rx = 0; rx < FarChunk::kColumnsX; ++rx)
    {
        for (int rz = 0; rz < FarChunk::kColumnsZ; ++rz)
        {
            const FarChunk::SurfaceCell& cell = lod.strata[FarChunk::index(rx, rz)];
            if (cell.block == BlockId::Air || cell.worldY == std::numeric_limits<int>::min())
            {
                continue;
            }

            const float worldY = static_cast<float>(cell.worldY + 1);
            const float minX = baseOrigin.x + static_cast<float>(rx) * step;
            const float maxX = baseOrigin.x + static_cast<float>(rx + 1) * step;
            const float minZ = baseOrigin.z + static_cast<float>(rz) * step;
            const float maxZ = baseOrigin.z + static_cast<float>(rz + 1) * step;

            const glm::vec3 p0{minX, worldY, minZ};
            const glm::vec3 p1{maxX, worldY, minZ};
            const glm::vec3 p2{maxX, worldY, maxZ};
            const glm::vec3 p3{minX, worldY, maxZ};

            const auto& uv = blockUVTable_[toIndex(cell.block)].faces[toIndex(BlockFace::Top)];
            const std::uint32_t lightingData = packVertexLighting(packLightLevels(kMaxLightLevel, 0));

            const std::uint32_t baseIndex = static_cast<std::uint32_t>(chunk.meshData.vertices.size());
            chunk.meshData.vertices.push_back(Vertex{p0, normal, glm::vec2{0.0f, 0.0f}, uv.base, uv.size, lightingData});
            chunk.meshData.vertices.push_back(Vertex{p1, normal, glm::vec2{1.0f, 0.0f}, uv.base, uv.size, lightingData});
            chunk.meshData.vertices.push_back(Vertex{p2, normal, glm::vec2{1.0f, 1.0f}, uv.base, uv.size, lightingData});
            chunk.meshData.vertices.push_back(Vertex{p3, normal, glm::vec2{0.0f, 1.0f}, uv.base, uv.size, lightingData});

            chunk.meshData.indices.push_back(baseIndex + 0);
            chunk.meshData.indices.push_back(baseIndex + 1);
            chunk.meshData.indices.push_back(baseIndex + 2);
            chunk.meshData.indices.push_back(baseIndex + 0);
            chunk.meshData.indices.push_back(baseIndex + 2);
            chunk.meshData.indices.push_back(baseIndex + 3);
        }
    }
}

void ChunkManager::Impl::buildChunkMeshAsync(Chunk& chunk)
{
    std::vector<BlockId> chunkBlocks;
    std::vector<std::uint8_t> chunkLightLevels;
    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        chunk.meshData.clear();

        if (!chunk.hasBlocks.load(std::memory_order_acquire))
        {
            chunk.meshReady.store(true, std::memory_order_release);
            return;
        }

        chunkBlocks = chunk.blocks;
        chunkLightLevels = chunk.lightLevels;
    }

    MeshData meshData;

    const int baseWorldX = chunk.coord.x * kChunkSizeX;
    const int baseWorldY = chunk.minWorldY;
    const int baseWorldZ = chunk.coord.z * kChunkSizeZ;
    const glm::vec3 chunkOrigin(static_cast<float>(baseWorldX), static_cast<float>(baseWorldY), static_cast<float>(baseWorldZ));

    auto isInsideChunk = [](const glm::ivec3& local) noexcept
    {
        return local.x >= 0 && local.x < kChunkSizeX &&
               local.y >= 0 && local.y < kChunkSizeY &&
               local.z >= 0 && local.z < kChunkSizeZ;
    };

    auto localToWorld = [&](int lx, int ly, int lz) -> glm::ivec3
    {
        return glm::ivec3(baseWorldX + lx, baseWorldY + ly, baseWorldZ + lz);
    };

    auto sampleBlock = [&](int lx, int ly, int lz) -> BlockId
    {
        if (lx >= 0 && lx < kChunkSizeX && ly >= 0 && ly < kChunkSizeY && lz >= 0 && lz < kChunkSizeZ)
        {
            return chunkBlocks[blockIndex(lx, ly, lz)];
        }

        const glm::ivec3 worldPos = localToWorld(lx, ly, lz);
        const glm::ivec3 sampleChunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
        auto sampleChunk = getChunkShared(sampleChunkCoord);
        if (!sampleChunk || worldPos.y < sampleChunk->minWorldY || worldPos.y > sampleChunk->maxWorldY)
        {
            return BlockId::Air;
        }

        const glm::ivec3 local = localBlockCoords(worldPos, sampleChunkCoord);
        if (local.x < 0 || local.x >= kChunkSizeX ||
            local.z < 0 || local.z >= kChunkSizeZ)
        {
            return BlockId::Air;
        }

        std::lock_guard<std::mutex> sampleLock(sampleChunk->meshMutex);
        const int localY = worldPos.y - sampleChunk->minWorldY;
        return sampleChunk->blocks[blockIndex(local.x, localY, local.z)];
    };

    auto samplePackedLight = [&](int lx, int ly, int lz) -> std::uint8_t
    {
        if (lx >= 0 && lx < kChunkSizeX && ly >= 0 && ly < kChunkSizeY && lz >= 0 && lz < kChunkSizeZ)
        {
            return chunkLightLevels[blockIndex(lx, ly, lz)];
        }

        const glm::ivec3 worldPos = localToWorld(lx, ly, lz);
        if (worldPos.y < 0)
        {
            return packLightLevels(0, 0);
        }

        const glm::ivec3 sampleChunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
        auto sampleChunk = getChunkShared(sampleChunkCoord);
        if (!sampleChunk || worldPos.y < sampleChunk->minWorldY || worldPos.y > sampleChunk->maxWorldY)
        {
            return packLightLevels(kMaxLightLevel, 0);
        }

        const glm::ivec3 local = localBlockCoords(worldPos, sampleChunkCoord);
        if (local.x < 0 || local.x >= kChunkSizeX ||
            local.z < 0 || local.z >= kChunkSizeZ)
        {
            return packLightLevels(kMaxLightLevel, 0);
        }

        std::lock_guard<std::mutex> sampleLock(sampleChunk->meshMutex);
        const int localY = worldPos.y - sampleChunk->minWorldY;
        return sampleChunk->lightLevels[blockIndex(local.x, localY, local.z)];
    };

    enum class Axis : int { X = 0, Y = 1, Z = 2 };
    enum class FaceDir : int { Negative = 0, Positive = 1 };

    struct FaceMaterial
    {
        glm::vec2 uvBase{0.0f};
        glm::vec2 uvSize{1.0f};
        glm::ivec3 uAxis{1, 0, 0};
        glm::ivec3 vAxis{0, 1, 0};
        BlockFace face{BlockFace::Top};
        std::array<std::uint32_t, 4> lightingData{};
        bool mergeable{true};

        bool operator==(const FaceMaterial& other) const noexcept
        {
            return uvBase == other.uvBase &&
                   uvSize == other.uvSize &&
                   uAxis == other.uAxis &&
                   vAxis == other.vAxis &&
                   face == other.face &&
                   lightingData == other.lightingData &&
                   mergeable == other.mergeable;
        }
    };

    struct MaskCell
    {
        bool exists{false};
        FaceMaterial material{};
    };

	    const std::array<glm::vec3, 3> axisNormals{
	        glm::vec3{1.0f, 0.0f, 0.0f},
	        glm::vec3{0.0f, 1.0f, 0.0f},
	        glm::vec3{0.0f, 0.0f, 1.0f}
	    };

        constexpr std::array<int, 4> kCornerUSigns{-1, 1, 1, -1};
        constexpr std::array<int, 4> kCornerVSigns{-1, -1, 1, 1};

	    auto faceFromNormal = [](const glm::vec3& normal) noexcept -> BlockFace
	    {
	        if (normal.y > 0.5f) return BlockFace::Top;
	        if (normal.y < -0.5f) return BlockFace::Bottom;
	        if (normal.x > 0.5f) return BlockFace::East;
	        if (normal.x < -0.5f) return BlockFace::West;
	        if (normal.z > 0.5f) return BlockFace::South;
	        return BlockFace::North;
	    };

	    auto faceSampleAxes = [](BlockFace face, glm::ivec3& uAxis, glm::ivec3& vAxis) noexcept
	    {
	        switch (face)
	        {
	        case BlockFace::Top:
	        case BlockFace::Bottom:
	            uAxis = glm::ivec3(1, 0, 0);
	            vAxis = glm::ivec3(0, 0, 1);
	            break;
	        case BlockFace::East:
	        case BlockFace::West:
	            uAxis = glm::ivec3(0, 1, 0);
	            vAxis = glm::ivec3(0, 0, 1);
	            break;
	        case BlockFace::South:
	        case BlockFace::North:
	        default:
	            uAxis = glm::ivec3(1, 0, 0);
	            vAxis = glm::ivec3(0, 1, 0);
	            break;
	        }
	    };

	    auto buildCornerLighting = [&](BlockFace face, const glm::ivec3& owningLocal) -> std::array<std::uint32_t, 4>
	    {
	        std::array<std::uint32_t, 4> cornerLighting{};
	        const glm::ivec3 outward = faceOffset(face);
	        glm::ivec3 sideU{0};
	        glm::ivec3 sideV{0};
	        faceSampleAxes(face, sideU, sideV);

	        for (std::size_t cornerIndex = 0; cornerIndex < cornerLighting.size(); ++cornerIndex)
	        {
	            const int uSign = kCornerUSigns[cornerIndex];
	            const int vSign = kCornerVSigns[cornerIndex];
	            const glm::ivec3 fallbackSample = owningLocal + outward;
	            const std::array<glm::ivec3, 4> lightSamples{
	                fallbackSample,
	                fallbackSample + sideU * uSign,
	                fallbackSample + sideV * vSign,
	                fallbackSample + sideU * uSign + sideV * vSign
	            };

	            int skySum = 0;
	            int blockSum = 0;
	            int validSamples = 0;
	            for (const glm::ivec3& samplePos : lightSamples)
	            {
	                const BlockId sampleLightBlock = sampleBlock(samplePos.x, samplePos.y, samplePos.z);
	                if (isOpaqueForLighting(sampleLightBlock))
	                {
	                    continue;
	                }

	                const std::uint8_t packedLight = samplePackedLight(samplePos.x, samplePos.y, samplePos.z);
	                skySum += static_cast<int>(skyLightFromPacked(packedLight));
	                blockSum += static_cast<int>(blockLightFromPacked(packedLight));
	                ++validSamples;
	            }

	            std::uint8_t averagedSky = 0;
	            std::uint8_t averagedBlock = 0;
	            if (validSamples > 0)
	            {
	                averagedSky = static_cast<std::uint8_t>((skySum + validSamples / 2) / validSamples);
	                averagedBlock = static_cast<std::uint8_t>((blockSum + validSamples / 2) / validSamples);
	            }
	            else
	            {
	                const std::uint8_t fallbackPacked =
	                    samplePackedLight(fallbackSample.x, fallbackSample.y, fallbackSample.z);
	                averagedSky = skyLightFromPacked(fallbackPacked);
	                averagedBlock = blockLightFromPacked(fallbackPacked);
	            }

	            const bool side1Solid =
	                isAoSolid(sampleBlock(owningLocal.x + sideU.x * uSign,
	                                      owningLocal.y + sideU.y * uSign,
	                                      owningLocal.z + sideU.z * uSign));
	            const bool side2Solid =
	                isAoSolid(sampleBlock(owningLocal.x + sideV.x * vSign,
	                                      owningLocal.y + sideV.y * vSign,
	                                      owningLocal.z + sideV.z * vSign));
	            const bool cornerSolid =
	                isAoSolid(sampleBlock(owningLocal.x + sideU.x * uSign + sideV.x * vSign,
	                                      owningLocal.y + sideU.y * uSign + sideV.y * vSign,
	                                      owningLocal.z + sideU.z * uSign + sideV.z * vSign));
	            const std::uint8_t aoLevel =
	                (side1Solid && side2Solid)
	                    ? static_cast<std::uint8_t>(3)
	                    : static_cast<std::uint8_t>(static_cast<int>(side1Solid) +
	                                                static_cast<int>(side2Solid) +
	                                                static_cast<int>(cornerSolid));

	            cornerLighting[cornerIndex] =
	                packVertexLighting(packLightLevels(averagedSky, averagedBlock), aoLevel);
	        }

	        return cornerLighting;
	    };

	    auto makeMaterial = [&](BlockId block, const glm::vec3& normal, const glm::ivec3& owningLocal) -> FaceMaterial
	    {
	        FaceMaterial material{};
	        material.mergeable = !isAlphaCutoutBlock(block);
	        const BlockFace face = faceFromNormal(normal);

	        material.face = face;
            material.lightingData = buildCornerLighting(face, owningLocal);

	        if (blockAtlasConfigured_)
	        {
	            const BlockUVSet& uvSet = blockUVTable_[toIndex(block)];
	            const FaceUV& faceUV = uvSet.faces[toIndex(face)];
            material.uvBase = faceUV.base;
            material.uvSize = faceUV.size;
        }
        else
        {
            material.uvBase = glm::vec2(0.0f);
            material.uvSize = glm::vec2(1.0f);
        }

        switch (face)
        {
        case BlockFace::Top:
            material.uAxis = glm::ivec3(1, 0, 0);
            material.vAxis = glm::ivec3(0, 0, 1);
            break;
        case BlockFace::Bottom:
            material.uAxis = glm::ivec3(1, 0, 0);
            material.vAxis = glm::ivec3(0, 0, -1);
            break;
        case BlockFace::East:
            material.uAxis = glm::ivec3(0, 0, 1);
            material.vAxis = glm::ivec3(0, 1, 0);
            break;
        case BlockFace::West:
            material.uAxis = glm::ivec3(0, 0, -1);
            material.vAxis = glm::ivec3(0, 1, 0);
            break;
        case BlockFace::South:
            material.uAxis = glm::ivec3(-1, 0, 0);
            material.vAxis = glm::ivec3(0, 1, 0);
            break;
        case BlockFace::North:
        default:
            material.uAxis = glm::ivec3(1, 0, 0);
            material.vAxis = glm::ivec3(0, 1, 0);
            break;
        }

        return material;
    };

    auto emitQuad = [&](Axis axis, FaceDir dir, int slice, int bStart, int cStart, int bSize, int cSize, const FaceMaterial& material)
    {
        const int a = static_cast<int>(axis);
        const int b = (a + 1) % 3;
        const int c = (a + 2) % 3;

        glm::vec3 normal = axisNormals[a];
        if (dir == FaceDir::Negative)
        {
            normal = -normal;
        }

        glm::vec3 base(0.0f);
        base[a] = static_cast<float>(slice);
        base[b] = static_cast<float>(bStart);
        base[c] = static_cast<float>(cStart);

        glm::vec3 du(0.0f);
        du[b] = static_cast<float>(bSize);

        glm::vec3 dv(0.0f);
        dv[c] = static_cast<float>(cSize);

        std::array<glm::vec3, 4> positions{
            chunkOrigin + base,
            chunkOrigin + base + du,
            chunkOrigin + base + du + dv,
            chunkOrigin + base + dv
        };

            std::array<std::uint32_t, 4> cornerLighting = material.lightingData;
	        if (dir == FaceDir::Negative)
	        {
	            std::swap(positions[1], positions[3]);
                std::swap(cornerLighting[1], cornerLighting[3]);
	        }

            const int diagonal02 =
                lightingMetricFromPackedVertex(cornerLighting[0]) +
                lightingMetricFromPackedVertex(cornerLighting[2]);
            const int diagonal13 =
                lightingMetricFromPackedVertex(cornerLighting[1]) +
                lightingMetricFromPackedVertex(cornerLighting[3]);
            const bool flipDiagonal = diagonal13 > diagonal02;

	        const glm::vec3 uAxisVec = glm::vec3(material.uAxis);
	        const glm::vec3 vAxisVec = glm::vec3(material.vAxis);

	        const std::size_t vertexStart = meshData.vertices.size();
        for (int i = 0; i < 4; ++i)
        {
            const glm::vec3& pos = positions[i];

            Vertex vertex{};
            vertex.position = pos;
	            vertex.normal = normal;
	            vertex.tileCoord = glm::vec2(glm::dot(pos, uAxisVec), glm::dot(pos, vAxisVec));
	            vertex.atlasBase = material.uvBase;
	            vertex.atlasSize = material.uvSize;
	            vertex.lightingData = cornerLighting[i];
	            meshData.vertices.push_back(vertex);
	        }

            if (flipDiagonal)
            {
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 0));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 1));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 3));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 1));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 2));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 3));
            }
            else
            {
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 0));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 1));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 2));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 2));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 3));
	            meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 0));
            }
	    };

    auto greedyMeshAxis = [&](Axis axis)
    {
        const int dims[3] = {kChunkSizeX, kChunkSizeY, kChunkSizeZ};
        const int a = static_cast<int>(axis);
        const int b = (a + 1) % 3;
        const int c = (a + 2) % 3;

        const int sizeA = dims[a];
        const int sizeB = dims[b];
        const int sizeC = dims[c];

        std::vector<MaskCell> mask(static_cast<std::size_t>(sizeB * sizeC));

        auto maskIndex = [&](int bi, int ci) -> int
        {
            return bi * sizeC + ci;
        };

        for (int dirIndex = 0; dirIndex < 2; ++dirIndex)
        {
            const FaceDir dir = static_cast<FaceDir>(dirIndex);

            for (int slice = 0; slice <= sizeA; ++slice)
            {
                std::fill(mask.begin(), mask.end(), MaskCell{});

                for (int bi = 0; bi < sizeB; ++bi)
                {
                    for (int ci = 0; ci < sizeC; ++ci)
                    {
                        const int maskIdx = maskIndex(bi, ci);
                        MaskCell cell{};

                        const glm::ivec3 positiveLocal{
                            (a == 0) ? slice : ((b == 0) ? bi : ci),
                            (a == 1) ? slice : ((b == 1) ? bi : ci),
                            (a == 2) ? slice : ((b == 2) ? bi : ci)
                        };

                        const glm::ivec3 negativeLocal{
                            (a == 0) ? slice - 1 : ((b == 0) ? bi : ci),
                            (a == 1) ? slice - 1 : ((b == 1) ? bi : ci),
                            (a == 2) ? slice - 1 : ((b == 2) ? bi : ci)
                        };

                        const BlockId positiveBlock = sampleBlock(positiveLocal.x, positiveLocal.y, positiveLocal.z);
                        const BlockId negativeBlock = sampleBlock(negativeLocal.x, negativeLocal.y, negativeLocal.z);

                        glm::ivec3 owningLocal{0};
                        bool createFace = false;

                        if (dir == FaceDir::Positive)
                        {
                            if (isInsideChunk(negativeLocal) && shouldRenderBlockFace(negativeBlock, positiveBlock))
                            {
                                owningLocal = negativeLocal;
                                createFace = true;
                            }
                        }
                        else
                        {
                            if (isInsideChunk(positiveLocal) && shouldRenderBlockFace(positiveBlock, negativeBlock))
                            {
                                owningLocal = positiveLocal;
                                createFace = true;
                            }
                        }

	                        if (createFace)
	                        {
	                            const glm::vec3 normal = axisNormals[a] * ((dir == FaceDir::Positive) ? 1.0f : -1.0f);
	                            cell.exists = true;
	                            const std::size_t blockIdx = blockIndex(owningLocal.x, owningLocal.y, owningLocal.z);
	                            const BlockId owningBlock = chunkBlocks[blockIdx];
	                            cell.material = makeMaterial(
	                                owningBlock,
	                                normal,
	                                owningLocal);
	                        }

                        mask[maskIdx] = cell;
                    }
                }

                for (int bi = 0; bi < sizeB; ++bi)
                {
                    int ci = 0;
                    while (ci < sizeC)
                    {
                        const int maskIdx = maskIndex(bi, ci);
                        const MaskCell& cell = mask[maskIdx];
                        if (!cell.exists)
                        {
                            ++ci;
                            continue;
                        }

                        const FaceMaterial material = cell.material;

                        int runLengthC = 1;
                        while (material.mergeable && ci + runLengthC < sizeC)
                        {
                            const MaskCell& nextCell = mask[maskIndex(bi, ci + runLengthC)];
                            if (!nextCell.exists || !(nextCell.material == material))
                            {
                                break;
                            }
                            ++runLengthC;
                        }

                        int runHeightB = 1;
                        while (material.mergeable && bi + runHeightB < sizeB)
                        {
                            bool rowMatches = true;
                            for (int offset = 0; offset < runLengthC; ++offset)
                            {
                                const MaskCell& rowCell = mask[maskIndex(bi + runHeightB, ci + offset)];
                                if (!rowCell.exists || !(rowCell.material == material))
                                {
                                    rowMatches = false;
                                    break;
                                }
                            }

                            if (!rowMatches)
                            {
                                break;
                            }

                            ++runHeightB;
                        }

                        emitQuad(axis, dir, slice, bi, ci, runHeightB, runLengthC, material);

                        for (int bOffset = 0; bOffset < runHeightB; ++bOffset)
                        {
                            for (int cOffset = 0; cOffset < runLengthC; ++cOffset)
                            {
                                mask[maskIndex(bi + bOffset, ci + cOffset)].exists = false;
                            }
                        }

                        ci += runLengthC;
                    }
                }
            }
        }
    };

    greedyMeshAxis(Axis::X);
    greedyMeshAxis(Axis::Y);
    greedyMeshAxis(Axis::Z);

    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        chunk.meshData = std::move(meshData);
    }
    chunk.meshReady.store(true, std::memory_order_release);
}

glm::ivec3 ChunkManager::Impl::worldToChunkCoords(int worldX, int worldY, int worldZ) noexcept
{
    return {floorDiv(worldX, kChunkSizeX), floorDiv(worldY, kChunkSizeY), floorDiv(worldZ, kChunkSizeZ)};
}

std::shared_ptr<Chunk> ChunkManager::Impl::acquireChunk(const glm::ivec3& coord)
{
    std::shared_ptr<Chunk> chunk;
    {
        std::lock_guard<std::mutex> lock(chunkPoolMutex_);
        if (!chunkPool_.empty())
        {
            chunk = std::move(chunkPool_.back());
            chunkPool_.pop_back();
        }
    }

    if (!chunk)
    {
        chunk = std::make_shared<Chunk>(coord);
    }

    chunk->reset(coord);
    return chunk;

}

std::shared_ptr<Chunk> ChunkManager::Impl::getChunkShared(const glm::ivec3& coord) noexcept
{
    std::lock_guard<std::mutex> lock(chunksMutex);
    auto it = chunks_.find(coord);
    return (it != chunks_.end()) ? it->second : nullptr;
}

std::shared_ptr<const Chunk> ChunkManager::Impl::getChunkShared(const glm::ivec3& coord) const noexcept
{
    std::lock_guard<std::mutex> lock(chunksMutex);
    auto it = chunks_.find(coord);
    if (it != chunks_.end())
    {
        return it->second;
    }
    return nullptr;
}

Chunk* ChunkManager::Impl::getChunk(const glm::ivec3& coord) noexcept
{
    return getChunkShared(coord).get();
}

const Chunk* ChunkManager::Impl::getChunk(const glm::ivec3& coord) const noexcept
{
    return getChunkShared(coord).get();
}

void ChunkManager::Impl::markNeighborsForRemeshingIfNeeded(const glm::ivec3& coord, int localX, int localY, int localZ)
{
    auto queueNeighbor = [&](const glm::ivec3& neighborCoord)
    {
        auto neighbor = getChunkShared(neighborCoord);
        if (!neighbor)
        {
            return;
        }

        requestChunkRemesh(neighbor);
    };

    if (localX == 0)
    {
        queueNeighbor(coord + glm::ivec3{-1, 0, 0});
    }

    if (localX == kChunkSizeX - 1)
    {
        queueNeighbor(coord + glm::ivec3{1, 0, 0});
    }

    if (localZ == 0)
    {
        queueNeighbor(coord + glm::ivec3{0, 0, -1});
    }

    if (localZ == kChunkSizeZ - 1)
    {
        queueNeighbor(coord + glm::ivec3{0, 0, 1});

    }

    if (localY == 0)
    {
        queueNeighbor(coord + glm::ivec3{0, -1, 0});
    }

    if (localY == kChunkSizeY - 1)
    {
        queueNeighbor(coord + glm::ivec3{0, 1, 0});
    }
}

void ChunkManager::Impl::requestChunkRemesh(const std::shared_ptr<Chunk>& chunk)
{
    if (!chunk)
    {
        return;
    }

    if (!chunk->hasBlocks.load(std::memory_order_acquire) &&
        chunk->indexCount.load(std::memory_order_acquire) == 0)
    {
        return;
    }

    const ChunkState state = chunk->state.load(std::memory_order_acquire);
    if (state == ChunkState::Generating || state == ChunkState::Meshing)
    {
        chunk->pendingMeshRefresh.store(true, std::memory_order_release);
        return;
    }

    if (state == ChunkState::Remeshing)
    {
        if (chunk->inFlight.load(std::memory_order_acquire) > 0)
        {
            chunk->pendingMeshRefresh.store(true, std::memory_order_release);
            return;
        }
    }

    if (state == ChunkState::Uploaded || state == ChunkState::Ready || state == ChunkState::Remeshing)
    {
        chunk->state.store(ChunkState::Remeshing, std::memory_order_release);
        enqueueJob(chunk, JobType::Mesh, chunk->coord);
    }
}

void ChunkManager::Impl::queueChunkForLightingRemesh(const std::shared_ptr<Chunk>& chunk)
{
    requestChunkRemesh(chunk);
}

std::uint8_t ChunkManager::Impl::packedLightAtWorld(const glm::ivec3& worldPos) const noexcept
{
    if (worldPos.y < 0)
    {
        return packLightLevels(0, 0);
    }

    const glm::ivec3 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
    auto chunk = getChunkShared(chunkCoord);
    if (!chunk)
    {
        return packLightLevels(kMaxLightLevel, 0);
    }

    if (worldPos.y < chunk->minWorldY || worldPos.y > chunk->maxWorldY)
    {
        return packLightLevels(kMaxLightLevel, 0);
    }

    const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
    if (local.x < 0 || local.x >= kChunkSizeX ||
        local.z < 0 || local.z >= kChunkSizeZ)
    {
        return packLightLevels(kMaxLightLevel, 0);
    }

    const int localY = worldPos.y - chunk->minWorldY;
    return chunk->lightLevels[blockIndex(local.x, localY, local.z)];
}

void ChunkManager::Impl::relightAroundChunk(const glm::ivec3& centerCoord)
{
    std::vector<std::shared_ptr<Chunk>> regionChunks;
    regionChunks.reserve(27);
    for (int dx = -1; dx <= 1; ++dx)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dz = -1; dz <= 1; ++dz)
            {
                auto chunk = getChunkShared(centerCoord + glm::ivec3(dx, dy, dz));
                if (chunk)
                {
                    regionChunks.push_back(std::move(chunk));
                }
            }
        }
    }

    if (regionChunks.empty())
    {
        return;
    }

    std::sort(regionChunks.begin(),
              regionChunks.end(),
              [](const std::shared_ptr<Chunk>& lhs, const std::shared_ptr<Chunk>& rhs)
              {
                  if (lhs->coord.x != rhs->coord.x)
                  {
                      return lhs->coord.x < rhs->coord.x;
                  }
                  if (lhs->coord.y != rhs->coord.y)
                  {
                      return lhs->coord.y < rhs->coord.y;
                  }
                  return lhs->coord.z < rhs->coord.z;
              });
    regionChunks.erase(std::unique(regionChunks.begin(), regionChunks.end()), regionChunks.end());

    std::unordered_map<glm::ivec3, std::shared_ptr<Chunk>, ChunkHasher> regionLookup;
    regionLookup.reserve(regionChunks.size());
    for (const auto& chunk : regionChunks)
    {
        regionLookup.emplace(chunk->coord, chunk);
    }

    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(regionChunks.size());
    for (auto& chunk : regionChunks)
    {
        locks.emplace_back(chunk->meshMutex);
    }

    std::vector<std::vector<std::uint8_t>> previousLights;
    previousLights.reserve(regionChunks.size());
    for (auto& chunk : regionChunks)
    {
        previousLights.push_back(chunk->lightLevels);
        std::fill(chunk->lightLevels.begin(), chunk->lightLevels.end(), packLightLevels(0, 0));
        chunk->lightBoundaryDirtyMask = 0;
    }

    struct LightNode
    {
        glm::ivec3 worldPos{0};
        std::uint8_t level{0};
    };

    std::deque<LightNode> skyQueue;
    std::deque<LightNode> blockQueue;

    auto accessRegionVoxel = [&](const glm::ivec3& worldPos) -> std::pair<Chunk*, std::size_t>
    {
        const glm::ivec3 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.y, worldPos.z);
        auto it = regionLookup.find(chunkCoord);
        if (it == regionLookup.end())
        {
            return {nullptr, 0};
        }

        const glm::ivec3 local = localBlockCoords(worldPos, chunkCoord);
        if (local.x < 0 || local.x >= kChunkSizeX ||
            local.z < 0 || local.z >= kChunkSizeZ ||
            worldPos.y < it->second->minWorldY ||
            worldPos.y > it->second->maxWorldY)
        {
            return {nullptr, 0};
        }

        const int localY = worldPos.y - it->second->minWorldY;
        return {it->second.get(), blockIndex(local.x, localY, local.z)};
    };

    auto seedSkyLight = [&](const glm::ivec3& worldPos, std::uint8_t level)
    {
        auto [chunk, idx] = accessRegionVoxel(worldPos);
        if (!chunk || isOpaqueForLighting(chunk->blocks[idx]))
        {
            return;
        }

        if (level > skyLightFromPacked(chunk->lightLevels[idx]))
        {
            setSkyLight(chunk->lightLevels[idx], level);
            if (level > 1)
            {
                skyQueue.push_back({worldPos, level});
            }
        }
    };

    auto seedBlockLight = [&](const glm::ivec3& worldPos, std::uint8_t level)
    {
        auto [chunk, idx] = accessRegionVoxel(worldPos);
        if (!chunk)
        {
            return;
        }

        if (level > blockLightFromPacked(chunk->lightLevels[idx]))
        {
            setBlockLight(chunk->lightLevels[idx], level);
            if (level > 1)
            {
                blockQueue.push_back({worldPos, level});
            }
        }
    };

    auto computeSkyLightFromAbove = [&](int worldX, int worldY, int worldZ) -> std::uint8_t
    {
        if (worldY < 0)
        {
            return 0;
        }

        int accumulatedAttenuation = 0;
        int scanWorldY = worldY;

        while (scanWorldY >= 0)
        {
            const glm::ivec3 scanPos(worldX, scanWorldY, worldZ);
            const glm::ivec3 chunkCoord = worldToChunkCoords(scanPos.x, scanPos.y, scanPos.z);

            std::shared_ptr<Chunk> chunk;
            auto regionIt = regionLookup.find(chunkCoord);
            if (regionIt != regionLookup.end())
            {
                chunk = regionIt->second;
            }
            else
            {
                chunk = getChunkShared(chunkCoord);
            }

            if (!chunk)
            {
                break;
            }

            const glm::ivec3 local = localBlockCoords(scanPos, chunkCoord);
            const int localX = local.x;
            const int localZ = local.z;
            if (localX < 0 || localX >= kChunkSizeX || localZ < 0 || localZ >= kChunkSizeZ)
            {
                break;
            }

            int localY = scanWorldY - chunk->minWorldY;
            for (; localY < kChunkSizeY; ++localY)
            {
                const BlockId block = chunk->blocks[blockIndex(localX, localY, localZ)];
                if (isOpaqueForLighting(block))
                {
                    return 0;
                }

                accumulatedAttenuation += static_cast<int>(blockLightingProperties(block).skyAttenuation);
                if (accumulatedAttenuation >= static_cast<int>(kMaxLightLevel))
                {
                    return 0;
                }
            }

            scanWorldY = chunk->maxWorldY + 1;
        }

        return static_cast<std::uint8_t>(
            std::max(0, static_cast<int>(kMaxLightLevel) - accumulatedAttenuation));
    };

    std::vector<std::shared_ptr<Chunk>> verticalOrder = regionChunks;
    std::sort(verticalOrder.begin(),
              verticalOrder.end(),
              [](const std::shared_ptr<Chunk>& lhs, const std::shared_ptr<Chunk>& rhs)
              {
                  if (lhs->coord.y != rhs->coord.y)
                  {
                      return lhs->coord.y > rhs->coord.y;
                  }
                  if (lhs->coord.x != rhs->coord.x)
                  {
                      return lhs->coord.x < rhs->coord.x;
                  }
                  return lhs->coord.z < rhs->coord.z;
              });

    for (const auto& chunk : verticalOrder)
    {
        const int baseWorldX = chunk->coord.x * kChunkSizeX;
        const int baseWorldZ = chunk->coord.z * kChunkSizeZ;

        for (int localX = 0; localX < kChunkSizeX; ++localX)
        {
            for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
            {
                const int worldX = baseWorldX + localX;
                const int worldZ = baseWorldZ + localZ;
                std::uint8_t incomingSky = computeSkyLightFromAbove(worldX, chunk->maxWorldY + 1, worldZ);

                for (int localY = kChunkSizeY - 1; localY >= 0; --localY)
                {
                    const std::size_t idx = blockIndex(localX, localY, localZ);
                    const BlockId block = chunk->blocks[idx];
                    const glm::ivec3 worldPos(worldX, chunk->minWorldY + localY, worldZ);

                    if (isOpaqueForLighting(block))
                    {
                        setSkyLight(chunk->lightLevels[idx], 0);
                        incomingSky = 0;
                    }
                    else
                    {
                        const std::uint8_t attenuation = blockLightingProperties(block).skyAttenuation;
                        incomingSky = static_cast<std::uint8_t>(
                            std::max(0, static_cast<int>(incomingSky) - static_cast<int>(attenuation)));
                        setSkyLight(chunk->lightLevels[idx], incomingSky);
                        if (incomingSky > 0)
                        {
                            skyQueue.push_back({worldPos, incomingSky});
                        }
                    }

                    const std::uint8_t emission = blockLightingProperties(block).blockEmission;
                    if (emission > 0)
                    {
                        setBlockLight(chunk->lightLevels[idx], emission);
                        blockQueue.push_back({worldPos, emission});
                    }
                }
            }
        }
    }

    for (auto& chunk : regionChunks)
    {
        for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
        {
            const glm::ivec3 neighborCoord = chunk->coord + faceOffset(face);
            if (regionLookup.find(neighborCoord) != regionLookup.end())
            {
                continue;
            }

            auto outsideChunk = getChunkShared(neighborCoord);
            if (!outsideChunk)
            {
                chunk->lightBoundaryDirtyMask |= static_cast<std::uint8_t>(1u << toIndex(face));
                continue;
            }

            const glm::ivec3 offset = faceOffset(face);
            for (int localX = 0; localX < kChunkSizeX; ++localX)
            {
                for (int localY = 0; localY < kChunkSizeY; ++localY)
                {
                    for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
                    {
                        if ((offset.x < 0 && localX != 0) ||
                            (offset.x > 0 && localX != kChunkSizeX - 1) ||
                            (offset.y < 0 && localY != 0) ||
                            (offset.y > 0 && localY != kChunkSizeY - 1) ||
                            (offset.z < 0 && localZ != 0) ||
                            (offset.z > 0 && localZ != kChunkSizeZ - 1))
                        {
                            continue;
                        }

                        const std::size_t idx = blockIndex(localX, localY, localZ);
                        const BlockId block = chunk->blocks[idx];
                        if (isOpaqueForLighting(block))
                        {
                            continue;
                        }

                        const glm::ivec3 worldPos(chunk->coord.x * kChunkSizeX + localX,
                                                  chunk->minWorldY + localY,
                                                  chunk->coord.z * kChunkSizeZ + localZ);
                        const std::uint8_t neighborPacked = packedLightAtWorld(worldPos + offset);
                        const std::uint8_t loss = propagationLossFor(block);
                        const std::uint8_t skySeed =
                            (skyLightFromPacked(neighborPacked) > loss)
                                ? static_cast<std::uint8_t>(skyLightFromPacked(neighborPacked) - loss)
                                : 0;
                        const std::uint8_t blockSeed =
                            (blockLightFromPacked(neighborPacked) > loss)
                                ? static_cast<std::uint8_t>(blockLightFromPacked(neighborPacked) - loss)
                                : 0;
                        if (skySeed > 0)
                        {
                            seedSkyLight(worldPos, skySeed);
                        }
                        if (blockSeed > 0)
                        {
                            seedBlockLight(worldPos, blockSeed);
                        }
                    }
                }
            }
        }
    }

    auto propagateLight = [&](std::deque<LightNode>& queue, bool skyChannel)
    {
        while (!queue.empty())
        {
            const LightNode node = queue.front();
            queue.pop_front();

            auto [sourceChunk, sourceIdx] = accessRegionVoxel(node.worldPos);
            if (!sourceChunk)
            {
                continue;
            }

            const std::uint8_t currentLevel =
                skyChannel ? skyLightFromPacked(sourceChunk->lightLevels[sourceIdx])
                           : blockLightFromPacked(sourceChunk->lightLevels[sourceIdx]);
            if (currentLevel != node.level || currentLevel == 0)
            {
                continue;
            }

            for (BlockFace face : {BlockFace::Top, BlockFace::Bottom, BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
            {
                const glm::ivec3 neighborPos = node.worldPos + faceOffset(face);
                auto [targetChunk, targetIdx] = accessRegionVoxel(neighborPos);
                if (!targetChunk)
                {
                    continue;
                }

                const BlockId targetBlock = targetChunk->blocks[targetIdx];
                if (isOpaqueForLighting(targetBlock))
                {
                    continue;
                }

                const std::uint8_t loss = propagationLossFor(targetBlock);
                if (currentLevel <= loss)
                {
                    continue;
                }

                const std::uint8_t nextLevel = static_cast<std::uint8_t>(currentLevel - loss);
                const std::uint8_t existingLevel =
                    skyChannel ? skyLightFromPacked(targetChunk->lightLevels[targetIdx])
                               : blockLightFromPacked(targetChunk->lightLevels[targetIdx]);
                if (nextLevel <= existingLevel)
                {
                    continue;
                }

                if (skyChannel)
                {
                    setSkyLight(targetChunk->lightLevels[targetIdx], nextLevel);
                }
                else
                {
                    setBlockLight(targetChunk->lightLevels[targetIdx], nextLevel);
                }

                if (nextLevel > 1)
                {
                    queue.push_back({neighborPos, nextLevel});
                }
            }
        }
    };

    propagateLight(skyQueue, true);
    propagateLight(blockQueue, false);

    std::vector<std::shared_ptr<Chunk>> changedChunks;
    changedChunks.reserve(regionChunks.size());
    for (std::size_t i = 0; i < regionChunks.size(); ++i)
    {
        if (regionChunks[i]->lightLevels != previousLights[i])
        {
            changedChunks.push_back(regionChunks[i]);
        }
    }

    locks.clear();

    for (const auto& chunk : changedChunks)
    {
        queueChunkForLightingRemesh(chunk);
    }
}







ColumnSample ChunkManager::Impl::sampleColumn(int worldX, int worldZ, int slabMinWorldY, int slabMaxWorldY) const
{
    const bool usesDefaultSlabBounds =
        slabMinWorldY == std::numeric_limits<int>::min() && slabMaxWorldY == std::numeric_limits<int>::max();
    if (slabMinWorldY > slabMaxWorldY)
    {
        std::swap(slabMinWorldY, slabMaxWorldY);
    }

    if (!surfaceMap_)
    {
        throw std::runtime_error("Surface map is not initialized");
    }
    if (!climateMap_)
    {
        throw std::runtime_error("Climate map is not initialized");
    }

    ColumnSample sample{};
    const terrain::SurfaceColumn& surfaceColumn = surfaceMap_->column(worldX, worldZ);
    const terrain::ClimateSample& climateSample = climateMap_->sample(worldX, worldZ);

    sample.dominantBiome = surfaceColumn.dominantBiome;
    sample.dominantWeight = surfaceColumn.dominantWeight;
    sample.surfaceHeight = surfaceColumn.surfaceHeight;
    sample.surfaceY = surfaceColumn.surfaceY;
    sample.originalSurfaceY = surfaceColumn.surfaceY;
    if (usesDefaultSlabBounds)
    {
        slabMinWorldY = sample.surfaceY;
        slabMaxWorldY = sample.surfaceY;
    }
    sample.minSurfaceY = std::min(sample.surfaceY, slabMinWorldY);
    sample.maxSurfaceY = std::max(sample.surfaceY, slabMaxWorldY);
    sample.soilCreepCoefficient = surfaceColumn.soilCreepCoefficient;
    sample.roughAmplitude = surfaceColumn.roughAmplitude;
    sample.hillAmplitude = surfaceColumn.hillAmplitude;
    sample.mountainAmplitude = surfaceColumn.mountainAmplitude;
    sample.dominantIsOcean = climateSample.dominantIsOcean;
    sample.distanceToCoast = climateSample.distanceToCoast;
    sample.distanceToShore = std::isfinite(climateSample.distanceToCoast)
                                 ? climateSample.distanceToCoast
                                 : std::numeric_limits<float>::infinity();
    sample.soilCreepOffset = 0.0f;

    sample.topBlendCount = std::min(climateSample.blendCount, sample.topBlendDebug.size());
    const glm::vec2 columnPos(static_cast<float>(worldX), static_cast<float>(worldZ));
    for (std::size_t i = 0; i < sample.topBlendCount; ++i)
    {
        const auto& srcBlend = climateSample.blends[i];
        auto& dstBlend = sample.topBlendDebug[i];
        dstBlend.biome = srcBlend.biome;
        dstBlend.weight = srcBlend.weight;
        dstBlend.aggregatedHeight = srcBlend.height;
        dstBlend.normalizedDistance = srcBlend.normalizedDistance;
        dstBlend.seedRadius = srcBlend.falloff;
        dstBlend.worldDistance = glm::length(columnPos - srcBlend.sitePosition);
        dstBlend.isOcean = srcBlend.biome && srcBlend.biome->isOcean();
    }

    sample.slabHasSolid = surfaceColumn.surfaceY >= slabMinWorldY;
    if (sample.slabHasSolid)
    {
        sample.slabHighestSolidY = std::min(surfaceColumn.surfaceY, slabMaxWorldY);
    }

    if (!std::isfinite(sample.distanceToShore))
    {
        if (sample.dominantBiome && sample.dominantBiome->isOcean())
        {
            sample.distanceToShore = 0.0f;
            sample.distanceToCoast = 0.0f;
        }
    }

    return sample;
}



void ChunkManager::Impl::generateSurfaceOnlyChunk(Chunk& chunk)
{
    std::lock_guard<std::mutex> lock(chunk.meshMutex);
    std::fill(chunk.blocks.begin(), chunk.blocks.end(), BlockId::Air);

    if (!chunk.lodData)
    {
        chunk.lodData = std::make_unique<FarChunk>();
    }

    FarChunk& lod = *chunk.lodData;
    lod.origin = glm::vec3(static_cast<float>(chunk.coord.x * kChunkSizeX),
                           static_cast<float>(chunk.minWorldY),
                           static_cast<float>(chunk.coord.z * kChunkSizeZ));
    lod.size = glm::ivec3{kChunkSizeX, kChunkSizeY, kChunkSizeZ};
    lod.lodStep = FarChunk::kColumnStep;
    lod.thickness = 1;

    const int baseWorldX = chunk.coord.x * kChunkSizeX;
    const int baseWorldZ = chunk.coord.z * kChunkSizeZ;
    const int slabMinWorldY = chunk.minWorldY;
    const int slabMaxWorldY = chunk.maxWorldY;

    bool anySolid = false;

    for (int rx = 0; rx < FarChunk::kColumnsX; ++rx)
    {
        for (int rz = 0; rz < FarChunk::kColumnsZ; ++rz)
        {
            int bestWorldY = std::numeric_limits<int>::min();
            BlockId bestBlock = BlockId::Air;
            int bestLocalX = -1;
            int bestLocalZ = -1;

            const auto considerTopBlock = [&](int candidateWorldY, BlockId candidateBlock, int localX, int localZ)
            {
                if (candidateBlock == BlockId::Air)
                {
                    return;
                }
                if (candidateWorldY < slabMinWorldY || candidateWorldY > slabMaxWorldY)
                {
                    return;
                }
                if (candidateWorldY > bestWorldY)
                {
                    bestWorldY = candidateWorldY;
                    bestBlock = candidateBlock;
                    bestLocalX = localX;
                    bestLocalZ = localZ;
                }
            };

            for (int localX = rx * FarChunk::kColumnStep;
                 localX < (rx + 1) * FarChunk::kColumnStep && localX < kChunkSizeX;
                 ++localX)
            {
                for (int localZ = rz * FarChunk::kColumnStep;
                     localZ < (rz + 1) * FarChunk::kColumnStep && localZ < kChunkSizeZ;
                     ++localZ)
                {
                    const int worldX = baseWorldX + localX;
                    const int worldZ = baseWorldZ + localZ;

                    ColumnSample sample = sampleColumn(worldX, worldZ, slabMinWorldY, slabMaxWorldY);
                    if (!sample.dominantBiome)
                    {
                        continue;
                    }

                    const BiomeDefinition& biome = *sample.dominantBiome;
                    if (sample.slabHasSolid)
                    {
                        const terrain::TerrainColumnBlocks resolvedBlocks =
                            terrain::resolveTerrainColumnBlocks(biome, sample, worldX, worldZ, globalSeaLevel_);
                        considerTopBlock(sample.slabHighestSolidY, resolvedBlocks.surfaceBlock, localX, localZ);
                    }

                    const auto& waterFill = biome.terrainSettings.waterFill;
                    if (waterFill.enabled && sample.surfaceY < globalSeaLevel_)
                    {
                        const int waterTopWorld = std::min(globalSeaLevel_, slabMaxWorldY);
                        int waterBottomWorld = std::max(sample.surfaceY + 1, slabMinWorldY);
                        if (waterFill.maxDepth > 0)
                        {
                            waterBottomWorld = std::max(waterBottomWorld, waterTopWorld - waterFill.maxDepth + 1);
                        }
                        if (waterBottomWorld <= waterTopWorld)
                        {
                            considerTopBlock(waterTopWorld, waterFill.block, localX, localZ);
                        }
                    }
                }
            }

            FarChunk::SurfaceCell cell{};

            if (bestLocalX >= 0 && bestBlock != BlockId::Air)
            {
                cell.worldY = bestWorldY;
                cell.block = bestBlock;

                const int localY = bestWorldY - chunk.minWorldY;
                if (localY >= 0 && localY < kChunkSizeY)
                {
                    chunk.blocks[blockIndex(bestLocalX, localY, bestLocalZ)] = bestBlock;
                    anySolid = true;
                }
            }

            lod.strata[FarChunk::index(rx, rz)] = cell;
        }
    }

    chunk.hasBlocks.store(anySolid, std::memory_order_release);
    if (anySolid)
    {
        columnManager_.updateChunk(chunk);
    }
    else
    {
        columnManager_.removeChunk(chunk);
    }
    invalidatePredictedColumn({chunk.coord.x, chunk.coord.z});
}

void ChunkManager::Impl::generateChunkBlocks(Chunk& chunk)
{
    std::vector<PendingStructureEdit> externalEdits;
    bool anySolid = false;

    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        std::fill(chunk.blocks.begin(), chunk.blocks.end(), BlockId::Air);

        const int baseWorldX = chunk.coord.x * kChunkSizeX;
        const int baseWorldZ = chunk.coord.z * kChunkSizeZ;

        if (surfaceMap_)
        {
            const int fragmentSize = terrain::SurfaceFragment::kSize;
            const int minFragmentX = floorDiv(baseWorldX - 1, fragmentSize);
            const int maxFragmentX = floorDiv(baseWorldX + kChunkSizeX, fragmentSize);
            const int minFragmentZ = floorDiv(baseWorldZ - 1, fragmentSize);
            const int maxFragmentZ = floorDiv(baseWorldZ + kChunkSizeZ, fragmentSize);

            for (int fx = minFragmentX; fx <= maxFragmentX; ++fx)
            {
                for (int fz = minFragmentZ; fz <= maxFragmentZ; ++fz)
                {
                    const auto& prefetchedFragment = surfaceMap_->getFragment({fx, fz});
                    (void)prefetchedFragment;
                }
            }
        }

        std::array<ColumnBuildResult, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)> columnResults{};

        auto setBlockDirect = [&](int localX, int localY, int localZ, BlockId block)
        {
            if (localX < 0 || localX >= kChunkSizeX || localZ < 0 || localZ >= kChunkSizeZ)
            {
                return;
            }
            if (localY < 0 || localY >= kChunkSizeY)
            {
                return;
            }
            chunk.blocks[blockIndex(localX, localY, localZ)] = block;
            if (block != BlockId::Air)
            {
                anySolid = true;
            }
        };

        terrain::ChunkGenerationSummary summary{};
        if (terrainGenerator_)
        {
            summary = terrainGenerator_->generateChunkColumns(chunk.coord,
                                                              chunk.minWorldY,
                                                              chunk.maxWorldY,
                                                              kChunkSizeX,
                                                              kChunkSizeY,
                                                              kChunkSizeZ,
                                                              setBlockDirect,
                                                              columnResults);
            anySolid = anySolid || summary.anySolid;
        }

        const bool slabContainsTerrain = summary.slabContainsTerrain;

        auto setOrQueueBlock = [&](int worldX, int worldY, int worldZ, BlockId block, bool replaceSolid)
        {
            const glm::ivec3 worldPos{worldX, worldY, worldZ};
            const glm::ivec3 targetChunk = worldToChunkCoords(worldX, worldY, worldZ);
            if (targetChunk == chunk.coord)
            {
                if (worldY < chunk.minWorldY || worldY > chunk.maxWorldY)
                {
                    return;
                }

                const glm::ivec3 local = localBlockCoords(worldPos, targetChunk);
                const int localY = worldY - chunk.minWorldY;
                BlockId& destination = chunk.blocks[blockIndex(local.x, localY, local.z)];
                if (!replaceSolid && destination != BlockId::Air)
                {
                    return;
                }

                destination = block;
                if (block != BlockId::Air)
                {
                    anySolid = true;
                }
            }
            else
            {
                if (block == BlockId::Air)
                {
                    return;
                }

                externalEdits.push_back(PendingStructureEdit{targetChunk, worldPos, block, replaceSolid});
            }
        };

        auto getLocalColumnSample = [&](int worldX, int worldZ) -> ColumnSample
        {
            if (worldX >= baseWorldX && worldX < baseWorldX + kChunkSizeX && worldZ >= baseWorldZ
                && worldZ < baseWorldZ + kChunkSizeZ)
            {
                const int localX = worldX - baseWorldX;
                const int localZ = worldZ - baseWorldZ;
                return columnResults[columnIndex(localX, localZ)].sample;
            }
            return sampleColumn(worldX, worldZ);
        };

        if (slabContainsTerrain)
        {
            constexpr float kTreeBiomeWeightThreshold = 0.55f;
            constexpr int kDefaultTreeMinHeight = 6;
            constexpr int kDefaultTreeMaxHeight = 8;
            constexpr int kDefaultTreeMaxRadius = 2;

            const int treePadding = std::max(kDefaultTreeMaxRadius, kTaigaSpruceMaxLeafRadius);
            const int minWorldX = baseWorldX - treePadding;
            const int maxWorldX = baseWorldX + kChunkSizeX + treePadding - 1;
            const int minWorldZ = baseWorldZ - treePadding;
            const int maxWorldZ = baseWorldZ + kChunkSizeZ + treePadding - 1;

            auto resolvedSurfaceBlockAt = [&](int worldX, int worldZ, const ColumnSample& sample) -> BlockId
            {
                if (!sample.dominantBiome)
                {
                    return BlockId::Air;
                }

                const terrain::TerrainColumnBlocks blocks =
                    terrain::resolveTerrainColumnBlocks(*sample.dominantBiome, sample, worldX, worldZ, globalSeaLevel_);
                return blocks.surfaceBlock;
            };

            auto canAnchorTaigaSpruce = [&](int originX, int originZ, int& outGroundWorldY) -> bool
            {
                int groundWorldY = std::numeric_limits<int>::min();

                for (int trunkX = 0; trunkX < 2; ++trunkX)
                {
                    for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
                    {
                        const ColumnSample baseSample = getLocalColumnSample(originX + trunkX, originZ + trunkZ);
                        if (!baseSample.dominantBiome || !terrain::isTaigaBiome(*baseSample.dominantBiome))
                        {
                            return false;
                        }
                        if (baseSample.dominantWeight < kTreeBiomeWeightThreshold)
                        {
                            return false;
                        }

                        const BlockId surfaceBlock =
                            resolvedSurfaceBlockAt(originX + trunkX, originZ + trunkZ, baseSample);
                        if (surfaceBlock != BlockId::Grass && surfaceBlock != BlockId::Podzol)
                        {
                            return false;
                        }

                        if (groundWorldY == std::numeric_limits<int>::min())
                        {
                            groundWorldY = baseSample.surfaceY;
                        }
                        else if (baseSample.surfaceY != groundWorldY)
                        {
                            return false;
                        }
                    }
                }

                for (int dx = -2; dx <= 3; ++dx)
                {
                    for (int dz = -2; dz <= 3; ++dz)
                    {
                        const ColumnSample neighborSample = getLocalColumnSample(originX + dx, originZ + dz);
                        if (!neighborSample.dominantBiome)
                        {
                            return false;
                        }
                        if (std::abs(neighborSample.surfaceY - groundWorldY) > 1)
                        {
                            return false;
                        }
                    }
                }

                outGroundWorldY = groundWorldY;
                return groundWorldY > 2;
            };

            auto placeTaigaSpruce = [&](int originX, int originZ, int groundWorldY)
            {
                const int trunkHeight = taigaSpruceTrunkHeight(originX, groundWorldY, originZ);
                const int bareTrunkHeight = taigaSpruceBareTrunkHeight(originX, groundWorldY, originZ);

                for (int trunkX = 0; trunkX < 2; ++trunkX)
                {
                    for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
                    {
                        for (int dy = 1; dy <= trunkHeight; ++dy)
                        {
                            setOrQueueBlock(originX + trunkX,
                                            groundWorldY + dy,
                                            originZ + trunkZ,
                                            BlockId::SpruceLog,
                                            true);
                        }
                    }
                }

                const int canopyBaseWorld = groundWorldY + bareTrunkHeight + 1;
                const int canopyTopWorld = groundWorldY + trunkHeight;
                const int totalLayers = std::max(1, canopyTopWorld - canopyBaseWorld + 1);

                for (int worldY = canopyBaseWorld; worldY <= canopyTopWorld; ++worldY)
                {
                    const int layerFromBottom = worldY - canopyBaseWorld;
                    const int radius = taigaSpruceLeafRadiusForLayer(layerFromBottom, totalLayers);
                    if (radius <= 0)
                    {
                        continue;
                    }

                    for (int worldX = originX - radius; worldX <= originX + 1 + radius; ++worldX)
                    {
                        for (int worldZ = originZ - radius; worldZ <= originZ + 1 + radius; ++worldZ)
                        {
                            if (!taigaSpruceLeafOccupiesCell(originX,
                                                             originZ,
                                                             worldX,
                                                             worldZ,
                                                             radius,
                                                             layerFromBottom,
                                                             totalLayers))
                            {
                                continue;
                            }

                            setOrQueueBlock(worldX, worldY, worldZ, BlockId::SpruceLeaves, false);
                        }
                    }
                }

                const int crownWorldY = canopyTopWorld + 1;
                for (int trunkX = 0; trunkX < 2; ++trunkX)
                {
                    for (int trunkZ = 0; trunkZ < 2; ++trunkZ)
                    {
                        setOrQueueBlock(originX + trunkX,
                                        crownWorldY,
                                        originZ + trunkZ,
                                        BlockId::SpruceLeaves,
                                        false);
                    }
                }
            };

            for (int worldX = minWorldX; worldX <= maxWorldX; ++worldX)
            {
                for (int worldZ = minWorldZ; worldZ <= maxWorldZ; ++worldZ)
                {
                    const ColumnSample columnSample = getLocalColumnSample(worldX, worldZ);
                    if (!columnSample.dominantBiome)
                    {
                        continue;
                    }

                    const BiomeDefinition& biome = *columnSample.dominantBiome;
                    if (!biome.generatesTrees)
                    {
                        continue;
                    }

                    if (columnSample.dominantWeight < kTreeBiomeWeightThreshold)
                    {
                        continue;
                    }

                    const int groundWorldY = columnSample.surfaceY;
                    if (groundWorldY <= 2)
                    {
                        continue;
                    }

                    if (terrain::isTaigaBiome(biome))
                    {
                        if (!shouldSpawnTaigaSpruce(biome, worldX, groundWorldY, worldZ))
                        {
                            continue;
                        }

                        int taigaGroundWorldY = std::numeric_limits<int>::min();
                        if (!canAnchorTaigaSpruce(worldX, worldZ, taigaGroundWorldY))
                        {
                            continue;
                        }

                        placeTaigaSpruce(worldX, worldZ, taigaGroundWorldY);
                        continue;
                    }

                    const int groundLocalY = groundWorldY - chunk.minWorldY;
                    if (groundLocalY < 0 || groundLocalY >= kChunkSizeY)
                    {
                        continue;
                    }

                    const int localX = worldX - baseWorldX;
                    const int localZ = worldZ - baseWorldZ;
                    const BlockId resolvedSurfaceBlock = resolvedSurfaceBlockAt(worldX, worldZ, columnSample);
                    if (localX >= 0 && localX < kChunkSizeX && localZ >= 0 && localZ < kChunkSizeZ)
                    {
                        const std::size_t blockIdx = blockIndex(localX, groundLocalY, localZ);
                        if (chunk.blocks[blockIdx] != resolvedSurfaceBlock)
                        {
                            continue;
                        }
                    }

                    const float density = noise_.fbm(static_cast<float>(worldX) * 0.05f,
                                                     static_cast<float>(worldZ) * 0.05f,
                                                     4,
                                                     0.55f,
                                                     2.0f);
                    const float normalizedDensity = std::clamp((density + 1.0f) * 0.5f, 0.0f, 1.0f);
                    const float randomValue = hashToUnitFloat(worldX, groundWorldY, worldZ);
                    const float spawnThresholdBase = 0.015f + normalizedDensity * 0.02f;
                    const float spawnThreshold =
                        std::clamp(spawnThresholdBase * std::max(biome.treeDensityMultiplier, 0.0f), 0.0f, 1.0f);
                    if (randomValue > spawnThreshold)
                    {
                        continue;
                    }

                    bool terrainSuitable = true;
                    for (int dx = -1; dx <= 1 && terrainSuitable; ++dx)
                    {
                        for (int dz = -1; dz <= 1; ++dz)
                        {
                            if (dx == 0 && dz == 0)
                            {
                                continue;
                            }

                            const ColumnSample neighborSample = getLocalColumnSample(worldX + dx, worldZ + dz);
                            const int neighborHeight = neighborSample.surfaceY;
                            if (std::abs(neighborHeight - groundWorldY) > 1)
                            {
                                terrainSuitable = false;
                                break;
                            }
                        }
                    }

                    if (!terrainSuitable)
                    {
                        continue;
                    }

                    int trunkHeight = kDefaultTreeMinHeight +
                                      static_cast<int>(hashToUnitFloat(worldX, groundWorldY + 1, worldZ) *
                                                       static_cast<float>(kDefaultTreeMaxHeight - kDefaultTreeMinHeight + 1));
                    trunkHeight = std::clamp(trunkHeight, kDefaultTreeMinHeight, kDefaultTreeMaxHeight);

                    for (int dy = 0; dy < trunkHeight; ++dy)
                    {
                        setOrQueueBlock(worldX, groundWorldY + dy, worldZ, BlockId::Wood, true);
                    }

                    const int canopyBaseWorld = groundWorldY + trunkHeight - 3;
                    const int canopyTopWorld = groundWorldY + trunkHeight;
                    for (int worldY = canopyBaseWorld; worldY <= canopyTopWorld; ++worldY)
                    {
                        const int layer = worldY - canopyBaseWorld;
                        int radius = 2;
                        if (worldY >= canopyTopWorld - 1)
                        {
                            radius = 1;
                        }

                        for (int dx = -radius; dx <= radius; ++dx)
                        {
                            for (int dz = -radius; dz <= radius; ++dz)
                            {
                                if (std::abs(dx) == radius && std::abs(dz) == radius && radius > 1)
                                {
                                    continue;
                                }

                                if (dx == 0 && dz == 0 && worldY <= groundWorldY + trunkHeight - 1)
                                {
                                    continue;
                                }

                                if (layer == 0 && std::abs(dx) + std::abs(dz) > 3)
                                {
                                    continue;
                                }

                                setOrQueueBlock(worldX + dx, worldY, worldZ + dz, BlockId::Leaves, false);
                            }
                        }
                    }
                }
            }
        }

        const bool appliedPending = applyPendingStructureEditsLocked(chunk);
        if (appliedPending)
        {
            anySolid = true;
        }

        chunk.hasBlocks.store(anySolid, std::memory_order_release);
        columnManager_.updateChunk(chunk);
    }

    if (!externalEdits.empty())
    {
        dispatchStructureEdits(externalEdits);
    }

    invalidatePredictedColumn({chunk.coord.x, chunk.coord.z});
}


bool ChunkManager::Impl::applyPendingStructureEditsLocked(Chunk& chunk)
{
    std::vector<PendingStructureEdit> edits;
    {
        std::lock_guard<std::mutex> lock(pendingStructureMutex_);
        auto it = pendingStructureEdits_.find(chunk.coord);
        if (it != pendingStructureEdits_.end())
        {
            edits = std::move(it->second);
            pendingStructureEdits_.erase(it);
        }
    }

    bool wroteSolid = false;
    for (const PendingStructureEdit& edit : edits)
    {
        const glm::ivec3 local = localBlockCoords(edit.worldPos, chunk.coord);
        if (local.x < 0 || local.x >= kChunkSizeX ||
            edit.worldPos.y < chunk.minWorldY || edit.worldPos.y > chunk.maxWorldY ||
            local.z < 0 || local.z >= kChunkSizeZ)
        {
            continue;
        }

        const int localY = edit.worldPos.y - chunk.minWorldY;
        BlockId& destination = chunk.blocks[blockIndex(local.x, localY, local.z)];
        if (!edit.replaceSolid && destination != BlockId::Air)
        {
            continue;
        }

        destination = edit.block;
        if (edit.block != BlockId::Air)
        {
            wroteSolid = true;
        }
    }

    return wroteSolid;
}

void ChunkManager::Impl::dispatchStructureEdits(const std::vector<PendingStructureEdit>& edits)
{
    if (edits.empty())
    {
        return;
    }

    std::unordered_set<glm::ivec3, ChunkHasher> touchedChunks;
    touchedChunks.reserve(edits.size());

    {
        std::lock_guard<std::mutex> lock(pendingStructureMutex_);
        for (const PendingStructureEdit& edit : edits)
        {
            pendingStructureEdits_[edit.chunkCoord].push_back(edit);
            touchedChunks.insert(edit.chunkCoord);
        }
    }

    for (const glm::ivec3& coord : touchedChunks)
    {
        auto chunk = getChunkShared(coord);
        if (!chunk)
        {
            continue;
        }

        ChunkState state = chunk->state.load(std::memory_order_acquire);
        if (state == ChunkState::Generating)
        {
            continue;
        }

        bool wroteSolid = false;
        {
            std::lock_guard<std::mutex> lock(chunk->meshMutex);
            wroteSolid = applyPendingStructureEditsLocked(*chunk);
            if (wroteSolid)
            {
                chunk->hasBlocks.store(true, std::memory_order_release);
                columnManager_.updateChunk(*chunk);
                invalidatePredictedColumn({chunk->coord.x, chunk->coord.z});
            }
        }

        if (!wroteSolid)
        {
            continue;
        }

        relightAroundChunk(coord);

        (void)state;
    }
}

bool ChunkManager::Impl::chunkHasSolidBlocks(const Chunk& chunk) noexcept
{
    return std::any_of(chunk.blocks.begin(), chunk.blocks.end(), [](BlockId block) { return block != BlockId::Air; });

}

ChunkManager::ChunkManager(unsigned seed)
    : impl_(std::make_unique<Impl>(seed))
{
}

ChunkManager::~ChunkManager() = default;

void ChunkManager::initializeRendering(ID3D12Device* device)
{
    impl_->initializeRendering(device);
}

void ChunkManager::setBlockTextureAtlasConfig(const BlockTextureAtlasConfig& config)
{
    impl_->setBlockTextureAtlasConfig(config);
}

void ChunkManager::update(const glm::vec3& cameraPos)
{
    impl_->update(cameraPos);
}

void ChunkManager::update(const glm::vec3& cameraPos, const glm::vec3& cameraForward)
{
    impl_->update(cameraPos, cameraForward);
}

WorldRenderData ChunkManager::buildRenderData(const Frustum& frustum) const
{
    return impl_->buildRenderData(frustum);
}

float ChunkManager::surfaceHeight(float worldX, float worldZ) const noexcept
{
    return impl_->surfaceHeight(worldX, worldZ);
}

terrain::ColumnSample ChunkManager::sampleColumnAt(const glm::vec3& worldPos,
                                                   int slabMinWorldY,
                                                   int slabMaxWorldY) const
{
    return impl_->sampleColumnAt(worldPos, slabMinWorldY, slabMaxWorldY);
}

void ChunkManager::clear()
{
    impl_->clear();
}

bool ChunkManager::destroyBlock(const glm::ivec3& worldPos)
{
    return impl_->destroyBlock(worldPos);
}

bool ChunkManager::placeBlock(const glm::ivec3& targetBlockPos,
                              const glm::ivec3& faceNormal,
                              BlockId block)
{
    return impl_->placeBlock(targetBlockPos, faceNormal, block);
}

RaycastHit ChunkManager::raycast(const glm::vec3& origin, const glm::vec3& direction) const
{
    return impl_->raycast(origin, direction);
}

void ChunkManager::updateHighlight(const glm::vec3& cameraPos, const glm::vec3& cameraDirection)
{
    impl_->updateHighlight(cameraPos, cameraDirection);
}

void ChunkManager::toggleViewDistance()
{
    impl_->toggleViewDistance();
}

int ChunkManager::viewDistance() const noexcept
{
    return impl_->viewDistance();
}

int ChunkManager::nearRenderDistance() const noexcept
{
    return impl_->nearRenderDistance();
}

int ChunkManager::farRenderDistanceBlocks() const noexcept
{
    return impl_->farRenderDistanceBlocks();
}

RenderDistanceSettings ChunkManager::renderDistanceSettings() const noexcept
{
    return impl_->renderDistanceSettings();
}

void ChunkManager::setRenderDistance(int distance) noexcept
{
    impl_->setRenderDistance(distance);
}

void ChunkManager::setNearRenderDistance(int chunks) noexcept
{
    impl_->setNearRenderDistance(chunks);
}

void ChunkManager::setFarRenderDistanceBlocks(int blocks) noexcept
{
    impl_->setFarRenderDistanceBlocks(blocks);
}

void ChunkManager::setFogStartBlocks(int blocks) noexcept
{
    impl_->setFogStartBlocks(blocks);
}

void ChunkManager::setLodEnabled(bool enabled)
{
    impl_->setLodEnabled(enabled);
}

bool ChunkManager::lodEnabled() const noexcept
{
    return impl_->lodEnabled();
}

void ChunkManager::setFarTerrainEnabled(bool enabled)
{
    impl_->setFarTerrainEnabled(enabled);
}

bool ChunkManager::farTerrainEnabled() const noexcept
{
    return impl_->farTerrainEnabled();
}

BlockId ChunkManager::blockAt(const glm::ivec3& worldPos) const noexcept
{
    return impl_->blockAt(worldPos);
}

LightSample ChunkManager::lightAt(const glm::ivec3& worldPos) const noexcept
{
    return impl_->lightAt(worldPos);
}

glm::vec3 ChunkManager::findSafeSpawnPosition(float worldX, float worldZ) const
{
    return impl_->findSafeSpawnPosition(worldX, worldZ);
}

void ChunkManager::beginSpawnPreload(const glm::vec3& spawnPos)
{
    impl_->beginSpawnPreload(spawnPos);
}

bool ChunkManager::isSpawnPreloadReady() const noexcept
{
    return impl_->isSpawnPreloadReady();
}

bool ChunkManager::playerReleaseReady() const noexcept
{
    return impl_->playerReleaseReady();
}

StreamingPhase ChunkManager::streamingPhase() const noexcept
{
    return impl_->streamingPhase();
}

void ChunkManager::setStartupEnabled(bool enabled) noexcept
{
    impl_->setStartupEnabled(enabled);
}

bool ChunkManager::startupEnabled() const noexcept
{
    return impl_->startupEnabled();
}

StreamingStatusSnapshot ChunkManager::streamingStatusSnapshot() const noexcept
{
    return impl_->streamingStatusSnapshot();
}

ChunkProfilingSnapshot ChunkManager::sampleProfilingSnapshot()
{
    return impl_->sampleProfilingSnapshot();
}

std::string ChunkManager::biomeNameAt(const glm::vec3& worldPos) const
{
    return impl_->biomeNameAt(worldPos);
}

