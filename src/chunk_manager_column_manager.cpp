// chunk_manager_column_manager.cpp
// Implements the internal column-height cache used by ChunkManager streaming and spawn queries.

#include "chunk_manager_support.h"

#include <algorithm>

namespace
{
[[nodiscard]] constexpr std::size_t blockIndex(int x, int y, int z) noexcept
{
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ) +
           static_cast<std::size_t>(z) * static_cast<std::size_t>(kChunkSizeX) +
           static_cast<std::size_t>(x);
}

[[nodiscard]] constexpr std::size_t columnIndex(int x, int z) noexcept
{
    return static_cast<std::size_t>(z) * static_cast<std::size_t>(kChunkSizeX) +
           static_cast<std::size_t>(x);
}
} // namespace

glm::ivec2 ColumnManager::columnKey(const glm::ivec3& chunkCoord, int localX, int localZ) noexcept
{
    return {chunkCoord.x * kChunkSizeX + localX, chunkCoord.z * kChunkSizeZ + localZ};
}

int ColumnManager::scanColumnHighestWorld(const ChunkBlockView& chunk, int localX, int localZ) noexcept
{
    if (chunk.blocks.size() < static_cast<std::size_t>(kChunkBlockCount))
    {
        return kNoHeight;
    }

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
    (void)inserted;
    it->second.slabHeights[chunkY] = highestWorldY;
    it->second.highestWorldY = computeHighest(it->second);
}

void ColumnManager::updateChunk(const ChunkBlockView& chunk)
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

void ColumnManager::updateChunkHeights(
    const glm::ivec3& chunkCoord,
    const std::array<int, static_cast<std::size_t>(kChunkSizeX * kChunkSizeZ)>& highestWorlds)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (int x = 0; x < kChunkSizeX; ++x)
    {
        for (int z = 0; z < kChunkSizeZ; ++z)
        {
            applyHeightLocked(columnKey(chunkCoord, x, z), chunkCoord.y, highestWorlds[columnIndex(x, z)]);
        }
    }
}

void ColumnManager::updateColumn(const ChunkBlockView& chunk, int localX, int localZ)
{
    const int highestWorld = scanColumnHighestWorld(chunk, localX, localZ);
    std::lock_guard<std::mutex> lock(mutex_);
    applyHeightLocked(columnKey(chunk.coord, localX, localZ), chunk.coord.y, highestWorld);
}

void ColumnManager::removeChunk(const glm::ivec3& chunkCoord)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (int x = 0; x < kChunkSizeX; ++x)
    {
        for (int z = 0; z < kChunkSizeZ; ++z)
        {
            applyHeightLocked(columnKey(chunkCoord, x, z), chunkCoord.y, kNoHeight);
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
