#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace terrain
{

struct FbmSettings
{
    float frequency{1.0f};
    int octaves{1};
    float gain{0.5f};
    float lacunarity{2.0f};
};

struct NoiseProfile
{
    FbmSettings main{0.01f, 6, 0.5f, 2.0f};
    FbmSettings medium{0.008f, 7, 0.5f, 2.0f};
    FbmSettings detail{0.04f, 8, 0.45f, 2.2f};
    FbmSettings mountain{0.004f, 5, 0.5f, 2.1f};
};

struct WorldgenProfile
{
    std::optional<unsigned> seedOverride{};
    std::string climateGenerator{"legacy"};
    NoiseProfile noise{};
    int seaLevel{20};

    [[nodiscard]] unsigned effectiveSeed(unsigned fallback) const noexcept
    {
        return seedOverride.value_or(fallback);
    }

    static WorldgenProfile load(const std::filesystem::path& path);
};

} // namespace terrain

