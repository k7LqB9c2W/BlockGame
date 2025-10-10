#include "terrain/worldgen_profile.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <toml++/toml.h>

namespace terrain
{
namespace
{
float readFloat(const toml::table& table, std::string_view key, float fallback)
{
    if (auto value = table[key].value<double>())
    {
        return static_cast<float>(*value);
    }
    if (auto valueF = table[key].value<float>())
    {
        return *valueF;
    }
    return fallback;
}

int readInt(const toml::table& table, std::string_view key, int fallback)
{
    if (auto value = table[key].value<std::int64_t>())
    {
        return static_cast<int>(*value);
    }
    return fallback;
}

void applyFbmSettings(const toml::table& noiseTable,
                      std::string_view key,
                      FbmSettings& settings,
                      const std::filesystem::path& filePath)
{
    const toml::table* fbmTable = noiseTable[key].as_table();
    if (!fbmTable)
    {
        return;
    }

    settings.frequency = readFloat(*fbmTable, "frequency", settings.frequency);
    settings.gain = readFloat(*fbmTable, "gain", settings.gain);
    settings.lacunarity = readFloat(*fbmTable, "lacunarity", settings.lacunarity);
    settings.octaves = readInt(*fbmTable, "octaves", settings.octaves);

    if (settings.frequency < 0.0f || !std::isfinite(settings.frequency))
    {
        std::ostringstream oss;
        oss << "Noise frequency for '" << key << "' in " << filePath << " must be non-negative";
        throw std::runtime_error(oss.str());
    }

    if (settings.octaves <= 0)
    {
        std::ostringstream oss;
        oss << "Noise octaves for '" << key << "' in " << filePath << " must be positive";
        throw std::runtime_error(oss.str());
    }

    if (!std::isfinite(settings.gain) || !std::isfinite(settings.lacunarity))
    {
        std::ostringstream oss;
        oss << "Noise parameters for '" << key << "' in " << filePath << " must be finite";
        throw std::runtime_error(oss.str());
    }
}

} // namespace

WorldgenProfile WorldgenProfile::load(const std::filesystem::path& path)
{
    WorldgenProfile profile{};
    if (!std::filesystem::exists(path))
    {
        return profile;
    }

    toml::table table = toml::parse_file(path.string());

    if (auto seedValue = table["seed"].value<std::int64_t>())
    {
        if (*seedValue < 0 || *seedValue > static_cast<std::int64_t>(std::numeric_limits<unsigned>::max()))
        {
            std::ostringstream oss;
            oss << "Seed value out of range in " << path;
            throw std::runtime_error(oss.str());
        }
        profile.seedOverride = static_cast<unsigned>(*seedValue);
    }

    if (auto climate = table["climate_generator"].value<std::string>())
    {
        profile.climateGenerator = *climate;
    }

    if (auto seaLevelValue = table["sea_level"].value<std::int64_t>())
    {
        profile.seaLevel = static_cast<int>(*seaLevelValue);
    }

    if (const toml::table* noiseTable = table["noise"].as_table())
    {
        applyFbmSettings(*noiseTable, "main", profile.noise.main, path);
        applyFbmSettings(*noiseTable, "medium", profile.noise.medium, path);
        applyFbmSettings(*noiseTable, "detail", profile.noise.detail, path);
        applyFbmSettings(*noiseTable, "mountain", profile.noise.mountain, path);
    }

    return profile;
}

} // namespace terrain

