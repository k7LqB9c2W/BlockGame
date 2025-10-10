#include "terrain/biome_database.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#include <toml++/toml.h>

#include "../chunk_manager.h"

namespace terrain
{
namespace
{
std::string toLowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string requireString(const toml::table& table,
                          std::string_view key,
                          const std::filesystem::path& filePath)
{
    if (auto value = table[key].value<std::string>())
    {
        if (!value->empty())
        {
            return *value;
        }
    }

    std::ostringstream oss;
    oss << "Missing or empty string field '" << key << "' in " << filePath;
    throw std::runtime_error(oss.str());
}

bool requireBool(const toml::table& table, std::string_view key, const std::filesystem::path& filePath)
{
    if (auto value = table[key].value<bool>())
    {
        return *value;
    }

    std::ostringstream oss;
    oss << "Missing boolean field '" << key << "' in " << filePath;
    throw std::runtime_error(oss.str());
}

float requireFloat(const toml::table& table, std::string_view key, const std::filesystem::path& filePath)
{
    if (auto value = table[key].value<double>())
    {
        return static_cast<float>(*value);
    }
    if (auto valueF = table[key].value<float>())
    {
        return *valueF;
    }

    std::ostringstream oss;
    oss << "Missing floating-point field '" << key << "' in " << filePath;
    throw std::runtime_error(oss.str());
}

int requireInt(const toml::table& table, std::string_view key, const std::filesystem::path& filePath)
{
    if (auto value = table[key].value<std::int64_t>())
    {
        return static_cast<int>(*value);
    }

    std::ostringstream oss;
    oss << "Missing integer field '" << key << "' in " << filePath;
    throw std::runtime_error(oss.str());
}

BlockId parseBlockId(const std::string& text, const std::filesystem::path& filePath)
{
    const std::string lower = toLowerCopy(text);
    if (lower == "air") return BlockId::Air;
    if (lower == "grass") return BlockId::Grass;
    if (lower == "wood") return BlockId::Wood;
    if (lower == "leaves") return BlockId::Leaves;
    if (lower == "sand") return BlockId::Sand;
    if (lower == "water") return BlockId::Water;
    if (lower == "stone") return BlockId::Stone;

    std::ostringstream oss;
    oss << "Unknown block id '" << text << "' in " << filePath;
    throw std::runtime_error(oss.str());
}

std::vector<std::string> parseFlags(const toml::table& table, const std::filesystem::path& filePath)
{
    const toml::node_view flagsNode = table["flags"];
    const toml::array* flagsArray = flagsNode.as_array();
    if (!flagsArray)
    {
        std::ostringstream oss;
        oss << "Missing array field 'flags' in " << filePath;
        throw std::runtime_error(oss.str());
    }

    std::vector<std::string> flags;
    flags.reserve(flagsArray->size());
    for (const toml::node& node : *flagsArray)
    {
        if (const auto* str = node.as_string())
        {
            const std::string value = str->get();
            if (value.empty())
            {
                std::ostringstream oss;
                oss << "Encountered empty flag in " << filePath;
                throw std::runtime_error(oss.str());
            }
            flags.push_back(value);
        }
        else
        {
            std::ostringstream oss;
            oss << "Non-string flag entry in " << filePath;
            throw std::runtime_error(oss.str());
        }
    }

    return flags;
}

void validateHeights(const BiomeDefinition& definition, const std::filesystem::path& filePath)
{
    if (definition.minHeight > definition.maxHeight)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' has min_height greater than max_height in " << filePath;
        throw std::runtime_error(oss.str());
    }
}

void validateNumericRanges(const BiomeDefinition& definition, const std::filesystem::path& filePath)
{
    if (!std::isfinite(definition.heightOffset) || !std::isfinite(definition.heightScale)
        || !std::isfinite(definition.maxGradient) || !std::isfinite(definition.baseSlopeBias)
        || !std::isfinite(definition.treeDensityMultiplier) || !std::isfinite(definition.footprintMultiplier))
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' contains non-finite numeric fields in " << filePath;
        throw std::runtime_error(oss.str());
    }

    if (definition.footprintMultiplier <= 0.0f)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have footprint_multiplier > 0 in " << filePath;
        throw std::runtime_error(oss.str());
    }

    if (definition.maxGradient < 0.0f)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have non-negative max_gradient in " << filePath;
        throw std::runtime_error(oss.str());
    }

    if (definition.heightScale < 0.0f)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have non-negative height_scale in " << filePath;
        throw std::runtime_error(oss.str());
    }

    if (definition.treeDensityMultiplier < 0.0f)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have non-negative tree_density_multiplier in " << filePath;
        throw std::runtime_error(oss.str());
    }
}

} // namespace

bool BiomeDefinition::hasFlag(std::string_view flag) const noexcept
{
    std::string normalized(flag);
    normalized = toLowerCopy(normalized);
    return flagLookup_.find(normalized) != flagLookup_.end();
}

void BiomeDefinition::setFlags(std::vector<std::string> flags)
{
    flags_ = std::move(flags);
    flagLookup_.clear();
    oceanFlag_ = false;
    littleMountainsFlag_ = false;

    for (std::string& flag : flags_)
    {
        const std::string normalized = toLowerCopy(flag);
        flag = normalized;
        flagLookup_.insert(normalized);
        if (normalized == "ocean")
        {
            oceanFlag_ = true;
        }
        else if (normalized == "little_mountains")
        {
            littleMountainsFlag_ = true;
        }
    }
}

BiomeDatabase::BiomeDatabase(const std::filesystem::path& directory)
{
    loadFromDirectory(directory);
}

const BiomeDefinition& BiomeDatabase::biome(const std::string& id) const
{
    const std::string normalized = toLowerCopy(id);
    auto it = indexById_.find(normalized);
    if (it == indexById_.end())
    {
        std::ostringstream oss;
        oss << "Biome '" << id << "' not found";
        throw std::runtime_error(oss.str());
    }

    return definitions_[it->second];
}

const BiomeDefinition* BiomeDatabase::tryGetBiome(const std::string& id) const noexcept
{
    const std::string normalized = toLowerCopy(id);
    auto it = indexById_.find(normalized);
    if (it == indexById_.end())
    {
        return nullptr;
    }
    return &definitions_[it->second];
}

const BiomeDefinition& BiomeDatabase::definitionByIndex(std::size_t index) const
{
    if (index >= definitions_.size())
    {
        throw std::out_of_range("Biome index out of range");
    }
    return definitions_[index];
}

void BiomeDatabase::loadFromDirectory(const std::filesystem::path& directory)
{
    namespace fs = std::filesystem;
    if (!fs::exists(directory))
    {
        std::ostringstream oss;
        oss << "Biome directory '" << directory << "' does not exist";
        throw std::runtime_error(oss.str());
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(directory))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        const fs::path& path = entry.path();
        if (path.extension() == ".toml")
        {
            files.push_back(path);
        }
    }

    std::sort(files.begin(), files.end());

    if (files.empty())
    {
        std::ostringstream oss;
        oss << "No biome configuration files found in " << directory;
        throw std::runtime_error(oss.str());
    }

    definitions_.clear();
    indexById_.clear();
    maxFootprintMultiplier_ = 1.0f;

    for (const fs::path& path : files)
    {
        BiomeDefinition definition = parseBiomeFile(path);
        const std::string normalizedId = toLowerCopy(definition.id);
        if (indexById_.find(normalizedId) != indexById_.end())
        {
            std::ostringstream oss;
            oss << "Duplicate biome id '" << definition.id << "' in " << path;
            throw std::runtime_error(oss.str());
        }

        definition.id = normalizedId;
        maxFootprintMultiplier_ = std::max(maxFootprintMultiplier_, definition.footprintMultiplier);
        const std::size_t index = definitions_.size();
        definitions_.push_back(std::move(definition));
        indexById_[normalizedId] = index;
    }
}

BiomeDefinition BiomeDatabase::parseBiomeFile(const std::filesystem::path& path)
{
    toml::table table = toml::parse_file(path.string());

    BiomeDefinition definition{};
    definition.id = requireString(table, "id", path);
    definition.name = requireString(table, "name", path);
    definition.surfaceBlock = parseBlockId(requireString(table, "surface_block", path), path);
    definition.fillerBlock = parseBlockId(requireString(table, "filler_block", path), path);
    definition.generatesTrees = requireBool(table, "generates_trees", path);
    definition.treeDensityMultiplier = requireFloat(table, "tree_density_multiplier", path);
    definition.heightOffset = requireFloat(table, "height_offset", path);
    definition.heightScale = requireFloat(table, "height_scale", path);
    definition.minHeight = requireInt(table, "min_height", path);
    definition.maxHeight = requireInt(table, "max_height", path);
    definition.baseSlopeBias = requireFloat(table, "base_slope_bias", path);
    definition.maxGradient = requireFloat(table, "max_gradient", path);
    definition.footprintMultiplier = requireFloat(table, "footprint_multiplier", path);
    definition.setFlags(parseFlags(table, path));

    validateHeights(definition, path);
    validateNumericRanges(definition, path);

    return definition;
}

} // namespace terrain

