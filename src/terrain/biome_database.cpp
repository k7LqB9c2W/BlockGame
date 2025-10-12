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

#include <glm/common.hpp>

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

std::uint16_t propertyBitFromString(std::string_view name, const std::filesystem::path& filePath)
{
    std::string lowered{name};
    lowered = toLowerCopy(std::move(lowered));
    using GP = BiomeDefinition::GenerationProperties;
    if (lowered == "hot") return GP::kHot;
    if (lowered == "temperate") return GP::kTemperate;
    if (lowered == "cold") return GP::kCold;
    if (lowered == "inland") return GP::kInland;
    if (lowered == "land") return GP::kLand;
    if (lowered == "ocean") return GP::kOcean;
    if (lowered == "wet") return GP::kWet;
    if (lowered == "neither_wet_nor_dry" || lowered == "neutral" || lowered == "neutral_hydration")
        return GP::kNeutralHydration;
    if (lowered == "dry") return GP::kDry;
    if (lowered == "barren") return GP::kBarren;
    if (lowered == "balanced") return GP::kBalanced;
    if (lowered == "overgrown") return GP::kOvergrown;
    if (lowered == "mountain") return GP::kMountain;
    if (lowered == "low_terrain" || lowered == "lowterrain") return GP::kLowTerrain;
    if (lowered == "anti_mountain" || lowered == "antimountain") return GP::kAntiMountain;

    std::ostringstream oss;
    oss << "Unknown biome property '" << name << "' in " << filePath;
    throw std::runtime_error(oss.str());
}

BiomeDefinition::GenerationProperties parseGenerationProperties(const toml::table& table,
                                                                const std::filesystem::path& filePath,
                                                                bool fillMissingGroups)
{
    const toml::node_view propertiesNode = table["properties"];
    const toml::array* propertiesArray = propertiesNode.as_array();
    if (!propertiesArray)
    {
        BiomeDefinition::GenerationProperties props{};
        if (fillMissingGroups)
        {
            props.fillMissingGroups();
        }
        return props;
    }

    BiomeDefinition::GenerationProperties result{};
    for (const toml::node& node : *propertiesArray)
    {
        if (const auto* str = node.as_string())
        {
            result.add(propertyBitFromString(str->get(), filePath));
        }
        else
        {
            std::ostringstream oss;
            oss << "Non-string property entry in " << filePath;
            throw std::runtime_error(oss.str());
        }
    }

    if (fillMissingGroups)
    {
        result.fillMissingGroups();
    }

    return result;
}

BiomeDefinition::InterpolationCurve parseInterpolationCurve(const std::string& value,
                                                            const std::filesystem::path& filePath)
{
    const std::string normalized = toLowerCopy(value);
    if (normalized == "none" || normalized == "step")
    {
        return BiomeDefinition::InterpolationCurve::Step;
    }
    if (normalized == "linear")
    {
        return BiomeDefinition::InterpolationCurve::Linear;
    }
    if (normalized == "square")
    {
        return BiomeDefinition::InterpolationCurve::Square;
    }

    std::ostringstream oss;
    oss << "Unknown interpolation_curve value '" << value << "' in " << filePath;
    throw std::runtime_error(oss.str());
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
    if (!std::isfinite(definition.radius) || definition.radius <= 0.0f)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have positive finite radius in " << filePath;
        throw std::runtime_error(oss.str());
    }
    if (!std::isfinite(definition.radiusVariation) || definition.radiusVariation < 0.0f)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have non-negative finite radius_variation in " << filePath;
        throw std::runtime_error(oss.str());
    }
    if (!std::isfinite(definition.spawnChance) || definition.spawnChance < 0.0f)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have non-negative finite spawn_chance in " << filePath;
        throw std::runtime_error(oss.str());
    }

    if (!std::isfinite(definition.interpolationWeight) || definition.interpolationWeight <= 0.0f)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have positive finite interpolation_weight in " << filePath;
        throw std::runtime_error(oss.str());
    }

    const auto& soil = definition.terrainSettings.soilCreep;
    if (!std::isfinite(soil.strength) || soil.strength < 0.0f)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have non-negative finite soil_creep.strength in " << filePath;
        throw std::runtime_error(oss.str());
    }
    if (soil.maxStep < 0 || soil.maxDepth < 0)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have non-negative soil_creep max_step/max_depth in " << filePath;
        throw std::runtime_error(oss.str());
    }

    const auto& stripes = definition.terrainSettings.stripes;
    if (!std::isfinite(stripes.noiseThreshold))
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have finite stripes.noise_threshold in " << filePath;
        throw std::runtime_error(oss.str());
    }
    if (stripes.period < 0 || stripes.thickness < 0)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have non-negative stripes period/thickness in " << filePath;
        throw std::runtime_error(oss.str());
    }
    if (stripes.noiseThreshold < 0.0f || stripes.noiseThreshold > 1.0f)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have stripes.noise_threshold between 0 and 1 in " << filePath;
        throw std::runtime_error(oss.str());
    }

    const auto& water = definition.terrainSettings.waterFill;
    if (water.maxDepth < 0)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' must have non-negative water_fill.max_depth in " << filePath;
        throw std::runtime_error(oss.str());
    }

    if (definition.minHeightLimit && definition.maxHeightLimit
        && *definition.minHeightLimit > *definition.maxHeightLimit)
    {
        std::ostringstream oss;
        oss << "Biome '" << definition.id << "' has min_height_limit greater than max_height_limit in " << filePath;
        throw std::runtime_error(oss.str());
    }
}

} // namespace

void BiomeDefinition::GenerationProperties::fillMissingGroups() noexcept
{
    const std::uint16_t val = bits;
    const std::uint16_t empty =
        static_cast<std::uint16_t>(~val & (~val >> 1) & (~val >> 2) & kMask);
    bits = static_cast<std::uint16_t>(val | empty | (empty << 1) | (empty << 2));
}

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

    for (std::string& flag : flags_)
    {
        const std::string normalized = toLowerCopy(flag);
        flag = normalized;
        flagLookup_.insert(normalized);
        if (normalized == "ocean")
        {
            oceanFlag_ = true;
        }
    }
}

float BiomeDefinition::applyHeightLimits(float height, float normalizedDistance) const noexcept
{
    if (!hasHeightLimits())
    {
        return height;
    }

    const float t = std::clamp(normalizedDistance, 0.0f, 1.0f);
    const float fade = glm::smoothstep(0.35f, 0.95f, t);
    if (fade <= 0.0f)
    {
        return height;
    }

    const float original = height;
    float result = height;

    if (minHeightLimit)
    {
        const float limit = static_cast<float>(*minHeightLimit);
        const float target = glm::mix(original, limit, fade);
        result = std::max(result, target);
    }

    if (maxHeightLimit)
    {
        const float limit = static_cast<float>(*maxHeightLimit);
        const float target = glm::mix(original, limit, fade);
        result = std::min(result, target);
    }

    return result;
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
        maxBiomeRadius_ = std::max(maxBiomeRadius_, definition.maxRadius());
        const std::size_t index = definitions_.size();
        definitions_.push_back(std::move(definition));
        indexById_[normalizedId] = index;
    }

    for (BiomeDefinition& definition : definitions_)
    {
        for (auto& transition : definition.transitionBiomes)
        {
            if (transition.biomeId.empty())
            {
                continue;
            }
            const BiomeDefinition* target = tryGetBiome(transition.biomeId);
            if (!target)
            {
                std::ostringstream oss;
                oss << "Transition biome '" << transition.biomeId << "' referenced by biome '"
                    << definition.id << "' was not found";
                throw std::runtime_error(oss.str());
            }
            transition.biome = target;
        }

        float accumulatedChance = 0.0f;
        for (auto& sub : definition.subBiomes)
        {
            if (sub.biomeId.empty())
            {
                continue;
            }
            const BiomeDefinition* target = tryGetBiome(sub.biomeId);
            if (!target)
            {
                std::ostringstream oss;
                oss << "Sub-biome '" << sub.biomeId << "' referenced by biome '" << definition.id
                    << "' was not found";
                throw std::runtime_error(oss.str());
            }
            sub.biome = target;
            accumulatedChance += std::max(sub.chance, 0.0f);
        }
        if (definition.subBiomeTotalChance <= 0.0f && accumulatedChance > 0.0f)
        {
            definition.subBiomeTotalChance = accumulatedChance;
        }
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
    if (const auto offsetValue = table["height_offset"].value<double>())
    {
        definition.heightOffset = static_cast<float>(*offsetValue);
    }
    else if (const auto offsetFloat = table["height_offset"].value<float>())
    {
        definition.heightOffset = *offsetFloat;
    }
    else
    {
        definition.heightOffset = 0.0f;
    }

    if (const auto scaleValue = table["height_scale"].value<double>())
    {
        definition.heightScale = static_cast<float>(*scaleValue);
    }
    else if (const auto scaleFloat = table["height_scale"].value<float>())
    {
        definition.heightScale = *scaleFloat;
    }
    else
    {
        definition.heightScale = 0.0f;
    }
    definition.minHeight = requireInt(table, "min_height", path);
    definition.maxHeight = requireInt(table, "max_height", path);
    if (const auto minLimitValue = table["min_height_limit"].value<std::int64_t>())
    {
        definition.minHeightLimit = static_cast<int>(*minLimitValue);
    }
    else if (const auto minLimitFloat = table["min_height_limit"].value<double>())
    {
        definition.minHeightLimit = static_cast<int>(std::lround(*minLimitFloat));
    }

    if (const auto maxLimitValue = table["max_height_limit"].value<std::int64_t>())
    {
        definition.maxHeightLimit = static_cast<int>(*maxLimitValue);
    }
    else if (const auto maxLimitFloat = table["max_height_limit"].value<double>())
    {
        definition.maxHeightLimit = static_cast<int>(std::lround(*maxLimitFloat));
    }
    definition.baseSlopeBias = requireFloat(table, "base_slope_bias", path);
    definition.maxGradient = requireFloat(table, "max_gradient", path);
    definition.footprintMultiplier =
        BiomeDefinition::clampFootprintMultiplier(requireFloat(table, "footprint_multiplier", path));
    if (const auto roughValue = table["roughness"].value<double>())
    {
        definition.roughness = static_cast<float>(*roughValue);
    }
    else if (const auto roughFloat = table["roughness"].value<float>())
    {
        definition.roughness = *roughFloat;
    }

    if (const auto hillsValue = table["hills"].value<double>())
    {
        definition.hills = static_cast<float>(*hillsValue);
    }
    else if (const auto hillsFloat = table["hills"].value<float>())
    {
        definition.hills = *hillsFloat;
    }

    if (const auto mountainsValue = table["mountains"].value<double>())
    {
        definition.mountains = static_cast<float>(*mountainsValue);
    }
    else if (const auto mountainsFloat = table["mountains"].value<float>())
    {
        definition.mountains = *mountainsFloat;
    }

    if (const auto keepValue = table["keep_original_terrain"].value<double>())
    {
        definition.keepOriginalTerrain = static_cast<float>(*keepValue);
    }
    else if (const auto keepFloat = table["keep_original_terrain"].value<float>())
    {
        definition.keepOriginalTerrain = *keepFloat;
    }

    definition.keepOriginalTerrain = std::clamp(definition.keepOriginalTerrain, 0.0f, 1.0f);

    const auto readFloatOr = [&](const toml::table& tbl, std::string_view key, float fallback) -> float {
        if (const auto val = tbl[key].value<double>())
        {
            return static_cast<float>(*val);
        }
        if (const auto valf = tbl[key].value<float>())
        {
            return *valf;
        }
        return fallback;
    };

    bool minRadiusProvided = false;
    bool maxRadiusProvided = false;
    float minRadius = readFloatOr(table, "min_radius", definition.radius);
    if (table.contains("min_radius"))
    {
        minRadiusProvided = true;
    }
    else if (table.contains("radius_min"))
    {
        minRadius = readFloatOr(table, "radius_min", definition.radius);
        minRadiusProvided = true;
    }

    float maxRadius = readFloatOr(table, "max_radius", definition.radius);
    if (table.contains("max_radius"))
    {
        maxRadiusProvided = true;
    }
    else if (table.contains("radius_max"))
    {
        maxRadius = readFloatOr(table, "radius_max", definition.radius);
        maxRadiusProvided = true;
    }

    if (const auto radiusValue = table["radius"].value<double>())
    {
        definition.radius = std::max(static_cast<float>(*radiusValue), 1.0f);
    }
    else if (const auto radiusFloat = table["radius"].value<float>())
    {
        definition.radius = std::max(*radiusFloat, 1.0f);
    }

    if (const auto radiusVarValue = table["radius_variation"].value<double>())
    {
        definition.radiusVariation = std::max(static_cast<float>(*radiusVarValue), 0.0f);
    }
    else if (const auto radiusVarFloat = table["radius_variation"].value<float>())
    {
        definition.radiusVariation = std::max(*radiusVarFloat, 0.0f);
    }

    if (const auto fixedRadiusValue = table["fixed_radius"].value<bool>())
    {
        definition.fixedRadius = *fixedRadiusValue;
    }

    if (minRadiusProvided || maxRadiusProvided)
    {
        minRadius = std::max(minRadius, 1.0f);
        maxRadius = std::max(maxRadius, minRadius);
        definition.radius = 0.5f * (minRadius + maxRadius);
        definition.radiusVariation = std::max(0.5f * (maxRadius - minRadius), 0.0f);
    }

    definition.spawnChance = readFloatOr(table, "spawn_chance", definition.spawnChance);

    if (const auto smooth = table["smooth_beaches"].value<bool>())
    {
        definition.terrainSettings.smoothBeaches = *smooth;
    }

    if (const auto interpolationCurveValue = table["interpolation_curve"].value<std::string>())
    {
        definition.interpolationCurve = parseInterpolationCurve(*interpolationCurveValue, path);
    }

    if (const auto interpolationWeightValue = table["interpolation_weight"].value<double>())
    {
        definition.interpolationWeight = static_cast<float>(*interpolationWeightValue);
    }
    else if (const auto interpolationWeightFloat = table["interpolation_weight"].value<float>())
    {
        definition.interpolationWeight = *interpolationWeightFloat;
    }

    if (const toml::table* soilTable = table["soil_creep"].as_table())
    {
        auto& soil = definition.terrainSettings.soilCreep;
        if (const auto strengthValue = (*soilTable)["strength"].value<double>())
        {
            soil.strength = static_cast<float>(*strengthValue);
        }
        else if (const auto strengthFloat = (*soilTable)["strength"].value<float>())
        {
            soil.strength = *strengthFloat;
        }
        if (const auto maxStepValue = (*soilTable)["max_step"].value<std::int64_t>())
        {
            soil.maxStep = static_cast<int>(*maxStepValue);
        }
        if (const auto maxDepthValue = (*soilTable)["max_depth"].value<std::int64_t>())
        {
            soil.maxDepth = static_cast<int>(*maxDepthValue);
        }
    }

    if (const toml::table* stripeTable = table["stripes"].as_table())
    {
        auto& stripes = definition.terrainSettings.stripes;
        stripes.enabled = true;
        if (const auto enabledValue = (*stripeTable)["enabled"].value<bool>())
        {
            stripes.enabled = *enabledValue;
        }
        if (const auto blockValue = (*stripeTable)["block"].value<std::string>())
        {
            stripes.block = parseBlockId(*blockValue, path);
        }
        else
        {
            stripes.block = definition.fillerBlock;
        }
        if (const auto periodValue = (*stripeTable)["period"].value<std::int64_t>())
        {
            stripes.period = static_cast<int>(*periodValue);
        }
        if (const auto thicknessValue = (*stripeTable)["thickness"].value<std::int64_t>())
        {
            stripes.thickness = static_cast<int>(*thicknessValue);
        }
        if (const auto thresholdValue = (*stripeTable)["noise_threshold"].value<double>())
        {
            stripes.noiseThreshold = static_cast<float>(*thresholdValue);
        }
        else if (const auto thresholdFloat = (*stripeTable)["noise_threshold"].value<float>())
        {
            stripes.noiseThreshold = *thresholdFloat;
        }
    }

    if (const toml::table* waterTable = table["water_fill"].as_table())
    {
        auto& water = definition.terrainSettings.waterFill;
        water.enabled = true;
        if (const auto enabledValue = (*waterTable)["enabled"].value<bool>())
        {
            water.enabled = *enabledValue;
        }
        if (const auto blockValue = (*waterTable)["block"].value<std::string>())
        {
            water.block = parseBlockId(*blockValue, path);
        }
        else
        {
            water.block = BlockId::Water;
        }
        if (const auto maxDepthValue = (*waterTable)["max_depth"].value<std::int64_t>())
        {
            water.maxDepth = static_cast<int>(*maxDepthValue);
        }
    }

    std::vector<std::string> flags = parseFlags(table, path);
    definition.setFlags(flags);

    const bool hasExplicitProperties = table.contains("properties");
    if (hasExplicitProperties)
    {
        definition.properties = parseGenerationProperties(table, path, false);
        definition.properties.fillMissingGroups();
    }
    else
    {
        BiomeDefinition::GenerationProperties derived{};
        for (const std::string& flag : flags)
        {
            try
            {
                derived.add(propertyBitFromString(flag, path));
            }
            catch (const std::exception&)
            {
                // Ignore flags that are not mapped to generation properties.
            }
        }
        derived.fillMissingGroups();
        definition.properties = derived;
    }

    if (const toml::array* transitionArray = table["transition_biomes"].as_array())
    {
        definition.transitionBiomes.reserve(transitionArray->size());
        for (const toml::node& node : *transitionArray)
        {
            const toml::table* transitionTable = node.as_table();
            if (!transitionTable)
            {
                std::ostringstream oss;
                oss << "Transition biome entry is not a table in " << path;
                throw std::runtime_error(oss.str());
            }

            BiomeDefinition::TransitionBiomeDefinition entry{};
            entry.biomeId = toLowerCopy(requireString(*transitionTable, "id", path));
            entry.chance = readFloatOr(*transitionTable, "chance", entry.chance);
            if (const auto widthValue = (*transitionTable)["width"].value<std::int64_t>())
            {
                entry.width = std::max(1, static_cast<int>(*widthValue));
            }

            entry.propertyMask = parseGenerationProperties(*transitionTable, path, false);
            definition.transitionBiomes.push_back(std::move(entry));
        }
    }

    definition.maxSubBiomeCount = readFloatOr(table, "max_sub_biome_count", definition.maxSubBiomeCount);
    definition.subBiomeTotalChance = readFloatOr(table, "sub_biome_total_chance", definition.subBiomeTotalChance);
    if (const toml::array* subBiomes = table["sub_biomes"].as_array())
    {
        definition.subBiomes.reserve(subBiomes->size());
        for (const toml::node& node : *subBiomes)
        {
            const toml::table* subTable = node.as_table();
            if (!subTable)
            {
                std::ostringstream oss;
                oss << "Sub-biome entry is not a table in " << path;
                throw std::runtime_error(oss.str());
            }

            BiomeDefinition::SubBiomeDefinition sub{};
            sub.biomeId = toLowerCopy(requireString(*subTable, "id", path));
            sub.chance = readFloatOr(*subTable, "chance", sub.chance);
            sub.minRadius = readFloatOr(*subTable, "min_radius", sub.minRadius);
            sub.maxRadius = readFloatOr(*subTable, "max_radius", sub.maxRadius);
            definition.subBiomes.push_back(std::move(sub));
        }
    }

    if (definition.terrainSettings.waterFill.block == BlockId::Air)
    {
        definition.terrainSettings.waterFill.block = BlockId::Water;
    }
    if (!definition.terrainSettings.waterFill.enabled && definition.isOcean())
    {
        definition.terrainSettings.waterFill.enabled = true;
        definition.terrainSettings.waterFill.block = BlockId::Water;
        definition.terrainSettings.waterFill.maxDepth = 32;
    }

    validateHeights(definition, path);
    validateNumericRanges(definition, path);

    return definition;
}

} // namespace terrain

