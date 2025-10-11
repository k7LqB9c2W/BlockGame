#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum class BlockId : std::uint8_t;

namespace terrain
{

struct BiomeDefinition
{
    std::string id;
    std::string name;
    BlockId surfaceBlock;
    BlockId fillerBlock;
    bool generatesTrees{false};
    float treeDensityMultiplier{0.0f};
    float heightOffset{0.0f};
    float heightScale{0.0f};
    int minHeight{0};
    int maxHeight{0};
    float baseSlopeBias{0.0f};
    float maxGradient{0.0f};
    float footprintMultiplier{1.0f};
    float roughness{0.0f};
    float hills{0.0f};
    float mountains{0.0f};
    float keepOriginalTerrain{0.0f};
    float radius{256.0f};
    float radiusVariation{0.0f};
    float spawnChance{1.0f};
    enum class InterpolationCurve
    {
        Step,
        Linear,
        Square
    };
    InterpolationCurve interpolationCurve{InterpolationCurve::Square};
    float interpolationWeight{1.0f};
    struct SoilCreepSettings
    {
        float strength{0.0f};
        int maxStep{0};
        int maxDepth{0};
    };

    struct StripeSettings
    {
        bool enabled{false};
        BlockId block{};
        int period{4};
        int thickness{1};
        float noiseThreshold{0.0f};
    };

    struct WaterFillSettings
    {
        bool enabled{false};
        BlockId block{};
        int maxDepth{0};
    };

    struct TerrainSettings
    {
        bool smoothBeaches{false};
        SoilCreepSettings soilCreep{};
        StripeSettings stripes{};
        WaterFillSettings waterFill{};
    } terrainSettings{};

    [[nodiscard]] bool isOcean() const noexcept { return oceanFlag_; }
    [[nodiscard]] bool hasFlag(std::string_view flag) const noexcept;
    [[nodiscard]] const std::vector<std::string>& flags() const noexcept { return flags_; }

    void setFlags(std::vector<std::string> flags);
    [[nodiscard]] static float clampFootprintMultiplier(float value) noexcept
    {
        constexpr float kMinFootprintMultiplier = 0.25f;
        constexpr float kMaxFootprintMultiplier = 3.0f;
        return std::clamp(value, kMinFootprintMultiplier, kMaxFootprintMultiplier);
    }
    [[nodiscard]] float minRadius() const noexcept
    {
        return std::max(radius - radiusVariation, 1.0f);
    }
    [[nodiscard]] float maxRadius() const noexcept
    {
        return std::max(radius + radiusVariation, 1.0f);
    }

private:
    std::vector<std::string> flags_{};
    std::unordered_set<std::string> flagLookup_{};
    bool oceanFlag_{false};
};

class BiomeDatabase
{
public:
    explicit BiomeDatabase(const std::filesystem::path& directory = std::filesystem::path("assets/biomes"));

    [[nodiscard]] const BiomeDefinition& biome(const std::string& id) const;
    [[nodiscard]] const BiomeDefinition* tryGetBiome(const std::string& id) const noexcept;
    [[nodiscard]] const BiomeDefinition& definitionByIndex(std::size_t index) const;
    [[nodiscard]] const std::vector<BiomeDefinition>& definitions() const noexcept { return definitions_; }
    [[nodiscard]] std::size_t biomeCount() const noexcept { return definitions_.size(); }
    [[nodiscard]] float maxFootprintMultiplier() const noexcept { return maxFootprintMultiplier_; }
    [[nodiscard]] float maxBiomeRadius() const noexcept { return maxBiomeRadius_; }

private:
    void loadFromDirectory(const std::filesystem::path& directory);
    static BiomeDefinition parseBiomeFile(const std::filesystem::path& path);

    std::vector<BiomeDefinition> definitions_{};
    std::unordered_map<std::string, std::size_t> indexById_{};
    float maxFootprintMultiplier_{1.0f};
    float maxBiomeRadius_{256.0f};
};

} // namespace terrain
