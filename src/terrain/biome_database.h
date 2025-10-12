#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
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
    struct GenerationProperties
    {
        static constexpr std::uint16_t kMask = 0b001001001001001u;
        static constexpr std::uint16_t kHot = 1u << 0;
        static constexpr std::uint16_t kTemperate = 1u << 1;
        static constexpr std::uint16_t kCold = 1u << 2;
        static constexpr std::uint16_t kInland = 1u << 3;
        static constexpr std::uint16_t kLand = 1u << 4;
        static constexpr std::uint16_t kOcean = 1u << 5;
        static constexpr std::uint16_t kWet = 1u << 6;
        static constexpr std::uint16_t kNeutralHydration = 1u << 7;
        static constexpr std::uint16_t kDry = 1u << 8;
        static constexpr std::uint16_t kBarren = 1u << 9;
        static constexpr std::uint16_t kBalanced = 1u << 10;
        static constexpr std::uint16_t kOvergrown = 1u << 11;
        static constexpr std::uint16_t kMountain = 1u << 12;
        static constexpr std::uint16_t kLowTerrain = 1u << 13;
        static constexpr std::uint16_t kAntiMountain = 1u << 14;

        std::uint16_t bits{0};

        [[nodiscard]] bool empty() const noexcept { return bits == 0; }
        [[nodiscard]] bool has(std::uint16_t mask) const noexcept { return (bits & mask) == mask; }
        [[nodiscard]] std::uint16_t value() const noexcept { return bits; }
        [[nodiscard]] bool hasAny(std::uint16_t mask) const noexcept { return (bits & mask) != 0; }
        [[nodiscard]] bool isCoastal() const noexcept { return has(kLand) && has(kOcean); }

        void add(std::uint16_t mask) noexcept { bits |= mask; }
        void fillMissingGroups() noexcept;
    };

    struct TransitionBiomeDefinition
    {
        std::string biomeId;
        float chance{1.0f};
        int width{2};
        GenerationProperties propertyMask{};
        const BiomeDefinition* biome{nullptr};
    };

    struct SubBiomeDefinition
    {
        std::string biomeId;
        float chance{0.0f};
        float minRadius{0.0f};
        float maxRadius{0.0f};
        const BiomeDefinition* biome{nullptr};

        [[nodiscard]] float sampleRadius(float defaultRadius, float noise) const noexcept
        {
            const float low = minRadius > 0.0f ? minRadius : defaultRadius * 0.25f;
            const float high = maxRadius > 0.0f ? maxRadius : defaultRadius * 0.75f;
            return std::lerp(low, high, std::clamp(noise, 0.0f, 1.0f));
        }
    };

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
    std::optional<int> minHeightLimit{};
    std::optional<int> maxHeightLimit{};
    float radius{256.0f};
    float radiusVariation{0.0f};
    float spawnChance{1.0f};
    bool fixedRadius{false};
    GenerationProperties properties{};
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
    std::vector<TransitionBiomeDefinition> transitionBiomes{};
    std::vector<SubBiomeDefinition> subBiomes{};
    float maxSubBiomeCount{0.0f};
    float subBiomeTotalChance{0.0f};

    [[nodiscard]] bool isOcean() const noexcept { return oceanFlag_; }
    [[nodiscard]] bool hasFlag(std::string_view flag) const noexcept;
    [[nodiscard]] const std::vector<std::string>& flags() const noexcept { return flags_; }
    [[nodiscard]] const GenerationProperties& generationProperties() const noexcept { return properties; }
    [[nodiscard]] float applyHeightLimits(float height, float normalizedDistance) const noexcept;
    [[nodiscard]] bool hasHeightLimits() const noexcept
    {
        return minHeightLimit.has_value() || maxHeightLimit.has_value();
    }

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
