// mob_model.cpp
// Loads a Bedrock-style subset of mob geometry JSON and bakes it into a static bind-pose mesh for BlockGame.

#include "mob_model.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>

namespace
{
struct JsonArray;
struct JsonObject;

struct JsonValue
{
    using Storage = std::variant<std::nullptr_t,
                                 bool,
                                 double,
                                 std::string,
                                 std::shared_ptr<JsonArray>,
                                 std::shared_ptr<JsonObject>>;

    Storage storage{nullptr};

    [[nodiscard]] const bool* asBool() const noexcept
    {
        return std::get_if<bool>(&storage);
    }

    [[nodiscard]] const double* asNumber() const noexcept
    {
        return std::get_if<double>(&storage);
    }

    [[nodiscard]] const std::string* asString() const noexcept
    {
        return std::get_if<std::string>(&storage);
    }

    [[nodiscard]] const JsonArray* asArray() const noexcept;
    [[nodiscard]] const JsonObject* asObject() const noexcept;
};

struct JsonArray
{
    std::vector<JsonValue> values;
};

struct JsonObjectEntry
{
    std::string key;
    JsonValue value;
};

struct JsonObject
{
    std::vector<JsonObjectEntry> entries;

    [[nodiscard]] const JsonValue* find(std::string_view key) const noexcept
    {
        for (const JsonObjectEntry& entry : entries)
        {
            if (entry.key == key)
            {
                return &entry.value;
            }
        }
        return nullptr;
    }
};

const JsonArray* JsonValue::asArray() const noexcept
{
    if (const auto* array = std::get_if<std::shared_ptr<JsonArray>>(&storage))
    {
        return array->get();
    }
    return nullptr;
}

const JsonObject* JsonValue::asObject() const noexcept
{
    if (const auto* object = std::get_if<std::shared_ptr<JsonObject>>(&storage))
    {
        return object->get();
    }
    return nullptr;
}

class JsonParser
{
public:
    explicit JsonParser(std::string_view text)
        : text_(text)
    {
    }

    [[nodiscard]] JsonValue parse()
    {
        skipWhitespace();
        JsonValue value = parseValue();
        skipWhitespace();
        if (!atEnd())
        {
            throwError("unexpected trailing characters");
        }
        return value;
    }

private:
    [[nodiscard]] JsonValue parseValue()
    {
        skipWhitespace();
        if (atEnd())
        {
            throwError("unexpected end of input");
        }

        switch (peek())
        {
        case '{':
            return JsonValue{std::make_shared<JsonObject>(parseObject())};
        case '[':
            return JsonValue{std::make_shared<JsonArray>(parseArray())};
        case '"':
            return JsonValue{parseString()};
        case 't':
            consumeKeyword("true");
            return JsonValue{true};
        case 'f':
            consumeKeyword("false");
            return JsonValue{false};
        case 'n':
            consumeKeyword("null");
            return JsonValue{nullptr};
        default:
            if (peek() == '-' || std::isdigit(static_cast<unsigned char>(peek())))
            {
                return JsonValue{parseNumber()};
            }
            throwError("unexpected token");
        }
    }

    [[nodiscard]] JsonObject parseObject()
    {
        JsonObject object;
        expect('{');
        skipWhitespace();
        if (consumeIf('}'))
        {
            return object;
        }

        while (true)
        {
            if (peek() != '"')
            {
                throwError("expected object key");
            }

            JsonObjectEntry entry;
            entry.key = parseString();
            skipWhitespace();
            expect(':');
            entry.value = parseValue();
            object.entries.push_back(std::move(entry));
            skipWhitespace();

            if (consumeIf('}'))
            {
                return object;
            }

            expect(',');
            skipWhitespace();
        }
    }

    [[nodiscard]] JsonArray parseArray()
    {
        JsonArray array;
        expect('[');
        skipWhitespace();
        if (consumeIf(']'))
        {
            return array;
        }

        while (true)
        {
            array.values.push_back(parseValue());
            skipWhitespace();
            if (consumeIf(']'))
            {
                return array;
            }

            expect(',');
            skipWhitespace();
        }
    }

    [[nodiscard]] std::string parseString()
    {
        expect('"');
        std::string result;
        while (!atEnd())
        {
            const char ch = advance();
            if (ch == '"')
            {
                return result;
            }

            if (ch != '\\')
            {
                result.push_back(ch);
                continue;
            }

            if (atEnd())
            {
                throwError("unterminated string escape");
            }

            const char escaped = advance();
            switch (escaped)
            {
            case '"': result.push_back('"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            case 'u':
            {
                if (pos_ + 4 > text_.size())
                {
                    throwError("invalid unicode escape");
                }

                unsigned codePoint = 0;
                for (int i = 0; i < 4; ++i)
                {
                    const char hex = text_[pos_ + i];
                    codePoint <<= 4;
                    if (hex >= '0' && hex <= '9')
                    {
                        codePoint |= static_cast<unsigned>(hex - '0');
                    }
                    else if (hex >= 'a' && hex <= 'f')
                    {
                        codePoint |= static_cast<unsigned>(hex - 'a' + 10);
                    }
                    else if (hex >= 'A' && hex <= 'F')
                    {
                        codePoint |= static_cast<unsigned>(hex - 'A' + 10);
                    }
                    else
                    {
                        throwError("invalid unicode escape");
                    }
                }
                pos_ += 4;
                if (codePoint <= 0x7F)
                {
                    result.push_back(static_cast<char>(codePoint));
                }
                else if (codePoint <= 0x7FF)
                {
                    result.push_back(static_cast<char>(0xC0 | ((codePoint >> 6) & 0x1F)));
                    result.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
                }
                else
                {
                    result.push_back(static_cast<char>(0xE0 | ((codePoint >> 12) & 0x0F)));
                    result.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
                    result.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
                }
                break;
            }
            default:
                throwError("unsupported string escape");
            }
        }

        throwError("unterminated string");
    }

    [[nodiscard]] double parseNumber()
    {
        const char* begin = text_.data() + pos_;
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(begin, &end);
        if (end == begin || errno == ERANGE)
        {
            throwError("invalid number");
        }

        pos_ = static_cast<std::size_t>(end - text_.data());
        return value;
    }

    void consumeKeyword(std::string_view keyword)
    {
        if (text_.substr(pos_, keyword.size()) != keyword)
        {
            throwError("unexpected keyword");
        }
        pos_ += keyword.size();
    }

    void skipWhitespace()
    {
        while (!atEnd() && std::isspace(static_cast<unsigned char>(text_[pos_])))
        {
            ++pos_;
        }
    }

    void expect(char ch)
    {
        if (atEnd() || text_[pos_] != ch)
        {
            std::ostringstream stream;
            stream << "expected '" << ch << "'";
            throwError(stream.str());
        }
        ++pos_;
    }

    [[nodiscard]] bool consumeIf(char ch)
    {
        if (!atEnd() && text_[pos_] == ch)
        {
            ++pos_;
            return true;
        }
        return false;
    }

    [[nodiscard]] char peek() const noexcept
    {
        return text_[pos_];
    }

    [[nodiscard]] char advance() noexcept
    {
        return text_[pos_++];
    }

    [[nodiscard]] bool atEnd() const noexcept
    {
        return pos_ >= text_.size();
    }

    [[noreturn]] void throwError(const std::string& message) const
    {
        std::ostringstream stream;
        stream << "json parse error at byte " << pos_ << ": " << message;
        throw std::runtime_error(stream.str());
    }

    std::string_view text_;
    std::size_t pos_{0};
};

[[nodiscard]] std::optional<float> readFloat(const JsonValue* value) noexcept
{
    if (value == nullptr)
    {
        return std::nullopt;
    }

    if (const double* number = value->asNumber())
    {
        return static_cast<float>(*number);
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<bool> readBool(const JsonValue* value) noexcept
{
    if (value == nullptr)
    {
        return std::nullopt;
    }

    if (const bool* flag = value->asBool())
    {
        return *flag;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> readString(const JsonValue* value)
{
    if (value == nullptr)
    {
        return std::nullopt;
    }

    if (const std::string* text = value->asString())
    {
        return *text;
    }
    return std::nullopt;
}

[[nodiscard]] glm::vec3 readVec3(const JsonValue* value, const char* fieldName)
{
    const JsonArray* array = value != nullptr ? value->asArray() : nullptr;
    if (array == nullptr || array->values.size() != 3)
    {
        std::ostringstream stream;
        stream << "expected a 3-number array for '" << fieldName << "'";
        throw std::runtime_error(stream.str());
    }

    glm::vec3 result(0.0f);
    for (std::size_t i = 0; i < 3; ++i)
    {
        const double* number = array->values[i].asNumber();
        if (number == nullptr)
        {
            std::ostringstream stream;
            stream << "expected numeric values in '" << fieldName << "'";
            throw std::runtime_error(stream.str());
        }
        result[static_cast<int>(i)] = static_cast<float>(*number);
    }
    return result;
}

[[nodiscard]] glm::vec2 readVec2(const JsonValue* value, const char* fieldName)
{
    const JsonArray* array = value != nullptr ? value->asArray() : nullptr;
    if (array == nullptr || array->values.size() != 2)
    {
        std::ostringstream stream;
        stream << "expected a 2-number array for '" << fieldName << "'";
        throw std::runtime_error(stream.str());
    }

    glm::vec2 result(0.0f);
    for (std::size_t i = 0; i < 2; ++i)
    {
        const double* number = array->values[i].asNumber();
        if (number == nullptr)
        {
            std::ostringstream stream;
            stream << "expected numeric values in '" << fieldName << "'";
            throw std::runtime_error(stream.str());
        }
        result[static_cast<int>(i)] = static_cast<float>(*number);
    }
    return result;
}

[[nodiscard]] glm::ivec2 readIvec2(const JsonValue* value, const char* fieldName)
{
    const glm::vec2 values = readVec2(value, fieldName);
    return glm::ivec2(static_cast<int>(std::lround(values.x)),
                      static_cast<int>(std::lround(values.y)));
}

struct BedrockCube
{
    glm::vec3 origin{0.0f};
    glm::vec3 size{0.0f};
    glm::ivec2 uv{0};
    bool mirror{false};
    float inflate{0.0f};
};

struct BedrockBone
{
    std::string name;
    int parentIndex{-1};
    glm::vec3 pivot{0.0f};
    glm::vec3 bindRotation{0.0f};
    bool mirror{false};
    std::vector<BedrockCube> cubes;
};

struct ParsedModelGeometry
{
    glm::ivec2 textureSize{64, 32};
    std::vector<BedrockBone> bones;
};

[[nodiscard]] std::string stripMobIdSuffix(std::string fileName)
{
    constexpr std::string_view kGeoSuffix = ".geo";
    if (fileName.size() > kGeoSuffix.size() &&
        fileName.compare(fileName.size() - kGeoSuffix.size(), kGeoSuffix.size(), kGeoSuffix) == 0)
    {
        fileName.resize(fileName.size() - kGeoSuffix.size());
    }
    return fileName;
}

[[nodiscard]] std::filesystem::path defaultTexturePathForModel(const std::filesystem::path& sourcePath)
{
    const std::string stem = stripMobIdSuffix(sourcePath.stem().string());
    return sourcePath.parent_path() / (stem + ".png");
}

[[nodiscard]] glm::mat4 bedrockRotationMatrix(const glm::vec3& rotationDegrees)
{
    const glm::vec3 correctedRotation = -rotationDegrees;
    glm::mat4 matrix(1.0f);
    matrix = glm::rotate(matrix, glm::radians(correctedRotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
    matrix = glm::rotate(matrix, glm::radians(correctedRotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    matrix = glm::rotate(matrix, glm::radians(correctedRotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    return matrix;
}

[[nodiscard]] ParsedModelGeometry parseBoneGeometryArray(const JsonArray& bones,
                                                         const JsonObject& textureOwner,
                                                         bool modernDescriptionNames)
{
    ParsedModelGeometry parsed;
    const char* textureWidthName = modernDescriptionNames ? "texture_width" : "texturewidth";
    const char* textureHeightName = modernDescriptionNames ? "texture_height" : "textureheight";
    if (const auto textureWidth = readFloat(textureOwner.find(textureWidthName)))
    {
        parsed.textureSize.x = static_cast<int>(std::lround(*textureWidth));
    }
    if (const auto textureHeight = readFloat(textureOwner.find(textureHeightName)))
    {
        parsed.textureSize.y = static_cast<int>(std::lround(*textureHeight));
    }

    std::unordered_map<std::string, int> boneIndexByName;
    boneIndexByName.reserve(bones.values.size());

    for (const JsonValue& boneValue : bones.values)
    {
        const JsonObject* boneObject = boneValue.asObject();
        if (boneObject == nullptr)
        {
            throw std::runtime_error("each bone must be an object");
        }

        BedrockBone bone;
        const auto boneName = readString(boneObject->find("name"));
        if (!boneName || boneName->empty())
        {
            throw std::runtime_error("each bone must have a name");
        }
        bone.name = *boneName;
        bone.pivot = readVec3(boneObject->find("pivot"), "pivot");
        if (const JsonValue* bindPose = boneObject->find("bind_pose_rotation"))
        {
            bone.bindRotation = readVec3(bindPose, "bind_pose_rotation");
        }
        else if (const JsonValue* rotation = boneObject->find("rotation"))
        {
            bone.bindRotation = readVec3(rotation, "rotation");
        }
        bone.mirror = readBool(boneObject->find("mirror")).value_or(false);

        if (const auto parentName = readString(boneObject->find("parent")))
        {
            const auto parentIt = boneIndexByName.find(*parentName);
            if (parentIt == boneIndexByName.end())
            {
                std::ostringstream stream;
                stream << "bone '" << bone.name << "' references unknown parent '" << *parentName << "'";
                throw std::runtime_error(stream.str());
            }
            bone.parentIndex = parentIt->second;
        }

        if (const JsonValue* cubesValue = boneObject->find("cubes"))
        {
            const JsonArray* cubes = cubesValue->asArray();
            if (cubes == nullptr)
            {
                throw std::runtime_error("bone cubes must be an array");
            }

            for (const JsonValue& cubeValue : cubes->values)
            {
                const JsonObject* cubeObject = cubeValue.asObject();
                if (cubeObject == nullptr)
                {
                    throw std::runtime_error("each cube must be an object");
                }

                BedrockCube cube;
                cube.origin = readVec3(cubeObject->find("origin"), "origin");
                cube.size = readVec3(cubeObject->find("size"), "size");
                cube.uv = readIvec2(cubeObject->find("uv"), "uv");
                cube.mirror = readBool(cubeObject->find("mirror")).value_or(bone.mirror);
                cube.inflate = readFloat(cubeObject->find("inflate")).value_or(0.0f);
                bone.cubes.push_back(cube);
            }
        }

        boneIndexByName.emplace(bone.name, static_cast<int>(parsed.bones.size()));
        parsed.bones.push_back(std::move(bone));
    }

    return parsed;
}

[[nodiscard]] ParsedModelGeometry parseGeometryFromRoot(const JsonObject& root)
{
    if (const JsonValue* modernGeometry = root.find("minecraft:geometry"))
    {
        const JsonArray* geometryArray = modernGeometry->asArray();
        if (geometryArray == nullptr || geometryArray->values.empty())
        {
            throw std::runtime_error("'minecraft:geometry' must contain at least one geometry");
        }

        const JsonObject* geometryObject = geometryArray->values.front().asObject();
        if (geometryObject == nullptr)
        {
            throw std::runtime_error("geometry entry must be an object");
        }

        const JsonObject* description = geometryObject->find("description") != nullptr
                                            ? geometryObject->find("description")->asObject()
                                            : nullptr;
        const JsonArray* bones = geometryObject->find("bones") != nullptr
                                     ? geometryObject->find("bones")->asArray()
                                     : nullptr;
        if (description == nullptr || bones == nullptr)
        {
            throw std::runtime_error("modern geometry must include 'description' and 'bones'");
        }
        return parseBoneGeometryArray(*bones, *description, true);
    }

    for (const JsonObjectEntry& entry : root.entries)
    {
        if (entry.key == "format_version")
        {
            continue;
        }
        if (!entry.key.starts_with("geometry."))
        {
            continue;
        }

        const JsonObject* geometryObject = entry.value.asObject();
        if (geometryObject == nullptr)
        {
            throw std::runtime_error("legacy geometry entry must be an object");
        }

        const JsonArray* bones = geometryObject->find("bones") != nullptr
                                     ? geometryObject->find("bones")->asArray()
                                     : nullptr;
        if (bones == nullptr)
        {
            throw std::runtime_error("legacy geometry must include 'bones'");
        }
        return parseBoneGeometryArray(*bones, *geometryObject, false);
    }

    throw std::runtime_error("no supported geometry entry was found");
}

struct FaceUvRect
{
    glm::vec2 min{0.0f};
    glm::vec2 max{1.0f};
};

[[nodiscard]] FaceUvRect pixelRectToUv(const glm::ivec2& textureSize,
                                       float minU,
                                       float minV,
                                       float sizeU,
                                       float sizeV)
{
    const float safeWidth = static_cast<float>(std::max(textureSize.x, 1));
    const float safeHeight = static_cast<float>(std::max(textureSize.y, 1));
    return FaceUvRect{
        glm::vec2(minU / safeWidth, minV / safeHeight),
        glm::vec2((minU + sizeU) / safeWidth, (minV + sizeV) / safeHeight),
    };
}

void appendQuad(std::vector<MobVertex>& vertices,
                std::vector<std::uint32_t>& indices,
                const std::array<glm::vec3, 4>& positions,
                const glm::vec3& normal,
                FaceUvRect uvRect,
                bool mirrorU)
{
    const std::uint32_t baseIndex = static_cast<std::uint32_t>(vertices.size());

    glm::vec2 uv0(uvRect.min.x, uvRect.max.y);
    glm::vec2 uv1(uvRect.max.x, uvRect.max.y);
    glm::vec2 uv2(uvRect.max.x, uvRect.min.y);
    glm::vec2 uv3(uvRect.min.x, uvRect.min.y);
    if (mirrorU)
    {
        std::swap(uv0.x, uv1.x);
        std::swap(uv3.x, uv2.x);
    }

    vertices.push_back(MobVertex{positions[0], normal, uv0, glm::vec4(1.0f)});
    vertices.push_back(MobVertex{positions[1], normal, uv1, glm::vec4(1.0f)});
    vertices.push_back(MobVertex{positions[2], normal, uv2, glm::vec4(1.0f)});
    vertices.push_back(MobVertex{positions[3], normal, uv3, glm::vec4(1.0f)});

    indices.push_back(baseIndex + 0);
    indices.push_back(baseIndex + 1);
    indices.push_back(baseIndex + 2);
    indices.push_back(baseIndex + 0);
    indices.push_back(baseIndex + 2);
    indices.push_back(baseIndex + 3);
}

void bakeCube(std::vector<MobVertex>& vertices,
              std::vector<std::uint32_t>& indices,
              const glm::ivec2& textureSize,
              const glm::mat4& transform,
              const BedrockCube& cube)
{
    const glm::vec3 inflate(cube.inflate);
    const glm::vec3 minCorner = cube.origin - inflate;
    const glm::vec3 maxCorner = cube.origin + cube.size + inflate;

    const std::array<glm::vec3, 8> localCorners{{
        {minCorner.x, minCorner.y, minCorner.z},
        {maxCorner.x, minCorner.y, minCorner.z},
        {maxCorner.x, maxCorner.y, minCorner.z},
        {minCorner.x, maxCorner.y, minCorner.z},
        {minCorner.x, minCorner.y, maxCorner.z},
        {maxCorner.x, minCorner.y, maxCorner.z},
        {maxCorner.x, maxCorner.y, maxCorner.z},
        {minCorner.x, maxCorner.y, maxCorner.z},
    }};

    std::array<glm::vec3, 8> worldCorners{};
    for (std::size_t i = 0; i < localCorners.size(); ++i)
    {
        worldCorners[i] = glm::vec3(transform * glm::vec4(localCorners[i], 1.0f)) / 16.0f;
    }

    const glm::mat3 normalMatrix(transform);
    const float sizeX = std::abs(cube.size.x) + cube.inflate * 2.0f;
    const float sizeY = std::abs(cube.size.y) + cube.inflate * 2.0f;
    const float sizeZ = std::abs(cube.size.z) + cube.inflate * 2.0f;
    const float baseU = static_cast<float>(cube.uv.x);
    const float baseV = static_cast<float>(cube.uv.y);

    const FaceUvRect topUv = pixelRectToUv(textureSize, baseU + sizeZ, baseV, sizeX, sizeZ);
    const FaceUvRect bottomUv = pixelRectToUv(textureSize, baseU + sizeZ + sizeX, baseV, sizeX, sizeZ);
    const FaceUvRect leftUv = pixelRectToUv(textureSize, baseU, baseV + sizeZ, sizeZ, sizeY);
    const FaceUvRect frontUv = pixelRectToUv(textureSize, baseU + sizeZ, baseV + sizeZ, sizeX, sizeY);
    const FaceUvRect rightUv = pixelRectToUv(textureSize, baseU + sizeZ + sizeX, baseV + sizeZ, sizeZ, sizeY);
    const FaceUvRect backUv = pixelRectToUv(textureSize, baseU + sizeZ + sizeX + sizeZ, baseV + sizeZ, sizeX, sizeY);

    appendQuad(vertices,
               indices,
               {worldCorners[4], worldCorners[5], worldCorners[6], worldCorners[7]},
               glm::normalize(normalMatrix * glm::vec3(0.0f, 0.0f, 1.0f)),
               backUv,
               cube.mirror);
    appendQuad(vertices,
               indices,
               {worldCorners[1], worldCorners[0], worldCorners[3], worldCorners[2]},
               glm::normalize(normalMatrix * glm::vec3(0.0f, 0.0f, -1.0f)),
               frontUv,
               cube.mirror);
    appendQuad(vertices,
               indices,
               {worldCorners[0], worldCorners[4], worldCorners[7], worldCorners[3]},
               glm::normalize(normalMatrix * glm::vec3(-1.0f, 0.0f, 0.0f)),
               leftUv,
               cube.mirror);
    appendQuad(vertices,
               indices,
               {worldCorners[5], worldCorners[1], worldCorners[2], worldCorners[6]},
               glm::normalize(normalMatrix * glm::vec3(1.0f, 0.0f, 0.0f)),
               rightUv,
               cube.mirror);
    appendQuad(vertices,
               indices,
               {worldCorners[3], worldCorners[7], worldCorners[6], worldCorners[2]},
               glm::normalize(normalMatrix * glm::vec3(0.0f, 1.0f, 0.0f)),
               topUv,
               cube.mirror);
    appendQuad(vertices,
               indices,
               {worldCorners[0], worldCorners[1], worldCorners[5], worldCorners[4]},
               glm::normalize(normalMatrix * glm::vec3(0.0f, -1.0f, 0.0f)),
               bottomUv,
               cube.mirror);
}

[[nodiscard]] MobModel bakeMobModel(const std::filesystem::path& path, const ParsedModelGeometry& parsed)
{
    MobModel model;
    model.id = stripMobIdSuffix(path.stem().string());
    model.sourcePath = path;
    model.texturePath = defaultTexturePathForModel(path);
    model.textureSize = parsed.textureSize;

    for (const BedrockBone& bone : parsed.bones)
    {
        // Bedrock geometry authors cubes and pivots in model space. For the static bind-pose bake,
        // applying parent transforms again explodes child bones away from the body; preserve only
        // this bone's own bind rotation around its authored model-space pivot.
        const glm::mat4 transform = glm::translate(glm::mat4(1.0f), bone.pivot) *
                                    bedrockRotationMatrix(bone.bindRotation) *
                                    glm::translate(glm::mat4(1.0f), -bone.pivot);
        for (const BedrockCube& cube : bone.cubes)
        {
            bakeCube(model.vertices, model.indices, model.textureSize, transform, cube);
        }
    }

    if (model.vertices.empty())
    {
        throw std::runtime_error("mob geometry baked no vertices");
    }

    model.localBoundsMin = model.vertices.front().position;
    model.localBoundsMax = model.vertices.front().position;
    for (const MobVertex& vertex : model.vertices)
    {
        model.localBoundsMin = glm::min(model.localBoundsMin, vertex.position);
        model.localBoundsMax = glm::max(model.localBoundsMax, vertex.position);
    }

    return model;
}

[[nodiscard]] MobModel loadMobModelFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("failed to open mob model file");
    }

    std::ostringstream stream;
    stream << input.rdbuf();
    const std::string text = stream.str();
    JsonParser parser(text);
    const JsonValue rootValue = parser.parse();
    const JsonObject* root = rootValue.asObject();
    if (root == nullptr)
    {
        throw std::runtime_error("mob model root must be an object");
    }

    const ParsedModelGeometry parsed = parseGeometryFromRoot(*root);
    return bakeMobModel(path, parsed);
}

} // namespace

void MobModelLibrary::clear() noexcept
{
    models_.clear();
}

bool MobModelLibrary::loadDirectory(const std::filesystem::path& directory)
{
    clear();

    std::error_code ec;
    if (!std::filesystem::exists(directory, ec) || ec)
    {
        std::cerr << "Mob model directory not found: " << directory << std::endl;
        return false;
    }

    std::vector<std::filesystem::path> files;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }
        if (entry.path().extension() != ".json")
        {
            continue;
        }
        files.push_back(entry.path());
    }

    std::sort(files.begin(), files.end());

    for (const std::filesystem::path& path : files)
    {
        try
        {
            MobModel model = loadMobModelFile(path);
            models_[model.id] = std::move(model);
        }
        catch (const std::exception& ex)
        {
            std::cerr << "Failed to load mob model '" << path.string() << "': " << ex.what() << std::endl;
        }
    }

    return !models_.empty();
}

const MobModel* MobModelLibrary::find(std::string_view id) const noexcept
{
    const auto it = models_.find(std::string(id));
    if (it == models_.end())
    {
        return nullptr;
    }
    return &it->second;
}

std::vector<const MobModel*> MobModelLibrary::all() const
{
    std::vector<const MobModel*> result;
    result.reserve(models_.size());
    for (const auto& [id, model] : models_)
    {
        (void)id;
        result.push_back(&model);
    }
    std::sort(result.begin(),
              result.end(),
              [](const MobModel* lhs, const MobModel* rhs)
              {
                  return lhs->id < rhs->id;
              });
    return result;
}

std::size_t MobModelLibrary::size() const noexcept
{
    return models_.size();
}
