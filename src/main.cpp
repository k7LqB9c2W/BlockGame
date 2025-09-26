#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{
constexpr float kNearPlane = 0.1f;
constexpr float kFarPlane = 256.0f;
constexpr float kCameraEyeHeight = 1.7f;
constexpr float kEpsilon = 1e-6f;

constexpr int kChunkSizeX = 16;
constexpr int kChunkSizeY = 64;
constexpr int kChunkSizeZ = 16;
constexpr int kChunkBlockCount = kChunkSizeX * kChunkSizeY * kChunkSizeZ;
constexpr int kViewDistance = 4; // chunks around the player

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};

class Camera
{
public:
    glm::vec3 position{0.0f, 10.0f, 5.0f};
    float yaw{-90.0f};
    float pitch{0.0f};
    float moveSpeed{8.0f};
    float mouseSensitivity{0.12f};

    const glm::vec3& front() const noexcept { return front_; }
    const glm::vec3& up() const noexcept { return up_; }
    const glm::vec3& right() const noexcept { return right_; }
    const glm::vec3& worldUp() const noexcept { return worldUp_; }

    void processMouse(float xoffset, float yoffset)
    {
        xoffset *= mouseSensitivity;
        yoffset *= mouseSensitivity;

        yaw += xoffset;
        pitch += yoffset;
        pitch = std::clamp(pitch, -89.0f, 89.0f);
        updateVectors();
    }

    void updateVectors()
    {
        const float yawRad = glm::radians(yaw);
        const float pitchRad = glm::radians(pitch);

        glm::vec3 direction;
        direction.x = std::cos(yawRad) * std::cos(pitchRad);
        direction.y = std::sin(pitchRad);
        direction.z = std::sin(yawRad) * std::cos(pitchRad);
        front_ = glm::normalize(direction);

        glm::vec3 rightCandidate = glm::cross(front_, worldUp_);
        if (glm::length(rightCandidate) < kEpsilon)
        {
            rightCandidate = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        else
        {
            rightCandidate = glm::normalize(rightCandidate);
        }
        right_ = rightCandidate;
        up_ = glm::normalize(glm::cross(right_, front_));
    }

private:
    glm::vec3 front_{0.0f, 0.0f, -1.0f};
    glm::vec3 up_{0.0f, 1.0f, 0.0f};
    glm::vec3 right_{1.0f, 0.0f, 0.0f};
    glm::vec3 worldUp_{0.0f, 1.0f, 0.0f};
};

struct InputContext
{
    Camera* camera{nullptr};
    float lastX{0.0f};
    float lastY{0.0f};
    bool firstMouse{true};
};

void processInput(GLFWwindow* window, Camera& camera, float deltaTime)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    const float velocity = camera.moveSpeed * deltaTime;

    glm::vec3 forward = camera.front();
    forward.y = 0.0f;
    if (glm::length(forward) > kEpsilon)
    {
        forward = glm::normalize(forward);
    }

    glm::vec3 right = glm::cross(forward, camera.worldUp());
    if (glm::length(right) > kEpsilon)
    {
        right = glm::normalize(right);
    }
    else
    {
        right = camera.right();
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        camera.position += forward * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        camera.position -= forward * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        camera.position -= right * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        camera.position += right * velocity;
    }
}

void framebufferSizeCallback(GLFWwindow*, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    auto* input = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
    if (input == nullptr || input->camera == nullptr)
    {
        return;
    }

    if (input->firstMouse)
    {
        input->lastX = static_cast<float>(xpos);
        input->lastY = static_cast<float>(ypos);
        input->firstMouse = false;
    }

    const float xoffset = static_cast<float>(xpos) - input->lastX;
    const float yoffset = input->lastY - static_cast<float>(ypos);

    input->lastX = static_cast<float>(xpos);
    input->lastY = static_cast<float>(ypos);

    input->camera->processMouse(xoffset, yoffset);
}

[[nodiscard]] GLuint compileShader(GLenum type, const char* source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE)
    {
        GLint logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);

        std::string infoLog;
        if (logLength > 0)
        {
            infoLog.resize(static_cast<size_t>(logLength));
            GLsizei written = 0;
            glGetShaderInfoLog(shader, logLength, &written, infoLog.data());
            infoLog.resize(static_cast<size_t>(written));
        }
        if (infoLog.empty())
        {
            infoLog = "unknown error";
        }

        glDeleteShader(shader);
        throw std::runtime_error("Shader compilation failed: " + infoLog);
    }

    return shader;
}

[[nodiscard]] GLuint createProgram(const char* vertexSrc, const char* fragmentSrc)
{
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_FALSE)
    {
        GLint logLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

        std::string infoLog;
        if (logLength > 0)
        {
            infoLog.resize(static_cast<size_t>(logLength));
            GLsizei written = 0;
            glGetProgramInfoLog(program, logLength, &written, infoLog.data());
            infoLog.resize(static_cast<size_t>(written));
        }
        if (infoLog.empty())
        {
            infoLog = "unknown error";
        }

        glDeleteProgram(program);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        throw std::runtime_error("Program linkage failed: " + infoLog);
    }

    glDetachShader(program, vertexShader);
    glDetachShader(program, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

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

enum class BlockId : std::uint8_t
{
    Air = 0,
    Grass = 1
};

inline bool isSolid(BlockId block) noexcept
{
    return block != BlockId::Air;
}

inline std::size_t blockIndex(int x, int y, int z) noexcept
{
    return static_cast<std::size_t>(y) * (kChunkSizeX * kChunkSizeZ) + static_cast<std::size_t>(z) * kChunkSizeX + static_cast<std::size_t>(x);
}

inline float hashToUnitFloat(int x, int y, int z) noexcept
{
    std::uint32_t h = static_cast<std::uint32_t>(x * 374761393 + y * 668265263 + z * 2147483647);
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
}

class PerlinNoise
{
public:
    explicit PerlinNoise(unsigned seed = 2025u)
    {
        std::array<int, 256> temp;
        std::iota(temp.begin(), temp.end(), 0);

        std::mt19937 rng(seed);
        std::shuffle(temp.begin(), temp.end(), rng);

        for (int i = 0; i < 256; ++i)
        {
            permutation_[i] = permutation_[i + 256] = temp[static_cast<std::size_t>(i)];
        }
    }

    float noise(float x, float y) const noexcept
    {
        const int xi = static_cast<int>(std::floor(x)) & 255;
        const int yi = static_cast<int>(std::floor(y)) & 255;

        const float xf = x - std::floor(x);
        const float yf = y - std::floor(y);

        const float u = fade(xf);
        const float v = fade(yf);

        const int aa = permutation_[permutation_[xi] + yi];
        const int ab = permutation_[permutation_[xi] + yi + 1];
        const int ba = permutation_[permutation_[xi + 1] + yi];
        const int bb = permutation_[permutation_[xi + 1] + yi + 1];

        const float x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1.0f, yf), u);
        const float x2 = lerp(grad(ab, xf, yf - 1.0f), grad(bb, xf - 1.0f, yf - 1.0f), u);
        return lerp(x1, x2, v);
    }

    float fbm(float x, float y, int octaves, float persistence, float lacunarity) const noexcept
    {
        float amplitude = 1.0f;
        float frequency = 1.0f;
        float sum = 0.0f;
        float maxValue = 0.0f;

        for (int i = 0; i < octaves; ++i)
        {
            sum += noise(x * frequency, y * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        if (maxValue > 0.0f)
        {
            sum /= maxValue;
        }

        return sum;
    }

    float ridge(float x, float y, int octaves, float lacunarity, float gain) const noexcept
    {
        float sum = 0.0f;
        float amplitude = 0.5f;
        float frequency = 1.0f;
        float prev = 1.0f;

        for (int i = 0; i < octaves; ++i)
        {
            float n = noise(x * frequency, y * frequency);
            n = 1.0f - std::abs(n);
            n *= n;
            sum += n * amplitude * prev;
            prev = n;
            frequency *= lacunarity;
            amplitude *= gain;
        }

        return sum;
    }

private:
    std::array<int, 512> permutation_{};

    static float fade(float t) noexcept
    {
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }

    static float lerp(float a, float b, float t) noexcept
    {
        return a + t * (b - a);
    }

    static float grad(int hash, float x, float y) noexcept
    {
        const int h = hash & 7;
        const float u = h < 4 ? x : y;
        const float v = h < 4 ? y : x;
        return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
    }
};

struct Chunk
{
    explicit Chunk(const glm::ivec2& c)
        : coord(c), blocks(kChunkBlockCount, BlockId::Air)
    {
    }

    void releaseGPU()
    {
        if (ibo != 0)
        {
            glDeleteBuffers(1, &ibo);
            ibo = 0;
        }
        if (vbo != 0)
        {
            glDeleteBuffers(1, &vbo);
            vbo = 0;
        }
        if (vao != 0)
        {
            glDeleteVertexArrays(1, &vao);
            vao = 0;
        }
    }

    glm::ivec2 coord;
    std::vector<BlockId> blocks;
    GLuint vao{0};
    GLuint vbo{0};
    GLuint ibo{0};
    GLsizei indexCount{0};
    bool meshDirty{true};
};

struct ChunkHasher
{
    std::size_t operator()(const glm::ivec2& v) const noexcept
    {
        return static_cast<std::size_t>(v.x) * 73856093u ^ static_cast<std::size_t>(v.y) * 19349663u;
    }
};

struct Frustum
{
    std::array<glm::vec4, 6> planes{};

    static Frustum fromMatrix(const glm::mat4& matrix)
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

    [[nodiscard]] bool intersectsAABB(const glm::vec3& minCorner, const glm::vec3& maxCorner) const noexcept
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
};

class ChunkManager
{
public:
    explicit ChunkManager(unsigned seed)
        : noise_(seed)
    {
    }

    ~ChunkManager()
    {
        clear();
    }

    void update(const glm::vec3& cameraPos)
    {
        const int worldX = static_cast<int>(std::floor(cameraPos.x));
        const int worldZ = static_cast<int>(std::floor(cameraPos.z));
        const glm::ivec2 centerChunk = worldToChunkCoords(worldX, worldZ);

        std::unordered_set<glm::ivec2, ChunkHasher> needed;
        needed.reserve(static_cast<std::size_t>((2 * kViewDistance + 1) * (2 * kViewDistance + 1)));

        for (int dz = -kViewDistance; dz <= kViewDistance; ++dz)
        {
            for (int dx = -kViewDistance; dx <= kViewDistance; ++dx)
            {
                needed.insert(centerChunk + glm::ivec2(dx, dz));
            }
        }

        // Create missing chunks
        for (const glm::ivec2& coord : needed)
        {
            ensureChunk(coord);
        }

        // Remove chunks outside the view distance
        std::vector<glm::ivec2> toRemove;
        toRemove.reserve(chunks_.size());
        for (const auto& [coord, chunkPtr] : chunks_)
        {
            if (needed.find(coord) == needed.end())
            {
                toRemove.push_back(coord);
            }
        }

        for (const glm::ivec2& coord : toRemove)
        {
            auto it = chunks_.find(coord);
            if (it != chunks_.end())
            {
                it->second->releaseGPU();
                chunks_.erase(it);
            }
        }

        // Rebuild meshes that are marked dirty
        for (const auto& [coord, chunkPtr] : chunks_)
        {
            if (chunkPtr->meshDirty)
            {
                buildChunkMesh(*chunkPtr);
            }
        }
    }

    void render(GLuint shaderProgram, const glm::mat4& viewProj, const glm::vec3& cameraPos, const Frustum& frustum) const
    {
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "uViewProj"), 1, GL_FALSE, glm::value_ptr(viewProj));
        glUniform3fv(glGetUniformLocation(shaderProgram, "uLightDir"), 1, glm::value_ptr(lightDirection_));
        glUniform3fv(glGetUniformLocation(shaderProgram, "uCameraPos"), 1, glm::value_ptr(cameraPos));

        for (const auto& [coord, chunkPtr] : chunks_)
        {
            if (chunkPtr->indexCount == 0)
            {
                continue;
            }

            const glm::vec3 minCorner = glm::vec3(static_cast<float>(coord.x * kChunkSizeX), 0.0f, static_cast<float>(coord.y * kChunkSizeZ));
            const glm::vec3 maxCorner = minCorner + glm::vec3(static_cast<float>(kChunkSizeX), static_cast<float>(kChunkSizeY), static_cast<float>(kChunkSizeZ));

            if (!frustum.intersectsAABB(minCorner, maxCorner))
            {
                continue;
            }

            glBindVertexArray(chunkPtr->vao);
            glDrawElements(GL_TRIANGLES, chunkPtr->indexCount, GL_UNSIGNED_INT, nullptr);
        }

        glBindVertexArray(0);
        glUseProgram(0);
    }

    [[nodiscard]] float surfaceHeight(float worldX, float worldZ) const noexcept
    {
        const int wx = static_cast<int>(std::floor(worldX));
        const int wz = static_cast<int>(std::floor(worldZ));
        const glm::ivec2 chunkCoord = worldToChunkCoords(wx, wz);
        const int localX = wrapIndex(wx, kChunkSizeX);
        const int localZ = wrapIndex(wz, kChunkSizeZ);

        const Chunk* chunk = getChunk(chunkCoord);
        if (chunk == nullptr)
        {
            return 0.0f;
        }

        for (int y = kChunkSizeY - 1; y >= 0; --y)
        {
            if (isSolid(chunk->blocks[blockIndex(localX, y, localZ)]))
            {
                return static_cast<float>(y + 1);
            }
        }
        return 0.0f;
    }

    void clear()
    {
        for (auto& [coord, chunkPtr] : chunks_)
        {
            if (chunkPtr)
            {
                chunkPtr->releaseGPU();
            }
        }
        chunks_.clear();
    }

private:
    [[nodiscard]] static glm::ivec2 worldToChunkCoords(int worldX, int worldZ) noexcept
    {
        return {floorDiv(worldX, kChunkSizeX), floorDiv(worldZ, kChunkSizeZ)};
    }

    [[nodiscard]] BlockId blockAt(const glm::ivec3& worldPos) const noexcept
    {
        if (worldPos.y < 0 || worldPos.y >= kChunkSizeY)
        {
            return BlockId::Air;
        }

        const glm::ivec2 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.z);
        const Chunk* chunk = getChunk(chunkCoord);
        if (chunk == nullptr)
        {
            return BlockId::Air;
        }

        const int localX = wrapIndex(worldPos.x, kChunkSizeX);
        const int localZ = wrapIndex(worldPos.z, kChunkSizeZ);
        return chunk->blocks[blockIndex(localX, worldPos.y, localZ)];
    }

    [[nodiscard]] Chunk* getChunk(const glm::ivec2& coord) noexcept
    {
        auto it = chunks_.find(coord);
        return (it != chunks_.end()) ? it->second.get() : nullptr;
    }

    [[nodiscard]] const Chunk* getChunk(const glm::ivec2& coord) const noexcept
    {
        auto it = chunks_.find(coord);
        return (it != chunks_.end()) ? it->second.get() : nullptr;
    }

    void ensureChunk(const glm::ivec2& coord)
    {
        if (chunks_.find(coord) != chunks_.end())
        {
            return;
        }

        auto chunk = std::make_unique<Chunk>(coord);
        generateChunkBlocks(*chunk);
        chunk->meshDirty = true;
        Chunk* raw = chunk.get();
        chunks_.emplace(coord, std::move(chunk));
        markNeighborsDirty(coord);
        raw->meshDirty = true;
    }

    void markNeighborsDirty(const glm::ivec2& coord)
    {
        static const std::array<glm::ivec2, 4> offsets{
            glm::ivec2{1, 0},
            glm::ivec2{-1, 0},
            glm::ivec2{0, 1},
            glm::ivec2{0, -1}
        };

        for (const glm::ivec2& offset : offsets)
        {
            Chunk* neighbor = getChunk(coord + offset);
            if (neighbor != nullptr)
            {
                neighbor->meshDirty = true;
            }
        }
    }

    void generateChunkBlocks(Chunk& chunk)
    {
        const int baseWorldX = chunk.coord.x * kChunkSizeX;
        const int baseWorldZ = chunk.coord.y * kChunkSizeZ;

        for (int x = 0; x < kChunkSizeX; ++x)
        {
            for (int z = 0; z < kChunkSizeZ; ++z)
            {
                const int worldX = baseWorldX + x;
                const int worldZ = baseWorldZ + z;

                const float nx = static_cast<float>(worldX) * 0.035f;
                const float nz = static_cast<float>(worldZ) * 0.035f;

                float heightNoise = noise_.fbm(nx, nz, 4, 0.5f, 2.0f);
                float ridgeNoise = noise_.ridge(nx * 0.6f, nz * 0.6f, 3, 2.0f, 0.5f);
                float combined = heightNoise * 8.0f + ridgeNoise * 6.0f;
                combined += noise_.noise(nx * 2.2f, nz * 2.2f) * 1.5f;

                float targetHeight = 14.0f + combined;
                targetHeight = std::clamp(targetHeight, 1.0f, static_cast<float>(kChunkSizeY - 2));
                const int columnHeight = std::max(1, static_cast<int>(std::round(targetHeight)));

                for (int y = 0; y < kChunkSizeY; ++y)
                {
                    chunk.blocks[blockIndex(x, y, z)] = (y <= columnHeight) ? BlockId::Grass : BlockId::Air;
                }
            }
        }
    }

    void buildChunkMesh(Chunk& chunk)
    {
        static const std::array<glm::ivec3, 6> faceDirections{
            glm::ivec3{0, 0, -1}, // Front (-Z)
            glm::ivec3{0, 0, 1},  // Back (+Z)
            glm::ivec3{-1, 0, 0}, // Left (-X)
            glm::ivec3{1, 0, 0},  // Right (+X)
            glm::ivec3{0, -1, 0}, // Bottom (-Y)
            glm::ivec3{0, 1, 0}   // Top (+Y)
        };

        static const std::array<std::array<glm::vec3, 4>, 6> faceVertices{
            // Front (-Z)
            std::array<glm::vec3, 4>{
                glm::vec3{0.0f, 0.0f, 0.0f},
                glm::vec3{0.0f, 1.0f, 0.0f},
                glm::vec3{1.0f, 1.0f, 0.0f},
                glm::vec3{1.0f, 0.0f, 0.0f}},
            // Back (+Z)
            std::array<glm::vec3, 4>{
                glm::vec3{0.0f, 0.0f, 1.0f},
                glm::vec3{1.0f, 0.0f, 1.0f},
                glm::vec3{1.0f, 1.0f, 1.0f},
                glm::vec3{0.0f, 1.0f, 1.0f}},
            // Left (-X)
            std::array<glm::vec3, 4>{
                glm::vec3{0.0f, 0.0f, 1.0f},
                glm::vec3{0.0f, 1.0f, 1.0f},
                glm::vec3{0.0f, 1.0f, 0.0f},
                glm::vec3{0.0f, 0.0f, 0.0f}},
            // Right (+X)
            std::array<glm::vec3, 4>{
                glm::vec3{1.0f, 0.0f, 1.0f},
                glm::vec3{1.0f, 0.0f, 0.0f},
                glm::vec3{1.0f, 1.0f, 0.0f},
                glm::vec3{1.0f, 1.0f, 1.0f}},
            // Bottom (-Y)
            std::array<glm::vec3, 4>{
                glm::vec3{0.0f, 0.0f, 1.0f},
                glm::vec3{0.0f, 0.0f, 0.0f},
                glm::vec3{1.0f, 0.0f, 0.0f},
                glm::vec3{1.0f, 0.0f, 1.0f}},
            // Top (+Y)
            std::array<glm::vec3, 4>{
                glm::vec3{0.0f, 1.0f, 0.0f},
                glm::vec3{0.0f, 1.0f, 1.0f},
                glm::vec3{1.0f, 1.0f, 1.0f},
                glm::vec3{1.0f, 1.0f, 0.0f}},
        };

        static const std::array<glm::vec3, 6> faceNormals{
            glm::vec3{0.0f, 0.0f, -1.0f},
            glm::vec3{0.0f, 0.0f, 1.0f},
            glm::vec3{-1.0f, 0.0f, 0.0f},
            glm::vec3{1.0f, 0.0f, 0.0f},
            glm::vec3{0.0f, -1.0f, 0.0f},
            glm::vec3{0.0f, 1.0f, 0.0f}
        };

        std::vector<Vertex> vertices;
        vertices.reserve(4096);
        std::vector<std::uint32_t> indices;
        indices.reserve(6144);

        const int baseWorldX = chunk.coord.x * kChunkSizeX;
        const int baseWorldZ = chunk.coord.y * kChunkSizeZ;

        for (int x = 0; x < kChunkSizeX; ++x)
        {
            for (int y = 0; y < kChunkSizeY; ++y)
            {
                for (int z = 0; z < kChunkSizeZ; ++z)
                {
                    const BlockId block = chunk.blocks[blockIndex(x, y, z)];
                    if (!isSolid(block))
                    {
                        continue;
                    }

                    const glm::ivec3 worldPos{baseWorldX + x, y, baseWorldZ + z};

                    for (int face = 0; face < 6; ++face)
                    {
                        const glm::ivec3 neighborPos = worldPos + faceDirections[static_cast<std::size_t>(face)];
                        if (isSolid(blockAt(neighborPos)))
                        {
                            continue;
                        }

                        const glm::vec3 normal = faceNormals[static_cast<std::size_t>(face)];
                        glm::vec3 color = sideColor_;
                        if (normal.y > 0.5f)
                        {
                            color = topColor_;
                        }
                        else if (normal.y < -0.5f)
                        {
                            color = bottomColor_;
                        }
                        const float tint = hashToUnitFloat(worldPos.x, worldPos.y, worldPos.z) * 0.12f - 0.06f;
                        color += glm::vec3(tint);
                        color = glm::clamp(color, 0.0f, 1.0f);

                        const glm::vec3 blockOrigin(static_cast<float>(worldPos.x), static_cast<float>(worldPos.y), static_cast<float>(worldPos.z));
                        const std::size_t vertexStart = vertices.size();

                        for (int i = 0; i < 4; ++i)
                        {
                            Vertex vertex{};
                            vertex.position = blockOrigin + faceVertices[static_cast<std::size_t>(face)][static_cast<std::size_t>(i)];
                            vertex.normal = normal;
                            vertex.color = color;
                            vertices.push_back(vertex);
                        }

                        indices.push_back(static_cast<std::uint32_t>(vertexStart + 0));
                        indices.push_back(static_cast<std::uint32_t>(vertexStart + 1));
                        indices.push_back(static_cast<std::uint32_t>(vertexStart + 2));
                        indices.push_back(static_cast<std::uint32_t>(vertexStart + 2));
                        indices.push_back(static_cast<std::uint32_t>(vertexStart + 3));
                        indices.push_back(static_cast<std::uint32_t>(vertexStart + 0));
                    }
                }
            }
        }

        if (chunk.vao == 0)
        {
            glGenVertexArrays(1, &chunk.vao);
            glGenBuffers(1, &chunk.vbo);
            glGenBuffers(1, &chunk.ibo);

            glBindVertexArray(chunk.vao);
            glBindBuffer(GL_ARRAY_BUFFER, chunk.vbo);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, position)));
            glEnableVertexAttribArray(0);

            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));
            glEnableVertexAttribArray(1);

            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, color)));
            glEnableVertexAttribArray(2);
        }

        glBindVertexArray(chunk.vao);
        glBindBuffer(GL_ARRAY_BUFFER, chunk.vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)), vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, chunk.ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size() * sizeof(std::uint32_t)), indices.data(), GL_STATIC_DRAW);

        chunk.indexCount = static_cast<GLsizei>(indices.size());
        chunk.meshDirty = false;

        glBindVertexArray(0);
    }

    PerlinNoise noise_;
    std::unordered_map<glm::ivec2, std::unique_ptr<Chunk>, ChunkHasher> chunks_;
    const glm::vec3 lightDirection_{glm::normalize(glm::vec3(0.5f, -1.0f, 0.2f))};
    const glm::vec3 topColor_{0.26f, 0.72f, 0.32f};
    const glm::vec3 sideColor_{0.29f, 0.48f, 0.24f};
    const glm::vec3 bottomColor_{0.20f, 0.18f, 0.12f};
};

} // namespace

int main()
{
    if (glfwInit() != GLFW_TRUE)
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

    constexpr int kInitialWidth = 1280;
    constexpr int kInitialHeight = 720;

    GLFWwindow* window = glfwCreateWindow(kInitialWidth, kInitialHeight, "BlockGame", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    Camera camera;
    camera.updateVectors();

    InputContext inputContext;
    inputContext.camera = &camera;

    int windowWidth = 0;
    int windowHeight = 0;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    inputContext.lastX = static_cast<float>(windowWidth) * 0.5f;
    inputContext.lastY = static_cast<float>(windowHeight) * 0.5f;

    glfwSetWindowUserPointer(window, &inputContext);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    const char* vertexShaderSrc = R"(#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

uniform mat4 uViewProj;

out vec3 vNormal;
out vec3 vWorldPos;
out vec3 vColor;

void main()
{
    vNormal = aNormal;
    vWorldPos = aPos;
    vColor = aColor;
    gl_Position = uViewProj * vec4(aPos, 1.0);
}
)";

    const char* fragmentShaderSrc = R"(#version 330 core
out vec4 FragColor;

in vec3 vNormal;
in vec3 vWorldPos;
in vec3 vColor;

uniform vec3 uLightDir;
uniform vec3 uCameraPos;

void main()
{
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(-uLightDir);
    vec3 viewDir = normalize(uCameraPos - vWorldPos);
    float diff = max(dot(normal, lightDir), 0.0);
    float ambient = 0.35;
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);
    vec3 color = vColor * (ambient + diff) + vec3(0.1f) * spec;
    FragColor = vec4(color, 1.0);
}
)";

    GLuint shaderProgram = 0;
    try
    {
        shaderProgram = createProgram(vertexShaderSrc, fragmentShaderSrc);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Shader compilation failed: " << ex.what() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    ChunkManager chunkManager(1337u);
    chunkManager.update(camera.position);
    camera.position.y = chunkManager.surfaceHeight(camera.position.x, camera.position.z) + kCameraEyeHeight;

    float lastFrame = static_cast<float>(glfwGetTime());
    std::cout << "Controls: WASD to move, mouse to look, ESC to quit." << std::endl;

    while (!glfwWindowShouldClose(window))
    {
        const float currentFrame = static_cast<float>(glfwGetTime());
        const float deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window, camera, deltaTime);

        chunkManager.update(camera.position);
        camera.position.y = chunkManager.surfaceHeight(camera.position.x, camera.position.z) + kCameraEyeHeight;

        glClearColor(0.55f, 0.78f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
        framebufferWidth = std::max(framebufferWidth, 1);
        framebufferHeight = std::max(framebufferHeight, 1);
        const float aspect = static_cast<float>(framebufferWidth) / static_cast<float>(framebufferHeight);

        const glm::mat4 projection = glm::perspective(glm::radians(60.0f), aspect, kNearPlane, kFarPlane);
        const glm::mat4 view = glm::lookAt(camera.position, camera.position + camera.front(), camera.up());
        const glm::mat4 viewProj = projection * view;
        const Frustum frustum = Frustum::fromMatrix(viewProj);

        chunkManager.render(shaderProgram, viewProj, camera.position, frustum);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    chunkManager.clear();
    glDeleteProgram(shaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}




