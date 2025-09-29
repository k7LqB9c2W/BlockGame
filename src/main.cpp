#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>
#include <map>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace
{
constexpr float kNearPlane = 0.1f;
constexpr float kDefaultFarPlane = 256.0f;
constexpr float kFarPlanePadding = 96.0f;
constexpr float kCameraEyeHeight = 1.7f;
constexpr float kEpsilon = 1e-6f;
constexpr float kMaxRayDistance = 8.0f; // Maximum reach distance for block targeting

// Player physics constants
constexpr float kPlayerWidth = 0.6f;     // Player bounding box width
constexpr float kPlayerHeight = 1.8f;    // Player bounding box height
constexpr float kGravity = -20.0f;       // Gravity acceleration (negative = downward)
constexpr float kJumpVelocity = 8.0f;    // Initial jump velocity
constexpr float kTerminalVelocity = -50.0f; // Maximum fall speed
constexpr float kHorizontalDamping = 0.80f;     // Velocity damping when no input is held
constexpr float kGroundSnapTolerance = 1e-3f;   // Threshold for snapping the player to the ground
constexpr float kAxisCollisionEpsilon = 1e-4f;  // Padding used during axis sweeps

constexpr int kChunkSizeX = 16;
constexpr int kChunkSizeY = 64;
constexpr int kChunkSizeZ = 16;
constexpr int kChunkBlockCount = kChunkSizeX * kChunkSizeY * kChunkSizeZ;
constexpr int kAtlasTileSizePixels = 16;
constexpr int kDefaultViewDistance = 4;  // Default chunks around the player
constexpr int kExtendedViewDistance = 12; // Extended view distance for N toggle
constexpr int kMaxChunkJobsPerFrame = 12;
constexpr int kMaxRingsPerFrame = 1;
constexpr std::size_t kUploadBudgetBytesPerFrame = 4ull * 1024ull * 1024ull;
constexpr std::size_t kMinBufferSizeBytes = 4ull * 1024ull;

inline float computeFarPlaneForViewDistance(int viewDistance) noexcept
{
    const float horizontalSpan = static_cast<float>(viewDistance + 1) * static_cast<float>(std::max(kChunkSizeX, kChunkSizeZ));
    const float diagonal = std::sqrt(2.0f) * horizontalSpan;
    return std::max(diagonal + kFarPlanePadding, kDefaultFarPlane);
}

float kFarPlane = computeFarPlaneForViewDistance(kDefaultViewDistance);

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 tileCoord;
    glm::vec2 atlasBase;
    glm::vec2 atlasSize;
};

class Camera
{
public:
    glm::vec3 position{0.0f, 10.0f, 5.0f};
    float yaw{-90.0f};
    float pitch{0.0f};
    float moveSpeed{8.0f};
    float mouseSensitivity{0.12f};
    
    // Physics properties
    glm::vec3 velocity{0.0f, 0.0f, 0.0f};
    bool onGround{false};

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
    bool leftMousePressed{false};
    bool leftMouseJustPressed{false};
    bool rightMousePressed{false};
    bool rightMouseJustPressed{false};
    bool nKeyPressed{false};
    bool nKeyJustPressed{false};
    bool f1Pressed{false};
    bool f1JustPressed{false};
    bool showCoordinates{false};
    bool showRenderDistanceGUI{false};
    std::string inputBuffer{};
};

// Forward declaration
class ChunkManager;

struct RaycastHit
{
    bool hit{false};
    glm::ivec3 blockPos{0};
    glm::ivec3 faceNormal{0};
    float distance{0.0f};
};

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

void charCallback(GLFWwindow* window, unsigned int codepoint)
{
    auto* input = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
    if (input == nullptr)
    {
        return;
    }

    // Only accept input when GUI is active
    if (input->showRenderDistanceGUI && codepoint < 128)
    {
        // Only accept digits
        if (codepoint >= '0' && codepoint <= '9')
        {
            // Limit input length to prevent overflow
            if (input->inputBuffer.size() < 10)
            {
                input->inputBuffer += static_cast<char>(codepoint);
            }
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    auto* input = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
    if (input == nullptr)
    {
        return;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        bool wasPressed = input->leftMousePressed;
        input->leftMousePressed = (action == GLFW_PRESS);
        input->leftMouseJustPressed = input->leftMousePressed && !wasPressed;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        bool wasPressed = input->rightMousePressed;
        input->rightMousePressed = (action == GLFW_PRESS);
        input->rightMouseJustPressed = input->rightMousePressed && !wasPressed;
    }
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

struct LoadedTexture
{
    GLuint id{0};
    glm::ivec2 size{0};
};

[[nodiscard]] LoadedTexture loadTexture(const char* path)
{
    LoadedTexture texture{};
    glGenTextures(1, &texture.id);

    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_set_flip_vertically_on_load(false);
    unsigned char* data = stbi_load(path, &width, &height, &channels, 0);

    if (!data)
    {
        std::cerr << "Failed to load texture: " << path << std::endl;
        glDeleteTextures(1, &texture.id);
        texture.id = 0;
        texture.size = glm::ivec2(0);
        return texture;
    }

    GLenum format = GL_RGB;
    if (channels == 1)
    {
        format = GL_RED;
    }
    else if (channels == 3)
    {
        format = GL_RGB;
    }
    else if (channels == 4)
    {
        format = GL_RGBA;
    }

    glBindTexture(GL_TEXTURE_2D, texture.id);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 0);

    texture.size = glm::ivec2(width, height);
    std::cout << "Loaded texture: " << path << " (" << texture.size.x << "x" << texture.size.y << ")" << std::endl;

    return texture;
}

class Crosshair
{
public:
    Crosshair()
    {
        setupCrosshair();
    }

    ~Crosshair()
    {
        cleanup();
    }

    void render(int screenWidth, int screenHeight)
    {
        glDisable(GL_DEPTH_TEST);
        glUseProgram(shaderProgram_);
        
        if (screenSizeLocation_ >= 0)
        {
            glUniform2f(screenSizeLocation_,
                       static_cast<float>(screenWidth), static_cast<float>(screenHeight));
        }
        
        glBindVertexArray(vao_);
        glDrawArrays(GL_LINES, 0, 4);
        glBindVertexArray(0);
        
        glUseProgram(0);
        glEnable(GL_DEPTH_TEST);
    }

private:
    GLuint vao_{0};
    GLuint vbo_{0};
    GLuint shaderProgram_{0};
    GLint screenSizeLocation_{-1};

    void setupCrosshair()
    {
        // Crosshair vertices (two lines in normalized device coordinates)
        float crosshairSize = 0.02f;
        float vertices[] = {
            // Horizontal line
            -crosshairSize, 0.0f,
             crosshairSize, 0.0f,
            // Vertical line
             0.0f, -crosshairSize,
             0.0f,  crosshairSize
        };

        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
        
        glBindVertexArray(0);

        // Create crosshair shader
        const char* crosshairVertexShader = R"(#version 330 core
layout (location = 0) in vec2 aPos;

void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

        const char* crosshairFragmentShader = R"(#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 1.0, 1.0, 0.8);
}
)";

        try
        {
            shaderProgram_ = createProgram(crosshairVertexShader, crosshairFragmentShader);
        }
        catch (const std::exception& ex)
        {
            std::cerr << "Failed to create crosshair shader: " << ex.what() << std::endl;
        }

        if (shaderProgram_ != 0)
        {
            screenSizeLocation_ = glGetUniformLocation(shaderProgram_, "uScreenSize");
        }
        else
        {
            screenSizeLocation_ = -1;
        }
    }

    void cleanup()
    {
        if (vao_ != 0)
        {
            glDeleteVertexArrays(1, &vao_);
            vao_ = 0;
        }
        if (vbo_ != 0)
        {
            glDeleteBuffers(1, &vbo_);
            vbo_ = 0;
        }
        if (shaderProgram_ != 0)
        {
            glDeleteProgram(shaderProgram_);
            shaderProgram_ = 0;
        }
        screenSizeLocation_ = -1;
    }
};

#include "text_overlay.inl"

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
    Grass = 1,
    Log = 2,
    Count
};

constexpr std::size_t toIndex(BlockId block) noexcept
{
    return static_cast<std::size_t>(block);
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

constexpr std::size_t kBlockCount = toIndex(BlockId::Count);
constexpr std::size_t kBlockFaceCount = toIndex(BlockFace::Count);

enum class ChunkState : std::uint8_t
{
    Empty = 0,        // Not started
    Generating,       // Block generation in progress
    Meshing,         // Mesh building in progress
    Ready,           // Mesh ready for GPU upload
    Uploaded,        // Uploaded to GPU and ready to render
    Remeshing        // Currently remeshing but keep old mesh visible
};

enum class JobType : std::uint8_t
{
    Generate = 0,
    Mesh = 1
};

struct Chunk;

struct Job
{
    JobType type;
    glm::ivec2 chunkCoord;
    std::shared_ptr<Chunk> chunk;
    
    Job(JobType t, const glm::ivec2& coord, std::shared_ptr<Chunk> c)
        : type(t), chunkCoord(coord), chunk(std::move(c)) {}
};

class JobQueue
{
public:
    void push(const Job& job)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        priorityQueue_.push(wrap(job));
        condition_.notify_one();
    }

    bool tryPop(Job& job)
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

    Job waitAndPop()
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

    void stop()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shouldStop_.store(true, std::memory_order_release);
        condition_.notify_all();
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return priorityQueue_.empty();
    }

    void updatePriorityOrigin(const glm::ivec2& origin)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (origin == priorityOrigin_)
        {
            return;
        }

        priorityOrigin_ = origin;
        rebuildLocked();
    }

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
        bool operator()(const PrioritizedJob& lhs, const PrioritizedJob& rhs) const
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
    };

    PrioritizedJob wrap(const Job& job)
    {
        const int distance = manhattanDistance(job.chunkCoord, priorityOrigin_);
        const int bias = (job.type == JobType::Mesh) ? 0 : 1;
        const std::uint64_t sequence = nextSequence_++;
        return PrioritizedJob{job, distance, bias, sequence};
    }

    static int manhattanDistance(const glm::ivec2& a, const glm::ivec2& b) noexcept
    {
        return std::abs(a.x - b.x) + std::abs(a.y - b.y);
    }

    void rebuildLocked()
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

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> shouldStop_{false};
    glm::ivec2 priorityOrigin_{0, 0};
    std::priority_queue<PrioritizedJob, std::vector<PrioritizedJob>, JobComparer> priorityQueue_;
    std::uint64_t nextSequence_{0};
};


inline bool isSolid(BlockId block) noexcept
{
    return block != BlockId::Air;
}

inline std::size_t blockIndex(int x, int y, int z) noexcept
{
    return static_cast<std::size_t>(y) * (kChunkSizeX * kChunkSizeZ) + static_cast<std::size_t>(z) * kChunkSizeX + static_cast<std::size_t>(x);
}

inline std::size_t columnIndex(int x, int z) noexcept
{
    return static_cast<std::size_t>(z) * kChunkSizeX + static_cast<std::size_t>(x);
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

struct Chunk
{
    explicit Chunk(const glm::ivec2& c)
        : coord(c), blocks(kChunkBlockCount, BlockId::Air), state(ChunkState::Empty), columnMaxHeights(kChunkSizeX * kChunkSizeZ, -1)
    {
    }

    glm::ivec2 coord;
    std::vector<BlockId> blocks;
    std::atomic<ChunkState> state;
    std::vector<int> columnMaxHeights; // Highest solid block index per (x, z) column
    
    // GPU resources (main thread only)
    GLuint vao{0};
    GLuint vbo{0};
    GLuint ibo{0};
    GLsizei indexCount{0};
    std::size_t vertexCapacity{0};
    std::size_t indexCapacity{0};
    bool queuedForUpload{false};
    
    // Mesh data for async operations
    mutable std::mutex meshMutex;
    MeshData meshData;
    bool meshReady{false};
    std::atomic<int> inFlight{0};
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

struct ChunkShaderUniformLocations
{
    GLint uViewProj{-1};
    GLint uLightDir{-1};
    GLint uCameraPos{-1};
    GLint uAtlas{-1};
    GLint uHighlightedBlock{-1};
    GLint uHasHighlight{-1};
};

class ChunkManager
{
public:
    explicit ChunkManager(unsigned seed)
        : noise_(seed), shouldStop_(false), viewDistance_(kDefaultViewDistance), targetViewDistance_(kDefaultViewDistance)
    {
        kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);
        startWorkerThreads();
    }

    ~ChunkManager()
    {
        stopWorkerThreads();
        clear();
        destroyBufferPool();
    }

    void setAtlasTexture(GLuint texture) noexcept
    {
        atlasTexture_ = texture;
    }

    void setBlockTextureAtlasConfig(const glm::ivec2& textureSizePixels, int tileSizePixels)
    {
        if (tileSizePixels <= 0 || textureSizePixels.x <= 0 || textureSizePixels.y <= 0)
        {
            std::cerr << "Invalid block atlas dimensions provided" << std::endl;
            blockAtlasConfigured_ = false;
            return;
        }

        atlasTileScale_ = glm::vec2(
            static_cast<float>(tileSizePixels) / static_cast<float>(textureSizePixels.x),
            static_cast<float>(tileSizePixels) / static_cast<float>(textureSizePixels.y));

        for (auto& blockEntry : blockUVTable_)
        {
            for (auto& face : blockEntry.faces)
            {
                face.base = glm::vec2(0.0f);
                face.size = atlasTileScale_;
            }
        }

        auto assignFace = [&](BlockId block, BlockFace face, const glm::ivec2& tile)
        {
            const glm::vec2 base = glm::vec2(static_cast<float>(tile.x), static_cast<float>(tile.y)) * atlasTileScale_;
            auto& faceUV = blockUVTable_[toIndex(block)].faces[toIndex(face)];
            faceUV.base = base;
            faceUV.size = atlasTileScale_;
        };

        // Grass: tile row order follows OpenGL's bottom-left origin, row 0 is the bottom of the atlas.
        assignFace(BlockId::Grass, BlockFace::Top, {0, 0});
        assignFace(BlockId::Grass, BlockFace::Bottom, {0, 2});
        for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
        {
            assignFace(BlockId::Grass, face, {0, 1});
        }

        // Log block example: bark on row 3, caps on row 4. Update these indices to match your atlas.
        assignFace(BlockId::Log, BlockFace::Top, {0, 4});
        assignFace(BlockId::Log, BlockFace::Bottom, {0, 4});
        for (BlockFace face : {BlockFace::North, BlockFace::South, BlockFace::East, BlockFace::West})
        {
            assignFace(BlockId::Log, face, {0, 3});
        }

        // Add new block types here by mapping each face to a tile coordinate.
        blockAtlasConfigured_ = true;
    }

    void update(const glm::vec3& cameraPos)
    {
        const int worldX = static_cast<int>(std::floor(cameraPos.x));
        const int worldZ = static_cast<int>(std::floor(cameraPos.z));
        const glm::ivec2 centerChunk = worldToChunkCoords(worldX, worldZ);

        jobQueue_.updatePriorityOrigin(centerChunk);

        if (viewDistance_ > targetViewDistance_)
        {
            viewDistance_ = targetViewDistance_;
        }

        int jobBudget = kMaxChunkJobsPerFrame;

        for (int ring = 0; ring <= viewDistance_ && jobBudget > 0; ++ring)
        {
            RingProgress progress = ensureRing(centerChunk, ring, jobBudget);
            if (progress.budgetExhausted)
            {
                break;
            }
        }

        int ringsExpanded = 0;
        while (jobBudget > 0 && viewDistance_ < targetViewDistance_ && ringsExpanded < kMaxRingsPerFrame)
        {
            const int nextRing = viewDistance_ + 1;
            RingProgress progress = ensureRing(centerChunk, nextRing, jobBudget);

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

        removeDistantChunks(centerChunk);

        uploadReadyMeshes();
    }

    void render(GLuint shaderProgram, const glm::mat4& viewProj, const glm::vec3& cameraPos, const Frustum& frustum, const ChunkShaderUniformLocations& uniforms) const
    {
        glUseProgram(shaderProgram);
        if (uniforms.uViewProj >= 0)
        {
            glUniformMatrix4fv(uniforms.uViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
        }
        if (uniforms.uLightDir >= 0)
        {
            glUniform3fv(uniforms.uLightDir, 1, glm::value_ptr(lightDirection_));
        }
        if (uniforms.uCameraPos >= 0)
        {
            glUniform3fv(uniforms.uCameraPos, 1, glm::value_ptr(cameraPos));
        }
        
        // Pass highlighted block position to shader
        if (atlasTexture_ != 0)
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, atlasTexture_);
            if (uniforms.uAtlas >= 0)
            {
                glUniform1i(uniforms.uAtlas, 0);
            }
        }

        if (uniforms.uHighlightedBlock >= 0)
        {
            glUniform3f(uniforms.uHighlightedBlock,
                        static_cast<float>(highlightedBlock_.x),
                        static_cast<float>(highlightedBlock_.y),
                        static_cast<float>(highlightedBlock_.z));
        }
        if (uniforms.uHasHighlight >= 0)
        {
            glUniform1i(uniforms.uHasHighlight, hasHighlight_ ? 1 : 0);
        }

        std::vector<std::pair<glm::ivec2, std::shared_ptr<Chunk>>> snapshot;
        {
            std::lock_guard<std::mutex> lock(chunksMutex);
            snapshot.reserve(chunks_.size());
            for (const auto& entry : chunks_)
            {
                snapshot.push_back(entry);
            }
        }

        for (const auto& [coord, chunkPtr] : snapshot)
        {
            if (!chunkPtr)
            {
                continue;
            }

            ChunkState state = chunkPtr->state.load();
            if ((state != ChunkState::Uploaded && state != ChunkState::Remeshing) || chunkPtr->indexCount == 0)
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
        if (atlasTexture_ != 0)
        {
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glUseProgram(0);
    }

    [[nodiscard]] float surfaceHeight(float worldX, float worldZ) const noexcept
    {
        const int wx = static_cast<int>(std::floor(worldX));
        const int wz = static_cast<int>(std::floor(worldZ));
        const glm::ivec2 chunkCoord = worldToChunkCoords(wx, wz);
        const int localX = wrapIndex(wx, kChunkSizeX);
        const int localZ = wrapIndex(wz, kChunkSizeZ);

        auto chunk = getChunkShared(chunkCoord);
        if (!chunk)
        {
            return 0.0f;
        }

        int topBlock = -1;
        {
            std::lock_guard<std::mutex> lock(chunk->meshMutex);
            if (!chunk->columnMaxHeights.empty())
            {
                topBlock = chunk->columnMaxHeights[columnIndex(localX, localZ)];
            }
        }

        if (topBlock < 0)
        {
            return 0.0f;
        }

        return static_cast<float>(topBlock + 1);
    }

    void clear()
    {
        while (true)
        {
            std::vector<glm::ivec2> coords;
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
            for (const glm::ivec2& coord : coords)
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
                    recycleChunkGPU(*chunk);
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
    }

    bool destroyBlock(const glm::ivec3& worldPos)
    {
        if (worldPos.y < 0 || worldPos.y >= kChunkSizeY)
        {
            return false;
        }

        const glm::ivec2 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.z);
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

        const int localX = wrapIndex(worldPos.x, kChunkSizeX);
        const int localZ = wrapIndex(worldPos.z, kChunkSizeZ);
        const std::size_t blockIdx = blockIndex(localX, worldPos.y, localZ);

        {
            std::lock_guard<std::mutex> lock(chunk->meshMutex);
            if (!isSolid(chunk->blocks[blockIdx]))
            {
                return false;
            }

            chunk->blocks[blockIdx] = BlockId::Air;
            recomputeColumnHeight(*chunk, localX, localZ);
            chunk->state = ChunkState::Remeshing;
        }

        enqueueJob(chunk, JobType::Mesh, chunkCoord);

        // Only remesh neighbors if the destroyed block is on a chunk boundary
        markNeighborsForRemeshingIfNeeded(chunkCoord, localX, localZ);
        return true;
    }

    bool placeBlock(const glm::ivec3& targetBlockPos, const glm::ivec3& faceNormal)
    {
        // Calculate placement position by moving away from the target block face
        const glm::ivec3 placePos = targetBlockPos + faceNormal;
        
        // Check if placement position is valid
        if (placePos.y < 0 || placePos.y >= kChunkSizeY)
        {
            return false;
        }

        const glm::ivec2 chunkCoord = worldToChunkCoords(placePos.x, placePos.z);
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

        const int localX = wrapIndex(placePos.x, kChunkSizeX);
        const int localZ = wrapIndex(placePos.z, kChunkSizeZ);
        const std::size_t blockIdx = blockIndex(localX, placePos.y, localZ);

        {
            std::lock_guard<std::mutex> lock(chunk->meshMutex);
            // Don't place if there's already a block there
            if (isSolid(chunk->blocks[blockIdx]))
            {
                return false;
            }

            chunk->blocks[blockIdx] = BlockId::Grass;
            recomputeColumnHeight(*chunk, localX, localZ);
            chunk->state = ChunkState::Remeshing;
        }

        enqueueJob(chunk, JobType::Mesh, chunkCoord);

        // Only remesh neighbors if the placed block is on a chunk boundary
        markNeighborsForRemeshingIfNeeded(chunkCoord, localX, localZ);
        return true;
    }

    RaycastHit raycast(const glm::vec3& origin, const glm::vec3& direction) const
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

    void updateHighlight(const glm::vec3& cameraPos, const glm::vec3& cameraDirection)
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

    void toggleViewDistance()
    {
        try
        {
            if (targetViewDistance_ == kDefaultViewDistance)
            {
                std::cout << "Switching to extended render distance..." << std::endl;

                targetViewDistance_ = kExtendedViewDistance;
                kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);
                std::cout << "Extended render distance target: " << targetViewDistance_ << " chunks (total: "
                          << (2 * targetViewDistance_ + 1) * (2 * targetViewDistance_ + 1) << " chunks)" << std::endl;
            }
            else
            {
                std::cout << "Switching to default render distance..." << std::endl;

                targetViewDistance_ = kDefaultViewDistance;
                kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);
                std::cout << "Default render distance target: " << targetViewDistance_ << " chunks" << std::endl;
            }

            if (viewDistance_ > targetViewDistance_)
            {
                viewDistance_ = targetViewDistance_;
            }
        }
        catch (const std::exception& ex)
        {
            std::cerr << "Error toggling view distance: " << ex.what() << std::endl;
            targetViewDistance_ = kDefaultViewDistance;
            viewDistance_ = std::min(viewDistance_, targetViewDistance_);
            kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);
        }
    }

    [[nodiscard]] int viewDistance() const noexcept
    {
        return targetViewDistance_;
    }

    void setRenderDistance(int distance) noexcept
    {
        try
        {
            // Clamp distance to reasonable values (minimum 1, maximum 200 for safety)
            const int clampedDistance = std::max(1, std::min(distance, 200));
            targetViewDistance_ = clampedDistance;
            kFarPlane = computeFarPlaneForViewDistance(targetViewDistance_);
            
            if (viewDistance_ > targetViewDistance_)
            {
                viewDistance_ = targetViewDistance_;
            }
            
            std::cout << "Render distance set to: " << targetViewDistance_ << " chunks (total: "
                      << (2 * targetViewDistance_ + 1) * (2 * targetViewDistance_ + 1) << " chunks)" << std::endl;
        }
        catch (const std::exception& ex)
        {
            std::cerr << "Error setting render distance: " << ex.what() << std::endl;
        }
    }

    [[nodiscard]] BlockId blockAt(const glm::ivec3& worldPos) const noexcept
    {
        if (worldPos.y < 0 || worldPos.y >= kChunkSizeY)
        {
            return BlockId::Air;
        }

        const glm::ivec2 chunkCoord = worldToChunkCoords(worldPos.x, worldPos.z);
        auto chunk = getChunkShared(chunkCoord);
        if (!chunk)
        {
            return BlockId::Air;
        }

        const int localX = wrapIndex(worldPos.x, kChunkSizeX);
        const int localZ = wrapIndex(worldPos.z, kChunkSizeZ);
        return chunk->blocks[blockIndex(localX, worldPos.y, localZ)];
    }

    glm::vec3 findSafeSpawnPosition(float worldX, float worldZ) const
    {
        const float halfWidth = kPlayerWidth * 0.5f;
        
        // Start from the top of the world and work down
        for (int y = kChunkSizeY - 3; y >= 2; --y)
        {
            // Check if there's solid ground at this level
            bool hasGround = false;
            for (int dx = -1; dx <= 1; ++dx)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    if (isSolid(blockAt(glm::ivec3(
                        static_cast<int>(std::floor(worldX + dx * halfWidth)),
                        y - 1, 
                        static_cast<int>(std::floor(worldZ + dz * halfWidth))))))
                    {
                        hasGround = true;
                        break;
                    }
                }
                if (hasGround) break;
            }
            
            if (!hasGround) continue;
            
            // Check if there's enough vertical clearance for the player
            bool canFit = true;
            const int clearanceHeight = static_cast<int>(std::ceil(kPlayerHeight)) + 1;
            
            for (int checkY = y; checkY < y + clearanceHeight && checkY < kChunkSizeY; ++checkY)
            {
                // Check all blocks in the player's footprint
                for (int dx = -1; dx <= 1; ++dx)
                {
                    for (int dz = -1; dz <= 1; ++dz)
                    {
                        const float checkX = worldX + dx * halfWidth;
                        const float checkZ = worldZ + dz * halfWidth;
                        
                        if (isSolid(blockAt(glm::ivec3(
                            static_cast<int>(std::floor(checkX)),
                            checkY,
                            static_cast<int>(std::floor(checkZ))))))
                        {
                            canFit = false;
                            break;
                        }
                    }
                    if (!canFit) break;
                }
                if (!canFit) break;
            }
            
            if (canFit)
            {
                // Found a safe position - return camera position (eye level)
                const float safeY = static_cast<float>(y) + kCameraEyeHeight;
                std::cout << "Safe spawn found at height: " << safeY << " (feet at: " << y << ")" << std::endl;
                return glm::vec3(worldX, safeY, worldZ);
            }
        }
        
        // Fallback: spawn high above the world if no safe spot found
        std::cout << "Warning: No safe spawn found, spawning above terrain" << std::endl;
        const float fallbackY = static_cast<float>(kChunkSizeY - 5) + kCameraEyeHeight;
        return glm::vec3(worldX, fallbackY, worldZ);
    }

private:
    void startWorkerThreads()
    {
        const unsigned numThreads = std::max(1u, std::thread::hardware_concurrency() / 2);
        workerThreads_.reserve(numThreads);
        
        for (unsigned i = 0; i < numThreads; ++i)
        {
            workerThreads_.emplace_back(&ChunkManager::workerThreadFunction, this);
        }
    }

    void stopWorkerThreads()
    {
        shouldStop_ = true;
        jobQueue_.stop();
        
        for (auto& thread : workerThreads_)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }
        workerThreads_.clear();
    }

    void workerThreadFunction()
    {
        while (!shouldStop_)
        {
            try
            {
                Job job = jobQueue_.waitAndPop();
                processJob(job);
            }
            catch (const std::runtime_error&)
            {
                // Thread should exit (queue stopped)
                break;
            }
            catch (const std::exception& ex)
            {
                std::cerr << "Worker thread error: " << ex.what() << std::endl;
                // Continue running despite error
            }
        }
    }

    void enqueueJob(const std::shared_ptr<Chunk>& chunk, JobType type, const glm::ivec2& coord)
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

    void processJob(const Job& job)
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
            generateChunkBlocks(*chunk);
            chunk->state = ChunkState::Meshing;

            enqueueJob(chunk, JobType::Mesh, job.chunkCoord);
        }
        else if (job.type == JobType::Mesh)
        {
            buildChunkMeshAsync(*chunk);
            chunk->state = ChunkState::Ready;
            queueChunkForUpload(chunk);
        }
    }

    std::shared_ptr<Chunk> popNextChunkForUpload()
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

            chunk->queuedForUpload = false;
            return chunk;
        }
        return nullptr;
    }

    void queueChunkForUpload(const std::shared_ptr<Chunk>& chunk)
    {
        if (!chunk)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(uploadQueueMutex_);
        if (chunk->queuedForUpload)
        {
            return;
        }

        uploadQueue_.emplace_back(chunk);
        chunk->queuedForUpload = true;
    }

    void requeueChunkForUpload(const std::shared_ptr<Chunk>& chunk, bool toFront)
    {
        if (!chunk)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(uploadQueueMutex_);
        if (chunk->queuedForUpload)
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
        chunk->queuedForUpload = true;
    }

    struct BufferEntry
    {
        GLuint vao{0};
        GLuint vbo{0};
        GLuint ibo{0};
        std::size_t vertexCapacity{0};
        std::size_t indexCapacity{0};
    };

    static std::size_t bucketForSize(std::size_t bytes) noexcept
    {
        bytes = std::max(bytes, kMinBufferSizeBytes);
        bytes -= 1;
        bytes |= bytes >> 1;
        bytes |= bytes >> 2;
        bytes |= bytes >> 4;
        bytes |= bytes >> 8;
        bytes |= bytes >> 16;
#if SIZE_MAX > 0xffffffffu
        bytes |= bytes >> 32;
#endif
        return bytes + 1;
    }

    BufferEntry acquireBufferEntry(std::size_t vertexBytes, std::size_t indexBytes)
    {
        const std::size_t vertexBucket = bucketForSize(vertexBytes);
        const std::size_t indexBucket = bucketForSize(indexBytes);

        {
            std::lock_guard<std::mutex> lock(bufferPoolMutex_);
            auto it = bufferPool_.lower_bound(vertexBucket);
            while (it != bufferPool_.end())
            {
                auto& pool = it->second;
                for (std::size_t i = 0; i < pool.size(); ++i)
                {
                    if (pool[i].indexCapacity >= indexBucket)
                    {
                        auto itEntry = pool.begin() + static_cast<std::ptrdiff_t>(i);
                        BufferEntry entry = *itEntry;
                        pool.erase(itEntry);
                        if (pool.empty())
                        {
                            bufferPool_.erase(it);
                        }
                        return entry;
                    }
                }
                ++it;
            }
        }

        BufferEntry entry;
        entry.vertexCapacity = vertexBucket;
        entry.indexCapacity = indexBucket;

        glGenVertexArrays(1, &entry.vao);
        glGenBuffers(1, &entry.vbo);
        glGenBuffers(1, &entry.ibo);

        glBindVertexArray(entry.vao);

        glBindBuffer(GL_ARRAY_BUFFER, entry.vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(entry.vertexCapacity), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, position)));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, tileCoord)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, atlasBase)));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, atlasSize)));
        glEnableVertexAttribArray(4);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, entry.ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(entry.indexCapacity), nullptr, GL_DYNAMIC_DRAW);

        glBindVertexArray(0);

        return entry;
    }

    void releaseChunkBuffers(Chunk& chunk)
    {
        if (chunk.vao == 0)
        {
            chunk.vertexCapacity = 0;
            chunk.indexCapacity = 0;
            return;
        }

        BufferEntry entry{};
        entry.vao = chunk.vao;
        entry.vbo = chunk.vbo;
        entry.ibo = chunk.ibo;
        entry.vertexCapacity = chunk.vertexCapacity;
        entry.indexCapacity = chunk.indexCapacity;

        chunk.vao = 0;
        chunk.vbo = 0;
        chunk.ibo = 0;
        chunk.vertexCapacity = 0;
        chunk.indexCapacity = 0;
        chunk.indexCount = 0;

        std::lock_guard<std::mutex> lock(bufferPoolMutex_);
        bufferPool_[entry.vertexCapacity].push_back(entry);
    }

    void ensureChunkBuffers(Chunk& chunk, std::size_t vertexBytes, std::size_t indexBytes)
    {
        const std::size_t requiredVertex = std::max(vertexBytes, static_cast<std::size_t>(sizeof(Vertex)));
        const std::size_t requiredIndex = std::max(indexBytes, static_cast<std::size_t>(sizeof(std::uint32_t)));

        if (chunk.vao != 0 &&
            chunk.vertexCapacity >= requiredVertex &&
            chunk.indexCapacity >= requiredIndex)
        {
            return;
        }

        if (chunk.vao != 0)
        {
            releaseChunkBuffers(chunk);
        }

        BufferEntry entry = acquireBufferEntry(requiredVertex, requiredIndex);
        chunk.vao = entry.vao;
        chunk.vbo = entry.vbo;
        chunk.ibo = entry.ibo;
        chunk.vertexCapacity = entry.vertexCapacity;
        chunk.indexCapacity = entry.indexCapacity;
    }

    void recycleChunkGPU(Chunk& chunk)
    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        releaseChunkBuffers(chunk);
        chunk.meshData.clear();
        chunk.meshReady = false;
        chunk.queuedForUpload = false;
        chunk.indexCount = 0;
    }

    void destroyBufferPool()
    {
        std::lock_guard<std::mutex> lock(bufferPoolMutex_);
        for (auto& [_, entries] : bufferPool_)
        {
            for (auto& entry : entries)
            {
                if (entry.ibo != 0)
                {
                    glDeleteBuffers(1, &entry.ibo);
                }
                if (entry.vbo != 0)
                {
                    glDeleteBuffers(1, &entry.vbo);
                }
                if (entry.vao != 0)
                {
                    glDeleteVertexArrays(1, &entry.vao);
                }
            }
        }
        bufferPool_.clear();
    }
    struct RingProgress
    {
        bool fullyLoaded{false};
        bool budgetExhausted{false};
    };

    RingProgress ensureRing(const glm::ivec2& center, int radius, int& jobBudget)
    {
        bool missingFound = false;

        auto visitCoordinate = [&](const glm::ivec2& coord) -> bool
        {
            if (jobBudget <= 0)
            {
                return true;
            }

            if (ensureChunkAsync(coord))
            {
                --jobBudget;
                missingFound = true;
            }

            return jobBudget <= 0;
        };

        if (radius == 0)
        {
            if (visitCoordinate(center))
            {
                return RingProgress{false, true};
            }

            return RingProgress{!missingFound, false};
        }

        for (int dx = -radius; dx <= radius; ++dx)
        {
            if (visitCoordinate(center + glm::ivec2(dx, -radius)))
            {
                return RingProgress{false, true};
            }
            if (visitCoordinate(center + glm::ivec2(dx, radius)))
            {
                return RingProgress{false, true};
            }
        }

        for (int dz = -radius + 1; dz <= radius - 1; ++dz)
        {
            if (visitCoordinate(center + glm::ivec2(-radius, dz)))
            {
                return RingProgress{false, true};
            }
            if (visitCoordinate(center + glm::ivec2(radius, dz)))
            {
                return RingProgress{false, true};
            }
        }

        return RingProgress{!missingFound, false};
    }

    void removeDistantChunks(const glm::ivec2& center)
    {
        std::vector<glm::ivec2> toRemove;
        {
            std::lock_guard<std::mutex> lock(chunksMutex);
            toRemove.reserve(chunks_.size());
            for (const auto& [coord, chunkPtr] : chunks_)
            {
                const int dx = coord.x - center.x;
                const int dz = coord.y - center.y;
                if (std::max(std::abs(dx), std::abs(dz)) > targetViewDistance_)
                {
                    toRemove.push_back(coord);
                }
            }
        }

        for (const glm::ivec2& coord : toRemove)
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
                recycleChunkGPU(*chunk);
            }
        }
    }
    bool ensureChunkAsync(const glm::ivec2& coord)
    {
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

                chunk = std::make_shared<Chunk>(coord);
                chunk->state = ChunkState::Generating;
                chunks_.emplace(coord, chunk);
            }

            enqueueJob(chunk, JobType::Generate, coord);
            return true;
        }
        catch (const std::exception& ex)
        {
            std::cerr << "Error creating chunk at (" << coord.x << ", " << coord.y << "): " << ex.what() << std::endl;
            return false;
        }
    }


    void uploadReadyMeshes()
    {
        std::size_t remainingBudget = kUploadBudgetBytesPerFrame;
        bool uploadedAnything = false;

        while (remainingBudget > 0 || !uploadedAnything)
        {
            std::shared_ptr<Chunk> chunk = popNextChunkForUpload();
            if (!chunk)
            {
                break;
            }

            if (!chunk->meshReady || chunk->state.load() != ChunkState::Ready)
            {
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
                break;
            }

            uploadChunkMesh(*chunk);
            chunk->state = ChunkState::Uploaded;
            chunk->meshReady = false;
            uploadedAnything = true;

            if (totalBytes >= remainingBudget)
            {
                remainingBudget = 0;
            }
            else
            {
                remainingBudget -= totalBytes;
            }
        }
    }

    void uploadChunkMesh(Chunk& chunk)
    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);

        if (chunk.meshData.empty())
        {
            releaseChunkBuffers(chunk);
            chunk.meshData.clear();
            chunk.indexCount = 0;
            return;
        }

        const std::size_t vertexBytes = chunk.meshData.vertices.size() * sizeof(Vertex);
        const std::size_t indexBytes = chunk.meshData.indices.size() * sizeof(std::uint32_t);

        ensureChunkBuffers(chunk, vertexBytes, indexBytes);

        glBindVertexArray(chunk.vao);
        glBindBuffer(GL_ARRAY_BUFFER, chunk.vbo);
        if (vertexBytes > 0)
        {
            glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(vertexBytes), chunk.meshData.vertices.data());
        }

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, chunk.ibo);
        if (indexBytes > 0)
        {
            glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(indexBytes), chunk.meshData.indices.data());
        }

        chunk.indexCount = static_cast<GLsizei>(chunk.meshData.indices.size());
        glBindVertexArray(0);

        chunk.meshData.clear();
    }


    void buildChunkMeshAsync(Chunk& chunk)
    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        chunk.meshData.clear();

        const int baseWorldX = chunk.coord.x * kChunkSizeX;
        const int baseWorldZ = chunk.coord.y * kChunkSizeZ;
        const glm::vec3 chunkOrigin(static_cast<float>(baseWorldX), 0.0f, static_cast<float>(baseWorldZ));

        auto isInsideChunk = [](const glm::ivec3& local) noexcept
        {
            return local.x >= 0 && local.x < kChunkSizeX &&
                   local.y >= 0 && local.y < kChunkSizeY &&
                   local.z >= 0 && local.z < kChunkSizeZ;
        };

        auto localToWorld = [&](int lx, int ly, int lz) -> glm::ivec3
        {
            return glm::ivec3(baseWorldX + lx, ly, baseWorldZ + lz);
        };

        auto sampleBlock = [&](int lx, int ly, int lz) -> BlockId
        {
            if (ly < 0 || ly >= kChunkSizeY)
            {
                return BlockId::Air;
            }

            if (lx >= 0 && lx < kChunkSizeX && lz >= 0 && lz < kChunkSizeZ)
            {
                return chunk.blocks[blockIndex(lx, ly, lz)];
            }

            return blockAt(localToWorld(lx, ly, lz));
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

            bool operator==(const FaceMaterial& other) const noexcept
            {
                return uvBase == other.uvBase &&
                       uvSize == other.uvSize &&
                       uAxis == other.uAxis &&
                       vAxis == other.vAxis &&
                       face == other.face;
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

        auto makeMaterial = [&](BlockId block, const glm::vec3& normal) -> FaceMaterial
        {
            FaceMaterial material{};

            const BlockFace face = [&]() -> BlockFace
            {
                if (normal.y > 0.5f) return BlockFace::Top;
                if (normal.y < -0.5f) return BlockFace::Bottom;
                if (normal.x > 0.5f) return BlockFace::East;
                if (normal.x < -0.5f) return BlockFace::West;
                if (normal.z > 0.5f) return BlockFace::South;
                return BlockFace::North;
            }();

            material.face = face;

            if (blockAtlasConfigured_)
            {
                const BlockUVSet& uvSet = blockUVTable_[toIndex(block)];
                const FaceUV& faceUV = uvSet.faces[toIndex(face)];
                material.uvBase = faceUV.base;
                material.uvSize = faceUV.size;
            }
            else
            {
                // Fallback so we can still render even if the atlas has not been configured yet.
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

            if (dir == FaceDir::Negative)
            {
                std::swap(positions[1], positions[3]);
            }

            const glm::vec3 uAxisVec = glm::vec3(material.uAxis);
            const glm::vec3 vAxisVec = glm::vec3(material.vAxis);

            const std::size_t vertexStart = chunk.meshData.vertices.size();
            for (int i = 0; i < 4; ++i)
            {
                const glm::vec3& pos = positions[i];

                Vertex vertex{};
                vertex.position = pos;
                vertex.normal = normal;
                vertex.tileCoord = glm::vec2(glm::dot(pos, uAxisVec), glm::dot(pos, vAxisVec));
                vertex.atlasBase = material.uvBase;
                vertex.atlasSize = material.uvSize;
                chunk.meshData.vertices.push_back(vertex);
            }

            chunk.meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 0));
            chunk.meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 1));
            chunk.meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 2));
            chunk.meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 2));
            chunk.meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 3));
            chunk.meshData.indices.push_back(static_cast<std::uint32_t>(vertexStart + 0));
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

                            const bool positiveSolid = isSolid(sampleBlock(positiveLocal.x, positiveLocal.y, positiveLocal.z));
                            const bool negativeSolid = isSolid(sampleBlock(negativeLocal.x, negativeLocal.y, negativeLocal.z));

                            glm::ivec3 owningLocal{0};
                            bool createFace = false;

                            if (dir == FaceDir::Positive)
                            {
                                if (negativeSolid && !positiveSolid && isInsideChunk(negativeLocal))
                                {
                                    owningLocal = negativeLocal;
                                    createFace = true;
                                }
                            }
                            else
                            {
                                if (positiveSolid && !negativeSolid && isInsideChunk(positiveLocal))
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
                                const BlockId owningBlock = chunk.blocks[blockIdx];
                                cell.material = makeMaterial(owningBlock, normal);
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
                            while (ci + runLengthC < sizeC)
                            {
                                const MaskCell& nextCell = mask[maskIndex(bi, ci + runLengthC)];
                                if (!nextCell.exists || !(nextCell.material == material))
                                {
                                    break;
                                }
                                ++runLengthC;
                            }

                            int runHeightB = 1;
                            while (bi + runHeightB < sizeB)
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

        chunk.meshReady = true;
    }

    [[nodiscard]] static glm::ivec2 worldToChunkCoords(int worldX, int worldZ) noexcept
    {
        return {floorDiv(worldX, kChunkSizeX), floorDiv(worldZ, kChunkSizeZ)};
    }

    [[nodiscard]] std::shared_ptr<Chunk> getChunkShared(const glm::ivec2& coord) noexcept
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        auto it = chunks_.find(coord);
        return (it != chunks_.end()) ? it->second : nullptr;
    }

    [[nodiscard]] std::shared_ptr<const Chunk> getChunkShared(const glm::ivec2& coord) const noexcept
    {
        std::lock_guard<std::mutex> lock(chunksMutex);
        auto it = chunks_.find(coord);
        if (it != chunks_.end())
        {
            return it->second;
        }
        return nullptr;
    }

    [[nodiscard]] Chunk* getChunk(const glm::ivec2& coord) noexcept
    {
        return getChunkShared(coord).get();
    }

    [[nodiscard]] const Chunk* getChunk(const glm::ivec2& coord) const noexcept
    {
        return getChunkShared(coord).get();
    }


    void markNeighborsForRemeshingIfNeeded(const glm::ivec2& coord, int localX, int localZ)
    {
        auto queueNeighbor = [&](const glm::ivec2& neighborCoord)
        {
            auto neighbor = getChunkShared(neighborCoord);
            if (!neighbor)
            {
                return;
            }

            ChunkState neighborState = neighbor->state.load();
            if (neighborState != ChunkState::Uploaded && neighborState != ChunkState::Remeshing)
            {
                return;
            }

            neighbor->state = ChunkState::Remeshing;
            try
            {
                enqueueJob(neighbor, JobType::Mesh, neighborCoord);
            }
            catch (const std::exception& ex)
            {
                std::cerr << "Failed to queue remesh for neighbor (" << neighborCoord.x << ", " << neighborCoord.y << "): " << ex.what() << std::endl;
            }
        };

        if (localX == 0) // Left edge - affects left neighbor
        {
            queueNeighbor(coord + glm::ivec2{-1, 0});
        }

        if (localX == kChunkSizeX - 1) // Right edge - affects right neighbor
        {
            queueNeighbor(coord + glm::ivec2{1, 0});
        }

        if (localZ == 0) // Front edge - affects front neighbor
        {
            queueNeighbor(coord + glm::ivec2{0, -1});
        }

        if (localZ == kChunkSizeZ - 1) // Back edge - affects back neighbor
        {
            queueNeighbor(coord + glm::ivec2{0, 1});
        }
    }

    void recomputeColumnHeight(Chunk& chunk, int localX, int localZ) noexcept
    {
        const std::size_t idx = columnIndex(localX, localZ);
        int top = -1;
        for (int y = kChunkSizeY - 1; y >= 0; --y)
        {
            if (isSolid(chunk.blocks[blockIndex(localX, y, localZ)]))
            {
                top = y;
                break;
            }
        }
        chunk.columnMaxHeights[idx] = top;
    }

    void generateChunkBlocks(Chunk& chunk)
    {
        std::lock_guard<std::mutex> lock(chunk.meshMutex);
        const int baseWorldX = chunk.coord.x * kChunkSizeX;
        const int baseWorldZ = chunk.coord.y * kChunkSizeZ;

        for (int x = 0; x < kChunkSizeX; ++x)
        {
            for (int z = 0; z < kChunkSizeZ; ++z)
            {
                const int worldX = baseWorldX + x;
                const int worldZ = baseWorldZ + z;

                const float nx = static_cast<float>(worldX) * 0.01f;
                const float nz = static_cast<float>(worldZ) * 0.01f;

                // Main terrain with more octaves for rolling hills
                float mainTerrain = noise_.fbm(nx, nz, 6, 0.5f, 2.0f);
                
                // Mountain features with ridge noise
                float mountainNoise = noise_.ridge(nx * 0.4f, nz * 0.4f, 5, 2.1f, 0.5f);
                
                // Fine detail with high-frequency noise
                float detailNoise = noise_.fbm(nx * 4.0f, nz * 4.0f, 8, 0.45f, 2.2f);
                
                // Medium-scale features
                float mediumNoise = noise_.fbm(nx * 0.8f, nz * 0.8f, 7, 0.5f, 2.0f);
                
                // Combine all noise layers for natural variation
                float combined = mainTerrain * 12.0f +           // Base rolling hills
                               mountainNoise * 8.0f +            // Mountain ridges
                               mediumNoise * 4.0f +             // Medium features
                               detailNoise * 2.0f;              // Fine surface detail

                float targetHeight = 16.0f + combined;
                targetHeight = std::clamp(targetHeight, 2.0f, static_cast<float>(kChunkSizeY - 3));
                const int columnHeight = std::max(1, static_cast<int>(std::round(targetHeight)));
                const int topBlock = std::clamp(columnHeight, 0, kChunkSizeY - 1);
                chunk.columnMaxHeights[columnIndex(x, z)] = topBlock;

                for (int y = 0; y < kChunkSizeY; ++y)
                {
                    chunk.blocks[blockIndex(x, y, z)] = (y <= topBlock) ? BlockId::Grass : BlockId::Air;
                }
            }
        }
    }


    struct FaceUV
    {
        glm::vec2 base{0.0f}; // Normalized origin inside the atlas.
        glm::vec2 size{1.0f}; // Normalized tile dimensions.
    };

    struct BlockUVSet
    {
        std::array<FaceUV, kBlockFaceCount> faces{}; // One entry per cube face.
    };

    glm::vec2 atlasTileScale_{1.0f, 1.0f};
    std::array<BlockUVSet, kBlockCount> blockUVTable_{};
    bool blockAtlasConfigured_{false};

    std::deque<std::weak_ptr<Chunk>> uploadQueue_;
    std::mutex uploadQueueMutex_;
    std::map<std::size_t, std::vector<BufferEntry>> bufferPool_;
    std::mutex bufferPoolMutex_;
    PerlinNoise noise_;
    std::unordered_map<glm::ivec2, std::shared_ptr<Chunk>, ChunkHasher> chunks_;
    mutable std::mutex chunksMutex;
    const glm::vec3 lightDirection_{glm::normalize(glm::vec3(0.5f, -1.0f, 0.2f))};
    GLuint atlasTexture_{0};
    // Threading infrastructure
    JobQueue jobQueue_;
    std::vector<std::thread> workerThreads_;
    std::atomic<bool> shouldStop_;
    
    // Block highlighting
    glm::ivec3 highlightedBlock_{0};
    bool hasHighlight_{false};
    
    // View distance setting
    int viewDistance_;
    int targetViewDistance_;
};

// Collision detection helper functions
struct AABB
{
    glm::vec3 min;
    glm::vec3 max;
};

struct PlayerInputState
{
    glm::vec3 moveDirection{0.0f};
    bool jumpHeld{false};
};

inline AABB makePlayerAABB(const glm::vec3& position) noexcept
{
    const float halfWidth = kPlayerWidth * 0.5f;
    const glm::vec3 minCorner(position.x - halfWidth,
                              position.y - kCameraEyeHeight,
                              position.z - halfWidth);
    return AABB{minCorner, minCorner + glm::vec3(kPlayerWidth, kPlayerHeight, kPlayerWidth)};
}

inline bool overlaps1D(float minA, float maxA, float minB, float maxB) noexcept
{
    return (minA < maxB - kAxisCollisionEpsilon) && (maxA > minB + kAxisCollisionEpsilon);
}

struct AxisMoveResult
{
    float actualMove{0.0f};
    bool collided{false};
};

AxisMoveResult sweepPlayerAABB(AABB& box,
                               glm::vec3& position,
                               float move,
                               int axis,
                               const ChunkManager& chunkManager)
{
    AxisMoveResult result{move, false};
    if (std::abs(move) <= kAxisCollisionEpsilon)
    {
        if (move != 0.0f)
        {
            position[axis] += move;
            box.min[axis] += move;
            box.max[axis] += move;
        }
        return result;
    }

    const int otherAxis0 = (axis + 1) % 3;
    const int otherAxis1 = (axis + 2) % 3;
    const float minOther0 = box.min[otherAxis0];
    const float maxOther0 = box.max[otherAxis0];
    const float minOther1 = box.min[otherAxis1];
    const float maxOther1 = box.max[otherAxis1];

    int other0Min = static_cast<int>(std::floor(minOther0));
    int other0Max = static_cast<int>(std::floor(maxOther0));
    if (other0Max < other0Min)
    {
        other0Max = other0Min;
    }

    int other1Min = static_cast<int>(std::floor(minOther1));
    int other1Max = static_cast<int>(std::floor(maxOther1));
    if (other1Max < other1Min)
    {
        other1Max = other1Min;
    }

    auto layerHasCollision = [&](int primaryIndex) -> bool
    {
        for (int idx0 = other0Min; idx0 <= other0Max; ++idx0)
        {
            const float blockMin0 = static_cast<float>(idx0);
            const float blockMax0 = blockMin0 + 1.0f;
            if (!overlaps1D(minOther0, maxOther0, blockMin0, blockMax0))
            {
                continue;
            }

            for (int idx1 = other1Min; idx1 <= other1Max; ++idx1)
            {
                const float blockMin1 = static_cast<float>(idx1);
                const float blockMax1 = blockMin1 + 1.0f;
                if (!overlaps1D(minOther1, maxOther1, blockMin1, blockMax1))
                {
                    continue;
                }

                glm::ivec3 blockCoord(0);
                blockCoord[axis] = primaryIndex;
                blockCoord[otherAxis0] = idx0;
                blockCoord[otherAxis1] = idx1;

                if (isSolid(chunkManager.blockAt(blockCoord)))
                {
                    return true;
                }
            }
        }
        return false;
    };

    float allowed = move;
    if (move > 0.0f)
    {
        const float face = box.max[axis];
        const int firstBlock = static_cast<int>(std::floor(face - kAxisCollisionEpsilon)) + 1;
        const int lastBlock = static_cast<int>(std::floor(face + move + kAxisCollisionEpsilon));
        if (firstBlock <= lastBlock)
        {
            for (int primary = firstBlock; primary <= lastBlock; ++primary)
            {
                const float blockMin = static_cast<float>(primary);
                const float distance = blockMin - face;
                if (distance > allowed + kAxisCollisionEpsilon)
                {
                    break;
                }

                if (layerHasCollision(primary))
                {
                    allowed = std::min(allowed, std::max(distance - kAxisCollisionEpsilon, 0.0f));
                    result.collided = true;
                    break;
                }
            }
        }
        allowed = std::clamp(allowed, 0.0f, move);
    }
    else
    {
        const float face = box.min[axis];
        const int firstBlock = static_cast<int>(std::floor(face - kAxisCollisionEpsilon));
        const int lastBlock = static_cast<int>(std::floor(face + move - kAxisCollisionEpsilon));
        if (firstBlock >= lastBlock)
        {
            for (int primary = firstBlock; primary >= lastBlock; --primary)
            {
                const float blockMax = static_cast<float>(primary + 1);
                const float distance = blockMax - face;
                if (distance < allowed - kAxisCollisionEpsilon)
                {
                    break;
                }

                if (layerHasCollision(primary))
                {
                    allowed = std::max(allowed, std::min(distance + kAxisCollisionEpsilon, 0.0f));
                    result.collided = true;
                    break;
                }
            }
        }
        allowed = std::clamp(allowed, move, 0.0f);
    }

    position[axis] += allowed;
    box.min[axis] += allowed;
    box.max[axis] += allowed;
    result.actualMove = allowed;
    return result;
}

PlayerInputState computePlayerInputState(GLFWwindow* window,
                                         InputContext& inputContext,
                                         Camera& camera,
                                         ChunkManager& chunkManager)
{
    PlayerInputState state;

    bool nKeyCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS);
    inputContext.nKeyJustPressed = nKeyCurrentlyPressed && !inputContext.nKeyPressed;
    inputContext.nKeyPressed = nKeyCurrentlyPressed;
    if (inputContext.nKeyJustPressed && !inputContext.showRenderDistanceGUI)
    {
        // Show the GUI
        inputContext.showRenderDistanceGUI = true;
        inputContext.inputBuffer.clear();
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
    
    // Handle GUI input
    if (inputContext.showRenderDistanceGUI)
    {
        // Enter key to apply
        static bool enterKeyPressed = false;
        bool enterKeyCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS || 
                                          glfwGetKey(window, GLFW_KEY_KP_ENTER) == GLFW_PRESS);
        if (enterKeyCurrentlyPressed && !enterKeyPressed)
        {
            if (!inputContext.inputBuffer.empty())
            {
                try
                {
                    int distance = std::stoi(inputContext.inputBuffer);
                    chunkManager.setRenderDistance(distance);
                }
                catch (const std::exception& e)
                {
                    std::cerr << "Invalid render distance input: " << e.what() << std::endl;
                }
            }
            inputContext.showRenderDistanceGUI = false;
            inputContext.inputBuffer.clear();
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        enterKeyPressed = enterKeyCurrentlyPressed;
        
        // Escape key to cancel
        static bool escapeKeyPressed = false;
        bool escapeKeyCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS);
        if (escapeKeyCurrentlyPressed && !escapeKeyPressed)
        {
            inputContext.showRenderDistanceGUI = false;
            inputContext.inputBuffer.clear();
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        escapeKeyPressed = escapeKeyCurrentlyPressed;
        
        // Backspace to delete characters
        static bool backspaceKeyPressed = false;
        bool backspaceKeyCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_BACKSPACE) == GLFW_PRESS);
        if (backspaceKeyCurrentlyPressed && !backspaceKeyPressed)
        {
            if (!inputContext.inputBuffer.empty())
            {
                inputContext.inputBuffer.pop_back();
            }
        }
        backspaceKeyPressed = backspaceKeyCurrentlyPressed;
    }

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
        state.moveDirection += forward;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        state.moveDirection -= forward;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        state.moveDirection -= right;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        state.moveDirection += right;
    }

    state.jumpHeld = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
    return state;
}

void applyGroundSnap(Camera& camera, const ChunkManager& chunkManager)
{
    const float halfWidth = kPlayerWidth * 0.5f;
    const std::array<glm::vec2, 4> sampleOffsets = {
        glm::vec2{-halfWidth, -halfWidth},
        glm::vec2{halfWidth, -halfWidth},
        glm::vec2{-halfWidth, halfWidth},
        glm::vec2{halfWidth, halfWidth}
    };

    float highestSurface = -std::numeric_limits<float>::infinity();
    for (const glm::vec2& offset : sampleOffsets)
    {
        const float sampleX = camera.position.x + offset.x;
        const float sampleZ = camera.position.z + offset.y;
        highestSurface = std::max(highestSurface, chunkManager.surfaceHeight(sampleX, sampleZ));
    }

    if (highestSurface > -std::numeric_limits<float>::infinity())
    {
        const float desiredY = highestSurface + kCameraEyeHeight;
        if (desiredY <= camera.position.y + kGroundSnapTolerance && camera.velocity.y <= 0.0f)
        {
            camera.position.y = desiredY;
			camera.velocity.y = 0.0f;
			camera.onGround = true;
        }
    }
}

void updatePhysics(Camera& camera,
                   const ChunkManager& chunkManager,
                   const PlayerInputState& inputState,
                   float dt)
{
    camera.velocity.y += kGravity * dt;
    if (camera.velocity.y < kTerminalVelocity)
    {
        camera.velocity.y = kTerminalVelocity;
    }

    const glm::vec2 horizontalInput(inputState.moveDirection.x, inputState.moveDirection.z);
    if (glm::dot(horizontalInput, horizontalInput) > kEpsilon * kEpsilon)
    {
        glm::vec3 normalized = glm::normalize(glm::vec3(horizontalInput.x, 0.0f, horizontalInput.y));
        camera.velocity.x = normalized.x * camera.moveSpeed;
        camera.velocity.z = normalized.z * camera.moveSpeed;
    }
    else
    {
        camera.velocity.x *= kHorizontalDamping;
        camera.velocity.z *= kHorizontalDamping;

        if (std::abs(camera.velocity.x) < kAxisCollisionEpsilon)
        {
            camera.velocity.x = 0.0f;
        }
        if (std::abs(camera.velocity.z) < kAxisCollisionEpsilon)
        {
            camera.velocity.z = 0.0f;
        }
    }

    if (inputState.jumpHeld && camera.onGround)
    {
        camera.velocity.y = kJumpVelocity;
        camera.onGround = false;
    }

    glm::vec3 desiredMove = camera.velocity * dt;
    AABB box = makePlayerAABB(camera.position);

    auto moveAndResolveAxis = [&](int axis) -> AxisMoveResult
    {
        return sweepPlayerAABB(box, camera.position, desiredMove[axis], axis, chunkManager);
    };

    AxisMoveResult moveX = moveAndResolveAxis(0);
    if (std::abs(moveX.actualMove - desiredMove.x) > kAxisCollisionEpsilon)
    {
        camera.velocity.x = 0.0f;
    }

    AxisMoveResult moveZ = moveAndResolveAxis(2);
    if (std::abs(moveZ.actualMove - desiredMove.z) > kAxisCollisionEpsilon)
    {
        camera.velocity.z = 0.0f;
    }

    bool groundedThisStep = false;
    AxisMoveResult moveY = moveAndResolveAxis(1);
    if (std::abs(moveY.actualMove - desiredMove.y) > kAxisCollisionEpsilon)
    {
        camera.velocity.y = 0.0f;
        if (desiredMove.y < 0.0f && moveY.actualMove > desiredMove.y)
        {
            groundedThisStep = true;
        }
    }

    camera.onGround = groundedThisStep;
    if (camera.onGround)
    {
        applyGroundSnap(camera, chunkManager);
    }
}

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
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCharCallback(window, charCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    const char* vertexShaderSrc = R"(#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTileCoord;
layout (location = 3) in vec2 aAtlasBase;
layout (location = 4) in vec2 aAtlasSize;

uniform mat4 uViewProj;

out vec3 vNormal;
out vec3 vWorldPos;
out vec2 vTileCoord;
out vec2 vAtlasBase;
out vec2 vAtlasSize;

void main()
{
    vNormal = aNormal;
    vWorldPos = aPos;
    vTileCoord = aTileCoord;
    vAtlasBase = aAtlasBase;
    vAtlasSize = aAtlasSize;
    gl_Position = uViewProj * vec4(aPos, 1.0);
}
)";

    const char* fragmentShaderSrc = R"(#version 330 core
out vec4 FragColor;

in vec3 vNormal;
in vec3 vWorldPos;
in vec2 vTileCoord;
in vec2 vAtlasBase;
in vec2 vAtlasSize;

uniform sampler2D uAtlas;
uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform vec3 uHighlightedBlock;
uniform int uHasHighlight;

void main()
{
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(-uLightDir);
    vec3 viewDir = normalize(uCameraPos - vWorldPos);
    float diff = max(dot(normal, lightDir), 0.0);
    float ambient = 0.35;
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);

    vec2 tileUV = fract(vTileCoord);
    vec2 atlasUV = vAtlasBase + vAtlasSize * tileUV;
    vec3 textureColor = texture(uAtlas, atlasUV).rgb;
    vec3 color = textureColor * (ambient + diff) + vec3(0.1f) * spec;

    if (uHasHighlight == 1) {
        ivec3 currentBlock = ivec3(floor(vWorldPos));
        ivec3 targetBlock = ivec3(uHighlightedBlock);

        if (currentBlock == targetBlock) {
            color += vec3(0.3f);
            color = min(color, vec3(1.0));
        }
    }

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

    ChunkShaderUniformLocations chunkUniforms{};
    chunkUniforms.uViewProj = glGetUniformLocation(shaderProgram, "uViewProj");
    chunkUniforms.uLightDir = glGetUniformLocation(shaderProgram, "uLightDir");
    chunkUniforms.uCameraPos = glGetUniformLocation(shaderProgram, "uCameraPos");
    chunkUniforms.uAtlas = glGetUniformLocation(shaderProgram, "uAtlas");
    chunkUniforms.uHighlightedBlock = glGetUniformLocation(shaderProgram, "uHighlightedBlock");
    chunkUniforms.uHasHighlight = glGetUniformLocation(shaderProgram, "uHasHighlight");

    LoadedTexture blockAtlas = loadTexture("block_atlas.png");
    if (blockAtlas.id == 0)
    {
        glDeleteProgram(shaderProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glUseProgram(shaderProgram);
    if (chunkUniforms.uAtlas >= 0)
    {
        glUniform1i(chunkUniforms.uAtlas, 0);
    }
    glUseProgram(0);

    ChunkManager chunkManager(1337u);
    chunkManager.setAtlasTexture(blockAtlas.id);
    chunkManager.setBlockTextureAtlasConfig(blockAtlas.size, kAtlasTileSizePixels); // Map block faces to atlas tiles.
    chunkManager.update(camera.position);
    
    // Find a guaranteed safe spawn position above ground
    std::cout << "Finding safe spawn position..." << std::endl;
    camera.position = chunkManager.findSafeSpawnPosition(camera.position.x, camera.position.z);
    camera.velocity = glm::vec3(0.0f);
    camera.onGround = false;
    
    std::cout << "Player spawned at: (" << camera.position.x << ", " << camera.position.y << ", " << camera.position.z << ")" << std::endl;

    Crosshair crosshair;
    TextOverlay textOverlay;

    constexpr double kFixedTimeStep = 1.0 / 60.0;
    double previousTime = glfwGetTime();
    double accumulator = 0.0;
    std::cout << "Controls: WASD to move, mouse to look, SPACE to jump, N to set render distance, left-click to destroy blocks, right-click to place blocks, ESC to quit." << std::endl;

    while (!glfwWindowShouldClose(window))
    {
        const double currentTime = glfwGetTime();
        double frameTime = currentTime - previousTime;
        previousTime = currentTime;
        frameTime = std::min(frameTime, 0.25);
        accumulator += frameTime;

        glfwPollEvents();

        bool f1CurrentlyPressed = (glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS);
        bool f1JustPressed = f1CurrentlyPressed && !inputContext.f1Pressed;
        if (f1JustPressed)
        {
            inputContext.showCoordinates = !inputContext.showCoordinates;
        }
        inputContext.f1JustPressed = f1JustPressed;
        inputContext.f1Pressed = f1CurrentlyPressed;

        // Only close window with ESC if GUI is not active
        // (ESC to close GUI is handled in computePlayerInputState)
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS && !inputContext.showRenderDistanceGUI)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        auto* inputContextPtr = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
        while (accumulator >= kFixedTimeStep)
        {
            if (inputContextPtr)
            {
                PlayerInputState inputState = computePlayerInputState(window, *inputContextPtr, camera, chunkManager);
                updatePhysics(camera, chunkManager, inputState, static_cast<float>(kFixedTimeStep));
            }
            else
            {
                InputContext dummy;
                PlayerInputState inputState = computePlayerInputState(window, dummy, camera, chunkManager);
                updatePhysics(camera, chunkManager, inputState, static_cast<float>(kFixedTimeStep));
            }
            accumulator -= kFixedTimeStep;
        }

        // Update block highlighting based on crosshair
        chunkManager.updateHighlight(camera.position, camera.front());

        // Handle block destruction
        if (inputContext.leftMouseJustPressed)
        {
            RaycastHit hit = chunkManager.raycast(camera.position, camera.front());
            if (hit.hit)
            {
                chunkManager.destroyBlock(hit.blockPos);
            }
            inputContext.leftMouseJustPressed = false; // Reset the flag
        }

        // Handle block placement
        if (inputContext.rightMouseJustPressed)
        {
            RaycastHit hit = chunkManager.raycast(camera.position, camera.front());
            if (hit.hit)
            {
                chunkManager.placeBlock(hit.blockPos, hit.faceNormal);
            }
            inputContext.rightMouseJustPressed = false; // Reset the flag
        }

        chunkManager.update(camera.position);

        glClearColor(0.55f, 0.78f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
        framebufferWidth = std::max(framebufferWidth, 1);
        framebufferHeight = std::max(framebufferHeight, 1);
        const float aspect = static_cast<float>(framebufferWidth) / static_cast<float>(framebufferHeight);

        const float currentFarPlane = computeFarPlaneForViewDistance(chunkManager.viewDistance());
        kFarPlane = currentFarPlane;
        const glm::mat4 projection = glm::perspective(glm::radians(60.0f), aspect, kNearPlane, currentFarPlane);
        const glm::mat4 view = glm::lookAt(camera.position, camera.position + camera.front(), camera.up());
        const glm::mat4 viewProj = projection * view;
        const Frustum frustum = Frustum::fromMatrix(viewProj);

        chunkManager.render(shaderProgram, viewProj, camera.position, frustum, chunkUniforms);

        // Render crosshair on top of everything
        crosshair.render(framebufferWidth, framebufferHeight);

        if (inputContext.showCoordinates)
        {
            std::ostringstream coordStream;
            coordStream.setf(std::ios::fixed, std::ios::floatfield);
            coordStream.precision(1);
            coordStream << 'X' << ' ' << camera.position.x << ' '
                         << 'Y' << ' ' << camera.position.y << ' '
                         << 'Z' << ' ' << camera.position.z;
            textOverlay.render(coordStream.str(), 8.0f, 8.0f, framebufferWidth, framebufferHeight, 8.0f, glm::vec3(1.0f));
        }

        // Render render distance GUI
        if (inputContext.showRenderDistanceGUI)
        {
            // Calculate center of screen for the GUI
            float centerX = framebufferWidth * 0.5f;
            float centerY = framebufferHeight * 0.5f;
            
            // Draw semi-transparent background (using multiple overlapping lines to create a filled rectangle effect)
            float boxWidth = 400.0f;
            float boxHeight = 100.0f;
            float boxLeft = centerX - boxWidth * 0.5f;
            float boxTop = centerY - boxHeight * 0.5f;
            
            // Draw prompt text
            std::string promptText = "Enter render distance:";
            textOverlay.render(promptText, boxLeft + 20.0f, boxTop + 20.0f, framebufferWidth, framebufferHeight, 8.0f, glm::vec3(1.0f));
            
            // Draw input text with cursor
            std::string inputText = inputContext.inputBuffer;
            if (static_cast<int>(glfwGetTime() * 2) % 2 == 0)  // Blinking cursor
            {
                inputText += "_";
            }
            textOverlay.render(inputText, boxLeft + 20.0f, boxTop + 50.0f, framebufferWidth, framebufferHeight, 10.0f, glm::vec3(0.5f, 1.0f, 0.5f));
        }

        glfwSwapBuffers(window);
    }

    chunkManager.clear();
    if (blockAtlas.id != 0)
    {
        glDeleteTextures(1, &blockAtlas.id);
    }
    glDeleteProgram(shaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}



