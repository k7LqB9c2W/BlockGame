#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "TextureLoader.h"
#include "chunk_manager.h"

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


namespace
{
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



