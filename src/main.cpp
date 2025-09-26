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
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr float kNearPlane = 0.1f;
constexpr float kFarPlane = 150.0f;
constexpr float kEpsilon = 1e-6f;
constexpr float kCameraEyeHeight = 1.7f;

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
};

struct InstanceData
{
    glm::vec3 translation;
    float tint;
};

class Camera
{
public:
    glm::vec3 position{0.0f, 2.0f, 5.0f};
    float yaw{-90.0f};
    float pitch{0.0f};
    float moveSpeed{6.5f};
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

struct WorldData
{
    std::vector<InstanceData> instances;
    std::vector<float> columnHeights;
    int gridRadius{0};
    int gridDiameter{0};
    float baseElevation{1.0f};

    [[nodiscard]] float columnTop(int x, int z) const noexcept
    {
        if (std::abs(x) > gridRadius || std::abs(z) > gridRadius)
        {
            return baseElevation;
        }

        const int idx = (x + gridRadius) * gridDiameter + (z + gridRadius);
        return columnHeights[idx];
    }

    [[nodiscard]] float sampleHeight(const glm::vec3& position) const noexcept
    {
        const int cellX = static_cast<int>(std::floor(position.x + 0.5f));
        const int cellZ = static_cast<int>(std::floor(position.z + 0.5f));
        return columnTop(cellX, cellZ);
    }
};

WorldData generateWorld(int gridRadius, int maxHeight, unsigned seed)
{
    WorldData world;
    world.gridRadius = gridRadius;
    world.gridDiameter = 2 * gridRadius + 1;
    const std::size_t cellCount = static_cast<std::size_t>(world.gridDiameter) * world.gridDiameter;
    world.columnHeights.assign(cellCount, 0.0f);

    std::mt19937 rng(seed);
    const float baseHeight = std::max(1.0f, std::min(2.2f, static_cast<float>(maxHeight) - 1.0f));
    world.baseElevation = baseHeight;

    struct Hill
    {
        glm::vec2 center;
        float radius;
        float amplitude;
    };

    std::vector<Hill> hills;
    const int hillCount = std::max(2, gridRadius / 3);
    std::uniform_real_distribution<float> centerDist(static_cast<float>(-gridRadius), static_cast<float>(gridRadius));
    std::uniform_real_distribution<float> radiusDist(std::max(4.0f, gridRadius * 0.4f), static_cast<float>(gridRadius));
    std::uniform_real_distribution<float> amplitudeDist(0.6f, std::max(1.2f, static_cast<float>(maxHeight) - baseHeight));

    hills.reserve(static_cast<std::size_t>(hillCount));
    for (int i = 0; i < hillCount; ++i)
    {
        Hill hill{};
        hill.center = glm::vec2(centerDist(rng), centerDist(rng));
        hill.radius = radiusDist(rng);
        hill.amplitude = amplitudeDist(rng);
        hills.push_back(hill);
    }

    std::uniform_real_distribution<float> jitterDist(-0.2f, 0.35f);

    world.instances.reserve(cellCount * static_cast<std::size_t>(maxHeight));

    for (int x = -gridRadius; x <= gridRadius; ++x)
    {
        for (int z = -gridRadius; z <= gridRadius; ++z)
        {
            float height = baseHeight + jitterDist(rng);
            for (const Hill& hill : hills)
            {
                const float dx = static_cast<float>(x) - hill.center.x;
                const float dz = static_cast<float>(z) - hill.center.y;
                const float dist = std::sqrt(dx * dx + dz * dz);
                if (dist < hill.radius)
                {
                    const float t = 1.0f - (dist / hill.radius);
                    height += hill.amplitude * (t * t);
                }
            }

            height = std::clamp(height, 1.0f, static_cast<float>(maxHeight));
            int columnHeight = std::max(1, static_cast<int>(std::round(height)));
            columnHeight = std::min(columnHeight, maxHeight);

            const std::size_t idx = static_cast<std::size_t>(x + gridRadius) * world.gridDiameter + (z + gridRadius);
            world.columnHeights[idx] = static_cast<float>(columnHeight);

            for (int y = 0; y < columnHeight; ++y)
            {
                InstanceData instance{};
                instance.translation = glm::vec3(static_cast<float>(x), static_cast<float>(y) + 0.5f, static_cast<float>(z));
                instance.tint = std::uniform_real_distribution<float>(-0.08f, 0.18f)(rng);
                world.instances.push_back(instance);
            }
        }
    }

    return world;
}

GLuint createCubeBuffers(GLuint& instanceBuffer, const std::vector<InstanceData>& instances)
{
    static const std::array<Vertex, 36> vertices{
        Vertex{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
        Vertex{{0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
        Vertex{{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
        Vertex{{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
        Vertex{{-0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
        Vertex{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},

        Vertex{{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        Vertex{{0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        Vertex{{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        Vertex{{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        Vertex{{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        Vertex{{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},

        Vertex{{-0.5f, 0.5f, 0.5f}, {-1.0f, 0.0f, 0.0f}},
        Vertex{{-0.5f, 0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}},
        Vertex{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}},
        Vertex{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}},
        Vertex{{-0.5f, -0.5f, 0.5f}, {-1.0f, 0.0f, 0.0f}},
        Vertex{{-0.5f, 0.5f, 0.5f}, {-1.0f, 0.0f, 0.0f}},

        Vertex{{0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
        Vertex{{0.5f, 0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        Vertex{{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        Vertex{{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        Vertex{{0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
        Vertex{{0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},

        Vertex{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}},
        Vertex{{0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}},
        Vertex{{0.5f, -0.5f, 0.5f}, {0.0f, -1.0f, 0.0f}},
        Vertex{{0.5f, -0.5f, 0.5f}, {0.0f, -1.0f, 0.0f}},
        Vertex{{-0.5f, -0.5f, 0.5f}, {0.0f, -1.0f, 0.0f}},
        Vertex{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}},

        Vertex{{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
        Vertex{{-0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
        Vertex{{0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
        Vertex{{0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
        Vertex{{0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
        Vertex{{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    };

    GLuint vao = 0;
    GLuint vbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(Vertex) * vertices.size()), vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));

    glGenBuffers(1, &instanceBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, instanceBuffer);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(InstanceData) * instances.size()), instances.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(InstanceData), reinterpret_cast<void*>(offsetof(InstanceData, translation)));
    glVertexAttribDivisor(2, 1);

    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(InstanceData), reinterpret_cast<void*>(offsetof(InstanceData, tint)));
    glVertexAttribDivisor(3, 1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return vao;
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

    try
    {
        const char* vertexShaderSrc = R"(#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aOffset;
layout (location = 3) in float aTint;

uniform mat4 uViewProj;

out vec3 vNormal;
out vec3 vWorldPos;
out float vTint;

void main()
{
    vec3 worldPos = aPos + aOffset;
    vWorldPos = worldPos;
    vNormal = aNormal;
    vTint = aTint;
    gl_Position = uViewProj * vec4(worldPos, 1.0);
}
)";

        const char* fragmentShaderSrc = R"(#version 330 core
out vec4 FragColor;

in vec3 vNormal;
in vec3 vWorldPos;
in float vTint;

uniform vec3 uLightDir;
uniform vec3 uBaseColor;
uniform vec3 uCameraPos;

void main()
{
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(uCameraPos - vWorldPos);
    vec3 lightDir = normalize(-uLightDir);

    float diff = max(dot(normal, lightDir), 0.0);
    float ambient = 0.35;
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);

    vec3 base = clamp(uBaseColor + vec3(vTint), 0.0, 1.0);
    vec3 color = base * (ambient + diff) + vec3(0.15f) * spec;
    FragColor = vec4(color, 1.0);
}
)";

        GLuint shaderProgram = createProgram(vertexShaderSrc, fragmentShaderSrc);

        WorldData world = generateWorld(18, 6, 1337u);
        GLuint instanceBuffer = 0;
        GLuint cubeVAO = createCubeBuffers(instanceBuffer, world.instances);

        const GLint viewProjLocation = glGetUniformLocation(shaderProgram, "uViewProj");
        const GLint lightDirLocation = glGetUniformLocation(shaderProgram, "uLightDir");
        const GLint baseColorLocation = glGetUniformLocation(shaderProgram, "uBaseColor");
        const GLint cameraPosLocation = glGetUniformLocation(shaderProgram, "uCameraPos");

        const glm::vec3 lightDirection = glm::normalize(glm::vec3(0.5f, -1.0f, 0.2f));
        const glm::vec3 blockBaseColor = glm::vec3(0.24f, 0.62f, 0.28f);

        camera.position.y = world.sampleHeight(camera.position) + kCameraEyeHeight;

        float lastFrame = static_cast<float>(glfwGetTime());

        std::cout << "Controls: WASD to move, mouse to look, ESC to quit." << std::endl;

        while (!glfwWindowShouldClose(window))
        {
            const float currentFrame = static_cast<float>(glfwGetTime());
            const float deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;

            processInput(window, camera, deltaTime);
            camera.position.y = world.sampleHeight(camera.position) + kCameraEyeHeight;

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

            glUseProgram(shaderProgram);
            glUniformMatrix4fv(viewProjLocation, 1, GL_FALSE, glm::value_ptr(viewProj));
            glUniform3fv(lightDirLocation, 1, glm::value_ptr(lightDirection));
            glUniform3fv(baseColorLocation, 1, glm::value_ptr(blockBaseColor));
            glUniform3fv(cameraPosLocation, 1, glm::value_ptr(camera.position));

            glBindVertexArray(cubeVAO);
            glDrawArraysInstanced(GL_TRIANGLES, 0, 36, static_cast<GLsizei>(world.instances.size()));
            glBindVertexArray(0);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        glDeleteVertexArrays(1, &cubeVAO);
        glDeleteBuffers(1, &instanceBuffer);
        glUseProgram(0);
        glDeleteProgram(shaderProgram);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
