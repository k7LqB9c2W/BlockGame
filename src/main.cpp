#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "TextureLoader.h"
#include "camera.h"
#include "chunk_manager.h"
#include "input_context.h"
#include "renderer.h"
#include "text_overlay.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <deque>
#include <ctime>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <crtdbg.h>
#include <DbgHelp.h>
#endif


namespace
{
std::mutex gCrashLogMutex;
std::filesystem::path gCrashLogPath;

void appendCrashLog(std::string message);
#ifdef _WIN32
void appendStackTrace(EXCEPTION_POINTERS* exceptionPointers = nullptr);
void writeMiniDump(EXCEPTION_POINTERS* exceptionPointers);
int __cdecl crtReportHook(int reportType, char* message, int* returnValue);
#endif

void crashSignalHandler(int signalValue)
{
    const char* name = "unknown";
    switch (signalValue)
    {
        case SIGABRT:
            name = "SIGABRT";
            break;
#ifdef SIGSEGV
        case SIGSEGV:
            name = "SIGSEGV";
            break;
#endif
#ifdef SIGILL
        case SIGILL:
            name = "SIGILL";
            break;
#endif
#ifdef SIGFPE
        case SIGFPE:
            name = "SIGFPE";
            break;
#endif
#ifdef SIGTERM
        case SIGTERM:
            name = "SIGTERM";
            break;
#endif
    }
    appendCrashLog(std::string("signal: ") + name);
#ifdef _WIN32
    appendStackTrace();
    writeMiniDump(nullptr);
#endif
    std::_Exit(EXIT_FAILURE);
}

void appendCrashLog(std::string message)
{
    if (gCrashLogPath.empty())
    {
        return;
    }

    std::lock_guard<std::mutex> lock(gCrashLogMutex);

    std::ofstream out(gCrashLogPath, std::ios::app);
    if (!out)
    {
        return;
    }

    const auto now = std::chrono::system_clock::now();
    const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
    std::tm timeInfo{};
#ifdef _WIN32
    localtime_s(&timeInfo, &timestamp);
#else
    if (std::tm* local = std::localtime(&timestamp))
    {
        timeInfo = *local;
    }
#endif
    out << std::put_time(&timeInfo, "%Y-%m-%d %H:%M:%S") << " - " << message << '\n';
    out.flush();
}

#ifdef _WIN32
void appendStackTrace(EXCEPTION_POINTERS* exceptionPointers)
{
    constexpr USHORT kMaxFrames = 64;
    void* stack[kMaxFrames]{};
    const USHORT captured = CaptureStackBackTrace(0, kMaxFrames, stack, nullptr);

    std::ostringstream oss;
    oss << "stack:";
    for (USHORT i = 0; i < captured; ++i)
    {
        const auto address = reinterpret_cast<std::uintptr_t>(stack[i]);
        oss << "\n  [" << i << "] 0x" << std::hex << address << std::dec;
    }

    if (exceptionPointers && exceptionPointers->ExceptionRecord)
    {
        oss << "\n  exception code: 0x" << std::hex
            << static_cast<std::uint32_t>(exceptionPointers->ExceptionRecord->ExceptionCode) << std::dec;
    }

    appendCrashLog(oss.str());
}

void writeMiniDump(EXCEPTION_POINTERS* exceptionPointers)
{
    std::filesystem::path dumpPath;
    if (!gCrashLogPath.empty())
    {
        dumpPath = gCrashLogPath.parent_path() / "blockgame_crash.dmp";
    }
    else
    {
        std::error_code ec;
        dumpPath = std::filesystem::current_path(ec);
        if (ec)
        {
            return;
        }
        dumpPath /= "blockgame_crash.dmp";
    }

    HMODULE dbgHelp = LoadLibraryW(L"DbgHelp.dll");
    if (!dbgHelp)
    {
        appendCrashLog("minidump: failed to load DbgHelp.dll");
        return;
    }

    using MiniDumpWriteDumpFn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, MINIDUMP_TYPE,
                                              CONST PMINIDUMP_EXCEPTION_INFORMATION,
                                              CONST PMINIDUMP_USER_STREAM_INFORMATION,
                                              CONST PMINIDUMP_CALLBACK_INFORMATION);

    auto miniDumpWriteDump = reinterpret_cast<MiniDumpWriteDumpFn>(GetProcAddress(dbgHelp, "MiniDumpWriteDump"));
    if (!miniDumpWriteDump)
    {
        appendCrashLog("minidump: MiniDumpWriteDump not available");
        FreeLibrary(dbgHelp);
        return;
    }

    HANDLE file = CreateFileW(dumpPath.c_str(),
                              GENERIC_WRITE,
                              FILE_SHARE_READ,
                              nullptr,
                              CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL,
                              nullptr);
    if (file == INVALID_HANDLE_VALUE)
    {
        appendCrashLog("minidump: failed to create dump file");
        FreeLibrary(dbgHelp);
        return;
    }

    MINIDUMP_EXCEPTION_INFORMATION info{};
    info.ThreadId = GetCurrentThreadId();
    info.ExceptionPointers = exceptionPointers;
    info.ClientPointers = FALSE;

    const MINIDUMP_TYPE dumpType = static_cast<MINIDUMP_TYPE>(MiniDumpWithIndirectlyReferencedMemory | MiniDumpScanMemory);
    const BOOL dumpResult = miniDumpWriteDump(GetCurrentProcess(),
                                              GetCurrentProcessId(),
                                              file,
                                              dumpType,
                                              exceptionPointers ? &info : nullptr,
                                              nullptr,
                                              nullptr);
    CloseHandle(file);
    FreeLibrary(dbgHelp);

    appendCrashLog(dumpResult ? "minidump: written to blockgame_crash.dmp"
                              : "minidump: MiniDumpWriteDump failed");
}

int __cdecl crtReportHook(int reportType, char* message, int*)
{
    const char* text = message ? message : "<null>";
    appendCrashLog(std::string("CRT report[") + std::to_string(reportType) + "]: " + text);
    return FALSE; // allow default processing
}
#endif

void initializeCrashLogging(const std::filesystem::path& logPath)
{
    gCrashLogPath = logPath;

    // Ensure the log file exists so later appends succeed even if the program dies immediately.
    {
        std::ofstream out(gCrashLogPath, std::ios::app);
    }

    std::signal(SIGABRT, crashSignalHandler);
#ifdef SIGSEGV
    std::signal(SIGSEGV, crashSignalHandler);
#endif
#ifdef SIGILL
    std::signal(SIGILL, crashSignalHandler);
#endif
#ifdef SIGFPE
    std::signal(SIGFPE, crashSignalHandler);
#endif
#ifdef SIGTERM
    std::signal(SIGTERM, crashSignalHandler);
#endif

    std::set_terminate([]
    {
        if (auto current = std::current_exception())
        {
            try
            {
                std::rethrow_exception(current);
            }
            catch (const std::exception& e)
            {
                appendCrashLog(std::string("terminate: ") + e.what());
            }
            catch (...)
            {
                appendCrashLog("terminate: unknown exception");
            }
        }
        else
        {
            appendCrashLog("terminate: no active exception");
        }

#ifdef _WIN32
        appendStackTrace();
        writeMiniDump(nullptr);
#endif
        std::abort();
    });

#ifdef _WIN32
    _CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, crtReportHook);

    SetUnhandledExceptionFilter([](EXCEPTION_POINTERS* info) -> LONG
    {
        appendCrashLog("SEH crash");
        appendStackTrace(info);
        writeMiniDump(info);
        return EXCEPTION_EXECUTE_HANDLER;
    });

    // Avoid CRT abort dialog swallowing the process without logging.
#ifdef _DEBUG
    _set_abort_behavior(0, _CALL_REPORTFAULT);
#endif
#endif
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

void runStreamingValidationScenarios(ChunkManager& chunkManager, const glm::vec3& basePosition)
{
    std::cout << "Running streaming validation scenarios..." << std::endl;
    const std::array<glm::vec3, 6> offsets = {
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, static_cast<float>(kChunkSizeY * 6), 0.0f),
        glm::vec3(0.0f, static_cast<float>(-kChunkSizeY * 4), 0.0f),
        glm::vec3(static_cast<float>(kChunkSizeX * 3), static_cast<float>(kChunkSizeY * 2), static_cast<float>(kChunkSizeZ * 3)),
        glm::vec3(0.0f, static_cast<float>(kChunkSizeY * 12), 0.0f),
        glm::vec3(0.0f, static_cast<float>(-kChunkSizeY * 8), 0.0f)
    };

    for (const glm::vec3& offset : offsets)
    {
        const glm::vec3 target = basePosition + offset;
        std::cout << "  Probing stream at (" << target.x << ", " << target.y << ", " << target.z << ")" << std::endl;
        chunkManager.update(target);
        ChunkProfilingSnapshot sweep = chunkManager.sampleProfilingSnapshot();
        std::cout << "    Stream vertical radius " << sweep.verticalRadius
                  << ", uploads " << sweep.uploadedChunks
                  << " (deferrals: " << sweep.deferredUploads << ")" << std::endl;
    }

    chunkManager.update(basePosition);
    chunkManager.sampleProfilingSnapshot();
}


#include "text_overlay.h"

// Collision detection helper functions
struct AABB
{
    glm::vec3 min;
    glm::vec3 max;
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

int runGame()
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

    if (std::getenv("BLOCKGAME_STREAMING_TEST"))
    {
        runStreamingValidationScenarios(chunkManager, camera.position);
        chunkManager.update(camera.position);
    }

    Crosshair crosshair;
    TextOverlay textOverlay;

    constexpr double kFixedTimeStep = 1.0 / 60.0;
    double previousTime = glfwGetTime();
    double accumulator = 0.0;
    double fpsTimer = 0.0;
    int fpsFrameCount = 0;
    double fpsValue = 0.0;
#ifndef NDEBUG
    double profilingOverlayTimer = 0.0;
    std::string profilingOverlayText;
#endif
    std::cout << "Controls: WASD to move, mouse to look, SPACE to jump, N to set render distance, F2 to teleport, left-click to destroy blocks, right-click to place blocks, ESC to quit." << std::endl;

    while (!glfwWindowShouldClose(window))
    {
        const double currentTime = glfwGetTime();
        double frameTime = currentTime - previousTime;
        previousTime = currentTime;
        frameTime = std::min(frameTime, 0.25);
        accumulator += frameTime;
        fpsTimer += frameTime;
        ++fpsFrameCount;
        if (fpsTimer >= 1.0)
        {
            if (fpsTimer > 0.0)
            {
                fpsValue = static_cast<double>(fpsFrameCount) / fpsTimer;
            }
            fpsTimer = 0.0;
            fpsFrameCount = 0;
        }
#ifndef NDEBUG
        profilingOverlayTimer += frameTime;

        if (profilingOverlayTimer >= 1.0)
        {
            ChunkProfilingSnapshot snapshot = chunkManager.sampleProfilingSnapshot();
            std::ostringstream profilingStream;
            profilingStream.setf(std::ios::fixed, std::ios::floatfield);
            profilingStream << std::setprecision(2);

            const double uploadedKiB = static_cast<double>(snapshot.uploadedBytes) / 1024.0;
            profilingStream << "Gen " << snapshot.generatedChunks;
            if (snapshot.generatedChunks > 0)
            {
                profilingStream << " @" << snapshot.averageGenerationMs << "ms";
            }
            profilingStream << " | Mesh " << snapshot.meshedChunks;
            if (snapshot.meshedChunks > 0)
            {
                profilingStream << " @" << snapshot.averageMeshingMs << "ms";
            }
            profilingStream << " | Upload " << snapshot.uploadedChunks << " (" << uploadedKiB << " KiB)";
            if (snapshot.throttledUploads > 0)
            {
                profilingStream << " Throttle " << snapshot.throttledUploads;
            }
            if (snapshot.deferredUploads > 0)
            {
                profilingStream << " Def " << snapshot.deferredUploads;
            }
            if (snapshot.evictedChunks > 0)
            {
                profilingStream << " Evict " << snapshot.evictedChunks;
            }

            if (snapshot.workerThreads > 0)
            {
                profilingStream << " | Workers " << snapshot.workerThreads;
            }

            const int verticalSpan = (snapshot.verticalRadius * 2 + 1) * kChunkSizeY;
            profilingStream << " | View " << chunkManager.viewDistance()
                            << "x" << snapshot.verticalRadius
                            << " (" << verticalSpan << "h)";

            profilingOverlayText = profilingStream.str();
            profilingOverlayTimer = 0.0;
        }
#endif

        glfwPollEvents();

        bool f1CurrentlyPressed = (glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS);
        bool f1JustPressed = f1CurrentlyPressed && !inputContext.f1Pressed;
        if (f1JustPressed)
        {
            inputContext.showDebugOverlay = !inputContext.showDebugOverlay;
        }
        inputContext.f1JustPressed = f1JustPressed;
        inputContext.f1Pressed = f1CurrentlyPressed;

        bool f3CurrentlyPressed = (glfwGetKey(window, GLFW_KEY_F3) == GLFW_PRESS);
        bool f3JustPressed = f3CurrentlyPressed && !inputContext.f3Pressed;
        if (f3JustPressed)
        {
            inputContext.lodEnabled = !inputContext.lodEnabled;
            chunkManager.setLodEnabled(inputContext.lodEnabled);
        }
        inputContext.f3JustPressed = f3JustPressed;
        inputContext.f3Pressed = f3CurrentlyPressed;

        // Only close window with ESC if GUI is not active
        // (ESC to close GUI is handled in computePlayerInputState)
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS &&
            !inputContext.showRenderDistanceGUI && !inputContext.showTeleportGUI)
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
        if (!inputContext.showRenderDistanceGUI && !inputContext.showTeleportGUI && inputContext.leftMouseJustPressed)
        {
            RaycastHit hit = chunkManager.raycast(camera.position, camera.front());
            if (hit.hit)
            {
                chunkManager.destroyBlock(hit.blockPos);
            }
            inputContext.leftMouseJustPressed = false; // Reset the flag
        }
        else
        {
            inputContext.leftMouseJustPressed = false;
        }

        // Handle block placement
        if (!inputContext.showRenderDistanceGUI && !inputContext.showTeleportGUI && inputContext.rightMouseJustPressed)
        {
            RaycastHit hit = chunkManager.raycast(camera.position, camera.front());
            if (hit.hit)
            {
                chunkManager.placeBlock(hit.blockPos, hit.faceNormal);
            }
            inputContext.rightMouseJustPressed = false; // Reset the flag
        }
        else
        {
            inputContext.rightMouseJustPressed = false;
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

        const ChunkRenderData renderData = chunkManager.buildRenderData(frustum);
        renderWorldGeometry(shaderProgram, viewProj, camera.position, chunkUniforms, renderData);

        // Render crosshair on top of everything
        crosshair.render(framebufferWidth, framebufferHeight);

        const double currentFpsEstimate = (fpsFrameCount > 0 && fpsTimer > 0.0)
                                              ? static_cast<double>(fpsFrameCount) / fpsTimer
                                              : fpsValue;
        const float overlayMargin = 12.0f;
        const float debugFontSize = 20.0f;
        std::string debugOverlayText;
        int debugOverlayLineCount = 0;

        if (inputContext.showDebugOverlay)
        {
            std::ostringstream debugStream;
            debugStream.setf(std::ios::fixed, std::ios::floatfield);
            debugStream << "FPS: " << std::setprecision(0) << currentFpsEstimate << '\n';
            debugStream << std::setprecision(1);
            debugStream << "XYZ: " << camera.position.x << ", "
                        << camera.position.y << ", "
                        << camera.position.z << '\n';
            debugStream << "Biome: " << chunkManager.biomeNameAt(camera.position);

            debugOverlayText = debugStream.str();
            debugOverlayLineCount = 1 + static_cast<int>(std::count(debugOverlayText.begin(),
                                                                    debugOverlayText.end(),
                                                                    '\n'));

            textOverlay.render(debugOverlayText,
                               overlayMargin,
                               overlayMargin,
                               framebufferWidth,
                               framebufferHeight,
                               debugFontSize,
                               glm::vec3(1.0f));
        }

#ifndef NDEBUG
        if (!profilingOverlayText.empty())
        {
            float overlayY = overlayMargin;
            if (debugOverlayLineCount > 0)
            {
                float lineHeight = textOverlay.lineHeight(debugFontSize);
                if (lineHeight <= 0.0f)
                {
                    lineHeight = debugFontSize * 1.2f;
                }
                overlayY += lineHeight * static_cast<float>(debugOverlayLineCount) + 8.0f;
            }

            textOverlay.render(profilingOverlayText,
                               overlayMargin,
                               overlayY,
                               framebufferWidth,
                               framebufferHeight,
                               16.0f,
                               glm::vec3(0.85f, 0.95f, 1.0f));
        }
#endif

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
            std::string promptText = "Enter render distance (1-48):";
            textOverlay.render(promptText, boxLeft + 20.0f, boxTop + 20.0f, framebufferWidth, framebufferHeight, 8.0f, glm::vec3(1.0f));

            // Draw input text with cursor
            std::string inputText = inputContext.inputBuffer;
            if (static_cast<int>(glfwGetTime() * 2) % 2 == 0)  // Blinking cursor
            {
                inputText += "_";
            }
            textOverlay.render(inputText, boxLeft + 20.0f, boxTop + 50.0f, framebufferWidth, framebufferHeight, 10.0f, glm::vec3(0.5f, 1.0f, 0.5f));
        }

        if (inputContext.showTeleportGUI)
        {
            float centerX = framebufferWidth * 0.5f;
            float centerY = framebufferHeight * 0.5f;

            float boxWidth = 400.0f;
            float boxHeight = 120.0f;
            float boxLeft = centerX - boxWidth * 0.5f;
            float boxTop = centerY - boxHeight * 0.5f;

            std::string promptText = "Enter teleport target (x y z):";
            textOverlay.render(promptText,
                               boxLeft + 20.0f,
                               boxTop + 20.0f,
                               framebufferWidth,
                               framebufferHeight,
                               8.0f,
                               glm::vec3(1.0f));

            std::string inputText = inputContext.teleportBuffer;
            if (static_cast<int>(glfwGetTime() * 2) % 2 == 0)
            {
                inputText += "_";
            }
            textOverlay.render(inputText,
                               boxLeft + 20.0f,
                               boxTop + 55.0f,
                               framebufferWidth,
                               framebufferHeight,
                               10.0f,
                               glm::vec3(0.5f, 0.8f, 1.0f));
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

} // namespace

int main(int argc, char** argv)
{
    std::filesystem::path exePath;
    if (argc > 0 && argv[0] != nullptr)
    {
        std::error_code ec;
        exePath = std::filesystem::canonical(argv[0], ec);
        if (ec)
        {
            exePath = std::filesystem::absolute(argv[0], ec);
            if (ec)
            {
                exePath.clear();
            }
        }
    }

    std::filesystem::path exeDirectory;
    std::error_code dirEc;
    const bool pathIsDirectory = std::filesystem::is_directory(exePath, dirEc);
    if (exePath.empty() || (!dirEc && pathIsDirectory))
    {
        exeDirectory = exePath;
    }
    else
    {
        exeDirectory = exePath.parent_path();
    }

    if (exeDirectory.empty())
    {
        exeDirectory = std::filesystem::current_path();
    }

    std::filesystem::path logPath = exeDirectory / "blockgame_crash.log";

#ifndef NDEBUG
    std::cout << "Crash log path: " << logPath << '\n';
#endif

    initializeCrashLogging(logPath);

    try
    {
        return runGame();
    }
    catch (const std::exception& e)
    {
        appendCrashLog(std::string("uncaught exception: ") + e.what());
        std::cerr << "Unhandled exception: " << e.what() << '\n';
    }
    catch (...)
    {
        appendCrashLog("uncaught exception: unknown exception");
        std::cerr << "Unhandled non-standard exception" << std::endl;
    }

    return EXIT_FAILURE;
}
