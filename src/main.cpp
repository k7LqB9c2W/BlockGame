#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "camera.h"
#include "chunk_manager.h"
#include "input_context.h"
#include "renderer.h"
#include "terrain/terrain_generator.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

bool applyRenderDistanceInput(ChunkManager& chunkManager, const std::string& input)
{
    if (input.empty())
    {
        return false;
    }

    std::string normalized = input;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream stream(normalized);
    int nearDistance = 0;
    int farDistance = chunkManager.farRenderDistanceBlocks();
    if (!(stream >> nearDistance))
    {
        return false;
    }

    if (stream >> farDistance)
    {
        chunkManager.setFarRenderDistanceBlocks(farDistance);
    }
    chunkManager.setNearRenderDistance(nearDistance);
    return true;
}

bool applyTeleportInput(Camera& camera, const std::string& input)
{
    if (input.empty())
    {
        return false;
    }

    std::string normalized = input;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream stream(normalized);
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    if (!(stream >> x >> y >> z))
    {
        return false;
    }

    camera.position = glm::vec3(x, y, z);
    camera.velocity = glm::vec3(0.0f);
    camera.onGround = false;
    return true;
}

void drawCrosshairOverlay(int framebufferWidth, int framebufferHeight)
{
    const ImVec2 center(static_cast<float>(framebufferWidth) * 0.5f,
                        static_cast<float>(framebufferHeight) * 0.5f);
    constexpr float crosshairSize = 8.0f;
    constexpr float thickness = 2.0f;

    ImDrawList* drawList = ImGui::GetForegroundDrawList();
    drawList->AddLine(ImVec2(center.x - crosshairSize, center.y),
                      ImVec2(center.x + crosshairSize, center.y),
                      IM_COL32(255, 255, 255, 220),
                      thickness);
    drawList->AddLine(ImVec2(center.x, center.y - crosshairSize),
                      ImVec2(center.x, center.y + crosshairSize),
                      IM_COL32(255, 255, 255, 220),
                      thickness);
}

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

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    constexpr int kInitialWidth = 1280;
    constexpr int kInitialHeight = 720;

    GLFWwindow* window = glfwCreateWindow(kInitialWidth, kInitialHeight, "BlockGame", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return EXIT_FAILURE;
    }

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
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    Renderer renderer;
    try
    {
        renderer.initialize(window, kInitialWidth, kInitialHeight);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Failed to initialize D3D12 renderer: " << ex.what() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    EnvironmentState environment{};

    LoadedTexture blockAtlas;
    try
    {
        blockAtlas = renderer.loadTexture("block_atlas.png");
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Failed to load block atlas: " << ex.what() << std::endl;
        renderer.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    ChunkManager chunkManager(1337u);
    chunkManager.initializeRendering(renderer.device());
    chunkManager.setBlockTextureAtlasConfig(blockAtlas.size, kAtlasTileSizePixels); // Map block faces to atlas tiles.
    inputContext.lodEnabled = chunkManager.farTerrainEnabled();
    
    // Find a guaranteed safe spawn position above ground
    std::cout << "Finding safe spawn position..." << std::endl;
    camera.position = chunkManager.findSafeSpawnPosition(camera.position.x, camera.position.z);
    camera.velocity = glm::vec3(0.0f);
    camera.onGround = false;

    std::cout << "Player spawned at: (" << camera.position.x << ", " << camera.position.y << ", " << camera.position.z << ")" << std::endl;
    chunkManager.beginSpawnPreload(camera.position);

    if (std::getenv("BLOCKGAME_STREAMING_TEST"))
    {
        runStreamingValidationScenarios(chunkManager, camera.position);
        chunkManager.update(camera.position, camera.front());
    }

    constexpr double kFixedTimeStep = 1.0 / 60.0;
    double previousTime = glfwGetTime();
    double accumulator = 0.0;
    double fpsTimer = 0.0;
    int fpsFrameCount = 0;
    double fpsValue = 0.0;
    double loadingOverlayTimer = 0.0;
    std::string loadingOverlayText;
    double profilingOverlayTimer = 0.0;
    std::string profilingOverlayText;
    std::cout << "Controls: WASD to move, mouse to look, . to toggle mouse/UI control, SPACE to jump, N to set near/far render distance, F2 to teleport, F3 to toggle far terrain, left-click to destroy blocks, right-click to place blocks, ESC to quit." << std::endl;

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
        profilingOverlayTimer += frameTime;

        if (profilingOverlayTimer >= 1.0)
        {
            ChunkProfilingSnapshot snapshot = chunkManager.sampleProfilingSnapshot();
            const RendererProfilingSnapshot rendererSnapshot = renderer.profilingSnapshot();
            const char* phaseName = "steady";
            switch (snapshot.phase)
            {
            case StreamingPhase::SpawnResolve:
                phaseName = "spawn";
                break;
            case StreamingPhase::ExactPreload:
                phaseName = "preload";
                break;
            case StreamingPhase::InteractiveNearOnly:
                phaseName = "near";
                break;
            case StreamingPhase::FarRamp:
                phaseName = "ramp";
                break;
            case StreamingPhase::SteadyState:
            default:
                phaseName = "steady";
                break;
            }

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
            profilingStream << " | " << phaseName
                            << " Near " << chunkManager.nearRenderDistance()
                            << "x" << snapshot.verticalRadius
                            << " Far " << chunkManager.farRenderDistanceBlocks()
                            << " (" << verticalSpan << "h)";
            if (snapshot.exactChunksPending > 0 || snapshot.exactChunksReady > 0)
            {
                profilingStream << " | Exact " << snapshot.exactChunksReady
                                << " ready " << snapshot.exactChunksPending << " pending";
            }
            if (snapshot.farActiveTiles > 0 || snapshot.farDirtyTiles > 0)
            {
                profilingStream << " | FarTiles " << snapshot.farActiveTiles;
                if (snapshot.farDirtyTiles > 0)
                {
                    profilingStream << " dirty " << snapshot.farDirtyTiles;
                }
                if (snapshot.farShellTilesReady > 0)
                {
                    profilingStream << " ready " << snapshot.farShellTilesReady;
                }
                if (snapshot.farTilesBuilt > 0)
                {
                    profilingStream << " built " << snapshot.farTilesBuilt;
                }
                if (snapshot.farTilesQueued > 0)
                {
                    profilingStream << " q " << snapshot.farTilesQueued;
                }
            }
            profilingStream << " | UploadMs " << snapshot.uploadMsLastFrame;
            profilingStream << " | UpdateMs " << snapshot.updateMsLastFrame;
            if (snapshot.farCollectMsLastFrame > 0.0 || snapshot.farUploadMsLastFrame > 0.0)
            {
                profilingStream << " | FarCollect " << snapshot.farCollectMsLastFrame
                                << " FarUpload " << snapshot.farUploadMsLastFrame;
            }
            if (rendererSnapshot.atmosphereLutMs > 0.0 || rendererSnapshot.skyDrawMs > 0.0 ||
                rendererSnapshot.shadowDrawMs > 0.0 || rendererSnapshot.worldDrawMs > 0.0 ||
                rendererSnapshot.toneMapMs > 0.0)
            {
                profilingStream << " | Atmo " << rendererSnapshot.atmosphereLutMs
                                << " Sky " << rendererSnapshot.skyDrawMs
                                << " Shadow " << rendererSnapshot.shadowDrawMs
                                << " World " << rendererSnapshot.worldDrawMs
                                << " Tone " << rendererSnapshot.toneMapMs;
            }

            profilingOverlayText = profilingStream.str();
            profilingOverlayTimer = 0.0;
        }

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
            inputContext.lodEnabled = !chunkManager.farTerrainEnabled();
            chunkManager.setFarTerrainEnabled(inputContext.lodEnabled);
        }
        inputContext.f3JustPressed = f3JustPressed;
        inputContext.f3Pressed = f3CurrentlyPressed;

        const StreamingStatusSnapshot streamingStatus = chunkManager.streamingStatusSnapshot();
        const bool playerReleased = streamingStatus.playerReleaseReady;
        if (!playerReleased)
        {
            loadingOverlayTimer += frameTime;
            if (loadingOverlayTimer >= 0.2 || loadingOverlayText.empty())
            {
                const char* phaseName = "Loading";
                switch (streamingStatus.phase)
                {
                case StreamingPhase::SpawnResolve:
                    phaseName = "Resolving spawn";
                    break;
                case StreamingPhase::ExactPreload:
                    phaseName = "Preloading exact world";
                    break;
                case StreamingPhase::InteractiveNearOnly:
                    phaseName = "Stabilizing near world";
                    break;
                case StreamingPhase::FarRamp:
                    phaseName = "Streaming far terrain";
                    break;
                case StreamingPhase::SteadyState:
                default:
                    phaseName = "Streaming";
                    break;
                }

                std::ostringstream loadingStream;
                loadingStream << "Loading world...\n";
                loadingStream << phaseName << '\n';
                loadingStream << "Exact bubble: " << streamingStatus.exactReadyChunks
                              << " / " << streamingStatus.exactRequiredChunks << '\n';
                loadingStream << "Pending uploads: " << streamingStatus.exactPendingUploads << '\n';
                loadingStream << "Far tiles: " << streamingStatus.farReadyTiles
                              << " ready, " << streamingStatus.farQueuedTiles << " queued\n";
                loadingStream << streamingStatus.blockingReason;
                loadingOverlayText = loadingStream.str();
                loadingOverlayTimer = 0.0;
            }
        }
        else
        {
            loadingOverlayTimer = 0.0;
            loadingOverlayText.clear();
        }

        // Only allow ESC to quit while the game has mouse/camera capture.
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS &&
            isGameplayMouseCaptured(inputContext))
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        auto* inputContextPtr = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
        if (playerReleased)
        {
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
        }
        else
        {
            accumulator = 0.0;
        }

        if (playerReleased)
        {
            // Update block highlighting based on crosshair
            chunkManager.updateHighlight(camera.position, camera.front());

            if (isGameplayMouseCaptured(inputContext) && inputContext.leftMouseJustPressed)
            {
                RaycastHit hit = chunkManager.raycast(camera.position, camera.front());
                if (hit.hit)
                {
                    chunkManager.destroyBlock(hit.blockPos);
                }
                inputContext.leftMouseJustPressed = false;
            }

            if (isGameplayMouseCaptured(inputContext) && inputContext.rightMouseJustPressed)
            {
                RaycastHit hit = chunkManager.raycast(camera.position, camera.front());
                if (hit.hit)
                {
                    chunkManager.placeBlock(hit.blockPos, hit.faceNormal);
                }
                inputContext.rightMouseJustPressed = false;
            }
        }
        inputContext.leftMouseJustPressed = false;
        inputContext.rightMouseJustPressed = false;

        chunkManager.update(camera.position, camera.front());

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
        framebufferWidth = std::max(framebufferWidth, 1);
        framebufferHeight = std::max(framebufferHeight, 1);
        if (framebufferWidth != renderer.width() || framebufferHeight != renderer.height())
        {
            renderer.resize(framebufferWidth, framebufferHeight);
        }
        const float aspect = static_cast<float>(framebufferWidth) / static_cast<float>(framebufferHeight);

        const float currentFarPlane = computeFarPlaneForDistanceBlocks(chunkManager.farRenderDistanceBlocks());
        kFarPlane = currentFarPlane;
        const glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(60.0f), aspect, kNearPlane, currentFarPlane);
        const glm::mat4 view = glm::lookAt(camera.position, camera.position + camera.front(), camera.up());
        const glm::mat4 viewProj = projection * view;
        const Frustum frustum = Frustum::fromMatrix(viewProj);

        const auto updateEnvironment = [&]()
        {
            environment.fogStartBlocks = static_cast<float>(chunkManager.renderDistanceSettings().fogStartBlocks);
            environment.farDistanceBlocks = static_cast<float>(chunkManager.farRenderDistanceBlocks());

            const float dayAngle = ((environment.timeOfDay - 6.0f) / 24.0f) * glm::two_pi<float>();
            const float elevation = std::sin(dayAngle);
            const float azimuth = dayAngle - glm::half_pi<float>();
            const float horizontal = std::max(0.05f, std::cos(dayAngle));
            glm::vec3 sunDir{
                std::cos(azimuth) * horizontal,
                elevation,
                std::sin(azimuth) * horizontal
            };
            if (glm::length(sunDir) <= 1e-4f)
            {
                sunDir = glm::vec3(-0.35f, 0.9f, -0.2f);
            }
            environment.sunDirection = glm::normalize(sunDir);

            const float daylight = std::clamp(environment.sunDirection.y * 0.5f + 0.5f, 0.0f, 1.0f);
            const glm::vec3 sunriseTint{0.85f, 0.45f, 0.22f};
            const glm::vec3 middayTint{4.5f, 4.8f, 5.5f};
            environment.sunIlluminance = glm::mix(sunriseTint, middayTint, std::pow(daylight, 0.65f));
        };
        updateEnvironment();

        const glm::vec3 viewDirection = glm::normalize(camera.front());
        const float viewElevationDeg =
            glm::degrees(std::asin(std::clamp(viewDirection.y, -1.0f, 1.0f)));
        const float sunElevationDeg =
            glm::degrees(std::asin(std::clamp(environment.sunDirection.y, -1.0f, 1.0f)));
        const float sunViewDot =
            glm::dot(glm::normalize(environment.sunDirection), viewDirection);
        const float altitudeAboveGround =
            camera.position.y - chunkManager.surfaceHeight(camera.position.x, camera.position.z);
        const float fogSpanBlocks =
            std::max(environment.farDistanceBlocks - environment.fogStartBlocks, 0.0f);
        const bool nearHorizonView = std::abs(viewDirection.y) <= 0.08f;
        const bool lookingBelowHorizon = viewDirection.y < 0.0f;

        renderer.beginFrame(glm::vec4(0.10f, 0.16f, 0.26f, 1.0f));
        if (chunkManager.streamingPhase() != StreamingPhase::ExactPreload)
        {
            const WorldRenderData renderData = chunkManager.buildRenderData(frustum);
            renderer.renderWorld(renderData, view, projection, camera.position, blockAtlas, environment);
        }
        renderer.beginImGuiFrame();

        const double currentFpsEstimate = (fpsFrameCount > 0 && fpsTimer > 0.0)
                                              ? static_cast<double>(fpsFrameCount) / fpsTimer
                                              : fpsValue;
        std::string debugOverlayText;

        if (inputContext.showDebugOverlay)
        {
            std::ostringstream debugStream;
            debugStream.setf(std::ios::fixed, std::ios::floatfield);
            debugStream << "FPS: " << std::setprecision(0) << currentFpsEstimate << '\n';
            debugStream << std::setprecision(1);
            debugStream << "XYZ: " << camera.position.x << ", "
                        << camera.position.y << ", "
                        << camera.position.z << '\n';
            debugStream << "Biome: " << chunkManager.biomeNameAt(camera.position) << '\n';

            const RaycastHit debugHit = chunkManager.raycast(camera.position, camera.front());
            glm::vec3 samplePosition = camera.position;
            if (debugHit.hit)
            {
                samplePosition = glm::vec3(debugHit.blockPos);
            }

            const int columnX = static_cast<int>(std::floor(samplePosition.x));
            const int columnZ = static_cast<int>(std::floor(samplePosition.z));
            const terrain::ColumnSample columnSample = chunkManager.sampleColumnAt(samplePosition);

            debugStream << std::setprecision(2);
            const terrain::BiomeDefinition* dominantBiome = columnSample.dominantBiome;
            const char* dominantName = dominantBiome ? dominantBiome->name.c_str() : "(none)";
            debugStream << "Column [" << columnX << ", " << columnZ << "]\n";
            debugStream << "Dominant: "
                        << dominantName
                        << " (w=" << columnSample.dominantWeight << ")\n";

            debugStream << "SurfaceY: " << columnSample.surfaceY
                        << " h=" << columnSample.surfaceHeight
                        << " min=" << columnSample.minSurfaceY
                        << " max=" << columnSample.maxSurfaceY;
            if (columnSample.slabHasSolid && columnSample.slabHighestSolidY != std::numeric_limits<int>::min())
            {
                debugStream << " solidTop=" << columnSample.slabHighestSolidY;
            }
            debugStream << '\n';

            debugStream << "Soil: coeff=" << columnSample.soilCreepCoefficient
                        << " offset=" << columnSample.soilCreepOffset
                        << " originalY=" << columnSample.originalSurfaceY << '\n';

            debugStream << "Amplitude: rough=" << columnSample.roughAmplitude
                        << " hill=" << columnSample.hillAmplitude
                        << " mountain=" << columnSample.mountainAmplitude
                        << " shoreDist=" << columnSample.distanceToShore
                        << " domOcean=" << (columnSample.dominantIsOcean ? "true" : "false")
                        << " coastDist=";
            if (std::isfinite(columnSample.distanceToCoast))
            {
                debugStream << columnSample.distanceToCoast;
            }
            else
            {
                debugStream << "inf";
            }
            debugStream << '\n';

            debugStream << "Climate blend:";
            if (columnSample.topBlendCount == 0)
            {
                debugStream << " none\n";
            }
            else
            {
                debugStream << '\n';
                for (std::size_t i = 0; i < columnSample.topBlendCount; ++i)
                {
                    const auto& blend = columnSample.topBlendDebug[i];
                    const terrain::BiomeDefinition* blendBiome = blend.biome;
                    const char* blendName = blendBiome ? blendBiome->name.c_str() : "(none)";
                    debugStream << "  - "
                                << blendName
                                << " w=" << blend.weight
                                << " aggY=" << blend.aggregatedHeight
                                << " normDist=" << blend.normalizedDistance
                                << " radius=" << blend.seedRadius
                                << " worldDist=" << blend.worldDistance
                                << " ocean=" << (blend.isOcean ? "true" : "false")
                                << '\n';
                }
            }

            debugOverlayText = debugStream.str();
        }

        if (inputContext.showDebugOverlay && !debugOverlayText.empty())
        {
            ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.35f);
            ImGui::Begin("Debug Overlay",
                         nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs);
            ImGui::TextUnformatted(debugOverlayText.c_str());
            ImGui::End();
        }

        if (inputContext.showDebugOverlay)
        {
            ImGui::SetNextWindowPos(ImVec2(12.0f, 260.0f), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.85f);
            ImGui::Begin("Environment", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::TextUnformatted(inputContext.cameraMouseCaptured
                                       ? "Press . to release the mouse for UI."
                                       : "Press . again to return to camera look.");
            ImGui::Checkbox("Atmosphere", &environment.atmosphereEnabled);
            ImGui::SliderFloat("Time of Day", &environment.timeOfDay, 0.0f, 24.0f, "%.2f");
            ImGui::SliderFloat("Exposure", &environment.tonemap.exposure, 0.10f, 3.0f, "%.2f");
            ImGui::SliderFloat("White Point", &environment.tonemap.whitePoint, 2.0f, 16.0f, "%.2f");
            ImGui::SliderFloat("Aerial Distance (km)",
                               &environment.atmosphere.aerialPerspectiveDistanceKm,
                               4.0f,
                               64.0f,
                               "%.1f");
            ImGui::SliderFloat("Mie G", &environment.atmosphere.mieAnisotropy, 0.5f, 0.95f, "%.2f");
            ImGui::Text("Sun Dir: %.2f %.2f %.2f",
                        environment.sunDirection.x,
                        environment.sunDirection.y,
                        environment.sunDirection.z);
            ImGui::Separator();
            ImGui::TextUnformatted("Pass Isolation");
            ImGui::Checkbox("World Pass", &environment.debug.worldPassEnabled);
            ImGui::Checkbox("Sky Pass", &environment.debug.skyPassEnabled);
            ImGui::Checkbox("Aerial Perspective", &environment.debug.aerialPerspectiveEnabled);
            ImGui::Checkbox("Fog Fallback", &environment.debug.fogFallbackEnabled);
            ImGui::Checkbox("Shadows", &environment.debug.shadowsEnabled);
            ImGui::Separator();
            ImGui::TextUnformatted("View Diagnostics");
            ImGui::Text("View Y: %.3f (%.1f deg)", viewDirection.y, viewElevationDeg);
            ImGui::Text("Sun Y: %.3f (%.1f deg)", environment.sunDirection.y, sunElevationDeg);
            ImGui::Text("View.Sun: %.3f", sunViewDot);
            ImGui::Text("Above Ground: %.2f blocks", altitudeAboveGround);
            ImGui::Text("Fog Start/Far: %.0f / %.0f", environment.fogStartBlocks, environment.farDistanceBlocks);
            ImGui::Text("Fog Span: %.0f", fogSpanBlocks);
            ImGui::Text("Near Horizon: %s", nearHorizonView ? "yes" : "no");
            ImGui::Text("Looking Below Horizon: %s", lookingBelowHorizon ? "yes" : "no");
            ImGui::Separator();
            ImGui::TextUnformatted("How To Isolate");
            ImGui::TextWrapped("1. Disable Sky Pass. If the band stays, it is not the sky dome.");
            ImGui::TextWrapped("2. Disable Aerial Perspective. If the band disappears, the aerial LUT is the source.");
            ImGui::TextWrapped("3. Disable Fog Fallback with Aerial Perspective off. If the band still stays, it is world-side shading or geometry.");
            ImGui::End();
        }

        if (!playerReleased && !loadingOverlayText.empty())
        {
            ImGui::SetNextWindowPos(ImVec2(framebufferWidth * 0.5f, framebufferHeight * 0.5f),
                                    ImGuiCond_Always,
                                    ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowBgAlpha(0.50f);
            ImGui::Begin("Loading Overlay",
                         nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs);
            ImGui::TextUnformatted(loadingOverlayText.c_str());
            ImGui::End();
        }

        if (inputContext.showRenderDistanceGUI)
        {
            ImGui::SetNextWindowPos(ImVec2(framebufferWidth * 0.5f, framebufferHeight * 0.5f),
                                    ImGuiCond_Always,
                                    ImVec2(0.5f, 0.5f));
            ImGui::Begin("Render Distance",
                         nullptr,
                         ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::TextUnformatted("Enter near chunks and optional far blocks (e.g. 12 4800):");
            const bool submit = ImGui::InputText("##render-distance",
                                                 &inputContext.inputBuffer,
                                                 ImGuiInputTextFlags_EnterReturnsTrue);
            if (submit || ImGui::Button("Apply"))
            {
                if (!applyRenderDistanceInput(chunkManager, inputContext.inputBuffer))
                {
                    std::cerr << "Invalid render distance input: " << inputContext.inputBuffer << std::endl;
                }
                inputContext.showRenderDistanceGUI = false;
                inputContext.inputBuffer.clear();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape))
            {
                inputContext.showRenderDistanceGUI = false;
                inputContext.inputBuffer.clear();
            }
            ImGui::End();
        }

        if (inputContext.showTeleportGUI)
        {
            ImGui::SetNextWindowPos(ImVec2(framebufferWidth * 0.5f, framebufferHeight * 0.5f),
                                    ImGuiCond_Always,
                                    ImVec2(0.5f, 0.5f));
            ImGui::Begin("Teleport",
                         nullptr,
                         ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::TextUnformatted("Enter teleport target (x y z):");
            const bool submit = ImGui::InputText("##teleport",
                                                 &inputContext.teleportBuffer,
                                                 ImGuiInputTextFlags_EnterReturnsTrue);
            if (submit || ImGui::Button("Teleport"))
            {
                if (!applyTeleportInput(camera, inputContext.teleportBuffer))
                {
                    std::cerr << "Invalid teleport input: " << inputContext.teleportBuffer << std::endl;
                }
                inputContext.showTeleportGUI = false;
                inputContext.teleportBuffer.clear();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape))
            {
                inputContext.showTeleportGUI = false;
                inputContext.teleportBuffer.clear();
            }
            ImGui::End();
        }

        if (!profilingOverlayText.empty())
        {
            ImGui::SetNextWindowPos(ImVec2(12.0f, inputContext.showDebugOverlay ? 220.0f : 12.0f), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.30f);
            ImGui::Begin("Profiling Overlay",
                         nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs);
            ImGui::TextUnformatted(profilingOverlayText.c_str());
            ImGui::End();
        }

        if (playerReleased && isGameplayMouseCaptured(inputContext))
        {
            drawCrosshairOverlay(framebufferWidth, framebufferHeight);
        }

        renderer.endFrame();
    }

    chunkManager.clear();
    renderer.shutdown();
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

    std::error_code cwdEc;
    std::filesystem::current_path(exeDirectory, cwdEc);
    if (cwdEc)
    {
        std::cerr << "Failed to set working directory to executable directory: "
                  << cwdEc.message() << '\n';
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
