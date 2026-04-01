#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "camera.h"
#include "chunk_manager.h"
#include "input_context.h"
#include "mob_system.h"
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
#include <cctype>
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
#include <string_view>
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

extern "C"
{
// Export the DirectX 12 Agility SDK contract so d3d12.dll loads the local
// runtime from .\D3D12 before BlockGame creates its D3D12 device.
__declspec(dllexport) extern const UINT D3D12SDKVersion = 618;
__declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}

namespace
{
std::mutex gCrashLogMutex;
std::filesystem::path gCrashLogPath;
std::filesystem::path gCrashDumpPath;
std::filesystem::path gHangDumpPath;
std::mutex gSymbolMutex;
std::mutex gDiagnosticStateMutex;
std::deque<std::string> gDiagnosticBreadcrumbs;
std::string gDiagnosticPhase{"startup"};
std::atomic<std::uint64_t> gDiagnosticHeartbeatMicros{0};
std::atomic<std::uint64_t> gDiagnosticFrameCounter{0};
std::atomic<bool> gSymbolsReady{false};
std::atomic<bool> gHangWatchdogStop{false};
std::atomic<bool> gHangWatchdogDumped{false};
std::thread gHangWatchdogThread;
constexpr std::size_t kMaxDiagnosticBreadcrumbs = 64;
constexpr auto kHangWatchdogTimeout = std::chrono::seconds(15);

void appendCrashLog(std::string message);
[[nodiscard]] std::uint64_t steadyMicrosNow() noexcept;
void noteDiagnosticPhase(std::string_view phase, bool advanceFrame = false);
void appendDiagnosticSnapshot(const char* reason);
void initializeSymbolHandler(const std::filesystem::path& symbolRoot);
void shutdownSymbolHandler() noexcept;
void startHangWatchdog();
void stopHangWatchdog() noexcept;
void shutdownCrashLogging() noexcept;
void setProcessEnvironmentVariable(const char* name, const char* value);
void applyLaunchDebugOptions(int argc, char** argv);
#ifdef _WIN32
void appendStackTrace(EXCEPTION_POINTERS* exceptionPointers = nullptr);
void writeMiniDump(EXCEPTION_POINTERS* exceptionPointers);
void writeHangMiniDump();
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
    appendDiagnosticSnapshot("signal");
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

void setProcessEnvironmentVariable(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

[[nodiscard]] std::filesystem::path resolveDebugLogPath(const char* envVarName,
                                                        const std::filesystem::path& defaultPath)
{
    const char* value = std::getenv(envVarName);
    if (value != nullptr && *value != '\0')
    {
        return std::filesystem::path(value);
    }
    return defaultPath;
}

void resetDebugLogFile(const char* envVarName,
                       const std::filesystem::path& defaultPath,
                       std::string_view initialLine = {})
{
    const std::filesystem::path logPath = resolveDebugLogPath(envVarName, defaultPath);
    setProcessEnvironmentVariable(envVarName, logPath.string().c_str());
    std::error_code ec;
    const std::filesystem::path parentPath = logPath.parent_path();
    if (!parentPath.empty())
    {
        std::filesystem::create_directories(parentPath, ec);
    }

    std::ofstream out(logPath, std::ios::trunc);
    if (!out)
    {
        return;
    }
    if (!initialLine.empty())
    {
        out << initialLine << '\n';
    }
}

void applyLaunchDebugOptions(int argc, char** argv)
{
    bool enableGpuDebug = false;
    bool enableGpuBreak = false;
    bool enableGpuValidation = false;
    bool enableLodVisibilityDebug = false;
    bool enableExactUploadDebug = false;
    bool enableTerrainLogDebug = false;
    for (int i = 1; i < argc; ++i)
    {
        if (argv[i] == nullptr)
        {
            continue;
        }

        const std::string_view arg(argv[i]);
        if (arg == "--d3d-debug" || arg == "--gpu-debug" || arg == "--debug-gpu")
        {
            enableGpuDebug = true;
            continue;
        }

        if (arg == "--gpu-validate" || arg == "--d3d-validate")
        {
            enableGpuDebug = true;
            enableGpuValidation = true;
            continue;
        }

        if (arg == "--gpu-break" || arg == "--d3d-break")
        {
            enableGpuDebug = true;
            enableGpuValidation = true;
            enableGpuBreak = true;
            continue;
        }

        if (arg == "--lod-visibility-debug" || arg == "--lod-debug")
        {
            enableLodVisibilityDebug = true;
            continue;
        }

        if (arg == "--exact-upload-debug" || arg == "--chunk-upload-debug")
        {
            enableExactUploadDebug = true;
            continue;
        }

        if (arg == "--terrain-log-debug" || arg == "--debug-terrain-log")
        {
            enableTerrainLogDebug = true;
            continue;
        }

    }

    if (!enableGpuDebug && !enableLodVisibilityDebug && !enableExactUploadDebug && !enableTerrainLogDebug)
    {
        return;
    }

    if (enableGpuDebug)
    {
        resetDebugLogFile("BLOCKGAME_RENDER_DEBUG_LOG_FILE",
                          "gpudebug.log",
                          "gpu debug logging enabled");
        setProcessEnvironmentVariable("BLOCKGAME_RENDER_DEBUG_LOG", "1");
        setProcessEnvironmentVariable("BLOCKGAME_ENABLE_D3D12_DEBUG_LAYER", "1");
        setProcessEnvironmentVariable("BLOCKGAME_ENABLE_D3D12_DRED", "1");
        setProcessEnvironmentVariable("BLOCKGAME_ENABLE_D3D12_GPU_VALIDATION",
                                      enableGpuValidation ? "1" : "0");
        if (enableGpuBreak)
        {
            setProcessEnvironmentVariable("BLOCKGAME_BREAK_ON_D3D12_ERROR", "1");
        }
        else
        {
            setProcessEnvironmentVariable("BLOCKGAME_BREAK_ON_D3D12_ERROR", "0");
        }
    }

    if (enableLodVisibilityDebug)
    {
        setProcessEnvironmentVariable("BLOCKGAME_LOD_VIS_DEBUG", "1");
        resetDebugLogFile("BLOCKGAME_LOD_VIS_DEBUG_FILE",
                          "loddebug.log",
                          "lod visibility debug enabled");
    }

    if (enableExactUploadDebug)
    {
        setProcessEnvironmentVariable("BLOCKGAME_EXACT_UPLOAD_DEBUG", "1");
        resetDebugLogFile("BLOCKGAME_EXACT_UPLOAD_DEBUG_FILE",
                          "exactuploaddebug.log",
                          "exact upload debug enabled");
    }

    if (enableTerrainLogDebug)
    {
        setProcessEnvironmentVariable("BLOCKGAME_TERRAIN_DEBUG_LOG", "1");
        resetDebugLogFile("BLOCKGAME_TERRAIN_DEBUG_LOG_FILE",
                          "debug_terrain.log",
                          "terrain debug logging enabled");
    }

    std::vector<std::string> enabledOptions;
    if (enableGpuDebug)
    {
        enabledOptions.emplace_back("D3D12 debug logging");
        enabledOptions.emplace_back("D3D12 debug layer");
        enabledOptions.emplace_back("DRED");
        if (enableGpuValidation)
        {
            enabledOptions.emplace_back("GPU validation");
        }
        if (enableGpuBreak)
        {
            enabledOptions.emplace_back("break-on-error");
        }
    }
    if (enableLodVisibilityDebug)
    {
        enabledOptions.emplace_back("LOD visibility debug (loddebug.log)");
    }
    if (enableExactUploadDebug)
    {
        enabledOptions.emplace_back("exact upload debug (exactuploaddebug.log)");
    }
    if (enableTerrainLogDebug)
    {
        enabledOptions.emplace_back("terrain debug log (debug_terrain.log)");
    }

    if (!enabledOptions.empty())
    {
        std::cout << "Enabled launch options for this run:";
        for (std::size_t i = 0; i < enabledOptions.size(); ++i)
        {
            std::cout << (i == 0 ? " " : ", ") << enabledOptions[i];
        }
        std::cout << "." << '\n';
    }
}

[[nodiscard]] std::uint64_t steadyMicrosNow() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

void noteDiagnosticPhase(std::string_view phase, bool advanceFrame)
{
    if (advanceFrame)
    {
        gDiagnosticFrameCounter.fetch_add(1, std::memory_order_relaxed);
    }

    {
        std::lock_guard<std::mutex> lock(gDiagnosticStateMutex);
        if (gDiagnosticPhase != phase)
        {
            gDiagnosticPhase.assign(phase.data(), phase.size());
            gDiagnosticBreadcrumbs.emplace_back(gDiagnosticPhase);
            if (gDiagnosticBreadcrumbs.size() > kMaxDiagnosticBreadcrumbs)
            {
                gDiagnosticBreadcrumbs.pop_front();
            }
        }
    }

    gDiagnosticHeartbeatMicros.store(steadyMicrosNow(), std::memory_order_relaxed);
}

void appendDiagnosticSnapshot(const char* reason)
{
    std::string phase;
    std::deque<std::string> breadcrumbs;
    {
        std::lock_guard<std::mutex> lock(gDiagnosticStateMutex);
        phase = gDiagnosticPhase;
        breadcrumbs = gDiagnosticBreadcrumbs;
    }

    const std::uint64_t lastHeartbeatMicros = gDiagnosticHeartbeatMicros.load(std::memory_order_relaxed);
    const std::uint64_t nowMicros = steadyMicrosNow();
    const double heartbeatAgeMs =
        (lastHeartbeatMicros == 0 || nowMicros < lastHeartbeatMicros)
            ? 0.0
            : static_cast<double>(nowMicros - lastHeartbeatMicros) / 1000.0;

    std::ostringstream summary;
    summary.setf(std::ios::fixed, std::ios::floatfield);
    summary << std::setprecision(2)
            << "diagnostics: " << reason
            << " | phase=" << phase
            << " | frame=" << gDiagnosticFrameCounter.load(std::memory_order_relaxed)
            << " | heartbeat_age_ms=" << heartbeatAgeMs;
    appendCrashLog(summary.str());

    if (!breadcrumbs.empty())
    {
        std::ostringstream trail;
        trail << "breadcrumbs:";
        std::size_t index = 0;
        for (const std::string& crumb : breadcrumbs)
        {
            trail << "\n  [" << index++ << "] " << crumb;
        }
        appendCrashLog(trail.str());
    }
}

#ifdef _WIN32
[[nodiscard]] std::string symbolizeAddress(std::uintptr_t address)
{
    std::ostringstream oss;
    oss << "0x" << std::hex << address << std::dec;

    if (!gSymbolsReady.load(std::memory_order_acquire))
    {
        return oss.str();
    }

    std::lock_guard<std::mutex> lock(gSymbolMutex);

    HANDLE process = GetCurrentProcess();
    const DWORD64 address64 = static_cast<DWORD64>(address);
    const DWORD64 moduleBase = SymGetModuleBase64(process, address64);

    IMAGEHLP_MODULE64 moduleInfo{};
    moduleInfo.SizeOfStruct = sizeof(moduleInfo);
    if (moduleBase != 0 && SymGetModuleInfo64(process, address64, &moduleInfo))
    {
        const char* moduleName = moduleInfo.ModuleName[0] != '\0' ? moduleInfo.ModuleName : moduleInfo.ImageName;
        if (moduleName != nullptr && moduleName[0] != '\0')
        {
            oss << " " << std::filesystem::path(moduleName).filename().string();
        }
    }

    std::array<std::byte, sizeof(SYMBOL_INFO) + MAX_SYM_NAME> symbolStorage{};
    auto* symbol = reinterpret_cast<SYMBOL_INFO*>(symbolStorage.data());
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;

    DWORD64 displacement = 0;
    if (SymFromAddr(process, address64, &displacement, symbol))
    {
        oss << "!" << symbol->Name;
        if (displacement != 0)
        {
            oss << "+0x" << std::hex << displacement << std::dec;
        }
    }

    if (moduleBase != 0)
    {
        oss << " [rva 0x" << std::hex << (address64 - moduleBase) << std::dec << "]";
    }

    IMAGEHLP_LINE64 lineInfo{};
    lineInfo.SizeOfStruct = sizeof(lineInfo);
    DWORD lineDisplacement = 0;
    if (SymGetLineFromAddr64(process, address64, &lineDisplacement, &lineInfo))
    {
        oss << " at " << lineInfo.FileName << ":" << lineInfo.LineNumber;
        if (lineDisplacement != 0)
        {
            oss << "+0x" << std::hex << lineDisplacement << std::dec;
        }
    }

    return oss.str();
}

void initializeSymbolHandler(const std::filesystem::path& symbolRoot)
{
    std::lock_guard<std::mutex> lock(gSymbolMutex);
    if (gSymbolsReady.load(std::memory_order_acquire))
    {
        return;
    }

    std::string searchPath = symbolRoot.string();
    const std::filesystem::path parent = symbolRoot.parent_path();
    if (!parent.empty())
    {
        searchPath += ';';
        searchPath += parent.string();
    }

    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
    if (!SymInitialize(GetCurrentProcess(), searchPath.c_str(), TRUE))
    {
        appendCrashLog("symbols: SymInitialize failed error=" + std::to_string(GetLastError()));
        return;
    }

    gSymbolsReady.store(true, std::memory_order_release);
}

void shutdownSymbolHandler() noexcept
{
    std::lock_guard<std::mutex> lock(gSymbolMutex);
    if (!gSymbolsReady.exchange(false, std::memory_order_acq_rel))
    {
        return;
    }

    SymCleanup(GetCurrentProcess());
}

void writeMiniDumpTo(const std::filesystem::path& dumpPath,
                     EXCEPTION_POINTERS* exceptionPointers,
                     const char* label)
{
    std::error_code ec;
    const std::filesystem::path parentPath = dumpPath.parent_path();
    if (!parentPath.empty())
    {
        std::filesystem::create_directories(parentPath, ec);
    }

    HANDLE file = CreateFileW(dumpPath.c_str(),
                              GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_WRITE,
                              nullptr,
                              CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL,
                              nullptr);
    if (file == INVALID_HANDLE_VALUE)
    {
        appendCrashLog(std::string(label) + ": failed to create dump file error=" + std::to_string(GetLastError()));
        return;
    }

    MINIDUMP_EXCEPTION_INFORMATION info{};
    info.ThreadId = GetCurrentThreadId();
    info.ExceptionPointers = exceptionPointers;
    info.ClientPointers = FALSE;

    const MINIDUMP_TYPE dumpType = static_cast<MINIDUMP_TYPE>(
        MiniDumpWithDataSegs |
        MiniDumpWithHandleData |
        MiniDumpWithIndirectlyReferencedMemory |
        MiniDumpScanMemory |
        MiniDumpWithProcessThreadData |
        MiniDumpWithThreadInfo |
        MiniDumpWithUnloadedModules);

    const BOOL dumpResult = MiniDumpWriteDump(GetCurrentProcess(),
                                              GetCurrentProcessId(),
                                              file,
                                              dumpType,
                                              exceptionPointers ? &info : nullptr,
                                              nullptr,
                                              nullptr);
    const DWORD dumpError = dumpResult ? ERROR_SUCCESS : GetLastError();
    CloseHandle(file);

    if (dumpResult)
    {
        appendCrashLog(std::string(label) + ": written to " + dumpPath.filename().string());
    }
    else
    {
        appendCrashLog(std::string(label) + ": MiniDumpWriteDump failed error=" + std::to_string(dumpError));
    }
}

void writeHangMiniDump()
{
    const std::filesystem::path dumpPath =
        gHangDumpPath.empty() ? std::filesystem::path("blockgame_hang.dmp") : gHangDumpPath;
    writeMiniDumpTo(dumpPath, nullptr, "hang dump");
}

void startHangWatchdog()
{
    stopHangWatchdog();
    gHangWatchdogStop.store(false, std::memory_order_release);
    gHangWatchdogDumped.store(false, std::memory_order_release);
    gDiagnosticHeartbeatMicros.store(steadyMicrosNow(), std::memory_order_relaxed);

    gHangWatchdogThread = std::thread([]()
    {
        const std::uint64_t defaultTimeoutMicros = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(kHangWatchdogTimeout).count());
        const std::uint64_t shutdownTimeoutMicros = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::seconds(60)).count());
        while (!gHangWatchdogStop.load(std::memory_order_acquire))
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            const std::uint64_t lastHeartbeatMicros = gDiagnosticHeartbeatMicros.load(std::memory_order_relaxed);
            if (lastHeartbeatMicros == 0)
            {
                continue;
            }

            const std::uint64_t nowMicros = steadyMicrosNow();
            std::string phase;
            {
                std::lock_guard<std::mutex> lock(gDiagnosticStateMutex);
                phase = gDiagnosticPhase;
            }
            const bool shutdownPhase = phase.rfind("shutdown/", 0) == 0;
            const std::uint64_t timeoutMicros = shutdownPhase ? shutdownTimeoutMicros : defaultTimeoutMicros;
            if (nowMicros < lastHeartbeatMicros || (nowMicros - lastHeartbeatMicros) < timeoutMicros)
            {
                continue;
            }

            if (!gHangWatchdogDumped.exchange(true, std::memory_order_acq_rel))
            {
                appendDiagnosticSnapshot("hang watchdog timeout");
                writeHangMiniDump();
            }
        }
    });
}

void stopHangWatchdog() noexcept
{
    gHangWatchdogStop.store(true, std::memory_order_release);
    if (gHangWatchdogThread.joinable())
    {
        gHangWatchdogThread.join();
    }
}
#else
void initializeSymbolHandler(const std::filesystem::path&) {}
void shutdownSymbolHandler() noexcept {}
void startHangWatchdog() {}
void stopHangWatchdog() noexcept {}
#endif

#ifdef _WIN32
void appendStackTrace(EXCEPTION_POINTERS* exceptionPointers)
{
    constexpr USHORT kMaxFrames = 64;
    void* stack[kMaxFrames]{};
    const USHORT captured = CaptureStackBackTrace(0, kMaxFrames, stack, nullptr);

    std::ostringstream oss;
    if (exceptionPointers && exceptionPointers->ExceptionRecord)
    {
        const auto exceptionCode =
            static_cast<std::uint32_t>(exceptionPointers->ExceptionRecord->ExceptionCode);
        const auto exceptionAddress = reinterpret_cast<std::uintptr_t>(
            exceptionPointers->ExceptionRecord->ExceptionAddress);
        oss << "exception: code=0x" << std::hex << exceptionCode << std::dec
            << " address=" << symbolizeAddress(exceptionAddress) << '\n';
    }

    oss << "stack:";
    for (USHORT i = 0; i < captured; ++i)
    {
        const auto address = reinterpret_cast<std::uintptr_t>(stack[i]);
        oss << "\n  [" << i << "] " << symbolizeAddress(address);
    }

    appendCrashLog(oss.str());
}

void writeMiniDump(EXCEPTION_POINTERS* exceptionPointers)
{
    const std::filesystem::path dumpPath =
        gCrashDumpPath.empty() ? std::filesystem::path("blockgame_crash.dmp") : gCrashDumpPath;
    writeMiniDumpTo(dumpPath, exceptionPointers, "minidump");
}

int __cdecl crtReportHook(int reportType, char* message, int*)
{
    const char* text = message ? message : "<null>";
    appendCrashLog(std::string("CRT report[") + std::to_string(reportType) + "]: " + text);
    appendDiagnosticSnapshot("crt report");
    return FALSE; // allow default processing
}
#endif

[[nodiscard]] bool envFlagEnabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return false;
    }

    std::string lower(value);
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return lower != "0" && lower != "false" && lower != "off" && lower != "no";
}

[[nodiscard]] float envFloatOrDefault(const char* name, float fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return fallback;
    }

    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    return (end != value) ? parsed : fallback;
}

[[nodiscard]] bool tryGetEnvFloat(const char* name, float& outValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return false;
    }

    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value)
    {
        return false;
    }

    outValue = parsed;
    return true;
}

[[nodiscard]] bool tryGetEnvInt(const char* name, int& outValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return false;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value)
    {
        return false;
    }

    outValue = static_cast<int>(parsed);
    return true;
}

[[nodiscard]] constexpr int blocksToChunkRadiusCeil(int blocks) noexcept
{
    return std::max(1, (std::max(blocks, 1) + kChunkSizeX - 1) / kChunkSizeX);
}

[[nodiscard]] constexpr int chunkRadiusToBlocks(int chunks) noexcept
{
    return std::max(chunks, 1) * kChunkSizeX;
}

void applyCameraPose(Camera& camera,
                     const glm::vec3& position,
                     float yawDegrees,
                     float pitchDegrees);

enum class BenchmarkScenarioKind : std::uint8_t
{
    SpawnPreload = 0,
    FullExactPreload,
    PlayerIdleExactFill,
    PostReleaseExactFill,
    StationaryExactFill,
    PostReleaseExactSweepFill,
    StraightLineSprint,
    TurnHeavyTraversal,
    VerticalTravel
};

struct BenchmarkConfig
{
    bool enabled{false};
    BenchmarkScenarioKind scenario{BenchmarkScenarioKind::SpawnPreload};
    std::filesystem::path outputPath{};
    std::filesystem::path progressLogPath{};
    std::string buildConfig{"Unknown"};
    int exactChunks{kDefaultNearRenderDistance};
    int totalChunks{kDefaultTotalRenderDistanceChunks};
    int fogStartBlocks{kDefaultFarFogStartBlocks};
    int targetExactReadyChunks{30000};
    float altitudeOffsetBlocks{24.0f};
    glm::vec3 stationaryPosition{0.0f, 107.5f, 5.0f};
    float stationaryYawDegrees{0.0f};
    float stationaryPitchDegrees{-10.0f};
    double maxDurationSeconds{1800.0};
    double movementDurationSeconds{0.0};
    double cooldownDurationSeconds{0.0};
    double speedBlocksPerSecond{0.0};
    double turnSegmentSeconds{2.5};
    double verticalSpanBlocks{192.0};
    double progressLogIntervalSeconds{0.5};
    double sweepDegreesPerSecond{90.0};
};

struct BenchmarkSpikeRecord
{
    double elapsedSeconds{0.0};
    double frameMs{0.0};
    glm::vec3 cameraPosition{0.0f};
    glm::ivec3 cameraChunk{0};
    std::string phase{"steady_state"};
    std::string blockingReason{"ready"};
    std::string suspectedSource{"unknown"};
    double chunkUpdateMs{0.0};
    double chunkUpdateResidualMs{0.0};
    double chunkDenseResidencyMs{0.0};
    double chunkVerticalRadiusMs{0.0};
    double chunkPriorityUpdateMs{0.0};
    double chunkUploadBudgetMs{0.0};
    double chunkMissingScanMs{0.0};
    double chunkEnsureVolumeMs{0.0};
    double chunkSchedulingMs{0.0};
    double chunkEvictionMs{0.0};
    double chunkRelightMs{0.0};
    double chunkUploadMs{0.0};
    double chunkUploadQueueAgeMs{0.0};
    double chunkUploadQueuePickMs{0.0};
    double chunkPoolTrimMs{0.0};
    double chunkFarTerrainUpdateMs{0.0};
    double chunkColumnHeightLookupMs{0.0};
    double chunkColumnHeightSampleMs{0.0};
    double chunkUploadPrepareMs{0.0};
    double chunkUploadContextBeginMs{0.0};
    double chunkUploadFinalizeMs{0.0};
    double chunkCommitCollectMs{0.0};
    double chunkCommitChunkScanMs{0.0};
    double chunkCommitMeshLockWaitMs{0.0};
    double chunkCommitMeshLockedMs{0.0};
    double chunkCommitMeshStateMs{0.0};
    double chunkCommitPageStateMs{0.0};
    double chunkCommitReleaseMs{0.0};
    double chunkStartupStateMs{0.0};
    double chunkBenchmarkBookkeepingMs{0.0};
    double pollEventsMs{0.0};
    double buildRenderDataMs{0.0};
    double renderWorldCpuMs{0.0};
    double rendererWorldMs{0.0};
    double rendererShadowMs{0.0};
    double rendererSkyMs{0.0};
    double rendererAtmosphereMs{0.0};
    double rendererToneMapMs{0.0};
    double rendererPresentMs{0.0};
    double rendererEndFrameMs{0.0};
    int verticalRadius{0};
    int verticalRadiusDelta{0};
    int generatedChunks{0};
    int relitChunks{0};
    int relightBatches{0};
    int meshedChunks{0};
    int uploadedChunks{0};
    int uploadAttempts{0};
    int uploadQueueScanEntries{0};
    int uploadSkippedExpired{0};
    int uploadSkippedNotReady{0};
    int uploadSkippedPendingMesh{0};
    int uploadColumnLimited{0};
    int uploadBudgetDeferred{0};
    int uploadRetryFailures{0};
    int uploadScanLimitHits{0};
    int uploadBeginFailures{0};
    int uploadStalePendingMeshes{0};
    int jobBacklog{0};
    int uploadBacklog{0};
    int columnPrefetchBacklog{0};
    int exactPendingChunks{0};
    int missingChunks{0};
    std::uint64_t relightRegionChunks{0};
    std::uint64_t relightChangedChunks{0};
    std::uint64_t relightExternalSnapshotChunks{0};
    std::uint64_t relightSkyAboveChunkScans{0};
    std::uint64_t relightSkySeedNodes{0};
    std::uint64_t relightBlockSeedNodes{0};
    std::uint64_t relightSkyNodesProcessed{0};
    std::uint64_t relightBlockNodesProcessed{0};
};

struct BenchmarkSpikeSummary
{
    std::size_t countOver16_7Ms{0};
    std::size_t countOver33_3Ms{0};
    std::size_t countOver50Ms{0};
    std::size_t countOver100Ms{0};
    std::size_t countOver250Ms{0};
    std::size_t longestStreakOver33_3Ms{0};
    std::vector<BenchmarkSpikeRecord> worstSpikes;
};

struct BenchmarkRuntimeState
{
    bool started{false};
    bool completed{false};
    bool timedOut{false};
    double elapsedSeconds{0.0};
    double completionHoldSeconds{0.0};
    double exactRequiredStableSeconds{0.0};
    double nextProgressLogSeconds{0.0};
    double playerReleaseSeconds{-1.0};
    double steadyStateSeconds{-1.0};
    double fullExactReadySeconds{-1.0};
    int lastExactRequiredChunks{-1};
    int playerReleaseExactReadyChunks{-1};
    int playerReleaseExactRequiredChunks{-1};
    glm::vec3 spawnPosition{0.0f};
    glm::vec3 scenarioStartPosition{0.0f};
    glm::vec3 finalCameraPosition{0.0f};
    float scenarioStartYawDegrees{0.0f};
    float scenarioStartPitchDegrees{0.0f};
    float finalYawDegrees{0.0f};
    float finalPitchDegrees{0.0f};
    std::vector<double> frameTimesMs;
    std::vector<double> lodGpuCullMs;
    std::vector<double> lodIndirectBuildMs;
    std::vector<double> exactGpuSynthMs;
    std::vector<double> exactGpuStampMs;
    std::vector<double> exactGpuLightMs;
    std::vector<double> exactGpuFaceCountMs;
    std::vector<double> exactGpuFacePrefixMs;
    std::vector<double> exactGpuFaceEmitMs;
    std::vector<double> exactGpuTotalMs;
    std::vector<double> gpuLocalUsageMiB;
    std::vector<double> exactGpuTotalMiB;
    std::size_t currentSpikeStreakOver33_3Ms{0};
    BenchmarkSpikeSummary spikeSummary{};
};

struct BenchmarkFrameSummary
{
    std::size_t count{0};
    double averageMs{0.0};
    double medianMs{0.0};
    double p95Ms{0.0};
    double maxMs{0.0};
    double averageFps{0.0};
};

[[nodiscard]] const char* benchmarkScenarioName(BenchmarkScenarioKind scenario) noexcept
{
    switch (scenario)
    {
    case BenchmarkScenarioKind::SpawnPreload:
        return "spawn_preload";
    case BenchmarkScenarioKind::FullExactPreload:
        return "full_exact_preload";
    case BenchmarkScenarioKind::PlayerIdleExactFill:
        return "player_idle_exact_fill";
    case BenchmarkScenarioKind::PostReleaseExactFill:
        return "post_release_exact_fill";
    case BenchmarkScenarioKind::StationaryExactFill:
        return "stationary_exact_fill";
    case BenchmarkScenarioKind::PostReleaseExactSweepFill:
        return "post_release_exact_sweep_fill";
    case BenchmarkScenarioKind::StraightLineSprint:
        return "straight_line_sprint";
    case BenchmarkScenarioKind::TurnHeavyTraversal:
        return "turn_heavy_traversal";
    case BenchmarkScenarioKind::VerticalTravel:
        return "vertical_travel";
    default:
        return "unknown";
    }
}

[[nodiscard]] const char* streamingPhaseName(StreamingPhase phase) noexcept
{
    switch (phase)
    {
    case StreamingPhase::SpawnResolve:
        return "spawn_resolve";
    case StreamingPhase::ExactPreload:
        return "exact_preload";
    case StreamingPhase::InteractiveNearOnly:
        return "interactive_near_only";
    case StreamingPhase::FarRamp:
        return "far_ramp";
    case StreamingPhase::SteadyState:
    default:
        return "steady_state";
    }
}

[[nodiscard]] bool tryParseBenchmarkScenario(const std::string& text, BenchmarkScenarioKind& outScenario) noexcept
{
    if (text == "spawn_preload")
    {
        outScenario = BenchmarkScenarioKind::SpawnPreload;
        return true;
    }
    if (text == "full_exact_preload" || text == "fixed_exact_preload")
    {
        outScenario = BenchmarkScenarioKind::FullExactPreload;
        return true;
    }
    if (text == "player_idle_exact_fill" || text == "idle_exact_fill" || text == "realistic_exact_fill" ||
        text == "standing_exact_fill")
    {
        outScenario = BenchmarkScenarioKind::PlayerIdleExactFill;
        return true;
    }
    if (text == "post_release_exact_fill" || text == "release_exact_fill" || text == "spawn_exact_fill")
    {
        outScenario = BenchmarkScenarioKind::PostReleaseExactFill;
        return true;
    }
    if (text == "stationary_exact_fill" || text == "in_game_exact_fill" || text == "static_exact_fill")
    {
        outScenario = BenchmarkScenarioKind::StationaryExactFill;
        return true;
    }
    if (text == "post_release_exact_sweep_fill" || text == "release_exact_sweep_fill" ||
        text == "spawn_exact_sweep_fill" || text == "exact_sweep_fill")
    {
        outScenario = BenchmarkScenarioKind::PostReleaseExactSweepFill;
        return true;
    }
    if (text == "straight_line" || text == "straight_line_sprint")
    {
        outScenario = BenchmarkScenarioKind::StraightLineSprint;
        return true;
    }
    if (text == "turn_heavy" || text == "turn_heavy_traversal")
    {
        outScenario = BenchmarkScenarioKind::TurnHeavyTraversal;
        return true;
    }
    if (text == "vertical" || text == "vertical_travel")
    {
        outScenario = BenchmarkScenarioKind::VerticalTravel;
        return true;
    }
    return false;
}

[[nodiscard]] bool benchmarkScenarioStartsAtSpawn(BenchmarkScenarioKind scenario) noexcept
{
    return scenario == BenchmarkScenarioKind::SpawnPreload ||
           scenario == BenchmarkScenarioKind::FullExactPreload ||
           scenario == BenchmarkScenarioKind::PlayerIdleExactFill;
}

[[nodiscard]] bool benchmarkScenarioUsesFullStartupExactPreload(BenchmarkScenarioKind scenario) noexcept
{
    return scenario == BenchmarkScenarioKind::FullExactPreload;
}

[[nodiscard]] bool benchmarkScenarioUsesInteractiveRuntime(BenchmarkScenarioKind scenario) noexcept
{
    return scenario == BenchmarkScenarioKind::PlayerIdleExactFill;
}

[[nodiscard]] BenchmarkConfig loadBenchmarkConfig()
{
    BenchmarkConfig config;
    config.enabled = envFlagEnabled("BLOCKGAME_BENCHMARK");
    if (!config.enabled)
    {
        return config;
    }

    const char* scenarioValue = std::getenv("BLOCKGAME_BENCHMARK_SCENARIO");
    if (scenarioValue == nullptr || scenarioValue[0] == '\0')
    {
        config.enabled = false;
        return config;
    }

    std::string normalizedScenario = scenarioValue;
    std::transform(normalizedScenario.begin(),
                   normalizedScenario.end(),
                   normalizedScenario.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (!tryParseBenchmarkScenario(normalizedScenario, config.scenario))
    {
        config.enabled = false;
        return config;
    }

    if (const char* outputValue = std::getenv("BLOCKGAME_BENCHMARK_OUTPUT"))
    {
        config.outputPath = outputValue;
    }
    if (config.outputPath.empty())
    {
        std::error_code ec;
        std::filesystem::path cwd = std::filesystem::current_path(ec);
        if (ec)
        {
            cwd = ".";
        }
        config.outputPath = cwd / "artifacts" / "chunk_benchmark" /
            (std::string(benchmarkScenarioName(config.scenario)) + ".json");
    }
    if (const char* progressLogValue = std::getenv("BLOCKGAME_BENCHMARK_PROGRESS_LOG"))
    {
        config.progressLogPath = progressLogValue;
    }
    if (config.progressLogPath.empty())
    {
        config.progressLogPath = config.outputPath;
        config.progressLogPath.replace_extension(".log");
    }

    if (const char* buildConfigValue = std::getenv("BLOCKGAME_BENCHMARK_BUILD_CONFIG"))
    {
        config.buildConfig = buildConfigValue;
    }

    (void)tryGetEnvInt("BLOCKGAME_BENCHMARK_EXACT_CHUNKS", config.exactChunks);
    (void)tryGetEnvInt("BLOCKGAME_BENCHMARK_NEAR_CHUNKS", config.exactChunks);
    config.totalChunks = config.exactChunks;
    (void)tryGetEnvInt("BLOCKGAME_BENCHMARK_TOTAL_CHUNKS", config.totalChunks);
    int legacyFarBlocks = 0;
    if (tryGetEnvInt("BLOCKGAME_BENCHMARK_FAR_BLOCKS", legacyFarBlocks))
    {
        config.totalChunks = blocksToChunkRadiusCeil(legacyFarBlocks);
    }
    (void)tryGetEnvInt("BLOCKGAME_BENCHMARK_FOG_START_BLOCKS", config.fogStartBlocks);
    (void)tryGetEnvInt("BLOCKGAME_BENCHMARK_TARGET_CHUNKS", config.targetExactReadyChunks);
    config.altitudeOffsetBlocks =
        envFloatOrDefault("BLOCKGAME_BENCHMARK_ALTITUDE_OFFSET", config.altitudeOffsetBlocks);
    config.stationaryPosition.x =
        envFloatOrDefault("BLOCKGAME_BENCHMARK_STATIONARY_X", config.stationaryPosition.x);
    config.stationaryPosition.y =
        envFloatOrDefault("BLOCKGAME_BENCHMARK_STATIONARY_Y", config.stationaryPosition.y);
    config.stationaryPosition.z =
        envFloatOrDefault("BLOCKGAME_BENCHMARK_STATIONARY_Z", config.stationaryPosition.z);
    config.stationaryYawDegrees =
        envFloatOrDefault("BLOCKGAME_BENCHMARK_STATIONARY_YAW", config.stationaryYawDegrees);
    config.stationaryPitchDegrees =
        envFloatOrDefault("BLOCKGAME_BENCHMARK_STATIONARY_PITCH", config.stationaryPitchDegrees);
    config.maxDurationSeconds = std::max(
        1.0,
        static_cast<double>(
            envFloatOrDefault("BLOCKGAME_BENCHMARK_MAX_DURATION_SECONDS", static_cast<float>(config.maxDurationSeconds))));

    switch (config.scenario)
    {
    case BenchmarkScenarioKind::SpawnPreload:
    case BenchmarkScenarioKind::FullExactPreload:
    case BenchmarkScenarioKind::PlayerIdleExactFill:
    case BenchmarkScenarioKind::PostReleaseExactFill:
    case BenchmarkScenarioKind::StationaryExactFill:
    case BenchmarkScenarioKind::PostReleaseExactSweepFill:
        config.movementDurationSeconds = 0.0;
        config.cooldownDurationSeconds = 0.0;
        config.speedBlocksPerSecond = 0.0;
        break;
    case BenchmarkScenarioKind::StraightLineSprint:
        config.movementDurationSeconds = 18.0;
        config.cooldownDurationSeconds = 2.0;
        config.speedBlocksPerSecond = 96.0;
        break;
    case BenchmarkScenarioKind::TurnHeavyTraversal:
        config.movementDurationSeconds = 18.0;
        config.cooldownDurationSeconds = 2.0;
        config.speedBlocksPerSecond = 72.0;
        break;
    case BenchmarkScenarioKind::VerticalTravel:
        config.movementDurationSeconds = 16.0;
        config.cooldownDurationSeconds = 2.0;
        config.speedBlocksPerSecond = 0.0;
        break;
    }

    if (config.scenario == BenchmarkScenarioKind::StationaryExactFill)
    {
        config.targetExactReadyChunks = std::min(config.targetExactReadyChunks, 30000);
        config.maxDurationSeconds = std::min(config.maxDurationSeconds, 1800.0);
    }
    if (config.scenario == BenchmarkScenarioKind::PlayerIdleExactFill)
    {
        config.maxDurationSeconds = std::min(config.maxDurationSeconds, 300.0);
    }

    config.movementDurationSeconds =
        std::max(0.0, static_cast<double>(envFloatOrDefault("BLOCKGAME_BENCHMARK_DURATION",
                                                            static_cast<float>(config.movementDurationSeconds))));
    config.cooldownDurationSeconds =
        std::max(0.0, static_cast<double>(envFloatOrDefault("BLOCKGAME_BENCHMARK_COOLDOWN",
                                                            static_cast<float>(config.cooldownDurationSeconds))));
    config.speedBlocksPerSecond =
        std::max(0.0, static_cast<double>(envFloatOrDefault("BLOCKGAME_BENCHMARK_SPEED",
                                                            static_cast<float>(config.speedBlocksPerSecond))));
    config.turnSegmentSeconds =
        std::max(0.25, static_cast<double>(envFloatOrDefault("BLOCKGAME_BENCHMARK_TURN_SEGMENT_SECONDS",
                                                             static_cast<float>(config.turnSegmentSeconds))));
    config.verticalSpanBlocks =
        std::max(32.0, static_cast<double>(envFloatOrDefault("BLOCKGAME_BENCHMARK_VERTICAL_SPAN",
                                                             static_cast<float>(config.verticalSpanBlocks))));
    config.progressLogIntervalSeconds =
        std::max(0.1, static_cast<double>(envFloatOrDefault("BLOCKGAME_BENCHMARK_PROGRESS_LOG_INTERVAL",
                                                            static_cast<float>(config.progressLogIntervalSeconds))));
    config.sweepDegreesPerSecond =
        std::max(15.0, static_cast<double>(envFloatOrDefault("BLOCKGAME_BENCHMARK_SWEEP_DEGREES_PER_SECOND",
                                                             static_cast<float>(config.sweepDegreesPerSecond))));

    return config;
}

[[nodiscard]] double percentileFromSorted(const std::vector<double>& sortedValues, double percentile)
{
    if (sortedValues.empty())
    {
        return 0.0;
    }

    const double clamped = std::clamp(percentile, 0.0, 1.0);
    const std::size_t index = static_cast<std::size_t>(
        std::clamp(std::ceil(clamped * static_cast<double>(sortedValues.size())) - 1.0,
                   0.0,
                   static_cast<double>(sortedValues.size() - 1)));
    return sortedValues[index];
}

[[nodiscard]] BenchmarkFrameSummary summarizeFrameTimes(const std::vector<double>& frameTimesMs)
{
    BenchmarkFrameSummary summary{};
    summary.count = frameTimesMs.size();
    if (frameTimesMs.empty())
    {
        return summary;
    }

    std::vector<double> sorted = frameTimesMs;
    std::sort(sorted.begin(), sorted.end());
    const double totalMs = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    summary.averageMs = totalMs / static_cast<double>(sorted.size());
    summary.medianMs = percentileFromSorted(sorted, 0.50);
    summary.p95Ms = percentileFromSorted(sorted, 0.95);
    summary.maxMs = sorted.back();
    if (summary.averageMs > 0.0)
    {
        summary.averageFps = 1000.0 / summary.averageMs;
    }
    return summary;
}

[[nodiscard]] BenchmarkStageStats summarizeStageSamples(const std::vector<double>& samplesMs)
{
    BenchmarkStageStats stats{};
    if (samplesMs.empty())
    {
        return stats;
    }

    std::vector<double> sorted = samplesMs;
    std::sort(sorted.begin(), sorted.end());
    stats.count = sorted.size();
    stats.totalMs = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    stats.averageMs = stats.totalMs / static_cast<double>(stats.count);
    stats.medianMs = percentileFromSorted(sorted, 0.50);
    stats.p95Ms = percentileFromSorted(sorted, 0.95);
    stats.p99Ms = percentileFromSorted(sorted, 0.99);
    stats.maxMs = sorted.back();
    return stats;
}

[[nodiscard]] int benchmarkFloorDiv(int value, int divisor) noexcept
{
    int quotient = value / divisor;
    const int remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0)))
    {
        --quotient;
    }
    return quotient;
}

[[nodiscard]] glm::ivec3 benchmarkWorldToChunkCoords(const glm::vec3& worldPos) noexcept
{
    return {
        benchmarkFloorDiv(static_cast<int>(std::floor(worldPos.x)), kChunkSizeX),
        benchmarkFloorDiv(static_cast<int>(std::floor(worldPos.y)), kChunkSizeY),
        benchmarkFloorDiv(static_cast<int>(std::floor(worldPos.z)), kChunkSizeZ)};
}

[[nodiscard]] double rendererWorkMs(const RendererProfilingSnapshot& snapshot) noexcept
{
    return snapshot.atmosphereLutMs +
           snapshot.skyDrawMs +
           snapshot.shadowDrawMs +
           snapshot.worldDrawMs +
           snapshot.toneMapMs +
           snapshot.endFrameMs;
}

[[nodiscard]] std::string classifyBenchmarkSpikeSource(double frameMs,
                                                       const ChunkProfilingSnapshot& chunkSnapshot,
                                                       const RendererProfilingSnapshot& rendererSnapshot,
                                                       double pollEventsMs,
                                                       double buildRenderDataMs,
                                                       double renderWorldCpuMs) noexcept
{
    const double renderMs = rendererWorkMs(rendererSnapshot);
    const double updateMs = chunkSnapshot.updateMsLastFrame;
    const double updateResidualMs = chunkSnapshot.updateResidualMsLastFrame;
    const double denseResidencyMs = chunkSnapshot.denseResidencyMsLastFrame;
    const double verticalRadiusMs = chunkSnapshot.verticalRadiusMsLastFrame;
    const double priorityUpdateMs = chunkSnapshot.priorityUpdateMsLastFrame;
    const double uploadBudgetMs = chunkSnapshot.uploadBudgetMsLastFrame;
    const double missingScanMs = chunkSnapshot.missingScanMsLastFrame;
    const double ensureVolumeMs = chunkSnapshot.ensureVolumeMsLastFrame;
    const double schedulingMs = chunkSnapshot.schedulingMsLastFrame;
    const double evictionMs = chunkSnapshot.evictionMsLastFrame;
    const double relightMs = chunkSnapshot.relightMsLastFrame;
    const double presentMs = rendererSnapshot.presentMs;
    const double uploadMs = chunkSnapshot.uploadMsLastFrame;
    const double uploadQueuePickMs = chunkSnapshot.uploadQueuePickMsLastFrame;
    const double poolTrimMs = chunkSnapshot.poolTrimMsLastFrame;
    const double farTerrainUpdateMs = chunkSnapshot.farTerrainUpdateMsLastFrame;
    const double columnHeightLookupMs = chunkSnapshot.columnHeightLookupMsLastFrame;
    const double columnHeightSampleMs = chunkSnapshot.columnHeightSampleMsLastFrame;
    const double uploadPrepareMs = chunkSnapshot.uploadPrepareMsLastFrame;
    const double uploadContextBeginMs = chunkSnapshot.uploadContextBeginMsLastFrame;
    const double uploadFinalizeMs = chunkSnapshot.uploadFinalizeMsLastFrame;
    const double commitCollectMs = chunkSnapshot.commitCollectMsLastFrame;
    const double commitChunkScanMs = chunkSnapshot.commitChunkScanMsLastFrame;
    const double commitMeshLockWaitMs = chunkSnapshot.commitMeshLockWaitMsLastFrame;
    const double commitMeshLockedMs = chunkSnapshot.commitMeshLockedMsLastFrame;
    const double commitMeshStateMs = chunkSnapshot.commitMeshStateMsLastFrame;
    const double commitPageStateMs = chunkSnapshot.commitPageStateMsLastFrame;
    const double commitReleaseMs = chunkSnapshot.commitReleaseMsLastFrame;
    const double startupStateMs = chunkSnapshot.startupStateMsLastFrame;
    const double benchmarkBookkeepingMs = chunkSnapshot.benchmarkBookkeepingMsLastFrame;

    if (pollEventsMs >= 25.0 && pollEventsMs >= frameMs * 0.35)
    {
        return "poll_events";
    }
    if (buildRenderDataMs >= 25.0 && buildRenderDataMs >= frameMs * 0.35)
    {
        return "build_render_data";
    }
    if (renderWorldCpuMs >= 25.0 && renderWorldCpuMs >= frameMs * 0.35)
    {
        return "render_world_cpu";
    }
    if (presentMs >= 25.0 && presentMs >= frameMs * 0.35)
    {
        return "present_wait";
    }
    if (updateMs >= 25.0 && updateMs >= renderMs * 1.25)
    {
        if (missingScanMs >= 10.0 && missingScanMs >= updateMs * 0.35)
        {
            return "chunk_missing_scan";
        }
        if (verticalRadiusMs >= 10.0 && verticalRadiusMs >= updateMs * 0.35)
        {
            return "chunk_vertical_radius";
        }
        if (priorityUpdateMs >= 10.0 && priorityUpdateMs >= updateMs * 0.35)
        {
            return "chunk_priority_update";
        }
        if (uploadBudgetMs >= 10.0 && uploadBudgetMs >= updateMs * 0.35)
        {
            return "chunk_upload_budget";
        }
        if (uploadPrepareMs >= 10.0 && uploadPrepareMs >= updateMs * 0.35)
        {
            if (commitMeshLockWaitMs >= 10.0 && commitMeshLockWaitMs >= uploadPrepareMs * 0.30)
            {
                return "chunk_commit_mesh_lock_wait";
            }
            if (commitMeshLockedMs >= 10.0 && commitMeshLockedMs >= uploadPrepareMs * 0.30)
            {
                return "chunk_commit_mesh_locked";
            }
            if (commitCollectMs >= 10.0 && commitCollectMs >= uploadPrepareMs * 0.30)
            {
                return "chunk_commit_collect";
            }
            if (commitChunkScanMs >= 10.0 && commitChunkScanMs >= uploadPrepareMs * 0.30)
            {
                return "chunk_commit_chunk_scan";
            }
            if (commitMeshStateMs >= 10.0 && commitMeshStateMs >= uploadPrepareMs * 0.30)
            {
                return "chunk_commit_mesh_state";
            }
            if (commitPageStateMs >= 10.0 && commitPageStateMs >= uploadPrepareMs * 0.30)
            {
                return "chunk_commit_page_state";
            }
            if (commitReleaseMs >= 10.0 && commitReleaseMs >= uploadPrepareMs * 0.30)
            {
                return "chunk_commit_release";
            }
            return "chunk_upload_prepare";
        }
        if (uploadContextBeginMs >= 10.0 && uploadContextBeginMs >= updateMs * 0.35)
        {
            return "chunk_upload_begin";
        }
        if (uploadFinalizeMs >= 10.0 && uploadFinalizeMs >= updateMs * 0.35)
        {
            return "chunk_upload_finalize";
        }
        if (farTerrainUpdateMs >= 10.0 && farTerrainUpdateMs >= updateMs * 0.35)
        {
            return "chunk_far_terrain_update";
        }
        if (columnHeightLookupMs >= 10.0 && columnHeightLookupMs >= updateMs * 0.35)
        {
            return "chunk_column_height_lookup";
        }
        if (columnHeightSampleMs >= 10.0 && columnHeightSampleMs >= updateMs * 0.35)
        {
            return "chunk_column_height_sample";
        }
        if (ensureVolumeMs >= 10.0 && ensureVolumeMs >= updateMs * 0.35)
        {
            return "chunk_ensure_volume";
        }
        if (denseResidencyMs >= 10.0 && denseResidencyMs >= updateMs * 0.35)
        {
            return "chunk_dense_residency";
        }
        if (schedulingMs >= 10.0 && schedulingMs >= updateMs * 0.35)
        {
            return "chunk_scheduling";
        }
        if (evictionMs >= 10.0 && evictionMs >= updateMs * 0.35)
        {
            return "chunk_eviction";
        }
        if (relightMs >= 5.0 && relightMs >= updateMs * 0.2)
        {
            return "chunk_main_thread_relight";
        }
        if (uploadQueuePickMs >= 6.0 && uploadQueuePickMs >= updateMs * 0.25)
        {
            return "chunk_upload_queue_pick";
        }
        if (uploadMs >= 6.0 && uploadMs >= updateMs * 0.25)
        {
            return "chunk_upload";
        }
        if (poolTrimMs >= 6.0 && poolTrimMs >= updateMs * 0.25)
        {
            return "chunk_pool_trim";
        }
        if (startupStateMs >= 6.0 && startupStateMs >= updateMs * 0.25)
        {
            return "chunk_startup_state";
        }
        if (benchmarkBookkeepingMs >= 6.0 && benchmarkBookkeepingMs >= updateMs * 0.25)
        {
            return "chunk_benchmark_bookkeeping";
        }
        if (updateResidualMs >= 10.0 && updateResidualMs >= updateMs * 0.35)
        {
            return "chunk_update_unattributed";
        }
        if (chunkSnapshot.relightBatches > 0 || chunkSnapshot.relitChunks > 0)
        {
            return "chunk_update_relight";
        }
        if (chunkSnapshot.generatedChunks > 0 || chunkSnapshot.meshedChunks > 0)
        {
            return "chunk_update_streaming";
        }
        return "chunk_update";
    }
    if (renderMs >= 20.0 && renderMs >= frameMs * 0.35)
    {
        if (rendererSnapshot.shadowDrawMs >= rendererSnapshot.worldDrawMs &&
            rendererSnapshot.shadowDrawMs >= rendererSnapshot.skyDrawMs)
        {
            return "renderer_shadow";
        }
        if (rendererSnapshot.worldDrawMs >= rendererSnapshot.skyDrawMs)
        {
            return "renderer_world";
        }
        return "renderer_sky";
    }
    if (chunkSnapshot.generatedChunks > 0 || chunkSnapshot.meshedChunks > 0 || chunkSnapshot.uploadedChunks > 0)
    {
        return "streaming_pressure";
    }
    return "unknown";
}

void recordBenchmarkSpike(BenchmarkRuntimeState& runtimeState,
                          double frameMs,
                          const Camera& camera,
                          const ChunkProfilingSnapshot& chunkSnapshot,
                          const RendererProfilingSnapshot& rendererSnapshot,
                          const StreamingStatusSnapshot& streamingStatus,
                          double pollEventsMs,
                          double buildRenderDataMs,
                          double renderWorldCpuMs)
{
    if (frameMs > 16.7)
    {
        ++runtimeState.spikeSummary.countOver16_7Ms;
    }
    if (frameMs > 33.3)
    {
        ++runtimeState.spikeSummary.countOver33_3Ms;
        ++runtimeState.currentSpikeStreakOver33_3Ms;
        runtimeState.spikeSummary.longestStreakOver33_3Ms =
            std::max(runtimeState.spikeSummary.longestStreakOver33_3Ms,
                     runtimeState.currentSpikeStreakOver33_3Ms);
    }
    else
    {
        runtimeState.currentSpikeStreakOver33_3Ms = 0;
    }
    if (frameMs > 50.0)
    {
        ++runtimeState.spikeSummary.countOver50Ms;
    }
    if (frameMs > 100.0)
    {
        ++runtimeState.spikeSummary.countOver100Ms;
    }
    if (frameMs > 250.0)
    {
        ++runtimeState.spikeSummary.countOver250Ms;
    }

    constexpr std::size_t kMaxStoredSpikes = 12;
    if (frameMs <= 50.0)
    {
        return;
    }

    BenchmarkSpikeRecord record{};
    record.elapsedSeconds = runtimeState.elapsedSeconds;
    record.frameMs = frameMs;
    record.cameraPosition = camera.position;
    record.cameraChunk = benchmarkWorldToChunkCoords(camera.position);
    record.phase = streamingPhaseName(streamingStatus.phase);
    record.blockingReason = streamingStatus.blockingReason ? streamingStatus.blockingReason : "ready";
    record.suspectedSource = classifyBenchmarkSpikeSource(frameMs,
                                                          chunkSnapshot,
                                                          rendererSnapshot,
                                                          pollEventsMs,
                                                          buildRenderDataMs,
                                                          renderWorldCpuMs);
    record.chunkUpdateMs = chunkSnapshot.updateMsLastFrame;
    record.chunkUpdateResidualMs = chunkSnapshot.updateResidualMsLastFrame;
    record.chunkDenseResidencyMs = chunkSnapshot.denseResidencyMsLastFrame;
    record.chunkVerticalRadiusMs = chunkSnapshot.verticalRadiusMsLastFrame;
    record.chunkPriorityUpdateMs = chunkSnapshot.priorityUpdateMsLastFrame;
    record.chunkUploadBudgetMs = chunkSnapshot.uploadBudgetMsLastFrame;
    record.chunkMissingScanMs = chunkSnapshot.missingScanMsLastFrame;
    record.chunkEnsureVolumeMs = chunkSnapshot.ensureVolumeMsLastFrame;
    record.chunkSchedulingMs = chunkSnapshot.schedulingMsLastFrame;
    record.chunkEvictionMs = chunkSnapshot.evictionMsLastFrame;
    record.chunkRelightMs = chunkSnapshot.relightMsLastFrame;
    record.chunkUploadMs = chunkSnapshot.uploadMsLastFrame;
    record.chunkUploadQueueAgeMs = chunkSnapshot.uploadQueueAgeMsLastFrame;
    record.chunkUploadQueuePickMs = chunkSnapshot.uploadQueuePickMsLastFrame;
    record.chunkPoolTrimMs = chunkSnapshot.poolTrimMsLastFrame;
    record.chunkFarTerrainUpdateMs = chunkSnapshot.farTerrainUpdateMsLastFrame;
    record.chunkColumnHeightLookupMs = chunkSnapshot.columnHeightLookupMsLastFrame;
    record.chunkColumnHeightSampleMs = chunkSnapshot.columnHeightSampleMsLastFrame;
    record.chunkUploadPrepareMs = chunkSnapshot.uploadPrepareMsLastFrame;
    record.chunkUploadContextBeginMs = chunkSnapshot.uploadContextBeginMsLastFrame;
    record.chunkUploadFinalizeMs = chunkSnapshot.uploadFinalizeMsLastFrame;
    record.chunkCommitCollectMs = chunkSnapshot.commitCollectMsLastFrame;
    record.chunkCommitChunkScanMs = chunkSnapshot.commitChunkScanMsLastFrame;
    record.chunkCommitMeshLockWaitMs = chunkSnapshot.commitMeshLockWaitMsLastFrame;
    record.chunkCommitMeshLockedMs = chunkSnapshot.commitMeshLockedMsLastFrame;
    record.chunkCommitMeshStateMs = chunkSnapshot.commitMeshStateMsLastFrame;
    record.chunkCommitPageStateMs = chunkSnapshot.commitPageStateMsLastFrame;
    record.chunkCommitReleaseMs = chunkSnapshot.commitReleaseMsLastFrame;
    record.chunkStartupStateMs = chunkSnapshot.startupStateMsLastFrame;
    record.chunkBenchmarkBookkeepingMs = chunkSnapshot.benchmarkBookkeepingMsLastFrame;
    record.pollEventsMs = pollEventsMs;
    record.buildRenderDataMs = buildRenderDataMs;
    record.renderWorldCpuMs = renderWorldCpuMs;
    record.rendererWorldMs = rendererSnapshot.worldDrawMs;
    record.rendererShadowMs = rendererSnapshot.shadowDrawMs;
    record.rendererSkyMs = rendererSnapshot.skyDrawMs;
    record.rendererAtmosphereMs = rendererSnapshot.atmosphereLutMs;
    record.rendererToneMapMs = rendererSnapshot.toneMapMs;
    record.rendererPresentMs = rendererSnapshot.presentMs;
    record.rendererEndFrameMs = rendererSnapshot.endFrameMs;
    record.verticalRadius = chunkSnapshot.verticalRadius;
    record.verticalRadiusDelta = chunkSnapshot.verticalRadiusDelta;
    record.generatedChunks = chunkSnapshot.generatedChunks;
    record.relitChunks = chunkSnapshot.relitChunks;
    record.relightBatches = chunkSnapshot.relightBatches;
    record.meshedChunks = chunkSnapshot.meshedChunks;
    record.uploadedChunks = chunkSnapshot.uploadedChunks;
    record.uploadAttempts = chunkSnapshot.uploadAttemptsLastFrame;
    record.uploadQueueScanEntries = chunkSnapshot.uploadQueueScanEntriesLastFrame;
    record.uploadSkippedExpired = chunkSnapshot.uploadSkippedExpiredLastFrame;
    record.uploadSkippedNotReady = chunkSnapshot.uploadSkippedNotReadyLastFrame;
    record.uploadSkippedPendingMesh = chunkSnapshot.uploadSkippedPendingMeshLastFrame;
    record.uploadColumnLimited = chunkSnapshot.uploadColumnLimitedLastFrame;
    record.uploadBudgetDeferred = chunkSnapshot.uploadBudgetDeferredLastFrame;
    record.uploadRetryFailures = chunkSnapshot.uploadRetryFailuresLastFrame;
    record.uploadScanLimitHits = chunkSnapshot.uploadScanLimitHitsLastFrame;
    record.uploadBeginFailures = chunkSnapshot.uploadBeginFailuresLastFrame;
    record.uploadStalePendingMeshes = chunkSnapshot.uploadStalePendingMeshesLastFrame;
    record.jobBacklog = chunkSnapshot.jobQueueDepth;
    record.uploadBacklog = chunkSnapshot.uploadQueueDepth;
    record.columnPrefetchBacklog = chunkSnapshot.columnPrefetchQueueDepth;
    record.exactPendingChunks = chunkSnapshot.exactChunksPending;
    record.missingChunks = chunkSnapshot.missingChunks;
    record.relightRegionChunks = chunkSnapshot.relightRegionChunks;
    record.relightChangedChunks = chunkSnapshot.relightChangedChunks;
    record.relightExternalSnapshotChunks = chunkSnapshot.relightExternalSnapshotChunks;
    record.relightSkyAboveChunkScans = chunkSnapshot.relightSkyAboveChunkScans;
    record.relightSkySeedNodes = chunkSnapshot.relightSkySeedNodes;
    record.relightBlockSeedNodes = chunkSnapshot.relightBlockSeedNodes;
    record.relightSkyNodesProcessed = chunkSnapshot.relightSkyNodesProcessed;
    record.relightBlockNodesProcessed = chunkSnapshot.relightBlockNodesProcessed;

    std::vector<BenchmarkSpikeRecord>& spikes = runtimeState.spikeSummary.worstSpikes;
    spikes.push_back(std::move(record));
    std::sort(spikes.begin(),
              spikes.end(),
              [](const BenchmarkSpikeRecord& lhs, const BenchmarkSpikeRecord& rhs)
              {
                  return lhs.frameMs > rhs.frameMs;
              });
    if (spikes.size() > kMaxStoredSpikes)
    {
        spikes.resize(kMaxStoredSpikes);
    }
}

[[nodiscard]] glm::vec3 benchmarkDirectionForYaw(float yawDegrees) noexcept
{
    const float radians = glm::radians(yawDegrees);
    return glm::normalize(glm::vec3(std::cos(radians), 0.0f, std::sin(radians)));
}

void initializeBenchmarkCamera(Camera& camera,
                               const BenchmarkConfig& config,
                               BenchmarkRuntimeState& runtimeState)
{
    runtimeState.scenarioStartPosition = runtimeState.spawnPosition;
    runtimeState.scenarioStartPosition.y += config.altitudeOffsetBlocks;
    runtimeState.scenarioStartYawDegrees = camera.yaw;
    runtimeState.scenarioStartPitchDegrees = camera.pitch;

    switch (config.scenario)
    {
    case BenchmarkScenarioKind::StraightLineSprint:
        applyCameraPose(camera, runtimeState.scenarioStartPosition, -35.0f, -8.0f);
        break;
    case BenchmarkScenarioKind::TurnHeavyTraversal:
        applyCameraPose(camera, runtimeState.scenarioStartPosition, 0.0f, -10.0f);
        break;
    case BenchmarkScenarioKind::VerticalTravel:
        applyCameraPose(camera, runtimeState.scenarioStartPosition, -20.0f, -22.0f);
        break;
    case BenchmarkScenarioKind::PostReleaseExactSweepFill:
        applyCameraPose(camera,
                        runtimeState.spawnPosition,
                        runtimeState.scenarioStartYawDegrees,
                        runtimeState.scenarioStartPitchDegrees);
        break;
    case BenchmarkScenarioKind::StationaryExactFill:
        runtimeState.scenarioStartPosition = config.stationaryPosition;
        runtimeState.scenarioStartYawDegrees = config.stationaryYawDegrees;
        runtimeState.scenarioStartPitchDegrees = config.stationaryPitchDegrees;
        applyCameraPose(camera,
                        runtimeState.scenarioStartPosition,
                        runtimeState.scenarioStartYawDegrees,
                        runtimeState.scenarioStartPitchDegrees);
        break;
    case BenchmarkScenarioKind::SpawnPreload:
    case BenchmarkScenarioKind::FullExactPreload:
    case BenchmarkScenarioKind::PlayerIdleExactFill:
    case BenchmarkScenarioKind::PostReleaseExactFill:
    default:
        applyCameraPose(camera, runtimeState.spawnPosition, camera.yaw, camera.pitch);
        break;
    }
}

void applyBenchmarkCameraPose(Camera& camera,
                              const BenchmarkConfig& config,
                              const BenchmarkRuntimeState& runtimeState)
{
    if (!runtimeState.started)
    {
        return;
    }

    const double movementElapsed = std::min(runtimeState.elapsedSeconds, config.movementDurationSeconds);
    const glm::vec3 origin = runtimeState.scenarioStartPosition;

    switch (config.scenario)
    {
    case BenchmarkScenarioKind::StraightLineSprint:
    {
        const glm::vec3 direction = benchmarkDirectionForYaw(-35.0f);
        const float distance = static_cast<float>(movementElapsed * config.speedBlocksPerSecond);
        applyCameraPose(camera, origin + direction * distance, -35.0f, -8.0f);
        break;
    }
    case BenchmarkScenarioKind::TurnHeavyTraversal:
    {
        static constexpr std::array<float, 4> kYaws{0.0f, 90.0f, 180.0f, 270.0f};
        glm::vec3 offset(0.0f);
        const double segmentLength = config.speedBlocksPerSecond * config.turnSegmentSeconds;
        const int completedSegments = static_cast<int>(movementElapsed / config.turnSegmentSeconds);
        const double segmentProgressSeconds =
            movementElapsed - static_cast<double>(completedSegments) * config.turnSegmentSeconds;

        for (int segmentIndex = 0; segmentIndex < completedSegments; ++segmentIndex)
        {
            offset += benchmarkDirectionForYaw(kYaws[segmentIndex % kYaws.size()]) *
                      static_cast<float>(segmentLength);
        }

        const float currentYaw = kYaws[completedSegments % static_cast<int>(kYaws.size())];
        offset += benchmarkDirectionForYaw(currentYaw) *
                  static_cast<float>(segmentProgressSeconds * config.speedBlocksPerSecond);
        applyCameraPose(camera, origin + offset, currentYaw, -10.0f);
        break;
    }
    case BenchmarkScenarioKind::VerticalTravel:
    {
        const double halfDuration = std::max(config.movementDurationSeconds * 0.5, 0.001);
        float currentY = origin.y;
        if (movementElapsed <= halfDuration)
        {
            const float t = static_cast<float>(movementElapsed / halfDuration);
            currentY = origin.y + static_cast<float>(config.verticalSpanBlocks * t);
        }
        else
        {
            const float t = static_cast<float>((movementElapsed - halfDuration) / halfDuration);
            currentY = origin.y + static_cast<float>(config.verticalSpanBlocks * (1.0 - t));
        }
        applyCameraPose(camera, glm::vec3(origin.x, currentY, origin.z), -20.0f, -22.0f);
        break;
    }
    case BenchmarkScenarioKind::PostReleaseExactSweepFill:
    {
        const float yaw =
            runtimeState.scenarioStartYawDegrees +
            static_cast<float>(std::fmod(runtimeState.elapsedSeconds * config.sweepDegreesPerSecond, 360.0));
        applyCameraPose(camera, runtimeState.spawnPosition, yaw, runtimeState.scenarioStartPitchDegrees);
        break;
    }
    case BenchmarkScenarioKind::StationaryExactFill:
        applyCameraPose(camera,
                        config.stationaryPosition,
                        config.stationaryYawDegrees,
                        config.stationaryPitchDegrees);
        break;
    case BenchmarkScenarioKind::SpawnPreload:
    case BenchmarkScenarioKind::FullExactPreload:
    case BenchmarkScenarioKind::PlayerIdleExactFill:
    case BenchmarkScenarioKind::PostReleaseExactFill:
    default:
        break;
    }
}

bool resetBenchmarkProgressLog(const BenchmarkConfig& config)
{
    std::error_code ec;
    const std::filesystem::path parentPath = config.progressLogPath.parent_path();
    if (!parentPath.empty())
    {
        std::filesystem::create_directories(parentPath, ec);
        if (ec)
        {
            return false;
        }
    }

    std::ofstream out(config.progressLogPath, std::ios::trunc);
    if (!out)
    {
        return false;
    }

    out << "scenario=" << benchmarkScenarioName(config.scenario)
        << " build_config=" << config.buildConfig
        << " exact_chunks=" << config.exactChunks
        << " total_chunks=" << config.totalChunks
        << " fog_start_blocks=" << config.fogStartBlocks
        << " target_exact_ready_chunks=" << config.targetExactReadyChunks
        << '\n';
    return true;
}

void appendBenchmarkProgressLog(const BenchmarkConfig& config,
                                const BenchmarkRuntimeState& runtimeState,
                                const Camera& camera,
                                const ChunkProfilingSnapshot& profiling,
                                const StreamingStatusSnapshot& streamingStatus,
                                const char* eventTag = nullptr)
{
    std::ofstream out(config.progressLogPath, std::ios::app);
    if (!out)
    {
        return;
    }

    const double exactPercent = streamingStatus.exactRequiredChunks > 0
                                    ? (100.0 * static_cast<double>(streamingStatus.exactReadyChunks) /
                                       static_cast<double>(streamingStatus.exactRequiredChunks))
                                    : 0.0;

    out.setf(std::ios::fixed, std::ios::floatfield);
    out << std::setprecision(2);
    out << "t=" << runtimeState.elapsedSeconds << "s";
    if (eventTag != nullptr && eventTag[0] != '\0')
    {
        out << " event=" << eventTag;
    }
    out << " phase=" << streamingPhaseName(streamingStatus.phase)
        << " ready=" << streamingStatus.exactReadyChunks
        << " required=" << streamingStatus.exactRequiredChunks
        << " exact_pct=" << exactPercent
        << " pending_uploads=" << streamingStatus.exactPendingUploads
        << " exact_pending=" << profiling.exactChunksPending
        << " exact_pages=" << profiling.exactGpuPageCount
        << " exact_page_mib=" << (static_cast<double>(profiling.exactGpuPageBytes) / (1024.0 * 1024.0))
        << " exact_oflow=" << profiling.exactGpuBuildOverflows
        << " exact_rb_fail=" << profiling.exactGpuBuildReadbackFailures
        << " exact_res_fail=" << profiling.exactGpuBuildResourceFailures
        << " exact_stale=" << profiling.exactGpuBuildStaleCancels
        << " exact_submit=" << profiling.exactGpuBuildsSubmitted
        << " exact_commit=" << profiling.exactGpuBuildsCommitted
        << " exact_replace=" << profiling.exactGpuMeshReplacements
        << " exact_qbuild=" << profiling.exactGpuQueuedBuilds
        << " exact_pbuild=" << profiling.exactGpuPendingBuilds
        << " exact_cpu_ms={prep:" << profiling.exactGpuPrepareCpuMs
        << ",submit:" << profiling.exactGpuSubmitCpuMs
        << ",commit:" << profiling.exactGpuCommitCpuMs
        << ",wg:" << profiling.exactGpuWorldgenResolveMsLastCycle
        << ",prepass_rb:" << profiling.exactGpuPrepassFaceTotalsReadbackMsLastCycle
        << ",emit_meta:" << profiling.exactGpuEmitMetadataSyncMsLastCycle
        << ",sweep:" << profiling.exactGpuPageSweepMsLastCycle
        << ",emit_wait:" << profiling.exactGpuEmitWaitMsLastCycle
        << ",emit_fence:" << profiling.exactGpuEmitFenceLifetimeMsLastCycle << "}"
        << " exact_batch={submit:" << profiling.exactGpuSubmitBatchBuildsLastCycle
        << ",emit:" << profiling.exactGpuEmitBatchBuildsLastCycle << "}"
        << " exact_wg_miss=" << profiling.exactGpuWorldgenPageMissesLastCycle
        << " exact_emit_meta={pages:" << profiling.exactGpuEmitMetadataDirtyPagesLastCycle
        << ",kib:" << (static_cast<double>(profiling.exactGpuEmitMetadataUploadBytesLastCycle) / 1024.0) << "}"
        << " exact_emit_backlog={batches:" << profiling.exactGpuReadyForEmitBacklogBatchesLastCycle
        << ",builds:" << profiling.exactGpuReadyForEmitBacklogBuildsLastCycle
        << ",compute_before:" << profiling.exactGpuComputeInFlightBeforeEmitLastCycle
        << ",compute_after:" << profiling.exactGpuComputeInFlightAfterEmitLastCycle
        << ",blocking_age:" << profiling.exactGpuBlockingEmitBatchAgeMsLastCycle
        << ",blocking_builds:" << profiling.exactGpuBlockingEmitBatchBuildsLastCycle << "}"
        << " exact_sweep_pages=" << profiling.exactGpuPageSweepPagesLastCycle
        << " exact_gpu_ms={synth:" << profiling.exactGpuSynthMs
        << ",stamp:" << profiling.exactGpuStampMs
        << ",light:" << profiling.exactGpuLightMs
        << ",count:" << profiling.exactGpuFaceCountMs
        << ",prefix:" << profiling.exactGpuFacePrefixMs
        << ",allocate:" << profiling.exactGpuAllocateMs
        << ",emit:" << profiling.exactGpuFaceEmitMs
        << ",total:" << profiling.exactGpuTotalMs << "}"
        << " exact_mem_mib=" << (static_cast<double>(profiling.exactGpuTotalBytes) / (1024.0 * 1024.0))
        << " vram_local_mib=" << (static_cast<double>(profiling.gpuLocalUsageBytes) / (1024.0 * 1024.0))
        << "/" << (static_cast<double>(profiling.gpuLocalBudgetBytes) / (1024.0 * 1024.0))
        << " upload_q=" << profiling.uploadQueueDepth
        << " prefetch_q=" << profiling.columnPrefetchQueueDepth
        << " player_released=" << (runtimeState.playerReleaseSeconds >= 0.0 ? 1 : 0)
        << " release_s=" << runtimeState.playerReleaseSeconds
        << " steady_state_s=" << runtimeState.steadyStateSeconds
        << " full_ready_s=" << runtimeState.fullExactReadySeconds
        << " yaw=" << camera.yaw
        << " pitch=" << camera.pitch
        << " stable_required_s=" << runtimeState.exactRequiredStableSeconds
        << " pos=(" << camera.position.x << "," << camera.position.y << "," << camera.position.z << ")";
    if (streamingStatus.blockingReason != nullptr && streamingStatus.blockingReason[0] != '\0')
    {
        out << " blocking=\"" << streamingStatus.blockingReason << "\"";
    }
    out << '\n';
}

void writeJsonEscaped(std::ostream& out, const std::string& text)
{
    out.put('"');
    for (char ch : text)
    {
        switch (ch)
        {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out.put(ch);
            break;
        }
    }
    out.put('"');
}

void writeVec3Json(std::ostream& out, const glm::vec3& value)
{
    out << '[' << value.x << ',' << value.y << ',' << value.z << ']';
}

void writeIvec3Json(std::ostream& out, const glm::ivec3& value)
{
    out << '[' << value.x << ',' << value.y << ',' << value.z << ']';
}

void writeStageStatsJson(std::ostream& out, const BenchmarkStageStats& stats)
{
    out << "{"
        << "\"count\":" << stats.count
        << ",\"total_ms\":" << stats.totalMs
        << ",\"avg_ms\":" << stats.averageMs
        << ",\"median_ms\":" << stats.medianMs
        << ",\"p95_ms\":" << stats.p95Ms
        << ",\"p99_ms\":" << stats.p99Ms
        << ",\"max_ms\":" << stats.maxMs
        << "}";
}

void writeCountStatsJson(std::ostream& out, const BenchmarkStageStats& stats)
{
    out << "{"
        << "\"samples\":" << stats.count
        << ",\"total\":" << stats.totalMs
        << ",\"avg\":" << stats.averageMs
        << ",\"median\":" << stats.medianMs
        << ",\"p95\":" << stats.p95Ms
        << ",\"p99\":" << stats.p99Ms
        << ",\"max\":" << stats.maxMs
        << "}";
}

void writeQueueStatsJson(std::ostream& out, const BenchmarkQueueDepthStats& stats)
{
    out << "{"
        << "\"samples\":" << stats.sampleCount
        << ",\"avg_depth\":" << stats.averageDepth
        << ",\"median_depth\":" << stats.medianDepth
        << ",\"p95_depth\":" << stats.p95Depth
        << ",\"max_depth\":" << stats.maxDepth
        << "}";
}

void writeCacheStatsJson(std::ostream& out, const BenchmarkCacheStats& stats)
{
    out << "{"
        << "\"hits\":" << stats.hits
        << ",\"misses\":" << stats.misses
        << ",\"fills\":" << stats.fills
        << ",\"hit_rate\":" << stats.hitRate
        << "}";
}

void writeSpikeSummaryJson(std::ostream& out, const BenchmarkSpikeSummary& summary)
{
    out << "{"
        << "\"count_over_16_7_ms\":" << summary.countOver16_7Ms
        << ",\"count_over_33_3_ms\":" << summary.countOver33_3Ms
        << ",\"count_over_50_ms\":" << summary.countOver50Ms
        << ",\"count_over_100_ms\":" << summary.countOver100Ms
        << ",\"count_over_250_ms\":" << summary.countOver250Ms
        << ",\"longest_streak_over_33_3_ms\":" << summary.longestStreakOver33_3Ms
        << ",\"worst\":[";
    for (std::size_t i = 0; i < summary.worstSpikes.size(); ++i)
    {
        if (i != 0)
        {
            out << ',';
        }

        const BenchmarkSpikeRecord& spike = summary.worstSpikes[i];
        out << "{"
            << "\"elapsed_s\":" << spike.elapsedSeconds
            << ",\"frame_ms\":" << spike.frameMs
            << ",\"camera\":";
        writeVec3Json(out, spike.cameraPosition);
        out << ",\"camera_chunk\":";
        writeIvec3Json(out, spike.cameraChunk);
        out << ",\"phase\":";
        writeJsonEscaped(out, spike.phase);
        out << ",\"blocking_reason\":";
        writeJsonEscaped(out, spike.blockingReason);
        out << ",\"suspected_source\":";
        writeJsonEscaped(out, spike.suspectedSource);
        out << ",\"chunk_update_ms\":" << spike.chunkUpdateMs
            << ",\"chunk_update_residual_ms\":" << spike.chunkUpdateResidualMs
            << ",\"chunk_dense_residency_ms\":" << spike.chunkDenseResidencyMs
            << ",\"chunk_vertical_radius_ms\":" << spike.chunkVerticalRadiusMs
            << ",\"chunk_priority_update_ms\":" << spike.chunkPriorityUpdateMs
            << ",\"chunk_upload_budget_ms\":" << spike.chunkUploadBudgetMs
            << ",\"chunk_missing_scan_ms\":" << spike.chunkMissingScanMs
            << ",\"chunk_ensure_volume_ms\":" << spike.chunkEnsureVolumeMs
            << ",\"chunk_scheduling_ms\":" << spike.chunkSchedulingMs
            << ",\"chunk_eviction_ms\":" << spike.chunkEvictionMs
            << ",\"chunk_relight_ms\":" << spike.chunkRelightMs
            << ",\"chunk_upload_ms\":" << spike.chunkUploadMs
            << ",\"chunk_upload_queue_age_ms\":" << spike.chunkUploadQueueAgeMs
            << ",\"chunk_upload_queue_pick_ms\":" << spike.chunkUploadQueuePickMs
            << ",\"chunk_pool_trim_ms\":" << spike.chunkPoolTrimMs
            << ",\"chunk_far_terrain_update_ms\":" << spike.chunkFarTerrainUpdateMs
            << ",\"chunk_column_height_lookup_ms\":" << spike.chunkColumnHeightLookupMs
            << ",\"chunk_column_height_sample_ms\":" << spike.chunkColumnHeightSampleMs
            << ",\"chunk_upload_prepare_ms\":" << spike.chunkUploadPrepareMs
            << ",\"chunk_upload_context_begin_ms\":" << spike.chunkUploadContextBeginMs
            << ",\"chunk_upload_finalize_ms\":" << spike.chunkUploadFinalizeMs
            << ",\"chunk_commit_collect_ms\":" << spike.chunkCommitCollectMs
            << ",\"chunk_commit_chunk_scan_ms\":" << spike.chunkCommitChunkScanMs
            << ",\"chunk_commit_mesh_lock_wait_ms\":" << spike.chunkCommitMeshLockWaitMs
            << ",\"chunk_commit_mesh_locked_ms\":" << spike.chunkCommitMeshLockedMs
            << ",\"chunk_commit_mesh_state_ms\":" << spike.chunkCommitMeshStateMs
            << ",\"chunk_commit_page_state_ms\":" << spike.chunkCommitPageStateMs
            << ",\"chunk_commit_release_ms\":" << spike.chunkCommitReleaseMs
            << ",\"chunk_startup_state_ms\":" << spike.chunkStartupStateMs
            << ",\"chunk_benchmark_bookkeeping_ms\":" << spike.chunkBenchmarkBookkeepingMs
            << ",\"poll_events_ms\":" << spike.pollEventsMs
            << ",\"build_render_data_ms\":" << spike.buildRenderDataMs
            << ",\"render_world_cpu_ms\":" << spike.renderWorldCpuMs
            << ",\"renderer_world_ms\":" << spike.rendererWorldMs
            << ",\"renderer_shadow_ms\":" << spike.rendererShadowMs
            << ",\"renderer_sky_ms\":" << spike.rendererSkyMs
            << ",\"renderer_atmosphere_ms\":" << spike.rendererAtmosphereMs
            << ",\"renderer_tonemap_ms\":" << spike.rendererToneMapMs
            << ",\"renderer_present_ms\":" << spike.rendererPresentMs
            << ",\"renderer_end_frame_ms\":" << spike.rendererEndFrameMs
            << ",\"chunk_vertical_radius\":" << spike.verticalRadius
            << ",\"chunk_vertical_radius_delta\":" << spike.verticalRadiusDelta
            << ",\"generated_this_frame\":" << spike.generatedChunks
            << ",\"relit_this_frame\":" << spike.relitChunks
            << ",\"relight_batches_this_frame\":" << spike.relightBatches
            << ",\"meshed_this_frame\":" << spike.meshedChunks
            << ",\"uploaded_this_frame\":" << spike.uploadedChunks
            << ",\"upload_attempts_this_frame\":" << spike.uploadAttempts
            << ",\"upload_queue_scan_entries_this_frame\":" << spike.uploadQueueScanEntries
            << ",\"upload_skipped_expired_this_frame\":" << spike.uploadSkippedExpired
            << ",\"upload_skipped_not_ready_this_frame\":" << spike.uploadSkippedNotReady
            << ",\"upload_skipped_pending_mesh_this_frame\":" << spike.uploadSkippedPendingMesh
            << ",\"upload_column_limited_this_frame\":" << spike.uploadColumnLimited
            << ",\"upload_budget_deferred_this_frame\":" << spike.uploadBudgetDeferred
            << ",\"upload_retry_failures_this_frame\":" << spike.uploadRetryFailures
            << ",\"upload_scan_limit_hits_this_frame\":" << spike.uploadScanLimitHits
            << ",\"upload_begin_failures_this_frame\":" << spike.uploadBeginFailures
            << ",\"upload_stale_pending_meshes_this_frame\":" << spike.uploadStalePendingMeshes
            << ",\"job_backlog\":" << spike.jobBacklog
            << ",\"upload_backlog\":" << spike.uploadBacklog
            << ",\"column_prefetch_backlog\":" << spike.columnPrefetchBacklog
            << ",\"exact_pending_chunks\":" << spike.exactPendingChunks
            << ",\"missing_chunks\":" << spike.missingChunks
            << ",\"relight_region_chunks_this_frame\":" << spike.relightRegionChunks
            << ",\"relight_changed_chunks_this_frame\":" << spike.relightChangedChunks
            << ",\"relight_external_snapshot_chunks_this_frame\":" << spike.relightExternalSnapshotChunks
            << ",\"relight_sky_above_chunk_scans_this_frame\":" << spike.relightSkyAboveChunkScans
            << ",\"relight_sky_seed_nodes_this_frame\":" << spike.relightSkySeedNodes
            << ",\"relight_block_seed_nodes_this_frame\":" << spike.relightBlockSeedNodes
            << ",\"relight_sky_nodes_processed_this_frame\":" << spike.relightSkyNodesProcessed
            << ",\"relight_block_nodes_processed_this_frame\":" << spike.relightBlockNodesProcessed
            << "}";
    }
    out << "]}";
}

bool writeBenchmarkScenarioJson(const BenchmarkConfig& config,
                                const BenchmarkRuntimeState& runtimeState,
                                const BenchmarkFrameSummary& frameSummary,
                                const ChunkBenchmarkReport& report,
                                const ChunkProfilingSnapshot& finalProfiling,
                                const StreamingStatusSnapshot& streamingStatus,
                                const RenderDistanceSettings& renderSettings)
{
    std::error_code ec;
    const std::filesystem::path parentPath = config.outputPath.parent_path();
    if (!parentPath.empty())
    {
        std::filesystem::create_directories(parentPath, ec);
        if (ec)
        {
            return false;
        }
    }

    std::ofstream out(config.outputPath, std::ios::trunc);
    if (!out)
    {
        return false;
    }

    out.setf(std::ios::fixed, std::ios::floatfield);
    out << std::setprecision(4);

    const BenchmarkStageStats lodGpuCullStats = summarizeStageSamples(runtimeState.lodGpuCullMs);
    const BenchmarkStageStats lodIndirectBuildStats = summarizeStageSamples(runtimeState.lodIndirectBuildMs);
    const auto chooseExactGpuStats = [](const BenchmarkStageStats& reportStats,
                                        const std::vector<double>& sampledStats) -> BenchmarkStageStats
    {
        return reportStats.count > 0 ? reportStats : summarizeStageSamples(sampledStats);
    };
    const BenchmarkStageStats exactGpuSynthStats =
        chooseExactGpuStats(report.exactGpuSynthStage, runtimeState.exactGpuSynthMs);
    const BenchmarkStageStats exactGpuStampStats =
        chooseExactGpuStats(report.exactGpuStampStage, runtimeState.exactGpuStampMs);
    const BenchmarkStageStats exactGpuLightStats =
        chooseExactGpuStats(report.exactGpuLightStage, runtimeState.exactGpuLightMs);
    const BenchmarkStageStats exactGpuFaceCountStats =
        chooseExactGpuStats(report.exactGpuFaceCountStage, runtimeState.exactGpuFaceCountMs);
    const BenchmarkStageStats exactGpuFacePrefixStats =
        chooseExactGpuStats(report.exactGpuFacePrefixStage, runtimeState.exactGpuFacePrefixMs);
    const BenchmarkStageStats exactGpuAllocateStats = report.exactGpuAllocateStage;
    const BenchmarkStageStats exactGpuFaceEmitStats =
        chooseExactGpuStats(report.exactGpuFaceEmitStage, runtimeState.exactGpuFaceEmitMs);
    const BenchmarkStageStats exactGpuTotalStats =
        chooseExactGpuStats(report.exactGpuTotalStage, runtimeState.exactGpuTotalMs);

    out << "{";
    out << "\"schema_version\":2";
    out << ",\"scenario\":";
    writeJsonEscaped(out, benchmarkScenarioName(config.scenario));
    out << ",\"build_config\":";
    writeJsonEscaped(out, config.buildConfig);
    out << ",\"progress_log_path\":";
    writeJsonEscaped(out, config.progressLogPath.string());
    out << ",\"completed\":" << (runtimeState.completed ? "true" : "false");
    out << ",\"timed_out\":" << (runtimeState.timedOut ? "true" : "false");
    out << ",\"duration_seconds\":" << runtimeState.elapsedSeconds;
    out << ",\"max_duration_seconds\":" << config.maxDurationSeconds;
    out << ",\"movement_seconds\":" << config.movementDurationSeconds;
    out << ",\"cooldown_seconds\":" << config.cooldownDurationSeconds;
    out << ",\"target_exact_ready_chunks\":" << config.targetExactReadyChunks;
    out << ",\"render_settings\":{"
        << "\"exact_chunks\":" << renderSettings.exactChunks
        << ",\"total_chunks\":" << renderSettings.totalChunks
        << ",\"fog_start_blocks\":" << renderSettings.fogStartBlocks
        << ",\"lod_mode\":\""
        << (renderSettings.totalChunks > renderSettings.exactChunks ? "cpu_lod_active" : "exact_only")
        << "\""
        << "}";
    out << ",\"camera\":{"
        << "\"spawn\":";
    writeVec3Json(out, runtimeState.spawnPosition);
    out << ",\"scenario_start\":";
    writeVec3Json(out, runtimeState.scenarioStartPosition);
    out << ",\"final\":";
    writeVec3Json(out, runtimeState.finalCameraPosition);
    out << ",\"final_yaw\":" << runtimeState.finalYawDegrees
        << ",\"final_pitch\":" << runtimeState.finalPitchDegrees
        << "}";
    out << ",\"milestones\":{"
        << "\"player_release_seconds\":" << runtimeState.playerReleaseSeconds
        << ",\"steady_state_seconds\":" << runtimeState.steadyStateSeconds
        << ",\"full_exact_ready_seconds\":" << runtimeState.fullExactReadySeconds
        << ",\"player_release_exact_ready_chunks\":" << runtimeState.playerReleaseExactReadyChunks
        << ",\"player_release_exact_required_chunks\":" << runtimeState.playerReleaseExactRequiredChunks
        << "}";
    out << ",\"frame\":{"
        << "\"count\":" << frameSummary.count
        << ",\"avg_ms\":" << frameSummary.averageMs
        << ",\"median_ms\":" << frameSummary.medianMs
        << ",\"p95_ms\":" << frameSummary.p95Ms
        << ",\"max_ms\":" << frameSummary.maxMs
        << ",\"avg_fps\":" << frameSummary.averageFps
        << "}";
    out << ",\"frame_spikes\":";
    writeSpikeSummaryJson(out, runtimeState.spikeSummary);
    out << ",\"throughput\":{"
        << "\"generated_chunks\":" << report.generatedChunks
        << ",\"meshed_chunks\":" << report.meshedChunks
        << ",\"uploaded_chunks\":" << report.uploadedChunks
        << ",\"exact_gpu_builds_submitted\":" << finalProfiling.exactGpuBuildsSubmitted
        << ",\"exact_gpu_builds_committed\":" << finalProfiling.exactGpuBuildsCommitted
        << ",\"far_built_tiles\":" << report.farBuiltTiles
        << ",\"uploaded_bytes\":" << report.uploadedBytes
        << ",\"generated_chunks_per_sec\":"
        << (runtimeState.elapsedSeconds > 0.0 ? static_cast<double>(report.generatedChunks) / runtimeState.elapsedSeconds : 0.0)
        << ",\"meshed_chunks_per_sec\":"
        << (runtimeState.elapsedSeconds > 0.0 ? static_cast<double>(report.meshedChunks) / runtimeState.elapsedSeconds : 0.0)
        << ",\"uploaded_chunks_per_sec\":"
        << (runtimeState.elapsedSeconds > 0.0 ? static_cast<double>(report.uploadedChunks) / runtimeState.elapsedSeconds : 0.0)
        << ",\"exact_gpu_builds_submitted_per_sec\":"
        << (runtimeState.elapsedSeconds > 0.0 ? static_cast<double>(finalProfiling.exactGpuBuildsSubmitted) / runtimeState.elapsedSeconds : 0.0)
        << ",\"exact_gpu_builds_committed_per_sec\":"
        << (runtimeState.elapsedSeconds > 0.0 ? static_cast<double>(finalProfiling.exactGpuBuildsCommitted) / runtimeState.elapsedSeconds : 0.0)
        << "}";
    out << ",\"stages\":{"
        << "\"sample\":";
    writeStageStatsJson(out, report.sampleStage);
    out << ",\"generate\":";
    writeStageStatsJson(out, report.generateStage);
    out << ",\"relight\":";
    writeStageStatsJson(out, report.relightStage);
    out << ",\"mesh\":";
    writeStageStatsJson(out, report.meshStage);
    out << ",\"upload\":";
    writeStageStatsJson(out, report.uploadStage);
    out << ",\"update\":";
    writeStageStatsJson(out, report.updateStage);
    out << ",\"update_residual\":";
    writeStageStatsJson(out, report.updateResidualStage);
    out << ",\"dense_residency\":";
    writeStageStatsJson(out, report.denseResidencyStage);
    out << ",\"vertical_radius\":";
    writeStageStatsJson(out, report.verticalRadiusStage);
    out << ",\"priority_update\":";
    writeStageStatsJson(out, report.priorityUpdateStage);
    out << ",\"upload_budget_prep\":";
    writeStageStatsJson(out, report.uploadBudgetPrepStage);
    out << ",\"visible_scan\":";
    writeStageStatsJson(out, report.visibleScanStage);
    out << ",\"ensure_volume\":";
    writeStageStatsJson(out, report.ensureVolumeStage);
    out << ",\"ensure_volume_column_prep\":";
    writeStageStatsJson(out, report.ensureVolumeColumnPrepStage);
    out << ",\"ensure_volume_sort\":";
    writeStageStatsJson(out, report.ensureVolumeSortStage);
    out << ",\"ensure_volume_dispatch\":";
    writeStageStatsJson(out, report.ensureVolumeDispatchStage);
    out << ",\"ensure_volume_chunk_lookup\":";
    writeStageStatsJson(out, report.ensureVolumeChunkLookupStage);
    out << ",\"ensure_volume_enqueue\":";
    writeStageStatsJson(out, report.ensureVolumeEnqueueStage);
    out << ",\"scheduling\":";
    writeStageStatsJson(out, report.schedulingStage);
    out << ",\"eviction\":";
    writeStageStatsJson(out, report.evictionStage);
    out << ",\"main_thread_relight\":";
    writeStageStatsJson(out, report.mainThreadRelightStage);
    out << ",\"upload_drain\":";
    writeStageStatsJson(out, report.uploadDrainStage);
    out << ",\"upload_queue_pick\":";
    writeStageStatsJson(out, report.uploadQueuePickStage);
    out << ",\"pool_trim\":";
    writeStageStatsJson(out, report.poolTrimStage);
    out << ",\"far_terrain_update\":";
    writeStageStatsJson(out, report.farTerrainUpdateStage);
    out << ",\"column_height_lookup\":";
    writeStageStatsJson(out, report.columnHeightLookupStage);
    out << ",\"column_height_sample\":";
    writeStageStatsJson(out, report.columnHeightSampleStage);
    out << ",\"upload_prepare\":";
    writeStageStatsJson(out, report.uploadPrepareStage);
    out << ",\"upload_context_begin\":";
    writeStageStatsJson(out, report.uploadContextBeginStage);
    out << ",\"upload_finalize\":";
    writeStageStatsJson(out, report.uploadFinalizeStage);
    out << ",\"commit_collect\":";
    writeStageStatsJson(out, report.commitCollectStage);
    out << ",\"commit_chunk_scan\":";
    writeStageStatsJson(out, report.commitChunkScanStage);
    out << ",\"commit_mesh_lock_wait\":";
    writeStageStatsJson(out, report.commitMeshLockWaitStage);
    out << ",\"commit_mesh_locked\":";
    writeStageStatsJson(out, report.commitMeshLockedStage);
    out << ",\"commit_mesh_state\":";
    writeStageStatsJson(out, report.commitMeshStateStage);
    out << ",\"commit_page_state\":";
    writeStageStatsJson(out, report.commitPageStateStage);
    out << ",\"commit_release\":";
    writeStageStatsJson(out, report.commitReleaseStage);
    out << ",\"generate_blocks_mesh_lock\":";
    writeStageStatsJson(out, report.generateBlocksMeshLockStage);
    out << ",\"upload_chunk_mesh_lock\":";
    writeStageStatsJson(out, report.uploadChunkMeshLockStage);
    out << ",\"neighborhood_snapshot_lock\":";
    writeStageStatsJson(out, report.neighborhoodSnapshotLockStage);
    out << ",\"sky_light_cache_lock\":";
    writeStageStatsJson(out, report.skyLightCacheLockStage);
    out << ",\"startup_state\":";
    writeStageStatsJson(out, report.startupStateStage);
    out << ",\"benchmark_bookkeeping\":";
    writeStageStatsJson(out, report.benchmarkBookkeepingStage);
    out << ",\"far_build\":";
    writeStageStatsJson(out, report.farBuildStage);
    out << ",\"lod_gpu_synthesis\":";
    writeStageStatsJson(out, report.lodGpuSynthesisStage);
    out << ",\"lod_gpu_stamp\":";
    writeStageStatsJson(out, report.lodGpuStampStage);
    out << ",\"lod_gpu_face_build\":";
    writeStageStatsJson(out, report.lodGpuFaceBuildStage);
    out << ",\"lod_gpu_cull\":";
    writeStageStatsJson(out, lodGpuCullStats);
    out << ",\"lod_indirect_build\":";
    writeStageStatsJson(out, lodIndirectBuildStats);
    out << ",\"exact_gpu_synth\":";
    writeStageStatsJson(out, exactGpuSynthStats);
    out << ",\"exact_gpu_stamp\":";
    writeStageStatsJson(out, exactGpuStampStats);
    out << ",\"exact_gpu_light\":";
    writeStageStatsJson(out, exactGpuLightStats);
    out << ",\"exact_gpu_face_count\":";
    writeStageStatsJson(out, exactGpuFaceCountStats);
    out << ",\"exact_gpu_face_prefix\":";
    writeStageStatsJson(out, exactGpuFacePrefixStats);
    out << ",\"exact_gpu_allocate\":";
    writeStageStatsJson(out, exactGpuAllocateStats);
    out << ",\"exact_gpu_face_emit\":";
    writeStageStatsJson(out, exactGpuFaceEmitStats);
    out << ",\"exact_gpu_total\":";
    writeStageStatsJson(out, exactGpuTotalStats);
    out << ",\"exact_gpu_prepare_cpu\":";
    writeStageStatsJson(out, report.exactGpuPrepareCpuStage);
    out << ",\"exact_gpu_submit_cpu\":";
    writeStageStatsJson(out, report.exactGpuSubmitCpuStage);
    out << ",\"exact_gpu_commit_cpu\":";
    writeStageStatsJson(out, report.exactGpuCommitCpuStage);
    out << ",\"exact_gpu_worldgen_resolve\":";
    writeStageStatsJson(out, report.exactGpuWorldgenResolveStage);
    out << ",\"exact_gpu_prepass_face_totals_readback\":";
    writeStageStatsJson(out, report.exactGpuPrepassFaceTotalsReadbackStage);
    out << ",\"exact_gpu_emit_metadata_sync\":";
    writeStageStatsJson(out, report.exactGpuEmitMetadataSyncStage);
    out << ",\"exact_gpu_page_sweep\":";
    writeStageStatsJson(out, report.exactGpuPageSweepStage);
    out << ",\"exact_gpu_emit_wait\":";
    writeStageStatsJson(out, report.exactGpuEmitWaitStage);
    out << ",\"exact_gpu_emit_fence_lifetime\":";
    writeStageStatsJson(out, report.exactGpuEmitFenceLifetimeStage);
    out << ",\"exact_gpu_worldgen_page_misses\":";
    writeCountStatsJson(out, report.exactGpuWorldgenPageMisses);
    out << ",\"exact_gpu_emit_metadata_dirty_pages\":";
    writeCountStatsJson(out, report.exactGpuEmitMetadataDirtyPages);
    out << ",\"exact_gpu_emit_metadata_upload_bytes\":";
    writeCountStatsJson(out, report.exactGpuEmitMetadataUploadBytes);
    out << ",\"exact_gpu_page_sweep_pages\":";
    writeCountStatsJson(out, report.exactGpuPageSweepPages);
    out << ",\"exact_gpu_ready_for_emit_backlog_batches\":";
    writeCountStatsJson(out, report.exactGpuReadyForEmitBacklogBatches);
    out << ",\"exact_gpu_ready_for_emit_backlog_builds\":";
    writeCountStatsJson(out, report.exactGpuReadyForEmitBacklogBuilds);
    out << ",\"exact_gpu_compute_in_flight_before_emit\":";
    writeCountStatsJson(out, report.exactGpuComputeInFlightBeforeEmit);
    out << ",\"exact_gpu_compute_in_flight_after_emit\":";
    writeCountStatsJson(out, report.exactGpuComputeInFlightAfterEmit);
    out << ",\"exact_gpu_blocking_emit_batch_age\":";
    writeStageStatsJson(out, report.exactGpuBlockingEmitBatchAgeStage);
    out << ",\"exact_gpu_blocking_emit_batch_builds\":";
    writeCountStatsJson(out, report.exactGpuBlockingEmitBatchBuilds);
    out << ",\"exact_gpu_submit_batch_builds\":";
    writeCountStatsJson(out, report.exactGpuSubmitBatchBuilds);
    out << ",\"exact_gpu_emit_batch_builds\":";
    writeCountStatsJson(out, report.exactGpuEmitBatchBuilds);
    out << ",\"chunk_ready_latency\":";
    writeStageStatsJson(out, report.chunkReadyLatency);
    out << ",\"chunk_ready_wait_generate\":";
    writeStageStatsJson(out, report.chunkReadyWaitGenerateStage);
    out << ",\"chunk_ready_request_queued_generate\":";
    writeStageStatsJson(out, report.chunkReadyRequestQueuedGenerateStage);
    out << ",\"chunk_ready_request_queued_mesh\":";
    writeStageStatsJson(out, report.chunkReadyRequestQueuedMeshStage);
    out << ",\"chunk_ready_request_queued_prefetch\":";
    writeStageStatsJson(out, report.chunkReadyRequestQueuedPrefetchStage);
    out << ",\"chunk_ready_request_queued_bulk\":";
    writeStageStatsJson(out, report.chunkReadyRequestQueuedBulkStage);
    out << ",\"chunk_ready_request_latency_sensitive_outstanding\":";
    writeStageStatsJson(out, report.chunkReadyRequestLatencySensitiveOutstandingStage);
    out << ",\"chunk_ready_start_queued_generate\":";
    writeStageStatsJson(out, report.chunkReadyStartQueuedGenerateStage);
    out << ",\"chunk_ready_start_queued_mesh\":";
    writeStageStatsJson(out, report.chunkReadyStartQueuedMeshStage);
    out << ",\"chunk_ready_start_queued_prefetch\":";
    writeStageStatsJson(out, report.chunkReadyStartQueuedPrefetchStage);
    out << ",\"chunk_ready_start_queued_bulk\":";
    writeStageStatsJson(out, report.chunkReadyStartQueuedBulkStage);
    out << ",\"chunk_ready_start_active_generate\":";
    writeStageStatsJson(out, report.chunkReadyStartActiveGenerateStage);
    out << ",\"chunk_ready_start_active_mesh\":";
    writeStageStatsJson(out, report.chunkReadyStartActiveMeshStage);
    out << ",\"chunk_ready_start_active_prefetch\":";
    writeStageStatsJson(out, report.chunkReadyStartActivePrefetchStage);
    out << ",\"chunk_ready_start_active_bulk\":";
    writeStageStatsJson(out, report.chunkReadyStartActiveBulkStage);
    out << ",\"chunk_ready_start_latency_sensitive_outstanding\":";
    writeStageStatsJson(out, report.chunkReadyStartLatencySensitiveOutstandingStage);
    out << ",\"chunk_ready_generate\":";
    writeStageStatsJson(out, report.chunkReadyGenerateStage);
    out << ",\"chunk_ready_wait_mesh_enqueue\":";
    writeStageStatsJson(out, report.chunkReadyWaitMeshEnqueueStage);
    out << ",\"chunk_ready_wait_mesh_start\":";
    writeStageStatsJson(out, report.chunkReadyWaitMeshStartStage);
    out << ",\"chunk_ready_mesh\":";
    writeStageStatsJson(out, report.chunkReadyMeshStage);
    out << ",\"chunk_ready_wait_upload\":";
    writeStageStatsJson(out, report.chunkReadyWaitUploadStage);
    out << ",\"chunk_ready_upload_to_ready\":";
    writeStageStatsJson(out, report.chunkReadyUploadToReadyStage);
    out << ",\"upload_queue_age\":";
    writeStageStatsJson(out, report.uploadQueueAgeStage);
    out << ",\"structure_query\":";
    writeStageStatsJson(out, report.structureQueryStage);
    out << ",\"ensure_volume_columns_visited\":";
    writeStageStatsJson(out, report.ensureVolumeColumnsVisited);
    out << ",\"ensure_volume_candidates_built\":";
    writeStageStatsJson(out, report.ensureVolumeCandidatesBuilt);
    out << ",\"ensure_volume_existing_chunk_skips\":";
    writeStageStatsJson(out, report.ensureVolumeExistingChunkSkips);
    out << ",\"ensure_volume_column_cap_skips\":";
    writeStageStatsJson(out, report.ensureVolumeColumnCapSkips);
    out << "}";
    out << ",\"upload_detail\":{"
        << "\"queue_scan_entries\":";
    writeCountStatsJson(out, report.uploadQueueScanEntries);
    out << ",\"attempts_per_frame\":";
    writeCountStatsJson(out, report.uploadAttemptsPerFrame);
    out << ",\"uploaded_chunks_per_frame\":";
    writeCountStatsJson(out, report.uploadChunksPerFrame);
    out << ",\"uploaded_bytes_per_frame\":";
    writeCountStatsJson(out, report.uploadBytesPerFrame);
    out << ",\"expired_entries_per_frame\":";
    writeCountStatsJson(out, report.uploadExpiredEntriesPerFrame);
    out << ",\"skipped_not_ready_per_frame\":";
    writeCountStatsJson(out, report.uploadSkippedNotReadyPerFrame);
    out << ",\"skipped_pending_mesh_per_frame\":";
    writeCountStatsJson(out, report.uploadSkippedPendingMeshPerFrame);
    out << ",\"column_limited_per_frame\":";
    writeCountStatsJson(out, report.uploadColumnLimitedPerFrame);
    out << ",\"budget_deferred_per_frame\":";
    writeCountStatsJson(out, report.uploadBudgetDeferredPerFrame);
    out << ",\"retry_failures_per_frame\":";
    writeCountStatsJson(out, report.uploadRetryFailuresPerFrame);
    out << ",\"scan_limit_hits_per_frame\":";
    writeCountStatsJson(out, report.uploadScanLimitHitsPerFrame);
    out << ",\"begin_failures_per_frame\":";
    writeCountStatsJson(out, report.uploadBeginFailuresPerFrame);
    out << ",\"stale_pending_meshes_per_frame\":";
    writeCountStatsJson(out, report.uploadStalePendingMeshesPerFrame);
    out << "}";
    out << ",\"relight_detail\":{"
        << "\"vertical_radius_delta\":";
    writeCountStatsJson(out, report.verticalRadiusDelta);
    out << ",\"region_chunks\":";
    writeCountStatsJson(out, report.relightRegionChunks);
    out << ",\"changed_chunks\":";
    writeCountStatsJson(out, report.relightChangedChunks);
    out << ",\"external_snapshot_chunks\":";
    writeCountStatsJson(out, report.relightExternalSnapshotChunks);
    out << ",\"sky_above_chunk_scans\":";
    writeCountStatsJson(out, report.relightSkyAboveChunkScans);
    out << ",\"sky_seed_nodes\":";
    writeCountStatsJson(out, report.relightSkySeedNodes);
    out << ",\"block_seed_nodes\":";
    writeCountStatsJson(out, report.relightBlockSeedNodes);
    out << ",\"sky_nodes_processed\":";
    writeCountStatsJson(out, report.relightSkyNodesProcessed);
    out << ",\"block_nodes_processed\":";
    writeCountStatsJson(out, report.relightBlockNodesProcessed);
    out << "}";
    out << ",\"queues\":{"
        << "\"job_backlog\":";
    writeQueueStatsJson(out, report.jobQueueDepth);
    out << ",\"upload_backlog\":";
    writeQueueStatsJson(out, report.uploadQueueDepth);
    out << ",\"column_prefetch_backlog\":";
    writeQueueStatsJson(out, report.columnPrefetchQueueDepth);
    out << ",\"far_build_backlog\":";
    writeQueueStatsJson(out, report.farBuildQueueDepth);
    out << ",\"far_upload_backlog\":";
    writeQueueStatsJson(out, report.farUploadQueueDepth);
    out << "}";
    out << ",\"cache\":{"
        << "\"climate\":";
    writeCacheStatsJson(out, report.climateCache);
    out << ",\"surface\":";
    writeCacheStatsJson(out, report.surfaceCache);
    out << ",\"structure\":";
    writeCacheStatsJson(out, report.structureCache);
    out << "}";
    out << ",\"structures\":{"
        << "\"regions_built\":" << report.structureRegionsBuilt
        << ",\"query_avg_ms\":" << report.structureQueryStage.averageMs
        << ",\"cache_hit_rate\":" << report.structureCache.hitRate
        << "}";
    out << ",\"final_profiling\":{"
        << "\"pooled_chunks\":" << finalProfiling.pooledChunkCount
        << ",\"pooled_chunk_bytes\":" << finalProfiling.pooledChunkBytes
        << ",\"pooled_chunk_budget_bytes\":" << finalProfiling.pooledChunkBudgetBytes
        << ",\"vertical_radius\":" << finalProfiling.verticalRadius
        << ",\"vertical_radius_delta\":" << finalProfiling.verticalRadiusDelta
        << ",\"visible_scan_ms\":" << finalProfiling.missingScanMsLastFrame
        << ",\"update_residual_ms\":" << finalProfiling.updateResidualMsLastFrame
        << ",\"dense_residency_ms\":" << finalProfiling.denseResidencyMsLastFrame
        << ",\"ensure_volume_ms\":" << finalProfiling.ensureVolumeMsLastFrame
        << ",\"ensure_volume_column_prep_ms\":" << finalProfiling.ensureVolumeColumnPrepMsLastFrame
        << ",\"ensure_volume_sort_ms\":" << finalProfiling.ensureVolumeSortMsLastFrame
        << ",\"ensure_volume_dispatch_ms\":" << finalProfiling.ensureVolumeDispatchMsLastFrame
        << ",\"ensure_volume_chunk_lookup_ms\":" << finalProfiling.ensureVolumeChunkLookupMsLastFrame
        << ",\"ensure_volume_enqueue_ms\":" << finalProfiling.ensureVolumeEnqueueMsLastFrame
        << ",\"scheduling_ms\":" << finalProfiling.schedulingMsLastFrame
        << ",\"eviction_ms\":" << finalProfiling.evictionMsLastFrame
        << ",\"upload_drain_ms\":" << finalProfiling.uploadMsLastFrame
        << ",\"upload_queue_age_ms\":" << finalProfiling.uploadQueueAgeMsLastFrame
        << ",\"upload_queue_pick_ms\":" << finalProfiling.uploadQueuePickMsLastFrame
        << ",\"upload_attempts_last_frame\":" << finalProfiling.uploadAttemptsLastFrame
        << ",\"upload_queue_scan_entries_last_frame\":" << finalProfiling.uploadQueueScanEntriesLastFrame
        << ",\"upload_skipped_expired_last_frame\":" << finalProfiling.uploadSkippedExpiredLastFrame
        << ",\"upload_skipped_not_ready_last_frame\":" << finalProfiling.uploadSkippedNotReadyLastFrame
        << ",\"upload_skipped_pending_mesh_last_frame\":" << finalProfiling.uploadSkippedPendingMeshLastFrame
        << ",\"upload_column_limited_last_frame\":" << finalProfiling.uploadColumnLimitedLastFrame
        << ",\"upload_budget_deferred_last_frame\":" << finalProfiling.uploadBudgetDeferredLastFrame
        << ",\"upload_retry_failures_last_frame\":" << finalProfiling.uploadRetryFailuresLastFrame
        << ",\"upload_scan_limit_hits_last_frame\":" << finalProfiling.uploadScanLimitHitsLastFrame
        << ",\"upload_begin_failures_last_frame\":" << finalProfiling.uploadBeginFailuresLastFrame
        << ",\"upload_stale_pending_meshes_last_frame\":" << finalProfiling.uploadStalePendingMeshesLastFrame
        << ",\"uploaded_bytes_last_frame\":" << finalProfiling.uploadedBytesLastFrame
        << ",\"far_terrain_update_ms\":" << finalProfiling.farTerrainUpdateMsLastFrame
        << ",\"column_height_lookup_ms\":" << finalProfiling.columnHeightLookupMsLastFrame
        << ",\"column_height_sample_ms\":" << finalProfiling.columnHeightSampleMsLastFrame
        << ",\"upload_prepare_ms\":" << finalProfiling.uploadPrepareMsLastFrame
        << ",\"upload_context_begin_ms\":" << finalProfiling.uploadContextBeginMsLastFrame
        << ",\"upload_finalize_ms\":" << finalProfiling.uploadFinalizeMsLastFrame
        << ",\"commit_collect_ms\":" << finalProfiling.commitCollectMsLastFrame
        << ",\"commit_chunk_scan_ms\":" << finalProfiling.commitChunkScanMsLastFrame
        << ",\"commit_mesh_lock_wait_ms\":" << finalProfiling.commitMeshLockWaitMsLastFrame
        << ",\"commit_mesh_locked_ms\":" << finalProfiling.commitMeshLockedMsLastFrame
        << ",\"commit_mesh_state_ms\":" << finalProfiling.commitMeshStateMsLastFrame
        << ",\"commit_page_state_ms\":" << finalProfiling.commitPageStateMsLastFrame
        << ",\"commit_release_ms\":" << finalProfiling.commitReleaseMsLastFrame
        << ",\"relight_region_chunks_last_frame\":" << finalProfiling.relightRegionChunks
        << ",\"relight_changed_chunks_last_frame\":" << finalProfiling.relightChangedChunks
        << ",\"relight_external_snapshot_chunks_last_frame\":" << finalProfiling.relightExternalSnapshotChunks
        << ",\"relight_sky_above_chunk_scans_last_frame\":" << finalProfiling.relightSkyAboveChunkScans
        << ",\"relight_sky_seed_nodes_last_frame\":" << finalProfiling.relightSkySeedNodes
        << ",\"relight_block_seed_nodes_last_frame\":" << finalProfiling.relightBlockSeedNodes
        << ",\"relight_sky_nodes_processed_last_frame\":" << finalProfiling.relightSkyNodesProcessed
        << ",\"relight_block_nodes_processed_last_frame\":" << finalProfiling.relightBlockNodesProcessed
        << ",\"lod_active_tiles\":" << finalProfiling.farActiveTiles
        << ",\"lod_dirty_tiles\":" << finalProfiling.farDirtyTiles
        << ",\"lod_ready_tiles\":" << finalProfiling.farShellTilesReady
        << ",\"lod_tiles_built_last_update\":" << finalProfiling.farTilesBuilt
        << ",\"lod_tiles_queued\":" << finalProfiling.farTilesQueued
        << ",\"lod_tiles_pending_upload\":" << finalProfiling.farTilesPendingUpload
        << ",\"lod_build_avg_ms\":" << finalProfiling.farBuildMsAverage
        << ",\"lod_gpu_synthesis_ms\":" << finalProfiling.lodGpuSynthesisMs
        << ",\"lod_gpu_stamp_ms\":" << finalProfiling.lodGpuStampMs
        << ",\"lod_gpu_face_build_ms\":" << finalProfiling.lodGpuFaceBuildMs
        << ",\"lod_gpu_cull_ms\":" << finalProfiling.lodGpuCullMs
        << ",\"lod_indirect_build_ms\":" << finalProfiling.lodIndirectBuildMs
        << ",\"exact_gpu_synth_ms\":" << finalProfiling.exactGpuSynthMs
        << ",\"exact_gpu_stamp_ms\":" << finalProfiling.exactGpuStampMs
        << ",\"exact_gpu_light_ms\":" << finalProfiling.exactGpuLightMs
        << ",\"exact_gpu_face_count_ms\":" << finalProfiling.exactGpuFaceCountMs
        << ",\"exact_gpu_face_prefix_ms\":" << finalProfiling.exactGpuFacePrefixMs
        << ",\"exact_gpu_allocate_ms\":" << finalProfiling.exactGpuAllocateMs
        << ",\"exact_gpu_face_emit_ms\":" << finalProfiling.exactGpuFaceEmitMs
        << ",\"exact_gpu_total_ms\":" << finalProfiling.exactGpuTotalMs
        << ",\"exact_gpu_prepare_cpu_ms\":" << finalProfiling.exactGpuPrepareCpuMs
        << ",\"exact_gpu_submit_cpu_ms\":" << finalProfiling.exactGpuSubmitCpuMs
        << ",\"exact_gpu_commit_cpu_ms\":" << finalProfiling.exactGpuCommitCpuMs
        << ",\"exact_gpu_worldgen_resolve_ms\":" << finalProfiling.exactGpuWorldgenResolveMsLastCycle
        << ",\"exact_gpu_prepass_face_totals_readback_ms\":"
        << finalProfiling.exactGpuPrepassFaceTotalsReadbackMsLastCycle
        << ",\"exact_gpu_emit_metadata_sync_ms\":" << finalProfiling.exactGpuEmitMetadataSyncMsLastCycle
        << ",\"exact_gpu_page_sweep_ms\":" << finalProfiling.exactGpuPageSweepMsLastCycle
        << ",\"exact_gpu_emit_wait_ms\":" << finalProfiling.exactGpuEmitWaitMsLastCycle
        << ",\"exact_gpu_emit_fence_lifetime_ms\":" << finalProfiling.exactGpuEmitFenceLifetimeMsLastCycle
        << ",\"exact_gpu_worldgen_page_misses_last_cycle\":" << finalProfiling.exactGpuWorldgenPageMissesLastCycle
        << ",\"exact_gpu_emit_metadata_dirty_pages_last_cycle\":" << finalProfiling.exactGpuEmitMetadataDirtyPagesLastCycle
        << ",\"exact_gpu_emit_metadata_upload_bytes_last_cycle\":" << finalProfiling.exactGpuEmitMetadataUploadBytesLastCycle
        << ",\"exact_gpu_page_sweep_pages_last_cycle\":" << finalProfiling.exactGpuPageSweepPagesLastCycle
        << ",\"exact_gpu_ready_for_emit_backlog_batches_last_cycle\":" << finalProfiling.exactGpuReadyForEmitBacklogBatchesLastCycle
        << ",\"exact_gpu_ready_for_emit_backlog_builds_last_cycle\":" << finalProfiling.exactGpuReadyForEmitBacklogBuildsLastCycle
        << ",\"exact_gpu_compute_in_flight_before_emit_last_cycle\":" << finalProfiling.exactGpuComputeInFlightBeforeEmitLastCycle
        << ",\"exact_gpu_compute_in_flight_after_emit_last_cycle\":" << finalProfiling.exactGpuComputeInFlightAfterEmitLastCycle
        << ",\"exact_gpu_blocking_emit_batch_age_ms_last_cycle\":" << finalProfiling.exactGpuBlockingEmitBatchAgeMsLastCycle
        << ",\"exact_gpu_blocking_emit_batch_builds_last_cycle\":" << finalProfiling.exactGpuBlockingEmitBatchBuildsLastCycle
        << ",\"exact_gpu_submit_batch_builds_last_cycle\":" << finalProfiling.exactGpuSubmitBatchBuildsLastCycle
        << ",\"exact_gpu_emit_batch_builds_last_cycle\":" << finalProfiling.exactGpuEmitBatchBuildsLastCycle
        << ",\"lod_collect_ms\":" << finalProfiling.farCollectMsLastFrame
        << ",\"lod_upload_ms\":" << finalProfiling.farUploadMsLastFrame
        << ",\"structure_query_ms\":" << finalProfiling.structureQueryMs
        << ",\"structure_cache_hit_rate\":" << finalProfiling.structureCacheHitRate
        << ",\"structure_regions_built\":" << finalProfiling.structureRegionsBuilt
        << ",\"exact_cpu_authoritative_chunks\":" << finalProfiling.exactCpuAuthoritativeChunks
        << ",\"exact_gpu_resident_nonlocal_chunks\":" << finalProfiling.exactGpuResidentNonlocalChunks
        << ",\"exact_cpu_materializing_chunks\":" << finalProfiling.exactCpuMaterializingChunks
        << ",\"exact_gpu_pending_retire_chunks\":" << finalProfiling.exactGpuPendingRetireChunks
        << ",\"exact_gpu_page_bytes\":" << finalProfiling.exactGpuPageBytes
        << ",\"exact_gpu_column_bytes\":" << finalProfiling.exactGpuColumnBytes
        << ",\"exact_gpu_sparse_voxel_bytes\":" << finalProfiling.exactGpuSparseVoxelBytes
        << ",\"exact_gpu_voxel_bytes\":" << finalProfiling.exactGpuVoxelBytes
        << ",\"exact_gpu_light_scratch_bytes\":" << finalProfiling.exactGpuLightScratchBytes
        << ",\"exact_gpu_scratch_bytes\":" << finalProfiling.exactGpuScratchBytes
        << ",\"exact_gpu_upload_scratch_bytes\":" << finalProfiling.exactGpuUploadScratchBytes
        << ",\"exact_gpu_readback_bytes\":" << finalProfiling.exactGpuReadbackBytes
        << ",\"exact_gpu_total_bytes\":" << finalProfiling.exactGpuTotalBytes
        << ",\"gpu_local_usage_bytes\":" << finalProfiling.gpuLocalUsageBytes
        << ",\"gpu_local_budget_bytes\":" << finalProfiling.gpuLocalBudgetBytes
        << ",\"gpu_local_available_for_reservation_bytes\":" << finalProfiling.gpuLocalAvailableForReservationBytes
        << ",\"gpu_non_local_usage_bytes\":" << finalProfiling.gpuNonLocalUsageBytes
        << ",\"gpu_non_local_budget_bytes\":" << finalProfiling.gpuNonLocalBudgetBytes
        << ",\"ensure_volume_columns_visited_last_frame\":" << finalProfiling.ensureVolumeColumnsVisitedLastFrame
        << ",\"ensure_volume_candidates_built_last_frame\":" << finalProfiling.ensureVolumeCandidatesBuiltLastFrame
        << ",\"ensure_volume_existing_chunk_skips_last_frame\":" << finalProfiling.ensureVolumeExistingChunkSkipsLastFrame
        << ",\"ensure_volume_column_cap_skips_last_frame\":" << finalProfiling.ensureVolumeColumnCapSkipsLastFrame
        << ",\"column_prefetch_queue_depth\":" << finalProfiling.columnPrefetchQueueDepth
        << "}";
    out << ",\"final_streaming\":{"
        << "\"phase\":";
    writeJsonEscaped(out, streamingPhaseName(streamingStatus.phase));
    out << ",\"exact_ready_chunks\":" << streamingStatus.exactReadyChunks
        << ",\"exact_required_chunks\":" << streamingStatus.exactRequiredChunks
        << ",\"exact_pending_uploads\":" << streamingStatus.exactPendingUploads
        << ",\"far_active_tiles\":" << streamingStatus.farActiveTiles
        << ",\"far_dirty_tiles\":" << streamingStatus.farDirtyTiles
        << ",\"far_ready_tiles\":" << streamingStatus.farReadyTiles
        << ",\"far_queued_tiles\":" << streamingStatus.farQueuedTiles
        << ",\"far_pending_upload_tiles\":" << streamingStatus.farPendingUploadTiles
        << ",\"lod_active_tiles\":" << streamingStatus.farActiveTiles
        << ",\"lod_dirty_tiles\":" << streamingStatus.farDirtyTiles
        << ",\"lod_ready_tiles\":" << streamingStatus.farReadyTiles
        << ",\"lod_queued_tiles\":" << streamingStatus.farQueuedTiles
        << ",\"lod_pending_upload_tiles\":" << streamingStatus.farPendingUploadTiles
        << ",\"player_release_ready\":" << (streamingStatus.playerReleaseReady ? "true" : "false")
        << ",\"blocking_reason\":";
    writeJsonEscaped(out, streamingStatus.blockingReason ? streamingStatus.blockingReason : "");
    out << "}";
    out << "}";
    return true;
}

struct ScreenshotSweepConfig
{
    bool enabled{false};
    std::filesystem::path outputDir{};
    int initialSettleFrames{180};
    int settleFramesPerCapture{18};
    float heightOffsetBlocks{96.0f};
    std::vector<float> pitches{-8.0f, -4.0f, -2.0f, 0.0f, 2.0f, 4.0f, 8.0f, 15.0f, 30.0f, 45.0f};
    std::vector<float> yaws{0.0f, 30.0f, 60.0f, 90.0f, 120.0f, 150.0f, 180.0f, 210.0f, 240.0f, 270.0f, 300.0f, 330.0f};
};

struct ScreenshotSweepState
{
    bool initialized{false};
    glm::vec3 anchorPosition{0.0f};
    std::size_t poseIndex{0};
    int waitFramesRemaining{0};
    std::ofstream manifest;
};

struct ScreenshotReproConfig
{
    bool enabled{false};
    glm::vec3 position{0.0f};
    float yawDegrees{0.0f};
    float pitchDegrees{0.0f};
    bool useLookTarget{false};
    glm::vec3 lookTarget{0.0f};
    std::filesystem::path outputPath{};
    int settleFrames{20};
    bool waitForLodReady{true};
    double lodReadyTimeoutSeconds{120.0};
    bool writeLodDebugSnapshot{true};
};

struct ScreenshotReproState
{
    bool initialized{false};
    bool captureRequested{false};
    int waitFramesRemaining{0};
    double lodReadyWaitSeconds{0.0};
    bool lodTimeoutReported{false};
};

struct CapturePlacementAction
{
    glm::ivec3 targetBlockPos{0};
    glm::ivec3 faceNormal{0, 1, 0};
    BlockId block{BlockId::DebugLamp};
    bool applied{false};
};

struct CaptureOverridesConfig
{
    bool hasTimeOfDay{false};
    float timeOfDay{12.0f};
    bool hasExactChunks{false};
    int exactChunks{kDefaultNearRenderDistance};
    bool hasTotalChunks{false};
    int totalChunks{kDefaultTotalRenderDistanceChunks};
    bool hasFogStartBlocks{false};
    int fogStartBlocks{kDefaultFarFogStartBlocks};
    bool hasDebugView{false};
    TerrainDebugView terrainDebugView{TerrainDebugView::None};
    bool hasDirectSunEnabled{false};
    bool directSunEnabled{true};
    std::vector<CapturePlacementAction> placements;
};

[[nodiscard]] BlockId parseCaptureBlockId(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (text == "air")
    {
        return BlockId::Air;
    }
    if (text == "debug_lamp" || text == "lamp")
    {
        return BlockId::DebugLamp;
    }
    if (text == "grass")
    {
        return BlockId::Grass;
    }
    if (text == "stone")
    {
        return BlockId::Stone;
    }
    if (text == "sand")
    {
        return BlockId::Sand;
    }
    if (text == "water")
    {
        return BlockId::Water;
    }
    if (text == "wood" || text == "log")
    {
        return BlockId::Wood;
    }
    if (text == "leaves" || text == "leaf")
    {
        return BlockId::Leaves;
    }
    if (text == "spruce_log" || text == "sprucelog")
    {
        return BlockId::SpruceLog;
    }
    if (text == "spruce_leaves" || text == "spruceleaves")
    {
        return BlockId::SpruceLeaves;
    }
    if (text == "dark_oak_log" || text == "darkoaklog")
    {
        return BlockId::DarkOakLog;
    }
    if (text == "dark_oak_leaves" || text == "darkoakleaves")
    {
        return BlockId::DarkOakLeaves;
    }
    if (text == "birch_log" || text == "birchlog")
    {
        return BlockId::BirchLog;
    }
    if (text == "birch_leaves" || text == "birchleaves")
    {
        return BlockId::BirchLeaves;
    }
    if (text == "acacia_log" || text == "acacialog")
    {
        return BlockId::AcaciaLog;
    }
    if (text == "acacia_leaves" || text == "acacialeaves")
    {
        return BlockId::AcaciaLeaves;
    }
    if (text == "podzol")
    {
        return BlockId::Podzol;
    }

    return BlockId::DebugLamp;
}

[[nodiscard]] TerrainDebugView parseTerrainDebugView(int value) noexcept
{
    switch (value)
    {
    case 1:
        return TerrainDebugView::SkyLight;
    case 2:
        return TerrainDebugView::BlockLight;
    case 3:
        return TerrainDebugView::MipLevel;
    case 4:
        return TerrainDebugView::AmbientOcclusion;
    default:
        return TerrainDebugView::None;
    }
}

[[nodiscard]] const char* terrainDebugViewLabel(TerrainDebugView view) noexcept
{
    switch (view)
    {
    case TerrainDebugView::SkyLight:
        return "Sky Light";
    case TerrainDebugView::BlockLight:
        return "Block Light";
    case TerrainDebugView::MipLevel:
        return "Mip Level";
    case TerrainDebugView::AmbientOcclusion:
        return "AO";
    case TerrainDebugView::None:
    default:
        return "None";
    }
}

[[nodiscard]] const char* blockIdLabel(BlockId block) noexcept
{
    switch (block)
    {
    case BlockId::Air:
        return "Air";
    case BlockId::Grass:
        return "Grass";
    case BlockId::Wood:
        return "Wood";
    case BlockId::Leaves:
        return "Leaves";
    case BlockId::Sand:
        return "Sand";
    case BlockId::Water:
        return "Water";
    case BlockId::Stone:
        return "Stone";
    case BlockId::SpruceLog:
        return "SpruceLog";
    case BlockId::SpruceLeaves:
        return "SpruceLeaves";
    case BlockId::DarkOakLog:
        return "DarkOakLog";
    case BlockId::DarkOakLeaves:
        return "DarkOakLeaves";
    case BlockId::BirchLog:
        return "BirchLog";
    case BlockId::BirchLeaves:
        return "BirchLeaves";
    case BlockId::AcaciaLog:
        return "AcaciaLog";
    case BlockId::AcaciaLeaves:
        return "AcaciaLeaves";
    case BlockId::Podzol:
        return "Podzol";
    case BlockId::DebugLamp:
        return "DebugLamp";
    case BlockId::Count:
    default:
        return "Unknown";
    }
}

[[nodiscard]] constexpr std::array<BlockId, 16> placeableBlockOptions() noexcept
{
    return {
        BlockId::Grass,
        BlockId::Wood,
        BlockId::Leaves,
        BlockId::Sand,
        BlockId::Water,
        BlockId::Stone,
        BlockId::SpruceLog,
        BlockId::SpruceLeaves,
        BlockId::DarkOakLog,
        BlockId::DarkOakLeaves,
        BlockId::BirchLog,
        BlockId::BirchLeaves,
        BlockId::AcaciaLog,
        BlockId::AcaciaLeaves,
        BlockId::Podzol,
        BlockId::DebugLamp
    };
}

[[nodiscard]] std::vector<CapturePlacementAction> loadCapturePlacements()
{
    std::vector<CapturePlacementAction> actions;
    const char* value = std::getenv("BLOCKGAME_CAPTURE_PLACEMENTS");
    if (value == nullptr || value[0] == '\0')
    {
        return actions;
    }

    std::stringstream stream(value);
    std::string entry;
    while (std::getline(stream, entry, ';'))
    {
        if (entry.empty())
        {
            continue;
        }

        const std::size_t firstPipe = entry.find('|');
        const std::size_t secondPipe = entry.find('|', firstPipe == std::string::npos ? std::string::npos : firstPipe + 1);
        if (firstPipe == std::string::npos || secondPipe == std::string::npos)
        {
            continue;
        }

        CapturePlacementAction action;
        {
            std::string positionPart = entry.substr(0, firstPipe);
            std::replace(positionPart.begin(), positionPart.end(), ',', ' ');
            std::istringstream positionStream(positionPart);
            if (!(positionStream >> action.targetBlockPos.x >> action.targetBlockPos.y >> action.targetBlockPos.z))
            {
                continue;
            }
        }
        {
            std::string normalPart = entry.substr(firstPipe + 1, secondPipe - firstPipe - 1);
            std::replace(normalPart.begin(), normalPart.end(), ',', ' ');
            std::istringstream normalStream(normalPart);
            if (!(normalStream >> action.faceNormal.x >> action.faceNormal.y >> action.faceNormal.z))
            {
                continue;
            }
        }

        action.block = parseCaptureBlockId(entry.substr(secondPipe + 1));
        actions.push_back(action);
    }

    return actions;
}

[[nodiscard]] CaptureOverridesConfig loadCaptureOverridesConfig()
{
    CaptureOverridesConfig config;
    config.placements = loadCapturePlacements();

    config.hasTimeOfDay = tryGetEnvFloat("BLOCKGAME_CAPTURE_TIME_OF_DAY", config.timeOfDay);
    config.hasExactChunks = tryGetEnvInt("BLOCKGAME_CAPTURE_EXACT_CHUNKS", config.exactChunks);
    config.hasExactChunks = tryGetEnvInt("BLOCKGAME_CAPTURE_NEAR_CHUNKS", config.exactChunks) || config.hasExactChunks;
    config.totalChunks = config.exactChunks;
    config.hasTotalChunks = tryGetEnvInt("BLOCKGAME_CAPTURE_TOTAL_CHUNKS", config.totalChunks);
    int legacyFarBlocks = 0;
    if (tryGetEnvInt("BLOCKGAME_CAPTURE_FAR_BLOCKS", legacyFarBlocks))
    {
        config.totalChunks = blocksToChunkRadiusCeil(legacyFarBlocks);
        config.hasTotalChunks = true;
    }
    config.hasFogStartBlocks = tryGetEnvInt("BLOCKGAME_CAPTURE_FOG_START_BLOCKS", config.fogStartBlocks);
    if (const char* debugViewValue = std::getenv("BLOCKGAME_CAPTURE_DEBUG_VIEW"))
    {
        char* end = nullptr;
        const long parsed = std::strtol(debugViewValue, &end, 10);
        if (end != debugViewValue)
        {
            config.hasDebugView = true;
            config.terrainDebugView = parseTerrainDebugView(static_cast<int>(parsed));
        }
    }
    if (const char* directSunValue = std::getenv("BLOCKGAME_CAPTURE_DIRECT_SUN"))
    {
        config.hasDirectSunEnabled = true;
        config.directSunEnabled = envFlagEnabled("BLOCKGAME_CAPTURE_DIRECT_SUN");
        (void)directSunValue;
    }
    return config;
}

[[nodiscard]] ScreenshotSweepConfig loadScreenshotSweepConfig()
{
    ScreenshotSweepConfig config;
    config.enabled = envFlagEnabled("BLOCKGAME_SCREENSHOT_SWEEP");
    if (!config.enabled)
    {
        return config;
    }

    if (const char* pathValue = std::getenv("BLOCKGAME_SCREENSHOT_SWEEP_DIR"))
    {
        config.outputDir = pathValue;
    }

    if (config.outputDir.empty())
    {
        std::error_code ec;
        std::filesystem::path cwd = std::filesystem::current_path(ec);
        if (ec)
        {
            cwd = ".";
        }
        config.outputDir = cwd / "artifacts" / "horizon_sweep";
    }

    config.heightOffsetBlocks = envFloatOrDefault("BLOCKGAME_SCREENSHOT_SWEEP_HEIGHT_OFFSET",
                                                  config.heightOffsetBlocks);

    return config;
}

[[nodiscard]] ScreenshotReproConfig loadScreenshotReproConfig()
{
    ScreenshotReproConfig config;
    config.enabled = envFlagEnabled("BLOCKGAME_REPRO_CAPTURE");
    if (!config.enabled)
    {
        return config;
    }

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    if (!tryGetEnvFloat("BLOCKGAME_REPRO_X", x) ||
        !tryGetEnvFloat("BLOCKGAME_REPRO_Y", y) ||
        !tryGetEnvFloat("BLOCKGAME_REPRO_Z", z))
    {
        config.enabled = false;
        return config;
    }
    config.position = glm::vec3(x, y, z);

    float lookX = 0.0f;
    float lookY = 0.0f;
    float lookZ = 0.0f;
    const bool hasLookTarget =
        tryGetEnvFloat("BLOCKGAME_REPRO_LOOK_X", lookX) &&
        tryGetEnvFloat("BLOCKGAME_REPRO_LOOK_Y", lookY) &&
        tryGetEnvFloat("BLOCKGAME_REPRO_LOOK_Z", lookZ);
    if (hasLookTarget)
    {
        config.useLookTarget = true;
        config.lookTarget = glm::vec3(lookX, lookY, lookZ);
    }
    else
    {
        if (!tryGetEnvFloat("BLOCKGAME_REPRO_YAW", config.yawDegrees) ||
            !tryGetEnvFloat("BLOCKGAME_REPRO_PITCH", config.pitchDegrees))
        {
            config.enabled = false;
            return config;
        }
    }

    if (const char* outputValue = std::getenv("BLOCKGAME_REPRO_OUTPUT"))
    {
        config.outputPath = outputValue;
    }

    if (config.outputPath.empty())
    {
        std::error_code ec;
        std::filesystem::path cwd = std::filesystem::current_path(ec);
        if (ec)
        {
            cwd = ".";
        }
        config.outputPath = cwd / "artifacts" / "repro_capture" / "repro.bmp";
    }

    config.settleFrames = std::max(0, static_cast<int>(std::lround(
        envFloatOrDefault("BLOCKGAME_REPRO_SETTLE_FRAMES", static_cast<float>(config.settleFrames)))));
    if (const char* waitValue = std::getenv("BLOCKGAME_REPRO_WAIT_FOR_LOD_READY"))
    {
        config.waitForLodReady = envFlagEnabled("BLOCKGAME_REPRO_WAIT_FOR_LOD_READY");
        (void)waitValue;
    }
    config.lodReadyTimeoutSeconds =
        std::max(0.0, static_cast<double>(envFloatOrDefault("BLOCKGAME_REPRO_LOD_READY_TIMEOUT_SECONDS",
                                                            static_cast<float>(config.lodReadyTimeoutSeconds))));
    if (const char* writeDebugValue = std::getenv("BLOCKGAME_REPRO_WRITE_LOD_DEBUG"))
    {
        config.writeLodDebugSnapshot = envFlagEnabled("BLOCKGAME_REPRO_WRITE_LOD_DEBUG");
        (void)writeDebugValue;
    }

    return config;
}

[[nodiscard]] std::size_t screenshotSweepPoseCount(const ScreenshotSweepConfig& config) noexcept
{
    return config.pitches.size() * config.yaws.size();
}

[[nodiscard]] float screenshotSweepPitchForIndex(const ScreenshotSweepConfig& config, std::size_t poseIndex)
{
    return config.pitches[poseIndex / config.yaws.size()];
}

[[nodiscard]] float screenshotSweepYawForIndex(const ScreenshotSweepConfig& config, std::size_t poseIndex)
{
    return config.yaws[poseIndex % config.yaws.size()];
}

void applyCameraPose(Camera& camera,
                     const glm::vec3& position,
                     float yawDegrees,
                     float pitchDegrees)
{
    camera.position = position;
    camera.velocity = glm::vec3(0.0f);
    camera.onGround = true;
    camera.flyMode = false;
    camera.yaw = yawDegrees;
    camera.pitch = std::clamp(pitchDegrees, -89.0f, 89.0f);
    camera.updateVectors();
}

[[nodiscard]] glm::vec2 yawPitchFromLookTarget(const glm::vec3& position, const glm::vec3& lookTarget)
{
    const glm::vec3 direction = glm::normalize(lookTarget - position);
    const float yawDegrees = glm::degrees(std::atan2(direction.z, direction.x));
    const float pitchDegrees = glm::degrees(std::asin(std::clamp(direction.y, -1.0f, 1.0f)));
    return glm::vec2(yawDegrees, pitchDegrees);
}

[[nodiscard]] std::string screenshotSweepPoseLabel(std::size_t poseIndex, float yawDegrees, float pitchDegrees)
{
    const auto formatSigned = [](float value) -> std::string
    {
        std::ostringstream part;
        part << (value >= 0.0f ? 'p' : 'm')
             << std::setfill('0') << std::setw(3)
             << static_cast<int>(std::abs(std::round(value)));
        return part.str();
    };

    std::ostringstream name;
    name << "pose_" << std::setfill('0') << std::setw(3) << poseIndex
         << "_yaw_" << formatSigned(yawDegrees)
         << "_pitch_" << formatSigned(pitchDegrees)
         << ".bmp";
    return name.str();
}

void initializeCrashLogging(const std::filesystem::path& logPath)
{
    gCrashLogPath = logPath;
    gCrashDumpPath = logPath.parent_path() / "blockgame_crash.dmp";
    gHangDumpPath = logPath.parent_path() / "blockgame_hang.dmp";

    // Ensure the log file exists so later appends succeed even if the program dies immediately.
    {
        std::ofstream out(gCrashLogPath, std::ios::app);
    }

    initializeSymbolHandler(logPath.parent_path());
    noteDiagnosticPhase("startup/crash_logging");
    startHangWatchdog();

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

        appendDiagnosticSnapshot("terminate");
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
        appendDiagnosticSnapshot("seh");
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

void shutdownCrashLogging() noexcept
{
    stopHangWatchdog();
    shutdownSymbolHandler();
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
    int exactDistance = 0;
    int totalDistance = chunkManager.totalRenderDistanceChunks();
    if (!(stream >> exactDistance))
    {
        return false;
    }

    if (stream >> totalDistance)
    {
        chunkManager.setTotalRenderDistanceChunks(totalDistance);
    }
    chunkManager.setExactRenderDistanceChunks(exactDistance);
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
    camera.flyMode = false;
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
    constexpr float kSprintMultiplier = 1.3f;
    if (inputState.toggleFlightPressed)
    {
        camera.flyMode = !camera.flyMode;
        camera.velocity = glm::vec3(0.0f);
        camera.onGround = false;
    }

    const float currentMoveSpeed = camera.moveSpeed * (inputState.sprintHeld ? kSprintMultiplier : 1.0f);
    const glm::vec2 horizontalInput(inputState.moveDirection.x, inputState.moveDirection.z);
    if (glm::dot(horizontalInput, horizontalInput) > kEpsilon * kEpsilon)
    {
        glm::vec3 normalized = glm::normalize(glm::vec3(horizontalInput.x, 0.0f, horizontalInput.y));
        camera.velocity.x = normalized.x * currentMoveSpeed;
        camera.velocity.z = normalized.z * currentMoveSpeed;
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

    if (camera.flyMode)
    {
        float verticalDirection = 0.0f;
        if (inputState.ascendHeld)
        {
            verticalDirection += 1.0f;
        }
        if (inputState.descendHeld)
        {
            verticalDirection -= 1.0f;
        }
        camera.velocity.y = verticalDirection * currentMoveSpeed;
    }
    else
    {
        camera.velocity.y += kGravity * dt;
        if (camera.velocity.y < kTerminalVelocity)
        {
            camera.velocity.y = kTerminalVelocity;
        }
    }

    if (!camera.flyMode && inputState.jumpHeld && camera.onGround)
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
        if (!camera.flyMode && desiredMove.y < 0.0f && moveY.actualMove > desiredMove.y)
        {
            groundedThisStep = true;
        }
    }

    camera.onGround = !camera.flyMode && groundedThisStep;
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
    int exitCode = EXIT_SUCCESS;
    noteDiagnosticPhase("startup/window_ready");
    {
    Renderer renderer;
    try
    {
        noteDiagnosticPhase("startup/renderer_initialize");
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
    const bool benchmarkRequested = envFlagEnabled("BLOCKGAME_BENCHMARK");
    const BenchmarkConfig benchmarkConfig = loadBenchmarkConfig();
    if (benchmarkRequested && !benchmarkConfig.enabled)
    {
        std::cerr << "Invalid benchmark configuration. Set BLOCKGAME_BENCHMARK_SCENARIO to one of "
                  << "spawn_preload, full_exact_preload, player_idle_exact_fill, post_release_exact_fill, stationary_exact_fill, post_release_exact_sweep_fill, straight_line_sprint, "
                  << "turn_heavy_traversal, or vertical_travel."
                  << std::endl;
        renderer.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }
    const ScreenshotReproConfig screenshotReproConfig = loadScreenshotReproConfig();
    const ScreenshotSweepConfig screenshotSweepConfig = loadScreenshotSweepConfig();
    CaptureOverridesConfig captureOverrides = loadCaptureOverridesConfig();
    ScreenshotReproState screenshotReproState{};
    ScreenshotSweepState screenshotSweepState{};
    BenchmarkRuntimeState benchmarkState{};
    if (benchmarkConfig.enabled && (screenshotReproConfig.enabled || screenshotSweepConfig.enabled))
    {
        std::cerr << "Benchmark mode cannot run together with screenshot automation." << std::endl;
        renderer.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }
    if (screenshotReproConfig.enabled)
    {
        std::error_code ec;
        std::filesystem::create_directories(screenshotReproConfig.outputPath.parent_path(), ec);
        if (ec)
        {
            std::cerr << "Failed to create repro capture directory: "
                      << screenshotReproConfig.outputPath.parent_path() << std::endl;
            renderer.shutdown();
            glfwDestroyWindow(window);
            glfwTerminate();
            return EXIT_FAILURE;
        }

        std::cout << "Single repro capture enabled. Output: "
                  << screenshotReproConfig.outputPath << std::endl;
    }
    if (screenshotSweepConfig.enabled && !screenshotReproConfig.enabled)
    {
        std::error_code ec;
        std::filesystem::create_directories(screenshotSweepConfig.outputDir, ec);
        if (ec)
        {
            std::cerr << "Failed to create screenshot sweep directory: "
                      << screenshotSweepConfig.outputDir << std::endl;
            renderer.shutdown();
            glfwDestroyWindow(window);
            glfwTerminate();
            return EXIT_FAILURE;
        }

        screenshotSweepState.manifest.open(screenshotSweepConfig.outputDir / "captures.csv",
                                           std::ios::trunc);
        if (!screenshotSweepState.manifest)
        {
            std::cerr << "Failed to open screenshot sweep manifest in "
                      << screenshotSweepConfig.outputDir << std::endl;
            renderer.shutdown();
            glfwDestroyWindow(window);
            glfwTerminate();
            return EXIT_FAILURE;
        }
        screenshotSweepState.manifest << "file,yaw,pitch,pos_x,pos_y,pos_z\n";
        std::cout << "Screenshot sweep enabled. Output: "
                  << screenshotSweepConfig.outputDir << std::endl;
    }

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

    {
    ChunkManager chunkManager(1337u);
    noteDiagnosticPhase("startup/chunk_manager");
    chunkManager.initializeRendering(renderer.device());
    chunkManager.setBlockTextureAtlasConfig(BlockTextureAtlasConfig{
        blockAtlas.size,
        blockAtlas.tileSizePixels > 0 ? blockAtlas.tileSizePixels : kAtlasTileSizePixels,
        blockAtlas.tileStridePixels > 0 ? blockAtlas.tileStridePixels : kAtlasTileSizePixels,
        blockAtlas.tilePaddingPixels});
    if (captureOverrides.hasExactChunks)
    {
        chunkManager.setExactRenderDistanceChunks(captureOverrides.exactChunks);
    }
    if (captureOverrides.hasTotalChunks)
    {
        chunkManager.setTotalRenderDistanceChunks(captureOverrides.totalChunks);
    }
    if (captureOverrides.hasFogStartBlocks)
    {
        chunkManager.setFogStartBlocks(captureOverrides.fogStartBlocks);
    }
    if (captureOverrides.hasTimeOfDay)
    {
        environment.timeOfDay = captureOverrides.timeOfDay;
    }
    if (captureOverrides.hasDebugView)
    {
        environment.debug.terrainDebugView = captureOverrides.terrainDebugView;
    }
    if (captureOverrides.hasDirectSunEnabled)
    {
        environment.debug.directSunEnabled = captureOverrides.directSunEnabled;
    }
    if (benchmarkConfig.enabled)
    {
        chunkManager.setExactRenderDistanceChunks(benchmarkConfig.exactChunks);
        chunkManager.setTotalRenderDistanceChunks(benchmarkConfig.totalChunks);
        chunkManager.setFogStartBlocks(benchmarkConfig.fogStartBlocks);

        if (!benchmarkScenarioUsesInteractiveRuntime(benchmarkConfig.scenario))
        {
            environment.atmosphereEnabled = false;
            environment.debug.worldPassEnabled = true;
            environment.debug.skyPassEnabled = false;
            environment.debug.aerialPerspectiveEnabled = false;
            environment.debug.fogFallbackEnabled = true;
            environment.debug.shadowsEnabled = true;
            environment.debug.directSunEnabled = true;
            environment.debug.terrainDebugView = TerrainDebugView::None;
        }
        chunkManager.setBenchmarkMetricsEnabled(true);
        std::cout << "Chunk benchmark enabled for scenario '"
                  << benchmarkScenarioName(benchmarkConfig.scenario)
                  << "'. Output: " << benchmarkConfig.outputPath
                  << " Progress log: " << benchmarkConfig.progressLogPath << std::endl;
    }

    MobSystem mobSystem;
    std::unordered_map<std::string, LoadedTexture> mobTextureCache;
    std::unordered_set<std::string> missingMobTextures;
    noteDiagnosticPhase("startup/mobs");
    if (mobSystem.loadDefinitions(std::filesystem::path("assets") / "mobs"))
    {
        for (const MobModel* model : mobSystem.allModels())
        {
            if (model == nullptr || model->texturePath.empty())
            {
                continue;
            }

            const std::string textureKey = model->texturePath.generic_string();
            if (mobTextureCache.find(textureKey) != mobTextureCache.end() ||
                missingMobTextures.find(textureKey) != missingMobTextures.end())
            {
                continue;
            }

            std::error_code textureEc;
            if (!std::filesystem::exists(model->texturePath, textureEc) || textureEc)
            {
                missingMobTextures.insert(textureKey);
                continue;
            }

            try
            {
                mobTextureCache.emplace(textureKey, renderer.loadTexture(textureKey.c_str()));
            }
            catch (const std::exception& ex)
            {
                std::cerr << "Failed to load mob texture '" << textureKey << "': " << ex.what() << std::endl;
                missingMobTextures.insert(textureKey);
            }
        }

        std::cout << "Loaded " << mobSystem.definitionCount() << " mob model definition(s)." << std::endl;
    }
    else
    {
        std::cout << "No mob model definitions were loaded from assets/mobs." << std::endl;
    }

    const auto resolveMobTextureBinding = [&](const MobModel& model) -> MobTextureBinding
    {
        if (!model.texturePath.empty())
        {
            const std::string textureKey = model.texturePath.generic_string();
            const auto textureIt = mobTextureCache.find(textureKey);
            if (textureIt != mobTextureCache.end())
            {
                return MobTextureBinding{textureIt->second.srvGpu, true};
            }
        }
        return {};
    };

    const auto spawnPassiveMobNearPlayer = [&](const char* modelId, const char* displayName)
    {
        if (mobSystem.findModel(modelId) == nullptr)
        {
            std::cerr << "Cannot spawn " << displayName << ": assets/mobs/" << modelId
                      << ".geo.json was not loaded." << std::endl;
            return false;
        }

        glm::vec3 forward = camera.front();
        forward.y = 0.0f;
        if (glm::dot(forward, forward) <= 1e-4f)
        {
            forward = glm::vec3(0.0f, 0.0f, -1.0f);
        }
        else
        {
            forward = glm::normalize(forward);
        }

        const glm::vec3 spawnPos = glm::vec3(camera.position.x + forward.x * 3.0f,
                                             chunkManager.surfaceHeight(camera.position.x + forward.x * 3.0f,
                                                                        camera.position.z + forward.z * 3.0f),
                                             camera.position.z + forward.z * 3.0f);
        const float yawRadians = std::atan2(-forward.x, -forward.z);
        const bool spawned = mobSystem.spawn(modelId, spawnPos, yawRadians);
        if (spawned)
        {
            std::cout << "Spawned " << displayName << " at: ("
                      << spawnPos.x << ", " << spawnPos.y << ", " << spawnPos.z << ")" << std::endl;
        }
        return spawned;
    };

    const auto spawnPigNearPlayer = [&]()
    {
        return spawnPassiveMobNearPlayer("pig", "pig");
    };

    const auto spawnCowNearPlayer = [&]()
    {
        return spawnPassiveMobNearPlayer("cow", "cow");
    };
    
    // Find a guaranteed safe spawn position above ground
    std::cout << "Finding safe spawn position..." << std::endl;
    camera.position = chunkManager.findSafeSpawnPosition(camera.position.x, camera.position.z);
    camera.velocity = glm::vec3(0.0f);
    camera.onGround = false;
    camera.flyMode = false;

    std::cout << "Player spawned at: (" << camera.position.x << ", " << camera.position.y << ", " << camera.position.z << ")" << std::endl;
    benchmarkState.spawnPosition = camera.position;
    benchmarkState.finalCameraPosition = camera.position;
    benchmarkState.finalYawDegrees = camera.yaw;
    benchmarkState.finalPitchDegrees = camera.pitch;
    if (benchmarkConfig.enabled && benchmarkScenarioStartsAtSpawn(benchmarkConfig.scenario))
    {
        chunkManager.resetBenchmarkMetrics();
        if (!resetBenchmarkProgressLog(benchmarkConfig))
        {
            std::cerr << "Failed to initialize benchmark progress log at "
                      << benchmarkConfig.progressLogPath << std::endl;
            renderer.shutdown();
            glfwDestroyWindow(window);
            glfwTerminate();
            return EXIT_FAILURE;
        }
        benchmarkState.started = true;
        benchmarkState.elapsedSeconds = 0.0;
        benchmarkState.completionHoldSeconds = 0.0;
        benchmarkState.exactRequiredStableSeconds = 0.0;
        benchmarkState.nextProgressLogSeconds = 0.0;
        benchmarkState.playerReleaseSeconds = -1.0;
        benchmarkState.steadyStateSeconds = -1.0;
        benchmarkState.fullExactReadySeconds = -1.0;
        benchmarkState.lastExactRequiredChunks = -1;
        benchmarkState.playerReleaseExactReadyChunks = -1;
        benchmarkState.playerReleaseExactRequiredChunks = -1;
        benchmarkState.scenarioStartPosition = benchmarkState.spawnPosition;
        benchmarkState.scenarioStartYawDegrees = camera.yaw;
        benchmarkState.scenarioStartPitchDegrees = camera.pitch;
        benchmarkState.frameTimesMs.clear();
        benchmarkState.currentSpikeStreakOver33_3Ms = 0;
        benchmarkState.spikeSummary = BenchmarkSpikeSummary{};
    }
    chunkManager.setStartupExactPreloadChunks(
        benchmarkConfig.enabled && benchmarkScenarioUsesFullStartupExactPreload(benchmarkConfig.scenario)
            ? benchmarkConfig.exactChunks
            : kDefaultStartupExactPreloadChunks);
    chunkManager.beginSpawnPreload(camera.position);

    if (std::getenv("BLOCKGAME_STREAMING_TEST"))
    {
        runStreamingValidationScenarios(chunkManager, camera.position);
        chunkManager.setRenderSynchronization(renderer.frameFence(), renderer.lastSubmittedFrameFenceValue());
        chunkManager.update(camera.position, camera.front());
        renderer.setUploadSynchronization(chunkManager.uploadFence(),
                                          chunkManager.lastSubmittedUploadFenceValue(),
                                          chunkManager.farUploadFence(),
                                          chunkManager.lastSubmittedFarUploadFenceValue(),
                                          chunkManager.exactGpuFence(),
                                          chunkManager.lastSubmittedExactGpuFenceValue());
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
    std::cout << "Controls: WASD to move, mouse to look, hold CTRL to sprint, SPACE to jump, double-tap SPACE to toggle flight, SHIFT to descend while flying, . to toggle mouse/UI control, N to set exact/total render distance, F2 to teleport, E to choose block type, left-click to destroy blocks, right-click to place blocks, ESC to quit." << std::endl;

    while (!glfwWindowShouldClose(window))
    {
        noteDiagnosticPhase("frame/start", true);
        const auto frameCpuStart = std::chrono::steady_clock::now();
        double pollEventsMs = 0.0;
        double buildRenderDataMs = 0.0;
        double renderWorldCpuMs = 0.0;
        bool benchmarkRequestClose = false;
        const double currentTime = glfwGetTime();
        const double rawFrameTime = currentTime - previousTime;
        double frameTime = rawFrameTime;
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

        noteDiagnosticPhase("frame/poll_events");
        if (!benchmarkConfig.enabled && profilingOverlayTimer >= 1.0)
        {
            ChunkProfilingSnapshot snapshot = chunkManager.sampleProfilingSnapshot();
            const RendererProfilingSnapshot rendererSnapshot = renderer.profilingSnapshot();
            snapshot.lodGpuCullMs = rendererSnapshot.lodGpuCullMs;
            snapshot.lodIndirectBuildMs = rendererSnapshot.lodIndirectBuildMs;
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
                            << " Exact " << chunkManager.exactRenderDistanceChunks()
                            << "x" << snapshot.verticalRadius
                            << " Total " << chunkManager.totalRenderDistanceChunks() << "c"
                            << " (" << verticalSpan << "h)";
            if (snapshot.exactChunksPending > 0 || snapshot.exactChunksReady > 0)
            {
                profilingStream << " | Exact " << snapshot.exactChunksReady
                                << " ready " << snapshot.exactChunksPending << " pending";
            }
            if (snapshot.farActiveTiles > 0 || snapshot.farDirtyTiles > 0)
            {
                profilingStream << " | LOD " << snapshot.farShellTilesReady
                                << "/" << snapshot.farActiveTiles << " ready";
                if (snapshot.farDirtyTiles > 0)
                {
                    profilingStream << " dirty " << snapshot.farDirtyTiles;
                }
                if (snapshot.farTilesBuilt > 0)
                {
                    profilingStream << " built " << snapshot.farTilesBuilt;
                }
                if (snapshot.farTilesQueued > 0)
                {
                    profilingStream << " q " << snapshot.farTilesQueued;
                }
                if (snapshot.farTilesPendingUpload > 0)
                {
                    profilingStream << " up " << snapshot.farTilesPendingUpload;
                }
            }
            if (snapshot.pooledChunkBudgetBytes > 0)
            {
                const double pooledMiB = static_cast<double>(snapshot.pooledChunkBytes) / (1024.0 * 1024.0);
                const double poolBudgetMiB =
                    static_cast<double>(snapshot.pooledChunkBudgetBytes) / (1024.0 * 1024.0);
                profilingStream << " | Pool " << snapshot.pooledChunkCount
                                << " (" << pooledMiB << "/" << poolBudgetMiB << " MiB)";
            }
            profilingStream << " | UploadMs " << snapshot.uploadMsLastFrame;
            profilingStream << " | UpdateMs " << snapshot.updateMsLastFrame;
            if (snapshot.farCollectMsLastFrame > 0.0 || snapshot.farUploadMsLastFrame > 0.0)
            {
                profilingStream << " | LODCollect " << snapshot.farCollectMsLastFrame
                                << " LODUpload " << snapshot.farUploadMsLastFrame;
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

        const auto pollEventsStart = std::chrono::steady_clock::now();
        glfwPollEvents();
        pollEventsMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pollEventsStart).count();
        noteDiagnosticPhase("frame/input");

        bool f1CurrentlyPressed = (glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS);
        bool f1JustPressed = f1CurrentlyPressed && !inputContext.f1Pressed;
        if (f1JustPressed)
        {
            inputContext.showDebugOverlay = !inputContext.showDebugOverlay;
        }
        inputContext.f1JustPressed = f1JustPressed;
        inputContext.f1Pressed = f1CurrentlyPressed;

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
                    phaseName = "Streaming";
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
                loadingStream << "Pending exact GPU builds: " << streamingStatus.exactPendingUploads << '\n';
                if (chunkManager.totalRenderDistanceChunks() > chunkManager.exactRenderDistanceChunks())
                {
                    loadingStream << "LOD tiles: " << streamingStatus.farReadyTiles
                                  << " / " << streamingStatus.farActiveTiles
                                  << " ready, " << streamingStatus.farQueuedTiles << " queued";
                    if (streamingStatus.farPendingUploadTiles > 0)
                    {
                        loadingStream << ", " << streamingStatus.farPendingUploadTiles << " pending upload";
                    }
                    loadingStream << '\n';
                }
                loadingStream << "Total radius target: " << chunkManager.totalRenderDistanceChunks()
                              << " chunks";
                if (chunkManager.totalRenderDistanceChunks() > chunkManager.exactRenderDistanceChunks())
                {
                    loadingStream << " (LOD active beyond exact)";
                }
                loadingStream << '\n';
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

        if (benchmarkConfig.enabled)
        {
            inputContext.showDebugOverlay = false;
            inputContext.showControlsOverlay = false;
            inputContext.showRenderDistanceGUI = false;
            inputContext.showTeleportGUI = false;
            inputContext.cameraMouseCaptured = true;

            const bool benchmarkStartAllowed =
                benchmarkConfig.scenario == BenchmarkScenarioKind::StationaryExactFill ||
                playerReleased;
            if (!benchmarkScenarioStartsAtSpawn(benchmarkConfig.scenario) &&
                benchmarkStartAllowed &&
                !benchmarkState.started)
            {
                chunkManager.resetBenchmarkMetrics();
                if (!resetBenchmarkProgressLog(benchmarkConfig))
                {
                    std::cerr << "Failed to initialize benchmark progress log at "
                              << benchmarkConfig.progressLogPath << std::endl;
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                    exitCode = EXIT_FAILURE;
                    benchmarkState.completed = true;
                    benchmarkRequestClose = true;
                }
                else
                {
                    benchmarkState.started = true;
                    benchmarkState.timedOut = false;
                    benchmarkState.elapsedSeconds = 0.0;
                    benchmarkState.completionHoldSeconds = 0.0;
                    benchmarkState.exactRequiredStableSeconds = 0.0;
                    benchmarkState.nextProgressLogSeconds = 0.0;
                    benchmarkState.playerReleaseSeconds = -1.0;
                    benchmarkState.steadyStateSeconds = -1.0;
                    benchmarkState.fullExactReadySeconds = -1.0;
                    benchmarkState.lastExactRequiredChunks = -1;
                    benchmarkState.playerReleaseExactReadyChunks = -1;
                    benchmarkState.playerReleaseExactRequiredChunks = -1;
                    benchmarkState.frameTimesMs.clear();
                    benchmarkState.currentSpikeStreakOver33_3Ms = 0;
                    benchmarkState.spikeSummary = BenchmarkSpikeSummary{};
                    benchmarkState.exactGpuSynthMs.clear();
                    benchmarkState.exactGpuStampMs.clear();
                    benchmarkState.exactGpuLightMs.clear();
                    benchmarkState.exactGpuFaceCountMs.clear();
                    benchmarkState.exactGpuFacePrefixMs.clear();
                    benchmarkState.exactGpuFaceEmitMs.clear();
                    benchmarkState.exactGpuTotalMs.clear();
                    benchmarkState.gpuLocalUsageMiB.clear();
                    benchmarkState.exactGpuTotalMiB.clear();
                    initializeBenchmarkCamera(camera, benchmarkConfig, benchmarkState);
                }
            }

            if (benchmarkState.started && !benchmarkState.completed)
            {
                benchmarkState.elapsedSeconds += frameTime;
                if (benchmarkState.playerReleaseSeconds < 0.0 && playerReleased)
                {
                    benchmarkState.playerReleaseSeconds = benchmarkState.elapsedSeconds;
                    benchmarkState.playerReleaseExactReadyChunks = streamingStatus.exactReadyChunks;
                    benchmarkState.playerReleaseExactRequiredChunks = streamingStatus.exactRequiredChunks;
                }
                if (benchmarkState.steadyStateSeconds < 0.0 &&
                    streamingStatus.phase == StreamingPhase::SteadyState)
                {
                    benchmarkState.steadyStateSeconds = benchmarkState.elapsedSeconds;
                }
                const auto updateExactFillCompletion = [&]()
                {
                    const ChunkProfilingSnapshot benchmarkProfiling = chunkManager.sampleProfilingSnapshot();
                    if (streamingStatus.exactRequiredChunks == benchmarkState.lastExactRequiredChunks)
                    {
                        benchmarkState.exactRequiredStableSeconds += frameTime;
                    }
                    else
                    {
                        benchmarkState.lastExactRequiredChunks = streamingStatus.exactRequiredChunks;
                        benchmarkState.exactRequiredStableSeconds = 0.0;
                    }
                    const bool completedRequiredSweep =
                        benchmarkConfig.scenario != BenchmarkScenarioKind::PostReleaseExactSweepFill ||
                        benchmarkState.elapsedSeconds >=
                            (360.0 / std::max(1.0, benchmarkConfig.sweepDegreesPerSecond));
                    const bool discoverySettled =
                        benchmarkState.exactRequiredStableSeconds >= 5.0 &&
                        benchmarkProfiling.exactChunksPending == 0 &&
                        benchmarkProfiling.uploadQueueDepth == 0 &&
                        completedRequiredSweep;
                    const bool stationaryExactReady =
                        benchmarkConfig.scenario == BenchmarkScenarioKind::StationaryExactFill &&
                        streamingStatus.exactReadyChunks >= benchmarkConfig.targetExactReadyChunks &&
                        streamingStatus.exactPendingUploads == 0 &&
                        benchmarkProfiling.uploadQueueDepth == 0;
                    const bool fullExactReady =
                        benchmarkConfig.scenario != BenchmarkScenarioKind::StationaryExactFill &&
                        streamingStatus.phase == StreamingPhase::SteadyState &&
                        streamingStatus.exactRequiredChunks > 0 &&
                        streamingStatus.exactReadyChunks >= streamingStatus.exactRequiredChunks &&
                        streamingStatus.exactPendingUploads == 0 &&
                        discoverySettled;
                    if (benchmarkState.fullExactReadySeconds < 0.0 &&
                        (fullExactReady || stationaryExactReady))
                    {
                        benchmarkState.fullExactReadySeconds = benchmarkState.elapsedSeconds;
                    }
                    if (fullExactReady || stationaryExactReady)
                    {
                        benchmarkState.completionHoldSeconds += frameTime;
                    }
                    else
                    {
                        benchmarkState.completionHoldSeconds = 0.0;
                    }

                    if (benchmarkState.completionHoldSeconds >= 0.5)
                    {
                        benchmarkState.completed = true;
                        benchmarkRequestClose = true;
                        return true;
                    }
                    if (benchmarkState.elapsedSeconds >= benchmarkConfig.maxDurationSeconds)
                    {
                        benchmarkState.completed = true;
                        benchmarkState.timedOut = true;
                        benchmarkRequestClose = true;
                        return true;
                    }
                    return false;
                };

                const bool usesAutomatedCamera =
                    benchmarkConfig.scenario == BenchmarkScenarioKind::StationaryExactFill ||
                    benchmarkConfig.scenario == BenchmarkScenarioKind::StraightLineSprint ||
                    benchmarkConfig.scenario == BenchmarkScenarioKind::TurnHeavyTraversal ||
                    benchmarkConfig.scenario == BenchmarkScenarioKind::VerticalTravel ||
                    benchmarkConfig.scenario == BenchmarkScenarioKind::PostReleaseExactSweepFill;
                if (usesAutomatedCamera)
                {
                    applyBenchmarkCameraPose(camera, benchmarkConfig, benchmarkState);
                    if (benchmarkConfig.scenario != BenchmarkScenarioKind::StationaryExactFill &&
                        benchmarkConfig.scenario != BenchmarkScenarioKind::PostReleaseExactSweepFill &&
                        benchmarkState.elapsedSeconds >=
                        benchmarkConfig.movementDurationSeconds + benchmarkConfig.cooldownDurationSeconds)
                    {
                        benchmarkState.completed = true;
                        benchmarkRequestClose = true;
                    }
                }
                if (benchmarkConfig.scenario == BenchmarkScenarioKind::FullExactPreload ||
                    benchmarkConfig.scenario == BenchmarkScenarioKind::PlayerIdleExactFill ||
                    benchmarkConfig.scenario == BenchmarkScenarioKind::PostReleaseExactFill ||
                    benchmarkConfig.scenario == BenchmarkScenarioKind::StationaryExactFill ||
                    benchmarkConfig.scenario == BenchmarkScenarioKind::PostReleaseExactSweepFill)
                {
                    (void)updateExactFillCompletion();
                }
                else if (!usesAutomatedCamera && playerReleased)
                {
                    benchmarkState.completed = true;
                    benchmarkRequestClose = true;
                }

                benchmarkState.finalCameraPosition = camera.position;
                benchmarkState.finalYawDegrees = camera.yaw;
                benchmarkState.finalPitchDegrees = camera.pitch;

                const ChunkProfilingSnapshot progressProfiling = chunkManager.sampleProfilingSnapshot();
                const bool shouldWriteProgress =
                    benchmarkState.elapsedSeconds >= benchmarkState.nextProgressLogSeconds ||
                    benchmarkState.completed;
                if (shouldWriteProgress)
                {
                    appendBenchmarkProgressLog(benchmarkConfig,
                                               benchmarkState,
                                               camera,
                                               progressProfiling,
                                               streamingStatus,
                                               benchmarkState.completed
                                                   ? (benchmarkState.timedOut ? "timed_out" : "completed")
                                                   : "progress");
                    benchmarkState.nextProgressLogSeconds =
                        benchmarkState.elapsedSeconds + benchmarkConfig.progressLogIntervalSeconds;
                }
            }
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
            const bool scriptedBenchmarkRuntime =
                benchmarkConfig.enabled && !benchmarkScenarioUsesInteractiveRuntime(benchmarkConfig.scenario);
            if (screenshotSweepConfig.enabled || screenshotReproConfig.enabled || scriptedBenchmarkRuntime)
            {
                accumulator = 0.0;
            }
            else
            {
                while (accumulator >= kFixedTimeStep)
                {
                    if (inputContextPtr)
                    {
                        PlayerInputState inputState = computePlayerInputState(window, *inputContextPtr, camera, chunkManager);
                        updatePhysics(camera, chunkManager, inputState, static_cast<float>(kFixedTimeStep));
                        mobSystem.update(camera.position, chunkManager, static_cast<float>(kFixedTimeStep));
                    }
                    else
                    {
                        InputContext dummy;
                        PlayerInputState inputState = computePlayerInputState(window, dummy, camera, chunkManager);
                        updatePhysics(camera, chunkManager, inputState, static_cast<float>(kFixedTimeStep));
                        mobSystem.update(camera.position, chunkManager, static_cast<float>(kFixedTimeStep));
                    }
                    accumulator -= kFixedTimeStep;
                }
            }
        }
        else
        {
            accumulator = 0.0;
        }

        if (playerReleased)
        {
            const bool scriptedCaptureActive =
                screenshotSweepConfig.enabled ||
                screenshotReproConfig.enabled ||
                (benchmarkConfig.enabled && !benchmarkScenarioUsesInteractiveRuntime(benchmarkConfig.scenario));
            if (!scriptedCaptureActive)
            {
                chunkManager.updateHighlight(camera.position, camera.front());
            }

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
                    chunkManager.placeBlock(hit.blockPos,
                                           hit.faceNormal,
                                           inputContext.selectedPlacementBlock);
                }
                inputContext.rightMouseJustPressed = false;
            }
        }
        inputContext.leftMouseJustPressed = false;
        inputContext.rightMouseJustPressed = false;

        bool capturePlacementsApplied = true;
        if (playerReleased && !captureOverrides.placements.empty())
        {
            for (auto& action : captureOverrides.placements)
            {
                if (action.applied)
                {
                    continue;
                }

                const glm::ivec3 placedWorldPos = action.targetBlockPos + action.faceNormal;
                const BlockId existingPlacedBlock = chunkManager.blockAt(placedWorldPos);
                if (existingPlacedBlock == action.block)
                {
                    action.applied = true;
                    continue;
                }

                capturePlacementsApplied = false;
                if (action.block == BlockId::Air)
                {
                    if (existingPlacedBlock == BlockId::Air || chunkManager.destroyBlock(placedWorldPos))
                    {
                        action.applied = (chunkManager.blockAt(placedWorldPos) == BlockId::Air);
                    }
                    continue;
                }

                if (existingPlacedBlock != BlockId::Air)
                {
                    chunkManager.destroyBlock(placedWorldPos);
                    continue;
                }

                if (chunkManager.placeBlock(action.targetBlockPos, action.faceNormal, action.block))
                {
                    action.applied = true;
                }
            }

            capturePlacementsApplied = std::all_of(captureOverrides.placements.begin(),
                                                   captureOverrides.placements.end(),
                                                   [](const CapturePlacementAction& action)
                                                   {
                                                       return action.applied;
                                                   });
        }

        bool screenshotReproCaptureThisFrame = false;
        std::filesystem::path screenshotReproCapturePath;
        std::filesystem::path screenshotReproLodDebugPath;
        bool screenshotSweepCaptureThisFrame = false;
        std::filesystem::path screenshotSweepCapturePath;
        if (screenshotReproConfig.enabled && playerReleased)
        {
            inputContext.showDebugOverlay = false;
            inputContext.cameraMouseCaptured = true;

            if (!screenshotReproState.initialized)
            {
                screenshotReproState.initialized = true;
                screenshotReproState.waitFramesRemaining = screenshotReproConfig.settleFrames;
                screenshotReproState.lodReadyWaitSeconds = 0.0;
                screenshotReproState.lodTimeoutReported = false;
            }

            float reproYaw = screenshotReproConfig.yawDegrees;
            float reproPitch = screenshotReproConfig.pitchDegrees;
            if (screenshotReproConfig.useLookTarget)
            {
                const glm::vec2 yawPitch =
                    yawPitchFromLookTarget(screenshotReproConfig.position, screenshotReproConfig.lookTarget);
                reproYaw = yawPitch.x;
                reproPitch = yawPitch.y;
            }

            applyCameraPose(camera, screenshotReproConfig.position, reproYaw, reproPitch);

            if (!screenshotReproState.captureRequested)
            {
                if (!capturePlacementsApplied)
                {
                    screenshotReproState.waitFramesRemaining = screenshotReproConfig.settleFrames;
                    screenshotReproState.lodReadyWaitSeconds = 0.0;
                    screenshotReproState.lodTimeoutReported = false;
                }
                else
                {
                    const bool lodShellReady =
                        !screenshotReproConfig.waitForLodReady ||
                        (streamingStatus.farActiveTiles > 0 &&
                         streamingStatus.farReadyTiles >= streamingStatus.farActiveTiles);
                    const bool lodReadyTimedOut =
                        screenshotReproConfig.waitForLodReady &&
                        screenshotReproConfig.lodReadyTimeoutSeconds > 0.0 &&
                        screenshotReproState.lodReadyWaitSeconds >= screenshotReproConfig.lodReadyTimeoutSeconds;

                    if (!lodShellReady && !lodReadyTimedOut)
                    {
                        screenshotReproState.waitFramesRemaining = screenshotReproConfig.settleFrames;
                        screenshotReproState.lodReadyWaitSeconds += frameTime;
                    }
                    else
                    {
                        if (!lodShellReady && lodReadyTimedOut && !screenshotReproState.lodTimeoutReported)
                        {
                            screenshotReproState.lodTimeoutReported = true;
                            std::cout << "Repro capture LOD-ready wait timed out after "
                                      << screenshotReproState.lodReadyWaitSeconds
                                      << "s with "
                                      << streamingStatus.farReadyTiles << '/'
                                      << streamingStatus.farActiveTiles
                                      << " far tiles ready." << std::endl;
                        }

                        if (screenshotReproState.waitFramesRemaining > 0)
                        {
                            --screenshotReproState.waitFramesRemaining;
                        }
                        else
                        {
                            screenshotReproCapturePath = screenshotReproConfig.outputPath;
                            if (screenshotReproConfig.writeLodDebugSnapshot)
                            {
                                screenshotReproLodDebugPath = screenshotReproCapturePath;
                                screenshotReproLodDebugPath.replace_extension(".lod.json");
                            }
                            screenshotReproCaptureThisFrame = true;
                            screenshotReproState.captureRequested = true;
                            std::cout << "Capturing repro screenshot at XYZ: "
                                      << camera.position.x << ", "
                                      << camera.position.y << ", "
                                      << camera.position.z
                                      << " yaw=" << camera.yaw
                                      << " pitch=" << camera.pitch
                                      << " lodReady=" << streamingStatus.farReadyTiles
                                      << "/" << streamingStatus.farActiveTiles << std::endl;
                        }
                    }
                }
            }
        }
        if (screenshotSweepConfig.enabled && !screenshotReproConfig.enabled && playerReleased)
        {
            inputContext.showDebugOverlay = false;
            inputContext.cameraMouseCaptured = true;

            if (!screenshotSweepState.initialized)
            {
                screenshotSweepState.initialized = true;
                screenshotSweepState.anchorPosition = camera.position;
                screenshotSweepState.anchorPosition.y += screenshotSweepConfig.heightOffsetBlocks;
                screenshotSweepState.waitFramesRemaining = screenshotSweepConfig.initialSettleFrames;
            }

            const std::size_t totalSweepPoses = screenshotSweepPoseCount(screenshotSweepConfig);
            if (screenshotSweepState.poseIndex >= totalSweepPoses)
            {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            else
            {
                const float sweepYaw =
                    screenshotSweepYawForIndex(screenshotSweepConfig, screenshotSweepState.poseIndex);
                const float sweepPitch =
                    screenshotSweepPitchForIndex(screenshotSweepConfig, screenshotSweepState.poseIndex);
                applyCameraPose(camera,
                                screenshotSweepState.anchorPosition,
                                sweepYaw,
                                sweepPitch);

                if (!capturePlacementsApplied)
                {
                    screenshotSweepState.waitFramesRemaining = screenshotSweepConfig.settleFramesPerCapture;
                }
                else if (screenshotSweepState.waitFramesRemaining > 0)
                {
                    --screenshotSweepState.waitFramesRemaining;
                }
                else
                {
                    const std::string captureFileName =
                        screenshotSweepPoseLabel(screenshotSweepState.poseIndex, sweepYaw, sweepPitch);
                    screenshotSweepCapturePath = screenshotSweepConfig.outputDir / captureFileName;
                    screenshotSweepCaptureThisFrame = true;
                    screenshotSweepState.manifest << captureFileName << ','
                                                 << sweepYaw << ','
                                                 << sweepPitch << ','
                                                 << camera.position.x << ','
                                                 << camera.position.y << ','
                                                 << camera.position.z << '\n';
                    screenshotSweepState.manifest.flush();
                    ++screenshotSweepState.poseIndex;
                    screenshotSweepState.waitFramesRemaining = screenshotSweepConfig.settleFramesPerCapture;
                }
            }
        }

        chunkManager.setRenderSynchronization(renderer.frameFence(), renderer.lastSubmittedFrameFenceValue());
        noteDiagnosticPhase("frame/chunk_update");
        chunkManager.update(camera.position, camera.front());
        renderer.setUploadSynchronization(chunkManager.uploadFence(),
                                          chunkManager.lastSubmittedUploadFenceValue(),
                                          chunkManager.farUploadFence(),
                                          chunkManager.lastSubmittedFarUploadFenceValue(),
                                          chunkManager.exactGpuFence(),
                                          chunkManager.lastSubmittedExactGpuFenceValue());

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

        const RenderDistanceSettings projectionRenderSettings = chunkManager.renderDistanceSettings();
        const int visibleDistanceBlocks =
            chunkRadiusToBlocks(std::max(projectionRenderSettings.exactChunks, projectionRenderSettings.totalChunks));
        const float currentFarPlane = computeFarPlaneForDistanceBlocks(visibleDistanceBlocks);
        kFarPlane = currentFarPlane;
        const glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(60.0f), aspect, kNearPlane, currentFarPlane);
        const glm::mat4 view = glm::lookAt(camera.position, camera.position + camera.front(), camera.up());
        const glm::mat4 viewProj = projection * view;
        const Frustum frustum = Frustum::fromMatrix(viewProj);

        const auto updateEnvironment = [&]()
        {
            const RenderDistanceSettings renderSettings = chunkManager.renderDistanceSettings();
            const bool exactOnly = renderSettings.totalChunks <= renderSettings.exactChunks;
            const int hiddenExactPreloadChunks = exactOnly ? kHiddenExactPreloadBufferChunks : 0;
            const float exactVisibleDistanceBlocks =
                static_cast<float>(chunkRadiusToBlocks(renderSettings.exactChunks));
            const float totalVisibleDistanceBlocks =
                static_cast<float>(chunkRadiusToBlocks(std::max(renderSettings.exactChunks, renderSettings.totalChunks)));
            const float hiddenExactPreloadBlocks =
                static_cast<float>(chunkRadiusToBlocks(hiddenExactPreloadChunks));
            const float effectiveVisibleDistanceBlocks = exactOnly
                ? (exactVisibleDistanceBlocks + hiddenExactPreloadBlocks)
                : (totalVisibleDistanceBlocks + static_cast<float>(kChunkSizeX * 2));
            const float configuredFogStartBlocks =
                static_cast<float>(std::max(renderSettings.fogStartBlocks, 0));
            const float minFogStartBlocks = exactOnly
                ? std::max(24.0f, exactVisibleDistanceBlocks * 0.48f)
                : std::max(std::max(24.0f, exactVisibleDistanceBlocks * 0.35f),
                           effectiveVisibleDistanceBlocks * 0.42f);
            const float maxFogStartBlocks = exactOnly
                ? std::max(minFogStartBlocks + 16.0f, exactVisibleDistanceBlocks - 12.0f)
                : std::max(minFogStartBlocks + 16.0f, effectiveVisibleDistanceBlocks * 0.82f);

            environment.farDistanceBlocks = effectiveVisibleDistanceBlocks;
            environment.fogStartBlocks = std::min(
                std::clamp(configuredFogStartBlocks, minFogStartBlocks, maxFogStartBlocks),
                (exactOnly ? (exactVisibleDistanceBlocks - 8.0f) : (effectiveVisibleDistanceBlocks - 8.0f)));

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

        noteDiagnosticPhase("frame/render_begin");
        renderer.beginFrame(glm::vec4(120.0f / 255.0f,
                                      167.0f / 255.0f,
                                      255.0f / 255.0f,
                                      1.0f));
        if (chunkManager.streamingPhase() != StreamingPhase::ExactPreload)
        {
            noteDiagnosticPhase("frame/build_render_data");
            const auto buildRenderDataStart = std::chrono::steady_clock::now();
            WorldRenderData renderData = chunkManager.buildRenderData(frustum);
            mobSystem.appendRenderBatches(renderData, frustum, resolveMobTextureBinding);
            buildRenderDataMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - buildRenderDataStart).count();
            const auto renderWorldStart = std::chrono::steady_clock::now();
            noteDiagnosticPhase("frame/render_world");
            renderer.renderWorld(renderData, view, projection, camera.position, blockAtlas, environment);
            renderWorldCpuMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - renderWorldStart).count();
        }
        noteDiagnosticPhase("frame/ui");
        renderer.beginImGuiFrame();

        const double currentFpsEstimate = (fpsFrameCount > 0 && fpsTimer > 0.0)
                                             ? static_cast<double>(fpsFrameCount) / fpsTimer
                                             : fpsValue;
        std::string debugOverlayText;
        std::string lightingSnapshotText;
        std::string holeDebugSnapshotText;
        std::string lodDiagnosticsText;
        const std::filesystem::path lodDiagnosticsDumpPath = std::filesystem::path("artifacts") / "repro_capture" / "f1_lod_diagnostics.json";
        std::string hitBlockSummary{"none"};
        std::string hitBlockType{"none"};
        glm::vec3 samplePosition = camera.position;
        glm::ivec3 lightProbeBlock(static_cast<int>(std::floor(camera.position.x)),
                                   static_cast<int>(std::floor(camera.position.y)),
                                   static_cast<int>(std::floor(camera.position.z)));
        std::string probeBlockType{blockIdLabel(chunkManager.blockAt(lightProbeBlock))};
        LightSample probeLight = chunkManager.lightAt(lightProbeBlock);

        if (inputContext.showDebugOverlay)
        {
            std::ostringstream debugStream;
            debugStream.setf(std::ios::fixed, std::ios::floatfield);
            debugStream << "FPS: " << std::setprecision(0) << currentFpsEstimate << '\n';
             debugStream << std::setprecision(1);
             debugStream << "XYZ: " << camera.position.x << ", "
                         << camera.position.y << ", "
                         << camera.position.z << '\n';
             debugStream << "Move Mode: " << (camera.flyMode ? "Fly" : (camera.onGround ? "Ground" : "Air")) << '\n';
             debugStream << "Yaw/Pitch: " << camera.yaw << ", "
                         << camera.pitch << '\n';
             debugStream << std::setprecision(3);
             debugStream << "Front: " << camera.front().x << ", "
                         << camera.front().y << ", "
                         << camera.front().z << '\n';
             debugStream << std::setprecision(1);
             debugStream << "Biome: " << chunkManager.biomeNameAt(camera.position) << '\n';

            const RaycastHit debugHit = chunkManager.raycast(camera.position, camera.front());
            if (debugHit.hit)
            {
                samplePosition = glm::vec3(debugHit.blockPos);
                lightProbeBlock = debugHit.blockPos + debugHit.faceNormal;
                hitBlockType = blockIdLabel(chunkManager.blockAt(debugHit.blockPos));
                probeBlockType = blockIdLabel(chunkManager.blockAt(lightProbeBlock));

                std::ostringstream hitStream;
                hitStream << debugHit.blockPos.x << ", " << debugHit.blockPos.y << ", " << debugHit.blockPos.z
                          << " face " << debugHit.faceNormal.x << ", " << debugHit.faceNormal.y << ", " << debugHit.faceNormal.z;
                hitBlockSummary = hitStream.str();
                debugStream << "Hit Block: " << hitBlockSummary << " [" << hitBlockType << "]\n";
            }
            else
            {
                debugStream << "Hit Block: none\n";
            }

            probeLight = chunkManager.lightAt(lightProbeBlock);
            debugStream << "Probe Block: " << lightProbeBlock.x << ", "
                        << lightProbeBlock.y << ", "
                        << lightProbeBlock.z << " [" << probeBlockType << "]\n";
            debugStream << "Light Probe: sky=" << static_cast<int>(probeLight.sky)
                        << " block=" << static_cast<int>(probeLight.block) << '\n';
            debugStream << "Place Mode: "
                        << blockIdLabel(inputContext.selectedPlacementBlock)
                        << '\n';

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

            const RenderDistanceSettings renderSettings = chunkManager.renderDistanceSettings();
            const glm::ivec3 lightingDebugWorldPos(static_cast<int>(std::floor(samplePosition.x)),
                                                   static_cast<int>(std::floor(samplePosition.y)),
                                                   static_cast<int>(std::floor(samplePosition.z)));
            std::ostringstream snapshotStream;
            snapshotStream.setf(std::ios::fixed, std::ios::floatfield);
            snapshotStream << std::setprecision(1);
            snapshotStream << "XYZ: " << camera.position.x << ", " << camera.position.y << ", " << camera.position.z << '\n';
            snapshotStream << "Move Mode: " << (camera.flyMode ? "Fly" : (camera.onGround ? "Ground" : "Air")) << '\n';
            snapshotStream << "Yaw/Pitch: " << camera.yaw << ", " << camera.pitch << '\n';
            snapshotStream << "Hit Block: " << hitBlockSummary;
            if (hitBlockType != "none")
            {
                snapshotStream << " [" << hitBlockType << "]";
            }
            snapshotStream << '\n';
            snapshotStream << "Probe Block: " << lightProbeBlock.x << ", " << lightProbeBlock.y << ", " << lightProbeBlock.z
                           << " [" << probeBlockType << "]\n";
            snapshotStream << "Light Probe: sky=" << static_cast<int>(probeLight.sky)
                           << " block=" << static_cast<int>(probeLight.block) << '\n';
            snapshotStream << "Time/Exposure/WP: " << environment.timeOfDay << " / "
                           << environment.tonemap.exposure << " / "
                           << environment.tonemap.whitePoint << '\n';
            snapshotStream << "Exact/Total/Fog: " << renderSettings.exactChunks << " / "
                           << renderSettings.totalChunks << " / "
                           << renderSettings.fogStartBlocks << '\n';
            snapshotStream << "Terrain Debug: " << terrainDebugViewLabel(environment.debug.terrainDebugView) << '\n';
            snapshotStream << "Passes: world=" << (environment.debug.worldPassEnabled ? "on" : "off")
                           << " sky=" << (environment.debug.skyPassEnabled ? "on" : "off")
                           << " aerial=" << (environment.debug.aerialPerspectiveEnabled ? "on" : "off")
                           << " fog=" << (environment.debug.fogFallbackEnabled ? "on" : "off")
                           << " shadows=" << (environment.debug.shadowsEnabled ? "on" : "off")
                           << " sun=" << (environment.debug.directSunEnabled ? "on" : "off") << '\n';
            snapshotStream << "Enhanced Atmosphere: " << (environment.atmosphereEnabled ? "on" : "off")
                           << " mieG=" << environment.atmosphere.mieAnisotropy
                           << " aerialKm=" << environment.atmosphere.aerialPerspectiveDistanceKm << '\n';
            snapshotStream << "View/Sun: viewY=" << viewDirection.y
                           << " sunY=" << environment.sunDirection.y
                           << " dot=" << sunViewDot
                           << " nearHorizon=" << (nearHorizonView ? "yes" : "no")
                           << " belowHorizon=" << (lookingBelowHorizon ? "yes" : "no") << '\n';
            snapshotStream << "Streaming: exact " << streamingStatus.exactReadyChunks << "/" << streamingStatus.exactRequiredChunks
                           << " totalTarget=" << renderSettings.totalChunks
                           << " lod="
                           << (renderSettings.totalChunks > renderSettings.exactChunks ? "cpu_lod_active" : "exact_only");
            if (renderSettings.totalChunks > renderSettings.exactChunks)
            {
                snapshotStream << " lodTiles=" << streamingStatus.farReadyTiles << "/" << streamingStatus.farActiveTiles
                               << " queued=" << streamingStatus.farQueuedTiles;
                if (streamingStatus.farPendingUploadTiles > 0)
                {
                    snapshotStream << " pendingUpload=" << streamingStatus.farPendingUploadTiles;
                }
            }
            snapshotStream << '\n';
            snapshotStream << chunkManager.exactLightingDebugSnapshot(lightingDebugWorldPos);
            lightingSnapshotText = snapshotStream.str();

            const RecentEditHoleDebugSnapshot holeDebugSnapshot =
                chunkManager.recentEditHoleDebugSnapshot(camera.position);
            const LodDiagnosticsSnapshot lodDiagnosticsSnapshot =
                chunkManager.lodDiagnosticsSnapshot(camera.position);
            std::ostringstream holeStream;
            holeStream.setf(std::ios::fixed, std::ios::floatfield);
            holeStream << std::setprecision(2);
            if (!holeDebugSnapshot.hasRecentEdit)
            {
                holeStream << "No recent block edit tracked.\n";
                holeStream << "Break or place a block, wait for the flicker, then copy this panel.\n";
                holeStream << "The capture keeps a short event history for a few seconds after each edit.";
            }
            else
            {
                holeStream << "Recent Edit: " << holeDebugSnapshot.editKind
                           << " at " << holeDebugSnapshot.editWorldPos.x << ", "
                           << holeDebugSnapshot.editWorldPos.y << ", "
                           << holeDebugSnapshot.editWorldPos.z
                           << " chunk " << holeDebugSnapshot.editChunkCoord.x << ", "
                           << holeDebugSnapshot.editChunkCoord.y << ", "
                           << holeDebugSnapshot.editChunkCoord.z
                           << " age=" << holeDebugSnapshot.ageSeconds << "s\n";
                holeStream << "CameraChunkY: " << holeDebugSnapshot.cameraChunkY
                           << " verticalRadius=" << holeDebugSnapshot.verticalRadius << '\n';
                holeStream << "Events:\n";
                if (holeDebugSnapshot.recentEvents.empty())
                {
                    holeStream << "  (none)\n";
                }
                else
                {
                    for (const std::string& event : holeDebugSnapshot.recentEvents)
                    {
                        holeStream << "  " << event << '\n';
                    }
                }

                holeStream << "Tracked Chunks:\n";
                for (const RecentEditHoleChunkInfo& chunkInfo : holeDebugSnapshot.chunks)
                {
                    holeStream << "  [" << chunkInfo.coord.x << ", " << chunkInfo.coord.y << ", " << chunkInfo.coord.z << "] "
                               << (chunkInfo.present ? chunkInfo.stateLabel : "Missing")
                               << " idx=" << chunkInfo.indexCount
                               << " page=";
                    if (chunkInfo.bufferPageIndex == (std::numeric_limits<std::uint32_t>::max)())
                    {
                        holeStream << "none";
                    }
                    else
                    {
                        holeStream << chunkInfo.bufferPageIndex;
                    }
                    holeStream << " has=" << (chunkInfo.hasBlocks ? "y" : "n")
                               << " mesh=" << (chunkInfo.meshReady ? "y" : "n")
                               << " queued=" << (chunkInfo.queuedForUpload ? "y" : "n")
                               << " flight=" << chunkInfo.inFlight
                               << " span=[" << chunkInfo.columnMinChunkY << ", " << chunkInfo.columnMaxChunkY << "]"
                               << " h=";
                    if (chunkInfo.columnHeight == (std::numeric_limits<int>::min)())
                    {
                        holeStream << "none";
                    }
                    else
                    {
                        holeStream << chunkInfo.columnHeight;
                    }
                    holeStream << " src=" << chunkInfo.heightSource
                               << " evict=" << (chunkInfo.wouldEvict ? "yes" : "no")
                               << '\n';
                }
            }
            holeDebugSnapshotText = holeStream.str();

            std::ostringstream lodStream;
            lodStream.setf(std::ios::fixed, std::ios::floatfield);
            lodStream << std::setprecision(2);
            lodStream << "LOD Shell\n";
            lodStream << "Tiles: active=" << lodDiagnosticsSnapshot.activeTiles
                      << " ready=" << lodDiagnosticsSnapshot.readyTiles
                      << " dirty=" << lodDiagnosticsSnapshot.dirtyTiles
                      << " inFlight=" << lodDiagnosticsSnapshot.inFlightTiles << '\n';
            lodStream << "Build Avg ms: mesh=" << lodDiagnosticsSnapshot.averageBuildMs
                      << " synth=" << lodDiagnosticsSnapshot.averageGpuSynthesisMs
                      << " stamp=" << lodDiagnosticsSnapshot.averageGpuStampMs
                      << " face=" << lodDiagnosticsSnapshot.averageGpuFaceBuildMs << '\n';

            if (lodDiagnosticsSnapshot.tiles.empty())
            {
                lodStream << "No active LOD tiles near the camera.";
            }
            else
            {
                for (std::size_t i = 0; i < lodDiagnosticsSnapshot.tiles.size(); ++i)
                {
                    const LodDiagnosticsTileSnapshot& tile = lodDiagnosticsSnapshot.tiles[i];
                    lodStream << '\n'
                              << "#" << (i + 1)
                              << " L" << tile.level
                              << " [" << tile.tileCoord.x << ", " << tile.tileCoord.y << "]"
                              << " d2=" << tile.distanceSq
                              << " idx=" << tile.indexCount
                              << " active=" << (tile.active ? "y" : "n")
                              << " dirty=" << (tile.dirty ? "y" : "n")
                              << " flight=" << (tile.inFlight ? "y" : "n") << '\n';
                    lodStream << "  worldMin=[" << tile.worldMin.x << ", " << tile.worldMin.y << ", " << tile.worldMin.z
                              << "] worldMax=[" << tile.worldMax.x << ", " << tile.worldMax.y << ", " << tile.worldMax.z
                              << "] span=" << tile.chunkSpanBlocks
                              << " blockScale=" << tile.blockScaleBlocks << '\n';
                }
            }
            lodDiagnosticsText = lodStream.str();
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

        if (inputContext.showControlsOverlay)
        {
            static constexpr const char* kControlsHelp =
                "Controls\n"
                "W/A/S/D: Move\n"
                "Space: Jump\n"
                "Double-tap Space: Toggle flight\n"
                "Flight: Space up, Shift down\n"
                "Mouse: Look\n"
                "Left Mouse: Break block\n"
                "Right Mouse: Place block\n"
                "E: Open block picker\n"
                "L: Toggle Grass / DebugLamp placement\n"
                ". : Release or recapture mouse\n"
                "F1: Debug overlay, lighting lab, mob spawning, and LOD diagnostics\n"
                "F2: Teleport dialog\n"
                "N: Render distance dialog\n"
                "H: Toggle this help\n"
                "Esc: Quit while mouse is captured\n"
                "\n"
                "Lighting isolation\n"
                "1. Press F1, then press . to use the UI.\n"
                "2. Use Exact Only, World Only, No Sun, or the terrain debug modes.\n"
                "3. Use Copy Lighting Snapshot and send the text back.";

            ImGui::SetNextWindowPos(ImVec2(static_cast<float>(framebufferWidth) - 18.0f, 18.0f),
                                    ImGuiCond_Always,
                                    ImVec2(1.0f, 0.0f));
            ImGui::SetNextWindowBgAlpha(0.78f);
            ImGui::Begin("Controls",
                         nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs);
            ImGui::TextUnformatted(kControlsHelp);
            ImGui::End();
        }

        if (inputContext.showDebugOverlay)
        {
            ImGui::SetNextWindowPos(ImVec2(12.0f, 260.0f), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.85f);
            ImGui::Begin("Lighting Lab", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::TextUnformatted(inputContext.cameraMouseCaptured
                                       ? "Press . to release the mouse for UI. Press H for controls."
                                       : "Press . again to return to camera look. Press H for controls.");
            ImGui::Text("Placement Block: %s (press E to choose)",
                        blockIdLabel(inputContext.selectedPlacementBlock));
            ImGui::Separator();
            ImGui::TextUnformatted("Quick Isolate");

            const auto applyBeautyPreset = [&]()
            {
                // Optional cinematic-style path. Keep terrain tuning anchored to Base Game first.
                environment.atmosphereEnabled = true;
                environment.debug.worldPassEnabled = true;
                environment.debug.skyPassEnabled = true;
                environment.debug.aerialPerspectiveEnabled = true;
                environment.debug.fogFallbackEnabled = true;
                environment.debug.shadowsEnabled = true;
                environment.debug.directSunEnabled = true;
                environment.debug.aoIntensity = 1.0f;
                environment.debug.terrainDebugView = TerrainDebugView::None;
            };
            const auto applyWorldOnlyPreset = [&]()
            {
                // This is the canonical/default BlockGame look used for lighting evaluation.
                environment.atmosphereEnabled = false;
                environment.debug.worldPassEnabled = true;
                environment.debug.skyPassEnabled = false;
                environment.debug.aerialPerspectiveEnabled = false;
                environment.debug.fogFallbackEnabled = true;
                environment.debug.shadowsEnabled = true;
                environment.debug.directSunEnabled = true;
                environment.debug.aoIntensity = 1.0f;
                environment.debug.terrainDebugView = TerrainDebugView::None;
            };
            const auto applyNoSunPreset = [&]()
            {
                environment.debug.shadowsEnabled = false;
                environment.debug.directSunEnabled = false;
                environment.debug.aoIntensity = 1.0f;
                environment.debug.terrainDebugView = TerrainDebugView::None;
            };
            const auto applyDefaultsPreset = [&]()
            {
                // Reset to the shipping/default presentation, not the enhanced atmosphere mode.
                applyWorldOnlyPreset();
                environment.timeOfDay = 12.0f;
                environment.tonemap.exposure = 0.62f;
                environment.tonemap.whitePoint = 9.0f;
                environment.atmosphere.aerialPerspectiveDistanceKm = 12.0f;
                environment.atmosphere.mieAnisotropy = 0.76f;
                chunkManager.setExactRenderDistanceChunks(kDefaultNearRenderDistance);
                chunkManager.setTotalRenderDistanceChunks(kDefaultTotalRenderDistanceChunks);
                chunkManager.setFogStartBlocks(kDefaultFarFogStartBlocks);
            };

            if (ImGui::Button("Enhanced Atmo"))
            {
                applyBeautyPreset();
            }
            ImGui::SameLine();
            if (ImGui::Button("Base Game"))
            {
                applyWorldOnlyPreset();
            }
            ImGui::SameLine();
            if (ImGui::Button("No Sun"))
            {
                applyNoSunPreset();
            }
            ImGui::SameLine();
            if (ImGui::Button("Exact Only"))
            {
                chunkManager.setTotalRenderDistanceChunks(chunkManager.exactRenderDistanceChunks());
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset Defaults"))
            {
                applyDefaultsPreset();
            }

            if (ImGui::Button("None"))
            {
                environment.debug.terrainDebugView = TerrainDebugView::None;
            }
            ImGui::SameLine();
            if (ImGui::Button("Sky Light"))
            {
                environment.debug.terrainDebugView = TerrainDebugView::SkyLight;
            }
            ImGui::SameLine();
            if (ImGui::Button("Block Light"))
            {
                environment.debug.terrainDebugView = TerrainDebugView::BlockLight;
            }
            ImGui::SameLine();
            if (ImGui::Button("Mip"))
            {
                environment.debug.terrainDebugView = TerrainDebugView::MipLevel;
            }

            ImGui::Separator();
            ImGui::TextUnformatted("Environment");
            ImGui::TextWrapped("Base Game is the default lighting target. Enhanced Atmosphere is optional and should stay non-default when tuning terrain readability.");
            ImGui::Checkbox("Enhanced Atmosphere (Non-default)", &environment.atmosphereEnabled);
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
            ImGui::Checkbox("Sky Pass (Enhanced Atmo)", &environment.debug.skyPassEnabled);
            ImGui::Checkbox("Aerial Perspective (Enhanced Atmo)", &environment.debug.aerialPerspectiveEnabled);
            ImGui::Checkbox("Fog Fallback", &environment.debug.fogFallbackEnabled);
            ImGui::Checkbox("Shadows", &environment.debug.shadowsEnabled);
            ImGui::Checkbox("Direct Sun", &environment.debug.directSunEnabled);
            ImGui::SliderFloat("AO Intensity", &environment.debug.aoIntensity, 0.25f, 2.50f, "%.2f");
            int terrainDebugView = static_cast<int>(environment.debug.terrainDebugView);
            if (ImGui::Combo("Terrain Debug",
                             &terrainDebugView,
                             "None\0Sky Light\0Block Light\0Mip Level\0AO\0"))
            {
                environment.debug.terrainDebugView =
                    static_cast<TerrainDebugView>(terrainDebugView);
            }
            ImGui::Separator();
            ImGui::TextUnformatted("Render Distance");
            RenderDistanceSettings renderSettings = chunkManager.renderDistanceSettings();
            int exactChunks = renderSettings.exactChunks;
            if (ImGui::SliderInt("Exact Chunks", &exactChunks, 1, kMaxExactRenderDistanceChunks))
            {
                chunkManager.setExactRenderDistanceChunks(exactChunks);
                renderSettings.exactChunks = exactChunks;
            }
            int totalChunks = renderSettings.totalChunks;
            if (ImGui::SliderInt("Total Chunks (LOD Radius)", &totalChunks, 1, kMaxTotalRenderDistanceChunks))
            {
                chunkManager.setTotalRenderDistanceChunks(totalChunks);
                renderSettings.totalChunks = totalChunks;
            }
            int fogStartBlocks = renderSettings.fogStartBlocks;
            const int fogSliderMax =
                (std::max)(chunkRadiusToBlocks(std::max(renderSettings.exactChunks + 4, renderSettings.totalChunks)),
                           256);
            if (ImGui::SliderInt("Fog Start", &fogStartBlocks, 0, fogSliderMax))
            {
                chunkManager.setFogStartBlocks(fogStartBlocks);
                environment.fogStartBlocks = static_cast<float>(fogStartBlocks);
                renderSettings.fogStartBlocks = fogStartBlocks;
            }
            const bool lodActive = renderSettings.totalChunks > renderSettings.exactChunks;
            const double exactPercent = streamingStatus.exactRequiredChunks > 0
                                            ? (100.0 * static_cast<double>(streamingStatus.exactReadyChunks) /
                                               static_cast<double>(streamingStatus.exactRequiredChunks))
                                            : 100.0;
            const double lodPercent = streamingStatus.farActiveTiles > 0
                                          ? (100.0 * static_cast<double>(streamingStatus.farReadyTiles) /
                                             static_cast<double>(streamingStatus.farActiveTiles))
                                          : 100.0;
            ImGui::Text("Streaming: Exact %d/%d | Total Radius %d | LOD %s",
                        streamingStatus.exactReadyChunks,
                        streamingStatus.exactRequiredChunks,
                        renderSettings.totalChunks,
                        lodActive ? "active" : "off");
            ImGui::Text("Exact bubble: %d/%d ready (%.0f%%) | pending uploads %d",
                        streamingStatus.exactReadyChunks,
                        streamingStatus.exactRequiredChunks,
                        exactPercent,
                        streamingStatus.exactPendingUploads);
            if (lodActive)
            {
                ImGui::Text("LOD tiles: %d/%d ready (%.0f%%) | queued %d | pending upload %d | dirty %d",
                            streamingStatus.farReadyTiles,
                            streamingStatus.farActiveTiles,
                            lodPercent,
                            streamingStatus.farQueuedTiles,
                            streamingStatus.farPendingUploadTiles,
                            streamingStatus.farDirtyTiles);
                ImGui::TextWrapped("Only the inner Exact Radius streams real chunks. The outer %d chunks are LOD-only visual terrain until you move closer.",
                                   std::max(renderSettings.totalChunks - renderSettings.exactChunks, 0));
                ImGui::TextWrapped("GPU-backed distant terrain is active beyond the Exact Radius. Exact chunks still own gameplay, collision, and edits.");
            }
            ImGui::Separator();
            ImGui::TextUnformatted("Mobs");
            ImGui::Text("Definitions: %zu | Spawned: %zu",
                        mobSystem.definitionCount(),
                        mobSystem.instanceCount());
            const bool pigAvailable = (mobSystem.findModel("pig") != nullptr);
            const bool cowAvailable = (mobSystem.findModel("cow") != nullptr);
            if (!pigAvailable)
            {
                ImGui::BeginDisabled();
            }
            if (ImGui::Button("Spawn Pig"))
            {
                spawnPigNearPlayer();
            }
            if (!pigAvailable)
            {
                ImGui::EndDisabled();
                ImGui::TextUnformatted("Pig definition missing from assets/mobs.");
            }
            if (!cowAvailable)
            {
                ImGui::BeginDisabled();
            }
            if (ImGui::Button("Spawn Cow"))
            {
                spawnCowNearPlayer();
            }
            if (!cowAvailable)
            {
                ImGui::EndDisabled();
                ImGui::TextUnformatted("Cow definition missing from assets/mobs.");
            }
            ImGui::Separator();
            ImGui::TextUnformatted("View Diagnostics");
            ImGui::Text("Yaw/Pitch: %.1f / %.1f", camera.yaw, camera.pitch);
            ImGui::Text("Front: %.3f %.3f %.3f",
                        viewDirection.x,
                        viewDirection.y,
                        viewDirection.z);
            ImGui::Text("View Y: %.3f (%.1f deg)", viewDirection.y, viewElevationDeg);
            ImGui::Text("Sun Y: %.3f (%.1f deg)", environment.sunDirection.y, sunElevationDeg);
            ImGui::Text("View.Sun: %.3f", sunViewDot);
            ImGui::Text("Above Ground: %.2f blocks", altitudeAboveGround);
            ImGui::Text("Fog Start/Visible: %.0f / %.0f", environment.fogStartBlocks, environment.farDistanceBlocks);
            ImGui::Text("Fog Span: %.0f", fogSpanBlocks);
            ImGui::Text("Near Horizon: %s", nearHorizonView ? "yes" : "no");
            ImGui::Text("Looking Below Horizon: %s", lookingBelowHorizon ? "yes" : "no");
            ImGui::Text("Hit Block: %s", hitBlockSummary.c_str());
            ImGui::Text("Hit Block Type: %s", hitBlockType.c_str());
            ImGui::Text("Probe Block: %d %d %d", lightProbeBlock.x, lightProbeBlock.y, lightProbeBlock.z);
            ImGui::Text("Probe Block Type: %s", probeBlockType.c_str());
            ImGui::Text("Probe Light: sky=%d block=%d", static_cast<int>(probeLight.sky), static_cast<int>(probeLight.block));
            ImGui::Separator();
            ImGui::TextUnformatted("Snapshot");
            if (ImGui::Button("Copy Lighting Snapshot"))
            {
                ImGui::SetClipboardText(lightingSnapshotText.c_str());
            }
            ImGui::BeginChild("LightingSnapshot", ImVec2(430.0f, 220.0f), true, ImGuiWindowFlags_HorizontalScrollbar);
            ImGui::TextUnformatted(lightingSnapshotText.c_str());
            ImGui::EndChild();
            if (ImGui::Button("Copy Hole Snapshot"))
            {
                ImGui::SetClipboardText(holeDebugSnapshotText.c_str());
            }
            ImGui::BeginChild("HoleDebugSnapshot", ImVec2(430.0f, 200.0f), true, ImGuiWindowFlags_HorizontalScrollbar);
            ImGui::TextUnformatted(holeDebugSnapshotText.c_str());
            ImGui::EndChild();
            ImGui::Separator();
            ImGui::TextUnformatted("How To Isolate");
            ImGui::TextWrapped("1. Disable Sky Pass. If the band stays, it is not the sky dome.");
            ImGui::TextWrapped("2. Disable Aerial Perspective. If the band disappears, the aerial LUT is the source.");
            ImGui::TextWrapped("3. Use Exact Only to isolate near-chunk streaming behavior.");
            ImGui::TextWrapped("4. Use Sky Light or Block Light terrain debug to confirm whether the bad region is actually carrying broken light data.");
            ImGui::TextWrapped("5. Disable Fog Fallback with Aerial Perspective off. If the band still stays, it is world-side shading or geometry.");
            ImGui::End();
        }

        if (inputContext.showDebugOverlay)
        {
            ImGui::SetNextWindowPos(ImVec2(462.0f, 260.0f), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.85f);
            ImGui::Begin("LOD Diagnostics", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::TextUnformatted("Nearest active far-page shell state. Use this before guessing at parity or mesher bugs.");
            if (ImGui::Button("Copy LOD Snapshot"))
            {
                ImGui::SetClipboardText(lodDiagnosticsText.c_str());
            }
            ImGui::SameLine();
            if (ImGui::Button("Write LOD JSON"))
            {
                chunkManager.writeLodDebugSnapshot(lodDiagnosticsDumpPath, camera.position);
            }
            ImGui::TextWrapped("JSON path: %s", lodDiagnosticsDumpPath.string().c_str());
            ImGui::BeginChild("LodDiagnosticsSnapshot", ImVec2(520.0f, 420.0f), true, ImGuiWindowFlags_HorizontalScrollbar);
            ImGui::TextUnformatted(lodDiagnosticsText.c_str());
            ImGui::EndChild();
            ImGui::End();
        }

        if (!benchmarkConfig.enabled && !playerReleased && !loadingOverlayText.empty())
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
            ImGui::TextUnformatted("Enter exact chunks and optional total chunks (LOD beyond exact, e.g. 12 96):");
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

        if (inputContext.showBlockPickerGUI)
        {
            ImGui::SetNextWindowPos(ImVec2(framebufferWidth * 0.5f, framebufferHeight * 0.5f),
                                    ImGuiCond_Always,
                                    ImVec2(0.5f, 0.5f));
            ImGui::Begin("Block Picker",
                         nullptr,
                         ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::TextUnformatted("Choose the block placed by right-click.");
            ImGui::Text("Current: %s", blockIdLabel(inputContext.selectedPlacementBlock));
            ImGui::Separator();

            for (BlockId block : placeableBlockOptions())
            {
                const bool selected = inputContext.selectedPlacementBlock == block;
                if (ImGui::Selectable(blockIdLabel(block), selected))
                {
                    inputContext.selectedPlacementBlock = block;
                    inputContext.placeLampMode = (block == BlockId::DebugLamp);
                }
            }

            ImGui::Separator();
            if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGuiKey_Escape))
            {
                inputContext.showBlockPickerGUI = false;
            }
            ImGui::End();
        }

        if (!benchmarkConfig.enabled &&
            !screenshotSweepConfig.enabled &&
            !screenshotReproConfig.enabled &&
            !profilingOverlayText.empty())
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

        if (screenshotReproCaptureThisFrame)
        {
            if (!screenshotReproLodDebugPath.empty())
            {
                chunkManager.writeLodDebugSnapshot(screenshotReproLodDebugPath, camera.position);
            }
            renderer.requestScreenshot(screenshotReproCapturePath);
        }
        if (screenshotSweepCaptureThisFrame)
        {
            renderer.requestScreenshot(screenshotSweepCapturePath);
        }

        if (!benchmarkConfig.enabled &&
            !screenshotSweepConfig.enabled &&
            !screenshotReproConfig.enabled &&
            playerReleased &&
            isGameplayMouseCaptured(inputContext))
        {
            drawCrosshairOverlay(framebufferWidth, framebufferHeight);
        }

        noteDiagnosticPhase("frame/end_frame");
        renderer.endFrame();
        noteDiagnosticPhase("frame/presented");

        if (benchmarkConfig.enabled && benchmarkState.started)
        {
            const double frameCpuMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - frameCpuStart).count();
            benchmarkState.frameTimesMs.push_back(frameCpuMs);

            const ChunkProfilingSnapshot frameChunkSnapshot = chunkManager.sampleProfilingSnapshot();
            const RendererProfilingSnapshot frameRendererSnapshot = renderer.profilingSnapshot();
            benchmarkState.lodGpuCullMs.push_back(frameRendererSnapshot.lodGpuCullMs);
            benchmarkState.lodIndirectBuildMs.push_back(frameRendererSnapshot.lodIndirectBuildMs);
            benchmarkState.exactGpuSynthMs.push_back(frameChunkSnapshot.exactGpuSynthMs);
            benchmarkState.exactGpuStampMs.push_back(frameChunkSnapshot.exactGpuStampMs);
            benchmarkState.exactGpuLightMs.push_back(frameChunkSnapshot.exactGpuLightMs);
            benchmarkState.exactGpuFaceCountMs.push_back(frameChunkSnapshot.exactGpuFaceCountMs);
            benchmarkState.exactGpuFacePrefixMs.push_back(frameChunkSnapshot.exactGpuFacePrefixMs);
            benchmarkState.exactGpuFaceEmitMs.push_back(frameChunkSnapshot.exactGpuFaceEmitMs);
            benchmarkState.exactGpuTotalMs.push_back(frameChunkSnapshot.exactGpuTotalMs);
            benchmarkState.gpuLocalUsageMiB.push_back(
                static_cast<double>(frameChunkSnapshot.gpuLocalUsageBytes) / (1024.0 * 1024.0));
            benchmarkState.exactGpuTotalMiB.push_back(
                static_cast<double>(frameChunkSnapshot.exactGpuTotalBytes) / (1024.0 * 1024.0));
            const StreamingStatusSnapshot frameStreamingStatus = chunkManager.streamingStatusSnapshot();
            recordBenchmarkSpike(benchmarkState,
                                 frameCpuMs,
                                 camera,
                                 frameChunkSnapshot,
                                 frameRendererSnapshot,
                                 frameStreamingStatus,
                                 pollEventsMs,
                                 buildRenderDataMs,
                                 renderWorldCpuMs);
        }

        if (benchmarkRequestClose)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        if (screenshotReproConfig.enabled && screenshotReproState.captureRequested)
        {
            noteDiagnosticPhase("frame/request_close");
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }

    if (benchmarkConfig.enabled)
    {
        if (!benchmarkState.completed)
        {
            std::cerr << "Benchmark scenario '" << benchmarkScenarioName(benchmarkConfig.scenario)
                      << "' did not complete." << std::endl;
            exitCode = EXIT_FAILURE;
        }
        else
        {
            const BenchmarkFrameSummary frameSummary = summarizeFrameTimes(benchmarkState.frameTimesMs);
            const ChunkBenchmarkReport benchmarkReport = chunkManager.benchmarkReport();
            ChunkProfilingSnapshot finalProfiling = chunkManager.sampleProfilingSnapshot();
            const RendererProfilingSnapshot finalRendererProfiling = renderer.profilingSnapshot();
            finalProfiling.lodGpuCullMs = finalRendererProfiling.lodGpuCullMs;
            finalProfiling.lodIndirectBuildMs = finalRendererProfiling.lodIndirectBuildMs;
            const StreamingStatusSnapshot finalStreamingStatus = chunkManager.streamingStatusSnapshot();
            const RenderDistanceSettings finalRenderSettings = chunkManager.renderDistanceSettings();
            if (!writeBenchmarkScenarioJson(benchmarkConfig,
                                            benchmarkState,
                                            frameSummary,
                                            benchmarkReport,
                                            finalProfiling,
                                            finalStreamingStatus,
                                            finalRenderSettings))
            {
                std::cerr << "Failed to write benchmark scenario output to "
                          << benchmarkConfig.outputPath << std::endl;
                exitCode = EXIT_FAILURE;
            }
            else
            {
                std::cout << "Benchmark scenario written to " << benchmarkConfig.outputPath << std::endl;
            }
        }

        std::cout.flush();
        std::cerr.flush();
        std::_Exit(exitCode);
    }

    noteDiagnosticPhase("shutdown/detach_sync");
    renderer.setUploadSynchronization(nullptr, 0, nullptr, 0);
    chunkManager.setRenderSynchronization(nullptr, 0);
    noteDiagnosticPhase("shutdown/chunk_manager");
    }
    noteDiagnosticPhase("shutdown/renderer");
    renderer.shutdown();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return exitCode;
}

} // namespace

int main(int argc, char** argv)
{
    applyLaunchDebugOptions(argc, argv);

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

    int exitCode = EXIT_FAILURE;
    try
    {
        exitCode = runGame();
    }
    catch (const std::exception& e)
    {
        appendCrashLog(std::string("uncaught exception: ") + e.what());
        appendDiagnosticSnapshot("uncaught exception");
        std::cerr << "Unhandled exception: " << e.what() << '\n';
    }
    catch (...)
    {
        appendCrashLog("uncaught exception: unknown exception");
        appendDiagnosticSnapshot("uncaught exception");
        std::cerr << "Unhandled non-standard exception" << std::endl;
    }

    shutdownCrashLogging();
    return exitCode;
}
