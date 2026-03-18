#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <process.h>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "shader_manifest.h"

#pragma comment(lib, "d3dcompiler.lib")

namespace
{
namespace fs = std::filesystem;

std::string normalizeShaderDependencyKey(const fs::path& value)
{
    std::string key = value.lexically_normal().generic_string();
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char ch)
    {
        return static_cast<char>(std::tolower(ch));
    });
    return key;
}

std::string_view trimLeft(std::string_view value)
{
    const std::size_t firstNonWhitespace = value.find_first_not_of(" \t\r\n");
    if (firstNonWhitespace == std::string_view::npos)
    {
        return {};
    }
    return value.substr(firstNonWhitespace);
}

bool parseQuotedIncludeDirective(std::string_view line, std::string_view& includePath)
{
    line = trimLeft(line);
    if (line.empty() || line.front() != '#')
    {
        return false;
    }

    line.remove_prefix(1);
    line = trimLeft(line);
    if (!line.starts_with("include"))
    {
        return false;
    }

    line.remove_prefix(7);
    line = trimLeft(line);
    if (line.empty() || line.front() != '"')
    {
        return false;
    }

    line.remove_prefix(1);
    const std::size_t closingQuote = line.find('"');
    if (closingQuote == std::string_view::npos)
    {
        return false;
    }

    includePath = line.substr(0, closingQuote);
    return true;
}

bool isShaderTreeNewerThanOutput(const fs::path& sourcePath,
                                 const fs::file_time_type& outputTime,
                                 std::unordered_set<std::string>& visitedPaths)
{
    const fs::path normalizedSourcePath = sourcePath.lexically_normal();
    const std::string key = normalizeShaderDependencyKey(normalizedSourcePath);
    if (!visitedPaths.insert(key).second)
    {
        return true;
    }

    std::error_code ec;
    const fs::file_time_type sourceTime = fs::last_write_time(normalizedSourcePath, ec);
    if (ec || sourceTime > outputTime)
    {
        return false;
    }

    std::ifstream source(normalizedSourcePath);
    if (!source)
    {
        return false;
    }

    std::string line;
    while (std::getline(source, line))
    {
        std::string_view includePath;
        if (!parseQuotedIncludeDirective(line, includePath))
        {
            continue;
        }

        const fs::path childPath = (normalizedSourcePath.parent_path() / fs::path(includePath)).lexically_normal();
        if (!isShaderTreeNewerThanOutput(childPath, outputTime, visitedPaths))
        {
            return false;
        }
    }

    return true;
}

bool isShaderUpToDate(const fs::path& sourcePath, const fs::path& outputPath)
{
    std::error_code ec;
    if (!fs::exists(outputPath, ec) || ec)
    {
        return false;
    }

    const fs::file_time_type outputTime = fs::last_write_time(outputPath, ec);
    if (ec)
    {
        return false;
    }

    std::unordered_set<std::string> visitedPaths;
    return isShaderTreeNewerThanOutput(sourcePath, outputTime, visitedPaths);
}

std::wstring widen(std::string_view value)
{
    return std::wstring(value.begin(), value.end());
}

std::wstring dxcTargetProfile(std::string_view target)
{
    if (target.ends_with("5_0"))
    {
        std::wstring mapped = widen(target);
        mapped[mapped.size() - 3] = L'6';
        return mapped;
    }
    return widen(target);
}

int compileShaderWithDxc(const fs::path& dxcPath,
                         const fs::path& shaderRoot,
                         const fs::path& outputPath,
                         const ShaderCompileSpec& spec)
{
    const std::wstring sourcePath = (shaderRoot / spec.relativePath).wstring();
    const std::wstring outputFile = outputPath.wstring();
    const std::wstring includePath = shaderRoot.wstring();
    const std::wstring entryPoint = widen(spec.entryPoint);
    const std::wstring target = dxcTargetProfile(spec.target);

    std::vector<std::wstring> storage;
    storage.reserve(13);
    storage.push_back(dxcPath.wstring());
    storage.push_back(L"-E");
    storage.push_back(entryPoint);
    storage.push_back(L"-T");
    storage.push_back(target);
    storage.push_back(L"-Fo");
    storage.push_back(outputFile);
    storage.push_back(L"-I");
    storage.push_back(includePath);
    storage.push_back(L"-Ges");
    storage.push_back(L"-O3");
    storage.push_back(sourcePath);

    std::vector<const wchar_t*> argv;
    argv.reserve(storage.size() + 1);
    for (const std::wstring& arg : storage)
    {
        argv.push_back(arg.c_str());
    }
    argv.push_back(nullptr);

    const intptr_t result = _wspawnv(_P_WAIT, dxcPath.c_str(), argv.data());
    if (result == -1)
    {
        std::cerr << "failed to launch dxc: " << dxcPath.string() << '\n';
        return 1;
    }
    return static_cast<int>(result);
}

int compileShaderWithD3DCompiler(const fs::path& shaderRoot,
                                 const fs::path& outputPath,
                                 const ShaderCompileSpec& spec)
{
    Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
    Microsoft::WRL::ComPtr<ID3DBlob> errors;
    const fs::path sourcePath = shaderRoot / spec.relativePath;
    const std::wstring widePath = sourcePath.wstring();
    const HRESULT hr = D3DCompileFromFile(widePath.c_str(),
                                          nullptr,
                                          D3D_COMPILE_STANDARD_FILE_INCLUDE,
                                          spec.entryPoint,
                                          spec.target,
                                          D3DCOMPILE_ENABLE_STRICTNESS,
                                          0,
                                          &bytecode,
                                          &errors);
    if (FAILED(hr))
    {
        std::cerr << "shader compile failed: " << sourcePath.string();
        if (errors && errors->GetBufferSize() > 0)
        {
            std::cerr << ": ";
            std::cerr.write(static_cast<const char*>(errors->GetBufferPointer()),
                            static_cast<std::streamsize>(errors->GetBufferSize()));
        }
        std::cerr << '\n';
        return 1;
    }

    std::filesystem::create_directories(outputPath.parent_path());
    std::ofstream out(outputPath, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        std::cerr << "failed to open compiled shader output: " << outputPath.string() << '\n';
        return 1;
    }

    out.write(static_cast<const char*>(bytecode->GetBufferPointer()),
              static_cast<std::streamsize>(bytecode->GetBufferSize()));
    if (!out)
    {
        std::cerr << "failed to write compiled shader output: " << outputPath.string() << '\n';
        return 1;
    }

    return 0;
}

int compileShader(const fs::path& shaderRoot,
                  const fs::path& outputRoot,
                  const fs::path& dxcPath,
                  const ShaderCompileSpec& spec)
{
    const fs::path sourcePath = shaderRoot / spec.relativePath;
    const fs::path outputPath =
        outputRoot / compiledShaderFileName(spec.relativePath, spec.entryPoint, spec.target);

    if (isShaderUpToDate(sourcePath, outputPath))
    {
        std::cout << "skipped " << spec.relativePath << " -> " << outputPath.filename().string() << std::endl;
        return 0;
    }

    fs::create_directories(outputPath.parent_path());
    std::cout << "compiling " << spec.relativePath << " [" << spec.entryPoint << ", " << spec.target << "]";
    if (!dxcPath.empty())
    {
        std::cout << " with dxc";
    }
    std::cout << std::endl;

    const int result = !dxcPath.empty()
                           ? compileShaderWithDxc(dxcPath, shaderRoot, outputPath, spec)
                           : compileShaderWithD3DCompiler(shaderRoot, outputPath, spec);
    if (result != 0)
    {
        std::cerr << "shader compile failed: " << sourcePath.string() << '\n';
        return result;
    }

    std::cout << "compiled " << spec.relativePath << " -> " << outputPath.filename().string() << std::endl;
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 4)
    {
        std::cerr << "usage: shader_precompiler <shader_root> <output_root> [dxc_exe]\n";
        return 2;
    }

    const fs::path shaderRoot = argv[1];
    const fs::path outputRoot = argv[2];
    fs::path dxcPath;
    if (argc == 4)
    {
        dxcPath = argv[3];
    }
    else if (const char* dxcFromEnv = std::getenv("BLOCKGAME_DXC_EXE"))
    {
        dxcPath = dxcFromEnv;
    }

    if (!dxcPath.empty())
    {
        std::error_code ec;
        if (!fs::exists(dxcPath, ec) || ec)
        {
            std::cerr << "dxc executable not found: " << dxcPath.string() << '\n';
            return 2;
        }
    }

    int failures = 0;
    for (const ShaderCompileSpec& spec : kBlockGameShaderCompileSpecs)
    {
        failures += compileShader(shaderRoot, outputRoot, dxcPath, spec);
    }

    if (failures != 0)
    {
        std::cerr << "shader precompile failures: " << failures << '\n';
        return 1;
    }

    return 0;
}
