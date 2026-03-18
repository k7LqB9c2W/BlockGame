#include <d3dcompiler.h>
#include <wrl/client.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "shader_manifest.h"

#pragma comment(lib, "d3dcompiler.lib")

namespace
{
int compileShader(const std::filesystem::path& shaderRoot,
                  const std::filesystem::path& outputRoot,
                  const ShaderCompileSpec& spec)
{
    const std::filesystem::path sourcePath = shaderRoot / spec.relativePath;
    const std::filesystem::path outputPath =
        outputRoot / compiledShaderFileName(spec.relativePath, spec.entryPoint, spec.target);

    Microsoft::WRL::ComPtr<ID3DBlob> bytecode;
    Microsoft::WRL::ComPtr<ID3DBlob> errors;
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

    std::cout << "compiled " << spec.relativePath << " -> " << outputPath.filename().string() << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: shader_precompiler <shader_root> <output_root>\n";
        return 2;
    }

    const std::filesystem::path shaderRoot = argv[1];
    const std::filesystem::path outputRoot = argv[2];
    int failures = 0;
    for (const ShaderCompileSpec& spec : kBlockGameShaderCompileSpecs)
    {
        failures += compileShader(shaderRoot, outputRoot, spec);
    }

    if (failures != 0)
    {
        std::cerr << "shader precompile failures: " << failures << '\n';
        return 1;
    }

    return 0;
}
