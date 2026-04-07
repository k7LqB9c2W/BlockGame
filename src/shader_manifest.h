#pragma once

#include <array>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>

struct ShaderCompileSpec
{
    const char* relativePath;
    const char* entryPoint;
    const char* target;
};

inline constexpr std::array<ShaderCompileSpec, 50> kBlockGameShaderCompileSpecs{{
    {"world_vs.hlsl", "main", "vs_5_0"},
    {"mob_vs.hlsl", "main", "vs_5_0"},
    {"block_outline_vs.hlsl", "main", "vs_5_0"},
    {"shadow_vs.hlsl", "main", "vs_5_0"},
    {"exact_world_vs.hlsl", "main", "vs_5_0"},
    {"water_vs.hlsl", "main", "vs_5_0"},
    {"exact_shadow_vs.hlsl", "main", "vs_5_0"},
    {"shadow_ps.hlsl", "main", "ps_5_0"},
    {"world_near_ps.hlsl", "main", "ps_5_0"},
    {"water_ps.hlsl", "main", "ps_5_0"},
    {"world_translucent_ps.hlsl", "main", "ps_5_0"},
    {"world_far_ps.hlsl", "main", "ps_5_0"},
    {"mob_ps.hlsl", "main", "ps_5_0"},
    {"block_outline_ps.hlsl", "main", "ps_5_0"},
    {"depth_pyramid.hlsl", "DepthPyramidMain", "cs_5_0"},
    {"lod_gpu_cull.hlsl", "LodCullMain", "cs_5_0"},
    {"lod_gpu_cull.hlsl", "LodIndirectBuildMain", "cs_5_0"},
    {"exact_gpu_cull.hlsl", "ExactCullMain", "cs_5_0"},
    {"exact_gpu_cull.hlsl", "ExactIndirectBuildMain", "cs_5_0"},
    {"fullscreen_vs.hlsl", "main", "vs_5_0"},
    {"base_sky_ps.hlsl", "main", "ps_5_0"},
    {"clouds_vs.hlsl", "main", "vs_5_0"},
    {"clouds_ps.hlsl", "main", "ps_5_0"},
    {"oit_composite_ps.hlsl", "main", "ps_5_0"},
    {"tone_map_ps.hlsl", "main", "ps_5_0"},
    {"atmosphere_transmittance_ps.hlsl", "main", "ps_5_0"},
    {"atmosphere_multiscattering_ps.hlsl", "main", "ps_5_0"},
    {"atmosphere_skyview_ps.hlsl", "main", "ps_5_0"},
    {"atmosphere_sky_ps.hlsl", "main", "ps_5_0"},
    {"atmosphere_aerial_perspective_ps.hlsl", "main", "ps_5_0"},
    {"far_lod_column_atlas_update_canonical_cs.hlsl", "FarLodChunkSeedCacheMain", "cs_5_0"},
    {"far_lod_column_atlas_update_canonical_cs.hlsl", "FarLodColumnSampleCacheMain", "cs_5_0"},
    {"far_lod_column_atlas_update_canonical_cs.hlsl", "FarLodColumnAtlasUpdateMain", "cs_5_0"},
    {"far_lod_chunk_synth_cs.hlsl", "FarLodChunkSynthMain", "cs_5_0"},
    {"far_lod_chunk_structure_stamp_cs.hlsl", "FarLodChunkStructureStampMain", "cs_5_0"},
    {"far_lod_chunk_face_count_cs.hlsl", "FarLodChunkFaceCountMain", "cs_5_0"},
    {"far_lod_chunk_face_prefix_cs.hlsl", "FarLodChunkFacePrefixGroupMain", "cs_5_0"},
    {"far_lod_chunk_face_prefix_cs.hlsl", "FarLodChunkFacePrefixScanMain", "cs_5_0"},
    {"far_lod_chunk_face_prefix_cs.hlsl", "FarLodChunkFacePrefixAddMain", "cs_5_0"},
    {"far_lod_chunk_face_emit_cs.hlsl", "FarLodChunkFaceEmitMain", "cs_5_0"},
    {"exact_chunk_descriptor_gen_cs.hlsl", "ExactChunkDescriptorGenMain", "cs_6_6"},
    {"exact_chunk_synth_cs.hlsl", "ExactChunkSynthMain", "cs_6_6"},
    {"exact_chunk_structure_stamp_cs.hlsl", "ExactChunkStructureStampMain", "cs_6_6"},
    {"exact_chunk_halo_cache_cs.hlsl", "ExactChunkHaloCacheMain", "cs_6_6"},
    {"exact_chunk_light_cs.hlsl", "ExactChunkLightMain", "cs_6_6"},
    {"exact_chunk_seam_export_cs.hlsl", "ExactChunkSeamExportMain", "cs_6_6"},
    {"exact_chunk_face_count_cs.hlsl", "ExactChunkFaceCountMain", "cs_6_6"},
    {"exact_chunk_face_prefix_cs.hlsl", "ExactChunkFacePrefixMain", "cs_6_6"},
    {"exact_chunk_face_emit_cs.hlsl", "ExactChunkFaceEmitMain", "cs_6_6"},
    {"exact_chunk_draw_record_clear_cs.hlsl", "ExactChunkDrawRecordClearMain", "cs_6_6"},
}};

inline std::string sanitizeShaderName(std::string_view value)
{
    std::string result;
    result.reserve(value.size());
    for (char ch : value)
    {
        const unsigned char code = static_cast<unsigned char>(ch);
        result.push_back(std::isalnum(code) ? ch : '_');
    }
    return result;
}

inline std::string compiledShaderFileName(std::string_view relativePath,
                                          std::string_view entryPoint,
                                          std::string_view target)
{
    return sanitizeShaderName(relativePath) + "__" + sanitizeShaderName(entryPoint) + "__"
           + sanitizeShaderName(target) + ".cso";
}

inline std::filesystem::path compiledShaderPath(const std::filesystem::path& shaderRoot,
                                                std::string_view relativePath,
                                                std::string_view entryPoint,
                                                std::string_view target)
{
    return shaderRoot / "compiled" / compiledShaderFileName(relativePath, entryPoint, target);
}

inline std::filesystem::path compiledShaderPathForSource(const std::filesystem::path& sourcePath,
                                                         std::string_view entryPoint,
                                                         std::string_view target)
{
    return compiledShaderPath(sourcePath.parent_path(),
                              sourcePath.filename().string(),
                              entryPoint,
                              target);
}
