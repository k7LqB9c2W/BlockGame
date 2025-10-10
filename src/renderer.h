#pragma once

#include "chunk_manager.h"

void renderWorldGeometry(GLuint shaderProgram,
                         const glm::mat4& viewProj,
                         const glm::vec3& cameraPos,
                         const ChunkShaderUniformLocations& uniforms,
                         const ChunkRenderData& renderData);
