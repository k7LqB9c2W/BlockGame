#pragma once

#include "chunk_manager.h"

void renderWorldGeometry(GLuint shaderProgram,
                         GLuint farShaderProgram,
                         const glm::mat4& viewProj,
                         const glm::vec3& cameraPos,
                         const ChunkShaderUniformLocations& nearUniforms,
                         const FarTerrainShaderUniformLocations& farUniforms,
                         const WorldRenderData& renderData);
