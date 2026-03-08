#include "renderer.h"

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

void renderWorldGeometry(GLuint shaderProgram,
                         GLuint farShaderProgram,
                         const glm::mat4& viewProj,
                         const glm::vec3& cameraPos,
                         const ChunkShaderUniformLocations& nearUniforms,
                         const FarTerrainShaderUniformLocations& farUniforms,
                         const WorldRenderData& renderData)
{
    glUseProgram(shaderProgram);
    if (nearUniforms.uViewProj >= 0)
    {
        glUniformMatrix4fv(nearUniforms.uViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
    }
    if (nearUniforms.uLightDir >= 0)
    {
        glUniform3fv(nearUniforms.uLightDir, 1, glm::value_ptr(renderData.lightDirection));
    }
    if (nearUniforms.uCameraPos >= 0)
    {
        glUniform3fv(nearUniforms.uCameraPos, 1, glm::value_ptr(cameraPos));
    }

    if (renderData.atlasTexture != 0)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, renderData.atlasTexture);
        if (nearUniforms.uAtlas >= 0)
        {
            glUniform1i(nearUniforms.uAtlas, 0);
        }
    }

    if (nearUniforms.uHighlightedBlock >= 0)
    {
        glUniform3f(nearUniforms.uHighlightedBlock,
                    static_cast<float>(renderData.highlightedBlock.x),
                    static_cast<float>(renderData.highlightedBlock.y),
                    static_cast<float>(renderData.highlightedBlock.z));
    }
    if (nearUniforms.uHasHighlight >= 0)
    {
        glUniform1i(nearUniforms.uHasHighlight, renderData.hasHighlight ? 1 : 0);
    }

    for (const ChunkRenderBatch& batch : renderData.nearBatches)
    {
        if (batch.counts.empty())
        {
            continue;
        }

        glBindVertexArray(batch.vao);
        glMultiDrawElementsBaseVertex(GL_TRIANGLES,
                                      batch.counts.data(),
                                      GL_UNSIGNED_INT,
                                      batch.offsets.data(),
                                      static_cast<GLsizei>(batch.counts.size()),
                                      batch.baseVertices.data());
    }

    if (farShaderProgram != 0 && !renderData.farBatches.empty())
    {
        glUseProgram(farShaderProgram);
        if (farUniforms.uViewProj >= 0)
        {
            glUniformMatrix4fv(farUniforms.uViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
        }
        if (farUniforms.uLightDir >= 0)
        {
            glUniform3fv(farUniforms.uLightDir, 1, glm::value_ptr(renderData.lightDirection));
        }
        if (farUniforms.uAtlas >= 0)
        {
            glUniform1i(farUniforms.uAtlas, 0);
        }
        if (farUniforms.uCameraPos >= 0)
        {
            glUniform3fv(farUniforms.uCameraPos, 1, glm::value_ptr(cameraPos));
        }
        if (farUniforms.uFogColor >= 0)
        {
            glUniform3fv(farUniforms.uFogColor, 1, glm::value_ptr(renderData.fogColor));
        }
        if (farUniforms.uFogStart >= 0)
        {
            glUniform1f(farUniforms.uFogStart, renderData.fogStart);
        }
        if (farUniforms.uFogEnd >= 0)
        {
            glUniform1f(farUniforms.uFogEnd, renderData.fogEnd);
        }

        for (const ChunkRenderBatch& batch : renderData.farBatches)
        {
            if (batch.counts.empty())
            {
                continue;
            }

            glBindVertexArray(batch.vao);
            glMultiDrawElementsBaseVertex(GL_TRIANGLES,
                                          batch.counts.data(),
                                          GL_UNSIGNED_INT,
                                          batch.offsets.data(),
                                          static_cast<GLsizei>(batch.counts.size()),
                                          batch.baseVertices.data());
        }
    }

    glBindVertexArray(0);
    if (renderData.atlasTexture != 0)
    {
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glUseProgram(0);
}
