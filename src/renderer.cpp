#include "renderer.h"

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

void renderWorldGeometry(GLuint shaderProgram,
                         const glm::mat4& viewProj,
                         const glm::vec3& cameraPos,
                         const ChunkShaderUniformLocations& uniforms,
                         const ChunkRenderData& renderData)
{
    glUseProgram(shaderProgram);
    if (uniforms.uViewProj >= 0)
    {
        glUniformMatrix4fv(uniforms.uViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
    }
    if (uniforms.uLightDir >= 0)
    {
        glUniform3fv(uniforms.uLightDir, 1, glm::value_ptr(renderData.lightDirection));
    }
    if (uniforms.uCameraPos >= 0)
    {
        glUniform3fv(uniforms.uCameraPos, 1, glm::value_ptr(cameraPos));
    }

    if (renderData.atlasTexture != 0)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, renderData.atlasTexture);
        if (uniforms.uAtlas >= 0)
        {
            glUniform1i(uniforms.uAtlas, 0);
        }
    }

    if (uniforms.uHighlightedBlock >= 0)
    {
        glUniform3f(uniforms.uHighlightedBlock,
                    static_cast<float>(renderData.highlightedBlock.x),
                    static_cast<float>(renderData.highlightedBlock.y),
                    static_cast<float>(renderData.highlightedBlock.z));
    }
    if (uniforms.uHasHighlight >= 0)
    {
        glUniform1i(uniforms.uHasHighlight, renderData.hasHighlight ? 1 : 0);
    }

    for (const ChunkRenderBatch& batch : renderData.batches)
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

    glBindVertexArray(0);
    if (renderData.atlasTexture != 0)
    {
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glUseProgram(0);
}
