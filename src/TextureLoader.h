#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

struct LoadedTexture
{
    GLuint id{0};
    glm::ivec2 size{0};
};

LoadedTexture loadTexture(const char* path);
