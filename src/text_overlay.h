#pragma once

#include <glad/glad.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <string>
#include <vector>

class TextOverlay
{
public:
    TextOverlay();
    ~TextOverlay();

    TextOverlay(const TextOverlay&) = delete;
    TextOverlay& operator=(const TextOverlay&) = delete;
    TextOverlay(TextOverlay&&) = delete;
    TextOverlay& operator=(TextOverlay&&) = delete;

    void render(const std::string& text,
                float x,
                float y,
                int screenWidth,
                int screenHeight,
                float pixelHeight,
                const glm::vec3& color);

    [[nodiscard]] float lineHeight(float pixelHeight) const noexcept;

private:
    struct Glyph
    {
        bool valid{false};
        bool hasBitmap{false};
        float advance{0.0f};
        float offsetX{0.0f};
        float offsetY{0.0f};
        float width{0.0f};
        float height{0.0f};
        glm::vec2 uvMin{0.0f};
        glm::vec2 uvMax{0.0f};
    };

    struct Vertex
    {
        float x;
        float y;
        float u;
        float v;
    };

    void setupBuffers();
    void setupShader();
    bool loadFontAtlas(const std::string& path);
    void cleanup();
    [[nodiscard]] bool isReady() const noexcept;

    GLuint vao_{0};
    GLuint vbo_{0};
    GLuint shaderProgram_{0};
    GLuint fontTexture_{0};
    GLint screenSizeLocation_{-1};
    GLint colorLocation_{-1};
    GLint textureLocation_{-1};

    static constexpr int kFirstCodepoint = 32;
    static constexpr int kCodepointCount = 95;
    static constexpr float kBaseFontPixelHeight = 32.0f;

    float baseline_{0.0f};
    float baseLineHeight_{0.0f};
    int fallbackGlyphIndex_{-1};

    Glyph glyphs_[kCodepointCount];

    std::vector<Vertex> vertexBuffer_{};
};

