#include "text_overlay.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
[[nodiscard]] GLuint compileShader(GLenum type, const char* source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE)
    {
        GLint logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);

        std::string infoLog;
        if (logLength > 0)
        {
            infoLog.resize(static_cast<std::size_t>(logLength));
            GLsizei written = 0;
            glGetShaderInfoLog(shader, logLength, &written, infoLog.data());
            infoLog.resize(static_cast<std::size_t>(written));
        }

        glDeleteShader(shader);
        throw std::runtime_error("Text overlay shader compilation failed: " + infoLog);
    }

    return shader;
}

[[nodiscard]] GLuint createProgram(const char* vertexSrc, const char* fragmentSrc)
{
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_FALSE)
    {
        GLint logLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

        std::string infoLog;
        if (logLength > 0)
        {
            infoLog.resize(static_cast<std::size_t>(logLength));
            GLsizei written = 0;
            glGetProgramInfoLog(program, logLength, &written, infoLog.data());
            infoLog.resize(static_cast<std::size_t>(written));
        }

        glDeleteProgram(program);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        throw std::runtime_error("Text overlay program linkage failed: " + infoLog);
    }

    glDetachShader(program, vertexShader);
    glDetachShader(program, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

[[nodiscard]] std::vector<unsigned char> loadFileBytes(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        return {};
    }

    stream.seekg(0, std::ios::end);
    const std::streampos length = stream.tellg();
    if (length <= 0)
    {
        return {};
    }
    stream.seekg(0, std::ios::beg);

    std::vector<unsigned char> data(static_cast<std::size_t>(length));
    stream.read(reinterpret_cast<char*>(data.data()), length);
    if (!stream)
    {
        return {};
    }

    return data;
}
} // namespace

TextOverlay::TextOverlay()
{
    try
    {
        setupBuffers();
        setupShader();
        if (!loadFontAtlas("assets/fonts/arial.ttf"))
        {
            std::cerr << "Failed to load font atlas from assets/fonts/arial.ttf" << std::endl;
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Failed to initialize text overlay: " << ex.what() << std::endl;
    }
}

TextOverlay::~TextOverlay()
{
    cleanup();
}

void TextOverlay::render(const std::string& text,
                         float x,
                         float y,
                         int screenWidth,
                         int screenHeight,
                         float pixelHeight,
                         const glm::vec3& color)
{
    if (!isReady() || text.empty() || screenWidth <= 0 || screenHeight <= 0 || pixelHeight <= 0.0f)
    {
        return;
    }

    const float scale = pixelHeight / kBaseFontPixelHeight;
    const float baseline = baseline_ * scale;
    const float lineAdvance = (baseLineHeight_ + extraLineSpacing_) * scale;

    vertexBuffer_.clear();
    vertexBuffer_.reserve(text.size() * 6);

    float penX = x;
    float penY = y + baseline;

    for (char ch : text)
    {
        if (ch == '\n')
        {
            penX = x;
            penY += lineAdvance;
            continue;
        }

        int index = static_cast<int>(ch) - kFirstCodepoint;
        if (index < 0 || index >= kCodepointCount || !glyphs_[index].valid)
        {
            index = fallbackGlyphIndex_;
            if (index < 0 || index >= kCodepointCount || !glyphs_[index].valid)
            {
                continue;
            }
        }

        const Glyph& glyph = glyphs_[index];

        if (glyph.hasBitmap)
        {
            const float glyphX0 = penX + glyph.offsetX * scale;
            const float glyphY0 = penY + glyph.offsetY * scale;
            const float glyphX1 = glyphX0 + glyph.width * scale;
            const float glyphY1 = glyphY0 + glyph.height * scale;

            vertexBuffer_.push_back({glyphX0, glyphY0, glyph.uvMin.x, glyph.uvMin.y});
            vertexBuffer_.push_back({glyphX1, glyphY0, glyph.uvMax.x, glyph.uvMin.y});
            vertexBuffer_.push_back({glyphX1, glyphY1, glyph.uvMax.x, glyph.uvMax.y});

            vertexBuffer_.push_back({glyphX0, glyphY0, glyph.uvMin.x, glyph.uvMin.y});
            vertexBuffer_.push_back({glyphX1, glyphY1, glyph.uvMax.x, glyph.uvMax.y});
            vertexBuffer_.push_back({glyphX0, glyphY1, glyph.uvMin.x, glyph.uvMax.y});
        }

        penX += glyph.advance * scale;
    }

    if (vertexBuffer_.empty())
    {
        return;
    }

    glDisable(GL_DEPTH_TEST);
    const GLboolean wasCullFace = glIsEnabled(GL_CULL_FACE);
    if (wasCullFace)
    {
        glDisable(GL_CULL_FACE);
    }

    const GLboolean wasBlend = glIsEnabled(GL_BLEND);
    if (!wasBlend)
    {
        glEnable(GL_BLEND);
    }
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(shaderProgram_);
    glUniform2f(screenSizeLocation_, static_cast<float>(screenWidth), static_cast<float>(screenHeight));
    glUniform3f(colorLocation_, color.x, color.y, color.z);
    glUniform1i(textureLocation_, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fontTexture_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(vertexBuffer_.size() * sizeof(Vertex)),
                 vertexBuffer_.data(),
                 GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertexBuffer_.size()));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    if (!wasBlend)
    {
        glDisable(GL_BLEND);
    }
    if (wasCullFace)
    {
        glEnable(GL_CULL_FACE);
    }
    glEnable(GL_DEPTH_TEST);
}

float TextOverlay::lineHeight(float pixelHeight) const noexcept
{
    if (!isReady() || pixelHeight <= 0.0f)
    {
        return 0.0f;
    }

    return (baseLineHeight_ + extraLineSpacing_) * (pixelHeight / kBaseFontPixelHeight);
}

void TextOverlay::setupBuffers()
{
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, x)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, u)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void TextOverlay::setupShader()
{
    const char* vertexShaderSrc = R"(#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;

out vec2 vUV;

uniform vec2 uScreenSize;

void main()
{
    vec2 ndc = vec2((aPos.x / uScreenSize.x) * 2.0 - 1.0, 1.0 - (aPos.y / uScreenSize.y) * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUV = aUV;
}
)";

    const char* fragmentShaderSrc = R"(#version 330 core
in vec2 vUV;

out vec4 FragColor;

uniform vec3 uColor;
uniform sampler2D uFontTexture;

void main()
{
    float alpha = texture(uFontTexture, vUV).r;
    FragColor = vec4(uColor, alpha);
}
)";

    try
    {
        shaderProgram_ = createProgram(vertexShaderSrc, fragmentShaderSrc);
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        shaderProgram_ = 0;
    }

    if (shaderProgram_ != 0)
    {
        screenSizeLocation_ = glGetUniformLocation(shaderProgram_, "uScreenSize");
        colorLocation_ = glGetUniformLocation(shaderProgram_, "uColor");
        textureLocation_ = glGetUniformLocation(shaderProgram_, "uFontTexture");
    }
}

bool TextOverlay::loadFontAtlas(const std::string& path)
{
    if (shaderProgram_ == 0)
    {
        return false;
    }

    std::vector<unsigned char> fontData = loadFileBytes(path);
    if (fontData.empty())
    {
        return false;
    }

    stbtt_fontinfo fontInfo{};
    if (!stbtt_InitFont(&fontInfo, fontData.data(), 0))
    {
        return false;
    }

    float scale = stbtt_ScaleForPixelHeight(&fontInfo, kBaseFontPixelHeight);
    int ascent = 0;
    int descent = 0;
    int lineGap = 0;
    stbtt_GetFontVMetrics(&fontInfo, &ascent, &descent, &lineGap);
    baseline_ = static_cast<float>(ascent) * scale;
    baseLineHeight_ = static_cast<float>(ascent - descent + lineGap) * scale;
    const float computedGap = static_cast<float>(lineGap) * scale;
    const float minGap = kBaseFontPixelHeight * 0.15f;
    extraLineSpacing_ = std::max(minGap, computedGap);

    constexpr int atlasWidth = 512;
    constexpr int atlasHeight = 512;
    std::vector<unsigned char> atlas(atlasWidth * atlasHeight, 0);
    stbtt_pack_context packContext{};
    if (!stbtt_PackBegin(&packContext, atlas.data(), atlasWidth, atlasHeight, 0, 1, nullptr))
    {
        return false;
    }
    stbtt_PackSetOversampling(&packContext, 2, 2);

    std::array<stbtt_packedchar, kCodepointCount> packedChars{};
    if (!stbtt_PackFontRange(&packContext,
                             fontData.data(),
                             0,
                             kBaseFontPixelHeight,
                             kFirstCodepoint,
                             kCodepointCount,
                             packedChars.data()))
    {
        stbtt_PackEnd(&packContext);
        return false;
    }

    stbtt_PackEnd(&packContext);

    for (int i = 0; i < kCodepointCount; ++i)
    {
        const stbtt_packedchar& pc = packedChars[i];
        Glyph& glyph = glyphs_[i];
        glyph.valid = true;
        glyph.hasBitmap = (pc.x1 > pc.x0) && (pc.y1 > pc.y0);
        glyph.advance = pc.xadvance;
        glyph.offsetX = pc.xoff;
        glyph.offsetY = pc.yoff;
        glyph.width = static_cast<float>(pc.x1 - pc.x0);
        glyph.height = static_cast<float>(pc.y1 - pc.y0);
        glyph.uvMin = glm::vec2(static_cast<float>(pc.x0) / static_cast<float>(atlasWidth),
                                static_cast<float>(pc.y0) / static_cast<float>(atlasHeight));
        glyph.uvMax = glm::vec2(static_cast<float>(pc.x1) / static_cast<float>(atlasWidth),
                                static_cast<float>(pc.y1) / static_cast<float>(atlasHeight));
    }

    const int fallbackCodepoint = '?';
    const int fallbackIndex = fallbackCodepoint - kFirstCodepoint;
    if (fallbackIndex >= 0 && fallbackIndex < kCodepointCount && glyphs_[fallbackIndex].valid)
    {
        fallbackGlyphIndex_ = fallbackIndex;
    }
    else
    {
        fallbackGlyphIndex_ = -1;
    }

    if (fontTexture_ == 0)
    {
        glGenTextures(1, &fontTexture_);
    }

    glBindTexture(GL_TEXTURE_2D, fontTexture_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RED,
                 atlasWidth,
                 atlasHeight,
                 0,
                 GL_RED,
                 GL_UNSIGNED_BYTE,
                 atlas.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    return true;
}

void TextOverlay::cleanup()
{
    if (fontTexture_ != 0)
    {
        glDeleteTextures(1, &fontTexture_);
        fontTexture_ = 0;
    }
    if (shaderProgram_ != 0)
    {
        glDeleteProgram(shaderProgram_);
        shaderProgram_ = 0;
    }
    if (vbo_ != 0)
    {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    if (vao_ != 0)
    {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
}

bool TextOverlay::isReady() const noexcept
{
    return shaderProgram_ != 0 && fontTexture_ != 0;
}

