constexpr int kGlyphHeight = 7;

struct GlyphPattern
{
    std::array<std::uint8_t, kGlyphHeight> rows{};
    int width{0};
};

std::uint8_t rowMask(const char* pattern, int width) noexcept
{
    std::uint8_t mask = 0;
    if (!pattern)
    {
        return mask;
    }

    for (int i = 0; i < width && pattern[i] != '\0'; ++i)
    {
        if (pattern[i] == '1')
        {
            const int bit = width - 1 - i;
            if (bit >= 0 && bit < 8)
            {
                mask |= static_cast<std::uint8_t>(1 << bit);
            }
        }
    }

    return mask;
}

GlyphPattern makeGlyph(const std::array<const char*, kGlyphHeight>& rows, int width) noexcept
{
    GlyphPattern glyph{};
    glyph.width = width;
    for (int row = 0; row < kGlyphHeight; ++row)
    {
        glyph.rows[row] = rowMask(rows[row], width);
    }
    return glyph;
}

const GlyphPattern* glyphForChar(char c) noexcept
{
    switch (c)
    {
    case '0':
    {
        static const GlyphPattern glyph = makeGlyph({{"11111", "10001", "10001", "10001", "10001", "10001", "11111"}}, 5);
        return &glyph;
    }
    case '1':
    {
        static const GlyphPattern glyph = makeGlyph({{"00100", "01100", "00100", "00100", "00100", "00100", "01110"}}, 5);
        return &glyph;
    }
    case '2':
    {
        static const GlyphPattern glyph = makeGlyph({{"11111", "00001", "00001", "11111", "10000", "10000", "11111"}}, 5);
        return &glyph;
    }
    case '3':
    {
        static const GlyphPattern glyph = makeGlyph({{"11111", "00001", "00001", "11111", "00001", "00001", "11111"}}, 5);
        return &glyph;
    }
    case '4':
    {
        static const GlyphPattern glyph = makeGlyph({{"10001", "10001", "10001", "11111", "00001", "00001", "00001"}}, 5);
        return &glyph;
    }
    case '5':
    {
        static const GlyphPattern glyph = makeGlyph({{"11111", "10000", "10000", "11111", "00001", "00001", "11111"}}, 5);
        return &glyph;
    }
    case '6':
    {
        static const GlyphPattern glyph = makeGlyph({{"11111", "10000", "10000", "11111", "10001", "10001", "11111"}}, 5);
        return &glyph;
    }
    case '7':
    {
        static const GlyphPattern glyph = makeGlyph({{"11111", "00001", "00010", "00100", "01000", "01000", "01000"}}, 5);
        return &glyph;
    }
    case '8':
    {
        static const GlyphPattern glyph = makeGlyph({{"11111", "10001", "10001", "11111", "10001", "10001", "11111"}}, 5);
        return &glyph;
    }
    case '9':
    {
        static const GlyphPattern glyph = makeGlyph({{"11111", "10001", "10001", "11111", "00001", "00001", "11111"}}, 5);
        return &glyph;
    }
    case '-':
    {
        static const GlyphPattern glyph = makeGlyph({{"00000", "00000", "00000", "11111", "00000", "00000", "00000"}}, 5);
        return &glyph;
    }
    case '.':
    {
        static const GlyphPattern glyph = makeGlyph({{"0", "0", "0", "0", "0", "0", "1"}}, 1);
        return &glyph;
    }
    case 'X':
    {
        static const GlyphPattern glyph = makeGlyph({{"10001", "01010", "00100", "00100", "01010", "10001", "10001"}}, 5);
        return &glyph;
    }
    case 'Y':
    {
        static const GlyphPattern glyph = makeGlyph({{"10001", "01010", "00100", "00100", "00100", "00100", "00100"}}, 5);
        return &glyph;
    }
    case 'Z':
    {
        static const GlyphPattern glyph = makeGlyph({{"11111", "00001", "00010", "00100", "01000", "10000", "11111"}}, 5);
        return &glyph;
    }
    case ' ':
    {
        static const GlyphPattern glyph = makeGlyph({{"000", "000", "000", "000", "000", "000", "000"}}, 3);
        return &glyph;
    }
    case 'A':
    {
        static const GlyphPattern glyph = makeGlyph({{"01110", "10001", "10001", "11111", "10001", "10001", "10001"}}, 5);
        return &glyph;
    }
    case 'C':
    {
        static const GlyphPattern glyph = makeGlyph({{"01110", "10001", "10000", "10000", "10000", "10001", "01110"}}, 5);
        return &glyph;
    }
    case 'D':
    {
        static const GlyphPattern glyph = makeGlyph({{"11110", "10001", "10001", "10001", "10001", "10001", "11110"}}, 5);
        return &glyph;
    }
    case 'E':
    {
        static const GlyphPattern glyph = makeGlyph({{"11111", "10000", "10000", "11110", "10000", "10000", "11111"}}, 5);
        return &glyph;
    }
    case 'I':
    {
        static const GlyphPattern glyph = makeGlyph({{"111", "010", "010", "010", "010", "010", "111"}}, 3);
        return &glyph;
    }
    case 'N':
    {
        static const GlyphPattern glyph = makeGlyph({{"10001", "11001", "10101", "10101", "10011", "10001", "10001"}}, 5);
        return &glyph;
    }
    case 'R':
    {
        static const GlyphPattern glyph = makeGlyph({{"11110", "10001", "10001", "11110", "10100", "10010", "10001"}}, 5);
        return &glyph;
    }
    case 'S':
    {
        static const GlyphPattern glyph = makeGlyph({{"01111", "10000", "10000", "01110", "00001", "00001", "11110"}}, 5);
        return &glyph;
    }
    case 'T':
    {
        static const GlyphPattern glyph = makeGlyph({{"11111", "00100", "00100", "00100", "00100", "00100", "00100"}}, 5);
        return &glyph;
    }
    case ':':
    {
        static const GlyphPattern glyph = makeGlyph({{"0", "0", "1", "0", "1", "0", "0"}}, 1);
        return &glyph;
    }
    case 'a':
    {
        static const GlyphPattern glyph = makeGlyph({{"00000", "00000", "01110", "00001", "01111", "10001", "01111"}}, 5);
        return &glyph;
    }
    case 'c':
    {
        static const GlyphPattern glyph = makeGlyph({{"00000", "00000", "01110", "10000", "10000", "10001", "01110"}}, 5);
        return &glyph;
    }
    case 'd':
    {
        static const GlyphPattern glyph = makeGlyph({{"00001", "00001", "01111", "10001", "10001", "10001", "01111"}}, 5);
        return &glyph;
    }
    case 'e':
    {
        static const GlyphPattern glyph = makeGlyph({{"00000", "00000", "01110", "10001", "11111", "10000", "01111"}}, 5);
        return &glyph;
    }
    case 'i':
    {
        static const GlyphPattern glyph = makeGlyph({{"010", "000", "110", "010", "010", "010", "111"}}, 3);
        return &glyph;
    }
    case 'n':
    {
        static const GlyphPattern glyph = makeGlyph({{"00000", "00000", "10110", "11001", "10001", "10001", "10001"}}, 5);
        return &glyph;
    }
    case 'r':
    {
        static const GlyphPattern glyph = makeGlyph({{"00000", "00000", "10110", "11001", "10000", "10000", "10000"}}, 5);
        return &glyph;
    }
    case 's':
    {
        static const GlyphPattern glyph = makeGlyph({{"00000", "00000", "01111", "10000", "01110", "00001", "11110"}}, 5);
        return &glyph;
    }
    case 't':
    {
        static const GlyphPattern glyph = makeGlyph({{"00100", "00100", "11111", "00100", "00100", "00100", "00011"}}, 5);
        return &glyph;
    }
    default:
        return nullptr;
    }
}

class TextOverlay
{
public:
    TextOverlay()
    {
        setup();
    }

    ~TextOverlay()
    {
        cleanup();
    }

    void render(const std::string& text, float x, float y, int screenWidth, int screenHeight, float scale, const glm::vec3& color)
    {
        if (text.empty() || shaderProgram_ == 0 || screenWidth <= 0 || screenHeight <= 0)
        {
            return;
        }

        buildVertices(text, x, y, scale);
        if (vertexData_.empty())
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

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertexData_.size() * sizeof(float)), vertexData_.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertexData_.size() / 2));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

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

private:
    GLuint vao_{0};
    GLuint vbo_{0};
    GLuint shaderProgram_{0};
    GLint screenSizeLocation_{-1};
    GLint colorLocation_{-1};
    std::vector<float> vertexData_{};

    void setup()
    {
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        const char* vertexShaderSrc = R"(#version 330 core
layout (location = 0) in vec2 aPos;
uniform vec2 uScreenSize;

void main()
{
    vec2 ndc = vec2((aPos.x / uScreenSize.x) * 2.0 - 1.0, 1.0 - (aPos.y / uScreenSize.y) * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

        const char* fragmentShaderSrc = R"(#version 330 core
out vec4 FragColor;
uniform vec3 uColor;

void main()
{
    FragColor = vec4(uColor, 1.0);
}
)";

        try
        {
            shaderProgram_ = createProgram(vertexShaderSrc, fragmentShaderSrc);
        }
        catch (const std::exception& ex)
        {
            std::cerr << "Failed to create text overlay shader: " << ex.what() << std::endl;
            shaderProgram_ = 0;
        }

        if (shaderProgram_ != 0)
        {
            screenSizeLocation_ = glGetUniformLocation(shaderProgram_, "uScreenSize");
            colorLocation_ = glGetUniformLocation(shaderProgram_, "uColor");
        }
    }

    void cleanup()
    {
        if (vao_ != 0)
        {
            glDeleteVertexArrays(1, &vao_);
            vao_ = 0;
        }
        if (vbo_ != 0)
        {
            glDeleteBuffers(1, &vbo_);
            vbo_ = 0;
        }
        if (shaderProgram_ != 0)
        {
            glDeleteProgram(shaderProgram_);
            shaderProgram_ = 0;
        }
    }

    void buildVertices(const std::string& text, float x, float y, float scale)
    {
        vertexData_.clear();
        vertexData_.reserve(text.size() * 48);

        float cursorX = x;
        float cursorY = y;
        const float advancePadding = scale;

        for (char ch : text)
        {
            if (ch == '\n')
            {
                cursorX = x;
                cursorY += (static_cast<float>(kGlyphHeight) + 1.0f) * scale;
                continue;
            }

            const GlyphPattern* glyph = glyphForChar(ch);
            if (!glyph)
            {
                cursorX += (4.0f * scale) + advancePadding;
                continue;
            }

            appendGlyph(*glyph, cursorX, cursorY, scale);
            cursorX += (static_cast<float>(glyph->width) + 1.0f) * scale;
        }
    }

    void appendGlyph(const GlyphPattern& glyph, float originX, float originY, float scale)
    {
        for (int row = 0; row < kGlyphHeight; ++row)
        {
            std::uint8_t bits = glyph.rows[row];
            for (int col = 0; col < glyph.width; ++col)
            {
                const int bitIndex = glyph.width - 1 - col;
                if (((bits >> bitIndex) & 1u) == 0u)
                {
                    continue;
                }

                float x0 = originX + static_cast<float>(col) * scale;
                float y0 = originY + static_cast<float>(row) * scale;
                float x1 = x0 + scale;
                float y1 = y0 + scale;

                vertexData_.push_back(x0);
                vertexData_.push_back(y0);
                vertexData_.push_back(x1);
                vertexData_.push_back(y0);
                vertexData_.push_back(x1);
                vertexData_.push_back(y1);

                vertexData_.push_back(x0);
                vertexData_.push_back(y0);
                vertexData_.push_back(x1);
                vertexData_.push_back(y1);
                vertexData_.push_back(x0);
                vertexData_.push_back(y1);
            }
        }
    }
};
