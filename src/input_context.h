#pragma once

#include <string>

#include <glm/glm.hpp>

struct GLFWwindow;

class Camera;
class ChunkManager;

struct InputContext
{
    Camera* camera{nullptr};
    float lastX{0.0f};
    float lastY{0.0f};
    bool firstMouse{true};
    bool leftMousePressed{false};
    bool leftMouseJustPressed{false};
    bool rightMousePressed{false};
    bool rightMouseJustPressed{false};
    bool nKeyPressed{false};
    bool nKeyJustPressed{false};
    bool f2Pressed{false};
    bool f2JustPressed{false};
    bool f1Pressed{false};
    bool f1JustPressed{false};
    bool f3Pressed{false};
    bool f3JustPressed{false};
    bool lodEnabled{false};
    bool showCoordinates{false};
    bool showRenderDistanceGUI{false};
    bool showTeleportGUI{false};
    std::string inputBuffer{};
    std::string teleportBuffer{};
};

struct PlayerInputState
{
    glm::vec3 moveDirection{0.0f};
    bool jumpHeld{false};
};

void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void charCallback(GLFWwindow* window, unsigned int codepoint);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

PlayerInputState computePlayerInputState(GLFWwindow* window,
                                         InputContext& inputContext,
                                         Camera& camera,
                                         ChunkManager& chunkManager);
