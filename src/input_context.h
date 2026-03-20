#pragma once

#include <string>

#include <glm/glm.hpp>

// Needed for BlockId defaults used by the placement picker.
#include "chunk_manager.h"

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
    bool periodPressed{false};
    bool periodJustPressed{false};
    bool f2Pressed{false};
    bool f2JustPressed{false};
    bool f1Pressed{false};
    bool f1JustPressed{false};
    bool hKeyPressed{false};
    bool hKeyJustPressed{false};
    bool lKeyPressed{false};
    bool lKeyJustPressed{false};
    bool eKeyPressed{false};
    bool eKeyJustPressed{false};
    bool spacePressed{false};
    bool spaceJustPressed{false};
    bool cameraMouseCaptured{true};
    bool placeLampMode{false};
    BlockId selectedPlacementBlock{BlockId::Grass};
    bool showDebugOverlay{false};
    bool showControlsOverlay{false};
    bool showRenderDistanceGUI{false};
    bool showTeleportGUI{false};
    bool showBlockPickerGUI{false};
    double lastSpacePressTimeSeconds{-1.0};
    std::string inputBuffer{};
    std::string teleportBuffer{};
};

[[nodiscard]] inline bool isGameplayUiOpen(const InputContext& inputContext) noexcept
{
    return inputContext.showRenderDistanceGUI ||
           inputContext.showTeleportGUI ||
           inputContext.showBlockPickerGUI;
}

[[nodiscard]] inline bool isGameplayMouseCaptured(const InputContext& inputContext) noexcept
{
    return inputContext.cameraMouseCaptured &&
           !isGameplayUiOpen(inputContext);
}

struct PlayerInputState
{
    glm::vec3 moveDirection{0.0f};
    bool jumpHeld{false};
    bool ascendHeld{false};
    bool descendHeld{false};
    bool toggleFlightPressed{false};
};

void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void charCallback(GLFWwindow* window, unsigned int codepoint);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

PlayerInputState computePlayerInputState(GLFWwindow* window,
                                         InputContext& inputContext,
                                         Camera& camera,
                                         ChunkManager& chunkManager);
