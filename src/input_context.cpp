#include "input_context.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>

#include "camera.h"
#include "chunk_manager.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <exception>
#include <iostream>
#include <sstream>

void framebufferSizeCallback(GLFWwindow*, int width, int height)
{
    (void)width;
    (void)height;
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (ImGui::GetCurrentContext() != nullptr)
    {
        ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);
    }
    auto* input = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
    if (input == nullptr || input->camera == nullptr)
    {
        return;
    }

    const bool imguiCapturingMouse = ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse;
    if (!isGameplayMouseCaptured(*input) || imguiCapturingMouse)
    {
        input->firstMouse = true;
        return;
    }

    if (input->firstMouse)
    {
        input->lastX = static_cast<float>(xpos);
        input->lastY = static_cast<float>(ypos);
        input->firstMouse = false;
    }

    const float xoffset = static_cast<float>(xpos) - input->lastX;
    const float yoffset = input->lastY - static_cast<float>(ypos);

    input->lastX = static_cast<float>(xpos);
    input->lastY = static_cast<float>(ypos);

    input->camera->processMouse(xoffset, yoffset);
}

void charCallback(GLFWwindow* window, unsigned int codepoint)
{
    if (ImGui::GetCurrentContext() != nullptr)
    {
        ImGui_ImplGlfw_CharCallback(window, codepoint);
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/)
{
    if (ImGui::GetCurrentContext() != nullptr)
    {
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, 0);
    }
    auto* input = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
    if (input == nullptr)
    {
        return;
    }

    const bool imguiCapturingMouse = ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse;
    if (!isGameplayMouseCaptured(*input) || imguiCapturingMouse)
    {
        input->leftMousePressed = false;
        input->leftMouseJustPressed = false;
        input->rightMousePressed = false;
        input->rightMouseJustPressed = false;
        return;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        bool wasPressed = input->leftMousePressed;
        input->leftMousePressed = (action == GLFW_PRESS);
        input->leftMouseJustPressed = input->leftMousePressed && !wasPressed;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        bool wasPressed = input->rightMousePressed;
        input->rightMousePressed = (action == GLFW_PRESS);
        input->rightMouseJustPressed = input->rightMousePressed && !wasPressed;
    }
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (ImGui::GetCurrentContext() != nullptr)
    {
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (ImGui::GetCurrentContext() != nullptr)
    {
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    }
}

PlayerInputState computePlayerInputState(GLFWwindow* window,
                                         InputContext& inputContext,
                                         Camera& camera,
                                         ChunkManager& chunkManager)
{
    (void)chunkManager;
    static constexpr double kFlightToggleDoubleTapSeconds = 0.30;
    PlayerInputState state;

    bool nKeyCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS);
    inputContext.nKeyJustPressed = nKeyCurrentlyPressed && !inputContext.nKeyPressed;
    inputContext.nKeyPressed = nKeyCurrentlyPressed;
    if (inputContext.nKeyJustPressed && !inputContext.showRenderDistanceGUI && !inputContext.showTeleportGUI)
    {
        inputContext.showRenderDistanceGUI = true;
        inputContext.inputBuffer.clear();
        inputContext.firstMouse = true;
    }

    bool f2CurrentlyPressed = (glfwGetKey(window, GLFW_KEY_F2) == GLFW_PRESS);
    inputContext.f2JustPressed = f2CurrentlyPressed && !inputContext.f2Pressed;
    inputContext.f2Pressed = f2CurrentlyPressed;
    if (inputContext.f2JustPressed && !inputContext.showTeleportGUI && !inputContext.showRenderDistanceGUI)
    {
        inputContext.showTeleportGUI = true;
        inputContext.teleportBuffer.clear();
        inputContext.firstMouse = true;
    }

    bool periodCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_PERIOD) == GLFW_PRESS);
    inputContext.periodJustPressed = periodCurrentlyPressed && !inputContext.periodPressed;
    inputContext.periodPressed = periodCurrentlyPressed;
    if (inputContext.periodJustPressed && !inputContext.showRenderDistanceGUI && !inputContext.showTeleportGUI)
    {
        inputContext.cameraMouseCaptured = !inputContext.cameraMouseCaptured;
        inputContext.firstMouse = true;
        inputContext.leftMousePressed = false;
        inputContext.leftMouseJustPressed = false;
        inputContext.rightMousePressed = false;
        inputContext.rightMouseJustPressed = false;
    }

    bool hKeyCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS);
    inputContext.hKeyJustPressed = hKeyCurrentlyPressed && !inputContext.hKeyPressed;
    inputContext.hKeyPressed = hKeyCurrentlyPressed;
    if (inputContext.hKeyJustPressed && !inputContext.showRenderDistanceGUI && !inputContext.showTeleportGUI)
    {
        inputContext.showControlsOverlay = !inputContext.showControlsOverlay;
        inputContext.firstMouse = true;
    }

    bool lKeyCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS);
    inputContext.lKeyJustPressed = lKeyCurrentlyPressed && !inputContext.lKeyPressed;
    inputContext.lKeyPressed = lKeyCurrentlyPressed;
    if (inputContext.lKeyJustPressed && !inputContext.showRenderDistanceGUI && !inputContext.showTeleportGUI)
    {
        inputContext.placeLampMode = !inputContext.placeLampMode;
    }

    const bool captureMouse = isGameplayMouseCaptured(inputContext);
    glfwSetInputMode(window, GLFW_CURSOR, captureMouse ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);

    const bool spaceCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
    inputContext.spaceJustPressed = spaceCurrentlyPressed && !inputContext.spacePressed;
    inputContext.spacePressed = spaceCurrentlyPressed;

    const bool shiftCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ||
                                       (glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    if (captureMouse)
    {
        if (inputContext.spaceJustPressed)
        {
            const double nowSeconds = glfwGetTime();
            if (inputContext.lastSpacePressTimeSeconds >= 0.0 &&
                (nowSeconds - inputContext.lastSpacePressTimeSeconds) <= kFlightToggleDoubleTapSeconds)
            {
                state.toggleFlightPressed = true;
                inputContext.lastSpacePressTimeSeconds = -1.0;
            }
            else
            {
                inputContext.lastSpacePressTimeSeconds = nowSeconds;
            }
        }

        glm::vec3 forward = camera.front();
        forward.y = 0.0f;
        if (glm::length(forward) > kEpsilon)
        {
            forward = glm::normalize(forward);
        }

        glm::vec3 right = glm::cross(forward, camera.worldUp());
        if (glm::length(right) > kEpsilon)
        {
            right = glm::normalize(right);
        }
        else
        {
            right = camera.right();
        }

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        {
            state.moveDirection += forward;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        {
            state.moveDirection -= forward;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        {
            state.moveDirection -= right;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        {
            state.moveDirection += right;
        }

        if (camera.flyMode)
        {
            state.ascendHeld = spaceCurrentlyPressed;
            state.descendHeld = shiftCurrentlyPressed;
        }
        else
        {
            state.jumpHeld = spaceCurrentlyPressed;
        }
    }
    else
    {
        state.jumpHeld = false;
        state.ascendHeld = false;
        state.descendHeld = false;
        state.toggleFlightPressed = false;
        inputContext.lastSpacePressTimeSeconds = -1.0;
    }

    return state;
}

