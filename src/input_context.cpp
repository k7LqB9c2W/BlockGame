#include "input_context.h"

#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "camera.h"
#include "chunk_manager.h"

#include <glm/glm.hpp>

#include <exception>
#include <iostream>

void framebufferSizeCallback(GLFWwindow*, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    auto* input = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
    if (input == nullptr || input->camera == nullptr)
    {
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
    auto* input = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
    if (input == nullptr)
    {
        return;
    }

    if (input->showRenderDistanceGUI && codepoint < 128)
    {
        if (codepoint >= '0' && codepoint <= '9')
        {
            if (input->inputBuffer.size() < 10)
            {
                input->inputBuffer += static_cast<char>(codepoint);
            }
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/)
{
    auto* input = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
    if (input == nullptr)
    {
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

PlayerInputState computePlayerInputState(GLFWwindow* window,
                                         InputContext& inputContext,
                                         Camera& camera,
                                         ChunkManager& chunkManager)
{
    PlayerInputState state;

    bool nKeyCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS);
    inputContext.nKeyJustPressed = nKeyCurrentlyPressed && !inputContext.nKeyPressed;
    inputContext.nKeyPressed = nKeyCurrentlyPressed;
    if (inputContext.nKeyJustPressed && !inputContext.showRenderDistanceGUI)
    {
        inputContext.showRenderDistanceGUI = true;
        inputContext.inputBuffer.clear();
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }

    if (inputContext.showRenderDistanceGUI)
    {
        static bool enterKeyPressed = false;
        bool enterKeyCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS ||
                                          glfwGetKey(window, GLFW_KEY_KP_ENTER) == GLFW_PRESS);
        if (enterKeyCurrentlyPressed && !enterKeyPressed)
        {
            if (!inputContext.inputBuffer.empty())
            {
                try
                {
                    int distance = std::stoi(inputContext.inputBuffer);
                    chunkManager.setRenderDistance(distance);
                }
                catch (const std::exception& e)
                {
                    std::cerr << "Invalid render distance input: " << e.what() << std::endl;
                }
            }
            inputContext.showRenderDistanceGUI = false;
            inputContext.inputBuffer.clear();
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        enterKeyPressed = enterKeyCurrentlyPressed;

        static bool escapeKeyPressed = false;
        bool escapeKeyCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS);
        if (escapeKeyCurrentlyPressed && !escapeKeyPressed)
        {
            inputContext.showRenderDistanceGUI = false;
            inputContext.inputBuffer.clear();
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        escapeKeyPressed = escapeKeyCurrentlyPressed;

        static bool backspaceKeyPressed = false;
        bool backspaceKeyCurrentlyPressed = (glfwGetKey(window, GLFW_KEY_BACKSPACE) == GLFW_PRESS);
        if (backspaceKeyCurrentlyPressed && !backspaceKeyPressed)
        {
            if (!inputContext.inputBuffer.empty())
            {
                inputContext.inputBuffer.pop_back();
            }
        }
        backspaceKeyPressed = backspaceKeyCurrentlyPressed;
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

    state.jumpHeld = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
    return state;
}
