#pragma once

#include <glm/glm.hpp>

class Camera
{
public:
    glm::vec3 position{0.0f, 10.0f, 5.0f};
    float yaw{-90.0f};
    float pitch{0.0f};
    float moveSpeed{8.0f};
    float mouseSensitivity{0.12f};

    // Physics properties
    glm::vec3 velocity{0.0f, 0.0f, 0.0f};
    bool onGround{false};

    const glm::vec3& front() const noexcept;
    const glm::vec3& up() const noexcept;
    const glm::vec3& right() const noexcept;
    const glm::vec3& worldUp() const noexcept;

    void processMouse(float xoffset, float yoffset);
    void updateVectors();

private:
    glm::vec3 front_{0.0f, 0.0f, -1.0f};
    glm::vec3 up_{0.0f, 1.0f, 0.0f};
    glm::vec3 right_{1.0f, 0.0f, 0.0f};
    glm::vec3 worldUp_{0.0f, 1.0f, 0.0f};
};
