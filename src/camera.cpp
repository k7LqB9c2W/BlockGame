#include "camera.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

#include "chunk_manager.h"

const glm::vec3& Camera::front() const noexcept
{
    return front_;
}

const glm::vec3& Camera::up() const noexcept
{
    return up_;
}

const glm::vec3& Camera::right() const noexcept
{
    return right_;
}

const glm::vec3& Camera::worldUp() const noexcept
{
    return worldUp_;
}

void Camera::processMouse(float xoffset, float yoffset)
{
    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;

    yaw += xoffset;
    pitch += yoffset;
    pitch = std::clamp(pitch, -89.0f, 89.0f);
    updateVectors();
}

void Camera::updateVectors()
{
    const float yawRad = glm::radians(yaw);
    const float pitchRad = glm::radians(pitch);

    glm::vec3 direction;
    direction.x = std::cos(yawRad) * std::cos(pitchRad);
    direction.y = std::sin(pitchRad);
    direction.z = std::sin(yawRad) * std::cos(pitchRad);
    front_ = glm::normalize(direction);

    glm::vec3 rightCandidate = glm::cross(front_, worldUp_);
    if (glm::length(rightCandidate) < kEpsilon)
    {
        rightCandidate = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        rightCandidate = glm::normalize(rightCandidate);
    }
    right_ = rightCandidate;
    up_ = glm::normalize(glm::cross(right_, front_));
}
