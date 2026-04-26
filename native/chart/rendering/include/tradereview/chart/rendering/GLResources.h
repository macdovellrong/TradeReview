#pragma once

#include <QOpenGLFunctions_3_3_Core>

#include <string_view>

namespace tradereview::chart::rendering {

class GLBuffer final {
public:
    void create(QOpenGLFunctions_3_3_Core& gl);
    void destroy(QOpenGLFunctions_3_3_Core& gl);

    [[nodiscard]] GLuint id() const;
    [[nodiscard]] bool valid() const;

private:
    GLuint id_ = 0;
};

class GLVertexArray final {
public:
    void create(QOpenGLFunctions_3_3_Core& gl);
    void destroy(QOpenGLFunctions_3_3_Core& gl);

    [[nodiscard]] GLuint id() const;
    [[nodiscard]] bool valid() const;

private:
    GLuint id_ = 0;
};

class GLProgram final {
public:
    void create(QOpenGLFunctions_3_3_Core& gl, std::string_view vertex_source, std::string_view fragment_source);
    void destroy(QOpenGLFunctions_3_3_Core& gl);

    [[nodiscard]] GLuint id() const;
    [[nodiscard]] bool valid() const;

private:
    GLuint id_ = 0;
};

} // namespace tradereview::chart::rendering
