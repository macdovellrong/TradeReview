#include "tradereview/chart/rendering/GLResources.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tradereview::chart::rendering {
namespace {

GLuint compile_shader(QOpenGLFunctions_3_3_Core& gl, GLenum type, std::string_view source)
{
    const GLuint shader = gl.glCreateShader(type);
    const auto* raw_source = source.data();
    const GLint length = static_cast<GLint>(source.size());
    gl.glShaderSource(shader, 1, &raw_source, &length);
    gl.glCompileShader(shader);

    GLint ok = GL_FALSE;
    gl.glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok == GL_TRUE) {
        return shader;
    }

    GLint log_length = 0;
    gl.glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    std::string log(static_cast<std::size_t>(log_length), '\0');
    if (log_length > 0) {
        gl.glGetShaderInfoLog(shader, log_length, nullptr, log.data());
    }
    gl.glDeleteShader(shader);
    throw std::runtime_error("OpenGL shader compile failed: " + log);
}

} // namespace

void GLBuffer::create(QOpenGLFunctions_3_3_Core& gl)
{
    if (id_ == 0) {
        gl.glGenBuffers(1, &id_);
    }
}

void GLBuffer::destroy(QOpenGLFunctions_3_3_Core& gl)
{
    if (id_ != 0) {
        gl.glDeleteBuffers(1, &id_);
        id_ = 0;
    }
}

GLuint GLBuffer::id() const
{
    return id_;
}

bool GLBuffer::valid() const
{
    return id_ != 0;
}

void GLVertexArray::create(QOpenGLFunctions_3_3_Core& gl)
{
    if (id_ == 0) {
        gl.glGenVertexArrays(1, &id_);
    }
}

void GLVertexArray::destroy(QOpenGLFunctions_3_3_Core& gl)
{
    if (id_ != 0) {
        gl.glDeleteVertexArrays(1, &id_);
        id_ = 0;
    }
}

GLuint GLVertexArray::id() const
{
    return id_;
}

bool GLVertexArray::valid() const
{
    return id_ != 0;
}

void GLProgram::create(QOpenGLFunctions_3_3_Core& gl, std::string_view vertex_source, std::string_view fragment_source)
{
    if (id_ != 0) {
        return;
    }

    const GLuint vertex_shader = compile_shader(gl, GL_VERTEX_SHADER, vertex_source);
    GLuint fragment_shader = 0;
    try {
        fragment_shader = compile_shader(gl, GL_FRAGMENT_SHADER, fragment_source);
    } catch (...) {
        gl.glDeleteShader(vertex_shader);
        throw;
    }
    const GLuint program = gl.glCreateProgram();
    gl.glAttachShader(program, vertex_shader);
    gl.glAttachShader(program, fragment_shader);
    gl.glLinkProgram(program);
    gl.glDeleteShader(vertex_shader);
    gl.glDeleteShader(fragment_shader);

    GLint ok = GL_FALSE;
    gl.glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok == GL_TRUE) {
        id_ = program;
        return;
    }

    GLint log_length = 0;
    gl.glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
    std::string log(static_cast<std::size_t>(log_length), '\0');
    if (log_length > 0) {
        gl.glGetProgramInfoLog(program, log_length, nullptr, log.data());
    }
    gl.glDeleteProgram(program);
    throw std::runtime_error("OpenGL program link failed: " + log);
}

void GLProgram::destroy(QOpenGLFunctions_3_3_Core& gl)
{
    if (id_ != 0) {
        gl.glDeleteProgram(id_);
        id_ = 0;
    }
}

GLuint GLProgram::id() const
{
    return id_;
}

bool GLProgram::valid() const
{
    return id_ != 0;
}

} // namespace tradereview::chart::rendering
