#include "rendering/ShaderProgram.h"
#include <iostream>
#include <string>
#include <vector>
#include "utils/Logger.h"

namespace kfusion {
namespace rendering {

ShaderProgram::ShaderProgram() {
    initializeOpenGLFunctions();
}

ShaderProgram::~ShaderProgram() {
    if (program_id_) {
        glDeleteProgram(program_id_);
        program_id_ = 0;
    }
}

unsigned int ShaderProgram::compileShader(unsigned int type, const char* src) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        int log_len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
        std::vector<char> log(log_len > 0 ? log_len : 0);
        std::string log_text;
        if (log_len > 0) {
            glGetShaderInfoLog(shader, log_len, nullptr, log.data());
            log_text = log.data();
        }
        // Route through the project logger so a headless/offscreen run captures
        // the compile failure (previously a bare std::cerr a redirected run lost).
        KFLOG_ERROR("shader", "compile failed: " + log_text);
        std::cerr << "[Shader] Compile error: " << log_text << "\n";
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

bool ShaderProgram::load(const char* vert_src, const char* frag_src) {
    initializeOpenGLFunctions();

    // Compile first. Until a new program links, program_id_ keeps whatever valid
    // program it already had, so a failed reload can never leave a zero handle.
    unsigned int vert = compileShader(GL_VERTEX_SHADER,   vert_src);
    unsigned int frag = compileShader(GL_FRAGMENT_SHADER, frag_src);
    if (!vert || !frag) {
        if (vert) glDeleteShader(vert);
        if (frag) glDeleteShader(frag);
        return false;
    }

    unsigned int program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    // Attached shaders are released once linked (or not), regardless of outcome.
    glDeleteShader(vert);
    glDeleteShader(frag);

    int success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        int log_len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
        std::vector<char> log(log_len > 0 ? log_len : 0);
        std::string log_text;
        if (log_len > 0) {
            glGetProgramInfoLog(program, log_len, nullptr, log.data());
            log_text = log.data();
        }
        KFLOG_ERROR("shader", "link failed: " + log_text);
        std::cerr << "[Shader] Link error: " << log_text << "\n";
        glDeleteProgram(program);
        return false;
    }

    // The new program is valid: release the previous one BEFORE its handle is
    // overwritten, so repeated load() calls cannot leak a GL program object.
    if (program_id_) {
        glDeleteProgram(program_id_);
    }
    program_id_ = program;
    return true;
}

void ShaderProgram::use()   { if (program_id_) glUseProgram(program_id_); }
void ShaderProgram::disuse() { glUseProgram(0); }

bool ShaderProgram::setUniformMat4(const char* name, const float* data) {
    if (program_id_ == 0) return false;
    int loc = glGetUniformLocation(program_id_, name);
    if (loc < 0) return false;
    glUniformMatrix4fv(loc, 1, GL_FALSE, data);
    return true;
}

bool ShaderProgram::setUniformMat3(const char* name, const float* data) {
    if (program_id_ == 0) return false;
    int loc = glGetUniformLocation(program_id_, name);
    if (loc < 0) return false;
    glUniformMatrix3fv(loc, 1, GL_FALSE, data);
    return true;
}

bool ShaderProgram::setUniformVec3(const char* name, float x, float y, float z) {
    if (program_id_ == 0) return false;
    int loc = glGetUniformLocation(program_id_, name);
    if (loc < 0) return false;
    glUniform3f(loc, x, y, z);
    return true;
}

bool ShaderProgram::setUniformFloat(const char* name, float v) {
    if (program_id_ == 0) return false;
    int loc = glGetUniformLocation(program_id_, name);
    if (loc < 0) return false;
    glUniform1f(loc, v);
    return true;
}

bool ShaderProgram::setUniformInt(const char* name, int v) {
    if (program_id_ == 0) return false;
    int loc = glGetUniformLocation(program_id_, name);
    if (loc < 0) return false;
    glUniform1i(loc, v);
    return true;
}

} // namespace rendering
} // namespace kfusion
