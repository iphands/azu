#pragma once

#include <QOpenGLFunctions_3_3_Core>
#include <string>

namespace kfusion {
namespace rendering {

class ShaderProgram : protected QOpenGLFunctions_3_3_Core {
public:
    ShaderProgram();
    ~ShaderProgram();

    /**
     * Compile + link a fresh program. On ANY failure (compile or link) the
     * previously valid program, if any, is left untouched so a transient reload
     * failure cannot drop rendering to a zero handle mid-session. On success the
     * old program is deleted *before* its handle is overwritten, so a reload
     * never leaks the previous GL program object.
     */
    bool load(const char* vert_src, const char* frag_src);
    void use();
    void disuse();

    // Every setter returns false (and issues no GL call) when there is no bound
    // program, so a caller can tell "set" from "silently dropped on an invalid
    // program" instead of poking glGetUniformLocation(0, ...) (todo 28).
    bool setUniformMat4(const char* name, const float* data);
    bool setUniformMat3(const char* name, const float* data);
    bool setUniformVec3(const char* name, float x, float y, float z);
    bool setUniformFloat(const char* name, float v);
    bool setUniformInt(const char* name, int v);

    unsigned int programId() const { return program_id_; }
    bool isValid() const { return program_id_ != 0; }

private:
    unsigned int program_id_ = 0;

    unsigned int compileShader(unsigned int type, const char* src);
};

} // namespace rendering
} // namespace kfusion
