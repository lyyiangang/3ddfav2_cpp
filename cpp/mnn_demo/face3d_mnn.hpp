#ifndef FACE3D_MNN_HPP
#define FACE3D_MNN_HPP

#include <string>
#include <vector>
#include <memory>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>

class Face3d
{
    public:
        Face3d(const std::string& model_path);
        ~Face3d();

        std::vector<float> Predict(const uint8_t* buffer_120x120x3);

    private:
        std::unique_ptr<MNN::CV::ImageProcess> _preprocess;
        std::unique_ptr<MNN::Interpreter> _interpreter;
        MNN::Session* _session;
};
#endif