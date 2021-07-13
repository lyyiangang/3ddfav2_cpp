#include "face3d_mnn.hpp"
#include <iostream>

Face3d::Face3d(const std::string& model_path){
    float means[3] = {127.5, 127.5, 127.5};
    float std[3] = {1/127.5, 1/127.5, 1/127.5};
    _preprocess.reset(
        MNN::CV::ImageProcess::create(MNN::CV::ImageFormat::RGB,
        MNN::CV::BGR, \
        means, \
        3, \
        std, \
        3)
    );
    _interpreter.reset(MNN::Interpreter::createFromFile(model_path.c_str()));
    MNN::ScheduleConfig cfg;
    _session = _interpreter->createSession(cfg);
}

Face3d::~Face3d(){
}

std::vector<float> Face3d::Predict(const uint8_t* buffer_120x120x3){
    MNN::Tensor* input_ts = _interpreter->getSessionInput(_session, "input");
    _preprocess->convert(buffer_120x120x3, 120, 120, 0, input_ts);
    _interpreter->runSession(_session); 
    MNN::Tensor* lnd_output_ts = _interpreter->getSessionOutput(_session, "output");
    MNN::Tensor lnd_host(lnd_output_ts, MNN::Tensor::CAFFE);
    lnd_output_ts->copyToHostTensor(&lnd_host);
    std::vector<float> output_landmarks(lnd_host.elementSize(), 0);
    std::copy(lnd_host.host<float>(), lnd_host.host<float>() + lnd_host.elementSize(), output_landmarks.data());
    return output_landmarks;
}
