#pragma once

#include "ParameteredObject.h"

#include <opencv2/core/cuda.hpp>
class EAGLE_EXPORTS Algorithm : public TInterface<IID_Algorithm, EagleLib::ParameteredObject>
{
public:
    virtual void initialize(cv::cuda::Stream& stream) = 0;
    virtual void pre_process(cv::cuda::Stream& stream) = 0;
    virtual void process(cv::cuda::Stream& stream) = 0;
    virtual void post_process(cv::cuda::Stream& stream) = 0;
};