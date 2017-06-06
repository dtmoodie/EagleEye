#include "HistogramEqualization.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>

using namespace aq::Nodes;

bool HistogramEqualization::processImpl() {
    cv::cuda::GpuMat              output;
    std::vector<cv::cuda::GpuMat> channels;
    if (!per_channel) {
        cv::cuda::GpuMat hsv;
        cv::cuda::cvtColor(input->getGpuMat(stream()), hsv, cv::COLOR_BGR2HSV, 0, stream());
        cv::cuda::split(hsv, channels, stream());
        cv::cuda::GpuMat equalized;
        cv::cuda::equalizeHist(channels[2], equalized, stream());
        channels[2] = equalized;
        cv::cuda::merge(channels, hsv, stream());
        cv::cuda::cvtColor(hsv, output, cv::COLOR_HSV2BGR, 0, stream());
    } else {
        cv::cuda::split(input->getGpuMat(stream()), channels, stream());
        for (int i = 0; i < channels.size(); ++i) {
            cv::cuda::GpuMat equalized;
            cv::cuda::equalizeHist(channels[i], equalized, stream());
            channels[i] = equalized;
        }
        cv::cuda::merge(channels, output, stream());
    }

    output_param.updateData(output, input_param.getTimestamp(), _ctx.get());
    return true;
}

MO_REGISTER_CLASS(HistogramEqualization)

bool CLAHE::processImpl() {
    if (!_clahe || clip_limit_param.modified() || grid_size_param.modified()) {
        _clahe = cv::cuda::createCLAHE(clip_limit, cv::Size(grid_size, grid_size));
        clip_limit_param.modified(false);
        grid_size_param.modified(false);
    }
    cv::cuda::GpuMat hsv;
    cv::cuda::cvtColor(input->getGpuMat(stream()), hsv, cv::COLOR_BGR2HSV, 0, stream());
    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(hsv, channels, stream());
    _clahe->apply(channels[2], channels[2], stream());
    cv::cuda::GpuMat output;
    cv::cuda::merge(channels, hsv, stream());
    cv::cuda::cvtColor(hsv, output, cv::COLOR_HSV2BGR, 0, stream());
    output_param.updateData(output, mo::tag::_param = input_param, mo::tag::_context = _ctx.get());
    return true;
}

MO_REGISTER_CLASS(CLAHE)
