#include "FFT.h"



using namespace EagleLib;
using namespace EagleLib::Nodes;

bool FFT::ProcessImpl()
{
    cv::cuda::GpuMat padded;
    if(input->GetChannels() > 2)
    {
        LOG(debug) << "Too many channels, can only handle 1 or 2 channel input. Input has " << input->GetChannels() << " channels.";
        return false;
    }
    if(use_optimized_size)
    {
        int in_rows = input->GetSize().height;
        int in_cols = input->GetSize().width;
        int rows = cv::getOptimalDFTSize(in_rows);
        int cols = cv::getOptimalDFTSize(in_cols);
        cv::cuda::copyMakeBorder(input->GetGpuMat(*_ctx->stream), padded, 0, rows - in_rows, 0, cols - in_cols, cv::BORDER_CONSTANT, cv::Scalar::all(0), *_ctx->stream);
    }else
    {
        padded = input->GetGpuMat(*_ctx->stream);
    }
    if(padded.depth() != CV_32F)
    {
        cv::cuda::GpuMat float_img;
        padded.convertTo(float_img, CV_MAKETYPE(CV_32F, padded.channels()), *_ctx->stream);
        padded = float_img;
    }
    int flags = 0;
    if (dft_rows)
        flags = flags | cv::DFT_ROWS;
    if (dft_scale)
        flags = flags | cv::DFT_SCALE;
    if (dft_inverse)
        flags = flags | cv::DFT_INVERSE;
    if (dft_real_output)
        flags = flags | cv::DFT_REAL_OUTPUT;
    cv::cuda::GpuMat result;
    cv::cuda::dft(padded, result, input->GetSize(),flags, *_ctx->stream);
    coefficients_param.UpdateData(result, input_param.GetTimestamp(), _ctx);
    if(magnitude_param.HasSubscriptions())
    {
        cv::cuda::GpuMat magnitude;
        cv::cuda::magnitude(result,magnitude, *_ctx->stream);
        
        if(log_scale)
        {
            cv::cuda::add(magnitude, cv::Scalar::all(1), magnitude, cv::noArray(), -1, *_ctx->stream);
            cv::cuda::log(magnitude, magnitude, *_ctx->stream);
        }
        this->magnitude_param.UpdateData(magnitude, input_param.GetTimestamp(), _ctx);
    }
    if(phase_param.HasSubscriptions())
    {
        std::vector<cv::cuda::GpuMat> channels;
        cv::cuda::split(result, channels, Stream());
        cv::cuda::GpuMat phase;
        cv::cuda::phase(channels[0], channels[1], phase, false, Stream());
        this->phase_param.UpdateData(phase, input_param.GetTimestamp(), _ctx);
    }
    return true;
}

cv::Mat getShiftMat(cv::Size matSize)
{
    cv::Mat shift(matSize, CV_32F);
    for(int y = 0; y < matSize.height; ++y)
    {
        for(int x = 0; x < matSize.width; ++x)
        {
            shift.at<float>(y,x) = 1.0 - 2.0 * ((x+y)&1);
        }
    }

    return shift;
}

bool FFTPreShiftImage::ProcessImpl()
{
    if (d_shiftMat.size() != input->GetSize())
    {
        d_shiftMat.upload(getShiftMat(input->GetSize()), Stream());
    }
    cv::cuda::GpuMat result;
    cv::cuda::multiply(d_shiftMat, input->GetGpuMat(Stream()), result, 1, -1, Stream());
    output_param.UpdateData(result, input_param.GetTimestamp(), _ctx);
    return true;
}

bool FFTPostShift::ProcessImpl()
{
    if (d_shiftMat.size() != input->GetSize())
    {
        d_shiftMat.upload(getShiftMat(input->GetSize()), Stream());
        std::vector<cv::cuda::GpuMat> channels;
        channels.push_back(d_shiftMat);
        channels.push_back(d_shiftMat);
        cv::cuda::merge(channels, d_shiftMat, Stream());
    }
    cv::cuda::GpuMat result;
    cv::cuda::multiply(d_shiftMat, input->GetGpuMat(Stream()), result, 1 / float(input->GetSize().area()), -1, Stream());
    return true;
}

MO_REGISTER_CLASS(FFT);
MO_REGISTER_CLASS(FFTPreShiftImage);
MO_REGISTER_CLASS(FFTPostShift);
