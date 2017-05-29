#include "FFT.h"



using namespace aq;
using namespace aq::Nodes;

bool FFT::processImpl()
{
    cv::cuda::GpuMat padded;
    if(input->getChannels() > 2)
    {
        LOG(debug) << "Too many channels, can only handle 1 or 2 channel input. Input has " << input->getChannels() << " channels.";
        return false;
    }
    if(use_optimized_size)
    {
        int in_rows = input->getSize().height;
        int in_cols = input->getSize().width;
        int rows = cv::getOptimalDFTSize(in_rows);
        int cols = cv::getOptimalDFTSize(in_cols);
        cv::cuda::copyMakeBorder(input->getGpuMat(stream()), padded, 0, rows - in_rows, 0, cols - in_cols, cv::BORDER_CONSTANT, cv::Scalar::all(0), stream());
    }else
    {
        padded = input->getGpuMat(stream());
    }
    if(padded.depth() != CV_32F)
    {
        cv::cuda::GpuMat float_img;
        padded.convertTo(float_img, CV_MAKETYPE(CV_32F, padded.channels()), stream());
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
    cv::cuda::dft(padded, result, input->getSize(),flags, stream());
    coefficients_param.updateData(result, input_param.getTimestamp(), _ctx.get());
    if(magnitude_param.hasSubscriptions())
    {
        cv::cuda::GpuMat magnitude;
        cv::cuda::magnitude(result,magnitude, stream());
        
        if(log_scale)
        {
            cv::cuda::add(magnitude, cv::Scalar::all(1), magnitude, cv::noArray(), -1, stream());
            cv::cuda::log(magnitude, magnitude, stream());
        }
        this->magnitude_param.updateData(magnitude, input_param.getTimestamp(), _ctx.get());
    }
    if(phase_param.hasSubscriptions())
    {
        std::vector<cv::cuda::GpuMat> channels;
        cv::cuda::split(result, channels, stream());
        cv::cuda::GpuMat phase;
        cv::cuda::phase(channels[0], channels[1], phase, false, stream());
        this->phase_param.updateData(phase, input_param.getTimestamp(), _ctx.get());
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

bool FFTPreShiftImage::processImpl()
{
    if (d_shiftMat.size() != input->getSize())
    {
        d_shiftMat.upload(getShiftMat(input->getSize()), stream());
    }
    cv::cuda::GpuMat result;
    cv::cuda::multiply(d_shiftMat, input->getGpuMat(stream()), result, 1, -1, stream());
    output_param.updateData(result, input_param.getTimestamp(), _ctx.get());
    return true;
}

bool FFTPostShift::processImpl()
{
    if (d_shiftMat.size() != input->getSize())
    {
        d_shiftMat.upload(getShiftMat(input->getSize()), stream());
        std::vector<cv::cuda::GpuMat> channels;
        channels.push_back(d_shiftMat);
        channels.push_back(d_shiftMat);
        cv::cuda::merge(channels, d_shiftMat, stream());
    }
    cv::cuda::GpuMat result;
    cv::cuda::multiply(d_shiftMat, input->getGpuMat(stream()), result, 1 / float(input->getSize().area()), -1, stream());
    return true;
}

MO_REGISTER_CLASS(FFT);
MO_REGISTER_CLASS(FFTPreShiftImage);
MO_REGISTER_CLASS(FFTPostShift);
