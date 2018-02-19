#include "flip.hpp"
#include "Aquila/nodes/NodeInfo.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

using namespace aq::nodes;
bool Flip::processImpl()
{
    auto state = input->getSyncState();
    if (state == input->DEVICE_UPDATED)
    {
        cv::cuda::GpuMat output;
        cv::cuda::flip(input->getGpuMat(stream()), output, axis.getValue(), stream());
        output_param.updateData(output, input_param.getTimestamp(), _ctx.get());
        return true;
    }
    else
    {
        if (state == input->HOST_UPDATED)
        {
            cv::Mat flipped;
            cv::flip(input->getMat(stream()), flipped, axis.getValue());
            output_param.updateData(flipped, input_param.getTimestamp(), _ctx.get());
            return true;
        }
        else if (state == input->SYNCED)
        {
            cv::Mat h_flipped;
            cv::flip(input->getMat(stream()), h_flipped, axis.getValue());
            cv::cuda::GpuMat d_flipped;
            cv::cuda::flip(input->getGpuMat(stream()), d_flipped, axis.getValue(), stream());
            output_param.updateData({h_flipped, d_flipped}, input_param.getTimestamp(), _ctx.get());
            return true;
        }
    }
    return false;
}

MO_REGISTER_CLASS(Flip)

bool Rotate::processImpl()
{
    cv::cuda::GpuMat rotated;
    auto size = input->getSize();
    cv::Mat rotation = cv::getRotationMatrix2D({size.width / 2.0f, size.height / 2.0f}, angle_degrees, 1.0);
    cv::cuda::warpAffine(input->getGpuMat(stream()),
                         rotated,
                         rotation,
                         size,
                         cv::INTER_CUBIC,
                         cv::BORDER_REFLECT,
                         cv::Scalar(),
                         stream());
    output_param.updateData(rotated, input_param.getTimestamp(), _ctx.get());
    return true;
}

MO_REGISTER_CLASS(Rotate)
