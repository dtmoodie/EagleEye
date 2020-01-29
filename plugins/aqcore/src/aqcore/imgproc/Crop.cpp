#include "Crop.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/types/ObjectDetection.hpp>
using namespace aq::nodes;

bool Crop::processImpl()
{
    const cv::Size size = input->getSize();
    cv::Rect2f bb = roi;
    boundingBoxToPixels(bb, size);

    if (input->getSyncState() < input->DEVICE_UPDATED)
    {
        auto ROI = input->getMat(stream())(bb);
        output_param.updateData(ROI.clone(), mo::tag::_param = input_param);
    }
    else
    {
        cv::cuda::GpuMat out;
        auto ROI = input->getGpuMat(stream())(bb);
        ROI.copyTo(out, stream());
        output_param.updateData(out, mo::tag::_param = input_param);
    }
    return true;
}

MO_REGISTER_CLASS(Crop)
