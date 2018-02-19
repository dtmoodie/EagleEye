#include "DrawRegionsOfInterest.hpp"
#include "Aquila/nodes/NodeInfo.hpp"
#include "Aquila/utilities/GpuDrawing.hpp"

using namespace aq::nodes;

bool DrawRegionsOfInterest::processImpl()
{
    cv::cuda::GpuMat draw_image;
    image->clone(draw_image, stream());
    auto image_size = image->getSize();
    for (const auto& roi : *bounding_boxes)
    {
        cv::Rect pixelRect;
        pixelRect.x = static_cast<int>(roi.x * image_size.width);
        pixelRect.y = static_cast<int>(roi.y * image_size.height);
        pixelRect.width = static_cast<int>(roi.width * image_size.width);
        pixelRect.height = static_cast<int>(roi.height * image_size.height);
        cv::cuda::rectangle(draw_image, pixelRect, cv::Scalar(0, 128, 0), 2, stream());
    }
    output_param.updateData(draw_image, image_param.getTimestamp(), _ctx.get());
    return true;
}

MO_REGISTER_CLASS(DrawRegionsOfInterest)
