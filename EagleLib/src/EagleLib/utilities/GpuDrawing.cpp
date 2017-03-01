#include "EagleLib/utilities/GpuDrawing.hpp"

void cv::cuda::rectangle(cv::cuda::GpuMat& img, const cv::Rect& rect, cv::Scalar color, int thickness, cv::cuda::Stream& stream)
{
    cv::Rect top(rect.tl(), cv::Size(rect.width, thickness));
    img(top).setTo(color, stream);

    top.y += rect.height - thickness;
    img(top).setTo(color, stream);

    top.width = thickness;
    top.height = rect.height;
    top.x = rect.x;
    top.y = rect.y;
    img(top).setTo(color, stream);

    top.x = rect.x + rect.width - thickness;
    img(top).setTo(color, stream);
}
