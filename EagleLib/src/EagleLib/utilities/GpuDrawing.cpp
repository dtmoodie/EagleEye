#include "EagleLib/utilities/GpuDrawing.hpp"
#include <opencv2/core/cuda.hpp>

void cv::cuda::rectangle(cv::cuda::GpuMat& img, const cv::Rect& rect_, cv::Scalar color, int thickness, cv::cuda::Stream& stream)
{
    cv::Rect img_rect({0,0}, img.size());
    cv::Rect rect = rect_ & img_rect;
    if(rect.size().area() == 0)
        return;

    cv::Rect top(rect.tl(), cv::Size(rect.width, thickness));
    img(top & img_rect).setTo(color, stream);

    top.y += rect.height - thickness;
    img(top & img_rect).setTo(color, stream);

    top.width = thickness;
    top.height = rect.height;
    top.x = rect.x;
    top.y = rect.y;
    img(top & img_rect).setTo(color, stream);

    top.x = rect.x + rect.width - thickness;
    img(top & img_rect).setTo(color, stream);
}
