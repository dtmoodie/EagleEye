#include "IDetector.hpp"

namespace aq
{
namespace nodes
{
std::vector<cv::Rect> IImageDetector::getRegions() const
{
    std::vector<cv::Rect2f> defaultROI;

    if (bounding_boxes)
    {
        defaultROI = *bounding_boxes;
    }
    else
    {
        defaultROI.push_back(cv::Rect2f(0, 0, 1.0, 1.0));
    }

    auto input_image_shape = input->getShape();
    if (input_detections != nullptr)
    {
        defaultROI.clear();
        for (const auto& itr : *input_detections)
        {
            defaultROI.emplace_back(itr.bounding_box.x / input_image_shape[2],
                                    itr.bounding_box.y / input_image_shape[1],
                                    itr.bounding_box.width / input_image_shape[2],
                                    itr.bounding_box.height / input_image_shape[1]);
        }
    }
    std::vector<cv::Rect> pixel_bounding_boxes;
    for (size_t i = 0; i < defaultROI.size(); ++i)
    {
        cv::Rect bb;
        bb.x = static_cast<int>(defaultROI[i].x * input_image_shape[2]);
        bb.y = static_cast<int>(defaultROI[i].y * input_image_shape[1]);
        bb.width = static_cast<int>(defaultROI[i].width * input_image_shape[2]);
        bb.height = static_cast<int>(defaultROI[i].height * input_image_shape[1]);
        if (bb.x + bb.width >= input_image_shape[2])
        {
            bb.x -= input_image_shape[2] - bb.width;
        }
        if (bb.y + bb.height >= input_image_shape[1])
        {
            bb.y -= input_image_shape[1] - bb.height;
        }
        bb.x = std::max(0, bb.x);
        bb.y = std::max(0, bb.y);
        pixel_bounding_boxes.push_back(bb);
    }
    return pixel_bounding_boxes;
}
}
}
