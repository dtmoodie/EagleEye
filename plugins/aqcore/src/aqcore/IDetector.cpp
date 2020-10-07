#include <ct/types/opencv.hpp>

#include "IDetector.hpp"

namespace aqcore
{
    std::vector<cv::Rect> IImageDetector::getRegions() const
    {
        std::vector<cv::Rect2f> defaultROI;

        if (regions_of_interest)
        {
            defaultROI = *regions_of_interest;
        }
        else
        {
            defaultROI.push_back(cv::Rect2f(0, 0, 1.0, 1.0));
        }

        auto input_image_shape = input->shape();
        if (input_detections != nullptr)
        {
            defaultROI.clear();
            auto bbs = input_detections->getComponent<aq::detection::BoundingBox2d>();
            for (const auto& itr : bbs)
            {
                defaultROI.emplace_back(itr.x / input_image_shape[2],
                                        itr.y / input_image_shape[1],
                                        itr.width / input_image_shape[2],
                                        itr.height / input_image_shape[1]);
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

} // namespace aqcore
