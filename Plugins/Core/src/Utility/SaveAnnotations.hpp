#pragma once
#include "Aquila/Nodes/Node.h"
#include "Aquila/ObjectDetection.hpp"

namespace aq
{
namespace Nodes
{
    class SaveAnnotations: public Node
    {
    public:
        MO_DERIVE(SaveAnnotations, Node)
            INPUT(SyncedMemory, input, nullptr)
            INPUT(std::vector<std::string>, labels, nullptr)
            OPTIONAL_INPUT(std::vector<DetectedObject>, detections, nullptr)

            PARAM(mo::WriteDirectory, output_directory, {})
            PARAM(int, object_class, 8)
            PARAM(std::string, image_stem, "image-")
            PARAM(std::string, annotation_stem, "annotation-")
            PARAM(bool, save_roi, false)
            TOOLTIP(save_roi, "If set to true, save only a cropped region of the image")
            //PARAM(mo::ReadFile, label_file, {})
            //MO_SLOT(void, click_left, std::string, cv::Point, int, cv::Mat)
            MO_SLOT(void, select_rect, std::string, cv::Rect, int, cv::Mat)
            MO_SLOT(void, on_class_change, int)
            MO_SLOT(void, on_key, int)
            STATUS(int, current_class, -1)
            STATUS(int, save_count, 0)
        MO_END
        SaveAnnotations();
    protected:
        bool ProcessImpl();
        void draw();
        //std::vector<std::string> _labels;
        //cv::Mat h_legend;
        std::vector<DetectedObject> _annotations;
        cv::Mat _original_image;
        cv::Mat _draw_image;
        cv::Mat h_lut;
    };
}
}
