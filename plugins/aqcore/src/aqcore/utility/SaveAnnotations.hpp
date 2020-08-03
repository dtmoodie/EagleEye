#pragma once

#include "Aquila/types/ObjectDetection.hpp"
#include <Aquila/types/SyncedMemory.hpp>

#include "Aquila/nodes/Node.hpp"

#include <MetaObject/types/file_types.hpp>

namespace aq
{
    namespace nodes
    {
        class SaveAnnotations : public Node
        {
          public:
            MO_DERIVE(SaveAnnotations, Node)
                INPUT(SyncedMemory, input)
                INPUT(CategorySet::ConstPtr, cats)
                INPUT(DetectedObjectSet, detections)

                PARAM(mo::WriteDirectory, output_directory, {})
                PARAM(int, object_class, 8)
                PARAM(std::string, image_stem, "image-")
                PARAM(std::string, annotation_stem, "annotation-")
                PARAM(bool, save_roi, false)
                // TOOLTIP(save_roi, "If set to true, save only a cropped region of the image")
                // PARAM(mo::ReadFile, label_file, {})
                // MO_SLOT(void, click_left, std::string, cv::Point, int, cv::Mat)
                MO_SLOT(void, select_rect, std::string, cv::Rect, int, cv::Mat)
                MO_SLOT(void, on_class_change, int)
                MO_SLOT(void, on_key, int)
                int current_class = -1;
                int save_count = 0;
                // STATUS(int, current_class, -1)
                // STATUS(int, save_count, 0)
            MO_END;
            SaveAnnotations();

          protected:
            bool processImpl();
            void draw();
            DetectedObjectSet _annotations;
            cv::Mat _original_image;
            cv::Mat _draw_image;
            std::shared_ptr<const aq::CategorySet> _cats;
        };
    } // namespace nodes
} // namespace aq
