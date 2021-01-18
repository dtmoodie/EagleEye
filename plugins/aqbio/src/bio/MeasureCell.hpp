#pragma once
#include "FindCellMembrane.hpp"
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedImage.hpp>

namespace aqbio
{

    class aqbio_EXPORT MeasureCell : public aq::nodes::Node
    {
      public:
        MO_DERIVE(MeasureCell, aq::nodes::Node)
            INPUT(Cell, cell)
            INPUT(std::string, image_name)
            INPUT(aq::SyncedImage, image)

            PARAM(float, stddev_thresh, 2.0f)
            PARAM(mo::WriteDirectory, out_dir, {})
            PARAM(float, pixel_per_um, 2.11f)

            OUTPUT(aq::SyncedImage, overlay, {})
        MO_END;

      protected:
        bool processImpl() override;

        struct Measurement
        {
            cv::Point outer0;
            cv::Point outer1;
            cv::Point inner0;
            cv::Point inner1;
            float inner_radius = 0;
            float outer_radius = 0;
            float inner_diameter = 0;
            float outer_diameter = 0;
            float diff = 0;
        };
    };
}
