#pragma once
#include "FindCellMembrane.hpp"
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>

namespace aq
{
namespace bio
{

class aqbio_EXPORT MeasureCell : public nodes::Node
{
  public:
    MO_DERIVE(MeasureCell, nodes::Node)
        INPUT(Cell, cell, nullptr)
        INPUT(std::string, image_name, nullptr)
        INPUT(aq::SyncedMemory, image, nullptr)
        PARAM(float, stddev_thresh, 2.0f)
        PARAM(mo::WriteDirectory, out_dir, {})
        PARAM(float, pixel_per_um, 2.11f)
        OUTPUT(aq::SyncedMemory, overlay, {})
    MO_END

  protected:
    virtual bool processImpl() override;

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
}
