#pragma once

#include "Aquila/nodes/Node.hpp"
namespace aq
{
namespace Nodes
{
class LegendDisplay: public Node
{
public:
    MO_DERIVE(LegendDisplay, Node)
        INPUT(std::vector<std::string>, labels, nullptr)
        MO_SLOT(void, click_left, std::string, cv::Point, int, cv::Mat)
        MO_SIGNAL(void, on_class_change, int)
        MO_SIGNAL(void, on_class_change, std::string)
    MO_END;
protected:
    bool ProcessImpl();
    cv::Mat h_lut, h_legend;
};
}
}
