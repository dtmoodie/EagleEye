#include "LegendDisplay.hpp"
#include "Aquila/gui/UiCallbackHandlers.h"
#include "Aquila/core/IDataStream.hpp"
#include "Aquila/utilities/cuda/CudaCallbacks.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <boost/lexical_cast.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
using namespace aq;
using namespace aq::Nodes;

void LegendDisplay::click_left(std::string window_name, cv::Point pt, int, cv::Mat)
{
    if(window_name == "legend")
    {
        //int idx = ((pt.y - 25) / 20) + 1;
        int idx = (pt.y - 5) / 20;

        sig_on_class_change(idx);
    }
}

bool LegendDisplay::processImpl()
{
    h_lut.create(1, labels->size(), CV_8UC3);
    for(int i = 0; i < labels->size(); ++i)
        h_lut.at<cv::Vec3b>(i) = cv::Vec3b(i*180 / labels->size(), 200, 255);
    cv::cvtColor(h_lut, h_lut, cv::COLOR_HSV2BGR);

    int legend_width = 100;
    int max_width = 0;
    for(const auto& label : *labels)
    {
        max_width = std::max<int>(max_width, label.size());
    }
    legend_width += max_width * 15;

    h_legend.create(labels->size() * 20 + 15, legend_width, CV_8UC3);
    h_legend.setTo(0);
    for(int i = 0; i < labels->size(); ++i)
    {
        cv::Vec3b color = h_lut.at<cv::Vec3b>(i);
        h_legend(cv::Rect(8, 5 + 20 * i, 50, 20)).setTo(color);

        cv::putText(h_legend, boost::lexical_cast<std::string>(i) + " " + (*labels)[i],
                    cv::Point(65, 25 + 20 * i),
                    cv::FONT_HERSHEY_COMPLEX, 0.7,
                    cv::Scalar(color[0], color[1], color[2]));
    }
    getDataStream()->getWindowCallbackManager()->imshow("legend", h_legend);
    return true;
}


MO_REGISTER_CLASS(LegendDisplay)
