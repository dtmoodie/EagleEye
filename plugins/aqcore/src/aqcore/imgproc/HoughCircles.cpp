#include "HoughCircles.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <opencv2/imgproc.hpp>
using namespace aq;
using namespace aq::nodes;

bool HoughCircle::processImpl()
{
    cv::Mat host_image = input->getMat(_ctx.get());
    std::vector<cv::Vec3f> circles_;
    std::vector<int> accumulations;
    cv::HoughCircles(host_image,
                     circles_,
                     accumulations,
                     cv::HOUGH_GRADIENT,
                     dp,
                     host_image.rows / 8,
                     upper_threshold,
                     center_threshold,
                     min_radius,
                     max_radius);
    circles.clear();
    if (circles_.size())
    {
        auto minmax = std::minmax_element(accumulations.begin(), accumulations.end());
        float range = (*minmax.second - *minmax.first);
        if (range == 0.0f)
        {
            range = 1.0f;
        }
        for (size_t i = 0; i < circles_.size(); ++i)
        {
            circles.emplace_back(
                (accumulations[i] - *minmax.first) / range, i, circles_[i][0], circles_[i][1], circles_[i][2]);
        }
    }
    circles_param.emitUpdate(input_param);
    // circles_param.updateData(circles, mo::tag::_param = input_param);

    if (drawn_circles_param.hasSubscriptions())
    {
        cv::Mat draw_image = host_image.clone();
        for (size_t i = 0; i < circles.size(); i++)
        {
            cv::Point center(cvRound(circles[i].origin.x()), cvRound(circles[i].origin.y()));
            int radius = cvRound(circles[i].radius);
            // circle center
            cv::circle(draw_image, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
            // circle outline
            cv::circle(draw_image, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
        }
        drawn_circles_param.updateData(draw_image, mo::tag::_param = input_param);
    }
    return true;
}

MO_REGISTER_CLASS(HoughCircle)
