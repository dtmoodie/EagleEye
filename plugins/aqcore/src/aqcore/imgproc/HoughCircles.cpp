#include <ct/types/opencv.hpp>

#include "HoughCircles.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <opencv2/imgproc.hpp>

namespace aqcore
{

    bool HoughCircle::processImpl()
    {
        mo::IAsyncStreamPtr_t stream = this->getStream();
        const cv::Mat host_image = input->mat(stream.get());
        std::vector<cv::Vec3f> detected_circles;
        std::vector<int> accumulations;
        cv::HoughCircles(host_image,
                         detected_circles,
                         // accumulations,
                         cv::HOUGH_GRADIENT,
                         dp,
                         host_image.rows / 8,
                         upper_threshold,
                         center_threshold,
                         min_radius,
                         max_radius);
        Output_t output;

        if (detected_circles.size())
        {
            output.resize(detected_circles.size());
            auto minmax = std::minmax_element(accumulations.begin(), accumulations.end());
            float range = (*minmax.second - *minmax.first);
            if (range == 0.0f)
            {
                range = 1.0f;
            }
            for (size_t i = 0; i < detected_circles.size(); ++i)
            {
                aq::detection::Confidence conf = 1.0;
                if (accumulations.size() == detected_circles.size())
                {
                    conf = (accumulations[i] - *minmax.first) / range;
                }
                const cv::Vec3f data = detected_circles[i];
                const float center_x = data[0];
                const float center_y = data[1];
                const float radius = data[2];
                aq::Circlef circle(center_x, center_y, radius);
                output.push_back(std::move(circle), std::move(conf));
            }
        }
        this->output.publish(std::move(output));
        // circles_param.updateData(circles, mo::tag::_param = input_param);

        /*if (drawn_circles_param.hasSubscriptions())
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
        }*/
        return true;
    }
} // namespace aqcore

using namespace aqcore;
MO_REGISTER_CLASS(HoughCircle)
