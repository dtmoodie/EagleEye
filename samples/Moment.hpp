#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
namespace vclick
{
    class Moment
    {
    public:
        Moment(float Px_ = 0.0f, float Py_ = 0.0f, float Pz_ = 0.0f);
        template<typename AR> void serialize(AR& ar);
        float Evaluate(cv::Mat mask, cv::Mat points, cv::Vec3f centroid);
        float Px, Py, Pz;
    };
}