#pragma once
#include <ct/reflect/reflect_data.hpp>
#include "aqpointclouds_export.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <cereal/cereal.hpp>

namespace point_clouds
{

class aqpointclouds_EXPORT Moment
{
public:
    Moment(float Px_ = 0.0f, float Py_ = 0.0f, float Pz_ = 0.0f);

    float evaluate(cv::Mat mask, cv::Mat points, cv::Vec3f centroid);
    REFLECT_INTERNAL_START(Moment)
        REFLECT_INTERNAL_MEMBER(float, Px)
        REFLECT_INTERNAL_MEMBER(float, Py)
        REFLECT_INTERNAL_MEMBER(float, Pz)
    REFLECT_INTERNAL_END;
};


}
