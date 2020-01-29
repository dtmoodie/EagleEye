#pragma once
#include "aqpointclouds/aqpointclouds_export.hpp"
#include <cereal/cereal.hpp>
#include <ct/reflect.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace point_clouds
{
class aqpointclouds_EXPORT Moment
{
  public:
    Moment(float Px_ = 0.0f, float Py_ = 0.0f, float Pz_ = 0.0f);

    float evaluate(cv::Mat mask, cv::Mat points, cv::Vec3f centroid);
    float Px, Py, Pz;
};
}

namespace ct
{

REFLECT_BEGIN(point_clouds::Moment)
    PUBLIC_ACCESS(Px)
    PUBLIC_ACCESS(Py)
    PUBLIC_ACCESS(Pz)
REFLECT_END;

}

