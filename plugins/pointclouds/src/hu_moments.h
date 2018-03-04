#pragma once
#include "aqpointclouds_export.hpp"
#include <cereal/cereal.hpp>
#include <ct/reflect/reflect_data.hpp>
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
    namespace reflect
    {
        REFLECT_DATA_START(point_clouds::Moment)
            REFLECT_DATA_MEMBER(Px)
            REFLECT_DATA_MEMBER(Py)
            REFLECT_DATA_MEMBER(Pz)
        REFLECT_DATA_END;
    }
}
