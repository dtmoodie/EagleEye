#include "hu_moments.h"
#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include <MetaObject/params/MetaParam.hpp>
#include <MetaObject/serialization/CerealPolicy.hpp>
#include <cereal/types/vector.hpp>
#include <ct/reflect/cereal.hpp>

using namespace point_clouds;

INSTANTIATE_META_PARAM(Moment);
INSTANTIATE_META_PARAM(std::vector<Moment>);

Moment::Moment(float Px_, float Py_, float Pz_) : Px(Px_), Py(Py_), Pz(Pz_)
{
}

float Moment::evaluate(cv::Mat mask, cv::Mat points, cv::Vec3f centroid)
{
    float value = 0;
    uchar* mask_ptr = mask.ptr<uchar>();
    cv::Vec3f* pts = points.ptr<cv::Vec3f>();
    const int num_points = mask.size().area();
    float count = 0.0f;
    for (int i = 0; i < num_points; ++i)
    {
        if (mask_ptr[i])
        {
            value +=
                pow(pts[i][0] - centroid[0], Px) * pow(pts[i][1] - centroid[1], Py) * pow(pts[i][2] - centroid[2], Pz);
            ++count;
        }
    }
    value /= count;
    return value;
}
