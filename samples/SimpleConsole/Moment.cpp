#include "Moment.hpp"
#include <MetaObject/params/IO/CerealPolicy.hpp>
#include <MetaObject/params/MetaParameter.hpp>
#include "MetaObject/params/detail/MetaParametersDetail.hpp"
#include <cereal/types/vector.hpp>


using namespace vclick;

INSTANTIATE_META_PARAM(Moment);
INSTANTIATE_META_PARAM(std::vector<Moment>);
Moment::Moment(float Px_, float Py_, float Pz_) :
    Px(Px_), Py(Py_), Pz(Pz_)
{

}

template<typename AR>
void Moment::serialize(AR& ar)
{
    ar(CEREAL_NVP(Px));
    ar(CEREAL_NVP(Py));
    ar(CEREAL_NVP(Pz));
}

float Moment::Evaluate(cv::Mat mask, cv::Mat points, cv::Vec3f centroid)
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
            value += pow(pts[i][0] - centroid[0], Px) * pow(pts[i][1] - centroid[1], Py) * pow(pts[i][2] - centroid[2], Pz);
            ++count;
        }
    }
    value /= count;
    return value;
}
