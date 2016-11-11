#include "BoundingBox.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"
#include "MetaObject/Parameters/MetaParameter.hpp"
#include <cereal/types/vector.hpp>
using namespace vclick;

INSTANTIATE_META_PARAMETER(BoundingBox);
INSTANTIATE_META_PARAMETER(std::vector<BoundingBox>);

cv::Mat BoundingBox::Contains(std::vector<cv::Vec3f>& points)
{
    return Contains(cv::Mat(1, points.size(), CV_32FC3, &points[0]));
}

cv::Mat BoundingBox::Contains(cv::Mat points)
{
    cv::Mat output_mask;
    output_mask.create(points.size(), CV_8UC1);
    const int num_points = points.size().area();
    uchar* mask_ptr = output_mask.ptr<uchar>();
    cv::Vec3f* pt_ptr = points.ptr<cv::Vec3f>();
    for (int i = 0; i < num_points; ++i)
    {
        const cv::Vec3f& pt = pt_ptr[i];
        mask_ptr[i] = Contains(pt);
    }
    return output_mask;
}

template<typename AR>
void BoundingBox::serialize(AR& ar)
{
    ar(CEREAL_NVP(x));
    ar(CEREAL_NVP(y));
    ar(CEREAL_NVP(z));
    ar(CEREAL_NVP(width));
    ar(CEREAL_NVP(height));
    ar(CEREAL_NVP(depth));
}