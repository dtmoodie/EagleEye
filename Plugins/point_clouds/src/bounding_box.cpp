#include "bounding_box.h"
#include <opencv2/calib3d.hpp>

using namespace EagleLib;
using namespace EagleLib::point_clouds;

namespace detail
{
    void contains_kernel(const cv::cuda::GpuMat& points, cv::cuda::GpuMat& mask, cv::Mat& T, cv::Scalar& size, cv::cuda::Stream& stream);
}


bounding_box::bounding_box(cv::Mat transform, cv::Scalar size):
    _size(size)
{
    transform.convertTo(_transform, CV_32F);
    if(_transform.rows == _transform.cols && _transform.rows == 4)
    {
        _inv_transform = _transform.inv();
    }
    if(_transform.cols == 1 && (_transform.rows == 4 || _transform.rows == 3))
    {
        
    }
    if(_transform.rows == 1 && (_transform.cols == 4 || _transform.cols == 3))
    {
        
    }
}
bool bounding_box::contains(const cv::Vec3f& point)
{
    return false;
}

cv::Mat bounding_box::contains(const cv::Mat& points)
{
    return cv::Mat();
}

cv::cuda::GpuMat bounding_box::contains(const cv::cuda::GpuMat& points, cv::cuda::Stream& stream)
{
    return cv::cuda::GpuMat();
}

void bounding_box::contains(cv::InputArray points, cv::InputOutputArray mask, cv::cuda::Stream& stream)
{
    CV_Assert(points.type() == mask.type());
    CV_Assert(points.depth() == CV_32F);
    if(points.type() == cv::_InputArray::CUDA_GPU_MAT)
    {
        //detail::contains_kernel(points.getGpuMat(), mask.getGpuMatRef(), _transform, _size, stream);
    }else
    {
        cv::Mat _points = points.getMat();
        cv::Mat& _mask = mask.getMatRef();
        _points = _points.reshape(1);
        cv::transform(_points, _points, _inv_transform);
        cv::inRange(_points, cv::Scalar::all(0), _size, mask);
    }
}