#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda_types.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_traits.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>

namespace aq
{
    class InputArray : public cv::InputArray
    {
    };

    class PtCloud2Mat
    {
    };

    class Mat2PtCloud
    {
    };
}
