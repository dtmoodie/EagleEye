#include "../Defs.hpp"
#include <memory>
namespace EagleLib
{
    class EAGLE_EXPORTS IOBufferPad
    {
        typedef std::shared_ptr<IOBufferPad> Ptr;
        enum DataType
        {
            Image,
            PointCloud,
            Tensor
        };
        DataType GetDataType() const = 0;
        int GetNumSamples() const = 0;
        cv::cuda::GpuMat& GetSample(int idx = 0) = 0;
    };
}