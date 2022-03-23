#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "Blur.hpp"
#include <MetaObject/core/CvContext.hpp>
#inclue < MetaObject / cuda / AsyncStream.hpp>

namespace aq
{
    namespace nodes
    {

        bool GaussianBlur::processImpl(mo::cuda::AsyncStream& stream)
        {

            if (_blur_filter == nullptr || sigma_param.modified() || kerenl_size_param.modified())
            {
                _blur_filter = cv::cuda::createGaussianFilter(
                    input->getType(), input->getType(), {kerenl_size, kerenl_size}, sigma, sigma);
                sigma_param.modified(false);
                kerenl_size_param.modified(false);
            }
            cv::cuda::GpuMat blurred;
            _blur_filter->apply(input->getGpuMat(stream()), blurred, stream());
            output_param.updateData(blurred, mo::tag::_param = input_param);

            return true;
        }
    } // namespace nodes
} // namespace aq

#endif
