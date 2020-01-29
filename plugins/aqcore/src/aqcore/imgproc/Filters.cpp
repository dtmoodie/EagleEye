#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "Filters.h"
#include <Aquila/nodes/NodeInfo.hpp>

using namespace aq;
using namespace aq::nodes;

bool Canny::processImpl()
{
    if (low_thresh_param.modified() || high_thresh_param.modified() || aperature_size_param.modified() ||
        L2_gradient_param.modified() || detector == nullptr)
    {
        detector = cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, aperature_size, L2_gradient);
    }
    cv::cuda::GpuMat edges;
    detector->detect(input->getGpuMat(stream()), edges, stream());
    edges_param.updateData(edges, mo::tag::_param = input_param, _ctx.get());
    return true;
}

MO_REGISTER_CLASS(Canny)
#endif
