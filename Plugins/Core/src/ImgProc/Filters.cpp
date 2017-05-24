#include "Filters.h"



using namespace aq;
using namespace aq::Nodes;

bool Canny::processImpl()
{
    if(low_thresh_param.modified() || 
        high_thresh_param.modified() || 
        aperature_size_param.modified() || 
        L2_gradient_param.modified() || 
        detector == nullptr)
    {
        detector = cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, aperature_size, L2_gradient);
    }
    cv::cuda::GpuMat edges;
    detector->detect(input->getGpuMat(stream()), edges, stream());
    edges_param.updateData(edges, input_param.getTimestamp(), _ctx);
    return true;
}




MO_REGISTER_CLASS(Canny)

