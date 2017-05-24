#include "Filters.h"



using namespace aq;
using namespace aq::Nodes;

bool Canny::ProcessImpl()
{
    if(low_thresh_param._modified || 
        high_thresh_param._modified || 
        aperature_size_param._modified || 
        L2_gradient_param._modified || 
        detector == nullptr)
    {
        detector = cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, aperature_size, L2_gradient);
    }
    cv::cuda::GpuMat edges;
    detector->detect(input->getGpuMat(Stream()), edges, Stream());
    edges_param.UpdateData(edges, input_param.GetTimestamp(), _ctx);
    return true;
}




MO_REGISTER_CLASS(Canny)

