#include "Filters.h"



using namespace EagleLib;
using namespace EagleLib::Nodes;

bool Canny::ProcessImpl()
{
    if(low_thresh_param.modified || 
        high_thresh_param.modified || 
        aperature_size_param.modified || 
        L2_gradient_param.modified || 
        detector == nullptr)
    {
        detector = cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, aperature_size, L2_gradient);
    }
    cv::cuda::GpuMat edges;
    detector->detect(input->GetGpuMat(Stream()), edges, Stream());
    edges_param.UpdateData(edges, input_param.GetTimestamp(), _ctx);
    return true;
}




MO_REGISTER_CLASS(Canny)

