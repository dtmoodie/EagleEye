
#include "FastMS.h"
#include <EagleLib/Nodes/NodeInfo.hpp>

#ifdef FASTMS_FOUND

using namespace EagleLib;
using namespace EagleLib::Nodes;


bool FastMumfordShah::ProcessImpl()
{
    if(!solver)
        solver.reset(new Solver());
    cv::Mat h_img = input->GetMat(Stream());

    Par param;
    param.lambda = lambda;
    param.alpha = alpha;
    param.temporal = temporal;
    param.iterations = iterations;
    param.stop_eps = epsilon;
    param.stop_k = stop_k;
    param.adapt_params = adapt_params;
    param.weight = weight;
    param.edges = overlay_edges;
    stream.waitForCompletion();
    cv::Mat result = solver->run(h_img, param);
    segmented_param.UpdateData(result, input_param.GetTimestamp(), _ctx);
    return true;
}



MO_REGISTER_CLASS(FastMumfordShah);

#endif
