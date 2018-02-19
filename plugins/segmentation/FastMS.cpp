
#ifdef FASTMS_FOUND
#include "FastMS.h"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include <Aquila/nodes/NodeInfo.hpp>

using namespace aq;
using namespace aq::nodes;

bool FastMumfordShah::processImpl()
{
    if (!solver)
        solver.reset(new Solver());
    cv::Mat h_img = input->getMat(stream());

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
    stream().waitForCompletion();
    cv::Mat result = solver->run(h_img, param);
    segmented_param.updateData(result, input_param.getTimestamp(), _ctx.get());
    return true;
}

MO_REGISTER_CLASS(FastMumfordShah);

#endif
