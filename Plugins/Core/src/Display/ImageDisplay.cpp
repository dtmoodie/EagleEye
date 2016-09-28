#include "ImageDisplay.h"

#include <MetaObject/Thread/InterThread.hpp>
#include "../remotery/lib/Remotery.h"
#include <EagleLib/utilities/CudaCallbacks.hpp>
#include <EagleLib/utilities/UiCallbackHandlers.h>
#include "EagleLib/profiling.h"


using namespace EagleLib;
using namespace EagleLib::Nodes;

bool QtImageDisplay::ProcessImpl()
{
    cv::Mat mat;
    if(image)
    {
        cv::Mat mat = image->GetMat(*_ctx->stream);
    }
    if(cpu_mat)
    {
        mat = *cpu_mat;
    }
    
    std::string name = GetTreeName();
    if(!mat.empty())
    {
        EagleLib::cuda::enqueue_callback(
            [mat, name]()->void
        {
            cv::imshow(name, mat);
        }, *_ctx->stream);
    }
    
    return true;
}

bool KeyPointDisplay::ProcessImpl()
{
    return true;
}

bool FlowVectorDisplay::ProcessImpl()
{
    return true;
}

bool HistogramDisplay::ProcessImpl()
{
    return true;
}

bool DetectionDisplay::ProcessImpl()
{
    return true;
}

bool OGLImageDisplay::ProcessImpl()
{
    return true;
}


MO_REGISTER_CLASS(QtImageDisplay);
MO_REGISTER_CLASS(KeyPointDisplay);
MO_REGISTER_CLASS(FlowVectorDisplay);
MO_REGISTER_CLASS(HistogramDisplay);
MO_REGISTER_CLASS(DetectionDisplay);
MO_REGISTER_CLASS(OGLImageDisplay);


