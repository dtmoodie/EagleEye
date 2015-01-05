#pragma once
/*  The root node is of critical to dynamic recompiles since this node contains the exception handling and
 *  file monitoring code for dynamically re-compiling code
 *
 *
 *
 *
 *
 *
 *
 *
 *
*/


#include "nodes/Node.h"
#include "../RuntimeObjectSystem/RuntimeObjectSystem.h"

namespace EagleLib
{
    class Root: public Node
    {
    public:
        Root();
        ~Root();
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img);
    };
    boost::shared_ptr<RuntimeObjectSystem> objSystem;

}

