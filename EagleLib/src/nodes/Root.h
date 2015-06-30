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
/*

#include "nodes/Node.h"
#ifdef RCC_ENABLED
#include "../RuntimeObjectSystem/RuntimeObjectSystem.h"
#endif
namespace EagleLib
{
    class Root: public Node
    {
    public:
        Root();
        ~Root();
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
#ifdef RCC_ENABLED
private:
    boost::shared_ptr<RuntimeObjectSystem> m_pRuntimeObjSystem;
    boost::shared_ptr<ICompilerLogger> m_pCompileLogger;
#endif

    };

}

*/
