#include "nodes/VideoProc/OpticalFlow.h"

#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
using namespace EagleLib;

#ifdef RCC_ENABLED
    #include "../RuntimeObjectSystem/ObjectInterfacePerModule.h"
    #if __linux
    RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_cuda -lopencv_cudaoptflow")
    #endif

#endif

