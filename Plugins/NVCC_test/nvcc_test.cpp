#include "nvcc_test.h"
#include "nvcc_test.cuh"
#include <EagleLib/Nodes/NodeInfo.hpp>
#include "RuntimeSourceDependency.h"
#include <EagleLib/Detail/PluginExport.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>


using namespace EagleLib;
using namespace EagleLib::Nodes;
SETUP_PROJECT_IMPL

bool nvcc_test::ProcessImpl()
{
    //run_kernel()
    return false;
}


MO_REGISTER_CLASS(nvcc_test)

