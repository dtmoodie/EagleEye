#include "src/precompiled.hpp"
#include <EagleLib/rcc/external_includes/cv_cudafeatures3d.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    namespace Nodes
    {
    
    class Sobel: public Node
    {
    public:


    };

    class Canny: public Node
    {
        cv::Ptr<cv::cuda::CannyEdgeDetector> detector;
    public:
        MO_DERIVE(Canny, Node);
            PARAM(double, low_thresh, 0.0);
            PARAM(double, high_thresh, 20.0);
            PARAM(int, aperature_size, 3);
            PARAM(bool, L2_gradient, false);
            INPUT(SyncedMemory, input, nullptr);
            OUTPUT(SyncedMemory, edges, SyncedMemory());
        MO_END;
    protected:
        bool ProcessImpl();
    };

    class Laplacian: public Node
    {
    public:
    
    };
    class BiLateral: public Node
    {
    public:
    
    };
    class MeanShiftFilter: public Node
    {
    public:
    
    };
    class MeanShiftProc: public Node
    {
    public:
    
    };
    class MeanShiftSegmentation: public Node
    {
    public:
    
    };
    }
}
