#include "src/precompiled.hpp"
#include <Aquila/rcc/external_includes/cv_cudafeatures3d.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aq
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
        bool processImpl();
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
