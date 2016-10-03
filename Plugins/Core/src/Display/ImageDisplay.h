#include "src/precompile.hpp"
#include "EagleLib/nodes/Sink.h"
#include <EagleLib/rcc/external_includes/cv_highgui.hpp>
#include <EagleLib/rcc/external_includes/cv_core.hpp>
#include <EagleLib/utilities/CudaUtils.hpp>
#include <EagleLib/ObjectDetection.hpp>

#include <MetaObject/Parameters/ParameterMacros.hpp>
#include <MetaObject/Parameters/TypedInputParameter.hpp>

#include <opencv2/core/opengl.hpp>
namespace EagleLib
{
namespace Nodes
{
    class QtImageDisplay: public Node
    {
    public:
        MO_DERIVE(QtImageDisplay, Node);
            INPUT(TS<SyncedMemory>, image, nullptr);
            INPUT(cv::Mat, cpu_mat, nullptr);
        MO_END;
    protected:
        bool ProcessImpl();
    };
    class OGLImageDisplay: public Node
    {
    public:
        MO_DERIVE(OGLImageDisplay, Node);
            INPUT(TS<SyncedMemory>, image, nullptr);
        MO_END;
        bool ProcessImpl();
    };
    class KeyPointDisplay: public Node
    {
    public:
        MO_DERIVE(KeyPointDisplay, Node);
            INPUT(TS<SyncedMemory>, image, nullptr);
            INPUT(TS<SyncedMemory>, synced_points,  nullptr);
            INPUT(cv::cuda::GpuMat, gpu_points, nullptr);
            INPUT(cv::Mat, cpu_points, nullptr);
        MO_END;
    protected:
        bool ProcessImpl();
    };
    class FlowVectorDisplay: public Node
    {
    public:
        MO_DERIVE(FlowVectorDisplay, Node);
            INPUT(TS<SyncedMemory>, image, nullptr);
        MO_END;
    protected:
        bool ProcessImpl();
    };

    class HistogramDisplay: public Node
    {
    public:
        MO_DERIVE(HistogramDisplay, Node);
        MO_END;
    protected:
        bool ProcessImpl();
    };
    class DetectionDisplay: public Node
    {
    public:
        MO_DERIVE(DetectionDisplay, Node);
        INPUT(TS<SyncedMemory>, image, nullptr);
        MO_END;
    protected:
        bool ProcessImpl();
    };
} // namespace Nodes
} // namespace EagleLib
