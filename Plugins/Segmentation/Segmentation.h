#include "Aquila/Nodes/Node.h"
#include <Aquila/types/SyncedMemory.hpp>
#include <Aquila/rcc/external_includes/cv_cudabgsegm.hpp>
#include "Aquila/utilities/CudaUtils.hpp"
#include <Aquila/metatypes/SyncedMemoryMetaParams.hpp>
#include "Segmentation_impl.h"
#ifdef FASTMS_FOUND
#include "libfastms/solver/solver.h"
#endif
#include <MetaObject/Detail/MetaObjectMacros.hpp>



namespace aq
{
    namespace Nodes
    {

    class OtsuThreshold: public Node
    {
    public:
        MO_DERIVE(OtsuThreshold, Node)
            INPUT(SyncedMemory, image, nullptr)
            OPTIONAL_INPUT(SyncedMemory, histogram, nullptr)
            OPTIONAL_INPUT(SyncedMemory, range, nullptr)
            OUTPUT(SyncedMemory, output, SyncedMemory())
        MO_END
    protected:
        bool ProcessImpl();
    };

    class MOG2: public Node
    {
    public:
        MO_DERIVE(MOG2, Node)
            INPUT(SyncedMemory, image, nullptr)
            PARAM(int, history, 500)
            PARAM(double, threshold, 15)
            PARAM(bool, detect_shadows, true)
            PARAM(double, learning_rate, 1.0)
            OUTPUT(SyncedMemory, background, SyncedMemory())
        MO_END;

    protected:
        bool ProcessImpl();
        cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> mog2;
    };

    class Watershed: public Node
    {
    public:
        MO_DERIVE(Watershed, Node)
            INPUT(SyncedMemory, image, nullptr)
            INPUT(SyncedMemory, marker_mask, nullptr)
            OUTPUT(SyncedMemory, mask, SyncedMemory())
        MO_END;
    protected:
        bool ProcessImpl();
    };

    void kmeans_impl(cv::cuda::GpuMat input, cv::cuda::GpuMat& labels, cv::cuda::GpuMat& clusters, int k, cv::cuda::Stream stream, cv::cuda::GpuMat weights = cv::cuda::GpuMat());


    class KMeans: public Node
    {
        cv::cuda::HostMem hostBuf;
    public:
        MO_DERIVE(KMeans, Node)
            INPUT(SyncedMemory, image, nullptr)
            ENUM_PARAM(flags, cv::KMEANS_PP_CENTERS, cv::KMEANS_RANDOM_CENTERS, cv::KMEANS_USE_INITIAL_LABELS)
            PARAM(int, k, 10)
            PARAM(int, iterations, 100)
            PARAM(double, epsilon, 0.1)
            PARAM(int, attempts, 1)
            PARAM(double, color_weight, 1.0)
            PARAM(double, distance_weight, 1.0)
            OUTPUT(SyncedMemory, clusters, SyncedMemory())
            OUTPUT(SyncedMemory, labels, SyncedMemory())
            OUTPUT(double, compactness, 0.0)
        MO_END;
    protected:
        bool ProcessImpl();
    };

    class MeanShift: public Node
    {
        cv::cuda::GpuMat blank;

    public:
        MO_DERIVE(MeanShift, Node)
            INPUT(SyncedMemory, image, nullptr)
            PARAM(int, spatial_radius, 5)
            PARAM(int, color_radius, 5)
            PARAM(int, min_size, 5)
            PARAM(int, max_iters, 5)
            PARAM(double, epsilon, 1.0)
            OUTPUT(SyncedMemory, output, SyncedMemory())
        MO_END
        bool ProcessImpl();
    };

    class ManualMask: public Node
    {
        enum MaskType
        {
            Circular = 0,
            Rectangular = 1
        };
        cv::cuda::GpuMat mask;
    public:
        ManualMask();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
    #ifdef FASTMS_FOUND
    class SLaT : public Node
    {
        cv::cuda::HostMem imageBuffer;
        cv::Mat lab;
        cv::Mat smoothed_32f;
        cv::Mat lab_32f;
        cv::Mat tensor;
        cv::Mat labels;
        cv::Mat centers;
        boost::shared_ptr<Solver> solver;
    public:
        SLaT();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
#endif
    }
}
