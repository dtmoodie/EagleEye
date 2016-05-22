#include "EagleLib/nodes/Node.h"
#include <EagleLib/rcc/external_includes/cv_cudabgsegm.hpp>
#include "EagleLib/utilities/CudaUtils.hpp"
#include "Segmentation_impl.h"
#include "libfastms/solver/solver.h"
#include "EagleLib/Defs.hpp"
#include <EagleLib/Project_defs.hpp>
SETUP_PROJECT_DEF

namespace EagleLib
{
    namespace Nodes
    {
    
    class OtsuThreshold: public Node
    {
    public:
        OtsuThreshold();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class SegmentMOG2: public Node
    {

        cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> mog2;
    public:
        SegmentMOG2();
        virtual void NodeInit(bool firstInit);
        virtual void Serialize(ISimpleSerializer *pSerializer);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class SegmentWatershed: public Node
    {
    public:
        SegmentWatershed();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
    class SegmentCPMC: public Node
    {
        enum BackgroundInitialization
        {
            AllBoarders = 0,
            TopAndSides = 1,
            Top = 2,
            Sides = 3
        };

    public:
        SegmentCPMC();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class SegmentGrabCut: public Node
    {
        cv::Mat bgdModel;
        cv::Mat fgdModel;
        ConstBuffer<cv::Mat> maskBuf;
    public:
        SegmentGrabCut();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
    void kmeans_impl(cv::cuda::GpuMat input, cv::cuda::GpuMat& labels, cv::cuda::GpuMat& clusters, int k, cv::cuda::Stream stream, cv::cuda::GpuMat weights = cv::cuda::GpuMat());
    class KMeans: public Node
    {
    public:
        KMeans();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class SegmentKMeans: public Node
    {
//        cv::cuda::HostMem labels;
//        cv::cuda::HostMem clusters;
        cv::cuda::HostMem hostBuf;
    public:
        SegmentKMeans();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class SegmentMeanShift: public Node
    {
        cv::cuda::GpuMat blank;
        cv::cuda::HostMem dest;
    public:
        SegmentMeanShift();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
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
    }

}
