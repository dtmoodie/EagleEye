#include "precompiled.hpp"
#include "EagleLib/nodes/Node.h"
#include <MetaObject/MetaObject.hpp>

#include "opencv2/cudastereo.hpp"
#include "EagleLib/utilities/CudaUtils.hpp"
namespace EagleLib
{
    namespace Nodes
    {
        class StereoBase: public Node
        {
        public:
            MO_DERIVE(StereoBase, Node)
                INPUT(SyncedMemory, left_image, nullptr);
                INPUT(SyncedMemory, right_image, nullptr);
                OUTPUT(SyncedMemory, disparity, SyncedMemory);
                PARAM(int, num_disparities, 64)
            MO_END;
        };
    class StereoBM: public StereoBase
    {
        cv::Ptr<cv::cuda::StereoBM> stereoBM;
    public:
        MO_DERIVE(StereoBM, StereoBase)
            PARAM(int, block_size, 19);
        MO_END;
    protected:
        bool ProcessImpl();
    };
    class StereoBilateralFilter: public Node
    {
    public:
    };

    class StereoBeliefPropagation: public Node
    {
        cv::Ptr<cv::cuda::StereoBeliefPropagation> bp;
    public:
        StereoBeliefPropagation();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class StereoConstantSpaceBP: public StereoBase
    {
        cv::Ptr<cv::cuda::StereoConstantSpaceBP> csbp;
    public:
        MO_DERIVE(StereoConstantSpaceBP, StereoBase)
            PARAM(int, num_levels, 4);
            PARAM(int, nr_plane, 4);
            PARAM(int, num_iterations, 8);
            ENUM_PARAM(message_type, CV_16SC1, CV_32FC1);
        MO_END;
    protected:
        bool ProcessImpl();
    };
    class UndistortStereo: public Node
    {
        cv::cuda::GpuMat mapY, mapX;
        cv::cuda::HostMem X, Y;

    public:
        MO_DERIVE(UndistortStereo, Node)
            INPUT(SyncedMemory, input, nullptr);
            OUTPUT(SyncedMemory, undistorted, SyncedMemory());
            INPUT(cv::Mat, camera_matrix, nullptr);
            INPUT(cv::Mat, distortion_matrix, nullptr);
            INPUT(cv::Mat, rotation_matrix, nullptr);
            INPUT(cv::Mat, projection_matrix, nullptr);
            ENUM_PARAM(interpolation_method, cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC, cv::INTER_AREA, cv::INTER_LANCZOS4, cv::INTER_MAX );
            ENUM_PARAM(boarder_mode, cv::BORDER_CONSTANT, cv::BORDER_REPLICATE, cv::BORDER_REFLECT, cv::BORDER_WRAP, cv::BORDER_REFLECT_101, cv::BORDER_ISOLATED);
            OUTPUT(SyncedMemory, mapX, SyncedMemory());
            OUTPUT(SyncedMemory, mapY, SyncedMemory());
        MO_END;
    protected:
        bool ProcessImpl();
        UndistortStereo();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
    }
}
