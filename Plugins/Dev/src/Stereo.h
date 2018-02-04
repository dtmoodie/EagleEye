#include "Aquila/nodes/Node.hpp"
#include "Aquila/utilities/cuda/CudaUtils.hpp"
#include "opencv2/cudastereo.hpp"
#include "precompiled.hpp"
#include <Aquila/types/SyncedMemory.hpp>
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/params/ParamMacros.hpp>
#include <MetaObject/types/file_types.hpp>
#include <opencv2/imgproc.hpp>
namespace aq
{
    namespace nodes
    {
        class StereoBase : public Node
        {
          public:
            MO_DERIVE(StereoBase, Node)
            INPUT(SyncedMemory, left_image, nullptr)
            INPUT(SyncedMemory, right_image, nullptr)
            OUTPUT(SyncedMemory, disparity, SyncedMemory())
            PARAM(int, num_disparities, 64)
            MO_END
        };
        class StereoBM : public StereoBase
        {
          public:
            MO_DERIVE(StereoBM, StereoBase)
            PARAM(int, block_size, 19)
            MO_END
          protected:
            bool processImpl();
            cv::Ptr<cv::cuda::StereoBM> stereoBM;
        };
        class StereoBilateralFilter : public Node
        {
          public:
            MO_DERIVE(StereoBilateralFilter, Node)

            MO_END
          protected:
            bool processImpl() { return false; }
        };

        class StereoBeliefPropagation : public StereoBase
        {
          public:
            MO_DERIVE(StereoBeliefPropagation, Node)
            PARAM(int, num_iters, 5)
            PARAM(int, num_levels, 5)
            ENUM_PARAM(message_type, CV_16S, CV_32F)
            MO_END

          protected:
            bool processImpl();
            cv::Ptr<cv::cuda::StereoBeliefPropagation> bp;
        };

        class StereoConstantSpaceBP : public StereoBase
        {
            cv::Ptr<cv::cuda::StereoConstantSpaceBP> csbp;

          public:
            MO_DERIVE(StereoConstantSpaceBP, StereoBase)
            PARAM(int, num_levels, 4)
            PARAM(int, nr_plane, 4)
            PARAM(int, num_iterations, 8)
            ENUM_PARAM(message_type, CV_16SC1, CV_32FC1)
            MO_END
          protected:
            bool processImpl();
        };

        class UndistortStereo : public Node
        {
          public:
            MO_DERIVE(UndistortStereo, Node)
            INPUT(SyncedMemory, input, nullptr)
            OUTPUT(SyncedMemory, undistorted, SyncedMemory())
            INPUT(cv::Mat, camera_matrix, nullptr)
            INPUT(cv::Mat, distortion_matrix, nullptr)
            INPUT(cv::Mat, rotation_matrix, nullptr)
            INPUT(cv::Mat, projection_matrix, nullptr)
            ENUM_PARAM(interpolation_method,
                       cv::INTER_NEAREST,
                       cv::INTER_LINEAR,
                       cv::INTER_CUBIC,
                       cv::INTER_AREA,
                       cv::INTER_LANCZOS4,
                       cv::INTER_MAX)
            ENUM_PARAM(boarder_mode,
                       cv::BORDER_CONSTANT,
                       cv::BORDER_REPLICATE,
                       cv::BORDER_REFLECT,
                       cv::BORDER_WRAP,
                       cv::BORDER_REFLECT_101,
                       cv::BORDER_ISOLATED)
            OUTPUT(SyncedMemory, mapX, SyncedMemory())
            OUTPUT(SyncedMemory, mapY, SyncedMemory())
            MO_END
          protected:
            bool processImpl();
        };
    }
}
