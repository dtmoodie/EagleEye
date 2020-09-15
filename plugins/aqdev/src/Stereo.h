#include <Aquila/types/SyncedImage.hpp>

#include "Aquila/nodes/Node.hpp"
#include "opencv2/cudastereo.hpp"

#include <aqcore/OpenCVCudaNode.hpp>

#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/params/ParamMacros.hpp>
#include <MetaObject/types/file_types.hpp>

#include "opencv2/cudastereo.hpp"
#include <opencv2/imgproc.hpp>

namespace aqdev
{

    class StereoBase : public aqcore::OpenCVCudaNode
    {
      public:
        MO_DERIVE(StereoBase, aqcore::OpenCVCudaNode)
            INPUT(aq::SyncedImage, left_image)
            INPUT(aq::SyncedImage, right_image)
            OUTPUT(aq::SyncedImage, disparity)
            PARAM(int, num_disparities, 64)
        MO_END;
    };
    class StereoBM : public StereoBase
    {
      public:
        MO_DERIVE(StereoBM, StereoBase)
            PARAM(int, block_size, 19)
        MO_END;

      protected:
        bool processImpl() override;
        cv::Ptr<cv::cuda::StereoBM> stereoBM;
    };
    class StereoBilateralFilter : public aqcore::OpenCVCudaNode
    {
      public:
        MO_DERIVE(StereoBilateralFilter, aqcore::OpenCVCudaNode)

        MO_END;

      protected:
        bool processImpl() { return false; }
    };

    class StereoBeliefPropagation : public StereoBase
    {
      public:
        MO_DERIVE(StereoBeliefPropagation, StereoBase)
            PARAM(int, num_iters, 5)
            PARAM(int, num_levels, 5)
            ENUM_PARAM(message_type, CV_16S, CV_32F)
        MO_END;

      protected:
        bool processImpl() override;
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
        MO_END;

      protected:
        bool processImpl() override;
    };

    class UndistortStereo : public aqcore::OpenCVCudaNode
    {
      public:
        MO_DERIVE(UndistortStereo, aqcore::OpenCVCudaNode)
            INPUT(aq::SyncedImage, input)

            INPUT(cv::Mat, camera_matrix)
            INPUT(cv::Mat, distortion_matrix)
            INPUT(cv::Mat, rotation_matrix)
            INPUT(cv::Mat, projection_matrix)

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

            OUTPUT(aq::SyncedImage, mapX)
            OUTPUT(aq::SyncedImage, mapY)
            OUTPUT(aq::SyncedImage, undistorted)

        MO_END;

      protected:
        cv::cuda::GpuMat m_map_x;
        cv::cuda::GpuMat m_map_y;
        bool processImpl() override;
    };

} // namespace aqdev
