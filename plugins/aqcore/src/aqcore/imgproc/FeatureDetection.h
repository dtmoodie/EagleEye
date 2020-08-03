#include "Aquila/rcc/external_includes/cv_cudafeatures2d.hpp"
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedImage.hpp>

#include <RuntimeObjectSystem/RuntimeInclude.h>
#include <RuntimeObjectSystem/RuntimeSourceDependency.h>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace cv
{
    namespace cuda
    {
        class CornersDetector;
        class CornernessCriteria;
    } // namespace cuda
} // namespace cv

namespace aq
{
    namespace nodes
    {

        class GoodFeaturesToTrack : public Node
        {
            cv::Ptr<cv::cuda::CornersDetector> detector;
            void update_detector(int depth);
            void detect(const cv::cuda::GpuMat& img,
                        int frame_number,
                        const cv::cuda::GpuMat& mask,
                        cv::cuda::Stream& stream);

          public:
            MO_DERIVE(GoodFeaturesToTrack, Node)
                PARAM(int, max_corners, 1000)
                PARAM(double, quality_level, 0.01)
                PARAM(double, min_distance, 0.0)
                PARAM(int, block_size, 3)
                PARAM(bool, use_harris, false)
                RANGED_PARAM(double, harris_K, 0.04, 0.01, 1.0)
                INPUT(SyncedImage, input)
                OPTIONAL_INPUT(SyncedImage, mask)
                OUTPUT(SyncedImage, key_points)
                STATUS(int, num_corners, 0)
            MO_END

          protected:
            virtual bool processImpl() override;
        };

        class FastFeatureDetector : public Node
        {
          public:
            MO_DERIVE(FastFeatureDetector, Node)
                PARAM(int, threshold, 10)
                PARAM(bool, use_nonmax_suppression, true);
                ENUM_PARAM(fast_type,
                           cv::cuda::FastFeatureDetector::TYPE_5_8,
                           cv::cuda::FastFeatureDetector::TYPE_7_12,
                           cv::cuda::FastFeatureDetector::TYPE_9_16)
                PARAM(int, max_points, 5000)

                INPUT(SyncedImage, input)
                OPTIONAL_INPUT(SyncedImage, mask)
                OUTPUT(SyncedImage, keypoints)
                STATE(cv::Ptr<cv::cuda::Feature2DAsync>, detector, cv::Ptr<cv::cuda::Feature2DAsync>())
            MO_END

          protected:
            virtual bool processImpl() override;
        };

        class ORBFeatureDetector : public Node
        {
          public:
            MO_DERIVE(ORBFeatureDetector, Node)
                PARAM(int, num_features, 500);
                PARAM(float, scale_factor, 1.2);
                PARAM(int, num_levels, 8);
                PARAM(int, edge_threshold, 31);
                PARAM(int, first_level, 0);
                PARAM(int, WTA_K, 2);
                ENUM_PARAM(score_type, cv::ORB::kBytes, cv::ORB::HARRIS_SCORE, cv::ORB::FAST_SCORE);
                PARAM(int, patch_size, 31);
                PARAM(int, fast_threshold, 20);
                PARAM(bool, blur_for_descriptor, true);

                STATE(cv::Ptr<cv::cuda::ORB>, detector, cv::Ptr<cv::cuda::ORB>());

                INPUT(SyncedImage, input);
                OPTIONAL_INPUT(SyncedImage, mask);

                OUTPUT(SyncedImage, keypoints);
                OUTPUT(SyncedImage, descriptors);
            MO_END;

          protected:
            virtual bool processImpl() override;
        };

        class CornerHarris : public Node
        {
            cv::Ptr<cv::cuda::CornernessCriteria> detector;

          public:
            MO_DERIVE(CornerHarris, Node)
                PARAM(int, block_size, 3)
                PARAM(int, sobel_aperature_size, 5)
                PARAM(double, harris_free_parameter, 1.0)

                INPUT(SyncedImage, input)
                OUTPUT(SyncedImage, score)
            MO_END

          protected:
            virtual bool processImpl() override;
        };

        class CornerMinEigenValue : public Node
        {
            cv::Ptr<cv::cuda::CornernessCriteria> detector;

          public:
            MO_DERIVE(CornerMinEigenValue, Node)
                PARAM(int, block_size, 3)
                PARAM(int, sobel_aperature_size, 5)
                PARAM(double, harris_free_parameter, 1.0)
                INPUT(SyncedImage, input)
                OUTPUT(SyncedImage, score)
            MO_END

          protected:
            virtual bool processImpl() override;
        };
    } // namespace nodes
} // namespace aq
