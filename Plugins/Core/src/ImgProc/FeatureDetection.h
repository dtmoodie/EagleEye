
#include "src/precompiled.hpp"
//#include "EagleLib/Nodes/VideoProc/Tracking.hpp"
#include "EagleLib/rcc/external_includes/cv_cudafeatures2d.hpp"

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace EagleLib
{
    namespace Nodes
    {
        class GoodFeaturesToTrack : public Node
        {
            cv::Ptr<cv::cuda::CornersDetector> detector;
            void update_detector(int depth);
            void detect(const cv::cuda::GpuMat& img, int frame_number, const cv::cuda::GpuMat& mask, cv::cuda::Stream& stream);
        public:
            MO_DERIVE(GoodFeaturesToTrack, Node);
                PARAM(int, max_corners, 1000);
                PARAM(double, quality_level, 0.01);
                PARAM(double, min_distance, 0.0);
                PARAM(int, block_size, 3);
                PARAM(bool, use_harris, false);
                RANGED_PARAM(double, harris_K, 0.04, 0.01, 1.0);
                INPUT(SyncedMemory, input, nullptr);
                OPTIONAL_INPUT(SyncedMemory, mask, nullptr);
                OUTPUT(SyncedMemory, key_points, SyncedMemory());
                STATUS(int, num_corners, 0);
            MO_END;
        protected:
            bool ProcessImpl();

            
        };

        class FastFeatureDetector : public Node
        {
        public:
            MO_DERIVE(FastFeatureDetector, Node);
                PARAM(int, threshold, 10);
                PARAM(bool, use_nonmax_suppression, true);
                ENUM_PARAM(fast_type, cv::cuda::FastFeatureDetector::TYPE_5_8, cv::cuda::FastFeatureDetector::TYPE_7_12, cv::cuda::FastFeatureDetector::TYPE_9_16);
                PARAM(int, max_points, 5000);
                INPUT(SyncedMemory, input, nullptr);
                OPTIONAL_INPUT(SyncedMemory, mask, nullptr);
                OUTPUT(SyncedMemory, keypoints, SyncedMemory());
                PROPERTY(cv::Ptr<cv::cuda::Feature2DAsync>, detector, cv::Ptr<cv::cuda::Feature2DAsync>());
            MO_END;
        protected:
            bool ProcessImpl();
        };

        class ORBFeatureDetector : public Node
        {
        public:
            MO_DERIVE(ORBFeatureDetector, Node);
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
                INPUT(SyncedMemory, input, nullptr);
                OPTIONAL_INPUT(SyncedMemory, mask, nullptr);
                PROPERTY(cv::Ptr<cv::cuda::ORB>, detector, cv::Ptr<cv::cuda::ORB>());
                OUTPUT(SyncedMemory, keypoints, SyncedMemory());
                OUTPUT(SyncedMemory, descriptors, SyncedMemory());
            MO_END;
        protected:
            bool ProcessImpl();

        };



        class CornerHarris : public Node
        {
            cv::Ptr<cv::cuda::CornernessCriteria> detector;
        public:
            MO_DERIVE(CornerHarris, Node)
                PARAM(int, block_size, 3);
                PARAM(int, sobel_aperature_size, 5);
                PARAM(double, harris_free_parameter, 1.0);
                INPUT(SyncedMemory, input, nullptr);
                OUTPUT(SyncedMemory, score, SyncedMemory());
            MO_END;
        protected:
            bool ProcessImpl();
        };
        class CornerMinEigenValue : public Node
        {
            cv::Ptr<cv::cuda::CornernessCriteria> detector;
        public:
            MO_DERIVE(CornerMinEigenValue, Node)
                PARAM(int, block_size, 3);
                PARAM(int, sobel_aperature_size, 5);
                PARAM(double, harris_free_parameter, 1.0);
                INPUT(SyncedMemory, input, nullptr);
                OUTPUT(SyncedMemory, score, SyncedMemory());
                MO_END;
        protected:
            bool ProcessImpl();
        };
    }
}
