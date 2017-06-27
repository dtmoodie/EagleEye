#pragma once
#include "precompiled.hpp"
#include <boost/circular_buffer.hpp>
#include <MetaObject/params/Types.hpp>
#include <Aquila/types/SyncedMemory.hpp>
namespace aq
{
    typedef std::vector<cv::Point2f> ImagePoints;
    typedef std::vector<cv::Point3f> ObjectPoints;
    namespace nodes
    {
        class FindCheckerboard : public Node
        {
        public:
            MO_DERIVE(FindCheckerboard, Node)
                INPUT(SyncedMemory, input, nullptr);
                PARAM(int, num_corners_x, 6);
                PARAM(int, num_corners_y, 9);
                PARAM(double, corner_distance, 18.75);
                OUTPUT(ImagePoints, image_points, ImagePoints());
                OUTPUT(ObjectPoints, object_points, ObjectPoints());
                OUTPUT(SyncedMemory, drawn_corners, SyncedMemory());
            MO_END;

            bool processImpl();
            //virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
        };
        class LoadCameraCalibration : public Node
        {
            cv::Mat K;
            cv::Mat dist;
        public:
            MO_DERIVE(LoadCameraCalibration, Node)
            MO_END;
        private:
            bool processImpl() { return false; }

        };

        class CalibrateCamera: public Node
        {
            std::vector<ImagePoints> image_point_collection;
            std::vector<ObjectPoints> object_point_collection;
            std::vector<cv::Vec2f> image_point_centroids;
            std::vector<cv::Point3f> objectPoints3d;
            cv::Mat K;
            cv::Mat distortionCoeffs;
            boost::recursive_mutex pointCollectionMtx;
            cv::Size imgSize;
        public:
            MO_DERIVE(CalibrateCamera, Node)
                INPUT(SyncedMemory, image, nullptr);
                INPUT(ImagePoints, image_points, nullptr);
                INPUT(ObjectPoints, object_points, nullptr);
                PARAM(float, min_pixel_distance, 10.0f);
                MO_SLOT(void, Save);
                MO_SLOT(void, Clear);
                MO_SLOT(void, ForceCalibration);
                MO_SLOT(void, SaveCalibration);
                STATUS(cv::Mat, camera_matrix, cv::Mat());
                STATUS(cv::Mat, distortion_matrix, cv::Mat());
                PARAM(mo::WriteFile, save_file, mo::WriteFile("CameraMatrix.yml"));
                STATUS(double, reprojection_error, 0.0);
                OUTPUT(std::vector<cv::Mat>, rotation_vecs, std::vector<cv::Mat>());
                OUTPUT(std::vector<cv::Mat>, translation_vecs, std::vector<cv::Mat>());
                PROPERTY(int, lastCalibration, 0);
            MO_END;
            bool processImpl();

        };

        class CalibrateStereoPair: public Node
        {
            boost::circular_buffer<cv::Vec2f> centroidHistory1;
            boost::circular_buffer<cv::Vec2f> centroidHistory2;

            std::vector<ObjectPoints> objectPointCollection;

            std::vector<ImagePoints> imagePointCollection1;
            std::vector<ImagePoints> imagePointCollection2;

            std::vector<cv::Vec2f> imagePointCentroids1;
            std::vector<cv::Vec2f> imagePointCentroids2;

            std::vector<cv::Vec2f> imagePointCentroids;

            cv::Mat K1, K2, dist1, dist2, Rot, Trans, Ess, Fun;
            cv::Mat R1, R2, P1, P2, Q;

        public:
            MO_DERIVE(CalibrateStereoPair, Node);
                INPUT(SyncedMemory, image, nullptr);
                INPUT(ImagePoints, camera_points_1, nullptr);
                INPUT(ImagePoints, camera_points_2, nullptr);
                INPUT(ObjectPoints, object_points, nullptr);
                INPUT(cv::Mat, camera_matrix_1, nullptr);
                INPUT(cv::Mat, camera_matrix_2, nullptr);
                INPUT(cv::Mat, distortion_matrix_1, nullptr);
                INPUT(cv::Mat, distortion_matrix_2, nullptr);
                MO_SLOT(void, Save);
                MO_SLOT(void, Clear);
                OUTPUT(cv::Mat, rotation_matrix, cv::Mat());
                OUTPUT(cv::Mat, translation_matrix, cv::Mat());
                OUTPUT(cv::Mat, essential_matrix, cv::Mat());
                OUTPUT(cv::Mat, fundamental_matrix, cv::Mat());
                PROPERTY(int, lastCalibration, 0);
                STATUS(int, image_pairs, 0);
                STATUS(double, reprojection_error, 0.0);
                MO_END;

        protected:
            bool processImpl();
        };

        class ReadStereoCalibration: public Node
        {

        public:
            MO_DERIVE(ReadStereoCalibration, Node)
                PARAM(mo::ReadFile, calibration_file, mo::ReadFile("StereoCalibration.yml"));
                OUTPUT(cv::Mat, camera_matrix_1, cv::Mat());
                OUTPUT(cv::Mat, camera_matrix_2, cv::Mat());
                OUTPUT(cv::Mat, distortion_matrix_1, cv::Mat());
                OUTPUT(cv::Mat, distortion_matrix_2, cv::Mat());
                OUTPUT(cv::Mat, rotation_matrix, cv::Mat());
                OUTPUT(cv::Mat, translation_matrix, cv::Mat());
                OUTPUT(cv::Mat, essential_matrix, cv::Mat());
                OUTPUT(cv::Mat, fundamental_matrix, cv::Mat());
                OUTPUT(cv::Mat, R1, cv::Mat());
                OUTPUT(cv::Mat, R2, cv::Mat());
                OUTPUT(cv::Mat, P1, cv::Mat());
                OUTPUT(cv::Mat, P2, cv::Mat());
                OUTPUT(cv::Mat, Q, cv::Mat());
                MO_SLOT(void, OnCalibrationFileChange, mo::Context*, mo::IParam*);
            MO_END;
        protected:
            bool processImpl();
        };

        class ReadCameraCalibration: public Node
        {
            cv::Mat K, dist;
        public:
            MO_DERIVE(ReadCameraCalibration, Node)
                PARAM(mo::ReadFile, calibration_file, mo::ReadFile("CameraCalibration.yml"));
                OUTPUT(cv::Mat, camera_matrix, cv::Mat());
                OUTPUT(cv::Mat, distortion_matrix, cv::Mat());
            MO_END;
        protected:
            bool processImpl(){ return false; }
        };
    }
}
