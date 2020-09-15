#pragma once
#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/nodes/Node.hpp>

#include <MetaObject/types/file_types.hpp>
#include <boost/circular_buffer.hpp>

namespace aqdev
{
    typedef std::vector<cv::Point2f> ImagePoints;
    typedef std::vector<cv::Point3f> ObjectPoints;

    class FindCheckerboard : public aq::nodes::Node
    {
      public:
        MO_DERIVE(FindCheckerboard, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)

            PARAM(int, num_corners_x, 6)
            PARAM(int, num_corners_y, 9)
            PARAM(float, corner_distance, 18.75f)

            OUTPUT(ImagePoints, image_points)
            OUTPUT(ObjectPoints, object_points)
            OUTPUT(aq::SyncedImage, drawn_corners)
        MO_END;

        bool processImpl();

      private:
        ImagePoints m_image_points;
        ObjectPoints m_object_points;
    };
    class LoadCameraCalibration : public aq::nodes::Node
    {
        cv::Mat K;
        cv::Mat dist;

      public:
        MO_DERIVE(LoadCameraCalibration, aq::nodes::Node)
        MO_END;

      private:
        bool processImpl() override { return false; }
    };

    class CalibrateCamera : public aq::nodes::Node
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
        MO_DERIVE(CalibrateCamera, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)
            INPUT(ImagePoints, image_points)
            INPUT(ObjectPoints, object_points)

            PARAM(float, min_pixel_distance, 10.0f)
            MO_SLOT(void, save)
            MO_SLOT(void, clear)
            MO_SLOT(void, forceCalibration)
            MO_SLOT(void, saveCalibration)

            PARAM(mo::WriteFile, save_file, mo::WriteFile("CameraMatrix.yml"))

            OUTPUT(std::vector<cv::Mat>, rotation_vecs)
            OUTPUT(std::vector<cv::Mat>, translation_vecs)
        MO_END;

      private:
        cv::Mat camera_matrix;
        cv::Mat distortion_matrix;
        double reprojection_error = 0.0;
        int last_calibration = 0;
        bool processImpl() override;
    };

    class CalibrateStereoPair : public aq::nodes::Node
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
        MO_DERIVE(CalibrateStereoPair, aq::nodes::Node)

            INPUT(aq::SyncedImage, input)
            INPUT(ImagePoints, camera_points_1)
            INPUT(ImagePoints, camera_points_2)
            INPUT(ObjectPoints, object_points)
            INPUT(cv::Mat, camera_matrix_1)
            INPUT(cv::Mat, camera_matrix_2)
            INPUT(cv::Mat, distortion_matrix_1)
            INPUT(cv::Mat, distortion_matrix_2)

            MO_SLOT(void, save)
            MO_SLOT(void, clear)

            OUTPUT(cv::Mat, rotation_matrix)
            OUTPUT(cv::Mat, translation_matrix)
            OUTPUT(cv::Mat, essential_matrix)
            OUTPUT(cv::Mat, fundamental_matrix)
        MO_END;

      protected:
        bool processImpl() override;

      private:
        int last_calibration = 0;
        int image_pairs = 0;
        double reprojection_error = 0.0;
    };

    class ReadStereoCalibration : public aq::nodes::Node
    {

      public:
        MO_DERIVE(ReadStereoCalibration, aq::nodes::Node)
            PARAM(mo::ReadFile, calibration_file, mo::ReadFile("StereoCalibration.yml"));

            PARAM_UPDATE_SLOT(calibration_file)

            OUTPUT(cv::Mat, camera_matrix_1)
            OUTPUT(cv::Mat, camera_matrix_2)
            OUTPUT(cv::Mat, distortion_matrix_1)
            OUTPUT(cv::Mat, distortion_matrix_2)
            OUTPUT(cv::Mat, rotation_matrix)
            OUTPUT(cv::Mat, translation_matrix)
            OUTPUT(cv::Mat, essential_matrix)
            OUTPUT(cv::Mat, fundamental_matrix)
            OUTPUT(cv::Mat, R1)
            OUTPUT(cv::Mat, R2)
            OUTPUT(cv::Mat, P1)
            OUTPUT(cv::Mat, P2)
            OUTPUT(cv::Mat, Q)

        MO_END;

      protected:
        bool processImpl() override;
    };

    class ReadCameraCalibration : public aq::nodes::Node
    {
        cv::Mat K, dist;

      public:
        MO_DERIVE(ReadCameraCalibration, aq::nodes::Node)
            PARAM(mo::ReadFile, calibration_file, mo::ReadFile("CameraCalibration.yml"));

            OUTPUT(cv::Mat, camera_matrix)
            OUTPUT(cv::Mat, distortion_matrix)
        MO_END;

      protected:
        bool processImpl() override { return false; }
    };
} // namespace aqdev
