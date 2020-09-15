#include <ct/types/opencv.hpp>

#include "Calibration.h"

#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"

#include <Aquila/nodes/NodeInfo.hpp>

#include <Aquila/rcc/external_includes/cv_calib3d.hpp>
#include <Aquila/rcc/external_includes/cv_cudaarithm.hpp>
#include <Aquila/rcc/external_includes/cv_cudaimgproc.hpp>
#include <Aquila/rcc/external_includes/cv_highgui.hpp>

#include <RuntimeObjectSystem/IRuntimeObjectSystem.h>

namespace aqdev
{

    IPerModuleInterface* GetModule() { return PerModuleInterface::GetInstance(); }

    bool FindCheckerboard::processImpl()
    {
        if (num_corners_x_param.getModified() || num_corners_y_param.getModified() ||
            corner_distance_param.getModified() || m_object_points.empty())
        {
            m_object_points.resize(num_corners_x * num_corners_y);
            int count = 0;
            for (int i = 0; i < num_corners_y; ++i)
            {
                for (int j = 0; j < num_corners_x; ++j, ++count)
                {
                    m_object_points[count] = cv::Point3f(corner_distance * j, corner_distance * i, 0);
                }
            }
            num_corners_x_param.setModified(false);
            num_corners_y_param.setModified(false);
            corner_distance_param.setModified(false);
        }
        mo::IAsyncStreamPtr_t stream = this->getStream();
        mo::IDeviceStream* dev_stream = stream->getDeviceStream();

        auto& mat = input->getMat(stream.get());

        bool found = cv::findChessboardCorners(mat, cv::Size(num_corners_x, num_corners_y), m_image_points);

        if (drawn_corners.getNumSubscribers() > 0)
        {
            cv::Mat display = mat.clone();
            cv::drawChessboardCorners(display, cv::Size(num_corners_x, num_corners_y), m_image_points, found);
            drawn_corners.publish(display, mo::tags::param = &input_param);
        }
        image_points.publish(m_image_points, mo::tags::param = &input_param);
        object_points.publish(m_object_points, mo::tags::param = &input_param);
        return found;
    }

    void CalibrateCamera::save()
    {
        cv::FileStorage fs;
        fs.open(save_file.string(), cv::FileStorage::WRITE);
        if (fs.isOpened())
        {
            fs << "Camera Matrix" << camera_matrix;
            fs << "Distortion Matrix" << distortion_matrix;
        }
    }
    void CalibrateCamera::clear()
    {
        boost::recursive_mutex::scoped_lock lock(pointCollectionMtx);
        object_point_collection.clear();
        image_point_collection.clear();
    }

    void CalibrateCamera::forceCalibration() {}

    void CalibrateCamera::saveCalibration() {}

    bool CalibrateCamera::processImpl()
    {
        if (image_points->size() != object_points->size())
        {
            MO_LOG(trace, "image_points->size() != object_points->size()");
            return false;
        }
        // cv::Size size = image->getSize();
        cv::Vec2f centroid(0, 0);
        for (int i = 0; i < image_points->size(); ++i)
        {
            centroid += cv::Vec2f((*image_points)[i].x, (*image_points)[i].y);
        }
        centroid /= float(image_points->size());
        float minDist = std::numeric_limits<float>::max();
        for (auto& other : image_point_centroids)
        {
            double dist = cv::norm(other - centroid);
            if (dist < minDist)
                minDist = dist;
        }
        if (minDist > min_pixel_distance)
        {
            image_point_collection.push_back(*image_points);
            object_point_collection.push_back(*object_points);
            image_point_centroids.push_back(centroid);
        }
        if (object_point_collection.size() > last_calibration + 10)
        {
            const auto size = this->input->size();
            cv::Size sz(size(0), size(1));
            std::vector<cv::Mat> tvecs;
            std::vector<cv::Mat> rvecs;
            double quality = cv::calibrateCamera(
                object_point_collection, image_point_collection, sz, camera_matrix, distortion_matrix, rvecs, tvecs);

            this->translation_vecs.publish(tvecs, mo::tags::param = &this->input_param);
            this->rotation_vecs.publish(rvecs, mo::tags::param = &this->input_param);

            if (quality < 1)
            {
                // consider disabling the optimization because we have sufficient quality
            }

            reprojection_error = quality;
            last_calibration = int(object_point_collection.size());
        }
        return true;
    }

    void CalibrateStereoPair::clear() {}

    void CalibrateStereoPair::save()
    {
        cv::FileStorage fs("StereoCalibration.yml", cv::FileStorage::WRITE);
        fs << "K1" << K1;
        fs << "D1" << dist1;
        fs << "K2" << K2;
        fs << "D2" << dist2;
        fs << "Rotation" << Rot;
        fs << "Translation" << Trans;
        fs << "Essential" << Ess;
        fs << "Fundamental" << Fun;
        fs << "R1" << R1;
        fs << "R2" << R2;
        fs << "P1" << P1;
        fs << "P2" << P2;
        fs << "Q" << Q;
    }

    bool CalibrateStereoPair::processImpl()
    {
        if (camera_points_1->size() != camera_points_2->size())
            return false;

        cv::Vec2f centroid1(0, 0);
        cv::Vec2f centroid2(0, 0);
        for (int i = 0; i < camera_points_1->size(); ++i)
        {
            centroid1 += cv::Vec2f((*camera_points_1)[i].x, (*camera_points_1)[i].y);
            centroid2 += cv::Vec2f((*camera_points_2)[i].x, (*camera_points_2)[i].y);
        }
        centroid1.val[0] /= camera_points_1->size();
        centroid1.val[1] /= camera_points_1->size();
        centroid2.val[0] /= camera_points_2->size();
        centroid2.val[1] /= camera_points_2->size();
        centroidHistory1.push_back(centroid1);
        centroidHistory2.push_back(centroid2);
        float minDist1 = std::numeric_limits<float>::max();
        float minDist2 = std::numeric_limits<float>::max();

        for (cv::Vec2f& other : imagePointCentroids1)
        {
            double dist = cv::norm(other - centroid1);
            if (dist < minDist1)
                minDist1 = dist;
        }
        for (cv::Vec2f& other : imagePointCentroids2)
        {
            double dist = cv::norm(other - centroid2);
            if (dist < minDist2)
                minDist2 = dist;
        }
        if (minDist1 < 100 || minDist2 < 100)
        {
            return false;
        }
        cv::Vec2f motionSum1(0, 0);
        cv::Vec2f motionSum2(0, 0);

        for (int i = 1; i < centroidHistory1.size(); ++i)
        {
            motionSum1 += centroidHistory1[i] - centroidHistory1[i - 1];
        }

        for (int i = 1; i < centroidHistory2.size(); ++i)
        {
            motionSum2 += centroidHistory2[i] - centroidHistory2[i - 1];
        }
        imagePointCollection1.push_back(*camera_points_1);
        imagePointCollection2.push_back(*camera_points_2);
        objectPointCollection.push_back(*object_points);

        image_pairs = imagePointCollection1.size();

        if (int(imagePointCollection1.size()) > last_calibration + 20)
        {
            const auto sz = input->size();
            const cv::Size size(sz(0), sz(1));
            cv::Mat rmatrix;
            cv::Mat tmatrix;
            cv::Mat E;
            cv::Mat F;
            reprojection_error = cv::stereoCalibrate(objectPointCollection,
                                                     imagePointCollection1,
                                                     imagePointCollection2,
                                                     *camera_matrix_1,
                                                     *distortion_matrix_1,
                                                     *camera_matrix_2,
                                                     *distortion_matrix_2,
                                                     size,
                                                     rmatrix,
                                                     tmatrix,
                                                     E,
                                                     F);
            this->rotation_matrix.publish(rmatrix, mo::tags::param = &this->input_param);
            this->translation_matrix.publish(tmatrix, mo::tags::param = &this->input_param);
            this->essential_matrix.publish(E, mo::tags::param = &this->input_param);
            this->fundamental_matrix.publish(F, mo::tags::param = &this->input_param);

            save();
        }
        return true;
    }

    void ReadStereoCalibration::on_calibration_file_modified(const mo::IParam&,
                                                             mo::Header,
                                                             mo::UpdateFlags,
                                                             mo::IAsyncStream&)
    {
        cv::FileStorage fs(calibration_file.string(), cv::FileStorage::READ);
        cv::Mat K1;
        cv::Mat K2;
        cv::Mat D1;
        cv::Mat D2;
        cv::Mat R;
        cv::Mat T;
        cv::Mat E;
        cv::Mat F;
        cv::Mat R1;
        cv::Mat R2;
        cv::Mat P1;
        cv::Mat P2;
        cv::Mat Q;

        fs["K1"] >> K1;
        fs["K2"] >> K2;
        fs["D1"] >> D1;
        fs["D2"] >> D2;
        fs["Rotation"] >> R;
        fs["Translation"] >> T;
        fs["Essential"] >> E;
        fs["Fundamental"] >> F;
        fs["R1"] >> R1;
        fs["R2"] >> R2;
        fs["P1"] >> P1;
        fs["P2"] >> P2;
        fs["Q"] >> Q;
        this->camera_matrix_1.publish(K1);
        this->camera_matrix_2.publish(K2);
        this->distortion_matrix_1.publish(D1);
        this->distortion_matrix_2.publish(D2);
        this->rotation_matrix.publish(R);
        this->translation_matrix.publish(T);
        this->essential_matrix.publish(E);
        this->fundamental_matrix.publish(F);
        this->R1.publish(R1);
        this->R2.publish(R2);
        this->P1.publish(P1);
        this->P2.publish(P2);
        this->Q.publish(Q);
    }

    bool ReadStereoCalibration::processImpl() { return true; }
} // namespace aqdev

using namespace aqdev;
MO_REGISTER_CLASS(CalibrateCamera)
MO_REGISTER_CLASS(CalibrateStereoPair)
MO_REGISTER_CLASS(FindCheckerboard)
MO_REGISTER_CLASS(LoadCameraCalibration)
MO_REGISTER_CLASS(ReadStereoCalibration)
MO_REGISTER_CLASS(ReadCameraCalibration)
