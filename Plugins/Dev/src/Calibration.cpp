#include "Calibration.h"
#include <EagleLib/rcc/external_includes/cv_calib3d.hpp>
#include <EagleLib/rcc/external_includes/cv_highgui.hpp>
#include <EagleLib/Nodes/VideoProc/Tracking.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>

#include <IRuntimeObjectSystem.h>


using namespace EagleLib;
using namespace EagleLib::Nodes;
IPerModuleInterface* GetModule()
{
    return PerModuleInterface::GetInstance();
}
SETUP_PROJECT_IMPL
bool FindCheckerboard::ProcessImpl()
{
    if(num_corners_x_param.modified || num_corners_y_param.modified ||
        corner_distance_param.modified || object_points.empty())
    {
        object_points.resize(num_corners_x * num_corners_y);
        int count = 0;
        for (int i = 0; i < num_corners_y; ++i)
        {
            for (int j = 0; j < num_corners_x; ++j, ++count)
            {
                object_points[count] = cv::Point3f(corner_distance*j, corner_distance*i, 0);
            }
        }
        num_corners_x_param.modified = false;
        num_corners_y_param.modified = false;
        corner_distance_param.modified = false;
    }
    auto& mat = input->GetMat(*_ctx->stream);
    _ctx->stream->waitForCompletion();
    bool found = cv::findChessboardCorners(mat, cv::Size(num_corners_x, num_corners_y), image_points);
    if (drawn_corners_param.HasSubscriptions())
    {
        cv::Mat display = mat.clone();
        cv::drawChessboardCorners(display, cv::Size(num_corners_x, num_corners_y), image_points, found);
        drawn_corners_param.UpdateData(display, input_param.GetTimestamp(), _ctx);
    }
    return found;
}


void CalibrateCamera::Save()
{
    cv::FileStorage fs;
    fs.open(save_file.string(), cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "Camera Matrix" << camera_matrix;
        fs << "Distortion Matrix" << distortion_matrix;
    }
 
}
void CalibrateCamera::Clear()
{
    boost::recursive_mutex::scoped_lock lock(pointCollectionMtx);
    object_point_collection.clear();
    image_point_collection.clear();
}

void CalibrateCamera::ForceCalibration()
{

}
void CalibrateCamera::SaveCalibration()
{

}
bool CalibrateCamera::ProcessImpl()
{
    if(image_points->size() != object_points->size())
    {
        LOG(trace) << "image_points->size() != object_points->size()";
        return false;
    }
    cv::Size size = image->GetSize();
    cv::Vec2f centroid(0,0);
    for (int i = 0; i < image_points->size(); ++i)
    {
        centroid += cv::Vec2f((*image_points)[i].x, (*image_points)[i].y);
    }
    centroid /= float(image_points->size());
    float minDist = std::numeric_limits<float>::max();
    for(auto& other : image_point_centroids)
    {
        float dist = cv::norm(other - centroid);
        if (dist < minDist)
            minDist = dist;
    }
    if(minDist > min_pixel_distance)
    {
        image_point_collection.push_back(*image_points);
        object_point_collection.push_back(*object_points);
        image_point_centroids.push_back(centroid);
    }
    if(object_point_collection.size() > lastCalibration + 10)
    {
        std::vector<cv::Mat> rvecs;
        std::vector<cv::Mat> tvecs;
        double quality = cv::calibrateCamera(
            object_point_collection, 
            image_point_collection,
            image->GetSize(), 
            camera_matrix, 
            distortion_matrix, 
            rotation_vecs, 
            translation_vecs);
        if(quality < 1)
        {
            // consider disabling the optimization because we have sufficient quality
        }
        camera_matrix_param.Commit();
        distortion_matrix_param.Commit();
        rotation_vecs_param.Commit();
        translation_vecs_param.Commit();
        reprojection_error_param.UpdateData(quality);
        lastCalibration = object_point_collection.size();
    }
    return true;
}


void CalibrateStereoPair::Clear()
{

}
void CalibrateStereoPair::Save()
{
    cv::FileStorage fs("StereoCalibration.yml", cv::FileStorage::WRITE);
    fs << "K1" << *camera_matrix_1;
    fs << "D1" << *distortion_matrix_1;
    fs << "K2" << *camera_matrix_2;
    fs << "D2" << *distortion_matrix_2;
    fs << "Rotation" << rotation_matrix;
    fs << "Translation" << translation_matrix;
    fs << "Essential" << essential_matrix;
    fs << "Fundamental" << fundamental_matrix;
    fs << "R1" << R1;
    fs << "R2" << R2;
    fs << "P1" << P1;
    fs << "P2" << P2;
    fs << "Q" << Q;
}
bool CalibrateStereoPair::ProcessImpl()
{
    if(camera_points_1->size() != camera_points_2->size())
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
        float dist = cv::norm(other - centroid1);
        if (dist < minDist1)
            minDist1 = dist;
    }
    for (cv::Vec2f& other : imagePointCentroids2)
    {
        float dist = cv::norm(other - centroid2);
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
    image_pairs_param.UpdateData(imagePointCollection1.size());
    if(imagePointCollection1.size() > lastCalibration + 20)
    {
        reprojection_error = cv::stereoCalibrate(
            objectPointCollection, 
            imagePointCollection1, 
            imagePointCollection2, 
            *camera_matrix_1,
            *distortion_matrix_1, 
            *camera_matrix_2, 
            *distortion_matrix_2, 
            image->GetSize(), 
            rotation_matrix, 
            translation_matrix, 
            essential_matrix, 
            fundamental_matrix);
        reprojection_error_param.Commit();
        rotation_matrix_param.Commit();
        translation_matrix_param.Commit();
        essential_matrix_param.Commit();
        fundamental_matrix_param.Commit();
        Save();
    }
    return true;
}



void ReadStereoCalibration::OnCalibrationFileChange(mo::Context* ctx, mo::IParameter* param)
{
    cv::FileStorage fs(calibration_file.string(), cv::FileStorage::READ);
    fs["K1"] >> camera_matrix_1;
    fs["K2"] >> camera_matrix_2;
    fs["D1"] >> distortion_matrix_1;
    fs["D2"] >> distortion_matrix_2;
    fs["Rotation"] >> rotation_matrix;
    fs["Translation"] >> translation_matrix;
    fs["Essential"] >> essential_matrix;
    fs["Fundamental"] >> fundamental_matrix;
    fs["R1"] >> R1;
    fs["R2"] >> R2;
    fs["P1"] >> P1;
    fs["P2"] >> P2;
    fs["Q"] >> Q;
    camera_matrix_1_param.Commit();
    camera_matrix_2_param.Commit();
    distortion_matrix_1_param.Commit();
    distortion_matrix_2_param.Commit();
    rotation_matrix_param.Commit();
    translation_matrix_param.Commit();
    essential_matrix_param.Commit();
    fundamental_matrix_param.Commit();
    R1_param.Commit();
    R2_param.Commit();
    P1_param.Commit();
    P2_param.Commit();
    Q_param.Commit();
}

bool ReadStereoCalibration::ProcessImpl()
{
    return true;
}

/*cv::cuda::GpuMat ReadCameraCalibration::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(_parameters[0]->changed)
    {
        std::string path = getParameter<Parameters::ReadFile>(0)->Data()->string();
        cv::FileStorage fs(path, cv::FileStorage::READ);
        fs["Camera Matrix"] >> K;
        fs["Distortion Matrix"] >> dist;

        updateParameter("Camera Matrix", K);
        updateParameter("Distortion Matrix", dist);

    }
    return img;
}

*/
MO_REGISTER_CLASS(CalibrateCamera)
MO_REGISTER_CLASS(CalibrateStereoPair)
MO_REGISTER_CLASS(FindCheckerboard)
MO_REGISTER_CLASS(LoadCameraCalibration)
MO_REGISTER_CLASS(ReadStereoCalibration)
MO_REGISTER_CLASS(ReadCameraCalibration)
