#include "Calibration.h"
#include <external_includes/cv_calib3d.hpp>
#include <external_includes/cv_highgui.hpp>
#include <nodes/VideoProc/Tracking.hpp>
#include <external_includes/cv_cudaarithm.hpp>
#include <external_includes/cv_cudaimgproc.hpp>

#include "Manager.h"
#include <UI/InterThread.hpp>
using namespace EagleLib;

IPerModuleInterface* GetModule()
{
    return PerModuleInterface::GetInstance();
}
SETUP_PROJECT_IMPL
void FindCheckerboard::Init(bool firstInit)
{
	
	if (firstInit)
	{
		updateParameter("Num corners X", int(6));
		updateParameter("Num corners Y", int(9));
		updateParameter("Corner distance", double(18.75), Parameters::Parameter::Control, "Distance between corners in mm");
		addInputParameter<TrackSparseFunctor>("Sparse tracking functor");
	}
	
    updateParameterPtr("Image points 2d", &imagePoints);
    updateParameterPtr("Object points 3d", &objectPoints);
}
cv::cuda::GpuMat FindCheckerboard::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
	int numY = *getParameter<int>(1)->Data();
	int numX = *getParameter<int>(0)->Data();
	double dx = *getParameter<double>(2)->Data();
	if (parameters[0]->changed || parameters[1]->changed || parameters[2]->changed)
	{
        objectPoints.resize(numY * numX);
        int count = 0;
		for (int i = 0; i < numY; ++i)
		{
            for (int j = 0; j < numX; ++j, ++count)
			{
                objectPoints[count] = cv::Point3f(dx*j, dx*i, 0);
			}
		}
		parameters[0]->changed = false;
		parameters[1]->changed = false;
		parameters[2]->changed = false;
	}
	TrackSparseFunctor* tracker = getParameter<TrackSparseFunctor>("Sparse tracking functor")->Data();
	bool found = false;
	cv::Mat h_corners;

	if (img.channels() == 3)
		cv::cuda::cvtColor(img, currentGreyFrame, cv::COLOR_BGR2GRAY, 0, stream);
	else
		currentGreyFrame = img;

	if (tracker)
	{
		if (prevFramePoints.empty())
		{
			//log(Status, "Initializing with CPU corner finder routine");
			NODE_LOG(info) << "Initializing with CPU corner finder routine";
			currentGreyFrame.download(h_img, stream);
			stream.waitForCompletion();

			found = cv::findChessboardCorners(h_img, cv::Size(numX, numY), imagePoints);
			if (found)
			{
				TIME
				prevFramePoints.upload(imagePoints, stream);
				prevFramePoints = prevFramePoints.reshape(2, 1);
				prevFramePoints.copyTo(currentFramePoints, stream);
				cv::drawChessboardCorners(h_img, cv::Size(numX, numY), imagePoints, found);
				Parameters::UI::UiCallbackService::Instance()->post(boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow), fullTreeName, h_img));
				prevGreyFrame = currentGreyFrame;
				TIME
			}
		}
		else
		{
			TIME
				// Track points with GPU tracker
				(*tracker)(prevGreyFrame, currentGreyFrame, prevFramePoints, currentFramePoints, status, error, stream);
			// Find the centroid of the new points
			int goodPoints = cv::cuda::countNonZero(status);
			if (goodPoints == numX*numY)
			{
				//log(Status, "Tracking successful with tracker");
				NODE_LOG(info) << "Tracking successful with tracker";
				prevGreyFrame = currentGreyFrame;
				prevFramePoints = currentFramePoints;
				currentFramePoints.download(imagePoints, stream);
				stream.waitForCompletion();
				h_corners = cv::Mat(imagePoints);
				found = true;
			}
			else
			{
				prevFramePoints.release();
			}
			TIME
		}
	}
	else
	{
		//log(Status, "Relying on CPU corner finder routine");
		NODE_LOG(info) << "Relying on CPU corner finder routine";
		currentGreyFrame.download(h_img, stream);
		stream.waitForCompletion();

		found = cv::findChessboardCorners(h_img, cv::Size(numX, numY), imagePoints);
		if (found)
		{
			h_corners = cv::Mat(imagePoints);
			cv::drawChessboardCorners(h_img, cv::Size(numX, numY), imagePoints, found);
			Parameters::UI::UiCallbackService::Instance()->post(boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow), fullTreeName, h_img));
		}
		else
		{
			
			//log(Warning, "Could not find checkerboard pattern");
			NODE_LOG(warning) << "Could not find checkerboard pattern";
		}
	}
	if (!found)
		imagePoints.clear();
	return img;
}
void LoadCameraCalibration::Init(bool firstInit)
{

}
cv::cuda::GpuMat LoadCameraCalibration::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
	return img;
}


void CalibrateCamera::Init(bool firstInit)
{
    updateParameter<boost::function<void(void)>>("Clear", boost::bind(&CalibrateCamera::clear, this));					// 0
    updateParameter<boost::function<void(void)>>("Force calibration", boost::bind(&CalibrateCamera::calibrate, this));	// 1
	updateParameter<boost::function<void(void)>>("Save calibration", boost::bind(&CalibrateCamera::save, this));		// 2

    if(firstInit)
    {
		addInputParameter<ImagePoints>("Image points");					//3 
		addInputParameter<ObjectPoints>("Object points");					// 4
		updateParameter("Min pixel distance", float(10.0));																// 5
		
		updateParameter<Parameters::WriteFile>("Save file", Parameters::WriteFile("Camera Matrix.yml"));
		updateParameter("Enabled", true);
    }
    updateParameterPtr("Image points 2d", &imagePointCollection, Parameters::Parameter::Output);
    updateParameterPtr("Object points 3d", &objectPointCollection, Parameters::Parameter::Output);
	updateParameter("Camera matrix", K, Parameters::Parameter::State);
	updateParameter("Distortion matrix", distortionCoeffs, Parameters::Parameter::State);
    lastCalibration = 0;
}
void CalibrateCamera::save()
{
	auto K = getParameter<cv::Mat>("Camera matrix");
	auto file = getParameter<Parameters::WriteFile>("Save file");
	auto dist = getParameter<cv::Mat>("Distortion matrix");
	if (K && file && dist)
	{
		cv::FileStorage fs;
		fs.open(file->Data()->string(), cv::FileStorage::WRITE);
		if (fs.isOpened())
		{
			fs << "Camera Matrix" << *K->Data();
			fs << "Distortion Matrix" << *dist->Data();
		}
	}
}
void CalibrateCamera::clear()
{
    boost::recursive_mutex::scoped_lock lock(pointCollectionMtx);
    objectPointCollection.clear();
    imagePointCollection.clear();
}
void callibrateCameraThread(std::vector<cv::Mat>& imagePointCollection, std::vector<std::vector<cv::Point3f>>& objectPointCollection)
{
    
}

cv::cuda::GpuMat CalibrateCamera::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
	auto imagePoints = getParameter<ImagePoints>(3)->Data();
    auto objPoints = getParameter<ObjectPoints>(4)->Data();
	if (imagePoints == nullptr || objPoints == nullptr)
    {
        //log(Warning, "Image points or object points not defined");
		NODE_LOG(info) << "Image points or object points not defined";
		return img;
    }

	if (imagePoints->size() == objPoints->size())
	{
		imgSize = img.size(); 
		TIME
			cv::Vec2f centroid(0, 0);
		for (int i = 0; i < imagePoints->size(); ++i)
		{
			centroid += cv::Vec2f((*imagePoints)[i].x, (*imagePoints)[i].y);
		}
		centroid /= float(imagePoints->size());

		float minDist = std::numeric_limits<float>::max();
		for (cv::Vec2f& other : imagePointCentroids)
		{
			float dist = cv::norm(other - centroid);
			if (dist < minDist)
				minDist = dist;
		}
		TIME
		if (minDist > *getParameter<float>("Min pixel distance")->Data())
		{
            //log(Status, "Adding frame to collection");
			NODE_LOG(info) << "Adding frame to collection";
			imagePointCollection.push_back(*imagePoints);
			objectPointCollection.push_back(*objPoints);
			imagePointCentroids.push_back(centroid);

			if (objectPointCollection.size() > lastCalibration + 10 && *getParameter<bool>("Enabled")->Data())
			{
				TIME
					calibrate();
				TIME
            }
			else
			{
				//log(Status, "Waiting for more images before calibration");
				NODE_LOG(info) << "Waiting for more images before calibration";
			}
		}
    }else
    {
        //log(Warning, "# image points doesn't match # object points");
		NODE_LOG(warning) << "# image points doesn't match # object points";
    }
    return img;
}
void CalibrateCamera::calibrate()
{
    //log(Status, "Calibrating camera with " + boost::lexical_cast<std::string>(imagePointCollection.size()) + " images");
	NODE_LOG(info) << "Calibrating camera with " << imagePointCollection.size() << " images";
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    double quality = cv::calibrateCamera(objectPointCollection, imagePointCollection,
                        imgSize, K, distortionCoeffs,rvecs,tvecs);
    if(quality < 1)
    {
        //log(Status, "Sufficient calibration achieved, turning off calibration routine");
		NODE_LOG(info) << "Sufficient calibration achieved, turning off calibration routine";
        enabled = false;
    }
	updateParameter("Camera matrix", K, Parameters::Parameter::State);
    updateParameter("Distortion matrix", distortionCoeffs, Parameters::Parameter::State);
	updateParameter("Reprojection error", quality, Parameters::Parameter::State);
    lastCalibration = objectPointCollection.size();
}

void CalibrateStereoPair::Init(bool firstInit)
{
	updateParameter<boost::function<void(void)>>("Clear", boost::bind(&CalibrateStereoPair::clear, this));
	if (firstInit)
	{
		addInputParameter<ImagePoints>("Camera 1 points");
		addInputParameter<ImagePoints>("Camera 2 points");
		addInputParameter<ObjectPoints>("Object points 3d");

		addInputParameter<cv::Mat>("Camera 1 Matrix");
		addInputParameter<cv::Mat>("Camera 2 matrix");

		addInputParameter<cv::Mat>("Distortion matrix 1");
		addInputParameter<cv::Mat>("Distortion matrix 2");
	}
    //updateParameter<Parameters::WriteFile>("Save file", Parameters::WriteFile("Stereo_calibration.yml"));
    //RegisterParameterCallback("Save file", boost::bind(&CalibrateStereoPair::save, this));
    updateParameter<boost::function<void(void)>>("Save calibration", boost::bind(&CalibrateStereoPair::save, this));
    updateParameterPtr("Rotation matrix", &Rot);
    updateParameterPtr("Translation matrix", &Trans);
    updateParameterPtr("Essential matrix", &Ess);
    updateParameterPtr("Fundamental matrix", &Fun);
    lastCalibration = 0;
    centroidHistory1.set_capacity(20);
    centroidHistory2.set_capacity(20);
}
void CalibrateStereoPair::clear()
{

}
void CalibrateStereoPair::save()
{
    auto K1_ = getParameter<cv::Mat>("Camera 1 Matrix")->Data();
    auto K2_ = getParameter<cv::Mat>("Camera 2 matrix")->Data();
    auto d1_ = getParameter<cv::Mat>("Distortion matrix 1")->Data();
    auto d2_ = getParameter<cv::Mat>("Distortion matrix 2")->Data();

    if (K1_ == nullptr)
        K1_ = &K1;
    if (K2_ == nullptr)
        K2_ = &K2;
    if (d1_ == nullptr)
        d1_ = &dist1;
    if (d2_ == nullptr)
        d2_ = &dist2;
    //Parameters::WriteFile* saveFile = getParameter<Parameters::WriteFile>("Save file")->Data();
    cv::FileStorage fs("StereoCalibration.yml", cv::FileStorage::WRITE);
    fs << "K1" << *K1_;
    fs << "D1" << *d1_;
    fs << "K2" << *K2_;
    fs << "D2" << *d2_;
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

cv::cuda::GpuMat CalibrateStereoPair::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
	auto pts1 = getParameter<ImagePoints>("Camera 1 points")->Data();
	auto pts2 = getParameter<ImagePoints>("Camera 2 points")->Data();
	auto objPts = getParameter<ObjectPoints>("Object points 3d")->Data();

	if (!pts1 || !pts2 || !objPts)
    {
        //log(Warning, "Image points or object points not selected");
		NODE_LOG(warning) << "Image points or object points not selected";
		return img;
    }
	if (pts1->size() != pts2->size() || pts1->size() != objPts->size() || objPts->empty())
    {
        //log(Warning, );
		NODE_LOG(warning) << "Image points not found or not equal to object point size";
		return img;
    }

    // Calculate centroid for each image
    cv::Vec2f centroid1(0,0);
    cv::Vec2f centroid2(0,0);
    for(int i = 0; i < pts1->size(); ++i)
    {
        centroid1 += cv::Vec2f((*pts1)[i].x, (*pts1)[i].y);
        centroid2 += cv::Vec2f((*pts2)[i].x, (*pts2)[i].y);
    }
    centroid1.val[0] /= pts1->size();
    centroid1.val[1] /= pts1->size();
    centroid2.val[0] /= pts2->size();
    centroid2.val[1] /= pts2->size();
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
    if(minDist1 < 100 || minDist2 < 100)
    {
        //log(Status, "Insufficient movement to add points");
		NODE_LOG(info) << "Insufficient movement to add points";
        return img;
    }
    // Check if there is little motion in the image
    cv::Vec2f motionSum1(0,0);
    cv::Vec2f motionSum2(0,0);

    for(int i = 1; i < centroidHistory1.size(); ++i)
    {
        motionSum1 += centroidHistory1[i] - centroidHistory1[i-1];
    }

    for(int i = 1; i < centroidHistory2.size(); ++i)
    {
        motionSum2 += centroidHistory2[i] - centroidHistory2[i-1];
    }
    updateParameter("Motion vector 1", motionSum1);
    updateParameter("Motion vector 2", motionSum2);
    if(cv::norm(motionSum1) > 20 || cv::norm(motionSum2) > 20)
    {
        //log(Status, "Too much movement to record image");
		NODE_LOG(info) << "Too much movement to record image";
        return img;
    }

    //log(Status, "Adding image to collection");
	NODE_LOG(info) << "Adding image to collection";
	imagePointCollection1.push_back(*pts1);
	imagePointCollection2.push_back(*pts2);
	objectPointCollection.push_back(*objPts);
    NODE_LOG(info) << "Image pairs: " << imagePointCollection1.size();

	auto K1_ = getParameter<cv::Mat>("Camera 1 Matrix")->Data();
	auto K2_ = getParameter<cv::Mat>("Camera 2 matrix")->Data();
	auto d1_ = getParameter<cv::Mat>("Distortion matrix 1")->Data();
	auto d2_ = getParameter<cv::Mat>("Distortion matrix 2")->Data();

	if (K1_ == nullptr)
		K1_ = &K1;
	if (K2_ == nullptr)
		K2_ = &K2;
	if (d1_ == nullptr)
		d1_ = &dist1;
	if (d2_ == nullptr)
        d2_ = &dist2;


    if (imagePointCollection1.size() > lastCalibration + 20 &&
		imagePointCollection2.size() == imagePointCollection1.size() && 
		imagePointCollection1.size() == objectPointCollection.size())
	{
        NODE_LOG(info) << "Running calibration with " << objectPointCollection.size() << " image pairs";
		double reprojError = cv::stereoCalibrate(objectPointCollection, imagePointCollection1, imagePointCollection2, *K1_, *d1_, *K2_, *d2_, img.size(), Rot, Trans, Ess, Fun);

        cv::stereoRectify(*K1_, *d1_, *K2_, *d2_, img.size(), Rot,Trans, R1,R2, P1, P2, Q);
        NODE_LOG(info) << "Calibration complete";
		lastCalibration = imagePointCollection1.size();
		updateParameter("Reprojection error", reprojError);
        save();
	}

    return img;
}

void ReadStereoCalibration::Init(bool firstInit)
{
    updateParameter("Calibration file", Parameters::ReadFile("StereoCalibration.yml"));
    updateParameter("K1", K1);
    updateParameter("K2", K2);
    updateParameter("dist1", dist1);
    updateParameter("dist2", dist2);
    updateParameter("Rotation", Rot);
    updateParameter("Translation", Trans);
    updateParameter("Essential", Ess);
    updateParameter("Fundamental", Fun);
    updateParameter("R1", R1);
    updateParameter("R2", R2);
    updateParameter("P1", P1);
    updateParameter("P2", P2);
    updateParameter("Q", Q);
}

cv::cuda::GpuMat ReadStereoCalibration::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(parameters[0]->changed)
    {
        std::string path = getParameter<Parameters::ReadFile>(0)->Data()->string();
        cv::FileStorage fs(path, cv::FileStorage::READ);
        fs["K1"] >> K1;
        fs["K2"] >> K2;
        fs["D1"] >> dist1;
        fs["D2"] >> dist2;
        fs["Rotation"] >> Rot;
        fs["Translation"] >> Trans;
        fs["Essential"] >> Ess;
        fs["Fundamental"] >> Fun;
        fs["R1"] >> R1;
        fs["R2"] >> R2;
        fs["P1"] >> P1;
        fs["P2"] >> P2;
        fs["Q"] >> Q;
        updateParameter("K1", K1);
        updateParameter("K2", K2);
        updateParameter("dist1", dist1);
        updateParameter("dist2", dist2);
        updateParameter("Rotation", Rot);
        updateParameter("Translation", Trans);
        updateParameter("Essential", Ess);
        updateParameter("Fundamental", Fun);
        updateParameter("R1", R1);
        updateParameter("R2", R2);
        updateParameter("P1", P1);
        updateParameter("P2", P2);
        updateParameter("Q", Q);
    }
    return img;
}

void ReadCameraCalibration::Init(bool firstInit)
{
    updateParameter("Calibration file", Parameters::ReadFile("CameraCalibration.yml"));
    updateParameter("Camera Matrix", K);
    updateParameter("Distortion Matrix", dist);
}

cv::cuda::GpuMat ReadCameraCalibration::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(parameters[0]->changed)
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

NODE_DEFAULT_CONSTRUCTOR_IMPL(CalibrateCamera)
NODE_DEFAULT_CONSTRUCTOR_IMPL(CalibrateStereoPair)
NODE_DEFAULT_CONSTRUCTOR_IMPL(FindCheckerboard)
NODE_DEFAULT_CONSTRUCTOR_IMPL(LoadCameraCalibration)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ReadStereoCalibration)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ReadCameraCalibration)
