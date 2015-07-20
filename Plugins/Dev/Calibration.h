#pragma once
#include "nodes/Node.h"

#include "nodes/Node.h"
#ifdef __cplusplus
extern "C"{
#endif
    CV_EXPORTS IPerModuleInterface* GetModule();

#ifdef __cplusplus
}
#endif
namespace EagleLib
{
	typedef std::vector<cv::Point2f> ImagePoints;
	typedef std::vector<cv::Point3f> ObjectPoints;
	class FindCheckerboard : public Node
	{
		ImagePoints imagePoints;
		ObjectPoints objectPoints;
		cv::cuda::GpuMat currentGreyFrame;
		cv::cuda::GpuMat prevFramePoints;
		cv::cuda::GpuMat currentFramePoints;
		cv::cuda::GpuMat prevGreyFrame;
		cv::cuda::GpuMat status;
		cv::cuda::GpuMat error;
		cv::cuda::HostMem h_img;
	public:
		FindCheckerboard();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
	};
	class LoadCameraCalibration : public Node
	{
		cv::Mat K;
		cv::Mat dist;
	public:
		LoadCameraCalibration();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);

	};

    class CalibrateCamera: public Node
    {

        std::vector<ImagePoints> imagePointCollection;
        std::vector<ObjectPoints> objectPointCollection;
        std::vector<cv::Vec2f> imagePointCentroids;
        std::vector<cv::Point3f> objectPoints3d;
		cv::Mat K;
		cv::Mat distortionCoeffs;
        boost::recursive_mutex pointCollectionMtx;
        int lastCalibration;
        cv::Size imgSize;
    public:
		virtual void save();
        virtual void clear();
        virtual void calibrate();
        CalibrateCamera();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class CalibrateStereoPair: public Node
    {
		std::vector<ObjectPoints> objectPointCollection;

        std::vector<ImagePoints> imagePointCollection1;
        std::vector<ImagePoints> imagePointCollection2;
        
		std::vector<cv::Vec2f> imagePointCentroids1;
		std::vector<cv::Vec2f> imagePointCentroids2;

		cv::Mat K1, K2, dist1, dist2, Rot, Trans, Ess, Fun;
        
		int lastCalibration;

    public:
		virtual void clear();
		CalibrateStereoPair();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
