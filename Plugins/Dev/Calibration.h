#pragma once


#include "nodes/Node.h"
#ifdef __cplusplus
extern "C"{
#endif
    CV_EXPORTS IPerModuleInterface* CALL GetModule();

#ifdef __cplusplus
}
#endif
namespace EagleLib
{
    class CalibrateCamera: public Node
    {

        cv::cuda::GpuMat prevFramePoints;
        cv::cuda::GpuMat currentFramePoints;
        cv::cuda::GpuMat prevGreyFrame;
        cv::cuda::GpuMat currentGreyFrame;
        cv::cuda::GpuMat status;
        cv::cuda::GpuMat error;
        cv::cuda::HostMem h_img;
        cv::cuda::HostMem corners;
        std::vector<cv::Mat> imagePointCollection;
        std::vector<std::vector<cv::Point3f>> objectPointCollection;
        std::vector<cv::Vec2f> imagePointCentroids;
        std::vector<cv::Point3f> objectPoints3d;
        boost::recursive_mutex pointCollectionMtx;
        int lastCalibration;
        cv::Size imgSize;
    public:
        virtual void clear();
        virtual void calibrate();
        CalibrateCamera();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class CalibrateStereoPair: public Node
    {

        std::vector<cv::Mat> imagePointCollection1;
        std::vector<cv::Mat> imagePointCollection2;
        std::vector<std::vector<cv::Point3f>> objectPointCollection;
        std::vector<cv::Vec2f> imagePointCentroids;
        std::vector<cv::Point3f> objectPoints3d;
        boost::recursive_mutex pointCollectionMtx;
		int lastCalibration;

    public:
		virtual void clear();
		CalibrateStereoPair();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
