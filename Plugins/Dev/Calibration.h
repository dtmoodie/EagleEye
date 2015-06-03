#pragma once


#include "nodes/Node.h"

namespace EagleLib
{
    class CalibrateCamera: public Node
    {
        cv::cuda::HostMem h_img;
        std::vector<std::vector<cv::Point2f>> imagePointCollection;
        std::vector<std::vector<cv::Point3f>> objectPointCollection;
        std::vector<cv::Point3f> objectPoints3d;
        boost::recursive_mutex pointCollectionMtx;
    public:
        virtual void clear();
        CalibrateCamera();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };

    class CalibrateStereoPair: public Node
    {
    public:
        CalibrateStereoPair();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
