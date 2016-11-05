#pragma once
#include <boost/function.hpp>
#include <boost/any.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <opencv2/core/cuda.hpp>
#include <map>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    typedef boost::function<void(
        cv::cuda::GpuMat,       // reference image
        cv::cuda::GpuMat,       // Current image
        cv::cuda::GpuMat,       // Reference points
        cv::cuda::GpuMat&,      // Tracked points (gets updated by the tracker
        cv::cuda::GpuMat&,      // status output
        cv::cuda::GpuMat&,      // error output
        cv::cuda::Stream&)>      // stream
            TrackSparseFunctor;

    typedef boost::function<void(
        cv::cuda::GpuMat,           // Input image
        cv::cuda::GpuMat,           // Mask
        cv::cuda::GpuMat&,          // Keypoints
        cv::cuda::GpuMat&,          // Descriptors
        cv::cuda::Stream&)>          // Stream
            DetectAndComputeFunctor;

    class CV_EXPORTS Correspondence
    {
        int frameIndex;
        int keyFrameIndex;

    };

    class CV_EXPORTS KeyFrame
    {
        enum VariableType
        {
            Pose = 0,
            CameraMatrix,
            Homography,
            CoordinateSystem,
            KeyPoints,
            Descriptors
        };
        std::map<VariableType, boost::any> data;
        std::map<int, boost::shared_ptr<Correspondence>> correspondences;
    public:
        KeyFrame(cv::cuda::GpuMat img_, int idx_);
        cv::cuda::GpuMat img;
        int frameIndex;

        bool setPose(cv::Mat pose);
        bool setPoseCoordinateSystem(std::string coordinateSyste);
        bool setCamera(cv::Mat cameraMatrix);
        cv::cuda::GpuMat& getKeyPoints();
        cv::cuda::GpuMat& getDescriptors();
        bool setCorrespondene(int otherIdx, cv::cuda::GpuMat thisFramePts, cv::cuda::GpuMat otherFramePts);
        bool getCorrespondence(int otherIdx, cv::cuda::GpuMat& thisFramePts, cv::cuda::GpuMat& otherFramePts);
        bool getCorrespondence(int otherIdx, cv::Mat& homography);
        bool getHomography(int otherIdx, cv::Mat& homography);
        bool hasCorrespondence(int otherIdx);
    };

    struct CV_EXPORTS TrackedFrame
    {
        TrackedFrame(cv::cuda::GpuMat img_, int idx_): keyFrame(img_, idx_){}
        KeyFrame keyFrame;
        cv::cuda::GpuMat trackedPoints;
        cv::cuda::GpuMat status;
        cv::cuda::GpuMat error;
        float trackingQuality;
    };
    struct CV_EXPORTS TrackingResults
    {
        int KeyFrameIdx;
        int TrackedFrameIdx;
        cv::cuda::GpuMat d_keyFramePts;
        cv::cuda::GpuMat d_trackedFramePts;
        cv::cuda::GpuMat d_status;
        cv::cuda::HostMem h_keyFramePts;
        cv::cuda::HostMem h_trackedFramePts;
        cv::cuda::HostMem h_status;
        cv::Mat homography;
        bool preFilter;
        boost::condition_variable cv;
        boost::mutex mtx;
        float quality;
        bool calculated;
    };

    struct TrackedObject
    {

    };
}
