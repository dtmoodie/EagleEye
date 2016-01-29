#pragma once
#include "EagleLib/nodes/Node.h"
#include "EagleLib/rcc/external_includes/cv_videoio.hpp"
#include <EagleLib/utilities/CudaUtils.hpp>

#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    namespace Nodes
    {
    enum SourceType
    {
        v4l2src = 0,
        rtspsrc = 1
    };
    enum VideoType
    {
        h264 = 0,
        mjpg = 1
    };
    class Camera: public Node
    {
        virtual bool changeStream(const std::string& gstreamParams);
        virtual bool changeStream(int device);
        cv::VideoCapture cam;
        
		virtual void read_image();
		boost::thread read_thread;
		EagleLib::concurrent_notifier<cv::cuda::GpuMat> notifier;
    public:
        Camera();
        ~Camera();
        virtual void Init(bool firstInit);
        virtual void Serialize(ISimpleSerializer *pSerializer);
        virtual bool SkipEmpty() const;
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    class GStreamerCamera: public Node
    {
        cv::VideoCapture cam;
        cv::cuda::HostMem hostBuf;

        void setString();
    public:
        GStreamerCamera();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
        virtual bool SkipEmpty() const;
    };

    class CV_EXPORTS RTSPCamera: public Node
    {
        //cv::cuda::GpuMat output;
        cv::VideoCapture cam;
        int putItr;
        int bufferSize;
        //std::vector<cv::cuda::HostMem> hostBuffer;
        concurrent_notifier<cv::Mat> notifier;
        cv::cuda::HostMem* currentNewestFrame;
        boost::mutex mtx;
        boost::thread processingThread;

        void setString();
        void readImage_thread();
    public:
        RTSPCamera();
        ~RTSPCamera();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
        virtual bool SkipEmpty() const;

    };
    }
}
