#include "EagleLib/nodes/Sink.h"

#include <EagleLib/rcc/external_includes/cv_highgui.hpp>
#include <EagleLib/rcc/external_includes/cv_core.hpp>
#include <opencv2/core/opengl.hpp>
#include <EagleLib/utilities/CudaUtils.hpp>
#include <EagleLib/ObjectDetection.hpp>

namespace EagleLib
{
    namespace Nodes
    {
    class QtImageDisplay: public CpuSink
    {
        std::string prevName;
    public:
        QtImageDisplay();
        virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream);
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual void doProcess(const cv::Mat& mat, double timestamp, int frame_number, cv::cuda::Stream& stream);
    };
    class OGLImageDisplay: public Node
    {
        std::string prevName;
    public:
        OGLImageDisplay();

        
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    };
    class KeyPointDisplay: public Node
    {
        //ConstEventBuffer<std::pair<cv::cuda::HostMem, cv::cuda::HostMem>> hostData;
        int displayType;
    public:

        KeyPointDisplay();
        virtual void NodeInit(bool firstInit);
        TS<SyncedMemory> doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        cv::Mat uicallback();
        virtual void Serialize(ISimpleSerializer *pSerializer);
    };
    class FlowVectorDisplay: public Node
    {
        // First buffer is the image, second is a pair of the points to be used
        //ConstEventBuffer<cv::cuda::HostMem[4]> hostData;
        //void display(cv::cuda::GpuMat img, cv::cuda::GpuMat initial, cv::cuda::GpuMat final, cv::cuda::GpuMat mask, std::string& name, cv::cuda::Stream);
    public:
        std::string displayName;
        FlowVectorDisplay();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        //cv::Mat uicallback();
        virtual void Serialize(ISimpleSerializer *pSerializer);
    };

    class HistogramDisplay: public Node
    {

    public:
        ConstBuffer<cv::cuda::HostMem> histograms;
        void displayHistogram();
        HistogramDisplay();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    class DetectionDisplay: public Node
    {
    public:
        ConstBuffer<std::pair<cv::cuda::HostMem, std::vector<DetectedObject>>> hostData;
        void displayCallback();
        DetectionDisplay();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    } // namespace Nodes
} // namespace EagleLib
