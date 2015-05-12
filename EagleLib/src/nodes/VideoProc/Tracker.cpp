#include "nodes/VideoProc/Tracker.h"


using namespace EagleLib;
void KeyFrameTracker::Init(bool firstInit)
{
    if(firstInit)
    {
        auto detector = NodeManager::getInstance().addNode("GoodFeaturesToTrackDetector");
        detector->getParameter<bool>("Enabled")->data = false;
        auto tracker = NodeManager::getInstance().addNode("SparsePyrLKOpticalFlow");
        children.push_back(detector);
        children.push_back(tracker);

    }
}

cv::cuda::GpuMat KeyFrameTracker::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{

}

void CMTTracker::Init(bool firstInit)
{

}

cv::cuda::GpuMat CMTTracker::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{

}

void TLDTracker::Init(bool firstInit)
{

}

cv::cuda::GpuMat TLDTracker::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{

}
