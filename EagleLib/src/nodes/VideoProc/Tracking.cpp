#include "nodes/VideoProc/Tracking.hpp"

using namespace EagleLib;

KeyFrame::KeyFrame(cv::cuda::GpuMat img_, int idx_):
    img(img_), frameIndex(idx_){}

bool KeyFrame::setPose(cv::Mat pose)
{
    return false;
}

bool KeyFrame::setPoseCoordinateSystem(std::string coordinateSyste)
{
    return false;
}

bool KeyFrame::setCamera(cv::Mat cameraMatrix)
{
    return false;
}

bool KeyFrame::setCorrespondene(int otherIdx, cv::cuda::GpuMat thisFramePts, cv::cuda::GpuMat otherFramePts)
{
    return false;
}

bool KeyFrame::getCorrespondence(int otherIdx, cv::cuda::GpuMat& thisFramePts, cv::cuda::GpuMat& otherFramePts)
{

    return false;
}

bool KeyFrame::getCorrespondence(int otherIdx, cv::Mat& homography)
{
    return false;
}

bool KeyFrame::getHomography(int otherIdx, cv::Mat& homography)
{
    auto itr = correspondences.find(otherIdx);
    if(itr == correspondences.end())
        return false;

}

bool KeyFrame::hasCorrespondence(int otherIdx)
{
    return correspondences.find(otherIdx) != correspondences.end();
}
