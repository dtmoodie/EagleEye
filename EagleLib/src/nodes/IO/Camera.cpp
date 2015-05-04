#include "nodes/IO/Camera.h"

using namespace EagleLib;

void Camera::changeStream(int device)
{
    cam = cv::VideoCapture(device);
    cv::Mat img;
    if(cam.read(img))
    {
        hostBuffer = cv::cuda::HostMem(img.size(),img.type());
    }

}
void Camera::changeStream(const std::string &gstreamParams)
{

}

void Camera::Init(bool firstInit)
{
    updateParameter<int>("Camera Number", 0);
    updateParameter<std::string>("Gstreamer stream ", "");
    changeStream(0);
}

cv::cuda::GpuMat Camera::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
{
    if(parameters[0]->changed)
        changeStream(getParameter<int>(0)->data);
    if(parameters[1]->changed)
        changeStream(getParameter<std::string>(1)->data);

    cam.read(hostBuffer.createMatHeader());
    img.upload(hostBuffer,stream);

    return img;
}
bool Camera::SkipEmpty() const
{
    return false;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(Camera)
