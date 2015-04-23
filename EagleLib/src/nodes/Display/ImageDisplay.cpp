#include "nodes/Display/ImageDisplay.h"

using namespace EagleLib;

NODE_DEFAULT_CONSTRUCTOR_IMPL(QtImageDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(OGLImageDisplay)

QtImageDisplay::QtImageDisplay(boost::function<void(cv::Mat, Node*)> cpuCallback_)
{
    cpuDisplayCallback = cpuCallback_;
}
QtImageDisplay::QtImageDisplay(boost::function<void (cv::cuda::GpuMat, Node*)> gpuCallback_)
{
    gpuDisplayCallback = gpuCallback_;
}
void QtImageDisplay::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("Name", std::string(), Parameter::Control, "Set name for window");
    }
}

cv::cuda::GpuMat
QtImageDisplay::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream stream)
{
    if(img.channels() != 1 && img.channels() != 3)
    {
        log(Warning, "Image has " + boost::lexical_cast<std::string>(img.channels()) + " channels! Cannot display!");
        return img;
    }
    if(img.empty())
        return img;
    if(gpuDisplayCallback)
    {
        gpuDisplayCallback(img, this);
        return img;
    }
    cv::Mat h_img;
    img.download(h_img, stream);
    stream.waitForCompletion();
    if(cpuDisplayCallback)
    {
        cpuDisplayCallback(h_img, this);
        return img;
    }
    std::string name = getParameter<std::string>(0)->data;
    if(name.size() == 0)
    {
        name = fullTreeName;
    }
    try
    {
        cv::imshow(name, h_img);
        cv::waitKey(1);
    }catch(cv::Exception &err)
    {
        log(Warning, err.what());
    }
    parameters[0]->changed = false;
    return img;
}


OGLImageDisplay::OGLImageDisplay(boost::function<void(cv::cuda::GpuMat,Node*)> gpuCallback_)
{
    gpuDisplayCallback = gpuCallback_;
}

void OGLImageDisplay::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("Default Name", std::string("Default Name"), Parameter::Control, "Set name for window");
    }
}

cv::cuda::GpuMat OGLImageDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
{
    if(parameters[0]->changed)
    {
        cv::destroyWindow(prevName);
        prevName = getParameter<std::string>(0)->data;
        parameters[0]->changed = false;
        cv::namedWindow(prevName, cv::WINDOW_OPENGL);
    }
    if(gpuDisplayCallback)
    {
        gpuDisplayCallback(img, this);
        return img;
    }
    cv::namedWindow(prevName, cv::WINDOW_OPENGL);
    try
    {
        cv::imshow(prevName, img);
    }catch(cv::Exception &e)
    {
        log(Error, "This node needs to be run from the UI / main thread. ");
    }

    return img;
}
