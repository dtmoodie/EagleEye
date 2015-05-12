#include "nodes/Display/ImageDisplay.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
using namespace EagleLib;

NODE_DEFAULT_CONSTRUCTOR_IMPL(QtImageDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(OGLImageDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(KeyPointDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(FlowVectorDisplay)
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
struct UserData
{
    UserData(cv::cuda::HostMem img, QtImageDisplay* node_): displayImage(img), node(node_){}
    cv::cuda::HostMem displayImage;
    QtImageDisplay* node;
};

void QtImageDisplay_cpuCallback(int status, void* userData)
{
    UserData* tmp = (UserData*)userData;
    tmp->node->displayImage(tmp->displayImage);
    delete tmp;
}

void QtImageDisplay::displayImage(cv::cuda::HostMem image)
{
    if(cpuDisplayCallback)
    {
        cpuDisplayCallback(image.createMatHeader(), this);
        return;
    }
    std::string name = getParameter<std::string>(0)->data;
    if(name.size() == 0)
    {
        name = fullTreeName;
    }
    try
    {
        cv::imshow(name, image.createMatHeader());
        cv::waitKey(1);
    }catch(cv::Exception &err)
    {
        log(Warning, err.what());
    }
    parameters[0]->changed = false;
}

cv::cuda::GpuMat
QtImageDisplay::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
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

    img.download(hostImage, stream);
    stream.enqueueHostCallback(QtImageDisplay_cpuCallback, new UserData(hostImage,this));
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

cv::cuda::GpuMat OGLImageDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
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

// This gets called in the user interface thread for drawing and displaying, after data is downloaded from the gpu
cv::Mat KeyPointDisplay::uicallback()
{
    if(displayType == 0)
    {
        EventBuffer<std::pair<cv::cuda::HostMem,cv::cuda::HostMem>>* buffer = hostData.waitBack();
        cv::Scalar color = getParameter<cv::Scalar>(3)->data;
        int radius = getParameter<int>(2)->data;
        if(buffer)
        {
            cv::Mat keyPoints = buffer->data.first.createMatHeader();
            cv::Mat hostImage = buffer->data.second.createMatHeader();
            cv::Vec2f* pts = keyPoints.ptr<cv::Vec2f>(0);
            for(int i = 0; i < keyPoints.cols; ++i, ++pts)
            {
                cv::circle(hostImage, cv::Point(pts->val[0], pts->val[1]), radius, color, 1);
            }
            return hostImage;
        }
        return cv::Mat();
    }
    return cv::Mat();
}


void KeyPointDisplay_callback(int status, void* userData)
{
    KeyPointDisplay* node = (KeyPointDisplay*)userData;
    if(node->uiThreadCallback)
        return node->uiThreadCallback(boost::bind(&KeyPointDisplay::uicallback,node), node);
    cv::Mat img = node->uicallback();
    try
    {
        if(!img.empty())
            cv::imshow(node->fullTreeName, img);
    }catch(cv::Exception &e)
    {
        std::cout << e.what() << std::endl;
    }

}

void KeyPointDisplay::Init(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Device keypoints");
        addInputParameter<cv::Mat>("Host keypoints");
        updateParameter("Radius", int(5));
        updateParameter("Color", cv::Scalar(255,0,0));
        hostData.resize(20);
        displayType = -1;
    }
}
void KeyPointDisplay::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(hostData);
}
cv::cuda::GpuMat KeyPointDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* d_mat = getParameter<cv::cuda::GpuMat*>(0)->data;

    if(d_mat)
    {
        auto buffer = hostData.getFront();
        d_mat->download(buffer->data.first, stream);
        img.download(buffer->data.second, stream);
        buffer->fillEvent.record(stream);
        stream.enqueueHostCallback(KeyPointDisplay_callback, this);
        displayType = 0;
        return img;
    }
    auto h_mat = getParameter<cv::Mat*>(1)->data;
    if(h_mat)
    {

    }
    return img;
}
void FlowVectorDisplay_callback(int status, void* userData)
{
    FlowVectorDisplay* node = (FlowVectorDisplay*)userData;
    if(node)
    {
        if(node->uiThreadCallback)
        {
            node->uiThreadCallback(boost::bind(&FlowVectorDisplay::uicallback, node), node);
            return;
        }
        cv::Mat img = node->uicallback();
        if(!img.empty())
            cv::imshow(node->fullTreeName, img);
    }
}

void FlowVectorDisplay::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(hostData);
}

void FlowVectorDisplay::Init(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Device initial poitns");
        addInputParameter<cv::cuda::GpuMat>("Device current points");
        addInputParameter<cv::cuda::GpuMat>("Point mask");
        updateParameter("Good Color", cv::Scalar(0,255,0));
        updateParameter("Bad Color", cv::Scalar(0,0,255));
    }
}

cv::cuda::GpuMat FlowVectorDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* d_initial = getParameter<cv::cuda::GpuMat*>(0)->data;
    cv::cuda::GpuMat* d_final = getParameter<cv::cuda::GpuMat*>(1)->data;
    cv::cuda::GpuMat* d_mask = getParameter<cv::cuda::GpuMat*>(2)->data;
    if(d_initial && d_final)
    {
        auto buffer = hostData.getFront();
        img.download(buffer->data[0], stream);
        if(d_mask && !d_mask->empty())
            d_mask->download(buffer->data[1], stream);
        d_initial->download(buffer->data[2], stream);
        d_final->download(buffer->data[3], stream);
        buffer->fillEvent.record(stream);
        stream.enqueueHostCallback(FlowVectorDisplay_callback, this);
    }
    return img;
}

cv::Mat FlowVectorDisplay::uicallback()
{
    EventBuffer<cv::cuda::HostMem[4]>* buffer = hostData.waitBack();
    if(buffer)
    {
        cv::Mat img = buffer->data[0].createMatHeader();
        cv::Mat mask = buffer->data[1].createMatHeader();
        cv::Mat initial = buffer->data[2].createMatHeader();
        cv::Mat final = buffer->data[3].createMatHeader();
        cv::Scalar color = getParameter<cv::Scalar>(3)->data;
        // Iterate through all points and draw them vectors
        if(mask.empty())
        {
            for(int i = 0; i < final.cols; ++i)
            {
                cv::line(img, initial.at<cv::Point2f>(i), final.at<cv::Point2f>(i),color);
            }
        }
        return img;
    }
    return cv::Mat();
}
