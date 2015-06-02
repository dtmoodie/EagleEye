#include "nodes/Display/ImageDisplay.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
using namespace EagleLib;

NODE_DEFAULT_CONSTRUCTOR_IMPL(QtImageDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(OGLImageDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(KeyPointDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(FlowVectorDisplay)
NODE_DEFAULT_CONSTRUCTOR_IMPL(HistogramDisplay)
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
    Node::Init(firstInit);
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
    //tmp->node->displayImage(tmp->displayImage);
    EagleLib::UIThreadCallback::getInstance().addCallback(boost::bind(&QtImageDisplay::displayImage, tmp->node, tmp->displayImage));
    delete tmp;
}

void QtImageDisplay::displayImage(cv::cuda::HostMem image)
{
//    if(cpuDisplayCallback)
//    {
//        cpuDisplayCallback(image.createMatHeader(), this);
//        return;
//    }
    std::string name = getParameter<std::string>(0)->data;
    if(name.size() == 0)
    {
        name = fullTreeName;
    }
    try
    {
        cv::imshow(name, image.createMatHeader());
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
    Node::Init(firstInit);
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
    Node::Init(firstInit);
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
            cv::imshow(node->displayName, img);
    }
}

void FlowVectorDisplay::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(hostData);
}

void FlowVectorDisplay::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Device initial poitns");
        addInputParameter<cv::cuda::GpuMat>("Device current points");
        addInputParameter<cv::cuda::GpuMat>("Point mask");
        updateParameter("Good Color", cv::Scalar(0,255,0));
        updateParameter("Bad Color", cv::Scalar(0,0,255));
        updateParameter<boost::function<void(cv::cuda::GpuMat, cv::cuda::GpuMat, cv::cuda::GpuMat, cv::cuda::GpuMat, std::string&, cv::cuda::Stream)>>
                ("Display functor", boost::bind(&FlowVectorDisplay::display, this, _1, _2, _3, _4, _5, _6));
    }
}
void FlowVectorDisplay::display(cv::cuda::GpuMat img, cv::cuda::GpuMat initial,
                                cv::cuda::GpuMat final, cv::cuda::GpuMat mask,
                                std::string &name, cv::cuda::Stream stream)
{
    displayName = name;
    auto buffer = hostData.getFront();
    img.download(buffer->data[0], stream);
    if(!mask.empty())
        mask.download(buffer->data[1], stream);
    initial.download(buffer->data[2], stream);
    final.download(buffer->data[3], stream);
    buffer->fillEvent.record(stream);
    stream.enqueueHostCallback(FlowVectorDisplay_callback, this);
}

cv::cuda::GpuMat FlowVectorDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* d_initial = getParameter<cv::cuda::GpuMat*>(0)->data;
    cv::cuda::GpuMat* d_final = getParameter<cv::cuda::GpuMat*>(1)->data;
    cv::cuda::GpuMat* d_mask = getParameter<cv::cuda::GpuMat*>(2)->data;
    if(d_initial && d_final)
    {
        display(img, *d_initial, *d_final, d_mask ? *d_mask : cv::cuda::GpuMat(), fullTreeName, stream);
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
void histogramDisplayCallback(int status, void* userData)
{
    HistogramDisplay* node = (HistogramDisplay*)userData;
    UIThreadCallback::getInstance().addCallback(boost::bind(&HistogramDisplay::displayHistogram, node));

}
void HistogramDisplay::displayHistogram()
{
    cv::cuda::HostMem* dataPtr = histograms.getBack();
    cv::Mat data = dataPtr->createMatHeader();
    if(data.channels() != 1)
    {
        log(Error, "Currently only supports 1 channel histograms, input has " + boost::lexical_cast<std::string>(data.channels()) + " channels");
        return;
    }
    double minVal, maxVal;
    int minIdx, maxIdx;
    cv::minMaxIdx(data, &minVal, &maxVal, &minIdx, &maxIdx);
    cv::Mat img(100, data.cols*5,CV_8U, cv::Scalar(0));
    updateParameter("Min value", minVal, Parameter::State);
    updateParameter("Min bin", minIdx, Parameter::State);
    updateParameter("Max value", maxVal, Parameter::State);
    updateParameter("Max bin", maxIdx, Parameter::State);
    for(int i = 0; i < data.cols; ++i)
    {
        double height = data.at<int>(i);
        height -= minVal;
        height /=(maxVal - minVal);
        height *= 100;
        cv::rectangle(img, cv::Rect(i*5, 100 - (int)height, 5, 100), cv::Scalar(255),-1);
    }
    cv::imshow(fullTreeName, img);

}

void HistogramDisplay::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Input");
    }
}

cv::cuda::GpuMat HistogramDisplay::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* input = getParameter<cv::cuda::GpuMat*>(0)->data;
    if(input)
    {
        if(input->rows == 1 || input->cols == 1)
        {
            input->download(*histograms.getFront(), stream);
            stream.enqueueHostCallback(histogramDisplayCallback,this);
            return img;
        }
    }
    if(img.rows == 1 || img.cols == 1)
    {
        img.download(*histograms.getFront(), stream);
        stream.enqueueHostCallback(histogramDisplayCallback,this);
    }
    // Currently assuming the input image is a histogram
    return img;
}
