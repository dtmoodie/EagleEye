#include "nodes/IO/Camera.h"

using namespace EagleLib;

bool Camera::changeStream(int device)
{
    try
    {
        log(Status, "Setting camera to device: " + boost::lexical_cast<std::string>(device));
        cam.release();
        cam = cv::VideoCapture(device);
        return cam.isOpened();
    }catch(cv::Exception &e)
    {
        log(Error, e.what());
        return false;
    }

}
bool Camera::changeStream(const std::string &gstreamParams)
{
    try
    {
        log(Status, "Setting camera with gstreamer settings: " + gstreamParams);
        cam.release();
        cam = cv::VideoCapture(gstreamParams);
        return cam.isOpened(); 
    }catch(cv::Exception &e)
    {
        log(Error, e.what());
        return false;
    }

}
Camera::~Camera()
{
//    acquisitionThread.interrupt();
//    acquisitionThread.join();
}

void Camera::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        updateParameter<int>("Camera Number", 0);
        updateParameter<std::string>("Gstreamer stream ", "rtsp://root:12369pp@192.168.0.6/axis-media/media.amp");
        parameters[0]->changed = false;
        parameters[1]->changed = false;
    }
}
void Camera::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(cam);
}

cv::cuda::GpuMat Camera::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(parameters[0]->changed)
    {
        parameters[0]->changed = false;
        changeStream(*getParameter<int>(0)->Data());
    }
    if(parameters[1]->changed)
    {
        parameters[1]->changed = false;
        changeStream(*getParameter<std::string>(1)->Data());
    }
    if(cam.isOpened())
    {
		cam.set(cv::CAP_PROP_ZOOM, 20); 
        cam.read(hostBuf);
        img.upload(hostBuf,stream);
    }
    updateParameter("Output", img, Parameters::Parameter::Output);
    return img;
}
bool Camera::SkipEmpty() const
{
    return false;
}
void GStreamerCamera::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        Parameters::Parameter::EnumParameter param;
        param.addEnum(ENUM(v4l2src));
		Parameters::Parameter::EnumParameter type;
        type.addEnum(ENUM(h264));
        updateParameter("Source type", param);
        updateParameter("Source encoding", type);
        updateParameter<std::string>("Source", "/dev/video0");
        updateParameter("Width", int(1920));
        updateParameter("Height", int(1080));
        updateParameter<std::string>("Framerate", "30/1");
        updateParameter("Queue", true);
        updateParameter<std::string>("Username", "");
        updateParameter<std::string>("Password", "");
        setString();
        //updateParameter("Gstreamer string", "v4l2src device=/dev/video0 ! video/x-h264, width=1920, height=1080, framerate=30/1 ! queue ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink");
    }
}
void GStreamerCamera::setString()
{
    std::stringstream str;
    SourceType src = (SourceType)getParameter<Parameters::Parameter::EnumParameter>(0)->Data()->getValue();
    VideoType encoding = (VideoType)getParameter<Parameters::Parameter::EnumParameter>(1)->Data()->getValue();

    switch(src)
    {
    case v4l2src:
        str << "v4l2src ";
        break;
    case rtspsrc:
        str << "rtspsrc ";
        break;
    }
    if(src == v4l2src)
        
    if(src == rtspsrc)
    {
        std::string userName = *getParameter<std::string>(7)->Data();
        std::string pass = *getParameter<std::string>(8)->Data();
        if(userName.size() && pass.size())
        {
            str << "location=rtsp://" << userName << ":" << pass << "@" << *getParameter<std::string>(2)->Data() << " ! ";
        }else
        {
            str << "location=rtsp://" << *getParameter<std::string>(2)->Data() << " ! ";
        }
        if(encoding == h264)
        {
            str << " rtph264depay ! avdec_h264 ! ";
        }
    }
    
    if(src == v4l2src)
    {
        str << "device=" << *getParameter<std::string>(2)->Data() << " ! ";
        if(encoding == h264)
            str << "video/x-h264, width=";
        str << *getParameter<int>(3)->Data();
        str << ", height=";
        str << *getParameter<int>(4)->Data();
        str << ", framerate=";
        str << *getParameter<std::string>(5)->Data();
        str << " ! ";
    }


    if(*getParameter<bool>(6)->Data())
        str << "queue ! ";
    if(encoding == h264 && src == v4l2src)
    {

        str << "h264parse ! avdec_h264 ! videoconvert ! video/x-raw, width=";
        str << *getParameter<int>(3)->Data();
        str << ", height=";
        str << *getParameter<int>(4)->Data();
    }
    str << " ! appsink";
    std::string result = str.str();
    std::cout << result << std::endl;
    // "v4l2src device=/dev/video0 ! video/x-h264, width=1920, height=1080, framerate=30/1 ! queue ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink"
    updateParameter<std::string>("Gstreamer string", result);
    cam.release();
    try
    {
        cam.open(result);
    }catch(cv::Exception &e)
    {
        log(Error, e.what());
        return;
    }

    if(cam.isOpened())
        log(Status, "Successfully opened camera");
    else
        log(Error, "Failed to open camera");

    for(size_t i = 0; i < parameters.size(); ++i)
    {
        parameters[i]->changed = false;
    }
}

// rtsp://192.168.10.35:554/axis-media/media.amp?videocodec=h264

cv::cuda::GpuMat GStreamerCamera::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(parameters[0]->changed ||
        parameters[1]->changed ||
        parameters[2]->changed ||
        parameters[3]->changed ||
        parameters[4]->changed ||
        parameters[5]->changed ||
        parameters[6]->changed ||
        parameters[7]->changed ||
        parameters[8]->changed)
    {
        setString();
    }
    if(cam.isOpened())
    {
        if(cam.read(hostBuf))
        {
            img.upload(hostBuf,stream);
        }

    }
    updateParameter("Output", img, Parameters::Parameter::Output);
    return img;
}
bool GStreamerCamera::SkipEmpty() const
{
    return false;
}
void RTSPCamera::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
		Parameters::Parameter::EnumParameter param;
        param.addEnum(ENUM(rtspsrc));
		Parameters::Parameter::EnumParameter type;
        type.addEnum(ENUM(h264));
        updateParameter("Source type", param);
        updateParameter("Source encoding", type);
        updateParameter<std::string>("Source", "192.168.10.35:554/axis-media/media.amp");
        updateParameter<std::string>("Username", "");
        updateParameter<std::string>("Password", "");
        setString();
        //updateParameter("Gstreamer string", "v4l2src device=/dev/video0 ! video/x-h264, width=1920, height=1080, framerate=30/1 ! queue ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink");
    }
}
void RTSPCamera::setString()
{
    std::stringstream str;
    SourceType src = (SourceType)getParameter<Parameters::Parameter::EnumParameter>(0)->Data()->getValue();
    VideoType encoding = (VideoType)getParameter<Parameters::Parameter::EnumParameter>(1)->Data()->getValue();
    std::string result;
    if(src == rtspsrc)
    {
        if(encoding == h264)
        {
			std::string userName = *getParameter<std::string>("Username")->Data();
            std::string pw = *getParameter<std::string>("Password")->Data();
            str << "rtspsrc location=rtsp://";
            if(userName.size())
                str << userName << ":";
            if(pw.size())
                str << pw << "@";
            str << *getParameter<std::string>(2)->Data() << " ! ";
            str << "rtph264depay ! ";
            str << "avdec_h264 ! ";
            str << "appsink";
            result = str.str();
            std::cout << result << std::endl;
            updateParameter<std::string>("Gstreamer string", result);
        }

        if(encoding == mjpg)
        {

        }
    }
    cam.release();
    try
    {
        cam.open(result);
    }catch(cv::Exception &e)
    {
        log(Error, e.what());
        return;
    }

    if(cam.isOpened())
        log(Status, "Successfully opened camera");
    else
        log(Error, "Failed to open camera");

    for(size_t i = 0; i < parameters.size(); ++i)
    {
        parameters[i]->changed = false;
    }
}

cv::cuda::GpuMat RTSPCamera::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    if(parameters[0]->changed ||
       parameters[1]->changed ||
       parameters[2]->changed ||
       parameters[3]->changed ||
       parameters[4]->changed)
    {
        setString();
    }
    if(cam.isOpened())
    {
        if(cam.read(hostBuf))
        {
            img.upload(hostBuf,stream);
        }
    }
    updateParameter("Output", img, Parameters::Parameter::Output);
    return img;

}

bool RTSPCamera::SkipEmpty() const
{
    return false;
}
NODE_DEFAULT_CONSTRUCTOR_IMPL(Camera)
NODE_DEFAULT_CONSTRUCTOR_IMPL(GStreamerCamera)
NODE_DEFAULT_CONSTRUCTOR_IMPL(RTSPCamera)
