#include "nodes/IO/Camera.h"
#include "../remotery/lib/Remotery.h"

using namespace EagleLib;

bool Camera::changeStream(int device)
{
    getParameter<std::string>("Gstreamer stream")->Data()->clear();
    try
    {
		NODE_LOG(info) << "Setting camera to device: " << device;
        cam.release();
        cam = cv::VideoCapture(device);
        return cam.isOpened();
    }catch(cv::Exception &e)
    {
		NODE_LOG(error) << e.what();
        return false;
    }
}
bool Camera::changeStream(const std::string &gstreamParams)
{
	if (!gstreamParams.size())
		return false;
    try
    {
		NODE_LOG(info) << "Setting camera with gstreamer settings: " << gstreamParams;
        cam.release();
        cam = cv::VideoCapture(gstreamParams);
        return cam.isOpened(); 
    }catch(cv::Exception &e)
    {
        //log(Error, e.what());
		NODE_LOG(error) << e.what();
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
        updateParameter<std::string>("Gstreamer stream", "rtsp://root:12369pp@192.168.0.6/axis-media/media.amp");
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
		if (!hostBuf.empty())
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
        Parameters::EnumParameter param;
        param.addEnum(ENUM(v4l2src));
		Parameters::EnumParameter type;
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
    SourceType src = (SourceType)getParameter<Parameters::EnumParameter>(0)->Data()->getValue();
    VideoType encoding = (VideoType)getParameter<Parameters::EnumParameter>(1)->Data()->getValue();

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
        //log(Error, e.what());
		NODE_LOG(error) << e.what();
        return;
    }

	if (cam.isOpened())
	{
		NODE_LOG(info) << "Successfully opened camera";
	}
	else
	{
		NODE_LOG(error) << "Failed to open camera";
	}
		

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
    currentNewestFrame = nullptr;
    hostBuffer.resize(5);
    bufferSize = 5;
    putItr = 0;
    if(firstInit)
    {
		Parameters::EnumParameter param;
        param.addEnum(ENUM(rtspsrc));
		Parameters::EnumParameter type;
        type.addEnum(ENUM(h264));
        updateParameter("Source type", param);
        updateParameter("Source encoding", type);
        updateParameter<std::string>("Source URI", "192.168.1.150");
        updateParameter<unsigned short>("Port", 554);
        updateParameter<std::string>("Path", "/axis-media/media.amp");
        updateParameter<std::string>("Username", "root");
        updateParameter<std::string>("Password", "12369pp");
        updateParameter<unsigned short>("Width", 1920);
        updateParameter<unsigned short>("Height", 1080);
        updateParameter("Output", cv::cuda::GpuMat(), Parameters::Parameter::Output);
    }
	for (auto itr = parameters.begin(); itr != parameters.end(); ++itr)
	{
		(*itr)->changed = false;
	}
}
void RTSPCamera::readImage_thread()
{
	while (!boost::this_thread::interruption_requested())
	{
		if (cam.isOpened())
		{
			rmt_ScopedCPUSample(RTSPCamera_readImage);
			try
			{
				cam.read(hostBuffer[putItr % bufferSize]);
				boost::mutex::scoped_lock lock(mtx);
				notifier.push(&hostBuffer[putItr % bufferSize]);
				++putItr;
				if (putItr == 1000)
					putItr = 0;
			}
			catch (...)
			{
				NODE_LOG(error) << "Error in reading image";
			}
		}
	}    
}
RTSPCamera::~RTSPCamera()
{
    processingThread.interrupt();
    processingThread.join();
}

void RTSPCamera::setString()
{
    processingThread.interrupt();
//    log(Status, "Setting up RTSP Camera");
	NODE_LOG(info) << "Setting up RTSP camera";
    std::stringstream str;
    SourceType src = (SourceType)getParameter<Parameters::EnumParameter>(0)->Data()->getValue();
    VideoType encoding = (VideoType)getParameter<Parameters::EnumParameter>(1)->Data()->getValue();
    std::string result;
	//rtspsrc location=rtsp://root:12369pp@192.168.1.52:554/axis-media/media.amp ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink
    if(src == rtspsrc)
    {
        if(encoding == h264)
        {
			std::string userName = *getParameter<std::string>("Username")->Data();
            std::string pw = *getParameter<std::string>("Password")->Data();
            str << "rtspsrc location=rtsp://";
            if(userName.size())
            {
                str << userName;
                if(pw.size())
                    str << ":" << pw;
                str << "@";
            }
            str << *getParameter<std::string>(2)->Data();       // URI = 192.168.1.150
            str << ":" << *getParameter<unsigned short>(3)->Data();    // Port = 554
            str << *getParameter<std::string>(4)->Data();       // Path = /axis-media/media.amp
            str << " ! ";
            str << "rtph264depay ! h264parse ! ";
#ifdef JETSON
            str << "omxh264dec ! ";
#else
            str << "avdec_h264 ! ";
#endif
            str << "videoconvert ! video/x-raw, ";
            str << "width=" << *getParameter<unsigned short>("Width")->Data();
            str << ", height=" << *getParameter<unsigned short>("Height")->Data();
            str << " ! appsink";
            result = str.str();
            std::cout << result << std::endl;
            updateParameter<std::string>("Gstreamer string", result);
        }

        if(encoding == mjpg)
        {

        }
    }
    processingThread.join();
    cam.release();
    try
    {
		cam = cv::VideoCapture(result, CV_CAP_GSTREAMER);
    }catch(cv::Exception &e)
    {
		NODE_LOG(error) << e.what();
        return;
    }

    if(cam.isOpened())
    {
		NODE_LOG(info) << "Successfully opened camera";
        processingThread = boost::thread(boost::bind(&RTSPCamera::readImage_thread, this));
    }
	else
	{
		NODE_LOG(error) << "Failed to open camera";
	}

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
    cv::cuda::HostMem* data = nullptr;
	
	if (notifier.try_pop(data) &&data && !data->empty())
    {
        output.upload(*data, stream);
        updateParameter("Output", output, Parameters::Parameter::Output);
        return output;
    }
	onUpdate();
    return cv::cuda::GpuMat();
}

bool RTSPCamera::SkipEmpty() const
{
    return false;
}
NODE_DEFAULT_CONSTRUCTOR_IMPL(Camera)
NODE_DEFAULT_CONSTRUCTOR_IMPL(GStreamerCamera)
NODE_DEFAULT_CONSTRUCTOR_IMPL(RTSPCamera)