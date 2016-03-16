#include "rtsp.h"
#include "ObjectInterfacePerModule.h"
#include <signals/logging.hpp>
#include "EagleLib/ParameteredObjectImpl.hpp"
using namespace EagleLib;

std::string frame_grabber_rtsp::frame_grabber_rtsp_info::GetObjectName()
{
    return "frame_grabber_rtsp";
}
std::string frame_grabber_rtsp::frame_grabber_rtsp_info::GetObjectTooltip()
{
    return "";
}
std::string frame_grabber_rtsp::frame_grabber_rtsp_info::GetObjectHelp()
{
    return "";
}
int frame_grabber_rtsp::frame_grabber_rtsp_info::CanLoadDocument(const std::string& document) const
{
    std::string rtsp("rtsp");
    if(document.compare(0, rtsp.length(), rtsp) == 0)
    {
        return 10;
    }
    return 0;
}

int frame_grabber_rtsp::frame_grabber_rtsp_info::LoadTimeout() const
{
    return 3000;
}

int frame_grabber_rtsp::frame_grabber_rtsp_info::Priority() const
{
    return 9;
}
frame_grabber_rtsp::frame_grabber_rtsp():
    frame_grabber_gstreamer()
{
    frame_count = 0;
    _reconnect = false;
}
TS<SyncedMemory> frame_grabber_rtsp::GetNextFrameImpl(cv::cuda::Stream& stream)
{
    if(_reconnect)
        LoadFile();
    try
    {
        if (h_cam)
        {
            cv::Mat h_mat;
            if (h_cam->read(h_mat))
            {
                if (!h_mat.empty())
                {
                    cv::cuda::GpuMat d_mat;
                    d_mat.upload(h_mat, stream);
                    current_frame = TS<SyncedMemory>(h_cam->get(cv::CAP_PROP_POS_MSEC), (int)frame_count++, h_mat, d_mat);
                    return TS<SyncedMemory>(current_frame.timestamp, current_frame.frame_number, current_frame.clone(stream));
                }
            }
        }
    }catch(cv::Exception &e)
    {

    }
    catch(...)
    {
    
    }
    
    return TS<SyncedMemory>();
}
void frame_grabber_rtsp::Init(bool firstInit)
{
    frame_grabber_cv::Init(firstInit);
    _reconnect = false;
    updateParameter<std::function<void(void)>>("Reconnect", [this]()->void
    {
        this->_reconnect = true;
    });
}
bool frame_grabber_rtsp::LoadFile(const std::string& file_path)
{
    //rtspsrc location=rtsp://root:12369pp@192.168.1.52:554/axis-media/media.amp ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink
    std::string file_to_load;
    if(loaded_document.size() && !file_path.size())
        file_to_load = loaded_document;
    else
        file_to_load = file_path;
#ifdef JETSON
    std::string gstreamer_string = "rtspsrc location=" + file_path + " ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink";
#else
    std::string gstreamer_string = "rtspsrc location=" + file_to_load + " ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink";
#endif
	
	h_cam.release();
	LOG(info) << "Attemping to load " << file_to_load;
    LOG(debug) << "Gstreamer string: " << gstreamer_string;
    _reconnect = false;
    playback_frame_number = -1;
	try
	{
		h_cam.reset(new cv::VideoCapture());
		if (h_cam)
		{
            if (h_cam->open(gstreamer_string, cv::CAP_GSTREAMER))
			{
				loaded_document = file_to_load;
				playback_frame_number = h_cam->get(cv::CAP_PROP_POS_FRAMES) + 1;

				return true;
			}
		}
	}
	catch (cv::Exception& e)
	{
	}
	return false;
}

shared_ptr<ICoordinateManager> frame_grabber_rtsp::GetCoordinateManager()
{
    return coordinate_manager;
}
static frame_grabber_rtsp::frame_grabber_rtsp_info info;
REGISTERCLASS(frame_grabber_rtsp, &info);
