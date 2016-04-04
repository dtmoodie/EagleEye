#include "Freenect.h"
#include "libfreenect/libfreenect.hpp"
#include "freenect.cuh"
#include "EagleLib/rcc/SystemTable.hpp"
#include <boost/lexical_cast.hpp>

using namespace EagleLib;
IPerModuleInterface* GetModule()
{
    return PerModuleInterface::GetInstance();
}

class MyFreenectDevice : public Freenect::FreenectDevice
{
public:
    MyFreenectDevice(freenect_context *_ctx, int _index)
        : Freenect::FreenectDevice(_ctx, _index),
          m_new_rgb_frame(false), m_new_depth_frame(false)
    {
        setDepthFormat(FREENECT_DEPTH_REGISTERED);
		auto video_mode = freenect_find_video_mode(FREENECT_RESOLUTION_HIGH, FREENECT_VIDEO_RGB);
		auto depth_mode = freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED);
		m_buffer_video.create(video_mode.height, video_mode.width, CV_8UC3);
		m_buffer_depth.create(depth_mode.height, depth_mode.width, CV_16U);
    }

    // Do not call directly, even in child
    void VideoCallback(void *_rgb, uint32_t timestamp)
    {
        std::lock_guard<std::mutex> lock(m_rgb_mutex);
        uint8_t* rgb = static_cast<uint8_t*>(_rgb);
        memcpy(rgb, m_buffer_video.data, getVideoBufferSize());
        m_new_rgb_frame = true;
		video_timestamp = timestamp;
    }

    // Do not call directly, even in child
    void DepthCallback(void *_depth, uint32_t timestamp)
    {
        std::lock_guard<std::mutex> lock(m_depth_mutex);
        uint16_t* depth = static_cast<uint16_t*>(_depth);
        memcpy(depth, m_buffer_depth.data, getDepthBufferSize());
        m_new_depth_frame = true;
		depth_timestamp = timestamp;
    }

    bool getRGB(cv::Mat h_buffer, uint32_t& timestamp)
    {
        std::lock_guard<std::mutex> lock(m_rgb_mutex);

        if (!m_new_rgb_frame)
            return false;

		m_buffer_video.copyTo(h_buffer);
		timestamp = video_timestamp;
        m_new_rgb_frame = false;

        return true;
    }

    bool getDepth(cv::Mat h_buffer, uint32_t& timestamp)
    {
        std::lock_guard<std::mutex> lock(m_depth_mutex);

        if (!m_new_depth_frame)
            return false;
		m_buffer_depth.copyTo(h_buffer);
		timestamp = depth_timestamp;
        m_new_depth_frame = false;

        return true;
    }

private:
    std::mutex m_rgb_mutex;
    std::mutex m_depth_mutex;
	cv::Mat m_buffer_video;
	cv::Mat m_buffer_depth;
	uint32_t video_timestamp;
	uint32_t depth_timestamp;
    bool m_new_rgb_frame;
    bool m_new_depth_frame;
};

using namespace EagleLib;

freenect::~freenect()
{
}
bool freenect::LoadFile(const std::string& file_path)
{
	auto idx = file_path.find("freenect/");
	if (idx != std::string::npos)
	{
		auto substr = file_path.substr(idx + 9);
		auto systemTable = PerModuleInterface::GetInstance()->GetSystemTable();
		auto freenect = systemTable->GetSingleton<Freenect::Freenect>();
		if (!freenect)
		{
			freenect = new Freenect::Freenect();
			systemTable->SetSingleton<Freenect::Freenect>(freenect);
		}
		try
		{
			_myDevice = &freenect->createDevice<MyFreenectDevice>(boost::lexical_cast<int>(substr));
		}
		catch (std::runtime_error & e)
		{
			LOG(error) << e.what();
			_myDevice = nullptr;
			return false;
		}
		_myDevice->startVideo();
		_myDevice->startDepth();
		return true;
	}
	return false;
}
rcc::shared_ptr<ICoordinateManager> freenect::GetCoordinateManager()
{
	return _coordinate_manager;
}
TS<SyncedMemory> freenect::GetFrameImpl(int index, cv::cuda::Stream& stream)
{
	return TS<SyncedMemory>();
}
TS<SyncedMemory> freenect::GetNextFrameImpl(cv::cuda::Stream& stream)
{
	if (_myDevice)
	{
		cv::Mat depth;
		uint32_t timestamp;
		_myDevice->getDepth(depth, timestamp);
		return TS<SyncedMemory>((double)timestamp, (int)timestamp, depth);
	}
	return TS<SyncedMemory>();
}

void freenect::Serialize(ISimpleSerializer* pSerializer)
{
	SERIALIZE(_myDevice);
	SERIALIZE(_freenect);
}
std::string freenect::frame_grabber_freenect_info::GetObjectName()
{
	return "freenect";
}
std::string freenect::frame_grabber_freenect_info::GetObjectTooltip()
{
	return "";
}
std::string freenect::frame_grabber_freenect_info::GetObjectHelp()
{
	return "";
}
int freenect::frame_grabber_freenect_info::CanLoadDocument(const std::string& document) const
{
	if (document.find("freenect/") != std::string::npos)
	{
		return 10;
	}
	return 0;
}
int freenect::frame_grabber_freenect_info::Priority() const
{
	return 0;
}
int freenect::frame_grabber_freenect_info::LoadTimeout() const
{
	return 1000;
}

static freenect::frame_grabber_freenect_info info;
REGISTERCLASS(freenect, &info);