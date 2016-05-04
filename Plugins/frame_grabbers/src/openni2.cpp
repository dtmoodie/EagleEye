#include "openni2.h"

using namespace EagleLib;

frame_grabber_openni2_info::frame_grabber_openni2_info()
{

}
std::string frame_grabber_openni2_info::GetObjectName()
{
	return "frame_grabber_openni2";
}
int frame_grabber_openni2_info::CanLoadDocument(const std::string& document) const
{
	std::string doc = document;
	std::transform(doc.begin(), doc.end(), doc.begin(), ::tolower);
	if (doc == "openni2")
	{
		return 10;
	}
	try
	{
		int index = boost::lexical_cast<int>(doc);
		if (index == cv::CAP_OPENNI2 || index == cv::CAP_OPENNI2_ASUS)
		{
			return 10;
		}
	}
	catch (boost::bad_lexical_cast &e)
	{
	}
	return 0;
}

int frame_grabber_openni2_info::Priority() const
{
	return 0;
}

bool frame_grabber_openni2::LoadFile(const std::string& file_path)
{
	std::string doc = file_path;
	std::transform(doc.begin(), doc.end(), doc.begin(), ::tolower);
	if (doc == "openni2")
	{
		return h_LoadFile("1600");
	}
	try
	{
		int index = boost::lexical_cast<int>(doc);
		if (index == cv::CAP_OPENNI2 || index == cv::CAP_OPENNI2_ASUS)
		{
			return h_LoadFile(doc);
		}
	}
	catch (boost::bad_lexical_cast &e)
	{
	}
	return false;
}

TS<SyncedMemory> frame_grabber_openni2::GetFrameImpl(int index, cv::cuda::Stream& stream)
{
	return TS<SyncedMemory>();
}

TS<SyncedMemory> frame_grabber_openni2::GetNextFrameImpl(cv::cuda::Stream& stream)
{
	if (h_cam)
	{
		cv::Mat point_cloud;
		cv::Mat depth;
		h_cam->retrieve(depth, cv::CAP_OPENNI_DEPTH_MAP);
		if (h_cam->retrieve(point_cloud, cv::CAP_OPENNI_POINT_CLOUD_MAP))
		{
			if (!point_cloud.empty())
			{
				double ts = h_cam->get(cv::CAP_PROP_POS_MSEC);
				int fn = h_cam->get(cv::CAP_PROP_POS_FRAMES);
				return TS<SyncedMemory>(ts, fn, point_cloud);
			}
		}
	}
	return TS<SyncedMemory>();
}
rcc::shared_ptr<ICoordinateManager> frame_grabber_openni2::GetCoordinateManager()
{
	return _coordinate_manager;
}
static frame_grabber_openni2_info g_info;

REGISTERCLASS(frame_grabber_openni2, &g_info);