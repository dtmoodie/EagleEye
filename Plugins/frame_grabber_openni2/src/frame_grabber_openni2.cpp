#include "frame_grabber_openni2.h"
#include "openni2_initializer.h"
#include <EagleLib/ICoordinateManager.h>
using namespace EagleLib;



std::string frame_grabber_openni2_info::GetObjectName()
{
	return "frame_grabber_openni2";
}

int frame_grabber_openni2_info::CanLoadDocument(const std::string& document) const
{
	std::string doc = document;
	std::transform(doc.begin(), doc.end(), doc.begin(), ::tolower);
	std::string openni("openni::");
	if(doc.compare(0, openni.length(), openni) == 0)
	{
		// Check if valid uri
		initializer_NI2::instance();
		openni::Array<openni::DeviceInfo> devices;
		openni::OpenNI::enumerateDevices(&devices);
		for(int i = 0; i < devices.getSize(); ++i)
		{
			auto uri = devices[i].getUri();
			auto len = strlen(uri);
			if(document.compare(openni.length(),len, uri) == 0)
			{
				return 11;
			}
		}
		return 5;
	}
	return 0;
}

int frame_grabber_openni2_info::LoadTimeout() const
{
	return 1000;
}

std::vector<std::string> frame_grabber_openni2_info::ListLoadableDocuments()
{
	initializer_NI2::instance();
	openni::Array<openni::DeviceInfo> devices;
	openni::OpenNI::enumerateDevices(&devices);
	
	std::vector<std::string> output;
	for(int i = 0; i < devices.getSize(); ++i)
	{
		auto uri = devices[i].getUri();
		output.push_back(std::string("OpenNI::") + std::string(uri));
	}
	return output;
}



frame_grabber_openni2::frame_grabber_openni2()
{
	
}
int frame_grabber_openni2::GetNumFrames()
{
	return -1;
}
bool frame_grabber_openni2::LoadFile(const std::string& file_path)
{
	std::string doc = file_path;
	std::transform(doc.begin(), doc.end(), doc.begin(), ::tolower);
	std::string openni("openni::");
	if(doc.compare(0, openni.length(), openni) == 0)
	{
		initializer_NI2::instance();
		std::string uri = file_path.substr(openni.length());
		if(uri.size())
		{
			openni::Array<openni::DeviceInfo> devices;
			openni::OpenNI::enumerateDevices(&devices);
		}else
		{
			_device.reset(new openni::Device());
			openni::Status rc;
			rc = _device->open(openni::ANY_DEVICE);
			if(rc != openni::STATUS_OK)
			{
				LOG(info) << "Unable to connect to openni2 compatible device";
				return false;
			}
			LOG(info) << "Connected to device " << _device->getDeviceInfo().getUri();
			return true;
		}
	}
	return false;
}

TS<SyncedMemory> frame_grabber_openni2::GetFrameImpl(int index, cv::cuda::Stream& stream)
{
	return TS<SyncedMemory>();
}

TS<SyncedMemory> frame_grabber_openni2::GetNextFrameImpl(cv::cuda::Stream& stream)
{
	return TS<SyncedMemory>();
}
rcc::shared_ptr<ICoordinateManager> frame_grabber_openni2::GetCoordinateManager()
{
	return rcc::shared_ptr<ICoordinateManager>();
}
static frame_grabber_openni2_info g_inst;
REGISTERCLASS(frame_grabber_openni2, &g_inst);