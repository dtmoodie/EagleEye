#include "frame_grabber_openni2.h"
#include "openni2_initializer.h"
#include <EagleLib/ICoordinateManager.h>
#include "EagleLib/rcc/ObjectManager.h"
using namespace EagleLib;
SETUP_PROJECT_IMPL


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
	return 10000;
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
	_is_stream = true;
}
frame_grabber_openni2::~frame_grabber_openni2()
{
	if(_depth)
	{
		_depth->removeNewFrameListener(this);
		_depth->stop();
		if(_device)
		{
			_device->close();
		}
	}
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
				LOG(info) << "Unable to connect to openni2 compatible device: " << openni::OpenNI::getExtendedError();
				return false;
			}
			_depth.reset(new openni::VideoStream());
			rc = _depth->create(*_device, openni::SENSOR_DEPTH);
			if( rc != openni::STATUS_OK)
			{
				LOG(info) << "Unable to retrieve depth stream: " << openni::OpenNI::getExtendedError();
				return false;
			}
			_depth->addNewFrameListener(this);
			_depth->start();
			LOG(info) << "Connected to device " << _device->getDeviceInfo().getUri();
			return true;
		}
	}
	return false;
}
void frame_grabber_openni2::onNewFrame(openni::VideoStream& stream)
{
	openni::Status rc = stream.readFrame(&_frame);
	if(rc != openni::STATUS_OK)
	{
		LOG(debug) << "Unable to read new depth frame: " << openni::OpenNI::getExtendedError();
		return;
	}
	int height = _frame.getHeight();
	int width = _frame.getWidth();
	auto ts = _frame.getTimestamp();
	auto fn = _frame.getFrameIndex();
	int scale = 1;
	switch(_frame.getVideoMode().getPixelFormat())
	{
	case openni::PIXEL_FORMAT_DEPTH_100_UM:
		scale = 10;
	case openni::PIXEL_FORMAT_DEPTH_1_MM:	
		cv::Mat data(height, width, CV_16U, (ushort*)_frame.getData());
		cv::Mat XYZ;
		XYZ.create(height, width, CV_32FC3);
		for(int i = 0; i < height; ++i)
		{
			ushort* ptr = data.ptr<ushort>(i);
			cv::Vec3f* pt_ptr = XYZ.ptr<cv::Vec3f>(i);
			for(int j = 0; j < width; ++j)
			{
				openni::CoordinateConverter::convertDepthToWorld(*_depth, j, i, ptr[j], &pt_ptr[j].val[0], &pt_ptr[j].val[1], &pt_ptr[j].val[2]);
			}
		}
		
        FrameGrabberBuffered::PushFrame(TS<SyncedMemory>(double(ts), fn, XYZ), false);
		break;
	}
}

rcc::shared_ptr<ICoordinateManager> frame_grabber_openni2::GetCoordinateManager()
{
	return rcc::shared_ptr<ICoordinateManager>();
}
static frame_grabber_openni2_info g_inst;
REGISTERCLASS(frame_grabber_openni2, &g_inst);
