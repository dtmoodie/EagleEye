#pragma once

#include "OpenNI.h"
#include <EagleLib/frame_grabber_base.h>
#include "RuntimeLinkLibrary.h"

RUNTIME_COMPILER_LINKLIBRARY("OpenNI2.lib");

namespace EagleLib
{
	class PLUGIN_EXPORTS frame_grabber_openni2_info: public FrameGrabberInfo
	{
	public:
		virtual std::string GetObjectName();
		virtual int CanLoadDocument(const std::string& document) const;
		virtual int LoadTimeout() const;
		virtual std::vector<std::string> ListLoadableDocuments();
	};
	class PLUGIN_EXPORTS frame_grabber_openni2: public FrameGrabberBuffered
	{
		std::shared_ptr<openni::Device> _device;
		std::shared_ptr<openni::VideoStream> _depth;
	public:
		frame_grabber_openni2();
		virtual int GetNumFrames();
		virtual bool LoadFile(const std::string& file_path);
		virtual TS<SyncedMemory> GetFrameImpl(int index, cv::cuda::Stream& stream);
		virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream);
		virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
	};
}