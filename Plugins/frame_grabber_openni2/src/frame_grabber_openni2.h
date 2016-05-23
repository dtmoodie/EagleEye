#pragma once

#include "OpenNI.h"
#include <EagleLib/frame_grabber_base.h>
#include "RuntimeLinkLibrary.h"
SETUP_PROJECT_DEF
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
	class PLUGIN_EXPORTS frame_grabber_openni2: public FrameGrabberBuffered, public openni::VideoStream::NewFrameListener
	{
		openni::VideoFrameRef _frame;
		std::shared_ptr<openni::Device> _device;
		std::shared_ptr<openni::VideoStream> _depth;
		//boost::circular_buffer<TS<SyncedMemory>> _buffer;
	public:
		frame_grabber_openni2();
		~frame_grabber_openni2();
		
		virtual bool LoadFile(const std::string& file_path);
		virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
		int GetNumFrames();
		void onNewFrame(openni::VideoStream& stream);
	};
}