#pragma once

#include "gstreamer.h"

namespace EagleLib
{
	class PLUGIN_EXPORTS frame_grabber_html5 : public frame_grabber_gstreamer
	{
	public:
		class PLUGIN_EXPORTS frame_grabber_html5_info: public FrameGrabberInfo
		{
		public:
			virtual std::string GetObjectName();
			virtual int CanLoadDocument(const std::string& document) const;
			virtual int Priority() const;
		};

		frame_grabber_html5();
		virtual bool LoadFile(const std::string& file_path);
		virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
	};

}