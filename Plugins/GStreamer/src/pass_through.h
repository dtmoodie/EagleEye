#pragma once

#include "gstreamer.hpp"
namespace EagleLib
{
	namespace Nodes
	{
		class PLUGIN_EXPORTS h264_pass_through: public gstreamer_sink_base
		{
		public:
			class h264_pass_through_info : public Nodes::NodeInfo
			{
			public:
				h264_pass_through_info();
			};
			virtual void NodeInit(bool firstInit);
			virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> img, cv::cuda::Stream &stream);
		};

	}
}
