#include "pass_through.h"
#include "parameters/ParameteredObjectImpl.hpp"

using namespace EagleLib;
using namespace EagleLib::Nodes;

void h264_pass_through::NodeInit(bool firstInit)
{
	if (firstInit)
	{
		updateParameter<std::string>("gstreamer string", "");
		updateParameter<bool>("active", true);
	}
}

TS<SyncedMemory> h264_pass_through::doProcess(TS<SyncedMemory> img, cv::cuda::Stream &stream)
{
	if (_parameters[0]->changed)
	{
		create_pipeline(*getParameter<std::string>(0)->Data());
		if (get_pipeline_state() != GST_STATE_PLAYING && _pipeline)
			start_pipeline();
		_parameters[0]->changed = false;
	}
	if(getParameter(1)->changed)
	{
		if (*getParameter<bool>(1)->Data())
		{
			if (get_pipeline_state() != GST_STATE_PLAYING && _pipeline)
				start_pipeline();
		}
		else
		{
			pause_pipeline();
		}
		getParameter(1)->changed = false;
	}
	

	return img;
}

h264_pass_through::h264_pass_through_info::h264_pass_through_info():
	NodeInfo("h264_pass_through", { "Utilities" })
{

}
static h264_pass_through::h264_pass_through_info g_info;
REGISTERCLASS(h264_pass_through, &g_info);


