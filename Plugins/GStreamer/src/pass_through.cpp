#include "pass_through.h"
#include <EagleLib/Nodes/NodeInfo.hpp>

using namespace EagleLib;
using namespace EagleLib::Nodes;
h264_pass_through::h264_pass_through()
{

}
bool h264_pass_through::ProcessImpl()
{
    if(gstreamer_string_param.modified)
    {
        create_pipeline(gstreamer_string);
        if (get_pipeline_state() != GST_STATE_PLAYING && _pipeline)
            start_pipeline();
        gstreamer_string_param.modified = false;
        valve = gst_bin_get_by_name(GST_BIN(_pipeline), "myvalve");
        if(!valve)
        {
            LOG_NODE(warning) << "No valve found in pipeline with name 'myvalve'";
        }
    }
    if(active_param.modified)
    {
        if(active)
        {
            g_object_set(valve, "drop", false, NULL);
        }else
        {
            g_object_set(valve, "drop", true, NULL);
        }
        active_param.modified = false;
    }
    return true;
}


MO_REGISTER_CLASS(h264_pass_through);


