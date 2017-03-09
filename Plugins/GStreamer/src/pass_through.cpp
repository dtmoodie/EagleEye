#include "pass_through.h"
#include <Aquila/Nodes/NodeInfo.hpp>

using namespace aq;
using namespace aq::Nodes;

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
        previously_active = false;
    }
    if(active_param.modified && active != previously_active)
    {
        if(active)
        {
            g_object_set(valve, "drop", false, NULL);
            previously_active = true;
        }else
        {
            g_object_set(valve, "drop", true, NULL);
            previously_active = false;
        }
        active_param.modified = false;
    }
    return true;
}


MO_REGISTER_CLASS(h264_pass_through);


