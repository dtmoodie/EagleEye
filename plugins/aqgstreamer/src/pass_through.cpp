#include <ct/types/opencv.hpp>

#include "pass_through.h"
#include <Aquila/nodes/NodeInfo.hpp>

namespace aqgstreamer
{

    bool H264PassThrough::processImpl()
    {
        if (pipeline_param.getModified())
        {
            createPipeline(pipeline);
            if (getPipelineState() != GST_STATE_PLAYING && m_pipeline)
            {
                startPipeline();
            }
            pipeline_param.setModified(false);
            m_valve = gst_bin_get_by_name(GST_BIN(m_pipeline.get()), "myvalve");
            if (!m_valve)
            {
                this->getLogger().warn("No valve found in pipeline with name 'myvalve'");
            }
            m_previously_active = false;
        }
        if (active_param.getModified() && active != m_previously_active)
        {
            if (active)
            {
                g_object_set(m_valve, "drop", false, NULL);
                m_previously_active = true;
            }
            else
            {
                g_object_set(m_valve, "drop", true, NULL);
                m_previously_active = false;
            }
            active_param.setModified(false);
        }
        return true;
    }
} // namespace aqgstreamer

using namespace aqgstreamer;
MO_REGISTER_CLASS(H264PassThrough);
