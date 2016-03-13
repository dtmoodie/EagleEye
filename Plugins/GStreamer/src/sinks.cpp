#include "sinks.hpp"
#include <gst/gst.h>
#include "EagleLib/ParameteredObjectImpl.hpp"
using namespace EagleLib;
using namespace EagleLib::Nodes;

tcpserver::tcpserver():
    gstreamer_sink_base()
{
    _initialized = false;
    
}

tcpserver::~tcpserver()
{

}

#define ENUM_FEATURE(Enum, feature)     \
    bool has_##feature = false;         \
    if(check_feature(#feature)) {       \
        Enum.addEnum(__COUNTER__, #feature); \
        has_##feature = true; \
    }

void tcpserver::Init(bool firstInit)
{
    if(firstInit)
    {
        Parameters::EnumParameter encoders;
        encoders.addEnum(-1, "Select encoder");
    
        if(check_feature("matroskamux"))
        {
            ENUM_FEATURE(encoders, openh264enc)
            ENUM_FEATURE(encoders, avenc_h264)
            ENUM_FEATURE(encoders, omxh264enc)
        }
        if(check_feature("webmmux"))
        {
            ENUM_FEATURE(encoders, omxvp8enc)
            ENUM_FEATURE(encoders, vp8enc)
        }
        updateParameter("Encoder", encoders);
        auto interfaces = get_interfaces();
        Parameters::EnumParameter interfaces_;
        for(int i = 0; i < interfaces.size(); ++i)
        {
            interfaces_.addEnum(i, interfaces[i]);
        }
        updateParameter("Interfaces", interfaces_);
    }
}

TS<SyncedMemory> tcpserver::doProcess(TS<SyncedMemory> img, cv::cuda::Stream &stream)
{
    if(!_initialized || getParameter("Encoder")->changed || getParameter("Interfaces")->changed)
    {
        auto encoder = getParameter<Parameters::EnumParameter>("Encoder")->Data();
        if(encoder->getValue() != -1)
        {
            std::string name = encoder->getEnum();
            std::stringstream ss;
            ss << "appsrc name=mysource ! videoconvert ! ";
            ss << name;
            if(name == "openh264enc" || name == "avenc_h264" || name == "omxh264enc")
            {
                ss << " ! matroskamux streamable=true ! tcpserversink host=";
            }
            else if(name == "omxvp8enc" || name == "vp8enc")
            {
                ss << " ! webmmux ! tcpserversink host=";
            }
            ss << getParameter<Parameters::EnumParameter>("Interfaces")->Data()->getEnum();
            ss << " port=8080";
            _initialized = create_pipeline(ss.str());
            if(_initialized)
            {
                getParameter("Encoder")->changed = false;
                getParameter("Interfaces")->changed = false;
            }
        }
    }else
    {
        return gstreamer_sink_base::doProcess(img, stream);
    }

    return img;
}

static EagleLib::Nodes::NodeInfo g_registerer_tcpserver_sink("tcpserver", { "Image", "Sink" });
REGISTERCLASS(tcpserver, &g_registerer_tcpserver_sink);