#include "sinks.hpp"
#include <gst/gst.h>
#include <EagleLib/Nodes/NodeInfo.hpp>

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
    if(check_feature(#feature)) {       \
        Enum.addEnum(__COUNTER__, #feature); \
    }

void tcpserver::NodeInit(bool firstInit)
{
    if(firstInit)
    {
        encoders.addEnum(-1, "Select encoder");
    
        if(check_feature("matroskamux"))
        {
            if(check_feature("openh264enc"))
                encoders.addEnum(0, "openh264enc");
            if(check_feature("avenc_h264"))
                encoders.addEnum(1, "avenc_h264");
            if(check_feature("omxh264enc"))
                encoders.addEnum(2, "omxh264enc");
        }
        if(check_feature("webmmux"))
        {
            if(check_feature("omxvp8enc"))
                encoders.addEnum(3, "omxvp8enc");
            if(check_feature("vp8enc"))
                encoders.addEnum(4, "vp8enc");
        }
        encoders_param.Commit();
        auto interfaces = get_interfaces();
        for(int i = 0; i < interfaces.size(); ++i)
        {
            this->interfaces.addEnum(i, interfaces[i]);
        }
        this->interfaces_param.Commit();
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


MO_REGISTER_CLASS(tcpserver);




void BufferedHeartbeatRtsp::NodeInit(bool firstInit)
{

}

