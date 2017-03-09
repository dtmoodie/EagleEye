#include "BandwidthNode.hpp"
#include "Aquila/Nodes/NodeInfo.hpp"

//#include <MetaObject/Parameters/IO/TextPolicy.hpp>
//#include <MetaObject/Parameters/IO/CerealPolicy.hpp>
#include <pcap/ipnet.h>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

#ifndef _MSC_VER
#include <sys/socket.h>
#endif

#include "RuntimeLinkLibrary.h"
using namespace aq;
using namespace aq::Nodes;

#ifdef _MSC_VER
RUNTIME_COMPILER_LINKLIBRARY("wpcap.lib")

#else

#endif

//INSTANTIATE_META_PARAMETER(std::vector<std::string>);
BandwidthUsage::~BandwidthUsage()
{
    for(auto& thread : monitor_threads)
    {
        thread.interrupt();
        thread.join();
    }
}
std::string getIp(pcap_if_t* inter)
{
    
    uchar* ptr = (uchar*)inter->addresses->next->addr->sa_data;
    std::vector<uchar> test(ptr, ptr + 14);
    std::stringstream ss;

    for(int i = 0; i < test.size(); ++i)
    {
        if(test[i])
        {
            ss << (int)test[i] << ".";
        }
    }
    return ss.str();
}
std::vector<std::string> BandwidthUsage::listDevices()
{
    PcapDevices d;
    auto alldevs = d.get_devices();
    std::vector<std::string> output;
    for(auto& dev : alldevs)
    {
        output.emplace_back(getIp(dev));
    }
    return output;
}


bool BandwidthUsage::ProcessImpl()
{
    if(ports_param.modified && selected_device != -1)
    {
        monitors.reserve(ports.size());
        monitor_threads.reserve(ports.size());
        
        auto alldevs = d.get_devices();
        
        for(int i = 0; i < ports.size(); ++i)
        {
            monitors.emplace_back(alldevs[selected_device]);
        }
        for(const auto& m : monitors)
        {
            monitor_threads.emplace_back(&BandwidthMonitor::run, m);
        }

        ports_param.modified = false;
    }
    _modified = true;
    return true;
}

MO_REGISTER_CLASS(BandwidthUsage);
