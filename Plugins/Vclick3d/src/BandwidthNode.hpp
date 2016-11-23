#pragma once

#include "EagleLib/Nodes/Node.h"
#include "BandwidthMonitor.hpp"
#include "PcapDevices.hpp"

#include <boost/thread.hpp>
namespace EagleLib
{
namespace Nodes
{
    class BandwidthUsage: public Node
    {
    public:
        ~BandwidthUsage();
        MO_DERIVE(BandwidthUsage, Node)
            STATUS(std::vector<std::string>, devices, listDevices());
            PARAM(int, selected_device, -1);
            PARAM(std::vector<unsigned short>, ports, std::vector<unsigned short>());
            OUTPUT(std::vector<double>, usage_bytes_per_second, std::vector<double>());
        MO_END;
    protected:
        bool ProcessImpl();
    private:
        std::vector<boost::thread> monitor_threads;
        std::vector<BandwidthMonitor> monitors;
        std::vector<std::string> listDevices();
        PcapDevices d;
    };
} // namespace Nodes
} // namespace EagleLib