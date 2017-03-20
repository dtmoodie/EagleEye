#ifdef HAVE_PCAP
#include "PcapDevices.hpp"

#include <iostream>
#include <fstream>


PcapDevices::PcapDevices()
{
    char errbuf[PCAP_ERRBUF_SIZE];
    if (pcap_findalldevs(&devices, errbuf) == -1) {
        throw std::runtime_error(
            "PcapDevices::PcapDevices:  Could not find any devices");
    }
}

PcapDevices::~PcapDevices()
{
    if (devices) pcap_freealldevs(devices);
}

std::vector<pcap_if_t*> PcapDevices::get_devices() const
{
    decltype(get_devices()) result;
    for (auto* d = devices; d != nullptr; d = d->next) {
        result.emplace_back(d);
    }
    return result;
}
#endif
