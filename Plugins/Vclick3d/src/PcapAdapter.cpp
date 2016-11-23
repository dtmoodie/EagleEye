#include "PcapAdapter.hpp"

#include <cassert>
#include <stdexcept>


PcapAdapter::PcapAdapter(pcap_if_t* device, OpenMode openmode)
{
    open(device, openmode);
}

PcapAdapter::~PcapAdapter()
{
    if (handle) pcap_close(handle);
}

void PcapAdapter::open(pcap_if_t* device, OpenMode openmode)
{
    switch (openmode) {
    case OpenMode::Live:
        open_live(device);
        break;
    case OpenMode::Dead:
        open_dead(device);
        break;
    case OpenMode::Offline:
        open_offline(device);
        break;
    }
}

pcap_t* PcapAdapter::get() const
{
    return handle;
}

void PcapAdapter::open_live(pcap_if_t* device)
{
    char errbuf[PCAP_ERRBUF_SIZE];
    if ((handle = pcap_open_live(device->name,       // device
                                         MaxPacketSize,      // snaplen
                                         1,                  // promisc
                                         1000,               // read timeout
                                         errbuf)) == nullptr) {
        throw std::runtime_error(std::string(errbuf));
    }
}

void PcapAdapter::open_dead(pcap_if_t* device)
{
    assert(false);
    // not yet implemented
}

void PcapAdapter::open_offline(pcap_if_t* device)
{
    assert(false);
    // not yet implemented
}