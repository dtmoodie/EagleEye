#pragma once

#ifndef MPS_PCAP_DEVICES_HPP
#define MPS_PCAP_DEVICES_HPP
#include "Vclick3dExport.hpp"
#include <vector>

#include <pcap.h>


/**
 * \brief Simple RAII wrapper for pcap_if_t
 *
 */
class Vclick3d_EXPORT PcapDevices {
public:
    PcapDevices();
    ~PcapDevices();

    std::vector<pcap_if_t*> get_devices() const;

private:
    pcap_if_t* devices {};
};

#endif // MPS_PCAP_DEVICES_HPP