#pragma once

#ifndef PCAP_ADAPTER_HPP
#define PCAP_ADAPTER_HPP
#include "Vclick3dExport.hpp"
#include <pcap.h>


/**
 * \brief Simple RAII wrapper for pcap_t
 *
 */
class Vclick3d_EXPORT PcapAdapter {
public:
    enum class OpenMode {
        Dead, Live, Offline
    };

    PcapAdapter() = default;
    PcapAdapter(pcap_if_t* device, OpenMode openmode);
    ~PcapAdapter();

    void open(pcap_if_t* device, OpenMode openmode);
    void open_live(pcap_if_t* device);
    void open_dead(pcap_if_t* device);
    void open_offline(pcap_if_t* device);
    pcap_t* get() const;

private:
    static constexpr unsigned short MaxPacketSize { 65535 };

private:
    pcap_t* handle {};
};

#endif // PCAP_ADAPTER_HPP
