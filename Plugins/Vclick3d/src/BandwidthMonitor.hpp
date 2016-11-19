#pragma once

#ifndef BANDWIDTH_MONITOR_HPP
#define BANDWIDTH_MONITOR_HPP

#include "Vclick3dExport.hpp"

#include <chrono>
#include <mutex>
#include <string>
#include <pcap.h>

#include "PcapAdapter.hpp"


// to convert to KB, MB, etc.
static constexpr double BytesPerKilo { 1024 };

/**
 * \brief A BandwidthMonitor is associated with one pcap_if and tracks the used bandwidth per time.
 *
 */
class Vclick3d_EXPORT BandwidthMonitor {
public:
    BandwidthMonitor(pcap_if_t* device);

    // to be run in a seperate thread for each BandwidthMonitor
    void run();

private:
    // saves Bytes and Time to a file, along with a usage statistic 
    void save_snapshot() const;
    void update_snapshot(std::chrono::seconds& acc_time);

private:
    // static so multiple instances accumulate to same variable
    static unsigned long long Bytes;
    static std::chrono::seconds Time;
    static std::mutex Mutex;

    // used for saving snapshots of the accumulated values
    // after a certain time period
    std::chrono::system_clock::time_point m_time_point;
    std::chrono::seconds m_snapshot_time { 30 };
    const std::string bytefile { "bytes.txt" };
    const std::string timefile { "time.txt" };

    // so run() can open the pcap_of for the device
    pcap_if_t* m_device;
    // the pcap_if used to track bandwidth usage
    PcapAdapter m_adapter;
};

// returns the most readable (>= 1) representation of bytes, stops at GB
Vclick3d_EXPORT std::pair<unsigned long long, std::string> convert_bytes(unsigned long long bytes);

#endif // BANDWIDTH_MONITOR_HPP