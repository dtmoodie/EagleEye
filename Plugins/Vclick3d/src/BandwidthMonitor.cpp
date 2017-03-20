#ifdef HAVE_PCAP
#include "BandwidthMonitor.hpp"

#include <fstream>


unsigned long long BandwidthMonitor::Bytes {};
std::chrono::seconds BandwidthMonitor::Time { 0 };
std::mutex BandwidthMonitor::Mutex {};
    
BandwidthMonitor::BandwidthMonitor(pcap_if_t* device)
    :m_device { device }
{
    // Initializes previous values from files if Bytes and Time are 0
    // This is intended to run once at the start of the program (when they are 0)
    if (!Bytes) {
        static std::mutex is_mutex;
        std::lock_guard<std::mutex> guard(is_mutex);
        std::ifstream is(bytefile);
        if (is) {
            unsigned long long b;
            is >> b;
            Bytes = b;
        }
    }

    if (!Time.count()) {
        static std::mutex is_mutex;
        std::lock_guard<std::mutex> guard(is_mutex);
        std::ifstream is(timefile);
        if (is) {
            std::chrono::seconds::rep sec;
            is >> sec;
            Time = std::chrono::seconds(sec);
        }
    }
}

void BandwidthMonitor::run()
{
    m_adapter.open_live(m_device);

    m_time_point = std::chrono::system_clock::now();
    pcap_pkthdr* header {};
    const u_char* data;
    int res {};
    std::chrono::seconds acc_time {};
    while ((res = pcap_next_ex(m_adapter.get(), &header, &data) >= 0)) {
        if (!res) continue;  // timeout
            
        std::lock_guard<std::mutex> guard(Mutex);
        Bytes += header->caplen;

        update_snapshot(acc_time);
    }
}

void BandwidthMonitor::update_snapshot(std::chrono::seconds& acc_time)
{
    // Bug introduced while outsourcing, since Mutex already locked from run()
    //std::lock_guard<std::mutex> guard(Mutex);

    auto then = m_time_point;
    m_time_point = std::chrono::system_clock::now();
    acc_time += std::chrono::duration_cast<std::chrono::seconds>(m_time_point - then);
    if (acc_time >= m_snapshot_time) {
        Time += m_snapshot_time;
        acc_time -= m_snapshot_time;
        save_snapshot();
    }
}

void BandwidthMonitor::save_snapshot() const
{
    static std::mutex os_mutex;
    std::lock_guard<std::mutex> lk(os_mutex);

    std::ofstream osbytes(bytefile);
    osbytes << Bytes;

    std::ofstream osttime(timefile);
    osttime << Time.count();

    auto byteval = convert_bytes(Bytes);

    std::ofstream osusage("usage.txt");
    osusage << byteval.first << byteval.second
        << " in the past "
        << std::chrono::duration_cast<std::chrono::hours>(Time).count() / 24
        << " days.\n";
}

/**
 * \brief Divides value of bytes by 1024, until a number >= 1 is reached
 * \return Returns a std::pair: first=new byte value, second=corresponding unit
 */
std::pair<unsigned long long, std::string> convert_bytes(unsigned long long bytes)
{
    // convert to most convenient unit
    auto count = 0;
    auto value = bytes;
    while (value / BytesPerKilo >= 1) {
        ++count;
        value /= BytesPerKilo;
    }

    std::string unit;
    switch (count) {
    case 0: unit = "bytes"; break;
    case 1: unit = "KB"; break;
    case 2: unit = "MB"; break;
    default: unit = "GB"; break;
    }
    return { value, unit };
}
#endif
