#include "html5.h"
using namespace EagleLib;


frame_grabber_html5::frame_grabber_html5()
{

}

bool frame_grabber_html5::LoadFile(const std::string& file_path)
{
    std::string http("http://");
    if (file_path.compare(0, http.length(), http) == 0)
    {
        // Grab everything between http:// and : for the host
        auto pos = file_path.find(':');
        std::stringstream ss;
        ss << "tcpclientsrc host=";
        if (pos != std::string::npos)
        {
            std::string host = file_path.substr(6, pos - 6);
            std::string ip = file_path.substr(pos + 1);
            ss << host;
            ss << ip;
        }

        //h_LoadFile("tcpclientsrc ")
    }
    return false;
}

rcc::shared_ptr<ICoordinateManager> frame_grabber_html5::GetCoordinateManager()
{
    return coordinate_manager;
}

std::string frame_grabber_html5::frame_grabber_html5_info::GetObjectName()
{
    return "frame_grabber_html5";
}
int frame_grabber_html5::frame_grabber_html5_info::CanLoadDocument(const std::string& document) const
{
    std::string http("http://");
    if (document.compare(0, http.length(), http) == 0)
    {
        return 10;
    }
    return 0;
}
int frame_grabber_html5::frame_grabber_html5_info::Priority() const
{
    return 0;
}