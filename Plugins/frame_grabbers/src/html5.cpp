#ifdef HAVE_GSTREAMER
#include "html5.h"
#include "precompiled.hpp"

using namespace EagleLib;
using namespace EagleLib::Nodes;

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
    return rcc::shared_ptr<ICoordinateManager>();
}


int frame_grabber_html5::CanLoadDocument(const std::string& document) const
{
    std::string http("http://");
    if (document.compare(0, http.length(), http) == 0)
    {
        return 10;
    }
    return 0;
}

#endif // HAVE_GSTREAMER