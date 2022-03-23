#ifdef HAVE_GSTREAMER
#include "html5.h"
#include "precompiled.hpp"
#include <Aquila/framegrabbers/GrabberInfo.hpp>

using namespace aq;
using namespace aq::nodes;
// gst-launch-1.0 tcpclientsrc host=192.168.0.99 port=80 ! matroskademux ! h264parse ! avdec_h264 ! videoconvert !
// autovideosink
bool GrabberHTML::loadData(const std::string& file_path)
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
            std::string host = file_path.substr(http.size(), file_path.size() - http.size());
            pos = file_path.find(':', http.size() + 1);
            std::string port = file_path.substr(pos + 1);
            ss << "host=";
            ss << host;
            ss << "port=";
            ss << port;
            ss << " ! matroskademux ! h264parse ! avdec_h264 ! videoconvert ! appsink";
            return GrabberGstreamer::loadData(ss.str());
        }
    }
    return false;
}

int GrabberHTML::canLoad(const std::string& document)
{
    std::string http("http://");
    if (document.compare(0, http.length(), http) == 0)
    {
        return 9;
    }
    return 0;
}
MO_REGISTER_CLASS(GrabberHTML)
#endif // HAVE_GSTREAMER
