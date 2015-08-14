#include "gstreamer.hpp"

using namespace EagleLib;

void RTSP_server::setup()
{
	// gst-launch-1.0 -v videotestsrc ! openh264enc ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=8004

    pipeline = gst_pipeline_new("RTSP server");
    source_OpenCV = gst_element_factory_make("appsrc", "Source_OpenCV");

    gst_bin_add(GST_BIN(pipeline), source_OpenCV);

	encoder = gst_element_factory_make("openh264enc", "openh264enc0");

	gst_bin_add(GST_BIN(pipeline), encoder);

	payloader = gst_element_factory_make("rtph264pay", "rtph264pay0");
	// http://lists.freedesktop.org/archives/gstreamer-devel/2011-July/032472.html
	g_object_set(payloader, "config-interval", 1, "pt", 96);

	gst_bin_add(GST_BIN(pipeline), payloader);

	udpSink = gst_element_factory_make("udpsink", "udpsink0");
	g_object_set(udpSink, "host", "127.0.0.1", "port", 8004);

	gst_bin_add(GST_BIN(pipeline), udpSink);
	
	
	

}

void RTSP_server::Init(bool firstInit)
{

}

cv::cuda::GpuMat RTSP_server::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(RTSP_server)
