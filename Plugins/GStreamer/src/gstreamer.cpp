#include "gstreamer.hpp"
#include <gst/video/video.h>
#include <gst/app/gstappsrc.h>
using namespace EagleLib;

SETUP_PROJECT_IMPL

typedef class RTSP_server App;
static gboolean
bus_message(GstBus * bus, GstMessage * message, App * app)
{
	BOOST_LOG_TRIVIAL(debug) << "Received message type: " << gst_message_type_get_name(GST_MESSAGE_TYPE(message));

	switch (GST_MESSAGE_TYPE(message)) {
	case GST_MESSAGE_ERROR: 
	{
		GError *err = NULL;
		gchar *dbg_info = NULL;

		gst_message_parse_error(message, &err, &dbg_info);
		BOOST_LOG_TRIVIAL(error) << "Error from element " << GST_OBJECT_NAME(message->src) << ": " << err->message;
		BOOST_LOG_TRIVIAL(error) << "Debugging info: " << (dbg_info) ? dbg_info : "none";
		g_error_free(err);
		g_free(dbg_info);
		g_main_loop_quit(app->glib_MainLoop);
		break;
	}
	case GST_MESSAGE_EOS:
		g_main_loop_quit(app->glib_MainLoop);
		break; 
	default:
		break;
	}
	return TRUE;
}

static gboolean cb_need_data(GstElement *appsrc,
						guint  unused_size,
						App*    user_data)
{
	static int count = 0;
	if (count == 0)
	{
		cv::Mat frameimage(user_data->imgSize, CV_8UC3);
		static GstClockTime timestamp = 0;
		GstBuffer *buffer;
		guint buffersize;
		GstFlowReturn ret;
		GstMapInfo info;
		buffersize = frameimage.cols * frameimage.rows * frameimage.channels();
		buffer = gst_buffer_new_and_alloc(buffersize);
		if (gst_buffer_map(buffer, &info, (GstMapFlags)GST_MAP_WRITE)) {
			memcpy(info.data, frameimage.data, buffersize);
			gst_buffer_unmap(buffer, &info);
		}
		else
		{
			BOOST_LOG_TRIVIAL(error) << "Unable to map image data to buffer";
		}
		ret = gst_app_src_push_buffer((GstAppSrc*)user_data->source_OpenCV, buffer);
		if (ret != GST_FLOW_OK) {
			BOOST_LOG_TRIVIAL(error) << "something wrong in cb_need_data";
			g_main_loop_quit(user_data->glib_MainLoop);
		}
		//gst_buffer_unref(buffer);
		++count;
	}
	return TRUE;
	
}
static void start_feed(GstElement * pipeline, guint size, App * app){
	if (app->sourceid == 0){
		app->sourceid = g_timeout_add(67, (GSourceFunc)cb_need_data, app);
	}
}
static void stop_feed(GstElement * pipeline, App *app)
{
	if (app->sourceid != 0) {
		GST_DEBUG("stop feeding");
		g_source_remove(app->sourceid);
		app->sourceid = 0;
	}
}
RTSP_server::~RTSP_server()
{
	g_main_loop_quit(glib_MainLoop);
	g_main_loop_unref(glib_MainLoop);
	glibThread.join(); 
}

void RTSP_server::gst_loop()
{
	if (!g_main_loop_is_running(glib_MainLoop))
	{
		g_main_loop_run(glib_MainLoop);
	}
}


void RTSP_server::setup()
{
	// gst-launch-1.0 -v videotestsrc ! openh264enc ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=8004
	gst_debug_set_active(1);

	if (!gst_is_initialized())
	{
		char** argv; // = { "-vvv" };
		argv = new char*{ "-vvv" };
		int argc = 1;
		gst_init(&argc, &argv);
	}
		
	if (!glib_MainLoop)
		glib_MainLoop = g_main_loop_new(NULL, 0);

	glibThread = boost::thread(boost::bind(&RTSP_server::gst_loop, this));
	GError* error = nullptr;
	pipeline = gst_parse_launch("appsrc name=mysource ! openh264enc ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=8004", &error);
	if (error != nullptr)
	{
		NODE_LOG(error) << "Error parsing pipeline " << error->message;
	}
	source_OpenCV = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
	
	GstCaps* caps = gst_caps_new_simple("video/x-raw",
		"format", G_TYPE_STRING, "RGB24",
		"width", G_TYPE_INT, imgSize.width,
		"height", G_TYPE_INT, imgSize.height,
		"framerate", GST_TYPE_FRACTION, 15, 1,
		"pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1,
		NULL);

	if (caps == nullptr)
	{
		NODE_LOG(error) << "Error creating caps for appsrc";
	}

	gst_app_src_set_caps(GST_APP_SRC(source_OpenCV), caps);
	g_object_set(G_OBJECT(source_OpenCV), 
		"stream-type", GST_APP_STREAM_TYPE_STREAM, 
		"format", GST_FORMAT_TIME, 
		NULL);
	
	g_signal_connect(source_OpenCV, "need-data", G_CALLBACK(start_feed), this);
	g_signal_connect(source_OpenCV, "enough-data", G_CALLBACK(stop_feed), this);

	// Error callback
	auto bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
	CV_Assert(bus);
	gst_bus_add_watch(bus, (GstBusFunc)bus_message, this);
	gst_object_unref(bus);

	GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
	CV_Assert(ret != GST_STATE_CHANGE_FAILURE);
	//Create_PipelineGraph(pipeline);
}

void RTSP_server::Init(bool firstInit)
{
	updateParameter<std::string>("Address", "127.0.0.1");
	updateParameter<unsigned short>("Port", 8004);
	sourceid = 0;
	glib_MainLoop = nullptr;
	if (firstInit)
	{
		source_OpenCV = nullptr;
		pipeline = nullptr;
		encoder = nullptr;
		payloader = nullptr;
		udpSink = nullptr;
		
		bufferPool.resize(5);
	}
}
void RTSP_server::push_image()
{
	static GstClockTime timestamp = 0;

	GstBuffer* buffer;
	auto h_buffer = bufferPool.getBack();
	if (h_buffer)
	{
		int bufferlength = h_buffer->cols * h_buffer->rows * h_buffer->channels();
		buffer = gst_buffer_new_and_alloc(bufferlength);
		
		GstMapInfo map;
		gst_buffer_map(buffer, &map, (GstMapFlags)GST_MAP_WRITE);
		memcpy(map.data, h_buffer->data, map.size);
		gst_buffer_unmap(buffer, &map);

		/*GST_BUFFER_PTS(buffer) = timestamp;

		GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, 2);
		timestamp += GST_BUFFER_DURATION(buffer);*/

		
		GstFlowReturn rw;
		g_signal_emit_by_name(source_OpenCV, "push-buffer", buffer, &rw);

		if (rw != GST_FLOW_OK)
		{
			NODE_LOG(error) << "Error pushing buffer into appsrc " << rw;
		}
	}
}

void RTSP_server::Serialize(ISimpleSerializer* pSerializer)
{
	Node::Serialize(pSerializer);
	SERIALIZE(source_OpenCV);
	SERIALIZE(pipeline);
	SERIALIZE(encoder);
	SERIALIZE(payloader);
	SERIALIZE(udpSink);
}
void RTSP_serverCallback(int status, void* userData)
{
	static_cast<RTSP_server*>(userData)->push_image();
}
cv::cuda::GpuMat RTSP_server::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
	if (imgSize != img.size())
	{
		imgSize = img.size();
		setup();
	}
	if (!g_main_loop_is_running(glib_MainLoop))
	{
		NODE_LOG(error) << "Main glib loop not running";
		return img;
	}
	/*auto buffer = bufferPool.getFront();
	img.download(*buffer, stream);
	stream.enqueueHostCallback(RTSP_serverCallback, this);*/
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(RTSP_server)

