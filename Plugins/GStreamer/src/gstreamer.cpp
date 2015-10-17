#include "gstreamer.hpp"
#include <gst/video/video.h>
#include <gst/app/gstappsrc.h>

#include <QtNetwork/qnetworkinterface.h>
#include <Manager.h>
#include <SystemTable.hpp>

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


static void start_feed(GstElement * pipeline, guint size, App * app)
{
	app->feed_enabled = true;
}
static void stop_feed(GstElement * pipeline, App *app)
{
	app->feed_enabled = false;
}
// This only actually gets called when gstreamer.cpp gets recompiled
RTSP_server::~RTSP_server()
{
    NODE_LOG(info) << "RTSP server destructor";
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PAUSED);
    CV_Assert(ret != GST_STATE_CHANGE_FAILURE);
#ifdef _MSC_VER
    PerModuleInterface::GetInstance()->GetSystemTable()->SetSingleton<GMainLoop>(nullptr);
#endif
	g_main_loop_quit(glib_MainLoop);
	glibThread.join(); 
	//g_main_loop_unref(glib_MainLoop);
	g_signal_handler_disconnect(source_OpenCV, need_data_id);
	g_signal_handler_disconnect(source_OpenCV, enough_data_id);
	//gst_object_unref(pipeline);
	//gst_object_unref(source_OpenCV);
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
	// gst-launch-1.0 -v videotestsrc ! videoconvert ! openh264enc ! rtph264pay config-interval=1 pt=96 ! tcpserversink host=192.168.1.208 port=8004
	gst_debug_set_active(1);

	if (!gst_is_initialized())
	{
		char** argv; // = { "-vvv" };
		argv = new char*{ "-vvv" };
		int argc = 1;
		gst_init(&argc, &argv);
	}

	if (!glib_MainLoop)
	{
		glib_MainLoop = g_main_loop_new(NULL, 0);
	}
	glibThread = boost::thread(boost::bind(&RTSP_server::gst_loop, this));
	GError* error = nullptr;
	std::stringstream ss;
	ss << "appsrc name=mysource ! videoconvert ! ";
#ifdef JETSON 
	ss << "omxh264enc ! ";
#else
	ss << "openh264enc ! ";
#endif
	ss << "rtph264pay config-interval=1 pt=96 ! gdppay ! tcpserversink host=";

	foreach(auto inter, QNetworkInterface::allInterfaces())
	{
		if (inter.flags().testFlag(QNetworkInterface::IsUp) && 
		   !inter.flags().testFlag(QNetworkInterface::IsLoopBack))
		{
			foreach(auto entry, inter.addressEntries())
			{
				if (inter.hardwareAddress() != "00:00:00:00:00:00" && entry.ip().toString().contains("."))
				{
					NODE_LOG(info) << "Setting interface to " << inter.name().toStdString() << " " << 
						entry.ip().toString().toStdString() << " " << inter.hardwareAddress().toStdString();
					ss << entry.ip().toString().toStdString();
					break;
				}
			}
		}
	}
	ss << " port=";
    ss << *getParameter<unsigned short>("Port")->Data();
	std::string pipestr = ss.str();
	NODE_LOG(info) << pipestr;
    updateParameter<std::string>("gst pipeline", pipestr);
    if (!pipeline)
	    pipeline = gst_parse_launch(pipestr.c_str(), &error);
	if (error != nullptr)
	{
		NODE_LOG(error) << "Error parsing pipeline " << error->message;
	}
    if (!source_OpenCV)
	    source_OpenCV = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
	
	GstCaps* caps = gst_caps_new_simple(
        "video/x-raw",
		"format", G_TYPE_STRING, "BGR",
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
	g_object_set(
        G_OBJECT(source_OpenCV), 
		"stream-type", GST_APP_STREAM_TYPE_STREAM, 
		"format", GST_FORMAT_TIME, 
		NULL);
	
	need_data_id = g_signal_connect(source_OpenCV, "need-data", G_CALLBACK(start_feed), this);
	enough_data_id = g_signal_connect(source_OpenCV, "enough-data", G_CALLBACK(stop_feed), this);
	
	// Error callback
	auto bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
	CV_Assert(bus);
	gst_bus_add_watch(bus, (GstBusFunc)bus_message, this);
	gst_object_unref(bus);

	GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
	CV_Assert(ret != GST_STATE_CHANGE_FAILURE); 
}

void RTSP_server::Init(bool firstInit)
{
	if (firstInit) 
	{
		timestamp = 0;
		prevTime = clock();
        glib_MainLoop = nullptr;
        updateParameter<unsigned short>("Port", 8004);
        feed_enabled = false;
        source_OpenCV = nullptr;
        pipeline = nullptr;
	}
	bufferPool.resize(5);
}
void RTSP_server::push_image()
{
	GstBuffer* buffer;
	auto h_buffer = bufferPool.getBack();
	if (h_buffer)
	{
		int bufferlength = h_buffer->cols * h_buffer->rows * h_buffer->channels();
		buffer = gst_buffer_new_and_alloc(bufferlength);
		cv::Mat img = h_buffer->createMatHeader();
		GstMapInfo map;
		gst_buffer_map(buffer, &map, (GstMapFlags)GST_MAP_WRITE);
		memcpy(map.data, h_buffer->data, map.size);
		gst_buffer_unmap(buffer, &map);

		GST_BUFFER_PTS(buffer) = timestamp;
		
		GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(delta, GST_SECOND, 1000);
		timestamp += GST_BUFFER_DURATION(buffer);

		
		GstFlowReturn rw;
		g_signal_emit_by_name(source_OpenCV, "push-buffer", buffer, &rw);

		if (rw != GST_FLOW_OK)
		{
			NODE_LOG(error) << "Error pushing buffer into appsrc " << rw;
		}
		gst_buffer_unref(buffer);
	}
}

void RTSP_server::Serialize(ISimpleSerializer* pSerializer)
{
	Node::Serialize(pSerializer);
    SERIALIZE(glib_MainLoop);
    SERIALIZE(source_OpenCV);
    SERIALIZE(pipeline);
    SERIALIZE(need_data_id);
    SERIALIZE(enough_data_id);
    SERIALIZE(feed_enabled);
    SERIALIZE(delta);
    SERIALIZE(timestamp);
    SERIALIZE(prevTime);
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
	auto curTime = clock();
	delta = curTime - prevTime;
	prevTime = curTime;
	if (!g_main_loop_is_running(glib_MainLoop))
	{
		NODE_LOG(error) << "Main glib loop not running";
		return img;
	}
	if (feed_enabled)
	{
		auto buffer = bufferPool.getFront();
		img.download(*buffer, stream);
		stream.enqueueHostCallback(RTSP_serverCallback, this);
	}
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(RTSP_server);

