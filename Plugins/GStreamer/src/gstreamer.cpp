#include "gstreamer.hpp"
#include <gst/video/video.h>
#include <gst/app/gstappsrc.h>
#include "../remotery/lib/Remotery.h"
#include <QtNetwork/qnetworkinterface.h>
#include <EagleLib/rcc/SystemTable.hpp>


using namespace EagleLib;
using namespace EagleLib::Nodes;


SETUP_PROJECT_IMPL

typedef class RTSP_server App;
static gboolean
bus_message(GstBus * bus, GstMessage * message, App * app)
{
	BOOST_LOG_TRIVIAL(debug) << "Received message type: " << gst_message_type_get_name(GST_MESSAGE_TYPE(message));

	switch (GST_MESSAGE_TYPE(message)) 
	{
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
// This only actually gets called when gstreamer.cpp gets recompiled or the node is deleted
RTSP_server::~RTSP_server()
{
    NODE_LOG(info) << "RTSP server destructor";
	if (pipeline)
	{
		GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_NULL);
	}
    
#ifdef _MSC_VER
    PerModuleInterface::GetInstance()->GetSystemTable()->SetSingleton<GMainLoop>(nullptr);
#endif
	if (glib_MainLoop)
	{
		g_main_loop_quit(glib_MainLoop);
		glibThread.join();
		g_main_loop_unref(glib_MainLoop);
		
	}
	if (pipeline)
	{
		gst_object_unref(pipeline);
	}
	if (source_OpenCV)
	{
		g_signal_handler_disconnect(source_OpenCV, need_data_id);
		g_signal_handler_disconnect(source_OpenCV, enough_data_id);
		gst_object_unref(source_OpenCV);
	}
	
	
}

void RTSP_server::gst_loop()
{
	if (!g_main_loop_is_running(glib_MainLoop))
	{
		g_main_loop_run(glib_MainLoop);
	}
}
void RTSP_server::onPipeChange()
{
	std::string* str = getParameter<std::string>("gst pipeline")->Data();
	if (str->size())
	{
		setup(*str);
	}
}

void RTSP_server::setup(std::string pipeOverride)
{
	rmt_ScopedCPUSample(RTSP_server_setup);
	gst_debug_set_active(1);

	if (!gst_is_initialized())
	{
		char** argv;
		argv = new char*{ "-vvv" };
		int argc = 1;
		gst_init(&argc, &argv);
	}

	if (!glib_MainLoop)
	{
		glib_MainLoop = g_main_loop_new(NULL, 0);
	}
	glibThread = boost::thread(std::bind(&RTSP_server::gst_loop, this));
	GError* error = nullptr;
	std::stringstream ss;
	if (pipeOverride.size() == 0)
	{
		ss << "appsrc name=mysource ! videoconvert ! ";
#ifdef JETSON
		ss << "omxh264enc ! ";
#else
		ss << "openh264enc ! ";
#endif
		ss << "rtph264pay config-interval=1 pt=96 ! gdppay ! ";

		switch (getParameter<Parameters::EnumParameter>(0)->Data()->currentSelection)
		{
			case TCP:
			{
				ss << "tcpserversink host=";

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
								updateParameter<std::string>("Host", entry.ip().toString().toStdString());
								ss << entry.ip().toString().toStdString();
								break;
							}
						}
					}
				}
				break;
			}
			case UDP:
			{
				ss << "udpsink host=";
				std::string* host = getParameter<std::string>("Host")->Data();
				if (host->size())
				{
					ss << *host;
				}
				else
				{
					NODE_LOG(warning) << "host not set, setting to localhost";
					updateParameter<std::string>("Host", "127.0.0.1");
					ss << "127.0.0.1";
				}
				break;
			}
		}
		ss << " port=";
		ss << *getParameter<unsigned short>("Port")->Data();
		pipeOverride = ss.str();
	}
	else
	{
		updateParameter<std::string>("gst pipeline", pipeOverride);
	}
	
	NODE_LOG(info) << pipeOverride;

    
	if (pipeline)
	{
		GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_NULL);
		gst_object_unref(pipeline);
		g_signal_handler_disconnect(source_OpenCV, need_data_id);
		g_signal_handler_disconnect(source_OpenCV, enough_data_id);
		gst_object_unref(source_OpenCV);
	}


    
	pipeline = gst_parse_launch(pipeOverride.c_str(), &error);
	
	if (pipeline == nullptr)
	{
		NODE_LOG(error) << "Error parsing pipeline";
		feed_enabled = false;
		return;
	}
	
	if (error != nullptr)
	{
		NODE_LOG(error) << "Error parsing pipeline " << error->message;
	}
    
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
	if (!firstInit)
	{

	}
	if (firstInit)
	{
		timestamp = 0;
		prevTime = clock();
        
		Parameters::EnumParameter server_type;
		server_type.addEnum(ENUM(TCP));
		server_type.addEnum(ENUM(UDP));
		updateParameter("Server type", server_type);
		updateParameter<unsigned short>("Port", 8004);
		updateParameter<std::string>("Host", "")->SetTooltip("When TCP is selected, this is the address of the device to bind to, when UDP is selected this is the address of the device to receive the video stream")->type = Parameters::Parameter::Control;
		updateParameter<std::string>("gst pipeline", "");
	}
	bufferPool.resize(5);
}
void RTSP_server::push_image()
{
	rmt_ScopedCPUSample(RTSP_server_push_image);
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
		glibThread = boost::thread(std::bind(&RTSP_server::gst_loop, this));
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

RTSP_server::RTSP_server():
	Node()
{
	source_OpenCV = nullptr;
	pipeline = nullptr;
	glib_MainLoop = nullptr;
	feed_enabled = false;
	need_data_id = 0; 
	enough_data_id = 0;
}					
static EagleLib::Nodes::NodeInfo g_registerer_RTSP_server("RTSP_server", { "Image", "Sink" });

REGISTERCLASS(RTSP_server, &g_registerer_RTSP_server)


// http://cgit.freedesktop.org/gstreamer/gst-rtsp-server/tree/examples/test-appsrc.c

RTSP_server_new::RTSP_server_new()
{
	loop = nullptr;
	server = nullptr;
	factory = nullptr;
	pipeline = nullptr;
	appsrc = nullptr;
	clientCount = 0;
	connected = false;
	first_run = true;
}
RTSP_server_new::~RTSP_server_new()
{
	NODE_LOG(info) << "Shutting down rtsp server";
	if(pipeline)
		gst_element_set_state(pipeline, GST_STATE_NULL);
	g_main_loop_quit(loop);
	if(factory)
		gst_object_unref(factory);
	if (server)
	{
		g_source_remove(server_id);
		gst_object_unref(server);
	}
		
	glib_thread.join();
	if(loop)
		g_main_loop_unref(loop);
	if(pipeline)
		gst_object_unref(pipeline);
	if(appsrc)
		gst_object_unref(appsrc);
}
void RTSP_server_new::push_image()
{

}
void RTSP_server_new::onPipeChange()
{

}
void RTSP_server_new::glibThread()
{
	if (!g_main_loop_is_running(loop))
	{
		g_main_loop_run(loop);
	}
	BOOST_LOG_TRIVIAL(info) << "[RTSP Server] Gmain loop quitting";
}
void RTSP_server_new::setup(std::string pipeOverride)
{
	
}
void rtsp_server_new_need_data_callback(GstElement * appsrc, guint unused, gpointer user_data)
{
	auto node = static_cast<EagleLib::Nodes::RTSP_server_new*>(user_data);
	cv::cuda::HostMem* h_buffer = nullptr;
	node->notifier.wait_and_pop(h_buffer);
	if (h_buffer && node->connected)
	{
		int bufferlength = h_buffer->cols * h_buffer->rows * h_buffer->channels();
		auto buffer = gst_buffer_new_and_alloc(bufferlength);
		cv::Mat img = h_buffer->createMatHeader();
		GstMapInfo map;
		gst_buffer_map(buffer, &map, (GstMapFlags)GST_MAP_WRITE);
		memcpy(map.data, h_buffer->data, map.size);
		gst_buffer_unmap(buffer, &map);

		GST_BUFFER_PTS(buffer) = node->timestamp;

		GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(node->delta, GST_SECOND, 1000);
		node->timestamp += GST_BUFFER_DURATION(buffer);


		GstFlowReturn rw;
		g_signal_emit_by_name(appsrc, "push-buffer", buffer, &rw);

		if (rw != GST_FLOW_OK)
		{
			NODE_LOG(error) << "Error pushing buffer into appsrc " << rw;
		}
		gst_buffer_unref(buffer);
	}


}

static gboolean
bus_message_new(GstBus * bus, GstMessage * message, RTSP_server_new * app)
{
	BOOST_LOG_TRIVIAL(debug) << "Received message type: " << gst_message_type_get_name(GST_MESSAGE_TYPE(message));

	switch (GST_MESSAGE_TYPE(message))
	{
	case GST_MESSAGE_ERROR:
	{
		GError *err = NULL;
		gchar *dbg_info = NULL;

		gst_message_parse_error(message, &err, &dbg_info);
		BOOST_LOG_TRIVIAL(error) << "Error from element " << GST_OBJECT_NAME(message->src) << ": " << err->message;
		BOOST_LOG_TRIVIAL(error) << "Debugging info: " << (dbg_info) ? dbg_info : "none";
		g_error_free(err);
		g_free(dbg_info);
		g_main_loop_quit(app->loop);
		break;
	}
	case GST_MESSAGE_EOS:
		g_main_loop_quit(app->loop); 
		break;
	default:
		break;
	}
	return TRUE;
}
void client_close_handler(GstRTSPClient *client, EagleLib::Nodes::RTSP_server_new* node)
{
	node->clientCount--;
	BOOST_LOG_TRIVIAL(info) << "[RTSP Server] Client Disconnected " << node->clientCount << " " << client;
	if (node->clientCount == 0)
	{
		BOOST_LOG_TRIVIAL(info) << "[RTSP Server] Setting pipeline state to GST_STATE_NULL and unreffing old pipeline";
		gst_element_set_state(node->pipeline,GST_STATE_NULL);
		gst_object_unref(node->pipeline);
		gst_object_unref(node->appsrc);
		node->appsrc = nullptr;
		node->pipeline = nullptr;
		node->connected = false;
	}
}
void media_configure(GstRTSPMediaFactory * factory, GstRTSPMedia * media, EagleLib::Nodes::RTSP_server_new* node)
{
	if (node->imgSize.area() == 0)
	{
		return;
	}

	// get the element used for providing the streams of the media 
	node->pipeline = gst_rtsp_media_get_element(media);

	// get our appsrc, we named it 'mysrc' with the name property 
	node->appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(node->pipeline), "mysrc");

	BOOST_LOG_TRIVIAL(info) << "[RTSP Server] Configuring pipeline " << node->clientCount << " " << node->pipeline << " " << node->appsrc;

	// this instructs appsrc that we will be dealing with timed buffer 
	gst_util_set_object_arg(G_OBJECT(node->appsrc), "format", "time");

	// configure the caps of the video 
	g_object_set(G_OBJECT(node->appsrc), "caps",
		gst_caps_new_simple("video/x-raw",
			"format", G_TYPE_STRING, "BGR",
			"width", G_TYPE_INT, node->imgSize.width,
			"height", G_TYPE_INT, node->imgSize.height,
			"framerate", GST_TYPE_FRACTION, 30, 1, NULL), NULL);


	// install the callback that will be called when a buffer is needed 
	g_signal_connect(node->appsrc, "need-data", (GCallback)rtsp_server_new_need_data_callback, node);
	//gst_object_unref(appsrc);
	//gst_object_unref(element);
}
void new_client_handler(GstRTSPServer *server, GstRTSPClient *client, EagleLib::Nodes::RTSP_server_new* node)
{
	
	node->clientCount++;
	node->connected = true;
	BOOST_LOG_TRIVIAL(info) << "New client connected " << node->clientCount << " " << client;
	if (node->first_run)
	{
		g_signal_connect(node->factory, "media-configure", G_CALLBACK(media_configure), node);
	}
	g_signal_connect(client, "closed", G_CALLBACK(client_close_handler), node);
	node->first_run = false;
}


void RTSP_server_new::Init(bool firstInit)
{
	if (firstInit)
	{
		GstRTSPMountPoints *mounts = nullptr;
		gst_debug_set_active(1);
		if (firstInit)
		{
			timestamp = 0;
			prevTime = clock();
		}
		if (!gst_is_initialized())
		{
			char** argv;
			argv = new char*{ "-vvv" };
			int argc = 1;
			gst_init(&argc, &argv);
		}
		if (!loop)
			loop = g_main_loop_new(NULL, FALSE);

		// create a server instance 
		if (!server)
			server = gst_rtsp_server_new();
		// get the mount points for this server, every server has a default object
		// that be used to map uri mount points to media factories 
		if (!mounts)
			mounts = gst_rtsp_server_get_mount_points(server);

		// make a media factory for a test stream. The default media factory can use
		// gst-launch syntax to create pipelines.
		// any launch line works as long as it contains elements named pay%d. Each
		// element with pay%d names will be a stream 
		if (!factory)
		{
			factory = gst_rtsp_media_factory_new();
		}
		gst_rtsp_media_factory_set_shared(factory, true);

		gst_rtsp_media_factory_set_launch(factory,
			"( appsrc name=mysrc ! videoconvert ! openh264enc ! rtph264pay name=pay0 pt=96 )");

		// notify when our media is ready, This is called whenever someone asks for
		// the media and a new pipeline with our appsrc is created 
		//g_signal_connect(factory, "media-configure", G_CALLBACK(media_configure), this);

		// attach the test factory to the /test url 
		gst_rtsp_mount_points_add_factory(mounts, "/test", factory);

		// don't need the ref to the mounts anymore 
		g_object_unref(mounts);

		// attach the server to the default maincontext 
		server_id = gst_rtsp_server_attach(server, NULL);
		g_signal_connect(server, "client-connected",G_CALLBACK(new_client_handler), this);

		glib_thread = boost::thread(std::bind(&RTSP_server_new::glibThread, this));
	}
}
void RTSP_server_download_callback(int status, void* user_data)
{
	auto node = static_cast<RTSP_server_new*>(user_data);
	auto buf = node->hostBuffer.getBack();
	if (buf)
	{
		if (!buf->empty())
		{
			node->notifier.push(buf);
		}
	}
}
cv::cuda::GpuMat RTSP_server_new::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
	imgSize = img.size();
	auto curTime = clock();
	delta = curTime - prevTime;
	prevTime = curTime;
	auto buf = hostBuffer.getFront();
	img.download(*buf, stream);
	stream.enqueueHostCallback(RTSP_server_download_callback, this);
	return img;
}

static EagleLib::Nodes::NodeInfo g_registerer_RTSP_server_new("RTSP_server_new", { "Image", "Sink" });
REGISTERCLASS(RTSP_server_new, &g_registerer_RTSP_server_new);
