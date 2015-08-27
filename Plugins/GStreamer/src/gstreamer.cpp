#include "gstreamer.hpp"
#include <gst/video/video.h>
#include <gst/app/gstappsrc.h>
using namespace EagleLib;

SETUP_PROJECT_IMPL

typedef class RTSP_server App;
static gboolean
bus_message(GstBus * bus, GstMessage * message, App * app)
{
	//GST_DEBUG("got message %s",
	//	gst_message_type_get_name(GST_MESSAGE_TYPE(message)));
	BOOST_LOG_TRIVIAL(debug) << "Received message type: " << gst_message_type_get_name(GST_MESSAGE_TYPE(message));

	switch (GST_MESSAGE_TYPE(message)) {
	case GST_MESSAGE_ERROR: 
	{
		GError *err = NULL;
		gchar *dbg_info = NULL;

		gst_message_parse_error(message, &err, &dbg_info);
		BOOST_LOG_TRIVIAL(error) << "Error from element " << GST_OBJECT_NAME(message->src) << ": " << err->message;
		BOOST_LOG_TRIVIAL(error) << "Debugging info: " << (dbg_info) ? dbg_info : "none";
		//g_printerr("ERROR from element %s: %s\n",
		//	GST_OBJECT_NAME(message->src), err->message);
		//g_printerr("Debugging info: %s\n", (dbg_info) ? dbg_info : "none");
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
static gboolean read_data(App *app)
{
	GstBuffer *buffer;
	//guint8 *ptr;
	gint size;
	GstFlowReturn ret;

	//ptr = g_malloc(BUFF_SIZE);
	//g_assert(ptr);

	//size = fread(ptr, 1, BUFF_SIZE, app->file);

	//if (size == 0){
	//	ret = gst_app_src_end_of_stream(app->src);
	//	g_debug("eos returned %d at %d\n", ret, __LINE__);
	//	return FALSE;
	//}

	buffer = gst_buffer_new();
	auto h_buffer = app->bufferPool.getBack();
	if (h_buffer)
	{
		int bufferlength = h_buffer->cols * h_buffer->rows * h_buffer->channels();
		buffer = gst_buffer_new_and_alloc(bufferlength);
		

		//gst_buffer_pool_acquire_buffer(bufferPool, &buffer, nullptr);

		GstMapInfo map;
		gst_buffer_map(buffer, &map, (GstMapFlags)GST_MAP_WRITE);
		memcpy(map.data, h_buffer->data, map.size);
		gst_buffer_unmap(buffer, &map);
		size = h_buffer->rows * h_buffer->cols;

		ret = gst_app_src_push_buffer(GST_APP_SRC(app->source_OpenCV), buffer);
		gst_buffer_unref(buffer);
		if (ret != GST_FLOW_OK){
			g_debug("push buffer returned %d for %d bytes \n", ret, size);
			return FALSE;
		}	
	}
	return TRUE;
}

static void cb_need_data(GstElement *appsrc,
						guint  unused_size,
						App*    user_data)
{
	
}

static void stop_feed(GstElement * pipeline, App *app)
{
	if (app->sourceid != 0) {
		GST_DEBUG("stop feeding");
		//g_source_remove(app->sourceid);
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
void
Create_PipelineGraph(GstElement *pipeline) {
	bool debug_active = gst_debug_is_active();
	gst_debug_set_active(1);
	GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "PipelineGraph");
	gst_debug_set_active(debug_active);
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


	pipeline = gst_parse_launch("appsrc name=mysource  ! openh264enc ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=8004", nullptr);
	
	
	source_OpenCV = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");

	auto bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
	gst_object_unref(bus);
	CV_Assert(bus);

	g_signal_connect(source_OpenCV, "need-data", G_CALLBACK(cb_need_data), this);
	g_signal_connect(source_OpenCV, "enough-data", G_CALLBACK(stop_feed), this);

	/* add watch for messages */
	gst_bus_add_watch(bus, (GstBusFunc)bus_message, this);

	GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
	CV_Assert(ret != GST_STATE_CHANGE_FAILURE);
	Create_PipelineGraph(pipeline);
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
	setup();
}
void RTSP_server::push_image()
{
	static GstClockTime timestamp = 0;

	GstBuffer* buffer;
	auto h_buffer = bufferPool.getBack();
	if (h_buffer)
	{
		int bufferlength = h_buffer->cols * h_buffer->rows * h_buffer->channels();
		//buffer = bufferPool.getFront();
		//if (*buffer == nullptr)
		buffer = gst_buffer_new_and_alloc(bufferlength);
		
		//gst_buffer_pool_acquire_buffer(bufferPool, &buffer, nullptr);
		GstMapInfo map;
		gst_buffer_map(buffer, &map, (GstMapFlags)GST_MAP_WRITE);
		memcpy(map.data, h_buffer->data, map.size);
		gst_buffer_unmap(buffer, &map);

		GST_BUFFER_PTS(buffer) = timestamp;

		GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, 2);
		timestamp += GST_BUFFER_DURATION(buffer);


		GstCaps *caps_Source = NULL;

		caps_Source = gst_caps_new_simple("video/x-raw",
			"format", G_TYPE_STRING, "RGB",
			"width", G_TYPE_INT, h_buffer->cols,
			"height", G_TYPE_INT, h_buffer->rows,
			"bpp", G_TYPE_INT, 24,
			"depth", G_TYPE_INT, 24,
			"red_mask", G_TYPE_INT, 0x00ff0000,
			"green_mask", G_TYPE_INT, 0x0000ff00,
			"blue_mask", G_TYPE_INT, 0x000000ff,
			"framerate", GST_TYPE_FRACTION, 15, 1,
			"pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1,
			NULL);

		if (!GST_IS_CAPS(caps_Source)) {
			NODE_LOG(error)  << "Error creating Caps for OpenCV-Source";
			return;
		}

		gst_app_src_set_caps(GST_APP_SRC(source_OpenCV), caps_Source);
		//gst_buffer_set_caps(buffer, caps_Source);
		//g_object_set(G_OBJECT(buffer), "caps", caps_Source, NULL);
		gst_caps_unref(caps_Source);

		
		
		GstFlowReturn rw;

		//rw = gst_app_src_push_buffer(GST_APP_SRC(source_OpenCV), buffer);
		g_signal_emit_by_name(source_OpenCV, "push-buffer", buffer, &rw);

		if (rw != GST_FLOW_OK)
		{
			NODE_LOG(error) << "Error pushing buffer into appsrc " << rw;
		}
		//gst_buffer_unref(buffer);
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
	//SERIALIZE(glib_MainLoop);
	//SERIALIZE(imgSize);
	//SERIALIZE(h_buffer);
}
void RTSP_serverCallback(int status, void* userData)
{
	static_cast<RTSP_server*>(userData)->push_image();
}
cv::cuda::GpuMat RTSP_server::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
	if (!g_main_loop_is_running(glib_MainLoop))
	{
		NODE_LOG(error) << "Main glib loop not running";
		return img;
	}
	if (imgSize != img.size())
	{
		//gst_caps_new_simple
		//gst_video_format_new_caps(GST_VIDEO_FORMAT_RGB, 640, 480, 0, 1, 4, 3);
		//GstCaps* caps = gst_caps_new_simple("video/x-raw-rgb", "width", G_TYPE_INT, img.cols, "height", G_TYPE_INT, img.rows, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
		GstCaps* caps = gst_caps_new_simple("video/x-raw", 
			"format", G_TYPE_STRING, "RGB",
			"width", G_TYPE_INT, img.cols,
			"height", G_TYPE_INT, img.rows,
			"bpp", G_TYPE_INT, 24,
			"depth", G_TYPE_INT, 24,
			"red_mask", G_TYPE_INT, 0x00ff0000,
			"green_mask", G_TYPE_INT, 0x0000ff00,
			"blue_mask", G_TYPE_INT, 0x000000ff,
			"framerate", GST_TYPE_FRACTION, 15, 1,
			"pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1,
			NULL);

		if (caps == nullptr)
		{
			NODE_LOG(error) << "Error creating caps for appsrc";
		}
		//g_object_set(G_OBJECT(source_OpenCV), "caps", caps, NULL);
		gst_app_src_set_caps(GST_APP_SRC(source_OpenCV), caps);
		g_object_set(G_OBJECT(source_OpenCV), "stream-type", 0, "format", GST_FORMAT_TIME, NULL);
		imgSize = img.size();

	}
	auto buffer = bufferPool.getFront();
	img.download(*buffer, stream);
	stream.enqueueHostCallback(RTSP_serverCallback, this);
	//buffer->fillEvent.record(stream);
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(RTSP_server)


// ************************ Example ****************************
/*
#include <stdio.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

typedef struct {
	GstPipeline *pipeline;
	GstAppSrc *src;
	GstElement *sink;
	GstElement *decoder;
	GstElement *ffmpeg;
	GstElement *xvimagesink;
	GMainLoop *loop;
	guint sourceid;
	FILE *file;
}gst_app_t;

static gst_app_t gst_app;

#define BUFF_SIZE (1024)

static gboolean read_data(gst_app_t *app)
{
	GstBuffer *buffer;
	guint8 *ptr;
	gint size;
	GstFlowReturn ret;

	ptr = g_malloc(BUFF_SIZE);
	g_assert(ptr);

	size = fread(ptr, 1, BUFF_SIZE, app->file);

	if (size == 0){
		ret = gst_app_src_end_of_stream(app->src);
		g_debug("eos returned %d at %d\n", ret, __LINE__);
		return FALSE;
	}

	buffer = gst_buffer_new();
	GST_BUFFER_MALLOCDATA(buffer) = ptr;
	GST_BUFFER_SIZE(buffer) = size;
	GST_BUFFER_DATA(buffer) = GST_BUFFER_MALLOCDATA(buffer);

	ret = gst_app_src_push_buffer(app->src, buffer);

	if (ret != GST_FLOW_OK){
		g_debug("push buffer returned %d for %d bytes \n", ret, size);
		return FALSE;
	}

	if (size != BUFF_SIZE){
		ret = gst_app_src_end_of_stream(app->src);
		g_debug("eos returned %d at %d\n", ret, __LINE__);
		return FALSE;
	}

	return TRUE;
}

static void start_feed(GstElement * pipeline, guint size, gst_app_t *app)
{
	if (app->sourceid == 0) {
		GST_DEBUG("start feeding");
		app->sourceid = g_idle_add((GSourceFunc)read_data, app);
	}
}

static void stop_feed(GstElement * pipeline, gst_app_t *app)
{
	if (app->sourceid != 0) {
		GST_DEBUG("stop feeding");
		g_source_remove(app->sourceid);
		app->sourceid = 0;
	}
}

static void on_pad_added(GstElement *element, GstPad *pad)
{
	GstCaps *caps;
	GstStructure *str;
	gchar *name;
	GstPad *ffmpegsink;
	GstPadLinkReturn ret;

	g_debug("pad added");

	caps = gst_pad_get_caps(pad);
	str = gst_caps_get_structure(caps, 0);

	g_assert(str);

	name = (gchar*)gst_structure_get_name(str);

	g_debug("pad name %s", name);

	if (g_strrstr(name, "video")){

		ffmpegsink = gst_element_get_pad(gst_app.ffmpeg, "sink");
		g_assert(ffmpegsink);
		ret = gst_pad_link(pad, ffmpegsink);
		g_debug("pad_link returned %d\n", ret);
		gst_object_unref(ffmpegsink);
	}
	gst_caps_unref(caps);
}

static gboolean bus_callback(GstBus *bus, GstMessage *message, gpointer *ptr)
{
	gst_app_t *app = (gst_app_t*)ptr;

	switch (GST_MESSAGE_TYPE(message)){

	case GST_MESSAGE_ERROR:{
							   gchar *debug;
							   GError *err;

							   gst_message_parse_error(message, &err, &debug);
							   g_print("Error %s\n", err->message);
							   g_error_free(err);
							   g_free(debug);
							   g_main_loop_quit(app->loop);
	}
		break;

	case GST_MESSAGE_EOS:
		g_print("End of stream\n");
		g_main_loop_quit(app->loop);
		break;

	default:
		g_print("got message %s\n", \
			gst_message_type_get_name(GST_MESSAGE_TYPE(message)));
		break;
	}

	return TRUE;
}

int main(int argc, char *argv[])
{
	gst_app_t *app = &gst_app;
	GstBus *bus;
	GstStateChangeReturn state_ret;

	if (argc != 2){
		printf("File name not specified\n");
		return 1;
	}

	app->file = fopen(argv[1], "r");

	g_assert(app->file);

	gst_init(NULL, NULL);

	app->pipeline = (GstPipeline*)gst_pipeline_new("mypipeline");
	bus = gst_pipeline_get_bus(app->pipeline);
	gst_bus_add_watch(bus, (GstBusFunc)bus_callback, app);
	gst_object_unref(bus);

	app->src = (GstAppSrc*)gst_element_factory_make("appsrc", "mysrc");
	app->decoder = gst_element_factory_make("decodebin", "mydecoder");
	app->ffmpeg = gst_element_factory_make("ffmpegcolorspace", "myffmpeg");
	app->xvimagesink = gst_element_factory_make("xvimagesink", "myvsink");

	g_assert(app->src);
	g_assert(app->decoder);
	g_assert(app->ffmpeg);
	g_assert(app->xvimagesink);

	g_signal_connect(app->src, "need-data", G_CALLBACK(start_feed), app);
	g_signal_connect(app->src, "enough-data", G_CALLBACK(stop_feed), app);
	g_signal_connect(app->decoder, "pad-added", G_CALLBACK(on_pad_added), app->decoder);

	gst_bin_add_many(GST_BIN(app->pipeline), (GstElement*)app->src, app->decoder, app->ffmpeg, app->xvimagesink, NULL);

	if (!gst_element_link((GstElement*)app->src, app->decoder)){
		g_warning("failed to link src anbd decoder");
	}

	if (!gst_element_link(app->ffmpeg, app->xvimagesink)){
		g_warning("failed to link ffmpeg and xvsink");
	}

	state_ret = gst_element_set_state((GstElement*)app->pipeline, GST_STATE_PLAYING);
	g_warning("set state returned %d\n", state_ret);

	app->loop = g_main_loop_new(NULL, FALSE);
	printf("Running main loop\n");
	g_main_loop_run(app->loop);

	state_ret = gst_element_set_state((GstElement*)app->pipeline, GST_STATE_NULL);
	g_warning("set state null returned %d\n", state_ret);

	return 0;
}*/