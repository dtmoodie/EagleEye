#pragma once
#define PARAMETERS_USE_UI

#include "nodes/Node.h"
#include <EagleLib/Defs.hpp>
#include <gst/gst.h>
#include <gst/gstelement.h>
#include <gst/gstelementfactory.h>
#include <gst/gstutils.h>
#include <gst/gstpipeline.h>
#include <gst/app/gstappsrc.h>

SETUP_PROJECT_DEF
#ifdef _MSC_VER
RUNTIME_COMPILER_LINKLIBRARY("gstapp-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gstaudio-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gstbase-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gstcontroller-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gstnet-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gstpbutils-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gstreamer-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gstriff-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gstrtp-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gstrtsp-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gstsdp-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gsttag-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gstvideo-1.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("glib-2.0.lib");
RUNTIME_COMPILER_LINKLIBRARY("gobject-2.0.lib");
#endif

namespace EagleLib
{
    class RTSP_server: public Node
    {
		GstClockTime timestamp;
		time_t prevTime;
		time_t delta;
    public:
		enum ServerType
		{
			TCP = 0,
			UDP = 1
		};
		bool feed_enabled;
		GstElement* source_OpenCV;
		GstElement* pipeline;
		GMainLoop* glib_MainLoop;
		cv::Size imgSize;
		boost::thread glibThread;
		ConstBuffer<cv::cuda::HostMem> bufferPool;
		guint need_data_id;
		guint enough_data_id;

		void gst_loop();
		void Serialize(ISimpleSerializer* pSerializer);
		void push_image();
		void onPipeChange();
        void setup(std::string& pipeOverride = std::string());
        RTSP_server();
		~RTSP_server();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
/*
References
https://www.youtube.com/watch?v=7Xdry76ek5E
http://www.imgportal.net/home/wp-content/uploads/maris-script1.cpp
http://stackoverflow.com/questions/20219401/how-to-push-opencv-images-into-gstreamer-pipeline-to-stream-it-over-the-tcpserve
http://gstreamer.freedesktop.org/data/doc/gstreamer/head/manual/html/section-data-spoof.html

// GStreamer
#include <gstreamer-0.10/gst/gst.h>
#include <gstreamer-0.10/gst/gstelement.h>
#include <gstreamer-0.10/gst/gstpipeline.h>
#include <gstreamer-0.10/gst/gstutils.h>
#include <gstreamer-0.10/gst/app/gstappsrc.h>

// OpenCV
// #include "cv.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


GMainLoop *glib_MainLoop;
unsigned int heartbeat_Intervall; ///< In Milliseconds
boost::shared_ptr<boost::asio::deadline_timer> heartbeat;

GstElement *source_OpenCV;
guint64 imagecounter;

void
GLib_MainLoop() {
    if( !g_main_loop_is_running( glib_MainLoop ) ) {
        std::cout << "Starting glib_MainLoop..." << std::endl;
        g_main_loop_run( glib_MainLoop );
        std::cout << "Starting glib_MainLoop stopped." << std::endl;
    }
};

/// Creates an Image with a red filled Circle and the current Time displayed in it.
cv::Mat
Create_Image() {
    cv::Size size = cv::Size( 640, 480 );
    cv::Mat image = cv::Mat::zeros( size, CV_8UC3 );
    cv::Point center = cv::Point( size.width / 2, size.height / 2 );
    int thickness = -1;
    int lineType = 8;

    cv::circle( image, center, size.width / 4.0, cv::Scalar( 0, 0, 255 ), thickness, lineType );

    std::stringstream current_Time;
    boost::posix_time::time_facet *facet = new boost::posix_time::time_facet( "%Y.%m.%d %H:%M:%S.%f" );
    current_Time.imbue( std::locale( current_Time.getloc(), facet ) );
    current_Time << boost::posix_time::microsec_clock::universal_time();

    int font = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontsize = 1;
    int font_Thickness = 1;

    int baseline = 0;
    cv::Size textSize_01 = cv::getTextSize( current_Time.str(), font, fontsize, font_Thickness , &baseline );
    baseline += font_Thickness;

    cv::Point textOrg_01( ( image.cols - textSize_01.width ) / 2, ( image.rows + textSize_01.height * 2 ) / 2 );
    cv::Scalar textcolour = cv::Scalar( 0, 255, 0 );
    cv::putText( image, current_Time.str(), textOrg_01, font, fontsize, textcolour , font_Thickness , 1 );

    return image;
}

/// Creates a Graph of the created Pipeline including the contained Elements. The environment variable "GST_DEBUG_DUMP_DOT_DIR" must be set, e.g to /tmp/ to actually create the Graph.
/// Furthermore GST_DEBUG needs to be activated, e.g. with "GST_DEBUG=3".
/// So "GST_DEBUG=3 GST_DEBUG_DUMP_DOT_DIR=/tmp/" ./Sandbox would work.
/// The .dot file can be converted to a e.g. svg-Graphic with the following command (Package GraphViz): dot -Tsvg -oPipelineGraph.svg PipelineGraph.dot
void
Create_PipelineGraph( GstElement *pipeline ) {
    bool debug_active = gst_debug_is_active();
    gst_debug_set_active( 1 );
    GST_DEBUG_BIN_TO_DOT_FILE( GST_BIN( pipeline ), GST_DEBUG_GRAPH_SHOW_ALL, "PipelineGraph" );
    gst_debug_set_active( debug_active );
}

void
Push_new_Image( const boost::system::error_code &error ) {
    if( error != 0 ) {
        std::cout << "Error in Timer: " << error.message() << std::endl;
        return;
    }

    cv::Mat image = Create_Image();

    /// OpenCV handles image in BGR, so to get RGB, Channels R and B needs to be swapped.
    cv::cvtColor( image, image, CV_CVTIMG_SWAP_RB );

    {
        /// How do i get the actual bpp and depth out of the cv::Mat?
        GstCaps *caps = gst_caps_new_simple( "video/x-raw-rgb", "width", G_TYPE_INT, image.cols, "height", G_TYPE_INT, image.rows, "framerate", GST_TYPE_FRACTION, 0, 1, NULL );
        g_object_set( G_OBJECT( source_OpenCV ), "caps", caps, NULL );
        gst_caps_unref( caps );

        IplImage* img = new IplImage( image );
        uchar *IMG_data = ( uchar* ) img->imageData;

        GstBuffer *buffer;
        {
            int bufferlength = image.cols * image.rows * image.channels();
            buffer = gst_buffer_new_and_alloc( bufferlength );

            /// Copy Data from OpenCV to GStreamer
            memcpy( GST_BUFFER_DATA( buffer ), IMG_data, GST_BUFFER_SIZE( buffer ) );

            GST_BUFFER_DURATION( buffer ) = gst_util_uint64_scale( bufferlength, GST_SECOND, 1 );
        }

        /// Setting the Metadata for the image to be pushed.
        {
            GstCaps *caps_Source = NULL;

            std::stringstream video_caps_text;
            video_caps_text << "video/x-raw-rgb,width=(int)" << image.cols << ",height=(int)" << image.rows << ",framerate=(fraction)0/1";
            caps_Source = gst_caps_from_string( video_caps_text.str().c_str() );

            if( !GST_IS_CAPS( caps_Source ) ) {
                std::cout << "Error creating Caps for OpenCV-Source, exiting...";
                exit( 1 );
            }

            gst_app_src_set_caps( GST_APP_SRC( source_OpenCV ), caps_Source );
            gst_buffer_set_caps( buffer, caps_Source );
            gst_caps_unref( caps_Source );
        }

        /// Setting a continues timestamp
        GST_BUFFER_TIMESTAMP( buffer ) = gst_util_uint64_scale( imagecounter * 20, GST_MSECOND, 1 );
        imagecounter += 1;

        /// Push Buffer into GStreamer-Pipeline
        GstFlowReturn rw;
        rw = gst_app_src_push_buffer( GST_APP_SRC( source_OpenCV ), buffer );

        if( rw != GST_FLOW_OK ) {
            std::cout << "Error push buffer to GStreamer-Pipeline, exiting...";

            exit( 1 );
        } else {
            std::cout << "GST_FLOW_OK " << "imagecounter: " << imagecounter << std::endl;
        }

    }

    /// Renew the Heartbeat-Timer
    heartbeat->expires_from_now( boost::posix_time::milliseconds( heartbeat_Intervall ) );
    heartbeat->async_wait( Push_new_Image );
}

int
main( int argc, char **argv ) {
    std::cout << "Sandbox started." << std::endl;

    /// ####################
    /// Initialise Sandbox
    /// ####################
    boost::shared_ptr<boost::asio::io_service> io_service = boost::make_shared<boost::asio::io_service>();
    boost::shared_ptr<boost::asio::io_service::work> work = boost::make_shared<boost::asio::io_service::work>( *io_service );
    boost::shared_ptr<boost::thread_group> threadgroup = boost::make_shared<boost::thread_group>();

    /// io_service Callback for continuously feeding into the pipeline of GStreamer.
    /// I've using to push the Buffer into GStreamer as i come available instead of getting informed about an empty pipeline by GStreamer-Signals.
    heartbeat_Intervall = 1000; ///< In Milliseconds
    heartbeat = boost::make_shared<boost::asio::deadline_timer>( ( *( io_service.get() ) ) );

    std::cout << "Initialise GStreamer..." << std::endl;
    gst_init( &argc, &argv );

    glib_MainLoop = g_main_loop_new( NULL, 0 );

    std::cout << "Start GLib_MainLoop..." << std::endl;
    io_service->post( GLib_MainLoop );

    /// Create some Workerthreads
    for( std::size_t i = 0; i < 3; ++i )  {
        threadgroup->create_thread( boost::bind( &boost::asio::io_service::run, &( *io_service ) ) );
    }

    /// ####################
    /// Do the actual Work
    /// ####################
    GstElement *pipeline;
    GstElement *converter_FFMpegColorSpace;
    GstElement *converter_VP8_Encoder;
    GstElement *muxer_WebM;
    GstElement *sink_TCPServer;


    /// Create GStreamer Elements

    pipeline = gst_pipeline_new( "OpenCV_to_TCPServer" );

    if( !pipeline ) {
        std::cout << "Error creating Pipeline, exiting...";
        return 1;
    }

    {
        source_OpenCV = gst_element_factory_make( "appsrc", "Source_OpenCV" );

        if( !source_OpenCV ) {
            std::cout << "Error creating OpenCV-Source, exiting...";
            return 1;
        }

        gst_bin_add( GST_BIN( pipeline ), source_OpenCV );
    }

    {
        converter_FFMpegColorSpace = gst_element_factory_make( "ffmpegcolorspace", "Converter_FFMpegColorSpace" );

        if( !converter_FFMpegColorSpace ) {
            std::cout << "Error creating Converter_FFMpegColorSpace, exiting...";
            return 1;
        }

        gst_bin_add( GST_BIN( pipeline ), converter_FFMpegColorSpace );
    }

    {
        converter_VP8_Encoder = gst_element_factory_make( "vp8enc", "Converter_VP8_Encoder" );

        if( !converter_VP8_Encoder ) {
            std::cout << "Error creating Converter_VP8_Encoder, exiting...";
            return 1;
        }

        gst_bin_add( GST_BIN( pipeline ), converter_VP8_Encoder );
    }

    {
        muxer_WebM = gst_element_factory_make( "webmmux", "Muxer_WebM" );

        if( !muxer_WebM ) {
            std::cout << "Error creating Muxer_WebM, exiting...";
            return 1;
        }

        gst_bin_add( GST_BIN( pipeline ), muxer_WebM );
    }

    {
        sink_TCPServer = gst_element_factory_make( "tcpserversink", "Sink_TCPServer" );

        if( !sink_TCPServer ) {
            std::cout << "Error creating Sink_TCPServer, exiting...";
            return 1;
        }

        gst_bin_add( GST_BIN( pipeline ), sink_TCPServer );
    }


    /// Link GStreamer Elements

    if( !gst_element_link( source_OpenCV, converter_FFMpegColorSpace ) ) {
        std::cout << "Error linking creating source_OpenCV to converter_FFMpegColorSpace, exiting...";
        return 2;
    }

    if( !gst_element_link( converter_FFMpegColorSpace, converter_VP8_Encoder ) ) {
        std::cout << "Error linking creating converter_FFMpegColorSpace to converter_VP8_Encoder, exiting...";
        return 2;
    }

    if( !gst_element_link( converter_VP8_Encoder, muxer_WebM ) ) {
        std::cout << "Error linking creating converter_VP8_Encoder to muxer_WebM, exiting...";
        return 2;
    }

    if( !gst_element_link( muxer_WebM, sink_TCPServer ) ) {
        std::cout << "Error linking creating muxer_WebM to sink_TCPServer, exiting...";
        return 2;
    }


    /// Set State of the GStreamer Pipeline to Playing
    GstStateChangeReturn ret = gst_element_set_state( pipeline, GST_STATE_PLAYING );

    if( ret == GST_STATE_CHANGE_FAILURE ) {
        std::cout << "Error setting GStreamer-Pipeline to playing.";
        return 2;
    }

    Create_PipelineGraph( pipeline );


    /// Start the Heartbeat, that continously creates new Images
    heartbeat->expires_from_now( boost::posix_time::milliseconds( heartbeat_Intervall ) );
    heartbeat->async_wait( Push_new_Image );

    /// ####################
    /// Shutdown the Sandbox
    /// ####################
    std::cout << "Wait some Seconds before joining all Threads and shutdown the Sandbox..." << std::endl;
    boost::this_thread::sleep( boost::posix_time::seconds( 4 ) );

    std::cout << "Shutdown Sandbox..." << std::endl;
    g_main_loop_quit( glib_MainLoop );
    io_service->stop();

    while( !io_service->stopped() ) {
        boost::this_thread::sleep( boost::posix_time::seconds( 1 ) );
    }

    work.reset();
    threadgroup->join_all();

    g_main_loop_unref( glib_MainLoop );

    threadgroup.reset();
    work.reset();
    io_service.reset();

    std::cout << "Sandbox stopped" << std::endl;
}


*/
