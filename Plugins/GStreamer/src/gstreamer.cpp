#include "gstreamer.hpp"

using namespace EagleLib;

void RTSP_server::setup()
{
    pipeline = gst_pipeline_new("RTSP server");
    source_OpenCV = gst_element_factory_make("appsrc", "Source_OpenCV");
    gst_bin_add(GST_BIN(pipeline), source_OpenCV);



}

void RTSP_server::Init(bool firstInit)
{

}

cv::cuda::GpuMat RTSP_server::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(RTSP_server)
