#include "ImageDisplay.h"
#include "../precompiled.hpp"
#include <MetaObject/Thread/InterThread.hpp>
#include <Aquila/utilities/CudaCallbacks.hpp>
#include <Aquila/utilities/UiCallbackHandlers.h>
#include <MetaObject/Logging/Profiling.hpp>
#include "MetaObject/Parameters/detail/TypedInputParameterPtrImpl.hpp"
#include "MetaObject/Parameters/detail/TypedParameterPtrImpl.hpp"


using namespace aq;
using namespace aq::Nodes;

bool QtImageDisplay::ProcessImpl()
{
    cv::Mat mat;
    boost::optional<mo::time_t> ts;
    bool overlay = overlay_timestamp;
    if(image && !image->empty())
    {
        mat = image->GetMat(Stream());
        ts = image_param.GetTimestamp();
    }
    if(cpu_mat)
    {
        mat = *cpu_mat;
    }

    std::string name = GetTreeName();
    if(!mat.empty())
    {
        size_t gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);

        aq::cuda::enqueue_callback_async(
            [mat, name, overlay, ts, this]()->void
        {
            PROFILE_RANGE(imshow);
            cv::Mat draw_img = mat;
            if(overlay && ts)
            {
                draw_img = mat.clone();
                std::stringstream ss;
                ss << "Timestamp: " << ts;
                cv::putText(mat, ss.str(), cv::Point(20,40), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0,255,0));
            }
            GetDataStream()->GetWindowCallbackManager()->imshow(name, draw_img);
        }, gui_thread_id, Stream());

        return true;
    }
    return false;
}


bool KeyPointDisplay::ProcessImpl()
{
    return true;
}

bool FlowVectorDisplay::ProcessImpl()
{
    return true;
}

bool HistogramDisplay::ProcessImpl()
{
    if(draw.empty())
    {
        cv::Mat h_draw;
        h_draw.create(100, 256, CV_MAKE_TYPE(CV_8U, histogram->GetChannels()));
        h_draw.setTo(cv::Scalar::all(0));
        // Add tick marks to the top of the histogram
        cv::line(h_draw, {256/2, 0}, {256/2, 5}, cv::Scalar::all(255));
        cv::line(h_draw, {256/4, 0}, {256/4, 3}, cv::Scalar::all(255));
        cv::line(h_draw, {3*256/4, 0}, {3*256/4, 3}, cv::Scalar::all(255));
        draw.upload(h_draw, Stream());
    }
    cv::cuda::GpuMat output_image;
    cv::cuda::drawHistogram(histogram->GetGpuMat(Stream()), output_image, cv::noArray(), Stream());
    std::string name = GetTreeName();
    size_t gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);
    //draw.copyTo(output_image, Stream());
    cv::cuda::add(output_image, draw, output_image, cv::noArray(), -1, Stream());
    aq::cuda::enqueue_callback_async(
                [name, output_image, this]()->void
    {
        PROFILE_RANGE(imshow);
        GetDataStream()->GetWindowCallbackManager()->imshowd(name, output_image, cv::WINDOW_OPENGL);
    }, gui_thread_id, Stream());
    return true;
}

bool HistogramOverlay::ProcessImpl()
{
    if(draw.empty())
    {
        cv::Mat h_draw;
        h_draw.create(100, 256, CV_MAKE_TYPE(CV_8U, histogram->GetChannels()));
        h_draw.setTo(cv::Scalar::all(0));
        // Add tick marks to the top of the histogram
        cv::line(h_draw, {256/2, 0}, {256/2, 5}, cv::Scalar::all(255));
        cv::line(h_draw, {256/4, 0}, {256/4, 3}, cv::Scalar::all(255));
        cv::line(h_draw, {3*256/4, 0}, {3*256/4, 3}, cv::Scalar::all(255));
        draw.upload(h_draw, Stream());
    }
    cv::cuda::GpuMat output_image;
    image->Clone(output_image, Stream());

    cv::cuda::drawHistogram(histogram->GetGpuMat(Stream()), output_image(cv::Rect(0,0, 256, 100)), cv::noArray(), Stream());
    //draw.copyTo(output_image(cv::Rect(0,0,256,100)), Stream());
    cv::cuda::add(output_image(cv::Rect(0,0,256,100)), draw, output_image(cv::Rect(0,0,256,100)),
                  cv::noArray(), -1, Stream());
    output_param.UpdateData(output_image, image_param.GetTimestamp(), _ctx);
    return true;
}

bool DetectionDisplay::ProcessImpl()
{
    return true;
}

bool OGLImageDisplay::ProcessImpl()
{
    std::string name = GetTreeName();
    size_t gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);
    cv::cuda::GpuMat gpumat = image->GetGpuMat(Stream());
    auto ts = image_param.GetTimestamp();
    if(!_prev_time)
        _prev_time = ts;
    auto prev = _prev_time;
    aq::cuda::enqueue_callback_async(
                [name, this, gpumat, ts, prev]()->void
    {
        PROFILE_RANGE(imshow);
        GetDataStream()->GetWindowCallbackManager()->imshowd(name, gpumat, cv::WINDOW_OPENGL);
        if(ts)
        {
            //std::cout << *ts - *prev  << std::endl;
        }
    }, gui_thread_id, Stream());
    _prev_time = ts;
    return true;
}


MO_REGISTER_CLASS(QtImageDisplay)
MO_REGISTER_CLASS(KeyPointDisplay)
MO_REGISTER_CLASS(FlowVectorDisplay)
MO_REGISTER_CLASS(HistogramDisplay)
MO_REGISTER_CLASS(HistogramOverlay)
MO_REGISTER_CLASS(DetectionDisplay)
MO_REGISTER_CLASS(OGLImageDisplay)
