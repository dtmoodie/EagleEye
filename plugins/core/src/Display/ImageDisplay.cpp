#include "ImageDisplay.h"
#include "../precompiled.hpp"
#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include <Aquila/gui/UiCallbackHandlers.h>
#include <Aquila/utilities/cuda/CudaCallbacks.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/thread/InterThread.hpp>

using namespace aq;
using namespace aq::nodes;

bool QtImageDisplay::processImpl()
{
    cv::Mat mat;
    boost::optional<mo::Time_t> ts;
    bool overlay = overlay_timestamp;
    if (image && !image->empty())
    {
        mat = image->getMat(stream());
        ts = image_param.getTimestamp();
    }
    if (cpu_mat)
    {
        mat = *cpu_mat;
    }

    std::string name = getTreeName();
    if (!mat.empty())
    {
        size_t gui_thread_id = mo::ThreadRegistry::instance()->getThread(mo::ThreadRegistry::GUI);

        aq::cuda::enqueue_callback_async(
            [mat, name, overlay, ts, this]() -> void {
                PROFILE_RANGE(imshow);
                cv::Mat draw_img = mat;
                if (overlay && ts)
                {
                    draw_img = mat.clone();
                    std::stringstream ss;
                    ss << "Timestamp: " << ts;
                    cv::putText(mat, ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0));
                }
                getGraph()->getWindowCallbackManager()->imshow(name, draw_img);
            },
            gui_thread_id,
            stream());

        return true;
    }
    return false;
}

bool KeyPointDisplay::processImpl()
{
    return true;
}

bool FlowVectorDisplay::processImpl()
{
    return true;
}

bool HistogramDisplay::processImpl()
{
    if (draw.empty())
    {
        cv::Mat h_draw;
        h_draw.create(100, 256, CV_MAKE_TYPE(CV_8U, histogram->getChannels()));
        h_draw.setTo(cv::Scalar::all(0));
        // Add tick marks to the top of the histogram
        cv::line(h_draw, {256 / 2, 0}, {256 / 2, 5}, cv::Scalar::all(255));
        cv::line(h_draw, {256 / 4, 0}, {256 / 4, 3}, cv::Scalar::all(255));
        cv::line(h_draw, {3 * 256 / 4, 0}, {3 * 256 / 4, 3}, cv::Scalar::all(255));
        draw.upload(h_draw, stream());
    }
    cv::cuda::GpuMat output_image;
    cv::cuda::drawHistogram(histogram->getGpuMat(stream()), output_image, cv::noArray(), stream());
    std::string name = getTreeName();
    size_t gui_thread_id = mo::ThreadRegistry::instance()->getThread(mo::ThreadRegistry::GUI);
    // draw.copyTo(output_image, stream());
    cv::cuda::add(output_image, draw, output_image, cv::noArray(), -1, stream());
    aq::cuda::enqueue_callback_async(
        [name, output_image, this]() -> void {
            PROFILE_RANGE(imshow);
            getGraph()->getWindowCallbackManager()->imshowd(name, output_image, cv::WINDOW_OPENGL);
        },
        gui_thread_id,
        stream());
    return true;
}

bool HistogramOverlay::processImpl()
{
    if (draw.empty())
    {
        cv::Mat h_draw;
        h_draw.create(100, 256, CV_MAKE_TYPE(CV_8U, histogram->getChannels()));
        h_draw.setTo(cv::Scalar::all(0));
        // Add tick marks to the top of the histogram
        cv::line(h_draw, {256 / 2, 0}, {256 / 2, 5}, cv::Scalar::all(255));
        cv::line(h_draw, {256 / 4, 0}, {256 / 4, 3}, cv::Scalar::all(255));
        cv::line(h_draw, {3 * 256 / 4, 0}, {3 * 256 / 4, 3}, cv::Scalar::all(255));
        draw.upload(h_draw, stream());
    }
    cv::cuda::GpuMat output_image;
    image->clone(output_image, stream());

    cv::cuda::drawHistogram(
        histogram->getGpuMat(stream()), output_image(cv::Rect(0, 0, 256, 100)), cv::noArray(), stream());
    // draw.copyTo(output_image(cv::Rect(0,0,256,100)), stream());
    cv::cuda::add(output_image(cv::Rect(0, 0, 256, 100)),
                  draw,
                  output_image(cv::Rect(0, 0, 256, 100)),
                  cv::noArray(),
                  -1,
                  stream());
    output_param.updateData(output_image, image_param.getTimestamp(), _ctx.get());
    return true;
}

bool DetectionDisplay::processImpl()
{
    return true;
}

bool OGLImageDisplay::processImpl()
{
    std::string name = getTreeName();
    size_t gui_thread_id = mo::ThreadRegistry::instance()->getThread(mo::ThreadRegistry::GUI);
    if (m_use_opengl && _ctx->device_id != -1)
    {
        cv::cuda::GpuMat gpumat = image->getGpuMat(stream());
        auto ts = image_param.getTimestamp();
        if (!_prev_time)
            _prev_time = ts;
        auto prev = _prev_time;
        aq::cuda::enqueue_callback_async(
            [name, this, gpumat, ts, prev]() -> void {
                PROFILE_RANGE(imshow);
                try
                {
                    getGraph()->getWindowCallbackManager()->imshowd(name, gpumat, cv::WINDOW_OPENGL);
                }
                catch (mo::ExceptionWithCallStack<cv::Exception>& e)
                {
                    m_use_opengl = false;
                }
            },
            gui_thread_id,
            stream());
        _prev_time = ts;
    }
    else
    {
        cv::Mat mat;// = image->getMat(stream());
        if(_ctx->device_id == -1)
        {
            mat = image->getMatNoSync();
            mo::ThreadSpecificQueue::push([mat, this, name]()
            {
                PROFILE_RANGE(imshow);
                try
                {
                    getGraph()->getWindowCallbackManager()->imshow(name, mat);
                }
                catch (mo::ExceptionWithCallStack<cv::Exception>& e)
                {
                }
            }, gui_thread_id);
        }
        else
        {
            mat = image->getMat(stream());
            aq::cuda::enqueue_callback_async(
                [name, this, mat]() -> void {
                    PROFILE_RANGE(imshow);
                    try
                    {
                        getGraph()->getWindowCallbackManager()->imshow(name, mat);
                    }
                    catch (mo::ExceptionWithCallStack<cv::Exception>& e)
                    {
                    }
                },
                gui_thread_id,
                stream());
        }

    }
    return true;
}

MO_REGISTER_CLASS(QtImageDisplay)
MO_REGISTER_CLASS(KeyPointDisplay)
MO_REGISTER_CLASS(FlowVectorDisplay)
MO_REGISTER_CLASS(HistogramDisplay)
MO_REGISTER_CLASS(HistogramOverlay)
MO_REGISTER_CLASS(DetectionDisplay)
MO_REGISTER_CLASS(OGLImageDisplay)
