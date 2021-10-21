#include "ImageDisplay.h"
#include "../precompiled.hpp"

#include <Aquila/gui/UiCallbackHandlers.h>

#include <MetaObject/core/metaobject_config.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/params/detail/TInputParamPtrImpl.hpp>
#include <MetaObject/params/detail/TParamPtrImpl.hpp>

#include <MetaObject/thread/ThreadRegistry.hpp>

using namespace aq;
using namespace aq::nodes;

namespace std
{
    ostream& operator<<(ostream& os, const cv::cuda::GpuMat& mat)
    {
        os << mat.type() << ' ' << mat.size();
        return os;
    }
} // namespace std

bool QtImageDisplay::processImpl()
{
    cv::Mat mat;
    mo::OptionalTime ts;
    bool overlay = overlay_timestamp;
    bool sync = false;
    if (input && !input->empty())
    {
        mo::IAsyncStreamPtr_t stream = this->getStream();
        // TODO use proper context based processImpl overloads
        mat = input->mat(stream.get());
        ts = input_param.getNewestTimestamp();
    }

    std::string name = getName();
    if (!mat.empty())
    {
        mo::IAsyncStream::Ptr_t gui_thread = mo::ThreadRegistry::instance()->getThread(mo::ThreadRegistry::GUI);
        if (sync)
        {
            gui_thread->pushWork([mat, name, overlay, ts, this, gui_thread](mo::IAsyncStream* stream) -> void {
                PROFILE_RANGE(imshow);
                MO_ASSERT(gui_thread.get() == stream);
                cv::Mat draw_img = mat;
                if (overlay && ts)
                {
                    draw_img = mat.clone();
                    std::stringstream ss;
                    ss << "Timestamp: " << ts;
                    cv::putText(mat, ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0));
                }
                getGraph()->getObject<WindowCallbackHandler>()->imshow(name, draw_img);
            });
        }
        else
        {
            cv::Mat draw_img = mat;
            if (overlay && ts)
            {
                draw_img = mat.clone();
                std::stringstream ss;
                ss << "Timestamp: " << ts;
                cv::putText(mat, ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0));
            }
            auto graph = getGraph();
            MO_ASSERT(graph != nullptr);
            auto window_manager = graph->getObject<WindowCallbackHandler>();
            MO_ASSERT(window_manager != nullptr);
            window_manager->imshow(name, draw_img);
        }

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

#if MO_OPENCV_HAVE_CUDA
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
MO_REGISTER_CLASS(HistogramDisplay)

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
MO_REGISTER_CLASS(HistogramOverlay)
#endif
bool DetectionDisplay::processImpl()
{
    return true;
}

bool OGLImageDisplay::processImpl()
{
    std::string name = getName();
    mo::IAsyncStreamPtr_t gui_stream = mo::ThreadRegistry::instance()->getThread(mo::ThreadRegistry::GUI);
    rcc::shared_ptr<OGLImageDisplay> self(*this);
    mo::IAsyncStreamPtr_t my_stream = this->getStream();
    mo::OptionalTime ts = input_param.getNewestHeader()->timestamp;

    if (m_use_opengl && gui_stream->isDeviceStream())
    {
        auto current_data = this->input_param.getCurrentData(gui_stream.get());
        cv::cuda::GpuMat gpumat = this->input->gpuMat(gui_stream->getDeviceStream());

        auto device_work_func = [this, name, self, ts, gpumat, current_data](mo::IAsyncStream*) {
            PROFILE_RANGE(imshow);
            try
            {
                auto graph = getGraph();
                if (graph)
                {
                    auto cbm = graph->getObject<WindowCallbackHandler>();
                    if (cbm)
                    {
                        cbm->imshowd(name, gpumat, cv::WINDOW_OPENGL);
                    }
                }
            }
            catch (mo::TExceptionWithCallstack<cv::Exception>& e)
            {
                m_use_opengl = false;
            }
        };

        if (my_stream != gui_stream)
        {
            gui_stream->pushWork(std::move(device_work_func));
        }
        else
        {
            device_work_func(my_stream.get());
        }
    }

    /*if (m_use_opengl && _ctx->device_id != -1)
    {

        if (!_prev_time)
            _prev_time = ts;
        auto prev = _prev_time;
        aq::cuda::enqueue_callback_async(
            [name, ptr, this, gpumat, ts, prev]() -> void {
                PROFILE_RANGE(imshow);
                try
                {
                    auto graph = getGraph();
                    if (graph)
                    {
                        auto cbm = graph->getWindowCallbackManager();
                        if (cbm)
                        {
                            cbm->imshowd(name, gpumat, cv::WINDOW_OPENGL);
                        }
                    }
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
        cv::Mat mat; // = image->getMat(stream());
        if (_ctx->device_id == -1)
        {
            mat = image->getMatNoSync();
            mo::ThreadSpecificQueue::push(
                [mat, this, ptr, name]() {
                    PROFILE_RANGE(imshow);
                    try
                    {
                        getGraph()->getWindowCallbackManager()->imshow(name, mat);
                    }
                    catch (mo::ExceptionWithCallStack<cv::Exception>& e)
                    {
                    }
                },
                gui_thread_id);
        }
        else
        {
            mat = image->getMat(stream());
            aq::cuda::enqueue_callback_async(
                [name, this, ptr, mat]() -> void {
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
    }*/
    return true;
}

MO_REGISTER_CLASS(QtImageDisplay)
MO_REGISTER_CLASS(KeyPointDisplay)
MO_REGISTER_CLASS(FlowVectorDisplay)
MO_REGISTER_CLASS(DetectionDisplay)
MO_REGISTER_CLASS(OGLImageDisplay)
