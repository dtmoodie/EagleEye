#include "EagleLib/utilities/WindowCallbackManager.h"
#include "EagleLib/utilities/UiCallbackHandlers.h"
#include "ObjectInterfacePerModule.h"
#include "EagleLib/rcc/SystemTable.hpp"
#include <EagleLib/utilities/CudaCallbacks.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace EagleLib;

/*void WindowHandler::mouseCallback(int ev, int x, int y, int flags, void* manager)
{
    WindowHandler* mgr = static_cast<WindowHandler*>(manager);
    switch(ev):
    {
    case cv::EVENT_MBUTTONDOWN:
    case cv::EVENT_LBUTTONDOWN:
    case cv::EVENT_RBUTTONDOWN:
    {
        mgr->_dragging = true;
        mgr->_flags = flags;
        mgr->_event = ev;
        break;
    }
    case cv::EVENT_MOUSEMOVE:
    {
        if(mgr->_dragging == true)
        {
            mgr->_points.emplace_back(x, y);
        }
        break;
    }
    case cv::EVENT_MBUTTONUP:
    case cv::EVENT_LBUTTONUP:
    case cv::EVENT_RBUTTONUP:
    {
        if(mgr->_dragging == true)
        {
            std::vector<cv::Point2f> homogeneous_points;
            homogeneous_points.reserve(mgr->_points.size());
            for(const auto& pt : mgr->_points)
            {
                homogeneous_points.emplace_back(pt.x / mgr->_image_size.width, pt.y / mgr->_image_size.height);
            }
            mgr->_parent->sig_on_points(mgr->_window_name, mgr->_flags, homogeneous_points);

            cv::Rect rect = cv::boundingRect(mgr->_points);
            cv::Rect2f homo_rect(rect.x / mgr->_image_size.width, rect.y / mgr->_image_size.height,
                                 rect.width / mgr->_image_size.width, rect.height / mgr->_image_size.height);
            mgr->_parent->sig_on_rect(mgr->_window_name, mgr->_flags, homo_rect);
        }
        break;
    }

    }
}

void WindowCallbackHandler::onCb(int ev, int x, int y, int flags)
{

}

void guiThreadFuncCpu(std::string name, cv::Mat mat, WindowHandler* mgr)
{
    cv::namedWindow(name);
    cv::setMouseCallback(name, &WindowCallbackHandlerCB, mgr);
    cv::imshow(name, mat);
}

void guiThreadFuncGpu(std::string name, cv::cuda::GpuMat mat, WindowHandler* mgr)
{

}

void WindowCallbackHandler::imshow(const std::string& name, const cv::Mat& mat)
{

}

void WindowCallbackHandler::imshow(const std::string& name, const cv::cuda::GpuMat& mat, cv::cuda::Stream& stream)
{

}

void WindowCallbackHandler::imshow(const std::string& name, const SyncedMemory& mat, cv::cuda::Stream& stream)
{
    std::shared_ptr<WindowHandler>& handler = _handlers[name];
    if(!handler)
    {
        handler.reset(new WindowHandler());
        handler->_parent = this;
        handler->_image_size = mat.GetSize();
    }

    auto state = mat.GetSyncState();
    const cv::Mat& h_mat = mat.GetMat(stream);

    size_t gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);
    std::function<void(void)> func = std::bind(&WindowCallbackHandler::guiThreadFuncCpu, name, h_mat, handler.get());

    if(state == mat.DEVICE_UPDATED)
    {
        cuda::enqueue_callback_async(func, gui_thread_id, stream);
    }else
    {
        mo::ThreadSpecificQueue::Push(func, gui_thread_id, this);
    }
}
*/

