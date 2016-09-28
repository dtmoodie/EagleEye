#include "EagleLib/utilities/UiCallbackHandlers.h"
#include "Remotery.h"
#include "ObjectInterfacePerModule.h"
#include "EagleLib/rcc/external_includes/cv_core.hpp"
#include "EagleLib/rcc/external_includes/cv_highgui.hpp"
#include "EagleLib/rcc/SystemTable.hpp"

#include <MetaObject/Thread/ThreadRegistry.hpp>
#include <MetaObject/Thread/InterThread.hpp>

#include <boost/thread/mutex.hpp>

using namespace EagleLib;


static void on_mouse_click(int event, int x, int y, int flags, void* callback_handler)
{
    auto ptr =  static_cast<std::pair<std::string, WindowCallbackHandler*>*>(callback_handler);
    ptr->second->handle_click(event, x, y, flags, ptr->first);
}

void WindowCallbackHandler::imshow(const std::string& window_name, cv::Mat img, int flags)
{
    auto gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);
    if (mo::GetThisThread() != gui_thread_id)
    {
        mo::ThreadSpecificQueue::Push(std::bind(&WindowCallbackHandler::imshow, this, window_name, img, flags), gui_thread_id);
        return;
    }
    // The below code should only execute if this is on the GUI thread, thus it doesn't need locking
    auto itr = windows.find(window_name);
    if (itr == windows.end())
    {
        cv::namedWindow(window_name, flags);
        windows[window_name] = this;
        auto itr = windows.find(window_name);
        cv::setMouseCallback(window_name, on_mouse_click, &(*itr));
    }
    if (!dragging[window_name])
        cv::imshow(window_name, img);
}
void WindowCallbackHandler::imshowd(const std::string& window_name, cv::cuda::GpuMat img, int flags)
{
    auto gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);
    if (mo::GetThisThread() != gui_thread_id)
    {
        mo::ThreadSpecificQueue::Push(std::bind(&WindowCallbackHandler::imshowd, this, window_name, img, flags), gui_thread_id);
        return;
    }
    auto itr = windows.find(window_name);
    if (itr == windows.end())
    {
        cv::namedWindow(window_name, flags);
        windows[window_name] = this;
        auto itr = windows.find(window_name);
        cv::setMouseCallback(window_name, on_mouse_click, &(*itr));
    }
    if (!dragging[window_name])
        cv::imshow(window_name, img);
}

void WindowCallbackHandler::Init(bool firstInit)
{
    if(firstInit)
    {
    
    }else
    {
        for(auto& itr : windows)
        {
            itr.second = this;
            cv::setMouseCallback(itr.first, on_mouse_click, &itr);
        }
    }
}

WindowCallbackHandler::WindowCallbackHandler()
{
}

MO_REGISTER_OBJECT(WindowCallbackHandler);


void WindowCallbackHandler::handle_click(int event, int x, int y, int flags, const std::string& win_name_)
{
    cv::Point pt(x, y);
    std::string win_name = win_name_;
    switch (event)
    {
    case cv::EVENT_MOUSEMOVE:
    {
        if(flags & cv::EVENT_FLAG_LBUTTON)
            dragged_points[win_name].push_back(pt);
        sig_move_mouse(win_name, pt, flags);
        sig_click(win_name, pt, flags);
        break;
    }
    case cv::EVENT_LBUTTONDOWN:
    {
        dragging[win_name] = true;
        drag_start[win_name] = pt;
        dragged_points[win_name].clear();
        dragged_points[win_name].push_back(pt);
        sig_click_left(win_name, pt, flags);
        sig_click(win_name, pt, flags);
        break;
    }
    case cv::EVENT_RBUTTONDOWN:
    {
        dragging[win_name] = true;
        drag_start[win_name] = cv::Point(x, y);
        sig_click_right(win_name, pt, flags);
        sig_click(win_name, pt, flags);
        break;
    }
    case cv::EVENT_MBUTTONDOWN:
    {
        dragging[win_name] = true;
        drag_start[win_name] = cv::Point(x, y);
        sig_click_middle(win_name, pt, flags);
        sig_click(win_name, pt, flags);
        break;
    }
    case cv::EVENT_LBUTTONUP:
    {
        dragging[win_name] = false;
        cv::Rect rect(drag_start[win_name], cv::Point(x, y));
        if(rect.width != 0 && rect.height != 0)
            sig_select_rect(win_name, rect, flags);
        sig_select_points(win_name, dragged_points[win_name], flags);
        dragged_points[win_name].clear();
        break;
    }
    case cv::EVENT_RBUTTONUP:
    {
        dragging[win_name] = false;
        cv::Rect rect(drag_start[win_name], cv::Point(x, y));
        if (rect.width != 0 && rect.height != 0)
            sig_select_rect(win_name, rect, flags);
        break;
    }
    case cv::EVENT_MBUTTONUP:
    {
        dragging[win_name] = false;
        cv::Rect rect(drag_start[win_name], cv::Point(x, y));
        if (rect.width != 0 && rect.height != 0)
            sig_select_rect(win_name, rect, flags);
        break;
    }
    case cv::EVENT_LBUTTONDBLCLK:
    {
        flags += 64;
        sig_click_left(win_name, pt, flags);
        break;
    }
    case cv::EVENT_RBUTTONDBLCLK:
    {
        flags += 64;
        sig_click_right(win_name, pt, flags);
        break;
    }
    case cv::EVENT_MBUTTONDBLCLK:
    {
        flags += 64;
        sig_click_middle(win_name, pt, flags);
        break;
    }
    case cv::EVENT_MOUSEWHEEL:
    {

        break;
    }
    case cv::EVENT_MOUSEHWHEEL:
    {

        break;
    }
    }
}



