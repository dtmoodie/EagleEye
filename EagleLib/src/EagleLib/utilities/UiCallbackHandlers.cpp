#include "UiCallbackHandlers.h"
#include "opencv2/highgui.hpp"
#include "Events.h"
#include <ObjectInterfacePerModule.h>
#include <SystemTable.hpp>
#include <boost/thread/mutex.hpp>
using namespace EagleLib;


static void on_mouse_click(int event, int x, int y, int flags, void* callback_handler)
{
    auto ptr =  static_cast<std::pair<std::string, WindowCallbackHandler*>*>(callback_handler);
    ptr->second->handle_click(event, x, y, flags, ptr->first);
}

void WindowCallbackHandler::imshow(const std::string& window_name, cv::InputArray img)
{
    cv::imshow(window_name, img);
    auto itr = windows.find(window_name);
    if (itr == windows.end())
    {
        windows[window_name] = this;    
        auto itr = windows.find(window_name);
        cv::setMouseCallback(window_name, on_mouse_click, &(*itr));
    }
}

WindowCallbackHandler* WindowCallbackHandler::instance(size_t stream_id)
{
    static boost::mutex mtx;
    static std::map<size_t, WindowCallbackHandler*> inst;

    auto itr = inst.find(stream_id);
    if (itr == inst.end())
    {
        auto ptr = new WindowCallbackHandler();
        inst[stream_id] = ptr;
        return ptr;
    }
    return itr->second;
}

WindowCallbackHandler::WindowCallbackHandler()
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table)
    {
        auto signalHandler = table->GetSingleton<ISignalHandler>();

        sig_click_right = signalHandler->GetSignalSafe<boost::signals2::signal<void(std::string, cv::Point, int)>>("click_right");
        sig_click_left = signalHandler->GetSignalSafe<boost::signals2::signal<void(std::string, cv::Point, int)>>("click_left");
        sig_click_middle = signalHandler->GetSignalSafe<boost::signals2::signal<void(std::string, cv::Point, int)>>("click_middle");
        sig_move_mouse = signalHandler->GetSignalSafe<boost::signals2::signal<void(std::string, cv::Point, int)>>("move_mouse");

        sig_click = signalHandler->GetSignalSafe<boost::signals2::signal<void(std::string, cv::Point, int)>>("click");
        sig_select_rect = signalHandler->GetSignalSafe<boost::signals2::signal<void(std::string, cv::Rect, int)>>("rect_select");

    }
}
void WindowCallbackHandler::handle_click(int event, int x, int y, int flags, const std::string& win_name)
{
    cv::Point pt(x, y);
    switch (event)
    {
    case cv::EVENT_MOUSEMOVE:
    {
        (*sig_move_mouse)(win_name, pt, flags);
        (*sig_click)(win_name, pt, flags);
        break;
    }
    case cv::EVENT_LBUTTONDOWN:
    {
        drag_start[win_name] = pt;
        (*sig_click_left)(win_name, pt, flags);
        (*sig_click)(win_name, pt, flags);
        break;
    }
    case cv::EVENT_RBUTTONDOWN:
    {
        drag_start[win_name] = cv::Point(x, y);
        (*sig_click_right)(win_name, pt, flags);
        (*sig_click)(win_name, pt, flags);
        break;
    }
    case cv::EVENT_MBUTTONDOWN:
    {
        drag_start[win_name] = cv::Point(x, y);
        (*sig_click_middle)(win_name, pt, flags);
        (*sig_click)(win_name, pt, flags);
        break;
    }
    case cv::EVENT_LBUTTONUP:
    {
        cv::Rect rect(drag_start[win_name], cv::Point(x, y));
        if(rect.width != 0 && rect.height != 0)
            (*sig_select_rect)(win_name, rect, flags);
        break;
    }
    case cv::EVENT_RBUTTONUP:
    {
        cv::Rect rect(drag_start[win_name], cv::Point(x, y));
        if (rect.width != 0 && rect.height != 0)
            (*sig_select_rect)(win_name, rect, flags);
        break;
    }
    case cv::EVENT_MBUTTONUP:
    {
        cv::Rect rect(drag_start[win_name], cv::Point(x, y));
        if (rect.width != 0 && rect.height != 0)
            (*sig_select_rect)(win_name, rect, flags);
        break;
    }
    case cv::EVENT_LBUTTONDBLCLK:
    {
        flags += 64;
        (*sig_click_left)(win_name, pt, flags);
        break;
    }
    case cv::EVENT_RBUTTONDBLCLK:
    {
        flags += 64;
        (*sig_click_right)(win_name, pt, flags);
        break;
    }
    case cv::EVENT_MBUTTONDBLCLK:
    {
        flags += 64;
        (*sig_click_middle)(win_name, pt, flags);
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



