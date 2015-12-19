#include "UiCallbackHandlers.h"
#include "opencv2/highgui.hpp"
#include "Events.h"
#include <ObjectInterfacePerModule.h>
#include <SystemTable.hpp>
using namespace EagleLib;


static void on_mouse_click(int event, int x, int y, int, void* callback_handler)
{

}

WindowCallbackHandler* WindowCallbackHandler::instance()
{
    static WindowCallbackHandler* inst = nullptr;
    if (inst == nullptr)
    {
        inst = new WindowCallbackHandler();
    }
    return inst;
}

WindowCallbackHandler::WindowCallbackHandler()
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table)
    {
        auto signalHandler = table->GetSingleton<ISignalHandler>();

        sig_click_right = signalHandler->GetSignalSafe<boost::signals2::signal<void(std::string, cv::Point)>>("click_right");
        sig_click_left = signalHandler->GetSignalSafe<boost::signals2::signal<void(std::string, cv::Point)>>("click_left");
        sig_click_middle = signalHandler->GetSignalSafe<boost::signals2::signal<void(std::string, cv::Point)>>("click_middle");

        sig_click = signalHandler->GetSignalSafe<boost::signals2::signal<void(std::string, cv::Point, int)>>("click");
        sig_select_rect = signalHandler->GetSignalSafe<boost::signals2::signal<void(std::string, cv::Rect, int)>>("rect_select");

    }
}
void WindowCallbackHandler::handle_click(int event, int x, int y, int flags, void* callback_handler)
{
    switch (event)
    {
    case cv::EVENT_MOUSEMOVE:
    {

    }
    case cv::EVENT_LBUTTONDOWN:
    {
        drag_start.x = x;
        drag_start.y = y;
    }
    case cv::EVENT_RBUTTONDOWN:
    {
        drag_start.x = x;
        drag_start.y = y;
    }
    case cv::EVENT_MBUTTONDOWN:
    {
        drag_start.x = x;
        drag_start.y = y;
    }
    case cv::EVENT_LBUTTONUP:
    {

    }
    case cv::EVENT_RBUTTONUP:
    {

    }
    case cv::EVENT_MBUTTONUP:
    {

    }
    case cv::EVENT_LBUTTONDBLCLK:
    {

    }
    case cv::EVENT_RBUTTONDBLCLK:
    {

    }
    case cv::EVENT_MBUTTONDBLCLK:
    {

    }
    case cv::EVENT_MOUSEWHEEL:
    {

    }
    case cv::EVENT_MOUSEHWHEEL:
    {

    }
    }
}



