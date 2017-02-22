#include "EagleLib/utilities/UiCallbackHandlers.h"
#include "ObjectInterfacePerModule.h"
#include "EagleLib/rcc/external_includes/cv_core.hpp"
#include "EagleLib/rcc/external_includes/cv_highgui.hpp"
#include "EagleLib/rcc/SystemTable.hpp"

#include <MetaObject/Thread/ThreadRegistry.hpp>
#include <MetaObject/Thread/InterThread.hpp>

#include <boost/thread/mutex.hpp>

using namespace EagleLib;
WindowCallbackHandler::EventLoop::EventLoop()
{
    //size_t gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);

    //mo::ThreadSpecificQueue::Push(, gui_thread_id, this)
}

WindowCallbackHandler::EventLoop::~EventLoop()
{

}

WindowCallbackHandler::EventLoop* WindowCallbackHandler::EventLoop::Instance()
{
    static EventLoop* g_inst = nullptr;
    if(g_inst == nullptr)
        g_inst = new EventLoop();
    return g_inst;
}

void WindowCallbackHandler::EventLoop::run()
{
    int key = cv::waitKey(1);
    if(key != -1)
    {
        std::unique_lock<std::mutex> lock(mtx);
        for(auto& itr : handlers)
        {
            itr->sig_on_key(key);
        }
    }
}

void WindowCallbackHandler::EventLoop::Register(WindowCallbackHandler* handler)
{
    std::unique_lock<std::mutex> lock(mtx);
    handlers.push_back(handler);
}

void WindowCallbackHandler::on_mouse_click(int event, int x, int y, int flags, void* callback_handler)
{
    auto ptr =  static_cast<WindowCallbackHandler::WindowHandler*>(callback_handler);
    ptr->on_mouse(event, x, y, flags);
}

void WindowCallbackHandler::imshow(const std::string& window_name, cv::Mat img, int flags)
{
    auto gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);
    if (mo::GetThisThread() != gui_thread_id)
    {
        mo::ThreadSpecificQueue::Push(std::bind(&WindowCallbackHandler::imshow, this, window_name, img, flags), gui_thread_id);
        return;
    }
    std::shared_ptr<WindowHandler>& handler = windows[window_name];

    if(!handler)
    {
        handler.reset(new WindowHandler());
        cv::namedWindow(window_name, flags);
        cv::setMouseCallback(window_name, on_mouse_click, handler.get());
        handler->parent = this;
        handler->win_name = window_name;
    }
    if (!handler->dragging)
    {
        cv::imshow(window_name, img);
        handler->displayed_image = img;
    }
    int key = cv::waitKey(1);
    if(key != -1)
    {
        sig_on_key(key);
    }
}
void WindowCallbackHandler::imshowd(const std::string& window_name, cv::cuda::GpuMat img, int flags)
{
    auto gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);
    if (mo::GetThisThread() != gui_thread_id)
    {
        mo::ThreadSpecificQueue::Push(std::bind(&WindowCallbackHandler::imshowd, this, window_name, img, flags), gui_thread_id);
        return;
    }
    std::shared_ptr<WindowHandler>& handler = windows[window_name];
    if(!handler)
    {
        handler.reset(new WindowHandler());
        cv::namedWindow(window_name, flags);
        cv::setMouseCallback(window_name, on_mouse_click, handler.get());
        handler->parent = this;
        handler->win_name = window_name;
    }
    if (!handler->dragging)
    {
        cv::imshow(window_name, img);
    }
    int key = cv::waitKey(1);
    if(key != -1)
    {
        sig_on_key(key);
    }
}
void WindowCallbackHandler::imshowb(const std::string& window_name, cv::ogl::Buffer buffer, int flags)
{
    auto gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);
    if (mo::GetThisThread() != gui_thread_id)
    {
        mo::ThreadSpecificQueue::Push(std::bind(&WindowCallbackHandler::imshowb, this, window_name, buffer, flags), gui_thread_id);
        return;
    }
    std::shared_ptr<WindowHandler>& handler = windows[window_name];
    if(!handler)
    {
        handler.reset(new WindowHandler());
        cv::namedWindow(window_name, flags);
        cv::setMouseCallback(window_name, on_mouse_click, handler.get());
        handler->parent = this;
        handler->win_name = window_name;
    }
    if (!handler->dragging)
    {
        cv::imshow(window_name, buffer);
    }
    int key = cv::waitKey(1);
    if(key != -1)
    {
        sig_on_key(key);
    }
}

WindowCallbackHandler::WindowCallbackHandler()
{

}
void WindowCallbackHandler::Init(bool firstInit)
{
    mo::IMetaObject::Init(firstInit);
    EventLoop::Instance()->Register(this);
}

MO_REGISTER_OBJECT(WindowCallbackHandler);


void WindowCallbackHandler::WindowHandler::on_mouse(int event, int x, int y, int flags)
{
    cv::Point pt(x, y);
    double aspect_ratio = cv::getWindowProperty(this->win_name, cv::WND_PROP_ASPECT_RATIO);

    //pt.y *= aspect_ratio;
    switch (event)
    {
    case cv::EVENT_MOUSEMOVE:
    {
        if(flags & cv::EVENT_FLAG_LBUTTON)
            dragged_points.push_back(pt);
        parent->sig_move_mouse(win_name, pt, flags, displayed_image);
        parent->sig_click(win_name, pt, flags, displayed_image);
        break;
    }
    case cv::EVENT_LBUTTONDOWN:
    {
        dragging = true;
        drag_start = pt;
        dragged_points.clear();
        dragged_points.push_back(pt);
        parent->sig_click_left(win_name, pt, flags, displayed_image);
        parent->sig_click(win_name, pt, flags, displayed_image);
        break;
    }
    case cv::EVENT_RBUTTONDOWN:
    {
        dragging = true;
        drag_start = cv::Point(x, y);
        parent->sig_click_right(win_name, pt, flags, displayed_image);
        parent->sig_click(win_name, pt, flags, displayed_image);
        break;
    }
    case cv::EVENT_MBUTTONDOWN:
    {
        dragging = true;
        drag_start = cv::Point(x, y);
        parent->sig_click_middle(win_name, pt, flags, displayed_image);
        parent->sig_click(win_name, pt, flags, displayed_image);
        break;
    }
    case cv::EVENT_LBUTTONUP:
    {
        dragging = false;
        cv::Rect rect(drag_start, cv::Point(x, y));
        if(rect.width != 0 && rect.height != 0)
            parent->sig_select_rect(win_name, rect, flags, displayed_image);
        parent->sig_select_points(win_name, dragged_points, flags, displayed_image);
        dragged_points.clear();
        break;
    }
    case cv::EVENT_RBUTTONUP:
    {
        dragging = false;
        cv::Rect rect(drag_start, cv::Point(x, y));
        if (rect.width != 0 && rect.height != 0)
            parent->sig_select_rect(win_name, rect, flags, displayed_image);
        break;
    }
    case cv::EVENT_MBUTTONUP:
    {
        dragging = false;
        cv::Rect rect(drag_start, cv::Point(x, y));
        if (rect.width != 0 && rect.height != 0)
            parent->sig_select_rect(win_name, rect, flags, displayed_image);
        break;
    }
    case cv::EVENT_LBUTTONDBLCLK:
    {
        flags += 64;
        parent->sig_click_left(win_name, pt, flags, displayed_image);
        break;
    }
    case cv::EVENT_RBUTTONDBLCLK:
    {
        flags += 64;
        parent->sig_click_right(win_name, pt, flags, displayed_image);
        break;
    }
    case cv::EVENT_MBUTTONDBLCLK:
    {
        flags += 64;
        parent->sig_click_middle(win_name, pt, flags, displayed_image);
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

