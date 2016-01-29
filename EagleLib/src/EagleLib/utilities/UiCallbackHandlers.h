#pragma once

#include "EagleLib/Defs.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <set>
#include <memory>
#include "EagleLib/Signals.h"
#include "IObject.h"
#include "EagleLib/rcc/shared_ptr.hpp"
namespace EagleLib
{
    class WindowCallbackHandlerManager;
    // Single instance per stream
    class EAGLE_EXPORTS WindowCallbackHandler: public TInterface<IID_IOBJECT,IObject>
    {
        friend class WindowCallbackHandlerManager;
        void set_stream(size_t stream);
    public:
        enum 
        {
            PAUSE_DRAG = 1 << 31
        };

        WindowCallbackHandler();

        void Init(bool firstInit);
        //static WindowCallbackHandler* instance(size_t stream_id = 0);
        
        void handle_click(int event, int x, int y, int flags, const std::string& win_name);
        void imshow(const std::string& window_name, cv::InputArray img, int flags = 0);


        Signals::signal<void(std::string, cv::Point, int)>* sig_click_right;
        Signals::signal<void(std::string, cv::Point, int)>* sig_click_left;
        Signals::signal<void(std::string, cv::Point, int)>* sig_click_middle;
        Signals::signal<void(std::string, cv::Point, int)>* sig_move_mouse;

        

        Signals::signal<void(std::string, cv::Point, int)>* sig_click;
        Signals::signal<void(std::string, cv::Rect, int)>* sig_select_rect;
        Signals::signal<void(std::string, std::vector<cv::Point>, int)>* sig_select_points;
    private:
        std::map<std::string, bool> dragging;
        std::map<std::string, cv::Point> drag_start;
        std::map<std::string, std::vector<cv::Point>> dragged_points;
        
        std::map<std::string, WindowCallbackHandler*> windows;
    };
} // namespace EagleLib