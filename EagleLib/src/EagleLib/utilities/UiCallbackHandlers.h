#pragma once

#include "EagleLib/Defs.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <boost/signals2.hpp>
#include <set>
#include <memory>
namespace EagleLib
{
    class EAGLE_EXPORTS WindowCallbackHandler
    {
    public:
        WindowCallbackHandler();
        static WindowCallbackHandler* instance(size_t stream_id = 0);

        void handle_click(int event, int x, int y, int flags, const std::string& win_name);
        void imshow(const std::string& window_name, cv::InputArray img);


        boost::signals2::signal<void(std::string, cv::Point, int)>* sig_click_right;
        boost::signals2::signal<void(std::string, cv::Point, int)>* sig_click_left;
        boost::signals2::signal<void(std::string, cv::Point, int)>* sig_click_middle;
        boost::signals2::signal<void(std::string, cv::Point, int)>* sig_move_mouse;

        boost::signals2::signal<void(std::string, cv::Point, int)>* sig_click;
        boost::signals2::signal<void(std::string, cv::Rect, int) >* sig_select_rect;
    private:
        std::map<std::string,cv::Point> drag_start;
        
        std::map<std::string, WindowCallbackHandler*> windows;
    };
} // namespace EagleLib