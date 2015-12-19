#pragma once


#include <opencv2/core/types.hpp>
#include <boost/signals2.hpp>
namespace EagleLib
{
    class WindowCallbackHandler
    {
    public:
        WindowCallbackHandler();
        static WindowCallbackHandler* instance();
        void handle_click(int event, int x, int y, int flags, void* callback_handler);
        boost::signals2::signal<void(std::string, cv::Point)>* sig_click_right;
        boost::signals2::signal<void(std::string, cv::Point)>* sig_click_left;
        boost::signals2::signal<void(std::string, cv::Point)>* sig_click_middle;

        boost::signals2::signal<void(std::string, cv::Point, int)>* sig_click;
        boost::signals2::signal<void(std::string, cv::Rect, int) >* sig_select_rect;
    private:
        cv::Point drag_start;
        
        
    };



}