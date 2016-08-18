#pragma once

#include "EagleLib/Defs.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>
#include <set>
#include <memory>
#include "EagleLib/Signals.h"
#include "IObject.h"
#include <shared_ptr.hpp>
#include <MetaObject/IMetaObject.hpp>
#include <MetaObject/Detail/MetaObjectMacros.hpp>
#include <MetaObject/Signals/detail/SignalMacros.hpp>
namespace EagleLib
{
    class IDataStream;
    class WindowCallbackHandlerManager;
    // Single instance per stream
    class EAGLE_EXPORTS WindowCallbackHandler: public TInterface<IID_IOBJECT,mo::IMetaObject>
    {
        friend class WindowCallbackHandlerManager;
    public:
        enum 
        {
            PAUSE_DRAG = 1 << 31
        };
        static rcc::shared_ptr<WindowCallbackHandler> create();
        WindowCallbackHandler();

        void Init(bool firstInit);
        
        void handle_click(int event, int x, int y, int flags, const std::string& win_name);
        void imshow(const std::string& window_name, cv::Mat img, int flags = 0);
        void imshowd(const std::string& window_name, cv::cuda::GpuMat img, int flags = 0);

        MO_BEGIN(WindowCallbackHandler, mo::IMetaObject);
            MO_SIGNAL(void, click_right, std::string, cv::Point, int);
            MO_SIGNAL(void, click_left, std::string, cv::Point, int);
            MO_SIGNAL(void, click_middle, std::string, cv::Point, int);
            MO_SIGNAL(void, move_mouse, std::string, cv::Point, int);
            MO_SIGNAL(void, click, std::string, cv::Point, int);
            MO_SIGNAL(void, select_rect, std::string, cv::Rect, int);
            MO_SIGNAL(void, select_points, std::string, std::vector<cv::Point>, int);
        MO_END;

    private:
        std::map<std::string, bool> dragging;
        std::map<std::string, cv::Point> drag_start;
        std::map<std::string, std::vector<cv::Point>> dragged_points;
        
        std::map<std::string, WindowCallbackHandler*> windows;
    };
} // namespace EagleLib