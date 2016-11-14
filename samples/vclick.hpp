#ifdef HAVE_WT
#pragma once
#include "WebSink.hpp"
#include "TParameterResource.hpp"
#include "BoundingBox.hpp"

#include <Wt/WApplication>
#include <EagleLib/Nodes/Node.h>
#include <MetaObject/Parameters/TypedParameterPtr.hpp>
#include <sstream>
#include <mutex>

namespace vclick
{
    class WebUi: public Wt::WApplication
    {
    public:
        // Blocking call, run on own thread.
        static void StartServer(int argc, char** argv, rcc::shared_ptr<WebSink> sink);
        WebUi(const Wt::WEnvironment& env);
        ~WebUi();
    private:
        void handleMove(int index, float x, float y, float z);
        void handleRotate(int index, float x, float y, float z);
        void handleResize(int index, float x, float y, float z);
        void handleActivate();
        void handleAddbb();
        void handleRebuildModel();
        void handleKeydown(int value);
        
        Wt::WText* render_window;
        Wt::JSignal<int, float, float, float> onMove;
        Wt::JSignal<int, float, float, float> onResize;
        Wt::JSignal<int, float, float, float> onRotate;
        Wt::JSignal<int> onKeydown;
        Wt::JSlot* update;
        
        TParameterResource<cv::Mat>* foregroundStream;
        TParameterResource<EagleLib::SyncedMemory>* backgroundStream;
        TParameterResource<std::vector<BoundingBox>>* boundingBoxStream;
    };
}
#endif
