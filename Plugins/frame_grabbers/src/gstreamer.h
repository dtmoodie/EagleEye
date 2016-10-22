#pragma once
#ifdef HAVE_GSTREAMER
#include "cv_capture.h"
#include "EagleLib/ICoordinateManager.h"
#include "RuntimeLinkLibrary.h"

RUNTIME_COMPILER_LINKLIBRARY("gstapp-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gstaudio-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gstbase-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gstcontroller-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gstnet-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gstpbutils-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gstreamer-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gstriff-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gstrtp-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gstrtsp-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gstsdp-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gsttag-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gstvideo-1.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("gobject-2.0.lib")
RUNTIME_COMPILER_LINKLIBRARY("glib-2.0.lib")

namespace EagleLib
{
    namespace Nodes
    {
        class PLUGIN_EXPORTS frame_grabber_gstreamer: public frame_grabber_cv
        {
        public:
            MO_DERIVE(frame_grabber_gstreamer, frame_grabber_cv)
                PARAM(bool, loop, true);
                MO_SIGNAL(void, eof);
            MO_END;

            frame_grabber_gstreamer();
            ~frame_grabber_gstreamer();
            virtual bool LoadFile(const std::string& file_path);
            virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
            TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream);
            static int CanLoadDocument(const std::string& document);
            static std::vector<std::string> ListLoadableDocuments();
        protected:
            rcc::shared_ptr<ICoordinateManager>     coordinate_manager;
            std::string                             loaded_file;
            TS<SyncedMemory>                        current_frame;
        };
    }
}
#endif // HAVE_GSTREAMER