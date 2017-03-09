#pragma once

#include "cv_capture.h"
#include "Aquila/ICoordinateManager.h"
#include "RuntimeLinkLibrary.h"


namespace aq
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
