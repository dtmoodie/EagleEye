#pragma once
#include "Aquila/framegrabbers/IFrameGrabber.hpp"

namespace aq
{
    namespace nodes
    {
        class FrameGrabberDirectory: public IFrameGrabber
        {
        public:
            static int canLoadPath(const std::string& doc);

            MO_DERIVE(FrameGrabberDirectory, IFrameGrabber)
                STATUS(int, frame_index, 0)
                MO_SIGNAL(void, eos)
            MO_END

            virtual bool loadData(const std::string file_path) override;
            virtual void restart() override;

        protected:
            virtual bool processImpl() override;
        private:
            std::string                     loaded_file;
            std::vector<std::string>        files_on_disk;
            rcc::shared_ptr<IGrabber>       fg; // internal frame grabber used for loading the actual files
        };
    }
}
