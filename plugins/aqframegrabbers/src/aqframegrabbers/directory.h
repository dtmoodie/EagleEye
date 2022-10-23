#pragma once
#include <aqframegrabbers_export.hpp>

#include "Aquila/framegrabbers/IFrameGrabber.hpp"

#include <boost/fiber/future.hpp>

namespace aqframegrabbers
{

    class aqframegrabbers_EXPORT FrameGrabberDirectory : public aq::nodes::IFrameGrabber
    {
      public:
        static int canLoadPath(const std::string& doc);
        static int loadTimeout();

        MO_DERIVE(FrameGrabberDirectory, aq::nodes::IFrameGrabber)
            STATUS(int, frame_index, 0)
            MO_SIGNAL(void, onEos)
            MO_SIGNAL(void, update)
            PARAM(bool, synchronous, false)
            PARAM(bool, eos, false)
            MO_SLOT(void, nextFrame)
            MO_SLOT(void, prevFrame)

        MO_END;

        void initCustom(bool first_init) override;
        bool loadData(const std::string file_path) override;
        void restart() override;

      protected:
        bool processImpl() override;
        void prefetch(int dist = 1);

      private:
        std::string loaded_file;
        std::vector<std::string> files_on_disk;
        rcc::shared_ptr<aq::nodes::IGrabber> fg; // internal frame grabber used for loading the actual files
        bool step = false;
        bool m_prefetching = false;
        mo::Thread m_prefetch_thread;
        mo::IAsyncStreamPtr_t m_prefetch_stream;
        boost::fibers::promise<bool> m_prefetch_promise;
    };
} // namespace aqframegrabbers
