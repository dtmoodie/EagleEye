#ifndef AQGSTREAMER_GLIB_THREAD_HPP
#define AQGSTREAMER_GLIB_THREAD_HPP

#include "aqgstreamer/aqgstreamer_export.hpp"

#include <MetaObject/thread/ConditionVariable.hpp>
#include <MetaObject/thread/Mutex.hpp>

#include <boost/thread.hpp>
#include <gst/gst.h>

namespace aqgstreamer
{
    class aqgstreamer_EXPORT GLibThread : public std::enable_shared_from_this<GLibThread>
    {

        boost::thread m_thread;
        GMainLoop* m_main_loop = nullptr;
        mutable mo::ConditionVariable m_cv;
        mutable mo::Mutex_t m_mtx;
        mo::IAsyncStreamPtr_t m_stream;

        void loop();

      public:
        GLibThread();
        ~GLibThread();
        static std::shared_ptr<GLibThread> instance();
        static std::shared_ptr<GLibThread> instance(SystemTable* table);
        // gobject main event loop
        GMainLoop* getMainLoop();

        void stopThread();

        void startThread();

        size_t getThreadId();

        void yield();

        mo::IAsyncStreamPtr_t getStream() const;
    };
} // namespace aqgstreamer

#endif // AQGSTREAMER_GLIB_THREAD_HPP
