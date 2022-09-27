#include "glib_thread.h"
#include "MetaObject/core/SystemTable.hpp"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/ThreadInfo.hpp>
#include <MetaObject/thread/ThreadRegistry.hpp>

#include <opencv2/core.hpp>

namespace aqgstreamer
{
    struct GlibEventHandler;

    // This is the idle callback which calls MetaObject's GUI events on Unix from
    // the glib thread. This is an ugly hack for now :/
    gboolean glibIdle(gpointer user_data)
    {
        mo::IAsyncStream* stream = static_cast<mo::IAsyncStream*>(user_data);
        stream->synchronize();
        return TRUE;
    }

    struct GlibEventHandler
    {
        GlibEventHandler() {}

        // handle events from the glib thread, this should be called by a glib callback
        void handleEvent() {}

        // Called from the emitting thread
        void onEvent()
        {
            // Emit a glib signal from the emitting thread that will call handleEvent on the glib
            // event loop
        }
    };

    GLibThread::GLibThread()
    {

        if (!m_main_loop)
        {
            MO_LOG(info, "Creating new glib event loop");
            m_main_loop = g_main_loop_new(NULL, 0);
        }
        if (g_main_loop_is_running(m_main_loop))
        {
            MO_LOG(debug, "glib main loop already running");
            return;
        }
        m_thread = boost::thread(boost::bind(&GLibThread::loop, this));
    }

    GLibThread::~GLibThread() { stopThread(); }

    void GLibThread::loop()
    {
        auto cv_cpu_allocator = SystemTable::instance()->getSingletonOptional<cv::MatAllocator>();
        if (!g_main_loop_is_running(m_main_loop))
        {
            mo::IAsyncStreamPtr_t stream;
            {
                mo::Mutex_t::Lock_t lock(m_mtx);
                mo::initThread();
                stream = mo::IAsyncStream::create("glib thread");
                mo::IAsyncStream::setCurrent(stream);
                mo::setThisThreadName("glib thread");
            }
            m_stream = stream;
            m_cv.notify_all();
#ifndef _MSC_VER
            mo::ThreadRegistry::instance()->setGUIStream(stream);
            g_idle_add(&glibIdle, stream.get());

// Ideally we can just use a notifier instead of an idle func
// TODO make me work...
// GlibEventHandler handler;
#endif
            g_main_loop_run(m_main_loop);
        }
        std::cout << "GlibThread loop exit" << std::endl;
    }

    // TODO initialize at plugin load
    std::shared_ptr<GLibThread> GLibThread::instance()
    {
        auto table = PerModuleInterface::GetInstance()->GetSystemTable();
        return instance(table);
    }

    std::shared_ptr<GLibThread> GLibThread::instance(SystemTable* table) { return table->getSingleton<GLibThread>(); }

    GMainLoop* GLibThread::getMainLoop()
    {
        if (!m_main_loop)
        {
            MO_LOG(info, "Creating new glib event loop");
            m_main_loop = g_main_loop_new(NULL, 0);
        }
        return m_main_loop;
    }

    mo::IAsyncStreamPtr_t GLibThread::getStream() const
    {
        mo::Mutex_t::Lock_t lock(m_mtx);
        mo::IAsyncStreamPtr_t output = m_stream.lock();
        if (!output)
        {
            m_cv.wait(lock);
            output = m_stream.lock();
        }
        return output;
    }

    void GLibThread::stopThread()
    {
        if (g_main_loop_is_running(m_main_loop))
        {
            g_main_loop_quit(m_main_loop);
        }
        m_thread.interrupt();
        m_thread.join();
    }

    void GLibThread::startThread() {}

    size_t GLibThread::getThreadId() { return mo::getThreadId(m_thread); }
} // namespace aqgstreamer
