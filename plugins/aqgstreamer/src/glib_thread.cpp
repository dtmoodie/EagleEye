#include "glib_thread.h"
#include "MetaObject/core/SystemTable.hpp"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/thread/ThreadInfo.hpp>
#include <MetaObject/thread/ThreadRegistry.hpp>

namespace aqgstreamer
{
    struct GlibEventHandler;

    // This is the idle callback which calls MetaObject's GUI events on Unix from
    // the glib thread. This is an ugly hack for now :/
    gboolean glibIdle(gpointer user_data)
    {
        GLibThread* obj = static_cast<GLibThread*>(user_data);
        obj->yield();
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

    GLibThread::~GLibThread()
    {
        MO_LOG(info, "Cleaning up glib thread");
        stopThread();
    }

    void GLibThread::loop()
    {
        if (!g_main_loop_is_running(m_main_loop))
        {
#ifndef _MSC_VER
            mo::ThreadRegistry::instance()->registerThread(mo::ThreadRegistry::GUI);
            g_idle_add(&glibIdle, this);

// Ideally we can just use a notifier instead of an idle func
// TODO make me work...
// GlibEventHandler handler;
#endif
            {
                mo::Mutex_t::Lock_t lock(m_mtx);

                m_stream = mo::IAsyncStream::create("glib thread");
            }
            m_cv.notify_all();

            MO_LOG(info, "glib event loop starting");
            g_main_loop_run(m_main_loop);
        }
        MO_LOG(info, "glib event loop ending");
    }

    mo::IAsyncStreamPtr_t GLibThread::getStream() const
    {
        mo::Mutex_t::Lock_t lock(m_mtx);
        while (!m_stream)
        {
            m_cv.wait(lock);
        }
        return m_stream;
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

    void GLibThread::stopThread()
    {
        MO_LOG(info, "Stopping glib thread");
        g_main_loop_quit(m_main_loop);
        m_thread.interrupt();
        m_thread.join();
    }

    void GLibThread::yield() { m_stream->synchronize(); }

    void GLibThread::startThread() {}

    size_t GLibThread::getThreadId() { return mo::getThreadId(m_thread); }
} // namespace aqgstreamer
