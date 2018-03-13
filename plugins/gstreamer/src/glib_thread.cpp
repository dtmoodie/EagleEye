#include "glib_thread.h"
#include "MetaObject/core/SystemTable.hpp"
#include <MetaObject/thread/ThreadRegistry.hpp>
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/thread/boost_thread.hpp>
#include <MetaObject/thread/InterThread.hpp>

struct GlibEventHandler;

// This is the idle callback which calls MetaObject's GUI events on Unix from
// the glib thread. This is an ugly hack for now :/
gboolean glibIdle(gpointer /*user_data*/)
{
    mo::ThreadSpecificQueue::run();
    return TRUE;
}

struct GlibEventHandler
{
    GlibEventHandler():
        notifier(mo::ThreadSpecificQueue::registerNotifier(std::bind(&GlibEventHandler::onEvent, this)))
    {

    }

    // handle events from the glib thread, this should be called by a glib callback
    void handleEvent()
    {
        mo::ThreadSpecificQueue::run();
    }

    // Called from the emitting thread
    void onEvent()
    {
        // Emit a glib signal from the emitting thread that will call handleEvent on the glib
        // event loop
    }
    mo::ThreadSpecificQueue::ScopedNotifier notifier;
};



glib_thread::glib_thread()
{
    _main_loop = nullptr;
}

glib_thread::~glib_thread()
{
    stop_thread();
}

void glib_thread::loop()
{
    if (!g_main_loop_is_running(_main_loop))
    {
#ifndef _MSC_VER
        mo::ThreadRegistry::instance()->registerThread(mo::ThreadRegistry::GUI);
        g_idle_add(&glibIdle, nullptr);

        // Ideally we can just use a notifier instead of an idle func
        // TODO make me work...
        // GlibEventHandler handler;
#endif
        context = mo::Context::create("glib thread");
        MO_LOG(info) << "glib event loop starting";
        g_main_loop_run(_main_loop);

    }
    MO_LOG(info) << "glib event loop ending";
}


// TODO initialize at plugin load
glib_thread* glib_thread::instance()
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    auto instance = table->getSingleton<glib_thread>();
    if (!instance)
    {
        MO_LOG(info) << "Creating new instance of glib_thread";
        instance = std::make_shared<glib_thread>();
        table->setSingleton(instance);
    }
    return instance.get();
}

GMainLoop* glib_thread::get_main_loop()
{
    if (!_main_loop)
    {
        MO_LOG(info) << "Creating new glib event loop";
        _main_loop = g_main_loop_new(NULL, 0);
    }
    return _main_loop;
}

void glib_thread::stop_thread()
{
    g_main_loop_quit(_main_loop);
    _thread.interrupt();
    _thread.join();
}

void glib_thread::start_thread()
{
    if (!_main_loop)
    {
        MO_LOG(info) << "Creating new glib event loop";
        _main_loop = g_main_loop_new(NULL, 0);
    }
    if (g_main_loop_is_running(_main_loop))
    {
        MO_LOG(debug) << "glib main loop already running";
        return;
    }
    _thread = boost::thread(boost::bind(&glib_thread::loop, this));
}

size_t glib_thread::get_thread_id()
{
    return mo::getThreadId(_thread);
}
