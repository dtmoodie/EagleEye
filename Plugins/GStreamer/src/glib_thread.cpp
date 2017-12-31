#include "glib_thread.h"
#include "Aquila/rcc/SystemTable.hpp"

#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/thread/boost_thread.hpp>

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
        MO_LOG(info) << "glib event loop starting";
        g_main_loop_run(_main_loop);
    }
    MO_LOG(info) << "glib event loop ending";
}

glib_thread* glib_thread::instance()
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    auto instance = table->getSingleton<glib_thread>();
    if (!instance)
    {
        MO_LOG(info) << "Creating new instance of glib_thread";
        instance = table->setSingleton(new glib_thread());
    }
    return instance;
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