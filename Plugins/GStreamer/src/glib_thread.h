#pragma once

// trivial does not need RCC support
#include <gst/gst.h>
#include <boost/thread.hpp>

class PLUGIN_EXPORTS glib_thread
{
protected:
    boost::thread _thread;
    GMainLoop* _main_loop;
    glib_thread();
    ~glib_thread();
    void loop();
public:
    static glib_thread* instance();
    // gobject main event loop
    GMainLoop* get_main_loop();

    void stop_thread();

    void start_thread();
};