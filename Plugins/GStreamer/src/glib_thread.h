#pragma once

#include "GStreamerExport.hpp"
#include <gst/gst.h>
#include <boost/thread.hpp>

class GStreamer_EXPORT glib_thread
{
protected:
    boost::thread _thread;
    GMainLoop* _main_loop;
    
    void loop();
    
public:
    glib_thread();
    ~glib_thread();
    static glib_thread* instance();
    // gobject main event loop
    GMainLoop* get_main_loop();

    void stop_thread();

    void start_thread();

    size_t get_thread_id();
};
