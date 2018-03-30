#pragma once

#include "aqgstreamer_export.hpp"
#include <MetaObject/core/Context.hpp>
#include <boost/thread.hpp>
#include <gst/gst.h>

class aqgstreamer_EXPORT glib_thread
{
  protected:
    boost::thread _thread;
    GMainLoop* _main_loop;
    mutable boost::condition_variable cv;
    mutable boost::mutex mtx;
    std::shared_ptr<mo::Context> context;
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

    std::shared_ptr<mo::Context> getContext() const;
};
