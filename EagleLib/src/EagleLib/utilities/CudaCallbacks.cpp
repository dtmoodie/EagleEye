#include "EagleLib/utilities/CudaCallbacks.hpp"

#include <MetaObject/Logging/Log.hpp>
#include <MetaObject/Thread/InterThread.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/microsec_time_clock.hpp>
#include <boost/date_time/time_duration.hpp>

using namespace EagleLib::cuda;

void EagleLib::cuda::ICallback::cb_func_async_event_loop(int status, void* user_data)
{
    auto cb = static_cast<ICallbackEventLoop*>(user_data);
    mo::ThreadSpecificQueue::Push([cb]()
    {
        cb->run();
        delete cb;
    }, cb->event_loop_thread_id);
}

void EagleLib::cuda::ICallback::cb_func_async(int status, void* user_data)
{
#ifdef _MSC_VER
    pplx::create_task([user_data]()->void
    {
        auto cb = static_cast<ICallback*>(user_data);
        auto start = clock();
        cb->run();
        LOG(trace) << "Callback execution time: " << clock() - start << " ms";
        delete cb;
    });
#else
    auto cb = static_cast<ICallback*>(user_data);
    auto start = clock();
    cb->run();
    LOG(trace) << "Callback execution time: " << clock() - start << " ms";
    delete cb;
#endif
}
void EagleLib::cuda::ICallback::cb_func(int status, void* user_data)
{
    auto cb = static_cast<ICallback*>(user_data);
    cb->run();
    delete cb;
}
ICallback::~ICallback()
{
}


scoped_stream_timer::scoped_stream_timer(cv::cuda::Stream& stream, const std::string& scope_name) : _stream(stream), _scope_name(scope_name)
{
    start_time = new boost::posix_time::ptime();
    EagleLib::cuda::enqueue_callback<boost::posix_time::ptime*, void>(start_time, 
        [](boost::posix_time::ptime* start)
    {
        *start = boost::posix_time::microsec_clock::universal_time(); 
    }, stream);
}
struct scoped_event_timer_data
{
    std::string scope_name;
    boost::posix_time::ptime* start_time;
};
scoped_stream_timer::~scoped_stream_timer()
{
    scoped_event_timer_data data;
    data.scope_name = _scope_name;
    data.start_time = start_time;
    EagleLib::cuda::enqueue_callback<scoped_event_timer_data, void>(data, [](scoped_event_timer_data data)
    {
        LOG(trace) << "[" << data.scope_name << "] executed in " <<
            boost::posix_time::time_duration(boost::posix_time::microsec_clock::universal_time() - *data.start_time).total_microseconds() << " us";
        delete data.start_time;
    }, _stream);
}

EagleLib::pool::ObjectPool<cv::cuda::Event> EagleLib::cuda::scoped_event_stream_timer::eventPool;

struct scoped_event_data
{
    EagleLib::pool::Ptr<cv::cuda::Event> startEvent;
    EagleLib::pool::Ptr<cv::cuda::Event> endEvent;
    std::string _scope_name;
};

scoped_event_stream_timer::scoped_event_stream_timer(cv::cuda::Stream& stream, const std::string& scope_name) :
_stream(stream), _scope_name(scope_name)
{
    startEvent = eventPool.get_object();
    endEvent = eventPool.get_object();
    startEvent->record(stream);
}
scoped_event_stream_timer::~scoped_event_stream_timer()
{
    endEvent->record(_stream);
    scoped_event_data data;
    data.startEvent = startEvent;
    data.endEvent = endEvent;
    data._scope_name = _scope_name;
    EagleLib::cuda::enqueue_callback_async<scoped_event_data, void>(data, 
        [](scoped_event_data data)->void
    {
        LOG(trace) << "[" << data._scope_name << "] executed in " << cv::cuda::Event::elapsedTime((*data.startEvent.get()), (*data.endEvent.get())) << " ms";
    }, _stream);
}

LambdaCallback<void>::LambdaCallback(const std::function<void()>& f): 
    func(f) 
{

}
LambdaCallback<void>::~LambdaCallback()
{

}
void LambdaCallback<void>::run()
{
    func();
    promise.set_value();
}
LambdaCallbackEvent<void>::LambdaCallbackEvent(const std::function<void()>& f) :
    func(f)
{
    this->event_loop_thread_id = 0;
}
LambdaCallbackEvent<void>::~LambdaCallbackEvent()
{

}
void LambdaCallbackEvent<void>::run()
{
    func();
    promise.set_value();
}
