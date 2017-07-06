#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <MetaObject/thread/InterThread.hpp>
#include <MetaObject/logging/logging.hpp>


int main()
{
    //aq::SetupLogging();
    MO_LOG(info) << "Main thread started";
    size_t main_thread_id = mo::getThisThread();
    boost::thread thread(boost::bind<void>([main_thread_id]()->void
    {
        while (!boost::this_thread::interruption_requested())
        {
            MO_LOG(info) << "Launching callback from work thread";
            mo::ThreadSpecificQueue::push(
                boost::bind<void>([]()->void
            {
                MO_LOG(info) << "Running callback from main thread";
            }), main_thread_id);
        }
    }));

    boost::posix_time::ptime start = boost::posix_time::microsec_clock::universal_time();
    while (boost::posix_time::time_duration(boost::posix_time::microsec_clock::universal_time() - start).total_seconds() < 60)
    {
        mo::ThreadSpecificQueue::run();
    }
    thread.interrupt();
    thread.join();
    //aq::ShutdownLogging();
    return 0;
}
