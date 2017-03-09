#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <MetaObject/Thread/InterThread.hpp>
#include <MetaObject/Logging/Log.hpp>


int main()
{
    //aq::SetupLogging();
    LOG(info) << "Main thread started";
    size_t main_thread_id = mo::GetThisThread();
    boost::thread thread(boost::bind<void>([main_thread_id]()->void
    {
        while (!boost::this_thread::interruption_requested())
        {
            LOG(info) << "Launching callback from work thread";
            mo::ThreadSpecificQueue::Push(
                boost::bind<void>([]()->void
            {
                LOG(info) << "Running callback from main thread";
            }), main_thread_id);
        }
    }));

    boost::posix_time::ptime start = boost::posix_time::microsec_clock::universal_time();
    while (boost::posix_time::time_duration(boost::posix_time::microsec_clock::universal_time() - start).total_seconds() < 60)
    {
        mo::ThreadSpecificQueue::Run();
    }
    thread.interrupt();
    thread.join();
    //aq::ShutdownLogging();
    return 0;
}
