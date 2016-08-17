#include "parameters/UI/InterThread.hpp"
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/log/trivial.hpp>
#include <MetaObject/Logging/Log.hpp>
#include "EagleLib/Logging.h"

int main()
{
    EagleLib::SetupLogging();
    LOG(info) << "Main thread started";
    boost::thread thread(boost::bind<void>([]()->void
    {
        while (!boost::this_thread::interruption_requested())
        {
            LOG(info) << "Launching callback from work thread";
            Parameters::UI::UiCallbackService::Instance()->post(
                boost::bind<void>([]()->void
            {
                LOG(info) << "Running callback from main thread";
            }));
        }
    }));

    boost::posix_time::ptime start = boost::posix_time::microsec_clock::universal_time();
    while (boost::posix_time::time_duration(boost::posix_time::microsec_clock::universal_time() - start).total_seconds() < 60)
    {
        Parameters::UI::UiCallbackService::Instance()->run();
    }
    thread.interrupt();
    thread.join();
    EagleLib::ShutdownLogging();
    return 0;
}
