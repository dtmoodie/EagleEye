#include "HeartBeatBuffer.h"
#include <RuntimeObjectSystem/shared_ptr.hpp>

using namespace aq;
using namespace aq::nodes;
#define CATCH_MACRO                                                         \
    catch (boost::thread_resource_error& err)                               \
{                                                                           \
    MO_LOG(error) << err.what();                                               \
}                                                                           \
    catch (boost::thread_interrupted& err)                                  \
{                                                                           \
    MO_LOG(error) << "Thread interrupted";                                     \
    /* Needs to pass this back up to the chain to the processing thread.*/  \
    /* That way it knowns it needs to exit this thread */                   \
    throw err;                                                              \
}                                                                           \
    catch (boost::thread_exception& err)                                    \
{                                                                           \
    MO_LOG(error) << err.what();                                               \
}                                                                           \
    catch (cv::Exception &err)                                              \
{                                                                           \
    MO_LOG(error) << err.what();                                               \
}                                                                           \
    catch (boost::exception &err)                                           \
{                                                                           \
    MO_LOG(error) << "Boost error";                                            \
}                                                                           \
    catch (std::exception &err)                                             \
{                                                                           \
    MO_LOG(error) << err.what();                                               \
}                                                                           \
    catch (...)                                                             \
{                                                                           \
    MO_LOG(error) << "Unknown exception";                                      \
}

/*void HeartBeatBuffer::nodeInit(bool firstInit)
{
    if (firstInit)
    {
        updateParameter<int>("Buffer size", 30);
        updateParameter<double>("Heartbeat frequency", 1.0)->SetTooltip("Seconds between heartbeat images");
        updateParameter<bool>("Active", false);
        RegisterParameterCallback(2, boost::bind(&HeartBeatBuffer::onActivation, this));
        lastTime = clock();
        activated = false;
        
    }
    image_buffer.set_capacity(*getParameter<int>(0)->Data());
}
void HeartBeatBuffer::onActivation()
{
    activated = true;
}*/
/*TS<SyncedMemory> HeartBeatBuffer::process(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    if (boost::this_thread::interruption_requested())
        return input;

    if (input.empty() && SkipEmpty())
    {
        MO_LOG(trace) << " Skipped due to empty input";
    }
    else
    {
        if (_parameters[0]->changed)
        {
            image_buffer.set_capacity(*getParameter<int>(0)->Data());
            _parameters[0]->changed = false;
        }
        try
        {
            if (activated)
            {
                for (auto itr : image_buffer)
                {
                    for (auto childItr : _children)
                    {
                        itr = childItr->doProcess(itr, stream);
                    }
                } 
                activated = false;
                image_buffer.clear();
            }
            auto currentTime = clock();
            if ((double(currentTime) - double(lastTime)) / 1000 > *getParameter<double>(1)->Data() || 
                *getParameter<bool>(2)->Data())
            {
                lastTime = currentTime;
                // Send heartbeat
                for (auto itr : children)
                {
                    input = itr->process(input, stream);
                }
            }
            else
            {
                image_buffer.push_back(input);
            }
        }CATCH_MACRO
    }
    return input;
}*/



//NODE_DEFAULT_CONSTRUCTOR_IMPL(HeartBeatBuffer, Utility)
