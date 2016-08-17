#include "HeartBeatBuffer.h"
#include <shared_ptr.hpp>
#include <parameters/ParameteredObjectImpl.hpp>
using namespace EagleLib;
using namespace EagleLib::Nodes;
#define CATCH_MACRO                                                         \
    catch (boost::thread_resource_error& err)                               \
{                                                                           \
    NODE_LOG(error) << err.what();                                          \
}                                                                           \
    catch (boost::thread_interrupted& err)                                  \
{                                                                           \
    NODE_LOG(error) << "Thread interrupted";                                \
    /* Needs to pass this back up to the chain to the processing thread.*/  \
    /* That way it knowns it needs to exit this thread */                   \
    throw err;                                                              \
}                                                                           \
    catch (boost::thread_exception& err)                                    \
{                                                                           \
    NODE_LOG(error) << err.what();                                          \
}                                                                           \
    catch (cv::Exception &err)                                              \
{                                                                           \
    NODE_LOG(error) << err.what();                                          \
}                                                                           \
    catch (boost::exception &err)                                           \
{                                                                           \
    NODE_LOG(error) << "Boost error";                                       \
}                                                                           \
    catch (std::exception &err)                                             \
{                                                                           \
    NODE_LOG(error) << err.what();                                            \
}                                                                           \
    catch (...)                                                             \
{                                                                           \
    NODE_LOG(error) << "Unknown exception";                                 \
}

void HeartBeatBuffer::NodeInit(bool firstInit)
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
}
TS<SyncedMemory> HeartBeatBuffer::process(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    if (boost::this_thread::interruption_requested())
        return input;

    if (input.empty() && SkipEmpty())
    {
        NODE_LOG(trace) << " Skipped due to empty input";
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
                    for (auto childItr : children)
                    {
                        itr = childItr->doProcess(itr, stream);
                    }
                } 
                activated = false;
                image_buffer.clear();
            }
            auto currentTime = clock();
            if ((double(currentTime) - double(lastTime)) / 1000 > *getParameter<double>(1)->Data() || *getParameter<bool>(2)->Data())
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
}



NODE_DEFAULT_CONSTRUCTOR_IMPL(HeartBeatBuffer, Utility)
