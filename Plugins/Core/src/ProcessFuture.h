#pragma once
#include <src/precompiled.hpp>

#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>

namespace aq
{
    namespace nodes
    {
        class ProcessFuture: public Node
        {
            boost::thread _thread;
            std::condition_variable_any _cv;
            bool _pause;
        public:

            ProcessFuture();
            ~ProcessFuture();
            virtual void nodeInit(bool firstInit);
            virtual void SetGraph(IGraph* stream);
            virtual TS<SyncedMemory> process(TS<SyncedMemory>& input, cv::cuda::Stream& stream);
            virtual TS<SyncedMemory> doProcess(TS<SyncedMemory>& input, cv::cuda::Stream& stream);
            virtual void process_ahead();
            virtual void stop_thread();
            virtual void start_thread();
        };
    }
}