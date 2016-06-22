#pragma once
#include "EagleLib/nodes/Node.h"

#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>

namespace EagleLib
{
    namespace Nodes
    {
        class ProcessFuture: public Node
        {
            boost::thread _thread;
            std::condition_variable_any _cv;
			bool _pause;
        public:
            struct ProcessFutureInfo: public NodeInfo
            {
				ProcessFutureInfo();
                virtual std::string GetObjectTooltip();
                virtual std::string GetObjectHelp();
            };

            ProcessFuture();
            ~ProcessFuture();
            virtual void NodeInit(bool firstInit);
			virtual void SetDataStream(IDataStream* stream);
            virtual TS<SyncedMemory> process(TS<SyncedMemory>& input, cv::cuda::Stream& stream);
			virtual TS<SyncedMemory> doProcess(TS<SyncedMemory>& input, cv::cuda::Stream& stream);
            virtual void process_ahead();
			virtual void stop_thread();
			virtual void start_thread();
        };
    }
}