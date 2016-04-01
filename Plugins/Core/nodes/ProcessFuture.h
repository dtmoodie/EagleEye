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
        public:
            struct ProcessFutureInfo: public NodeInfo
            {
                virtual std::string GetObjectName();
                virtual std::string GetObjectTooltip();
                virtual std::string GetObjectHelp();
		        // Get the organizational hierarchy of this node, ie Image -> Processing -> ConvertToGrey
                virtual std::vector<const char*> GetNodeHierarchy();
            };

            ProcessFuture();
            ~ProcessFuture();
            virtual void Init(bool firstInit);
            virtual TS<SyncedMemory> process(TS<SyncedMemory>& input, cv::cuda::Stream& stream);
            virtual void process_ahead();
        };
    }
}