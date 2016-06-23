#pragma once
#include <EagleLib/nodes/Node.h>
#include <EagleLib/ParameteredIObjectImpl.hpp>
namespace EagleLib
{
    namespace Nodes
    {
        class PLUGIN_EXPORTS result_cache: public Node
        {
        public:
            virtual TS<SyncedMemory> process(TS<SyncedMemory>& input, cv::cuda::Stream& stream);
            virtual bool pre_check(const TS<SyncedMemory>& input);

            BEGIN_PARAMS(result_cache);
                PARAM(std::vector<TS<SyncedMemory>>, cache, std::vector<TS<SyncedMemory>>());
            END_PARAMS;
        };
    }
}