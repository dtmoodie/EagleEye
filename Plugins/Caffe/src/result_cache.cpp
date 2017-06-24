#include "result_cache.h"
//#include "Aquila/utilities/helper_macros.hpp"

#include "Aquila/core/Logger.hpp"

using namespace aq;
using namespace aq::nodes;

/*TS<SyncedMemory> result_cache::process(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    TS < SyncedMemory> output = input;

    if (children.size() == 0)
        return output;
    std::vector<Node::Ptr>  children_;
    {
        // Prevents adding of children while running, debatable how much this is needed
        std::lock_guard<std::recursive_mutex> lock(mtx);
        children_ = children;
    }
    TS<SyncedMemory> result = output;
    for (size_t i = 0; i < children_.size(); ++i)
    {
        if (children_[i] != nullptr)
        {
            try
            {
                result = children_[i]->process(result, stream);
            }CATCH_MACRO
        }
        else
        {
            ui_collector::set_node_name(getFullTreeName());
            NODE_LOG(error) << "Null child with idx: " + boost::lexical_cast<std::string>(i);
        }
    }
    cache.push_back(result);
    ui_collector::set_node_name(getFullTreeName());

    return output;
}

bool result_cache::pre_check(const TS<SyncedMemory>& input)
{
    return true;
}

static aq::nodes::NodeInfo g_registerer_result_cache("result_cache", { "utilities"});
REGISTERCLASS(result_cache, &g_registerer_result_cache);*/
