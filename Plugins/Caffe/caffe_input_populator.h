#pragma once
#include <EagleLib/nodes/Node.h>
#include <MetaObject/Parameters/ParameterMacros.hpp>
namespace EagleLib
{
    namespace Nodes
    {
        class PLUGIN_EXPORTS caffe_input_populator: public Node
        {
        public:
            std::vector<std::pair<int,int>> sample_permutation;

            MO_DERIVE(caffe_input_populator, Node);
                PARAM(bool, shuffle, true);
                STATUS(int, sample_index, 0);
                PARAM(int, blob_index, 0);
                PARAM(std::string, blob_name, "");
                //MO_SLOT(void, fill_blobs);
            MO_END;

            
        };
    }
}