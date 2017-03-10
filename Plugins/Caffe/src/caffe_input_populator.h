#pragma once
#include <Aquila/Nodes/Node.h>
#include <MetaObject/Parameters/ParameterMacros.hpp>
#include "CaffeExport.hpp"
namespace aq
{
    namespace Nodes
    {
        class Caffe_EXPORT caffe_input_populator: public Node
        {
        public:
            std::vector<std::pair<int,int>> sample_permutation;

            MO_DERIVE(caffe_input_populator, Node)
                PARAM(bool, shuffle, true)
                STATUS(int, sample_index, 0)
                PARAM(int, blob_index, 0)
                PARAM(std::string, blob_name, "")
            MO_END;
        };
    }
}
