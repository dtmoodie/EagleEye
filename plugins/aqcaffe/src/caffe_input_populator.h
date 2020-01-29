#pragma once
#include <Aquila/nodes/Node.hpp>
#include <MetaObject/params/ParamMacros.hpp>
#include "CaffeExport.hpp"
namespace aq
{
    namespace nodes
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
