#pragma once
#include "EagleLib/Nodes/Node.h"
#include <MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp>
#include <cereal/archives/json.hpp>
#include <fstream>
namespace EagleLib
{
    namespace Nodes
    {
        class JSONWriter: public Node
        {
        public:
            JSONWriter();
            ~JSONWriter();
            MO_DERIVE(JSONWriter, Node);
                PARAM(mo::WriteFile, output_file, mo::WriteFile("output_file.json"));
                MO_SLOT(void, on_output_file_modified, mo::Context*, mo::IParameter*);
                MO_SLOT(void, on_input_set, mo::Context*, mo::IParameter*)
            MO_END;
        protected:
            bool ProcessImpl();
            std::ofstream ofs;
            std::shared_ptr<cereal::JSONOutputArchive> ar;
        };
    }
}