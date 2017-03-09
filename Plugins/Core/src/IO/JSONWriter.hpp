#pragma once
#include "Aquila/Nodes/Node.h"
#include <MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp>
#include <cereal/archives/json.hpp>
#include <fstream>
namespace aq
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

        class JSONReader: public Node
        {
        public:
            JSONReader();
            MO_DERIVE(JSONReader, Node);
                PARAM(mo::ReadFile, input_file, mo::ReadFile("output_file.json"));
            MO_END;
        protected:
            bool ProcessImpl();
            std::shared_ptr<cereal::JSONInputArchive> ar;
            std::ifstream ifs;
            mo::InputParameter* input;
        };

    }
}