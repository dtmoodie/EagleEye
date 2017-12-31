#pragma once
#include "Aquila/nodes/Node.hpp"
#include <MetaObject/serialization/SerializationFactory.hpp>
#include <cereal/archives/json.hpp>
#include <fstream>
namespace aq
{
    namespace nodes
    {
        class JSONWriter : public Node
        {
          public:
            JSONWriter();
            ~JSONWriter();
            MO_DERIVE(JSONWriter, Node)
            PARAM(mo::WriteFile, output_file, mo::WriteFile("output_file.json"))
            PARAM_UPDATE_SLOT(output_file)
            MO_SLOT(void,
                    on_input_set,
                    mo::IParam*,
                    mo::Context*,
                    mo::OptionalTime_t,
                    size_t,
                    const std::shared_ptr<mo::ICoordinateSystem>&,
                    mo::UpdateFlags)
            MO_END;

          protected:
            bool processImpl();
            std::ofstream ofs;
            std::shared_ptr<cereal::JSONOutputArchive> ar;
        };

        class JSONReader : public Node
        {
          public:
            JSONReader();
            MO_DERIVE(JSONReader, Node)
            PARAM(mo::ReadFile, input_file, mo::ReadFile("output_file.json"))
            MO_END;

          protected:
            bool processImpl();
            std::shared_ptr<cereal::JSONInputArchive> ar;
            std::ifstream ifs;
            mo::InputParam* input;
        };
    }
}
