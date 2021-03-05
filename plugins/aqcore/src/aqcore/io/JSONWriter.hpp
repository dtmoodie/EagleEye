#pragma once
#include "Aquila/nodes/Node.hpp"
#include <MetaObject/serialization/JSONPrinter.hpp>
#include <MetaObject/types/file_types.hpp>

#include <fstream>
#include <memory>
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
                MO_SLOT(void, on_input_set, const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&)
            MO_END;

          protected:
            bool processImpl();
            std::ofstream ofs;
            std::shared_ptr<mo::JSONSaver> m_ar;
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
            std::unique_ptr<mo::JSONLoader> m_ar;
            std::ifstream ifs;
            mo::ISubscriber* input;
        };
    } // namespace nodes
} // namespace aq
