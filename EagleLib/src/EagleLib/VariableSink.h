#pragma once

#include "IVariableSink.h"
#include <vector>
#include <string>
#include <fstream>

namespace EagleLib
{
    class VariableSink: public IVariableSink
    {
    public:
        VariableSink();
        virtual ~VariableSink();
        virtual void SerializeVariables(unsigned long long frame_number, mo::IVariableManager* manager);
    };

    class CSV_VariableSink: public IVariableSink
    {
    public:
        CSV_VariableSink(const std::string& output_file);
        virtual ~CSV_VariableSink();
        
        virtual void SerializeVariables(unsigned long long frame_number, mo::IVariableManager* manager);

        std::string SerializeExample(unsigned long long frame_number, mo::IVariableManager* manager);

        std::vector<std::string> ListSerializableVariables(mo::IVariableManager* manager);
        void AddVariable(const std::string& name);
        void SetLayout(const std::vector<std::string>& layout);
        std::vector<std::string> GetLayout();

        void open(const std::string& filename);
        void close();
    protected:
        // List of all parameters to serialize and in what order, by frame_number is the first value, 
        // followed by all values listed in this vector in the order
        // of this vector
        std::vector<std::string> _serialization_layout;
        std::ofstream _ofs;
    };
}
