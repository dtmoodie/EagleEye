#include "EagleLib/ParameterBuffer.h"
using namespace EagleLib;
ParameterBuffer::ParameterBuffer(int size)
{
    _initial_size = size;
}
boost::any& ParameterBuffer::GetParameter(mo::TypeInfo type, const std::string& name, int frameNumber)
{
    std::lock_guard<std::mutex> lock(mtx);
    auto& param_buffer = _parameter_map[type][name];
    param_buffer.set_capacity(_initial_size);
    for (auto& param : param_buffer)
    {
        if (param.frame_number == frameNumber)
        {
            return param;
        }
    }
    param_buffer.push_back(FN<boost::any>(frameNumber));
    return param_buffer.back();
}
void ParameterBuffer::SetBufferSize(int size)
{
    for (auto& types : _parameter_map)
    {
        for (auto& buffers : types.second)
        {
            buffers.second.set_capacity(size);
        }
    }
    _initial_size = size;
}
