#include "opencv.hpp"
#include "Parameters.hpp"
#include <opencv2/core/base.hpp>

using namespace Parameters::Persistence::cv;

std::map<Loki::TypeInfo, std::pair<InterpreterRegistry::SerializerFunction, InterpreterRegistry::DeSerializerFunction >> InterpreterRegistry::registry;

void InterpreterRegistry::RegisterFunction(Loki::TypeInfo& type, SerializerFunction serializer, DeSerializerFunction deserializer)
{
	registry[type] = std::make_pair(serializer, deserializer);
}

std::pair<InterpreterRegistry::SerializerFunction, InterpreterRegistry::DeSerializerFunction >& InterpreterRegistry::GetInterpretingFunction(Loki::TypeInfo& type)
{
	if (registry.find(type) == registry.end())
		::cv::error(::cv::Error::StsAssert, "Datatype not registered to the registry", CV_Func, __FILE__, __LINE__);
	return registry[type];
}
void Parameters::Persistence::cv::Serialize(::cv::FileStorage* fs, Parameters::Parameter* param)
{
	InterpreterRegistry::GetInterpretingFunction(param->GetTypeInfo()).first(fs, param);
}
void Parameters::Persistence::cv::DeSerialize(::cv::FileNode* fs, Parameters::Parameter* param)
{
	InterpreterRegistry::GetInterpretingFunction(param->GetTypeInfo()).second(fs, param);
}
Parameters::Parameter* Parameters::Persistence::cv::DeSerialize(::cv::FileNode* fs)
{
	//TODO object factory based on serialized type
	return nullptr;
}