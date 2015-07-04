#pragma once


// Check if being built with OpenCV

#include <map>
#include <functional>
#include <opencv2/core/persistence.hpp>
#include <LokiTypeInfo.h>
#include <boost/lexical_cast.hpp>
//#include <Parameters.hpp>

namespace Parameters
{
	class Parameter;
	namespace Persistence
	{
		namespace cv
		{

			class InterpreterRegistry
			{
				
			public:
				typedef std::function<void(::cv::FileStorage*, Parameters::Parameter*)> SerializerFunction;
				typedef std::function<void(::cv::FileNode*, Parameters::Parameter*)> DeSerializerFunction;
				static void RegisterFunction(Loki::TypeInfo& type, SerializerFunction serializer, DeSerializerFunction deserializer);
				static std::pair<SerializerFunction,	DeSerializerFunction >& GetInterpretingFunction(Loki::TypeInfo& type);
			private:
				// Mapping from Loki::typeinfo to file writing functors
				static	std::map<Loki::TypeInfo, std::pair<SerializerFunction, DeSerializerFunction >> registry;
			};
			void Serialize(::cv::FileStorage* fs, Parameters::Parameter* param);
			void DeSerialize(::cv::FileNode* fs, Parameters::Parameter* param);
			Parameters::Parameter* DeSerialize(::cv::FileNode* fs);

			template<typename T> void Serializer(::cv::FileStorage* fs, Parameters::Parameter* param)
			{
				ITypedParameter<T>* typedParam = dynamic_cast<ITypedParameter<T>*>(param);
				if (typedParam)
				{
					const std::string& toolTip = typedParam->GetTooltip();
					(*fs) << typedParam->GetName().c_str() << "{";
					(*fs) << "Data" << *typedParam->Data();
					(*fs) << "Type" << typedParam->GetTypeInfo().name();
					if (toolTip.size())
						(*fs) << "ToolTip" << toolTip;
					(*fs) << "}";
				}
			}
			template<typename T> void DeSerializer(::cv::FileNode* fs, Parameters::Parameter* param)
			{
				ITypedParameter<T>* typedParam = dynamic_cast<ITypedParameter<T>*>(param);
				if (typedParam)
				{
					::cv::FileNode myNode = (*fs)[param->GetName()];
					std::string type = (std::string)myNode["Type"];
					if (type == param->GetTypeInfo().name())
						myNode["Data"] >> *typedParam->Data();
					else
						::cv::error(::cv::Error::StsAssert, "Datatype " + std::string(param->GetTypeInfo().name()) + " requested, but " + type + " found in file", CV_Func, __FILE__, __LINE__);
				}
			}
			template<typename T> class PersistencePolicy
			{
				//static const Registerer<T> registerer = Registerer<T>();
			public:
				PersistencePolicy()
				{
					InterpreterRegistry::RegisterFunction(Loki::TypeInfo(typeid(T)), std::bind(Serializer<T>, std::placeholders::_1, std::placeholders::_2), std::bind(DeSerializer<T>, std::placeholders::_1, std::placeholders::_2));
				}
			};
		}
	}
	
}
