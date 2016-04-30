#pragma once
#include "EagleLib/Defs.hpp"

#include "IObject.h"

#include <parameters/ParameteredObject.h>
namespace EagleLib
{
	class EAGLE_EXPORTS ParameteredIObject : public IObject, public Parameters::ParameteredObject
	{
	public:
		ParameteredIObject();
		virtual void Serialize(ISimpleSerializer* pSerializer);
		virtual void Init(const cv::FileNode& configNode);
		virtual void Init(bool firstInit);
		virtual void SerializeAllParams(ISimpleSerializer* pSerializer);
	};

}
