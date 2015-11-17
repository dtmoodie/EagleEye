#pragma once
#include "Node.h"
namespace EagleLib
{
	class EAGLE_EXPORTS ProcessingNode
	{
	public:
		virtual NodeType GetType() const;
	};
}