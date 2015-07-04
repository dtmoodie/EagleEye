#pragma once


#include "OpenCV.hpp"
namespace Parameters
{
	namespace Persistence
	{

		template<typename T> class PersistencePolicy : public cv::PersistencePolicy<T>
		{
		public:
			
		};
	}
}

