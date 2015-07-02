#pragma once


#include "OpenCV.hpp"




namespace Parameters
{
	template<typename T> class PersistencePolicy: public cvPersistencePolicy<T>
	{
		PersistencePolicy():
			cvPersistencePolicy<T>() {}
	};
}

