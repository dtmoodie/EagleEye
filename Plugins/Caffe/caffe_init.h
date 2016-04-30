#pragma once

namespace EagleLib
{
	class PLUGIN_EXPORTS caffe_init_singleton
	{
		caffe_init_singleton();
	public:
		static caffe_init_singleton* inst();
	};	
}