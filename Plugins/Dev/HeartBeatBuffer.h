#pragma once
#include "EagleLib/nodes/Node.h"
#include <boost/circular_buffer.hpp>

namespace EagleLib
{
    namespace Nodes
    {
	class HeartBeatBuffer : public Node
	{
		boost::circular_buffer<cv::cuda::GpuMat> image_buffer;
		time_t lastTime;
		bool activated;
		void onActivation();
	public:
		HeartBeatBuffer();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat process(cv::cuda::GpuMat& img, cv::cuda::Stream& steam );
	};
    }
}