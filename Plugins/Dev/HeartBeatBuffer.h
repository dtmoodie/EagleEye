#pragma once
#include "EagleLib/nodes/Node.h"
#include <boost/circular_buffer.hpp>

namespace EagleLib
{
    namespace Nodes
    {
	class HeartBeatBuffer : public Node
	{
		boost::circular_buffer<TS<SyncedMemory>> image_buffer;
		time_t lastTime;
		bool activated;
		void onActivation();
	public:
		HeartBeatBuffer();
		virtual void Init(bool firstInit);
        TS<SyncedMemory> process(TS<SyncedMemory>& input, cv::cuda::Stream& stream);
		//virtual cv::cuda::GpuMat process(cv::cuda::GpuMat& img, cv::cuda::Stream& steam );
	};
    }
}