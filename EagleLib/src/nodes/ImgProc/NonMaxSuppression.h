#pragma once

#include "nodes/Node.h"


namespace EagleLib
{
    class MinMax: public Node
    {
    public:
        MinMax();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
	class Threshold : public Node
	{
	public:
		Threshold();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
	};

    class NonMaxSuppression: public Node
    {
    public:
        NonMaxSuppression();
        void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
}
