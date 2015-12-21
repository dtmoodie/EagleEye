#pragma once
#include "nodes/Node.h"
#include "EagleLib/utilities/CudaUtils.hpp"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    class FrameRate: public Node
    {
        boost::posix_time::ptime prevTime;

    public:
        FrameRate();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
	
	class FrameLimiter : public Node
	{
		boost::posix_time::ptime lastTime;
	public:
		FrameLimiter();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
	};

    class CreateMat: public Node
    {
        cv::cuda::GpuMat createdMat;
    public:
        CreateMat();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    class SetMatrixValues: public Node
    {
        bool qualifiersSetup;
    public:
        SetMatrixValues();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
	class Resize : public Node
	{
		BufferPool<cv::cuda::GpuMat, EventPolicy> bufferPool;
	public:
		Resize();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
	};
	class Subtract : public Node
	{
	public:
		Subtract();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
	};
}
