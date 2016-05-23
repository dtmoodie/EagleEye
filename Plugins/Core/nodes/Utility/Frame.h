#pragma once
#include "EagleLib/nodes/Node.h"
#include "EagleLib/utilities/CudaUtils.hpp"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    namespace Nodes
    {
    class FrameRate: public Node
    {
        boost::posix_time::ptime prevTime;

    public:
        FrameRate();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
	
	class FrameLimiter : public Node
	{
		boost::posix_time::ptime lastTime;
	public:
		FrameLimiter();
		virtual void NodeInit(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
	};

    class CreateMat: public Node
    {
        cv::cuda::GpuMat createdMat;
    public:
        CreateMat();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    class SetMatrixValues: public Node
    {
        bool qualifiersSetup;
    public:
        SetMatrixValues();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
	class Resize : public Node
	{
		BufferPool<cv::cuda::GpuMat, EventPolicy> bufferPool;
	public:
		Resize();
		virtual void NodeInit(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
	};
	class Subtract : public Node
	{
	public:
		Subtract();
		virtual void NodeInit(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
	};
    }
}
