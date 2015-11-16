#pragma once
#include "nodes/Node.h"
#include <EagleLib/utilities/CudaUtils.hpp>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    class ConvertToGrey: public Node
    {
    public:
        ConvertToGrey();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };

    class ConvertToHSV: public Node
    {
    public:
        ConvertToHSV();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
	class ConvertToLab : public Node
	{
	public:
		ConvertToLab();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
	};
	class ConvertColorspace : public Node
	{
		BufferPool<cv::cuda::GpuMat, EventPolicy> resultBuffer;
	public:
		ConvertColorspace();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
	};
	class Magnitude : public Node
	{
		cv::cuda::GpuMat magnitude;
	public:
		Magnitude();
		virtual void Init(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
	};
    class ExtractChannels: public Node
    {
        int channelNum;
        ConstBuffer<std::vector<cv::cuda::GpuMat>> channelsBuffer;
        //std::vector<cv::cuda::GpuMat> channels;
    public:
        ExtractChannels();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    };
    class ConvertDataType: public Node
    {
    public:
        ConvertDataType();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    class Merge: public Node
    {
        bool qualifiersSet;
        cv::cuda::GpuMat mergedChannels;
    public:
        Merge();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream);
    };
    class Reshape: public Node
    {
    public:
        Reshape();
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
