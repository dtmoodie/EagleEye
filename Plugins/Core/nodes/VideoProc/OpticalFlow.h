#pragma once
#include <EagleLib/nodes/Node.h>
#include <EagleLib/rcc/external_includes/cv_cudaoptflow.hpp>
#include "EagleLib/utilities/CudaUtils.hpp"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
{
    namespace Nodes
    {
    
    class SparsePyrLKOpticalFlow: public Node
    {
        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> optFlow;
        std::vector<cv::cuda::GpuMat> prev_grey;
		//cv::cuda::GpuMat prev_grey;
		cv::cuda::GpuMat prev_key_points;
		void set_reference(cv::cuda::GpuMat& ref_image, cv::cuda::GpuMat& ref_points, size_t frame_number, cv::cuda::Stream& stream);


    public:
        SparsePyrLKOpticalFlow();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual void Serialize(ISimpleSerializer *pSerializer);
    };

	class DensePyrLKOpticalFlow : public Node
	{
		cv::Ptr<cv::cuda::DensePyrLKOpticalFlow> opt_flow;
		cv::cuda::GpuMat prevGreyImg;
		cv::cuda::GpuMat greyImg;
		cv::cuda::GpuMat flow;
	public:

		DensePyrLKOpticalFlow();
		virtual void NodeInit(bool firstInit);
		virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
	};



    class BroxOpticalFlow: public Node
    {
    public:
        BroxOpticalFlow();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    };
    class SparseOpticalFlow: public Node
    {
        cv::Ptr<cv::cuda::SparseOpticalFlow> optFlow;
    public:

    };

    class DenseOpticalFlow: public Node
    {
        public:
    };
    }
}
