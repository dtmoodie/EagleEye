#pragma once
#include <nodes/Node.h>
#include <Manager.h>


namespace EagleLib
{
	class CV_EXPORTS BroxOpticalFlow : public Node
    {
    public:
        BroxOpticalFlow();
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img);
        //virtual void getInputs();
    private:
        cv::cuda::GpuMat prevFrame;
    };

	class CV_EXPORTS PyrLKOpticalFlow : public Node
	{
	public:
        PyrLKOpticalFlow();
		cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img);
		void sparse(cv::cuda::GpuMat& img, cv::cuda::GpuMat& pts, cv::cuda::GpuMat& results, cv::cuda::GpuMat* error = 0);
		void dense(cv::cuda::GpuMat& img, cv::cuda::GpuMat& u, cv::cuda::GpuMat& v, cv::cuda::GpuMat* error = 0);
		void setReference(cv::cuda::GpuMat& img, cv::cuda::GpuMat* refPts);
        virtual void getInputs();
    private:
        cv::cuda::GpuMat refImg;
		cv::cuda::GpuMat refPts;
		cv::cuda::GpuMat prevPts;
	};


	// Static object forces constructor to be called at startup
//	REGISTER_TYPE(PyrLKOpticalFlow);
	/*
	class PyrLKOpticalFlowFactory : public NodeFactory
    {
    public:
        PyrLKOpticalFlowFactory()
        {
            Node::registerType("PyrLKOpticalFlow", this);
        }
        virtual boost::shared_ptr<Node> create()
        {
            return boost::shared_ptr<Node>(new PyrLKOpticalFlow());
        }
    };
	static PyrLKOpticalFlowFactory global_PyrLKOpticalFlowFactory;*/
}

//using namespace EagleLib;
//REGISTERCLASS(BroxOpticalFlow)
//REGISTERCLASS(PyrLKOpticalFlow)
