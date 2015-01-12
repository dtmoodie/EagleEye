#include <nodes/Node.h>



namespace EagleLib
{
    class BroxOpticalFlow : public Node
    {
    public:
        BroxOpticalFlow();
        cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img);
        //virtual void getInputs();
    private:
        cv::cuda::GpuMat prevFrame;
    };

    class PyrLKOpticalFlow: public Node
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
    static PyrLKOpticalFlowFactory global_PyrLKOpticalFlowFactory;  // Static object forces constructor to be called at startup
}
