
#include "FastMS.h"


#ifdef FASTMS_FOUND
using namespace EagleLib;
void FastMumfordShah::Init(bool firstInit)
{
	Node::Init(firstInit);
	if (firstInit)
	{
		updateParameter("Lambda", double(0.1))->SetTooltip("For bigger values, number of discontinuities will be smaller, for smaller values more discontinuities")->type;
		updateParameter("Alpha", double(20.0))->SetTooltip("For bigger values, solution will be more flat, for smaller values, solution will be more rough.");
		updateParameter("Temporal", double(0.0))->SetTooltip("For bigger values, solution will be driven to be similar to the previous frame, smaller values will allow for more interframe independence");
		updateParameter("Iterations", int(10000))->SetTooltip("Max number of iterations to perform");
		updateParameter("Epsilon", double(5e-5));
		updateParameter("Stop K", int(10))->SetTooltip("How often epsilon should be evaluated and checked");
		updateParameter("Adapt Params", false)->SetTooltip("If true: lambda and alpha will be adapted so that the solution will look more or less the same, for one and the same input image and for different scalings.");
		updateParameter("Weight", false)->SetTooltip("If true: The regularizer will be adjust to smooth less at pixels with high edge probability");
		updateParameter("Overlay edges", false);
		solver.reset(new Solver());
	}	
}

cv::cuda::GpuMat FastMumfordShah::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    img.download(h_img, stream);
    Par param;
	param.lambda = *getParameter<double>(0)->Data();
	param.alpha = *getParameter<double>(1)->Data();
	param.temporal = *getParameter<double>(2)->Data();
	param.iterations = *getParameter<int>(3)->Data();
	param.stop_eps = *getParameter<double>(4)->Data();
	param.stop_k = *getParameter<int>(5)->Data();
	param.adapt_params = *getParameter<bool>(6)->Data();
	param.weight = *getParameter<bool>(7)->Data();
	param.edges = *getParameter<bool>(8)->Data();
	stream.waitForCompletion();
	cv::Mat result = solver->run(h_img.createMatHeader(), param);
	img.upload(result, stream);
	return img;
}
void FastMumfordShah::Serialize(ISimpleSerializer* pSerializer)
{
	Node::Serialize(pSerializer);
	SERIALIZE(solver);
	SERIALIZE(h_img);
}
NODE_DEFAULT_CONSTRUCTOR_IMPL(FastMumfordShah);
REGISTER_NODE_HIERARCHY(FastMumfordShah, Image, Processing, Segmentation);
#endif
