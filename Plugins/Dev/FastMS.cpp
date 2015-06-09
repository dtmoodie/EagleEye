
#include "FastMS.h"


#ifdef FASTMS_FOUND
using namespace EagleLib;
void SegmentFastMumfordShah::Init(bool firstInit)
{
	updateParameter("Lambda", double(0.1), Parameter::Control, "For bigger values, number of discontinuities will be smaller, for smaller values more discontinuities");
	updateParameter("Alpha", double(20.0), Parameter::Control, "For bigger values, solution will be more flat, for smaller values, solution will be more rough.");
	updateParameter("Temporal", double(0.0), Parameter::Control, "For bigger values, solution will be driven to be similar to the previous frame, smaller values will allow for more interframe independence");
	updateParameter("Iterations", int(10000), Parameter::Control, "Max number of iterations to perform");
	updateParameter("Epsilon", double(5e-5));
	updateParameter("Stop K", int(10), Parameter::Control, "How often epsilon should be evaluated and checked");
	updateParameter("Adapt Params", false, Parameter::Control, "If true: lambda and alpha will be adapted so that the solution will look more or less the same, for one and the same input image and for different scalings.");
	updateParameter("Weight", false, Parameter::Control, "If true: The regularizer will be adjust to smooth less at pixels with high edge probability");
	updateParameter("Overlay edges", false);
}

cv::cuda::GpuMat SegmentFastMumfordShah::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    img.download(h_img, stream);
    Par param;
	param.lambda = getParameter<double>(0)->data;
	param.alpha = getParameter<double>(1)->data;
	param.temporal = getParameter<double>(2)->data;
	param.iterations = getParameter<int>(3)->data;
	param.stop_eps = getParameter<double>(4)->data;
	param.stop_k = getParameter<int>(5)->data;
	param.adapt_params = getParameter<bool>(6)->data;
	param.weight = getParameter<bool>(7)->data;
	param.edges = getParameter<bool>(8)->data;
	cv::Mat result = solver.run(h_img.createMatHeader(), param);
	img.upload(result, stream);
	return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(SegmentFastMumfordShah);
#endif
