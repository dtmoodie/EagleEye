#include "nodes/ParallelStack.h"
#include <boost/thread/future.hpp>
using namespace EagleLib;



ParallelStack::ParallelStack()
{

}

ParallelStack::~ParallelStack()
{

}

cv::cuda::GpuMat
ParallelStack::process(cv::cuda::GpuMat img, cv::cuda::Stream& stream)
{
    //std::vector<boost::promise<cv::cuda::GpuMat> > retVals(children.size());
    boost::promise<cv::cuda::GpuMat> retVal;
    threads.reserve(children.size());
    for(auto itr = children.begin(); itr != children.end(); ++itr)
    {
        //threads.push_back(boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(/*cast to the correct function call to avoid mis-resolution*/(void(Node::*)(cv::cuda::GpuMat&, boost::promise<cv::cuda::GpuMat>&))&Node::doProcess, *itr, img, boost::ref(retVal)))));
    }
    for(size_t i = 0; i < children.size(); ++i)
    {
        retVal.get_future().get();
    }
	return retVal.get_future().get();
}

REGISTERCLASS(ParallelStack)
