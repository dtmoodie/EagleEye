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
ParallelStack::process(cv::cuda::GpuMat img)
{
    //std::vector<boost::promise<cv::cuda::GpuMat> > retVals(children.size());
    boost::promise<cv::cuda::GpuMat> retVal;
    threads.reserve(children.size());
    for(int i = 0; i < children.size(); ++i)
    {
        threads.push_back(boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(&Node::doProcess, children[i], img, boost::ref(retVal)))));
    }
    for(int i = 0; i < children.size(); ++i)
    {
        retVal.get_future().get();
    }
	return retVal.get_future().get();
}
