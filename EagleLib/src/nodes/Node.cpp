#include "nodes/Node.h"
#include "../RuntimeObjectSystem/ObjectInterfacePerModule.h"
using namespace EagleLib;
Node::Node()
{

}

Node::~Node()
{

}
int Node::addChild(Node* child)
{
    boost::shared_ptr<Node> ptr(child);
    return addChild(ptr);
}

int Node::addChild(boost::shared_ptr<Node> child)
{
    if(errorCallback)
        child->errorCallback = errorCallback;
    if(statusCallback)
        child->statusCallback = statusCallback;
    if(warningCallback)
        child->warningCallback = warningCallback;
    for(int i = 0; i < child->parameters.size(); ++i)
    {
        childParameters.push_back(std::make_pair(i,child->parameters[i]));
    }
    children.push_back(child);
    return children.size() -1;
}
void
Node::getInputs()
{

}
void
Node::removeChild(boost::shared_ptr<Node> child)
{
    for(int i = 0; i < children.size(); ++i)
    {
        if(child == children[i])
        {
            children.erase(children.begin() + i);
            for(int j = 0; j < childParameters.size(); ++j)
            {
                if(childParameters[j].first == i)
                    childParameters.erase(childParameters.begin() + j);
            }
        }
    }
}

void
Node::removeChild(int idx)
{
    children.erase(children.begin() + idx);
    for(int j = 0; j < childParameters.size(); ++j)
    {
        if(childParameters[j].first == idx)
            childParameters.erase(childParameters.begin() + j);
    }

}

cv::cuda::GpuMat
Node::process(cv::cuda::GpuMat &img)
{
    try
    {
        return doProcess(img);
    }catch(cv::Exception &e)
    {
        if(errorCallback)
        {
         std::string message = std::string(__FUNCTION__) + std::string(e.what());
            errorCallback(message);
        }
    }
}

cv::cuda::GpuMat
Node::doProcess(cv::cuda::GpuMat& img)
{
    return img;
}
void
Node::doProcess(cv::cuda::GpuMat& img, boost::promise<cv::cuda::GpuMat> &retVal)
{
    retVal.set_value(process(img));
}

void
Node::registerDisplayCallback(boost::function<void(cv::Mat)>& f)
{
    cpuCallback = f;
}

void
Node::registerDisplayCallback(boost::function<void(cv::cuda::GpuMat)>& f)
{
    gpuCallback = f;
}

void
Node::spawnDisplay()
{

}
void
Node::killDisplay()
{

}
std::string
Node::getName()
{
    std::string name;
    if(parent != NULL)
        parent->getName() + "/" + name;
    return name;
}


REGISTERCLASS( Node );
