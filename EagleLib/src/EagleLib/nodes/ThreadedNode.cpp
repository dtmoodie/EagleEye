#include "EagleLib/nodes/ThreadedNode.h"
#include <MetaObject/Thread/InterThread.hpp>
#include <MetaObject/Thread/BoostThread.h>
using namespace EagleLib;
using namespace EagleLib::Nodes;

ThreadedNode::ThreadedNode()
{
    _run = false;
    StartThread();
    _thread_context.thread_id = mo::GetThreadId(_processing_thread);
    _run = true;
}

ThreadedNode::~ThreadedNode()
{
    StopThread();
}

void ThreadedNode::StopThread()
{
    _processing_thread.interrupt();
    _processing_thread.join();
}

void ThreadedNode::PauseThread()
{
    _run = false;
}

void ThreadedNode::ResumeThread()
{
    _run = true;
}

void ThreadedNode::StartThread()
{
    _processing_thread = boost::thread(boost::bind(&ThreadedNode::processingFunction, this));
}

void ThreadedNode::Process()
{

}

/*bool ThreadedNode::ConnectInput(rcc::shared_ptr<Node> node, 
                                const std::string& input_name, 
                                const std::string& output_name, 
                                mo::ParameterTypeFlags type)
{

}*/

Node::Ptr ThreadedNode::AddChild(Node* child)
{
    auto ptr = Node::AddChild(child);
    child->SetContext(&_thread_context);

    return ptr;
}

Node::Ptr ThreadedNode::AddChild(Node::Ptr child)
{
    auto ptr = Node::AddChild(child);
    child->SetContext(&_thread_context);

    return ptr;
}

void ThreadedNode::processingFunction()
{
    while(!boost::this_thread::interruption_requested())
    {
        if(_run)
        {
            mo::ThreadSpecificQueue::Run(_thread_context.thread_id);
            for(auto& child : _children)
            {
                child->Process();
            }
        }else
        {
            boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
        }    
    }
}

MO_REGISTER_OBJECT(ThreadedNode);