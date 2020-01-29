#include "SubGraph.hpp"
#include "MetaObject/thread/ThreadPool.hpp"
#include <Aquila/nodes/NodeInfo.hpp>

namespace aq
{
namespace nodes
{

SubGraph::SubGraph()
{
    m_dirty = true;
    m_thread = mo::ThreadPool::instance()->requestThread();
    m_thread_context = m_thread.getContext();
    _ctx = m_thread_context;
    m_thread_connection = m_thread.setInnerLoop(getSlot_loop<int(void)>());
    m_thread.setThreadName("SubGraph");
    m_thread.start();
}

SubGraph::~SubGraph()
{
    m_thread.stop();
}

void SubGraph::setTreeName(const std::string& name)
{
    Node::setTreeName(name);
    m_thread_context->setName(name);
    m_thread.setThreadName(name);
}

SubGraph::Ptr SubGraph::addChild(INode* child)
{
    auto ptr = Node::addChild(child);
    child->setContext(m_thread_context, true);
    return ptr;
}

SubGraph::Ptr SubGraph::addChild(const Ptr& child)
{
    auto ptr = Node::addChild(child);
    child->setContext(m_thread_context, true);
    return ptr;
}


void SubGraph::startThread()
{
    m_thread.start();
}

void SubGraph::stopThread()
{
    m_thread.stop();
}

void SubGraph::pauseThread()
{
    m_thread.stop();
}

void SubGraph::resumeThread()
{
    m_thread.start();
}


void SubGraph::node_updated(INode*)
{
    m_dirty = true;
}

void SubGraph::update()
{
    m_dirty = true;
}

void SubGraph::param_updated(mo::IMetaObject*, mo::IParam*)
{
    m_dirty = true;
}

void SubGraph::param_added(mo::IMetaObject*, mo::IParam*)
{
    m_dirty = true;
}


bool SubGraph::processImpl()
{
    return true;
}

bool SubGraph::process()
{
    return true;
}

int SubGraph::loop()
{
    processChildren();
    return 0;
}


}
}

using namespace aq::nodes;
MO_REGISTER_CLASS(SubGraph)
