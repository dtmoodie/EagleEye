#include "SubGraph.hpp"
#include "MetaObject/thread/ThreadPool.hpp"
#include <Aquila/nodes/NodeInfo.hpp>

#include <MetaObject/core/SystemTable.hpp>
namespace aq
{
    namespace nodes
    {

        SubGraph::SubGraph()
        {
            m_dirty = true;
            m_quit = false;

            m_thread = SystemTable::instance()->getSingleton<mo::ThreadPool>()->requestThread();
            m_thread->setName("SubGraph");
            m_thread_stream = m_thread->asyncStream();
            m_thread_stream->setName("SubGraph");
        }

        SubGraph::~SubGraph() { m_quit = true; }

        void SubGraph::setName(const std::string& name)
        {
            Node::setName(name);
            m_thread->setName(name);
            m_thread_stream->setName(name);
        }

        void SubGraph::addChild(Ptr child)
        {
            Node::addChild(child);
            child->setStream(m_thread_stream);
        }

        void SubGraph::startThread()
        {
            m_quit = false;
            m_thread_stream->pushWork([this](mo::IAsyncStream& stream)
            {
                MO_ASSERT(&stream == this->m_thread_stream.get());
                this->loop();
            });
        }

        void SubGraph::stopThread() { m_quit = true; }

        void SubGraph::node_updated(INode*) { m_dirty = true; }

        void SubGraph::update() { m_dirty = true; }

        void SubGraph::param_updated(mo::IMetaObject*, mo::IParam*) { m_dirty = true; }

        void SubGraph::param_added(mo::IMetaObject*, mo::IParam*) { m_dirty = true; }

        bool SubGraph::processImpl() { return true; }

        bool SubGraph::process() { return true; }

        void SubGraph::loop()
        {
            MO_ASSERT(m_thread_stream != nullptr);
            processChildren(*m_thread_stream);
            if (!m_quit)
            {
                m_thread_stream->pushWork([this](mo::IAsyncStream& stream)
                {
                    MO_ASSERT(&stream == this->m_thread_stream.get());
                    this->loop();
                });
            }
        }

    } // namespace nodes
} // namespace aq

using namespace aq::nodes;
MO_REGISTER_CLASS(SubGraph)
