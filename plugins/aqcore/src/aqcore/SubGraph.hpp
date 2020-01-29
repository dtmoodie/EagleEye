#pragma once
#include <Aquila/nodes/Node.hpp>
#include <MetaObject/thread/ThreadHandle.hpp>

namespace aq
{
namespace nodes
{
    class SubGraph: public Node
    {
    public:
        SubGraph();
        ~SubGraph();

        MO_DERIVE(SubGraph, Node)
            MO_SLOT(int, loop)

            MO_SLOT(void, startThread)
            MO_SLOT(void, stopThread)
            MO_SLOT(void, pauseThread)
            MO_SLOT(void, resumeThread)

            MO_SLOT(void, node_updated, INode*)
            MO_SLOT(void, update)
            MO_SLOT(void, param_updated, mo::IMetaObject*, mo::IParam*)
            MO_SLOT(void, param_added, mo::IMetaObject*, mo::IParam*)

        MO_END

        virtual void setTreeName(const std::string& name) override;
        virtual Ptr addChild(INode* child) override;
        virtual Ptr addChild(const Ptr& child) override;

    protected:
        virtual bool processImpl() override;
        virtual bool process() override;
    private:

        mo::ThreadHandle m_thread;
        mo::ConnectionPtr_t m_thread_connection;
        volatile bool m_dirty;
        mo::ContextPtr_t m_thread_context;
    };
}
}
