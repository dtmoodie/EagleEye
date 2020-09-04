#pragma once
#include <Aquila/nodes/Node.hpp>
#include <MetaObject/thread/ConditionVariable.hpp>
#include <MetaObject/thread/Mutex.hpp>
#include <MetaObject/thread/ThreadHandle.hpp>
namespace aq
{
    namespace nodes
    {
        class SubGraph : public Node
        {
          public:
            SubGraph();
            ~SubGraph();

            MO_DERIVE(SubGraph, Node)

                MO_SLOT(void, startThread)
                MO_SLOT(void, stopThread)

                MO_SLOT(void, node_updated, INode*)
                MO_SLOT(void, update)
                MO_SLOT(void, param_updated, mo::IMetaObject*, mo::IParam*)
                MO_SLOT(void, param_added, mo::IMetaObject*, mo::IParam*)

            MO_END;

            virtual void setName(const std::string& name) override;
            virtual void addChild(Ptr child) override;

          protected:
            virtual bool processImpl() override;
            virtual bool process() override;

          private:
            void loop();
            std::shared_ptr<mo::Thread> m_thread;
            mo::ConnectionPtr_t m_thread_connection;
            std::atomic<bool> m_dirty;
            std::atomic<bool> m_quit;
            std::shared_ptr<mo::IAsyncStream> m_thread_stream;
        };
    } // namespace nodes
} // namespace aq