#pragma once
#include "EagleLib/nodes/Node.h"
#include <boost/thread.hpp>

namespace EagleLib
{
    namespace Nodes
    {
        class EAGLE_EXPORTS ThreadedNode: public Node
        {
        public:
            ThreadedNode();
            ~ThreadedNode();
            
            void Process();
            
            Node::Ptr AddChild(Node* child);
            Node::Ptr AddChild(Node::Ptr child);
            /*bool      ConnectInput(rcc::shared_ptr<Node> node, 
                                   const std::string& input_name, 
                                   const std::string& output_name, 
                                   mo::ParameterTypeFlags type = mo::StreamBuffer_e);*/

            MO_BEGIN(ThreadedNode, Node);
                MO_SLOT(void, StopThread);
                MO_SLOT(void, PauseThread);
                MO_SLOT(void, ResumeThread);
                MO_SLOT(void, StartThread);
            MO_END;
        protected:

        private:
            void processingFunction();
            mo::Context _thread_context;
            boost::thread _processing_thread;
            bool _run;
        };
    }
}