#pragma once

namespace EagleLib
{
    class Node;
    class NodeNotifiable
    {
    protected:
        Node* m_node;
    public:
        NodeNotifiable(Node* node_ = nullptr)
        {            m_node = node_;        }
        // Updates the node pointer on recompile, set to nullptr on delete
        virtual void updateNode(Node* node_)
        {            m_node = node_;        }
    };
}
