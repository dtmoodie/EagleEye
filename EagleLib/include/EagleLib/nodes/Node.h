#pragma once

/*
 *  Nodes are the base processing object of this library.  Nodes can encapsulate all aspects of image processing from
 *  a single operation to a parallel set of intercommunicating processing stacks.  Nodes achieve this through one or more
 *  of the following concepts:
 *
 *  1) Processing node - Input image -> output image
 *  2) Function node - publishes a boost::function object to be used by sibling nodes
 *  3) Object node - publishes an object to be used by sibling nodes
 *  4) Serial collection nodes - Input image gets processed by all child nodes in series
 *  5) Parallel collection nodes - Input image gets processed by all child nodes in parallel. One thread per node
 *
 *  Nodes should be organized in a tree structure.  Each node will be accessible by name from the top of the tree via /parent/...../treeName where
 *  treeName is the unique name associated with that node.  The parameters of that node can be accessed via /parent/...../treeName:name.
 *  Nodes should be iterable by their parents by insertion order.  They should be accessible by sibling nodes.
 *
 *  Parameters:
 *  - Four main types of parameters, input, output, control and status.
 *  -- Input parameters should be defined in the expecting node by their internal name, datatype and the name of the output assigned to this input
 *  -- Output parameters should be defined by their datatype, memory location and
 *  - Designs:
 *  -- Could have a vector of parameter objects for each type.  The parameter object would contain the tree name of the parameter,
 *      the parameter type, and a pointer to the parameter
 *  -- Other considerations include using a boost::shared_ptr to the parameter, in which case constructing node and any other node that uses the parameter would share access.
 *      this has the advantage that the parameters don't need to be updated when an object swap occurs, since they aren't deleted.
 *      This would be nice for complex parameter objects, but it has the downside of functors not being updated correctly, which isn't such a big deal because the
 *      developer should just update functors accordingly in the init(bool) function.
 *
*/

// In library includes
#include "EagleLib/Detail/Export.hpp"
#include "EagleLib/Algorithm.h"
#include "EagleLib/SyncedMemory.h"
#include "EagleLib/utilities/CudaUtils.hpp"

// RCC includes
#include <IObject.h>
#include <IObjectInfo.h>
#include <ObjectInterfacePerModule.h>
#include <RuntimeLinkLibrary.h>
#include <shared_ptr.hpp>

// Dependent in house libraries
#include <MetaObject/Signals/TypedSlot.hpp>
#include <MetaObject/Signals/detail/SlotMacros.hpp>
#include <MetaObject/Signals/detail/SignalMacros.hpp>
#include <MetaObject/Detail/MetaObjectMacros.hpp>
#include <MetaObject/IMetaObject.hpp>
#include <MetaObject/IMetaObjectInfo.hpp>

// Dependent 3rd party libraries
#include <opencv2/core/cuda.hpp>
#include <EagleLib/rcc/external_includes/cv_core.hpp>
#include <EagleLib/rcc/external_includes/cv_highgui.hpp>

#include <boost/function.hpp>
#include <boost/log/attributes/scoped_attribute.hpp>
#include <boost/log/expressions/keyword.hpp>
#include <boost/log/attributes/mutable_constant.hpp>

// stl
#include <vector>
#include <type_traits>

namespace mo
{
    class IVariableManager;
}
namespace EagleLib
{
    namespace Nodes
    {
        class Node;
        class NodeImpl;
    }
    class IDataStream;
    class NodeFactory;
}


#define NODE_DEFAULT_CONSTRUCTOR_IMPL(NodeName, ...) \
NodeName::NodeName()                            \
{                                               \
}                                                \
static NodeInfo g_registerer_##NodeName(#NodeName, {STRINGIFY(__VA_ARGS__) });\
REGISTERCLASS(NodeName, &g_registerer_##NodeName)


#define SET_NODE_TOOLTIP(name, tooltip) g_registerer_##name.node_tooltip = std::string(tooltip);
#define SET_NODE_HELP(name, help) g_registerer_##name.node_help = std::string(help);

namespace EagleLib
{
    namespace Nodes
    {
    struct EAGLE_EXPORTS NodeInfoRegisterer
    {
        NodeInfoRegisterer(const char* nodeName, const char** nodeInfo);
        NodeInfoRegisterer(const char* nodeName, std::initializer_list<char const*> nodeInfo);
    };

    struct EAGLE_EXPORTS NodeInfo: public mo::IMetaObjectInfo
    {
        NodeInfo(const char* name, std::initializer_list<char const*> nodeInfo);
        int GetInterfaceId() const;
        std::string GetObjectName() const;
        std::string GetObjectTooltip() const;
        std::string GetObjectHelp() const;
        std::string Print() const;
        // Get the organizational hierarchy of this node, ie Image -> Processing -> ConvertToGrey
        virtual std::vector<const char*> GetNodeHierarchy() const;
        
        // List of nodes that need to be in the direct parental tree of this node, in required order
        virtual std::vector<std::vector<std::string>> GetParentalDependencies() const;
        
        // List of nodes that must exist in this data stream, but do not need to be in the direct parental tree of this node
        virtual std::vector<std::vector<std::string>> GetNonParentalDependencies() const;

        // Given the variable manager for a datastream, look for missing dependent variables and return a list of candidate nodes that provide those variables
        virtual std::vector<std::string> CheckDependentVariables(mo::IVariableManager* var_manager_) const;

        std::string node_name;
        std::string node_tooltip;
        std::string node_help;
        std::vector<const char*> node_hierarchy;
    };

    class EAGLE_EXPORTS Node: public TInterface<IID_NodeObject, Algorithm>
    {
    public:
        typedef rcc::shared_ptr<Node> Ptr;

        Node();
        virtual ~Node();

        std::string                     GetName() const;
        std::vector<rcc::weak_ptr<Node>>              GetParents();
        
        virtual Node::Ptr               AddChild(Node* child);
        virtual Node::Ptr               AddChild(Node::Ptr child);
        virtual Node::Ptr               GetChild(const std::string& treeName);
        virtual Node::Ptr               GetChild(const int& index);
        virtual void                    RemoveChild(const std::string& name);
        virtual void                    RemoveChild(Node::Ptr node);
        virtual void                    RemoveChild(Node* node);
        virtual void                    RemoveChild(rcc::weak_ptr<Node> node);
        virtual void                    RemoveChild(int idx);

        virtual void                    SwapChildren(int idx1, int idx2);
        virtual void                    SwapChildren(const std::string& name1, const std::string& name2);
        virtual void                    SwapChildren(Node::Ptr child1, Node::Ptr child2);

        virtual std::vector<Node*>      GetNodesInScope();
        virtual Node *                  GetNodeInScope(const std::string& name);
        virtual void                    GetNodesInScope(std::vector<Node*>& nodes);
        
        virtual void                    SetDataStream(IDataStream* stream);
        virtual IDataStream*            GetDataStream();

        void                            SetTreeName(const std::string& name);
        std::string                     GetTreeName() const;
        void                            SetTreeRoot(const std::string& name);
        std::string                     GetTreeRoot() const;
        virtual void                    AddParent(Node *parent);
        
        virtual void                    Init(bool firstInit);
        virtual void                    NodeInit(bool firstInit);

        virtual void                    Init(const std::string& configFile);
        virtual void                    Init(const cv::FileNode& configNode);

        virtual void                    Serialize(ISimpleSerializer *pSerializer);
        virtual void                    Serialize(cv::FileStorage& fs);

        
        
        bool                                                                externalDisplay;
        bool                                                                enabled;
        
        void                             Clock(int line_num);
        
        MO_BEGIN(Node)
            MO_SLOT(void, reset);
            MO_SIGNAL(void, node_updated, Node*);
        MO_END

    protected:
        friend class NodeFactory;
        IDataStream*                                                             _dataStream;
        long long                                                                _current_timestamp;
        std::shared_ptr<mo::IVariableManager>                                    _variable_manager;
        std::vector<Node::Ptr>                                              children;
        std::vector<rcc::weak_ptr<Node>>                                                     _parents;
        // Name as placed in the tree ie: RootNode/SerialStack/Sobel-1
        std::string                                                                treeRoot;
        // Name as it is stored in the children map, should be unique at this point in the tree. IE: Sobel-1
        std::string                                                                treeName;
    private:

        void ClearProcessingTime();
        void EndProcessingTime();
        
        std::shared_ptr<NodeImpl>                                               pImpl_;
        unsigned int                                                            rmt_hash;
        unsigned int                                                            rmt_cuda_hash;
    };
    } // namespace Nodes
} // namespace EagleLib