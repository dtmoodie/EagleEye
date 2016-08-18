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
    class NodeManager;
    class IDataStream;
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

    struct EAGLE_EXPORTS NodeInfo: public IObjectInfo
    {
        NodeInfo(const char* name, std::initializer_list<char const*> nodeInfo);
        virtual int GetObjectInfoType();
        virtual std::string GetObjectName();
        virtual std::string GetObjectTooltip();
        virtual std::string GetObjectHelp();
        // Get the organizational hierarchy of this node, ie Image -> Processing -> ConvertToGrey
        virtual std::vector<const char*> GetNodeHierarchy();
        
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

    class EAGLE_EXPORTS Node: public TInterface<IID_NodeObject, mo::IMetaObject>
    {
    public:
        typedef rcc::shared_ptr<Node> Ptr;

        Node();
        virtual ~Node();
        virtual cv::cuda::GpuMat        process(cv::cuda::GpuMat& img, cv::cuda::Stream& steam = cv::cuda::Stream::Null());
        virtual TS<SyncedMemory>        process(TS<SyncedMemory>& input, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual bool                    pre_check(const TS<SyncedMemory>& input);
        virtual cv::cuda::GpuMat        doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual TS<SyncedMemory>        doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream);
        std::string                     getName() const;
        std::string                     getTreeName();
        std::string                     getFullTreeName();
        Node *                          getParent();
        virtual void                    getInputs();
        virtual void                    updateParent();
        virtual void                    registerDisplayCallback(boost::function<void(cv::Mat, Node*)>& f);
        virtual void                    registerDisplayCallback(boost::function<void(cv::cuda::GpuMat, Node*)>& f);
        virtual void                    spawnDisplay();
        virtual void                    killDisplay();
        virtual Node::Ptr               addChild(Node* child);
        virtual Node::Ptr               addChild(Node::Ptr child);
        virtual Node::Ptr               getChild(const std::string& treeName);
        virtual Node::Ptr               getChild(const int& index);
        virtual Node*                   getChildRecursive(std::string treeName_);
        virtual void                    removeChild(const std::string& name);
        virtual void                    removeChild(Node::Ptr node);
        virtual void                    removeChild(Node* node);
        virtual void                    removeChild(rcc::weak_ptr<Node> node);
        virtual void                    removeChild(int idx);
        virtual void                    swapChildren(int idx1, int idx2);
        virtual void                    swapChildren(const std::string& name1, const std::string& name2);
        virtual void                    swapChildren(Node::Ptr child1, Node::Ptr child2);
        virtual std::vector<Node*>      getNodesInScope();
        virtual Node *                  getNodeInScope(const std::string& name);
        virtual void                    getNodesInScope(std::vector<Node*>& nodes);
        virtual void                    SetDataStream(IDataStream* stream);
        virtual IDataStream*            GetDataStream();

        virtual void                    setTreeName(const std::string& name);
        virtual void                    setFullTreeName(const std::string& name);
        virtual void                    setParent(Node *parent);
        virtual Node*                   swap(Node *other);
        virtual void                    Init(bool firstInit = true);
        virtual void                    NodeInit(bool firstInit = true);
        virtual void                    Init(const std::string& configFile);
        virtual void                    Init(const cv::FileNode& configNode);
        virtual void                    Serialize(ISimpleSerializer *pSerializer);
        virtual void                    Serialize(cv::FileStorage& fs);
        virtual bool                    SkipEmpty() const;
        long long                       GetTimestamp() const;
        std::vector<std::pair<time_t, int>> GetProfileTimings() const;
        std::vector<Node::Ptr>                                              children;
        bool                                                                externalDisplay;
        bool                                                                enabled;
        bool                                                                profile;
        double                           GetProcessingTime() const;
        void                             Clock(int line_num);
        
        MO_BEGIN(Node)
            MO_SLOT(void, reset);
            MO_SIGNAL(void, node_updated, Node*);
        MO_END

    protected:
        IDataStream*                                                             _dataStream;
        long long                                                                _current_timestamp;
        std::shared_ptr<mo::IVariableManager>                                    _variable_manager;

        rcc::weak_ptr<Node>                                                     parent;
        // Name as placed in the tree ie: RootNode/SerialStack/Sobel-1
        std::string                                                                fullTreeName;
        // Name as it is stored in the children map, should be unique at this point in the tree. IE: Sobel-1
        std::string                                                                treeName;
    private:
        friend class EagleLib::NodeManager;

        void ClearProcessingTime();
        void EndProcessingTime();
        
        std::shared_ptr<NodeImpl>                                               pImpl_;
        unsigned int                                                            rmt_hash;
        unsigned int                                                            rmt_cuda_hash;
    };
    } // namespace Nodes
} // namespace EagleLib
