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
#include "EagleLib/IDataStream.hpp"
#include "EagleLib/Algorithm.h"
#include "EagleLib/SyncedMemory.h"
#include "EagleLib/utilities/CudaUtils.hpp"
#include <EagleLib/IO/serialize.hpp>

// RCC includes
#include <IObject.h>
#include <IObjectInfo.h>
#include <ObjectInterfacePerModule.h>
#include <RuntimeLinkLibrary.h>
#include <shared_ptr.hpp>

// Dependent in house libraries
#include <MetaObject/MetaObject.hpp>

// Dependent 3rd party libraries
#include <opencv2/core/cuda.hpp>
#include <EagleLib/rcc/external_includes/cv_core.hpp>
#include <EagleLib/rcc/external_includes/cv_highgui.hpp>

#include <string>

#define SCOPED_PROFILE_NODE mo::scoped_profile COMBINE(scoped_profile, __LINE__)((this->GetTreeName() + "::" + __FUNCTION__), &_rmt_hash, &_rmt_cuda_hash, &Stream());
#define LOG_NODE(severity) BOOST_LOG_TRIVIAL(severity) << "[" << this->GetTreeName() << "::" << __FUNCTION__ <<  "] - "
namespace mo
{
    class IVariableManager;
    typedef std::shared_ptr<IVariableManager> IVariableManagerPtr;
}
namespace EagleLib
{
    namespace Nodes
    {
        class Node;
        class NodeImpl;
    }
    class IDataStream;
    class DataStream;
    class NodeFactory;
}

namespace EagleLib
{
namespace Nodes
{
    struct NodeInfo;
    
    class EAGLE_EXPORTS Node:
            public TInterface<ctcrc32("EagleLib::Nodes::Node"), Algorithm>
    {
    public:
        typedef NodeInfo InterfaceInfo;
        typedef rcc::shared_ptr<Node> Ptr;
        typedef rcc::weak_ptr<Node>   WeakPtr;
        typedef std::vector<Ptr> VecPtr;

        Node();
        virtual bool                    Process();

        virtual void                    AddParent(Node *parent);

        std::vector<WeakPtr>            GetParents() const;

        virtual bool                    ConnectInput(Ptr output_node,
                                                     const std::string& output_name, 
                                                     const std::string& input_name, 
                                                     mo::ParameterTypeFlags type = mo::StreamBuffer_e);
        virtual bool                    ConnectInput(Ptr output_node,
                                                     mo::IParameter* output_param,
                                                     mo::InputParameter* input_param,
                                                     mo::ParameterTypeFlags type = mo::StreamBuffer_e);

        virtual Ptr                     AddChild(Node* child);
        virtual Ptr                     AddChild(Node::Ptr child);

        virtual Ptr                     GetChild(const std::string& treeName);
        virtual Ptr                     GetChild(const int& index);
        virtual VecPtr                  GetChildren();

        virtual void                    RemoveChild(const std::string& name);
        virtual void                    RemoveChild(Ptr node);
        virtual void                    RemoveChild(Node* node);
        virtual void                    RemoveChild(WeakPtr node);
        virtual void                    RemoveChild(int idx);

        virtual void                    SwapChildren(int idx1, int idx2);
        virtual void                    SwapChildren(const std::string& name1, const std::string& name2);
        virtual void                    SwapChildren(Node::Ptr child1, Node::Ptr child2);

        virtual void                    SetDataStream(IDataStream* stream);
        virtual IDataStream*            GetDataStream();
        virtual mo::IVariableManagerPtr GetVariableManager();

        void                            SetUniqueId(int id);
        std::string                     GetTreeName();
        std::string                     GetTreeName() const;
        void                            SetTreeName(const std::string& name);
        
        virtual void                    Init(bool firstInit);
        virtual void                    NodeInit(bool firstInit);
        virtual void                    PostSerializeInit();

        virtual void                    Serialize(ISimpleSerializer *pSerializer);
        inline cv::cuda::Stream&        Stream(){ CV_Assert(_ctx); return _ctx->GetStream();}
        
        InputState CheckInputs();

        MO_BEGIN(Node)
            MO_SLOT(void, reset)
            MO_SIGNAL(void, node_updated, Node*)
        MO_END

    protected:
        friend class NodeFactory;
        friend class IDataStream;
        friend class EagleLib::DataStream;

        virtual std::vector<Node*>                        GetNodesInScope();
        virtual Node *                                    GetNodeInScope(const std::string& name);
        virtual void                                      GetNodesInScope(std::vector<Node*>& nodes);
        virtual mo::IParameter* AddParameter(std::shared_ptr<mo::IParameter> param);
        virtual mo::IParameter* AddParameter(mo::IParameter* param);

        friend bool EagleLib::DeSerialize(cereal::JSONInputArchive& ar, Node* obj);
        
        void onParameterUpdate(mo::Context* ctx, mo::IParameter* param);
        // The data stream is kinda the graph owner, it produces data and pushes
        // It also calls Process for all of the children of the data stream
        // it onto the graph.
        
        // Current timestamp of the frame that this node is processing / processed last
        long long                                       _current_timestamp;
        // The variable manager is one object shared within a processing graph
        // that has knowledge of all inputs and outputs within the graph
        // It handles creating buffers, setting up contexts and all connecting nodes
        mo::IVariableManagerPtr                         _variable_manager;
        bool                                            _modified;
        // The children of a node are all nodes accepting inputs
        // from this node
        VecPtr                                          _children;
        rcc::weak_ptr<IDataStream>                      _data_stream;
        int _unique_id;
        std::vector<WeakPtr>                            _parents;
        unsigned int                                    _rmt_hash;
        unsigned int                                    _rmt_cuda_hash;
        std::string                                     name;
    private:
        std::shared_ptr<NodeImpl>                       _pimpl_node;
    };
} // namespace Nodes
} // namespace EagleLib
