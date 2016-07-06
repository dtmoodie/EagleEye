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
#include "EagleLib/Defs.hpp"
//#include "EagleLib/ParameteredObject.h"
#include "EagleLib/Algorithm.h"
#include "EagleLib/SyncedMemory.h"
#include "EagleLib/utilities/CudaUtils.hpp"

// RCC includes
#include <IObject.h>
#include <IObjectInfo.h>
#include <ObjectInterfacePerModule.h>
#include <RuntimeLinkLibrary.h>
#include "EagleLib/rcc/shared_ptr.hpp"

// Dependent in house libraries
#include <parameters/LokiTypeInfo.h>
#include <parameters/Parameters.hpp>
#include <parameters/Types.hpp>
#include <signals/connection.h>

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

#define TIME Clock(__LINE__);

#ifdef _MSC_VER

#define NODE_LOG(severity)                                                                                                                              \
    /*BOOST_LOG_SCOPED_THREAD_ATTR("NodeName", boost::log::attributes::mutable_constant<std::string>(fullTreeName));*/                                        \
    /*BOOST_LOG_SCOPED_THREAD_ATTR("Node", boost::log::attributes::mutable_constant<const Node*>(this));*/                                                    \
    LOG(severity)

#else
#define NODE_LOG(severity)     LOG(severity)
#endif

namespace Parameters
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
        virtual std::vector<std::string> CheckDependentVariables(Parameters::IVariableManager* var_manager_) const;

        std::string node_name;
        std::string node_tooltip;
        std::string node_help;
        std::vector<const char*> node_hierarchy;
    };




    class EAGLE_EXPORTS Node: public TInterface<IID_NodeObject, ParameteredIObject>
    {
    public:
        typedef rcc::shared_ptr<Node> Ptr;
        
        Node();
        virtual ~Node();
        /**
         * @brief process Gets called by processing threads and parent nodes.  Process should be left alone because
         * @brief it is where all of the exception handling occurs
         * @param img input image
         * @param steam input stream
         * @return output image
         */
        virtual cv::cuda::GpuMat        process(cv::cuda::GpuMat& img, cv::cuda::Stream& steam = cv::cuda::Stream::Null());
        virtual TS<SyncedMemory>        process(TS<SyncedMemory>& input, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual bool pre_check(const TS<SyncedMemory>& input);
        /**
         * @brief doProcess this is the most used node and where the bulk of the work is performed.
         * @param img input image
         * @param stream input stream
         * @return output image
         */
        virtual cv::cuda::GpuMat        doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual TS<SyncedMemory>        doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream);
        virtual void                    reset();

        virtual Parameters::Parameter* addParameter(Parameters::Parameter::Ptr param);
        virtual Parameters::Parameter* addParameter(Parameters::Parameter* param);
        template<typename T> Parameters::Parameter* updateParameterPtr(const std::string& name, T* data, cv::cuda::Stream* stream = nullptr)
        {
            return ParameteredIObject::updateParameterPtr(name, data, GetTimestamp(), stream);
        }

        template<typename T> Parameters::Parameter* updateParameterPtr(const std::string& name, T* data, long long timestamp, cv::cuda::Stream* stream = nullptr)
        {
            return ParameteredIObject::updateParameterPtr(name, data, timestamp, stream);
        }
        
        template<typename T> Parameters::Parameter* updateParameter(const std::string& name, const T& data, cv::cuda::Stream* stream = nullptr)
        {
            return ParameteredIObject::updateParameter(name, data, GetTimestamp(), stream);
        }
        template<typename T> Parameters::Parameter* updateParameter(const std::string& name, const T& data, long long timestamp, cv::cuda::Stream* stream = nullptr)
        {
            return ParameteredIObject::updateParameter(name, data, timestamp, stream);
        }
        
        template<typename T> Parameters::Parameter* updateParameter(size_t idx, const T data, cv::cuda::Stream* stream = nullptr)
        {
            return ParameteredIObject::updateParameter(idx, data, GetTimestamp(), stream);
        }
        template<typename T> Parameters::Parameter* updateParameter(size_t idx, const T data, long long timestamp, cv::cuda::Stream* stream = nullptr)
        {
            return ParameteredIObject::updateParameter(idx, data, timestamp, stream);
        }
        /**
         * @brief getName depricated?  Idea was to recursively go through parent nodes and rebuild my tree name, useful I guess once
         * @brief node swapping and moving is implemented
         * @return
         */
        std::string                        getName() const;
        std::string                        getTreeName();
        std::string                     getFullTreeName();
        Node *getParent();
        /**
         * @brief getInputs [DEPRICATED]
         */
        virtual void                    getInputs();
        /**
         * @brief log internally used to log node status, warnings, and errors
         * @param level
         * @param msg
         */
        //virtual void                    log(boost::log::trivial::severity_level level, const std::string& msg);

        virtual void updateParent();



        // ****************************************************************************************************************
        //
        //                                    Display functions
        //
        // ****************************************************************************************************************
        // Register a function for displaying CPU images
         virtual void registerDisplayCallback(boost::function<void(cv::Mat, Node*)>& f);
        // Register a function for displaying GPU images
         virtual void registerDisplayCallback(boost::function<void(cv::cuda::GpuMat, Node*)>& f);
        // Spawn an external display just for this node, with name = treeName
         virtual void spawnDisplay();
        // Kill any spawned external displays
         virtual void killDisplay();

        // ****************************************************************************************************************
        //
        //                                    Child adding and deleting
        //
        // ****************************************************************************************************************
        virtual Node::Ptr               addChild(Node* child);
        virtual Node::Ptr               addChild(Node::Ptr child);
        virtual Node::Ptr               getChild(const std::string& treeName);
        virtual Node::Ptr               getChild(const int& index);
        virtual Node*                    getChildRecursive(std::string treeName_);
        virtual void                    removeChild(const std::string& name);
        virtual void                    removeChild(Node::Ptr node);
        virtual void                    removeChild(int idx);
        virtual void                    swapChildren(int idx1, int idx2);
        virtual void                    swapChildren(const std::string& name1, const std::string& name2);
        virtual void                    swapChildren(Node::Ptr child1, Node::Ptr child2);
        virtual std::vector<Node*>        getNodesInScope();
        virtual Node *                    getNodeInScope(const std::string& name);
        virtual void                    getNodesInScope(std::vector<Node*>& nodes);
        virtual void                    SetDataStream(IDataStream* stream);
        virtual IDataStream*            GetDataStream();
        
        // ****************************************************************************************************************
        //
        //                                    Name and accessing functions
        //
        // ****************************************************************************************************************
        /**
         * @brief setTreeName
         * @param name
         */
        virtual void setTreeName(const std::string& name);
        /**
         * @brief setFullTreeName
         * @param name
         */
        virtual void setFullTreeName(const std::string& name);
        /**
         * @brief setParent
         * @param parent
         */
        virtual void setParent(Node *parent);
        


        //******************************************************************************************************************
        //
        //                                    Parameter updating, getting and searching
        //
        
        virtual std::vector<std::string> listInputs();

        virtual std::vector<std::string> listParameters();

        
        void RegisterSignalConnection(std::shared_ptr<Signals::connection> connection);
        

        
        // ****************************************************************************************************************
        //
        //                                    Dynamic reloading and persistence
        //
        // ****************************************************************************************************************

        /**
         * @brief swap is supposed to swap positions of this node and other node.  This would swap it into the same place
         * @brief wrt other's parent, and swap all children.  This would not do anything to the parameters.  Not implemented
         * @brief and tested yet.
         * @param other
         * @return
         */
        virtual Node *swap(Node *other);

        virtual void Init(bool firstInit = true);
        virtual void NodeInit(bool firstInit = true);
        /**
         * @brief Init [DEPRICATED], would be used to load the configuration of this node froma  file
         * @param configFile
         */
        virtual void Init(const std::string& configFile);
        /**
         * @brief Init is used to reload a node based on a saved .yml configuration.
         * @brief base implementation reloads children, parameters and name information.
         * @brief override to extend to include other node specific persistence.
         * @brief Needs to match with the Serialize(cv::FileStorage& fs) function.
         * @param configNode is the fileNode representing the configuration of this node
         */
        virtual void Init(const cv::FileNode& configNode);
        /**
         * @brief Serialize serializes in and out the variables of a node during runtime object swapping.
         * @brief The default implementation handles serializing name parameters, children and parameters.
         * @param pSerializer is the serializer object that stores variables during swap
         */
        virtual void Serialize(ISimpleSerializer *pSerializer);
        /**
         * @brief Serialize [DEPRICATED] initial intention was to use this for persistence
         * @param fs
         */
        virtual void Serialize(cv::FileStorage& fs);
        /**
         * @brief SkipEmpty is used to flag if a node's doProcess function should be called on an empty input img.
         * @brief by default, all nodes just skip processing on an empty input.  For file loading nodes, this isn't desirable
         * @brief so for any nodes that generate an output, they override this function to return false.
         * @return true if this node should skip empty images, false otherwise.
         */
        virtual bool SkipEmpty() const;

        long long GetTimestamp() const;

        std::vector<std::pair<time_t, int>> GetProfileTimings() const;

        // ****************************************************************************************************************
        //
        //                                    Members
        //
        // ****************************************************************************************************************
        // Vector of child nodes
        std::vector<Node::Ptr>                                              children;

        // Constant name that describes the node ie: Sobel
        //std::string                                                            nodeName;
        
        // Vector of parameters for this node
        //std::vector< Parameters::Parameter::Ptr >                            parameters;

        /* True if spawnDisplay has been called, in which case results should be drawn and displayed on a window with the name treeName */
        bool                                                                externalDisplay;
        // Toggling this disables a node's doProcess code from ever being called
        bool                                                                enabled;

        bool                                                                profile;

        
        
        void onParameterAdded();
        double GetProcessingTime() const;
        void Clock(int line_num);
        
    protected:
        IDataStream*                                                             _dataStream;
        long long                                                                _current_timestamp;
        std::shared_ptr<Parameters::IVariableManager>                            _variable_manager;

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
