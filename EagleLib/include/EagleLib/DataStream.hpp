#pragma once
#include "IDataStream.hpp"

#include <MetaObject/MetaObject.hpp>
#include <MetaObject/Thread/Thread.hpp>
#include <MetaObject/Thread/ThreadHandle.hpp>
#include <EagleLib/utilities/UiCallbackHandlers.h>
#include <boost/thread.hpp>
#include <opencv2/core/cuda.hpp>


#define DS_END_(N) \
SIGNAL_INFO_END(N) \
SLOT_INFO_END(N) \
PARAMETER_INFO_END(N) \
SIGNALS_END(N) \
SLOT_END(N) \
void InitParameters(bool firstInit) \
{ \
    init_parameters_(firstInit, mo::_counter_<N - 1>()); \
    _init_parent_params(firstInit); \
} \
void SerializeParameters(ISimpleSerializer* pSerializer) \
{ \
    _serialize_parameters(pSerializer, mo::_counter_<N - 1>()); \
    _serialize_parent_params(pSerializer); \
} \
static const int _DS_N_ = N

namespace EagleLib
{
    class EAGLE_EXPORTS DataStream : public IDataStream
    {
    public:
        DataStream();
        virtual ~DataStream();
        MO_BEGIN(DataStream)
            MO_SIGNAL(void, StartThreads)
            MO_SIGNAL(void, StopThreads)

            MO_SLOT(void, StartThread)
            MO_SLOT(void, StopThread)
            MO_SLOT(void, PauseThread)
            MO_SLOT(void, ResumeThread)
            MO_SLOT(void, node_updated, Nodes::Node*)
            MO_SLOT(void, update)
            MO_SLOT(void, parameter_updated, mo::IMetaObject*, mo::IParameter*)
            MO_SLOT(void, parameter_added, mo::IMetaObject*, mo::IParameter*)
            MO_SLOT(void, run_continuously, bool)
            MO_SLOT(int, process)
        DS_END_(__COUNTER__);

        std::vector<rcc::weak_ptr<EagleLib::Nodes::Node>> GetTopLevelNodes();
        virtual mo::Context*                              GetContext() const;
        virtual void                                      InitCustom(bool firstInit);
        virtual rcc::weak_ptr<IViewManager>               GetViewManager();
        virtual rcc::weak_ptr<ICoordinateManager>         GetCoordinateManager();
        virtual rcc::weak_ptr<IRenderEngine>              GetRenderingEngine();
        virtual rcc::weak_ptr<ITrackManager>              GetTrackManager();
        virtual std::shared_ptr<mo::IVariableManager>     GetVariableManager();
        virtual mo::RelayManager*                         GetRelayManager();
        virtual IParameterBuffer*                         GetParameterBuffer();
        virtual rcc::weak_ptr<WindowCallbackHandler>    GetWindowCallbackManager();
        virtual std::vector<rcc::shared_ptr<Nodes::Node>> GetNodes() const;
        virtual std::vector<rcc::shared_ptr<Nodes::Node>> GetAllNodes() const;
        virtual bool                                      LoadDocument(const std::string& document, const std::string& prefered_loader = "");
        virtual std::vector<rcc::shared_ptr<Nodes::Node>> AddNode(const std::string& nodeName);
        virtual void                                      AddNode(rcc::shared_ptr<Nodes::Node> node);
        virtual void                                      AddNodeNoInit(rcc::shared_ptr<Nodes::Node> node);
        virtual void                                      AddNodes(std::vector<rcc::shared_ptr<Nodes::Node>> node);
        virtual void                                      RemoveNode(rcc::shared_ptr<Nodes::Node> node);
        virtual void                                      RemoveNode(Nodes::Node* node);
        virtual Nodes::Node*                              GetNode(const std::string& nodeName);
        virtual bool                                      SaveStream(const std::string& filename);
        virtual bool                                      LoadStream(const std::string& filename);
        template<class T> void                            load(T& ar);
        template<class T> void                            save(T& ar) const;

        void AddVariableSink(IVariableSink* sink);
        void RemoveVariableSink(IVariableSink* sink);
    protected:
        friend class IDataStream;
        virtual void AddChildNode(rcc::shared_ptr<Nodes::Node> node);
        virtual void RemoveChildNode(rcc::shared_ptr<Nodes::Node> node);
        virtual std::unique_ptr<ISingleton>& GetSingleton(mo::TypeInfo type);
        virtual std::unique_ptr<ISingleton>& GetIObjectSingleton(mo::TypeInfo type);

        std::map<mo::TypeInfo, std::unique_ptr<ISingleton>>       _singletons;
        std::map<mo::TypeInfo, std::unique_ptr<ISingleton>>       _iobject_singletons;
        int                                                       stream_id;
        size_t                                                    _thread_id;
        rcc::shared_ptr<IViewManager>                             view_manager;
        rcc::shared_ptr<ICoordinateManager>                       coordinate_manager;
        rcc::shared_ptr<IRenderEngine>                            rendering_engine;
        rcc::shared_ptr<ITrackManager>                            track_manager;
        std::shared_ptr<mo::IVariableManager>                     variable_manager;
        std::shared_ptr<mo::RelayManager>                         relay_manager;
        std::shared_ptr<IParameterBuffer>                         _parameter_buffer;
        std::mutex                                                nodes_mtx;
        mo::ThreadHandle                                          _processing_thread;
        volatile bool                                             dirty_flag;
        std::vector<IVariableSink*>                               variable_sinks;
        // These are threads for attempted connections
        std::vector<boost::thread*>                               connection_threads;
        std::vector<rcc::shared_ptr<Nodes::Node>>                 top_level_nodes;
        std::vector<rcc::weak_ptr<Nodes::Node>>                   child_nodes;
        rcc::shared_ptr<WindowCallbackHandler>                    _window_callback_handler;
        unsigned int                                              _rmt_hash;
        unsigned int                                              _rmt_cuda_hash;
    };
}
