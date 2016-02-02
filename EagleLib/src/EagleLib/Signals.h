#pragma once
#include "EagleLib/Defs.hpp"
#include "Signals/signal_manager.h"
namespace EagleLib
{
    // Class that can send and receive signals
    
    

    // Basically reimplements some of the stuff from Signals::signal_manager, but with
    // access restrictions and stream segregation
    class EAGLE_EXPORTS SignalManager: public Signals::signal_manager
    {
    public:
        template<typename T, typename C> std::shared_ptr<Signals::connection> Connect(const std::string& name, const std::function<T>& f, C* receiver, int stream_index = -1, boost::thread::id dest_thread = boost::this_thread::get_id())
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto&sig = GetSignal(name, Loki::TypeInfo(typeid(T)), stream_index);
            Signals::signal<T>* typedSignal = nullptr;
            if (!sig)
            {
                typedSignal = new Signals::signal<T>();
                sig.reset(typedSignal);
            }else
            {
                typedSignal = std::dynamic_pointer_cast<Signals::signal<T>>(sig).get();
            }
            ConnectionInfo cinfo;
            cinfo.ptr = (void*)receiver;
            cinfo.sender = false;
            cinfo.signal_name = name;
            cinfo.object_type = Loki::TypeInfo(typeid(C));
            cinfo.signal_signature = Loki::TypeInfo(typeid(T));
            connections.push_back(cinfo);
            return typedSignal->connect(f, dest_thread);
        }


        template<typename T, typename C> Signals::signal<T>* GetSignal(const std::string& name, C* sender, int stream_index = -1)
        {
            std::lock_guard<std::mutex> lock(mtx);

            auto&sig = GetSignal(name, Loki::TypeInfo(typeid(T)), stream_index);
            if (!sig)
                sig.reset(new Signals::signal<T>());

            ConnectionInfo cinfo;
            cinfo.sender = true;
            cinfo.ptr = (void*)sender;
            cinfo.signal_name = name;
            cinfo.object_type = Loki::TypeInfo(typeid(C));
            cinfo.signal_signature = Loki::TypeInfo(typeid(T));
            connections.push_back(cinfo);
            return std::dynamic_pointer_cast<Signals::signal<T>>(sig).get();
        }

    private:
        struct ConnectionInfo
        {
            bool sender; // True if sender, false if receiver
            Loki::TypeInfo object_type;
            Loki::TypeInfo signal_signature;
            void* ptr;
            std::string signal_name;
        };
        std::shared_ptr<Signals::signal_base>& GetSignal(const std::string& name, Loki::TypeInfo type, int stream_index);
        // The signal manager object stores signals by the stream index, signature, and name
        std::map<int,  // Stream index
        std::map<Loki::TypeInfo, // Signal signature
        std::map<std::string, // Signal name
            std::shared_ptr<Signals::signal_base> >>> stream_specific_signals;
        
        /*std::map<Loki::TypeInfo, // Signal signature
        std::map<std::string, // Signal name
            std::shared_ptr<Signals::signal_base> >> global_signals;*/

        std::list<ConnectionInfo> connections;
        std::mutex mtx;
    
    };
}