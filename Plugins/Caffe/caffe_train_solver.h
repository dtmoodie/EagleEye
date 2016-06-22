#pragma once
#include <caffe/solver.hpp>
#include <caffe/parallel.hpp>
#include <EagleLib/nodes/Node.h>
#include <EagleLib/ParameteredIObjectImpl.hpp>

namespace EagleLib
{
    namespace detail
    {
        template<typename T> class caffe_solver: public caffe::P2PSync<T>
    {
    
    };
    }
    

    namespace Nodes
    {
        class PLUGIN_EXPORTS caffe_solver: public Node
        {
        protected:
            std::shared_ptr<EagleLib::detail::caffe_solver<float>> _float_solver;
        public:
            enum LearningPolicies
            {
                fixed,
                step,
                exponential,
                inverse,
                multistep,
                poly,
                sigmoid
            };
            caffe_solver();
            virtual void NodeInit(bool firstInit);
            virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream);
            virtual bool pre_check(const TS<SyncedMemory>& input);
            BEGIN_PARAMS(caffe_solver);
                PARAM(Parameters::ReadFile, solver_description, "");
                PARAM(Parameters::ReadFile, network_description, "");
                PARAM(int, test_interval, 1000);
                PARAM(double, base_learning_rate, 0.001);
                PARAM(double, momentum, 0.9);
                PARAM(double, gamma, 0.1);
                PARAM(int, step_size, 20000);
                PARAM(int, snapshot_interval, 10000);
                PARAM(std::string, snapshot_prefix, "snapshots/");
                PARAM(boost::shared_ptr<caffe::Solver<float> >, solver, nullptr);
                PARAM(boost::shared_ptr<caffe::Net<float>>, neural_network, nullptr);
            END_PARAMS;
        };
        class PLUGIN_EXPORTS caffe_network: public Node
        {
        public:
            caffe_network();
            BEGIN_PARAMS(caffe_network)
                PARAM(boost::shared_ptr<caffe::Net<float>>, neural_network, NULL);
                PARAM(Parameters::ReadFile, nn_description, "");
            END_PARAMS;
            virtual TS<SyncedMemory> doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream);
            virtual bool pre_check(const TS<SyncedMemory>& input);
        };
    }
}