#include "caffe_train_solver.h"
#include <boost/filesystem.hpp>
#include <caffe/util/upgrade_proto.hpp>
#ifdef _MSC_VER
  #ifdef _DEBUG
	RUNTIME_COMPILER_LINKLIBRARY("libcaffe-d.lib");
  #else
    RUNTIME_COMPILER_LINKLIBRARY("libcaffe.lib");
    RUNTIME_COMPILER_LINKLIBRARY("proto.lib");
    RUNTIME_COMPILER_LINKLIBRARY("libprotobuf.lib");
  #endif
#else
#endif

using namespace EagleLib;
using namespace EagleLib::Nodes;
caffe_solver::caffe_solver():
    Node()
{

}
void caffe_solver::NodeInit(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<boost::shared_ptr<caffe::Net<float>>>("network");
    }
    Parameters::EnumParameter solver_type;
    solver_type.addEnum(ENUM(caffe::SolverParameter::ADADELTA));
    solver_type.addEnum(ENUM(caffe::SolverParameter::ADAGRAD));
    solver_type.addEnum(ENUM(caffe::SolverParameter::ADAM));
    solver_type.addEnum(ENUM(caffe::SolverParameter::SGD));
    solver_type.addEnum(ENUM(caffe::SolverParameter::NESTEROV));
    solver_type.addEnum(ENUM(caffe::SolverParameter::RMSPROP));
    solver_type.currentSelection = 3;
    updateParameter("solver type", solver_type);
    Parameters::EnumParameter lr_policy;
    lr_policy.addEnum(ENUM(fixed));
    lr_policy.addEnum(ENUM(step));
    lr_policy.addEnum(ENUM(exponential));
    lr_policy.addEnum(ENUM(inverse));
    lr_policy.addEnum(ENUM(multistep));
    lr_policy.addEnum(ENUM(poly));
    lr_policy.addEnum(ENUM(sigmoid));
    lr_policy.currentSelection = 1;
    updateParameter("learning rate policy", lr_policy);
}

TS<SyncedMemory> caffe_solver::doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    if(solver_description_param.changed && solver_description.size())
    {
        if(boost::filesystem::is_regular_file(solver_description))
        {
            caffe::SolverParameter solver_params;
            caffe::ReadSolverParamsFromTextFileOrDie(solver_description.string(), &solver_params);
            if(snapshot_prefix.size())
            {
                //solver_params.clear_snapshot_prefix();
                //*solver_params.mutable_snapshot_prefix() = snapshot_prefix;
            }
            
            solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_params));
            solver_param.type = Parameters::Parameter::Output;
            if(solver_params.has_test_interval())
                test_interval = solver_params.test_interval();
            if(solver_params.has_base_lr())
                base_learning_rate = solver_params.base_lr();
            if(solver_params.has_gamma())
                gamma = solver_params.gamma();
            neural_network = solver->net();
            auto input_blobs = neural_network->input_blobs();
            updateParameter("input blobs", input_blobs);
        }
        solver_description_param.changed = false;
    }
    if(!solver)
    {
        caffe::SolverParameter solver_params;
        
    }
    return input;
}
bool caffe_solver::pre_check(const TS<SyncedMemory>& input)
{
    return true;
}
caffe_network::caffe_network():
    Nodes::Node()
{

}
TS<SyncedMemory> Nodes::caffe_network::doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    // asdf
    if(nn_description_param.changed && nn_description.size())
    {
        if(boost::filesystem::is_regular_file(nn_description))
        {
            neural_network.reset(new caffe::Net<float>(nn_description.string(), caffe::TRAIN));
            neural_network_param.type = Parameters::Parameter::Output;
            auto inputs = neural_network->input_blobs();
            updateParameter("input blobs", inputs)->type = Parameters::Parameter::Output;
            nn_description_param.changed = false;
        }        
    }
    return input;
}
bool caffe_network::pre_check(const TS<SyncedMemory>& input)
{
    return true;
}

static EagleLib::Nodes::NodeInfo g_registerer_caffe_solver("caffe_solver", { "caffe", "training" });
static EagleLib::Nodes::NodeInfo g_registerer_caffe_network("caffe_network", { "caffe", "training" });
REGISTERCLASS(caffe_solver, &g_registerer_caffe_solver)
REGISTERCLASS(caffe_network, &g_registerer_caffe_network)