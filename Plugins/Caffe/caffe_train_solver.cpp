#include "caffe_train_solver.h"
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <caffe/util/upgrade_proto.hpp>
#include <EagleLib/Nodes/NodeInfo.hpp>
#include <algorithm>
#include "Caffe.h"
#include "caffe_include.h"
#include "caffe_init.h"
using namespace EagleLib;
using namespace EagleLib::Nodes;


void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
    std::vector<std::string> model_names;
    boost::split(model_names, model_list, boost::is_any_of(","));
    for (int i = 0; i < model_names.size(); ++i) {
        LOG(info) << "Finetuning from " << model_names[i];
        solver->net()->CopyTrainedLayersFrom(model_names[i]);
        for (int j = 0; j < solver->test_nets().size(); ++j) {
            solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
        }
    }
}
bool caffe_solver::ProcessImpl()
{
    if(::caffe::Caffe::mode() != ::caffe::Caffe::GPU)
        ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
    if(solver_description_param.modified && solver_description.size())
    {
        if(boost::filesystem::is_regular_file(solver_description))
        {
            caffe::SolverParameter solver_params;
            try
            {
                caffe::ReadSolverParamsFromTextFileOrDie(solver_description.string(), &solver_params);
                
                if(snapshot_prefix.size())
                {
                    //solver_params.clear_snapshot_prefix();
                    //*solver_params.mutable_snapshot_prefix() = snapshot_prefix;
                }
                solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_params));
            }catch(caffe::ExceptionWithCallStack<std::string>& e)
            {
                throw mo::ExceptionWithCallStack<std::string>(std::string(e), e.CallStack());
            }
            catch(caffe::IExceptionWithCallStackBase& e)
            {
                throw mo::ExceptionWithCallStack<std::string>(std::string(""), e.CallStack());
            }
            
            
            if(solver_params.has_test_interval())
                test_interval = solver_params.test_interval();
            if(solver_params.has_base_lr())
                base_learning_rate = solver_params.base_lr();
            if(solver_params.has_gamma())
                gamma = solver_params.gamma();
            neural_network = solver->net();
            auto idx = neural_network->input_blob_indices();
            auto names = neural_network->blob_names();
            std::vector<std::string> input_names;
            for(auto i : idx)
            {
                input_names.push_back(names[i]);
            }
            auto input_blobs_ = neural_network->input_blobs();
            // map input blobs
            
            for(int i = 0; i < input_blobs_.size(); ++i)
            {
                auto wrapped_blob = EagleLib::Nodes::CaffeBase::WrapBlob(*input_blobs_[i]);
                input_blobs[input_names[i]] = wrapped_blob;
            }
            input_blobs_param.Commit();
        }
        if(weight_files.size())
        {
            std::stringstream ss;
            for(int i = 0; i < weight_files.size(); ++i)
            {
                if(boost::filesystem::is_regular_file(weight_files[i]))
                {
                    if(i != 0)
                        ss << ",";
                    ss << weight_files[i];
                }                
            }
            //CopyLayers(solver.get(), ss.str());
        }
        if(previous_solver_state.size())
        {
            if(boost::filesystem::is_regular_file(previous_solver_state))
                solver->Restore(previous_solver_state.string().c_str());
        }
        solver_description_param.modified = false;
    }
    if(!solver)
    {
        caffe::SolverParameter solver_params;
    }
    if(solver && (input_blobs_param.HasSubscriptions() || input_blobs.empty()))
    {
        sig_fill_blobs();
        
        solver->Step(steps_per_iteration);
        if(input_blobs.empty())
            sig_update();
        return true;
    }
    return false;
}

/*caffe_network::caffe_network():
    Nodes::Node()
{

}
TS<SyncedMemory> Nodes::caffe_network::doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    // asdf
    if(nn_description_param.modified && nn_description.size())
    {
        if(boost::filesystem::is_regular_file(nn_description))
        {
            neural_network.reset(new caffe::Net<float>(nn_description.string(), caffe::TRAIN));
            
            auto inputs = neural_network->input_blobs();
            //updateParameter("input blobs", inputs)->type = Parameters::Parameter::Output;
            nn_description_param.modified = false;
        }        
    }
    return input;
}
*/








MO_REGISTER_CLASS(caffe_solver);
//REGISTERCLASS(caffe_network, &g_registerer_caffe_network);

