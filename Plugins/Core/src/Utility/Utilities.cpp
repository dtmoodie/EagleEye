#include "Utilities.h"
#include <boost/lexical_cast.hpp>
#include <EagleLib/Nodes/NodeInfo.hpp>
using namespace EagleLib;
using namespace EagleLib::Nodes;

/*


bool functionQualifier(Parameters::Parameter* parameter)
{
    if (parameter->GetTypeInfo() == mo::TypeInfo(typeid(boost::function<void(void)>)))
    {
        if (parameter->type & Parameters::Parameter::Output || parameter->type & Parameters::Parameter::Control)
            return true;

    }
    return false;
}

void SyncFunctionCall::NodeInit(bool firstInit)
{
    updateParameter<boost::function<void(void)>>("Call all input functions", boost::bind(&SyncFunctionCall::call, this));
    if(firstInit)
    {
        addInputParameter<boost::function<void(void)>>("Input 1")->SetQualifier(boost::bind(&functionQualifier, _1));
    }
}

void SyncFunctionCall::call()
{
    for(int i = 1; i < _parameters.size(); ++i)
    {
        auto param = dynamic_cast<Parameters::ITypedParameter<boost::function<void(void)>>*>(_parameters[i]);
        if(param)
        {
            if(param->Data() != nullptr)
                (*param->Data())();
        }
    }
}

cv::cuda::GpuMat SyncFunctionCall::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    bool full = true;
    for(int i = 1; i < _parameters.size(); ++i)
    {
        auto param = dynamic_cast<Parameters::ITypedParameter<boost::function<void(void)>>*>(_parameters[i]);
        
        if(param)
        {
            if(param->Data() == nullptr)
                full = false;
        }
    }
    if(full == true)
    {
        addInputParameter<boost::function<void(void)>>("Input " + boost::lexical_cast<std::string>(_parameters.size()))->SetQualifier(boost::bind(&functionQualifier, _1));
    }
    return img;
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(SyncFunctionCall, Utility)
*/

bool RegionOfInterest::ProcessImpl()
{
    if(roi.area())
    {
        auto img_roi = cv::Rect(cv::Point(0,0), image->GetSize());
        auto used_roi = img_roi & roi;

        std::vector<cv::Mat> h_mats;
        std::vector<cv::cuda::GpuMat> d_mats;
        const auto& d_inputs = image->GetGpuMatVec(Stream());
        const auto& h_inputs = image->GetMatVec(Stream());
        for(int i = 0; i < d_inputs.size(); ++i)
        {
            d_mats.push_back(d_inputs[i](used_roi));
            h_mats.push_back(h_inputs[i](used_roi));
        }
        ROI_param.UpdateData(SyncedMemory(h_mats, d_mats), image_param.GetTimestamp(), _ctx);
        return true;
    }
    return false;
}
MO_REGISTER_CLASS(RegionOfInterest);
void ExportRegionsOfInterest::NodeInit(bool firstInit)
{
    output.SetMtx(_mtx);
    output.UpdatePtr(&rois);
    output.SetContext(_ctx);
    output.SetName("output");
    output.SetFlags(mo::ParameterType::Output_e);
    AddParameter(&output);
}

bool ExportRegionsOfInterest::ProcessImpl()
{
    return true;
}
MO_REGISTER_CLASS(ExportRegionsOfInterest)
