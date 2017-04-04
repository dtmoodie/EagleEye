#include "Utilities.h"
#include <boost/lexical_cast.hpp>
#include <Aquila/Nodes/NodeInfo.hpp>
using namespace aq;
using namespace aq::Nodes;

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
        //auto img_roi = cv::Rect2f(cv::Point2f(0.0,0.0), image->GetSize());
        auto img_roi = cv::Rect2f(0.0f, 0.0f, 1.0f, 1.0f);
        auto used_roi = img_roi & roi;
        //cv::Rect2f img_size(cv::Point2f(0.0f, 0.0f), image->GetSize());
        auto img_size = image->GetSize();
        cv::Rect pixel_roi;
        pixel_roi.x = used_roi.x * img_size.width;
        pixel_roi.y = used_roi.y * img_size.height;
        pixel_roi.width = used_roi.width * img_size.width;
        pixel_roi.height = used_roi.height * img_size.height;
        pixel_roi = pixel_roi & cv::Rect(cv::Point(), img_size);


        std::vector<cv::Mat> h_mats;
        std::vector<cv::cuda::GpuMat> d_mats;
        std::vector<SyncedMemory::SYNC_STATE> state;
        const int num = image->GetNumMats();
        h_mats.resize(num);
        d_mats.resize(num);
        state.resize(num);
        for(int i = 0; i < num; ++i)
        {
            state[i] = image->GetSyncState(i);
            if(state[i] == SyncedMemory::HOST_UPDATED)
            {
                // host is ahead
                h_mats[i] = image->GetMat(Stream(), i)(pixel_roi);
            }else if(state[i] == SyncedMemory::SYNCED)
            {
                h_mats[i] = image->GetMat(Stream(), i)(pixel_roi);
                d_mats[i] = image->GetGpuMat(Stream(), i)(pixel_roi);
            }else
            {
                // device is ahead
                d_mats[i] = image->GetGpuMat(Stream(), i)(pixel_roi);
            }
        }
        ROI_param.UpdateData(SyncedMemory(h_mats, d_mats, state), image_param.GetTimestamp(), _ctx);
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
    output.AppendFlags(mo::Unstamped_e);
    AddParameter(&output);
}

bool ExportRegionsOfInterest::ProcessImpl()
{
    return true;
}
MO_REGISTER_CLASS(ExportRegionsOfInterest)
