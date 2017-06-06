#include "Utilities.h"
#include <Aquila/nodes/NodeInfo.hpp>
#include <boost/lexical_cast.hpp>
using namespace aq;
using namespace aq::Nodes;

/*


bool functionQualifier(Parameters::Parameter* parameter)
{
    if (parameter->getTypeInfo() == mo::TypeInfo(typeid(boost::function<void(void)>)))
    {
        if (parameter->type & Parameters::Parameter::Output || parameter->type & Parameters::Parameter::Control)
            return true;

    }
    return false;
}

void SyncFunctionCall::nodeInit(bool firstInit)
{
    updateParameter<boost::function<void(void)>>("Call all input functions", boost::bind(&SyncFunctionCall::call, this));
    if(firstInit)
    {
        addInputParam<boost::function<void(void)>>("Input 1")->SetQualifier(boost::bind(&functionQualifier, _1));
    }
}

void SyncFunctionCall::call()
{
    for(int i = 1; i < _parameters.size(); ++i)
    {
        auto param = dynamic_cast<Parameters::ITParam<boost::function<void(void)>>*>(_parameters[i]);
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
        auto param = dynamic_cast<Parameters::ITParam<boost::function<void(void)>>*>(_parameters[i]);

        if(param)
        {
            if(param->Data() == nullptr)
                full = false;
        }
    }
    if(full == true)
    {
        addInputParam<boost::function<void(void)>>("Input " + boost::lexical_cast<std::string>(_parameters.size()))->SetQualifier(boost::bind(&functionQualifier, _1));
    }
    return img;
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(SyncFunctionCall, Utility)
*/

bool RegionOfInterest::processImpl() {
    if (roi.area()) {
        //auto img_roi = cv::Rect2f(cv::Point2f(0.0,0.0), image->getSize());
        auto img_roi  = cv::Rect2f(0.0f, 0.0f, 1.0f, 1.0f);
        auto used_roi = img_roi & roi;
        //cv::Rect2f img_size(cv::Point2f(0.0f, 0.0f), image->getSize());
        auto     img_size = image->getSize();
        cv::Rect pixel_roi;
        pixel_roi.x      = used_roi.x * img_size.width;
        pixel_roi.y      = used_roi.y * img_size.height;
        pixel_roi.width  = used_roi.width * img_size.width;
        pixel_roi.height = used_roi.height * img_size.height;
        pixel_roi        = pixel_roi & cv::Rect(cv::Point(), img_size);

        std::vector<cv::Mat>                  h_mats;
        std::vector<cv::cuda::GpuMat>         d_mats;
        std::vector<SyncedMemory::SYNC_STATE> state;
        const int                             num = image->getNumMats();
        h_mats.resize(num);
        d_mats.resize(num);
        state.resize(num);
        for (int i = 0; i < num; ++i) {
            state[i] = image->getSyncState(i);
            if (state[i] == SyncedMemory::HOST_UPDATED) {
                // host is ahead
                h_mats[i] = image->getMat(stream(), i)(pixel_roi);
            } else if (state[i] == SyncedMemory::SYNCED) {
                h_mats[i] = image->getMat(stream(), i)(pixel_roi);
                d_mats[i] = image->getGpuMat(stream(), i)(pixel_roi);
            } else {
                // device is ahead
                d_mats[i] = image->getGpuMat(stream(), i)(pixel_roi);
            }
        }
        ROI_param.updateData(SyncedMemory(h_mats, d_mats, state), image_param.getTimestamp(), _ctx.get(), image_param.getCoordinateSystem(), image_param.getFrameNumber());
        return true;
    }
    return false;
}
MO_REGISTER_CLASS(RegionOfInterest);

void ExportRegionsOfInterest::nodeInit(bool firstInit) {
    output.setMtx(_mtx);
    output.updatePtr(&rois);
    output.setContext(_ctx.get());
    output.setName("output");
    output.setFlags(mo::ParamFlags::Output_e);
    output.appendFlags(mo::Unstamped_e);
    addParameter(&output);
}

bool ExportRegionsOfInterest::processImpl() {
    return true;
}
MO_REGISTER_CLASS(ExportRegionsOfInterest)
