#include "super_resolution.h"
#include <EagleLib/ParameteredObjectImpl.hpp>
using namespace EagleLib;
using namespace EagleLib::Nodes;
my_frame_source::my_frame_source()
{
    current_source = nullptr;
    current_stream = nullptr;
}
void my_frame_source::nextFrame(cv::OutputArray frame)
{
    if(current_source && current_stream)
        frame.getGpuMatRef() = current_source->GetGpuMatMutable(*current_stream);
}
void my_frame_source::reset()
{
    current_source = nullptr;
    current_stream = nullptr;
}
void my_frame_source::input_frame(TS<SyncedMemory>& image, cv::cuda::Stream& stream)
{
    current_source = &image;
    current_stream = &stream;
}

void super_resolution::Init(bool firstInit)
{
    frame_source.reset(new my_frame_source());
    super_res = cv::superres::createSuperResolution_BTVL1_CUDA();
    super_res->setInput(frame_source);
    if(firstInit)
    {
        //auto scale                  = super_res->getScale();
        //auto iterations             = super_res->getIterations();
        auto tau                    = super_res->getTau();
        auto lambda                 = super_res->getLabmda();
        auto alpha                  = super_res->getAlpha();
        auto kernel_size            = super_res->getKernelSize();
        auto gaussian_kernel_size   = super_res->getBlurKernelSize();
        auto blur_sigma             = super_res->getBlurSigma();
        auto temporal_area_radius   = super_res->getTemporalAreaRadius();
        super_res->setScale(2);
        super_res->setIterations(50);
        updateParameter("scale", 2);                                    // 0
        updateParameter("iterations", 50);                              // 1
        updateParameter("tau", tau);                                    // 2
        updateParameter("lambda", lambda);                              // 3
        updateParameter("Alpha", alpha);                                // 4
        updateParameter("kernel size", kernel_size);                    // 5
        updateParameter("blur kernel size", gaussian_kernel_size);      // 6
        updateParameter("blur sigma", blur_sigma);                      // 7
        updateParameter("Temporal area radius", temporal_area_radius);  // 8

    }else
    {
        for(int i = 0; i < _parameters.size(); ++i)
        {
            _parameters[i]->changed = true;
        }
    }
}
void super_resolution::doProcess(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    if(_parameters[0]->changed)
    {
        super_res->setScale(*getParameter<int>(0)->Data()); _parameters[0]->changed = false;
    }
    if(_parameters[1]->changed)
    {
        super_res->setIterations(*getParameter<int>(1)->Data()); _parameters[1]->changed = false;
    }
    if(_parameters[2]->changed)
    {
        super_res->setTau(*getParameter<double>(2)->Data()); _parameters[2]->changed = false;
    }
    if (_parameters[3]->changed)
    {
        super_res->setLabmda(*getParameter<double>(3)->Data()); _parameters[3]->changed = false;
    }
    if (_parameters[4]->changed)
    {
        super_res->setAlpha(*getParameter<double>(4)->Data()); _parameters[4]->changed = false;
    }
    if (_parameters[5]->changed)
    {
        super_res->setKernelSize(*getParameter<int>(5)->Data()); _parameters[5]->changed = false;
    }
    if (_parameters[6]->changed)
    {
        super_res->setBlurKernelSize(*getParameter<int>(6)->Data()); _parameters[6]->changed = false;
    }
    if (_parameters[7]->changed)
    {
        super_res->setBlurSigma(*getParameter<double>(7)->Data()); _parameters[7]->changed = false;
    }
    if (_parameters[8]->changed)
    {
        super_res->setTemporalAreaRadius(*getParameter<int>(8)->Data()); _parameters[8]->changed = false;
    }
    frame_source->input_frame(input, stream);
    cv::cuda::GpuMat result;
    super_res->nextFrame(result);
    updateParameter("super resolution", result)->type = Parameters::Parameter::Output;
    
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(super_resolution, Image, Processing)
