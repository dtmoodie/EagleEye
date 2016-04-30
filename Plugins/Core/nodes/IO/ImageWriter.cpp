#include "nodes/IO/ImageWriter.h"
#include <EagleLib/rcc/external_includes/cv_imgcodec.hpp>
#include "../remotery/lib/Remotery.h"
#include "EagleLib/utilities/CudaCallbacks.hpp"
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaarithm.hpp>
#include <parameters/ParameteredObjectImpl.hpp>
#include <boost/lexical_cast.hpp>
using namespace EagleLib;
using namespace EagleLib::Nodes;

void ImageWriterCallback(int status, void* data)
{
    ImageWriter* node = static_cast<ImageWriter*>(data);
    node->writeImage();
}

void ImageWriter::requestWrite()
{
    writeRequested = true;
}

void ImageWriter::writeImage()
{
	rmt_ScopedCPUSample(ImageWriter_writeImage);
    try
    {
        cv::imwrite(baseName +"-"+ boost::lexical_cast<std::string>(frameCount) + extension, h_buf);
    }catch(cv::Exception &e)
    {
        //log(Error, e.what());
		NODE_LOG(error) << e.what();
        return;
    }
    ++frameCount;
}

void ImageWriter::Init(bool firstInit)
{
    writeRequested = false;
    frameCount = 0;
    frameSkip = 0;
    baseName = "Image";
	Parameters::EnumParameter param;
    param.addEnum(ENUM(jpg));
    param.addEnum(ENUM(png));
    param.addEnum(ENUM(tiff));
    param.addEnum(ENUM(bmp));

    updateParameter<std::string>("Base name", "Image");
    updateParameter("Extension", param);
    updateParameter("Frequency", -1);
    updateParameter<boost::function<void(void)>>("Save image", boost::bind(&ImageWriter::requestWrite, this))->type = Parameters::Parameter::Output;
	updateParameter<Parameters::WriteDirectory>("Save Directory", Parameters::WriteDirectory("F:/temp"));
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Input image device");
        addInputParameter<cv::Mat>("Input image host");
    }
}

cv::cuda::GpuMat ImageWriter::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(_parameters[0]->changed)
    {
		std::string& tmp = *getParameter<std::string>(0)->Data();
		if (tmp.size())
			baseName = tmp;
		else
		{
			NODE_LOG(warning) << "Empty base name passed in";
		}
    }
    if(_parameters[1]->changed || extension.size() == 0)
    {
		Extensions ext = (Extensions)getParameter<Parameters::EnumParameter>(1)->Data()->getValue();
        switch (ext)
        {
        case jpg:
            extension = ".jpg";
            break;
        case png:
            extension = ".png";
            break;
        case tiff:
            extension = ".tif";
            break;
        case bmp:
            extension = ".bmp";
            break;
        default:
            extension = ".jpg";
            break;
        }
    }
	int freq = *getParameter<int>(2)->Data();
    if((writeRequested || (frameSkip >= freq && freq != -1)) && baseName.size() && extension.size())
    {
        img.download(h_buf, stream);
        stream.enqueueHostCallback(ImageWriterCallback, this);
        writeRequested = false;
    }
    ++frameSkip;
    return img; 
}
void ImageWriter::doProcess(TS<SyncedMemory> &img, cv::cuda::Stream &stream)
{
    std::string dir = getParameter<Parameters::WriteDirectory>("Save Directory")->Data()->string();
    if(dir.empty())
        dir = ".";
    if (_parameters[0]->changed)
    {
        std::string& tmp = *getParameter<std::string>(0)->Data();
        if (tmp.size())
            baseName = tmp;
        else
        {
            NODE_LOG(warning) << "Empty base name passed in";
        }
    }
    if (_parameters[1]->changed || extension.size() == 0)
    {
        Extensions ext = (Extensions)getParameter<Parameters::EnumParameter>(1)->Data()->getValue();
        switch (ext)
        {
        case jpg:
            extension = ".jpg";
            break;
        case png:
            extension = ".png";
            break;
        case tiff:
            extension = ".tif";
            break;
        case bmp:
            extension = ".bmp";
            break;
        default:
            extension = ".jpg";
            break;
        }
    }
    cv::Mat mat;
    if(auto gpu_input_ptr = getParameter<cv::cuda::GpuMat>("Input image device")->Data())
    {
        //if(gpu_input_ptr->depth() != CV_8U)
        if(gpu_input_ptr->depth() != CV_8UC1 && gpu_input_ptr->channels() == 1)
        {
            cv::cuda::GpuMat normalized;
            cv::cuda::normalize(*gpu_input_ptr,normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::noArray(), stream);
            normalized.download(mat, stream);
        }else
        {
            gpu_input_ptr->download(mat, stream);
        }
            
        
    }else
    {
        if(auto cpu_input_ptr = getParameter<cv::Mat>("Input image host")->Data())
        {
            mat = *cpu_input_ptr;
        }else
        {
            mat = img.GetMat(stream);
        }
    }
    
    int freq = *getParameter<int>(2)->Data();
    if ((writeRequested || (frameSkip >= freq) || freq == -1) && baseName.size() && extension.size())
    {
        cuda::enqueue_callback_async([mat, this, dir]()->void
        {
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(4) << frameCount++;
            cv::imwrite( dir + "/" + baseName + "-" + ss.str() + extension, mat);
        }, stream);
    }
    ++frameSkip;
}
NODE_DEFAULT_CONSTRUCTOR_IMPL(ImageWriter, Image, Sink)
