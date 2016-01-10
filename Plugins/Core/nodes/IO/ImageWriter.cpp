#include "nodes/IO/ImageWriter.h"
#include <external_includes/cv_imgcodec.hpp>
#include "../remotery/lib/Remotery.h"

using namespace EagleLib;

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
	updateParameter<Parameters::WriteDirectory>("Save Diretory", Parameters::WriteDirectory(""));
}

cv::cuda::GpuMat ImageWriter::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(parameters[0]->changed)
    {
		std::string& tmp = *getParameter<std::string>(0)->Data();
		if (tmp.size())
			baseName = tmp;
		else
		{
			NODE_LOG(warning) << "Empty base name passed in";
		}
    }
    if(parameters[1]->changed || extension.size() == 0)
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

NODE_DEFAULT_CONSTRUCTOR_IMPL(ImageWriter, Image, Sink)
