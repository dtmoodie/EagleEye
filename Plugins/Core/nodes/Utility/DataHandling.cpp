#include "nodes/Utility/DataHandling.h"
#include "Remotery.h"
#include <parameters/ParameteredObjectImpl.hpp>
#include <boost/lexical_cast.hpp>
using namespace EagleLib;
using namespace EagleLib::Nodes;

void GetOutputImage::Init(bool firstInit)
{
    if(firstInit)
        addInputParameter<cv::cuda::GpuMat>("Input");
}

cv::cuda::GpuMat GetOutputImage::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* input = getParameter<cv::cuda::GpuMat>("Input")->Data();
    if(input == nullptr)
    {
        //log(Status, "Input not defined");
		NODE_LOG(info) << "Input not defined";
        return img;
    }
    if(input->empty())
    {
        //log(Status, "Input is empty");
		NODE_LOG(info) << "Input is empty";
        return img;
    }
    return *input;
}
cv::cuda::GpuMat ExportInputImage::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
	{
		rmt_ScopedCPUSample(ExportInputImage_updateParameter);
		updateParameter("Output image", img, &stream);
	}
    
    return img;
}

void ExportInputImage::Init(bool firstInit)
{
    updateParameter("Output image", cv::cuda::GpuMat())->type = Parameters::Parameter::Output;
}

void ImageInfo::Init(bool firstInit)
{
	Parameters::EnumParameter dataType;
    dataType.addEnum(ENUM(CV_8U));
    dataType.addEnum(ENUM(CV_8S));
    dataType.addEnum(ENUM(CV_16U));
    dataType.addEnum(ENUM(CV_16S));
    dataType.addEnum(ENUM(CV_32S));
    dataType.addEnum(ENUM(CV_32F));
    dataType.addEnum(ENUM(CV_64F));
    updateParameter<Parameters::EnumParameter>("Type",dataType)->type = Parameters::Parameter::State;
}
cv::cuda::GpuMat ImageInfo::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{

    auto param = getParameter<Parameters::EnumParameter>(0);
    if(param->Data()->currentSelection != img.type())
    {
        param->Data()->currentSelection = img.type();
        _parameters[0]->changed = true;
        
    }
    //std::stringstream str;
    //str << "[" << img.cols << "x" << img.rows << "x" << img.channels() << "]" << " " << img.depth();
    //log(Status, str.str());
	NODE_LOG(info) << "[" << img.cols << "x" << img.rows << "x" << img.channels() << "]" << " " << img.depth();
	updateParameter<int>("Depth", img.depth())->type = Parameters::Parameter::State;
	updateParameter<int>("Rows", img.rows)->type = Parameters::Parameter::State;
	updateParameter<int>("Cols", img.cols)->type = Parameters::Parameter::State;
	updateParameter<int>("Channels", img.channels())->type = Parameters::Parameter::State;
	updateParameter<int>("Step", img.step)->type = Parameters::Parameter::State;
	updateParameter<int>("Ref count", *img.refcount)->type = Parameters::Parameter::State;
    return img;
}
void Mat2Tensor::Init(bool firstInit)
{
	Parameters::EnumParameter dataType;
    dataType.addEnum(ENUM(CV_8U));
    dataType.addEnum(ENUM(CV_8S));
    dataType.addEnum(ENUM(CV_16U));
    dataType.addEnum(ENUM(CV_16S));
    dataType.addEnum(ENUM(CV_32S));
    dataType.addEnum(ENUM(CV_32F));
    dataType.addEnum(ENUM(CV_64F));
    updateParameter<Parameters::EnumParameter>("Tensor Type",dataType);
    updateParameter("Include Position", true);
}
cv::cuda::GpuMat Mat2Tensor::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    int type = getParameter<Parameters::EnumParameter>(0)->Data()->currentSelection;
    bool position = *getParameter<bool>(1)->Data();
    int newCols = img.channels();
    if(position)
        newCols += 2;
    int rows = img.size().area();
    TIME
    if((position && positionMat.empty()) || _parameters[0]->changed)
    {
        cv::Mat h_positionMat(img.size().area(), 2, type);
        int row = 0;
        for(int y = 0; y < img.rows; ++y)
        {
            for(int x = 0; x < img.cols; ++x, ++row)
            {
                if(type == CV_8U)
                {
                    h_positionMat.at<uchar>(row,0) = x;
                    h_positionMat.at<uchar>(row,1) = y;
                }
                if(type == CV_8S)
                {
                    h_positionMat.at<char>(row,0) = x;
                    h_positionMat.at<char>(row,1) = y;
                }
                if(type == CV_16U)
                {
                    h_positionMat.at<unsigned short>(row,0) = x;
                    h_positionMat.at<unsigned short>(row,1) = y;
                }
                if(type == CV_32S)
                {
                    h_positionMat.at<int>(row,0) = x;
                    h_positionMat.at<int>(row,1) = y;
                }
                if(type == CV_32F)
                {
                    h_positionMat.at<float>(row,0) = x;
                    h_positionMat.at<float>(row,1) = y;
                }
                if(type == CV_64F)
                {
                    h_positionMat.at<double>(row,0) = x;
                    h_positionMat.at<double>(row,1) = y;
                }
            }
        }
        positionMat.upload(h_positionMat, stream);
        _parameters[0]->changed = false;
    }
    TIME
    auto buf = bufferPool.getFront();
    auto typeBuf = bufferPool.getFront();
    TIME
    if(position && !positionMat.empty())
    {
        TIME
        //buf->data.create(rows, newCols, type);

        if(buf->data.rows != rows || buf->data.cols != newCols || buf->data.type() != type)
            buf->data = cv::cuda::createContinuous(rows, newCols, type);
        TIME
        img.convertTo(typeBuf->data, type,stream);
        TIME
        typeBuf->data.reshape(1, rows).copyTo(buf->data(cv::Rect(0,0,img.channels(),rows)),stream);
        TIME
        positionMat.copyTo(buf->data(cv::Rect(img.channels(),0,2,rows)), stream);
        TIME
        return buf->data;
    }else
    {
        if(typeBuf->data.size() != img.size() || typeBuf->data.type() != type)
            typeBuf->data = cv::cuda::createContinuous(img.size(), type);
        img.convertTo(typeBuf->data, type, stream);
        TIME
        return typeBuf->data.reshape(1, rows);
    }
    return img;
}
cv::cuda::GpuMat ConcatTensor::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    bool full = true;
    std::vector<cv::cuda::GpuMat*> inputs;
    int type = -1;
    for(int i = 1; i < _parameters.size(); ++i)
    {
        auto param = dynamic_cast<Parameters::ITypedParameter<cv::cuda::GpuMat>*>(_parameters[i]);
        if(param)
        {
            if(param->Data() == nullptr)
                full = false;
            else
                inputs.push_back(param->Data());
        }
    }
    if(full == true)
    {
        addInputParameter<cv::cuda::GpuMat>("Input " + boost::lexical_cast<std::string>(_parameters.size()-1));
    }
    int cols = 0;
    int rows = 0;
    for(int i = 0; i < inputs.size(); ++i)
    {
        cols += inputs[i]->cols;
        rows = inputs[i]->rows;
        if(type == -1)
            type = inputs[i]->type();
        else
            if(type != inputs[i]->type())
                throw cv::Exception(0, "Datatype mismatch!",__FUNCTION__, __FILE__, __LINE__);
    }
    Buffer<cv::cuda::GpuMat, EventPolicy>* buf = d_buffer.getFront();
    buf->data.create(rows, cols, type);
    int colItr = 0;
    for(int i = 0; i < inputs.size(); ++i)
    {
        inputs[i]->copyTo(buf->data(cv::Rect(colItr,0, inputs[i]->cols, rows)), stream);
        colItr += inputs[i]->cols;
    }
    if(buf->data.empty())
        return img;
    else
        return buf->data;
}

void ConcatTensor::Init(bool firstInit)
{
    updateParameter("Include Input", true);
    addInputParameter<cv::cuda::GpuMat>("Input 0");
	
}
cv::cuda::GpuMat LagBuffer::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
	if (_parameters[0]->changed)
	{
		putItr = 0;
		getItr = 0;
		imageBuffer.resize(lagFrames);
		_parameters[0]->changed = false;
	}
	if (lagFrames == 0)
		return img;
	imageBuffer[putItr % lagFrames] = img;
	++putItr;
	{
		if (putItr > 1000 && getItr > 1000)
		{
			putItr -= 1000;
			getItr -= 1000;
		}
	}
	if ((putItr - getItr) == lagFrames)
	{
		cv::cuda::GpuMat out = imageBuffer[getItr % lagFrames];
		++getItr;
		return out;
	}
	return cv::cuda::GpuMat();
}
void LagBuffer::Init(bool firstInit)
{
	imageBuffer.resize(20);
	putItr = 0;
	getItr = 0;
	lagFrames = 20;
	_parameters.push_back(new Parameters::TypedInputParameterCopy<unsigned int>("Lag frames", 
								&lagFrames, Parameters::Parameter::ParameterType(Parameters::Parameter::Input | Parameters::Parameter::Control), 
								"Number of frames for this video stream to lag behind"));

	//	updateParameter<unsigned int>("Lag frames", &lagFrames, Parameters::Parameter::Control, "Number of frames for this video stream to lag behind");
}
cv::cuda::GpuMat CameraSync::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
	if (_parameters[0]->changed)
	{
		int offset = *getParameter<int>(0)->Data();
		if (offset == 0)
		{
			updateParameter<unsigned int>("Camera 1 offset", 0)->type =  Parameters::Parameter::Output;
			updateParameter<unsigned int>("Camera 2 offset", 0)->type =  Parameters::Parameter::Output;
		}
		else
		{
			if (offset < 0)
			{
				updateParameter<unsigned int>("Camera 1 offset", abs(offset))->type =  Parameters::Parameter::Output;
				updateParameter<unsigned int>("Camera 2 offset", 0)->type =  Parameters::Parameter::Output;
			}
			else
			{
				updateParameter<unsigned int>("Camera 1 offset", 0)->type = Parameters::Parameter::Output;
				updateParameter<unsigned int>("Camera 2 offset", abs(offset))->type =  Parameters::Parameter::Output;
			}
		}	
		_parameters[0]->changed = false;
	}
	return img;
}
bool CameraSync::SkipEmpty() const
{
	return false;
}
void CameraSync::Init(bool firstInit)
{
	updateParameter<int>("Camera offset", 0);
	updateParameter<unsigned int>("Camera 1 offset", 0)->type =  Parameters::Parameter::Output;
	updateParameter<unsigned int>("Camera 2 offset", 0)->type = Parameters::Parameter::Output;
	
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(GetOutputImage, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ImageInfo, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ExportInputImage, Image, Extractor)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Mat2Tensor, Converter)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ConcatTensor, Tensor, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(LagBuffer, Utility)
NODE_DEFAULT_CONSTRUCTOR_IMPL(CameraSync, Utility)

