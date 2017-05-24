#include "DataHandling.h"


#include <boost/lexical_cast.hpp>
using namespace aq;
using namespace aq::Nodes;

bool PlaybackInfo::processImpl()
{
    auto current_ts = input_param.getTimestamp();
    double framerate = 30.0;
    if(current_ts && last_timestamp)
    {
        mo::Time_t ts_delta = (*current_ts - *last_timestamp);
        framerate = 1000.0 / ts_delta.value();
    }

    auto now = boost::posix_time::microsec_clock::local_time();
    double processrate = 1000.0 / (double)boost::posix_time::time_duration(now - last_iteration_time).total_milliseconds();

    framerate_param.updateData(processrate);
    source_framerate_param.updateData(framerate);
    playrate_param.updateData(processrate / framerate);
    last_timestamp = input_param.getTimestamp();
    last_iteration_time = now;
    return true;
}
MO_REGISTER_CLASS(PlaybackInfo);
bool ImageInfo::processImpl()
{
    auto ts = input_param.getTimestamp();
    auto shape = input->GetShape();
    count_param.updateData(shape[0], ts, _ctx);
    height_param.updateData(shape[1], ts, _ctx);
    width_param.updateData(shape[2], ts, _ctx);
    channels_param.updateData(shape[3], ts, _ctx);
    return true;
}

MO_REGISTER_CLASS(ImageInfo);
bool Mat2Tensor::processImpl()
{
    int new_channels = input->getChannels();
    if(include_position)
    {
        new_channels += 2;
    }
    return false;
}


/*cv::cuda::GpuMat Mat2Tensor::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    int type = getParameter<Parameters::EnumParameter>(0)->Data()->currentSelection;
    bool position = *getParameter<bool>(1)->Data();
    int newCols = img.channels();
    if(position)
        newCols += 2;
    int rows = img.size().area();
    cv::cuda::GpuMat typed, continuous;
    if(img.type() == type)
    {
        typed = img;
    }    
    else
    {
        cv::cuda::createContinuous(img.size(), type, typed);
        img.convertTo(typed, type, stream);
    }
    if(!typed.isContinuous())
    {
        cv::cuda::createContinuous(typed.size(), typed.type(), continuous);
    }else
    {
        continuous = typed;
    }
    cv::cuda::GpuMat output;
    cv::cuda::createContinuous(rows, newCols, type, output);
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
    if(position)
    {
        positionMat.copyTo(output.colRange(img.channels(), output.cols), stream);
    }
    continuous.reshape(1, rows).copyTo(output.colRange(0, img.channels()), stream);
    return output;

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
        addInputParam<cv::cuda::GpuMat>("Input " + boost::lexical_cast<std::string>(_parameters.size()-1));
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

void ConcatTensor::nodeInit(bool firstInit)
{
    updateParameter("Include Input", true);
    addInputParam<cv::cuda::GpuMat>("Input 0");
    
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
void LagBuffer::nodeInit(bool firstInit)
{
    imageBuffer.resize(20);
    putItr = 0;
    getItr = 0;
    lagFrames = 20;
    _parameters.push_back(new Parameters::TypedInputParamCopy<unsigned int>("Lag frames", 
                                &lagFrames, Parameters::Parameter::ParameterType(Parameters::Parameter::Input | Parameters::Parameter::Control), 
                                "Number of frames for this video stream to lag behind"));

    //    updateParameter<unsigned int>("Lag frames", &lagFrames, Parameters::Parameter::Control, "Number of frames for this video stream to lag behind");
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
void CameraSync::nodeInit(bool firstInit)
{
    updateParameter<int>("Camera offset", 0);
    updateParameter<unsigned int>("Camera 1 offset", 0)->type =  Parameters::Parameter::Output;
    updateParameter<unsigned int>("Camera 2 offset", 0)->type = Parameters::Parameter::Output;
    
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(getOutputImage, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ImageInfo, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ExportInputImage, Image, Extractor)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Mat2Tensor, Converter)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ConcatTensor, Tensor, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(LagBuffer, Utility)
NODE_DEFAULT_CONSTRUCTOR_IMPL(CameraSync, Utility)

*/
