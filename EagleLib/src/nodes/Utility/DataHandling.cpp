#include "nodes/Utility/DataHandling.h"

using namespace EagleLib;

void GetOutputImage::Init(bool firstInit)
{
    if(firstInit)
        addInputParameter<cv::cuda::GpuMat>("Input");
}

cv::cuda::GpuMat GetOutputImage::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* input = getParameter<cv::cuda::GpuMat*>("Input")->data;
    if(input == nullptr)
    {
        log(Status, "Input not defined");
        return img;
    }
    if(input->empty())
    {
        log(Status, "Input is empty");
        return img;
    }
    return *input;
}
cv::cuda::GpuMat ExportInputImage::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    updateParameter("Output image", img, Parameter::Output);
    return img;
}

void ExportInputImage::Init(bool firstInit)
{

}

void ImageInfo::Init(bool firstInit)
{
    EnumParameter dataType;
    dataType.addEnum(ENUM(CV_8U));
    dataType.addEnum(ENUM(CV_8S));
    dataType.addEnum(ENUM(CV_16U));
    dataType.addEnum(ENUM(CV_16S));
    dataType.addEnum(ENUM(CV_32S));
    dataType.addEnum(ENUM(CV_32F));
    dataType.addEnum(ENUM(CV_64F));
    updateParameter<EnumParameter>("Type",dataType, Parameter::State);
}
cv::cuda::GpuMat ImageInfo::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{

    auto param = getParameter<EnumParameter>(0);
    if(param->data.currentSelection != img.type())
    {
        param->data.currentSelection = img.type();
        parameters[0]->changed = true;
        parameters[0]->onUpdate();
    }
    updateParameter<int>("Depth",img.depth(), Parameter::State);
    updateParameter<int>("Rows",img.rows, Parameter::State);
    updateParameter<int>("Cols", img.cols, Parameter::State);
    updateParameter<int>("Channels", img.channels(), Parameter::State);
    updateParameter<int>("Step", img.step, Parameter::State);
    updateParameter<int>("Ref count", *img.refcount, Parameter::State);
    return img;
}
void Mat2Tensor::Init(bool firstInit)
{
    EnumParameter dataType;
    dataType.addEnum(ENUM(CV_8U));
    dataType.addEnum(ENUM(CV_8S));
    dataType.addEnum(ENUM(CV_16U));
    dataType.addEnum(ENUM(CV_16S));
    dataType.addEnum(ENUM(CV_32S));
    dataType.addEnum(ENUM(CV_32F));
    dataType.addEnum(ENUM(CV_64F));
    updateParameter<EnumParameter>("Tensor Type",dataType);
    updateParameter("Include Position", true);
}
cv::cuda::GpuMat Mat2Tensor::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    int type = getParameter<EnumParameter>(0)->data.currentSelection;
    bool position = getParameter<bool>(1)->data;
    int newCols = img.channels();
    if(position)
        newCols += 2;
    int rows = img.size().area();
    TIME
    if((position && positionMat.empty()) || parameters[0]->changed)
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
        parameters[0]->changed = false;
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
    for(int i = 1; i < parameters.size(); ++i)
    {
        auto param = boost::dynamic_pointer_cast<TypedParameter<cv::cuda::GpuMat*>, Parameter>(parameters[i]);
        if(param)
        {
            if(param->data == nullptr)
                full = false;
            else
                inputs.push_back(param->data);
        }
    }
    if(full == true)
    {
        addInputParameter<cv::cuda::GpuMat>("Input " + boost::lexical_cast<std::string>(parameters.size()-1));
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

NODE_DEFAULT_CONSTRUCTOR_IMPL(GetOutputImage)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ImageInfo)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ExportInputImage)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Mat2Tensor)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ConcatTensor)
