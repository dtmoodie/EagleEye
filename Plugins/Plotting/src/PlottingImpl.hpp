#pragma once

#include "QVector"

#include "Plotting.h"
#include "PlottingExport.hpp"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
#include "boost/circular_buffer.hpp"
#include "opencv2/core.hpp"
#include <memory>
#include <mutex>
#include <opencv2/core/cuda.hpp>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

class QWidget;

namespace aq
{
    class Plotting_EXPORT QtPlotterImpl : public QtPlotter
    {
      protected:
        int channels;
        cv::Size size;
        std::shared_ptr<QWidget> controlWidget;
        std::vector<std::shared_ptr<mo::Connection>> connections;
        std::mutex mtx;

        std::vector<std::shared_ptr<mo::UI::qt::IParameterProxy>> parameterProxies;
        std::vector<mo::IParam*> parameters;
        // Parameters::Converters::Double::IConverter* converter;
      public:
        QtPlotterImpl();
        ~QtPlotterImpl();
        virtual QWidget* GetControlWidget(QWidget* parent);
        virtual void Serialize(ISimpleSerializer* pSerializer);
        template <typename T>
        typename mo::ITParam<T>* GetParameter(const std::string& name)
        {
            for (auto& itr : parameters)
            {
                if (itr->GetName() == name)
                {
                    return dynamic_cast<typename mo::ITParam<T>*>(itr);
                }
            }
            return nullptr;
        }
        template <typename T>
        typename mo::ITParam<T>::Ptr GetParameter(size_t index)
        {
            if (index < parameters.size())
                return std::dynamic_pointer_cast<typename mo::ITParam<T>>(parameters[index]);
            return typename mo::ITParam<T>::Ptr();
        }

        virtual void OnParameterUpdate(cv::cuda::Stream* stream);
    };

    class Plotting_EXPORT HistoryPlotter : public QtPlotterImpl
    {
      protected:
        std::vector<boost::circular_buffer<double>> channelData;
        HistoryPlotter();
        void on_history_size_change();

      public:
        virtual void PlotInit(bool firstInit);
    };
    class Plotting_EXPORT StaticPlotter : public QtPlotterImpl
    {
      protected:
        QVector<QVector<double>> channelData;

      public:
    };

    template <typename T>
    class Plotting_EXPORT VectorPlotter : public T
    {
        virtual void OnParameterUpdate(cv::cuda::Stream* stream) {}
    };

    template <typename T>
    class Plotting_EXPORT MatrixPlotter : public T
    {
    };

    template <typename T>
    class Plotting_EXPORT ScalarPlotter : public T
    {
    };
}

template <typename T>
T* getParameterPtr(mo::IParam* param)
{
    auto typedParam = dynamic_cast<mo::ITParam<T>*>(param);
    if (typedParam)
        return typedParam->Data();
    return nullptr;
}

cv::Size inline getSize(mo::IParam* param)
{
    auto gpuParam = getParameterPtr<cv::cuda::GpuMat>(param);
    if (gpuParam)
        return gpuParam->size();
    auto cpuParam = getParameterPtr<cv::Mat>(param);
    if (cpuParam)
        return cpuParam->size();
    auto hostParam = getParameterPtr<cv::cuda::HostMem>(param);
    if (hostParam)
        return hostParam->size();

    return cv::Size(0, 0);
}
int inline getChannels(mo::IParam* param)
{
    auto gpuParam = getParameterPtr<cv::cuda::GpuMat>(param);
    if (gpuParam)
        return gpuParam->channels();
    auto cpuParam = getParameterPtr<cv::Mat>(param);
    if (cpuParam)
        return cpuParam->channels();
    auto hostParam = getParameterPtr<cv::cuda::HostMem>(param);
    if (hostParam)
        return hostParam->channels();
    return 0;
}
QVector<double> inline getParamArrayDataHelper(cv::Mat h_data, int channel)
{
    QVector<double> output;
    CV_Assert((h_data.rows == 1 || h_data.cols == 1) && channel < h_data.channels());
    const int numItems = h_data.size().area();
    output.reserve(numItems);

    for (int i = 0; i < numItems; ++i)
    {
        if (h_data.depth() == CV_8U)
            output.push_back(h_data.at<uchar>(i));
        if (h_data.depth() == CV_8UC3)
            output.push_back(h_data.at<cv::Vec2b>(i).val[channel]);
        if (h_data.depth() == CV_16U)
            output.push_back(h_data.at<unsigned short>(i));
        if (h_data.depth() == CV_16UC3)
            output.push_back(h_data.at<cv::Vec3w>(i).val[channel]);
        if (h_data.depth() == CV_16S)
            output.push_back(h_data.at<short>(i));
        if (h_data.depth() == CV_16SC3)
            output.push_back(h_data.at<cv::Vec3s>(i).val[channel]);
        if (h_data.depth() == CV_32S)
            output.push_back(h_data.at<int>(i));
        if (h_data.depth() == CV_32SC3)
            output.push_back(h_data.at<cv::Vec3i>(i).val[channel]);
        if (h_data.depth() == CV_32F)
            output.push_back(h_data.at<float>(i));
        if (h_data.depth() == CV_32FC3)
            output.push_back(h_data.at<cv::Vec3f>(i).val[channel]);
        if (h_data.depth() == CV_64F)
            output.push_back(h_data.at<double>(i));
        if (h_data.depth() == CV_64FC3)
            output.push_back(h_data.at<cv::Vec3d>(i).val[channel]);
    }
    return output;
}

QVector<double> inline getParamArrayData(mo::IParam* param, int channel)
{
    auto gpuParam = getParameterPtr<cv::cuda::GpuMat>(param);
    if (gpuParam)
        return getParamArrayDataHelper(cv::Mat(*gpuParam), channel);
    auto cpuParam = getParameterPtr<cv::Mat>(param);
    if (cpuParam)
        return getParamArrayDataHelper(*cpuParam, channel);
    auto hostParam = getParameterPtr<cv::cuda::HostMem>(param);
    if (hostParam)
        return getParamArrayDataHelper(hostParam->createMatHeader(), channel);
    auto vecParam = getParameterPtr<std::vector<double>>(param);
    if (vecParam)
        return getParamArrayDataHelper(cv::Mat(*vecParam), 0);
    return QVector<double>();
}

template <typename T>
bool inline getData(mo::IParam* param, double& data)
{
    auto ptr = getParameterPtr<T>(param);
    if (ptr)
    {
        data = *ptr;
        return true;
    }
    return false;
}
template <typename T>
bool inline getDataVec(mo::IParam* param, int channel, double& data)
{
    auto ptr = getParameterPtr<T>(param);
    if (ptr)
    {
        data = ptr->val[channel];
        return true;
    }
    return false;
}

double inline getParamData(mo::Parameter* data, int channel)
{
    double output;
    if (getData<double>(data, output))
        return output;
    if (getData<int>(data, output))
        return output;
    if (getData<char>(data, output))
        return output;
    if (getData<unsigned char>(data, output))
        return output;
    if (getData<float>(data, output))
        return output;
    if (getData<unsigned int>(data, output))
        return output;
    if (getData<short>(data, output))
        return output;
    if (getData<unsigned short>(data, output))
        return output;
    if (getDataVec<cv::Vec2b>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec3b>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec4b>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec2s>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec3s>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec4s>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec2i>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec3i>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec4i>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec6i>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec8i>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec2d>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec3d>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec4d>(data, channel, output))
        return output;
    if (getDataVec<cv::Vec6d>(data, channel, output))
        return output;
    if (getDataVec<cv::Scalar>(data, channel, output))
        return output;
}

// Helper policies
struct DefaultChannelPolicy
{
    static bool acceptsChannels(int channels) { return channels != 0; }
};

struct SingleChannelPolicy
{
    static bool acceptsChannels(int channels) { return channels == 1; }
};

struct MultiChannelPolicy
{
    static bool acceptsChannels(int channels) { return channels > 1; }
};

struct DefaultSizePolicy
{
    static bool acceptsSize(cv::Size size) { return true; }
};

struct VectorSizePolicy
{
    static bool acceptsSize(cv::Size size) { return size.width == 1 || size.height == 1; }
};

struct MatrixSizePolicy
{
    static bool acceptsSize(cv::Size size) { return size.width != 1 && size.height != 1; }
};

struct DefaultTypePolicy
{
    static bool acceptsType(mo::Parameter::Ptr param) { return true; }
};

template <typename T>
struct TypePolicy
{
    static bool acceptsType(mo::Parameter* param) { return mo::TypeInfo(typeid(T)) == param->getTypeInfo(); }
};
struct StaticPlotPolicy
{
    int size;
    int channel;
    QVector<double> data;
    void addPlotData(mo::IParam* param, cv::cuda::Stream* stream) { data = getParamArrayData(param, channel); }

    QVector<double>& getPlotData() { return data; }

    void setSize(int size_) { size = size_; }
    void setChannel(int channel_) { channel = channel_; }
};

struct SlidingWindowPlotPolicy
{
    int channel;

    boost::circular_buffer<double> plotData;
    void addPlotData(mo::IParam* param) { plotData.push_back(getParamData(param, channel)); }

    QVector<double> getPlotData()
    {
        QVector<double> data;
        data.reserve(plotData.size());
        for (size_t i = 0; i < plotData.size(); ++i)
        {
            data.push_back(plotData[i]);
        }
        return data;
    }

    void setSize(int size) { plotData.set_capacity(size); }
};
