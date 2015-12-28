#pragma once
#include "Parameters.hpp"
#include "QVector"

#include "opencv2/core.hpp"
#include <opencv2/core/cuda.hpp>
#include "boost/circular_buffer.hpp"
#include <mutex>
#include "EagleLib/Defs.hpp"
#include "Plotting.h"
#include <memory>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
#include "Converters/DoubleConverter.hpp"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace Parameters
{
	namespace UI
	{
		namespace qt
		{
			class IParameterProxy;
		}
	}
}

class QWidget;


namespace EagleLib
{
	class EAGLE_EXPORTS QtPlotterImpl : public QtPlotter
	{
	protected:
		int channels;
		cv::Size size;
		std::shared_ptr<QWidget> controlWidget;
		std::vector<boost::signals2::connection> connections;
		std::mutex mtx;
		
		std::vector<std::shared_ptr<Parameters::UI::qt::IParameterProxy>> parameterProxies;
		std::vector<std::shared_ptr<Parameters::Parameter>> parameters;
		Parameters::Converters::Double::IConverter* converter;
	public:
		QtPlotterImpl();
		~QtPlotterImpl();
		virtual QWidget* GetControlWidget(QWidget* parent);
		virtual void Serialize(ISimpleSerializer *pSerializer);
		//virtual void Init(bool firstInit);
		template<typename T> typename Parameters::ITypedParameter<T>::Ptr GetParameter(const std::string& name)
		{
			for (auto& itr : parameters)
			{
				if (itr->GetName() == name)
				{
					return std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(itr);
				}
			}
            return typename Parameters::ITypedParameter<T>::Ptr();
		}
		template<typename T> typename Parameters::ITypedParameter<T>::Ptr GetParameter(size_t index)
		{
			if(index < parameters.size())
				return std::dynamic_pointer_cast<typename Parameters::ITypedParameter<T>>(parameters[index]);
            return typename Parameters::ITypedParameter<T>::Ptr();
		}

		//virtual void HandleData(double data, int row, int col, int channel) = 0;

		virtual void OnParameterUpdate(cv::cuda::Stream* stream);
	};

	class EAGLE_EXPORTS HistoryPlotter : public QtPlotterImpl
	{
	protected:
		std::vector<boost::circular_buffer<double>> channelData;
		HistoryPlotter();
		void on_history_size_change();

	public:
		virtual void Init(bool firstInit);
	};
	class EAGLE_EXPORTS StaticPlotter : public QtPlotterImpl
	{
	protected:
		QVector<QVector<double>> channelData;


	public:
	};

	template<typename T> class EAGLE_EXPORTS VectorPlotter : public T
	{
		virtual void OnParameterUpdate(cv::cuda::Stream* stream)
		{

		}
	};
	template<typename T> class EAGLE_EXPORTS MatrixPlotter : public T
	{

	};

	template<typename T> class EAGLE_EXPORTS ScalarPlotter : public T
	{

	};


}




template<typename T> T* getParameterPtr(Parameters::Parameter::Ptr param)
{
	auto typedParam = std::dynamic_pointer_cast<Parameters::ITypedParameter<T>>(param);
	if (typedParam)
		return typedParam->Data();
	return nullptr;
}


cv::Size inline getSize(Parameters::Parameter::Ptr param)
{
    //auto gpuParam = EagleLib::getParameterPtr<cv::cuda::GpuMat>(param);
	auto gpuParam = getParameterPtr<cv::cuda::GpuMat>(param);
    if(gpuParam)
        return gpuParam->size();
    auto cpuParam = getParameterPtr<cv::Mat>(param);
    if(cpuParam)
        return cpuParam->size();
    auto hostParam = getParameterPtr<cv::cuda::HostMem>(param);
    if(hostParam)
        return hostParam->size();

    return cv::Size(0,0);
}
int inline getChannels(Parameters::Parameter::Ptr param)
{
    auto gpuParam = getParameterPtr<cv::cuda::GpuMat>(param);
    if(gpuParam)
        return gpuParam->channels();
    auto cpuParam = getParameterPtr<cv::Mat>(param);
    if(cpuParam)
        return cpuParam->channels();
    auto hostParam = getParameterPtr<cv::cuda::HostMem>(param);
    if(hostParam)
        return hostParam->channels();
    return 0;
}
QVector<double> inline getParamArrayDataHelper(cv::Mat h_data, int channel)
{
    QVector<double> output;
    CV_Assert((h_data.rows == 1 || h_data.cols == 1) && channel < h_data.channels());
    const int numItems = h_data.size().area();
    output.reserve(numItems);

    for(int i = 0; i < numItems; ++i)
    {
        if(h_data.depth() == CV_8U)
            output.push_back(h_data.at<uchar>(i));
        if(h_data.depth() == CV_8UC3)
            output.push_back(h_data.at<cv::Vec2b>(i).val[channel]);
        if(h_data.depth() == CV_16U)
            output.push_back(h_data.at<unsigned short>(i));
        if(h_data.depth() == CV_16UC3)
            output.push_back(h_data.at<cv::Vec3w>(i).val[channel]);
        if(h_data.depth() == CV_16S)
            output.push_back(h_data.at<short>(i));
        if(h_data.depth() == CV_16SC3)
            output.push_back(h_data.at<cv::Vec3s>(i).val[channel]);
        if(h_data.depth() == CV_32S)
            output.push_back(h_data.at<int>(i));
        if(h_data.depth() == CV_32SC3)
            output.push_back(h_data.at<cv::Vec3i>(i).val[channel]);
        if(h_data.depth() == CV_32F)
            output.push_back(h_data.at<float>(i));
        if(h_data.depth() == CV_32FC3)
            output.push_back(h_data.at<cv::Vec3f>(i).val[channel]);
        if(h_data.depth() == CV_64F)
            output.push_back(h_data.at<double>(i));
        if(h_data.depth() == CV_64FC3)
            output.push_back(h_data.at<cv::Vec3d>(i).val[channel]);
    }
    return output;
}

QVector<double> inline getParamArrayData(Parameters::Parameter::Ptr param, int channel)
{
    auto gpuParam = getParameterPtr<cv::cuda::GpuMat>(param);
    if(gpuParam)
        return getParamArrayDataHelper(cv::Mat(*gpuParam), channel);
    auto cpuParam = getParameterPtr<cv::Mat>(param);
    if(cpuParam)
        return getParamArrayDataHelper(*cpuParam, channel);
    auto hostParam = getParameterPtr<cv::cuda::HostMem>(param);
    if(hostParam)
        return getParamArrayDataHelper(hostParam->createMatHeader(), channel);
    auto vecParam = getParameterPtr<std::vector<double>>(param);
    if(vecParam)
        return getParamArrayDataHelper(cv::Mat(*vecParam), 0);
    return QVector<double>();
}
template<typename T> bool inline getData(Parameters::Parameter::Ptr param, double& data)
{
    auto ptr = getParameterPtr<T>(param);
    if(ptr)
    {
        data = *ptr;
        return true;
    }
    return false;
}
template<typename T> bool inline getDataVec(Parameters::Parameter::Ptr param, int channel, double& data)
{
    auto ptr = getParameterPtr<T>(param);
    if(ptr)
    {
        data = ptr->val[channel];
        return true;
    }
    return false;
}

double inline getParamData(Parameters::Parameter::Ptr data, int channel)
{
    double output;
    if(getData<double>(data, output))
        return output;
    if(getData<int>(data,output))
        return output;
    if(getData<char>(data,output))
        return output;
    if(getData<unsigned char>(data,output))
        return output;
    if(getData<float>(data,output))
        return output;
    if(getData<unsigned int>(data,output))
        return output;
    if(getData<short>(data,output))
        return output;
    if(getData<unsigned short>(data,output))
        return output;
    if(getDataVec<cv::Vec2b>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec3b>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec4b>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec2s>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec3s>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec4s>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec2i>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec3i>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec4i>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec6i>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec8i>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec2d>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec3d>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec4d>(data, channel, output))
        return output;
    if(getDataVec<cv::Vec6d>(data, channel, output))
        return output;
    if(getDataVec<cv::Scalar>(data, channel, output))
        return output;
}


// Helper policies
struct DefaultChannelPolicy
{
    static bool acceptsChannels(int channels){return channels != 0;}
};

struct SingleChannelPolicy
{
    static bool acceptsChannels(int channels){ return channels == 1;}
};

struct MultiChannelPolicy
{
    static bool acceptsChannels(int channels){ return channels > 1;}
};

struct DefaultSizePolicy
{
    static bool acceptsSize(cv::Size size){ return true;}
};

struct VectorSizePolicy
{
    static bool acceptsSize(cv::Size size){ return size.width == 1 || size.height == 1;}
};

struct MatrixSizePolicy
{
    static bool acceptsSize(cv::Size size){ return size.width != 1 && size.height != 1;}
};

struct DefaultTypePolicy
{
    static bool acceptsType(Parameters::Parameter::Ptr param)
    {
        return true;
    }
};

template<typename T> struct TypePolicy
{
    static bool acceptsType(Parameters::Parameter::Ptr param)
    {
	return Loki::TypeInfo(typeid(T)) == param->GetTypeInfo();
    }
};
struct StaticPlotPolicy
{
    int size;
    int channel;
    QVector<double> data;
    void addPlotData(Parameters::Parameter::Ptr param, cv::cuda::Stream* stream)
    {
        data = getParamArrayData(param, channel);
    }

    QVector<double>& getPlotData()
    {
        return data;
    }

    void setSize(int size_)
    {
        size = size_;
    }
    void setChannel(int channel_)
    {
        channel = channel_;
    }
};

struct SlidingWindowPlotPolicy
{
    int channel;

    boost::circular_buffer<double> plotData;
    void addPlotData(Parameters::Parameter::Ptr param)
    {
        plotData.push_back(getParamData(param, channel));
    }

    QVector<double> getPlotData()
    {
        QVector<double> data;
        data.reserve(plotData.size());
        for(size_t i = 0; i < plotData.size(); ++i)
        {
            data.push_back(plotData[i]);
        }
        return data;
    }

    void setSize(int size)
    {
        plotData.set_capacity(size);
    }
};
