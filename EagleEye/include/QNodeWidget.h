#pragma once
#include "nodes/Node.h"
#include "qgraphicsitem.h"
#include "qwidget.h"
#include <boost/type_traits.hpp>
namespace Ui {
	class QNodeWidget;
}
class CV_EXPORTS QNodeWidget : public QWidget
{
	Q_OBJECT
public:
	QNodeWidget(QWidget* parent = nullptr, EagleLib::Node* node = nullptr);
	~QNodeWidget();

private:
	/*void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<bool>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<bool*>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<std::string>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<std::string*>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<float>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<float*>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<double>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<double*>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<int>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<int*>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<cv::Vec3f>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<cv::Vec3f*>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<cv::Vec3b>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<cv::Vec3b*>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<cv::Mat>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<cv::cuda::GpuMat>> parameter);
	void handleParameter(boost::shared_ptr<EagleLib::TypedParameter<boost::function<void(void)>>> parameter);*/
	
	template<typename T> T* getParameter(EagleLib::Parameter::Ptr parameter)
	{
		if (parameter->typeName == typeid(T).name())
		{
            typename EagleLib::TypedParameter<T>::Ptr typedParam = boost::dynamic_pointer_cast<EagleLib::TypedParameter<T>, EagleLib::Parameter>(parameter);
			return &typedParam->data;
		}
		if (parameter->typeName == typeid(T*).name())
		{
            typename EagleLib::TypedParameter<T*>::Ptr typedParam = boost::dynamic_pointer_cast<EagleLib::TypedParameter<T*>, EagleLib::Parameter>(parameter);
			return typedParam->data;
		}
		return nullptr;
	}	
	
	Ui::QNodeWidget* ui;
	
};
// Interface class for the interop class
class CV_EXPORTS IQNodeInterop: public QObject
{
	Q_OBJECT

public:
	void updateUi();


private:


};

template<typename T> class QNodeInterop : IQNodeInterop
{
public:

	QNodeInterop(QNodeWidget* parent, EagleLib::TypedParameter<T>::Ptr parameter);
	~QNodeInterop();

	std::enable_if<std::is_floating_point<T>::value, void>::type updateUi();


private:
	EagleLib::TypedParameter<T>::Ptr parameter;
};