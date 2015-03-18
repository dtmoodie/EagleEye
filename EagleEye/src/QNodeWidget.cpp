#include "QNodeWidget.h"

#include "ui_QNodeWidget.h"




QNodeWidget::QNodeWidget(QWidget* parent, EagleLib::Node* node) :
	QWidget(parent),
	ui(new Ui::QNodeWidget())
{
	ui->setupUi(this);
	if (node)
	{
		ui->nodeName->setText(QString::fromStdString(node->fullTreeName));
		for (int i = 0; i < node->parameters.size(); ++i)
		{
				
		}
	}
}
QNodeWidget::~QNodeWidget()
{

}
/*
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<bool>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<bool*>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<std::string>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<std::string*>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<float>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<float*>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<double>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<double*>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<int>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<int*>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<cv::Vec3f>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<cv::Vec3f*>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<cv::Vec3b>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<cv::Vec3b*>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<cv::Mat>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<cv::cuda::GpuMat>> parameter)
{

}
void QNodeWidget::handleParameter(boost::shared_ptr<EagleLib::TypedParameter<boost::function<void(void)>>> parameter)
{

}*/