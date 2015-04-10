#include "QNodeWidget.h"

#include "ui_QNodeWidget.h"




QNodeWidget::QNodeWidget(QWidget* parent, EagleLib::Node* node) :
	QWidget(parent),
	ui(new Ui::QNodeWidget())
{
	ui->setupUi(this);
	if (node)
	{
		nodeId = node->GetObjectId();
		ui->nodeName->setText(QString::fromStdString(node->fullTreeName));
		ui->verticalLayout->setSpacing(0);
		for (int i = 0; i < node->parameters.size(); ++i)
		{
			auto interop = new IQNodeInterop(node->parameters[i], this);
			interop->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
			interop->setMaximumWidth(this->width());
			ui->verticalLayout->addWidget(interop);
		}
	}
}
QNodeWidget::~QNodeWidget()
{

}
EagleLib::Node* QNodeWidget::getNode()
{
	if (nodeId.IsValid())
	{
		return EagleLib::NodeManager::getInstance().getNode(nodeId);
	}
	return nullptr;
}
IQNodeProxy* dispatchParameter(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter)
{
	if (parameter->typeName == typeid(double).name())
		return new QNodeProxy<double>(parent, parameter);

	if (parameter->typeName == typeid(int).name())
		return new QNodeProxy<int>(parent, parameter);

	if (parameter->typeName == typeid(unsigned int).name())
		return new QNodeProxy<unsigned int>(parent, parameter);

	if (parameter->typeName == typeid(unsigned char).name())
		return new QNodeProxy<unsigned char>(parent, parameter);

	if (parameter->typeName == typeid(char).name())
		return new QNodeProxy<char>(parent, parameter);
	if (parameter->typeName == typeid(std::string).name())
		return new QNodeProxy<std::string>(parent, parameter);
	if (parameter->typeName == typeid(boost::filesystem::path).name())
		return new QNodeProxy<boost::filesystem::path>(parent, parameter);

	/*if (parameter->typeName == typeid(cv::Scalar).name())
	return new QNodeProxy<cv::Scalar>(parent, parameter);

	

	if (parameter->typeName == typeid(cv::Mat).name())
	return new QNodeProxy<cv::Mat>(parent, parameter);*/

	if (parameter->typeName == typeid(bool).name())
		return new QNodeProxy<bool>(parent, parameter);

	return nullptr;
}

