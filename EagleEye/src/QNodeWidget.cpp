#include "QNodeWidget.h"

#include "ui_QNodeWidget.h"
#include <boost/bind.hpp>



IQNodeInterop::IQNodeInterop(boost::shared_ptr<EagleLib::Parameter> parameter_, QWidget* parent) :
    QWidget(parent)
{
    layout = new QGridLayout(this);
    layout->setVerticalSpacing(0);
    nameElement = new QLabel(QString::fromStdString(parameter_->name), parent);
    proxy = dispatchParameter(this, parameter_);
    if (proxy)
    {
        layout->addWidget(proxy->getWidget(), 0, 1);
        layout->addWidget(proxy->getTypename(), 0, 2);
    }
    layout->addWidget(nameElement, 0, 0);
    nameElement->setToolTip(QString::fromStdString(parameter_->toolTip));
    parameter_->updateCallback = boost::bind(&IQNodeInterop::onParameterUpdate,this, _1);
}

IQNodeInterop::~IQNodeInterop()
{
    delete proxy;
}

void IQNodeInterop::updateUi()
{
    if (proxy)
        proxy->updateUi();
}

void IQNodeInterop::on_valueChanged(double value)
{
    if (proxy)
        proxy->onUiUpdated();
}

void IQNodeInterop::on_valueChanged(int value)
{
    if (proxy)
        proxy->onUiUpdated();
}

void IQNodeInterop::on_valueChanged(bool value)
{
    if (proxy)
        proxy->onUiUpdated();
}
void IQNodeInterop::on_valueChanged(QString value)
{
    if (proxy)
        proxy->onUiUpdated();
}

void IQNodeInterop::on_valueChanged()
{
    if (proxy)
        proxy->onUiUpdated();
}
void IQNodeInterop::onParameterUpdate(boost::shared_ptr<EagleLib::Parameter> parameter)
{
    updateUi();
}

QNodeWidget::QNodeWidget(QWidget* parent, EagleLib::Node* node) :
    QWidget(parent),
	ui(new Ui::QNodeWidget())
{
	ui->setupUi(this);
	if (node)
	{
        ui->chkEnabled->setChecked(node->enabled);
        connect(ui->chkEnabled, SIGNAL(clicked(bool)), this, SLOT(on_enableClicked(bool)));
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
{}

void QNodeWidget::on_enableClicked(bool state)
{    EagleLib::NodeManager::getInstance().getNode(nodeId)->enabled = state;     }

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
    if(parameter->type & EagleLib::Parameter::Control)
    {
        if (parameter->typeInfo == Loki::TypeInfo(typeid(double)))
            return new QNodeProxy<double, false>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(int)))
            return new QNodeProxy<int, false>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(unsigned int)))
            return new QNodeProxy<unsigned int, false>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(unsigned char)))
            return new QNodeProxy<unsigned char, false>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(char)))
            return new QNodeProxy<char, false>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(std::string)))
            return new QNodeProxy<std::string, false>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(boost::filesystem::path)))
            return new QNodeProxy<boost::filesystem::path, false>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(bool)))
            return new QNodeProxy<bool, false>(parent, parameter);
    }

    if(parameter->type & EagleLib::Parameter::Output || parameter->type & EagleLib::Parameter::State)
    {
        if (parameter->typeInfo == Loki::TypeInfo(typeid(double)))
            return new QNodeProxy<double, true>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(int)))
            return new QNodeProxy<int, true>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(unsigned int)))
            return new QNodeProxy<unsigned int, true>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(unsigned char)))
            return new QNodeProxy<unsigned char, true>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(char)))
            return new QNodeProxy<char, true>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(std::string)))
            return new QNodeProxy<std::string, true>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(boost::filesystem::path)))
            return new QNodeProxy<boost::filesystem::path, true>(parent, parameter);

        if (parameter->typeInfo == Loki::TypeInfo(typeid(bool)))
            return new QNodeProxy<bool, true>(parent, parameter);
    }


	/*if (parameter->typeName == typeid(cv::Scalar).name())
	return new QNodeProxy<cv::Scalar>(parent, parameter);

	

	if (parameter->typeName == typeid(cv::Mat).name())
	return new QNodeProxy<cv::Mat>(parent, parameter);*/



	return nullptr;
}

