#pragma once
#include "nodes/Node.h"
#include "qgraphicsitem.h"
#include "qwidget.h"
#include <boost/type_traits.hpp>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <QLineEdit>
#include <qgridlayout.h>
#include <qsizepolicy.h>
#include <boost/filesystem.hpp>
#include <qfiledialog.h>
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

	Ui::QNodeWidget* ui;
	ObjectId nodeId;
	ObjectId nodeParentId;
};


class IQNodeProxy
{
public:
	IQNodeProxy(){}
	virtual ~IQNodeProxy(){};
	virtual void updateUi() = 0;
	virtual void onUiUpdated() = 0;
	virtual QWidget* getWidget() = 0;
	boost::shared_ptr<EagleLib::Parameter> parameter;
};
class IQNodeInterop;
IQNodeProxy* dispatchParameter(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter);


// Interface class for the interop class
class CV_EXPORTS IQNodeInterop: public QWidget
{
	Q_OBJECT
public:
	IQNodeInterop(boost::shared_ptr<EagleLib::Parameter> parameter_, QWidget* parent = nullptr) :
		QWidget(parent)
	{
		layout = new QGridLayout(this);
		layout->setVerticalSpacing(0);
		nameElement = new QLabel(QString::fromStdString(parameter_->name), parent);
		proxy = dispatchParameter(this, parameter_);
		if (proxy)
			layout->addWidget(proxy->getWidget(), 0, 1);
		layout->addWidget(nameElement, 0, 0);
		nameElement->setToolTip(QString::fromStdString(parameter_->toolTip));
	}
	virtual ~IQNodeInterop()
	{
		delete proxy;
	}
	virtual void updateUi()
	{
		if (proxy)
			proxy->updateUi();
	}

private slots:
	void on_valueChanged(double value)
	{
		if (proxy)
			proxy->onUiUpdated();
	}
	void on_valueChanged(int value)
	{
		if (proxy)
			proxy->onUiUpdated();
	}
	void on_valueChanged(bool value)
	{
		if (proxy)
			proxy->onUiUpdated();
	}
	void on_valueChanged(QString value)
	{
		if (proxy)
			proxy->onUiUpdated();
	}
	void on_valueChanged()
	{
		if (proxy)
			proxy->onUiUpdated();
	}

protected:
	QLabel* nameElement;	
	IQNodeProxy* proxy;
	QGridLayout* layout;
};

// Proxy class for handling
template<typename T, typename Enable = void> class QNodeProxy : public IQNodeProxy
{

};



// Proxy class for handling
template<typename T>
class QNodeProxy<T, typename std::enable_if<std::is_floating_point<T>::value, void>::type> : public IQNodeProxy
{
public:
	QNodeProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_)
	{
		parameter = parameter_;
		box = new QDoubleSpinBox(parent);
		box->setMaximum(std::numeric_limits<T>::max());
		box->setMinimum(std::numeric_limits<T>::min());
		box->setValue(*EagleLib::getParameter<T>(parameter_));
		parent->connect(box, SIGNAL(valueChanged(double)), parent, SLOT(on_valueChanged(double)));
	}

	virtual void updateUi()
	{
		box->setValue(*EagleLib::getParameter<T>(parameter));
	}
	virtual void onUiUpdated()
	{
		*EagleLib::getParameter<T>(parameter) = box->value();
	}
	virtual QWidget* getWidget() { return box; }
private:
	QDoubleSpinBox* box;
};
template<typename T>
class QNodeProxy<T, typename std::enable_if<std::is_integral<T>::value, void>::type> : public IQNodeProxy
{
public:
	QNodeProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_)
	{
		parameter = parameter_;
		box = new QSpinBox(parent);
		box->setMaximum(std::numeric_limits<T>::max());
		box->setMinimum(std::numeric_limits<T>::min());
		box->setValue(*EagleLib::getParameter<T>(parameter_));
		parent->connect(box, SIGNAL(valueChanged(int)), parent, SLOT(on_valueChanged(int)));
	}
	virtual void updateUi()
	{
		box->setValue(*EagleLib::getParameter<T>(parameter));
	}
	virtual void onUiUpdated()
	{
		*EagleLib::getParameter<T>(parameter) = box->value();
	}
	virtual QWidget* getWidget() { return box; }
private:
	QSpinBox* box;
};
template<>
class QNodeProxy<bool, void>: public IQNodeProxy
{
public:
	QNodeProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_)	
	{
		box = new QCheckBox(parent);
		parent->connect(box, SIGNAL(stateChanged(int)), parent, SLOT(on_valueChanged(int)));
		parameter=parameter_;
	}
	virtual void updateUi()
	{	box->setChecked(*EagleLib::getParameter<bool>(parameter));	}
	virtual void onUiUpdated()
	{	*EagleLib::getParameter<bool>(parameter) = box->isChecked();	}
	virtual QWidget* getWidget() { return box; }
private:
	QCheckBox* box;
};

template<>
class QNodeProxy<std::string, void> : public IQNodeProxy
{
public:
	QNodeProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_)
	{
		box = new QLineEdit(parent);
		box->setText(QString::fromStdString(*EagleLib::getParameter<std::string>(parameter_)));
		parent->connect(box, SIGNAL(textChanged(QString)), parent, SLOT(on_valueChanged(QString)));
		parameter = parameter_;
	}
	virtual void updateUi()
	{
		box->setText(QString::fromStdString(*EagleLib::getParameter<std::string>(parameter)));
	}
	virtual void onUiUpdated()
	{
		*EagleLib::getParameter<std::string>(parameter) = box->text().toStdString();
	}
	virtual QWidget* getWidget() { return box; }
private:
	QLineEdit* box;
};

template<>
class QNodeProxy<boost::filesystem::path, void> : public IQNodeProxy
{
public:
	QNodeProxy(IQNodeInterop* parent_, boost::shared_ptr<EagleLib::Parameter> parameter_)
	{
		parent = parent_;
		button = new QPushButton(parent);
		button->setText(QString::fromStdString(EagleLib::getParameter<boost::filesystem::path>(parameter_)->string()));
		if (!button->text().size())
			button->setText("Select a file");
		parent->connect(button, SIGNAL(clicked()), parent, SLOT(on_valueChanged()));
		parameter = parameter_;
	}
	virtual void updateUi()
	{
		std::string fileName = EagleLib::getParameter<boost::filesystem::path>(parameter)->string();
		if (fileName.size())
			button->setText(QString::fromStdString(fileName));
		else
			button->setText("Select a file");
	}
	virtual void onUiUpdated()
	{
		QString filename = QFileDialog::getOpenFileName(parent, "Select file");
		*EagleLib::getParameter<boost::filesystem::path>(parameter) = boost::filesystem::path(filename.toStdString());
		button->setText(filename);
		button->setToolTip(filename);
		parameter->changed = true;
	}
	virtual QWidget* getWidget() { return button; }
private:
	QPushButton* button;
	QWidget* parent;
	
};