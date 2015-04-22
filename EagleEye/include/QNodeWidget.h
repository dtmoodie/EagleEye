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
#include <QLayout>
#include <qsizepolicy.h>
#include <boost/filesystem.hpp>
#include <qfiledialog.h>
#include <QLineEdit>
#include <QComboBox>
#include <type.h>
#include <boost/thread/mutex.hpp>
namespace Ui {
	class QNodeWidget;
}
class IQNodeInterop;
class QNodeWidget;
class IQNodeProxy;
class QInputProxy;

class CV_EXPORTS QNodeWidget : public QWidget
{
	Q_OBJECT
public:
	QNodeWidget(QWidget* parent = nullptr, EagleLib::Node* node = nullptr);
	~QNodeWidget();
	EagleLib::Node* getNode();
    void setSelected(bool state);
    void updateUi();
    // Used for thread safety
    void on_nodeUpdate();
    void on_logReceive(EagleLib::Verbosity verb, const std::string& msg, EagleLib::Node* node);
private slots:
    void on_enableClicked(bool state);
    void on_status(const std::string& msg, EagleLib::Node* node);
    void on_warning(const std::string& msg, EagleLib::Node* node);
    void on_error(const std::string& msg, EagleLib::Node* node);
    void on_critical(const std::string& msg, EagleLib::Node* node);
    void log(EagleLib::Verbosity verb, const std::string& msg, EagleLib::Node* node);
signals:
    void eLog(EagleLib::Verbosity verb, const std::string& msg, EagleLib::Node* node);
private:

	Ui::QNodeWidget* ui;
	ObjectId nodeId;
	ObjectId nodeParentId;
    QLineEdit* statusDisplay;
    QLineEdit* warningDisplay;
    QLineEdit* errorDisplay;
    QLineEdit* criticalDisplay;
    std::vector<boost::shared_ptr<IQNodeInterop>> interops;
};


class IQNodeProxy
{
public:
	IQNodeProxy(){}
    virtual ~IQNodeProxy(){}
    virtual void updateUi(bool init = false) = 0;
	virtual void onUiUpdated() = 0;
	virtual QWidget* getWidget() = 0;
    virtual QWidget* getTypename()
    {        return new QLabel(QString::fromStdString(type_info::demangle(parameter->typeInfo.name())));    }
	boost::shared_ptr<EagleLib::Parameter> parameter;
};
IQNodeProxy* dispatchParameter(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter, EagleLib::Node* node);


// Interface class for the interop class
class CV_EXPORTS IQNodeInterop: public QWidget
{
	Q_OBJECT
public:
    IQNodeInterop(boost::shared_ptr<EagleLib::Parameter> parameter_, QWidget* parent = nullptr, EagleLib::Node* node_= nullptr);
    virtual ~IQNodeInterop();
    virtual void updateUi();
    IQNodeProxy* proxy;
    boost::shared_ptr<EagleLib::Parameter> parameter;
private slots:
    void on_valueChanged(double value);
    void on_valueChanged(int value);
    void on_valueChanged(bool value);
    void on_valueChanged(QString value);
    void on_valueChanged();
    void onParameterUpdate(boost::shared_ptr<EagleLib::Parameter> parameter);
protected:
	QLabel* nameElement;	
    QGridLayout* layout;
    ObjectId nodeId;
};
// Class for UI elements relavent to finding valid input parameters
class QInputProxy: public IQNodeProxy
{
public:
    QInputProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter, EagleLib::Node* node_);
    virtual void onUiUpdated();
    virtual void updateUi(bool init = false);
    virtual QWidget* getWidget();
private:
    ObjectId nodeId;
    QComboBox* box;
};

// Proxy class for handling
template<typename T, bool display, typename Enable = void> class QNodeProxy : public IQNodeProxy
{

};



// **************************************************************************************************************
template<typename T>
class QNodeProxy<T, false, typename std::enable_if<std::is_floating_point<T>::value, void>::type> : public IQNodeProxy
{
public:
	QNodeProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_)
	{
		parameter = parameter_;
		box = new QDoubleSpinBox(parent);
		box->setMaximum(std::numeric_limits<T>::max());
		box->setMinimum(std::numeric_limits<T>::min());
        box->setMaximumWidth(100);
        updateUi(true);
		parent->connect(box, SIGNAL(valueChanged(double)), parent, SLOT(on_valueChanged(double)));
	}
    virtual void updateUi(bool init = false)
	{
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock) return;
        if(auto param = EagleLib::getParameterPtr<T>(parameter))
            box->setValue(*param);
	}
	virtual void onUiUpdated()
	{
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<T>(parameter))
            *param = box->value();
        parameter->changed = true;
	}
	virtual QWidget* getWidget() { return box; }
private:
	QDoubleSpinBox* box;
};
// **************************************************************************************************************
template<typename T>
class QNodeProxy<T, true, typename std::enable_if<std::is_floating_point<T>::value || std::is_integral<T>::value, void>::type> : public IQNodeProxy
{
public:
    QNodeProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_)
    {
        parameter = parameter_;
        box = new QLabel(parent);
        updateUi(true);
    }
    virtual void updateUi(bool init = false)
    {
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<T>(parameter))
            box->setText(QString::number(*param));
    }
    virtual void onUiUpdated()
    {
    }
    virtual QWidget* getWidget() { return box; }
private:
    QLabel* box;
};
// **************************************************************************************************************
template<typename T>
class QNodeProxy<T, false, typename std::enable_if<std::is_integral<T>::value, void>::type> : public IQNodeProxy
{
public:
	QNodeProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_)
	{
		parameter = parameter_;
        box = new QSpinBox(parent);
        if(std::numeric_limits<T>::max() > std::numeric_limits<int>::max())
            box->setMaximum(std::numeric_limits<int>::max());
        else
            box->setMaximum(std::numeric_limits<T>::max());
        box->setMinimum(std::numeric_limits<T>::min());
        updateUi(true);
		parent->connect(box, SIGNAL(valueChanged(int)), parent, SLOT(on_valueChanged(int)));
	}
    virtual void updateUi(bool init = false)
	{
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<T>(parameter))
            box->setValue(*param);
	}
	virtual void onUiUpdated()
	{
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<T>(parameter))
            *param = box->value();
        parameter->changed = true;
	}
	virtual QWidget* getWidget() { return box; }
private:
	QSpinBox* box;
};
// **************************************************************************************************************
template<>
class QNodeProxy<bool, false, void>: public IQNodeProxy
{
public:
	QNodeProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_)	
	{
        box = new QCheckBox(parent);
		parameter=parameter_;
        updateUi(true);
        parent->connect(box, SIGNAL(stateChanged(int)), parent, SLOT(on_valueChanged(int)));
	}
    virtual void updateUi(bool init = false)
    {
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<bool>(parameter))
            box->setChecked(*param);
    }
	virtual void onUiUpdated()
    {
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<bool>(parameter))
            *param = box->isChecked();
        parameter->changed = true;
    }
	virtual QWidget* getWidget() { return box; }
private:
	QCheckBox* box;
};
template<>
class QNodeProxy<bool, true, void>: public IQNodeProxy
{
public:
    QNodeProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_)
    {
        box = new QCheckBox(parent);
        box->setCheckable(false);
        parameter=parameter_;
        updateUi(true);
    }
    virtual void updateUi(bool init = false)
    {
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<bool>(parameter))
            box->setChecked(*param);
    }
    virtual void onUiUpdated()
    {	}
    virtual QWidget* getWidget() { return box; }
private:
    QCheckBox* box;
};
// **************************************************************************************************************
template<>
class QNodeProxy<std::string, false, void> : public IQNodeProxy
{
public:
	QNodeProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_)
	{
		box = new QLineEdit(parent);
        parameter = parameter_;
        updateUi(true);
        parent->connect(box, SIGNAL(editingFinished()), parent, SLOT(on_valueChanged()));

	}
    virtual void updateUi(bool init = false)
	{
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<std::string>(parameter))
            if(parameter->changed || init)
                box->setText(QString::fromStdString(*param));
	}
	virtual void onUiUpdated()
	{
        if(auto param = EagleLib::getParameterPtr<std::string>(parameter))
            *param = box->text().toStdString();
        parameter->changed = true;
	}
	virtual QWidget* getWidget() { return box; }
private:
	QLineEdit* box;
};
template<>
class QNodeProxy<std::string, true, void> : public IQNodeProxy
{
public:
    QNodeProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_)
    {
        box = new QLineEdit(parent);
        parameter = parameter_;
        updateUi(true);
    }
    virtual void updateUi(bool init = false)
    {
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<std::string>(parameter))
            if(parameter->changed || init)
                box->setText(QString::fromStdString(*param));
    }
    virtual void onUiUpdated()
    {
    }
    virtual QWidget* getWidget() { return box; }
private:
    QLineEdit* box;
};
// **************************************************************************************************************
template<>
class QNodeProxy<boost::filesystem::path, false, void> : public IQNodeProxy
{
public:
	QNodeProxy(IQNodeInterop* parent_, boost::shared_ptr<EagleLib::Parameter> parameter_)
	{
        parent = parent_;
		button = new QPushButton(parent);
        parameter = parameter_;
        updateUi(true);
        parent->connect(button, SIGNAL(clicked()), parent, SLOT(on_valueChanged()));

	}
    virtual void updateUi(bool init = false)
	{
        std::string fileName;
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<boost::filesystem::path>(parameter))
            fileName = param->string();
        lock.unlock();
		if (fileName.size())
			button->setText(QString::fromStdString(fileName));
		else
			button->setText("Select a file");
	}
	virtual void onUiUpdated()
	{
		QString filename = QFileDialog::getOpenFileName(parent, "Select file");
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<boost::filesystem::path>(parameter))
            *param = boost::filesystem::path(filename.toStdString());
        lock.unlock();
		button->setText(filename);
		button->setToolTip(filename);
		parameter->changed = true;
	}
	virtual QWidget* getWidget() { return button; }
private:
	QPushButton* button;
    QWidget* parent;
};

template<>
class QNodeProxy<boost::filesystem::path, true, void> : public IQNodeProxy
{
public:
    QNodeProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_)
    {
        parameter = parameter_;
        box = new QLineEdit(parent);
        updateUi();
    }
    virtual void updateUi(bool init = false)
    {
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<boost::filesystem::path>(parameter))
            box->setText(QString::fromStdString(param->string()));
    }
    virtual void onUiUpdated()
    {
    }
    virtual QWidget* getWidget() { return box; }
private:
    QLineEdit* box;
};

template<>
class QNodeProxy<boost::function<void(void)>, false, void> : public IQNodeProxy
{
public:
    QNodeProxy(IQNodeInterop* parent_, boost::shared_ptr<EagleLib::Parameter> parameter_)
    {
        parent = parent_;
        parameter = parameter_;
        button = new QPushButton(parent);
        parent->connect(button, SIGNAL(clicked()), parent, SLOT(on_valueChanged()));
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        button->setText(QString::fromStdString(parameter_->name));

    }
    virtual void updateUi(bool init = false)
    {

    }
    virtual void onUiUpdated()
    {
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        auto function = EagleLib::getParameterPtr<boost::function<void(void)>>(parameter);
        if(function)
        {
            (*function)();
        }
    }
    virtual QWidget* getWidget() { return button; }
private:
    QPushButton* button;
    QWidget* parent;
};

template<>
class QNodeProxy<EagleLib::EnumParameter, false, void> : public IQNodeProxy
{
public:
    QNodeProxy(IQNodeInterop* parent_, boost::shared_ptr<EagleLib::Parameter> parameter_)
    {
        parent = parent_;
        parameter = parameter_;
        box = new QComboBox(parent);
        updateUi(true);
        parent->connect(box, SIGNAL(currentIndexChanged(int)), parent_, SLOT(on_valueChanged(int)));

    }
    virtual void updateUi(bool init = false)
    {
        boost::mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        EagleLib::EnumParameter* param = EagleLib::getParameterPtr<EagleLib::EnumParameter>(parameter);
        if(init)
            for(int i = 0; i < param->enumerations.size(); ++i)
                box->addItem(QString::fromStdString(param->enumerations[i]));
    }
    virtual void onUiUpdated()
    {
        boost::mutex::scoped_lock lock(parameter->mtx);
        EagleLib::EnumParameter* param = EagleLib::getParameterPtr<EagleLib::EnumParameter>(parameter);
        param->currentSelection = param->values[box->currentIndex()];
        parameter->changed = true;
    }
    virtual QWidget* getWidget(){return box;}

private:
    QWidget* parent;
    QComboBox* box;
};

