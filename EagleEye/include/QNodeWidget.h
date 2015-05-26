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
#include <boost/thread/recursive_mutex.hpp>
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
    QNodeWidget(QWidget* parent = nullptr, EagleLib::Node::Ptr node = EagleLib::Node::Ptr());
	~QNodeWidget();
    EagleLib::Node::Ptr getNode();
    void setSelected(bool state);
    void updateUi(bool parameterUpdate = false);
    // Used for thread safety
    void on_nodeUpdate();
    void on_logReceive(EagleLib::Verbosity verb, const std::string& msg, EagleLib::Node* node);
    bool eventFilter(QObject *object, QEvent *event);
    void addParameterWidgetMap(QWidget* widget, EagleLib::Parameter::Ptr param);
    QWidget* mainWindow;
private slots:
    void on_enableClicked(bool state);
    void on_profileClicked(bool state);
    void on_status(const std::string& msg, EagleLib::Node* node);
    void on_warning(const std::string& msg, EagleLib::Node* node);
    void on_error(const std::string& msg, EagleLib::Node* node);
    void on_critical(const std::string& msg, EagleLib::Node* node);
    void on_profile(const std::string& msg, EagleLib::Node* node);

    void log(EagleLib::Verbosity verb, const std::string& msg, EagleLib::Node* node);
signals:
    void eLog(EagleLib::Verbosity verb, const std::string& msg, EagleLib::Node* node);
    void parameterClicked(EagleLib::Parameter::Ptr param);
private:
    std::map<QWidget*, EagleLib::Parameter::Ptr> widgetParamMap;
	Ui::QNodeWidget* ui;
    EagleLib::Node::Ptr node;
    QLineEdit* profileDisplay;
    QLineEdit* statusDisplay;
    QLineEdit* warningDisplay;
    QLineEdit* errorDisplay;
    QLineEdit* criticalDisplay;
    std::vector<boost::shared_ptr<IQNodeInterop>> interops;
    QNodeWidget* parentWidget;
    std::vector<QNodeWidget*> childWidgets;
};

class DraggableLabel: public QLabel
{
    EagleLib::Parameter::Ptr param;
public:
    DraggableLabel(QString name, EagleLib::Parameter::Ptr param_);
    void dropEvent(QDropEvent* event);
    void dragLeaveEvent(QDragLeaveEvent* event);
    void dragMoveEvent(QDragMoveEvent* event);
};

class IQNodeProxy
{
public:
	IQNodeProxy(){}
    virtual ~IQNodeProxy(){}
    virtual void updateUi(bool init = false) = 0;
    virtual void onUiUpdated(QWidget* widget = 0) = 0;
    virtual QWidget* getWidget(int num = 0) = 0;
    virtual int getNumWidgets(){return 1;}
    virtual QWidget* getTypename()
#ifdef _MSC_VER
	{
		return new QLabel(QString::fromStdString(parameter->typeInfo.name()));
	}
#else
    {        return new QLabel(QString::fromStdString(TypeInfo::demangle(parameter->typeInfo.name())));    }
#endif
	boost::shared_ptr<EagleLib::Parameter> parameter;
};
IQNodeProxy* dispatchParameter(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter, EagleLib::Node::Ptr node);


// Interface class for the interop class
class CV_EXPORTS IQNodeInterop: public QWidget
{
	Q_OBJECT
public:
    IQNodeInterop(boost::shared_ptr<EagleLib::Parameter> parameter_, QNodeWidget* parent = nullptr, EagleLib::Node::Ptr node_= EagleLib::Node::Ptr());
    virtual ~IQNodeInterop();

    IQNodeProxy* proxy;
    boost::shared_ptr<EagleLib::Parameter> parameter;
    boost::signals2::connection bc;
    boost::posix_time::ptime previousUpdateTime;
public slots:
    virtual void updateUi();
private slots:
    void on_valueChanged(double value);
    void on_valueChanged(int value);
    void on_valueChanged(bool value);
    void on_valueChanged(QString value);
    void on_valueChanged();
    void onParameterUpdate(boost::shared_ptr<EagleLib::Parameter> parameter);
    void onParameterUpdate();
signals:
    void updateNeeded();
protected:
	QLabel* nameElement;	
    QGridLayout* layout;
    EagleLib::Node::Ptr node;
};


// Class for UI elements relavent to finding valid input parameters
class QInputProxy: public IQNodeProxy
{
public:
    QInputProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter, EagleLib::Node::Ptr node_);
    virtual void onUiUpdated(QWidget* widget);
    virtual void updateUi(bool init = false);
    virtual QWidget* getWidget(int num = 0);
private:
    EagleLib::Node::Ptr node;
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
        box->setSingleStep(0.01);
        updateUi(true);
		parent->connect(box, SIGNAL(valueChanged(double)), parent, SLOT(on_valueChanged(double)));
	}
    virtual void updateUi(bool init = false)
	{
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock) return;
        if(auto param = EagleLib::getParameterPtr<T>(parameter))
            box->setValue(*param);
	}
    virtual void onUiUpdated(QWidget* widget)
	{
        boost::recursive_mutex::scoped_lock lock(parameter->mtx);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<T>(parameter))
            *param = box->value();
        parameter->changed = true;
	}
    virtual QWidget* getWidget(int num = 0) { return box; }
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
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<T>(parameter))
            box->setText(QString::number(*param));
    }
    virtual void onUiUpdated(QWidget* widget)
    {
    }
    virtual QWidget* getWidget(int num = 0) { return box; }
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
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<T>(parameter))
            box->setValue(*param);
	}
    virtual void onUiUpdated(QWidget* widget)
	{
        boost::recursive_mutex::scoped_lock lock(parameter->mtx);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<T>(parameter))
            *param = box->value();
        parameter->changed = true;
	}
    virtual QWidget* getWidget(int num = 0) { return box; }
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
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<bool>(parameter))
            box->setChecked(*param);
    }
    virtual void onUiUpdated(QWidget* widget)
    {
        boost::recursive_mutex::scoped_lock lock(parameter->mtx);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<bool>(parameter))
            *param = box->isChecked();
        parameter->changed = true;
    }
    virtual QWidget* getWidget(int num = 0) { return box; }
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
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<bool>(parameter))
            box->setChecked(*param);
    }
    virtual void onUiUpdated(QWidget* widget)
    {	}
    virtual QWidget* getWidget(int num = 0) { return box; }
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
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<std::string>(parameter))
            if(parameter->changed || init)
                box->setText(QString::fromStdString(*param));
	}
    virtual void onUiUpdated(QWidget* widget)
	{
        boost::recursive_mutex::scoped_lock lock(parameter->mtx);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<std::string>(parameter))
            *param = box->text().toStdString();
        parameter->changed = true;
	}
    virtual QWidget* getWidget(int num = 0) { return box; }
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
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<std::string>(parameter))
            if(parameter->changed || init)
                box->setText(QString::fromStdString(*param));
    }
    virtual void onUiUpdated(QWidget* widget)
    {
    }
    virtual QWidget* getWidget(int num = 0) { return box; }
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
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
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
    virtual void onUiUpdated(QWidget* widget)
    {
        QString filename;
        QNodeWidget* nodeWidget = dynamic_cast<QNodeWidget*>(parent->parentWidget());

        if(nodeWidget)
            filename = QFileDialog::getSaveFileName(nodeWidget->mainWindow, "Select file");
        else
            filename = QFileDialog::getSaveFileName(parent, "Select file");
        boost::recursive_mutex::scoped_lock lock(parameter->mtx);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<boost::filesystem::path>(parameter))
            *param = boost::filesystem::path(filename.toStdString());
        lock.unlock();
		button->setText(filename);
		button->setToolTip(filename);
		parameter->changed = true;
	}
    virtual QWidget* getWidget(int num = 0) { return button; }
private:
	QPushButton* button;
    IQNodeInterop* parent;
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
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        if(auto param = EagleLib::getParameterPtr<boost::filesystem::path>(parameter))
            box->setText(QString::fromStdString(param->string()));
    }
    virtual void onUiUpdated(QWidget* widget)
    {
    }
    virtual QWidget* getWidget(int num = 0) { return box; }
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
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        button->setText(QString::fromStdString(parameter_->name));

    }
    virtual void updateUi(bool init = false)
    {

    }
    virtual void onUiUpdated(QWidget* widget)
    {
        boost::recursive_mutex::scoped_lock lock(parameter->mtx);
        if(!lock)
            return;
        auto function = EagleLib::getParameterPtr<boost::function<void(void)>>(parameter);
        if(function)
        {
            (*function)();
        }
    }
    virtual QWidget* getWidget(int num = 0) { return button; }
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
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        EagleLib::EnumParameter* param = EagleLib::getParameterPtr<EagleLib::EnumParameter>(parameter);
        if(init)
            for(int i = 0; i < param->enumerations.size(); ++i)
                box->addItem(QString::fromStdString(param->enumerations[i]));
        box->setCurrentIndex(param->currentSelection);

    }
    virtual void onUiUpdated(QWidget* widget)
    {
        boost::recursive_mutex::scoped_lock lock(parameter->mtx);
        if(!lock)
            return;
        EagleLib::EnumParameter* param = EagleLib::getParameterPtr<EagleLib::EnumParameter>(parameter);
        param->currentSelection = box->currentIndex();
        parameter->changed = true;
    }
    virtual QWidget* getWidget(int num = 0){return box;}

private:
    QWidget* parent;
    QComboBox* box;
};

template<>
class QNodeProxy<EagleLib::EnumParameter, true, void> : public IQNodeProxy
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
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        EagleLib::EnumParameter* param = EagleLib::getParameterPtr<EagleLib::EnumParameter>(parameter);
        if(init)
        {
            for(int i = 0; i < param->enumerations.size(); ++i)
                box->addItem(QString::fromStdString(param->enumerations[i]));
        }
        if(parameter->changed)
        {
            box->setCurrentIndex(param->currentSelection);
        }
    }
    virtual void onUiUpdated(QWidget* widget)
    {

    }
    virtual QWidget* getWidget(int num = 0){return box;}

private:
    QWidget* parent;
    QComboBox* box;
};

template<>
class QNodeProxy<cv::Scalar, false, void> : public IQNodeProxy
{
public:
    QNodeProxy(IQNodeInterop* parent_, boost::shared_ptr<EagleLib::Parameter> parameter_)
    {
        parent = parent_;
        parameter = parameter_;
        boxes[0] = new QDoubleSpinBox(parent);
        boxes[1] = new QDoubleSpinBox(parent);
        boxes[2] = new QDoubleSpinBox(parent);

        boxes[0]->setMinimum(std::numeric_limits<double>::min());
        boxes[0]->setMaximum(std::numeric_limits<double>::max());
        boxes[1]->setMinimum(std::numeric_limits<double>::min());
        boxes[1]->setMaximum(std::numeric_limits<double>::max());
        boxes[2]->setMinimum(std::numeric_limits<double>::min());
        boxes[2]->setMaximum(std::numeric_limits<double>::max());

        boxes[0]->setMaximumWidth(100);
        boxes[1]->setMaximumWidth(100);
        boxes[2]->setMaximumWidth(100);

        updateUi(true);
        parent->connect(boxes[0], SIGNAL(valueChanged(double)), parent_, SLOT(on_valueChanged(double)));
        parent->connect(boxes[1], SIGNAL(valueChanged(double)), parent_, SLOT(on_valueChanged(double)));
        parent->connect(boxes[2], SIGNAL(valueChanged(double)), parent_, SLOT(on_valueChanged(double)));
    }
    virtual void updateUi(bool init = false)
    {
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        cv::Scalar* param = EagleLib::getParameterPtr<cv::Scalar>(parameter);
        boxes[0]->setValue(param->val[0]);
        boxes[1]->setValue(param->val[1]);
        boxes[2]->setValue(param->val[2]);
    }
    virtual void onUiUpdated(QWidget* widget)
    {
        boost::recursive_mutex::scoped_lock lock(parameter->mtx);
        if(!lock)
            return;
        cv::Scalar* param = EagleLib::getParameterPtr<cv::Scalar>(parameter);
        param->val[0] = boxes[0]->value();
        param->val[1] = boxes[1]->value();
        param->val[2] = boxes[2]->value();
        parameter->changed = true;
    }
    virtual QWidget* getWidget(int num = 0){
        return boxes[num];
    }
    virtual int getNumWidgets()
    {
        return 3;
    }

private:
    QWidget* parent;
    QDoubleSpinBox* boxes[3];
};
template<>
class QNodeProxy<cv::Scalar, true, void> : public IQNodeProxy
{
public:
    QNodeProxy(IQNodeInterop* parent_, boost::shared_ptr<EagleLib::Parameter> parameter_)
    {
        parent = parent_;
        parameter = parameter_;
        boxes[0] = new QLabel(parent);
        boxes[1] = new QLabel(parent);
        boxes[2] = new QLabel(parent);
        updateUi(true);
    }
    virtual void updateUi(bool init = false)
    {
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        cv::Scalar* param = EagleLib::getParameterPtr<cv::Scalar>(parameter);
        boxes[0]->setText(QString::number(param->val[0]));
        boxes[1]->setText(QString::number(param->val[1]));
        boxes[2]->setText(QString::number(param->val[2]));
    }
    virtual void onUiUpdated(QWidget* widget)
    {

    }
    virtual QWidget* getWidget(int num = 0){
        return boxes[num];
    }
    virtual int getNumWidgets()
    {
        return 3;
    }

private:
    QWidget* parent;
    QLabel* boxes[3];
};
template<typename T>
class QNodeProxy<std::vector<T>, false, void> : public IQNodeProxy
{
public:
    QNodeProxy(IQNodeInterop* parent_, boost::shared_ptr<EagleLib::Parameter> parameter_)
    {
        parent = parent_;
        parameter = parameter_;

        size = new QLabel(parent_);
        index = new QSpinBox(parent_);
        value = new QDoubleSpinBox(parent_);
        value->setMaximum(std::numeric_limits<T>::max());
        value->setMinimum(std::numeric_limits<T>::min());
        value->setMaximumWidth(100);
        prevIndex = 0;
        updateUi(true);
        parent_->connect(index, SIGNAL(valueChanged(int)), parent_, SLOT(on_valueChanged(int)));
        parent_->connect(value, SIGNAL(valueChanged(double)), parent_, SLOT(on_valueChanged(double)));
    }
    virtual void updateUi(bool init = false)
    {
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        std::vector<T>* param = EagleLib::getParameterPtr<std::vector<T>>(parameter);
        if(param->size())
            index->setMaximum(param->size()-1);
        else
            index->setMaximum(0);
        index->setMinimum(0);
        if(index->value() < param->size())
        {
            value->setValue((*param)[index->value()]);
        }
    }
    virtual void onUiUpdated(QWidget* widget)
    {

        boost::recursive_mutex::scoped_lock lock(parameter->mtx);
        if(!lock)
            return;
        std::vector<T>* param = EagleLib::getParameterPtr<std::vector<T>>(parameter);
        if(widget == index)
        {
            if(index->value() < param->size())
            {
                value->setValue((*param)[index->value()]);
            }
            return;
        }
        if(index->value() < param->size())
        {
            (*param)[index->value()] = value->value();
        }
        parameter->changed = true;

    }
    virtual QWidget* getWidget(int num = 0)
    {
        if(num == 0)
            return size;
        if(num == 1)
            return index;
        if(num == 2)
            return value;
        return nullptr;
    }
    virtual int getNumWidgets()
    {
        return 3;
    }

private:
    QWidget* parent;
    QLabel* size;
    QSpinBox* index;
    QDoubleSpinBox* value;
    int prevIndex;
};

template<typename T>
class QNodeProxy<std::vector<T>, true, void> : public IQNodeProxy
{
public:
    QNodeProxy(IQNodeInterop* parent_, boost::shared_ptr<EagleLib::Parameter> parameter_)
    {
        parent = parent_;
        parameter = parameter_;
        size = new QLabel(parent_);
        index = new QSpinBox(parent_);
        value = new QLabel(parent_);

        parent_->connect(index, SIGNAL(valueChanged(int)), parent_, SLOT(on_valueChanged(int)));
        updateUi(true);
    }
    virtual void updateUi(bool init = false)
    {
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        std::vector<T>* param = EagleLib::getParameterPtr<std::vector<T>>(parameter);
        if(param->size())
            index->setMaximum(param->size()-1);
        else
            index->setMaximum(0);
        index->setMinimum(0);
        if(index->value() < param->size())
        {
            value->setText(QString::number(((*param)[index->value()])));
        }
    }
    virtual void onUiUpdated(QWidget* widget)
    {
        updateUi();
    }
    virtual QWidget* getWidget(int num = 0)
    {
        if(num == 0)
            return size;
        if(num == 1)
            return index;
        if(num == 2)
            return value;
        return nullptr;
    }
    virtual int getNumWidgets()
    {
        return 3;
    }

private:
    QWidget* parent;
    QLabel* size;
    QSpinBox* index;
    QLabel* value;
};

template<typename T1, typename T2>
class QNodeProxy<std::vector<std::pair<T1, T2>>, false, void> : public IQNodeProxy
{
public:
    QNodeProxy(IQNodeInterop* parent_, boost::shared_ptr<EagleLib::Parameter> parameter_)
    {
        parent = parent_;
        parameter = parameter_;

        size = new QLabel(parent_);
        index = new QSpinBox(parent_);
        value1 = new QDoubleSpinBox(parent_);
        value1->setMaximum(std::numeric_limits<T1>::max());
        value1->setMinimum(std::numeric_limits<T1>::min());
        value1->setMaximumWidth(100);

        value2 = new QDoubleSpinBox(parent_);
        value2->setMaximum(std::numeric_limits<T2>::max());
        value2->setMinimum(std::numeric_limits<T2>::min());
        value2->setMaximumWidth(100);

        prevIndex = 0;
        updateUi(true);
        parent_->connect(index, SIGNAL(valueChanged(int)), parent_, SLOT(on_valueChanged(int)));
        parent_->connect(value1, SIGNAL(valueChanged(double)), parent_, SLOT(on_valueChanged(double)));
        parent_->connect(value2, SIGNAL(valueChanged(double)), parent_, SLOT(on_valueChanged(double)));
    }
    virtual void updateUi(bool init = false)
    {
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        std::vector<std::pair<T1,T2>>* param = EagleLib::getParameterPtr<std::vector<std::pair<T1, T2>>>(parameter);
        if(param->size())
            index->setMaximum(param->size()-1);
        else
            index->setMaximum(0);
        index->setMinimum(0);
        if(index->value() < param->size())
        {
            value1->setValue((*param)[index->value()].first);
            value2->setValue((*param)[index->value()].second);
        }
    }
    virtual void onUiUpdated(QWidget* widget)
    {

        boost::recursive_mutex::scoped_lock lock(parameter->mtx);
        if(!lock)
            return;
        std::vector<std::pair<T1, T2>>* param = EagleLib::getParameterPtr<std::vector<std::pair<T1, T2>>>(parameter);
        if(widget == index)
        {
            if(index->value() < param->size())
            {
                value1->setValue((*param)[index->value()].first);
                value2->setValue((*param)[index->value()].second);
            }
            return;
        }
        if(widget == value1)
        {
            if(index->value() < param->size())
            {
                (*param)[index->value()].first = value1->value();
            }
        }
        if(widget == value2)
        {
            if(index->value() < param->size())
            {
                (*param)[index->value()].second = value2->value();
            }
        }

        parameter->changed = true;
    }
    virtual QWidget* getWidget(int num = 0)
    {
        if(num == 0)
            return size;
        if(num == 1)
            return index;
        if(num == 2)
            return value1;
        if(num == 3)
            return value2;
        return nullptr;
    }
    virtual int getNumWidgets()
    {
        return 4;
    }

private:
    QWidget* parent;
    QLabel* size;
    QSpinBox* index;
    QDoubleSpinBox* value1;
    QDoubleSpinBox* value2;
    int prevIndex;
};
template<typename T1, typename T2>
class QNodeProxy<std::vector<std::pair<T1, T2>>, true, void> : public IQNodeProxy
{
public:
    QNodeProxy(IQNodeInterop* parent_, boost::shared_ptr<EagleLib::Parameter> parameter_)
    {
        parent = parent_;
        parameter = parameter_;

        size = new QLabel(parent_);
        index = new QSpinBox(parent_);
        value1 = new QLabel(parent_);
        value2 = new QLabel(parent_);

        prevIndex = 0;
        updateUi(true);
        parent_->connect(index, SIGNAL(valueChanged(int)), parent_, SLOT(on_valueChanged(int)));
    }
    virtual void updateUi(bool init = false)
    {
        boost::recursive_mutex::scoped_lock lock(parameter->mtx, boost::try_to_lock);
        if(!lock)
            return;
        std::vector<std::pair<T1,T2>>* param = EagleLib::getParameterPtr<std::vector<std::pair<T1, T2>>>(parameter);
        if(param->size())
            index->setMaximum(param->size()-1);
        else
            index->setMaximum(0);
        index->setMinimum(0);
        if(index->value() < param->size())
        {
            value1->setText(QString::number((*param)[index->value()].first));
            value2->setText(QString::number((*param)[index->value()].second));
        }
    }
    virtual void onUiUpdated(QWidget* widget)
    {

        boost::recursive_mutex::scoped_lock lock(parameter->mtx);
        if(!lock)
            return;
        std::vector<std::pair<T1, T2>>* param = EagleLib::getParameterPtr<std::vector<std::pair<T1, T2>>>(parameter);
        if(widget == index)
        {
            if(index->value() < param->size())
            {
                value1->setText(QString::number((*param)[index->value()].first));
                value2->setText(QString::number((*param)[index->value()].second));
            }
            return;
        }
        parameter->changed = true;
    }
    virtual QWidget* getWidget(int num = 0)
    {
        if(num == 0)
            return size;
        if(num == 1)
            return index;
        if(num == 2)
            return value1;
        if(num == 3)
            return value2;
        return nullptr;
    }
    virtual int getNumWidgets()
    {
        return 4;
    }

private:
    QWidget* parent;
    QLabel* size;
    QSpinBox* index;
    QLabel* value1;
    QLabel* value2;
    int prevIndex;
};
