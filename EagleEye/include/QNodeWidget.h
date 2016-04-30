#pragma once
#include "EagleLib/nodes/Node.h"
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
#include <parameters/type.h>
#include <boost/thread/recursive_mutex.hpp>
#include <parameters/UI/Qt.hpp>
#include <EagleLib/rcc/shared_ptr.hpp>
#include <EagleLib/DataStreamManager.h>

namespace Ui {
	class QNodeWidget;
    class DataStreamWidget;
}
class IQNodeInterop;
class QNodeWidget;
class IQNodeProxy;
class QInputProxy;

// Class for UI elements relavent to finding valid input parameters
class QInputProxy : public QWidget
{
	Q_OBJECT
    int prevIdx;
	void onParamDelete(Parameters::Parameter* parameter);
public:
	Parameters::InputParameter* inputParameter;
	QInputProxy(Parameters::Parameter* parameter, EagleLib::Nodes::Node::Ptr node_, QWidget* parent);
	virtual void updateUi(bool init = false);
	virtual QWidget* getWidget(int num = 0);
private slots:
	void on_valueChanged(int);
private:
	std::shared_ptr<Signals::connection> bc;
	std::shared_ptr<Signals::connection> dc;
	EagleLib::Nodes::Node::Ptr node;
	QComboBox* box;
};

class CV_EXPORTS QNodeWidget : public QWidget
{
	Q_OBJECT
public:
    QNodeWidget(QWidget* parent = nullptr, EagleLib::Nodes::Node::Ptr node = EagleLib::Nodes::Node::Ptr());
	~QNodeWidget();
    EagleLib::Nodes::Node::Ptr getNode();
    void setSelected(bool state);
    void updateUi(bool parameterUpdate = false, EagleLib::Nodes::Node* node = nullptr);
    // Used for thread safety
    void on_nodeUpdate();
    void on_logReceive(boost::log::trivial::severity_level verb, const std::string& msg);
    bool eventFilter(QObject *object, QEvent *event);
    void addParameterWidgetMap(QWidget* widget, Parameters::Parameter::Ptr param);
    QWidget* mainWindow;
private slots:
    void on_enableClicked(bool state);
    void on_profileClicked(bool state);


	void log(boost::log::trivial::severity_level verb, const std::string& msg);
signals:
	void eLog(boost::log::trivial::severity_level verb, const std::string& msg);
	void parameterClicked(Parameters::Parameter* param, QPoint pos);
private:
	QLineEdit* profileDisplay;
    QLineEdit* traceDisplay;
    QLineEdit* debugDisplay;
    QLineEdit* infoDisplay;
    QLineEdit* warningDisplay;
    QLineEdit* errorDisplay;


	std::map<QWidget*, Parameters::Parameter*> widgetParamMap;
	Ui::QNodeWidget* ui;
    EagleLib::Nodes::Node::Ptr node;
	std::vector<Parameters::UI::qt::IParameterProxy::Ptr> parameterProxies;
	std::vector<QInputProxy*> inputProxies;
    //std::vector<boost::shared_ptr<IQNodeInterop>> interops;
    QNodeWidget* parentWidget;
    std::vector<QNodeWidget*> childWidgets;
    std::shared_ptr<Signals::connection> log_connection;
};

class CV_EXPORTS DataStreamWidget: public QWidget
{
    Q_OBJECT
public:
    DataStreamWidget(QWidget* parent = nullptr, EagleLib::DataStream::Ptr stream = EagleLib::DataStream::Ptr());
    ~DataStreamWidget();

    EagleLib::DataStream::Ptr GetStream();
    void SetSelected(bool state);
    void update_ui();

signals:


private:
    EagleLib::DataStream::Ptr _dataStream;
    Ui::DataStreamWidget* ui;
    std::vector<QInputProxy*> inputProxies;
    std::vector<Parameters::UI::qt::IParameterProxy::Ptr> parameterProxies;
    std::map<QWidget*, Parameters::Parameter*> widgetParamMap;
};

class DraggableLabel: public QLabel
{
    Parameters::Parameter::Ptr param;
public:
    DraggableLabel(QString name, Parameters::Parameter::Ptr param_);
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
    {        return new QLabel(QString::fromStdString(parameter->GetTypeInfo().name()));    }
	boost::shared_ptr<Parameters::Parameter> parameter;
};
IQNodeProxy* dispatchParameter(IQNodeInterop* parent, Parameters::Parameter::Ptr parameter, EagleLib::Nodes::Node::Ptr node);


// Interface class for the interop class
class CV_EXPORTS IQNodeInterop: public QWidget
{
	Q_OBJECT
public:
    IQNodeInterop(Parameters::Parameter::Ptr parameter_, QNodeWidget* parent = nullptr, EagleLib::Nodes::Node::Ptr node_= EagleLib::Nodes::Node::Ptr());
    virtual ~IQNodeInterop();

    IQNodeProxy* proxy;
	Parameters::Parameter::Ptr parameter;
    std::shared_ptr<Signals::connection> bc;
    boost::posix_time::ptime previousUpdateTime;
public slots:
    virtual void updateUi();
private slots:
    void on_valueChanged(double value);
    void on_valueChanged(int value);
    void on_valueChanged(bool value);
    void on_valueChanged(QString value);
    void on_valueChanged();
	void onParameterUpdate(Parameters::Parameter::Ptr parameter);
    void onParameterUpdate();
signals:
    void updateNeeded();
protected:
	QLabel* nameElement;	
    QGridLayout* layout;
    EagleLib::Nodes::Node::Ptr node;
};

