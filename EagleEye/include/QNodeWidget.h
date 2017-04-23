#pragma once
#include "Aquila/Nodes/Node.h"
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

#include <boost/thread/recursive_mutex.hpp>

#include <RuntimeObjectSystem/shared_ptr.hpp>
#include <MetaObject/Parameters/UI/Qt/IParameterProxy.hpp>

namespace Ui {
    class QNodeWidget;
    class DataStreamWidget;
}
class IQNodeInterop;
class QNodeWidget;
class IQNodeProxy;
class QInputProxy;

// Class for UI elements relavent to finding valid input parameters
// TODO update in case input parameter is deleted on object recompile
class QInputProxy : public QWidget
{
    Q_OBJECT
public:
    QInputProxy(mo::IParameter* parameter, rcc::weak_ptr<aq::Nodes::Node> node_, QWidget* parent);
    void updateParameter(mo::IParameter* parameter);
    virtual void updateUi(mo::IParameter*, mo::Context*, bool init = false);
    virtual QWidget* getWidget(int num = 0);
    bool eventFilter(QObject* obj, QEvent* event);

    mo::InputParameter* inputParameter;
private slots:
    void on_valueChanged(int);
private:
    int prevIdx;
    void onParamDelete(mo::IParameter const* parameter);
    std::shared_ptr<mo::Connection> update_connection;
    std::shared_ptr<mo::Connection> delete_connection;
    rcc::weak_ptr<aq::Nodes::Node> node;
    QComboBox* box;
    mo::TypedSlot<void(mo::IParameter*, mo::Context*)> onParameterUpdateSlot;
    mo::TypedSlot<void(mo::IParameter const*)> onParameterDeleteSlot;
};

class CV_EXPORTS QNodeWidget : public QWidget
{
    Q_OBJECT
public:
    QNodeWidget(QWidget* parent = nullptr, rcc::weak_ptr<aq::Nodes::Node> node = rcc::weak_ptr<aq::Nodes::Node>());
    ~QNodeWidget();
    rcc::weak_ptr<aq::Nodes::Node> getNode();
    void setSelected(bool state);
    void updateUi(bool parameterUpdate = false, aq::Nodes::Node* node = nullptr);
    // Used for thread safety
    void on_nodeUpdate();
    void on_logReceive(boost::log::trivial::severity_level verb, const std::string& msg);
    bool eventFilter(QObject *object, QEvent *event);
    void addParameterWidgetMap(QWidget* widget, mo::IParameter* param);
    QWidget* mainWindow;
private slots:
    void on_enableClicked(bool state);
    void on_profileClicked(bool state);

    void log(boost::log::trivial::severity_level verb, const std::string& msg);
signals:
    void eLog(boost::log::trivial::severity_level verb, const std::string& msg);
    void parameterClicked(mo::IParameter* param, QPoint pos);
private:
    QLineEdit* profileDisplay;
    QLineEdit* traceDisplay;
    QLineEdit* debugDisplay;
    QLineEdit* infoDisplay;
    QLineEdit* warningDisplay;
    QLineEdit* errorDisplay;

    void on_object_recompile(mo::IMetaObject* obj);
    std::map<QWidget*, mo::IParameter*> widgetParamMap;
    Ui::QNodeWidget* ui;
    rcc::weak_ptr<aq::Nodes::Node> node;
    std::map<std::string, mo::UI::qt::IParameterProxy::Ptr> parameterProxies;
    std::map<std::string, std::shared_ptr<QInputProxy>> inputProxies;
    QNodeWidget* parentWidget;
    std::vector<QNodeWidget*> childWidgets;
    std::shared_ptr<mo::Connection> log_connection;
    std::shared_ptr<mo::Connection> _recompile_connection;
};

class CV_EXPORTS DataStreamWidget: public QWidget
{
    Q_OBJECT
public:
    DataStreamWidget(QWidget* parent = nullptr, aq::IDataStream::Ptr stream = aq::IDataStream::Ptr());
    ~DataStreamWidget();

    rcc::weak_ptr<aq::IDataStream> GetStream();
    void SetSelected(bool state);
    void update_ui();

signals:


private:
    aq::IDataStream::Ptr _dataStream;
    Ui::DataStreamWidget* ui;
    std::vector<QInputProxy*> inputProxies;
    std::map<std::string, mo::UI::qt::IParameterProxy::Ptr> parameterProxies;
    std::map<QWidget*, mo::IParameter*> widgetParamMap;
};

class DraggableLabel: public QLabel
{
    mo::IParameter* param;
public:
    DraggableLabel(QString name, mo::IParameter* param_);
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
    mo::IParameter* parameter;
};
IQNodeProxy* dispatchParameter(IQNodeInterop* parent, mo::IParameter* parameter, rcc::weak_ptr<aq::Nodes::Node> node);


// Interface class for the interop class
class CV_EXPORTS IQNodeInterop: public QWidget
{
    Q_OBJECT
public:
    IQNodeInterop(mo::IParameter* parameter_, QNodeWidget* parent = nullptr, rcc::weak_ptr<aq::Nodes::Node> node_= rcc::weak_ptr<aq::Nodes::Node>());
    virtual ~IQNodeInterop();

    IQNodeProxy* proxy;
    mo::IParameter* parameter;
    std::shared_ptr<mo::Connection> bc;
    boost::posix_time::ptime previousUpdateTime;
public slots:
    virtual void updateUi();
private slots:
    void on_valueChanged(double value);
    void on_valueChanged(int value);
    void on_valueChanged(bool value);
    void on_valueChanged(QString value);
    void on_valueChanged();
    void onParameterUpdate(mo::IParameter* parameter, mo::Context* ctx);
    //void onParameterUpdate();
signals:
    void updateNeeded();
protected:
    mo::TypedSlot<void(mo::IParameter*, mo::Context*)> onParameterUpdateSlot;
    QLabel* nameElement;    
    QGridLayout* layout;
    rcc::weak_ptr<aq::Nodes::Node> node;
};

