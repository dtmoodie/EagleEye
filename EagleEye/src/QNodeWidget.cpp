#include "QNodeWidget.h"

#include "ui_QNodeWidget.h"
#include <boost/bind.hpp>
#include <QPalette>
#include <QDateTime>


IQNodeInterop::IQNodeInterop(boost::shared_ptr<EagleLib::Parameter> parameter_, QNodeWidget* parent, EagleLib::Node::Ptr node_) :
    QWidget(parent),
    parameter(parameter_),
    node(node_)
{
    layout = new QGridLayout(this);
    layout->setVerticalSpacing(0);
    nameElement = new QLabel(QString::fromStdString(parameter_->name), parent);
    nameElement->setSizePolicy(QSizePolicy::Policy::MinimumExpanding, QSizePolicy::Policy::Fixed);
    proxy = dispatchParameter(this, parameter_, node_);
    int pos = 1;
    if (proxy)
    {
        int numWidgets = proxy->getNumWidgets();
        for(int i = 0; i < numWidgets; ++i, ++pos)
        {
            layout->addWidget(proxy->getWidget(i), 0, pos);
        }
    }
    layout->addWidget(nameElement, 0, 0);
    nameElement->installEventFilter(parent);
    nameElement->setToolTip(QString::fromStdString(parameter_->toolTip));
    connect(this, SIGNAL(updateNeeded()), this, SLOT(updateUi()), Qt::QueuedConnection);
    bc = parameter->onUpdate.connect(boost::bind(&IQNodeInterop::onParameterUpdate, this));

	QLabel* typeElement = new QLabel(QString::fromStdString(TypeInfo::demangle(parameter_->typeInfo.name())));

    typeElement->installEventFilter(parent);
    parent->addParameterWidgetMap(typeElement, parameter_);
    layout->addWidget(typeElement, 0, pos);
}

IQNodeInterop::~IQNodeInterop()
{
    delete proxy;
}

void IQNodeInterop::onParameterUpdate()
{
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = currentTime - previousUpdateTime;
    if(delta.total_milliseconds() > 30)
    {
        previousUpdateTime = currentTime;
        emit updateNeeded();
    }
}

void IQNodeInterop::updateUi()
{
    if (proxy)
        proxy->updateUi(false);
}

void IQNodeInterop::on_valueChanged(double value)
{
    if (proxy)
        proxy->onUiUpdated(static_cast<QWidget*>(sender()));
}

void IQNodeInterop::on_valueChanged(int value)
{
    if (proxy)
        proxy->onUiUpdated(static_cast<QWidget*>(sender()));
}

void IQNodeInterop::on_valueChanged(bool value)
{
    if (proxy)
        proxy->onUiUpdated(static_cast<QWidget*>(sender()));
}
void IQNodeInterop::on_valueChanged(QString value)
{
    if (proxy)
        proxy->onUiUpdated(static_cast<QWidget*>(sender()));
}

void IQNodeInterop::on_valueChanged()
{
    if (proxy)
        proxy->onUiUpdated(static_cast<QWidget*>(sender()));
}
void IQNodeInterop::onParameterUpdate(boost::shared_ptr<EagleLib::Parameter> parameter)
{
    updateUi();
}
DraggableLabel::DraggableLabel(QString name, EagleLib::Parameter::Ptr param_):
    QLabel(name), param(param_)
{
    setAcceptDrops(true);
}

void DraggableLabel::dropEvent(QDropEvent* event)
{
    QLabel::dropEvent(event);
    return;
}

void DraggableLabel::dragLeaveEvent(QDragLeaveEvent* event)
{
    QLabel::dragLeaveEvent(event);
    return;
}

void DraggableLabel::dragMoveEvent(QDragMoveEvent* event)
{
    QLabel::dragMoveEvent(event);
    return;
}


QNodeWidget::QNodeWidget(QWidget* parent, EagleLib::Node::Ptr node_) :
    mainWindow(parent),
    ui(new Ui::QNodeWidget()),
    node(node_)
{
	ui->setupUi(this);
    statusDisplay = new QLineEdit();
    warningDisplay = new QLineEdit();
    errorDisplay = new QLineEdit();
    criticalDisplay = new QLineEdit();
    profileDisplay = new QLineEdit();
    ui->verticalLayout->addWidget(profileDisplay);
    ui->verticalLayout->addWidget(statusDisplay);
    ui->verticalLayout->addWidget(warningDisplay);
    ui->verticalLayout->addWidget(errorDisplay);
    ui->verticalLayout->addWidget(criticalDisplay);
    profileDisplay->hide();
    statusDisplay->hide();
    warningDisplay->hide();
    errorDisplay->hide();
    criticalDisplay->hide();
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
    connect(this, SIGNAL(eLog(EagleLib::Verbosity,std::string,EagleLib::Node*)),
            this, SLOT(log(EagleLib::Verbosity,std::string,EagleLib::Node*)));

    if (node != nullptr)
	{
        ui->chkEnabled->setChecked(node->enabled);
        ui->profile->setChecked(node->profile);
        connect(ui->chkEnabled, SIGNAL(clicked(bool)), this, SLOT(on_enableClicked(bool)));
        connect(ui->profile, SIGNAL(clicked(bool)), this, SLOT(on_profileClicked(bool)));
        ui->nodeName->setText(QString::fromStdString(node->nodeName));
        ui->nodeName->setToolTip(QString::fromStdString(node->fullTreeName));
        ui->nodeName->setMaximumWidth(200);
        ui->verticalLayout->setSpacing(0);
        for (size_t i = 0; i < node->parameters.size(); ++i)
		{
            auto interop = new IQNodeInterop(node->parameters[i], this, node);
            interops.push_back(boost::shared_ptr<IQNodeInterop>(interop));
            //interop->setSizePolicy(QSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum));
            ui->verticalLayout->addWidget(interop);
		}
        node->onUpdate = boost::bind(&QNodeWidget::updateUi, this, true);
        node->messageCallback = boost::bind(&QNodeWidget::on_logReceive,this, _1, _2, _3);
	}
}
bool QNodeWidget::eventFilter(QObject *object, QEvent *event)
{
    if(event->type() == QEvent::MouseButtonPress)
    {
        auto itr = widgetParamMap.find(static_cast<QWidget*>(object));
        if(itr != widgetParamMap.end())
        {
            emit parameterClicked(itr->second);
            return true;
        }
        return false;
    }
    return false;
}

void QNodeWidget::addParameterWidgetMap(QWidget* widget, EagleLib::Parameter::Ptr param)
{
    if(widgetParamMap.find(widget) == widgetParamMap.end())
        widgetParamMap[widget] = param;
}

void QNodeWidget::updateUi(bool parameterUpdate)
{
    if(node == nullptr)
        return;
    ui->processingTime->setText(QString::number(node->processingTime));
    if(parameterUpdate)
    {
        if(node->parameters.size() != interops.size())
        {
            for(size_t i = 0; i < node->parameters.size(); ++i)
            {
                bool found = false;
                for(size_t j = 0; j < interops.size(); ++j)
                {
                    if(node->parameters[i] == interops[j]->parameter)
                        found = true;
                }
                if(found == false)
                {
                    // Need to add a new interop for this node
                    auto interop = new IQNodeInterop(node->parameters[i], this, node);
                    interops.push_back(boost::shared_ptr<IQNodeInterop>(interop));
                    ui->verticalLayout->addWidget(interop);
                }
            }
        }
        for(size_t i = 0; i < interops.size(); ++i)
        {
            interops[i]->updateUi();
        }
    }
}
void QNodeWidget::on_nodeUpdate()
{

}
void QNodeWidget::log(EagleLib::Verbosity verb, const std::string &msg, EagleLib::Node* node)
{
    switch(verb)
    {
    case EagleLib::Profiling:
        on_profile(msg, node);
    case EagleLib::Status:
        on_status(msg,node);
        return;
    case EagleLib::Warning:
        on_warning(msg,node);
        return;
    case EagleLib::Error:
        on_error(msg,node);
        return;
    case EagleLib::Critical:
        on_critical(msg,node);
        return;
    }
}


void QNodeWidget::on_logReceive(EagleLib::Verbosity verb, const std::string& msg, EagleLib::Node* node)
{
    emit eLog(verb, msg, node);
}



QNodeWidget::~QNodeWidget()
{

}

void QNodeWidget::on_enableClicked(bool state)
{    node->enabled = state;     }
void QNodeWidget::on_profileClicked(bool state)
{
    if(node != nullptr)
        node->profile = state;
}

EagleLib::Node::Ptr QNodeWidget::getNode()
{
    return node;
}

void QNodeWidget::on_profile(const std::string &msg, EagleLib::Node *node)
{

}

void QNodeWidget::on_status(const std::string& msg, EagleLib::Node* node)
{
    statusDisplay->show();
    statusDisplay->setText(QDateTime::currentDateTime().toString("[hh:mm.ss.zzz] ") + " Status: " + QString::fromStdString(msg));
    update();
}

void QNodeWidget::on_warning(const std::string& msg, EagleLib::Node* node)
{
    warningDisplay->show();
    warningDisplay->setText(QDateTime::currentDateTime().toString("[hh:mm.ss.zzz] ") + "Warning: " + QString::fromStdString(msg));
    update();
}

void QNodeWidget::on_error(const std::string& msg, EagleLib::Node* node)
{
    errorDisplay->show();
    errorDisplay->setText(QDateTime::currentDateTime().toString("[hh:mm.ss.zzz] ") + "Error: " + QString::fromStdString(msg));
    update();
}

void QNodeWidget::on_critical(const std::string& msg, EagleLib::Node* node)
{
    criticalDisplay->show();
    criticalDisplay->setText("Critical: " + QString::fromStdString(msg));
    update();
}
void QNodeWidget::setSelected(bool state)
{
    QPalette pal(palette());
    if(state == true)
        pal.setColor(QPalette::Background, Qt::green);
    else
        pal.setColor(QPalette::Background, Qt::gray);
    setAutoFillBackground(true);
    setPalette(pal);
}
QInputProxy::QInputProxy(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter_, EagleLib::Node::Ptr node_):
    node(node_)
{
    box = new QComboBox(parent);
    box->setMaximumWidth(200);
    parameter = parameter_;
    updateUi();
    parent->connect(box, SIGNAL(currentIndexChanged(int)), parent, SLOT(on_valueChanged(int)));
}

void QInputProxy::onUiUpdated(QWidget* sender)
{
    QString inputName = box->currentText();
    if(inputName.size() == 0)
        return;
    //auto tokens = inputName.split(":");
    //auto sourceNode = node->getNodeInScope(tokens[0].toStdString());
    //if(sourceNode == nullptr)
      //  return;
    //auto param = sourceNode->getParameter(tokens[1].toStdString());
    //if(param)
    parameter->setSource(inputName.toStdString());
    parameter->changed = true;

}
QWidget* QInputProxy::getWidget(int num)
{
    return box;
}
// This only needs to be called when a new output parameter is added to the environment......
// The best way of doing this would be to add a signal that each node emits when a new parameter is added
// to the environment, this signal is then caught and used to update all input proxies.
void QInputProxy::updateUi(bool init)
{
    QString currentItem = box->currentText();
    parameter->update();
    auto inputs = node->findCompatibleInputs(parameter);
    box->clear();
    box->addItem("");
    for(size_t i = 0; i < inputs.size(); ++i)
    {
        QString text = QString::fromStdString(inputs[i]);
        box->addItem(text);
    }
    if(parameter->inputName.size())
    {
        QString inputName = QString::fromStdString(parameter->inputName);
        if(box->currentText() != inputName)
            box->setCurrentText(inputName);
        return;
    }
    if(currentItem.size())
        box->setCurrentText(currentItem);
}

template<typename T> bool acceptsType(Loki::TypeInfo& type)
{
    return Loki::TypeInfo(typeid(T)) == type || Loki::TypeInfo(typeid(T*)) == type || Loki::TypeInfo(typeid(T&)) == type;
}
#define MAKE_TYPE(type) if(acceptsType<type>(parameter->typeInfo)) return new QNodeProxy<type, false>(parent, parameter);
#define MAKE_TYPE_(type) if(acceptsType<type>(parameter->typeInfo)) return new QNodeProxy<type, true>(parent, parameter);


IQNodeProxy* dispatchParameter(IQNodeInterop* parent, boost::shared_ptr<EagleLib::Parameter> parameter, EagleLib::Node::Ptr node)
{
    if(parameter->type & EagleLib::Parameter::Input)
    {
        // Create a ui for finding relavent inputs
        return new QInputProxy(parent, parameter, node);
    }
    typedef std::vector<std::pair<int,double>> vec_ID;
    typedef std::vector<std::pair<double,double>> vec_DD;
    typedef std::vector<std::pair<double,int>> vec_DI;
    typedef std::vector<std::pair<int, int>> vec_II;
    if(parameter->type & EagleLib::Parameter::Control)
    {
        MAKE_TYPE(double);
        MAKE_TYPE(float);
        MAKE_TYPE(int);
        MAKE_TYPE(unsigned int);
        MAKE_TYPE(short);
        MAKE_TYPE(unsigned short);
        MAKE_TYPE(char);
        MAKE_TYPE(unsigned char);
        MAKE_TYPE(unsigned long);
        MAKE_TYPE(std::string);
        MAKE_TYPE(boost::filesystem::path);
        MAKE_TYPE(bool);
        MAKE_TYPE(boost::function<void(void)>);
        MAKE_TYPE(EagleLib::EnumParameter);
        MAKE_TYPE(cv::Scalar);
        MAKE_TYPE(std::vector<int>);
        MAKE_TYPE(std::vector<double>);
        MAKE_TYPE(vec_ID);
        MAKE_TYPE(vec_DD);
        MAKE_TYPE(vec_DI);
        MAKE_TYPE(vec_II);
        MAKE_TYPE(EagleLib::ReadDirectory);
    }

    if(parameter->type & EagleLib::Parameter::Output || parameter->type & EagleLib::Parameter::State)
    {
        MAKE_TYPE_(double);
        MAKE_TYPE_(float);
        MAKE_TYPE_(int);
        MAKE_TYPE_(unsigned int);
        MAKE_TYPE_(short);
        MAKE_TYPE_(unsigned short);
        MAKE_TYPE_(char);
        MAKE_TYPE_(unsigned char);
        MAKE_TYPE_(unsigned long);
        MAKE_TYPE_(std::string);
        MAKE_TYPE_(boost::filesystem::path);
        MAKE_TYPE_(bool);
        MAKE_TYPE_(EagleLib::EnumParameter);
        MAKE_TYPE_(cv::Scalar);
        MAKE_TYPE_(std::vector<int>);
        MAKE_TYPE_(std::vector<double>);
        MAKE_TYPE_(vec_ID);
        MAKE_TYPE_(vec_DD);
        MAKE_TYPE_(vec_DI);
        MAKE_TYPE_(vec_II);
        MAKE_TYPE_(EagleLib::ReadDirectory);
    }
	return nullptr;
}


