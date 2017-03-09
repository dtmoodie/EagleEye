#include "QNodeWidget.h"

#include "ui_QNodeWidget.h"
#include "ui_datastreamwidget.h"

#include <boost/bind.hpp>
#include <QPalette>
#include <QDateTime>
#include "qevent.h"

#include "Aquila/logger.hpp"
#include <Aquila/Nodes/IFrameGrabber.hpp>
#include <MetaObject/Logging/Log.hpp>
#include <MetaObject/Parameters/UI/WidgetFactory.hpp>
#include <MetaObject/Parameters/IVariableManager.h>


IQNodeInterop::IQNodeInterop(mo::IParameter* parameter_, 
                             QNodeWidget* parent, 
                             rcc::weak_ptr<aq::Nodes::Node> node_) :
    QWidget(parent),
    parameter(parameter_),
    node(node_)
{
    layout = new QGridLayout(this);
    layout->setVerticalSpacing(0);
    nameElement = new QLabel(QString::fromStdString(parameter_->GetName()), parent);
    nameElement->setSizePolicy(QSizePolicy::Policy::MinimumExpanding, QSizePolicy::Policy::Fixed);
    //proxy = dispatchParameter(this, parameter_, node_);
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
    connect(this, SIGNAL(updateNeeded()), this, SLOT(updateUi()), Qt::QueuedConnection);
    onParameterUpdateSlot = mo::TypedSlot<void(mo::IParameter*, mo::Context*)>(std::bind(&IQNodeInterop::onParameterUpdate, this, std::placeholders::_1, std::placeholders::_2));
    bc = parameter->RegisterUpdateNotifier(&onParameterUpdateSlot);
    
    QLabel* typeElement = new QLabel(QString::fromStdString(parameter_->GetTypeInfo().name()));

    typeElement->installEventFilter(parent);
    parent->addParameterWidgetMap(typeElement, parameter_);
    layout->addWidget(typeElement, 0, pos);
}

IQNodeInterop::~IQNodeInterop()
{
    delete proxy;
}

/*void IQNodeInterop::onParameterUpdate()
{
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = currentTime - previousUpdateTime;
    if(delta.total_milliseconds() > 30)
    {
        previousUpdateTime = currentTime;
        emit updateNeeded();
    }
}*/

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
void IQNodeInterop::onParameterUpdate(mo::IParameter* parameter, mo::Context* ctx)
{
    updateUi();
}
DraggableLabel::DraggableLabel(QString name, mo::IParameter* param_):
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

QNodeWidget::QNodeWidget(QWidget* parent, rcc::weak_ptr<aq::Nodes::Node> node_) :
    mainWindow(parent),
    ui(new Ui::QNodeWidget()),
    node(node_)
{
    ui->setupUi(this);
    profileDisplay = new QLineEdit();
    traceDisplay = new QLineEdit();
    debugDisplay = new QLineEdit();
    infoDisplay = new QLineEdit();
    warningDisplay = new QLineEdit();
    errorDisplay = new QLineEdit();
    ui->gridLayout->addWidget(profileDisplay, 1, 0, 1, 2);
    ui->gridLayout->addWidget(traceDisplay, 2, 0, 1, 2);
    ui->gridLayout->addWidget(debugDisplay, 3, 0, 1, 2);
    ui->gridLayout->addWidget(infoDisplay, 4, 0, 1, 2);
    ui->gridLayout->addWidget(warningDisplay, 5, 0, 1, 2);
    ui->gridLayout->addWidget(errorDisplay, 6, 0, 1, 2);
    profileDisplay->hide();
    traceDisplay->hide();
    debugDisplay->hide();
    infoDisplay->hide();
    warningDisplay->hide();
    errorDisplay->hide();
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
    connect(this, SIGNAL(eLog(boost::log::trivial::severity_level, std::string)),
        this, SLOT(log(boost::log::trivial::severity_level, std::string)));

    if (node != nullptr)
    {
//        ui->chkEnabled->setChecked(node->enabled);
        //ui->profile->setChecked(node->profile);
        connect(ui->chkEnabled, SIGNAL(clicked(bool)), this, SLOT(on_enableClicked(bool)));
        connect(ui->profile, SIGNAL(clicked(bool)), this, SLOT(on_profileClicked(bool)));
        ui->nodeName->setText(QString::fromStdString(node->GetTypeName()));
        ui->nodeName->setToolTip(QString::fromStdString(node->GetTreeName()));
        ui->nodeName->setMaximumWidth(200);
        ui->gridLayout->setSpacing(0);
        auto parameters = node->GetDisplayParameters();
        for (size_t i = 0; i < parameters.size(); ++i)
        {
            int col = 0;
            if (parameters[i]->CheckFlags(mo::Input_e))
            {
                QInputProxy* proxy = new QInputProxy(parameters[i], node, this);
                ui->gridLayout->addWidget(proxy->getWidget(0), i+7, col, 1,1);
                inputProxies[parameters[i]->GetTreeName()] = std::shared_ptr<QInputProxy>(proxy);
                ++col;
            }
            
            if(parameters[i]->CheckFlags(mo::Control_e)|| parameters[i]->CheckFlags(mo::State_e) || parameters[i]->CheckFlags(mo::Output_e))
            {
                //auto interop = Parameters::UI::qt::WidgetFactory::Instance()->Createhandler(parameters[i]);
                auto proxy = mo::UI::qt::WidgetFactory::Instance()->CreateProxy(parameters[i]);
                if (proxy)
                {
                    parameterProxies[parameters[i]->GetTreeName()] = proxy;
                    auto widget = proxy->GetParameterWidget(this);
                    widget->installEventFilter(this);
                    widgetParamMap[widget] = parameters[i];
                    ui->gridLayout->addWidget(widget, i + 7, col, 1,1);
                }
            }
        }
        /*log_connection = aq::ui_collector::get_object_log_handler(node->getFullTreeName()).connect(std::bind(&QNodeWidget::on_logReceive, this, std::placeholders::_1, std::placeholders::_2));
        if (auto stream = node->GetDataStream())
        {
            if (auto sig_mgr = stream->GetSignalManager())
            {
                _recompile_connection = sig_mgr->connect<void(aq::ParameteredIObject*)>("object_recompiled", std::bind(&QNodeWidget::on_object_recompile, this, std::placeholders::_1));
            }
            else
            {
                LOG(debug) << "stream->GetSignalManager() failed";
            }
        }
        else
        {
            LOG(debug) << "No stream attached to node";
        }*/
    }
}
void QNodeWidget::on_object_recompile(mo::IMetaObject* obj)
{
    if(node)
    {
        if (obj->GetObjectId() == node->GetObjectId())
        {
            auto params = node->GetDisplayParameters();
            int idx = -1;
            for (auto& param : params)
            {
                ++idx;
                {
                    auto itr = parameterProxies.find(param->GetTreeName());
                    if (itr != parameterProxies.end())
                    {
                        itr->second->SetParameter(param);
                        break;
                    }
                }
                {
                    auto itr = inputProxies.find(param->GetTreeName());
                    if (itr != inputProxies.end())
                    {
                        itr->second->updateParameter(param);
                        break;
                    }
                }
                LOG(debug) << "Adding new parameter after recompile: " << param->GetTreeName();
                int col = 0;
                if (param->CheckFlags(mo::Input_e))
                {
                    QInputProxy* proxy = new QInputProxy(param, node, this);
                    ui->gridLayout->addWidget(proxy->getWidget(0), idx + 7, col, 1, 1);
                    inputProxies[param->GetTreeName()] = std::shared_ptr<QInputProxy>(proxy);
                    ++col;
                }

                if (param->CheckFlags(mo::Control_e) || param->CheckFlags(mo::State_e) || param->CheckFlags(mo::Output_e))
                {
                    //auto interop = Parameters::UI::qt::WidgetFactory::Instance()->Createhandler(param);
                    auto proxy = mo::UI::qt::WidgetFactory::Instance()->CreateProxy(param);
                    if (proxy)
                    {
                        parameterProxies[param->GetTreeName()] = proxy;
                        auto widget = proxy->GetParameterWidget(this);
                        widget->installEventFilter(this);
                        widgetParamMap[widget] = param;
                        ui->gridLayout->addWidget(widget, idx + 7, col, 1, 1);
                    }
                }
            }
        }
    }
}
bool QNodeWidget::eventFilter(QObject *object, QEvent *event)
{
    if(event->type() == QEvent::MouseButtonPress)
    {
        QMouseEvent* mev = dynamic_cast<QMouseEvent*>(event);
        if (mev->button() == Qt::RightButton)
        {
            
            auto itr = widgetParamMap.find(static_cast<QWidget*>(object));
            if (itr != widgetParamMap.end())
            {
                emit parameterClicked(itr->second, mev->pos());
                return true;
            }
        }
    }
    return false;
}

void QNodeWidget::addParameterWidgetMap(QWidget* widget, mo::IParameter* param)
{
    
}

void QNodeWidget::updateUi(bool parameterUpdate, aq::Nodes::Node *node_)
{
    if(node == nullptr)
        return;
    //ui->processingTime->setText(QString::number(node->GetProcessingTime()));
    /*if (node->profile)
    {
        auto timings = node->GetProfileTimings();
        if (timings.size() > 1)
        {
            profileDisplay->show();
            std::stringstream text;
            for (int i = 1; i < timings.size(); ++i)
            {
                text << "(" << timings[i - 1].second << "," << timings[i].second << ")-" << timings[i].first - timings[i - 1].first << " ";
            }
            profileDisplay->setText(QString::fromStdString(text.str()));
        }
    }
    else
    {
        profileDisplay->hide();
    }*/
    if(parameterUpdate && node_ == node.Get())
    {
        auto parameters = node->GetDisplayParameters();
        if (parameters.size() != (parameterProxies.size() + inputProxies.size()))
        {
            for(size_t i = 0; i < parameters.size(); ++i)
            {
                bool found = false;
                // Check normal parameters
                {
                    auto itr = parameterProxies.find(parameters[i]->GetTreeName());
                    if(itr != parameterProxies.end())
                    {
                        if(itr->second->CheckParameter(parameters[i]));
                        found = true;
                    }
                }
                // Check inputs
                if(!found)
                {
                    auto itr = inputProxies.find(parameters[i]->GetTreeName());
                    if(itr != inputProxies.end())
                    {
                        found = itr->second->inputParameter == dynamic_cast<mo::InputParameter*>(parameters[i]);
                    }
                }
                
                if(found == false)
                {
                    int col = 0;
                    if (parameters[i]->CheckFlags(mo::Input_e))
                    {
                        QInputProxy* proxy = new QInputProxy(parameters[i], node, this);
                        ui->gridLayout->addWidget(proxy->getWidget(0), i + 7, col, 1, 1);
                        inputProxies[parameters[i]->GetTreeName()] = std::shared_ptr<QInputProxy>(proxy);
                        ++col;
                    }

                    if (parameters[i]->CheckFlags(mo::Control_e) || parameters[i]->CheckFlags(mo::State_e) || parameters[i]->CheckFlags(mo::Output_e))
                    {
                        //auto interop = Parameters::UI::qt::WidgetFactory::Instance()->Createhandler(parameters[i]);
                        auto proxy = mo::UI::qt::WidgetFactory::Instance()->CreateProxy(parameters[i]);
                        if (proxy)
                        {
                            parameterProxies[parameters[i]->GetTreeName()] = proxy;
                            auto widget = proxy->GetParameterWidget(this);
                            widget->installEventFilter(this);
                            widgetParamMap[widget] = parameters[i];
                            ui->gridLayout->addWidget(widget, i + 7, col, 1, 1);
                        }
                    }
                }
            }
        }
    }
    if(parameterUpdate && node_ != node.Get())
    {
        for(auto& proxy : inputProxies)
        {
            proxy.second->updateUi(nullptr, nullptr, true);
        }
    }
}
void QNodeWidget::on_nodeUpdate()
{

}
void QNodeWidget::log(boost::log::trivial::severity_level verb, const std::string& msg)
{
    switch(verb)
    {
    case(boost::log::trivial::trace):
    {
        traceDisplay->setText(QString::fromStdString(msg));
        traceDisplay->show();
        break;
    }
    case(boost::log::trivial::debug) :
    {
        debugDisplay->setText(QString::fromStdString(msg));
        debugDisplay->show();
        break;
    }
    case(boost::log::trivial::info) :
    {
        infoDisplay->setText(QString::fromStdString(msg));
        infoDisplay->show();
        break;
    }
    case(boost::log::trivial::warning) :
    {
        warningDisplay->setText(QString::fromStdString(msg));
        warningDisplay->show();
        break;
    }
    case(boost::log::trivial::error) :
    {
        errorDisplay->setText(QString::fromStdString(msg));
        errorDisplay->show();
        break;
    }
    case(boost::log::trivial::fatal):
    {
        errorDisplay->setText(QString::fromStdString(msg));
        errorDisplay->show();
        break;
    }
    }
}



void QNodeWidget::on_logReceive(boost::log::trivial::severity_level verb, const std::string& msg)
{
    emit eLog(verb, msg);
}



QNodeWidget::~QNodeWidget()
{
    //aq::ui_collector::removeNodeCallbackHandler(node.get(), boost::bind(&QNodeWidget::on_logReceive, this, _1, _2));
}

void QNodeWidget::on_enableClicked(bool state)
{    
    //node->enabled = state;     
}
void QNodeWidget::on_profileClicked(bool state)
{
    /*if(node != nullptr)
        node->profile = state;*/
    
}

rcc::weak_ptr<aq::Nodes::Node> QNodeWidget::getNode()
{
    return node;
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

DataStreamWidget::DataStreamWidget(QWidget* parent, aq::IDataStream::Ptr stream):
    QWidget(parent), 
    _dataStream(stream),
    ui(new Ui::DataStreamWidget())
{
    ui->setupUi(this);
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
    /*if(auto fg = stream->GetFrameGrabber())
        ui->lbl_loaded_document->setText(QString::fromStdString(fg->GetSourceFilename()));*/
    //ui->lbl_stream_id->setText(QString::number(stream->get_stream_id()));
    update_ui();
}
void DataStreamWidget::update_ui()
{
    //auto fg = _dataStream->GetFrameGrabber();
    /*if (fg != nullptr)
    {
        auto parameters = fg->getParameters();
        for (int i = 0; i < parameters.size(); ++i)
        {
            int col = 0;
            if (parameters[i]->type & mo::IParameter::Input)
            {
                
            }

            if (parameters[i]->type & mo::IParameter::Control || parameters[i]->type & mo::IParameter::State || parameters[i]->type & mo::IParameter::Output)
            {
                auto interop = Parameters::UI::qt::WidgetFactory::Instance()->Createhandler(parameters[i]);
                if (interop)
                {
                    parameterProxies[parameters[i]->GetTreeName()] = interop;
                    auto widget = interop->GetParameterWidget(this);
                    widget->installEventFilter(this);
                    widgetParamMap[widget] = parameters[i];
                    ui->parameter_layout->addWidget(widget, i, col, 1, 1);
                }
            }
        }
    }*/
}

DataStreamWidget::~DataStreamWidget()
{

}

rcc::weak_ptr<aq::IDataStream> DataStreamWidget::GetStream()
{
    return _dataStream;
}
void DataStreamWidget::SetSelected(bool state)
{
    QPalette pal(palette());
    if (state == true)
        pal.setColor(QPalette::Background, Qt::green);
    else
        pal.setColor(QPalette::Background, Qt::gray);
    setAutoFillBackground(true);
    setPalette(pal);
}



QInputProxy::QInputProxy(mo::IParameter* parameter_, rcc::weak_ptr<aq::Nodes::Node> node_, QWidget* parent):
    node(node_), QWidget(parent)
{
    box = new QComboBox(this);
    box->setMinimumWidth(200);
    box->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    //bc = parameter_->RegisterNotifier(boost::bind(&QInputProxy::updateUi, this, false));
    //dc = parameter_->RegisterDeleteNotifier(std::bind(&QInputProxy::onParamDelete, this, std::placeholders::_1));
    onParameterUpdateSlot = mo::TypedSlot<void(mo::IParameter*, mo::Context*)>(std::bind(&QInputProxy::updateUi, this, std::placeholders::_1, std::placeholders::_2, false));
    onParameterDeleteSlot = mo::TypedSlot<void(mo::IParameter const*)>(std::bind(&QInputProxy::onParamDelete, this, std::placeholders::_1));
    update_connection = parameter_->RegisterUpdateNotifier(&onParameterUpdateSlot);
    delete_connection = parameter_->RegisterDeleteNotifier(&onParameterDeleteSlot);
    //dc = parameter_->RegisterDeleteNotifier(&onParameterUpdateSlot);
    inputParameter = dynamic_cast<mo::InputParameter*>(parameter_);
    updateUi(nullptr, nullptr, true);
    connect(box, SIGNAL(currentIndexChanged(int)), this, SLOT(on_valueChanged(int)));
    prevIdx = 0;
    box->installEventFilter(this);
}

void QInputProxy::updateParameter(mo::IParameter* parameter)
{
    delete_connection = parameter->RegisterDeleteNotifier(&onParameterDeleteSlot);
    update_connection = parameter->RegisterUpdateNotifier(&onParameterUpdateSlot);
    inputParameter = dynamic_cast<mo::InputParameter*>(parameter);
}

void QInputProxy::onParamDelete(mo::IParameter const* parameter)
{
    inputParameter = nullptr;
}
void QInputProxy::on_valueChanged(int idx)
{
    if(idx == prevIdx)
        return;
    prevIdx = idx;
    if (idx == 0)
    {
        inputParameter->SetInput(nullptr);
        return;
    }
    QString inputName = box->currentText();
    LOG(debug) << "Input changed to " << inputName.toStdString() << " for variable " << dynamic_cast<mo::IParameter*>(inputParameter)->GetName();
    auto tokens = inputName.split(":");
    //auto var_manager = node->GetVariableManager();
    //auto tmp_param = var_manager->GetOutputParameter(inputName.toStdString());
    
    //auto sourceNode = node->getNodeInScope(tokens[0].toStdString());
    //DOIF(sourceNode == nullptr, "Unable to find source node"; return;, debug);
    //auto param = sourceNode->getParameter(tokens[1].toStdString());
    /*if(tmp_param)
    {
        inputParameter->SetInput(tmp_param);
        dynamic_cast<mo::IParameter*>(inputParameter)->changed = true;
    }*/
}
QWidget* QInputProxy::getWidget(int num)
{
    return box;
}
bool QInputProxy::eventFilter(QObject* obj, QEvent* event)
{
    if(obj == box)
    {
        if(auto focus_event = dynamic_cast<QFocusEvent*>(event))
        {
            updateUi(nullptr, nullptr, false);
            return true;
        }
    }
    return false;
}
// This only needs to be called when a new output parameter is added to the environment......
// The best way of doing this would be to add a signal that each node emits when a new parameter is added
// to the environment, this signal is then caught and used to update all input proxies.
void QInputProxy::updateUi(mo::IParameter*, mo::Context*, bool init)
{
    auto var_man = node->GetVariableManager();
    auto inputs = var_man->GetOutputParameters(inputParameter->GetTypeInfo());
    if(box->count() == 0)
    {
        box->addItem(QString::fromStdString(dynamic_cast<mo::IParameter*>(inputParameter)->GetTreeName()));    
    }
    for(size_t i = 0; i < inputs.size(); ++i)
    {
        QString text = QString::fromStdString(inputs[i]->GetTreeName());
        bool found = false;
        for(int j = 0; j < box->count(); ++j)
        {
            if(box->itemText(j).compare(text) == 0)
                found = true;
        }
        if(!found)
            box->addItem(text);
    }
    auto input = inputParameter->GetInputParam();
    if (input)
    {
        auto currentInputName = input->GetTreeName();
        if (currentInputName.size())
        {
            QString inputName = QString::fromStdString(currentInputName);
            if (box->currentText() != inputName)
                box->setCurrentText(inputName);
            return;
        }
    }
}
