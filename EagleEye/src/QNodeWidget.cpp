#include "QNodeWidget.h"

#include "ui_QNodeWidget.h"
#include <boost/bind.hpp>
#include <QPalette>
#include <QDateTime>
#include "qevent.h"
#include "logger.hpp"

IQNodeInterop::IQNodeInterop(Parameters::Parameter::Ptr parameter_, QNodeWidget* parent, EagleLib::Node::Ptr node_) :
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
    nameElement->setToolTip(QString::fromStdString(parameter_->GetTooltip()));
    connect(this, SIGNAL(updateNeeded()), this, SLOT(updateUi()), Qt::QueuedConnection);
	bc = parameter->RegisterNotifier(boost::bind(&IQNodeInterop::onParameterUpdate, this));

    QLabel* typeElement = new QLabel(QString::fromStdString(parameter_->GetTypeInfo().name()));

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
void IQNodeInterop::onParameterUpdate(Parameters::Parameter::Ptr parameter)
{
    updateUi();
}
DraggableLabel::DraggableLabel(QString name, Parameters::Parameter::Ptr param_):
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
    traceDisplay = new QLineEdit();
    debugDisplay = new QLineEdit();
    infoDisplay = new QLineEdit();
    warningDisplay = new QLineEdit();
    errorDisplay = new QLineEdit();
    ui->gridLayout->addWidget(traceDisplay, 1, 0, 1, 2);
    ui->gridLayout->addWidget(debugDisplay, 2, 0, 1, 2);
    ui->gridLayout->addWidget(infoDisplay, 3, 0, 1, 2);
    ui->gridLayout->addWidget(warningDisplay, 4, 0, 1, 2);
    ui->gridLayout->addWidget(errorDisplay, 5, 0, 1, 2);
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
        ui->chkEnabled->setChecked(node->enabled);
        ui->profile->setChecked(node->profile);
        connect(ui->chkEnabled, SIGNAL(clicked(bool)), this, SLOT(on_enableClicked(bool)));
        connect(ui->profile, SIGNAL(clicked(bool)), this, SLOT(on_profileClicked(bool)));
        ui->nodeName->setText(QString::fromStdString(node->nodeName));
        ui->nodeName->setToolTip(QString::fromStdString(node->fullTreeName));
        ui->nodeName->setMaximumWidth(200);
        ui->gridLayout->setSpacing(0);
		
        for (size_t i = 0; i < node->parameters.size(); ++i)
		{
			int col = 0;
			if (node->parameters[i]->type & Parameters::Parameter::Input)
			{
				QInputProxy* proxy = new QInputProxy(node->parameters[i], node, this);
				ui->gridLayout->addWidget(proxy->getWidget(0), i+5, col, 1,1);
				inputProxies.push_back(proxy);
				++col;
			}
			
            if(node->parameters[i]->type & Parameters::Parameter::Control || node->parameters[i]->type & Parameters::Parameter::State || node->parameters[i]->type & Parameters::Parameter::Output)
			{
				auto interop = Parameters::UI::qt::WidgetFactory::Createhandler(node->parameters[i]);
				if (interop)
				{
					parameterProxies.push_back(interop);
					auto widget = interop->GetParameterWidget(this);
					widget->installEventFilter(this);
					widgetParamMap[widget] = node->parameters[i];
					ui->gridLayout->addWidget(widget, i + 5, col, 1,1);
				}
			}
			
		}
        node->onUpdate = boost::bind(&QNodeWidget::updateUi, this, true, _1);
        //node->messageCallback = boost::bind(&QNodeWidget::on_logReceive,this, _1, _2);
        EagleLib::ui_collector::addNodeCallbackHandler(node.get(), boost::bind(&QNodeWidget::on_logReceive, this, _1, _2));
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

void QNodeWidget::addParameterWidgetMap(QWidget* widget, Parameters::Parameter::Ptr param)
{
    
}

void QNodeWidget::updateUi(bool parameterUpdate, EagleLib::Node *node_)
{
    if(node == nullptr)
        return;
    ui->processingTime->setText(QString::number(node->GetProcessingTime()));
    if(parameterUpdate && node_ == node.get())
    {
		if (node->parameters.size() != parameterProxies.size())
        {
            for(size_t i = 0; i < node->parameters.size(); ++i)
            {
                bool found = false;
				for (size_t j = 0; j < parameterProxies.size(); ++j)
                {
					if (parameterProxies[j]->CheckParameter(node->parameters[i].get()))
                        found = true;
                }
				for (size_t j = 0; j < inputProxies.size(); ++j)
				{
					
					if (node->parameters[i]->type & Parameters::Parameter::Input && 
						inputProxies[j]->inputParameter == std::dynamic_pointer_cast<Parameters::InputParameter>(node->parameters[i]))
					{
						found = true;
					}
				}
                if(found == false)
                {
					int col = 0;
					if (node->parameters[i]->type & Parameters::Parameter::Input)
					{
						QInputProxy* proxy = new QInputProxy(node->parameters[i], node, this);
						ui->gridLayout->addWidget(proxy->getWidget(0), i + 5, col, 1, 1);
						inputProxies.push_back(proxy);
						++col;
					}

                    if (node->parameters[i]->type & Parameters::Parameter::Control || node->parameters[i]->type & Parameters::Parameter::State  || node->parameters[i]->type & Parameters::Parameter::Output)
					{
						auto interop = Parameters::UI::qt::WidgetFactory::Createhandler(node->parameters[i]);
						if (interop)
						{
							parameterProxies.push_back(interop);
							auto widget = interop->GetParameterWidget(this);
							widget->installEventFilter(this);
							widgetParamMap[widget] = node->parameters[i];
							ui->gridLayout->addWidget(widget, i + 5, col, 1, 1);
							//ui->gridLayout->addWidget(interop->GetParameterWidget(this), i + 5, col, 1, 1);
						}
					}
                }
            }
        }

    }
    if(parameterUpdate && node_ != node.get())
    {
        for (size_t i = 0; i < inputProxies.size(); ++i)
        {
            inputProxies[i]->updateUi(true);
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
    }
}



void QNodeWidget::on_logReceive(boost::log::trivial::severity_level verb, const std::string& msg)
{
    emit eLog(verb, msg);
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
QInputProxy::QInputProxy(std::shared_ptr<Parameters::Parameter> parameter_, EagleLib::Node::Ptr node_, QWidget* parent):
	node(node_), QWidget(parent)
{
    box = new QComboBox(this);
	box->setMinimumWidth(200);
	box->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	//box->setToolTip(QString::fromStdString(parameter_->GetTooltip()));

	bc = parameter_->RegisterNotifier(boost::bind(&QInputProxy::updateUi, this, false));
    inputParameter = std::dynamic_pointer_cast<Parameters::InputParameter>(parameter_);
    updateUi(true);
    connect(box, SIGNAL(currentIndexChanged(int)), this, SLOT(on_valueChanged(int)));
    prevIdx = 0;
}

void QInputProxy::on_valueChanged(int idx)
{
    if(idx == prevIdx)
        return;
	if (idx == 0)
	{
		inputParameter->SetInput(Parameters::Parameter::Ptr());
		return;
	}
		
    prevIdx = idx;
    QString inputName = box->currentText();
    auto tokens = inputName.split(":");
    auto sourceNode = node->getNodeInScope(tokens[0].toStdString());
    if(sourceNode == nullptr)
        return;
    auto param = sourceNode->getParameter(tokens[1].toStdString());
    if(param)
    {
        inputParameter->SetInput(param);
        std::dynamic_pointer_cast<Parameters::Parameter>(param)->changed = true;
    }
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
    if(init)
    {
        auto inputs = node->findCompatibleInputs(inputParameter);
        box->clear();
        box->addItem(QString::fromStdString(std::dynamic_pointer_cast<Parameters::Parameter>(inputParameter)->GetTreeName()));
        for(size_t i = 0; i < inputs.size(); ++i)
        {
            QString text = QString::fromStdString(inputs[i]);
            box->addItem(text);
        }
    }
	auto input = inputParameter->GetInput();
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

/*template<typename T> bool acceptsType(Loki::TypeInfo& type)
{
    return Loki::TypeInfo(typeid(T)) == type || Loki::TypeInfo(typeid(T*)) == type || Loki::TypeInfo(typeid(T&)) == type;
}
#define MAKE_TYPE(type) if(acceptsType<type>(parameter->typeInfo)) return new QNodeProxy<type, false>(parent, parameter);
#define MAKE_TYPE_(type) if(acceptsType<type>(parameter->typeInfo)) return new QNodeProxy<type, true>(parent, parameter);


IQNodeProxy* dispatchParameter(IQNodeInterop* parent, Parameters::Parameter::Ptr parameter, EagleLib::Node::Ptr node)
{
    if(parameter->type & Parameters::Parameters::ParameterInput)
    {
        // Create a ui for finding relavent inputs
        return new QInputProxy(parent, parameter, node);
    }
    typedef std::vector<std::pair<int,double>> vec_ID;
    typedef std::vector<std::pair<double,double>> vec_DD;
    typedef std::vector<std::pair<double,int>> vec_DI;
    typedef std::vector<std::pair<int, int>> vec_II;
    if(parameter->type & Parameters::Parameters::ParameterControl)
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

    if(parameter->type & Parameters::Parameters::ParameterOutput || parameter->type & Parameters::Parameters::ParameterState)
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
*/

