#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "qpluginloader.h"
#include "Manager.h"
#include <qfiledialog.h>
#include <nodes/Node.h>
#include <QNodeWidget.h>

#include "Plugins.h"
#include <opencv2/calib3d.hpp>
#include <qgraphicsproxywidget.h>
#include "QGLWidget"
#include <QGraphicsSceneMouseEvent>
#include <Manager.h>
#include "settingdialog.h"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/common.hpp>
#include <boost/log/exceptions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/expressions/attr.hpp>
#include <boost/log/attributes/time_traits.hpp>
#include <boost/log/expressions/formatters.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/attributes/named_scope.hpp>
#include <boost/log/expressions/formatters/named_scope.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>

#include <UI/InterThread.hpp>

int static_errorHandler( int status, const char* func_name,const char* err_msg, const char* file_name, int line, void* userdata )
{
	return 0;
}

static void processThread(std::vector<EagleLib::Node::Ptr>* parentList, boost::timed_mutex *mtx);
static void process(std::vector<EagleLib::Node::Ptr>* parentList, boost::timed_mutex *mtx);
MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow),
	rccSettings(new RCCSettingsDialog(this)),
	plotWizardDialog(new PlotWizardDialog(this)),
	settingsDialog(new SettingDialog(this))
{
	boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
	boost::log::add_common_attributes();
	boost::log::core::get()->add_global_attribute("Scope", boost::log::attributes::named_scope());
    // https://gist.github.com/xiongjia/e23b9572d3fc3d677e3d
	/*boost::log::add_console_LOG_TRIVIAL(std::cout, boost::log::keywords::format = (
		boost::log::expressions::stream
		<< boost::log::expressions::format_date_time< boost::posix_time::ptime >("TimeStamp", "[%H:%M:%S") << " "
		<< boost::log::expressions::attr< boost::thread::id >("ThreadID") << "]"
		<< boost::log::trivial::severity << " "
		<< boost::log::expressions::smessage));
	boost::log::add_common_attributes();
	boost::log::core::get()->add_global_attribute("Scope", boost::log::attributes::named_scope());
	boost::log::core::get()->add_global_attribute("ThreadID", boost::log::attributes::current_thread_id());*/
	//boost::log::add_console_LOG_TRIVIAL(std::cout, boost::log::keywords::format = "%TimeStamp% - %LineID% %Severity% %ThreadID% - %Message%");

	auto consoleFmtTimeStamp = boost::log::expressions::
		format_date_time<boost::posix_time::ptime>("TimeStamp", "%M:%S.%f");
	auto fmtThreadId = boost::log::expressions::
		attr<boost::log::attributes::current_thread_id::value_type>("ThreadID");
	auto fmtSeverity = boost::log::expressions::
		attr<boost::log::trivial::severity_level>("Severity");

	auto fmtScope = boost::log::expressions::format_named_scope("Scope",
		boost::log::keywords::format = "%n(%f:%l)",
		boost::log::keywords::iteration = boost::log::expressions::reverse,
		boost::log::keywords::depth = 2);

	boost::log::formatter consoleFmt =
		boost::log::expressions::format("%1%<%2%,%3%> %4%")
		% consoleFmtTimeStamp					// 1
		% fmtThreadId							// 2
		% fmtSeverity							// 3
		% boost::log::expressions::smessage;	// 4

	auto consoleSink = boost::log::add_console_log(std::clog);
	consoleSink->set_formatter(consoleFmt);


	auto fmtTimeStamp = boost::log::expressions::
		format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S.%f");
	boost::log::formatter logFmt =
		boost::log::expressions::format("[%1%] (%2%) [%3%] [%4%] %5%")
		% fmtTimeStamp 
		% fmtThreadId 
		% fmtSeverity
		% fmtScope
		% boost::log::expressions::smessage;
	

	auto fsSink = boost::log::add_file_log(
		boost::log::keywords::file_name = "test_%Y-%m-%d_%H-%M-%S.%N.log",
		boost::log::keywords::rotation_size = 10 * 1024 * 1024,
		boost::log::keywords::min_free_space = 30 * 1024 * 1024,
		boost::log::keywords::open_mode = std::ios_base::app);
	fsSink->set_formatter(logFmt);
	fsSink->locked_backend()->auto_flush(true);


    qRegisterMetaType<std::string>("std::string");
    qRegisterMetaType<cv::cuda::GpuMat>("cv::cuda::GpuMat");
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<EagleLib::Node::Ptr>("EagleLib::Node::Ptr");
    qRegisterMetaType<EagleLib::Node*>("EagleLib::Node*");
	qRegisterMetaType<boost::log::trivial::severity_level>("boost::log::trivial::severity_level");
    qRegisterMetaType<boost::function<cv::Mat(void)>>("boost::function<cv::Mat(void)>");
    qRegisterMetaType<boost::function<void(void)>>("boost::function<void(void)>");
    qRegisterMetaType<boost::function<void()>>("boost::function<void()>");
    qRegisterMetaType<Parameters::Parameter::Ptr>("Parameters::Parameter::Ptr");
    qRegisterMetaType<size_t>("size_t");

    ui->setupUi(this);

    fileMonitorTimer = new QTimer(this);
    fileMonitorTimer->start(1000);
    connect(fileMonitorTimer, SIGNAL(timeout()), this, SLOT(onTimeout()));
    nodeListDialog = new NodeListDialog(this);
    nodeListDialog->hide();
    connect(nodeListDialog, SIGNAL(nodeConstructed(EagleLib::Node::Ptr)),
        this, SLOT(onNodeAdd(EagleLib::Node::Ptr)));
	
	nodeGraph = new QGraphicsScene(this);
    //connect(nodeGraph, SIGNAL(selectionChanged()), this, SLOT(on_selectionChanged()));
	nodeGraphView = new NodeView(nodeGraph);
	connect(nodeGraphView, SIGNAL(selectionChanged(QGraphicsProxyWidget*)), this, SLOT(onSelectionChanged(QGraphicsProxyWidget*)));
	nodeGraphView->setInteractive(true);
    nodeGraphView->setViewport(new QGLWidget());
    nodeGraphView->setDragMode(QGraphicsView::ScrollHandDrag);
    ui->gridLayout->addWidget(nodeGraphView, 2, 0, 1,4);
    currentSelectedNodeWidget = nullptr;
    Parameters::UI::UiCallbackService::Instance()->setCallback(boost::bind(&MainWindow::processingThread_uiCallback, this, _1));
    rccSettings->hide();
    plotWizardDialog->hide();


    cv::redirectError(&static_errorHandler);



    // For some reason this doesn't work on linux :/

    //connect(this, SIGNAL(uiCallback(boost::function<void()>)),
    //        this, SLOT(on_uiCallback(boost::function<void()>)), Qt::QueuedConnection);
    connect(this, &MainWindow::uiCallback, this, &MainWindow::on_uiCallback, Qt::QueuedConnection);

    //connect(this, SIGNAL(uiCallback()), this, SLOT(on_uiCallback()), Qt::QueuedConnection);

    connect(nodeGraphView, SIGNAL(plotData(Parameters::Parameter::Ptr)), plotWizardDialog, SLOT(plotParameter(Parameters::Parameter::Ptr)));
    //connect(this, SIGNAL(eLOG_TRIVIAL(QString)), this, SLOT(LOG_TRIVIAL(QString)), Qt::QueuedConnection);
    connect(this, SIGNAL(oglDisplayImage(std::string,cv::cuda::GpuMat)), this, SLOT(onOGLDisplay(std::string,cv::cuda::GpuMat)), Qt::QueuedConnection);
    connect(this, SIGNAL(qtDisplayImage(std::string,cv::Mat)), this, SLOT(onQtDisplay(std::string,cv::Mat)), Qt::QueuedConnection);
    connect(nodeGraphView, SIGNAL(startThread()), this, SLOT(startProcessingThread()));
    connect(nodeGraphView, SIGNAL(stopThread()), this, SLOT(stopProcessingThread()));
    connect(nodeGraphView, SIGNAL(widgetDeleted(QNodeWidget*)), this, SLOT(onWidgetDeleted(QNodeWidget*)));
    connect(ui->actionSave, SIGNAL(triggered()), this, SLOT(onSaveClicked()));
    connect(ui->actionLoad, SIGNAL(triggered()), this, SLOT(onLoadClicked()));
    connect(ui->actionLoad_Plugin, SIGNAL(triggered()), this, SLOT(onLoadPluginClicked()));
    connect(this, SIGNAL(uiNeedsUpdate()), this, SLOT(onUiUpdate()), Qt::QueuedConnection);

    connect(this, SIGNAL(onNewParameter(EagleLib::Node*)), this, SLOT(on_NewParameter(EagleLib::Node*)), Qt::QueuedConnection);

    connect(ui->actionRCC_settings, SIGNAL(triggered()), this, SLOT(displayRCCSettings()));
    //connect(plotWizardDialog, SIGNAL(on_plotAdded(PlotWindow*)), this, SLOT(onPlotAdd(PlotWindow*)));
    connect(this, SIGNAL(pluginLoaded()), plotWizardDialog, SLOT(setup()));

	/* Instantiate several useful types since compilation is currently setup to not compile against the types used in eagle lib */
    Parameters::TypedParameter<Parameters::WriteDirectory>("Instantiation");
    Parameters::TypedParameter<Parameters::WriteFile>("Instantiation");
    Parameters::TypedParameter<Parameters::ReadDirectory>("Instantiation");
    Parameters::TypedParameter<Parameters::ReadFile>("Instantiation");
    Parameters::TypedParameter<Parameters::EnumParameter>("Instantiation");
	Parameters::TypedParameter<boost::filesystem::path>("Instantiation");
	Parameters::TypedParameter<boost::function<void(void)>>("Instantiation");

    EagleLib::UIThreadCallback::getInstance().setUINotifier(boost::bind(&MainWindow::uiNotifier, this));
    boost::function<void(const std::string&, int)> f = boost::bind(&MainWindow::onCompileLog, this, _1, _2);
    EagleLib::NodeManager::getInstance().setCompileCallback(f);
    QDir dir(QDir::currentPath());

#ifdef _MSC_VER
	std::string str = dir.absolutePath().toStdString();
#ifdef _DEBUG
	dir.cd("../Debug");
#else
	dir.cd("../Release");
#endif
	str = dir.absolutePath().toStdString();
	QFileInfoList files = dir.entryInfoList(QStringList("*.dll"));
#else
	dir.cd("Plugins");
    QFileInfoList files = dir.entryInfoList(QStringList("lib*.so"));
#endif
    for(int i = 0; i < files.size(); ++i)
    {
        if(files[i].isFile())
        {
            //EagleLib::NodeManager::getInstance().loadModule(files[i].absoluteFilePath().toStdString());
            EagleLib::loadPlugin(files[i].absoluteFilePath().toStdString());
        }
    }
    nodeListDialog->update();
    emit pluginLoaded();


    startProcessingThread();
	rccSettings->updateDisplay();
}

MainWindow::~MainWindow()
{
	stopProcessingThread();
	cv::destroyAllWindows();
    delete ui;
}
void MainWindow::closeEvent(QCloseEvent *event)
{
    stopProcessingThread();
    cv::destroyAllWindows();
    QMainWindow::closeEvent(event);
}
void MainWindow::displayRCCSettings()
{
    rccSettings->show();
}

void MainWindow::onCompileLog(const std::string& msg, int level)
{
	emit eLog(QString::fromStdString(msg));
	//ui->console->appendPlainText(QString::fromStdString(msg));
}

void MainWindow::on_pushButton_clicked()
{
    nodeListDialog->show();
}
void
MainWindow::onError(const std::string &error)
{
    return;
}
void
MainWindow::onStatus(const std::string &status)
{

}
void MainWindow::onPlotAdd(PlotWindow* plot)
{
    ui->plotTabs->addTab(plot,plot->getPlotName());
}
void MainWindow::onPlotRemove(PlotWindow* plot)
{

}

void saveWidgetPosition(NodeView* nodeView, cv::FileStorage& fs, EagleLib::Node::Ptr node, int& count)
{
    QGraphicsProxyWidget* widget = nodeView->getWidget(node->GetObjectId());
    if(widget)
    {
        fs << "{:";
        fs << "Name" << node->fullTreeName;
        fs << "x" << widget->pos().x();
        fs << "y" << widget->pos().y();
        fs << "}";
        ++count;
    }
    for(size_t i = 0; i < node->children.size(); ++i)
    {
        saveWidgetPosition(nodeView, fs, node->children[i], count);
    }
}

void
MainWindow::onSaveClicked()
{
    auto file = QFileDialog::getSaveFileName(this, "File to save to");
    if(file.size() == 0)
        return;
    stopProcessingThread();
    cv::FileStorage fs;
    fs.open(file.toStdString(), cv::FileStorage::WRITE);
    EagleLib::NodeManager::getInstance().saveNodes(parentList,fs);
    // Save node widget positions
    fs << "WidgetPositions" << "[";
    int count = 0;
    for(size_t i = 0; i <parentList.size(); ++i)
    {
        saveWidgetPosition(nodeGraphView, fs, parentList[i], count);
    }
    fs << "]";
    fs.release();
    startProcessingThread();
}

void MainWindow::on_uiCallback(boost::function<void()> f)
{
    if(f)
        f();
}


void MainWindow::processingThread_uiCallback(boost::function<void ()> f)
{
    emit uiCallback(f);
}

void
MainWindow::onLoadClicked()
{
    auto file = QFileDialog::getOpenFileName(this, "Load file");
    if(file.size() == 0)
        return;
    stopProcessingThread();
    std::vector<EagleLib::Node::Ptr> nodes = EagleLib::NodeManager::getInstance().loadNodes(file.toStdString());
    cv::FileStorage fs;
	try
	{
		fs.open(file.toStdString(), cv::FileStorage::READ);
	}
	catch (cv::Exception &e)
	{
		BOOST_LOG_TRIVIAL(error) << "Failed to load file " << file.toStdString() << " " << e.what();
		return;
	}
    
    positionMap.clear();
    try
    {
        cv::FileNode positions = fs["WidgetPositions"];
        for(cv::FileNodeIterator it = positions.begin(); it != positions.end(); ++it)
        {
            std::string name = (std::string)(*it)["Name"];
            float x = (float)(*it)["x"];
            float y = (float)(*it)["y"];
            //cv::Vec2f pos = (cv::Vec2f)(*it)["Position"];
            positionMap[name] = cv::Vec2f(x,y);
        }

    }catch(cv::Exception &e)
    {
        std::cout << e.what() << std::endl;
    }


    if(nodes.size())
    {
        for(size_t i = 0; i < widgets.size(); ++i)
        {
            delete widgets[i];
        }
        widgets.clear();
        currentSelectedNodeWidget = nullptr;
        parentList = nodes;
        for(size_t i =0; i < parentList.size(); ++i)
        {
            addNode(parentList[i]);
        }
        for(size_t i = 0; i < widgets.size(); ++i)
        {
            widgets[i]->updateUi();
        }
    }
    startProcessingThread();
}
void MainWindow::onLoadPluginClicked()
{
#ifdef _MSC_VER
    QString filename = QFileDialog::getOpenFileName(this, "Select file", QString(), QString("*.dll"));
#else
    QString filename = QFileDialog::getOpenFileName(this, "Select file", QString(), QString("*.so"));
#endif
    if(filename.size() == 0)
        return;
	filename = QDir::toNativeSeparators(filename);
    //if(EagleLib::NodeManager::getInstance().loadModule(filename.toStdString()))
    if(EagleLib::loadPlugin(filename.toStdString()))
    {
        nodeListDialog->update();
        emit pluginLoaded();
    }
}
void MainWindow::on_NewParameter(EagleLib::Node* node)
{
    for(size_t i = 0; i < widgets.size(); ++i)
    {
        widgets[i]->updateUi(true, node);
    }
}
// Called from the processing thread, that's why we need a queued connection here.
void MainWindow::newParameter(EagleLib::Node* node)
{
    emit onNewParameter(node);
}

void
MainWindow::onTimeout()
{
    static bool swapRequired = false;
    static bool joined = false;
    auto start = boost::posix_time::microsec_clock::universal_time();
    //EagleLib::UIThreadCallback::getInstance().processAllCallbacks();
    auto ms = boost::posix_time::time_duration(boost::posix_time::microsec_clock::universal_time() - start).total_milliseconds();
    if(ms > 30)
    {
        LOG_TRIVIAL(warning) << "UI callbacks taking " << ms << " milliseconds to complete";
    }
    //Parameters::UI::UiCallbackService::Instance()->run();
    for(size_t i = 0; i < widgets.size(); ++i)
    {
        widgets[i]->updateUi(false);
    }

    if(swapRequired)
    {
        if(!processingThread.try_join_for(boost::chrono::milliseconds(200)) && !joined)
        {
            LOG_TRIVIAL(info) <<"Processing thread not joined, cannot perform object swap";
            return;
        }else
        {
			LOG_TRIVIAL(info) << "Processing thread joined";
            joined = true;
        }
        if(EagleLib::NodeManager::getInstance().CheckRecompile(true))
        {
           // Still compiling
			LOG_TRIVIAL(info) << "Still compiling";
        }else
        {
			LOG_TRIVIAL(info) << "Recompile complete";
            processingThread = boost::thread(boost::bind(&processThread, &parentList, &parentMtx));
            swapRequired = false;
        }
        return;
    }
    if(EagleLib::NodeManager::getInstance().CheckRecompile(false))
    {
		LOG_TRIVIAL(info) << "Recompiling.....";
        swapRequired = true;
        joined = false;
        processingThread.interrupt();
        return;
    }
}
void MainWindow::log(QString message)
{
    ui->console->appendPlainText(message);
}
// Called from the processing thread
void MainWindow::oglDisplay(cv::cuda::GpuMat img, EagleLib::Node* node)
{
    emit oglDisplayImage(node->fullTreeName, img);
}
void MainWindow::qtDisplay(cv::Mat img, EagleLib::Node *node)
{
    emit qtDisplayImage(node->fullTreeName, img);
}

void MainWindow::onOGLDisplay(std::string name, cv::cuda::GpuMat img)
{
    cv::namedWindow(name, cv::WINDOW_OPENGL);
    cv::imshow(name, img);
}
void MainWindow::onQtDisplay(std::string name, cv::Mat img)
{
    cv::namedWindow(name);
    cv::imshow(name, img);
}
void MainWindow::onQtDisplay(boost::function<cv::Mat(void)> function, EagleLib::Node* node)
{
    cv::Mat img = function();
    cv::namedWindow(node->fullTreeName);
    cv::imshow(node->fullTreeName, img);
}

void MainWindow::addNode(EagleLib::Node::Ptr node)
{
    if(node->nodeName == "OGLImageDisplay")
    {
        node->gpuDisplayCallback = boost::bind(&MainWindow::oglDisplay, this, _1, _2);
    }
    if(node->nodeName == "QtImageDisplay")
    {
        node->cpuDisplayCallback = boost::bind(&MainWindow::qtDisplay, this, _1, _2);
    }
    if(node->nodeName == "KeyPointDisplay")
    {
        node->cpuDisplayCallback = boost::bind(&MainWindow::qtDisplay, this, _1, _2);
    }
    node->onParameterAdded.connect(boost::bind(&MainWindow::newParameter,this, _1));
    QNodeWidget* nodeWidget = new QNodeWidget(0, node);
    connect(nodeWidget, SIGNAL(parameterClicked(Parameters::Parameter::Ptr, QPoint)), nodeGraphView, SLOT(on_parameter_clicked(Parameters::Parameter::Ptr, QPoint)));
    auto proxyWidget = nodeGraph->addWidget(nodeWidget);

    auto itr = positionMap.find(node->fullTreeName);
    if(itr != positionMap.end())
    {
        cv::Vec2f pt = itr->second;
        proxyWidget->setPos(pt.val[0], pt.val[1]);
    }else
    {
        if (currentSelectedNodeWidget)
        {
            int yOffset = 0;
            if(currentNode != nullptr)
            {
                auto parentNode = currentNode->getParent();
                if(parentNode)
                {
                    auto itr = std::find(parentNode->children.begin(), parentNode->children.end(), node);
                    if(itr != parentNode->children.end())
                    {
                        auto idx = std::distance(itr, parentNode->children.begin());
                        yOffset -= idx*100;
                    }
                }
                auto parentWidget = nodeGraphView->getParent(node);
                if(parentWidget)
                    proxyWidget->setPos(parentWidget->pos() + QPointF(500, yOffset));
                else
                    proxyWidget->setPos(currentSelectedNodeWidget->pos() + QPointF(500, yOffset));
            }
        }
    }
    nodeGraphView->addWidget(proxyWidget, node->GetObjectId());
    nodeGraphView->setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);

    QGraphicsProxyWidget* prevWidget = currentSelectedNodeWidget;
    auto prevNode = currentNode;
    widgets.push_back(nodeWidget);
    currentSelectedNodeWidget = proxyWidget;
    currentNode = node;
    for(size_t i = 0; i < node->children.size(); ++i)
    {
        addNode(node->children[i]);
    }
    if(!prevWidget)
    {
        nodeWidget->setSelected(true);
        currentSelectedNodeWidget = proxyWidget;
        currentNode = prevNode;
    }else
    {
        currentSelectedNodeWidget = prevWidget;
    }
}
void MainWindow::updateLines()
{

}

void 
MainWindow::onNodeAdd(EagleLib::Node::Ptr node)
{	
    EagleLib::Node::Ptr prevNode = currentNode;
    if(currentNode != nullptr)
    {
        boost::recursive_mutex::scoped_lock(currentNode->mtx);
        currentNode->addChild(node);
    }else
    {

    }
    addNode(node);
    for(size_t i = 0; i < widgets.size(); ++i)
    {
        widgets[i]->updateUi();
    }
    if(node->getParent() == nullptr)
    {
        boost::timed_mutex::scoped_lock lock(parentMtx, boost::chrono::milliseconds(1000));
        if(lock.owns_lock())
        {
            parentList.push_back(node);
        }else
        {
            stopProcessingThread();
            parentList.push_back(node);
            startProcessingThread();
        }


    }
    if(currentNode == nullptr)
    {
        currentNode = node;
    }else
    {
        currentNode = prevNode;
    }
}
void MainWindow::onWidgetDeleted(QNodeWidget* widget)
{
    auto itr = std::find(widgets.begin(), widgets.end(), widget);
    if(itr != widgets.end())
        widgets.erase(itr);
    boost::mutex::scoped_lock(parentMtx);
    auto parentItr = std::find(parentList.begin(), parentList.end(), widget->getNode());
    if(parentItr != parentList.end())
        parentList.erase(parentItr);
}
void
MainWindow::uiNotifier()
{
    emit uiNeedsUpdate();
}
void MainWindow::onUiUpdate()
{
    EagleLib::UIThreadCallback::getInstance().processAllCallbacks();
}

void
MainWindow::onSelectionChanged(QGraphicsProxyWidget* widget)
{
    if(widget == nullptr)
    {
        if(currentSelectedNodeWidget)
            currentSelectedNodeWidget->setZValue(0);
        currentSelectedNodeWidget = nullptr;
        currentNode = EagleLib::Node::Ptr();
        return;
    }
    if(currentSelectedNodeWidget)
        if(auto oldWidget = dynamic_cast<QNodeWidget*>(currentSelectedNodeWidget->widget()))
            oldWidget->setSelected(false);
    currentSelectedNodeWidget = widget;
    currentSelectedNodeWidget->setZValue(1);
    if(auto ptr = dynamic_cast<QNodeWidget*>(widget->widget()))
    {
        currentNode = ptr->getNode();
        ptr->setSelected(true);
    }
}


void process(std::vector<EagleLib::Node::Ptr>* nodes, boost::timed_mutex* mtx)
{
    static std::vector<cv::cuda::GpuMat> images;
    static std::vector<cv::cuda::Stream> streams;
    boost::timed_mutex::scoped_lock lock(*mtx);
    if(nodes->size() != streams.size())
        streams.resize(nodes->size());
    if(nodes->size() != images.size())
        images.resize(nodes->size());
    for (size_t i = 0; i < nodes->size(); ++i)
    {
        (*nodes)[i]->process(images[i], streams[i]);
    }

}

void processThread(std::vector<EagleLib::Node::Ptr>* parentList, boost::timed_mutex *mtx)
{
	BOOST_LOG_TRIVIAL(info) << "Processing thread started" << std::endl;
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::ptime end = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta;
    while (!boost::this_thread::interruption_requested())
    {
        try
        {
			EagleLib::ProcessingThreadCallback::Run();
			Parameters::UI::ProcessingThreadCallbackService::run();
            process(parentList, mtx);
			end = boost::posix_time::microsec_clock::universal_time();
			delta = end - start;
			start = end;
			if (delta.total_milliseconds() < 15 || parentList->size() == 0)
				boost::this_thread::sleep_for(boost::chrono::milliseconds(15 - delta.total_milliseconds()));
        }catch(boost::thread_interrupted& err)
        {
			err;
			BOOST_LOG_TRIVIAL(info) << "Processing thread interrupted";
            break;
        }
        
    }
    BOOST_LOG_TRIVIAL(info) << "Interrupt requested, processing thread ended";
}
void MainWindow::startProcessingThread()
{
    processingThread = boost::thread(boost::bind(&processThread, &parentList, &parentMtx));
}
// So the problem here is that cv::imshow operates on the main thread, thus if the main thread blocks
// because it's waiting for processingThread to join, then cv::imshow will block, thus causing deadlock.
// What we need is a signal beforehand that will disable all imshow's before a delete.
void MainWindow::stopProcessingThread()
{
    processingThread.interrupt();
    processingThread.join();
}

void MainWindow::on_actionLog_settings_triggered()
{
    settingsDialog->show();   
}
void MainWindow::on_btnClear_clicked()
{
    stopProcessingThread();

    parentList.clear();
    startProcessingThread();
}

void MainWindow::on_btnStart_clicked()
{
    startProcessingThread();
}

void MainWindow::on_btnStop_clicked()
{
    stopProcessingThread();
}
