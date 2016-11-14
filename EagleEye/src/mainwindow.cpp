#define PARAMETERS_GENERATE_UI
#define HAVE_OPENCV
#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "FileOrFolderDialog.h"
#include <qfiledialog.h>
#include <qgraphicsproxywidget.h>
#include "QGLWidget"
#include <QGraphicsSceneMouseEvent>

#include "settingdialog.h"
#include "dialog_network_stream_selection.h"
#include <QNodeWidget.h>

#include "bookmark_dialog.h"
//#include <GL/gl.h>
//#include <GL/glu.h>


#include <shared_ptr.hpp>
#include <EagleLib/rcc/SystemTable.hpp>
#include <EagleLib/utilities/ogl_allocators.h>
#include "EagleLib/utilities/CpuMatAllocators.h"
#include <EagleLib/Logging.h>

#include "EagleLib/utilities/BufferPool.hpp"

#include "EagleLib/Signals.h"
#include "EagleLib/logger.hpp"
#include "EagleLib/Plugins.h"
#include <EagleLib/Nodes/Node.h>
#include <EagleLib/Nodes/NodeFactory.h>
#include <EagleLib/utilities/ColorMapperFactory.hpp>

#include <MetaObject/Logging/Log.hpp>
#include <signal.h>



void sig_handler(int s)
{
    LOG(error) << "Caught signal " << s << " with callstack:\n" << mo::print_callstack(0,true);
    if (s == 2)
        exit(EXIT_FAILURE);
}
static void process(std::vector<EagleLib::Nodes::Node::Ptr>* parentList, boost::timed_mutex *mtx);
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    rccSettings(new RCCSettingsDialog(this)),
    plotWizardDialog(new PlotWizardDialog(this)),
    settingsDialog(new SettingDialog(this))
{
    EagleLib::ColorMapperFactory::Instance()->Save("test.xml");
    signal(SIGINT, sig_handler);
    signal(SIGILL, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGSEGV, sig_handler);

    // Create the processing thread opengl context for creating buffers and uploading them
    processing_thread_context = nullptr;
    processing_thread_upload_window = nullptr;
    //updateParameterPtr("file load path", &file_load_path);
    VariableStorage::Instance()->LoadParams(this, "MainWindow");
    
    cv::Mat::setDefaultAllocator(EagleLib::CpuPinnedAllocator::instance());

    EagleLib::CpuDelayedDeallocationPool::instance()->deallocation_delay = 1000;
    
    EagleLib::SetupLogging();
    //EagleLib::ui_collector::addGenericCallbackHandler(boost::bind(&MainWindow::process_log_message, this, _1, _2));
    //logging_connection = EagleLib::ui_collector::get_log_handler().connect(std::bind(&MainWindow::process_log_message, this, std::placeholders::_1, std::placeholders::_2));
    

    qRegisterMetaType<std::string>("std::string");
    qRegisterMetaType<cv::cuda::GpuMat>("cv::cuda::GpuMat");
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<EagleLib::Nodes::Node::Ptr>("EagleLib::Nodes::Node::Ptr");
    qRegisterMetaType<EagleLib::Nodes::Node*>("EagleLib::Nodes::Node*");
    qRegisterMetaType<boost::log::trivial::severity_level>("boost::log::trivial::severity_level");
    qRegisterMetaType<boost::function<cv::Mat(void)>>("boost::function<cv::Mat(void)>");
    qRegisterMetaType<boost::function<void(void)>>("boost::function<void(void)>");
    qRegisterMetaType<boost::function<void()>>("boost::function<void()>");
    qRegisterMetaType<mo::IParameter::Ptr>("mo::IParameter::Ptr");
    qRegisterMetaType<size_t>("size_t");
    qRegisterMetaType<std::pair<void*,mo::TypeInfo>>("std::pair<void*,mo::TypeInfo>");

    ui->setupUi(this);

    fileMonitorTimer = new QTimer(this);
    fileMonitorTimer->start(1000);
    QObject::connect(fileMonitorTimer, SIGNAL(timeout()), this, SLOT(onTimeout()));
    nodeListDialog = new NodeListDialog(this);
    nodeListDialog->setParent(this);
    nodeListDialog->hide();
    //nodeListDialog->setup_signals(&_ui_manager);
    //nodeListDialog->SetupSignals(&_ui_manager);
    //_signal_connections.push_back(_ui_manager.connect<void(std::string)>("add_node", std::bind(&MainWindow::onNodeAdd, this, std::placeholders::_1), this, Signals::get_this_thread()));

    /*connect(nodeListDialog, SIGNAL(nodeConstructed(EagleLib::Nodes::Node::Ptr)),
        this, SLOT(onNodeAdd(EagleLib::Nodes::Node::Ptr)));*/
    
    nodeGraph = new QGraphicsScene(this);
    //connect(nodeGraph, SIGNAL(selectionChanged()), this, SLOT(on_selectionChanged()));
    nodeGraphView = new NodeView(nodeGraph);
    QObject::connect(nodeGraphView, SIGNAL(selectionChanged(QGraphicsProxyWidget*)), this, SLOT(onSelectionChanged(QGraphicsProxyWidget*)));
    nodeGraphView->setInteractive(true);
    nodeGraphView->setViewport(new QGLWidget());
    nodeGraphView->setDragMode(QGraphicsView::ScrollHandDrag);
    ui->gridLayout->addWidget(nodeGraphView, 2, 0, 1,4);
    currentSelectedNodeWidget = nullptr;
    currentSelectedStreamWidget = nullptr;
    //Parameters::UI::UiCallbackService::Instance()->setCallback(boost::bind(&MainWindow::processingThread_uiCallback, this, _1, _2));
    rccSettings->hide();
    plotWizardDialog->hide();

    bookmarks = new bookmark_dialog(this);
    bookmarks->setParent(this);
    bookmarks->hide();

    
    persistence_timer = new QTimer(this);
    persistence_timer->start(500); // save setting every half second
    QObject::connect(bookmarks, SIGNAL(open_file(QString)), this, SLOT(load_file(QString)));
    QObject::connect(persistence_timer, SIGNAL(timeout()), this, SLOT(on_persistence_timeout()));
    QObject::connect(this, &MainWindow::uiCallback, this, &MainWindow::on_uiCallback, Qt::QueuedConnection);
    QObject::connect(nodeGraphView, SIGNAL(plotData(mo::IParameter*)), plotWizardDialog, SLOT(plotParameter(mo::IParameter*)));
    QObject::connect(this, SIGNAL(eLog(QString)), this, SLOT(log(QString)), Qt::QueuedConnection);
    QObject::connect(this, SIGNAL(oglDisplayImage(std::string,cv::cuda::GpuMat)), this, SLOT(onOGLDisplay(std::string,cv::cuda::GpuMat)), Qt::QueuedConnection);
    QObject::connect(this, SIGNAL(qtDisplayImage(std::string,cv::Mat)), this, SLOT(onQtDisplay(std::string,cv::Mat)), Qt::QueuedConnection);
    QObject::connect(nodeGraphView, SIGNAL(startThread()), this, SLOT(startProcessingThread()));
    QObject::connect(nodeGraphView, SIGNAL(stopThread()), this, SLOT(stopProcessingThread()));
    QObject::connect(nodeGraphView, SIGNAL(widgetDeleted(QNodeWidget*)), this, SLOT(onWidgetDeleted(QNodeWidget*)));
    QObject::connect(nodeGraphView, SIGNAL(widgetDeleted(DataStreamWidget*)), this, SLOT(onWidgetDeleted(DataStreamWidget*)));
    QObject::connect(ui->actionSave, SIGNAL(triggered()), this, SLOT(onSaveClicked()));
    QObject::connect(ui->actionLoad, SIGNAL(triggered()), this, SLOT(onLoadClicked()));
    QObject::connect(ui->actionOpen_file, SIGNAL(triggered()), this, SLOT(onLoadFileClicked()));
    QObject::connect(ui->actionOpen_directory, SIGNAL(triggered()), this, SLOT(onLoadDirectoryClicked()));
    QObject::connect(ui->actionLoad_Plugin, SIGNAL(triggered()), this, SLOT(onLoadPluginClicked()));
    QObject::connect(this, SIGNAL(uiNeedsUpdate()), this, SLOT(onUiUpdate()), Qt::QueuedConnection);
    //connect(ui->actionBookmarks, SIGNAL(triggered()), this, SLOT()
    QObject::connect(this, SIGNAL(onNewParameter(EagleLib::Nodes::Node*)), this, SLOT(on_NewParameter(EagleLib::Nodes::Node*)), Qt::QueuedConnection);

    QObject::connect(ui->actionRCC_settings, SIGNAL(triggered()), this, SLOT(displayRCCSettings()));
    //connect(plotWizardDialog, SIGNAL(on_plotAdded(PlotWindow*)), this, SLOT(onPlotAdd(PlotWindow*)));
    QObject::connect(this, SIGNAL(pluginLoaded()), plotWizardDialog, SLOT(setup()));
#ifdef _MSC_VER
#ifdef _DEBUG
    LOG(info) << "EagleEye log messages, built with msvc version " << _MSC_VER << " in Debug mode.";
#else
    LOG(info) << "EagleEye log messages, built with msvc version " << _MSC_VER << " in Release mode.";
#endif
#else

#endif
    
    //EagleLib::UIThreadCallback::getInstance().setUINotifier(boost::bind(&MainWindow::uiNotifier, this));
    std::function<void(const std::string, int)> f = std::bind(&MainWindow::onCompileLog, this, std::placeholders::_1, std::placeholders::_2);
    //EagleLib::ObjectManager::Instance().setCompileCallback(f);
    mo::MetaObjectFactory::Instance()->SetCompileCallback(f);
    QDir dir(QDir::currentPath());

#ifdef _MSC_VER
    std::string str = dir.absolutePath().toStdString();
#ifdef _DEBUG
    dir.cd("../Debug");
#else
    dir.cd("../RelWithDebInfo");
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
            EagleLib::loadPlugin(files[i].absoluteFilePath().toStdString());
        }
    }
    nodeListDialog->update();
    emit pluginLoaded();
    
    /*auto allocator = PerModuleInterface::GetInstance()->GetSystemTable()->GetSingleton<EagleLib::ogl_allocator>();
    if (allocator)
    {
        cv::cuda::GpuMat::setDefaultAllocator(allocator);
    }*/
    cv::cuda::GpuMat::setDefaultAllocator(EagleLib::CombinedAllocator::Instance(100000000, 500000));
    rccSettings->updateDisplay();
    
    _sig_manager = mo::RelayManager::Instance();
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    table->SetSingleton(_sig_manager);
    if (table)
    {
        this->SetupSignals(_sig_manager);
    }
    startProcessingThread();
    mo::ThreadRegistry::Instance()->RegisterThread(mo::ThreadRegistry::GUI);
    mo::ThreadSpecificQueue::RegisterNotifier(std::bind(&MainWindow::uiNotifier, this));

}

MainWindow::~MainWindow()
{
    VariableStorage::Instance()->SaveParams(this, "MainWindow");
    stopProcessingThread();
    cv::destroyAllWindows();
    EagleLib::ShutdownLogging();
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

void saveWidgetPosition(NodeView* nodeView, cv::FileStorage& fs, EagleLib::Nodes::Node::Ptr node, int& count)
{
    QGraphicsProxyWidget* widget = nodeView->getWidget(node->GetObjectId());
    if(widget)
    {
        fs << "{:";
        fs << "Name" << node->GetTreeName();
        fs << "x" << widget->pos().x();
        fs << "y" << widget->pos().y();
        fs << "}";
        ++count;
    }
    auto children = node->GetChildren();
    for(size_t i = 0; i < children.size(); ++i)
    {
        saveWidgetPosition(nodeView, fs, children[i], count);
    }
}

void
MainWindow::onSaveClicked()
{
    auto file = QFileDialog::getSaveFileName(this, "File to save to");
    if(file.size() == 0)
        return;
    stopProcessingThread();
    //cv::FileStorage fs;
    //fs.open(file.toStdString(), cv::FileStorage::WRITE);
    //EagleLib::NodeManager::getInstance().saveNodes(parentList,fs);

    // Save node widget positions
    /*fs << "WidgetPositions" << "[";
    int count = 0;
    for(size_t i = 0; i <parentList.size(); ++i)
    {
        saveWidgetPosition(nodeGraphView, fs, parentList[i], count);
    }
    fs << "]";
    fs.release();
    startProcessingThread();*/
}

void MainWindow::on_uiCallback(boost::function<void()> f, std::pair<void*, mo::TypeInfo> source)
{
    static boost::posix_time::ptime last_end;
    try
    {
        //rmt_ScopedCPUSample(on_uiCallback);
        if (f)
        {
            //if(Parameters::UI::InvalidCallbacks::check_valid(source.first))
              //  f();
            
        }
    }
    catch (cv::Exception &e)
    {
        LOG(error) << e.what();
    }
    catch (...)
    {

    }
    
}


void MainWindow::processingThread_uiCallback(boost::function<void ()> f, std::pair<void*, mo::TypeInfo> source)
{
    emit uiCallback(f, source);
}

void
MainWindow::onLoadClicked()
{
    auto file = QFileDialog::getOpenFileName(this, "Load file", "", "*.yml");
    if(file.size() == 0)
        return;
    stopProcessingThread();
    //std::vector<EagleLib::Nodes::Node::Ptr> nodes = EagleLib::NodeManager::getInstance().loadNodes(file.toStdString());
    /*cv::FileStorage fs;
    try
    {
        fs.open(file.toStdString(), cv::FileStorage::READ);
    }
    catch (cv::Exception &e)
    {
        LOG(error) << "Failed to load file " << file.toStdString() << " " << e.what();
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
    startProcessingThread();*/
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

void MainWindow::onLoadFileClicked()
{
    QString filename = QDir::toNativeSeparators(QFileDialog::getOpenFileName(this, "Select file", QString::fromStdString(file_load_path)));
    std::string std_filename = filename.toStdString();
    boost::filesystem::path path(std_filename);
    
    if(boost::filesystem::is_directory(path))
        file_load_path = std_filename;
    else
        file_load_path = path.parent_path().string();
    
    load_file(filename);    
}
void MainWindow::onLoadDirectoryClicked()
{
    QString filename = QDir::toNativeSeparators(QFileDialog::getExistingDirectory(this, "Select directory", QString::fromStdString(dir_load_path)));
    std::string std_filename = filename.toStdString();
    boost::filesystem::path path(std_filename);

    if (boost::filesystem::is_directory(path))
        dir_load_path = std_filename;
    load_file(filename);
}
void MainWindow::load_file(QString filename, QString preferred_loader)
{
    if (EagleLib::IDataStream::CanLoadDocument(filename.toStdString()) || filename.size() == 0)
    {
        auto stream = EagleLib::IDataStream::Create(filename.toStdString(), preferred_loader.toStdString());
        if(stream)
        {
            data_streams.push_back(stream);
            stream->StartThread();
            auto data_stream_widget = new DataStreamWidget(0, stream);
            auto proxyWidget = nodeGraph->addWidget(data_stream_widget);
            nodeGraphView->addWidget(proxyWidget, stream.Get());
            current_stream = stream;
            data_streams.push_back(stream);
            data_stream_widgets.push_back(data_stream_widget);
        }      
        bookmarks->append_history(filename.toStdString());
    }
}
void MainWindow::parameter_added(EagleLib::Nodes::Node* node)
{
    for(size_t i = 0; i < widgets.size(); ++i)
    {
        widgets[i]->updateUi(true, node);
    }
}

void MainWindow::onTimeout()
{
    for(size_t i = 0; i < widgets.size(); ++i)
    {
        widgets[i]->updateUi(false);
    }
    if (mo::MetaObjectFactory::Instance()->CheckCompile())
    {
        stopProcessingThread();
        return;
    }
    if(mo::MetaObjectFactory::Instance()->IsCurrentlyCompiling())
    {
        LOG(info) << "Still compiling";
        return;
    }
    if(mo::MetaObjectFactory::Instance()->IsCompileComplete())
    {
        LOG(info) << "Recompile complete";
        mo::MetaObjectFactory::Instance()->SwapObjects();
        startProcessingThread();
    }
}
void MainWindow::on_persistence_timeout()
{
    VariableStorage::Instance()->SaveUI();
}
void MainWindow::log(QString message)
{
    ui->console->appendPlainText(message);
}


void MainWindow::addNode(EagleLib::Nodes::Node::Ptr node)
{
    // Check if this node already exists
    DOIF_LOG_PASS(nodeGraphView->getWidget(node->GetObjectId()), return, debug);
    
    QNodeWidget* nodeWidget = new QNodeWidget(0, node);
    QObject::connect(nodeWidget, SIGNAL(parameterClicked(mo::IParameter*, QPoint)), nodeGraphView, SLOT(on_parameter_clicked(mo::IParameter*, QPoint)));
    auto proxyWidget = nodeGraph->addWidget(nodeWidget);

    auto itr = positionMap.find(node->GetTreeName());
    if(itr != positionMap.end())
    {
        cv::Vec2f pt = itr->second;
        proxyWidget->setPos(pt.val[0], pt.val[1]);
    }else
    {
        int yOffset = 0;
        if (currentSelectedNodeWidget)
        {
            if(currentNode != nullptr)
            {
                auto parents = currentNode->GetParents();
                for(auto parentNode : parents)
                {
                    auto children = parentNode->GetChildren();
                    auto itr = std::find(children.begin(), children.end(), node.Get());
                    if(itr != children.end())
                    {
                        auto idx = std::distance(itr, children.begin());
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
        if (currentSelectedStreamWidget)
        {
            proxyWidget->setPos(currentSelectedStreamWidget->pos() + QPointF(500, yOffset));
        }        
        else
        {
            if(currentSelectedNodeWidget)
            {
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
    auto children = node->GetChildren();
    for(size_t i = 0; i < children.size(); ++i)
    {
        addNode(children[i]);
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
    startProcessingThread(); // need to do something to prevent calling this from adding children
}
void MainWindow::updateLines()
{

}
void MainWindow::node_update(EagleLib::Nodes::Node*)
{
    
}
void 
MainWindow::onNodeAdd(std::string node_name)
{   
    //rcc::weak_ptr<EagleLib::Nodes::Node> prevNode = currentNode;
    stopProcessingThread();
    std::vector<rcc::shared_ptr<EagleLib::Nodes::Node>> added_nodes;
    if(currentNode != nullptr)
    {
        //boost::recursive_mutex::scoped_lock lock(currentNode->_mtx);
        added_nodes = EagleLib::NodeFactory::Instance()->AddNode(node_name, currentNode.Get());
    }else
    {
        if(current_stream != nullptr)
        {
            added_nodes = EagleLib::NodeFactory::Instance()->AddNode(node_name, current_stream.Get());
        }
        else
        {
            // Node added without a parent node or data stream, node top level node in returned tree is a new parent node
        }
    }
    for (auto& node : added_nodes)
    {
        addNode(node);
    }
    dirty = true;
}
void MainWindow::onWidgetDeleted(QNodeWidget* widget)
{
    nodeGraphView->removeWidget(widget->getNode()->GetObjectId());
    auto itr = std::find(widgets.begin(), widgets.end(), widget);
    if(itr != widgets.end())
        widgets.erase(itr);
    boost::mutex::scoped_lock(parentMtx);
    auto parentItr = std::find(parentList.begin(), parentList.end(), widget->getNode().Get());
    if(parentItr != parentList.end())
        parentList.erase(parentItr);
    
}
void MainWindow::onWidgetDeleted(DataStreamWidget* widget)
{
    if(currentSelectedStreamWidget)
        if(currentSelectedStreamWidget->widget() == widget)
            currentSelectedStreamWidget = nullptr;

    auto itr = std::find(data_stream_widgets.begin(), data_stream_widgets.end(), widget);
    
    if(itr != data_stream_widgets.end())
        data_stream_widgets.erase(itr);
    auto stream = widget->GetStream();
    auto itr2 = std::find(data_streams.begin(), data_streams.end(), stream.Get());
    if(itr2 != data_streams.end())
        data_streams.erase(itr2);
    if(current_stream.Get() == stream.Get())
        current_stream.reset();
}
void
MainWindow::uiNotifier()
{
    emit uiNeedsUpdate();
}
void MainWindow::onUiUpdate()
{
    //EagleLib::UIThreadCallback::getInstance().processAllCallbacks();
    
    try
    {
        //Signals::thread_specific_queue::run_once();
        mo::ThreadSpecificQueue::RunOnce();
    }catch(...)
    {
    
    }
    
}

void
MainWindow::onSelectionChanged(QGraphicsProxyWidget* widget)
{
    if(widget == nullptr)
    {
        if(currentSelectedNodeWidget)
            currentSelectedNodeWidget->setZValue(0);
        currentSelectedNodeWidget = nullptr;
        currentSelectedStreamWidget = nullptr;
        currentNode = EagleLib::Nodes::Node::Ptr();
        return;
    }
    if(currentSelectedNodeWidget)
        if(auto oldWidget = dynamic_cast<QNodeWidget*>(currentSelectedNodeWidget->widget()))
            oldWidget->setSelected(false);
    if(currentSelectedStreamWidget)
        if(auto oldWidget = dynamic_cast<DataStreamWidget*>(currentSelectedStreamWidget->widget()))
            oldWidget->SetSelected(false);
    if(auto ptr = dynamic_cast<QNodeWidget*>(widget->widget()))
    {
        currentSelectedNodeWidget = widget;
        currentSelectedStreamWidget = nullptr;
        widget->setZValue(1);
        currentNode = ptr->getNode();
        current_stream.reset();
        ptr->setSelected(true);
    }
    
    if(auto ptr = dynamic_cast<DataStreamWidget*>(widget->widget()))
    {
        currentSelectedStreamWidget = widget;
        currentSelectedNodeWidget = nullptr;
        widget->setZValue(1);
        current_stream = ptr->GetStream();
        currentNode = rcc::shared_ptr<EagleLib::Nodes::Node>();
        ptr->SetSelected(true);
    }
}

void MainWindow::processThread()
{
    /*
    //auto handle = GetDC(0);
    //wglCreateContext(handle);
    LOG(info) << "Processing thread started" << std::endl;
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::ptime end = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta;
    
    
    while (!boost::this_thread::interruption_requested())
    {
        try
        {
            //EagleLib::ProcessingThreadCallback::Run();
            Parameters::UI::ProcessingThreadCallbackService::run();
            if (dirty)
            {
                dirty = false;
                process(&parentList, &parentMtx);
            }            
            end = boost::posix_time::microsec_clock::universal_time();
            delta = end - start;
            start = end;
            rmt_ScopedCPUSample(Idle);
            EagleLib::scoped_buffer::GarbageCollector::Run();
            if (delta.total_milliseconds() < 15 || parentList.size() == 0)
                boost::this_thread::sleep_for(boost::chrono::milliseconds(15 - delta.total_milliseconds()));
        }catch(boost::thread_interrupted& err)
        {
            (void)err;
            LOG(info) << "Processing thread interrupted";
            break;
        }
        
    }
    LOG(info) << "Interrupt requested, processing thread ended";*/
}
void MainWindow::process_log_message(boost::log::trivial::severity_level severity, std::string message)
{
    emit eLog(QString::fromStdString(message));
    //ui->console->appendPlainText();
}
void MainWindow::startProcessingThread()
{
    stopProcessingThread();
    sig_StartThreads();
}
// So the problem here is that cv::imshow operates on the main thread, thus if the main thread blocks
// because it's waiting for processingThread to join, then cv::imshow will block, thus causing deadlock.
// What we need is a signal beforehand that will disable all imshow's before a delete.
void MainWindow::stopProcessingThread()
{
    sig_StopThreads();
    processingThreadActive = false;
}

void MainWindow::on_actionLog_settings_triggered()
{
    settingsDialog->show();   
}

void MainWindow::on_actionOpen_Network_triggered()
{
    dialog_network_stream_selection dlg;
    dlg.exec();
    if(dlg.accepted)
    {
        load_file(dlg.url, dlg.preferred_loader);
    }
}

void MainWindow::on_actionBookmarks_triggered()
{
    bookmarks->show();
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
void MainWindow::on_nodeUpdate(EagleLib::Nodes::Node* node)
{
    dirty = true;
}
MO_REGISTER_CLASS(MainWindow)