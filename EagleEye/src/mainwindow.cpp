#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "qpluginloader.h"
#include "EagleLib.h"
#include <qfiledialog.h>
#include <nodes/Node.h>
#include <nodes/Display/ImageDisplay.h>
#include <QNodeWidget.h>

#include <nodes/ImgProc/FeatureDetection.h>
#include <nodes/SerialStack.h>
#include <nodes/VideoProc/OpticalFlow.h>
#include <nodes/IO/VideoLoader.h>
#include <opencv2/calib3d.hpp>
#include <qgraphicsproxywidget.h>
#include "QGLWidget"
#include <QGraphicsSceneMouseEvent>
#include <Manager.h>

int static_errorHandler( int status, const char* func_name,const char* err_msg, const char* file_name, int line, void* userdata )
{
	return 0;
}
static void getParentNodes(std::vector<ObjectId>* parentList, boost::mutex *mtx, std::vector<EagleLib::Node *> &nodes);
static void processThread(std::vector<ObjectId>* parentList, boost::recursive_mutex *mtx);
static void process(std::vector<ObjectId>* parentList, boost::recursive_mutex *mtx);
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    qRegisterMetaType<std::string>("std::string");
    qRegisterMetaType<cv::cuda::GpuMat>("cv::cuda::GpuMat");
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<EagleLib::Verbosity>("EagleLib::Verbosity");
    ui->setupUi(this);
    fileMonitorTimer = new QTimer(this);
    fileMonitorTimer->start(1000);
    connect(fileMonitorTimer, SIGNAL(timeout()), this, SLOT(onTimeout()));
    nodeListDialog = new NodeListDialog(this);
    nodeListDialog->hide();
	connect(nodeListDialog, SIGNAL(nodeConstructed(EagleLib::Node*)), 
		this, SLOT(onNodeAdd(EagleLib::Node*)));
	
	nodeGraph = new QGraphicsScene(this);
    connect(nodeGraph, SIGNAL(selectionChanged()), this, SLOT(on_selectionChanged()));
	nodeGraphView = new NodeView(nodeGraph);
	connect(nodeGraphView, SIGNAL(selectionChanged(QGraphicsProxyWidget*)), this, SLOT(onSelectionChanged(QGraphicsProxyWidget*)));
	nodeGraphView->setInteractive(true);
    nodeGraphView->setViewport(new QGLWidget());
	nodeGraphView->setDragMode(QGraphicsView::ScrollHandDrag);
	ui->gridLayout->addWidget(nodeGraphView, 1, 0);
	currentSelectedNodeWidget = nullptr;
    startProcessingThread();
	quit = false;
    cv::redirectError(&static_errorHandler);
    connect(this, SIGNAL(eLog(QString)), this, SLOT(log(QString)), Qt::QueuedConnection);
    connect(this, SIGNAL(oglDisplayImage(std::string,cv::cuda::GpuMat)), this, SLOT(onOGLDisplay(std::string,cv::cuda::GpuMat)), Qt::QueuedConnection);
    connect(this, SIGNAL(qtDisplayImage(std::string,cv::Mat)), this, SLOT(onQtDisplay(std::string,cv::Mat)), Qt::QueuedConnection);
    connect(nodeGraphView, SIGNAL(startThread()), this, SLOT(startProcessingThread()));
    connect(nodeGraphView, SIGNAL(stopThread()), this, SLOT(stopProcessingThread()));
    connect(nodeGraphView, SIGNAL(widgetDeleted(QNodeWidget*)), this, SLOT(onWidgetDeleted(QNodeWidget*)));
    boost::function<void(const std::string&, int)> f = boost::bind(&MainWindow::onCompileLog, this, _1, _2);
    EagleLib::NodeManager::getInstance().setCompileCallback(f);
}

MainWindow::~MainWindow()
{
	quit = true;
    processingThread.interrupt();
    processingThread.join();
    delete ui;
}
void MainWindow::onCompileLog(const std::string& msg, int level)
{
    ui->console->appendPlainText(QString::fromStdString(msg));
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
void
MainWindow::onTimeout()
{
    static bool swapRequired = false;
    static bool joined = false;
    for(int i = 0; i < widgets.size(); ++i)
    {
        widgets[i]->updateUi();
    }

    if(swapRequired)
    {
        if(!processingThread.try_join_for(boost::chrono::milliseconds(200)) && !joined)
        {
            log("Processing thread not joined, cannot perform object swap");
            return;
        }else
        {
            log("Processing thread joined");
            joined = true;
        }
        if(EagleLib::NodeManager::getInstance().CheckRecompile(true))
        {
           // Still compiling
            log("Still compiling");
        }else
        {
            log("Recompile complete");
            quit = false;
            processingThread = boost::thread(boost::bind(&processThread, &parentList, &parentMtx));
            swapRequired = false;
        }
        return;
    }
    if(EagleLib::NodeManager::getInstance().CheckRecompile(false))
    {
        log("Recompiling.....");
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

void 
MainWindow::onNodeAdd(EagleLib::Node* node)
{	
	if (currentNodeId.IsValid())
	{
		auto parent = EagleLib::NodeManager::getInstance().getNode(currentNodeId);
        if(parent)
            parent->addChild(node);
	}
    if(node->nodeName == "OGLImageDisplay")
    {
        node->gpuDisplayCallback = boost::bind(&MainWindow::oglDisplay, this, _1, _2);
    }
    if(node->nodeName == "QtImageDisplay")
    {
        node->cpuDisplayCallback = boost::bind(&MainWindow::qtDisplay, this, _1, _2);
    }

	// Add a new node widget to the graph
    boost::recursive_mutex::scoped_lock lock(parentMtx);
	QNodeWidget* nodeWidget = new QNodeWidget(0, node);
    auto proxyWidget = nodeGraph->addWidget(nodeWidget);

    nodeGraphView->addWidget(proxyWidget, node->GetObjectId());
    nodeGraphView->setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);

	if (currentSelectedNodeWidget)
	{
        proxyWidget->setPos(currentSelectedNodeWidget->pos() + QPointF(0, 50));
	}
	else
	{
        boost::recursive_mutex::scoped_try_lock lock(parentMtx);
		parentList.push_back(node->GetObjectId());
	}
    if(!currentSelectedNodeWidget)
    {
        nodeWidget->setSelected(true);
        currentSelectedNodeWidget = proxyWidget;
        currentNodeId = node->GetObjectId();
    }

    for(int i = 0; i < widgets.size(); ++i)
    {
        widgets[i]->updateUi();
    }
    widgets.push_back(nodeWidget);
}
void MainWindow::onWidgetDeleted(QNodeWidget* widget)
{
    auto itr = std::find(widgets.begin(), widgets.end(), widget);
    if(itr != widgets.end())
        widgets.erase(itr);
    boost::recursive_mutex::scoped_lock(parentMtx);
    auto parentItr = std::find(parentList.begin(), parentList.end(), widget->getNode()->GetObjectId());
    if(parentItr != parentList.end())
        parentList.erase(parentItr);
}

void
MainWindow::onSelectionChanged(QGraphicsProxyWidget* widget)
{
    if(widget == nullptr)
    {
        currentSelectedNodeWidget = nullptr;
        currentNodeId.SetInvalid();
        return;
    }
    if(currentSelectedNodeWidget)
        if(auto oldWidget = dynamic_cast<QNodeWidget*>(currentSelectedNodeWidget->widget()))
            oldWidget->setSelected(false);
    currentSelectedNodeWidget = widget;
    if(auto ptr = dynamic_cast<QNodeWidget*>(widget->widget()))
    {
        currentNodeId = ptr->getNode()->GetObjectId();
        ptr->setSelected(true);
    }
}

QList<EagleLib::Node*> MainWindow::getParentNodes()
{
	QList<EagleLib::Node*> nodes;
    for(int i = 0; i < parentList.size(); ++i)
	{
        auto node = EagleLib::NodeManager::getInstance().getNode(parentList[i]);
		if (node)
			nodes.push_back(node);
	}
	return nodes;
}
void getParentNodes(std::vector<ObjectId>* parentList, std::vector<EagleLib::Node*>& nodes, boost::recursive_mutex* mtx)
{
    nodes.clear();
    boost::recursive_mutex::scoped_lock lock(*mtx);
    nodes.reserve(parentList->size());
    for(int i = 0; i < parentList->size(); ++i)
    {
        auto node = EagleLib::NodeManager::getInstance().getNode((*parentList)[i]);
        if (node)
        {
            nodes.push_back(node);
        }
    }
    return;
}

void process(std::vector<ObjectId>* parentList, boost::recursive_mutex* mtx)
{

    std::vector<EagleLib::Node*> nodes;
    getParentNodes(parentList, nodes, mtx);
    static std::vector<cv::cuda::GpuMat> images;
    static std::vector<cv::cuda::Stream> streams;
    streams.resize(nodes.size());
    images.resize(nodes.size());
    int count = 0;
    for (auto it = nodes.begin(); it != nodes.end(); ++it, ++count)
    {
        (*it)->process(images[count], streams[count]);
    }
    if(nodes.size() == 0)
    {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(30));
    }
}

void processThread(std::vector<ObjectId>* parentList, boost::recursive_mutex *mtx)
{
    std::cout << "Processing thread started" << std::endl;
    while (!boost::this_thread::interruption_requested())
    {
        process(parentList, mtx);
    }
    std::cout << "Interrupt requested, processing thread ended" << std::endl;
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
