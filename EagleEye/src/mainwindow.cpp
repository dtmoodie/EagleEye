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

}
static void getParentNodes(std::vector<ObjectId>* parentList, boost::mutex *mtx, QList<EagleLib::Node *> &nodes);
static void processThread(std::vector<ObjectId>* parentList, boost::mutex* mtx);

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
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
    processingThread = boost::thread(boost::bind(&processThread, &parentList, &parentMtx));
	quit = false;
    cv::redirectError(&static_errorHandler);
    connect(this, SIGNAL(eLog(QString)), this, SLOT(log(QString)), Qt::QueuedConnection);
}

MainWindow::~MainWindow()
{
	quit = true;
    processingThread.interrupt();
    processingThread.join();
    delete ui;
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
    for(int i = 0; i < widgets.size(); ++i)
    {
        widgets[i]->updateUi();
    }
    if(swapRequired)
    {
        if(!processingThread.try_join_for(boost::chrono::milliseconds(30)))
        {
            log("Processing thread not joined, cannot perform object swap");
            return;
        }
        if(EagleLib::NodeManager::getInstance().CheckRecompile(true))
        {
           // Still compiling
            log("Still compiling");
        }else
        {
            log("Recompile complete");
            quit = false;
            processingThread = boost::thread(boost::bind(&MainWindow::process, this));
            swapRequired = false;
        }
        return;
    }
    if(EagleLib::NodeManager::getInstance().CheckRecompile(false))
    {
        log("Recompiling.....");
        swapRequired = true;
        processingThread.interrupt();
        processingThread.try_join_for(boost::chrono::milliseconds(30));
        return;
    }


}
void MainWindow::log(QString message)
{
    ui->console->appendPlainText(message);
}

void 
MainWindow::onNodeAdd(EagleLib::Node* node)
{	
	if (currentNodeId.IsValid())
	{
		auto parent = EagleLib::NodeManager::getInstance().getNode(currentNodeId);
		parent->addChild(node);
	}

	// Add a new node widget to the graph
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

void
MainWindow::onSelectionChanged(QGraphicsProxyWidget* widget)
{
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
void getParentNodes(std::vector<ObjectId>* parentList, boost::mutex* mtx, QList<EagleLib::Node*>& nodes)
{
    boost::mutex::scoped_try_lock lock(*mtx);
    if(!lock)
        return;
    for(int i = 0; i < parentList->size(); ++i)
    {
        auto node = EagleLib::NodeManager::getInstance().getNode((*parentList)[i]);
        if (node)
        {
            for(auto it = nodes.begin(); it != nodes.end(); ++it)
                if(*it == node)
                    continue;
            nodes.push_back(node);
        }

    }
    return;
}

void MainWindow::process()
{
	std::vector<cv::cuda::GpuMat> images;
    emit eLog("Processing thread started");
    while (!boost::this_thread::interruption_requested())
	{
		auto nodes = getParentNodes();
		images.resize(nodes.size());
		int count = 0;
		for (auto it = nodes.begin(); it != nodes.end(); ++it, ++count)
		{
			images[count] = (*it)->process(images[count]);
		}
        if(nodes.size() == 0)
            boost::this_thread::sleep_for(boost::chrono::milliseconds(30));
	}
    emit eLog("Processing thread ending");
}

void processThread(std::vector<ObjectId>* parentList, boost::mutex *mtx)
{
    std::vector<cv::cuda::GpuMat> images;
    QList<EagleLib::Node*> nodes;
    while (!boost::this_thread::interruption_requested())
    {
        getParentNodes(parentList, mtx, nodes);
        images.resize(nodes.size());
        int count = 0;
        for (auto it = nodes.begin(); it != nodes.end(); ++it, ++count)
        {
            images[count] = (*it)->process(images[count]);
        }
        if(nodes.size() == 0)
            boost::this_thread::sleep_for(boost::chrono::milliseconds(30));
    }
}
