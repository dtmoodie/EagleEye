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
	processingThread = boost::thread(boost::bind(&MainWindow::process, this));
	quit = false;
    //resizer = new WidgetResizer(nodeGraph);
}

MainWindow::~MainWindow()
{
	quit = true;
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
	EagleLib::NodeManager::getInstance().CheckRecompile();
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
	
    if (node->parentName.size())
	{
        auto parent = EagleLib::NodeManager::getInstance().getNode(node->parentName);
		auto parentWidget = nodeGraphView->getWidget(parent->GetObjectId());	
	}
	if (currentSelectedNodeWidget)
	{
        proxyWidget->setPos(currentSelectedNodeWidget->pos() + QPointF(0, 250));
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

void MainWindow::process()
{
	std::vector<cv::cuda::GpuMat> images;
    while (quit == false)
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
        if(EagleLib::NodeManager::getInstance().CheckRecompile())
        {

        }
	}
    std::cout << "Processing thread ending" << std::endl;
}

