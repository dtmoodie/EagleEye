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
	nodeGraph->addText("Test text");
	nodeGraphView = new NodeView(nodeGraph);
	connect(nodeGraphView, SIGNAL(selectionChanged(QGraphicsProxyWidget*)), this, SLOT(onSelectionChanged(QGraphicsProxyWidget*)));
	nodeGraphView->setInteractive(true);
	nodeGraphView->setViewport(new QGLWidget());
	nodeGraphView->setDragMode(QGraphicsView::ScrollHandDrag);
	ui->gridLayout->addWidget(nodeGraphView, 1, 0);
	currentSelectedNodeWidget = nullptr;
	processingThread = boost::thread(boost::bind(&MainWindow::process, this));
	quit = false;
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
	proxyWidget->setFlag(QGraphicsItem::ItemIsMovable);
	proxyWidget->setFlag(QGraphicsItem::ItemIsSelectable);
	proxyWidget->setFlag(QGraphicsItem::ItemIsFocusable);

	nodeGraphView->addWidget(proxyWidget, node->GetObjectId());
	
	if (node->parent.size())
	{
		auto parent = EagleLib::NodeManager::getInstance().getNode(node->parent);
		auto parentWidget = nodeGraphView->getWidget(parent->GetObjectId());	
	}
	if (currentSelectedNodeWidget)
	{
		proxyWidget->setPos(currentSelectedNodeWidget->pos() + QPointF(0, 250));
		auto widget = currentSelectedNodeWidget->widget();
		auto qnodeWidget = dynamic_cast<QNodeWidget*>(widget);
		if (qnodeWidget)
		{
			auto parentNode = qnodeWidget->getNode();
			if (parentNode)
			{
				parentNode->addChild(node);
			}
		}
	}
	else
	{
		parentList.push_back(node->GetObjectId());
	}
	currentSelectedNodeWidget = proxyWidget;
}

void
MainWindow::onSelectionChanged(QGraphicsProxyWidget* widget)
{
	currentSelectedNodeWidget = widget;
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
	while (!quit)
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
        EagleLib::NodeManager::getInstance().CheckRecompile();
	}
    std::cout << "Processing thread ending" << std::endl;
}
