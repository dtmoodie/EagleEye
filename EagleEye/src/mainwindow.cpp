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
	currentSelectedNode = nullptr;
}

MainWindow::~MainWindow()
{
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
	if (currentNodeId.IsValid)
	{
		auto parent = EagleLib::NodeManager::getInstance().getNode(currentNodeId);
		parent->addChild(node);
	}


	// Add a new node widget to the graph
	QNodeWidget* nodeWidget = new QNodeWidget(this, node);
	
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
		proxyWidget->setPos(currentSelectedNodeWidget->pos() + QPointF(0, 250));
	currentSelectedNodeWidget = proxyWidget;
}

void
MainWindow::onSelectionChanged(QGraphicsProxyWidget* widget)
{
	currentSelectedNodeWidget = widget;
}