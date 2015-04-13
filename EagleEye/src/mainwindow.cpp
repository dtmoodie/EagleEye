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
#include <statebox.h>
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
	currentSelectedNodeWidget = proxyWidget;
    currentNodeId = node->GetObjectId();
}

void
MainWindow::onSelectionChanged(QGraphicsProxyWidget* widget)
{
	currentSelectedNodeWidget = widget;
    auto ptr = dynamic_cast<QNodeWidget*>(widget->widget());
    if(ptr)
    {
        currentNodeId = ptr->getNode()->GetObjectId();
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
        EagleLib::NodeManager::getInstance().CheckRecompile();
	}
    std::cout << "Processing thread ending" << std::endl;
}
WidgetResizer::WidgetResizer(QGraphicsScene* scene_):
    scene(scene_),
    currentWidget(nullptr)
{
    if(scene_)
    {
        corners.push_back(new CornerGrabber());
        corners.push_back(new CornerGrabber());
        corners.push_back(new CornerGrabber());
        corners.push_back(new CornerGrabber());
        corners[0]->hide();
        corners[1]->hide();
        corners[2]->hide();
        corners[3]->hide();
        scene->addItem(corners[0]);
        scene->addItem(corners[1]);
        scene->addItem(corners[2]);
        scene->addItem(corners[3]);
        scene->addItem(this);
    }
}

bool WidgetResizer::sceneEventFilter(QGraphicsItem *watched, QEvent *event)
{
    if(watched != currentWidget) return false;
    std::cout << event->type() << std::endl;
}
void WidgetResizer::setWidget(QGraphicsProxyWidget* widget)
{
    auto pos = widget->pos();
    auto size = widget->size();
    corners[0]->setPos(pos);
    corners[1]->setPos(pos.x() + size.width(), pos.y());
    corners[2]->setPos(pos.x() + size.width(), pos.y() + size.height());
    corners[3]->setPos(pos.x(), pos.y() + size.height());
    corners[0]->show();
    corners[1]->show();
    corners[2]->show();
    corners[3]->show();
    widget->installSceneEventFilter(this);
    currentWidget = widget;
}


void WidgetResizer::mouseMoveEvent ( QGraphicsSceneMouseEvent * event )
{
    event->setAccepted(false);
}

void WidgetResizer::mouseMoveEvent(QGraphicsSceneDragDropEvent *event)
{
    event->setAccepted(false);
}

void WidgetResizer::mousePressEvent (QGraphicsSceneMouseEvent * event )
{
    event->setAccepted(false);
}

void WidgetResizer::mousePressEvent(QGraphicsSceneDragDropEvent *event)
{
 event->setAccepted(false);
}

void WidgetResizer::mouseReleaseEvent (QGraphicsSceneMouseEvent * event )
{
    event->setAccepted(false);
}
QRectF WidgetResizer::boundingRect() const
{
    if(currentWidget)
    {
        auto pos = currentWidget->pos();
        auto size = currentWidget->size();
        return QRectF(pos,size);
    }
    return QRectF(0,0,0,0);
}
void WidgetResizer::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{

}
