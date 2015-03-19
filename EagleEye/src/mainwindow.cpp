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

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //EagleLib::NodeManager::getInstance().addNode("TestNode");
    fileMonitorTimer = new QTimer(this);
    fileMonitorTimer->start(1000);
    connect(fileMonitorTimer, SIGNAL(timeout()), this, SLOT(onTimeout()));
    nodeListDialog = new NodeListDialog(this);
    nodeListDialog->hide();
	connect(nodeListDialog, SIGNAL(nodeConstructed(EagleLib::Node*)), 
		this, SLOT(onNodeAdd(EagleLib::Node*)));
	nodeGraph = new QGraphicsScene(this);
	nodeGraph->addText("Test text");
	nodeGraphView = new QGraphicsView(nodeGraph);
	nodeGraphView->setInteractive(true);
	nodeGraphView->setViewport(new QGLWidget());
	nodeGraphView->setDragMode(QGraphicsView::ScrollHandDrag);
	ui->gridLayout->addWidget(nodeGraphView, 1, 0);
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
	// Add a new node widget to the graph
	QNodeWidget* nodeWidget = new QNodeWidget(0, node);
	auto proxyWidget = nodeGraph->addWidget(nodeWidget);
	
}
