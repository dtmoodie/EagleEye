#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "qpluginloader.h"
#include "EagleLib.h"
#include <qfiledialog.h>
#include <nodes/Node.h>
#include <nodes/Display/ImageDisplay.h>


#include <nodes/ImgProc/FeatureDetection.h>
#include <nodes/SerialStack.h>
#include <nodes/VideoProc/OpticalFlow.h>
#include <nodes/IO/VideoLoader.h>
#include <opencv2/calib3d.hpp>



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    /*rootNode.errorCallback  = boost::bind(&MainWindow::onError,  this, _1);
	rootNode.statusCallback = boost::bind(&MainWindow::onStatus, this, _1);
	auto ptr = rootNode.addChild(new EagleLib::SerialStack());
    ptr->addChild(new EagleLib::IO::VideoLoader("/media/dan/Data/WeddingPhotos"));

    ptr->addChild(new EagleLib::Features2D::GoodFeaturesToTrackDetector());
    ptr->addChild(new EagleLib::PyrLKOpticalFlow());*/
    manager.addNode("TestNode");
    fileMonitorTimer = new QTimer(this);
    fileMonitorTimer->start(1000);
    connect(fileMonitorTimer, SIGNAL(timeout()), this, SLOT(onTimeout()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{



    /*QDir dir(qApp->applicationDirPath());
    dir.cd("plugins");
    QStringList filters;
    filters << "*.so";
    foreach(QString fileName, dir.entryList(filters))
    {
        QPluginLoader loader(dir.absoluteFilePath(fileName));
        QObject* plugin = loader.instance();
        if(plugin)
        {
            NodePluginFactory* NodePlugin = qobject_cast<NodePluginFactory*>(plugin);
        }
    }*/
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
    manager.CheckRecompile();
}
