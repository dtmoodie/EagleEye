#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "EagleLib.h"
#include <qfiledialog.h>
#include <nodes/Display/ImageDisplay.h>
#include <nodes/Node.h>
#include <nodes/ImgProc/FeatureDetection.h>
#include <nodes/SerialStack.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    rootNode.errorCallback = boost::bind(&MainWindow::onError, this, _1);
    rootNode.statusCallback = boost::bind(&MainWindow::onStatus, this, _1);
    int idx = rootNode.addChild(boost::shared_ptr<EagleLib::SerialStack>(new EagleLib::SerialStack()));
    rootNode.children[idx]->addChild(boost::shared_ptr<EagleLib::GoodFeaturesToTrackDetector>(new EagleLib::GoodFeaturesToTrackDetector(true)));
    rootNode.children[idx]->addChild(boost::shared_ptr<EagleLib::Node>(new EagleLib::ImageDisplay()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,"Select image file", "/home/dan/Pictures");
    if(fileName.size() == 0)
        return;
    cv::Mat mat = cv::imread(fileName.toStdString());
    if(mat.empty())
        return;
    cv::cuda::GpuMat d_mat(mat);
    if(d_mat.empty())
        return;
    if(d_mat.data == 0)
        return;
    baseNode.process(d_mat);

}
void
MainWindow::onError(std::string &error)
{
    return;
}
void
MainWindow::onStatus(std::string &status)
{

}
