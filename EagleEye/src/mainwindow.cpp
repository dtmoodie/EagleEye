#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "EagleLib.h"
#include <qfiledialog.h>
#include <nodes/Display/ImageDisplay.h>
#include <nodes/Node.h>
#include <nodes/VideoProc/FeatureDetection.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    baseNode.errorCallback = boost::bind(&MainWindow::onError, this, _1);
    baseNode.addChild(boost::shared_ptr<EagleLib::GoodFeaturesToTrackDetector>(new EagleLib::GoodFeaturesToTrackDetector()));
    int idx = baseNode.addChild(boost::shared_ptr<EagleLib::Node>(new EagleLib::ImageDisplay()));
    baseNode.children[idx]->drawResults = true;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,"Select image file");
    if(fileName.size() == 0)
        return;
    cv::Mat mat = cv::imread(fileName.toStdString());
    cv::cuda::GpuMat d_mat(mat);

    baseNode.process(d_mat);

}
void
MainWindow::onError(std::string &error)
{
    return;
}
