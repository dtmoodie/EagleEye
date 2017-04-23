#define PARAMETERS_GENERATE_UI
#define HAVE_OPENCV
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "FlowScene.hpp"
#include "FlowView.hpp"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    _flow_scene = new QtNodes::FlowScene();
    ui->main_layout->addWidget(new QtNodes::FlowView(_flow_scene));
}

MainWindow::~MainWindow()
{
    delete ui;
}

MO_REGISTER_CLASS(MainWindow)