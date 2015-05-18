#include "plotwindow.h"
#include "ui_plotwindow.h"

PlotWindow::PlotWindow(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PlotWindow)
{
    ui->setupUi(this);
    plot = new QCustomPlot(this);
    ui->gridLayout->addWidget(plot,0,0);
}

PlotWindow::~PlotWindow()
{
    delete ui;
}
