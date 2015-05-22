#include "plotwindow.h"
#include "ui_plotwindow.h"

PlotWindow::PlotWindow(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PlotWindow)
{
    ui->setupUi(this);
    plot = new QCustomPlot(this);
    ui->gridLayout->addWidget(plot,0,0);
    plot->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
}

PlotWindow::~PlotWindow()
{
    delete ui;
}
void PlotWindow::addPlotter(boost::shared_ptr<ParameterPlotter> plotter)
{

}
