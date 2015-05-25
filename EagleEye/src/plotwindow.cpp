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
    plot->installEventFilter(this);
    plot->setAcceptDrops(true);
}

PlotWindow::~PlotWindow()
{
    delete ui;
}
void PlotWindow::addPlotter(shared_ptr<EagleLib::QtPlotter> plotter)
{
    plots.push_back(plotter);
    plotter->addPlot(plot);
}

bool PlotWindow::eventFilter(QObject *obj, QEvent *ev)
{
    if(obj == plot)
    {
        if(ev->type() == QEvent::DragEnter)
        {
            QDragEnterEvent* dev = dynamic_cast<QDragEnterEvent*>(ev);
            dev->accept();
        }
        if(ev->type() == QEvent::Drop)
        {
            emit onDrop();
        }
    }
}
