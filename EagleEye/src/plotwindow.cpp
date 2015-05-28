#include "plotwindow.h"
#include "ui_plotwindow.h"
IPlotWindow::IPlotWindow(QWidget *parent):
    QWidget(parent){}



PlotWindow::PlotWindow(QWidget *parent) :
    IPlotWindow(parent),
    ui(new Ui::PlotWindow)
{
    ui->setupUi(this);
    plot = new QCustomPlot(this);
    ui->gridLayout->addWidget(plot,0,0);
    plot->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    plot->installEventFilter(this);
    plot->setAcceptDrops(true);
    plot->setInteractions((QCP::Interaction)255);
    rightClickMenu = new QMenu(this);
    QAction* resizeAction = rightClickMenu->addAction("Rescale plot");
    connect(resizeAction, SIGNAL(triggered()), this, SLOT(on_resizePlot_activated()));
}

PlotWindow::~PlotWindow()
{
    delete ui;
}
void PlotWindow::addPlotter(shared_ptr<EagleLib::QtPlotter> plotter)
{
    if(plotter->acceptsWidget(plot))
    {
        plotter->addPlot(plot);
        plots.push_back(plotter);
    }
}
void PlotWindow::on_resizePlot_activated()
{
    plot->rescaleAxes();
}

bool PlotWindow::eventFilter(QObject *obj, QEvent *ev)
{
    if(obj == plot)
    {
        if(ev->type() == QEvent::DragEnter)
        {
            QDragEnterEvent* dev = dynamic_cast<QDragEnterEvent*>(ev);
            dev->accept();
            return true;
        }
        if(ev->type() == QEvent::Drop)
        {
            emit onDrop();
            return true;
        }
        if(ev->type() == QEvent::MouseButtonPress)
        {
            QMouseEvent* mev = dynamic_cast<QMouseEvent*>(ev);
            if(mev == nullptr)
                return false;
            if(mev->button() == Qt::RightButton)
            {

                rightClickMenu->popup(mapToGlobal(mev->pos()));
            }
        }
    }
    return false;
}
MatrixViewWindow::MatrixViewWindow(QWidget *parent):
    IPlotWindow(parent)
{

}
MatrixViewWindow::~MatrixViewWindow()
{

}
void MatrixViewWindow::addPlotter(shared_ptr<EagleLib::QtPlotter> plotter)
{

}

