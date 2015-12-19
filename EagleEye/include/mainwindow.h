#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <nodes/Node.h>
#include <Manager.h>
#include <qtimer.h>
#include "NodeListDialog.h"
#include <qgraphicsscene.h>
#include <qgraphicsview.h>
#include "NodeView.h"
#include <qlist.h>
#include <vector>
#include <boost/thread.hpp>
#include "rccsettingsdialog.h"

#include "plotwizarddialog.h"


namespace Ui {
class MainWindow;
}
class SettingDialog;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void oglDisplay(cv::cuda::GpuMat img, EagleLib::Node *node);
    void qtDisplay(cv::Mat img, EagleLib::Node *node);
    void onCompileLog(const std::string& msg, int level);
    virtual void closeEvent(QCloseEvent *event);
    void processingThread_uiCallback(boost::function<void(void)> f, std::pair<void*, Loki::TypeInfo> source);
	void process_log_message(boost::log::trivial::severity_level severity, const std::string& message);
private slots:
    void on_pushButton_clicked();
    void onTimeout();
    void onNodeAdd(EagleLib::Node::Ptr node);
	void onSelectionChanged(QGraphicsProxyWidget* widget);
    void log(QString message);
    void onOGLDisplay(std::string name, cv::cuda::GpuMat img);
    void onQtDisplay(std::string name, cv::Mat img);
    void onQtDisplay(boost::function<cv::Mat(void)> function, EagleLib::Node* node);
    void stopProcessingThread();
    void startProcessingThread();
    void onWidgetDeleted(QNodeWidget* widget);
    void onSaveClicked();
    void onLoadClicked();
    void onLoadPluginClicked();
    void addNode(EagleLib::Node::Ptr node);
    void updateLines();
    void uiNotifier();
    void onUiUpdate();
    void on_NewParameter(EagleLib::Node* node);
    void newParameter(EagleLib::Node* node);
    void displayRCCSettings();
    void onPlotAdd(PlotWindow* plot);
    void onPlotRemove(PlotWindow* plot);
    void on_actionLog_settings_triggered();

    void on_btnClear_clicked();
    void on_uiCallback(boost::function<void()> f, std::pair<void*, Loki::TypeInfo> source);
    

    void on_btnStart_clicked();

    void on_btnStop_clicked();
    void on_nodeUpdate(EagleLib::Node* node);

signals:
    void onNewParameter(EagleLib::Node* node);
    void eLog(QString message);
    void oglDisplayImage(std::string name, cv::cuda::GpuMat img);
    void qtDisplayImage(std::string name, cv::Mat img);
    void qtDisplayImage(boost::function<cv::Mat(void)> function, EagleLib::Node* node);
    void uiCallback(boost::function<void()> f, std::pair<void*, Loki::TypeInfo> source);
    void uiNeedsUpdate();
    void pluginLoaded();


private:
    
    void onError(const std::string& error);
    void onStatus(const std::string& status);
    bool dirty;
    Ui::MainWindow *ui;
    QTimer* fileMonitorTimer;
    NodeListDialog* nodeListDialog;
	QGraphicsScene* nodeGraph;
	NodeView*	nodeGraphView;
	QGraphicsProxyWidget* currentSelectedNodeWidget;
    EagleLib::Node::Ptr currentNode;
    std::vector<EagleLib::Node::Ptr> parentList;
    boost::timed_mutex parentMtx;
    std::vector<QNodeWidget*> widgets;
    boost::thread processingThread;
    RCCSettingsDialog* rccSettings;
    std::map<std::string, cv::Vec2f> positionMap;
    PlotWizardDialog* plotWizardDialog;
	SettingDialog*  settingsDialog;
    
    void processThread();
	bool processingThreadActive;
};

#endif // MAINWINDOW_H
