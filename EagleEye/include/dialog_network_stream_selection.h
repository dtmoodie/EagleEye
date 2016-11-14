#ifndef DIALOG_NETWORK_STREAM_SELECTION_H
#define DIALOG_NETWORK_STREAM_SELECTION_H

#include <QDialog>
#include "user_interface_persistence.h"
#include <set>
namespace Ui {
class dialog_network_stream_selection;
}
class QListWidgetItem;
class dialog_network_stream_selection : public QDialog, public UIPersistence
{
    Q_OBJECT
public:
    typedef std::set<std::pair<std::string, std::string>> UrlHistory_t;
    explicit dialog_network_stream_selection(QWidget *parent = 0);
    ~dialog_network_stream_selection();
    QString url;
    QString preferred_loader;
    bool accepted;
    bool eventFilter(QObject *object, QEvent *event);
    std::vector<mo::IParameter*> GetParameters();
public slots:
    void accept();
    void cancel();
    void on_item_clicked(QListWidgetItem* item);
signals:
    void on_network_stream_selection(QString url);
    
private:
    void refresh_history();
    Ui::dialog_network_stream_selection *ui;
    UrlHistory_t url_history;
    mo::TypedParameterPtr<UrlHistory_t> url_history_param;
};

#endif // DIALOG_NETWORK_STREAM_SELECTION_H
