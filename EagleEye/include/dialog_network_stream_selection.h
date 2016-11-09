#ifndef DIALOG_NETWORK_STREAM_SELECTION_H
#define DIALOG_NETWORK_STREAM_SELECTION_H

#include <QDialog>
#include "user_interface_persistence.h"
#include <set>
namespace Ui {
class dialog_network_stream_selection;
}
class QListWidgetItem;
class dialog_network_stream_selection : public QDialog, public user_interface_persistence
{
    Q_OBJECT

public:
    typedef std::set<std::pair<std::string, std::string>> UrlHistory_t;
    MO_DERIVE(dialog_network_stream_selection, user_interface_persistence)
        PERSISTENT(UrlHistory_t, url_history);
    MO_END;
    explicit dialog_network_stream_selection(QWidget *parent = 0);
    ~dialog_network_stream_selection();
    QString url;
    QString preferred_loader;
    bool accepted;
    bool eventFilter(QObject *object, QEvent *event);
public slots:
    void accept();
    void cancel();
    void on_item_clicked(QListWidgetItem* item);
signals:
    void on_network_stream_selection(QString url);
    
private:
    void refresh_history();
    Ui::dialog_network_stream_selection *ui;
    
};

#endif // DIALOG_NETWORK_STREAM_SELECTION_H
