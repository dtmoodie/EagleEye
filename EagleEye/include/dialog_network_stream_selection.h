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
    explicit dialog_network_stream_selection(QWidget *parent = 0);
    ~dialog_network_stream_selection();
    QString url;
	QString preferred_loader;
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
    std::set<std::pair<std::string, std::string>> url_history;
};

#endif // DIALOG_NETWORK_STREAM_SELECTION_H
