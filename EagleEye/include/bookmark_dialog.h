#ifndef BOOKMARK_DIALOG_H
#define BOOKMARK_DIALOG_H

#include <QDialog>
#include "user_interface_persistence.h"
#include <set>
namespace Ui {
class bookmark_dialog;
}
class QListWidgetItem;

class bookmark_dialog : public QDialog, public user_interface_persistence
{
    Q_OBJECT
public:
    explicit bookmark_dialog(QWidget *parent = 0);
    ~bookmark_dialog();
    void update();
    MO_DERIVE(bookmark_dialog, user_interface_persistence)
        PERSISTENT(std::set<std::string>, history);
        PERSISTENT(std::set<std::string>, bookmarks);
    MO_END;
public slots:
    void append_history(std::string dir);

signals:
    void open_file(QString file);

private slots:
    void on_file_selected(QListWidgetItem* item);
    
private:
    Ui::bookmark_dialog *ui;
    
};

#endif // BOOKMARK_DIALOG_H
