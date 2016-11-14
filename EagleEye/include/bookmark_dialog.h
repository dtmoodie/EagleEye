#ifndef BOOKMARK_DIALOG_H
#define BOOKMARK_DIALOG_H

#include <QDialog>
#include "user_interface_persistence.h"
#include <set>
namespace Ui {
class bookmark_dialog;
}
class QListWidgetItem;

class bookmark_dialog : public QDialog, public UIPersistence
{
    Q_OBJECT
public:
    explicit bookmark_dialog(QWidget *parent = 0);
    ~bookmark_dialog();
    void update();
    std::vector<mo::IParameter*> GetParameters();
public slots:
    void append_history(std::string dir);

signals:
    void open_file(QString file);

private slots:
    void on_file_selected(QListWidgetItem* item);
    
private:
    Ui::bookmark_dialog *ui;
    std::set<std::string> bookmarks;
    std::set<std::string> history;
    mo::TypedParameterPtr<std::set<std::string>> history_param;
    mo::TypedParameterPtr<std::set<std::string>> bookmarks_param;
};

#endif // BOOKMARK_DIALOG_H
