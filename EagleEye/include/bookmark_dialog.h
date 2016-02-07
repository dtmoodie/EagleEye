#ifndef BOOKMARK_DIALOG_H
#define BOOKMARK_DIALOG_H

#include <QDialog>
#include "user_interface_persistence.h"
#include <set>
namespace Ui {
class bookmark_dialog;
}

class bookmark_dialog : public QDialog, public user_interface_persistence
{
    Q_OBJECT

public:
    explicit bookmark_dialog(QWidget *parent = 0);
    ~bookmark_dialog();
    void update();
public slots:
    void append_history(std::string dir);


private:
    Ui::bookmark_dialog *ui;
    std::set<std::string> history;
    std::set<std::string> bookmarks;
};

#endif // BOOKMARK_DIALOG_H
