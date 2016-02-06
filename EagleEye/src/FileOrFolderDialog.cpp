#include "FileOrFolderDialog.h"
#include <qpushbutton.h>

FileDialog::FileDialog(QWidget *parent)
    : QFileDialog(parent)
{
    setOption(QFileDialog::DontUseNativeDialog);
    setFileMode(QFileDialog::Directory);
    // setFileMode(QFileDialog::ExistingFiles);
    for (auto *pushButton : findChildren<QPushButton*>()) {
        if (pushButton->text() == "&Open" || pushButton->text() == "&Choose") {
            openButton = pushButton;
            break;
        }
    }
    disconnect(openButton, SIGNAL(clicked(bool)));
    connect(openButton, &QPushButton::clicked, this, &FileDialog::openClicked);
    treeView = findChild<QTreeView*>();
}

QStringList FileDialog::selected() const
{
    return selectedFilePaths;
}
void FileDialog::openClicked()
{
    selectedFilePaths.clear();
    for (const auto& modelIndex : treeView->selectionModel()->selectedIndexes()) {
        if (modelIndex.column() == 0)
            selectedFilePaths.append(directory().absolutePath() + modelIndex.data().toString());
    }
    std::string dbg = selectedFilePaths.at(0).toStdString();
    emit filesSelected(selectedFilePaths);
    hide();
}