DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR
cd ..
git pull origin aquila
cd Aquila
git pull origin master
cd dependencies/MetaObject
git pull origin master
cd ../pplx
git pull origin master
cd ../MetaObject/dependencies/rcc
git pull origin master
cd ../cereal
git pull origin experimental
