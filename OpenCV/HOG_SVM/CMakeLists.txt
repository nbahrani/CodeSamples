cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
ADD_LIBRARY(HOG_SVM hog_svm.cpp)
target_link_libraries( HOG_SVM NBA_Methods ${OpenCV_LIBS} )