cmake_minimum_required(VERSION 3.15...3.31)

project(MyProject VERSION 1.0
                  DESCRIPTION "Very nice project"
                  LANGUAGES CXX)

add_executable(out hello.cpp helloWorld.cpp helloWorld.h)