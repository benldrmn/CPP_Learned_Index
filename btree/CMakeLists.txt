# Copyright 2013 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 2.6)

project(cppbtree CXX)

option(build_tests "Build B-tree tests" OFF)

# CMake doesn't have a way to pure template library, 
add_library(cppbtree btree.h btree_map.h btree_set.h
            safe_btree.h safe_btree_map.h safe_btree_set.h)
set_target_properties(cppbtree PROPERTIES LINKER_LANGUAGE CXX)