#!/usr/bin/env bash
##
## @file run_unittests.sh
## @author Jijoong Moon <jijoong.moon@samsung.com>
## @date 03 April 2020
## @brief Run unit test for Machine learning API.
##

ret=0
pushd build

run_entry() {
  entry=$1
  ${e