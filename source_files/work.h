/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2017 TU Dresden, Center for Information Services and High
 * Performance Computing
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contact: daniel.hackenberg@tu-dresden.de
 *****************************************************************************/

#ifndef __FIRESTARTER__WORK_H
#define __FIRESTARTER__WORK_H

#include "firestarter_global.h"
#include <mm_malloc.h>

/*
 * function definitions
 */
$TEMPLATE work_h.function_definitions(dest,architectures)

/*
 * function that does the measurement
 */
extern void _work(volatile mydata_t* data, unsigned long long *high);

/*
 * loop executed by all threads, except the master thread
 */
extern void *thread(void *threaddata);

/*
 * init functions
 */
$TEMPLATE work_h.init_functions(dest,architectures)

/*
 * stress test functions
 */
$TEMPLATE work_h.stress_test_functions(dest,architectures)

/*
 * low load function
 */
int low_load_function(unsigned long long addrHigh,unsigned int period) __attribute__((noinline));
int low_load_function(unsigned long long addrHigh,unsigned int period);

#endif

