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

#include "firestarter_global.h"

typedef int (*init_function_t)(threaddata_t* threaddata);
typedef int (*kernel_function_t)(threaddata_t* threaddata);
typedef int base_function_t;

base_function_t select_base_function();
kernel_function_t get_working_function(base_function_t base_function, int num_threads);
size_t get_memory_size(base_function_t base_function);
init_function_t get_init_function(base_function_t base_function, int num_threads);
