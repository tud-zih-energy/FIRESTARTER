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

#include "firestarter.h"

#include "cpu.h"
#include "work.h"

base_function_t select_base_function()
{
    int result = FUNC_UNKNOWN;

    cpu_info_t* cpuinfo = NULL;
    cpuinfo = (cpu_info_t*)_mm_malloc(sizeof(cpu_info_t), 64);
    init_cpuinfo(cpuinfo, 0);

    if ((strcmp("GenuineIntel", cpuinfo->vendor) == 0) ||
        (strcmp("AuthenticAMD", cpuinfo->vendor) == 0))
    {
        switch (cpuinfo->family)
        {
$$ select function according to cpu family and model
$TEMPLATE lib_main_c.evaluate_environment_set_function_cases(dest, architectures, families)
        }
    }

    _mm_free(cpuinfo);

    return result;
}

size_t get_memory_size(base_function_t base_function, int num_threads)
{
    int FUNCTION = base_function | num_threads;

    switch(FUNCTION)
    {
$TEMPLATE lib_main_c.evaluate_environment_set_buffersize(dest, architectures)
    default:
        return 0;
    }
}

init_function_t get_init_function(base_function_t base_function, int num_threads)
{
    int FUNCTION = base_function | num_threads;

    switch (FUNCTION)
    {
$TEMPLATE lib_main_c.switch_init(dest, architectures)
    default:
        return NULL;
    }
}

kernel_function_t get_working_function(base_function_t base_function, int num_threads)
{
    int FUNCTION = base_function | num_threads;

    switch (FUNCTION)
    {
$TEMPLATE lib_main_c.switch_asm_work(dest, architectures)
    default:
        return NULL;
    }
}
