/*
 * Copyright (c) 2011-2017 by Lawrence Livermore National Security, LLC.
 * LLNL-CODE-645430
 *
 * Produced at Lawrence Livermore National Laboratory.
 * Written by  Marty McFadden, mcfadden8@llnl.gov
 *             Kathleen Shoga, shoga1@llnl.gov
 *             Barry Rountree rountree@llnl.gov
 *
 * All rights reserved.
 *
 * This file is part of msr-safe.
 *
 * msr-safe is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * msr-safe is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with msr-safe. If not, see <http://www.gnu.org/licenses/>.
 *
 * This material is based upon work supported by the U.S. Department of
 * Energy's Lawrence Livermore National Laboratory. Office of Science, under
 * Award number DE-AC52-07NA27344.
 */

/*
 * This file defines the ioctl interface for submitting a batch of MSR
 * requests.  This file should be distributed to a place where kernel
 * module interface header files are kept for a distribution running this
 * module so that user-space applications and libraries that wish to use
 * the batch interface may #include this file to get the interface.
 */

#ifndef MSR_SAFE_HEADER_INCLUDE
#define MSR_SAFE_HEADER_INCLUDE

#include <linux/ioctl.h>
#include <linux/types.h>

struct msr_batch_op
{
    __u16 cpu;     // In: CPU to execute {rd/wr}msr instruction
    __u16 isrdmsr; // In: 0=wrmsr, non-zero=rdmsr
    __s32 err;     // Out: set if error occurred with this operation
    __u32 msr;     // In: MSR Address to perform operation
    __u64 msrdata; // In/Out: Input/Result to/from operation
    __u64 wmask;   // Out: Write mask applied to wrmsr
};

/*
struct msr_batch_op foo[16];
// Populate that array of structs
struct msr_batch_array bar;
bar.numops=16;
bar.ops = foo;
*/


struct msr_batch_array
{
    __u32 numops;             // In: # of operations in operations array
    struct msr_batch_op *ops; // In: Array[numops] of operations
};

#define X86_IOC_MSR_BATCH   _IOWR('c', 0xA2, struct msr_batch_array)

#endif
