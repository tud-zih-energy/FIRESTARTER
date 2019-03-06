/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2019 TU Dresden, Center for Information Services and High
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

#ifndef __FIRESTARTER_GPU_H
#define __FIRESTARTER_GPU_H


typedef struct{
      int msize;        //Matrixsize to calculate on the GPU. Different msizes create different workloads...
      int use_double;   //If we want to use doubleprecision or not
      int use_device;   //number of devices to use
      int verbose;      //verbosity
      int loadingdone;  //variable to use if the initialization of GPUs are done
      int init_count;   // Counts devices that already initialized and probably started the workload 
      unsigned long long *loadvar; //provides termination information
      } gpustruct_t;

void * init_gpu(void * gpu);
#endif

