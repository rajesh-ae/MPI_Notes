/*
   This program reads some sample (ascii) data from a file in parallel using MPI I/O.
   This is the counterpart to the write_char_parallel.c program.

   Objective:
   We shall assume 6 processes are being used to read data.
   We want to read the string (characters) in file collectively 
   (using all MPI processes together) with the processes reading 
   only their own part of the string. We shall demonstrate this 
   using three approaches:
     1. Explicit offsets
     2. Individual file pointers
     3. Shared file pointers
   
   Compiling and running:
   mpicc read_char_parallel.c
   mpirun -np 6 ./a.out
   
   Input: (these should be in the current directory)
   file_exp_offset.dat, file_ind_ptr.dat, file_shr_ptr.dat
*/
#include <stdio.h>
#include <mpi.h>
#include <string.h>

#define MAX_STR_LEN 10

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);

  int rank,size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  if(size != 6){
    if(rank == 0) printf("\nERROR: This program should be run with 6 MPI processes.\n");
    MPI_Finalize();
    return 1;
  }

  MPI_File file_handle;
  MPI_Datatype char_array_mpi;

  char test_txt[MAX_STR_LEN];
  int i,iproc,arr_len_local,total_len,disp;

  // Set dta size to be read into each process
  switch(rank) {
    case 0:
      arr_len_local = 3;
      break;
    case 1:
      arr_len_local = 5;
      break;
    case 2:
      arr_len_local = 3;
      break;
    case 3:
      arr_len_local = 5;
      break;
    case 4:
      arr_len_local = 4;
      break;
    case 5:
      arr_len_local = 6;
      break;
    default:
      printf("wrong rank for the process\n");
      break;
  }
  
  // Everyone calculates the global length of data
  MPI_Allreduce(&arr_len_local, &total_len, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // Calculate displacement of local data in the global array
  MPI_Exscan(&arr_len_local,&disp,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if(rank == 0) disp = 0;

  // Create a MPI datatype for the global array 
  MPI_Type_create_subarray(1, &total_len, &arr_len_local, &disp, MPI_ORDER_C, MPI_CHAR, &char_array_mpi);
  MPI_Type_commit(&char_array_mpi);

  // Method 1: Using explicit offset
  MPI_File_open(MPI_COMM_WORLD, "file_exp_offset.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);
  MPI_File_read_at_all(file_handle, disp, test_txt,arr_len_local,MPI_CHAR,MPI_STATUS_IGNORE);
  MPI_File_close(&file_handle);
  
  // Terminate the string
  test_txt[arr_len_local] = '\0';
  
  // Print to screen
  if(rank == 0) printf("\nData from explicit offset method: \n");
  MPI_Barrier(MPI_COMM_WORLD); // Barriers are used only for better screen output ordering
  for(iproc=0; iproc<size; iproc++){
    if(rank == iproc){
      printf("Rank %d: %s \n", rank, test_txt);
      sprintf(test_txt, " "); // Empty the string for another reading
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  // Method 2: Using individual file pointers
  MPI_File_open(MPI_COMM_WORLD, "file_ind_ptr.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);
  MPI_File_set_view(file_handle, 0, MPI_CHAR, char_array_mpi,"native", MPI_INFO_NULL);
  MPI_File_read_all(file_handle, test_txt, arr_len_local, MPI_CHAR, MPI_STATUS_IGNORE);
  MPI_File_close(&file_handle);

  // Terminate the string
  test_txt[arr_len_local] = '\0';
  
  // Print to screen
  if(rank == 0) printf("\nData from individual file pointer method: \n");
  for(iproc=0; iproc<size; iproc++){
    if(rank == iproc){
      printf("Rank %d: %s \n", rank, test_txt);
      sprintf(test_txt, " "); // Empty the string for another reading
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  // Method 3: Using shared file pointers
  MPI_File_open(MPI_COMM_WORLD, "file_shr_ptr.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);
  MPI_File_read_ordered(file_handle, test_txt, arr_len_local, MPI_CHAR,MPI_STATUS_IGNORE);
  MPI_File_close(&file_handle);

  // Terminate the string
  test_txt[arr_len_local] = '\0';
  
  // Print to screen
  if(rank == 0) printf("\nData from shared file pointer method: \n");
  for(iproc=0; iproc<size; iproc++){
    if(rank == iproc){
      printf("Rank %d: %s \n", rank, test_txt);
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  MPI_Finalize();
  
  return 0;
}
