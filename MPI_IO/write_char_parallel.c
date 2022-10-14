/*
   This program writes some sample (ascii) data to a file in parallel using MPI I/O

   Objective:
   We shall assume 6 processes are being used to write data.
   Each of them contains a string of alphabets.

   We shall write these strings (characters) to file collectively 
   (using all MPI processes together). Also, we would like to have 
   the data in file to be in the original rank order.
   We shall demonstrate this using three approaches:
     1. Explicit offsets
     2. Individual file pointers
     3. Shared file pointers
   
   Compiling and running:
   mpicc write_char_parallel.c
   mpirun -np 6 ./a.out
   
   Output:
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

  // Prepare some sample data
  switch(rank) {
    case 0:
      sprintf(test_txt, "abc");
      break;
    case 1:
      sprintf(test_txt, "defgh");
      break;
    case 2:
      sprintf(test_txt, "ijk");
      break;
    case 3:
      sprintf(test_txt, "lmnop");
      break;
    case 4:
      sprintf(test_txt, "qrst");
      break;
    case 5:
      sprintf(test_txt, "uvwxyz");
      break;
    default:
      printf("wrong rank for the process\n");
      break;
  }
  
  arr_len_local = strlen(test_txt);
  
  // Print to screen the individual process data
  if(rank == 0) printf("\nIndividual process data: \n");
  for(iproc=0; iproc<size; iproc++){
    if(rank == iproc) printf("Rank %d: %s \n", rank, test_txt);
    MPI_Barrier(MPI_COMM_WORLD);
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
  MPI_File_delete("file_exp_offset.dat", MPI_INFO_NULL);
  MPI_File_open(MPI_COMM_WORLD, "file_exp_offset.dat", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file_handle);
  MPI_File_write_at_all(file_handle, disp, test_txt,arr_len_local,MPI_CHAR,MPI_STATUS_IGNORE);
  MPI_File_close(&file_handle);
  
  // Method 2: Using individual file pointers
  MPI_File_delete("file_ind_ptr.dat", MPI_INFO_NULL);
  MPI_File_open(MPI_COMM_WORLD, "file_ind_ptr.dat", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file_handle);
  MPI_File_set_view(file_handle, 0, MPI_CHAR, char_array_mpi,"native", MPI_INFO_NULL);
  MPI_File_write_all(file_handle, test_txt, arr_len_local, MPI_CHAR, MPI_STATUS_IGNORE);
  MPI_File_close(&file_handle);

  // Method 3: Using shared file pointers
  MPI_File_delete("file_shr_ptr.dat", MPI_INFO_NULL);
  MPI_File_open(MPI_COMM_WORLD, "file_shr_ptr.dat", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file_handle);
  MPI_File_write_ordered(file_handle, test_txt, arr_len_local, MPI_CHAR,MPI_STATUS_IGNORE);
  MPI_File_close(&file_handle);

  MPI_Finalize();
  
  return 0;
}
