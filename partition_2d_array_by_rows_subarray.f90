!   This program partitions a 2d array by rows among a given number of processes

!   Objective:
!   say, matrix A is (nrows x ncols)
!   assume nrows is divisible by nprocs , nrows_local = nrows/nprocs
!   then we want to distribute a as (nrows_local x ncols) sized arrays among all the processes
!   we would want to reassemble the original array (or similar) from such partitioned local arrays

!   Compiling and running:
!   mpif90 partition_2d_array_by_rows_subarray.f90
!   mpirun -np 6 ./a.out

    program partition_2d_array_by_rows
    use mpi
    implicit none

    integer :: i, j, k, l
    integer :: nrows, ncols, nrows_local
    integer,allocatable,dimension(:,:) :: A, A_local, B

    integer :: ierr, rank, nprocs
    integer(kind = mpi_address_kind) :: start, extent
    integer, dimension(2) :: starts, sizes, subsizes
    integer :: blocktype, resizedtype, intsize
    
    call mpi_init(ierr)
    call mpi_comm_rank(mpi_comm_world,rank,ierr)
    call mpi_comm_size(mpi_comm_world,nprocs,ierr)

    if(rank .eq. 0) write(*,'(a i2)') 'Number of processes = ',nprocs
    
    if(nprocs .ne. 6) then
      if(rank .eq. 0) then
        write(*,'(a i2)') 'Number of processes must be 6. Rerun the program with 6 MPI processes.'
        write(*,'(a i2)') 'Or change the number of rows to be an integer multiple of nprocs.'
      end if
      call mpi_finalize(ierr)
      stop
    end if

    nrows = 12  ! This is the row size of global array
    ncols = 5   ! This is the column size of global array
    nrows_local = nrows/nprocs   ! This is the row size of local (partitioned) array
    if(rank .eq. 0) write(*,'(a i2)') 'Number of rows to be given to each process = ',nrows_local

    allocate(A(nrows,ncols),A_local(nrows_local,ncols),B(nrows,ncols))

    ! In process 0, matrix A is created and printed to screen
    if(rank .eq. 0) then
    write(*,'(a)') "Matrix A in rank 0:"
      k = 1
      do i = 1,nrows
        do j = 1,ncols
          A(i,j) = k
          k = k+1
          write(*,'(i7)',advance="no") A(i,j)
        end do
        write(*,*)
      end do
      write(*,*)
    end if


    ! describe what these subblocks look like inside the full concatenated array
    sizes    = [ nrows, ncols ]
    subsizes = [ nrows_local, ncols ]
    starts   = [ 0, 0 ]
    
    ! A subarray type is created for MPI to handle nrows_local x ncols size arrays
    call MPI_Type_create_subarray( 2, sizes, subsizes, starts,     &
                                   MPI_ORDER_FORTRAN, MPI_INTEGER, &
                                   blocktype, ierr)

    ! This subarray type should be resized to be usable in collective operations
    ! The original extent of the type blocktype = nrows x ncols x MPI_INTEGER
    ! The modified extent of new type = nrows_local x MPI_INTEGER

    call MPI_Type_size(MPI_INTEGER, intsize, ierr)
    extent = intsize*nrows_local

    start = 0
    call MPI_Type_create_resized(blocktype, start, extent, resizedtype, ierr)
    call MPI_Type_commit(resizedtype, ierr)


    ! Partition and scatter the A matrix as nrows_local x ncols pieces
    ! Each process receives and stores their part as A_local
    call mpi_scatter(A,1,resizedtype,A_local,nrows_local*ncols,MPI_INTEGER,0,mpi_comm_world,ierr)

    ! Print to screen as a check on the partitioning
    call mpi_barrier(mpi_comm_world,ierr)
    if(rank .eq. 0) write(*,'(a)') "Partitions of matrix A :"
    do k = 0,nprocs-1
      if(rank .eq. k) then
        write(*,'(a i1 a)') "rank ", rank ," matrix : "
        do i = 1,nrows_local
          do j = 1,ncols
            write(*,'(i7)',advance="no") A_local(i,j)
          end do
          write(*,*)
        end do
        write(*,*)
      end if
      call mpi_barrier(mpi_comm_world,ierr)
    end do


    call MPI_Gather( A_local, nrows_local*ncols, MPI_INTEGER, &  ! everyone send nrows_local*ncols reals
                     A, 1, resizedtype,                   &  ! root gets 1 resized type from everyone
                     0, MPI_COMM_WORLD, ierr)

!     Print to screen as a check on the reassembling
    if(rank .eq. 0) then
      write(*,'(a)') "Reassembled matrix : "
      do i = 1,nrows
        do j = 1,ncols
          write(*,'(i7)',advance="no") A(i,j)
        end do
        write(*,*)
      end do
    end if
!    call mpi_barrier(mpi_comm_world,ierr)
!    call mpi_abort(mpi_comm_world,1,ierr)
    call mpi_finalize(ierr)
    end program partition_2d_array_by_rows
