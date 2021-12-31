!   This program partitions a 2d array by rows among a given number of processes

!   Objective:
!   say, matrix A is (nrows x ncols)
!   assume nrows is divisible by nprocs , nrows_local = nrows/nprocs
!   then we want to distribute a as (nrows_local x ncols) sized arrays among all the processes
!   we would want to reassemble the original array (or similar) from such partitioned local arrays

!   Compiling and running:
!   mpif90 partition_2d_array_by_rows_vector.f90
!   mpirun -np 6 ./a.out

    program partition_2d_array_by_rows
    use mpi
    implicit none

    integer :: i, j, k, l
    integer :: nrows, ncols, nrows_local
    integer,allocatable,dimension(:,:) :: A, A_local, B
    logical :: array_compare

    integer :: ierr, rank, nprocs
    integer(kind = mpi_address_kind) :: lb, extent
    integer :: rowtype, rowtype_resized
    integer,allocatable,dimension(:) :: counts, displs

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
    allocate(counts(nprocs),displs(nprocs))

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
    
    ! A strided vector type is created for MPI to handle nrows_local x ncols size arrays
    call mpi_type_vector(ncols, nrows_local, nrows, MPI_INTEGER, rowtype, ierr)
    call mpi_type_commit(rowtype, ierr)

    ! This vector type should be resized to be usable in collective operations
    ! The original extent of the type rowtype = nrows x ncols x MPI_INTEGER
    ! The modified extent of new type = nrows_local x MPI_INTEGER
    call mpi_type_get_extent(MPI_INTEGER, lb, extent, ierr)
    extent = extent * nrows_local

    call mpi_type_create_resized(rowtype, lb, extent, rowtype_resized, ierr)
    call mpi_type_commit(rowtype_resized, ierr)

    ! Partition and scatter the A matrix as nrows_local x ncols pieces
    ! Each process receives and stores their part as A_local
    call mpi_scatter(A,1,rowtype_resized,A_local,nrows_local*ncols,MPI_INTEGER,0,mpi_comm_world,ierr)

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
    
    ! Reassemble the parts A_local into a global array B in rank 0
!    call mpi_gather(A_local, nrows_local*ncols, MPI_INTEGER, &
!                   B, 1, rowtype_resized, 0, mpi_comm_world, ierr)
                   
    ! For everyone to gather the global matrix
    ! call mpi_allgather(A_local, nrows_local*ncols, MPI_INTEGER, &
    !                   A, 1, rowtype_resized, mpi_comm_world, ierr)

!   MPI_Gatherv works for all sizes
    do i = 1,nprocs
      counts(i) = 1     ! we will gather one of these new types from everyone
      displs(i) = i-1   ! the starting point of everyone's data
                        ! in the global array, in block extents
    end do
    call MPI_Gatherv( A_local, nrows_local*ncols, MPI_INTEGER, & ! I'm sending localsize**2 chars
                      B, counts, displs, rowtype_resized,&
                      0, MPI_COMM_WORLD, ierr)
    
!     Print to screen as a check on the reassembling and compare
    if(rank .eq. 0) then
      array_compare = .true.
      write(*,'(a)') "Reassembled matrix : "
      do i = 1,nrows
        do j = 1,ncols
          write(*,'(i7)',advance="no") B(i,j)
          if(B(i,j) .ne. A(i,j)) array_compare = .false.
        end do
        write(*,*)
      end do
      if(array_compare) then
        write(*,*) "Reassembled array comparison successful!"
      else
        write(*,*) "Reassembled array comparison failed!"
      end if
    end if

    call mpi_type_free(rowtype_resized,ierr)
    call mpi_type_free(rowtype,ierr)
    call mpi_finalize(ierr)
    end program partition_2d_array_by_rows

