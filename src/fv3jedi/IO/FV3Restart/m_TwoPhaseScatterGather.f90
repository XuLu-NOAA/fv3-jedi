module TwoPhaseScatterGather
  ! Inspired by a post by Jonathan Dursi at
  ! https://stackoverflow.com/questions/29325513/scatter-matrix-blocks-of-different-sizes-using-mpi
  use mpi
  use fv3jedi_geom_mod,             only: fv3jedi_geom
  implicit none

  ! GENERIC INTERFACES
  ! ------------------
  interface TwoPhaseScatter_Phase1
    procedure TwoPhaseScatter_Phase1_r4, TwoPhaseScatter_Phase1_r8
  end interface TwoPhaseScatter_Phase1

  interface TwoPhaseScatter_Phase2
    procedure TwoPhaseScatter_Phase2_r4, TwoPhaseScatter_Phase2_r8
  end interface TwoPhaseScatter_Phase2

  interface TwoPhaseGather_Phase1
    procedure TwoPhaseGather_Phase1_r4, TwoPhaseGather_Phase1_r8
  end interface TwoPhaseGather_Phase1

  interface TwoPhaseGather_Phase2
    procedure TwoPhaseGather_Phase2_r4, TwoPhaseGather_Phase2_r8
  end interface TwoPhaseGather_Phase2

  ! CUSTOM TYPES
  ! ------------
  type :: scatter_t
    logical :: lalloc = .false.
    integer, allocatable :: sendcounts_phase1(:), senddispls_phase1(:)
    integer, allocatable :: sendcounts_phase2(:), senddispls_phase2(:)
    integer :: vec_r4, localvec_r4
    integer :: vec_r8, localvec_r8
  endtype scatter_t

  type :: gather_t
    logical :: lalloc = .false.
    integer, allocatable :: recvcounts_phase1(:), recvdispls_phase1(:)
    integer, allocatable :: recvcounts_phase2(:), recvdispls_phase2(:)
    integer :: vec_r4, localvec_r4
    integer :: vec_r8, localvec_r8
  endtype gather_t

  ! SHARED COMMUNICATION STATE
  ! --------------------------
  type(scatter_t), allocatable, target :: TwoPhaseScatter(:)
  type(gather_t), allocatable, target  :: TwoPhaseGather(:)

  ! Dedicated Scatter Workspaces
  real(kind=4), allocatable, target :: scatter_workspace_r4(:,:,:)
  real(kind=8), allocatable, target :: scatter_workspace_r8(:,:,:)

  ! Dedicated Gather Workspaces
  real(kind=4), allocatable, target :: gather_workspace_r4(:,:,:)
  real(kind=8), allocatable, target :: gather_workspace_r8(:,:,:)

  ! Shared workspace for local type conversion (Scatter Phase 2)
  real(kind=4), allocatable, target :: scatter_recv_cast_workspace_r4(:,:,:)
  real(kind=4), allocatable, target :: gather_send_cast_workspace_r4(:,:,:)

  ! Geometrical arrays
  integer :: EWindex, NSindex, colComm, rowComm, rowrank, colrank
  integer(kind=4), allocatable :: NumColsPerRank(:), NumRowsPerRank(:)
  integer(kind=4), allocatable :: MyRowGlobal(:), MyColGlobal(:)
  integer(kind=4), allocatable :: MyRankInRowComm(:), MyRankInColComm(:)
  integer, allocatable :: ibegin(:), iend(:), jbegin(:), jend(:)
  integer :: layout(2), globalsizes(2), localsizes(2)

contains

  ! SUBROUTINES
  ! -----------
  subroutine TwoPhaseScatter_init(geom, npes, batch_size)
    implicit none
    type(fv3jedi_geom), intent(in):: geom
    integer, intent(in) :: npes, batch_size
    integer :: r, ierr
    integer :: rowSize, colSize
    integer :: myRowRank, myColRank
    integer :: mpicomm, wrank, wsize

    mpicomm = geom%f_comm%communicator()
    call MPI_Comm_rank(mpicomm, wrank, ierr)
    call MPI_Comm_size(mpicomm, wsize, ierr)

    layout = geom%layout

    EWindex = modulo(wrank,layout(1))
    NSindex = (wrank/layout(1))

    call MPI_Comm_split(mpicomm, NSindex, wrank, rowComm, ierr)
    call MPI_Comm_split(mpicomm, EWindex, wrank, colComm, ierr)

    call MPI_Comm_size(rowComm, rowSize, ierr)
    call MPI_Comm_size(colComm, colSize, ierr)

    ! Allocate based on actual comm sizes
    allocate(ibegin(0:rowSize-1), iend(0:rowSize-1))
    allocate(jbegin(0:colSize-1), jend(0:colSize-1))

    call MPI_AllGather(geom%isc,1,MPI_Integer,ibegin(0:),1,MPI_Integer, rowComm, ierr)
    call MPI_AllGather(geom%iec,1,MPI_Integer,iend(0:)  ,1,MPI_Integer, rowComm, ierr)
    call MPI_AllGather(geom%jsc,1,MPI_Integer,jbegin(0:),1,MPI_Integer, colComm, ierr)
    call MPI_AllGather(geom%jec,1,MPI_Integer,jend(0:)  ,1,MPI_Integer, colComm, ierr)

    ! Let other ranks know my row and column index
    allocate(MyRowGlobal(0:wsize-1), MyColGlobal(0:wsize-1))
    MyRowGlobal=-999; MyColGlobal=-999
    call MPI_AllGather(NSindex,1,MPI_Integer,MyRowGlobal,1,MPI_Integer, mpicomm, ierr)
    call MPI_AllGather(EWindex,1,MPI_Integer,MyColGlobal,1,MPI_Integer, mpicomm, ierr)

    ! Let other ranks know my rank in the row and column communicators
    call MPI_Comm_rank(rowComm, rowrank, ierr)
    call MPI_Comm_rank(colComm, colrank, ierr)
    allocate(MyRankInRowComm(0:wsize-1), MyRankInColComm(0:wsize-1))
    MyRankInRowComm=-999; MyRankInColComm=-999
    call MPI_AllGather(rowrank,1,MPI_Integer,MyRankInRowComm,1,MPI_Integer, mpicomm, ierr)
    call MPI_AllGather(colrank,1,MPI_Integer,MyRankInColComm,1,MPI_Integer, mpicomm, ierr)

    ! dimensions of my subdomain
    call MPI_Comm_rank(rowComm, myRowRank, ierr)  ! 0..rowSize-1
    call MPI_Comm_rank(colComm, myColRank, ierr)  ! 0..colSize-1
    localsizes(1) = iend(myRowRank) - ibegin(myRowRank) + 1
    localsizes(2) = jend(myColRank) - jbegin(myColRank) + 1

    ! Let other ranks in my row and column know how many rows and columns I have in my subdomain
    allocate(NumColsPerRank(0:rowSize-1))
    allocate(NumRowsPerRank(0:colSize-1))
    NumColsPerRank=-999; NumRowsPerRank=-999
    call MPI_Allgather(localsizes(1), 1, MPI_Integer, NumColsPerRank, 1, MPI_Integer, rowComm, ierr)
    call MPI_Allgather(localsizes(2), 1, MPI_Integer, NumRowsPerRank, 1, MPI_Integer, colComm, ierr)

    ! Horizontal dimensions of ensemble input files
    globalsizes(1) = geom%npx-1
    globalsizes(2) = geom%npy-1

    ! Allocate the structure
    ! ----------------------
    if (.not. allocated(TwoPhaseScatter)) allocate(TwoPhaseScatter(0:npes-1))

    ! Initialize all MPI Datatype handles to NULL
    ! (Guarantees safe datatype creation and teardown checks)
    ! -------------------------------------------------------
    do r = 0, npes-1
      ! Scatter handles
      TwoPhaseScatter(r)%localvec_r4 = MPI_DATATYPE_NULL
      TwoPhaseScatter(r)%localvec_r8 = MPI_DATATYPE_NULL
      TwoPhaseScatter(r)%vec_r4      = MPI_DATATYPE_NULL
      TwoPhaseScatter(r)%vec_r8      = MPI_DATATYPE_NULL
      TwoPhaseScatter(r)%lalloc      = .false.
    enddo

    ! Allocate the shared workspaces
    if (.not. allocated(scatter_workspace_r4)) then
      allocate(scatter_workspace_r4(globalsizes(1), localsizes(2), batch_size))
      !scatter_workspace_r4 = 0.0_4
    endif
    if (.not. allocated(scatter_workspace_r8)) then
      allocate(scatter_workspace_r8(globalsizes(1), localsizes(2), batch_size))
      !scatter_workspace_r8 = 0.0_8
    endif
    if (.not. allocated(scatter_recv_cast_workspace_r4)) then
      allocate(scatter_recv_cast_workspace_r4(localsizes(1), localsizes(2), batch_size))
      !scatter_recv_cast_workspace_r4 = 0.0_4
    endif

  end subroutine TwoPhaseScatter_init

  subroutine TwoPhaseGather_init(geom, npes, batch_size)
    implicit none
    type(fv3jedi_geom), intent(in):: geom
    integer, intent(in) :: npes, batch_size
    integer :: r, ierr
    integer :: rowSize, colSize
    integer :: myRowRank, myColRank
    integer :: mpicomm, wrank, wsize

    mpicomm = geom%f_comm%communicator()
    call MPI_Comm_rank(mpicomm, wrank, ierr)
    call MPI_Comm_size(mpicomm, wsize, ierr)

    layout = geom%layout

    EWindex = modulo(wrank,layout(1))
    NSindex = (wrank/layout(1))

    call MPI_Comm_split(mpicomm, NSindex, wrank, rowComm, ierr)
    call MPI_Comm_split(mpicomm, EWindex, wrank, colComm, ierr)

    call MPI_Comm_size(rowComm, rowSize, ierr)
    call MPI_Comm_size(colComm, colSize, ierr)

    ! Allocate based on actual comm sizes
    allocate(ibegin(0:rowSize-1), iend(0:rowSize-1))
    allocate(jbegin(0:colSize-1), jend(0:colSize-1))

    call MPI_AllGather(geom%isc,1,MPI_Integer,ibegin(0:),1,MPI_Integer, rowComm, ierr)
    call MPI_AllGather(geom%iec,1,MPI_Integer,iend(0:)  ,1,MPI_Integer, rowComm, ierr)
    call MPI_AllGather(geom%jsc,1,MPI_Integer,jbegin(0:),1,MPI_Integer, colComm, ierr)
    call MPI_AllGather(geom%jec,1,MPI_Integer,jend(0:)  ,1,MPI_Integer, colComm, ierr)

    ! Let other ranks know my row and column index
    allocate(MyRowGlobal(0:wsize-1), MyColGlobal(0:wsize-1))
    MyRowGlobal=-999; MyColGlobal=-999
    call MPI_AllGather(NSindex,1,MPI_Integer,MyRowGlobal,1,MPI_Integer, mpicomm, ierr)
    call MPI_AllGather(EWindex,1,MPI_Integer,MyColGlobal,1,MPI_Integer, mpicomm, ierr)

    ! Let other ranks know my rank in the row and column communicators
    call MPI_Comm_rank(rowComm, rowrank, ierr)
    call MPI_Comm_rank(colComm, colrank, ierr)
    allocate(MyRankInRowComm(0:wsize-1), MyRankInColComm(0:wsize-1))
    MyRankInRowComm=-999; MyRankInColComm=-999
    call MPI_AllGather(rowrank,1,MPI_Integer,MyRankInRowComm,1,MPI_Integer, mpicomm, ierr)
    call MPI_AllGather(colrank,1,MPI_Integer,MyRankInColComm,1,MPI_Integer, mpicomm, ierr)

    ! dimensions of my subdomain
    call MPI_Comm_rank(rowComm, myRowRank, ierr)  ! 0..rowSize-1
    call MPI_Comm_rank(colComm, myColRank, ierr)  ! 0..colSize-1
    localsizes(1) = iend(myRowRank) - ibegin(myRowRank) + 1
    localsizes(2) = jend(myColRank) - jbegin(myColRank) + 1

    ! Let other ranks in my row and column know how many rows and columns I have in my subdomain
    allocate(NumColsPerRank(0:rowSize-1))
    allocate(NumRowsPerRank(0:colSize-1))
    NumColsPerRank=-999; NumRowsPerRank=-999
    call MPI_Allgather(localsizes(1), 1, MPI_Integer, NumColsPerRank, 1, MPI_Integer, rowComm, ierr)
    call MPI_Allgather(localsizes(2), 1, MPI_Integer, NumRowsPerRank, 1, MPI_Integer, colComm, ierr)

    ! Horizontal dimensions of ensemble input files
    globalsizes(1) = geom%npx-1
    globalsizes(2) = geom%npy-1

    ! Allocate the structure
    ! ----------------------
    if (.not. allocated(TwoPhaseGather))  allocate(TwoPhaseGather(0:npes-1))

    ! Initialize all MPI Datatype handles to NULL
    ! (Guarantees safe datatype creation and teardown checks)
    ! -------------------------------------------------------
    do r = 0, npes-1
      ! Gather handles
      TwoPhaseGather(r)%localvec_r4  = MPI_DATATYPE_NULL
      TwoPhaseGather(r)%localvec_r8  = MPI_DATATYPE_NULL
      TwoPhaseGather(r)%vec_r4       = MPI_DATATYPE_NULL
      TwoPhaseGather(r)%vec_r8       = MPI_DATATYPE_NULL
      TwoPhaseGather(r)%lalloc       = .false.
    enddo

    ! Allocate the shared workspaces
    if (.not. allocated(gather_workspace_r4)) then
      allocate(gather_workspace_r4(globalsizes(1), localsizes(2), batch_size))
      !gather_workspace_r4 = 0.0_4
    endif
    if (.not. allocated(gather_workspace_r8)) then
      allocate(gather_workspace_r8(globalsizes(1), localsizes(2), batch_size))
      !gather_workspace_r8 = 0.0_8
    endif
    if (.not. allocated(gather_send_cast_workspace_r4)) then
      allocate(gather_send_cast_workspace_r4(localsizes(1), localsizes(2), batch_size))
      !gather_send_cast_workspace_r4 = 0.0_4
    endif
  end subroutine TwoPhaseGather_init

  subroutine TwoPhaseScatter_delete()
    implicit none
    integer :: r, ierr

    if (allocated(TwoPhaseScatter)) then
      do r = lbound(TwoPhaseScatter, 1), ubound(TwoPhaseScatter, 1)
        if (allocated(TwoPhaseScatter(r)%sendcounts_phase1)) deallocate(TwoPhaseScatter(r)%sendcounts_phase1)
        if (allocated(TwoPhaseScatter(r)%senddispls_phase1)) deallocate(TwoPhaseScatter(r)%senddispls_phase1)
        if (allocated(TwoPhaseScatter(r)%sendcounts_phase2)) deallocate(TwoPhaseScatter(r)%sendcounts_phase2)
        if (allocated(TwoPhaseScatter(r)%senddispls_phase2)) deallocate(TwoPhaseScatter(r)%senddispls_phase2)

        ! Safely free Datatypes ONLY if they were actually created
        if (TwoPhaseScatter(r)%localvec_r4 /= MPI_DATATYPE_NULL) call MPI_Type_free(TwoPhaseScatter(r)%localvec_r4, ierr)
        if (TwoPhaseScatter(r)%localvec_r8 /= MPI_DATATYPE_NULL) call MPI_Type_free(TwoPhaseScatter(r)%localvec_r8, ierr)

        if (TwoPhaseScatter(r)%vec_r4 /= MPI_DATATYPE_NULL) call MPI_Type_free(TwoPhaseScatter(r)%vec_r4, ierr)
        if (TwoPhaseScatter(r)%vec_r8 /= MPI_DATATYPE_NULL) call MPI_Type_free(TwoPhaseScatter(r)%vec_r8, ierr)
      enddo
      deallocate(TwoPhaseScatter)
    endif
    if (allocated(scatter_workspace_r4)) deallocate(scatter_workspace_r4)
    if (allocated(scatter_workspace_r8)) deallocate(scatter_workspace_r8)
    if (allocated(scatter_recv_cast_workspace_r4)) deallocate(scatter_recv_cast_workspace_r4)

    if(allocated(ibegin)) deallocate(ibegin)
    if(allocated(iend)) deallocate(iend)
    if(allocated(jbegin)) deallocate(jbegin)
    if(allocated(jend)) deallocate(jend)
    if(allocated(MyRowGlobal)) deallocate(MyRowGlobal)
    if(allocated(MyColGlobal)) deallocate(MyColGlobal)
    if(allocated(MyRankInRowComm)) deallocate(MyRankInRowComm)
    if(allocated(MyRankInColComm)) deallocate(MyRankInColComm)
    if(allocated(NumColsPerRank)) deallocate(NumColsPerRank)
    if(allocated(NumRowsPerRank)) deallocate(NumRowsPerRank)
  end subroutine TwoPhaseScatter_delete

  subroutine TwoPhaseGather_delete()
    implicit none
    integer :: r, ierr

    if (allocated(TwoPhaseGather)) then
      do r = lbound(TwoPhaseGather, 1), ubound(TwoPhaseGather, 1)
        if (allocated(TwoPhaseGather(r)%recvcounts_phase1)) deallocate(TwoPhaseGather(r)%recvcounts_phase1)
        if (allocated(TwoPhaseGather(r)%recvdispls_phase1)) deallocate(TwoPhaseGather(r)%recvdispls_phase1)
        if (allocated(TwoPhaseGather(r)%recvcounts_phase2)) deallocate(TwoPhaseGather(r)%recvcounts_phase2)
        if (allocated(TwoPhaseGather(r)%recvdispls_phase2)) deallocate(TwoPhaseGather(r)%recvdispls_phase2)

        ! Safely free Datatypes ONLY if they were actually created
        if (TwoPhaseGather(r)%localvec_r4 /= MPI_DATATYPE_NULL) call MPI_Type_free(TwoPhaseGather(r)%localvec_r4, ierr)
        if (TwoPhaseGather(r)%localvec_r8 /= MPI_DATATYPE_NULL) call MPI_Type_free(TwoPhaseGather(r)%localvec_r8, ierr)

        if (TwoPhaseGather(r)%vec_r4 /= MPI_DATATYPE_NULL) call MPI_Type_free(TwoPhaseGather(r)%vec_r4, ierr)
        if (TwoPhaseGather(r)%vec_r8 /= MPI_DATATYPE_NULL) call MPI_Type_free(TwoPhaseGather(r)%vec_r8, ierr)
      enddo
      deallocate(TwoPhaseGather)
    endif
    if (allocated(gather_workspace_r4))  deallocate(gather_workspace_r4)
    if (allocated(gather_workspace_r8))  deallocate(gather_workspace_r8)
    if (allocated(gather_send_cast_workspace_r4)) deallocate(gather_send_cast_workspace_r4)

    if(allocated(ibegin)) deallocate(ibegin)
    if(allocated(iend)) deallocate(iend)
    if(allocated(jbegin)) deallocate(jbegin)
    if(allocated(jend)) deallocate(jend)
    if(allocated(MyRowGlobal)) deallocate(MyRowGlobal)
    if(allocated(MyColGlobal)) deallocate(MyColGlobal)
    if(allocated(MyRankInRowComm)) deallocate(MyRankInRowComm)
    if(allocated(MyRankInColComm)) deallocate(MyRankInColComm)
    if(allocated(NumColsPerRank)) deallocate(NumColsPerRank)
    if(allocated(NumRowsPerRank)) deallocate(NumRowsPerRank)
  end subroutine TwoPhaseGather_delete

  subroutine TwoPhaseScatter_Phase1_r4(owner, rank, sendbuf, b_ind, req_p1)
    implicit none
    integer, intent(in) :: owner, rank
    real(kind=4), contiguous, target, asynchronous, intent(inout) :: sendbuf(:,:)
    integer, intent(in)  :: b_ind
    integer, intent(out) :: req_p1

    integer :: row, col, ierr, temptype
    integer(kind=MPI_ADDRESS_KIND) :: lb, extent
    integer :: myrow, mycol

    myrow = NSindex
    mycol = EWindex

    if (.not. TwoPhaseScatter(owner)%lalloc) then
      if (mycol == MyColGlobal(owner)) then
        allocate(TwoPhaseScatter(owner)%sendcounts_phase1(0:layout(2)-1))
        allocate(TwoPhaseScatter(owner)%senddispls_phase1(0:layout(2)-1))
        TwoPhaseScatter(owner)%senddispls_phase1(0) = 0
        TwoPhaseScatter(owner)%sendcounts_phase1(0) = globalsizes(1) * NumRowsPerRank(0)
        do row=1, layout(2)-1
          TwoPhaseScatter(owner)%sendcounts_phase1(row) = globalsizes(1) * NumRowsPerRank(row)
          TwoPhaseScatter(owner)%senddispls_phase1(row) = TwoPhaseScatter(owner)%senddispls_phase1(row-1) + &
                                                          TwoPhaseScatter(owner)%sendcounts_phase1(row-1)
        enddo
      endif
      allocate(TwoPhaseScatter(owner)%sendcounts_phase2(0:layout(1)-1))
      allocate(TwoPhaseScatter(owner)%senddispls_phase2(0:layout(1)-1))
      TwoPhaseScatter(owner)%senddispls_phase2(0) = 0
      TwoPhaseScatter(owner)%sendcounts_phase2(0) = NumColsPerRank(0)
      do col = 1, layout(1)-1
        TwoPhaseScatter(owner)%sendcounts_phase2(col) = NumColsPerRank(col)
        TwoPhaseScatter(owner)%senddispls_phase2(col) = TwoPhaseScatter(owner)%senddispls_phase2(col-1) + &
                                                        TwoPhaseScatter(owner)%sendcounts_phase2(col-1)
      enddo
      TwoPhaseScatter(owner)%lalloc = .true.
    endif

    if (TwoPhaseScatter(owner)%localvec_r4 == MPI_DATATYPE_NULL) then
      lb=0; extent=4
      call MPI_Type_vector(localsizes(2), 1, NumColsPerRank(mycol), MPI_REAL, temptype, ierr)
      call MPI_Type_create_resized(temptype, lb, extent, TwoPhaseScatter(owner)%localvec_r4, ierr)
      call MPI_Type_commit(TwoPhaseScatter(owner)%localvec_r4, ierr)
      call MPI_Type_free(temptype, ierr)
    endif

    if (mycol == MyColGlobal(owner) .and. TwoPhaseScatter(owner)%vec_r4 == MPI_DATATYPE_NULL) then
      lb=0; extent=4
      call MPI_Type_vector(localsizes(2), 1, globalsizes(1), MPI_REAL, temptype, ierr)
      call MPI_Type_create_resized(temptype, lb, extent, TwoPhaseScatter(owner)%vec_r4, ierr)
      call MPI_Type_commit(TwoPhaseScatter(owner)%vec_r4, ierr)
      call MPI_Type_free(temptype, ierr)
    endif

    if (mycol == MyColGlobal(owner)) then
      if (rank == owner) then
        call MPI_Iscatterv(sendbuf(1,1), TwoPhaseScatter(owner)%sendcounts_phase1, TwoPhaseScatter(owner)%senddispls_phase1, &
                           MPI_REAL, scatter_workspace_r4(1,1,b_ind), TwoPhaseScatter(owner)%sendcounts_phase1(myrow), &
                           MPI_REAL, MyRankInColComm(owner), colComm, req_p1, ierr)
      else
        call MPI_Iscatterv(MPI_BOTTOM, TwoPhaseScatter(owner)%sendcounts_phase1, TwoPhaseScatter(owner)%senddispls_phase1, &
                           MPI_REAL, scatter_workspace_r4(1,1,b_ind), TwoPhaseScatter(owner)%sendcounts_phase1(myrow), &
                           MPI_REAL, MyRankInColComm(owner), colComm, req_p1, ierr)
      endif
    else
      req_p1 = MPI_REQUEST_NULL
    endif
  end subroutine TwoPhaseScatter_Phase1_r4


  subroutine TwoPhaseScatter_Phase1_r8(owner, rank, sendbuf, b_ind, req_p1)
    implicit none
    integer, intent(in) :: owner, rank
    real(kind=8), contiguous, target, asynchronous, intent(inout) :: sendbuf(:,:)
    integer, intent(in)  :: b_ind
    integer, intent(out) :: req_p1

    integer :: row, col, ierr, temptype
    integer(kind=MPI_ADDRESS_KIND) :: lb, extent
    integer :: myrow, mycol

    myrow = NSindex
    mycol = EWindex

    if (.not. TwoPhaseScatter(owner)%lalloc) then
      if (mycol == MyColGlobal(owner)) then
        allocate(TwoPhaseScatter(owner)%sendcounts_phase1(0:layout(2)-1))
        allocate(TwoPhaseScatter(owner)%senddispls_phase1(0:layout(2)-1))
        TwoPhaseScatter(owner)%senddispls_phase1(0) = 0
        TwoPhaseScatter(owner)%sendcounts_phase1(0) = globalsizes(1) * NumRowsPerRank(0)
        do row=1, layout(2)-1
          TwoPhaseScatter(owner)%sendcounts_phase1(row) = globalsizes(1) * NumRowsPerRank(row)
          TwoPhaseScatter(owner)%senddispls_phase1(row) = TwoPhaseScatter(owner)%senddispls_phase1(row-1) + &
                                                          TwoPhaseScatter(owner)%sendcounts_phase1(row-1)
        enddo
      endif

      allocate(TwoPhaseScatter(owner)%sendcounts_phase2(0:layout(1)-1))
      allocate(TwoPhaseScatter(owner)%senddispls_phase2(0:layout(1)-1))
      TwoPhaseScatter(owner)%senddispls_phase2(0) = 0
      TwoPhaseScatter(owner)%sendcounts_phase2(0) = NumColsPerRank(0)
      do col = 1, layout(1)-1
        TwoPhaseScatter(owner)%sendcounts_phase2(col) = NumColsPerRank(col)
        TwoPhaseScatter(owner)%senddispls_phase2(col) = TwoPhaseScatter(owner)%senddispls_phase2(col-1) + &
                                                        TwoPhaseScatter(owner)%sendcounts_phase2(col-1)
      enddo
      TwoPhaseScatter(owner)%lalloc = .true.
    endif

    if (TwoPhaseScatter(owner)%localvec_r8 == MPI_DATATYPE_NULL) then
      lb=0; extent=8
      call MPI_Type_vector(localsizes(2), 1, NumColsPerRank(mycol), MPI_DOUBLE_PRECISION, temptype, ierr)
      call MPI_Type_create_resized(temptype, lb, extent, TwoPhaseScatter(owner)%localvec_r8, ierr)
      call MPI_Type_commit(TwoPhaseScatter(owner)%localvec_r8, ierr)
      call MPI_Type_free(temptype, ierr)
    endif

    if (mycol == MyColGlobal(owner) .and. TwoPhaseScatter(owner)%vec_r8 == MPI_DATATYPE_NULL) then
      lb=0; extent=8
      call MPI_Type_vector(localsizes(2), 1, globalsizes(1), MPI_DOUBLE_PRECISION, temptype, ierr)
      call MPI_Type_create_resized(temptype, lb, extent, TwoPhaseScatter(owner)%vec_r8, ierr)
      call MPI_Type_commit(TwoPhaseScatter(owner)%vec_r8, ierr)
      call MPI_Type_free(temptype, ierr)
    endif

    if (mycol == MyColGlobal(owner)) then
      if (rank == owner) then
        call MPI_Iscatterv(sendbuf(1,1), TwoPhaseScatter(owner)%sendcounts_phase1, TwoPhaseScatter(owner)%senddispls_phase1, &
                           MPI_DOUBLE_PRECISION, scatter_workspace_r8(1,1,b_ind), TwoPhaseScatter(owner)%sendcounts_phase1(myrow), &
                           MPI_DOUBLE_PRECISION, MyRankInColComm(owner), colComm, req_p1, ierr)
      else
        call MPI_Iscatterv(MPI_BOTTOM, TwoPhaseScatter(owner)%sendcounts_phase1, TwoPhaseScatter(owner)%senddispls_phase1, &
                           MPI_DOUBLE_PRECISION, scatter_workspace_r8(1,1,b_ind), TwoPhaseScatter(owner)%sendcounts_phase1(myrow), &
                           MPI_DOUBLE_PRECISION, MyRankInColComm(owner), colComm, req_p1, ierr)
      endif
    else
      req_p1 = MPI_REQUEST_NULL
    endif
  end subroutine TwoPhaseScatter_Phase1_r8


  subroutine TwoPhaseScatter_Phase2_r4(owner, rank, recvbuf, b_ind, req_p2)
    implicit none
    integer, intent(in) :: owner, rank
    real(kind=4), contiguous, target, asynchronous, intent(inout) :: recvbuf(:,:)
    integer, intent(in)  :: b_ind
    integer, intent(out) :: req_p2

    integer :: ierr, vec
    integer :: mycol

    mycol = EWindex

    if (mycol == MyColGlobal(owner)) then
      vec = TwoPhaseScatter(owner)%vec_r4
      call MPI_Iscatterv(scatter_workspace_r4(1,1,b_ind), TwoPhaseScatter(owner)%sendcounts_phase2, &
                         TwoPhaseScatter(owner)%senddispls_phase2, vec, recvbuf(1,1), &
                         TwoPhaseScatter(owner)%sendcounts_phase2(mycol), TwoPhaseScatter(owner)%localvec_r4, &
                         MyRankInRowComm(owner), rowComm, req_p2, ierr)
    else
      call MPI_Iscatterv(MPI_BOTTOM, TwoPhaseScatter(owner)%sendcounts_phase2, TwoPhaseScatter(owner)%senddispls_phase2, &
                         MPI_REAL, recvbuf(1,1), TwoPhaseScatter(owner)%sendcounts_phase2(mycol), &
                         TwoPhaseScatter(owner)%localvec_r4, MyRankInRowComm(owner), rowComm, req_p2, ierr)
    endif
  end subroutine TwoPhaseScatter_Phase2_r4


  subroutine TwoPhaseScatter_Phase2_r8(owner, rank, recvbuf, b_ind, req_p2)
    implicit none
    integer, intent(in) :: owner, rank
    real(kind=8), contiguous, target, asynchronous, intent(inout) :: recvbuf(:,:)
    integer, intent(in)  :: b_ind
    integer, intent(out) :: req_p2

    integer :: ierr, vec
    integer :: mycol

    mycol = EWindex

    if (mycol == MyColGlobal(owner)) then
      vec = TwoPhaseScatter(owner)%vec_r8
      call MPI_Iscatterv(scatter_workspace_r8(1,1,b_ind), TwoPhaseScatter(owner)%sendcounts_phase2, &
                         TwoPhaseScatter(owner)%senddispls_phase2, vec, recvbuf(1,1), &
                         TwoPhaseScatter(owner)%sendcounts_phase2(mycol), TwoPhaseScatter(owner)%localvec_r8, &
                         MyRankInRowComm(owner), rowComm, req_p2, ierr)
    else
      call MPI_Iscatterv(MPI_BOTTOM, TwoPhaseScatter(owner)%sendcounts_phase2, TwoPhaseScatter(owner)%senddispls_phase2, &
                         MPI_DOUBLE_PRECISION, recvbuf(1,1), TwoPhaseScatter(owner)%sendcounts_phase2(mycol), &
                         TwoPhaseScatter(owner)%localvec_r8, MyRankInRowComm(owner), rowComm, req_p2, ierr)
    endif
  end subroutine TwoPhaseScatter_Phase2_r8

  subroutine TwoPhaseGather_Phase1_r4(owner, rank, sendbuf, b_ind, req_p1)
    implicit none
    integer, intent(in) :: owner, rank
    real(kind=4), contiguous, target, asynchronous, intent(inout) :: sendbuf(:,:)
    integer, intent(in)  :: b_ind
    integer, intent(out) :: req_p1

    integer :: col, row, ierr, temptype, vec, localvec
    integer(kind=MPI_ADDRESS_KIND) :: lb, extent
    integer :: myrow, mycol

    myrow = NSindex
    mycol = EWindex

    if (.not. TwoPhaseGather(owner)%lalloc) then
      allocate(TwoPhaseGather(owner)%recvcounts_phase1(0:layout(1)-1))
      allocate(TwoPhaseGather(owner)%recvdispls_phase1(0:layout(1)-1))
      TwoPhaseGather(owner)%recvdispls_phase1(0) = 0
      TwoPhaseGather(owner)%recvcounts_phase1(0) = NumColsPerRank(0)
      do col = 1, layout(1)-1
        TwoPhaseGather(owner)%recvcounts_phase1(col) = NumColsPerRank(col)
        TwoPhaseGather(owner)%recvdispls_phase1(col) = TwoPhaseGather(owner)%recvdispls_phase1(col-1) + &
                                                       TwoPhaseGather(owner)%recvcounts_phase1(col-1)
      enddo

      if (mycol == MyColGlobal(owner)) then
        allocate(TwoPhaseGather(owner)%recvcounts_phase2(0:layout(2)-1))
        allocate(TwoPhaseGather(owner)%recvdispls_phase2(0:layout(2)-1))
        TwoPhaseGather(owner)%recvdispls_phase2(0) = 0
        TwoPhaseGather(owner)%recvcounts_phase2(0) = globalsizes(1) * NumRowsPerRank(0)
        do row=1, layout(2)-1
          TwoPhaseGather(owner)%recvcounts_phase2(row) = globalsizes(1) * NumRowsPerRank(row)
          TwoPhaseGather(owner)%recvdispls_phase2(row) = TwoPhaseGather(owner)%recvdispls_phase2(row-1) + &
                                                         TwoPhaseGather(owner)%recvcounts_phase2(row-1)
        enddo
      endif
      TwoPhaseGather(owner)%lalloc = .true.
    endif

    if (TwoPhaseGather(owner)%vec_r4 == MPI_DATATYPE_NULL) then
      lb=0; extent=4
      call MPI_Type_vector(localsizes(2), 1, globalsizes(1), MPI_REAL, temptype, ierr)
      call MPI_Type_create_resized(temptype, lb, extent, TwoPhaseGather(owner)%vec_r4, ierr)
      call MPI_Type_commit(TwoPhaseGather(owner)%vec_r4, ierr)
      call MPI_Type_free(temptype, ierr)
    endif

    if (TwoPhaseGather(owner)%localvec_r4 == MPI_DATATYPE_NULL) then
      lb=0; extent=4
      call MPI_Type_vector(localsizes(2), 1, NumColsPerRank(mycol), MPI_REAL, temptype, ierr)
      call MPI_Type_create_resized(temptype, lb, extent, TwoPhaseGather(owner)%localvec_r4, ierr)
      call MPI_Type_commit(TwoPhaseGather(owner)%localvec_r4, ierr)
      call MPI_Type_free(temptype, ierr)
    endif

    vec = TwoPhaseGather(owner)%vec_r4
    localvec = TwoPhaseGather(owner)%localvec_r4

    if (mycol == MyColGlobal(owner)) then
      call MPI_Igatherv(sendbuf(1,1), TwoPhaseGather(owner)%recvcounts_phase1(mycol), localvec, &
                        gather_workspace_r4(1,1,b_ind), TwoPhaseGather(owner)%recvcounts_phase1, &
                        TwoPhaseGather(owner)%recvdispls_phase1, vec, MyRankInRowComm(owner), &
                        rowComm, req_p1, ierr)
    else
      call MPI_Igatherv(sendbuf(1,1), TwoPhaseGather(owner)%recvcounts_phase1(mycol), localvec, &
                        MPI_BOTTOM, TwoPhaseGather(owner)%recvcounts_phase1, &
                        TwoPhaseGather(owner)%recvdispls_phase1, vec, MyRankInRowComm(owner), &
                        rowComm, req_p1, ierr)
    endif
  end subroutine TwoPhaseGather_Phase1_r4

  subroutine TwoPhaseGather_Phase1_r8(owner, rank, sendbuf, b_ind, req_p1)
    implicit none
    integer, intent(in) :: owner, rank
    real(kind=8), contiguous, target, asynchronous, intent(inout) :: sendbuf(:,:)
    integer, intent(in)  :: b_ind
    integer, intent(out) :: req_p1

    integer :: col, row, ierr, temptype, vec, localvec
    integer(kind=MPI_ADDRESS_KIND) :: lb, extent
    integer :: myrow, mycol

    myrow = NSindex
    mycol = EWindex

    if (.not. TwoPhaseGather(owner)%lalloc) then
      allocate(TwoPhaseGather(owner)%recvcounts_phase1(0:layout(1)-1))
      allocate(TwoPhaseGather(owner)%recvdispls_phase1(0:layout(1)-1))
      TwoPhaseGather(owner)%recvdispls_phase1(0) = 0
      TwoPhaseGather(owner)%recvcounts_phase1(0) = NumColsPerRank(0)
      do col = 1, layout(1)-1
        TwoPhaseGather(owner)%recvcounts_phase1(col) = NumColsPerRank(col)
        TwoPhaseGather(owner)%recvdispls_phase1(col) = TwoPhaseGather(owner)%recvdispls_phase1(col-1) + &
                                                       TwoPhaseGather(owner)%recvcounts_phase1(col-1)
      enddo

      if (mycol == MyColGlobal(owner)) then
        allocate(TwoPhaseGather(owner)%recvcounts_phase2(0:layout(2)-1))
        allocate(TwoPhaseGather(owner)%recvdispls_phase2(0:layout(2)-1))
        TwoPhaseGather(owner)%recvdispls_phase2(0) = 0
        TwoPhaseGather(owner)%recvcounts_phase2(0) = globalsizes(1) * NumRowsPerRank(0)
        do row=1, layout(2)-1
          TwoPhaseGather(owner)%recvcounts_phase2(row) = globalsizes(1) * NumRowsPerRank(row)
          TwoPhaseGather(owner)%recvdispls_phase2(row) = TwoPhaseGather(owner)%recvdispls_phase2(row-1) + &
                                                         TwoPhaseGather(owner)%recvcounts_phase2(row-1)
        enddo
      endif
      TwoPhaseGather(owner)%lalloc = .true.
    endif

    if (TwoPhaseGather(owner)%vec_r8 == MPI_DATATYPE_NULL) then
      lb=0; extent=8
      call MPI_Type_vector(localsizes(2), 1, globalsizes(1), MPI_DOUBLE_PRECISION, temptype, ierr)
      call MPI_Type_create_resized(temptype, lb, extent, TwoPhaseGather(owner)%vec_r8, ierr)
      call MPI_Type_commit(TwoPhaseGather(owner)%vec_r8, ierr)
      call MPI_Type_free(temptype, ierr)
    endif

    if (TwoPhaseGather(owner)%localvec_r8 == MPI_DATATYPE_NULL) then
      lb=0; extent=8
      call MPI_Type_vector(localsizes(2), 1, NumColsPerRank(mycol), MPI_DOUBLE_PRECISION, temptype, ierr)
      call MPI_Type_create_resized(temptype, lb, extent, TwoPhaseGather(owner)%localvec_r8, ierr)
      call MPI_Type_commit(TwoPhaseGather(owner)%localvec_r8, ierr)
      call MPI_Type_free(temptype, ierr)
    endif

    vec = TwoPhaseGather(owner)%vec_r8
    localvec = TwoPhaseGather(owner)%localvec_r8

    if (mycol == MyColGlobal(owner)) then
      call MPI_Igatherv(sendbuf(1,1), TwoPhaseGather(owner)%recvcounts_phase1(mycol), localvec, &
                        gather_workspace_r8(1,1,b_ind), TwoPhaseGather(owner)%recvcounts_phase1, &
                        TwoPhaseGather(owner)%recvdispls_phase1, vec, MyRankInRowComm(owner), &
                        rowComm, req_p1, ierr)
    else
      call MPI_Igatherv(sendbuf(1,1), TwoPhaseGather(owner)%recvcounts_phase1(mycol), localvec, &
                        MPI_BOTTOM, TwoPhaseGather(owner)%recvcounts_phase1, &
                        TwoPhaseGather(owner)%recvdispls_phase1, vec, MyRankInRowComm(owner), &
                        rowComm, req_p1, ierr)
    endif
  end subroutine TwoPhaseGather_Phase1_r8


  subroutine TwoPhaseGather_Phase2_r4(owner, rank, recvbuf, b_ind, req_p2)
    implicit none
    integer, intent(in) :: owner, rank
    real(kind=4), contiguous, target, asynchronous, intent(inout) :: recvbuf(:,:)
    integer, intent(in)  :: b_ind
    integer, intent(out) :: req_p2

    integer :: myrow, mycol, ierr

    myrow = NSindex
    mycol = EWindex

    if (mycol == MyColGlobal(owner)) then
      if (rank == owner) then
        call MPI_Igatherv(gather_workspace_r4(1,1,b_ind), TwoPhaseGather(owner)%recvcounts_phase2(myrow), MPI_REAL, &
                          recvbuf(1,1), TwoPhaseGather(owner)%recvcounts_phase2, &
                          TwoPhaseGather(owner)%recvdispls_phase2, MPI_REAL, MyRankInColComm(owner), &
                          colComm, req_p2, ierr)
      else
        call MPI_Igatherv(gather_workspace_r4(1,1,b_ind), TwoPhaseGather(owner)%recvcounts_phase2(myrow), MPI_REAL, &
                          MPI_BOTTOM, TwoPhaseGather(owner)%recvcounts_phase2, &
                          TwoPhaseGather(owner)%recvdispls_phase2, MPI_REAL, MyRankInColComm(owner), &
                          colComm, req_p2, ierr)
      endif
    else
      req_p2 = MPI_REQUEST_NULL
    endif
  end subroutine TwoPhaseGather_Phase2_r4


  subroutine TwoPhaseGather_Phase2_r8(owner, rank, recvbuf, b_ind, req_p2)
    implicit none
    integer, intent(in) :: owner, rank
    real(kind=8), contiguous, target, asynchronous, intent(inout) :: recvbuf(:,:)
    integer, intent(in)  :: b_ind
    integer, intent(out) :: req_p2

    integer :: myrow, mycol, ierr

    myrow = NSindex
    mycol = EWindex

    if (mycol == MyColGlobal(owner)) then
      if (rank == owner) then
        call MPI_Igatherv(gather_workspace_r8(1,1,b_ind), TwoPhaseGather(owner)%recvcounts_phase2(myrow), MPI_DOUBLE_PRECISION, &
                          recvbuf(1,1), TwoPhaseGather(owner)%recvcounts_phase2, &
                          TwoPhaseGather(owner)%recvdispls_phase2, MPI_DOUBLE_PRECISION, MyRankInColComm(owner), &
                          colComm, req_p2, ierr)
      else
        call MPI_Igatherv(gather_workspace_r8(1,1,b_ind), TwoPhaseGather(owner)%recvcounts_phase2(myrow), MPI_DOUBLE_PRECISION, &
                          MPI_BOTTOM, TwoPhaseGather(owner)%recvcounts_phase2, &
                          TwoPhaseGather(owner)%recvdispls_phase2, MPI_DOUBLE_PRECISION, MyRankInColComm(owner), &
                          colComm, req_p2, ierr)
      endif
    else
      req_p2 = MPI_REQUEST_NULL
    endif
  end subroutine TwoPhaseGather_Phase2_r8

end module TwoPhaseScatterGather
