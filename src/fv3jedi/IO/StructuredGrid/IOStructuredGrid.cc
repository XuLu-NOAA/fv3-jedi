/*
 * (C) Copyright 2025- UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include <netcdf.h>

#include <map>
#include <vector>

#include "atlas/functionspace.h"
#include "atlas/grid.h"
#include "atlas/grid/StructuredGrid.h"
#include "atlas/grid/Spacing.h"

#include "oops/base/GeometryData.h"
#include "oops/util/Logger.h"
#include "oops/util/stringFunctions.h"
#include "oops/util/Timer.h"

#include "fv3jedi/FieldMetadata/FieldsMetadata.h"
#include "fv3jedi/Geometry/Geometry.h"
#include "fv3jedi/Increment/Increment.h"
#include "fv3jedi/IO/StructuredGrid/IOStructuredGrid.h"
#include "fv3jedi/State/State.h"

#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include "atlas/array.h"
#include "atlas/field.h"

#include <iostream>
#include <sstream>
#include <unistd.h>  // getpid()
#include <limits>

namespace fv3jedi {
// -------------------------------------------------------------------------------------------------
static IOMaker<IOStructuredGrid> makerIOStructuredGrid_("structured grid");
static IOMaker<IOStructuredGrid> makerIOAuxGrid_("auxgrid");
// -------------------------------------------------------------------------------------------------
static inline void nc_rc(const int return_code, const std::string & operation) {
  if (return_code) {
    ABORT("IOStructuredGrid netCDF operation \'" + operation + "\' failed with error: "
          + nc_strerror(return_code));
  }
}

static size_t findNearestIndex(const std::vector<double>& arr, double value, bool lower_bound) {
  if (arr.empty()) return 0;
  size_t left = 0;
  size_t right = arr.size() - 1;
  if (lower_bound) {
    // Find first index where arr[idx] >= value
    while (left < right) {
      size_t mid = (left + right) / 2;
      if (arr[mid] < value) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    return left;
  } else {
    // Find last index where arr[idx] <= value
    while (left < right) {
      size_t mid = (left + right + 1) / 2;
      if (arr[mid] > value) {
        right = mid - 1;
      } else {
        left = mid;
      }
    }
    return left;
  }
}

// -------------------------------------------------------------------------------------------------
IOStructuredGrid::IOStructuredGrid(const Geometry & geom, const Parameters_ & params)
  : IOBase(geom, params.toConfiguration()),
    interpolator_(),
    interpolatorBack_(),
    params_(params),
    gridStr_(""),
    geom_(geom),
    writeFunctionSpace_(),
    readFunctionSpace_() {
  util::Timer timer(classname(), "IOStructuredGrid");
  oops::Log::trace() << classname() << " constructor starting" << std::endl;

  const std::string mode = (params_.mode.value() != boost::none) ? *params_.mode.value() : "write";

  std::string outputGridType = params.outputGridType.value();

  if (outputGridType == "latlon") {
    outputGridType = "L" + std::to_string(4*(geom.npx()-1)) + "x" +
                     std::to_string(2*(geom.npy()-1)+1);
  } else if (outputGridType == "gaussian") {
    outputGridType = "F" + std::to_string(geom.npy()-1);
  }

  if (outputGridType[0] == 'L') {
    gridStr_ = "latlon";
  } else if (outputGridType[0] == 'F') {
    gridStr_ = "gaussian";
  } else {
    ABORT("IOStructuredGrid: outputGridType must begin with L (latlon) or F (regular gaussian).");
  }

  const atlas::Grid outGrid(outputGridType);

  // All output points on rank 0 (as you had)
  std::vector<int> zeros(outGrid.size(), 0);
  const atlas::grid::Distribution dist(geom.getComm().size(), outGrid.size(), zeros.data());

  eckit::LocalConfiguration atlas_conf;
  atlas_conf.set("mpi_comm", geom.getComm().name());

  writeFunctionSpace_.reset(new atlas::functionspace::StructuredColumns(outGrid, dist, atlas_conf));

  // Model geom data
  oops::GeometryData geomData(geom.functionSpace(), geom.fields(), geom.levelsAreTopDown(),
                              geom.getComm());

  // -------------------------
  // WRITE-only mode
  // -------------------------
  if (mode == "write") {
    interpolator_.reset(new oops::GlobalInterpolator(params.toConfiguration(),
                                                     geomData,
                                                     *writeFunctionSpace_,
                                                     geom.getComm()));
    oops::Log::trace() << classname() << " constructor done (write)" << std::endl;
    return;
  }

  // -------------------------
  // READ/BOTH mode: build INPUT geometry from file
  // -------------------------
  if (params_.inputFilename.value() == boost::none) {
    ABORT("IOStructuredGrid: mode is 'read' or 'both' but no input filename specified");
  }
  const std::string inFile = *params_.inputFilename.value();

  size_t nLat = 0, nLon = 0;
  std::vector<double> file_lats;
  std::vector<double> file_lons;

  if (geom_.getComm().rank() == 0) {
    int ncid;
    nc_rc(nc_open(inFile.c_str(), NC_NOWRITE, &ncid), "nc_open " + inFile);

    int dim_gridyt, dim_gridxt;
    nc_rc(nc_inq_dimid(ncid, "grid_yt", &dim_gridyt), "nc_inq_dimid grid_yt");
    nc_rc(nc_inq_dimlen(ncid, dim_gridyt, &nLat), "nc_inq_dimlen grid_yt");
    nc_rc(nc_inq_dimid(ncid, "grid_xt", &dim_gridxt), "nc_inq_dimid grid_xt");
    nc_rc(nc_inq_dimlen(ncid, dim_gridxt, &nLon), "nc_inq_dimlen grid_xt");

    file_lats.resize(nLat);
    file_lons.resize(nLon);

    int var_gridyt, var_gridxt;
    nc_rc(nc_inq_varid(ncid, "grid_yt", &var_gridyt), "nc_inq_varid grid_yt");
    nc_rc(nc_get_var_double(ncid, var_gridyt, file_lats.data()), "nc_get_var_double grid_yt");

    nc_rc(nc_inq_varid(ncid, "grid_xt", &var_gridxt), "nc_inq_varid grid_xt");
    nc_rc(nc_get_var_double(ncid, var_gridxt, file_lons.data()), "nc_get_var_double grid_xt");

    nc_rc(nc_close(ncid), "nc_close");

    oops::Log::info() << "Input file grid: nLon=" << nLon << ", nLat=" << nLat << std::endl;
  }

  // Broadcast dims and coordinates
  geom_.getComm().broadcast(nLat, 0);
  geom_.getComm().broadcast(nLon, 0);
  if (geom_.getComm().rank() != 0) {
    file_lats.resize(nLat);
    file_lons.resize(nLon);
  }
  geom_.getComm().broadcast(file_lats, 0);
  geom_.getComm().broadcast(file_lons, 0);

  // Use L-grid ONLY for indexing/distribution (NOT for geometry)
  const std::string inputGridType = "L" + std::to_string(nLon) + "x" + std::to_string(nLat);
  const atlas::Grid inputGrid(inputGridType);

  // -----------------------------------------------------------------------------
  // Build distributed structured FunctionSpace for the global file grid
  // -----------------------------------------------------------------------------
  atlas::grid::Partitioner partitioner("equal_regions");
  
  // --- IMPORTANT: enable halo/ghosts so haloExchange is real and seam stencils exist ---
  atlas_conf.set("halo", 2);
  
  readFunctionSpace_.reset(new atlas::functionspace::StructuredColumns(inputGrid, partitioner, atlas_conf));
  
  // --- Override lonlat using file grid_xt/grid_yt ---
  atlas::FieldSet structuredAux;
  
  atlas::Field lonlat = readFunctionSpace_->createField<double>(
      atlas::option::name("lonlat") | atlas::option::variables(2));
  
  auto lonlatView = atlas::array::make_view<double,2>(lonlat);
  auto gidxView   = atlas::array::make_view<atlas::gidx_t,1>(readFunctionSpace_->global_index());
  auto ghostView  = atlas::array::make_view<int,1>(readFunctionSpace_->ghost());
  
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (atlas::idx_t p = 0; p < lonlatView.shape(0); ++p) {
    lonlatView(p,0) = nan;
    lonlatView(p,1) = nan;
  }
  
  // owned fill
  for (atlas::idx_t p = 0; p < lonlatView.shape(0); ++p) {
    if (ghostView(p) != 0) continue;              // owned only
    const auto gidx = gidxView(p);
    if (gidx <= 0) continue;
    const std::size_t g = static_cast<std::size_t>(gidx) - 1;
    const std::size_t i = g % nLon;
    const std::size_t j = g / nLon;
    lonlatView(p,0) = file_lons[i];
    lonlatView(p,1) = file_lats[j];               // flipLat already autodetected earlier in your version
  }
  
  // populate ghost lonlat consistently (NOW this matters because ghost>0)
  readFunctionSpace_->haloExchange(lonlat);
  
  structuredAux.add(lonlat);
  
  // quick check: report ghost count (rank0)
  if (geom.getComm().rank() == 0) {
    std::int64_t ng = 0;
    for (atlas::idx_t p = 0; p < ghostView.shape(0); ++p) ng += (ghostView(p) != 0);
    oops::Log::info() << "[CHECK-HALO] structured FS halo enabled: ghost points on rank0=" << ng << std::endl;
  }
  
  oops::GeometryData structuredGeomData(*readFunctionSpace_,
                                        structuredAux,
                                        geom.levelsAreTopDown(),
                                        geom.getComm());
  
  // Reverse interpolator (structured -> model)
  eckit::LocalConfiguration interpConfig = params.toConfiguration();
  if (!interpConfig.has("local interpolator type")) {
    interpConfig.set("local interpolator type", "oops unstructured grid interpolator");
  }
  
  interpolatorBack_.reset(new oops::GlobalInterpolator(interpConfig,
                                                       structuredGeomData,
                                                       geom.functionSpace(),
                                                       geom.getComm()));
  
  // Forward interpolator if needed
  if (mode == "both") {
    interpolator_.reset(new oops::GlobalInterpolator(params.toConfiguration(),
                                                     geomData,
                                                     *writeFunctionSpace_,
                                                     geom.getComm()));
  }

  oops::Log::trace() << classname() << " constructor done (" << mode << ")" << std::endl;
}
// -------------------------------------------------------------------------------------------------
IOStructuredGrid::~IOStructuredGrid() {
  util::Timer timer(classname(), "~IOStructuredGrid");
  oops::Log::trace() << classname() << " destructor starting" << std::endl;
  oops::Log::trace() << classname() << " destructor done" << std::endl;
}

// -------------------------------------------------------------------------------------------------
// //XL
void IOStructuredGrid::loadAkBkOnce_(int nLevModel) const {
  std::call_once(akbk_once_, [&]() {
    const int rank = geom_.getComm().rank();
    eckit::LocalConfiguration cfg = params_.toConfiguration();
    const std::string coeffFile = *params_.akbk.value();
    const size_t expect = static_cast<size_t>(nLevModel + 1);
    int ncid;
    nc_rc(nc_open(coeffFile.c_str(), NC_NOWRITE, &ncid), "nc_open " + coeffFile);
    auto read_time_axis = [&](const char * varname, std::vector<double> & out) {
      int varid;
      nc_rc(nc_inq_varid(ncid, varname, &varid), std::string("nc_inq_varid ") + varname);
      int ndims = 0;
      nc_rc(nc_inq_varndims(ncid, varid, &ndims), std::string("nc_inq_varndims ") + varname);
      if (ndims != 2) {
        if (rank == 0) oops::Log::error() << varname << " is not 2D (Time,xaxis_1)" << std::endl;
        ABORT(std::string(varname) + " wrong rank");
      }
      int dimids[NC_MAX_DIMS];
      nc_rc(nc_inq_vardimid(ncid, varid, dimids), std::string("nc_inq_vardimid ") + varname);
      size_t ntime = 0, naxis = 0;
      nc_rc(nc_inq_dimlen(ncid, dimids[0], &ntime), std::string("nc_inq_dimlen Time for ") + varname);
      nc_rc(nc_inq_dimlen(ncid, dimids[1], &naxis), std::string("nc_inq_dimlen xaxis_1 for ") + varname);
      if (ntime < 1) ABORT(std::string(varname) + " has Time dim < 1");
      if (naxis != expect) {
        if (rank == 0) {
          oops::Log::error() << varname << " axis length mismatch: expected "
                             << expect << " got " << naxis << std::endl;
        }
        ABORT(std::string(varname) + " wrong axis length");
      }
      out.assign(expect, 0.0);
      size_t start[2] = {0, 0};
      size_t count[2] = {1, expect};
      // Works even if stored as float; netCDF will convert to double.
      nc_rc(nc_get_vara_double(ncid, varid, start, count, out.data()),
            std::string("nc_get_vara_double ") + varname);
    };
    read_time_axis("ak", ak_);
    read_time_axis("bk", bk_);
    nc_rc(nc_close(ncid), "nc_close " + coeffFile);
    akbk_nlev_model_ = nLevModel;
  });
  if (akbk_nlev_model_ != nLevModel) {
    ABORT("ak/bk cache mismatch: different nLevModel than previously loaded");
  }
}

void IOStructuredGrid::read(State & x,
                            const eckit::LocalConfiguration & fileionames,
                            const eckit::LocalConfiguration & fileioscaling) const {
  util::Timer timer(classname(), "read(State vertical remap)");

  using clock_t = std::chrono::steady_clock;
  auto sec = [](clock_t::time_point a, clock_t::time_point b) {
    return std::chrono::duration<double>(b - a).count();
  };

  const auto t_total0 = clock_t::now();
  const int rank = geom_.getComm().rank();

  auto log0t = [&](const std::string &s, double v) {
    if (rank == 0) oops::Log::info() << s << v << " s" << std::endl;
  };

  const std::string inFile = *params_.inputFilename.value();
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // ============================================================
  // 0) Read file dims + coords (rank0) and broadcast
  // ============================================================
  size_t nLat = 0, nLon = 0, nLevFile = 0;
  std::vector<double> file_lats, file_lons;
  int fileNorthToSouth_i = 0;

  auto t_dims0 = clock_t::now();
  if (rank == 0) {
    int ncid;
    nc_rc(nc_open(inFile.c_str(), NC_NOWRITE, &ncid), "nc_open " + inFile);

    int dim_gridyt, dim_gridxt, dim_pfull;
    nc_rc(nc_inq_dimid(ncid, "grid_yt", &dim_gridyt), "nc_inq_dimid grid_yt");
    nc_rc(nc_inq_dimlen(ncid, dim_gridyt, &nLat), "nc_inq_dimlen grid_yt");
    nc_rc(nc_inq_dimid(ncid, "grid_xt", &dim_gridxt), "nc_inq_dimid grid_xt");
    nc_rc(nc_inq_dimlen(ncid, dim_gridxt, &nLon), "nc_inq_dimlen grid_xt");

    nc_rc(nc_inq_dimid(ncid, "pfull", &dim_pfull), "nc_inq_dimid pfull");
    nc_rc(nc_inq_dimlen(ncid, dim_pfull, &nLevFile), "nc_inq_dimlen pfull");

    file_lats.resize(nLat);
    file_lons.resize(nLon);

    int var_gridyt, var_gridxt;
    nc_rc(nc_inq_varid(ncid, "grid_yt", &var_gridyt), "nc_inq_varid grid_yt");
    nc_rc(nc_get_var_double(ncid, var_gridyt, file_lats.data()), "nc_get_var grid_yt");
    nc_rc(nc_inq_varid(ncid, "grid_xt", &var_gridxt), "nc_inq_varid grid_xt");
    nc_rc(nc_get_var_double(ncid, var_gridxt, file_lons.data()), "nc_get_var grid_xt");

    fileNorthToSouth_i = (nLat >= 2 && file_lats[0] > file_lats[nLat - 1]) ? 1 : 0;

    nc_rc(nc_close(ncid), "nc_close " + inFile);
  }

  geom_.getComm().broadcast(nLat, 0);
  geom_.getComm().broadcast(nLon, 0);
  geom_.getComm().broadcast(nLevFile, 0);

  if (rank != 0) {
    file_lats.resize(nLat);
    file_lons.resize(nLon);
  }
  geom_.getComm().broadcast(file_lats.begin(), file_lats.end(), 0);
  geom_.getComm().broadcast(file_lons.begin(), file_lons.end(), 0);
  geom_.getComm().broadcast(fileNorthToSouth_i, 0);

  log0t("[TIMER] read dims+coords+bcast: ", sec(t_dims0, clock_t::now()));

  // ============================================================
  // 1) Optional regional subset (lon/lat bounds) to reduce I/O
  //    - Uses params_.lon_min/lon_max/lat_min/lat_max when provided
  //    - Adds a small halo (buffer) around requested window
  // ============================================================
  size_t lat_start = 0, lat_count = nLat, lon_start = 0, lon_count = nLon;
  const size_t halo = 10;

  int use_regional_subset_i = 0;  // broadcastable
  
  double lon_min=-180.0, lon_max=180.0, lat_min=-90.0, lat_max=90.0;

  if (params_.lon_min.value() != boost::none) { lon_min = *params_.lon_min.value(); use_regional_subset_i = 1; }
  if (params_.lon_max.value() != boost::none) { lon_max = *params_.lon_max.value(); use_regional_subset_i = 1; }
  if (params_.lat_min.value() != boost::none) { lat_min = *params_.lat_min.value(); use_regional_subset_i = 1; }
  if (params_.lat_max.value() != boost::none) { lat_max = *params_.lat_max.value(); use_regional_subset_i = 1; }

  const bool fileNorthToSouth = (fileNorthToSouth_i != 0);

  const double lat_lo_user = std::min(lat_min, lat_max);
  const double lat_hi_user = std::max(lat_min, lat_max);
  const double lon_lo_user = std::min(lon_min, lon_max);
  const double lon_hi_user = std::max(lon_min, lon_max);

  auto bound_index = [](const std::vector<double> &a, double v, bool left) -> size_t {
    const size_t n = a.size();
    if (n == 0) return 0;
    if (left) {
      if (v <= a.front()) return 0;
      if (v >= a.back())  return n-1;
      auto it = std::lower_bound(a.begin(), a.end(), v);
      return static_cast<size_t>(std::distance(a.begin(), it));
    } else {
      if (v <= a.front()) return 0;
      if (v >= a.back())  return n-1;
      auto it = std::upper_bound(a.begin(), a.end(), v);
      if (it == a.begin()) return 0;
      --it;
      return static_cast<size_t>(std::distance(a.begin(), it));
    }
  };

  auto t_subset0 = clock_t::now();
  if (rank == 0 && use_regional_subset_i == 1) {
    double search_lon_min = lon_lo_user;
    double search_lon_max = lon_hi_user;

    // If file uses 0..360 and user gave -180..180, convert into 0..360
    if (!file_lons.empty() && file_lons.back() > 180.0) {
      oops::Log::info() << "User bounds: lon=[" << lon_lo_user << "," << lon_hi_user
                        << "] lat=[" << lat_lo_user << "," << lat_hi_user << "]" << std::endl;
      if (search_lon_min < 0.0) search_lon_min += 360.0;
      if (search_lon_max < 0.0) search_lon_max += 360.0;
      oops::Log::info() << "Converted to file lon=[0..360): lon=[" << search_lon_min
                        << "," << search_lon_max << "]" << std::endl;
      if (search_lon_max < search_lon_min) {
        ABORT("Dateline crossing subset is not supported by current subset logic");
      }
    }

    // Latitude search in ascending array
    std::vector<double> lats_asc = file_lats;
    if (fileNorthToSouth) std::reverse(lats_asc.begin(), lats_asc.end());

    size_t idx_lat_lo = bound_index(lats_asc, lat_lo_user, true);
    size_t idx_lat_hi = bound_index(lats_asc, lat_hi_user, false);
    if (idx_lat_lo > idx_lat_hi) std::swap(idx_lat_lo, idx_lat_hi);

    size_t idx_lon_lo = bound_index(file_lons, search_lon_min, true);
    size_t idx_lon_hi = bound_index(file_lons, search_lon_max, false);
    if (idx_lon_lo > idx_lon_hi) std::swap(idx_lon_lo, idx_lon_hi);

    // Expand by halo
    idx_lat_lo = (idx_lat_lo > halo) ? (idx_lat_lo - halo) : 0;
    idx_lat_hi = std::min(idx_lat_hi + halo, nLat - 1);
    idx_lon_lo = (idx_lon_lo > halo) ? (idx_lon_lo - halo) : 0;
    idx_lon_hi = std::min(idx_lon_hi + halo, nLon - 1);

    // Map back to file indexing if file lat is descending
    size_t lat_start_file = 0, lat_end_file = 0;
    if (fileNorthToSouth) {
      lat_start_file = (nLat - 1) - idx_lat_hi;
      lat_end_file   = (nLat - 1) - idx_lat_lo;
    } else {
      lat_start_file = idx_lat_lo;
      lat_end_file   = idx_lat_hi;
    }

    lat_start = lat_start_file;
    lat_count = lat_end_file - lat_start_file + 1;
    lon_start = idx_lon_lo;
    lon_count = idx_lon_hi - idx_lon_lo + 1;

    const size_t lat_end_print = lat_start + lat_count - 1;
    const size_t lon_end_print = lon_start + lon_count - 1;

    oops::Log::info()
      << "Regional subset indices: "
      << "  Latitude:  [" << lat_start << ":" << lat_end_print << "] = "
      << file_lats[lat_start] << " to " << file_lats[lat_end_print]
      << "  Longitude: [" << lon_start << ":" << lon_end_print << "] = " 
      << file_lons[lon_start] << " to " << file_lons[lon_end_print]
      << "  Data reduction: " << lat_count << " x " << lon_count
      << " (was " << nLat << " x " << nLon << ")" << std::endl;
  }
  log0t("[TIMER] subset compute: ", sec(t_subset0, clock_t::now()));

  geom_.getComm().broadcast(use_regional_subset_i, 0);
  const bool use_regional_subset = (use_regional_subset_i != 0);
  geom_.getComm().broadcast(lat_start, 0);
  geom_.getComm().broadcast(lat_count, 0);
  geom_.getComm().broadcast(lon_start, 0);
  geom_.getComm().broadcast(lon_count, 0);

  const size_t lat_end = lat_start + lat_count - 1;
  const size_t lon_end = lon_start + lon_count - 1;

  // ============================================================
  // 2) Pull State FieldSet and get model levels (81)
  // ============================================================
  atlas::FieldSet fieldsModelAll;
  x.toFieldSet(fieldsModelAll);

  if (!fieldsModelAll.has("air_temperature")) {
    ABORT("State missing air_temperature (needed to infer nLevModel)");
  }
  atlas::Field & tField = fieldsModelAll.field("air_temperature");
  const int nLevModel = (tField.rank() >= 2) ? static_cast<int>(tField.shape(1)) : 1;

  if (rank == 0) {
    oops::Log::info() << "[CHECK-NLEV] file pfull=" << nLevFile
                      << " model levels=" << nLevModel << std::endl;
  }
  if (nLevModel <= 1) ABORT("Model State appears not to have 3D levels (nLevModel<=1)");

  // ============================================================
  // 3) Load ak/bk once (cached member function you added)
  // ============================================================
  loadAkBkOnce_(nLevModel);  // fills ak_ and bk_ (size nLevModel+1)

  // ============================================================
  // 4) Variable mapping - read from configuration
  // ============================================================
  struct Var3D { std::string stateName; std::string fileName; };
  std::vector<Var3D> vars3d;
  
  // Get state variables from State object
  const auto& stateVars = x.variables();
  
  // Get field io names mapping from fileionames config (if provided)
  // NOTE: In many call sites, "fileionames" is already the mapping (model_name -> file_var_name),
  // not a parent node that contains a nested "field io names" key. Support both layouts.
  std::map<std::string, std::string> ioNameMap;
  if (fileionames.has("field io names")) {
    eckit::LocalConfiguration ionames = fileionames.getSubConfiguration("field io names");
    std::vector<std::string> keys = ionames.keys();
    for (const auto& key : keys) {
      ioNameMap[key] = ionames.getString(key);
    }
  } else {
    std::vector<std::string> keys = fileionames.keys();
    for (const auto& key : keys) {
      // Defensive: only pull string values; if a non-string sneaks in, skip it.
      try {
        ioNameMap[key] = fileionames.getString(key);
      } catch (...) {
        // ignore non-string entries
      }
    }
  }

  // Build vars3d list from state variables
  // Skip 2D variables (surface pressure will be handled separately)
  const std::vector<std::string> skip2D = {"air_pressure_at_surface", "surface_pressure"};
  
  for (size_t i = 0; i < stateVars.size(); ++i) {
    std::string varName = stateVars[i].name();
    
    // Skip 2D variables
    if (std::find(skip2D.begin(), skip2D.end(), varName) != skip2D.end()) {
      continue;
    }
    
    // Determine file name - use mapping if provided, otherwise use state name
    std::string fileName = varName;
    if (ioNameMap.count(varName) > 0) {
      fileName = ioNameMap[varName];
    }
    
    vars3d.push_back({varName, fileName});
  }
  
  if (rank == 0) {
    oops::Log::info() << "[I/O] Reading " << vars3d.size() << " 3D variables:" << std::endl;
    for (const auto& v : vars3d) {
      oops::Log::info() << "  " << v.stateName << " <- " << v.fileName << std::endl;
    }
  }

  // Verify all required variables exist in State
  for (const auto & v : vars3d) {
    if (!fieldsModelAll.has(v.stateName)) {
      if (rank == 0) oops::Log::error() << "State missing required field: " << v.stateName << std::endl;
      ABORT("State missing required 3D field");
    }
  }
  if (!fieldsModelAll.has("air_pressure_at_surface")) {
    ABORT("State missing air_pressure_at_surface");
  }

  const std::string dpresName = "__dpres__";
  const std::string psName    = "__pressfc__";

  // ============================================================
  // 5) Create source fields on structured FS (levels = nLevFile) and
  //    temporary target fields on model grid (levels = nLevFile)
  // ============================================================
  atlas::FieldSet srcFS;
  atlas::FieldSet tgtGlobal;

  auto make_model_tmp = [&](const std::string & name, int levels) -> atlas::Field {
    return geom_.functionSpace().createField<double>(
      atlas::option::name(name) | atlas::option::levels(levels));
  };

  for (const auto & v : vars3d) {
    srcFS.add(readFunctionSpace_->createField<double>(
      atlas::option::name(v.stateName) | atlas::option::levels(static_cast<int>(nLevFile))));
    tgtGlobal.add(make_model_tmp(v.stateName, static_cast<int>(nLevFile)));
  }

  // dpres (3D, nLevFile) and pressfc (2D)
  srcFS.add(readFunctionSpace_->createField<double>(
    atlas::option::name(dpresName) | atlas::option::levels(static_cast<int>(nLevFile))));
  tgtGlobal.add(make_model_tmp(dpresName, static_cast<int>(nLevFile)));

  srcFS.add(readFunctionSpace_->createField<double>(
    atlas::option::name(psName) | atlas::option::levels(1)));
  tgtGlobal.add(make_model_tmp(psName, 1));

  // required metadata for interpolator
  auto set_interp_type = [&](atlas::FieldSet & fs, const std::string & val) {
    for (auto & f : fs) {
      if (!f.metadata().has("interp_type")) f.metadata().set("interp_type", val);
    }
  };
  set_interp_type(srcFS, "default");
  set_interp_type(tgtGlobal, "default");

  // ============================================================
  // 6) Read slabs (rank0) + bcast + fill structured srcFS + haloExchange
  // ============================================================
  double t_read_rank0 = 0.0, t_bcast = 0.0, t_fill = 0.0;

  const auto gidxView  = atlas::array::make_view<atlas::gidx_t,1>(readFunctionSpace_->global_index());
  const auto ghostView = atlas::array::make_view<int,1>(readFunctionSpace_->ghost());

  // ============================================================
  // OPTIMIZED I/O v3: Parallel read, sequential broadcast
  // Read all variables in parallel, then broadcast one at a time
  // ============================================================
  const int n_ranks = geom_.getComm().size();
  const int n_reader_ranks = std::min(n_ranks, 16);
  
  if (rank == 0) {
    oops::Log::info() << "[I/O HYBRID] Using " << n_reader_ranks 
                      << " parallel readers with sequential broadcasts" << std::endl;
  }
  
  // Open NetCDF file for reader ranks
  int ncid = -1;
  const bool i_am_reader = (rank < n_reader_ranks);
  
  if (i_am_reader) {
    nc_rc(nc_open(inFile.c_str(), NC_NOWRITE, &ncid), "nc_open " + inFile);
  }
  
  // Timing
  double t_read_total = 0.0;
  double t_bcast_total = 0.0;
  double t_fill_total = 0.0;
  
  // Structure to hold variable info
  struct VarInfo {
    std::string fileName;
    std::string stateName;
    int nLevels;
    bool is3D;
  };
  
  // Build list of all variables to read
  std::vector<VarInfo> all_vars;
  for (const auto & v : vars3d) {
    all_vars.push_back({v.fileName, v.stateName, static_cast<int>(nLevFile), true});
  }
  all_vars.push_back({"dpres", dpresName, static_cast<int>(nLevFile), true});
  all_vars.push_back({"pressfc", psName, 1, false});
  
  // PHASE 1: All readers read their assigned variables in PARALLEL
  auto t_read_start = clock_t::now();
  
  std::map<int, std::vector<float>> my_data;  // var_idx -> data
  
  for (int v = 0; v < static_cast<int>(all_vars.size()); ++v) {
    int reader_rank = v % n_reader_ranks;
    
    if (rank == reader_rank) {
      const auto & var = all_vars[v];
      const size_t plane = lat_count * lon_count;
      const size_t slabSize = plane * (var.is3D ? static_cast<size_t>(var.nLevels) : 1);
      
      std::vector<float> data(slabSize);
      
      int varid;
      nc_rc(nc_inq_varid(ncid, var.fileName.c_str(), &varid), 
            "nc_inq_varid " + var.fileName);
      
      if (var.is3D) {
        size_t start[4] = {0, 0, lat_start, lon_start};
        size_t count[4] = {1, static_cast<size_t>(var.nLevels), lat_count, lon_count};
        nc_rc(nc_get_vara_float(ncid, varid, start, count, data.data()),
              "nc_get_vara_float " + var.fileName);
      } else {
        size_t start[3] = {0, lat_start, lon_start};
        size_t count[3] = {1, lat_count, lon_count};
        nc_rc(nc_get_vara_float(ncid, varid, start, count, data.data()),
              "nc_get_vara_float " + var.fileName);
      }
      
      my_data[v] = std::move(data);
      
      const double mb = (double(slabSize) * sizeof(float)) / (1024.0*1024.0);
      oops::Log::info() << "[I/O rank " << rank << "] Read " << var.fileName 
                        << " (" << mb << " MB)" << std::endl;
    }
  }
  
  t_read_total = sec(t_read_start, clock_t::now());
  
  if (ncid >= 0) {
    nc_rc(nc_close(ncid), "nc_close " + inFile);
  }
  
  // PHASE 2: Broadcast variables SEQUENTIALLY (one at a time)
  auto t_bcast_start = clock_t::now();
  
  for (int v = 0; v < static_cast<int>(all_vars.size()); ++v) {
    const auto & var = all_vars[v];
    int reader_rank = v % n_reader_ranks;
    
    const size_t plane = lat_count * lon_count;
    const size_t slabSize = plane * (var.is3D ? static_cast<size_t>(var.nLevels) : 1);
    
    std::vector<float> data(slabSize);
    
    // Reader copies from its stored data
    if (rank == reader_rank) {
      data = my_data[v];
    }
    
    // Sequential broadcast (one at a time)
    geom_.getComm().broadcast(data.begin(), data.end(), reader_rank);
    
    // PHASE 3: All ranks fill field immediately after receiving
    auto t_fill_start = clock_t::now();
    
    atlas::Field * field_ptr = nullptr;
    if (var.is3D) {
      field_ptr = &srcFS.field(var.stateName);
    } else {
      field_ptr = &srcFS.field(var.stateName);
    }
    
    auto v_view = atlas::array::make_view<double, 2>(*field_ptr);
    const int nLev = var.nLevels;
    
    // Init to NaN
    for (atlas::idx_t p = 0; p < v_view.shape(0); ++p)
      for (int k = 0; k < nLev; ++k)
        v_view(p, k) = nan;
    
    // Fill owned points
    for (atlas::idx_t p = 0; p < v_view.shape(0); ++p) {
      if (ghostView(p) != 0) continue;
      const auto gidx = gidxView(p);
      if (gidx <= 0) continue;
      
      const std::size_t g = static_cast<std::size_t>(gidx) - 1;
      const std::size_t i = g % nLon;
      const std::size_t j = g / nLon;
      if (i >= nLon || j >= nLat) continue;
      
      if (use_regional_subset) {
        if (i < lon_start || i > lon_end || j < lat_start || j > lat_end) continue;
      }
      
      const std::size_t ii = i - lon_start;
      const std::size_t jj = j - lat_start;
      
      if (var.is3D) {
        for (int k = 0; k < nLev; ++k) {
          const size_t idx = (static_cast<size_t>(k) * lat_count + jj) * lon_count + ii;
          v_view(p, k) = static_cast<double>(data[idx]);
        }
      } else {
        const size_t idx = jj * lon_count + ii;
        v_view(p, 0) = static_cast<double>(data[idx]);
      }
    }
    
    readFunctionSpace_->haloExchange(*field_ptr);
    t_fill_total += sec(t_fill_start, clock_t::now());
  }
  
  t_bcast_total = sec(t_bcast_start, clock_t::now());
  
  // Report timing
  double t_read_max = t_read_total;
  geom_.getComm().allReduceInPlace(t_read_max, eckit::mpi::Operation::MAX);
  
  log0t("[TIMER HYBRID I/O] NetCDF read (parallel, max): ", t_read_max);
  log0t("[TIMER HYBRID I/O] Broadcast (sequential, total): ", t_bcast_total);
  log0t("[TIMER HYBRID I/O] Fill fields (total): ", t_fill_total);

  // ============================================================
  // 7) Horizontal interpolation: structured(Global) -> model(Global)
  // ============================================================
  for (auto & f : tgtGlobal) {
    auto v = atlas::array::make_view<double, 2>(f);
    for (atlas::idx_t p = 0; p < v.shape(0); ++p)
      for (int k = 0; k < v.shape(1); ++k)
        v(p, k) = nan;
  }

  auto t_h0 = clock_t::now();
  interpolatorBack_->apply(srcFS, tgtGlobal);
  log0t("[TIMER] horizontal interp (file->model @file-levels): ", sec(t_h0, clock_t::now()));

  // views needed for vertical work
  auto dpresV = atlas::array::make_view<double, 2>(tgtGlobal.field(dpresName)); // (nodes,nLevFile)
  auto psV    = atlas::array::make_view<double, 2>(tgtGlobal.field(psName));    // (nodes,1)

  // model ps field (State)
  auto psState = atlas::array::make_view<double, 2>(fieldsModelAll.field("air_pressure_at_surface"));

  // ============================================================
  // 8) Vertical remap helper: log-pressure linear interpolation
  //    p_src must be strictly increasing (top->bottom).
  // ============================================================
  auto interp_logp = [&](const std::vector<double> & p_src,
                         const std::vector<double> & x_src,
                         double p_tgt) -> double {
    const size_t n = p_src.size();
    if (n < 2) return x_src.empty() ? nan : x_src.front();

    if (p_tgt <= p_src.front()) return x_src.front();
    if (p_tgt >= p_src.back())  return x_src.back();

    auto it = std::upper_bound(p_src.begin(), p_src.end(), p_tgt);
    size_t k1 = std::max<size_t>(1, static_cast<size_t>(it - p_src.begin())) - 1;
    size_t k2 = k1 + 1;

    const double p1 = std::max(1.0, p_src[k1]);
    const double p2 = std::max(1.0, p_src[k2]);
    const double x1 = x_src[k1];
    const double x2 = x_src[k2];

    const double w = (std::log(std::max(1.0, p_tgt)) - std::log(p1)) /
                     (std::log(p2) - std::log(p1));
    return x1 + w * (x2 - x1);
  };

  // ============================================================
  // 9) Vertical remap Global->81 for each 3D variable
  // ============================================================
  auto t_v0 = clock_t::now();

  // pressure work buffers (allocated once, reused)
  std::vector<double> p_int_src(nLevFile + 1);
  std::vector<double> p_mid_src(nLevFile);
  std::vector<double> p_int_tgt(nLevModel + 1);
  std::vector<double> p_mid_tgt(nLevModel);
  std::vector<double> x_src(nLevFile);

  for (const auto & vinfo : vars3d) {
    auto srcVar = atlas::array::make_view<double, 2>(tgtGlobal.field(vinfo.stateName));             // (nodes,nLevFile)
    auto dstVar = atlas::array::make_view<double, 2>(fieldsModelAll.field(vinfo.stateName));    // (nodes,nLevModel)

    const atlas::idx_t npts = dstVar.shape(0);

    for (atlas::idx_t p = 0; p < npts; ++p) {
      // ---- source interface/mid pressures from dpres + ps ----
      p_int_src[0] = 0.0;
      for (size_t k = 0; k < nLevFile; ++k) {
        double dp = dpresV(p, static_cast<int>(k));
        if (!std::isfinite(dp) || dp < 0.0) dp = 0.0;
        p_int_src[k + 1] = p_int_src[k] + dp;
      }

      double ps_src = psV(p, 0);
      if (!std::isfinite(ps_src) || ps_src <= 0.0) ps_src = p_int_src[nLevFile];

      const double sumdp = p_int_src[nLevFile];
      const double scale = (sumdp > 0.0 && ps_src > 0.0) ? (ps_src / sumdp) : 1.0;

      for (size_t k = 0; k <= nLevFile; ++k) p_int_src[k] *= scale;

      for (size_t k = 0; k < nLevFile; ++k) {
        p_mid_src[k] = 0.5 * (p_int_src[k] + p_int_src[k + 1]);
        if (p_mid_src[k] < 1.0) p_mid_src[k] = 1.0;
      }

      // ---- target interface/mid pressures from ak/bk + ps_tgt ----
      double ps_tgt = psV(p, 0);  // prefer interpolated global pressfc
      if (!std::isfinite(ps_tgt) || ps_tgt <= 0.0) ps_tgt = psState(p, 0); // fallback to State ps
      if (!std::isfinite(ps_tgt) || ps_tgt <= 0.0) ps_tgt = ps_src;        // last fallback

      for (int k = 0; k <= nLevModel; ++k) {
        p_int_tgt[k] = ak_[k] + bk_[k] * ps_tgt;
        if (p_int_tgt[k] < 1.0) p_int_tgt[k] = 1.0;
      }
      for (int k = 0; k < nLevModel; ++k) {
        p_mid_tgt[k] = 0.5 * (p_int_tgt[k] + p_int_tgt[k + 1]);
        if (p_mid_tgt[k] < 1.0) p_mid_tgt[k] = 1.0;
      }

      // ---- source profile values at this point ----
      for (size_t k = 0; k < nLevFile; ++k) {
        x_src[k] = srcVar(p, static_cast<int>(k));
      }

      // ---- interpolate to model levels ----
      for (int k = 0; k < nLevModel; ++k) {
        dstVar(p, k) = interp_logp(p_mid_src, x_src, p_mid_tgt[k]);
      }
    }
  }

  log0t("[TIMER] vertical remap (file-levels->model-levels): ", sec(t_v0, clock_t::now()));

  // ============================================================
  // 10) Update State surface pressure with interpolated pressfc (optional but consistent)
  // ============================================================
  for (atlas::idx_t p = 0; p < psState.shape(0); ++p) {
    const double v = psV(p, 0);
    if (std::isfinite(v) && v > 0.0) psState(p, 0) = v;
  }

  // ============================================================
  // 11) Copy back to State
  // ============================================================
  auto t_from0 = clock_t::now();
  x.fromFieldSet(fieldsModelAll);
  log0t("[TIMER] fromFieldSet: ", sec(t_from0, clock_t::now()));

  log0t("[TIMER] TOTAL read(): ", sec(t_total0, clock_t::now()));
}
// // XL
// -------------------------------------------------------------------------------------------------
void IOStructuredGrid::read(Increment & dx, const eckit::LocalConfiguration & fileionames,
                                const eckit::LocalConfiguration & fileioscaling) const {
  ABORT("IOStructuredGrid::read(Increment) not implemented");
}

// -------------------------------------------------------------------------------------------------

template <typename T>
void IOStructuredGrid::interpAndWrite(const T & obj, const std::string & label,
                                      const eckit::LocalConfiguration & fileionames,
                                      const eckit::LocalConfiguration & fileioscaling) const {
  util::Timer timer(classname(), "write " + label);
  oops::Log::trace() << classname() << " write " << label << " starting" << std::endl;

  // Create field sets
  atlas::FieldSet fieldsCubeSphere;
  atlas::FieldSet fieldsGeographic;
  obj.toFieldSet(fieldsCubeSphere);

  // Apply interpolation
  interpolator_->apply(fieldsCubeSphere, fieldsGeographic);

  // Write to disk if rank 0
  if (geom_.getComm().rank() == 0) {
    this->writeStructuredFields(fieldsGeographic, obj.validTime(), fileionames, fileioscaling);
  }

  oops::Log::trace() << classname() << " write " << label << " done" << std::endl;
//  geom_.getComm().broadcast(fieldsStructured, 0);
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::write(const State & x, const eckit::LocalConfiguration & fileionames,
                                const eckit::LocalConfiguration & fileioscaling) const {
  this->interpAndWrite(x, "state", fileionames, fileioscaling);
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::write(const Increment & dx, const eckit::LocalConfiguration & fileionames,
                             const eckit::LocalConfiguration & fileioscaling) const {
  this->interpAndWrite(dx, "increment", fileionames, fileioscaling);
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::writeStructuredFields(const atlas::FieldSet & fields,
                                             const util::DateTime & time,
                                             const eckit::LocalConfiguration & ioNames,
                                             const eckit::LocalConfiguration & ioScaling) const {
  // NetCDF IDs
  // ----------
  int fileId;

  // Dimension indices
  int latId;
  int lonId;
  int levId;
  int edgId;
  int forId;
  int timId;

  // Variable indices
  int fIv;
  std::map<std::string, int> fieldIvs;

  // Get ak/bk for writing
  // ---------------------
  std::vector<double> ak = geom_.ak();
  std::vector<double> bk = geom_.bk();

  // Get the name of the file and adjust with datetime
  // -------------------------------------------------
  std::string pathFile = params_.filename.value();

  // For backward compatibility add some things to the filename if not already present
  if (pathFile.find("%Y") == std::string::npos) {
    pathFile += "%Y%m%d_%H%M%Sz";
  }
  if (pathFile.find(".nc") == std::string::npos) {
    pathFile += ".nc4";
  }

  // Format the datetime string
  pathFile = time.formatString(pathFile);

  // Replace member number (ensemble applciaitons)
  util::stringfunctions::swapNameMember(params_.toConfiguration(), pathFile);

  // Create a file to write fields into
  // ----------------------------------
  nc_rc(nc_create(pathFile.c_str(), NC_CLOBBER | NC_NETCDF4, &fileId), "nc_create" + pathFile);

  // Create regular grid for determining lat/lon values
  // --------------------------------------------------
  const atlas::RegularGrid regGrid(writeFunctionSpace_->grid());

  // Define the dimensions in the file
  // ---------------------------------
  const int nLat = regGrid.ny();
  const int nLon = regGrid.nx();
  const int nLev = geom_.npz();
  const int nEdg = geom_.npz() + 1;
  const int nFor = 4;
  const int nTim = 1;

  nc_rc(nc_def_dim(fileId, params_.latName.value().c_str(), nLat, &latId), "nc_def_dim (lat)");
  nc_rc(nc_def_dim(fileId, params_.lonName.value().c_str(), nLon, &lonId), "nc_def_dim (lon)");
  nc_rc(nc_def_dim(fileId, params_.levName.value().c_str(), nLev, &levId), "nc_def_dim (lev)");
  nc_rc(nc_def_dim(fileId, params_.edgName.value().c_str(), nEdg, &edgId), "nc_def_dim (edg)");
  nc_rc(nc_def_dim(fileId, params_.forName.value().c_str(), nFor, &forId), "nc_def_dim (for)");
  nc_rc(nc_def_dim(fileId, params_.timName.value().c_str(), nTim, &timId), "nc_def_dim (tim)");

  // Define the dimensions variables in the file
  // -------------------------------------------
  std::vector<double> latArr(nLat);
  std::vector<double> lonArr(nLon);
  std::vector<int> levArr(nLev);
  std::vector<int> edgArr(nEdg);
  std::vector<int> forArr(nFor);
  std::vector<int> timArr(nTim);

  for (int i = 0; i < nLat; ++i) {
    latArr[i] = regGrid.y(nLat - 1 - i);
  }
  for (int i = 0; i < nLon; ++i) {
    lonArr[i] = regGrid.x(i);
  }
  for (int i = 0; i < nLev; ++i) {
    levArr[i] = i + 1;
  }
  for (int i = 0; i < nEdg; ++i) {
    edgArr[i] = i + 1;
  }
  for (int i = 0; i < nFor; ++i) {
    forArr[i] = i + 1;
  }
  for (int i = 0; i < nTim; ++i) {
    timArr[i] = i + 1;
  }

  // Write the dimension variables (and attributes) to the file
  // ----------------------------------------------------------
  nc_rc(nc_def_var(fileId, params_.latName.value().c_str(), NC_DOUBLE, 1, &latId, &fIv),
        "nc_def_var (lat)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("degrees_north"), "degrees_north"),
        "nc_put_att_text (lat)");
  fieldIvs[params_.latName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.lonName.value().c_str(), NC_DOUBLE, 1, &lonId, &fIv),
        "nc_def_var (lon)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("degrees_east"), "degrees_east"),
        "nc_put_att_text (lon)");
  fieldIvs[params_.lonName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.levName.value().c_str(), NC_INT, 1, &levId, &fIv),
        "nc_def_var (lev)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"),
        "nc_put_att_text (lev)");
  fieldIvs[params_.levName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.edgName.value().c_str(), NC_INT, 1, &edgId, &fIv),
        "nc_def_var (edg)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"),
        "nc_put_att_text (edg)");
  fieldIvs[params_.edgName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.forName.value().c_str(), NC_INT, 1, &forId, &fIv),
        "nc_def_var (for)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"),
        "nc_put_att_text (for)");
  fieldIvs[params_.forName.value()] = fIv;

  nc_rc(nc_def_var(fileId, params_.timName.value().c_str(), NC_INT, 1, &timId, &fIv),
        "nc_def_var (tim)");
  nc_rc(nc_put_att_text(fileId, fIv, "units", strlen("1"), "1"),
        "nc_put_att_text (tim)");
  fieldIvs[params_.timName.value()] = fIv;

  // Define some categories of dimension IDs for fields
  // --------------------------------------------------
  std::map<int, std::vector<int>> fieldDims;
  fieldDims[nLev] = {timId, levId, latId, lonId};  // Fields at levels
  fieldDims[nEdg] = {timId, edgId, latId, lonId};  // Fields at edges
  fieldDims[4] = {timId, forId, latId, lonId};     // Fields at four levels
  fieldDims[1] = {timId, latId, lonId};            // Fields at surface
  fieldDims[0] = {timId, latId, lonId};            // Fields at surface

  // Set float precision for fields
  // ------------------------------
  const int floatPrecision = params_.floatPrecision.value();
  const int ncPrec = (floatPrecision == 4) ? NC_FLOAT : NC_DOUBLE;

  // Define all the fields that will be written
  // ------------------------------------------
  for (auto& field : fields) {
    // Get number of levels for this field
    const int nLevField = field.shape(1);

    // Get dimensions for this field from map
    auto it = fieldDims.find(nLevField);
    if (it == fieldDims.end()) {
      std::ostringstream oss;
      oss << "IOStructuredGrid::writeStructuredFields: "
          << "No entry in fieldDims for field '" << field.name()
          << "' with " << nLevField << " levels.";
      ABORT(oss.str());
    }
    const auto &dims = it->second;

    // Look for fieldname in the iofile configuration and use the value if key found
    const std::string fieldLong = field.name();
    const char * fieldLongC = fieldLong.c_str();

    // Get the fieldmetadata for this field
    const FieldMetadata & fieldMetadata = geom_.fieldsMetaData().getFieldMetadata(fieldLong);
    std::string unitsStr = fieldMetadata.getVarUnits();
    const char * units = unitsStr.c_str();

    std::string fieldName = fieldLong;
    if (ioNames.has(fieldName)) {
      fieldName = ioNames.getString(fieldLong);
    }

    // Define the field in the file
    nc_rc(nc_def_var(fileId, fieldName.c_str(), ncPrec, dims.size(), dims.data(), &fIv),
          "nc_def_var " + fieldName);
    nc_rc(nc_put_att_text(fileId, fIv, "units", strlen(units), units),
          "nc_put_att_text " + fieldName + " units");
    nc_rc(nc_put_att_text(fileId, fIv, "long_name", strlen(fieldLongC), fieldLongC),
          "nc_put_att_text " + fieldName + " long_name");

    // Insert field into the fieldIvs map
    fieldIvs[field.name()] = fIv;
  }

  // Write ak/bk to the file as global attributes
  // --------------------------------------------
  nc_rc(nc_put_att_double(fileId, NC_GLOBAL, "ak", NC_DOUBLE, ak.size(), ak.data()),
          "nc_put_att_double (ak)");
  nc_rc(nc_put_att_double(fileId, NC_GLOBAL, "bk", NC_DOUBLE, bk.size(), bk.data()),
          "nc_put_att_double (bk)");
  nc_rc(nc_put_att_text(fileId, NC_GLOBAL, "grid", strlen(gridStr_.c_str()), gridStr_.c_str()),
          "nc_put_att_text (grid)");
  nc_rc(nc_put_att_int(fileId, NC_GLOBAL, "im", NC_INT, 1, &nLon),
          "nc_put_att_int (im)");
  nc_rc(nc_put_att_int(fileId, NC_GLOBAL, "jm", NC_INT, 1, &nLat),
          "nc_put_att_int (im)");

  // End definition mode
  // -------------------
  nc_rc(nc_enddef(fileId), "nc_enddef");

  // Write coordinate data into the file
  // -----------------------------------
  nc_rc(nc_put_var_double(fileId, fieldIvs[params_.latName.value()], latArr.data()),
        "nc_put_var_double (lat)");
  nc_rc(nc_put_var_double(fileId, fieldIvs[params_.lonName.value()], lonArr.data()),
        "nc_put_var_double (lon)");
  nc_rc(nc_put_var_int(fileId, fieldIvs[params_.levName.value()], levArr.data()),
        "nc_put_var_int (lev)");
  nc_rc(nc_put_var_int(fileId, fieldIvs[params_.edgName.value()], edgArr.data()),
        "nc_put_var_int (edg)");
  nc_rc(nc_put_var_int(fileId, fieldIvs[params_.timName.value()], timArr.data()),
        "nc_put_var_int (tim)");

  // Write the fields into the file
  // ------------------------------
  for (auto& field : fields) {
    // Get number of levels for this field
    const int nLevField = field.shape(1);

    // Create a rank 2 view of the field
    const auto fieldView = atlas::array::make_view<double, 2>(field);

    // Vector to hold the packed field
    std::vector<double> values(nLat*nLon*nLevField);

    // Loop over dimensions and pack the field
    for (size_t k = 0; k < nLevField; ++k) {
      for (size_t j = 0; j < nLat; ++j) {
        for (size_t i = 0; i < nLon; ++i) {
          values[k*nLat*nLon + j*nLon + i] = fieldView((nLat - 1 - j) * nLon + i, k);
        }
      }
    }

    // Write the field to the file
    nc_rc(nc_put_var_double(fileId, fieldIvs[field.name()], values.data()),
          "nc_put_var_double " + field.name());
  }

  // Close netCDF file
  // -----------------
  nc_rc(nc_close(fileId), "nc_close");
}

// -------------------------------------------------------------------------------------------------

void IOStructuredGrid::print(std::ostream & os) const {
  os << classname() << " IO using Atlas Structured Grid";
}

// -------------------------------------------------------------------------------------------------

}  // namespace fv3jedi
