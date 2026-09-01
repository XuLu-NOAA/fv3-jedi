/*
 * (C) Copyright 2017-2020 UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include <filesystem>
#include <string>
#include <vector>

#include "eckit/config/LocalConfiguration.h"
#include "eckit/config/YAMLConfiguration.h"
#include "eckit/exception/Exceptions.h"
#include "eckit/filesystem/PathName.h"
#include "eckit/mpi/Comm.h"

#include "fv3jedi/Utilities/Traits.h"

#include "oops/mpi/mpi.h"
#include "oops/runs/Application.h"
#include "oops/runs/ConvertState.h"
#include "oops/runs/Run.h"
#include "oops/util/Logger.h"

namespace fv3jedi {

// -----------------------------------------------------------------------------
/// FV3-JEDI-specific ensemble ConvertState driver.
///
/// This intentionally mimics oops::EnsembleApplication for the member splitting,
/// but adds one FV3-JEDI-specific behavior:
///
///   If a member YAML contains:
///
///     working directory: ens_member_cwd/mem001
///
///   then all MPI ranks for that member chdir into that directory before
///   constructing/running ConvertState.
///
/// This avoids FMS/MPP runtime-file collisions such as warnfile.000000.out
/// when many FV3-JEDI members run inside one MPI executable.
// -----------------------------------------------------------------------------

class EnsConvertState : public oops::Application {
 public:
  explicit EnsConvertState(const eckit::mpi::Comm & comm = oops::mpi::world())
    : Application(comm) {}

  ~EnsConvertState() override {}

  int execute(const eckit::Configuration & fullConfig) const override {
    if (fullConfig.has("files") && fullConfig.has("members")) {
      throw eckit::BadParameter("EnsConvertState: 'files' and 'members' "
                                "cannot be specified at the same time", Here());
    }

    // Read member configurations.
    std::vector<eckit::LocalConfiguration> memberConfigs;

    if (fullConfig.has("files")) {
      for (const std::string & file : fullConfig.getStringVector("files")) {
        const eckit::PathName yamlPathFile(file);
        const eckit::YAMLConfiguration memberConfig(yamlPathFile);
        memberConfigs.push_back(eckit::LocalConfiguration(memberConfig));
      }
    } else if (fullConfig.has("members")) {
      memberConfigs = fullConfig.getSubConfigurations("members");
    } else {
      throw eckit::BadParameter("EnsConvertState: either 'files' or 'members' "
                                "must be specified in the configuration", Here());
    }

    const int nmembers = memberConfigs.size();
    const int ntasks = this->getComm().size();
    const int mytask = this->getComm().rank();

    if (nmembers <= 0) {
      throw eckit::BadParameter("EnsConvertState: no ensemble members found", Here());
    }

    if (ntasks % nmembers != 0) {
      throw eckit::BadParameter("EnsConvertState: total MPI task count must be "
                                "divisible by the number of members", Here());
    }

    const int tasksPerMember = ntasks / nmembers;
    const int mymember = mytask / tasksPerMember + 1;

    oops::Log::info() << "Running " << nmembers
                      << " FV3-JEDI ensemble ConvertState members handled by "
                      << ntasks << " total MPI tasks and "
                      << tasksPerMember << " MPI tasks per member."
                      << std::endl;

    // Create one communicator per member.
    const std::string commNameStr = "comm_member_" + std::to_string(mymember);
    eckit::mpi::Comm & commMember =
      this->getComm().split(mymember, commNameStr.c_str());

    const eckit::LocalConfiguration & memberConfig = memberConfigs[mymember - 1];

    // Save original cwd. This should be the common DATA directory when launched.
    const std::filesystem::path originalCwd = std::filesystem::current_path();

    std::filesystem::path memberCwd;
    bool useMemberCwd = false;

    if (memberConfig.has("working directory")) {
      const std::string cwdString = memberConfig.getString("working directory");

      if (!cwdString.empty()) {
        memberCwd = std::filesystem::path(cwdString);

        // Relative working directories are relative to the launch directory,
        // not to the member YAML location.
        if (memberCwd.is_relative()) {
          memberCwd = originalCwd / memberCwd;
        }

        useMemberCwd = true;
      }
    }

    if (useMemberCwd) {
      // Rank 0 of each member creates the directory. Then all local member ranks
      // enter it before FV3/FMS Geometry is constructed.
      if (commMember.rank() == 0) {
        std::filesystem::create_directories(memberCwd);
      }
      commMember.barrier();

      std::filesystem::current_path(memberCwd);

      if (commMember.rank() == 0) {
        oops::Log::info() << "FV3-JEDI EnsConvertState member " << mymember
                          << " using working directory: "
                          << std::filesystem::current_path().string()
                          << std::endl;
      }
    }

    // Run normal ConvertState inside the member communicator.
    oops::ConvertState<fv3jedi::Traits> app(commMember);
    const int rc = app.execute(memberConfig);

    // Avoid restoring cwd while other ranks in the member are still finishing.
    commMember.barrier();

    if (useMemberCwd) {
      std::filesystem::current_path(originalCwd);
    }

    return rc;
  }

 private:
  std::string appname() const override {
    return "fv3jedi::EnsConvertState";
  }
};

// -----------------------------------------------------------------------------

}  // namespace fv3jedi

// -----------------------------------------------------------------------------

int main(int argc, char ** argv) {
  oops::Run run(argc, argv);
  fv3jedi::EnsConvertState app;
  return run.execute(app);
}
