#ifndef MIRENA_CONTROL__MPC_CONTROLLER_HPP_
#define MIRENA_CONTROL__MPC_CONTROLLER_HPP_

#include <array>
#include <cstddef>
#include <string>

extern "C" {
#include "acados_solver_mpc_problem.h"
}

namespace mirena
{

/// Thin owner of the generated acados solver.
///
/// The horizon buffers live inside the controller and are handed to acados
/// row by row, so warm starting and reading the solution back involve no
/// allocation and no intermediate copies.  A failed solve leaves the previous
/// horizon in place, which is what the next warm start wants.
class MpcController
{
public:
  static constexpr int kNx = MPC_PROBLEM_NX;                 // 10
  static constexpr int kNu = MPC_PROBLEM_NU;                 // 3
  static constexpr int kN = MPC_PROBLEM_N;                   // shooting intervals
  static constexpr int kNpGlobal = MPC_PROBLEM_NP_GLOBAL;    // 2 * path points + 1
  static constexpr int kPathPoints = (kNpGlobal - 1) / 2;

  /// Row-major horizons; each row is contiguous and passed straight to acados.
  using StateHorizon = std::array<std::array<double, kNx>, kN + 1>;
  using InputHorizon = std::array<std::array<double, kNu>, kN>;

  struct Stats
  {
    double solve_time_ms;
    double max_residual;
    int sqp_iterations;
  };

  MpcController();
  ~MpcController();

  MpcController(const MpcController &) = delete;
  MpcController & operator=(const MpcController &) = delete;

  /// Length of one shooting interval, read back from the generated solver.
  double stepDuration() const noexcept {return step_duration_;}

  /// Warm start from the stored horizon, solve, and on success overwrite it.
  /// Returns the acados status (0 == converged).
  int solve(const double (& x0)[kNx], const double (& p_global)[kNpGlobal]);

  /// Discard the internal solver state after a failure.
  void reset();

  Stats stats() const;

  const StateHorizon & states() const noexcept {return states_;}
  const InputHorizon & inputs() const noexcept {return inputs_;}

private:
  mpc_problem_solver_capsule * capsule_ = nullptr;
  ocp_nlp_config * config_ = nullptr;
  ocp_nlp_dims * dims_ = nullptr;
  ocp_nlp_in * in_ = nullptr;
  ocp_nlp_out * out_ = nullptr;
  ocp_nlp_solver * solver_ = nullptr;

  StateHorizon states_{};
  InputHorizon inputs_{};
  double step_duration_ = 0.0;
};

}  // namespace mirena

#endif  // MIRENA_CONTROL__MPC_CONTROLLER_HPP_
