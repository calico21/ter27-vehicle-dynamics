#include "mirena_control/mpc_controller.hpp"

#include <algorithm>
#include <stdexcept>

namespace mirena
{

namespace
{
// acados takes non-const char* for field names in a few places.
char * field(const char * name)
{
  return const_cast<char *>(name);
}
}  // namespace

MpcController::MpcController()
{
  capsule_ = mpc_problem_acados_create_capsule();
  if (capsule_ == nullptr) {
    throw std::runtime_error("acados: failed to allocate solver capsule");
  }

  const int status = mpc_problem_acados_create(capsule_);
  if (status != 0) {
    mpc_problem_acados_free_capsule(capsule_);
    capsule_ = nullptr;
    throw std::runtime_error(
            "acados: solver creation failed with status " + std::to_string(status));
  }

  config_ = mpc_problem_acados_get_nlp_config(capsule_);
  dims_ = mpc_problem_acados_get_nlp_dims(capsule_);
  in_ = mpc_problem_acados_get_nlp_in(capsule_);
  out_ = mpc_problem_acados_get_nlp_out(capsule_);
  solver_ = mpc_problem_acados_get_nlp_solver(capsule_);

  double ts = 0.0;
  ocp_nlp_in_get(config_, dims_, in_, 0, field("Ts"), &ts);
  step_duration_ = ts;
}

MpcController::~MpcController()
{
  if (capsule_ != nullptr) {
    mpc_problem_acados_free(capsule_);
    mpc_problem_acados_free_capsule(capsule_);
  }
}

int MpcController::solve(const double (& x0)[kNx], const double (& p_global)[kNpGlobal])
{
  // Pin stage 0 to the measured state.
  double * x0_ptr = const_cast<double *>(x0);
  ocp_nlp_constraints_model_set(config_, dims_, in_, out_, 0, field("lbx"), x0_ptr);
  ocp_nlp_constraints_model_set(config_, dims_, in_, out_, 0, field("ubx"), x0_ptr);

  // Warm start from the previous solution; acados shifts internally.
  for (int i = 0; i < kN; ++i) {
    ocp_nlp_out_set(config_, dims_, out_, in_, i, field("x"), states_[i].data());
    ocp_nlp_out_set(config_, dims_, out_, in_, i, field("u"), inputs_[i].data());
  }

  mpc_problem_acados_set_p_global_and_precompute_dependencies(
    capsule_, const_cast<double *>(p_global), kNpGlobal);

  const int status = mpc_problem_acados_solve(capsule_);

  if (status == ACADOS_SUCCESS) {
    for (int i = 0; i <= kN; ++i) {
      ocp_nlp_out_get(config_, dims_, out_, i, field("x"), states_[i].data());
    }
    for (int i = 0; i < kN; ++i) {
      ocp_nlp_out_get(config_, dims_, out_, i, field("u"), inputs_[i].data());
    }
  }

  return status;
}

void MpcController::reset()
{
  mpc_problem_acados_reset(capsule_, 1);
}

MpcController::Stats MpcController::stats() const
{
  Stats s{0.0, 0.0, 0};

  double time_tot = 0.0;
  ocp_nlp_get(solver_, "time_tot", &time_tot);
  s.solve_time_ms = time_tot * 1e3;

  int sqp_iter = 0;
  ocp_nlp_get(solver_, "sqp_iter", &sqp_iter);
  s.sqp_iterations = sqp_iter;

  // The Python binding's "residuals" is a composite; the C API only exposes the
  // four norms individually, and asking it for an unknown field calls exit().
  double residuals[4] = {0.0, 0.0, 0.0, 0.0};
  ocp_nlp_get(solver_, "res_stat", &residuals[0]);
  ocp_nlp_get(solver_, "res_eq", &residuals[1]);
  ocp_nlp_get(solver_, "res_ineq", &residuals[2]);
  ocp_nlp_get(solver_, "res_comp", &residuals[3]);
  s.max_residual = *std::max_element(residuals, residuals + 4);

  return s;
}

}  // namespace mirena
