import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ngclearn import Context, MethodProcess
from ngclearn.components.neurons.spiking import RAFCell, RAFSRM
output_dir = "exp/spikeResponseModel/rafSRM"

def run_rafsrm_visualization(fname="raf_srm_sparse_point_queries.png"):
    """
    This routine runs the RAF-SRM versus RAF ODE model experiment, 
    tracking discrete points where the SRM is called as the ODE model 
    is run iteratively/continually.
    """
    ## sim-time config
    dt = 0.01
    total_time = 5.5
    time_steps = int(total_time / dt)
    n_units = 1
    # hyper-param config
    tau_v_param = 2.0
    tau_w_param = 2.0
    omega_param = 8.0
    dampen_param = -0.4
    thr_param = 1.0
    v_reset_param = -1.0
    w_reset_param = 0.0

    ## set up model context, pinning both RAF and RAF-SRM
    with Context("RAF_SRM_Context") as ctx:
        srm_pop = RAFSRM(
            "srm", n_units=n_units, tau_v=tau_v_param, tau_w=tau_w_param,
            omega=omega_param, dampen_factor=dampen_param, thr=thr_param,
            v_reset=v_reset_param, w_reset=w_reset_param
        )
        raf_cell = RAFCell(
            name="raf_cell",
            n_units=n_units,
            tau_v=tau_v_param,
            tau_w=tau_w_param,
            omega=omega_param,
            dampen_factor=dampen_param,
            thr=thr_param,
            v_reset=v_reset_param,
            w_reset=w_reset_param,
            post_spike_reset=False  
        )
        advance_raf = MethodProcess("adv_raf", use_jit=True) >> raf_cell.advance_state
        advance = (MethodProcess("adv", use_jit=True) >> srm_pop.advance_state)
        reset = (MethodProcess("rst", use_jit=True) >> srm_pop.reset >> raf_cell.reset)

    ## now set up model simulation/experiment
    reset.run() ## reset model dynamics

    ## pre-synaptic arrival times (scaled artificially)
    base_pulse = 150.0 * 0.835
    incoming_events = {
        1.00: [base_pulse, 0.0, 0.0],
        2.57: [base_pulse, 0.0, 0.0]
    }
    ## sparse coordinates chosen across stimulus window/timeline
    sparse_query_timestamps = [
        1.00, 1.40, 2.15, 2.57, 2.78, 2.79, 3.60, 4.40, 5.20
    ]

    target_neuron = 0
    time_axis = []
    ode_v_history = []
    ode_spike_times = []
    srm_eval_times = []
    srm_v_records = []
    srm_spike_times = []

    was_post_spike = False
    post_prespike_mask = jnp.ones((1, n_units))
    has_spiked_once = False ## ensure only 1st threshold breach time-stamp recorded
    for step in range(time_steps):
        t_curr = round(step * dt, 2)

        ## continually track reference LIF ODE every loop iteration
        event_t = jnp.zeros((1, n_units))
        if t_curr in incoming_events:
            event_t = jnp.array([incoming_events[t_curr]])
        raf_cell.j.set(event_t)
        advance_raf.run(dt=dt, t=t_curr)

        time_axis.append(t_curr)
        ode_v_history.append(float(raf_cell.v.get()[0, target_neuron]))

        ## check for spike + enforce first-occurrence constraint
        if float(raf_cell.s.get()[0, target_neuron]) > 0.0:
            if not has_spiked_once:
                ode_spike_times.append(t_curr)
                has_spiked_once = True

        ## sparse event-driven tracking for RAF-SRM
        if t_curr in sparse_query_timestamps or was_post_spike:
            if t_curr in incoming_events:
                srm_pop.current_j.set(jnp.array([incoming_events[t_curr]]))
            elif was_post_spike:
                srm_pop.j_v.set(srm_pop.j_v.get() * post_prespike_mask)
                srm_pop.j_w.set(srm_pop.j_w.get() * post_prespike_mask)
                srm_pop.current_j.set(jnp.zeros((1, n_units)))
            else:
                srm_pop.current_j.set(jnp.zeros((1, n_units)))

            advance.run(dt=dt, t=t_curr)
            srm_eval_times.append(t_curr)
            srm_v_records.append(float(srm_pop.v.get()[0, target_neuron]))

            if float(srm_pop.s.get()[0, target_neuron]) > 0.0:
                srm_spike_times.append(t_curr)
            if jnp.any(srm_pop.s.get() > 0.0):
                was_post_spike = True
                post_prespike_mask = 1. - (srm_pop.s.get() > 0.)
            else:
                was_post_spike = False
                post_prespike_mask = jnp.ones((1, n_units))

    ## ------ plot setup ------
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ## plot contus reference background trace line from RAF's ODE
    ax.plot(time_axis, ode_v_history, label="RAF ODE Trajectory", color="#1f77b4", linewidth=2.0, alpha=0.7)
    ## overlay highly sparse point-query outputs as red dots
    ax.scatter(srm_eval_times, srm_v_records, color="red", edgecolor="black", s=70, zorder=5, label="RAF-SRM Point-Queries")
    ## extract absolute bottom layout data coordinate boundary dynamically
    bottom_y_boundary = ax.get_ylim()[0] ## grab lower float bound
    event_times_only = [1.00, 2.57]
    ax.scatter(
        event_times_only, [bottom_y_boundary] * len(event_times_only),
        color="#4B0082", edgecolor="black", marker="^", s=250, zorder=4,
        label="External Event Injections (Pre-Spikes)"
    )
    ## draw threshold boundary
    ax.axhline(y=thr_param, color="red", linestyle="--", alpha=0.5, label=f"Threshold ($\\theta$ = {thr_param:.1f})")
    ## draw a black tick directly on top of threshold line
    for spike_t in ode_spike_times:
        ax.vlines(
            x=spike_t, ymin=0.8, ymax=1.2, colors="black", linestyles="solid", 
            linewidth=4, zorder=6, label="Detected Spike Event"
        )
    ax.set_xlabel("Simulated Time (ms)", fontsize=15)
    ax.set_ylabel("Membrane Potential (mV)", fontsize=15)
    ## deduplicate handles + labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ## draw legend
    leg = ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="lower left")
    ## legend override/hook: explicitly compress icon sizes down to uniform scale
    for handle in leg.legend_handles:
        if hasattr(handle, 'set_sizes'):
            handle.set_sizes([40.0])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(0.0, 5.5)
    # re-apply axis bounds to prevent padding shift
    ax.set_ylim(bottom_y_boundary, ax.get_ylim()[1])

    plt.savefig(fname, dpi=300)
    #print(f"Single-spike validation diagram successfully written to disk: '{fname}'")

## run experiment
if __name__ == "__main__":
    fname = f"{output_dir}/raf_srm_point_queries.png"
    run_rafsrm_visualization(fname)

