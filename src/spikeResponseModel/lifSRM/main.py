import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ngclearn import Context, MethodProcess
from ngclearn.components.neurons.spiking import LIFCell, LIFSRM
output_dir = "exp/spikeResponseModel/lifSRM"

def run_lifsrm_visualization(fname="lif_srm_sparse_point_queries.png"):
    """
    This routine runs the LIF-SRM versus LIF ODE model experiment,
    tracking discrete points where the SRM is called as the ODE model
    is run iteratively/continually.
    """
    
    ## sim-time config
    dt = 0.1
    total_time = 15.0
    time_steps = int(total_time / dt)
    n_units = 1
    ## lif config
    tau_m_param = 5.0 ## ms
    thr_param = -52.0 ## mV
    v_rest_param = -65.0 ## mV
    v_reset_param = -60.0 ## mV

    ## set up model context, pinning both LIF and LIF-SRM 
    with Context("LIF_SRM_Context") as ctx:
        srm_pop = LIFSRM(
            name="srm",
            n_units=n_units,
            tau_m=tau_m_param,
            thr=thr_param,
            v_rest=v_rest_param,
            v_reset=v_reset_param
        )
        lif_cell = LIFCell(
            name="lif_cell",
            n_units=n_units,
            tau_m=tau_m_param,
            thr=thr_param,
            v_rest=v_rest_param,
            v_reset=v_reset_param,
            refract_time=0.,
            tau_theta=0.,
            theta_plus=0.,
            conduct_leak=1.
        )
        advance_lif = (MethodProcess("adv_lif", use_jit=True) >> lif_cell.advance_state)
        advance = (MethodProcess("adv", use_jit=True) >> srm_pop.advance_state)
        reset = (MethodProcess("rst", use_jit=True)
                 >> srm_pop.reset
                 >> lif_cell.reset
        )

    reset.run()

    ## set up stepped staircase pre-synaptic spike series
    base_pulse = 290.0
    incoming_events = {
        2.0: [0.0, base_pulse, 0.0],
        3.5: [0.0, base_pulse, 0.0],  # Spike occurs at t=3.6ms
        8.0: [0.0, base_pulse, 0.0],
        9.5: [0.0, base_pulse, 0.0]   # Spike occurs at t=9.6ms
    }

    ## set up query targets - these are spaced down ODE curves to prevent dot overlap
    sparse_query_timestamps = [
        1.0,  ## baseline
        2.0,  ## first pulse arrival
        2.8,  ## midway up first step
        3.5,  ## one step before first spike (-dt)
        3.6,  ## moment of 1st spike
        4.6,  ## spaced recovery (snapshot)
        6.5,  ## intermediate rest floor
        8.0,  ## third pulse arrival
        8.8,  ## midway up second step
        9.5,  ## one step before second spike (-dt)
        9.6,  ## moment of 2nd spike
        10.6, ## spaced recovery (snapshot)
        13.5  ## late-stage recovery point
    ]

    target_neuron = 1
    time_axis = []
    ode_v_history = []
    ode_spike_times = []

    srm_eval_times = []
    srm_v_records = []

    was_post_spike = False
    post_prespike_mask = jnp.ones((1, 3))

    for step in range(time_steps):
        t_curr = round(step * dt, 1)
        ## continuous reference tracking
        event_t = jnp.zeros((1, n_units))
        if t_curr in incoming_events:
            event_t = jnp.array([incoming_events[t_curr]])

        v_before_step = float(lif_cell.v.get()[0, target_neuron])
        j_val = float(event_t[0, target_neuron])
        v_next_est = v_before_step + (dt / tau_m_param) * (j_val - (v_before_step - v_rest_param))

        lif_cell.j.set(event_t)
        advance_lif.run(dt=dt, t=t_curr)

        time_axis.append(t_curr)
        if float(lif_cell.s.get()[0, target_neuron]) > 0.0:
            ode_v_history.append(v_next_est)  ## patch peak voltage to history line
            ode_spike_times.append(t_curr)
        else:
            ode_v_history.append(float(lif_cell.v.get()[0, target_neuron]))

        ## track event-stream for LIF-SRM
        is_event_step = t_curr in incoming_events
        is_query_step = t_curr in sparse_query_timestamps

        if is_event_step or is_query_step or was_post_spike:
            if t_curr in incoming_events:
                event_vector = incoming_events[t_curr]
                srm_pop.current_j.set(jnp.array([event_vector]))
            else:
                srm_pop.j_lowpass.set(srm_pop.j_lowpass.get() * post_prespike_mask)
                srm_pop.current_j.set(jnp.zeros((1, n_units)))

            advance.run(dt=dt, t=t_curr)

            srm_eval_times.append(t_curr)

            if t_curr in ode_spike_times:
                srm_v_records.append(v_next_est)
            else:
                srm_v_records.append(float(srm_pop.v.get()[0, target_neuron]))

            if jnp.any(srm_pop.s.get() > 0.0):
                was_post_spike = True
                post_prespike_mask = 1. - (srm_pop.current_j.get() > 0.)
            else:
                was_post_spike = False
                post_prespike_mask = jnp.ones((1, 3))

    ## ----- set up plotting -----
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ## plot continuous reference ODE curve
    ax.plot(
        time_axis, ode_v_history, 
        label="LIF ODE Trajectory", 
        color="#1f77b4", linewidth=2.0, alpha=0.7
    )

    ## plot SRM query points (as circles)
    ax.scatter(
        srm_eval_times, srm_v_records, color="red", 
        edgecolor="black", s=75, zorder=5, 
        label="LIF-SRM Point-Queries"
    )

    bottom_y_boundary = ax.get_ylim()[0]  # extract lower bound
    event_times_only = list(incoming_events.keys())
    ax.scatter(
        event_times_only, [bottom_y_boundary] * len(event_times_only),
        color="#4B0082", edgecolor="black", marker="^", s=180, zorder=4,
        label="External Event Injections (Pre-Spikes)"
    )

    ## draw threshold boundary
    ax.axhline(y=thr_param, color="red", linestyle="--", alpha=0.5, label=f"Threshold ($\\theta$ = {thr_param:.1f} mV)")
    ## draw black ticks centered on top of threshold line (height window: -53.5 to -50.5 mV)
    for spike_t in ode_spike_times:
        ax.vlines(
            x=spike_t, ymin=thr_param - 1.5, ymax=thr_param + 1.5, 
            colors="black", linestyles="solid", linewidth=4, zorder=6, 
            label="Detected Spike Event"
        )

    ax.set_xlabel("Simulated Time (ms)", fontsize=15)
    ax.set_ylabel("Membrane Potential (mV)", fontsize=15)

    ## de-duplicatae handles/labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    leg = ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="center right") ## draw key

    ## legend override/hook - compress icon sizes down to uniform scale (s=40)
    for handle in leg.legend_handles:
        if hasattr(handle, 'set_sizes'):
            handle.set_sizes([40.0])

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(0.0, total_time)
    ## force axis constraints to lock bottom_y_boundary smoothly onto lower frame edge
    ax.set_ylim(bottom_y_boundary, ax.get_ylim()[1])

    plt.savefig(fname, dpi=300)
    #print(f"Refined line diagram successfully saved to disk as '{fname}'.")

## run experiment
if __name__ == "__main__":
    fname = f"{output_dir}/lif_srm_point_queries.png"
    run_lifsrm_visualization(fname)

