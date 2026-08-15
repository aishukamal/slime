import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action
from slime.utils.phase_callback import load_phase_callback


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    release_train = args.release_train
    phase_cb = load_phase_callback(args)

    # allocate the GPUs
    if phase_cb:
        phase_cb.on_phase_begin("init", "both")

    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if phase_cb:
        phase_cb.on_phase_end("init", "both")

    # async train loop.
    #
    # Lock protocol: acquire TRAINER first, then SAMPLER (global order).
    # In async pipelining, gen N+1 overlaps with train N.  We acquire both
    # locks, dispatch gen, then train — sampler lock is held through
    # training until collection.  With per-job sampler groups this is a
    # no-op on the orchestrator side (no contention).
    if phase_cb:
        phase_cb.on_phase_begin("generate", "sampler", {"rollout_id": args.start_rollout_id})
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Collect the pending generation (sampler lock already held from dispatch).
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)
            rollout_data_next_future = None
            if phase_cb:
                phase_cb.on_phase_end("generate", "sampler", {"rollout_id": rollout_id})

        if release_train:
            if phase_cb:
                phase_cb.on_phase_begin("create", "trainer", {"rollout_id": rollout_id})
            actor_model.create()
            if phase_cb:
                phase_cb.on_phase_end("create", "trainer", {"rollout_id": rollout_id})

        if phase_cb:
            phase_cb.on_phase_begin("train", "trainer", {"rollout_id": rollout_id})

        # Dispatch next generation while training — acquire SAMPLER after
        # TRAINER to respect lock order, then dispatch the remote call.
        if rollout_id + 1 < args.num_rollout:
            if phase_cb:
                phase_cb.on_phase_begin("generate", "sampler", {"rollout_id": rollout_id + 1})
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if phase_cb:
            phase_cb.on_phase_end("train", "trainer", {"rollout_id": rollout_id})

        if release_train or (rollout_id + 1) % args.update_weights_interval == 0:
            # Collect any pending generation before weight sync (needs both pools).
            if rollout_data_next_future is not None:
                rollout_data_curr_ref = ray.get(rollout_data_next_future)
                rollout_data_next_future = None
                if phase_cb:
                    phase_cb.on_phase_end("generate", "sampler", {"rollout_id": rollout_id})
            if phase_cb:
                phase_cb.on_phase_begin("weight_sync", "both", {"rollout_id": rollout_id})
            actor_model.update_weights()
            if phase_cb:
                phase_cb.on_phase_end("weight_sync", "both", {"rollout_id": rollout_id})

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            if phase_cb:
                phase_cb.on_phase_begin("eval", "sampler", {"rollout_id": rollout_id})
            ray.get(rollout_manager.eval.remote(rollout_id))
            if phase_cb:
                phase_cb.on_phase_end("eval", "sampler", {"rollout_id": rollout_id})

    if phase_cb:
        phase_cb.close()
    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
