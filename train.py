import jax
import os
from utils import cond_mkdir
import tqdm
import orbax.checkpoint as orbax
from flax.training import checkpoints
import wandb


def train(
    key,
    state,
    dataset_state,
    loss_fn,
    validation_fn,
    dataset_sampler_fn,
    ckpt,
    epochs,
    epochs_till_checkpoint,
    logging_root,
    experiment_name,
):
    root_path = os.path.join(logging_root, experiment_name)
    checkpoints_dir = os.path.join(root_path, "checkpoints")
    cond_mkdir(checkpoints_dir)

    train_losses = []
    loss_infos = []

    @jax.jit
    def update(state, normalized_tcoords, source_boundary_values, dirichlet_mask):
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, normalized_tcoords, source_boundary_values, dirichlet_mask
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, loss_info

    for epoch in tqdm.tqdm(range(epochs)):

        key, dataset_key = jax.random.split(key)
        dataset_state, normalized_tcoords, gt = dataset_sampler_fn(
            dataset_key, dataset_state
        )

        state, train_loss, loss_info = update(
            state,
            normalized_tcoords,
            gt["source_boundary_values"],
            gt["dirichlet_mask"],
        )

        train_losses.append(train_loss.item())
        loss_infos.append(loss_info)

        if not epoch % epochs_till_checkpoint and epoch:
            orbax_checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())
            ckpt["dataset"] = dataset_state
            ckpt["model"] = state
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoints_dir,
                target=ckpt,
                step=epoch,
                overwrite=True,
                keep=2,
                orbax_checkpointer=orbax_checkpointer,
            )
            wandb.log(
                {
                    "train_loss": train_losses[-1],
                    "dirichlet_loss": loss_infos[-1]["dirichlet_loss"].item(),
                    "diff_constraint_hom_loss": loss_infos[-1][
                        "diff_constraint_hom_loss"
                    ].item(),
                },
                step=epoch,
            )
            print(f"Epoch: {epoch} train_loss: {train_losses[-1]}")

            validation_fn(state, epoch)
