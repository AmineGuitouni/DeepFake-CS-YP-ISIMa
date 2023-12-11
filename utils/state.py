import torch

class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self,
                 epoch,
                 paths_file, val_paths_file,
                 model,
                 optimizer,
                 scheduler):
        self.epoch = epoch
        self.paths_file = paths_file
        self.val_paths_file = val_paths_file
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::
        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "paths_file": self.paths_file,
            "val_paths_file": self.val_paths_file,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "schedule_state": self.scheduler.state_dict()
        }

    def apply_snapshot(self, obj):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = obj["epoch"]
        self.paths_file = obj["paths_file"]
        self.val_paths_file = obj["val_paths_file"]
        self.model.load_state_dict(obj["model_state"])
        self.optimizer.load_state_dict(obj["optimizer_state"])
        self.scheduler.load_state_dict(obj["schedule_state"])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot, device_id)
