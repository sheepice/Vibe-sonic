from ..velocity.dagger_policy import DaggerPPO

class StoneMoEPPO(DaggerPPO):
    def _extra_backward(self):
        super()._extra_backward()
        # DAgger/upper-body preservation may change the last actor batch. Use the
        # corresponding raw obs for a fresh routing graph; no privileged labels.
        obs=self._raw_actor._latest_update_obs
        if obs is None:return
        self._raw_actor._adapter_cond(obs)
        p=self._raw_actor.last_routes
        balance=(p.mean(0)-1/p.shape[-1]).square().sum()
        (0.01*balance).backward()
