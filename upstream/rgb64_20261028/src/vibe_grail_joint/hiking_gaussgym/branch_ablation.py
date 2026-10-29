"""Evaluation-only causal cut of ViBe attention output; geometry RGB stays intact."""
import torch

class AttentionBranchCut:
    def __init__(self, actor, worlds=(1,)):
        self.worlds=worlds
        self.enabled=True
        self.calls=0
        self.max_removed_norm=0.
        self.handles=[]
        assert len(actor.extracted_adapter_groups)==1
        for group in actor.extracted_adapter_groups:
            assert actor.extractors[group].latent_dim==128
            self.handles.append(actor.extractors[group].register_forward_hook(self._cut))

    def _cut(self, module, inputs, output):
        if not self.enabled:return output
        assert output.shape==(2,128)
        result=output.clone()
        ids=list(self.worlds)
        self.max_removed_norm=max(self.max_removed_norm,float(output[ids].norm(dim=-1).max()))
        result[ids]=0
        self.calls+=1
        return result

    @torch.no_grad()
    def verify(self,actor,obs):
        # A causal intervention test: changing direct image inputs cannot change
        # the cut world's action when geometry latent and all other inputs are fixed.
        self.enabled=False
        baseline=actor(obs,stochastic_output=False).clone()
        self.enabled=True
        cut=actor(obs,stochastic_output=False).clone()
        changed=obs.clone()
        changed['kv_tokens']['img_tokens'][1].fill_(3.14159)
        changed['q_cls'][1].fill_(-2.71828)
        altered=actor(changed,stochastic_output=False)
        preserved=float((baseline[0]-cut[0]).abs().max())
        independent=float((altered[1]-cut[1]).abs().max())
        assert preserved<1e-6 and independent<1e-6,(preserved,independent)
        return dict(normal_action_max_error=preserved,cut_action_image_invariance_error=independent,
                    initial_action_change_max=float((baseline[1]-cut[1]).abs().max()),
                    geometry_latent_fixed_for_causal_test=True)

    def audit(self):
        return dict(intervention='zero_128d_attention_output_after_extractor',cut_worlds=list(self.worlds),
                    calls=self.calls,max_removed_output_l2=self.max_removed_norm,
                    rgb_input='normal in both worlds',geometry_rgb='normal in both worlds',
                    preserved=['6d reference twist','8d independent command','672d geometry memory',
                               'SONIC motion tokens and proprioception','all checkpoint weights'],
                    implementation='forward hook, not structural pruning; no optimizer or training')
