# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch 

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from isaacgymenvs.learning.amp_players import AMPPlayerContinuous
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.algos_torch.players import rescale_actions


class ProAMPPlayerContinuous(AMPPlayerContinuous):

    def __init__(self, params):
        super().__init__(params)
        self.states_h, self.states_p = None, None
        return

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model_h.load_state_dict(checkpoint['model'])
        self.model_p.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model_h.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            self.model_p.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

        if self._normalize_amp_input:
            checkpoint = torch_ext.load_checkpoint(fn)
            self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        return

    def _build_net(self, config):
        self.model_h, self.model_p = self.network.build(config)
        self.model_h.to(self.device)
        self.model_p.to(self.device)
        self.model_h.eval()
        self.model_p.eval()
        assert self.model_h.is_rnn() == self.model_p.is_rnn(), "human and prosthesis model's input must be same"
        self.is_rnn = self.model_h.is_rnn()

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()
        return

    def get_action(self, obs_dict, is_deterministic=False):
        obs = obs_dict['obs']
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict_h = {'is_train': False, 'prev_actions': None, 'obs': obs, 'rnn_states': self.states_h}
        input_dict_p = {'is_train': False, 'prev_actions': None, 'obs': obs, 'rnn_states': self.states_p}
        with torch.no_grad():
            res_dict_h = self.model_h(input_dict_h)
            res_dict_p = self.model_p(input_dict_p)
        mu_h, mu_p = res_dict_h['mus'], res_dict_p['mus']
        action_h, action_p = res_dict_h['actions'], res_dict_p['actions']
        self.states_h, self.states_p = res_dict_h['rnn_states'], res_dict_p['rnn_states']
        if is_deterministic:
            current_action_h, current_action_p = mu_h, mu_p
        else:
            current_action_h, current_action_p = action_h, action_p
        if self.has_batch_dimension == False:
            current_action_h = torch.squeeze(current_action_h.detach())
            current_action_p = torch.squeeze(current_action_p.detach())

        # print(torch.sum(torch.abs(current_action_h - current_action_p)).item())
        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action_h, -1.0, 1.0))
        else:
            return current_action_h
