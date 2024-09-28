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
from isaacgymenvs.humanipro.pro_common_player import ProCommonPlayer
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.algos_torch.players import rescale_actions


class ProAMPPlayerContinuous(ProCommonPlayer):

    def __init__(self, params):
        config = params['config']

        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._disc_reward_scale = config['disc_reward_scale']
        self._print_disc_prediction = config.get('print_disc_prediction', False)

        super().__init__(params)

        self.states_h, self.states_p = None, None
        return

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model_h.load_state_dict(checkpoint['model_h'])
        self.model_p.load_state_dict(checkpoint['model_p'])
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

    def _build_net(self, config_h, config_p):
        super()._build_net(config_h, config_p)

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config_p['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()
        return

    def _post_step(self, info):
        super()._post_step(info)
        if self._print_disc_prediction:
            self._amp_debug(info)
        return

    def _build_net_config(self):
        config_h, config_p = super()._build_net_config()
        if (hasattr(self, 'env')):
            config_h['amp_input_shape'] = self.env.amp_observation_space.shape
            config_p['amp_input_shape'] = self.env.amp_observation_space.shape
        else:
            config_h['amp_input_shape'] = self.env_info['amp_observation_space']
            config_p['amp_input_shape'] = self.env_info['amp_observation_space']

        return config_h, config_p

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs.to(self.device))
            amp_rewards = self._calc_amp_rewards(amp_obs.to(self.device))
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)
        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1.0 / (1.0 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r

    def get_action(self, obs, is_deterministic = False):
        obs = obs['obs']
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
            mu_h[:, -4:] = mu_p
            current_action = mu_h
        else:
            action_h[:, -4:] = action_p
            current_action = action_h
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action
