from typing import Dict, List

import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from ray.rllib import SampleBatch
from ray.rllib.models import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.typing import TensorType

from gym_lr.rl_src.configurations import LSTM_SIZE

torch, nn = try_import_torch()


class MyTorchRNNModel(TorchRNN, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 fc_size=[64, 64],
                 lstm_state_size=LSTM_SIZE):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
        elif isinstance(action_space, MultiDiscrete):
            self.action_dim = np.sum(action_space.nvec)
        elif action_space.shape is not None:
            self.action_dim = int(np.product(action_space.shape))
        else:
            self.action_dim = int(len(action_space))

        lstm_input = self.obs_size
        if model_config["lstm_use_prev_action"]:
            lstm_input += self.num_outputs  # action space
        if model_config["lstm_use_prev_reward"]:
            lstm_input += 1  # reward

        self.lstm = nn.LSTM(lstm_input, self.lstm_state_size, batch_first=True)

        # initializer = normc_initializer(1.0)
        initializer = torch.nn.init.xavier_uniform_
        activation_fn = nn.Tanh

        # action branch
        h_layer_input = self.lstm_state_size
        layers = []
        for h_layer in fc_size:
            linear = nn.Linear(h_layer_input, h_layer)
            initializer(linear.weight)
            layers.append(linear)
            layers.append(activation_fn())
            h_layer_input = h_layer

        layers.append(nn.Linear(h_layer_input, num_outputs))
        self.action_branch = nn.Sequential(*layers)

        # value branch
        h_layer_input = self.lstm_state_size
        layers = []
        for h_layer in fc_size:
            linear = nn.Linear(h_layer_input, h_layer)
            initializer(linear.weight)
            layers.append(linear)
            layers.append(activation_fn())
            h_layer_input = h_layer

        layers.append(nn.Linear(h_layer_input, 1))
        self.value_branch = nn.Sequential(*layers)

        if model_config["lstm_use_prev_action"]:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = \
                ViewRequirement(SampleBatch.ACTIONS, space=self.action_space,
                                shift=-1)
        if model_config["lstm_use_prev_reward"]:
            self.view_requirements[SampleBatch.PREV_REWARDS] = \
                ViewRequirement(SampleBatch.REWARDS, shift=-1)

        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            np.zeros(self.lstm_state_size, np.float32),
            np.zeros(self.lstm_state_size, np.float32)
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        # prev_a_r = []
        # if self.model_config["lstm_use_prev_action"]:
        #     if isinstance(self.action_space, (Discrete, MultiDiscrete)):
        #         prev_a = one_hot(inputs[SampleBatch.PREV_ACTIONS].float(), self.action_space)
        #     else:
        #         prev_a = inputs[SampleBatch.PREV_ACTIONS].float()
        #     prev_a_r.append(torch.reshape(prev_a, [-1, self.action_dim]))
        # if self.model_config["lstm_use_prev_reward"]:
        #     prev_a_r.append(torch.reshape(inputs[SampleBatch.PREV_REWARDS].float(), [-1, 1]))
        #
        # if prev_a_r:
        #     inputs = torch.cat([inputs] + prev_a_r, dim=1)

        self._features, [h, c] = self.lstm(
            inputs, [torch.unsqueeze(state[0], 0),
                     torch.unsqueeze(state[1], 0)])
        action_out = self.action_branch(self._features)

        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    @override(TorchRNN)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        assert seq_lens is not None
        # Push obs through "unwrapped" net's `forward()` first.
        wrapped_out = input_dict["obs_flat"]

        # Concat. prev-action/reward if required.
        prev_a_r = []
        if self.model_config["lstm_use_prev_action"]:
            if isinstance(self.action_space, (Discrete, MultiDiscrete)):
                prev_a = one_hot(input_dict[SampleBatch.PREV_ACTIONS].float(),
                                 self.action_space)
            else:
                prev_a = input_dict[SampleBatch.PREV_ACTIONS].float()
            prev_a_r.append(torch.reshape(prev_a, [-1, self.action_dim]))
        if self.model_config["lstm_use_prev_reward"]:
            prev_a_r.append(
                torch.reshape(input_dict[SampleBatch.PREV_REWARDS].float(),
                              [-1, 1]))

        if prev_a_r:
            wrapped_out = torch.cat([wrapped_out] + prev_a_r, dim=1)

        # Then through our LSTM.
        input_dict["obs_flat"] = wrapped_out
        return super().forward(input_dict, state, seq_lens)
