import json
import os
import sys
import numpy as np
import copy
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import is_default_gpu
from utils.misc import length2mask
from utils.logger import print_progress

from models.model_HAMT import VLNBertCMT, Critic

from .eval_utils import cal_dtw

from .agent_base import BaseAgent


class Seq2SeqCMTAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # # For now, the agent can't pick which forward move to make - just the one in the middle
    # env_actions = {
    #   'left': (0,-1, 0), # left
    #   'right': (0, 1, 0), # right
    #   'up': (0, 0, 1), # up
    #   'down': (0, 0,-1), # down
    #   'forward': (1, 0, 0), # forward
    #   '<end>': (0, 0, 0), # <end>
    #   '<start>': (0, 0, 0), # <start>
    #   '<ignore>': (0, 0, 0)  # <ignore>
    # }
    # for k, v in env_actions.items():
    #     env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank

        # Models
        self._build_model()

        if self.args.world_size > 1:
            self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
            self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

        self.models = (self.vln_bert, self.critic)
        self.device = torch.device('cuda:%d'%self.rank) #TODO

        # Optimizers
        if self.args.optim == 'rms':
            optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            optimizer = torch.optim.SGD
        else:
            assert False
        if self.default_gpu:
            print('Optimizer: %s' % self.args.optim)

        self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, reduction="none")

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
        
        self.max_history = 50
        self.tour_prev_ended = [False] * self.env.batch_size
        self.history = None         # adaptive batch size
        self.history_raw = None        # full batch size
        self.history_raw_length = None        # full batch size

    def _build_model(self):
        self.vln_bert = VLNBertCMT(self.args).cuda()
        self.critic = Critic(self.args).cuda()

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]
        
        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor)
        mask = torch.from_numpy(mask)
        return seq_tensor.long().cuda(), mask.cuda(), seq_lengths

    def _cand_pano_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        ob_cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        ob_lens = []
        ob_img_fts, ob_ang_fts, ob_nav_types = [], [], []
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            cand_img_fts, cand_ang_fts, cand_nav_types = [], [], []
            cand_pointids = np.zeros((self.args.views, ), dtype=np.bool)
            for j, cc in enumerate(ob['candidate']):
                cand_img_fts.append(cc['feature'][:self.args.image_feat_size])
                cand_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                cand_pointids[cc['pointId']] = True
                cand_nav_types.append(1)
            # add [STOP] feature
            cand_img_fts.append(np.zeros((self.args.image_feat_size, ), dtype=np.float32))
            cand_ang_fts.append(np.zeros((self.args.angle_feat_size, ), dtype=np.float32))
            cand_img_fts = np.vstack(cand_img_fts)
            cand_ang_fts = np.vstack(cand_ang_fts)
            cand_nav_types.append(2)
            
            # add pano context
            pano_fts = ob['feature'][~cand_pointids]
            cand_pano_img_fts = np.concatenate([cand_img_fts, pano_fts[:, :self.args.image_feat_size]], 0)
            cand_pano_ang_fts = np.concatenate([cand_ang_fts, pano_fts[:, self.args.image_feat_size:]], 0)
            cand_nav_types.extend([0] * (self.args.views - np.sum(cand_pointids)))

            ob_lens.append(len(cand_nav_types))
            ob_img_fts.append(cand_pano_img_fts)
            ob_ang_fts.append(cand_pano_ang_fts)
            ob_nav_types.append(cand_nav_types)

        # pad features to max_len
        max_len = max(ob_lens)
        for i in range(len(obs)):
            num_pads = max_len - ob_lens[i]
            ob_img_fts[i] = np.concatenate([ob_img_fts[i], \
                np.zeros((num_pads, ob_img_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_ang_fts[i] = np.concatenate([ob_ang_fts[i], \
                np.zeros((num_pads, ob_ang_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_nav_types[i] = np.array(ob_nav_types[i] + [0] * num_pads)

        ob_img_fts = torch.from_numpy(np.stack(ob_img_fts, 0)).cuda()
        ob_ang_fts = torch.from_numpy(np.stack(ob_ang_fts, 0)).cuda()
        ob_nav_types = torch.from_numpy(np.stack(ob_nav_types, 0)).cuda()

        return ob_img_fts, ob_ang_fts, ob_nav_types, ob_lens, ob_cand_lens

    def _candidate_variable(self, obs):
        cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        max_len = max(cand_lens)
        cand_img_feats = np.zeros((len(obs), max_len, self.args.image_feat_size), dtype=np.float32)
        cand_ang_feats = np.zeros((len(obs), max_len, self.args.angle_feat_size), dtype=np.float32)
        cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int64)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                cand_img_feats[i, j] = cc['feature'][:self.args.image_feat_size]
                cand_ang_feats[i, j] = cc['feature'][self.args.image_feat_size:]
                cand_nav_types[i, j] = 1
            cand_nav_types[i, cand_lens[i]-1] = 2

        cand_img_feats = torch.from_numpy(cand_img_feats).cuda()
        cand_ang_feats = torch.from_numpy(cand_ang_feats).cuda()
        cand_nav_types = torch.from_numpy(cand_nav_types).cuda()
        return cand_img_feats, cand_ang_feats, cand_nav_types, cand_lens

    def _future_variable(self, obs, t):
        fut_img_feats = np.zeros((len(obs), self.args.image_feat_size), np.float32)
        fut_ang_feats = np.zeros((len(obs), self.args.angle_feat_size), np.float32)
        episode_not_ended = np.zeros((len(obs)), np.int64)
        for i, ob in enumerate(obs):
            if t < len(ob['gt_path_feats']):
                fut_img_feats[i] = ob['gt_path_feats'][t][ob['gt_view_idxs'][t], :self.args.image_feat_size]
                fut_ang_feats[i] = ob['gt_angle_feats'][t]
                episode_not_ended[i] = 1
        fut_img_feats = torch.from_numpy(fut_img_feats).cuda()
        fut_ang_feats = torch.from_numpy(fut_ang_feats).cuda()
    
        if self.args.hist_enc_pano:
            fut_pano_img_feats = np.zeros((len(obs), self.args.views, self.args.image_feat_size), np.float32)
            fut_pano_ang_feats = np.zeros((len(obs), self.args.views, self.args.angle_feat_size), np.float32)
            for i, ob in enumerate(obs):
                if t < len(ob['gt_path_feats']):
                    fut_pano_img_feats[i] = ob['gt_path_feats'][t][:, :self.args.image_feat_size]
                    fut_pano_ang_feats[i] = ob['gt_path_feats'][t][:, self.args.image_feat_size:]
            fut_pano_img_feats = torch.from_numpy(fut_pano_img_feats).cuda()
            fut_pano_ang_feats = torch.from_numpy(fut_pano_ang_feats).cuda()
        else:
            fut_pano_img_feats, fut_pano_ang_feats = None, None
    
        return (fut_img_feats, fut_ang_feats, fut_pano_img_feats, fut_pano_ang_feats), episode_not_ended
    
    def _history_variable(self, obs):
        hist_img_feats = np.zeros((len(obs), self.args.image_feat_size), np.float32)
        for i, ob in enumerate(obs):
            hist_img_feats[i] = ob['feature'][ob['viewIndex'], :self.args.image_feat_size]
        hist_img_feats = torch.from_numpy(hist_img_feats).cuda()

        if self.args.hist_enc_pano:
            hist_pano_img_feats = np.zeros((len(obs), self.args.views, self.args.image_feat_size), np.float32)
            hist_pano_ang_feats = np.zeros((len(obs), self.args.views, self.args.angle_feat_size), np.float32)
            for i, ob in enumerate(obs):
                hist_pano_img_feats[i] = ob['feature'][:, :self.args.image_feat_size]
                hist_pano_ang_feats[i] = ob['feature'][:, self.args.image_feat_size:]
            hist_pano_img_feats = torch.from_numpy(hist_pano_img_feats).cuda()
            hist_pano_ang_feats = torch.from_numpy(hist_pano_ang_feats).cuda()
        else:
            hist_pano_img_feats, hist_pano_ang_feats = None, None

        return hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    # def make_equiv_action(self, a_t, obs, traj=None):
    #     """
    #     Interface between Panoramic view and Egocentric view
    #     It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
    #     """
    #     def take_action(i, name):
    #         if type(name) is int:       # Go to the next view
    #             self.env.env.sims[i].makeAction([name], [0], [0])
    #         else:                       # Adjust
    #             self.env.env.sims[i].makeAction(*self.env_actions[name])
    #
    #     for i, ob in enumerate(obs):
    #         action = a_t[i]
    #         if action != -1:            # -1 is the <stop> action
    #             select_candidate = ob['candidate'][action]
    #             src_point = ob['viewIndex']
    #             trg_point = select_candidate['pointId']
    #             src_level = (src_point ) // 12  # The point idx started from 0
    #             trg_level = (trg_point ) // 12
    #             while src_level < trg_level:    # Tune up
    #                 take_action(i, 'up')
    #                 src_level += 1
    #             while src_level > trg_level:    # Tune down
    #                 take_action(i, 'down')
    #                 src_level -= 1
    #             while self.env.env.sims[i].getState()[0].viewIndex != trg_point:    # Turn right until the target
    #                 take_action(i, 'right')
    #             assert select_candidate['viewpointId'] == \
    #                    self.env.env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
    #             take_action(i, select_candidate['idx'])
    #
    #             state = self.env.env.sims[i].getState()[0]
    #             if traj is not None:
    #                 traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def _edit_history_length(self):
        for i_batch, hist in enumerate(self.history):
            l_hist = len(hist)
            if l_hist > self.max_history:
                delta = l_hist - self.max_history
                self.history[i_batch] = hist[:1] + hist[delta + 1:]
                for k in range(5):
                    self.history_raw[i_batch][k] = self.history_raw[i_batch][k][delta:]
                self.history_raw_length[i_batch] -= delta

    def _pad_hist_embeds(self):
        max_length = max([len(hist) for hist in self.history])
        hist_embeds = []
        for hist in self.history:
            hist_embeds.append(F.pad(input=torch.stack(hist), pad=(0, 0, 0, max_length-len(hist)), value=0))
        return torch.stack(hist_embeds)
        
    def _oracle_phrase(self):
        if self.env.extra_obs is not None:
            self.env.extra_obs = [ob for ob in self.env.extra_obs if ob is not None]
            future_lens = []
            for ob in self.env.extra_obs:
                if len(ob['gt_path_feats']) > self.vln_bert.vis_config.max_action_steps:
                    ob['gt_path_feats'] = ob['gt_path_feats'][-self.vln_bert.vis_config.max_action_steps:]
                    ob['gt_angle_feats'] = ob['gt_angle_feats'][-self.vln_bert.vis_config.max_action_steps:]
                    ob['gt_view_idxs'] = ob['gt_view_idxs'][-self.vln_bert.vis_config.max_action_steps:]
                future_lens.append(len(ob['gt_path_feats']))
            for t in range(max(future_lens)):
                feats, episode_not_ended = self._future_variable(self.env.extra_obs, t)
                t_hist_inputs = {
                    'mode': 'history',
                    'hist_img_feats': feats[0],
                    'hist_ang_feats': feats[1],
                    'hist_pano_img_feats': feats[2],
                    'hist_pano_ang_feats': feats[3],
                    'ob_step': [t],
                }
                t_hist_embeds = self.vln_bert(**t_hist_inputs)

                self.history_raw_length += episode_not_ended
                for i_batch, hist_embed in enumerate(t_hist_embeds):
                    if episode_not_ended[i_batch]:
                        self.history[i_batch].append(hist_embed)
                        for k in range(4):
                            self.history_raw[i_batch][k].append(feats[k][i_batch])
                        self.history_raw[i_batch][4].append(t)
            self._edit_history_length()
        
        
    def rollout(self, train_ml=None, train_rl=True, reset=True, extended_history=False, sep_hist=False, rebuild=False):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs(t=0)
        not_ended_obs = [ob for ob in obs if ob is not None]
        tour_prev_not_ended = ~np.array(self.tour_prev_ended)
        tour_not_ended = ~(np.array(self.env.tour_ended)[tour_prev_not_ended])
        batch_size = tour_not_ended.sum()

        # rebuild history after grad update
        if extended_history and self.history is not None:
            self.history_raw = [hist for i, hist in enumerate(self.history_raw) if tour_not_ended[i]]
            self.history_raw_length = self.history_raw_length[tour_not_ended]
            if rebuild:
                if sep_hist:
                    self.history = [[h_embed] for h_embed in self.vln_bert('history', ob_step=[-1]).expand(batch_size, -1)]
                else:
                    self.history = [[h_embed] for h_embed in self.vln_bert('history').expand(batch_size, -1)]
                for i_batch, hist_seq in enumerate(self.history_raw):
                    img_feats, ang_feats, pano_img_feats, pano_ang_feats, ob_step = hist_seq
                    hist_inputs = {
                        'mode': 'history',
                        'hist_img_feats': torch.stack(img_feats),
                        'hist_ang_feats': torch.stack(ang_feats),
                        'hist_pano_img_feats': torch.stack(pano_img_feats),
                        'hist_pano_ang_feats': torch.stack(pano_ang_feats),
                        'ob_step': ob_step,
                    }
                    hist_embeds = self.vln_bert(**hist_inputs)
                    self.history[i_batch] += [h_embed for h_embed in hist_embeds]
            else:
                self.history = [hist for i, hist in enumerate(self.history) if tour_not_ended[i]]
            
        # oracle phase-1
        if self.env.iterative:
            self._oracle_phrase()
    
        # Language input
        txt_ids, txt_masks, txt_lens = self._language_variable(not_ended_obs)

        ''' Language BERT '''
        language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
        txt_embeds = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'scan': ob['scan'],
            'gt_path': ob['gt_path'],
            'gt_length': ob['distance'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in not_ended_obs]

        # # Init the reward shaping
        # last_dist = np.zeros(batch_size, np.float32)
        # last_ndtw = np.zeros(batch_size, np.float32)
        # for i, ob in enumerate(not_ended_obs):   # The init distance from the view point to the target
        #     last_dist[i] = ob['distance']
        #     path_act = [vp[0] for vp in traj[i]['path']]
        #     last_ndtw[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

        if traj[0]['instr_id'] == '498_0':
            k = 1
        # Initialization the tracking state
        ended = np.array([False] * batch_size)

        # Init the logs
        inflection_list = [[] for _ in range(batch_size)]
        loss_list = []
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.

        # for backtrack
        visited = [set() for _ in range(batch_size)]
        
        # init history
        if not extended_history or self.history is None:
            first_ep = True
            self.history = [[h_embed] for h_embed in self.vln_bert('history').expand(batch_size, -1)]  # global embedding
            self.history_raw = [[[] for _ in range(5)] for _ in range(batch_size)]
            self.history_raw_length = np.zeros(batch_size, np.int64)
        else:
            first_ep = False
            if sep_hist:
                pivot = self.history_raw_length.copy()
                current_hist_tag = self.vln_bert('history').expand(batch_size, -1)
                for i_batch, hist_tag in enumerate(current_hist_tag):
                    self.history[i_batch].append(hist_tag)
                self._edit_history_length()
                
        for t in range(self.args.max_action_len):
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(not_ended_obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(not_ended_obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()
            
            ''' Visual BERT '''
            hist_lens = list(self.history_raw_length + 2) if sep_hist and not first_ep else list(self.history_raw_length + 1)
            hist_embeds = self._pad_hist_embeds()
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,    # history before t step
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True if self.feedback == 'sample' else False
            }
            t_outputs = self.vln_bert(**visual_inputs)
            logit = t_outputs[0]
            if self.feedback == 'sample':
                h_t = t_outputs[1]
                hidden_states.append(h_t)

            if train_ml is not None:
                # Supervised training
                target = self._teacher_action(not_ended_obs, ended)
                step_loss = self.criterion(logit, target)
                if self.args.inflection_weighting:
                    loss_list.append(step_loss)
                else:
                    ml_loss += step_loss.sum()


            # # mask logit where the agent backtracks in observation in evaluation
            # if self.args.no_cand_backtrack:
            #     bt_masks = torch.zeros(ob_nav_types.size()).bool()
            #     for ob_id, ob in enumerate(not_ended_obs):
            #         visited[ob_id].add(ob['viewpoint'])
            #         for c_id, c in enumerate(ob['candidate']):
            #             if c['viewpointId'] in visited[ob_id]:
            #                 bt_masks[ob_id][c_id] = True
            #     bt_masks = bt_masks.cuda()
            #     logit.masked_fill_(bt_masks, -float('inf'))

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (ob_cand_lens[i]-1) or next_id == self.args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # get history input embeddings
            if train_rl or ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len-1)):
                # DDP error: RuntimeError: Expected to mark a variable ready only once.
                # It seems that every output from DDP should be used in order to perform correctly
                hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(not_ended_obs)
                prev_act_angle = np.zeros((len(not_ended_obs), self.args.angle_feat_size), np.float32)
                for i, next_id in enumerate(cpu_a_t):
                    if next_id != -1:
                        prev_act_angle[i] = not_ended_obs[i]['candidate'][next_id]['feature'][-self.args.angle_feat_size:]
                prev_act_angle = torch.from_numpy(prev_act_angle).cuda()
                t_hist_inputs = {
                    'mode': 'history',
                    'hist_img_feats': hist_img_feats,
                    'hist_ang_feats': prev_act_angle,
                    'hist_pano_img_feats': hist_pano_img_feats,
                    'hist_pano_ang_feats': hist_pano_ang_feats,
                    'ob_step': [t],
                }
                t_hist_embeds = self.vln_bert(**t_hist_inputs)
                for i_batch, i_ended in enumerate(ended):
                    if not (i_ended or cpu_a_t[i_batch] == -1):
                        self.history_raw_length[i_batch] += 1
                        self.history[i_batch].append(t_hist_embeds[i_batch])
                        self.history_raw[i_batch][0].append(hist_img_feats[i_batch])
                        self.history_raw[i_batch][1].append(prev_act_angle[i_batch])
                        self.history_raw[i_batch][2].append(hist_pano_img_feats[i_batch])
                        self.history_raw[i_batch][3].append(hist_pano_ang_feats[i_batch])
                        self.history_raw[i_batch][4].append(t)
                self._edit_history_length()

                
            # Make action and get the new state
            obs, inflection = self.env.step(cpu_a_t, t=t+1, traj=traj)
            not_ended_obs = [ob for ob in obs if ob is not None]
            if self.args.inflection_weighting:
                for i_batch, i in enumerate(inflection):
                    if not ended[i_batch] and cpu_a_t[i_batch] == -1:
                        inflection_list[i_batch].append(2)
                    else:
                        inflection_list[i_batch].append(i)
            

            # if train_rl:
            #     # Calculate the mask and reward
            #     dist = np.zeros(batch_size, np.float32)
            #     ndtw_score = np.zeros(batch_size, np.float32)
            #     reward = np.zeros(batch_size, np.float32)
            #     mask = np.ones(batch_size, np.float32)
            #     for i, ob in enumerate(obs):
            #         dist[i] = ob['distance']
            #         path_act = [vp[0] for vp in traj[i]['path']]
            #         ndtw_score[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']
            #
            #         if ended[i]:
            #             reward[i] = 0.0
            #             mask[i] = 0.0
            #         else:
            #             action_idx = cpu_a_t[i]
            #             # Target reward
            #             if action_idx == -1:                              # If the action now is end
            #                 if dist[i] < 3.0:                             # Correct
            #                     reward[i] = 2.0 + ndtw_score[i] * 2.0
            #                 else:                                         # Incorrect
            #                     reward[i] = -2.0
            #             else:                                             # The action is not end
            #                 # Path fidelity rewards (distance & nDTW)
            #                 reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
            #                 ndtw_reward = ndtw_score[i] - last_ndtw[i]
            #                 if reward[i] > 0.0:                           # Quantification
            #                     reward[i] = 1.0 + ndtw_reward
            #                 elif reward[i] < 0.0:
            #                     reward[i] = -1.0 + ndtw_reward
            #                 else:
            #                     raise NameError("The action doesn't change the move")
            #                 # Miss the target penalty
            #                 if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
            #                     reward[i] -= (1.0 - last_dist[i]) * 2.0
            #     rewards.append(reward)
            #     masks.append(mask)
            #     last_dist[:] = dist
            #     last_ndtw[:] = ndtw_score

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        # if train_rl:
        #     if self.args.ob_type == 'pano':
        #         ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
        #         ob_masks = length2mask(ob_lens).logical_not()
        #     elif self.args.ob_type == 'cand':
        #         ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
        #         ob_masks = length2mask(ob_cand_lens).logical_not()
        #
        #     ''' Visual BERT '''
        #     visual_inputs = {
        #         'mode': 'visual',
        #         'txt_embeds': txt_embeds,
        #         'txt_masks': txt_masks,
        #         'hist_embeds': self.history,
        #         'hist_lens': hist_lens,
        #         'ob_img_feats': ob_img_feats,
        #         'ob_ang_feats': ob_ang_feats,
        #         'ob_nav_types': ob_nav_types,
        #         'ob_masks': ob_masks,
        #         'return_states': True
        #     }
        #     _, last_h_ = self.vln_bert(**visual_inputs)
        #
        #     rl_loss = 0.
        #
        #     # NOW, A2C!!!
        #     # Calculate the final discounted reward
        #     last_value__ = self.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
        #     discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
        #     for i in range(batch_size):
        #         if not ended[i]:        # If the action is not ended, use the value function as the last reward
        #             discount_reward[i] = last_value__[i]
        #
        #     length = len(rewards)
        #     total = 0
        #     for t in range(length-1, -1, -1):
        #         discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
        #         mask_ = torch.from_numpy(masks[t]).cuda()
        #         clip_reward = discount_reward.copy()
        #         r_ = torch.from_numpy(clip_reward).cuda()
        #         v_ = self.critic(hidden_states[t])
        #         a_ = (r_ - v_).detach()
        #
        #         t_policy_loss = (-policy_log_probs[t] * a_ * mask_).sum()
        #         t_critic_loss = (((r_ - v_) ** 2) * mask_).sum() * 0.5 # 1/2 L2 loss
        #
        #         rl_loss += t_policy_loss + t_critic_loss
        #         if self.feedback == 'sample':
        #             rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_).sum()
        #
        #         self.logs['critic_loss'].append(t_critic_loss.item())
        #         self.logs['policy_loss'].append(t_policy_loss.item())
        #
        #         total = total + np.sum(masks[t])
        #     self.logs['total'].append(total)
        #
        #     # Normalize the loss function
        #     if self.args.normalize_loss == 'total':
        #         rl_loss /= total
        #     elif self.args.normalize_loss == 'batch':
        #         rl_loss /= batch_size
        #     else:
        #         assert self.args.normalize_loss == 'none'
        #
        #     self.loss += rl_loss
        #     self.logs['RL_loss'].append(rl_loss.item()) # critic loss + policy loss + entropy loss

        # oracle phase-2

        if sep_hist and not first_ep:
            for i_batch, p in enumerate(pivot):
                self.history[i_batch] = self.history[i_batch][:p+1] + self.history[i_batch][p+2:]
        if self.env.iterative and not self.env.check_reach():
            self._oracle_phrase()
            
        if train_ml is not None:
            if self.args.inflection_weighting:
                loss_list = torch.stack(loss_list).t()
                tot_weight = 0
                for i, inflections in enumerate(inflection_list):
                    inflections = np.array(inflections, dtype=np.float)
                    inflect_weight = (inflections > 0).sum() / (inflections == 2).sum()
                    inflections[inflections == 2] = inflect_weight
                    tot_weight += inflections.sum()
                    ml_loss += (loss_list[i] * torch.from_numpy(inflections).cuda()).sum()
                self.loss += ml_loss * train_ml / tot_weight
                self.logs['IL_loss'].append((ml_loss * train_ml / tot_weight).item())
            else:
                self.loss += ml_loss * train_ml / batch_size
                self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.
        
        self.tour_prev_ended = copy.deepcopy(self.env.tour_ended)
        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super().test(iters=iters, extended_history=self.args.extended_history, sep_hist=self.args.sep_hist)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback
        
        self.vln_bert.train()
        self.critic.train()

        self.losses = []
        self.tour_prev_ended = [False] * self.env.batch_size
        self.history = None
        self.history_raw = None
        self.history_raw_length = None
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            if self.args.iterative:
                while True:
                    self.loss = 0
                    if feedback == 'teacher':
                        self.feedback = 'teacher'
                        self.rollout(train_ml=self.args.teacher_weight, train_rl=False,
                                     extended_history=self.args.extended_history, sep_hist=self.args.sep_hist,
                                     rebuild=self.args.rebuild, **kwargs)
                    elif feedback == 'sample':  # agents in IL and RL separately
                        if self.args.ml_weight != 0:
                            self.feedback = 'teacher'
                            self.rollout(train_ml=self.args.ml_weight, train_rl=False,
                                         extended_history=self.args.extended_history, sep_hist=self.args.sep_hist,
                                         rebuild=self.args.rebuild, **kwargs)
                        self.feedback = 'sample'
                        self.rollout(train_ml=None, train_rl=True, **kwargs)
                    else:
                        assert False
                    self.loss.backward()
                    if self.env.check_last():
                        self.tour_prev_ended = [False] * self.env.batch_size
                        self.history = None
                        self.history_raw = None
                        self.history_raw_length = None
                        break
            else:
                self.loss = 0
                if feedback == 'teacher':
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
                elif feedback == 'sample':  # agents in IL and RL separately
                    if self.args.ml_weight != 0:
                        self.feedback = 'teacher'
                        self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
                    self.feedback = 'sample'
                    self.rollout(train_ml=None, train_rl=True, **kwargs)
                else:
                    assert False
                self.loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            if self.args.aug is None:
                print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1
