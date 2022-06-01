''' Batched Room-to-Room navigation environment '''

import json
import os
from itertools import chain
import numpy as np
import math
import random
import networkx as nx
from collections import defaultdict, namedtuple

from r2r.data_utils import load_nav_graphs, angle_feature
from r2r.eval_utils import cal_dtw, cal_cls

ERROR_MARGIN = 3.0
ANGLE_INC = np.pi / 6.
WorldState = namedtuple(
  "WorldState",
  ["scan_id", "viewpoint_id", "view_index", "heading", "elevation"]
)

# TODO: remember to download total_adj_list.json and angle_feature.npy
class EnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, feat_db=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.feat_db = feat_db
        self.world_states = [None] * batch_size
        with open("../datasets/total_adj_list.json") as f:
            self.adj_dict = json.load(f)

    # def _make_id(self, scanId, viewpointId):
    #     return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        elevation = 0.0
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            if scanId is None:
                continue
            view_index = (12 * round(elevation / ANGLE_INC + 1) + round(heading / ANGLE_INC) % 12)
            heading = (view_index - (12 * round(elevation / ANGLE_INC + 1))) * ANGLE_INC
            self.world_states[i] = WorldState(
                scan_id=scanId,
                viewpoint_id=viewpointId,
                view_index=view_index,
                heading=heading,
                elevation=elevation
            )

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, state in enumerate(self.world_states):
            feature = self.feat_db.get_image_feature(state.scan_id, state.viewpoint_id)
            feature_states.append((feature, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, action in enumerate(actions):
            if action >= 0:
                long_id = "%s_%s_%s" % (self.world_states[i].scan_id,
                                        self.world_states[i].viewpoint_id,
                                        self.world_states[i].view_index)
                self.next_point = self.adj_dict[long_id][action + 1]
                self.world_states[i] = WorldState(
                    scan_id=self.world_states[i].scan_id,
                    viewpoint_id=self.next_point['nextViewpointId'],
                    view_index=self.next_point['absViewIndex'],
                    heading=self.next_point['absViewIndex'] % 12 * ANGLE_INC,
                    elevation=(self.next_point['absViewIndex'] // 12 - 1) * ANGLE_INC
                )


class R2RBatch(object):
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(
        self, feat_db, instr_data, connectivity_dir,
        batch_size=64, angle_feat_size=4,
        seed=0, name=None, iterative=False
    ):
        self.env = EnvBatch(feat_db=feat_db, batch_size=batch_size)
        self.data = instr_data
        self.scans = set([x['scan'] for x in self.data])
        self.gt_trajs = self._get_gt_trajs(self.data)

        if name == 'aug':
            tour_data = json.load(open("../tours_iVLN_prevalent.json"))
            tour_data = tour_data["prevalent"]
        else:
            tour_data = json.load(open("../tours_iVLN.json"))
            tour_data = tour_data[name]
        self.tour_data = list(chain.from_iterable(tour_data.values()))
        self.tour_batch = None
        self.extra_obs = None

        self.connectivity_dir = connectivity_dir
        self.angle_feat_size = angle_feat_size
        self.name = name
        self.seed = seed
        random.seed(self.seed)
        if iterative:
            self.iterative = True
            self.data = {ex["instr_id"]:ex for ex in self.data}
            random.shuffle(self.tour_data)
        else:
            self.iterative = False
            random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self.tour_ended = [False] * self.batch_size
        self._load_nav_graphs()
        self.angle_feature = np.load("../datasets/angle_feature.npy")
        self.angle_direct_dict = {}
        for long_id, locs in self.env.adj_dict.items():
            scan_id, viewpoint_id, _ = long_id.split("_")
            if viewpoint_id not in self.angle_direct_dict:
                self.angle_direct_dict[viewpoint_id] = \
                    {loc['nextViewpointId']:loc['absViewIndex'] for loc in locs[1:]}
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))
        print('%s loaded with %d tours, using splits: %s' % (
            self.__class__.__name__, len(self.tour_data), self.name))

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
            
    def _get_gt_trajs(self, data):
        return {x['instr_id']: (x['scan'], x['path']) for x in data}

    def size(self):
        return len(self.data)

    def check_last(self):
        if self.tour_batch is None: # start of env
            return True
        else:
            last = True
            for i in range(self.batch_size):
                if self.batch[i] is not None and self.batch[i]['instr_id'] != self.tour_batch[i][-1]:
                    last = False
                else:
                    self.tour_ended[i] = True
            return last
    
    def check_reach(self):
        reach = True
        batch_tmp = self.batch
        self.batch = [None] * self.batch_size
        for i in range(self.batch_size):
            if batch_tmp[i] is not None:
                if batch_tmp[i]['path'][-1] != self.env.world_states[i].viewpoint_id:
                    self.batch[i] = {
                        'heading': self.env.world_states[i].heading,
                        'path': self.shortest_paths[self.env.world_states[i].scan_id][
                                    self.env.world_states[i].viewpoint_id][batch_tmp[i]['path'][-1]][:-1]
                    }
                    reach = False
                else:
                    self.batch[i] = {
                        'heading': self.env.world_states[i].heading,
                        'path': []
                    }
        if not reach:
            self.extra_obs = self._get_path_obs()
        self.batch = batch_tmp
        return reach

    def _next_minibatch(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if self.iterative:
            if self.check_last():
                batch = self.tour_data[self.ix: self.ix + batch_size]
                if len(batch) < batch_size:
                    random.shuffle(self.tour_data)
                    self.ix = batch_size - len(batch)
                    batch += self.tour_data[:self.ix]
                else:
                    self.ix += batch_size
                self.tour_batch = batch
                self.tour_batch_probe = np.array([0] * batch_size)
                self.tour_ended = [False] * self.batch_size
                self.extra_obs = None
            else:
                self.tour_batch_probe += 1
                batch_tmp = [self.data[self.tour_batch[i][self.tour_batch_probe[i]]] \
                                 if not self.tour_ended[i] else None for i in range(batch_size)]
                for i in range(batch_size):
                    if batch_tmp[i] is not None and self.env.world_states[i].viewpoint_id != batch_tmp[i]['path'][0]:
                        batch_tmp[i] = {
                            'heading': self.env.world_states[i].heading,
                            'path': self.shortest_paths[self.env.world_states[i].scan_id][
                                        self.env.world_states[i].viewpoint_id][batch_tmp[i]['path'][0]][:-1]
                        }
                self.batch = batch_tmp
                self.extra_obs = self._get_path_obs()
            self.batch = [self.data[self.tour_batch[i][self.tour_batch_probe[i]]] \
                            if not self.tour_ended[i] else None for i in range(batch_size)]
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
            self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            if self.iterative:
                random.shuffle(self.tour_data)
            else:
                random.shuffle(self.data)
        self.tour_batch = None
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.viewpoint_id == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.shortest_paths[state.scan_id][state.viewpoint_id][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        candidate = []
        long_id = "%s_%s_%s" % (scanId, viewpointId, viewId)
        next_points = self.env.adj_dict[long_id]
        for loc in next_points[1:]:
            angle_feat = angle_feature(loc['rel_heading'], loc['rel_elevation'], self.angle_feat_size)
            candidate.append({
                'heading': loc['rel_heading'],
                'elevation': loc['rel_elevation'],
                'scanId': scanId,
                'viewpointId': loc['nextViewpointId'],  # Next viewpoint id
                'pointId': loc['absViewIndex'],
                'feature': np.concatenate((feature[loc['absViewIndex']], angle_feat), -1)
            })
        return candidate

    def _teacher_path_action(self, state, path, t=None, shortest_teacher=False):
        if shortest_teacher:
            return self._shortest_path_action(state, path[-1])

        teacher_vp = None
        if t is not None:
            teacher_vp = path[t + 1] if t < len(path) - 1 else state.viewpoint_id
        else:
            if state.viewpoint_id in path:
                cur_idx = path.index(state.viewpoint_id)
                if cur_idx == len(path) - 1: # STOP
                    teacher_vp = state.viewpoint_id
                else:
                    teacher_vp = path[cur_idx + 1]
        return teacher_vp

    def _get_path_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            if item is None:
                obs.append(None)
                continue
            path_feats = []
            angle_feats = []
            path_view_id = round(item['heading'] / ANGLE_INC) % 12 + 12
            path_view_ids = [path_view_id]
            for j, viewpoint_id in enumerate(item['path']):
                feats = self.env.feat_db.get_image_feature(state.scan_id, viewpoint_id)
                feats = np.concatenate((feats, self.angle_feature[path_view_id]), -1)
                angle_feat = np.zeros(self.angle_feat_size, np.float32)
                if j != len(item['path']) - 1:
                    long_id = "%s_%s_%s" % (state.scan_id, viewpoint_id, path_view_id)
                    next_points = self.env.adj_dict[long_id]
                    for loc in next_points[1:]:
                        if loc['nextViewpointId'] == item['path'][j + 1]:
                            angle_feat = angle_feature(loc['rel_heading'], loc['rel_elevation'], self.angle_feat_size)
                            path_view_id = loc['absViewIndex']
                            break
                path_feats.append(feats)
                angle_feats.append(angle_feat)
                path_view_ids.append(path_view_id)
            obs.append({
                'gt_path_feats': path_feats,
                'gt_angle_feats': angle_feats,
                'gt_view_idxs': path_view_ids,
            })
        return obs

    def _get_obs(self, t=None, shortest_teacher=False):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            if item is None:
                obs.append(None)
                continue
            base_view_id = state.view_index

            if feature is None:
                feature = np.zeros((36, 2048))

            # Full features
            candidate = self.make_candidate(feature, state.scan_id, state.viewpoint_id, state.view_index)
            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            
            obs.append({
                'instr_id': item['instr_id'],
                'scan': state.scan_id,
                'viewpoint': state.viewpoint_id,
                'viewIndex': state.view_index,
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature,
                'candidate': candidate,
                'instruction': item['instruction'],
                'teacher': self._teacher_path_action(state, item['path'], t=t, shortest_teacher=shortest_teacher),
                'gt_path': item['path'],
                'path_id': item['path_id']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.shortest_distances[state.scan_id][state.viewpoint_id][item['path'][-1]]
        return obs

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(**kwargs)
        
        scanIds = [item['scan'] if item is not None else None for item in self.batch]
        viewpointIds = [item['path'][0] if item is not None else None for item in self.batch]
        headings = [item['heading'] if item is not None else None for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs(t=0)

    def step(self, actions, t=None, traj=None):
        ''' Take action (same interface as makeActions) '''
        full_actions = np.full(len(self.batch), -1, np.int64)
        counter = 0
        idmap = {}
        for i, b in enumerate(self.batch):
            if b is not None:
                full_actions[i] = actions[counter]
                idmap[counter] = i
                counter += 1
        self.env.makeActions(full_actions)
        for i, action in enumerate(actions):
            if action >= 0 and traj is not None:
                traj[i]['path'].append((self.env.world_states[idmap[i]].viewpoint_id,
                                        self.env.world_states[idmap[i]].heading,
                                        self.env.world_states[idmap[i]].elevation))
        return self._get_obs(t=t)


    ############### Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _eval_item(self, scan, path, gt_path):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = [x[0] for x in item['trajectory']]
            scan, gt_traj = self.gt_trajs[instr_id]
            traj_scores = self._eval_item(scan, traj, gt_traj)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
            metrics['gt_path'].append(item['gt_path'])
            metrics['gt_length'].append(item['gt_length'])

        
        avg_metrics = {
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
        }
        return avg_metrics, metrics