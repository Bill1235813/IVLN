class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env):
        self.env = env
        self.results = {}
        self.losses = [] # For learning agents

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'scan': v['scan'],
                           'trajectory': v['path'], 'gt_path': v['gt_path'],
                           'gt_length': v['gt_length']})
            if detailed_output:
                output[-1]['details'] = v['details']
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        self.tour_prev_ended = [False] * self.env.batch_size
        self.history = None
        self.history_raw = None
        self.history_raw_length = None
        if iters is not None:
            # For each time, it will run_r2r_il.sh the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
                if self.env.check_last():
                    self.tour_prev_ended = [False] * self.env.batch_size
                    self.history = None
                    self.history_raw = None
                    self.history_raw_length = None
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                if self.env.check_last():
                    self.tour_prev_ended = [False] * self.env.batch_size
                    self.history = None
                    self.history_raw = None
                    self.history_raw_length = None
                    if looped:
                        break


