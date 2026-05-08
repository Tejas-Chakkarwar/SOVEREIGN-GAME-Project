import gymnasium
import numpy as np
import networkx as nx

# action indices for political actions
SEEK_ALLIANCE = 0
IMPOSE_SANCTION = 1
ISSUE_THREAT = 2
NEGOTIATE = 3
DO_NOTHING = 4

# action indices for military actions
ADVANCE = 0
HOLD = 1
WITHDRAW = 2
STRIKE = 3

# controller labels
INVADER = 0
DEFENDER = 1
NEUTRAL = 2
CONTESTED = 3  # special case — not stored in M directly, M row is [0,0,0]

# node indices — I'll just use integers to index into arrays
# mapping: I0=0, D0=1, N0=2, C0=3, C1=4, C2=5, C3=6, C4=7, C5=8
I0 = 0
D0 = 1
N0 = 2
C0 = 3
C1 = 4
C2 = 5
C3 = 6
C4 = 7
C5 = 8

NUM_NODES = 9
T_MAX = 200

# reward weights
w_T = 0.30
w_R = 0.20
w_O = 0.25
w_L = 0.15
w_S = 0.20
w_I = 0.10


class SovereignEnv(gymnasium.Env):
    def __init__(self, use_legitimacy=True, use_occupation_cost=True,
                 use_neutral_posture=True, sanction_threshold=0.60):

        super().__init__()

        # ablation flags
        self.use_legitimacy = use_legitimacy
        self.use_occupation_cost = use_occupation_cost
        self.use_neutral_posture = use_neutral_posture
        self.sanction_threshold = sanction_threshold

        # action space: [political action, military action]
        self.action_space = gymnasium.spaces.MultiDiscrete([5, 4])

        # observation space: 9*3 + 9 + 9 + 4 = 49
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(49,), dtype=np.float32
        )

        # build the graph once — nodes don't change, just their attributes
        self._build_graph()

        # node attributes — these are fixed (resource and strategic values)
        # order: I0, D0, N0, C0, C1, C2, C3, C4, C5
        self.resource_value = np.array([0.8, 0.8, 0.6, 0.4, 0.4, 0.5, 0.4, 0.3, 0.3], dtype=np.float32)
        self.strategic_value = np.array([0.9, 0.9, 0.5, 0.6, 0.5, 0.9, 0.7, 0.5, 0.4], dtype=np.float32)

        # state will be initialized in reset()
        self.M = None
        self.U_I = None
        self.U_D = None
        self.U_N = None

        self.L = None      # legitimacy
        self.E = None      # supply index
        self.theta = None  # neutral posture
        self.t_occ = None  # occupation time

        self.t = None      # current timestep

        self.sanctions_active = None
        self.below_threshold_count = None
        self.neutral_joined_defender = None
        self.supply_routes_open = None
        self.neutral_allied_invader = None

        self.settlement_score = None
        self.last_3_actions = None  # list of military actions (last 3)

        self.insurgency_happened = False

        # track prev resource total for delta_resources in reward
        self.prev_invader_resources = 0.0

    def _build_graph(self):
        # build the 9-node graph according to the topology in the strategy
        self.G = nx.Graph()
        self.G.add_nodes_from(range(NUM_NODES))

        # edges — exactly as specified
        # N0=2, C0=3, C5=8, C2=5, I0=0, C3=6, D0=1, C4=7, C1=4
        edges = [
            (N0, C0),  # N0 -- C0
            (N0, C5),  # N0 -- C5
            (C0, C5),  # C0 -- C5
            (C0, C2),  # C0 -- C2
            (C0, I0),  # C0 -- I0
            (C5, C2),  # C5 -- C2
            (C5, C3),  # C5 -- C3
            (C2, I0),  # C2 -- I0
            (C2, C3),  # C2 -- C3
            (I0, C3),  # I0 -- C3
            (I0, C4),  # I0 -- C4
            (C3, D0),  # C3 -- D0
            (C4, C1),  # C4 -- C1
        ]
        self.G.add_edges_from(edges)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # M is 9x3: each row is one-hot for [Invader, Defender, Neutral]
        # contested = [0, 0, 0] (handled as a special case)
        self.M = np.zeros((NUM_NODES, 3), dtype=np.float32)

        # set initial controllers
        # I0 -> Invader, D0 -> Defender, C1 -> Defender, C3 -> Defender
        # N0, C0, C2, C4, C5 -> Neutral
        self.M[I0][INVADER] = 1
        self.M[D0][DEFENDER] = 1
        self.M[N0][NEUTRAL] = 1
        self.M[C0][NEUTRAL] = 1
        self.M[C1][DEFENDER] = 1
        self.M[C2][NEUTRAL] = 1
        self.M[C3][DEFENDER] = 1
        self.M[C4][NEUTRAL] = 1
        self.M[C5][NEUTRAL] = 1

        # unit counts
        self.U_I = np.zeros(NUM_NODES, dtype=np.float32)
        self.U_D = np.zeros(NUM_NODES, dtype=np.float32)
        self.U_N = np.zeros(NUM_NODES, dtype=np.float32)

        self.U_I[I0] = 15.0   # 12 ground + 3 strike (just track total)
        self.U_D[D0] = 7.0    # 6 ground + 1 strike
        self.U_N[N0] = 4.0    # non-combatant

        # state variables
        self.L = 1.0
        self.E = 1.0
        self.theta = 0.0
        self.t_occ = 0
        self.t = 0

        # threshold tracking
        self.sanctions_active = False
        self.below_threshold_count = 0
        self.neutral_joined_defender = False
        self.supply_routes_open = False
        self.neutral_allied_invader = False

        self.settlement_score = 0
        self.last_3_actions = []

        self.insurgency_happened = False
        self.prev_invader_resources = self._compute_invader_resources()

        return self._get_obs(), {}

    def _get_controller(self, node):
        # returns INVADER, DEFENDER, NEUTRAL, or CONTESTED
        row = self.M[node]
        if row[INVADER] == 1:
            return INVADER
        elif row[DEFENDER] == 1:
            return DEFENDER
        elif row[NEUTRAL] == 1:
            return NEUTRAL
        else:
            return CONTESTED

    def _set_controller(self, node, controller):
        # set who controls a node
        self.M[node] = 0
        if controller != CONTESTED:
            self.M[node][controller] = 1
        # if CONTESTED, leave as [0,0,0]

    def _get_invader_territories(self):
        # returns list of node indices controlled by invader
        result = []
        for i in range(NUM_NODES):
            if self._get_controller(i) == INVADER:
                result.append(i)
        return result

    def _get_defender_territories(self):
        result = []
        for i in range(NUM_NODES):
            if self._get_controller(i) == DEFENDER:
                result.append(i)
        return result

    def _compute_invader_resources(self):
        total = 0.0
        for node in self._get_invader_territories():
            total += self.resource_value[node]
        return total

    def _compute_supply_index(self):
        # E = fraction of invader-controlled territories connected to I0
        # connected means there's a path through invader-controlled nodes only
        invader_territories = self._get_invader_territories()
        if len(invader_territories) == 0:
            return 0.0
        if I0 not in invader_territories:
            # I0 is not under invader control, so supply is broken
            return 0.0

        # build subgraph of only invader-controlled nodes
        subgraph = self.G.subgraph(invader_territories)
        connected_count = 0
        for node in invader_territories:
            # check if this node can reach I0 in the subgraph
            if nx.has_path(subgraph, node, I0):
                connected_count += 1

        return connected_count / len(invader_territories)

    def _apply_political(self, a_pol):
        # apply the political action effect on L and theta
        # I'll handle theta updates in _update_theta separately for cleanliness
        if not self.use_legitimacy:
            return

        if a_pol == SEEK_ALLIANCE:
            self.L += 0.01
        elif a_pol == IMPOSE_SANCTION:
            self.L -= 0.02
            # target E penalty applied — I'll reduce E here as a penalty modifier
            # the strategy says "target E -0.03" so I'll reduce E directly
            self.E = max(0.0, self.E - 0.03)
        elif a_pol == ISSUE_THREAT:
            self.L -= 0.03
        elif a_pol == NEGOTIATE:
            self.L += 0.03
        elif a_pol == DO_NOTHING:
            if self.L < 0.5:
                self.L -= 0.01

        # clip L to valid range
        self.L = np.clip(self.L, 0.0, 1.0)

    def _apply_military(self, a_mil):
        # apply the military action
        if a_mil == ADVANCE:
            self._do_advance()
        elif a_mil == HOLD:
            self._do_hold()
        elif a_mil == WITHDRAW:
            self._do_withdraw()
        elif a_mil == STRIKE:
            self._do_strike()

        # track last 3 military actions
        self.last_3_actions.append(a_mil)
        if len(self.last_3_actions) > 3:
            self.last_3_actions.pop(0)

    def _do_advance(self):
        # claim an adjacent territory — pick the best adjacent non-invader territory
        # I'll try to advance into a contested or neutral territory adjacent to an invader node
        invader_territories = self._get_invader_territories()
        candidates = []
        for node in invader_territories:
            for neighbor in self.G.neighbors(node):
                ctrl = self._get_controller(neighbor)
                if ctrl != INVADER:
                    # check we have units at this invader node to move
                    if self.U_I[node] > 0:
                        candidates.append((node, neighbor))

        if len(candidates) == 0:
            return

        # pick first candidate — simple greedy for now
        # I'll pick the one with highest resource value in the neighbor
        best_src = -1
        best_dst = -1
        best_val = -1
        for src, dst in candidates:
            if self.resource_value[dst] > best_val:
                best_val = self.resource_value[dst]
                best_src = src
                best_dst = dst

        if best_dst == -1:
            return

        # mark as contested — combat will resolve in _resolve_combat
        self._set_controller(best_dst, CONTESTED)
        # move one unit forward to represent the push
        self.U_I[best_dst] += 1
        self.U_I[best_src] -= 1

        if self.use_legitimacy:
            self.L = max(0.0, self.L - 0.05)
        if self.use_occupation_cost:
            self.t_occ += 1

    def _do_hold(self):
        # hold — just increment t_occ if in non-home territory
        if self.use_occupation_cost:
            # check if invader occupies any territory other than I0
            invader_territories = self._get_invader_territories()
            for node in invader_territories:
                if node != I0:
                    self.t_occ += 1
                    break  # I'll count this as "occupying non-home territory"

    def _do_withdraw(self):
        # cede one contested territory back — pick the one with lowest strategic value
        # invader gives up one territory they control (not I0)
        invader_territories = self._get_invader_territories()
        non_home = []
        for node in invader_territories:
            if node != I0:
                non_home.append(node)

        if len(non_home) == 0:
            return

        # pick the one to withdraw from — lowest strategic value
        worst_node = non_home[0]
        worst_val = self.strategic_value[non_home[0]]
        for node in non_home[1:]:
            if self.strategic_value[node] < worst_val:
                worst_val = self.strategic_value[node]
                worst_node = node

        # move units back (return units to I0 if possible)
        self.U_I[I0] += self.U_I[worst_node]
        self.U_I[worst_node] = 0

        # return territory to neutral (or defender if defender had units there)
        if self.U_D[worst_node] > 0:
            self._set_controller(worst_node, DEFENDER)
        else:
            self._set_controller(worst_node, NEUTRAL)

        if self.use_legitimacy:
            self.L = min(1.0, self.L + 0.02)

        # reset t_occ if fully withdrawn (only I0 remains)
        new_invader_territories = self._get_invader_territories()
        if len(new_invader_territories) <= 1:  # only I0 or nothing
            if self.use_occupation_cost:
                self.t_occ = 0

    def _do_strike(self):
        # destroy one defender unit — pick a node where defender has units
        best_target = -1
        best_units = 0
        for node in range(NUM_NODES):
            if self.U_D[node] > best_units:
                best_units = self.U_D[node]
                best_target = node

        if best_target == -1:
            return

        self.U_D[best_target] -= 1
        self.U_D[best_target] = max(0.0, self.U_D[best_target])

        if self.use_legitimacy:
            self.L = max(0.0, self.L - 0.08)
        if self.use_occupation_cost:
            self.t_occ += 1

    def _defender_step(self):
        # rule-based defender policy — exactly as specified in strategy

        # priority 1: retake D0 if invader controls it
        if self._get_controller(D0) == INVADER:
            self._defender_attack(D0)
            return

        # priority 2: recapture C1 or C3 if invader controls them
        if self._get_controller(C1) == INVADER or self._get_controller(C3) == INVADER:
            # find the nearest invaded territory between C1 and C3
            targets = []
            if self._get_controller(C1) == INVADER:
                targets.append(C1)
            if self._get_controller(C3) == INVADER:
                targets.append(C3)

            # find which target has more invader units (more urgent)
            # or just pick the first one — I'll pick the one with most invader units
            target = targets[0]
            for t in targets[1:]:
                if self.U_I[t] < self.U_I[target]:
                    target = t
            self._defender_attack(target)
            return

        # priority 3: reinforce D0 if invader units are adjacent to D0
        invader_adjacent_to_d0 = False
        for neighbor in self.G.neighbors(D0):
            if self.U_I[neighbor] > 0:
                invader_adjacent_to_d0 = True
                break

        if invader_adjacent_to_d0:
            self._defender_reinforce(D0)
            return

        # priority 4: counterattack if E < 0.5 and defender has local advantage
        if self.E < 0.5:
            # check if defender has local advantage anywhere adjacent to invader territory
            defender_territories = self._get_defender_territories()
            for def_node in defender_territories:
                for neighbor in self.G.neighbors(def_node):
                    if self._get_controller(neighbor) == INVADER:
                        if self.U_D[def_node] > self.U_I[neighbor]:
                            # local advantage — counterattack
                            self._defender_attack(neighbor)
                            return

        # priority 5: hold
        # do nothing

    def _defender_attack(self, target_node):
        # defender moves one unit toward target_node
        # find a defender node adjacent to target that has units
        for neighbor in self.G.neighbors(target_node):
            if self.U_D[neighbor] > 0:
                # move one unit into the target
                self.U_D[target_node] += 1
                self.U_D[neighbor] -= 1
                # mark as contested
                if self._get_controller(target_node) != DEFENDER:
                    self._set_controller(target_node, CONTESTED)
                return

    def _defender_reinforce(self, target_node):
        # move one unit toward target_node from an adjacent defender node
        # find adjacent defender node with units
        for neighbor in self.G.neighbors(target_node):
            if self.U_D[neighbor] > 0 and self._get_controller(neighbor) == DEFENDER:
                self.U_D[target_node] += 1
                self.U_D[neighbor] -= 1
                return

    def _resolve_combat(self):
        # go through each contested or mixed-unit territory and resolve combat
        for node in range(NUM_NODES):
            i_units = self.U_I[node]
            d_units = self.U_D[node]

            if i_units <= 0 and d_units <= 0:
                continue

            # only resolve if both sides have units here
            if i_units > 0 and d_units > 0:
                # defender gets a bonus in D0 if defender controls it
                # wait — D0 bonus is "when Defender controls it"
                # but if both sides have units it's contested... I'll check original controller
                # I'll apply the bonus if D0 and it was defender-controlled before the attack
                effective_defender = d_units
                if node == D0:
                    # D0 home territory defense bonus
                    effective_defender = d_units * 1.2

                if i_units > effective_defender:
                    # attacker (invader) wins
                    self.U_D[node] -= 1
                    self.U_D[node] = max(0.0, self.U_D[node])
                    # move 1 attacker unit to territory (already there from advance)
                    self._set_controller(node, INVADER)
                else:
                    # defender wins
                    self.U_I[node] -= 1
                    self.U_I[node] = max(0.0, self.U_I[node])
                    # territory goes back to defender
                    if self.U_D[node] > 0:
                        self._set_controller(node, DEFENDER)
                    else:
                        self._set_controller(node, NEUTRAL)

    def _update_map(self):
        # update controllers based on who has units where
        # this is more of a cleanup — make sure controllers match unit presence
        for node in range(NUM_NODES):
            i_units = self.U_I[node]
            d_units = self.U_D[node]

            if i_units > 0 and d_units == 0:
                self._set_controller(node, INVADER)
            elif d_units > 0 and i_units == 0:
                self._set_controller(node, DEFENDER)
            elif i_units > 0 and d_units > 0:
                self._set_controller(node, CONTESTED)
            # if neither has units, keep the controller as is
            # (territory is "occupied" but units can move away)

    def _update_state_variables(self):
        # update E (supply index)
        self.E = self._compute_supply_index()
        self.t += 1

    def _update_theta(self, a_pol, a_mil):
        # neutral posture drift — this took me a while to figure out the formula
        if not self.use_neutral_posture:
            return

        mu = (0.04 * (1 - self.L)
              + 0.05 * (a_mil == ADVANCE)
              + 0.10 * (a_mil == STRIKE)
              - 0.04 * (a_pol == NEGOTIATE)
              - 0.03 * (a_pol == SEEK_ALLIANCE)
              + 0.03 * (self.t_occ / T_MAX))

        noise = np.random.normal(0, 0.02)
        self.theta = np.clip(self.theta + mu + noise, -1.0, 1.0)

    def _check_thresholds(self):
        # check all threshold-based events
        if not self.use_neutral_posture:
            return

        lift_threshold = self.sanction_threshold - 0.10

        # sanctions logic
        if self.theta > self.sanction_threshold and not self.sanctions_active:
            self.sanctions_active = True
            self.below_threshold_count = 0

        if self.sanctions_active:
            if self.theta < lift_threshold:
                self.below_threshold_count += 1
                if self.below_threshold_count >= 5:
                    self.sanctions_active = False
            else:
                self.below_threshold_count = 0

        # coalition — neutral joins defender (irreversible)
        if self.theta > 0.85 and not self.neutral_joined_defender:
            self.neutral_joined_defender = True
            self.U_D[D0] += 1
            self.U_D[C3] += 1
            if self.use_legitimacy:
                self.L = max(0.0, self.L - 0.10)

        # supply routes open — theta < -0.60
        if self.theta < -0.60 and not self.supply_routes_open:
            self.supply_routes_open = True

        # neutral allied with invader — theta < -0.85
        if self.theta < -0.85 and not self.neutral_allied_invader:
            self.neutral_allied_invader = True
            if self.use_legitimacy:
                self.L = max(0.0, self.L - 0.05)

        # note: DO_NOTHING theta effect is handled at the top of step()
        # before _apply_political is called, so no need to do it here

    def _insurgency_prob(self):
        if not self.use_occupation_cost:
            return 0.0
        return 1 - np.exp(-0.05 * self.t_occ)

    def _destroy_invader_unit(self):
        # destroy one invader unit in a random occupied territory
        invader_territories = self._get_invader_territories()
        occupied = []
        for node in invader_territories:
            if node != I0 and self.U_I[node] > 0:
                occupied.append(node)

        if len(occupied) == 0:
            # if no units outside I0, destroy one at I0
            if self.U_I[I0] > 0:
                self.U_I[I0] -= 1
            return

        # pick a random occupied territory to hit
        target = occupied[int(np.random.random() * len(occupied))]
        self.U_I[target] -= 1
        self.U_I[target] = max(0.0, self.U_I[target])

    def _check_terminal(self):
        # check all terminal conditions in order
        if self.use_legitimacy and self.L <= 0:
            return True, -50.0

        total_invader = float(np.sum(self.U_I))
        if total_invader == 0:
            return True, -30.0

        # settlement is checked separately in _check_settlement (called from step)
        # because it needs a_pol — so we just handle non-settlement terminals here

        if self.t >= T_MAX:
            return True, 0.0

        # all territories controlled by invader
        all_invader = True
        for node in range(NUM_NODES):
            if self._get_controller(node) != INVADER:
                all_invader = False
                break
        if all_invader:
            return True, 10.0

        return False, 0.0

    def _check_settlement(self, a_pol):
        # check negotiated settlement conditions
        # this is separate from _check_terminal because it needs a_pol
        if a_pol == NEGOTIATE:
            self.settlement_score += 1
        else:
            self.settlement_score -= 1
        self.settlement_score = int(np.clip(self.settlement_score, 0, 5))

        if self.settlement_score >= 5:
            if self.L >= 0.70:
                if abs(self.theta) <= 0.30:
                    if self.t_occ == 0:
                        # check last 3 military actions don't contain ADVANCE or STRIKE
                        bad_actions = False
                        for act in self.last_3_actions:
                            if act == ADVANCE or act == STRIKE:
                                bad_actions = True
                                break
                        if not bad_actions:
                            return True, 40.0

        return False, 0.0

    def _compute_reward(self):
        invader_territories = self._get_invader_territories()

        # resource territory reward
        resource_total = 0.0
        for node in invader_territories:
            resource_total += self.resource_value[node]

        # delta resources since last step
        current_resources = resource_total
        delta_resources = current_resources - self.prev_invader_resources
        self.prev_invader_resources = current_resources

        # positive reward
        r_pos = w_T * resource_total + w_R * delta_resources

        # occupation cost — with possible 30% reduction if supply routes open
        occ_cost = w_O * (self.t_occ / T_MAX)
        if self.supply_routes_open:
            occ_cost = occ_cost * 0.70

        # negative reward components
        r_neg = (occ_cost
                 + w_L * (1 - self.L)
                 + w_S * (1 if self.sanctions_active else 0) * (1 - self.E)
                 + w_I * float(self.insurgency_happened))

        reward = r_pos - r_neg
        return float(reward)

    def _get_obs(self):
        # build the 49-dim observation vector
        obs = np.concatenate([
            self.M.flatten(),          # 9*3 = 27
            self.U_I.astype(np.float32),  # 9
            self.U_D.astype(np.float32),  # 9
            np.array([self.L, self.E, self.theta, float(self.t_occ)], dtype=np.float32)  # 4
        ])
        return obs.astype(np.float32)

    def step(self, action):
        a_pol = int(action[0])
        a_mil = int(action[1])

        # DO_NOTHING has a theta effect that depends on t_occ
        # handling it here before _apply_political keeps things simple
        if a_pol == DO_NOTHING and self.use_neutral_posture:
            if self.t_occ > 0:
                self.theta = np.clip(self.theta + 0.01, -1.0, 1.0)

        self._apply_political(a_pol)
        self._apply_military(a_mil)
        self._defender_step()
        self._resolve_combat()
        self._update_map()
        self._update_state_variables()
        self._update_theta(a_pol, a_mil)
        self._check_thresholds()

        # check insurgency
        self.insurgency_happened = np.random.random() < self._insurgency_prob()
        if self.insurgency_happened:
            self._destroy_invader_unit()

        # check settlement first (updates settlement_score)
        settled, settlement_reward = self._check_settlement(a_pol)

        if settled:
            reward = self._compute_reward() + settlement_reward
            return self._get_obs(), reward, True, False, {}

        # check other terminal conditions
        done, terminal_reward = self._check_terminal()
        reward = self._compute_reward() + terminal_reward

        return self._get_obs(), reward, done, False, {}

    def render(self):
        # simple text render — just print the current state
        print(f"Step: {self.t}, L={self.L:.3f}, E={self.E:.3f}, theta={self.theta:.3f}, t_occ={self.t_occ}")
        print(f"Invader territories: {self._get_invader_territories()}")
        print(f"Sanctions: {self.sanctions_active}, Supply open: {self.supply_routes_open}")
