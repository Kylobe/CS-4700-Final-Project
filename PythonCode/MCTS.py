from ChessEnv import ChessEnv
import math
import chess
import chess.syzygy as syzygy
import numpy as np
import random
import torch
from bestmove import get_best_move

class Node:
    def __init__(self, args, board:chess.Board, terminated, win_val, parent=None, action:tuple[int, int, int] = None, prior=0):
        self.game = board
        self.args = args
        self.tensor_state:torch.Tensor = None
        self.parent:Node = parent
        self.action:tuple[int, int, int] = action
        self.children:list[Node] = []
        self.visit_count = 0
        self.win_count = 0
        self.terminated = terminated
        self.win_val = win_val
        self.prior = prior

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        favorite_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                favorite_child = child
                best_ucb = ucb
        return favorite_child

    def get_ucb(self, child):
        # Terminals: child.win_val is from child's POV
        if child.terminated:
            if child.win_val == -1:   # child is lost => parent wins
                q_val = 1.0
            elif child.win_val == 1:  # child is winning => parent loses
                q_val = 0.0
            else:                     # draw
                q_val = 0.5
            u = 0.0  #no exploration on terminal nodes
        else:
            if child.visit_count > 0:
                ratio = child.win_count / child.visit_count  # in [-1,1]
                q_val = 1 - ((ratio + 1) / 2.0)              # parent success prob in [0,1]
            else:
                q_val = 0.5  # neutral

            u = self.args['C'] * (np.sqrt(self.visit_count + 1) / (child.visit_count + 1)) * child.prior
        return q_val + u


    def back_propagate(self, val):
        self.win_count += val
        self.visit_count += 1
        if self.parent is not None:
            self.parent.back_propagate(-val)

    def expand(self, policy, legal_actions):
        planes, rows, cols = legal_actions
        for p, r, c in zip(planes, rows, cols):
            action = (p, r, c)
            prob = policy[action]
            temp_board = self.game.copy(stack=False)
            uci_action = ChessEnv.decode_action(action, temp_board)
            temp_board.push(uci_action)
            child_terminal = temp_board.is_game_over()
            child_win_val = 0
            if child_terminal:
                result = temp_board.result()
                if result != "1/2-1/2":
                    child_win_val = -1 #parent node delivered checkmate, child node loses.
                    self.terminated = True
                    self.win_val = 1
                    if not self.parent is None and not self.parent.parent is None and self.action == (24, 0, 6) and self.parent.action == (0, 6, 2) and self.parent.parent.action == (21, 7, 1) and action == (6, 0, 2):
                        print("found it!")
                    if not self.parent is None:
                        Node.back_prop_terminal(self.parent)
            child = Node(self.args, temp_board, child_terminal, child_win_val, self, action, prob)
            self.children.append(child)

    @staticmethod
    def back_prop_terminal(node):
        all_children_terminal = True
        all_children_winning = True
        for child in node.children:
            if not child.terminated:
                all_children_terminal = False
                all_children_winning = False
                break
            elif child.win_val < 1:
                all_children_winning = False
        if all_children_winning:
            node.win_val = -1
            node.terminated = True
            if not node.parent is None:
                node.parent.win_val = 1
                node.parent.terminated = True
                if not node.parent.parent is None:
                    Node.back_prop_terminal(node.parent.parent)
        elif all_children_terminal:
            node.win_val = 0
            node.terminated = True
            if not node.parent is None:
                Node.back_prop_terminal(node.parent)

class MCTS:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.root = None
        if args.get("use_tables"):
            tb_path = args.get("syzygy_path")  # e.g., "C:/chess_tb/syzygy"
            if tb_path:
                try:
                    self.tb = syzygy.open_tablebase(tb_path)
                except Exception:
                    self.tb = None
        else:
            self.tb = None

    def create_root(self, state):
        board:chess.Board = state
        tensor_state = ChessEnv.encode_board(board)
        terminated = board.is_game_over()
        value = 0
        if terminated:
            result = board.result()
            if result == "1-0":
                value = 1
            elif result == "0-1":
                value = -1
        if board.turn != chess.WHITE:
            value *= -1
        self.root = Node(self.args, board, terminated, value)
        self.root.tensor_state = tensor_state

    def advance_root(self, move:tuple[int, int, int]):
        for child in self.root.children:
            # child.action here should be the *move*, not the policy index
            if child.action[0] == move[0] and child.action[1] == move[1] and child.action[2] == move[2]:
                self.root = child
                self.root.parent = None
                return
        # If not found, rebuild from scratch:
        self.root = None

    @torch.no_grad
    def search(self, state, num_searches:int = None):
        if self.root is None:
            self.create_root(state)

        if num_searches is None:
            num_searches = self.args['num_searches']

        for _ in range(num_searches):
            if self.root.terminated:
                if self.root.win_val == 1:
                    action_probs = np.zeros((73, 8, 8), dtype=np.float32)
                    for child in self.root.children:
                        child:Node
                        if child.win_val == -1:
                            action_probs[child.action] = 1
                            return action_probs
                elif self.root.win_val == -1:
                    action_probs = ChessEnv.create_plane_action_mask(self.root.game)
                    action_probs /= action_probs.sum()
                    return action_probs
                else:
                    for child in self.root.children:
                        child:Node
                        if child.win_val == 0:
                            action_probs[child.action] = 1
                            return action_probs
            node = self.root
            while node.is_fully_expanded():
                node = node.select()
            
            value = node.win_val
            terminated = node.terminated

            # inside search(), at the leaf just before NN eval/expand
            if not terminated:
                # 3a) Try local Syzygy first
                best_move = None
                tb_value = None
                if self.tb is not None:
                    try:
                        # perfect game-theoretic outcome for side-to-move
                        wdl = self.tb.probe_wdl(node.game)  # -2,-1,0,+1,+2
                        tb_value = 1.0 if wdl > 0 else (-1.0 if wdl < 0 else 0.0)
                        # pick a TB-optimal move
                        try:
                            best_move = self.tb.probe_root(node.game)
                        except KeyError:
                            best_move = None
                    except KeyError:
                        pass  # not in tablebase

                # 3b) If no local TB move, optionally try Lichess API
                if best_move is None and self.args.get("use_lichess_tb_api", False):
                    try:
                        best_move = get_best_move(node.game)
                        # You don't get a numeric value from API easily; you can keep tb_value=None
                    except Exception:
                        best_move = None

                if best_move is not None:
                    # one-hot policy on the TB move
                    mask = ChessEnv.create_plane_action_mask(node.game)
                    policy = np.zeros_like(mask, dtype=np.float32)
                    idx = ChessEnv.encode_action(best_move, node.game.turn)  # make sure you have encode_action(turn-aware)
                    policy[idx] = 1.0
                    legal_actions = np.nonzero(policy)

                    # expand with TB policy (single child), and use tb_value if available
                    node.expand(policy, legal_actions)
                    value = node.win_val if tb_value is None else tb_value
                else:
                    # fall back to NN
                    mask = ChessEnv.create_plane_action_mask(node.game)
                    if node.tensor_state is None:
                        node.tensor_state = ChessEnv.encode_board(node.game)
                    state_input = node.tensor_state.unsqueeze(0)
                    policy_logits, value_t = self.model(state_input.to(self.model.device))
                    policy = torch.softmax(policy_logits, dim=1).squeeze(0).detach().cpu().numpy()
                    policy *= mask
                    s = policy.sum()
                    policy = policy / s if s > 0 else mask / max(mask.sum(), 1)
                    value = float(value_t.item())
                    legal_actions = np.nonzero(policy)
                    node.expand(policy, legal_actions)


            node.back_propagate(value)

        action_probs = np.zeros((73, 8, 8), dtype=np.float32)
        for child in self.root.children:
            child:Node
            action_probs[child.action] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


