from AlphaZeroChess import AlphaZeroChess, AlphaZero
from ChessEnv import ChessEnv
import torch
import torch.multiprocessing as mp
from torch import optim
from collections import deque
from torch.optim.lr_scheduler import CosineAnnealingLR

def reset_lr(optim, lr):
    for pg in optim.param_groups:
        pg['lr'] = lr

def run_worker(rank, model, args, return_queue):
    env = ChessEnv()
    alpha = AlphaZero(model, None, env, args)
    result = []
    for _ in range(args['num_self_play_iterations']):
        result += alpha.self_play()
    return_queue.put(result)

def main():
    mp.set_start_method("spawn", force=True)
    load_model = False
    args = {
        'C': 2,
        'material_weight': 0.5,
        'num_searches': 800,
        'num_iterations': 10,
        'num_self_play_iterations': 5,
        'epochs': 5,
        'num_processes': 1,
        'res_blocks': 12,
        'num_hidden': 128,
        'batch_size': 128,
        'phase': 0,
        'syzygy_path':"C:\chess_tb\syzygy",
        'base-lr': 0.1,
        'use_tables':False
    }

    env = ChessEnv()
    model = AlphaZeroChess(env, num_resBlocks=args['res_blocks'], num_hidden=128)
    if load_model:
        model.load_state_dict(torch.load("LatestAlphaChess.pt", map_location=torch.device('cpu')))
    model.share_memory()



    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    if load_model:
        optimizer.load_state_dict(torch.load("LatestOptimizer.pt", map_location=torch.device('cpu')))
    alpha = AlphaZero(model, optimizer, env, args)
    memory = deque(maxlen=50000)
    scheduler = CosineAnnealingLR(optimizer, T_max=50) 
    for iteration in range(args['num_iterations']):
        model.eval()
        return_queue = mp.Queue()
        processes = []

        for rank in range(args['num_processes']):
            p = mp.Process(target=run_worker, args=(rank, model, args, return_queue))
            p.start()
            processes.append(p)

        for _ in range(args['num_processes']):
            memory += return_queue.get()

        for p in processes:
            p.join()
        memory_list = list(memory)
        model.train()
        reset_lr(optimizer, args['base-lr'])

        # re-create scheduler for this cycle so it decays over `epochs`
        scheduler = CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=1e-5)
        for epoch in range(args['epochs']):
            avg_loss, avg_policy, avg_value = alpha.train(memory_list)
            scheduler.step()
            print(f"[Iter {iteration+1}] Epoch {epoch+1}: Total Loss = {avg_loss}, Policy Loss: {avg_policy}, Value Loss: {avg_value}")

        torch.save(model.state_dict(), f"Phase1AlphaChess{iteration}.pt")
        torch.save(optimizer.state_dict(), f"Phase1Optimizer{iteration}.pt")
    torch.save(model.state_dict(), f"LatestAlphaChess.pt")
    torch.save(optimizer.state_dict(), f"LatestOptimizer.pt")


if __name__ == "__main__":
    main()

