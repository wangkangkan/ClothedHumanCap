from lib.config import cfg, args

def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.networks.renderer import make_renderer
    from lib.evaluators import make_evaluator

    gpus = torch.cuda.device_count()
    print(gpus)
    print(torch.cuda.get_device_name(0))

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.train()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = renderer.render_deformation(batch)
        evaluator.evaluate(output, batch)
    evaluator.summarize()

if __name__ == '__main__':
    globals()['run_' + args.type]()
