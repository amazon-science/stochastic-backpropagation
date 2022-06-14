from dlhammer.bootstrap import bootstrap, logger


def main():
    cfg = bootstrap(print_cfg=False)
    cfg.DDP.ENABLE = False
    from lstr.utils.parser import postprocess_cfg
    cfg = postprocess_cfg(cfg)
    if cfg.MODEL.MODEL_NAME == 'E2E_LSTR':
        from lstr.models.lstr.eval import evaluate
        evaluate(cfg, logger)
    else:
        raise ValueError('Not implemented')


if __name__ == "__main__":
    main()
