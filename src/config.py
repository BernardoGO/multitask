class Config:
    _ENV_SPACE = None
    __USE_PRIOR_KNOWLEDGE__ = None
    num_context = None
    _ACTION_SPACE = None
    n_episodes = 0

    __USE_PRIOR_KNOWLEDGE__ = None#bool(int(args.k))
    __TRAIN_LAST_LAYER__ = None#bool(int(args.t))
    __COPY_LAST_WEIGHTS__ = None#bool(int(args.c))
    __SIZE__ = 0#int(args.s)
    __ENV__ = 'AirRaid-v0'
    #__COMMENT__ = "{}-Size{}-Lw{}-Tl{}-Uk{}".format(__ENV__, __SIZE__, __COPY_LAST_WEIGHTS__, __TRAIN_LAST_LAYER__, __USE_PRIOR_KNOWLEDGE__)
    __COMMENT__ = lambda : "{}-Size{}-Lw{}-Tl{}-Uk{}".format(Config.__ENV__, Config.__SIZE__, Config.__COPY_LAST_WEIGHTS__, Config.__TRAIN_LAST_LAYER__, Config.__USE_PRIOR_KNOWLEDGE__)

    contex =  [1,0]