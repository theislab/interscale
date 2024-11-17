sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'loss',
    'goal': 'minimize'
    }

sweep_config['metric'] = metric

parameters_dict = {
    'batch_size': {
        'values': [20, 32, 64]
        },
    'dropout': {
          'values': [0.3, 0.4, 0.5]
        },
    }

parameters_dict.update({
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # integers between 32 and 256
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })

parameters_dict.update({
    'epochs': {
        'value': 10}
    })

sweep_config['parameters'] = parameters_dict