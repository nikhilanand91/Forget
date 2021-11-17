from models.models import Model
from base.hparams import HParams
from open_lth.foundations import hparams as open_lth_hparams
from open_lth.models import registry as open_lth_registry


class CifarResnet(Model):
    """
    We get the CifarResnet model from OpenLTH to save some time.
    """

    def __init__(self, train_hparams: HParams):
        self.open_lth_params = open_lth_hparams(train_hparams.model,
                                                'kaiming_uniform',
                                                'uniform')

    def forward(self):
        pass

    def get_model(self):
        return open_lth_registry.get(self.open_lth_params).cuda()