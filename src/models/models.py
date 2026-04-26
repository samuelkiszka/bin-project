from src.models.vggembednet1 import VGGEmbedNet1
from src.models.vggembednet2 import VGGEmbedNet2
from src.models.vggembednet3 import VGGEmbedNet3
from src.models.vggembednet4 import VGGEmbedNet4
from src.models.vggembednet5 import VGGEmbedNet5
from src.models.vggembednet6 import VGGEmbedNet6
from src.models.vggembednet7 import VGGEmbedNet7
from src.models.vggembednet8 import VGGEmbedNet8
from src.models.vggembednet9 import VGGEmbedNet9
from src.models.vggembednet10 import VGGEmbedNet10
from src.models.vggembednet11 import VGGEmbedNet11
from src.models.vggembednet12 import VGGEmbedNet12
from src.models.vggembednet13 import VGGEmbedNet13


def get_models():
    return {
        'all': None,
        VGGEmbedNet1.NAME: VGGEmbedNet1,
        VGGEmbedNet2.NAME: VGGEmbedNet2,
        VGGEmbedNet3.NAME: VGGEmbedNet3,
        VGGEmbedNet4.NAME: VGGEmbedNet4,
        VGGEmbedNet5.NAME: VGGEmbedNet5,
        VGGEmbedNet6.NAME: VGGEmbedNet6,
        VGGEmbedNet7.NAME: VGGEmbedNet7,
        VGGEmbedNet8.NAME: VGGEmbedNet8,
        VGGEmbedNet9.NAME: VGGEmbedNet9,
        VGGEmbedNet10.NAME: VGGEmbedNet10,
        VGGEmbedNet11.NAME: VGGEmbedNet11,
        VGGEmbedNet12.NAME: VGGEmbedNet12,
        VGGEmbedNet13.NAME: VGGEmbedNet13,
    }