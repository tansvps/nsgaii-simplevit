import random
import string
from transformer_simplevit_V2_dropout import SimpleViT
import torch
class Population:
    def random_pop(self):
        init_chromosome = []
        type_transfm = []
        depth = random.choice([2,3,4,5,6,7,8])
        for i in range(depth):
            transformer_types = ['Transformer_FA', 'Transformer_AF', 'Transformer_AA', 'Transformer_FF']
            transformer_type = random.choice(transformer_types)
            # print("transformer_type", transformer_type)
            h = [1, 2, 4, 8, 16]
            heads = random.choice(h)
            mlp_d = [128, 256, 512, 1024, 2048, 4096]
            mlp_dim = random.choice(mlp_d)
            if transformer_type == 'Transformer_FA':
                vit_tuple = (mlp_dim,heads)
                init_chromosome.append(vit_tuple)
            elif transformer_type == 'Transformer_AF':
                vit_tuple = (heads,mlp_dim)
                init_chromosome.append(vit_tuple)
            elif transformer_type == 'Transformer_AA':
                heads1 = heads
                heads2 = random.choice(h)
                vit_tuple = (heads1,heads2)
                init_chromosome.append(vit_tuple)
            elif transformer_type == 'Transformer_FF':
                mlp_dim1 = mlp_dim
                mlp_dim2 = random.choice(mlp_d)
                vit_tuple = (mlp_dim1,mlp_dim2)
                init_chromosome.append(vit_tuple)
            type_transfm.append(transformer_type)

        return init_chromosome , type_transfm,depth

    def transfer_genotype(self,genotype,type_transfm,depth):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleViT(image_size=32, patch_size=4, num_classes=10, dim=256, depth=depth,genotype=genotype,type_transfm=type_transfm,dropout=0.1)
        model = model.to(device)
        param = sum(param.numel() for param in model.parameters())

        return model, param

