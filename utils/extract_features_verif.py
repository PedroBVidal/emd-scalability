import torch 
from tqdm import tqdm

def extract_embedding_verif(input_blob, model, fm='arcface', level=4):
    model.eval()
    labels = []

    with torch.no_grad():
        feature_bank = []
        feature_bank_center = []
        avgpool_bank_center = []
        weights = []
        
        print("input blob type")
        #print(input_blob)
        #print(input_blob.shape)
        #print(type(input_blob))

        for input_img in input_blob:
            input_img = input_img.reshape(1,3,112,112) 
            out = model(input_img.cuda('cuda:2'))
            fea = out['fea']
            if fm == 'arcface':
                if level == 4:
                    aux_f = out['embedding_44']
                    avg_pool = out['adpt_pooling_44']
                elif level == 8:
                    aux_f = out['embedding_88'] 
                    avg_pool = out['adpt_pooling_88']
                else:
                    aux_f = out['embedding_16']
                    avg_pool = out['adpt_pooling_16']
            elif fm == 'sphereface' or fm == 'cosface' or fm == 'facenet':
                aux_f = out['embedding']
                avg_pool = out['adpt_pooling']
            no_avg_feat = aux_f

            avgpool_bank_center.append(avg_pool.data)
            feature_bank.append(no_avg_feat.data)
            feature_bank_center.append(fea.data)


        feature_bank = torch.cat(feature_bank, dim=0)
        N, C, _, _ = feature_bank.size()
        feature_bank = feature_bank.view(N, C, -1)
        feature_bank_center = torch.cat(feature_bank_center, dim=0)
        avgpool_bank_center = torch.cat(avgpool_bank_center, dim=0).squeeze(-1).squeeze(-1)

    feature_bank = torch.nn.functional.normalize(feature_bank, p=2, dim=1)
    feature_bank_center = torch.nn.functional.normalize(feature_bank_center, p=2, dim=1)
    avgpool_bank_center = torch.nn.functional.normalize(avgpool_bank_center, p=2, dim=1)

    return feature_bank, feature_bank_center, avgpool_bank_center, labels, weights
