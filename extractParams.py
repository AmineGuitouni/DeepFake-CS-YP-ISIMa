import torch
import os
from models.mvssnet import get_mvss
from models.upernet import EncoderDecoder

def load_paper_model(model_type, load_path,image_size):
    if (model_type == 'mvssnet'):
        model = get_mvss(backbone='resnet50',
                            pretrained_base=True,
                            nclass=1,
                            constrain=True,
                            n_input=3,
                            ).cuda()
    elif (model_type == 'upernet'):
        model = EncoderDecoder(n_classes=1, img_size=image_size, bayar=False).cuda()
    elif (model_type == 'ours'):
        model = EncoderDecoder(n_classes=1, img_size=image_size, bayar=True).cuda()
    else:
        print("Unrecognized model %s" % model_type)
    
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        print("load %s finish" % (os.path.basename(load_path)))
        return model
    
    print("%s not exist" % load_path)
    return None

def load_ISMA_model(path_to_paper_model= "ShallowDeepFakesLocalization/checkPoints/9.pth",
                    path_to_cls = "checkPoints/cls_head/0.pth",
                    path_to_seg= "checkPoints/seg_head/0.pth"):
    
    model = load_paper_model("ours",path_to_paper_model).to("cuda")
    model.decoder.cls_head = torch.load(path_to_cls).to("cuda")
    model.decoder.seg_head = torch.load(path_to_seg).to("cuda")

    return model

def get_decoder(model):
    return model.decoder

def get_ppm_head(model):
    return get_decoder(model).ppm_last_conv

def get_ppm_conv(model):
    return get_decoder(model).ppm_conv

def get_ppm_pooling(model):
    return get_decoder(model).ppm_pooling

def get_encoder(model): #resnet
    return model.encoder

def get_cls_head(model): # calssification
    return get_decoder(model).cls_head

def get_seg_head(model): # segmentation (mask)
    return get_decoder(model).seg_head

def get_decoder_fpn_in(model):
    return get_decoder(model).fpn_in

def get_decoder_fpn_out(model):
    return get_decoder(model).fpn_out

def get_decoder_conv_fusion(model):
    return get_decoder(model).conv_fusion
