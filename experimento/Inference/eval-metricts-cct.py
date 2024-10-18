import os, sys
import torch
import time
import flops_counter
import json

def metrics():
    sys.path.append('C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/CCT')
    from models import CCT
    configmodel = json.load(open("C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/CCT/configs/finalizados/config_CCT1_CB.json"))
    num_classes = 3

    checkpoint = torch.load("C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/CCT/modelsaved/CCT_V1_CB/CCT_V1_CB_14_checkpoint.pth")

    model = CCT(num_classes=num_classes,
                        conf=configmodel['model'], testing=True, versionmode=1)
    try:
        if 'module.' in list(checkpoint['state_dict'].keys())[0]: ## Correção de nomenclatura dos pesos
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("Carregando CCT Epoca:", checkpoint['epoch'])

    model.eval()
    model.cuda()

    #burn-in with 200 images   
    for x in range(0,1000):
        image = torch.randn(1, 3, 360, 640).cuda()
        with torch.no_grad():
            output = model.forward(image, domain=2)


    #reporting results in fps:  
    total=0
    for x in range(0,1000):
        image = torch.randn(1, 3, 1024, 2048).cuda()
        with torch.no_grad():
            a = time.perf_counter()
            output = model.forward(image, domain=2)
            torch.cuda.synchronize()
            b = time.perf_counter()
            total+=b-a
    print("Semi 1024x2048: "+str(1000/total))

    #reporting results in fps:  
    total=0
    for x in range(0,1000):
        image = torch.randn(1, 3, 360, 640).cuda()
        with torch.no_grad():
            a = time.perf_counter()
            output = model.forward(image, domain=2)
            torch.cuda.synchronize()
            b = time.perf_counter()
            total+=b-a
    print("Semi 360x640: "+str(1000/total))


    batch = torch.FloatTensor(1, 3, 512, 1024).cuda()
    model = flops_counter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    _ = model(batch, domain=2)



    print('Flops:  {}'.format(flops_counter.flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + flops_counter.get_model_parameters_number(model))


    sys.path.append('C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/CCT')
    from models import CCT
    configmodel = json.load(open("C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/CCT/configs/finalizados/config_CCT1.json"))
    num_classes = 3

    checkpoint = torch.load("C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/CCT/modelsaved/CCT_V1/CCT_V1_40_checkpoint.pth")

    model = CCT(num_classes=num_classes,
                        conf=configmodel['model'], testing=True, versionmode=1)
    try:
        if 'module.' in list(checkpoint['state_dict'].keys())[0]: ## Correção de nomenclatura dos pesos
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("Carregando CCT Epoca:", checkpoint['epoch'])

    model.eval()
    model.cuda()

    #burn-in with 200 images   
    for x in range(0,200):
        image = torch.randn(1, 3, 360, 640).cuda()
        with torch.no_grad():
            output = model.forward(image, domain=2)


    #reporting results in fps:  
    total=0
    for x in range(0,1000):
        image = torch.randn(1, 3, 1024, 2048).cuda()
        with torch.no_grad():
            a = time.perf_counter()
            output = model.forward(image, domain=2)
            torch.cuda.synchronize()
            b = time.perf_counter()
            total+=b-a
    print("AD 1024x2048: "+str(1000/total))

    #reporting results in fps:  
    total=0
    for x in range(0,1000):
        image = torch.randn(1, 3, 360, 640).cuda()
        with torch.no_grad():
            a = time.perf_counter()
            output = model.forward(image, domain=2)
            torch.cuda.synchronize()
            b = time.perf_counter()
            total+=b-a
    print("AD 360x640: "+str(1000/total))


    batch = torch.FloatTensor(1, 3, 512, 1024).cuda()
    model = flops_counter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    _ = model(batch, domain=2)



    print('Flops:  {}'.format(flops_counter.flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + flops_counter.get_model_parameters_number(model))

if __name__=='__main__':
    metrics()