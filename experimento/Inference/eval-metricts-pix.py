import os, sys
import torch
import time
import flops_counter


def metrics():
    sys.path.append('C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/pixmatch')
    from models import get_model_test
    model = get_model_test()
    checkpoint = torch.load("C:/Users/will_/OneDrive/Documentos/GitHub/Light_Domain_Adaptation/pixmatch/checkpoint/GTA5ToCityscape_epoca_95.pth")
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    print("Carregando PIX Epoca:", checkpoint['epoch'])

    model.eval()
    model.cuda()

    #burn-in with 200 images   
    for x in range(0,1000):
        image = torch.randn(1, 3, 360, 640).cuda()
        with torch.no_grad():
            output = model.forward(image)


    #reporting results in fps:  
    total=0
    for x in range(0,1000):
        image = torch.randn(1, 3, 1024, 2048).cuda()
        with torch.no_grad():
            a = time.perf_counter()
            output = model.forward(image)
            torch.cuda.synchronize()
            b = time.perf_counter()
            total+=b-a
    print("1024x2048: "+str(1000/total))

    #reporting results in fps:  
    total=0
    for x in range(0,1000):
        image = torch.randn(1, 3, 360, 640).cuda()
        with torch.no_grad():
            a = time.perf_counter()
            output = model.forward(image)
            torch.cuda.synchronize()
            b = time.perf_counter()
            total+=b-a
    print("360x640: "+str(1000/total))


    batch = torch.FloatTensor(1, 3, 512, 1024).cuda()
    model = flops_counter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    _ = model(batch)



    print('Flops:  {}'.format(flops_counter.flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + flops_counter.get_model_parameters_number(model))


if __name__=='__main__':
    metrics()