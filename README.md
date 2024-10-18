<div align="center">    
 
## Light_Domain_Adaptation: Um estudo experimental sobre aplicação prática de métodos de deep learning à segmentação semântica de áreas navegáveis e não navegáveis em tempo real


</div>

<!-- TODO: Add video -->

### Resumo


 <div text-align="justify" align="justify">    
A direção autônoma tem potencial para transformar o transporte urbano ao aumentar a segurança
e eficiência, mas enfrenta desafios relacionados à interpretação do ambiente em tempo real. A
segmentação semântica é essencial nesse processo, permitindo que sistemas autônomos identifi-
quem áreas navegáveis e obstáculos. No Brasil, há desafios específicos devido às particularidades
das vias e à dificuldade de generalizar modelos treinados com dados internacionais, como o
conjunto CityScapes. Este trabalho propõe um modelo eficiente e leve de segmentação semân-
tica, baseado no LiteSeg e integrado às arquiteturas CCT e PixMatch, adaptado ao contexto
brasileiro. O conjunto de dados CityScapesBrazil, com 21.485 imagens, foi gerado para validar
o contexto de cenas urbana brasileira. O modelo alcançou 95,4% de IoU na classe “road” e 30
FPS em alta resolução, comprovando sua eficiência e capacidade de operação em tempo real. A
proposta oferece uma solução robusta e prática para a navegação autônoma em vias brasileiras,
com alta performance e compatível com dispositivos de recursos limitados.
</div>

### Como Executar

#### Dependências
 -  torch==2.4.1
 -  torchvision==0.19.1
 -  python=3.10.14
 -  albumentations==1.4.18
 -  tensorboard==2.17.1

Demais dependências na exportação do ambiente anaconda presente em  [`\environment.yaml`](./environment.yaml)

#### Preparação dos dados
Antes de iniciar é necessário preencher o diretório [`datasets\`](./datasets/) com os conjuntos [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) e [CityScapes](https://www.cityscapes-dataset.com/) seguindo a estrutura de pastas: 
```
datasets
├── CityScapeData
|   |── city_list
│   ├── gtFine
│   │   ├── train
│   │   │   ├── aachen
│   │   │   └── ...
│   │   └── val
│   └── leftImg8bit
│       ├── train
│       └── val
├── GTA5Dataset
|   |── gta5_list
│   ├── images
│   ├── labels
│   └── list
├── CityScapesBrazil
```

#### Treinamento CCT
Por padrão será utilizado o modelo prétreinado do LiteSeg para o conjunto CityScapes com anotações grosseiras. Temos o script principal para administrar o treinamento [`\CCT\train.py`](./CCT/train.py), e o arquivo json de configuração com todos os hiperparâmetros e controle de checkpoint [`\CCT\configs\<myconfig>.json`](./CCT/configs/)

```bash
python <rootProject>\CCT\train.py --config <rootProject>\CCT\configs\<myconfig>.json
```
#### Treinamento PIX
Por padrão será utilizado o modelo prétreinado do LiteSeg para o conjunto CityScapes com anotações grosseiras. Temos o script principal para administrar o treinamento [`\pixmatch\main.py`](./pixmatch/main.py), e o arquivo yaml de configuração com todos os hiperparâmetros e controle de checkpoint [`\pixmatch\configs\<myconfig>.yaml`](./pixmatch/configs/)

```bash
python <rootProject>\pixmatch\main.py --config <rootProject>\pixmatch\configs\<myconfig>.yaml
```

#### Evaluation
Para verificar a performance do modelo é utilizado os scripts:

#### CCT, preencha as linhas 10, 13, 74 e 77 com a época deseja
```bash
python <rootProject>\experimento\Inference\eval-metricts-cct.py
```
#### PIX, preencha a linha 11 e 13 com a época deseja
```bash
python <rootProject>\experimento\Inference\eval-metricts-pix.py
```

#### Saída esperada
```bash
Carregando <modelo> Epoca: <nº epoca>
AD 1024x2048: <FPS>
AD 360x640: <FPS>
Flops:  <N>GMac
Params: <N>M
```

#### Citação   
```bibtex
@inproceedings{ferreira2024LDA,
  author    = {William Luis Alves Ferreira},
  title     = {Um estudo experimental sobre aplicação prática de métodos de deep
learning à segmentação semântica de áreas navegáveis e não
navegáveis em tempo real},
  year         = {2024},
  howpublished = {\url{https://github.com/illiamw/Light_Domain_Adaptation}},
  note         = {Acesso em: XX out. 2XXX}
}
```
